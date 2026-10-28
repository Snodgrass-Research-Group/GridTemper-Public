"""Reactive power placement — ReactivePowerPlacement.m + PVPQSwitching.m.

Algorithm doc: conversion_manual/algorithms/07_refinement.md (both the active
path and the dormant placement loop).

Deviations applied:
* **DV3** — slack relocation uses the largest generator's *bus id*, not the
  generator row index MATLAB used as a bus number.
* **DV13** — the incidence is rebuilt correctly (via ``build_incidence``), unlike
  ReactivePowerPlacement.m:64's scrambled ``[Indx1;Indx1]`` that made the final
  detailed AC power flow spuriously singular. With the fix the detailed solve
  actually runs (see DEVIATIONS.md; pending the author's confirmation).
* **DV4** — PVPQSwitching takes ``s_base`` and the complete SNet matrix as
  explicit parameters (MATLAB referenced out-of-scope ``options.SBase`` /
  ``PF.SNetComplete``).

PVPQSwitching source: Synthetic-Grid-Generator/KCL-KVL Compliant Grid
Generator/Matlab Functions/PVPQSwitching.m (READ-ONLY sibling repo, D17).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..config import CaseConfig, TemperOptions
from ..grid_model import GridModel, PowerFlowResult, SingularNetworkError
from ..powerflow.ac import ACPowerFlow


@dataclass
class SwitchState:
    """Mutable bus-type / SNet state threaded through PVPQSwitching (the dormant
    reactive-limit loop). Fields mirror the ``ACPF`` struct members the switching
    reads/writes; ``resolve`` re-runs the detailed AC solve between passes."""

    n_bus: int
    snet: np.ndarray            # (n_bus, n_scenarios) complex, per unit (mutable)
    v: np.ndarray               # solved complex voltages
    snet_calc: np.ndarray       # 100·V·conj(Y·V) from the last solve (complex, MVA)
    v_set: np.ndarray           # PV setpoints (complex)
    gen_list: np.ndarray        # PV bus ids (0-based)
    load_list: np.ndarray       # PQ bus ids
    slack: np.ndarray           # slack bus ids
    high_q: float
    low_q: float


class ReactivePlacement:
    """Final pipeline stage: bus typing, AC solves, and (behind
    ``reactive.run_placement_loop``) the sync-condenser placement loop."""

    def __init__(
        self,
        case: CaseConfig,
        options: TemperOptions,
        reference_ybus: sp.spmatrix | None = None,
    ):
        self.case = case
        self.options = options
        #: YBus_new dump for the simple-path override (quirk Q1); None computes it.
        self.reference_ybus = reference_ybus

    # ------------------------------------------------------------------ run
    def run(
        self,
        model: GridModel,
        A: sp.csc_matrix,
        snet: np.ndarray,
        run_detailed: bool = True,
    ) -> tuple[GridModel, PowerFlowResult]:
        """Active path (algorithms/07 §Active).

        Clamp ``B ≤ 1``, derive the committed-generator / load bus sets from
        ``snet``, start flat with a single slack and every other bus PV
        (loadlist empty), normalize sign/scale, run the *simple* AC power flow
        (YBus_new override when ``reference_ybus`` is set) to get
        ``Branch.MaxACPF``, then the *detailed* solve for bus state, and finally
        relocate the slack to the largest generator's bus (DV3).

        Returns ``(model, simple_result)``; ``model.branches['MaxACPF']`` holds
        the per-line max |flow| (MVA).
        """
        m = model.copy()
        branches = m.branches
        n_bus = m.n_bus
        s_base = self.case.s_base_mva

        # Clamp line charging (crude series-compensation stand-in, quirk).
        b = branches["B"].to_numpy().copy()
        b[b > 1] = 1.0
        branches["B"] = b

        # Committed generators / bus sets from SNet.
        gen_bus = m.gens["bus"].to_numpy(np.int64)
        committed = np.real(snet[gen_bus]) > 0.1
        gen_buses = np.unique(gen_bus[committed])

        # Duplicate the (single) scenario, flip sign so generation is negative
        # (pfmiss convention), and auto-scale MW→pu when the total is large.
        acsnet = snet.astype(complex).copy()
        if np.sum(np.real(acsnet[gen_buses])) > 0:
            acsnet = -acsnet
        if np.sum(np.abs(np.real(acsnet))) > 20000:
            acsnet = acsnet / s_base

        # Bus types: one slack, everything else PV (all-PV start; loadlist empty).
        slack0 = self.case.slack_bus_internal_matlab - 1   # 1-based internal → 0-based
        slack = np.array([slack0], dtype=np.int64)
        pv = np.setdiff1d(np.arange(n_bus), slack)
        pq = np.array([], dtype=np.int64)
        v0 = np.ones(n_bus, dtype=complex)

        acpf = ACPowerFlow(
            self.options.newton,
            self.options.reactive,
            use_reference_ybus=self.reference_ybus is not None,
            reference_ybus=self.reference_ybus,
        )
        simple = acpf.solve(branches, A, v0, acsnet, slack, pv, pq)
        # ReactivePowerPlacement drops non-converged scenarios before taking the
        # per-line max; with the single base-case scenario, a non-converged solve
        # leaves an empty set -> MaxACPF is NaN (exactly the stage8 dump, where the
        # base-case AC power flow does not converge — see the B6 finding).
        n_branch = m.n_branch
        branches["MaxACPF"] = (
            simple.flow_max_mva if simple.converged else np.full(n_branch, np.nan)
        )
        m.log["acpf_simple_converged"] = bool(simple.converged)

        if run_detailed:
            try:
                detailed, stats = acpf.solve_detailed(branches, A, v0, acsnet, slack, pv, pq)
                m.log["acpf_detailed"] = {
                    "converged": bool(detailed.converged),
                    "iterations": detailed.iterations,
                    "losses_mw": stats.losses_mw,
                }
            except SingularNetworkError as exc:  # would only happen with real singularity
                m.log["acpf_detailed"] = f"singular: {exc}"

        # Bus typing + slack relocation to the biggest generator (DV3: use the
        # generator's bus id, not its row index).
        bus_type = np.full(n_bus, 2, dtype=int)   # PV
        bus_type[slack0] = 3
        max_gen_bus = int(gen_bus[np.argmax(m.gens["MaxMW"].to_numpy())])
        bus_type[max_gen_bus] = 3                 # relocated slack (both stay type 3, as MATLAB)
        m.buses["bus_type"] = bus_type
        m.branches = branches
        return m, simple

    # ------------------------------------------------------- placement loop
    def _placement_loop(self, *args, **kwargs):
        """The MATLAB ``while Go`` loop (Go = 0 today). Not reached on the active
        path; requires ``_pv_pq_switching``. Gated by
        ``reactive.run_placement_loop`` (default False)."""
        raise NotImplementedError(
            "reactive placement loop is dormant (run_placement_loop=False); "
            "enable + implement per algorithms/07 when needed"
        )

    # --------------------------------------------------------- PV/PQ switch
    @staticmethod
    def _pv_pq_switching(
        state: SwitchState,
        snet_complete: np.ndarray,
        s_base: float,
        max_scenario: int,
        resolve: Callable[[SwitchState], SwitchState],
    ) -> SwitchState:
        """PVPQSwitching.m (external repo): two passes of reactive-limit bus
        reclassification.

        Each pass: from the last solve's ``SNetCalc``, restore to PV the buses
        scheduled at the high/low Q limit whose voltage is off-nominal (resetting
        their SNet to real power only); clamp Q and demote to PQ the buses whose
        *solved* Q exceeds the limits; reset PV setpoints to 1 pu (keep angle) and
        re-solve, warm-starting the next pass.

        DV4: ``s_base`` and ``snet_complete`` (the full MW/MVar scenario matrix)
        are explicit parameters (MATLAB referenced out-of-scope ``options.SBase``
        / ``PF.SNetComplete``); ``resolve`` runs the detailed AC solve. Dormant —
        physics-tested only.
        """
        for _ in range(2):
            q_net = np.imag(state.snet_calc)
            vm = np.abs(state.v)

            at_high = np.flatnonzero(np.imag(state.snet[:, 0]) >= state.high_q)
            switch_high = at_high[vm[at_high] > 1.05]
            state.snet[switch_high, 0] = np.real(snet_complete[switch_high, max_scenario]) / s_base
            state.gen_list = np.union1d(state.gen_list, switch_high)

            at_low = np.flatnonzero(np.imag(state.snet[:, 0]) <= state.low_q)
            switch_low = at_low[vm[at_low] < 0.95]
            state.snet[switch_low, 0] = np.real(snet_complete[switch_low, max_scenario]) / s_base
            state.gen_list = np.union1d(state.gen_list, switch_low)

            over_q = np.flatnonzero(q_net > state.high_q)
            under_q = np.flatnonzero(q_net < state.low_q)
            state.snet[over_q, 0] = (
                1j * state.high_q / s_base + np.real(snet_complete[over_q, max_scenario]) / s_base
            )
            state.snet[under_q, 0] = (
                1j * state.low_q / s_base + np.real(snet_complete[under_q, max_scenario]) / s_base
            )
            state.load_list = np.union1d(state.load_list, np.union1d(over_q, under_q))
            state.gen_list = np.setdiff1d(
                np.arange(state.n_bus), np.union1d(state.load_list, state.slack)
            )

            angle = np.angle(state.v_set[state.gen_list])
            state.v_set[state.gen_list] = np.exp(1j * angle)  # PV setpoints → 1 pu, keep angle
            state = resolve(state)
        return state

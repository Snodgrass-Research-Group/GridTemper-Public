"""AC backend interface + the well-posed case shared by all three solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ...config import CaseConfig
from ...grid_model import GridModel


@dataclass(frozen=True)
class ACCase:
    """A well-posed AC power-flow case (one slack, PV generators, PQ loads).

    Derived from the pipeline's ``GridModel`` + ``snet`` — NOT the MATLAB-faithful
    all-PV setup (which does not converge, B6). Real-power quantities follow the
    natural sign convention (``gen_p_mw`` / ``load_p_mw`` both positive); each
    backend applies its own internal signs.
    """

    n_bus: int
    vn_kv: np.ndarray            # nominal kV per bus
    slack_bus: int               # 0-based
    from_bus: np.ndarray         # 0-based branch endpoints
    to_bus: np.ndarray
    r_pu: np.ndarray             # per-branch series R, pu on s_base
    x_pu: np.ndarray
    b_pu: np.ndarray             # per-branch total line charging, pu on s_base
    is_transformer: np.ndarray   # bool per branch (Tap != 0)
    gen_bus: np.ndarray          # PV bus ids (0-based)
    gen_p_mw: np.ndarray         # PV real power (generation, positive)
    load_bus: np.ndarray         # PQ bus ids
    load_p_mw: np.ndarray        # load real power (consumption, positive)
    load_q_mvar: np.ndarray
    s_base_mva: float
    v_setpoint: float = 1.0

    @property
    def n_branch(self) -> int:
        """Number of branches in the case."""
        return self.from_bus.size

    @classmethod
    def from_model(cls, model: GridModel, snet: np.ndarray, case: CaseConfig) -> "ACCase":
        """Translate the processed grid into a solvable case.

        Bus types: the config slack; committed generator buses (net real
        injection > 0.1 MW) as PV; every other bus with load as PQ. Real power
        comes from ``snet`` (the dispatched net injection); reactive load comes
        from the demand table (MVAr).
        """
        n_bus = model.n_bus
        s_base = case.s_base_mva
        slack = case.slack_bus_internal_matlab - 1
        net_p = np.real(snet)

        # Reactive load aggregated per bus from the demand table (MW+jMVAr).
        q_by_bus = np.zeros(n_bus)
        if model.demand is not None and "bus" in model.demand.columns:
            dbus = model.demand["bus"].to_numpy(np.int64)
            np.add.at(q_by_bus, dbus, np.imag(model.demand["S_pu"].to_numpy()))

        gen_candidates = np.unique(model.gens["bus"].to_numpy(np.int64))
        gen_bus = gen_candidates[(net_p[gen_candidates] > 0.1) & (gen_candidates != slack)]
        is_gen = np.zeros(n_bus, bool)
        is_gen[gen_bus] = True

        others = np.arange(n_bus)
        load_mask = (~is_gen) & (others != slack) & (
            (np.abs(net_p) > 1e-6) | (np.abs(q_by_bus) > 1e-6)
        )
        load_bus = np.flatnonzero(load_mask)

        b = model.branches
        return cls(
            n_bus=n_bus,
            vn_kv=b_nom_kv(model),
            slack_bus=slack,
            from_bus=b["from_bus"].to_numpy(np.int64),
            to_bus=b["to_bus"].to_numpy(np.int64),
            r_pu=b["R"].to_numpy(float),
            x_pu=b["X"].to_numpy(float),
            b_pu=b["B"].to_numpy(float),
            is_transformer=(b["Tap"].to_numpy() != 0),
            gen_bus=gen_bus,
            gen_p_mw=net_p[gen_bus],
            load_bus=load_bus,
            load_p_mw=-net_p[load_bus],           # consumption positive (net_p is negative at loads)
            load_q_mvar=q_by_bus[load_bus],
            s_base_mva=s_base,
        )


def b_nom_kv(model: GridModel) -> np.ndarray:
    """Per-bus nominal kV (``NomKV``; falls back to 1.0 where missing/zero)."""
    kv = model.buses["NomKV"].to_numpy(float) if "NomKV" in model.buses.columns else np.ones(model.n_bus)
    kv = np.where(np.isnan(kv) | (kv <= 0), 1.0, kv)
    return kv


@dataclass(frozen=True)
class ACBackendResult:
    """Unified AC solve output across backends."""

    backend: str
    converged: bool
    v_bus: np.ndarray            # complex per-unit bus voltages (n_bus,)
    flow_from_mva: np.ndarray    # complex send-end branch flow (MVA)
    flow_to_mva: np.ndarray      # complex receive-end
    flow_max_mva: np.ndarray     # max(|from|, |to|) per branch (sizing input)
    losses_mw: float


@runtime_checkable
class ACSolverBackend(Protocol):
    """Contract every AC power-flow backend implements.

    A backend is a small object with a ``name`` and two methods: it reports
    whether it can run in this environment (:meth:`available`) and it solves a
    well-posed :class:`ACCase`, returning a unified :class:`ACBackendResult`
    (:meth:`solve`). Concrete implementations live in ``builtin.py``,
    ``pandapower_backend.py``, and ``powerworld_backend.py``.
    """

    name: str

    def available(self) -> bool:
        """Whether this backend can run here (deps + external tools present)."""
        ...

    def solve(self, case: ACCase) -> ACBackendResult:
        """Solve ``case`` and return the bus voltages, branch flows, and losses."""
        ...


def get_backend(name: str, **kwargs) -> ACSolverBackend:
    """Instantiate a backend by name: ``builtin`` | ``pandapower`` | ``powerworld``."""
    name = name.lower()
    if name == "builtin":
        from .builtin import BuiltinBackend

        return BuiltinBackend(**kwargs)
    if name == "pandapower":
        from .pandapower_backend import PandaPowerBackend

        return PandaPowerBackend(**kwargs)
    if name == "powerworld":
        from .powerworld_backend import PowerWorldBackend

        return PowerWorldBackend(**kwargs)
    raise ValueError(
        f"unknown ac_backend {name!r}; choose 'builtin', 'pandapower', or 'powerworld'"
    )


def solve_ac(
    model: GridModel,
    snet: np.ndarray,
    case: CaseConfig,
    backend: str = "builtin",
    **backend_kwargs,
) -> ACBackendResult:
    """Build the well-posed :class:`ACCase` and solve it with the named backend."""
    ac_case = ACCase.from_model(model, snet, case)
    return get_backend(backend, **backend_kwargs).solve(ac_case)

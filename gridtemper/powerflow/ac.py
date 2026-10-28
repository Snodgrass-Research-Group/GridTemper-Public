"""AC power flow drivers — ACPowerFlowSimple.m + ACPowerFlowFinal.m, unified.

Algorithm doc: conversion_manual/algorithms/06_powerflow_drivers.md.
Bug fix DV10 applies: Ybus is COMPUTED from branch data; the MATLAB behavior of
overriding it with the YBus_new.m dump (ACPowerFlowSimple only) is available as
``use_reference_ybus`` (validation only).

``add_reactive_support`` raises NotImplementedError — the MATLAB
``ACAddReactiveSupport`` function does not exist anywhere (algorithms/08).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import maximum_bipartite_matching

from ..config import NewtonOptions, ReactiveOptions
from ..grid_model import PowerFlowResult, SingularNetworkError
from .newton import NewtonSolver
from .ybus import YbusBuilder

_CONDEST_SINGULAR = 1e16  # MATLAB condest(YBus) > 1e16 -> SingularY


@dataclass(frozen=True)
class ACStats:
    """ACPowerFlowFinal PF.Stats: top-30 voltage/angle extremes, losses."""

    max_v: np.ndarray
    min_v: np.ndarray
    max_angle_diff: np.ndarray
    mean_angle_diff: float
    losses_mw: float
    loss_percent: float


def _sprank(A: sp.spmatrix) -> int:
    """Structural rank (MATLAB ``sprank``) = size of the maximum bipartite
    matching between buses (rows) and branches (columns)."""
    match = maximum_bipartite_matching(A.tocsr(), perm_type="column")
    return int((match != -1).sum())


def is_singular(A: sp.csc_matrix, ybus: sp.spmatrix) -> str | None:
    """ACPowerFlowFinal's singularity test (returns a reason, or None).

    Matches MATLAB exactly (verified against a MATLAB probe on the base case,
    where all three are False): isolated buses (``ARank < n_bus``), structural
    rank deficiency (``sprank(A) < n_bus``), or ill-conditioning
    (``condest(YBus[1:,1:]) > 1e16``).
    """
    n_bus = A.shape[0]
    degree = np.asarray(np.abs(A).sum(axis=1)).ravel()
    if np.any(degree == 0):
        return f"{int((degree == 0).sum())} isolated buses (ARank < n_bus)"
    if _sprank(A) < n_bus:
        return "structural rank deficiency (sprank < n_bus)"
    if _condest(ybus[1:, 1:]) > _CONDEST_SINGULAR:
        return "ill-conditioned reduced Ybus (condest > 1e16)"
    return None


def _condest(m: sp.spmatrix) -> float:
    """Estimate ``cond_1(m) = ||m||_1 · ||m^{-1}||_1`` (MATLAB ``condest``) via a
    sparse LU and one-norm estimation. Returns ``inf`` if the LU is singular."""
    m = m.tocsc()
    try:
        lu = spla.splu(m)
    except RuntimeError:
        return np.inf
    n = m.shape[0]
    inv_op = spla.LinearOperator(
        (n, n),
        matvec=lambda x: lu.solve(x),
        rmatvec=lambda x: lu.solve(x, trans="T"),
        dtype=m.dtype,
    )
    try:
        return float(spla.onenormest(m) * spla.onenormest(inv_op))
    except Exception:  # pragma: no cover - defensive
        return np.inf


class ACPowerFlow:
    """MATLAB-faithful AC driver — the unified ACPowerFlowSimple/Final path used
    by the reactive stage (see module docstring). Composes a :class:`NewtonSolver`
    and :class:`YbusBuilder`; distinct from the well-posed backend solve."""

    def __init__(
        self,
        newton_options: NewtonOptions,
        reactive_options: ReactiveOptions,
        builder: YbusBuilder | None = None,
        use_reference_ybus: bool = False,
        reference_ybus: sp.spmatrix | None = None,
    ):
        self.newton = NewtonSolver(newton_options)
        self.reactive_options = reactive_options
        self.builder = builder or YbusBuilder()
        self.use_reference_ybus = use_reference_ybus
        self.reference_ybus = reference_ybus

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _branch_arrays(branches: pd.DataFrame):
        return (
            branches["from_bus"].to_numpy(np.int64),
            branches["to_bus"].to_numpy(np.int64),
            branches["R"].to_numpy(float),
            branches["X"].to_numpy(float),
            branches["B"].to_numpy(float),
        )

    def _line_flows(self, v, yprim, A, from_bus, to_bus):
        vdrop = A.T @ v
        iline = yprim @ vdrop
        send = 100.0 * v[from_bus] * np.conj(iline)
        recv = 100.0 * v[to_bus] * np.conj(iline)
        return send, recv, np.maximum(np.abs(send), np.abs(recv))

    # -------------------------------------------------------------- simple
    def solve(
        self,
        branches: pd.DataFrame,
        A: sp.csc_matrix,
        v0: np.ndarray,
        snet_pu: np.ndarray,
        slack: np.ndarray,
        pv: np.ndarray,
        pq: np.ndarray,
        contingency: int | None = None,
    ) -> PowerFlowResult:
        """ACPowerFlowSimple.m — one NR solve + series line flows.

        With ``use_reference_ybus`` the solve uses the YBus_new.m dump (quirk Q1
        / DV10) instead of the computed Ybus; the series ``YPrim`` used for line
        flows is always computed from R/X.
        """
        A = A.tocsc()
        from_bus, to_bus, r, x, b = self._branch_arrays(branches)
        yset = self.builder.build_ac(A, from_bus, to_bus, r, x, b, contingency)
        if self.use_reference_ybus:
            if self.reference_ybus is None:
                raise ValueError("use_reference_ybus=True but no reference_ybus provided")
            ybus = self.reference_ybus.tocsc()
        else:
            ybus = yset.ybus_long

        res = self.newton.solve(ybus, v0, snet_pu, slack, pv, pq)
        send, recv, flow_max = self._line_flows(res.v, yset.yprim, A, from_bus, to_bus)
        angle_diff = np.abs(np.asarray(A.T @ np.rad2deg(np.angle(res.v))).ravel())
        return PowerFlowResult(
            v_bus=res.v,
            flow_send_mva=send,
            flow_recv_mva=recv,
            flow_max_mva=flow_max,
            converged=res.converged,
            iterations=res.iterations,
            angle_diff_deg=angle_diff,
            ds_dv=res.ds_dv,
            history=res.history,
        )

    # ------------------------------------------------------------ detailed
    def solve_detailed(
        self,
        branches: pd.DataFrame,
        A: sp.csc_matrix,
        v0: np.ndarray,
        snet_pu: np.ndarray,
        slack: np.ndarray,
        pv: np.ndarray,
        pq: np.ndarray,
        contingency: int | None = None,
    ) -> tuple[PowerFlowResult, ACStats]:
        """ACPowerFlowFinal.m — computed Ybus (long/lossless per options), rich
        singularity check (node degree, connectivity, ``condest > 1e16``),
        ``dV/dQ``, line flows, and summary stats.

        Raises
        ------
        SingularNetworkError
            When the reduced Ybus is rank-deficient or ill-conditioned — this is
            the base-case behavior today (isolated/ill-conditioned network), so
            the caller (ReactivePowerPlacement) skips the solve, exactly like the
            MATLAB ``if ~PF.SingularY`` guard.
        NotImplementedError
            If ``add_reactive_support`` (the missing ACAddReactiveSupport path).
        """
        if getattr(self.reactive_options, "add_reactive_support", False):
            raise NotImplementedError(
                "ACAddReactiveSupport is missing from every repo (algorithms/08)"
            )
        A = A.tocsc()
        from_bus, to_bus, r, x, b = self._branch_arrays(branches)
        yset = self.builder.build_ac(A, from_bus, to_bus, r, x, b, contingency)
        opts = self.reactive_options
        if opts.lossless_lines:
            ybus = 1j * (yset.bbus_long if opts.long_line else yset.bbus_short)
        else:
            ybus = yset.ybus_long if opts.long_line else yset.ybus_short

        reason = is_singular(A, ybus)
        if reason is not None:
            raise SingularNetworkError(f"AC network singular: {reason}")

        res = self.newton.solve(ybus, v0, snet_pu, slack, pv, pq)
        send, recv, flow_max = self._line_flows(res.v, yset.yprim, A, from_bus, to_bus)
        angle_diff = np.abs(np.asarray(A.T @ np.rad2deg(np.angle(res.v))).ravel())

        vabs = np.abs(res.v)
        losses = float(np.sum(np.real(send - recv)))
        denom = np.sum(np.abs(np.real(snet_pu))) / 2.0
        stats = ACStats(
            max_v=np.sort(vabs)[::-1][:30],
            min_v=np.sort(vabs)[:30],
            max_angle_diff=np.sort(angle_diff)[::-1][:30],
            mean_angle_diff=float(np.mean(angle_diff)) if angle_diff.size else 0.0,
            losses_mw=losses,
            loss_percent=losses / denom if denom else 0.0,
        )
        result = PowerFlowResult(
            v_bus=res.v,
            flow_send_mva=send,
            flow_recv_mva=recv,
            flow_max_mva=flow_max,
            converged=res.converged,
            iterations=res.iterations,
            angle_diff_deg=angle_diff,
            ds_dv=res.ds_dv,
            history=res.history,
        )
        return result, stats

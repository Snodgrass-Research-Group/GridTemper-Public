"""DC (B–θ) power flow — DCPowerFlowSimple.m.

Algorithm doc: conversion_manual/algorithms/06_powerflow_drivers.md §DCPowerFlowSimple.
Slack is bus 0 unconditionally (MATLAB used bus 1; flows are slack-invariant).
Raises SingularNetworkError instead of returning a SingularB flag.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla

from ..grid_model import PowerFlowResult, SingularNetworkError
from .ybus import YbusBuilder


class DCPowerFlow:
    """DC (B–θ) power-flow solver — DCPowerFlowSimple.m (see module docstring)."""

    def __init__(self, builder: YbusBuilder | None = None):
        self.builder = builder or YbusBuilder()

    def solve(
        self,
        x_pu: np.ndarray,
        snet_pu: np.ndarray,
        A: sp.csc_matrix,
        contingency: int | None = None,
    ) -> PowerFlowResult:
        """Solve ``Bbus[1:,1:]·θ = Re(snet)[1:]``; flows ``= 100·Bprim·Aᵀ·[0;θ]`` MW.

        Parameters
        ----------
        x_pu : per-branch reactance (pu).
        snet_pu : complex bus injection **in per unit** (MATLAB feeds
            ``real(SNet)/SBase``); only the real part is used.
        A : incidence (n_bus × n_branch).

        Raises
        ------
        SingularNetworkError
            If a bus is isolated or the network splits into islands (replaces
            the ``SingularB`` return / ``ARank<buses`` / ``sprank`` checks).
        """
        A = A.tocsc()
        bbus, bprim, a_con = self.builder.build_dc(A, x_pu, contingency)
        graph = a_con if contingency is not None else A

        degree = np.asarray(np.abs(graph).sum(axis=1)).ravel()
        n_comp = csg.connected_components(graph @ graph.T, directed=False)[0]
        if np.any(degree == 0) or n_comp > 1:
            raise SingularNetworkError(
                f"DC network singular: {int((degree == 0).sum())} isolated buses, "
                f"{n_comp} connected components"
            )

        b_reduced = bbus[1:, 1:].tocsc()
        theta = spla.spsolve(b_reduced, np.real(snet_pu)[1:])
        theta_full = np.concatenate([[0.0], theta])

        pline_send = 100.0 * np.asarray((bprim @ A.T) @ theta_full).ravel()
        theta_deg = np.rad2deg(theta_full)
        angle_diff = np.abs(np.asarray(A.T @ theta_deg).ravel())
        v_bus = np.exp(1j * theta_full)  # |V| = 1, angle = θ (legacy VBusPolar)
        return PowerFlowResult(
            v_bus=v_bus,
            flow_send_mva=pline_send,
            flow_recv_mva=None,
            flow_max_mva=np.abs(pline_send),
            converged=True,
            iterations=1,
            angle_diff_deg=angle_diff,
            ds_dv=None,
            history=None,
        )

    def solve_many(
        self,
        x_pu: np.ndarray,
        snet_pu: np.ndarray,
        A: sp.csc_matrix,
        contingency: int | None = None,
    ) -> np.ndarray:
        """DC power flow for many scenarios sharing ONE Bbus factorization.

        All scenarios use the same reactances, so ``Bbus`` is built and factored
        once and every scenario is a solve against the cached LU (batched RHS) —
        far faster than calling :meth:`solve` per scenario (this is the
        ``parfor`` over scenarios in RealPowerLoop, done right).

        Parameters
        ----------
        x_pu : per-branch reactance (pu).
        snet_pu : complex bus injections, shape (n_bus, n_scenarios), per unit.
        A : incidence (n_bus × n_branch).

        Returns
        -------
        numpy.ndarray
            Real branch flows, shape (n_branch, n_scenarios), MW.
        """
        A = A.tocsc()
        snet_pu = np.atleast_2d(snet_pu.T).T if snet_pu.ndim == 1 else snet_pu
        bbus, bprim, a_con = self.builder.build_dc(A, x_pu, contingency)
        graph = a_con if contingency is not None else A
        degree = np.asarray(np.abs(graph).sum(axis=1)).ravel()
        n_comp = csg.connected_components(graph @ graph.T, directed=False)[0]
        if np.any(degree == 0) or n_comp > 1:
            raise SingularNetworkError("DC network singular/disconnected")

        lu = spla.splu(bbus[1:, 1:].tocsc())      # factor once
        pp = np.real(snet_pu)[1:, :]              # (n_bus-1, n_scen)
        theta = lu.solve(pp)                      # batched solve, one factorization
        theta_full = np.vstack([np.zeros((1, theta.shape[1])), theta])
        return 100.0 * np.asarray((bprim @ A.T) @ theta_full)

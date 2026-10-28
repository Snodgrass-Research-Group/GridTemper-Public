"""Built-in AC backend — the from-scratch scipy Newton-Raphson solver.

Solves the well-posed :class:`ACCase` (slack + PV + PQ). Uses full-Newton
damping by default (unlike the MATLAB-faithful 0.1 under-relaxation), since here
the goal is an actual converged solution, not bit-reproduction of MATLAB.
"""

from __future__ import annotations

import numpy as np

from ...config import NewtonOptions
from ..newton import NewtonSolver
from ..ybus import YbusBuilder
from ...topology.incidence import build_incidence
from .base import ACBackendResult, ACCase


class BuiltinBackend:
    """From-scratch scipy Newton-Raphson AC backend (see module docstring).

    The recommended default: pure numpy/scipy, no external tools, and it solves
    the full TX-2030 case.
    """

    name = "builtin"

    def __init__(self, newton_options: NewtonOptions | None = None):
        # Real-solve defaults: full Newton step, tight tolerance.
        self.newton_options = newton_options or NewtonOptions(
            tol=1e-8, max_iter=50, damping=1.0
        )

    def available(self) -> bool:
        """Always ``True`` — depends only on numpy/scipy."""
        return True

    def solve(self, case: ACCase) -> ACBackendResult:
        """Build Ybus from ``case``, run Newton-Raphson, return voltages/flows/losses."""
        A = build_incidence(case.from_bus, case.to_bus, case.n_bus)
        yset = YbusBuilder().build_ac(
            A, case.from_bus, case.to_bus, case.r_pu, case.x_pu, case.b_pu
        )
        ybus = yset.ybus_long

        # NewtonSolver solves v·conj(Y·v) = snet, i.e. snet is the net power
        # *injected into the network* (generation positive, load negative), per
        # unit on s_base.
        s_base = case.s_base_mva
        snet_pu = np.zeros(case.n_bus, dtype=complex)
        np.add.at(snet_pu, case.gen_bus, case.gen_p_mw / s_base)
        np.add.at(snet_pu, case.load_bus, -(case.load_p_mw + 1j * case.load_q_mvar) / s_base)

        v0 = np.ones(case.n_bus, dtype=complex)
        res = NewtonSolver(self.newton_options).solve(
            ybus, v0, snet_pu, np.array([case.slack_bus]), case.gen_bus, case.load_bus
        )

        # Series line flows (YPrim from R/X only), in MVA.
        iline = yset.yprim @ (A.T @ res.v)
        send = s_base * res.v[case.from_bus] * np.conj(iline)
        recv = s_base * res.v[case.to_bus] * np.conj(iline)
        return ACBackendResult(
            backend=self.name,
            converged=res.converged,
            v_bus=res.v,
            flow_from_mva=send,
            flow_to_mva=recv,
            flow_max_mva=np.maximum(np.abs(send), np.abs(recv)),
            losses_mw=float(np.sum(np.real(send - recv))),
        )

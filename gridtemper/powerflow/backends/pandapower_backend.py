"""PandaPower AC backend.

Builds a pandapower network from the :class:`ACCase` — buses (with nominal kV),
one ext_grid at the slack, PV generators, PQ loads, and each branch as a
per-unit series impedance (``create_impedances``) plus the aggregated line
charging (Σ B/2) as bus shunts — then runs ``pp.runpp``.
"""

from __future__ import annotations

import numpy as np

from .base import ACBackendResult, ACCase


class PandaPowerBackend:
    """AC backend that solves via pandapower's ``runpp`` (see module docstring).

    Matches the builtin backend on small/medium cases; the full TX-2030 case may
    not converge (needs the ``[pandapower]`` extra).
    """

    name = "pandapower"

    def __init__(
        self,
        algorithm: str = "nr",
        init: str = "flat",
        min_impedance_pu: float = 1e-3,
        max_iteration: int = 50,
    ):
        self.algorithm = algorithm
        self.init = init
        #: Series |Z| below this (pu) is raised to the floor — very-low-impedance
        #: branches otherwise ill-condition pandapower's Jacobian. 0 disables.
        self.min_impedance_pu = min_impedance_pu
        self.max_iteration = max_iteration

    def available(self) -> bool:
        """``True`` if the optional ``pandapower`` package is importable."""
        try:
            import pandapower  # noqa: F401

            return True
        except ImportError:
            return False

    def solve(self, case: ACCase) -> ACBackendResult:
        """Build a pandapower net from ``case``, run ``runpp``, return the result."""
        import pandapower as pp

        net = pp.create_empty_network(sn_mva=case.s_base_mva)
        pp.create_buses(net, case.n_bus, vn_kv=case.vn_kv)  # indices 0..n_bus-1 == our bus ids
        pp.create_ext_grid(net, bus=int(case.slack_bus), vm_pu=case.v_setpoint)
        if case.gen_bus.size:
            pp.create_gens(net, case.gen_bus, p_mw=case.gen_p_mw, vm_pu=case.v_setpoint)
        if case.load_bus.size:
            pp.create_loads(net, case.load_bus, p_mw=case.load_p_mw, q_mvar=case.load_q_mvar)

        # Series R/X as per-unit impedances (pu on s_base). Regularize very-low
        # |Z| branches, which otherwise ill-condition pandapower's NR Jacobian.
        r_pu = case.r_pu.copy()
        x_pu = case.x_pu.copy()
        if self.min_impedance_pu > 0:
            z = np.abs(r_pu + 1j * x_pu)
            low = z < self.min_impedance_pu
            scale = np.where(low & (z > 0), self.min_impedance_pu / np.maximum(z, 1e-12), 1.0)
            r_pu = r_pu * scale
            x_pu = np.where(low & (z == 0), self.min_impedance_pu, x_pu * scale)
        pp.create_impedances(
            net,
            from_buses=case.from_bus,
            to_buses=case.to_bus,
            rft_pu=r_pu,
            xft_pu=x_pu,
            sn_mva=case.s_base_mva,
        )
        # Line charging Σ B/2 per bus as shunts (positive B is capacitive ->
        # negative q_mvar in pandapower's absorb-positive convention).
        idx = np.concatenate([case.from_bus, case.to_bus])
        b_matrix = np.bincount(
            idx, weights=np.concatenate([case.b_pu / 2, case.b_pu / 2]), minlength=case.n_bus
        )
        shunt_bus = np.flatnonzero(np.abs(b_matrix) > 1e-12)
        if shunt_bus.size:
            pp.create_shunts(
                net, shunt_bus, q_mvar=-b_matrix[shunt_bus] * case.s_base_mva, p_mw=0.0
            )

        try:
            pp.runpp(
                net,
                algorithm=self.algorithm,
                init=self.init,
                max_iteration=self.max_iteration,
                calculate_voltage_angles=True,
            )
            converged = bool(net.converged)
        except Exception:
            converged = False

        if not converged:
            nan_bus = np.full(case.n_bus, np.nan, dtype=complex)
            nan_br = np.full(case.n_branch, np.nan, dtype=complex)
            return ACBackendResult(self.name, False, nan_bus, nan_br, nan_br,
                                   np.full(case.n_branch, np.nan), float("nan"))

        vm = net.res_bus.vm_pu.to_numpy()
        va = np.deg2rad(net.res_bus.va_degree.to_numpy())
        v_bus = vm * np.exp(1j * va)
        ri = net.res_impedance
        send = ri.p_from_mw.to_numpy() + 1j * ri.q_from_mvar.to_numpy()
        recv = ri.p_to_mw.to_numpy() + 1j * ri.q_to_mvar.to_numpy()
        return ACBackendResult(
            backend=self.name,
            converged=True,
            v_bus=v_bus,
            flow_from_mva=send,
            flow_to_mva=recv,
            flow_max_mva=np.maximum(np.abs(send), np.abs(recv)),
            losses_mw=float(ri.pl_mw.sum()),
        )

"""Economic dispatch with reserve-based decommitment — UCED.m.

Algorithm doc: conversion_manual/algorithms/03_dispatch.md §UCED.

Backends (D2): gurobipy primary when importable/licensed; the always-available
fallback is an exact KKT/bisection solver for this separable convex QP
(single equality constraint + bounds) — no extra dependencies.

The QP solved repeatedly (gurobi convention, no ½ factor):

    min  Σ (C_i·P_i² + B_i·P_i)   s.t.  Σ P_i = total_load,  lb ≤ P ≤ MaxMW

Note (logged as **DV12**): MATLAB's ``quadprog`` fallback passes ``H = diag(C)``,
so it minimizes ``½ Σ C·P² + Σ B·P`` — half the quadratic weight gurobi uses —
while the marginal-cost ranking is computed as ``B + 2·C·P`` (the gurobi
convention). GridTemper uses the gurobi objective in *both* backends so the
ranking is self-consistent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import DispatchOptions
from ..grid_model import DispatchResult

try:  # optional primary backend (D2); license checked lazily on first solve
    import gurobipy as _gp

    _HAVE_GUROBI = True
except Exception:  # pragma: no cover - environment without gurobipy
    _HAVE_GUROBI = False

_DROP_TOL = 1e-3  # MATLAB "throw out generators with < 1e-3 dispatch"


class EconomicDispatch:
    """Single-period dispatch QP + greedy decommitment loop.

    Decommitment removes the ``decommit_batch`` (3) highest marginal-cost units
    per step until committed capacity ≤ load·(1+reserve) — full flow (including
    the infeasibility ladders lb=MinMW → lb=0 → lb=1 → MinMW=0) in the algorithm
    doc. The re-solve inside the loop deliberately does **not** refresh the
    setpoints before re-ranking, matching UCED.m (the ``GenMWSetpoint = ...``
    line is commented out there).
    """

    def __init__(self, options: DispatchOptions):
        self.options = options
        self.backend = options.backend
        self._auto = self.backend == "auto"
        if self._auto:
            self.backend = "gurobi" if _HAVE_GUROBI else "fallback"
        if self.backend == "gurobi" and not _HAVE_GUROBI:
            raise RuntimeError("backend='gurobi' but gurobipy is not importable")

    # ------------------------------------------------------------------ public
    def solve(self, gens: pd.DataFrame, total_load_mw: float) -> DispatchResult:
        """Run the full UCED flow; returns the committed fleet + setpoints.

        Parameters
        ----------
        gens : pandas.DataFrame
            Generator cost/limit table with columns ``A``/``B``/``C`` (cost
            coefficients), ``MinMW``/``MaxMW`` (limits), and ``NumberOfBus``.
            Extra columns (e.g. the ``bus`` internal id) are carried through to
            the committed subset.
        total_load_mw : float
            Total system load to serve (MW).

        Returns
        -------
        DispatchResult
            Setpoints aligned to the committed rows, the committed table, the
            realized quadratic objective, decommitment iteration count, and the
            per-solve status history.
        """
        reserve = self.options.reserve
        batch = self.options.decommit_batch
        floor = self.options.min_fleet_size
        d = float(total_load_mw)

        gens = gens.reset_index(drop=True).copy()
        statuses: list[str] = []

        def cols():
            return (
                gens["A"].to_numpy(float),
                gens["B"].to_numpy(float),
                gens["C"].to_numpy(float),
                gens["MinMW"].to_numpy(float),
                gens["MaxMW"].to_numpy(float),
            )

        a, b, c, min_mw, max_mw = cols()
        # lb "kind" tracks MATLAB's single reused `lb` variable: min|zero|one.
        lb_kind = "min"

        def lb_of(kind: str) -> np.ndarray:
            if kind == "zero":
                return np.zeros(len(gens))
            if kind == "one":
                return np.ones(len(gens))
            return gens["MinMW"].to_numpy(float)

        # --- initial solve, lb = MinMW ------------------------------------
        x, status = self._solve_qp(c, b, lb_of(lb_kind), max_mw, d)
        statuses.append(status)
        if x is None:
            x = min_mw.copy()

        if max_mw.sum() > d * (1 + reserve):
            # re-solve with lb = 0, then drop ~zero-dispatch units
            x0, s = self._solve_qp(c, b, lb_of("zero"), max_mw, d)
            statuses.append(s)
            if x0 is not None:
                x = x0
            keep = x >= _DROP_TOL
            gens = gens[keep].reset_index(drop=True)
            x = x[keep]
            a, b, c, min_mw, max_mw = cols()
            lb_kind = "min"
            # re-solve with lb = MinMW
            x1, status = self._solve_qp(c, b, lb_of(lb_kind), max_mw, d)
            statuses.append(status)
            if x1 is not None:
                x = x1
            if status == "infeasible":  # sum(MinMW) > load: relax floor to 1 MW
                lb_kind = "one"
                x2, status = self._solve_qp(c, b, lb_of(lb_kind), max_mw, d)
                statuses.append(status)
                if x2 is not None:
                    x = x2
            highest = self._rank_marginal(b, c, x, batch)

            # --- decommitment loop ----------------------------------------
            iters = 0
            while max_mw.sum() > d * (1 + reserve) and len(gens) > floor:
                if max_mw.sum() - max_mw[highest].sum() <= d:
                    break  # removing the batch would strand the load
                gens = gens.drop(index=highest).reset_index(drop=True)
                x = np.delete(x, highest)
                a, b, c, min_mw, max_mw = cols()
                lb_kind = "one" if status == "infeasible" else "min"
                # NOTE: x is NOT refreshed from this solve (UCED.m line 119 is
                # commented) — the re-ranking below uses the stale setpoints.
                _, status = self._solve_qp(c, b, lb_of(lb_kind), max_mw, d)
                statuses.append(status)
                highest = self._rank_marginal(b, c, x, batch)
                iters += 1
        else:
            iters = 0

        # --- final solve --------------------------------------------------
        a, b, c, min_mw, max_mw = cols()
        xf, status = self._solve_qp(c, b, lb_of(lb_kind), max_mw, d)
        statuses.append(status)
        if xf is not None:
            x = xf
        if status == "infeasible":  # last resort: drop all MinMW floors to 0
            gens = gens.copy()
            gens["MinMW"] = 0.0
            a, b, c, min_mw, max_mw = cols()
            xf, status = self._solve_qp(c, b, lb_of(lb_kind), max_mw, d)
            statuses.append(status)
            if xf is not None:
                x = xf
            keep = x >= _DROP_TOL
            gens = gens[keep].reset_index(drop=True)
            x = x[keep]

        objective = float(np.sum(c * x**2 + b * x)) if len(x) else 0.0
        return DispatchResult(
            setpoints_mw=x,
            committed_gens=gens.reset_index(drop=True),
            objective=objective,
            decommit_iterations=iters,
            solver_statuses=tuple(statuses),
        )

    # ------------------------------------------------------------- internals
    @staticmethod
    def _rank_marginal(b, c, x, k: int) -> np.ndarray:
        """Indices of the ``k`` highest marginal-cost units (B + 2·C·P).

        Mirrors ``maxk(MarginalCost, 3)``: the k largest, order irrelevant for
        removal (we drop them as a set).
        """
        marginal = b + 2.0 * c * x
        k = min(k, marginal.size)
        return np.argsort(marginal)[::-1][:k]

    def _solve_qp(self, c, b, lb, ub, total: float):
        """One QP solve on the selected backend; returns ``(x_or_None, status)``.

        ``x`` is ``None`` when the problem is infeasible (equality unmeetable
        within the bounds), so callers keep the previous setpoints exactly as
        MATLAB does on a gurobi ``INFEASIBLE`` return.
        """
        if self.backend == "gurobi":
            try:
                return self._solve_qp_gurobi(c, b, lb, ub, total)
            except _gp.GurobiError as exc:
                if not self._auto:
                    raise
                # 'auto' with an unusable licence (e.g. the size-limited one
                # bundled with pip gurobipy): use the exact fallback instead.
                warnings.warn(f"gurobipy unusable ({exc}); dispatch uses the "
                              "built-in fallback solver", RuntimeWarning,
                              stacklevel=2)
                self.backend = "fallback"
        return self._solve_qp_fallback(c, b, lb, ub, total)

    @staticmethod
    def _solve_qp_gurobi(c, b, lb, ub, total: float):
        n = len(b)
        m = _gp.Model()
        m.Params.OutputFlag = 0
        x = m.addVars(n, lb=list(lb), ub=list(ub))
        m.setObjective(
            _gp.quicksum(c[i] * x[i] * x[i] + b[i] * x[i] for i in range(n)),
            _gp.GRB.MINIMIZE,
        )
        m.addConstr(_gp.quicksum(x[i] for i in range(n)) == total)
        m.optimize()
        if m.Status in (_gp.GRB.OPTIMAL, _gp.GRB.SUBOPTIMAL):
            return np.array([x[i].X for i in range(n)]), "optimal"
        return None, "infeasible"

    @staticmethod
    def _solve_qp_fallback(c, b, lb, ub, total: float):
        """Exact separable-QP solver by KKT bisection on the equality price λ.

        For ``C_i > 0`` the stationary dispatch is ``P_i(λ) = clip((λ - B_i) /
        (2 C_i), lb_i, ub_i)``; for ``C_i = 0`` it is bang-bang on the sign of
        ``B_i − λ``. ``Σ P_i(λ)`` is non-decreasing in λ, so bisection finds the
        λ meeting ``Σ P = total``. Any residual left by ``C_i = 0`` degeneracy
        is distributed across units with headroom.
        """
        c = np.asarray(c, float)
        b = np.asarray(b, float)
        lb = np.asarray(lb, float)
        ub = np.asarray(ub, float)
        if total > ub.sum() + 1e-6 or total < lb.sum() - 1e-6:
            return None, "infeasible"

        def dispatch(lam: float) -> np.ndarray:
            x = np.empty_like(b)
            quad = c > 0
            x[quad] = np.clip((lam - b[quad]) / (2 * c[quad]), lb[quad], ub[quad])
            lin = ~quad
            x[lin] = np.where(b[lin] < lam, ub[lin], lb[lin])
            return x

        lo = float(np.min(b - 2 * c * np.abs(ub) - 1.0))
        hi = float(np.max(b + 2 * c * np.abs(ub) + 1.0))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if dispatch(mid).sum() < total:
                lo = mid
            else:
                hi = mid
        x = dispatch(0.5 * (lo + hi))
        residual = total - x.sum()
        if abs(residual) > 1e-9:
            if residual > 0:
                room = ub - x
            else:
                room = lb - x  # negative headroom
            total_room = room.sum()
            if abs(total_room) > 1e-12:
                x = x + room * (residual / total_room)
        return x, "optimal"

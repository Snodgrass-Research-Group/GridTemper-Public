"""Newton-Raphson power-flow solver — pfsolve.m + pfmiss.m + pflowjac.m + UpdateVBus.m.

Algorithm doc: conversion_manual/algorithms/05_powerflow_core.md §Newton core.
Bug fix DV9 applies (globals → constructor args + result history).

Sign convention (pfmiss.m): ``snet`` has power LEAVING the bus positive, so
generation is NEGATIVE here. Callers flip signs, not this module.

Everything stays sparse — the full Jacobian at 7132 buses would be 1.6 GB dense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..config import NewtonOptions


@dataclass(frozen=True)
class NewtonResult:
    """Outcome of one Newton-Raphson power-flow solve (pfsolve.m return values)."""

    v: np.ndarray                    # solved complex voltages
    converged: bool
    singular_jacobian: bool          # dx-magnitude heuristic (pfsolve CheckJac)
    iterations: int
    ds_dv: sp.spmatrix | None        # last Jacobian dS/dV (dVdQ input downstream)
    history: list | None             # (mismatch, v) per iteration if requested


def _sub(m: sp.spmatrix, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
    """Sparse submatrix ``m[rows][:, cols]`` (row then column fancy indexing)."""
    return m.tocsr()[rows][:, cols]


class _ReducedSystem:
    """Precompute the reduced-Jacobian assembly so the sparsity *pattern* (the
    symbolic structure) is reused across NR iterations — only the numeric data is
    recomputed each step. The Jacobian dS/dδ and dS/dV keep a fixed pattern (the
    Ybus pattern plus the diagonal) every iteration, so the gather (source data →
    block data) and scatter (blocks → assembled CSC) index maps are constant.

    Two-block form with ``pvpq = [pv, pq]``:
    ``J = [[Re dSdδ[pvpq,pvpq], Re dSdV[pvpq,pq]], [Im dSdδ[pq,pvpq], Im dSdV[pq,pq]]]``.
    """

    def __init__(self, dsdd, dsdv, pv, pq):
        self.pvpq = np.concatenate([pv, pq]).astype(np.int64)
        self.pq = np.asarray(pq, dtype=np.int64)
        # gather indices into each source's canonical-CSR data for the 4 blocks
        self._ga = self._gather(dsdd, self.pvpq, self.pvpq)
        self._gb = self._gather(dsdv, self.pvpq, self.pq)
        self._gc = self._gather(dsdd, self.pq, self.pvpq)
        self._gd = self._gather(dsdv, self.pq, self.pq)
        # learn the assembled CSC pattern + the concat(blocks)->CSC scatter perm
        blocks = [
            self._block(dsdd, self.pvpq, self.pvpq),
            self._block(dsdv, self.pvpq, self.pq),
            self._block(dsdd, self.pq, self.pvpq),
            self._block(dsdv, self.pq, self.pq),
        ]
        self.sizes = [b.nnz for b in blocks]
        probes, off = [], 0
        for b, n in zip(blocks, self.sizes):
            p = b.copy()
            p.data = np.arange(off, off + n, dtype=float)
            probes.append(p)
            off += n
        rjac0 = sp.bmat([[probes[0], probes[1]], [probes[2], probes[3]]], format="csc")
        rjac0.sort_indices()
        self.indptr, self.indices, self.shape = rjac0.indptr, rjac0.indices, rjac0.shape
        self.perm = rjac0.data.astype(np.int64)

    @staticmethod
    def _block(m, rows, cols) -> sp.csc_matrix:
        blk = (m.tocsr()[rows][:, cols]).tocsc()
        blk.sort_indices()
        return blk

    @classmethod
    def _gather(cls, m, rows, cols) -> np.ndarray:
        m = m.tocsr()
        m.sort_indices()
        probe = m.copy()
        probe.data = np.arange(m.nnz, dtype=float)
        blk = cls._block(probe, rows, cols)
        return blk.data.astype(np.int64)

    def assemble(self, dsdd, dsdv) -> sp.csc_matrix:
        """Assemble the reduced Jacobian for this iteration, reusing the cached
        sparsity pattern and gather/scatter maps (only the numeric data is new)."""
        dsdd = dsdd.tocsr()
        dsdd.sort_indices()
        dsdv = dsdv.tocsr()
        dsdv.sort_indices()
        parts = [
            dsdd.data[self._ga].real,
            dsdv.data[self._gb].real,
            dsdd.data[self._gc].imag,
            dsdv.data[self._gd].imag,
        ]
        data = np.concatenate(parts)[self.perm] if self.perm.size else np.zeros(0)
        return sp.csc_matrix((data, self.indices, self.indptr), shape=self.shape)


class NewtonSolver:
    """Full-Jacobian NR with PV/PQ lists supplied per solve (types are imposed
    at solve time, enabling the PVPQSwitching loop to reclassify buses)."""

    def __init__(self, options: NewtonOptions):
        self.options = options

    def solve(
        self,
        ybus: sp.csc_matrix,
        v0: np.ndarray,
        snet_pu: np.ndarray,
        slack: np.ndarray,
        pv: np.ndarray,
        pq: np.ndarray,
    ) -> NewtonResult:
        """pfsolve.m: iterate until max reduced mismatch < tol (5e-5) or max_iter
        (100); the polar update is damped by ``options.damping`` (0.1 — quirk Q2).

        Reduced system (genlist = ``pv``, loadlist = ``pq``; slack excluded):
        variables ``[δ_pv; δ_pq; |V|_pq]``, equations
        ``[Re(miss)_pv; Re(miss)_pq; Im(miss)_pq]``.
        """
        ybus = ybus.tocsc()
        v = np.asarray(v0, dtype=complex).copy()
        pv = np.asarray(pv, dtype=np.int64)
        pq = np.asarray(pq, dtype=np.int64)
        pvpq = np.concatenate([pv, pq])
        tol = self.options.tol
        damping = self.options.damping
        keep_history = self.options.keep_history

        miss = self._mismatch(ybus, v, snet_pu)
        rmiss = np.concatenate([miss[pvpq].real, miss[pq].imag])
        error = float(np.max(np.abs(rmiss))) if rmiss.size else 0.0

        history: list | None = [(miss.copy(), v.copy())] if keep_history else None
        singular = False
        check_jac = True
        ds_dv: sp.spmatrix | None = None
        reduced: _ReducedSystem | None = None  # built once, its pattern reused every iteration
        it = 0
        while error > tol and it < self.options.max_iter:
            dsdd, dsdv = self._jacobian(ybus, v)
            ds_dv = dsdv
            if reduced is None:
                reduced = _ReducedSystem(dsdd, dsdv, pv, pq)
            rjac = reduced.assemble(dsdd, dsdv)
            dx = spla.spsolve(rjac, -rmiss)
            dx = np.atleast_1d(dx)
            if check_jac and dx.size:
                adx = np.abs(dx)
                if adx.max() > 1e25 or adx.min() < 1e-25:
                    singular = True
                    check_jac = False
            v = self._apply_update(v, dx, pv, pq, damping)
            miss = self._mismatch(ybus, v, snet_pu)
            rmiss = np.concatenate([miss[pvpq].real, miss[pq].imag])
            error = float(np.max(np.abs(rmiss))) if rmiss.size else 0.0
            it += 1
            if history is not None:
                history.append((miss.copy(), v.copy()))

        return NewtonResult(v, error <= tol, singular, it, ds_dv, history)

    @staticmethod
    def _mismatch(ybus, v, snet_pu) -> np.ndarray:
        """pfmiss.m: ``v·conj(Y·v) − snet``, full complex vector."""
        ibus = ybus @ v
        return v * np.conj(ibus) - snet_pu

    @staticmethod
    def _jacobian(ybus, v) -> tuple[sp.spmatrix, sp.spmatrix]:
        """pflowjac.m: ``(dS/dδ, dS/dV)``, all buses, sparse throughout."""
        ibus = ybus @ v
        conj_ibus = np.conj(ibus)
        vmag = np.abs(v)
        conj_y = ybus.conj()
        diag_v = sp.diags(v)
        diag_conj_v = sp.diags(np.conj(v))
        dsdd = 1j * sp.diags(conj_ibus * v) - 1j * (diag_v @ conj_y @ diag_conj_v)
        dsdv = sp.diags(conj_ibus * (v / vmag)) + diag_v @ conj_y @ sp.diags(np.conj(v) / vmag)
        return dsdd.tocsr(), dsdv.tocsr()

    @staticmethod
    def _apply_update(v, dx, pv, pq, damping: float) -> np.ndarray:
        """UpdateVBus.m: ``dx = [δ_pv; δ_pq; |V|_pq]``, damped polar update."""
        delta = np.angle(v)
        vmag = np.abs(v)
        npv = pv.size
        npq = pq.size
        delta[pv] += damping * dx[:npv]
        delta[pq] += damping * dx[npv : npv + npq]
        vmag[pq] += damping * dx[npv + npq : npv + 2 * npq]
        return vmag * np.exp(1j * delta)

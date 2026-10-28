"""Perf tests: NR symbolic-structure reuse + scenario parallelism/batching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from gridtemper.powerflow.dc import DCPowerFlow
from gridtemper.powerflow.newton import NewtonSolver, _ReducedSystem, _sub
from gridtemper.powerflow.scenarios import solve_scenarios


def _four_bus_jacobian():
    y = 1.0 / (0.02 + 0.06j)
    dense = np.array([
        [2 * y, -y, -y, 0],
        [-y, 2 * y, 0, -y],
        [-y, 0, 2 * y, -y],
        [0, -y, -y, 2 * y],
    ])
    ybus = sp.csc_matrix(dense)
    v = np.array([1.0, 0.99 - 0.02j, 1.01 + 0.01j, 0.98 - 0.03j])
    return NewtonSolver._jacobian(ybus, v)


def test_reduced_system_matches_bruteforce():
    """The cached reduced-Jacobian assembly equals the direct bmat form."""
    dsdd, dsdv = _four_bus_jacobian()
    pv = np.array([1])
    pq = np.array([2, 3])
    pvpq = np.concatenate([pv, pq])
    got = _ReducedSystem(dsdd, dsdv, pv, pq).assemble(dsdd, dsdv).toarray()
    exp = sp.bmat(
        [
            [_sub(dsdd, pvpq, pvpq).real, _sub(dsdv, pvpq, pq).real],
            [_sub(dsdd, pq, pvpq).imag, _sub(dsdv, pq, pq).imag],
        ]
    ).toarray()
    assert np.allclose(got, exp)


def test_reduced_system_reused_across_data():
    """Pattern built once, reused for new Jacobian data (different v)."""
    dsdd0, dsdv0 = _four_bus_jacobian()
    rs = _ReducedSystem(dsdd0, dsdv0, np.array([1]), np.array([2, 3]))
    # a second Jacobian at a different operating point (same pattern)
    ybus = (dsdd0 != 0).astype(complex)  # reuse the structure
    v2 = np.array([1.0, 0.95 + 0.05j, 1.02 - 0.01j, 0.9 + 0.1j])
    dsdd2, dsdv2 = NewtonSolver._jacobian(sp.csc_matrix(ybus), v2)
    pvpq = np.array([1, 2, 3])
    pq = np.array([2, 3])
    got = rs.assemble(dsdd2, dsdv2).toarray()
    exp = sp.bmat(
        [
            [_sub(dsdd2, pvpq, pvpq).real, _sub(dsdv2, pvpq, pq).real],
            [_sub(dsdd2, pq, pvpq).imag, _sub(dsdv2, pq, pq).imag],
        ]
    ).toarray()
    assert np.allclose(got, exp)


def test_dc_solve_many_matches_single(reference_dir, matlab_csv_dir):
    A = sio.loadmat(reference_dir / "stage6_incidence.mat")["A"].tocsc()
    x = pd.read_csv(matlab_csv_dir / "stage7_branch.csv")["X"].to_numpy(float)
    snet = np.real(sio.loadmat(reference_dir / "stage5_snet.mat")["SNet"].ravel()) / 100.0
    dc = DCPowerFlow()
    single = dc.solve(x, snet.astype(complex), A).flow_send_mva
    multi = dc.solve_many(x, np.column_stack([snet, 2 * snet]).astype(complex), A)
    assert multi.shape == (len(x), 2)
    assert np.allclose(multi[:, 0], single, atol=1e-6)
    assert np.allclose(multi[:, 1], 2 * single, atol=1e-6)   # DC is linear in injection


def _square(x: int) -> int:  # module-level -> picklable for the process pool
    return x * x


def test_solve_scenarios_serial_and_parallel():
    xs = [1, 2, 3, 4, 5]
    assert solve_scenarios(_square, xs, n_workers=1) == [1, 4, 9, 16, 25]
    assert solve_scenarios(_square, xs, n_workers=2) == [1, 4, 9, 16, 25]

"""P2c tests: Ybus + Newton-Raphson + DC/AC drivers.

Dump comparisons (ybus_calc, dc_reference) validate the linear algebra to
machine precision; analytic 2-bus / all-PV cases validate the NR core; the
singularity check is validated against a MATLAB probe (the base case is NOT
singular on the correct incidence — stage8's SingularY=1 is the DV13 bug)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp

from gridtemper.config import NewtonOptions
from gridtemper.powerflow.ac import is_singular
from gridtemper.powerflow.dc import DCPowerFlow
from gridtemper.powerflow.newton import NewtonSolver
from gridtemper.powerflow.ybus import YbusBuilder


@pytest.fixture(scope="module")
def base_case(reference_dir, matlab_csv_dir):
    """Incidence A, branch R/X/B, and SNet for the TX-2030 base case."""
    A = sio.loadmat(reference_dir / "stage6_incidence.mat")["A"].tocsc()
    b = pd.read_csv(matlab_csv_dir / "stage7_branch.csv")
    snet = sio.loadmat(reference_dir / "stage5_snet.mat")["SNet"].ravel()
    return {
        "A": A,
        "from": b["FromInt"].to_numpy() - 1,
        "to": b["ToInt"].to_numpy() - 1,
        "R": b["R"].to_numpy(float),
        "X": b["X"].to_numpy(float),
        "B": b["B"].to_numpy(float),
        "snet": snet,
    }


# ------------------------------------------------------------------- Ybus
def test_build_ac_matches_ybus_calc(base_case, reference_dir):
    yb = YbusBuilder()
    yset = yb.build_ac(base_case["A"], base_case["from"], base_case["to"],
                       base_case["R"], base_case["X"], base_case["B"])
    ref = sio.loadmat(reference_dir / "ybus_calc.mat")
    for name, got in [("YBusShort", yset.ybus_short), ("YBusLong", yset.ybus_long),
                      ("YPrim", yset.yprim), ("BBusShort", yset.bbus_short),
                      ("BBusLong", yset.bbus_long)]:
        diff = abs(got - ref[name].tocsc())
        assert (diff.max() if diff.nnz else 0.0) < 1e-9, name
    assert np.allclose(yset.b_matrix, ref["BMatrix"].ravel())


def test_build_dc_matches(base_case, reference_dir):
    yb = YbusBuilder()
    bbus, _, _ = yb.build_dc(base_case["A"], base_case["X"])
    ref = sio.loadmat(reference_dir / "dc_reference.mat")["BBusDC"].tocsc()
    assert abs(bbus - ref).max() < 1e-9


def test_dc_flows_match(base_case, reference_dir):
    dc = DCPowerFlow()
    res = dc.solve(base_case["X"], np.real(base_case["snet"]) / 100.0, base_case["A"])
    ref = sio.loadmat(reference_dir / "dc_reference.mat")["DCPFlows"].ravel()
    assert res.flow_send_mva.shape == ref.shape
    assert np.max(np.abs(res.flow_send_mva - ref)) < 1e-6


# ----------------------------------------------------------------- Newton
def _two_bus_ybus(z=0.02 + 0.06j):
    y = 1.0 / z
    return sp.csc_matrix(np.array([[y, -y], [-y, y]]))


def test_newton_2bus_converges():
    """Slack + one PQ load. Converges to a power-balanced solution."""
    ybus = _two_bus_ybus()
    snet = np.array([0.0, 0.3 + 0.1j])          # load leaves bus 1 (positive)
    v0 = np.array([1.0 + 0j, 1.0 + 0j])
    for damping in (0.1, 1.0):
        solver = NewtonSolver(NewtonOptions(damping=damping, tol=5e-5, max_iter=100))
        res = solver.solve(ybus, v0, snet, slack=np.array([0]), pv=np.array([], int), pq=np.array([1]))
        assert res.converged, f"damping={damping}"
        # power balance at the PQ bus: v·conj(Y·v) == snet
        s = res.v * np.conj(ybus @ res.v)
        assert abs(s[1] - snet[1]) < 1e-4


def test_newton_all_pv_no_pq():
    """All-PV configuration (loadlist empty) — the base-case setup. The reduced
    Jacobian has only the angle block; the solver must handle npq == 0."""
    ybus = _two_bus_ybus()
    snet = np.array([0.0, -0.2 + 0j])           # real injection at PV bus 1
    v0 = np.array([1.0 + 0j, 1.0 + 0j])
    solver = NewtonSolver(NewtonOptions(damping=1.0, tol=5e-5, max_iter=100))
    res = solver.solve(ybus, v0, snet, slack=np.array([0]), pv=np.array([1]), pq=np.array([], int))
    assert res.converged
    assert np.isclose(np.abs(res.v[1]), 1.0)     # PV magnitude held at setpoint


def test_jacobian_stays_sparse():
    ybus = _two_bus_ybus()
    v = np.array([1.0 + 0j, 0.98 - 0.01j])
    dsdd, dsdv = NewtonSolver._jacobian(ybus, v)
    assert sp.issparse(dsdd) and sp.issparse(dsdv)
    # mismatch matches the definition v·conj(Y·v) − snet
    snet = np.zeros(2, complex)
    miss = NewtonSolver._mismatch(ybus, v, snet)
    assert np.allclose(miss, v * np.conj(ybus @ v))


# ------------------------------------------------------------ singularity
def test_base_case_not_singular(base_case):
    """The correct incidence is NOT singular (ARank=sprank=n_bus, condest≈5e6) —
    matches the MATLAB probe. stage8's SingularY=1 is the DV13 ANet bug, not a
    real property of the network."""
    yb = YbusBuilder()
    yset = yb.build_ac(base_case["A"], base_case["from"], base_case["to"],
                       base_case["R"], base_case["X"], base_case["B"])
    assert is_singular(base_case["A"], yset.ybus_long) is None


def test_singular_on_isolated_bus():
    """An isolated bus (zero incidence row) is detected as singular."""
    # 3 buses, 1 branch between 0 and 1; bus 2 isolated.
    A = sp.csc_matrix(np.array([[1.0], [-1.0], [0.0]]))
    ybus = sp.csc_matrix(np.diag([1.0 + 0j, 1.0 + 0j, 0.0 + 0j]))
    assert is_singular(A, ybus) is not None

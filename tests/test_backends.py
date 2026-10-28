"""Tests for the selectable AC backends (builtin / pandapower / powerworld).

The builtin and pandapower solvers are cross-validated on a small case (they
must agree). PowerWorld needs SimAuto + a case-building-capable install, so its
solve is environment-dependent — only the deterministic aux format is unit
tested. The full-case builtin solve is checked end-to-end via the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from gridtemper.config import CaseConfig, TemperOptions, load_toml
from gridtemper.powerflow.backends import ACCase, get_backend, solve_ac


def _three_bus() -> ACCase:
    # slack(0) + PV gen(1, 100 MW) + PQ load(2, 80+j20). Branches 0-1, 1-2.
    return ACCase(
        n_bus=3, vn_kv=np.array([138.0, 138.0, 138.0]), slack_bus=0,
        from_bus=np.array([0, 1]), to_bus=np.array([1, 2]),
        r_pu=np.array([0.01, 0.02]), x_pu=np.array([0.05, 0.08]), b_pu=np.array([0.02, 0.03]),
        is_transformer=np.array([False, False]),
        gen_bus=np.array([1]), gen_p_mw=np.array([100.0]),
        load_bus=np.array([2]), load_p_mw=np.array([80.0]), load_q_mvar=np.array([20.0]),
        s_base_mva=100.0,
    )


def test_builtin_solves_and_is_physical():
    r = get_backend("builtin").solve(_three_bus())
    assert r.converged
    vm = np.abs(r.v_bus)
    assert vm[0] == pytest.approx(1.0)          # slack held
    assert vm[1] == pytest.approx(1.0)          # PV held
    assert vm[2] < 1.0                           # load bus sags
    assert r.losses_mw > 0                       # positive series losses


def test_builtin_pandapower_agree():
    pp_backend = get_backend("pandapower")
    if not pp_backend.available():
        pytest.skip("pandapower not installed")
    case = _three_bus()
    a = get_backend("builtin").solve(case)
    b = pp_backend.solve(case)
    assert a.converged and b.converged
    assert np.allclose(np.abs(a.v_bus), np.abs(b.v_bus), atol=1e-4)
    assert np.allclose(np.rad2deg(np.angle(a.v_bus)), np.rad2deg(np.angle(b.v_bus)), atol=1e-2)
    assert np.allclose(a.flow_max_mva, b.flow_max_mva, rtol=1e-3)


def test_accase_from_model_partitions_buses(reference_dir, matlab_csv_dir):
    """ACCase.from_model derives a valid slack/PV/PQ partition from the model."""
    import pandas as pd
    import scipy.io as sio

    from gridtemper.grid_model import GridModel

    b7 = pd.read_csv(matlab_csv_dir / "stage7_branch.csv")
    bus4 = pd.read_csv(matlab_csv_dir / "stage4_bus.csv")
    g4 = pd.read_csv(matlab_csv_dir / "stage4_gen.csv")
    snet = sio.loadmat(reference_dir / "stage5_snet.mat")["SNet"].ravel()
    branches = b7.copy()
    branches["from_bus"] = b7["FromInt"].to_numpy() - 1
    branches["to_bus"] = b7["ToInt"].to_numpy() - 1
    gens = pd.DataFrame({"bus": g4["NumberOfBus"].astype(int) - 1, "MaxMW": g4["MaxMW"]})
    model = GridModel(buses=bus4.copy(), branches=branches, gens=gens, gencost=pd.DataFrame())

    case = ACCase.from_model(model, snet, CaseConfig())
    assert case.n_bus == 7132
    assert case.slack_bus == 1746
    assert case.gen_bus.size > 0 and case.load_bus.size > 0
    # slack, PV, PQ are disjoint
    assert case.slack_bus not in case.gen_bus and case.slack_bus not in case.load_bus
    assert np.intersect1d(case.gen_bus, case.load_bus).size == 0


def test_toml_loader(tmp_path):
    p = tmp_path / "gridtemper.toml"
    p.write_text(
        '[solver]\nac_backend = "pandapower"\n[case]\nslack_bus = 100\n'
        '[dispatch]\nreserve = 0.2\n'
    )
    case, options = load_toml(p)
    assert options.ac_backend == "pandapower"
    assert case.slack_bus_internal_matlab == 100
    assert options.dispatch.reserve == pytest.approx(0.2)


def test_ac_backend_validation():
    with pytest.raises(ValueError, match="ac_backend"):
        TemperOptions(ac_backend="nope")
    with pytest.raises(ValueError):
        get_backend("nope")


def test_powerworld_aux_format(tmp_path):
    """The generated aux uses PowerWorld's own concise field names (verified via
    a SaveData round-trip). No SimAuto needed for this unit test."""
    from gridtemper.powerflow.backends.powerworld_backend import PowerWorldBackend

    aux = tmp_path / "case.aux"
    circuits = PowerWorldBackend()._write_aux(_three_bus(), aux)
    text = aux.read_text()
    assert "Bus (Number,Name,NomkV,Slack)" in text
    assert "Branch (BusNumFrom,BusNumTo,Circuit,R,X,B,Status)" in text
    assert '"YES"' in text                       # slack flagged
    assert list(circuits) == [1, 1]              # distinct pairs -> circuit 1


def test_pipeline_ac_solution_builtin(reference_dir):
    """End-to-end: the builtin backend produces a converged AC solution on the
    full TX-2030 case (the MATLAB-faithful path never converges — B6)."""
    from gridtemper.pipeline import GridTemper

    ybus = sp.load_npz(reference_dir / "ybus_new.npz")
    opts = TemperOptions()
    opts.ac_backend = "builtin"
    result = GridTemper(CaseConfig(), opts, reference_ybus=ybus).run()
    ac = result.ac_solution
    assert ac is not None and ac.converged
    vm = np.abs(ac.v_bus)
    assert (vm > 0.8).all() and (vm < 1.2).all()   # physical voltage band
    assert ac.losses_mw > 0


def test_powerworld_open_pwb():
    """Open-existing-.PWB mode: solve a real PowerWorld case via SimAuto.
    Skips unless PowerWorld/SimAuto and a .PWB are available (this dev machine
    has both, but CI won't)."""
    from gridtemper.config import REPO_ROOT
    from gridtemper.powerflow.backends.powerworld_backend import PowerWorldBackend

    pwb = REPO_ROOT / "powerworld_cases" / "Texas7k_2030_20220718.PWB"
    if not pwb.exists():
        pytest.skip("no PowerWorld .PWB case available")
    backend = PowerWorldBackend(pwb_path=str(pwb))
    if not backend.available():
        pytest.skip("PowerWorld SimAuto not available")
    result = backend.solve(_three_bus())  # case is ignored in open mode
    assert result.converged
    vm = np.abs(result.v_bus)
    assert len(vm) > 1000                          # the full case loaded
    assert (vm > 0.8).all() and (vm < 1.2).all()   # physical voltage band


def test_solve_ac_helper_small():
    """solve_ac builds the case + runs the backend in one call (via a tiny model)."""
    import pandas as pd

    from gridtemper.grid_model import GridModel

    buses = pd.DataFrame({"NomKV": [138.0, 138.0, 138.0]})
    branches = pd.DataFrame({
        "from_bus": [0, 1], "to_bus": [1, 2],
        "R": [0.01, 0.02], "X": [0.05, 0.08], "B": [0.0, 0.0], "Tap": [0, 0],
    })
    gens = pd.DataFrame({"bus": [1], "MaxMW": [100.0]})
    demand = pd.DataFrame({"bus": [2], "S_pu": [80.0 + 20.0j]})
    model = GridModel(buses=buses, branches=branches, gens=gens, gencost=pd.DataFrame(), demand=demand)
    snet = np.array([-30.0, 50.0, -80.0], dtype=complex)  # gen 50 at bus1, load 80 at bus2
    r = solve_ac(model, snet, CaseConfig(slack_bus_internal_matlab=1), backend="builtin")
    assert r.converged

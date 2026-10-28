"""P3 tests: reactive placement (active path vs stage8) + dormant refinement.

KEY CONTEXT: today's MATLAB final AC stage is non-functional — the simple solve
does not converge (stage8_branch.MaxACPF is 100% NaN) and the detailed solve is
spuriously singular (the DV13 ANet bug). So the active-path validation is the
deterministic setup (bus types, VBusSet, B clamp, LoadList) plus faithful
reproduction of the non-convergence. DV13 is fixed, so our detailed solve is no
longer singular (validated in P2c)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp

from gridtemper.config import CaseConfig, TemperOptions
from gridtemper.grid_model import GridModel
from gridtemper.refinement.reactive import ReactivePlacement, SwitchState


@pytest.fixture(scope="module")
def base_model(reference_dir, matlab_csv_dir):
    """Base-case GridModel + incidence + snet + YBus_new, from the dumps."""
    A = sio.loadmat(reference_dir / "stage6_incidence.mat")["A"].tocsc()
    ybus_new = sp.load_npz(reference_dir / "ybus_new.npz")
    b7 = pd.read_csv(matlab_csv_dir / "stage7_branch.csv")
    g4 = pd.read_csv(matlab_csv_dir / "stage4_gen.csv")
    bus4 = pd.read_csv(matlab_csv_dir / "stage4_bus.csv")
    snet = sio.loadmat(reference_dir / "stage5_snet.mat")["SNet"].ravel()
    branches = b7.copy()
    branches["from_bus"] = b7["FromInt"].to_numpy() - 1
    branches["to_bus"] = b7["ToInt"].to_numpy() - 1
    gens = pd.DataFrame({"bus": g4["NumberOfBus"].astype(int) - 1, "MaxMW": g4["MaxMW"]})
    model = GridModel(buses=bus4.copy(), branches=branches, gens=gens, gencost=pd.DataFrame())
    return {"model": model, "A": A, "snet": snet, "ybus_new": ybus_new}


# ------------------------------------------------- ReactivePlacement (active)
def test_reactive_setup_matches_stage8(base_model, reference_dir):
    rp = ReactivePlacement(CaseConfig(), TemperOptions(), reference_ybus=base_model["ybus_new"])
    out, _ = rp.run(base_model["model"], base_model["A"], base_model["snet"], run_detailed=False)
    acpf = sio.loadmat(reference_dir / "stage8_acpf.mat")

    bus_type = out.buses["bus_type"].to_numpy()
    assert int((bus_type == 2).sum()) == acpf["GenList"].size == 7130   # PV buses
    assert acpf["LoadList"].size == 0                                   # all-PV start
    assert int((bus_type == 3).sum()) == 2                              # slack + relocated slack
    assert np.allclose(acpf["VBusSet"], 1.0)                            # flat start
    assert out.branches["B"].to_numpy().max() <= 1.0                    # B clamp


def test_reactive_reproduces_nonconvergence(base_model, matlab_csv_dir):
    """The base-case simple AC solve does not converge -> MaxACPF is NaN, exactly
    like stage8 (the whole final AC stage is non-functional today; B6)."""
    rp = ReactivePlacement(CaseConfig(), TemperOptions(), reference_ybus=base_model["ybus_new"])
    out, simple = rp.run(base_model["model"], base_model["A"], base_model["snet"], run_detailed=False)
    assert not simple.converged
    assert np.all(np.isnan(out.branches["MaxACPF"].to_numpy()))
    ref = pd.read_csv(matlab_csv_dir / "stage8_branch.csv")["MaxACPF"].to_numpy()
    assert np.all(np.isnan(ref))                                        # stage8 is NaN too


def test_reactive_detailed_not_singular_dv13(base_model):
    """DV13 fixed: with the correct incidence the detailed solve is NOT singular,
    so it actually runs (MATLAB's SingularY=1 came from the scrambled ANet)."""
    rp = ReactivePlacement(CaseConfig(), TemperOptions(), reference_ybus=base_model["ybus_new"])
    out, _ = rp.run(base_model["model"], base_model["A"], base_model["snet"], run_detailed=True)
    assert isinstance(out.log["acpf_detailed"], dict)                   # ran (not "singular: ...")
    assert "converged" in out.log["acpf_detailed"]


# ------------------------------------------------- PVPQSwitching (dormant)
def test_pv_pq_switching_partition_and_clamp():
    n = 6
    s_base = 100.0
    # bus 2 solved Q over the high limit, bus 3 under the low limit.
    snet_calc = np.zeros(n, complex)
    snet_calc[2] = 900j
    snet_calc[3] = -1300j
    state = SwitchState(
        n_bus=n,
        snet=np.zeros((n, 1), complex),
        v=np.ones(n, complex),
        snet_calc=snet_calc,
        v_set=np.ones(n, complex),
        gen_list=np.array([1, 2, 3, 4, 5]),
        load_list=np.array([], dtype=int),
        slack=np.array([0]),
        high_q=800.0,
        low_q=-1200.0,
    )
    snet_complete = np.zeros((n, 1))       # real MW schedule (all zero here)
    out = ReactivePlacement._pv_pq_switching(state, snet_complete, s_base, 0, resolve=lambda s: s)

    # over/under-Q buses demoted to PQ; the three lists partition all buses.
    assert set([2, 3]).issubset(set(out.load_list.tolist()))
    assert 2 not in out.gen_list and 3 not in out.gen_list
    union = np.union1d(np.union1d(out.gen_list, out.load_list), out.slack)
    assert np.array_equal(np.sort(union), np.arange(n))                 # partition covers all
    assert np.intersect1d(out.gen_list, out.load_list).size == 0        # disjoint
    # clamped Q at the limits (per unit)
    assert out.snet[2, 0].imag == pytest.approx(800.0 / s_base)
    assert out.snet[3, 0].imag == pytest.approx(-1200.0 / s_base)


# ------------------------------------------------- RealPowerLoop (dormant)
def test_real_power_loop_smoke():
    """Structural smoke test on a synthetic 3-bus / 2-line case (no transformers):
    the loop resizes overloaded lines, re-conductors, and terminates."""
    from gridtemper.conductors.library import ConductorLibrary
    from gridtemper.refinement.real_power import RealPowerLoop
    from gridtemper.topology.incidence import build_incidence

    lib = ConductorLibrary.from_reference([69, 138, 345])
    A = build_incidence(np.array([0, 1]), np.array([1, 2]), 3)
    branches = pd.DataFrame(
        {
            "Tap": [0, 0], "Resize": [1, 1], "Circuit": [1, 1], "circuits": [1.0, 1.0],
            "FromNomKV": [138.0, 138.0], "ToNomKV": [138.0, 138.0],
            "LimMVAA": [50.0, 50.0], "cond_index": [-1, -1],
            "MaxCondIndex": [40, 40], "MinCondIndex": [0, 0],
            "gmd_index": [500, 500], "length_km": [10.0, 12.0],
            "R": [0.0, 0.0], "X": [0.01, 0.01], "B": [0.0, 0.0], "Undersized": [0, 0],
        }
    )
    model = GridModel(buses=pd.DataFrame({"i": range(3)}), branches=branches,
                      gens=pd.DataFrame(), gencost=pd.DataFrame())
    snet = np.array([2.0 + 0j, 0.0 + 0j, -2.0 + 0j])   # 200 MW across the path
    opts = TemperOptions()
    opts.resize.random_resize = False
    out = RealPowerLoop(CaseConfig(), opts, lib).run(model, A, snet)
    assert "cond_index" in out.branches.columns
    assert (out.branches["cond_index"].to_numpy() >= 0).all()          # conductors assigned
    assert len(out.branches) == 2

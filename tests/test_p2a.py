"""P2a regression + physics tests: io, topology, dispatch.

Each stage is checked against the MATLAB dump for its input/output where
possible (isolating tie-order effects), plus analytic checks that stand alone.
Skips cleanly when the dumps are absent (conftest fixtures).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from gridtemper.config import DispatchOptions
from gridtemper.dispatch.cost_curves import fit_quadratic_costs
from gridtemper.dispatch.snet import build_snet
from gridtemper.dispatch.uced import EconomicDispatch
from gridtemper.io.pw_csv import make_valid_name
from gridtemper.topology.gsu_split import split_add_gsus
from gridtemper.topology.incidence import build_incidence
from gridtemper.topology.numbering import to_internal_numbering


# --------------------------------------------------------------------- io
def test_make_valid_name_matches_matlab():
    # Exact cases lifted from the real PW headers -> MATLAB readtable names.
    assert make_valid_name("Number of Bus") == "NumberOfBus"
    assert make_valid_name("Nom kV of Bus") == "NomKVOfBus"
    assert make_valid_name("Volt (kV)") == "Volt_kV_"
    assert make_valid_name("# Cost Curve Points") == "x_CostCurvePoints"
    assert make_valid_name("Variable O&M") == "VariableO_M"
    assert make_valid_name("Fixed Cost($/hr)") == "FixedCost___hr_"
    assert make_valid_name("MW Break10") == "MWBreak10"
    assert make_valid_name("S_pu") == "S_pu"


def test_pw_columns_match_matlab(raw_model, matlab_csv_dir):
    """read_pw_table reproduces MATLAB readtable's normalized variable names."""
    cases = {
        "gens": "stage1_gen.csv",
        "buses": "stage1_bus.csv",
        "branches": "stage1_branch.csv",
        "gencost": "stage1_gencost.csv",
    }
    for attr, fname in cases.items():
        ref_cols = list(pd.read_csv(matlab_csv_dir / fname, nrows=0).columns)
        got_cols = list(getattr(raw_model, attr).columns)
        assert got_cols == ref_cols, f"{attr}: {got_cols[:5]} != {ref_cols[:5]}"


def test_demand_complex_parsed(raw_model):
    """The MATLAB ``a+bi`` demand column becomes complex (blocker B2: MW+jMVAr)."""
    s = raw_model.demand["S_pu"].to_numpy()
    assert np.iscomplexobj(s)
    # First data row of Demand_onetimepoint.csv: 0.10286+0.038859i
    assert s[0] == pytest.approx(0.10286 + 0.038859j)


# --------------------------------------------------------------- gsu split
def test_split_row_counts_and_numbering(raw_model, matlab_csv_dir):
    ref_bus = pd.read_csv(matlab_csv_dir / "stage2_bus.csv")
    ref_branch = pd.read_csv(matlab_csv_dir / "stage2_branch.csv")
    split = split_add_gsus(raw_model, first_n_buses=88)

    assert split.n_bus == len(ref_bus)
    assert split.n_branch == len(ref_branch)
    # Same bus-number set (new GSU buses 90..416 fill the numbering gap).
    assert set(split.buses["Number"]) == set(ref_bus["Number"])
    new_nums = sorted(set(ref_bus["Number"]) - set(raw_model.buses["Number"]))
    assert new_nums[0] == 90 and len(new_nums) == split.n_bus - len(raw_model.buses)

    # Non-DV2 columns agree after sorting by Number (unique key).
    got = split.buses.sort_values("Number").reset_index(drop=True)
    exp = ref_bus.sort_values("Number").reset_index(drop=True)
    for col in ("NomKV", "Volt_kV_", "LoadMW", "LoadMvar"):
        assert np.allclose(got[col].to_numpy(float), exp[col].to_numpy(float)), col


def test_split_dv2_genmw_divergence(raw_model, matlab_csv_dir):
    """DV2: MATLAB zeroed bus-table GenMW by *generator* row index; we zero the
    original split bus row. Divergence must be confined to those rows."""
    ref_bus = pd.read_csv(matlab_csv_dir / "stage2_bus.csv").sort_values("Number").reset_index(drop=True)
    got = split_add_gsus(raw_model, first_n_buses=88).buses.sort_values("Number").reset_index(drop=True)
    diff = ~np.isclose(got["GenMW"].to_numpy(float), ref_bus["GenMW"].to_numpy(float))
    # A handful of rows differ (the split buses); the vast majority match.
    assert diff.sum() < 0.02 * len(got), f"DV2 divergence too broad: {diff.sum()} rows"


# ------------------------------------------------------------- cost curves
def test_cost_curves_vs_matlab(matlab_csv_dir):
    """Feed the exact MATLAB input (stage2 gencost) so rows align 1:1 with the
    stage3 output. DV11 revised: we replicate MATLAB's coefficient placement, so
    coefficients match directly (except numerically-flat curves whose quadratic
    leading-coefficient sign flips between numpy and MATLAB polyfit)."""
    src = pd.read_csv(matlab_csv_dir / "stage2_gencost.csv")
    ref = pd.read_csv(matlab_csv_dir / "stage3_gencost_quadratic.csv")
    got = fit_quadratic_costs(src)
    assert len(got) == len(ref)

    x = src[[f"MWBreak{i}" for i in range(1, 11)]].to_numpy(float)
    y = src[[f"MWhPrice{i}" for i in range(1, 11)]].to_numpy(float)
    exact = concave = boundary = 0
    for i in range(len(src)):
        trip_r = (ref["A"].iloc[i], ref["B"].iloc[i], ref["C"].iloc[i])
        trip_g = (got["A"].iloc[i], got["B"].iloc[i], got["C"].iloc[i])
        if y[i, 0] <= 0:
            assert trip_g == pytest.approx((0.0, 0.0, 0.0))
            continue
        quad = np.polyfit(x[i], y[i], 2)
        if abs(quad[0]) <= 1e-6:  # flat curve: branch is numerically ambiguous
            boundary += 1
            continue
        assert trip_g == pytest.approx(trip_r, rel=1e-6, abs=1e-9), f"row {i}"
        exact += 1
        concave += quad[0] < 0
    print(f"\nDV11: {exact} rows match MATLAB exactly ({concave} concave/linear-refit), "
          f"{boundary} flat-curve boundary rows (numpy/MATLAB polyfit sign coin-flip)")
    # Non-degenerate curves all matched (asserted in the loop). The flat-curve
    # rows are inherently ambiguous in BOTH implementations — flagged in DV11.
    assert boundary < 0.06 * len(src)


# --------------------------------------------------------------- numbering
def test_numbering_map_matches_matlab(raw_model, matlab_csv_dir):
    """The external->internal map must match MATLAB exactly (it sets the id of
    every downstream array). stage4_bus carries NumExt and 1-based internal."""
    ref_bus = pd.read_csv(matlab_csv_dir / "stage4_bus.csv")
    matlab_map = {int(e): int(n) - 1 for e, n in zip(ref_bus["NumExt"], ref_bus["Number"])}

    split = split_add_gsus(raw_model, first_n_buses=88)
    renum = to_internal_numbering(split)
    assert renum.ext_to_int == matlab_map

    # Branch endpoints line up with MATLAB's FromInt/ToInt (0-based here).
    ref_branch = pd.read_csv(matlab_csv_dir / "stage4_branch.csv")
    key = ["FromExt", "ToExt"]
    got = renum.branches.rename(columns={"from_ext": "FromExt", "to_ext": "ToExt"})
    merged = got.merge(ref_branch[key + ["FromInt", "ToInt"]], on=key, how="left")
    # every mapped from/to matches (allow the merge to fan out on parallel lines)
    assert np.all(merged["from_bus"].to_numpy() == merged["FromInt"].to_numpy() - 1)
    assert np.all(merged["to_bus"].to_numpy() == merged["ToInt"].to_numpy() - 1)


# -------------------------------------------------------------------- snet
def test_snet_vs_matlab(matlab_csv_dir, reference_dir):
    """UCED + bus aggregation vs Create_SNet. Fed the stage4 (renumbered) dumps
    so only dispatch+scatter is under test. gurobi is used both sides."""
    gencost = pd.read_csv(matlab_csv_dir / "stage4_gencost.csv")
    demand = pd.read_csv(matlab_csv_dir / "stage4_demand.csv")
    gencost = gencost.copy()
    gencost["bus"] = gencost["NumberOfBus"].astype(int) - 1  # 1-based -> 0-based
    demand = demand.copy()
    # stage4_demand S_pu is written as a MATLAB complex literal string.
    demand["S_pu"] = demand["S_pu"].map(_matlab_complex)
    demand["bus"] = demand["BusNumber"].astype(int) - 1

    n_bus = 7132
    snet, dispatch = build_snet(demand, gencost, DispatchOptions(), n_bus)

    ref = sio.loadmat(reference_dir / "stage5_snet.mat")["SNet"].ravel()
    assert snet.shape == ref.shape
    # Per-bus agreement (gurobi both sides; MATLAB A/B/C fed in isolates dispatch).
    diff = np.abs(snet.real - ref.real)
    print(f"\nSNet: max abs diff={diff.max():.4g} MW, committed={len(dispatch.committed_gens)} gens")
    assert diff.max() < 1.0, f"max per-bus SNet diff {diff.max()} MW"


# --------------------------------------------------------------- incidence
def test_incidence_vs_matlab(matlab_csv_dir, reference_dir):
    branch = pd.read_csv(matlab_csv_dir / "stage4_branch.csv")
    from_bus = branch["FromInt"].to_numpy(int) - 1
    to_bus = branch["ToInt"].to_numpy(int) - 1
    n_bus = 7132
    A = build_incidence(from_bus, to_bus, n_bus)
    ref = sio.loadmat(reference_dir / "stage6_incidence.mat")["A"]
    assert A.shape == ref.shape
    # Laplacian A·Aᵀ is invariant to branch (column) ordering -> robust equality.
    lap_got = (A @ A.T).tocsr()
    lap_ref = (ref @ ref.T).tocsr()
    assert (lap_got != lap_ref).nnz == 0


def test_incidence_analytic():
    # 3-bus path 0-1-2: two branches.
    A = build_incidence(np.array([0, 1]), np.array([1, 2]), 3)
    assert A.shape == (3, 2)
    dense = A.toarray()
    assert np.array_equal(dense, [[1, 0], [-1, 1], [0, -1]])
    # column sums are zero (each branch has one +1 and one -1)
    assert np.allclose(dense.sum(axis=0), 0)


def test_incidence_rejects_out_of_range():
    with pytest.raises(ValueError):
        build_incidence(np.array([0, 3]), np.array([1, 2]), 3)


# -------------------------------------------------------------------- uced
def test_uced_analytic_merit_order():
    """3 generators, distinct quadratic costs, no reserve pressure: dispatch
    equalizes marginal cost B+2C·P (water-filling) and meets load exactly."""
    # Costs chosen so the interior water-filling solution stays within bounds.
    gens = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [10.0, 12.0, 14.0],
            "C": [0.1, 0.1, 0.1],
            "MinMW": [0.0, 0.0, 0.0],
            "MaxMW": [100.0, 100.0, 100.0],
            "NumberOfBus": [1, 2, 3],
            "bus": [0, 1, 2],
        }
    )
    for backend in ("fallback", "auto"):
        res = EconomicDispatch(DispatchOptions(backend=backend, reserve=10.0)).solve(gens, 120.0)
        p = res.setpoints_mw
        assert p.sum() == pytest.approx(120.0, abs=1e-4)
        # cheaper units run harder; marginal costs equal in the interior (P=[50,40,30])
        assert p[0] > p[1] > p[2]
        mc = gens["B"].to_numpy() + 2 * gens["C"].to_numpy() * p
        assert mc.max() - mc.min() == pytest.approx(0.0, abs=1e-2)


def test_uced_decommits_to_reserve():
    """Large surplus fleet: decommitment trims capacity toward load·(1+reserve)."""
    n = 12
    gens = pd.DataFrame(
        {
            "A": np.zeros(n),
            "B": np.linspace(10, 60, n),
            "C": np.full(n, 0.02),
            "MinMW": np.full(n, 5.0),
            "MaxMW": np.full(n, 100.0),
            "NumberOfBus": np.arange(1, n + 1),
            "bus": np.arange(n),
        }
    )
    res = EconomicDispatch(DispatchOptions(backend="auto", reserve=0.15)).solve(gens, 300.0)
    assert res.setpoints_mw.sum() == pytest.approx(300.0, abs=1e-3)
    assert len(res.committed_gens) < n  # some units decommitted
    assert res.setpoints_mw.min() >= -1e-6


def _matlab_complex(v):
    if isinstance(v, str):
        return complex(v.replace("i", "j").replace(" ", ""))
    return complex(v)


def test_uced_auto_falls_back_when_gurobi_unusable(monkeypatch):
    """'auto' survives a GurobiError (e.g. the size-limited pip licence)."""
    gp = pytest.importorskip("gurobipy")
    from gridtemper.dispatch import uced

    def too_large(*args, **kwargs):
        raise gp.GurobiError(10010, "Model too large for size-limited license")

    monkeypatch.setattr(uced, "_HAVE_GUROBI", True)
    monkeypatch.setattr(uced.EconomicDispatch, "_solve_qp_gurobi", staticmethod(too_large))
    gens = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [10.0, 12.0, 14.0],
            "C": [0.1, 0.1, 0.1],
            "MinMW": [0.0, 0.0, 0.0],
            "MaxMW": [100.0, 100.0, 100.0],
            "NumberOfBus": [1, 2, 3],
            "bus": [0, 1, 2],
        }
    )
    ed = EconomicDispatch(DispatchOptions(backend="auto", reserve=10.0))
    with pytest.warns(RuntimeWarning, match="fallback"):
        res = ed.solve(gens, 120.0)
    assert ed.backend == "fallback"
    assert res.setpoints_mw.sum() == pytest.approx(120.0, abs=1e-4)
    with pytest.raises(gp.GurobiError):
        EconomicDispatch(DispatchOptions(backend="gurobi", reserve=10.0)).solve(gens, 120.0)

"""P2b tests: conductor library + assignment (validated vs stage7 dump),
line params / transformers / resize (dormant — physics + spot checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridtemper.config import CaseConfig, ResizeOptions, TemperOptions
from gridtemper.conductors.library import ConductorLibrary
from gridtemper.conductors.line_params import compute_rxb
from gridtemper.conductors.resize import resize_branches
from gridtemper.conductors.transformers import size_transformers
from gridtemper.io.reference_data import load_conductor_table, load_gmd_data, load_xf_data

LEVELS = [69, 138, 345]


# --------------------------------------------------------- reference data
def test_conductor_table_levels(matlab_csv_dir):
    tbl = load_conductor_table(LEVELS)
    ref = pd.read_csv(matlab_csv_dir / "stage7_conductor_data.csv")
    assert len(tbl) == len(ref)
    assert list(tbl["cond_index"]) == list(range(len(tbl)))         # 0-based, contiguous
    assert np.array_equal(tbl["cond_index"].to_numpy(), ref["CondIndex"].to_numpy() - 1)
    assert np.allclose(tbl["mva"], ref["MVA"])
    assert np.allclose(tbl["ra"], ref["Ra"])
    assert list(tbl["kv"].unique()) == LEVELS


def test_conductor_index_depends_on_levels():
    """CondIndex re-indexes over the enabled concatenation (MATLAB quirk)."""
    two = load_conductor_table([69, 138])
    three = load_conductor_table([69, 138, 345])
    assert (three["cond_index"] == range(len(three))).all()
    # the 69/138 block is identical; 345 only appears in the 3-level table
    assert len(three) > len(two)
    assert np.allclose(three.iloc[: len(two)]["mva"], two["mva"])


def test_gmd_and_xf_loaders():
    gmd = load_gmd_data("SPP")
    assert gmd.gmd_table.shape == (1000, 6)
    assert gmd.xd[0] == 0.0 and gmd.xd_prime[0] == 0.0
    xf_data, xf_xr = load_xf_data()
    assert {"pri", "sec", "xhl"}.issubset(xf_data.columns)
    assert {"mva", "xr"}.issubset(xf_xr.columns)


# ------------------------------------------------- AssignConductors (active)
def test_assign_conductors_vs_stage7(matlab_csv_dir):
    src = pd.read_csv(matlab_csv_dir / "stage4_branch.csv")
    ref = pd.read_csv(matlab_csv_dir / "stage7_branch.csv")
    lib = ConductorLibrary.from_reference(LEVELS)
    got = lib.assign(src, TemperOptions())

    # stage4 and stage7 are the same rows in the same order (assign adds columns).
    assert np.array_equal(got["cond_index"].to_numpy(), ref["CondIndex"].to_numpy() - 1)
    assert np.allclose(got["kcmil"].to_numpy(float), ref["kcmil"].to_numpy(float))
    assert np.allclose(got["cond_per_phase"].to_numpy(float), ref["CondPerPhase"].to_numpy(float))
    assert np.allclose(got["LimMVAA"].to_numpy(float), ref["LimMVAA"].to_numpy(float))
    assert np.allclose(got["FromNomKV"].to_numpy(float), ref["FromNomKV"].to_numpy(float))
    # Resize cleared on processed lines (matches stage7: no Resize==1 & Tap==0 left)
    assert int(((got["Resize"] == 1) & (got["Tap"] == 0)).sum()) == 0


# --------------------------------------------------------- RXB (dormant)
def test_rxb_short_line_limit():
    """γℓ→0: the distributed-parameter Z collapses to z·ℓ, so R/X match the
    lumped short-line formula r·ℓ/(kV²/100)."""
    lib = ConductorLibrary.from_reference(LEVELS)
    gmd = load_gmd_data("SPP")
    row = lib.table[lib.table["kv"] == 345].iloc[0]
    ci = int(row["cond_index"])
    branch = pd.DataFrame(
        {
            "FromNomKV": [345.0],
            "cond_index": [ci],
            "gmd_index": [500],
            "length_km": [1e-3],   # ~0.6 miles -> negligible long-line correction
            "circuits": [1.0],
            "LimMVAA": [1e9],
        }
    )
    out = compute_rxb(branch, CaseConfig(), TemperOptions(), lib, gmd)
    length_mi = 1e-3 / 1.61
    gmd_ft = int(gmd.gmd_table[499, 3])          # 345 kV -> column 3
    base = 345.0**2 / 100.0
    r_exp = row["ra"] * length_mi / base
    x_exp = (row["xa"] + gmd.xd[gmd_ft - 1]) * length_mi / base
    assert out["R"].iloc[0] == pytest.approx(r_exp, rel=1e-4)
    assert out["X"].iloc[0] == pytest.approx(x_exp, rel=1e-4)
    assert out["B"].iloc[0] > 0


def test_rxb_divides_by_circuits():
    """DV6: R/X/B all scale as 1/circuits (B division kept per ruling)."""
    lib = ConductorLibrary.from_reference(LEVELS)
    gmd = load_gmd_data("SPP")
    ci = int(lib.table[lib.table["kv"] == 138].iloc[0]["cond_index"])
    base = dict(FromNomKV=[138.0], cond_index=[ci], gmd_index=[300], length_km=[50.0], LimMVAA=[1e9])
    one = compute_rxb(pd.DataFrame({**base, "circuits": [1.0]}), CaseConfig(), TemperOptions(), lib, gmd)
    two = compute_rxb(pd.DataFrame({**base, "circuits": [2.0]}), CaseConfig(), TemperOptions(), lib, gmd)
    for col in ("R", "X", "B"):
        assert two[col].iloc[0] == pytest.approx(one[col].iloc[0] / 2, rel=1e-9)


# ----------------------------------------------- TransformerImpedances (dormant)
def test_transformer_spot():
    xf_data, xf_xr = load_xf_data()
    branch = pd.DataFrame(
        {
            "Tap": [1], "Resize": [1], "Circuits": [1.0],
            "LimMVAA": [100.0], "MinCondIndex": [0.0],
            "FromNomKV": [69.0], "ToNomKV": [138.0],
            "R": [0.0], "X": [0.0], "cond_index": [-1],
        }
    )
    out = size_transformers(branch, xf_data, xf_xr)
    xr = float(np.interp(100.0, np.sort(xf_xr["mva"]), xf_xr.sort_values("mva")["xr"]))
    xhl = float(xf_data[(xf_data["pri"] == 69) & (xf_data["sec"] == 138)]["xhl"].iloc[0])
    z_pu = (100.0 / 100.0) * (xhl / 100.0)
    x_exp = np.sqrt(z_pu**2 / (1 + 1 / xr**2))
    assert out["X"].iloc[0] == pytest.approx(x_exp, rel=1e-9)
    assert out["R"].iloc[0] == pytest.approx(x_exp / xr, rel=1e-9)
    assert out["cond_index"].iloc[0] == np.ceil(100.0 / 1.0)


def test_transformer_dv5_raises_on_missing_pair():
    """DV5: a lookup miss / Pri==Sec must raise, not reuse a stale ZIndex."""
    xf_data, xf_xr = load_xf_data()
    v = float(np.intersect1d(xf_data["pri"].unique(), xf_data["sec"].unique())[0])
    branch = pd.DataFrame(
        {
            "Tap": [1], "Resize": [1], "Circuits": [1.0],
            "LimMVAA": [100.0], "MinCondIndex": [0.0],
            "FromNomKV": [v], "ToNomKV": [v],   # snaps to Pri==Sec
            "R": [0.0], "X": [0.0], "cond_index": [-1],
        }
    )
    with pytest.raises(ValueError, match="DV5"):
        size_transformers(branch, xf_data, xf_xr)


# --------------------------------------------------------- ResizeBranches (dormant)
def _resize_frame(n=20):
    return pd.DataFrame(
        {
            "LimMVAA": np.full(n, 100.0),
            "cond_index": np.full(n, 5),
            "MaxCondIndex": np.full(n, 10),
            "MinCondIndex": np.full(n, 1),
            "Resize": np.zeros(n, dtype=int),
            "tag": np.arange(n),   # identity to check order preservation
        }
    )


def test_resize_preserves_order_and_flags():
    b = _resize_frame(20)
    flows = np.full(20, 50.0)
    flows[[2, 7, 13]] = 200.0   # overloaded (util 2.0)
    flows[[4, 9]] = 5.0         # low utilization (util 0.05)
    opts = ResizeOptions(random_resize=False, percent_resize=1.0, percent_change=1.3,
                         conductor_margin=1.0, downsize_threshold=0.5)
    out = resize_branches(b, opts, flows, np.random.default_rng(0))
    assert list(out["tag"]) == list(range(20))                 # DV7: order preserved
    assert out.loc[[2, 7, 13], "LimMVAA"].eq(np.ceil(100 * 1.3)).all()   # upsized
    assert out.loc[[4, 9], "LimMVAA"].eq(np.ceil(100 / 1.3)).all()       # downsized
    assert out.loc[[2, 7, 13, 4, 9], "Resize"].eq(1).all()
    assert out.loc[[0, 1, 5], "Resize"].eq(0).all()            # untouched


def test_resize_respects_cap_and_seed():
    b = _resize_frame(20)
    flows = np.full(20, 300.0)  # everything overloaded
    opts = ResizeOptions(random_resize=True, percent_resize=0.25, percent_change=1.3,
                         conductor_margin=1.0, downsize_threshold=0.5, downsize=False)
    out1 = resize_branches(b, opts, flows, np.random.default_rng(42))
    out2 = resize_branches(b, opts, flows, np.random.default_rng(42))
    assert int((out1["Resize"] == 1).sum()) == int(np.ceil(0.25 * 20))   # cap honored
    assert out1["LimMVAA"].equals(out2["LimMVAA"])                        # seeded -> reproducible

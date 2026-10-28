"""Validate the outputs of scripts/convert_mat_data.py against values read
directly from the MATLAB sources (spot checks, not full dumps — these do not
need dump_reference.m)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

DATA = Path(__file__).parents[1] / "gridtemper" / "data"
REF = Path(__file__).parent / "reference"


def test_conductor_table():
    df = pd.read_parquet(DATA / "conductors.parquet")
    assert sorted(df["kv"].unique()) == [69, 138, 230, 345, 500, 765]
    assert len(df) == 148
    # First row of ConductorData69.mat: 24 MVA, 69 kV, 80 kcmil, 1/phase,
    # Ra=1.114, Xa=0.553, XaP=0.124 (read via scipy during conversion dev).
    r0 = df[df["kv"] == 69].iloc[0]
    assert r0["mva"] == 24 and r0["size_kcmil"] == 80
    assert r0["ra"] == pytest.approx(1.114) and r0["xa_prime"] == pytest.approx(0.124)


def test_gmd_spp_step_expansion():
    """GMDDataSPP.m 69 kV: I1=[1;72;93;...;1000], GMD=[4;7;8;...;15] with
    MATLAB's overlapping inclusive ranges — later assignments win at the
    boundaries (slot 72 is 7, not 4)."""
    df = pd.read_parquet(DATA / "gmd_tables.parquet")
    spp69 = df[(df["region"] == "SPP") & (df["voltage_kv"] == 69)].set_index("slot")["gmd_ft"]
    assert len(spp69) == 1000
    assert spp69.loc[1] == 4
    assert spp69.loc[71] == 4
    assert spp69.loc[72] == 7          # boundary: second range overwrites
    assert spp69.loc[93] == 8
    assert spp69.loc[1000] == 15


def test_gmd_xd_vectors():
    xd = pd.read_parquet(DATA / "gmd_xd.parquet")
    spp = xd[xd["region"] == "SPP"].set_index("gmd_ft")
    # GMDDataSPP.m: Xd = [0; 0.0841; 0.1333; ...], XdPrime = [0; 0.0205; ...]
    assert spp.loc[1, "xd"] == 0.0
    assert spp.loc[2, "xd"] == pytest.approx(0.0841)
    assert spp.loc[2, "xd_prime"] == pytest.approx(0.0205)
    # physics constants: identical across all regions
    for region in xd["region"].unique():
        r = xd[xd["region"] == region].set_index("gmd_ft")
        assert np.allclose(r[["xd", "xd_prime"]], spp[["xd", "xd_prime"]])


@pytest.mark.skipif(
    not (REF / "ybus_new.npz").exists(),
    reason="run scripts/convert_mat_data.py first (tests/reference/ is gitignored)",
)
def test_ybus_new_reference():
    ybus = sp.load_npz(REF / "ybus_new.npz")
    assert ybus.shape == (7132, 7132)
    # First assignments in YBus_new.m (1-based → 0-based):
    #   Ybus(1,1) = 2.2855 - j*142.5150 ; Ybus(1,89) = -0.5714 + j*35.6288
    assert ybus[0, 0] == pytest.approx(2.2855 - 142.515j, abs=1e-4)
    assert ybus[0, 88] == pytest.approx(-0.5714 + 35.6288j, abs=1e-4)
    # A Ybus is structurally symmetric
    assert (ybus != ybus.T).nnz == 0

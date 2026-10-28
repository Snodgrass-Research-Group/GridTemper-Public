"""One-time conversion of the MATLAB reference data to native formats (D5).

The converted outputs (the ``gridtemper/data/*.parquet`` files) are already
committed, so you do NOT need to run this to use GridTemper. It is kept for
provenance and to regenerate the data if a source table changes.

To re-run you must supply the MATLAB sources, which are NOT part of this repo —
they live in the source ``TX-2030-Case-Updates`` repository under ``Functions/``.
Copy that ``Functions/`` directory to the GridTemper repo root (or edit the
``FUNCTIONS`` path below to point at it), then run from the repo root:

    python -m uv run python scripts/convert_mat_data.py

Produces:

* ``gridtemper/data/conductors.parquet``  ← Functions/ConductorDataTables/ConductorData{kV}.mat
* ``gridtemper/data/gmd_tables.parquet``  ← Functions/GMD Statistics and lookup tables/GMDData*.m
* ``gridtemper/data/gmd_xd.parquet``      ← Xd / XdPrime vectors from the same files
* ``tests/reference/ybus_new.npz``        ← Functions/YBus_new.m (sparse, regression reference)
* ``gridtemper/data/xf_data.parquet`` + ``xf_xr.parquet``
      ← Data.mat holds these as MATLAB *table* objects that scipy cannot read;
        they are exported to CSV by matlab_reference/dump_reference.m and
        converted here on a second run once those CSVs exist.

Notes recorded during conversion (2026-07-12):

* QNCAResults.mat needs NO conversion: main.m loads its ``params`` struct but
  overwrites every field the pipeline reads (TotalPaths, Scenarios, nodes)
  before use — the load is vestigial.
* Functions/Ybus.mat has no consumer (CALL_TREE.md) — not converted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
#: MATLAB sources are not committed to GridTemper (see module docstring); drop
#: the source repo's ``Functions/`` here, or repoint this, to re-run.
FUNCTIONS = REPO_ROOT / "Functions"
DATA_OUT = REPO_ROOT / "gridtemper" / "data"
REF_OUT = REPO_ROOT / "tests" / "reference"

CONDUCTOR_KVS = [69, 138, 230, 345, 500, 765]
#: Column names as declared in CreateConductorData.m (snake_cased).
CONDUCTOR_COLUMNS = ["mva", "kv", "size_kcmil", "cond_per_phase", "ra", "xa", "xa_prime", "cond_index"]

GMD_DIR = FUNCTIONS / "GMD Statistics and lookup tables"
GMD_VOLTAGES = [69, 138, 230, 345, 500, 765]


def convert_conductors() -> None:
    """All six per-kV conductor tables into one Parquet with a kv column.

    The MATLAB per-file CondIndex column (col 8) is dropped: it restarts per
    file and CreateConductorData.m overwrites it after concatenation anyway.
    ConductorLibrary.from_reference() re-derives it for the enabled levels.
    """
    frames = []
    for kv in CONDUCTOR_KVS:
        mat = sio.loadmat(FUNCTIONS / "ConductorDataTables" / f"ConductorData{kv}.mat")
        arr = mat[f"ConductorData{kv}"]
        df = pd.DataFrame(arr, columns=CONDUCTOR_COLUMNS)
        if not (df["kv"] == kv).all():
            raise ValueError(f"ConductorData{kv}.mat contains rows with kV != {kv}")
        frames.append(df.drop(columns="cond_index"))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DATA_OUT / "conductors.parquet", index=False)
    print(f"conductors.parquet: {len(out)} rows, kV levels {sorted(out['kv'].unique().astype(int).tolist())}")


def _parse_matlab_vector(text: str) -> np.ndarray:
    """Parse a MATLAB numeric vector body ('1;72;93' or '4;7;8')."""
    return np.array([float(tok) for tok in re.split(r"[;,\s]+", text.strip()) if tok])


def _expand_step_function(i1: np.ndarray, gmd: np.ndarray, n: int = 1000) -> np.ndarray:
    """Replicate ``GMDxx(I1(i):I1(i+1)) = GMD(i)`` (1-based, inclusive)."""
    out = np.zeros(n)
    for i in range(len(i1) - 1):
        out[int(i1[i]) - 1 : int(i1[i + 1])] = gmd[i]
    return out


def convert_gmd_tables() -> None:
    """Parse every GMDData<Region>.m into long-format Parquet tables.

    gmd_tables.parquet: region, voltage_kv, slot (1..1000), gmd_ft
    gmd_xd.parquet:     region, gmd_ft (row index+ matches MATLAB 1-based
                        indexing by GMD), xd, xd_prime
    """
    table_rows, xd_frames = [], []
    files = sorted(GMD_DIR.glob("GMDData*.m"))
    if not files:
        raise FileNotFoundError(f"no GMDData*.m under {GMD_DIR}")
    for f in files:
        region = f.stem.removeprefix("GMDData")
        text = f.read_text()
        # Per-voltage step functions: GMD<kv>=zeros(1000,1); I1=[...]; GMD=[...];
        for kv in GMD_VOLTAGES:
            m = re.search(
                rf"GMD{kv}\s*=\s*zeros\(1000,1\);\s*I1\s*=\s*\[([^\]]+)\];\s*GMD\s*=\s*\[([^\]]+)\];",
                text,
            )
            if not m:
                raise ValueError(f"{f.name}: no step-function block for {kv} kV")
            i1 = _parse_matlab_vector(m.group(1))
            gmd = _parse_matlab_vector(m.group(2))
            if len(gmd) != len(i1) - 1:
                raise ValueError(f"{f.name} {kv} kV: len(GMD)={len(gmd)} != len(I1)-1={len(i1)-1}")
            expanded = _expand_step_function(i1, gmd)
            table_rows.append(
                pd.DataFrame({
                    "region": region,
                    "voltage_kv": kv,
                    "slot": np.arange(1, 1001),
                    "gmd_ft": expanded,
                })
            )
        # Xd / XdPrime lookup vectors (indexed by GMD in feet, 1-based)
        vecs = {}
        for name in ("Xd", "XdPrime"):
            m = re.search(rf"\b{name}\s*=\s*\[([^\]]+)\];", text)
            if not m:
                raise ValueError(f"{f.name}: no {name} vector")
            vecs[name] = _parse_matlab_vector(m.group(1))
        if len(vecs["Xd"]) != len(vecs["XdPrime"]):
            raise ValueError(f"{f.name}: Xd/XdPrime length mismatch")
        xd_frames.append(
            pd.DataFrame({
                "region": region,
                "gmd_ft": np.arange(1, len(vecs["Xd"]) + 1),  # MATLAB Xd(GMD) is 1-based
                "xd": vecs["Xd"],
                "xd_prime": vecs["XdPrime"],
            })
        )
    tables = pd.concat(table_rows, ignore_index=True)
    xd = pd.concat(xd_frames, ignore_index=True)
    tables.to_parquet(DATA_OUT / "gmd_tables.parquet", index=False)
    xd.to_parquet(DATA_OUT / "gmd_xd.parquet", index=False)
    regions = sorted(tables["region"].unique().tolist())
    # Xd/XdPrime are physics constants — flag if any region deviates
    ref = xd[xd["region"] == regions[0]][["xd", "xd_prime"]].to_numpy()
    identical = all(
        np.allclose(xd[xd["region"] == r][["xd", "xd_prime"]].to_numpy(), ref) for r in regions
    )
    print(f"gmd_tables.parquet: {len(regions)} regions {regions}")
    print(f"gmd_xd.parquet: Xd/XdPrime identical across regions: {identical}")


def convert_ybus_new() -> None:
    """Parse the generated YBus_new.m script into a sparse matrix (.npz).

    Lines look like: ``Ybus(    1,   89) =     -0.5714+ j*(   35.6288);``
    Kept ONLY as a regression reference (quirk Q1 / DEVIATIONS DV10).
    """
    text = (FUNCTIONS / "YBus_new.m").read_text()
    m = re.search(r"Ybus\s*=\s*sparse\((\d+)\)", text)
    n = int(m.group(1))
    entries = re.findall(
        r"Ybus\(\s*(\d+),\s*(\d+)\)\s*=\s*([-\d.eE+]+)\s*\+\s*j\*\(\s*([-\d.eE+]+)\s*\);", text
    )
    if not entries:
        raise ValueError("no Ybus assignments matched — check the regex against YBus_new.m")
    rows = np.array([int(e[0]) - 1 for e in entries])
    cols = np.array([int(e[1]) - 1 for e in entries])
    vals = np.array([float(e[2]) + 1j * float(e[3]) for e in entries])
    ybus = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    sp.save_npz(REF_OUT / "ybus_new.npz", ybus)
    print(f"ybus_new.npz: {n}x{n}, {ybus.nnz} nonzeros")


def convert_xf_tables() -> None:
    """Data.mat stores XFData/XFXR as MATLAB tables (MCOS) — unreadable by
    scipy. dump_reference.m exports them as CSVs; convert when present."""
    src = REF_OUT / "matlab_csv"
    xf_data, xf_xr = src / "xf_data.csv", src / "xf_xr.csv"
    if not (xf_data.exists() and xf_xr.exists()):
        print(
            "xf_data/xf_xr: SKIPPED — run matlab_reference/dump_reference.m first "
            f"(expects {xf_data} and {xf_xr}), then re-run this script."
        )
        return
    for csv, out in ((xf_data, "xf_data.parquet"), (xf_xr, "xf_xr.parquet")):
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower() for c in df.columns]
        df.to_parquet(DATA_OUT / out, index=False)
        print(f"{out}: {len(df)} rows, columns {list(df.columns)}")


def main() -> int:
    """Run every .mat/.m -> parquet/npz conversion in order."""
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    REF_OUT.mkdir(parents=True, exist_ok=True)
    convert_conductors()
    convert_gmd_tables()
    convert_ybus_new()
    convert_xf_tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())

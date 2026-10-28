"""Loaders for the converted reference data in ``gridtemper/data/``.

The Parquet/npz files are produced ONCE by ``scripts/convert_mat_data.py``
(D5) — Python never reads .mat at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DATA_DIR

#: MATLAB concatenates the per-kV conductor tables in this fixed order,
#: including only the levels present in options.VoltageLevels (CreateConductorData.m).
_CONDUCTOR_KV_ORDER = [69, 138, 230, 345, 500, 765]


def load_conductor_table(voltage_levels: list[int]) -> pd.DataFrame:
    """Conductor library filtered to ``voltage_levels`` — CreateConductorData.m.

    Reads ``data/conductors.parquet`` (all kV levels, already stored in
    ascending-kV block order), keeps the enabled levels, and re-indexes
    ``cond_index`` 0..N-1 over the concatenation — preserving the MATLAB
    behavior that CondIndex depends on which voltage levels are enabled
    (algorithms/04_conductors.md). 0-based (MATLAB's 1-based CondIndex minus 1).

    Parameters
    ----------
    voltage_levels : list[int]
        Enabled nominal kV levels (CaseConfig.voltage_levels), e.g. [69,138,345].

    Returns
    -------
    pandas.DataFrame
        Columns ``mva, kv, size_kcmil, cond_per_phase, ra, xa, xa_prime,
        cond_index`` — one row per conductor at an enabled level.
    """
    enabled = {int(kv) for kv in voltage_levels}
    df = pd.read_parquet(DATA_DIR / "conductors.parquet")
    # Boolean masking preserves the parquet's ascending-kV block order, which is
    # exactly the CreateConductorData concatenation order.
    df = df[df["kv"].isin(enabled)].reset_index(drop=True)
    df["cond_index"] = np.arange(len(df), dtype=np.int64)
    return df


@dataclass(frozen=True)
class GmdData:
    """GMD lookup bundle for one region — the GMDData<Region>.m return triple."""

    gmd_table: np.ndarray   # (1000, 6): percentile slot × voltage class → GMD ft
    xd: np.ndarray          # inductive spacing factor, indexed by GMD ft (0-based)
    xd_prime: np.ndarray    # capacitive spacing factor, indexed by GMD ft (0-based)


def load_gmd_data(region: str) -> GmdData:
    """GMD statistics for ``region`` (e.g. 'SPP') — replaces the
    ``feval(['GMDData' Region])`` dispatch in RXB.m.

    ``gmd_table[i, c]`` is the GMD (ft) for percentile slot ``i`` (0-based; the
    MATLAB ``GMDIndex`` 1..1000 minus 1) and voltage-class column ``c`` (0..5 for
    69/138/230/345/500/765 kV). ``xd``/``xd_prime`` are indexed by GMD in feet
    (0-based; MATLAB's 1-based ``Xd(GMD)`` minus 1).
    """
    tables = pd.read_parquet(DATA_DIR / "gmd_tables.parquet")
    t = tables[tables["region"] == region]
    if t.empty:
        raise ValueError(f"no GMD data for region {region!r} in gmd_tables.parquet")
    piv = t.pivot(index="slot", columns="voltage_kv", values="gmd_ft").sort_index()
    gmd_table = piv[_CONDUCTOR_KV_ORDER].to_numpy()

    xd_df = pd.read_parquet(DATA_DIR / "gmd_xd.parquet")
    xr = xd_df[xd_df["region"] == region].sort_values("gmd_ft")
    return GmdData(
        gmd_table=gmd_table,
        xd=xr["xd"].to_numpy(),
        xd_prime=xr["xd_prime"].to_numpy(),
    )


def load_xf_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transformer impedance tables ``(xf_data, xf_xr)`` from Data.mat —
    consumed by conductors.transformers (TransformerImpedances.m).

    ``xf_data`` columns: ``pri, sec, ter, xhl, xht, xlt`` (kV, kV, kV, %, %, %).
    ``xf_xr`` columns: ``mva, xr``.
    """
    xf_data = pd.read_parquet(DATA_DIR / "xf_data.parquet")
    xf_xr = pd.read_parquet(DATA_DIR / "xf_xr.parquet")
    return xf_data, xf_xr

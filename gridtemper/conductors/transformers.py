"""Transformer impedance/sizing — TransformerImpedances.m.

Algorithm doc: conversion_manual/algorithms/04_conductors.md §TransformerImpedances.
DORMANT (RealPowerLoop only); no base-pipeline dump — physics/spot tests only.
Bug fix DV5 applies (stale ZIndex on lookup miss → raise instead of silently
reusing the previous transformer's impedance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest(values: np.ndarray, query: float) -> float:
    """MATLAB ``interp1(u, u, q, 'nearest', 'extrap')`` — snap to the nearest
    value in the sorted unique grid (endpoints for out-of-range queries)."""
    return float(values[np.abs(values - query).argmin()])


def size_transformers(
    branches: pd.DataFrame,
    xf_data: pd.DataFrame,
    xf_xr: pd.DataFrame,
) -> pd.DataFrame:
    """Size transformer rows (``Tap != 0``) from the lookup tables.

    Steps (per the source): clamp ``LimMVAA`` up to the ``MinCondIndex`` floor
    (mark ``Undersized``); for ``Resize`` transformers, snap Pri/Sec kV to the
    ``xf_data`` grid, interpolate the X/R ratio (splitting into parallel units
    above 2000 MVA), read ``XHL`` from the (Pri, Sec) or swapped (Sec, Pri)
    entry, and compute ``R``/``X`` on the 100 MVA base. Sets ``cond_index`` to
    the per-unit MVA of an individual transformer (``ceil(LimMVAA/Circuits)``)
    for every transformer row — the MATLAB column-meaning overload, kept.

    Raises
    ------
    ValueError
        On a (Pri, Sec) lookup miss or ``Pri == Sec`` (DV5) — MATLAB left a
        stale ``ZIndex`` here, silently reusing a prior impedance.
    """
    b = branches.reset_index(drop=True).copy()
    n = len(b)
    xf = np.flatnonzero(b["Tap"].to_numpy() != 0)

    # .copy() because these are mutated in place (pandas>=3.0 returns read-only views).
    lim = b["LimMVAA"].to_numpy(dtype=float).copy()
    circuits = (b["Circuits"].to_numpy(dtype=float).copy() if "Circuits" in b.columns else np.ones(n))
    min_ci = b["MinCondIndex"].to_numpy(dtype=float)
    undersized = (b["Undersized"].to_numpy(dtype=float).copy() if "Undersized" in b.columns else np.zeros(n))
    from_kv = b["FromNomKV"].to_numpy(dtype=float).copy()
    to_kv = b["ToNomKV"].to_numpy(dtype=float).copy()
    r_pu = b["R"].to_numpy(dtype=float).copy() if "R" in b.columns else np.zeros(n)
    x_pu = b["X"].to_numpy(dtype=float).copy() if "X" in b.columns else np.zeros(n)

    # Clamp below-floor transformers up to their minimum size.
    below = lim < min_ci
    lim[below] = min_ci[below]
    undersized[below & (b["Tap"].to_numpy() != 0)] = 1

    pri = xf_data["pri"].to_numpy(dtype=float)
    sec = xf_data["sec"].to_numpy(dtype=float)
    xhl = xf_data["xhl"].to_numpy(dtype=float)
    uniq_pri = np.unique(pri)
    uniq_sec = np.unique(sec)
    xr_mva = xf_xr["mva"].to_numpy(dtype=float)
    xr_ratio = xf_xr["xr"].to_numpy(dtype=float)
    order = np.argsort(xr_mva)
    xr_mva, xr_ratio = xr_mva[order], xr_ratio[order]

    resize = b["Resize"].to_numpy()
    xf_resize = np.flatnonzero((resize != 0) & (b["Tap"].to_numpy() != 0))
    for i in xf_resize:
        from_kv[i] = _nearest(uniq_pri, from_kv[i])
        to_kv[i] = _nearest(uniq_sec, to_kv[i])

        if lim[i] < 50:
            ratio = 25.0
        elif lim[i] > 2000:
            circuits[i] = np.ceil(lim[i] / 2000)
            ratio = float(np.interp(lim[i] / circuits[i], xr_mva, xr_ratio))
        else:
            ratio = float(np.interp(lim[i], xr_mva, xr_ratio))

        if from_kv[i] == to_kv[i]:
            raise ValueError(
                f"transformer row {i}: Pri==Sec=={from_kv[i]} kV — no impedance "
                "lookup (DV5; MATLAB reused a stale ZIndex here)"
            )
        z_idx = np.flatnonzero((pri == from_kv[i]) & (sec == to_kv[i]))
        if z_idx.size == 0:
            z_idx = np.flatnonzero((sec == from_kv[i]) & (pri == to_kv[i]))
        if z_idx.size == 0:
            raise ValueError(
                f"transformer row {i}: no XFData entry for (Pri,Sec)="
                f"({from_kv[i]},{to_kv[i]}) kV (DV5)"
            )
        z_pu = (100.0 / lim[i]) * (xhl[z_idx[0]] / 100.0)
        x_val = np.sqrt(z_pu**2 / (1 + 1 / ratio**2))
        if np.isnan(x_val):
            raise ValueError(f"transformer row {i}: NaN reactance (XR={ratio}) (DV5)")
        x_pu[i] = x_val
        r_pu[i] = x_val / ratio

    b["LimMVAA"] = lim
    b["Circuits"] = circuits
    b["Undersized"] = undersized
    b["FromNomKV"] = from_kv
    b["ToNomKV"] = to_kv
    b["R"] = r_pu
    b["X"] = x_pu
    # CondIndex = per-transformer MVA (overloaded meaning for XF rows).
    cond_index = b["cond_index"].to_numpy(dtype=float).copy() if "cond_index" in b.columns else np.zeros(n)
    cond_index[xf] = np.ceil(lim[xf] / circuits[xf])
    b["cond_index"] = cond_index
    return b

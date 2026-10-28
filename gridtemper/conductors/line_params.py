"""Line R/X/B with long-line correction — RXB.m.

Algorithm doc: conversion_manual/algorithms/04_conductors.md §RXB.
DORMANT (RealPowerLoop only); no base-pipeline reference dump — validated with
physics identities (the γℓ→0 short-line limit).

DV6 (R, X, **and B** all divided by circuits) is REPLICATED per the author's
2026-07-21 ruling — dividing B by parallel circuits is physically dubious
(shunt admittances add), but kept until confirmed. See DEVIATIONS.md DV6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CaseConfig, TemperOptions
from ..io.reference_data import GmdData
from .library import ConductorLibrary


def _gmd_column(kv: int) -> int:
    """0-based voltage-class column in the GMD table (MATLAB floor(kv/115)+1,
    with 765 kV forced to the 6th column)."""
    return 5 if kv == 765 else int(np.floor(kv / 115))


def compute_rxb(
    branches: pd.DataFrame,
    case: CaseConfig,
    options: TemperOptions,
    library: ConductorLibrary,
    gmd: GmdData,
) -> pd.DataFrame:
    """Per-unit R/X/B for transmission lines from conductor data + GMD spacing +
    length, using the distributed-parameter (sinh/tanh) correction.

    Required branch columns: ``FromNomKV`` (kV), ``cond_index`` (0-based),
    ``gmd_index`` (1..1000 percentile draw), ``length_km``, ``circuits``.
    Writes ``R``/``X``/``B`` (pu on ``options`` s_base) and ``gmd_ft``; caps
    ``LimMVAA`` at the 45° stability limit ``s_base·sin45°/X``.

    Formulas (RXB.m, verbatim): with per-mile ``r = Ra``, ``x = Xa + Xd(GMD)``,
    ``b = j/((XaP + XdPrime(GMD))·1e6)``, ``z = r + jx``, ``γ = √(z·b)``,
    ``ℓ`` miles: ``Z = z·ℓ·sinh(γℓ)/(γℓ)`` and ``B = Im(b·ℓ·tanh(γℓ/2)/(γℓ/2))``,
    per-unit on ``kV²/100``, then divide by circuits (DV6).
    """
    b = branches.reset_index(drop=True).copy()
    n = len(b)
    lib = library.table
    ra = lib["ra"].to_numpy()
    xa = lib["xa"].to_numpy()
    xa_prime = lib["xa_prime"].to_numpy()

    length_mi = options.routing_factor * b["length_km"].to_numpy(float) / 1.61
    from_kv = b["FromNomKV"].to_numpy(float)
    cond_index = b["cond_index"].to_numpy(dtype=np.int64)
    gmd_index = b["gmd_index"].to_numpy(dtype=np.int64)

    r_pu = np.zeros(n)
    x_pu = np.zeros(n)
    b_pu = np.zeros(n)
    gmd_ft = np.zeros(n, dtype=np.int64)

    for kv in np.unique(from_kv):
        c1 = np.flatnonzero(from_kv == kv)
        col = _gmd_column(int(kv))
        gmd_here = gmd.gmd_table[gmd_index[c1] - 1, col].astype(np.int64)  # GMDIndex 1-based
        gmd_ft[c1] = gmd_here
        ci = cond_index[c1]
        r = ra[ci]
        x = xa[ci] + gmd.xd[gmd_here - 1]                    # Xd(GMD), 1-based
        bb = 1j / ((xa_prime[ci] + gmd.xd_prime[gmd_here - 1]) * 1e6)
        z = r + 1j * x
        gama = np.sqrt(z * bb)
        ell = length_mi[c1]
        gl = gama * ell
        z_long = z * ell * (np.sinh(gl) / gl)
        b_long = np.imag(bb * ell * (np.tanh(gl / 2) / (gl / 2)))
        base = kv**2 / 100.0
        r_pu[c1] = np.real(z_long) / base
        x_pu[c1] = np.imag(z_long) / base
        b_pu[c1] = b_long * base

    circuits = b["circuits"].to_numpy(float)
    # DV6: R, X, and B all divided by circuits (B division kept per ruling).
    r_pu /= circuits
    x_pu /= circuits
    b_pu /= circuits

    b["R"] = r_pu
    b["X"] = x_pu
    b["B"] = b_pu
    b["gmd_ft"] = gmd_ft

    stability_limit = case.s_base_mva * np.sin(np.deg2rad(45)) / x_pu
    b["LimMVAA"] = np.minimum(b["LimMVAA"].to_numpy(float), stability_limit)
    return b

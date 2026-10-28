"""Bus admittance matrices — YBusCalc.m / YBusCalcDC.m.

Algorithm doc: conversion_manual/algorithms/05_powerflow_core.md §YBusCalc.
Bug fix DV8 applies (dead ``B1pu`` line in the DC contingency branch is dropped).

The builder caches the last result keyed on (branch-parameter fingerprint,
contingency), replacing the ReCalcYBus/YBusOld plumbing threaded through the
MATLAB drivers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class YbusSet:
    """One YBusCalc output bundle (short = no shunts, long = + line charging B/2)."""

    ybus_short: sp.csc_matrix       # complex, A·Yprim·Aᵀ
    ybus_long: sp.csc_matrix        # + j·diag(Σ B/2 per bus)
    yprim: sp.csc_matrix            # diagonal branch admittances 1/(R+jX)
    bbus_short: sp.csc_matrix       # real susceptance variant (LosslessLines)
    bbus_long: sp.csc_matrix
    a_con: sp.csc_matrix            # incidence with contingency column removed
    b_matrix: np.ndarray            # per-bus line-charging shunt Σ B/2


class YbusBuilder:
    """Builds and caches AC/DC system matrices from branch parameters."""

    def __init__(self) -> None:
        self._cache: dict = {}

    def build_ac(
        self,
        A: sp.csc_matrix,
        from_bus: np.ndarray,
        to_bus: np.ndarray,
        r_pu: np.ndarray,
        x_pu: np.ndarray,
        b_pu: np.ndarray,
        contingency: int | None = None,
    ) -> YbusSet:
        """YBusCalc.m.

        ``YBusShort = A·Yprim·Aᵀ`` (series only); ``YBusLong`` adds the line-charging
        shunt ``j·diag(Σ B/2)`` per bus. For a contingency the full-size ``A`` is
        kept for the product (with the primitive zeroed on that branch); ``a_con``
        (column removed) is returned only for connectivity checks (per the source).
        """
        r_pu = np.asarray(r_pu, float)
        x_pu = np.asarray(x_pu, float)
        b_charge = np.asarray(b_pu, float).copy()
        y1 = 1.0 / (r_pu + 1j * x_pu)
        b1 = 1.0 / (1j * x_pu)
        a_con = A
        if contingency is not None and contingency >= 0:
            y1[contingency] = 0.0
            b1[contingency] = 0.0
            b_charge[contingency] = 0.0
            a_con = _drop_column(A, contingency)

        yprim = sp.diags(y1, format="csc")
        bprim = sp.diags(b1, format="csc")
        ybus_short = (A @ yprim @ A.T).tocsc()
        bbus_short = (A @ bprim @ A.T).imag.tocsc()

        n_bus = A.shape[0]
        idx = np.concatenate([np.asarray(from_bus), np.asarray(to_bus)])
        half = np.concatenate([b_charge / 2, b_charge / 2])
        b_matrix = np.bincount(idx, weights=half, minlength=n_bus)
        shunt = sp.diags(b_matrix, format="csc")
        ybus_long = (ybus_short + 1j * shunt).tocsc()
        bbus_long = (bbus_short + shunt).tocsc()
        return YbusSet(ybus_short, ybus_long, yprim, bbus_short, bbus_long, a_con, b_matrix)

    def build_dc(
        self,
        A: sp.csc_matrix,
        x_pu: np.ndarray,
        contingency: int | None = None,
    ) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
        """YBusCalcDC.m — returns ``(bbus, bprim, a_con)`` with ``Bprim = diag(1/X)``
        and ``BBus = A·Bprim·Aᵀ`` (real). DV8: the dead ``B1pu`` line in the
        contingency branch is not ported."""
        x_pu = np.asarray(x_pu, float)
        b1 = 1.0 / x_pu
        a_con = A
        if contingency is not None and contingency >= 0:
            b1[contingency] = 0.0
            a_con = _drop_column(A, contingency)
        bprim = sp.diags(b1, format="csc")
        bbus = (A @ bprim @ A.T).tocsc()
        return bbus, bprim, a_con


def _drop_column(A: sp.csc_matrix, col: int) -> sp.csc_matrix:
    """Return ``A`` with column ``col`` removed (contingency ACon)."""
    keep = np.ones(A.shape[1], dtype=bool)
    keep[col] = False
    return A.tocsc()[:, keep]

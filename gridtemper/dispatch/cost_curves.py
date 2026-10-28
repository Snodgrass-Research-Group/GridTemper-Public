"""Quadratic cost-coefficient fitting — costcurve_to_cubic.m (name notwithstanding).

Algorithm doc: conversion_manual/algorithms/03_dispatch.md §costcurve_to_cubic.

**DV11 (revised — replicate MATLAB, do NOT apply the originally-planned fix).**
When the quadratic fit is concave, MATLAB refits linear and stores ``C = slope``,
``B = intercept``, ``A = 0``. The manual first called this a bug and proposed
swapping to ``B = slope``, ``A = intercept``. Reference comparison showed that
would be *wrong*: the PW curve y-values are incremental ($/MWh) **prices**, so
the intercept is the marginal-cost *level*; UCED's marginal cost ``B + 2·C·P``
must therefore keep the intercept in ``B``. On TX-2030 this branch fires for
~273 generators, and the proposed fix would have driven their marginal cost to
≈ the slope (near zero), making them look almost free. Kept MATLAB's placement;
flagged for the author (see DEVIATIONS.md DV11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The PW piecewise curve is 10 breakpoint/price pairs (MATLAB reads 1..10).
_N_POINTS = 10
_BREAK_COLS = [f"MWBreak{i}" for i in range(1, _N_POINTS + 1)]
_PRICE_COLS = [f"MWhPrice{i}" for i in range(1, _N_POINTS + 1)]
#: MATLAB keeps ``ModifiedGencostInfo(:,1:22)`` (metadata incl. NumberOfBus,
#: MinMW, MaxMW) then appends A/B/C.
_N_META_COLS = 22


def fit_quadratic_costs(gencost: pd.DataFrame) -> pd.DataFrame:
    """Fit ``A + B·P + C·P²`` to each generator's 10-point PW piecewise curve.

    Per generator row (costcurve_to_cubic.m):

    * ``MWhPrice1 > 0``: ``polyfit(MWBreak, MWhPrice, 2)``. If the leading
      (quadratic) coefficient is negative the fit is concave, so refit degree 1
      and store ``C = slope, B = intercept, A = 0`` — MATLAB's placement, kept
      deliberately (DV11 revised: the intercept is the marginal-cost level and
      belongs in ``B``).
    * ``MWhPrice1 < 0``: ``A = B = C = 0``.
    * ``MWhPrice1 == 0``: untouched (stays 0).

    The MWhPrice values are incremental ($/MWh) prices, so the fitted polynomial
    is really a marginal-cost curve reused as a total-cost curve — preserved
    as-is (algorithm-for-algorithm, D13).

    Parameters
    ----------
    gencost : pandas.DataFrame
        Generator cost table with ``MWBreak1..10`` / ``MWhPrice1..10`` columns
        (MW / $ per MWh).

    Returns
    -------
    pandas.DataFrame
        The first 22 metadata columns plus float ``A``/``B``/``C`` coefficients
        (units: $/h, $/MWh, $/MWh²).
    """
    out = gencost.iloc[:, :_N_META_COLS].copy()
    n = len(out)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    x = gencost[_BREAK_COLS].to_numpy(dtype=float)
    y = gencost[_PRICE_COLS].to_numpy(dtype=float)
    price1 = y[:, 0]

    for i in range(n):
        if price1[i] > 0:
            quad = np.polyfit(x[i], y[i], 2)  # [C, B, A], highest power first
            if quad[0] < 0:  # concave — refit linear (MATLAB placement, DV11)
                lin = np.polyfit(x[i], y[i], 1)  # [slope, intercept]
                c[i] = lin[0]
                b[i] = lin[1]
                a[i] = 0.0
            else:
                c[i] = quad[0]
                b[i] = quad[1]
                a[i] = quad[2]
        # price1 < 0 -> zeros (already initialized); price1 == 0 -> untouched
    out["A"] = a
    out["B"] = b
    out["C"] = c
    return out

"""Net complex injection per bus — Create_SNet.m.

Algorithm doc: conversion_manual/algorithms/03_dispatch.md §Create_SNet.
Sign convention HERE: generation positive, load negative (SNet = gen − demand).
The Newton solver boundary flips this (pfmiss wants generation negative) —
that flip lives in refinement.reactive, exactly where MATLAB does it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DispatchOptions
from ..grid_model import DispatchResult
from .uced import EconomicDispatch


def build_snet(
    demand: pd.DataFrame,
    gencost_q: pd.DataFrame,
    dispatch_options: DispatchOptions,
    n_bus: int,
) -> tuple[np.ndarray, DispatchResult]:
    """Dispatch the fleet against total demand and aggregate to buses.

    Steps (Create_SNet.m):

    1. ``Load = real(S_pu)`` — the demand column is MW+jMVAr despite its name
       (blocker B2, confirmed MW); only the real part drives dispatch.
    2. ``UCED`` on the quadratic-cost table for total load.
    3. Scatter committed dispatch and demand to buses with ``np.bincount``
       (replaces ``accumarray(..., [7132 1])`` — size from data, D7).
    4. ``SNet = Σ dispatch − Σ demand`` per bus.

    Parameters
    ----------
    demand : pandas.DataFrame
        Renumbered demand with complex ``S_pu`` (MW+jMVAr) and 0-based ``bus``.
    gencost_q : pandas.DataFrame
        Renumbered quadratic-cost table (``A``/``B``/``C``, ``MinMW``/``MaxMW``,
        0-based ``bus``).
    dispatch_options : DispatchOptions
        UCED backend + reserve settings.
    n_bus : int
        Bus count (length of the returned vector).

    Returns
    -------
    (snet, dispatch) : (numpy.ndarray[complex], DispatchResult)
        ``snet`` is complex (n_bus,), MW with generation positive; today the
        imaginary part is zero (Create_SNet dispatches real power only) but the
        dtype is complex for the reactive future.
    """
    load_mw = np.real(demand["S_pu"].to_numpy())
    total_load = float(load_mw.sum())

    dispatch = EconomicDispatch(dispatch_options).solve(gencost_q, total_load)

    gen_bus = dispatch.committed_gens["bus"].to_numpy(dtype=np.int64)
    gen_by_bus = np.bincount(gen_bus, weights=dispatch.setpoints_mw, minlength=n_bus)

    load_bus = demand["bus"].to_numpy(dtype=np.int64)
    load_by_bus = np.bincount(load_bus, weights=load_mw, minlength=n_bus)

    snet = (gen_by_bus - load_by_bus).astype(np.complex128)
    return snet, dispatch

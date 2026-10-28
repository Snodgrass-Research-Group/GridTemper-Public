"""Branch MVA-limit resizing — ResizeBranches.m.

Algorithm doc: conversion_manual/algorithms/04_conductors.md §ResizeBranches.
DORMANT (RealPowerLoop only). Bug fix DV7 applies: the MATLAB non-random path
``sortrows(Branch,'PercentUtilized')`` permanently reorders the table and never
restores it; here selection is done on index arrays and the table order is
preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ResizeOptions


def resize_branches(
    branches: pd.DataFrame,
    options: ResizeOptions,
    flows_mva: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Upsize overloaded / downsize underutilized branches.

    Upsize candidates (``|flow|/LimMVAA > 1/conductor_margin`` and
    ``cond_index < MaxCondIndex``) get ``LimMVAA = ceil(LimMVAA·percent_change)``;
    downsize candidates (``util < downsize_threshold`` and
    ``cond_index > MinCondIndex``) get ``LimMVAA = ceil(LimMVAA/percent_change)``.
    Each side is capped at ``ceil(percent_resize·n_branch)`` rows, selected by
    seeded shuffle (``random_resize``) or by utilization ranking. Changed rows
    are flagged ``Resize = 1``. ``Overloaded``/``Undersized`` flag the rows that
    are over/under-utilized but pinned at the conductor limit (can't auto-resize).

    Required branch columns: ``LimMVAA``, ``cond_index``, ``MaxCondIndex``,
    ``MinCondIndex``, ``Resize``. ``flows_mva`` is aligned to the branch rows.

    Returns a new table (row order preserved — DV7).
    """
    b = branches.reset_index(drop=True).copy()
    n = len(b)
    lim = b["LimMVAA"].to_numpy(dtype=float).copy()  # mutated in place (pandas>=3.0 COW)
    cond = b["cond_index"].to_numpy(dtype=float)
    max_cond = b["MaxCondIndex"].to_numpy(dtype=float)
    min_cond = b["MinCondIndex"].to_numpy(dtype=float)
    resize = b["Resize"].to_numpy().astype(int).copy()

    util = np.abs(flows_mva) / lim  # PercentUtilized (computed once, on original limits)
    overloaded = np.zeros(n, dtype=int)
    undersized = np.zeros(n, dtype=int)
    cap = int(np.ceil(options.percent_resize * n))

    def pick(selectable: np.ndarray, ascending: bool) -> np.ndarray:
        cand = np.flatnonzero(selectable)
        if options.random_resize:
            cand = cand[rng.permutation(cand.size)]
        else:  # DV7: rank by utilization via index order, never reorder the table
            keys = util[cand] if ascending else -util[cand]
            cand = cand[np.argsort(keys, kind="stable")]
        return cand[:cap] if cand.size > cap else cand

    if options.upsize:
        all_over = util > 1.0 / options.conductor_margin
        selectable = all_over & (cond < max_cond)
        overloaded = all_over.astype(int) - selectable.astype(int)  # overloaded but at max cond
        sel = pick(selectable, ascending=False)
        lim[sel] = np.ceil(lim[sel] * options.percent_change)
        resize[sel] = 1

    if options.downsize:
        all_low = util < options.downsize_threshold
        selectable = (cond > min_cond) & all_low
        undersized = all_low.astype(int) - selectable.astype(int)
        sel = pick(selectable, ascending=True)
        lim[sel] = np.ceil(lim[sel] / options.percent_change)
        resize[sel] = 1

    b["PercentUtilized"] = util
    b["LimMVAA"] = lim
    b["Resize"] = resize
    b["Overloaded"] = overloaded
    b["Undersized"] = undersized
    return b

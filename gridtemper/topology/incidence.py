"""Bus–branch incidence matrix — BuildANet.m.

Algorithm doc: conversion_manual/algorithms/02_io_and_topology.md §BuildANet.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_incidence(
    from_bus: np.ndarray,
    to_bus: np.ndarray,
    n_bus: int,
) -> sp.csc_matrix:
    """Sparse incidence matrix (n_bus × n_branch): +1 at from-bus, −1 at to-bus.

    MATLAB ``BuildANet`` builds ``sparse([From;To],[idx;idx],[+1;-1])`` sized to
    the largest bus number, then pads rows up to ``NumBuses`` (buses with no
    incident line stay all-zero rows). This builds the full ``n_bus`` height
    directly — high-numbered isolated buses are the zero rows.

    Parameters
    ----------
    from_bus, to_bus : array_like of int
        0-based internal endpoint ids, one per branch.
    n_bus : int
        Total bus count (rows). Sizes come from data, never a constant (D7).

    Returns
    -------
    scipy.sparse.csc_matrix
        (n_bus × n_branch) incidence matrix.

    Raises
    ------
    ValueError
        If a branch references a bus outside ``[0, n_bus)`` — replaces main.m's
        ``A ~= AFull`` error (a bus id at/above the count).
    """
    from_bus = np.asarray(from_bus)
    to_bus = np.asarray(to_bus)
    if from_bus.shape != to_bus.shape:
        raise ValueError(f"from_bus/to_bus length mismatch: {from_bus.shape} vs {to_bus.shape}")
    n_branch = from_bus.size
    if n_branch == 0:
        return sp.csc_matrix((n_bus, 0))
    lo = min(int(from_bus.min()), int(to_bus.min()))
    hi = max(int(from_bus.max()), int(to_bus.max()))
    if lo < 0 or hi >= n_bus:
        raise ValueError(
            f"branch endpoint id out of range: [{lo}, {hi}] not within [0, {n_bus})"
        )
    idx = np.arange(n_branch)
    rows = np.concatenate([from_bus, to_bus])
    cols = np.concatenate([idx, idx])
    data = np.concatenate([np.ones(n_branch), -np.ones(n_branch)])
    return sp.coo_matrix((data, (rows, cols)), shape=(n_bus, n_branch)).tocsc()

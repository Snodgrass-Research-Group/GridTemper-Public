"""External→internal bus renumbering — main.m lines 39–59.

Algorithm doc: conversion_manual/algorithms/02_io_and_topology.md §Numbering.
Bug fix DV1 applies here (in-loop mutation replaced by a prebuilt map).
"""

from __future__ import annotations

import numpy as np

from ..grid_model import GridModel


def to_internal_numbering(model: GridModel) -> GridModel:
    """Assign 0-based internal bus ids and remap every table's bus references.

    MATLAB (main.m 39–59) loops over buses assigning internal ids 1..n and, in
    the same loop, remaps branch From/To, gen and gencost ``NumberOfBus``, and
    demand ``BusNumber`` from external numbers. ``GenIndex`` is matched on
    ``ModifiedGenInfo.NumberOfBus`` *while that column is being mutated* — it
    only works because internal ids ascend in processed order (DV1). Here we
    build the external→internal map once and vector-remap, which is order-safe.

    Internal id == 0-based row position in ``buses`` (grid_model conventions).
    The buses are taken in their current order (split_add_gsus already applied
    MATLAB's ``sortrows``), so row ``i`` is internal bus ``i``.

    Adds
    ----
    buses : ``external_num`` (original PW number).
    branches : ``from_ext``/``to_ext`` (original) and ``from_bus``/``to_bus``
        (0-based internal).
    gens, gencost : ``bus_ext`` (original) and ``bus`` (0-based internal).
    demand (if present) : ``bus_ext`` and ``bus``.
    Fills ``model.ext_to_int`` (external number → 0-based id).

    Returns a new GridModel (does not mutate the input).
    """
    out = model.copy()
    buses = out.buses.reset_index(drop=True)
    external = buses["Number"].to_numpy()
    buses["external_num"] = external
    # external PW number -> 0-based internal id (== row position). Built once
    # from the *pre-remap* numbers, so no in-loop-mutation hazard (DV1).
    ext_to_int = {int(ext): i for i, ext in enumerate(external)}
    out.buses = buses
    out.ext_to_int = ext_to_int

    def remap(values: np.ndarray) -> np.ndarray:
        try:
            return np.array([ext_to_int[int(v)] for v in values], dtype=np.int64)
        except KeyError as exc:  # a table references a bus not in the bus list
            raise ValueError(
                f"bus number {exc.args[0]} referenced but absent from the bus table"
            ) from exc

    branches = out.branches.reset_index(drop=True)
    branches["from_ext"] = branches["FromNumber"].to_numpy()
    branches["to_ext"] = branches["ToNumber"].to_numpy()
    branches["from_bus"] = remap(branches["from_ext"].to_numpy())
    branches["to_bus"] = remap(branches["to_ext"].to_numpy())
    out.branches = branches

    gens = out.gens.reset_index(drop=True)
    gens["bus_ext"] = gens["NumberOfBus"].to_numpy()
    gens["bus"] = remap(gens["bus_ext"].to_numpy())
    out.gens = gens

    gencost = out.gencost.reset_index(drop=True)
    gencost["bus_ext"] = gencost["NumberOfBus"].to_numpy()
    gencost["bus"] = remap(gencost["bus_ext"].to_numpy())
    out.gencost = gencost

    if out.demand is not None:
        demand = out.demand.reset_index(drop=True)
        demand["bus_ext"] = demand["BusNumber"].to_numpy()
        demand["bus"] = remap(demand["bus_ext"].to_numpy())
        out.demand = demand

    return out

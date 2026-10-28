"""Split multi-generator buses and add GSU branches — split_add_GSUs.m.

Algorithm doc: conversion_manual/algorithms/02_io_and_topology.md §split_add_GSUs.
Bug fix DV2 applies here (bus-table indexed by generator row index).

Faithful quirk (TX-2030): the loop counter indexes the bus/branch tables **by
row position**, not by bus number, so processing the k-th candidate generator
bus touches the k-th bus/branch *row*. Because the first bus rows are numbered
contiguously (1,2,3,…) while the generator-bus list skips missing bus 63, the
row and the bus number drift apart past that gap. This is replicated exactly to
reproduce the MATLAB dump; the ``first_n_buses`` cutoff (88) is a CaseConfig
constant, not generalized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..grid_model import GridModel


def split_add_gsus(
    model: GridModel,
    first_n_buses: int,
    export_dir: Path | None = None,
) -> GridModel:
    """Give each generator on a multi-generator candidate bus its own bus + GSU.

    Parameters
    ----------
    model : GridModel
        Raw PW tables (before renumbering), with MATLAB-normalized columns.
    first_n_buses : int
        Split candidates are the first N unique generator bus numbers
        (CaseConfig.gsu_split_first_n_buses; 88 for TX-2030).
    export_dir : Path or None
        If given, write the modified tables as CSVs (quirk Q4); None skips the
        side effect (the default).

    Returns
    -------
    GridModel
        New model with the split tables, sorted the way MATLAB's trailing
        ``sortrows`` calls leave them (buses by Number, branches by ToNumber,
        gens by NumberOfBus, gencost by NumberOfBus).
    """
    out = model.copy()
    # Originals (read-only sources for the row copies, matching MATLAB reading
    # BusInfo/BranchInfo while it mutates ModifiedBusInfo/ModifiedBranchInfo).
    bus = model.buses.reset_index(drop=True)
    branch = model.branches.reset_index(drop=True)
    gen = model.gens.reset_index(drop=True)

    mbus = bus.copy()
    mbranch = branch.copy()
    mgen = gen.copy()
    mgencost = model.gencost.reset_index(drop=True).copy()

    gen_bus_numbers = np.unique(gen["NumberOfBus"].to_numpy())  # sorted (MATLAB unique)
    candidates = gen_bus_numbers[:first_n_buses]
    # MATLAB: startgenbusnum = length(GenBuses)+2 (89+1). New buses fill the gap
    # above the last candidate bus number.
    next_bus_num = len(candidates) + 2

    gen_number = gen["NumberOfBus"].to_numpy()
    new_bus_rows: list[pd.Series] = []
    new_branch_rows: list[pd.Series] = []

    for row in range(len(candidates)):  # 0-based == MATLAB gennum-1
        current_gens = np.flatnonzero(gen_number == candidates[row])
        if current_gens.size <= 1:
            continue
        for cg in current_gens:
            new_name = f"{gen.at[cg, 'NameOfBus']} {next_bus_num}"
            mgen.at[cg, "NumberOfBus"] = next_bus_num
            mgen.at[cg, "NameOfBus"] = new_name

            # New bus = copy of the *row-position* candidate bus row.
            nb = bus.iloc[row].copy()
            nb["Number"] = next_bus_num
            nb["Name"] = new_name
            new_bus_rows.append(nb)

            # Original candidate bus row becomes a 345 kV network bus.
            mbus.at[row, "NomKV"] = 345
            mbus.at[row, "Volt_kV_"] = 345
            mbus.at[row, "PUVolt"] = 1
            # DV2 fix: zero the *original split bus row's* generation (the evident
            # intent). MATLAB indexed the bus table with the generator row index
            # ``CurrentGens(i)`` — a latent bug; annotate in the regression test.
            mbus.at[row, "GenMW"] = 0
            mbus.at[row, "GenMvar"] = 0

            # New GSU branch = copy of the candidate branch row, rewired to the
            # new generator bus.
            nbr = branch.iloc[row].copy()
            nbr["FromNumber"] = branch.at[row, "ToNumber"]
            nbr["ToNumber"] = next_bus_num
            nbr["FromNomKV"] = branch.at[row, "FromNomKV"]
            nbr["FromName"] = branch.at[row, "ToName"]
            nbr["ToName"] = new_name
            nbr["Resize"] = 1
            if nbr["BranchDeviceType"] == "Transformer":
                nbr["Tap"] = 1
            new_branch_rows.append(nbr)

            mgencost.at[cg, "NumberOfBus"] = next_bus_num
            mgencost.at[cg, "NameOfBus"] = new_name
            next_bus_num += 1

        # After splitting, the original candidate branch row: 345 kV To side,
        # Line unless the From side is not 345 kV (then a transformer).
        mbranch.at[row, "ToNomKV"] = 345
        mbranch.at[row, "BranchDeviceType"] = "Line"
        mbranch.at[row, "Xfrmr"] = "NO"
        mbranch.at[row, "Resize"] = 1
        if mbranch.at[row, "FromNomKV"] != 345:
            mbranch.at[row, "BranchDeviceType"] = "Transformer"
            mbranch.at[row, "Xfrmr"] = "YES"
        if mbranch.at[row, "BranchDeviceType"] == "Transformer":
            mbranch.at[row, "Tap"] = 1

    if new_bus_rows:
        mbus = pd.concat([mbus, pd.DataFrame(new_bus_rows)], ignore_index=True)
    if new_branch_rows:
        mbranch = pd.concat([mbranch, pd.DataFrame(new_branch_rows)], ignore_index=True)

    # Cleanup: NaN -> 0 for the power columns, add Type/Circuits.
    for col in ("LoadMW", "GenMW", "LoadMvar", "GenMvar", "SwitchedShuntsMvar"):
        mbus[col] = mbus[col].fillna(0)
    load_mw = mbus["LoadMW"].to_numpy()
    gen_mw = mbus["GenMW"].to_numpy()
    bus_type = np.zeros(len(mbus), dtype=int)
    is_pq = (load_mw != 0) | (gen_mw == 0)  # PQ if it has load or no generation
    bus_type[is_pq] = 1
    bus_type[~is_pq & (gen_mw != 0)] = 2    # else PV if it generates
    mbus["Type"] = bus_type
    mbus["Circuits"] = 1

    # MATLAB sortrows: Bus by all cols (col 1 Number is unique -> by Number),
    # Branch by col 3 (ToNumber), Gen by col 2 (NumberOfBus), GenCost by col 1
    # (NumberOfBus). Stable to match sortrows' tie handling.
    out.buses = mbus.sort_values("Number", kind="stable").reset_index(drop=True)
    out.branches = mbranch.sort_values("ToNumber", kind="stable").reset_index(drop=True)
    out.gens = mgen.sort_values("NumberOfBus", kind="stable").reset_index(drop=True)
    out.gencost = mgencost.sort_values("NumberOfBus", kind="stable").reset_index(drop=True)

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        out.buses.to_csv(export_dir / "Buses.csv", index=False)
        out.branches.to_csv(export_dir / "Branches.csv", index=False)
        out.gens.to_csv(export_dir / "GenBuses.csv", index=False)
        out.gencost.to_csv(export_dir / "GenCostCurves.csv", index=False)

    return out

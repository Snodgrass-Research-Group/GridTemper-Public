"""Conductor table + assignment — CreateConductorData.m / AssignConductors.m.

Algorithm doc: conversion_manual/algorithms/04_conductors.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TemperOptions
from ..io.reference_data import load_conductor_table


class ConductorLibrary:
    """Conductor electrical/thermal data for the enabled voltage levels.

    ``table`` columns: ``mva, kv, size_kcmil, cond_per_phase, ra, xa, xa_prime,
    cond_index`` (0-based, re-indexed over the enabled-level concatenation —
    MATLAB behavior; see algorithm doc).
    """

    def __init__(self, table: pd.DataFrame):
        self.table = table

    @classmethod
    def from_reference(cls, voltage_levels: list[int]) -> "ConductorLibrary":
        """Load data/conductors.parquet filtered to ``voltage_levels``."""
        return cls(table=load_conductor_table(voltage_levels))

    def assign(self, branches: pd.DataFrame, options: TemperOptions) -> pd.DataFrame:
        """AssignConductors.m: for lines flagged ``Resize`` (Tap == 0), pick the
        conductor whose MVA is nearest the per-circuit limit at the line's kV.

        Writes ``cond_index`` (0-based; MATLAB's 1-based CondIndex minus 1, with
        −1 for unassigned rows), ``kcmil``, ``cond_per_phase``, the exact table
        capacity × circuits into ``LimMVAA``, and the conductor kV into
        ``FromNomKV``; clears ``Resize`` on the processed lines.

        Parameters
        ----------
        branches : pandas.DataFrame
            Branch table with ``Resize``/``Tap``/``Circuit``/``LimMVAA``/
            ``FromNomKV`` columns (and ``MinkV`` if ``options.change_voltages``).
        options : TemperOptions
            ``change_voltages`` selects the (dormant) voltage-changing path.

        Returns
        -------
        pandas.DataFrame
            A new branch table with the conductor columns filled in.
        """
        b = branches.reset_index(drop=True).copy()
        n = len(b)
        lines = np.flatnonzero((b["Resize"].to_numpy() == 1) & (b["Tap"].to_numpy() == 0))

        circuit = b["Circuit"].to_numpy(dtype=float)
        lim = b["LimMVAA"].to_numpy(dtype=float).copy()      # mutated (pandas>=3.0 COW)
        from_kv = b["FromNomKV"].to_numpy(dtype=float).copy()
        cond_index = np.full(n, -1, dtype=np.int64)       # -1 == MATLAB's 0 (unassigned)
        kcmil = np.zeros(n)
        cond_per_phase = np.zeros(n)

        # Per-circuit rating (MATLAB divides LimMVAA by Circuit before the lookup).
        lim[lines] = lim[lines] / circuit[lines]

        tbl = self.table
        if options.change_voltages:
            self._assign_change_voltages(b, lines, lim, from_kv, cond_index, kcmil, cond_per_phase)
        else:
            for kv, grp in tbl.groupby("kv"):
                sel = lines[from_kv[lines] == kv]
                if sel.size == 0:
                    continue
                mva = grp["mva"].to_numpy()
                idx = np.abs(mva[None, :] - lim[sel][:, None]).argmin(axis=1)  # nearest MVA
                cond_index[sel] = grp["cond_index"].to_numpy()[idx]
                kcmil[sel] = grp["size_kcmil"].to_numpy()[idx]
                cond_per_phase[sel] = grp["cond_per_phase"].to_numpy()[idx]
                from_kv[sel] = kv                                   # ConductorData.kV(CondIndex)
                lim[sel] = circuit[sel] * mva[idx]                  # exact table capacity × circuits

        resize = b["Resize"].to_numpy().copy()
        resize[lines] = 0
        b["cond_index"] = cond_index
        b["kcmil"] = kcmil
        b["cond_per_phase"] = cond_per_phase
        b["LimMVAA"] = lim
        b["FromNomKV"] = from_kv
        b["Resize"] = resize
        return b

    def _assign_change_voltages(self, b, lines, lim, from_kv, cond_index, kcmil, cond_per_phase):
        """Dormant path (``options.ChangeVoltages``): nearest conductor over the
        whole table, bumped up to the branch's ``MinkV`` group if below minimum
        voltage. Untested (no reference dump); implemented for completeness."""
        tbl = self.table
        all_mva = tbl["mva"].to_numpy()
        all_kv = tbl["kv"].to_numpy()
        min_kv = b["MinkV"].to_numpy(dtype=float) if "MinkV" in b.columns else np.zeros(len(b))
        for i in lines:
            j = int(np.abs(all_mva - lim[i]).argmin())
            if all_kv[j] < min_kv[i]:
                grp = tbl[tbl["kv"] == min_kv[i]]
                gj = int(np.abs(grp["mva"].to_numpy() - lim[i]).argmin())
                j = int(grp["cond_index"].to_numpy()[gj])
            cond_index[i] = j
            row = tbl.iloc[j]
            from_kv[i] = row["kv"]
            kcmil[i] = row["size_kcmil"]
            cond_per_phase[i] = row["cond_per_phase"]
            lim[i] = float(b["Circuit"].iloc[i]) * row["mva"]

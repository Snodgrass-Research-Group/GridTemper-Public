"""GridModel: the single data structure every GridTemper stage consumes and
returns — the contract between pipeline modules.

Replaces the MATLAB workspace quartet ``ModifiedBusInfo`` / ``ModifiedBranchInfo``
/ ``ModifiedGenInfo`` / ``ModifiedGenCost_quadratic`` plus ``DemandInfo`` and the
internal/external numbering bookkeeping from ``main.m``.

Conventions (aligned with gridcrucible, D17):

* Internal bus ids are **0-based** and equal to the row position in ``buses``.
  External PowerWorld numbers are data (``external_num``), never indices.
  MATLAB's 1-based internal ids = ours + 1 (slack 1747 MATLAB == 1746 here).
* Power columns are MW / MVAr on ``s_base`` = 100 MVA unless suffixed ``_pu``.
* The incidence matrix is CSC sparse, +1 at from-bus, −1 at to-bus
  (n_bus × n_branch), matching BuildANet.m.

Column schemas are declared as constants so io, checkpoints, and tests can
validate them. Columns marked (PW) come straight from the PowerWorld CSVs and
keep their normalized CSV names; the io module pins the exact list against
the real files — do NOT invent PW column names elsewhere.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Column schemas — the conversion-added columns (authoritative). PW columns
# ride along and are finalized by the io module.
# ---------------------------------------------------------------------------

#: Columns GridTemper adds to the bus table (beyond the PW CSV columns).
BUS_ADDED_COLUMNS = [
    "external_num",   # original PW number (MATLAB Bus.NumExt)
    "bus_type",       # 1=PQ, 2=PV, 3=slack (MATLAB Bus.Type)
    "sync_cond",      # bool — synchronous condenser placed here (ReactivePowerPlacement)
    "v_set",          # voltage setpoint (pu) written back to PowerWorld (Bus.VSet)
]

#: Columns GridTemper adds to the branch table.
BRANCH_ADDED_COLUMNS = [
    "from_bus", "to_bus",         # 0-based internal endpoints (MATLAB FromInt/ToInt - 1)
    "from_ext", "to_ext",         # original PW numbers
    "tap",                        # 0 = line, nonzero = transformer (main.m convention)
    "resize",                     # bool — flagged for re-conductoring (AssignConductors)
    "cond_index",                 # conductor table row (lines) / per-XF MVA (transformers!)
    "min_cond_index", "max_cond_index",
    "circuits",                   # parallel circuits in the right of way
    "lim_mva",                    # MVA limit, all circuits (MATLAB LimMVAA)
    "r_pu", "x_pu", "b_pu",       # per-unit on s_base (MATLAB R/X/B)
    "kcmil", "cond_per_phase",    # conductor selection results
    "gmd_index", "gmd_ft",        # GMD draw (1..1000) and looked-up spacing
    "overloaded", "undersized",   # ResizeBranches diagnostics
    "max_dc_flow", "max_ac_flow", # max |flow| across scenarios (MaxDCPF/MaxACPF)
]

#: Columns GridTemper adds to the generator and gencost tables.
GEN_ADDED_COLUMNS = ["bus", "bus_ext"]          # 0-based internal / original PW number
GENCOST_ADDED_COLUMNS = ["bus", "bus_ext", "A", "B", "C"]  # quadratic cost: A + B·P + C·P²


def _to_parquet_safe(df: pd.DataFrame, path) -> None:
    """Write ``df`` to Parquet, expanding complex columns into re/im floats
    (Parquet has no complex type). Used by :meth:`GridModel.checkpoint`."""
    out = df
    complex_cols = [c for c in df.columns if np.iscomplexobj(df[c].to_numpy())]
    if complex_cols:
        out = df.copy()
        for c in complex_cols:
            vals = out.pop(c).to_numpy()
            out[f"{c}_re"] = np.real(vals)
            out[f"{c}_im"] = np.imag(vals)
    out.to_parquet(path, index=False)


class SingularNetworkError(RuntimeError):
    """Raised when the network is disconnected or the system matrix is
    singular — replaces MATLAB's ``SingularY`` / ``SingularB`` return flags
    (callers that tolerated them catch this instead)."""


@dataclass
class GridModel:
    """The case state passed between pipeline stages.

    Stages return a NEW GridModel (via :meth:`copy`) or document in their
    docstring exactly which columns they mutate — never both.
    """

    buses: pd.DataFrame
    branches: pd.DataFrame
    gens: pd.DataFrame
    gencost: pd.DataFrame
    #: Bus-level demand (from Demand_onetimepoint.csv); indexed like buses
    #: after renumbering. None until load_inputs runs.
    demand: pd.DataFrame | None = None
    #: external PW number -> 0-based internal id. Built by topology.numbering.
    ext_to_int: dict[int, int] = field(default_factory=dict)
    #: Free-form provenance notes (stage name -> summary) for run logs.
    log: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ sizes
    @property
    def n_bus(self) -> int:
        """Bus count — replaces every hard-coded MATLAB ``7132`` (quirk Q3)."""
        return len(self.buses)

    @property
    def n_branch(self) -> int:
        """Branch count — replaces ``params.TotalPaths``."""
        return len(self.branches)

    # ----------------------------------------------------------------- masks
    @property
    def is_line(self) -> np.ndarray:
        """Transmission-line mask — MATLAB ``TL = Branch.Tap == 0``."""
        return (self.branches["tap"] == 0).to_numpy()

    @property
    def is_transformer(self) -> np.ndarray:
        """Transformer mask — MATLAB ``XF = Branch.Tap ~= 0``."""
        return (self.branches["tap"] != 0).to_numpy()

    # ------------------------------------------------------------- utilities
    def copy(self) -> "GridModel":
        """Deep copy (stages that return new models start here)."""
        return _copy.deepcopy(self)

    def validate(self, expected_bus_count: int | None = None) -> None:
        """Invariant checks run between stages (cheap, always on).

        * added columns present once the owning stage has run,
        * from/to indices within [0, n_bus),
        * expected_bus_count from CaseConfig if provided (validation only).
        """
        nb = self.n_bus
        if {"from_bus", "to_bus"}.issubset(self.branches.columns):
            fb = self.branches["from_bus"].to_numpy()
            tb = self.branches["to_bus"].to_numpy()
            if fb.min() < 0 or tb.min() < 0 or fb.max() >= nb or tb.max() >= nb:
                raise ValueError("branch endpoints outside [0, n_bus)")
        if expected_bus_count is not None and nb != expected_bus_count:
            raise ValueError(
                f"bus count {nb} != expected {expected_bus_count} "
                "(CaseConfig.expected_bus_count; set to None to skip)"
            )

    def checkpoint(self, directory) -> None:
        """Write the tables to Parquet in ``directory`` (debugging aid that
        replaces inspecting the MATLAB workspace).

        Complex columns (e.g. demand ``S_pu``) are split into ``<col>_re`` /
        ``<col>_im`` float columns since Parquet has no complex type.
        """
        import pathlib

        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tables = {
            "buses": self.buses,
            "branches": self.branches,
            "gens": self.gens,
            "gencost": self.gencost,
        }
        if self.demand is not None:
            tables["demand"] = self.demand
        for name, df in tables.items():
            _to_parquet_safe(df, directory / f"{name}.parquet")


# ---------------------------------------------------------------------------
# Shared result objects (frozen — results are facts, not state)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispatchResult:
    """UCED output — replaces the ``[GenMWSetpoint, fval, Generators, iter]``
    return tuple (algorithms/03_dispatch.md)."""

    setpoints_mw: np.ndarray          # aligned to committed_gens rows
    committed_gens: pd.DataFrame      # surviving fleet ("GenCommit")
    objective: float                  # actual QP objective (MATLAB returned 0)
    decommit_iterations: int
    solver_statuses: tuple[str, ...]  # one entry per QP solve, for debugging


@dataclass(frozen=True)
class PowerFlowResult:
    """Common AC/DC power-flow output (algorithms/05, 06).

    Flows are MW/MVA on s_base (the MATLAB ``100 * ...`` convention).
    """

    v_bus: np.ndarray                 # complex bus voltages (DC: 1∠θ)
    flow_send_mva: np.ndarray         # complex send-end line flows
    flow_recv_mva: np.ndarray | None  # None for DC
    flow_max_mva: np.ndarray          # max(|send|,|recv|) per line — sizing input
    converged: bool
    iterations: int
    angle_diff_deg: np.ndarray        # |Aᵀ·angle| per line
    #: dS/dV from the last Jacobian (ACPowerFlowFinal.dVdQ input); AC-detailed only.
    ds_dv: sp.spmatrix | None = None
    history: list | None = None       # NewtonOptions.keep_history

"""GridTemper pipeline orchestration — main.m.

Algorithm doc: conversion_manual/algorithms/01_main_pipeline.md.
Assembled in P4, after all subsystems are converted and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .conductors.library import ConductorLibrary
from .config import CaseConfig, TemperOptions
from .dispatch.cost_curves import fit_quadratic_costs
from .dispatch.snet import build_snet
from .grid_model import DispatchResult, GridModel, PowerFlowResult
from .io.pw_csv import read_pw_table
from .powerflow.backends import ACBackendResult, solve_ac
from .refinement.reactive import ReactivePlacement
from .refinement.real_power import RealPowerLoop
from .topology.gsu_split import split_add_gsus
from .topology.incidence import build_incidence
from .topology.numbering import to_internal_numbering


@dataclass(frozen=True)
class GridTemperResult:
    """Everything a pipeline run produces: the refined grid, dispatch, both AC
    results, the incidence matrix, and a per-stage summary log."""

    model: GridModel
    snet: np.ndarray                 # complex per-bus injection (MW, generation positive)
    dispatch: DispatchResult
    acpf: PowerFlowResult            # MATLAB-faithful simple-path result (reactive stage)
    ac_solution: ACBackendResult | None  # well-posed AC solve via the selected backend
    incidence: sp.csc_matrix
    stage_log: dict = field(default_factory=dict)


class GridTemper:
    """Runs the stages of main.m in order (config.STAGES); each stage logs a
    one-line summary and can checkpoint the GridModel to Parquet."""

    def __init__(
        self,
        case: CaseConfig | None = None,
        options: TemperOptions | None = None,
        reference_ybus: sp.spmatrix | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        self.case = case or CaseConfig()
        self.options = options or TemperOptions()
        #: YBus_new dump for the AC simple-path override (quirk Q1); None computes it.
        self.reference_ybus = reference_ybus
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def run(self) -> GridTemperResult:
        """Execute the pipeline stages in order (algorithms/01):
        load → split GSUs → fit costs → renumber → SNet(UCED) → incidence →
        assign conductors → [real-power loop, dormant] → reactive placement.
        """
        log: dict = {}

        model = self._load_inputs()
        log["load_inputs"] = f"{model.n_bus} buses, {model.n_branch} branches (raw)"

        model = split_add_gsus(model, self.case.gsu_split_first_n_buses, self.case.pw_export_dir)
        log["split_gsus"] = f"{model.n_bus} buses, {model.n_branch} branches after GSU split"
        self._checkpoint(model, "split_gsus")

        model.gencost = fit_quadratic_costs(model.gencost)
        log["fit_costs"] = f"quadratic A/B/C for {len(model.gencost)} generators"

        model = to_internal_numbering(model)
        log["renumber"] = f"{len(model.ext_to_int)} buses mapped ext->int (0-based)"
        self._checkpoint(model, "renumber")

        snet, dispatch = build_snet(
            model.demand, model.gencost, self.options.dispatch, model.n_bus
        )
        log["build_snet"] = (
            f"{len(dispatch.committed_gens)} gens committed, "
            f"total gen {np.real(snet[snet.real > 0]).sum():.0f} MW"
        )

        A = build_incidence(
            model.branches["from_bus"].to_numpy(),
            model.branches["to_bus"].to_numpy(),
            model.n_bus,
        )
        log["build_incidence"] = f"incidence {A.shape[0]}x{A.shape[1]}, {A.nnz} nnz"

        library = ConductorLibrary.from_reference(self.case.voltage_levels)
        model.branches = library.assign(model.branches, self.options)
        n_assigned = int((model.branches["cond_index"].to_numpy() >= 0).sum())
        log["assign_conductors"] = f"{n_assigned} lines conductored"
        self._checkpoint(model, "assign_conductors")

        if self.options.run_real_power_loop:
            model = RealPowerLoop(self.case, self.options, library).run(model, A, snet)
            log["real_power_loop"] = "ran (flag on)"
        else:
            log["real_power_loop"] = "skipped (dormant, default)"

        model, acpf = ReactivePlacement(self.case, self.options, self.reference_ybus).run(
            model, A, snet
        )
        log["reactive_placement"] = (
            f"AC simple converged={acpf.converged}; "
            f"detailed={model.log.get('acpf_detailed', 'n/a')}"
        )
        self._checkpoint(model, "reactive_placement")

        # Real (well-posed) AC solve via the selected backend — this is the
        # working AC solution, unlike the MATLAB-faithful non-convergent path.
        ac_solution: ACBackendResult | None = None
        backend_kwargs: dict = {}
        if self.options.ac_backend == "powerworld" and self.options.powerworld_pwb:
            backend_kwargs["pwb_path"] = self.options.powerworld_pwb
        try:
            ac_solution = solve_ac(
                model, snet, self.case, backend=self.options.ac_backend, **backend_kwargs
            )
            log["ac_solve"] = (
                f"{ac_solution.backend}: converged={ac_solution.converged}, "
                f"losses={ac_solution.losses_mw:.1f} MW"
            )
        except Exception as exc:  # a backend may be unavailable (pandapower/PowerWorld)
            log["ac_solve"] = f"{self.options.ac_backend} unavailable: {exc}"

        model.validate(self.case.expected_bus_count)
        return GridTemperResult(
            model=model, snet=snet, dispatch=dispatch, acpf=acpf,
            ac_solution=ac_solution, incidence=A, stage_log=log,
        )

    # ------------------------------------------------------------- internals
    def _load_inputs(self) -> GridModel:
        """convertPWtoTable ×5 (main.m); add Tap/Resize to the branch table."""
        d = self.case.input_dir
        branches = read_pw_table(d / self.case.branch_csv)
        branches["Tap"] = 0
        branches["Resize"] = 0
        return GridModel(
            buses=read_pw_table(d / self.case.bus_csv),
            branches=branches,
            gens=read_pw_table(d / self.case.gen_csv),
            gencost=read_pw_table(d / self.case.gencost_csv),
            demand=read_pw_table(d / self.case.demand_csv),
        )

    def _checkpoint(self, model: GridModel, stage: str) -> None:
        if self.checkpoint_dir is not None:
            model.checkpoint(self.checkpoint_dir / stage)

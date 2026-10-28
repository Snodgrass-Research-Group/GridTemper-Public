"""DC-power-flow line-sizing loop — RealPowerLoop.m (dormant, D9).

Algorithm doc: conversion_manual/algorithms/07_refinement.md §RealPowerLoop.
Gated by ``TemperOptions.run_real_power_loop`` (default False = today's MATLAB,
where the call is commented out). The loop needs per-line ``length_km`` /
``gmd_index`` for RXB, which the base TX-2030 branch table does not carry — this
is why the stage is dormant; it is exercised on synthetic branches only.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..config import CaseConfig, ResizeOptions, TemperOptions
from ..conductors.library import ConductorLibrary
from ..conductors.line_params import compute_rxb
from ..conductors.resize import resize_branches
from ..conductors.transformers import size_transformers
from ..grid_model import GridModel
from ..io.reference_data import GmdData, load_gmd_data, load_xf_data
from ..powerflow.dc import DCPowerFlow


class RealPowerLoop:
    """Iterate {resize → re-conductor → recompute impedances → DC PF} until
    (almost) no line is overloaded beyond its resizable range; the give-up
    criterion loosens after ``resize.max_iterations`` passes."""

    def __init__(
        self,
        case: CaseConfig,
        options: TemperOptions,
        library: ConductorLibrary,
        gmd: GmdData | None = None,
    ):
        self.case = case
        self.options = options
        self.library = library
        self.gmd = gmd if gmd is not None else load_gmd_data(case.regions[0])

    def run(self, model: GridModel, A: sp.csc_matrix, snet: np.ndarray) -> GridModel:
        """RealPowerLoop.m. Requires branch columns ``length_km``/``gmd_index``/
        ``circuits``/``cond_index``/``MaxCondIndex``/``MinCondIndex`` for RXB and
        resizing. ``snet`` is complex per-bus injection (MW)."""
        m = model.copy()
        branches = m.branches
        xf_data, xf_xr = load_xf_data()
        s_base = self.case.s_base_mva

        # RealPowerLoop forces these regardless of the caller's options.
        resize_opts = ResizeOptions(
            percent_resize=self.options.resize.percent_resize,
            percent_change=self.options.resize.percent_change,
            conductor_margin=self.options.resize.conductor_margin,
            downsize_threshold=self.options.resize.downsize_threshold,
            random_resize=self.options.resize.random_resize,
            upsize=True,
            downsize=True,
        )
        rng = np.random.default_rng(self.case.rng_seed)

        def reconductor(b):
            tl = b["tap"].to_numpy() == 0 if "tap" in b.columns else (b["Tap"].to_numpy() == 0)
            b = self.library.assign(b, self.options)
            # RXB only on the transmission lines; transformers get sized separately.
            lines = b[tl].copy()
            lines = compute_rxb(lines, self.case, self.options, self.library, self.gmd)
            b.loc[tl, ["R", "X", "B", "LimMVAA"]] = lines[["R", "X", "B", "LimMVAA"]].to_numpy()
            b = size_transformers(b, xf_data, xf_xr)
            return b

        # Initial pass: MaxDCPF seeded with the current limits (DCPF block is
        # commented out in the source), one resize + re-conductor.
        branches["MaxDCPF"] = branches["LimMVAA"].to_numpy()
        branches = resize_branches(branches, resize_opts, branches["MaxDCPF"].to_numpy(), rng)
        branches = reconductor(branches)

        term_criteria = 0
        overloaded = self._overloaded(branches, branches["MaxDCPF"].to_numpy())
        it = 0
        dc = DCPowerFlow()
        while np.count_nonzero(
            overloaded & (branches["MaxCondIndex"].to_numpy() - branches["cond_index"].to_numpy() > 1)
        ) > term_criteria:
            branches = resize_branches(branches, resize_opts, branches["MaxDCPF"].to_numpy(), rng)
            branches = reconductor(branches)
            res = dc.solve(branches["X"].to_numpy(float), np.real(snet) / s_base, A)
            max_flows = np.abs(res.flow_send_mva)
            branches["MaxDCPF"] = max_flows
            overloaded = self._overloaded(branches, max_flows)
            it += 1
            if it > self.options.resize.max_iterations:
                term_criteria += 1
            if it > 500:  # hard safety cap (the MATLAB loop trusts TermCriteria)
                break

        if branches["Resize"].to_numpy().any():
            branches = reconductor(branches)

        m.branches = branches
        return m

    def _overloaded(self, branches, max_flow) -> np.ndarray:
        """``(LimMVAA − MaxDCPF·margin < 0) & (cond_index < MaxCondIndex)``."""
        lim = branches["LimMVAA"].to_numpy(float)
        cond = branches["cond_index"].to_numpy(float)
        max_cond = branches["MaxCondIndex"].to_numpy(float)
        margin = self.options.resize.conductor_margin
        return (lim - max_flow * margin < 0) & (cond < max_cond)

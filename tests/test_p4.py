"""P4 end-to-end: the full GridTemper pipeline from raw PW CSVs.

Per-stage numerics are validated in test_p2a/b/c and test_p3; this checks the
stages integrate correctly and the aggregate outputs match the dumps (allowing
for the documented DV11 dispatch divergence and the B6 non-convergent AC)."""

from __future__ import annotations

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from gridtemper.config import CaseConfig, TemperOptions
from gridtemper.pipeline import GridTemper


def _run(reference_dir):
    ybus_new = sp.load_npz(reference_dir / "ybus_new.npz")
    gt = GridTemper(CaseConfig(), TemperOptions(), reference_ybus=ybus_new)
    return gt.run()


def test_pipeline_runs_end_to_end(reference_dir):
    result = _run(reference_dir)
    m = result.model
    assert m.n_bus == 7132
    assert m.n_branch == 9555
    # every stage logged
    expected_stages = {
        "load_inputs", "split_gsus", "fit_costs", "renumber", "build_snet",
        "build_incidence", "assign_conductors", "real_power_loop", "reactive_placement",
    }
    assert expected_stages.issubset(result.stage_log.keys())
    # invariants hold (validate() would raise otherwise)
    m.validate(expected_bus_count=7132)


def test_pipeline_dispatch_and_conductors(reference_dir):
    result = _run(reference_dir)
    # dispatch produced a committed fleet (~276; DV11 shifts a couple of gens)
    assert 260 <= len(result.dispatch.committed_gens) <= 290
    # conductors assigned to the resize lines (cond_index 0-based, -1 = unassigned)
    cond = result.model.branches["cond_index"].to_numpy()
    assert cond.min() == -1 and cond.max() >= 0
    assert int((cond >= 0).sum()) > 0


def test_pipeline_snet_close_to_stage5(reference_dir):
    """End-to-end SNet vs stage5. The pipeline uses GridTemper's own cost curves,
    so the DV11 flat-curve generators shift dispatch slightly — the bulk of buses
    still match and total injection is conserved."""
    result = _run(reference_dir)
    ref = sio.loadmat(reference_dir / "stage5_snet.mat")["SNet"].ravel()
    diff = np.abs(result.snet.real - ref.real)
    # net injection ~ 0 on both sides (generation balances load)
    assert abs(result.snet.real.sum()) < 1.0
    assert abs(ref.real.sum()) < 1.0
    # the vast majority of buses match; DV11 shifts only a handful
    n_close = int((diff < 1.0).sum())
    assert n_close > 0.98 * len(diff), f"only {n_close}/{len(diff)} buses within 1 MW"


def test_pipeline_final_ac_nonconvergent(reference_dir):
    """The final AC stage reproduces MATLAB's non-convergence (B6): MaxACPF NaN,
    bus types = 7130 PV + 2 slack."""
    result = _run(reference_dir)
    assert np.all(np.isnan(result.model.branches["MaxACPF"].to_numpy()))
    bt = result.model.buses["bus_type"].to_numpy()
    assert int((bt == 2).sum()) == 7130
    assert int((bt == 3).sum()) == 2

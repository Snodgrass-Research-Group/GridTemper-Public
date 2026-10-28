"""Thin CLI: build the TX-2030 config and run GridTemper end-to-end.

    python -m uv run python scripts/run_tx2030.py [--reference-ybus] [--checkpoint DIR]

``--reference-ybus`` loads the YBus_new.m dump (tests/reference/ybus_new.npz) so
the AC simple path reproduces the MATLAB behavior (quirk Q1 / DV10); without it
the Ybus is computed from branch data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scipy.sparse as sp

from gridtemper.config import AC_BACKENDS, CaseConfig, TemperOptions, load_toml
from gridtemper.pipeline import GridTemper


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, build the config, run the pipeline, and print a summary."""
    parser = argparse.ArgumentParser(description="Run the GridTemper TX-2030 pipeline.")
    parser.add_argument(
        "--ac-backend",
        choices=AC_BACKENDS,
        default=None,
        help="AC power-flow solver (default builtin, or from --config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="gridtemper.toml with solver/case/dispatch overrides",
    )
    parser.add_argument(
        "--reference-ybus",
        action="store_true",
        help="use the YBus_new.m dump for the AC simple path (MATLAB parity)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="directory to write per-stage Parquet checkpoints",
    )
    args = parser.parse_args(argv)

    case, options = load_toml(args.config) if args.config else (CaseConfig(), TemperOptions())
    if args.ac_backend:  # CLI overrides the TOML/default
        options.ac_backend = args.ac_backend

    reference_ybus = None
    if args.reference_ybus:
        path = Path(__file__).resolve().parents[1] / "tests" / "reference" / "ybus_new.npz"
        if not path.exists():
            print(f"ERROR: {path} not found — run scripts/convert_mat_data.py first", file=sys.stderr)
            return 1
        reference_ybus = sp.load_npz(path)

    gt = GridTemper(
        case,
        options,
        reference_ybus=reference_ybus,
        checkpoint_dir=args.checkpoint,
    )
    result = gt.run()

    print("=== GridTemper TX-2030 ===")
    for stage, summary in result.stage_log.items():
        print(f"  {stage:20s} {summary}")
    print(f"final: {result.model.n_bus} buses, {result.model.n_branch} branches, "
          f"objective ${result.dispatch.objective:,.0f}")
    if result.ac_solution is not None:
        r = result.ac_solution
        print(f"AC ({r.backend}): converged={r.converged}, losses={r.losses_mw:.1f} MW")
    return 0


if __name__ == "__main__":
    sys.exit(main())

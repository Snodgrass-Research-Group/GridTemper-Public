"""Shared fixtures: MATLAB reference dumps + tiny synthetic cases.

Reference dumps are produced once by matlab_reference/dump_reference.m into
tests/reference/. Tests that need a dump skip cleanly when it hasn't been
generated, so the suite runs without MATLAB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).parent / "reference"
MATLAB_CSV_DIR = REFERENCE_DIR / "matlab_csv"


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    if not REFERENCE_DIR.exists() or not any(REFERENCE_DIR.iterdir()):
        pytest.skip(
            "MATLAB reference dumps not generated yet — run "
            "matlab_reference/dump_reference.m"
        )
    return REFERENCE_DIR


@pytest.fixture(scope="session")
def matlab_csv_dir() -> Path:
    """Directory of the stage-by-stage CSV dumps (skip if not generated)."""
    if not MATLAB_CSV_DIR.exists() or not any(MATLAB_CSV_DIR.glob("stage*.csv")):
        pytest.skip(
            "MATLAB CSV dumps not generated yet — run "
            "matlab_reference/dump_reference.m"
        )
    return MATLAB_CSV_DIR


@pytest.fixture(scope="session")
def input_dir() -> Path:
    """The PowerWorld CSV inputs the pipeline reads (main.m)."""
    from gridtemper.config import CaseConfig

    return CaseConfig().input_dir


@pytest.fixture(scope="session")
def raw_model(input_dir):
    """A GridModel of the raw PW tables, before any transform (main.m step 2).

    Branches get the ``Tap``/``Resize`` columns main.m adds right after reading.
    """
    from gridtemper.grid_model import GridModel
    from gridtemper.io.pw_csv import read_pw_table

    branches = read_pw_table(input_dir / "Branches.csv")
    branches["Tap"] = 0
    branches["Resize"] = 0
    return GridModel(
        buses=read_pw_table(input_dir / "Buses.csv"),
        branches=branches,
        gens=read_pw_table(input_dir / "GenBuses.csv"),
        gencost=read_pw_table(input_dir / "GenCostCurves.csv"),
        demand=read_pw_table(input_dir / "Demand_onetimepoint.csv"),
    )

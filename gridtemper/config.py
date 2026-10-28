"""GridTemper run configuration.

Replaces three MATLAB mechanisms:

* the hand-edited setup block at the top of ``main.m`` (region, file paths,
  ``codectrl``, the hard-coded case constants — quirk Q3),
* the ``options`` struct from ``InitOptions.m``,
* the ``params`` struct loaded from ``QNCAResults.mat`` (only the fields the
  TX-2030 pipeline actually reads: everything else in that file is legacy
  QNCA state).

Two dataclasses, mirroring the gridcrucible convention (D17):

* :class:`CaseConfig` — *what system* to build: file paths, region, slack bus,
  voltage levels. A new case is a new ``CaseConfig``, never a code edit (D7).
* :class:`TemperOptions` — *how to run*: solver choices, loop knobs, feature
  flags. Defaults reproduce today's MATLAB behavior exactly, including the
  dormant stages being off (D9).

Sizes (bus counts, path counts) are ALWAYS derived from the data, never
configured — the MATLAB ``7132`` literals do not survive conversion.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

#: Valid ``ac_backend`` values (see TemperOptions.ac_backend).
AC_BACKENDS = ("builtin", "pandapower", "powerworld")

#: Repo root anchored on this file (GridTemper/gridtemper/config.py) so
#: relative input paths resolve regardless of the working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directory with the converted reference data (produced once by
#: ``scripts/convert_mat_data.py``).
DATA_DIR = Path(__file__).resolve().parent / "data"

#: Pipeline stage names in execution order (algorithms/01_main_pipeline.md).
STAGES = (
    "load_inputs",          # convertPWtoTable ×5
    "split_gsus",           # split_add_GSUs
    "fit_costs",            # costcurve_to_cubic
    "renumber",             # main.m internal-numbering loop
    "build_snet",           # Create_SNet → UCED
    "build_incidence",      # BuildANet
    "assign_conductors",    # CreateConductorData + AssignConductors
    "real_power_loop",      # RealPowerLoop (dormant — off by default)
    "reactive_placement",   # ReactivePowerPlacement (final active stage)
)


@dataclass
class CaseConfig:
    """What system to build. Defaults are the TX-2030 case from ``main.m``."""

    name: str = "TX-2030"
    #: EPIGRIDS region(s); index 0 selects the GMD lookup table (RXB.m).
    regions: list[str] = field(default_factory=lambda: ["SPP"])
    #: Voltage levels present in the system (InitOptions.VoltageLevels).
    voltage_levels: list[int] = field(default_factory=lambda: [69, 138, 345])
    #: Max line length (miles) per voltage level; must match voltage_levels.
    max_length_miles: list[float] = field(default_factory=lambda: [25, 65, 90])

    #: Input CSVs exported from PowerWorld (main.m lines 11-14, 30). Copied from
    #: the source repo's ``Functions/Input data from PW/`` into ``inputs/``.
    input_dir: Path = REPO_ROOT / "inputs"
    gen_csv: str = "GenBuses.csv"
    gencost_csv: str = "GenCostCurves.csv"
    bus_csv: str = "Buses.csv"
    branch_csv: str = "Branches.csv"
    demand_csv: str = "Demand_onetimepoint.csv"

    #: Slack bus as an EXTERNAL PowerWorld number (main.m hard-codes internal
    #: 1747; resolved to a 0-based internal index at load time — quirk Q3).
    slack_bus_internal_matlab: int = 1747

    #: split_add_GSUs candidate cutoff: only the first N unique generator bus
    #: numbers are split ("first 88 rows / up to bus 89, bus 63 missing").
    gsu_split_first_n_buses: int = 88

    #: Expected bus count AFTER GSU splitting — validation only, never sizing.
    #: (MATLAB hard-codes 7132 in Create_SNet/ReactivePowerPlacement.)
    expected_bus_count: int | None = 7132

    s_base_mva: float = 100.0
    #: Reproducibility seed for every stochastic choice (ResizeBranches
    #: shuffling). MATLAB used unseeded rand — D15/DEVIATIONS applies.
    rng_seed: int = 2030

    #: Optional directory to write the modified PW tables (split_add_GSUs
    #: wrote to 'Powerworld cases/' unconditionally — quirk Q4). None = don't.
    pw_export_dir: Path | None = None

    def __post_init__(self) -> None:
        if len(self.voltage_levels) != len(self.max_length_miles):
            # InitOptions.m kept this validation; so do we.
            raise ValueError(
                "Each voltage level needs an associated maximum line length: "
                f"{self.voltage_levels} vs {self.max_length_miles}"
            )


@dataclass
class NewtonOptions:
    """Newton-Raphson solver knobs (pfsolve.m / UpdateVBus.m)."""

    tol: float = 5e-5           # pfsolve.m mismatch tolerance
    max_iter: int = 100         # pfsolve.m iteration cap
    #: Under-relaxation factor — MATLAB hard-codes 0.1 via a global (quirk
    #: Q2). 1.0 = full Newton; keep 0.1 to reproduce MATLAB convergence.
    damping: float = 0.1
    keep_history: bool = False  # replaces the MissMatrix/VBusMatrix globals


@dataclass
class DispatchOptions:
    """Economic dispatch / UCED knobs."""

    #: 'auto' = gurobipy when importable, else the built-in fallback (D2).
    backend: str = "auto"       # 'auto' | 'gurobi' | 'fallback'
    reserve: float = 0.15       # options.Reserve
    #: Generators removed per decommitment step (UCED maxk(...,3)).
    decommit_batch: int = 3
    min_fleet_size: int = 5     # UCED while-loop floor
    zero_dispatch_tol: float = 1e-3


@dataclass
class ResizeOptions:
    """Branch resizing knobs (ResizeBranches.m / RealPowerLoop.m)."""

    percent_resize: float = 0.05     # options.PercentResize
    percent_change: float = 1.3      # options.PercentChange
    conductor_margin: float = 1.0    # options.ConductorMargin
    downsize_threshold: float = 0.5  # options.DownsizeThreshold
    random_resize: bool = True       # options.RandomResize (seeded in Python)
    upsize: bool = True
    downsize: bool = True
    max_iterations: int = 25         # RealPowerLoop soft cap before TermCriteria grows


@dataclass
class ReactiveOptions:
    """ReactivePowerPlacement knobs."""

    high_q_mvar: float = 800.0       # ACPF.HighQ — shunt reactor limit
    low_q_mvar: float = -1200.0      # ACPF.LowQ — shunt capacitor limit
    low_voltage_threshold: float = 0.8
    #: The full placement loop ('while Go') never runs in MATLAB (Go = 0).
    run_placement_loop: bool = False
    per_switch: float = 0.05         # PerSwitch — sync-cond demotion fraction
    long_line: bool = True           # ACPF.LongLine — include line charging
    lossless_lines: bool = False     # ACPF.LosslessLines


@dataclass
class TemperOptions:
    """How to run. Field-by-field port of InitOptions.m plus the flags that
    controlled dormant code paths (D9). Grouped sub-options above."""

    newton: NewtonOptions = field(default_factory=NewtonOptions)
    dispatch: DispatchOptions = field(default_factory=DispatchOptions)
    resize: ResizeOptions = field(default_factory=ResizeOptions)
    reactive: ReactiveOptions = field(default_factory=ReactiveOptions)

    # --- dormant-stage gates (all False = today's MATLAB behavior) ---
    run_real_power_loop: bool = False    # RealPowerLoop call is commented out
    export_matpower: bool = False        # Table2Matpower call is commented out
    add_reactive_support: bool = False   # ACAddReactiveSupport is MISSING (08 doc)
    #: Reproduce MATLAB bit-for-bit by loading the YBus_new.m reference matrix
    #: instead of computing Ybus (quirk Q1 / DEVIATIONS DV10). Validation only.
    use_reference_ybus: bool = False

    #: AC power-flow solver for the pipeline's real (well-posed) AC solve:
    #: 'builtin' (scipy NR, no extra deps) | 'pandapower' | 'powerworld' (SimAuto).
    #: Override in code, via gridtemper.toml, or the --ac-backend CLI flag.
    ac_backend: str = "builtin"
    #: For ac_backend='powerworld': open+solve this existing .PWB instead of
    #: building the processed case (whose creation some SimAuto installs refuse).
    powerworld_pwb: str | None = None

    # --- physics / misc (InitOptions.m) ---
    change_voltages: bool = False        # options.ChangeVoltages
    routing_factor: float = 1.0          # options.RoutingFactor
    length_modifier: float = 1.0         # options.LengthModifier
    theta_max_deg: float = 45.0          # options.ThetaMax (degrees — see 01 doc)
    n_workers: int = 1                   # collapses Parallel/parallelPC/ParallelLinux (D12)
    n_scenarios: int = 1                 # params.Scenarios
    debug: bool = False                  # codectrl.RunDebuggingCode analog

    def __post_init__(self) -> None:
        if self.ac_backend not in AC_BACKENDS:
            raise ValueError(f"ac_backend must be one of {AC_BACKENDS}, got {self.ac_backend!r}")


def load_toml(path: str | Path) -> tuple[CaseConfig, TemperOptions]:
    """Load a ``gridtemper.toml`` into ``(CaseConfig, TemperOptions)``.

    Recognized keys (all optional)::

        [solver]
        ac_backend = "pandapower"    # builtin | pandapower | powerworld

        [case]
        slack_bus = 1747             # 1-based internal slack index
        rng_seed  = 2030

        [dispatch]
        backend = "auto"             # auto | gurobi | fallback
        reserve = 0.15

    Unknown keys are ignored. This is the file-based form of the AC-backend
    toggle; equivalent to setting ``TemperOptions.ac_backend`` in code.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    case = CaseConfig()
    options = TemperOptions()

    solver = data.get("solver", {})
    if "ac_backend" in solver:
        options.ac_backend = str(solver["ac_backend"]).lower()
    if "newton_damping" in solver:
        options.newton.damping = float(solver["newton_damping"])
    if "powerworld_pwb" in solver:
        options.powerworld_pwb = str(solver["powerworld_pwb"])

    c = data.get("case", {})
    if "slack_bus" in c:
        case.slack_bus_internal_matlab = int(c["slack_bus"])
    if "rng_seed" in c:
        case.rng_seed = int(c["rng_seed"])

    d = data.get("dispatch", {})
    if "backend" in d:
        options.dispatch.backend = str(d["backend"]).lower()
    if "reserve" in d:
        options.dispatch.reserve = float(d["reserve"])

    options.__post_init__()  # revalidate ac_backend
    return case, options

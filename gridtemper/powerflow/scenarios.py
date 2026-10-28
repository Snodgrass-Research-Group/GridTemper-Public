"""Solve independent power-flow scenarios, optionally in parallel (D12).

Two complementary tools:

* :func:`solve_scenarios` — a spawn-safe process-pool map for *independent*
  solves (e.g. AC snapshots, whose Jacobians differ so nothing is shared).
* :meth:`gridtemper.powerflow.dc.DCPowerFlow.solve_many` — batches many DC
  scenarios through a *single* Bbus factorization (they share the system
  matrix), which is far faster than re-solving each.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def solve_scenarios(
    solve_one: Callable[[T], R],
    scenarios: Sequence[T],
    n_workers: int = 1,
) -> list[R]:
    """Apply ``solve_one`` to each scenario, in parallel when worthwhile.

    Runs serially when ``n_workers <= 1`` or there is a single scenario;
    otherwise a spawn-safe ``ProcessPoolExecutor`` with ``n_workers`` workers.

    ``solve_one`` must be picklable (a module-level function or a picklable
    callable such as a frozen dataclass) — a spawn requirement on Windows.
    """
    items = list(scenarios)
    if n_workers <= 1 or len(items) <= 1:
        return [solve_one(s) for s in items]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(solve_one, items))

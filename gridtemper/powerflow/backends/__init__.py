"""Selectable AC power-flow backends.

Three solvers behind one interface (chosen via ``TemperOptions.ac_backend`` /
``gridtemper.toml`` / ``--ac-backend``):

* ``builtin``    — the from-scratch scipy Newton-Raphson solver (no extra deps).
* ``pandapower`` — pandapower's ``runpp`` (extra ``[pandapower]``).
* ``powerworld`` — PowerWorld Simulator via SimAuto (extra ``[powerworld]``,
  Windows + a PowerWorld install).

Unlike the MATLAB-faithful ReactivePlacement path (all-PV, fixed |V|=1, which
does not converge — blocker B6), every backend solves the same *well-posed*
:class:`ACCase`: one slack bus, committed generators as PV, loads as PQ. This
gives an actual AC solution and lets the three solvers be compared directly.
"""

from __future__ import annotations

from .base import ACBackendResult, ACCase, ACSolverBackend, get_backend, solve_ac

__all__ = ["ACBackendResult", "ACCase", "ACSolverBackend", "get_backend", "solve_ac"]

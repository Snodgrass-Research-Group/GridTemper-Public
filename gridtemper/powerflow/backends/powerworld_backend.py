"""PowerWorld Simulator AC backend via SimAuto (COM automation).

Two modes:

* **build** (default) — generate a PowerWorld ``.aux`` from the :class:`ACCase`,
  load it into a fresh SimAuto case, solve, and read back voltages/flows. This
  needs a SimAuto install that permits *building* a case from automation; some
  (e.g. this dev machine) can open+solve existing cases but silently refuse to
  create objects, in which case ``solve`` raises a clear error.
* **open** (``pwb_path=...``) — open an existing ``.PWB`` case, solve it, and
  return its full solution. Works wherever SimAuto opens cases. The returned
  arrays are sized to the *opened case*, not the passed ACCase.

Requires ``pywin32`` (extra ``[powerworld]``) and a PowerWorld install;
``available()`` is False otherwise, so the backend can be skipped cleanly.

R/X/C are per-unit on the 100 MVA system base — the same base PowerWorld uses by
default — so they map straight through. Bus numbers are 1-based in PowerWorld
(our 0-based id + 1).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .base import ACBackendResult, ACCase


class PowerWorldBackend:
    """AC backend that solves via PowerWorld Simulator + SimAuto (see module
    docstring) in either build mode or open-existing-``.PWB`` mode."""

    name = "powerworld"

    #: Tried in order; the versioned ProgID targets the full Simulator.
    PROG_IDS = ("pwrworld.SimulatorAuto.24", "pwrworld.SimulatorAuto")

    def __init__(self, solve_method: str = "RECTNEWT", prog_id: str | None = None,
                 pwb_path: str | None = None, keep_aux: bool = False):
        self.solve_method = solve_method
        self.prog_id = prog_id
        #: If set, ``solve`` opens this .PWB and solves it (open mode) instead of
        #: building the ACCase (build mode).
        self.pwb_path = pwb_path
        self.keep_aux = keep_aux

    def _dispatch(self):
        import win32com.client as win32

        ids = (self.prog_id,) if self.prog_id else self.PROG_IDS
        last: Exception | None = None
        for pid in ids:
            try:
                return win32.Dispatch(pid)
            except Exception as exc:  # pragma: no cover - COM env dependent
                last = exc
        raise RuntimeError(f"could not dispatch PowerWorld SimAuto ({ids}): {last}")

    def available(self) -> bool:
        """``True`` if pywin32 can dispatch a PowerWorld SimAuto COM object here."""
        try:
            self._dispatch()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ aux
    def _write_aux(self, case: ACCase, path: Path) -> np.ndarray:
        """Write the case to a PowerWorld .aux file, using PowerWorld's own
        concise field names (verified against a ``SaveData`` round-trip). Returns
        the per-branch circuit ids assigned (to match flows back afterward)."""
        n1 = case.slack_bus + 1  # 1-based slack
        lines: list[str] = []

        # Buses ---------------------------------------------------------------
        lines.append("Bus (Number,Name,NomkV,Slack)")
        lines.append("{")
        for b in range(case.n_bus):
            slack = "YES" if (b + 1) == n1 else "NO"
            lines.append(f'  {b + 1} "Bus {b + 1}" {case.vn_kv[b]:.4f} "{slack}"')
        lines.append("}")

        # Generators (PV buses + a reference gen at the slack) ----------------
        lines.append("Gen (BusNum,ID,MW,VoltSet,Status,MWMax,MWMin,MvarMax,MvarMin)")
        lines.append("{")
        gen_p = {int(b): float(p) for b, p in zip(case.gen_bus, case.gen_p_mw)}
        for b in sorted(set(gen_p) | {case.slack_bus}):
            mw = gen_p.get(b, 0.0)
            lines.append(
                f'  {b + 1} "1" {mw:.4f} {case.v_setpoint:.4f} "Closed" 99900 -99900 99900 -99900'
            )
        lines.append("}")

        # Loads ---------------------------------------------------------------
        lines.append("Load (BusNum,ID,SMW,SMvar,Status)")
        lines.append("{")
        for b, p, q in zip(case.load_bus, case.load_p_mw, case.load_q_mvar):
            lines.append(f'  {int(b) + 1} "1" {float(p):.4f} {float(q):.4f} "Closed"')
        lines.append("}")

        # Branches (unique circuit id per parallel pair, for flow matching) ---
        lines.append("Branch (BusNumFrom,BusNumTo,Circuit,R,X,B,Status)")
        lines.append("{")
        pair_count: dict[tuple[int, int], int] = {}
        circuits = np.empty(case.n_branch, dtype=int)
        for i in range(case.n_branch):
            f, t = int(case.from_bus[i]), int(case.to_bus[i])
            pair_count[(f, t)] = pair_count.get((f, t), 0) + 1
            circuits[i] = pair_count[(f, t)]
            lines.append(
                f'  {f + 1} {t + 1} "{circuits[i]}" '
                f"{case.r_pu[i]:.8f} {case.x_pu[i]:.8f} {case.b_pu[i]:.8f} \"Closed\""
            )
        lines.append("}")
        path.write_text("\n".join(lines))
        return circuits

    # ---------------------------------------------------------------- solve
    def solve(self, case: ACCase) -> ACBackendResult:
        """Solve ``case`` via SimAuto: open + solve ``pwb_path`` if set (open
        mode), else build a ``.aux`` from ``case``, load, and solve (build mode)."""
        if self.pwb_path is not None:
            return self._solve_pwb(self.pwb_path)
        tmpdir = Path(tempfile.mkdtemp(prefix="gridtemper_pw_"))
        aux = tmpdir / "case.aux"
        circuits = self._write_aux(case, aux)

        sa = self._dispatch()
        try:
            _check(sa.RunScriptCommand("NewCase;"))
            _check(sa.RunScriptCommand("EnterMode(EDIT);"))
            _check(sa.ProcessAuxFile(str(aux)))
            # Verify the case was actually built. Some SimAuto installs/licenses
            # open+solve existing cases fine but cannot CREATE objects from
            # automation (aux/ChangeParameters silently no-op) — fail loudly
            # rather than "solve" an empty case.
            built = sa.GetParametersMultipleElement("Bus", ["BusNum"], "")
            if _err(built) or built[1] is None or len(built[1][0]) != case.n_bus:
                got = 0 if (_err(built) or built[1] is None) else len(built[1][0])
                raise RuntimeError(
                    f"SimAuto did not build the case ({got}/{case.n_bus} buses created). "
                    "This PowerWorld/SimAuto install can open+solve existing .PWB cases but "
                    "does not support building a case from automation. Use a SimAuto that "
                    "allows case creation, or open a .PWB template."
                )
            _check(sa.RunScriptCommand("EnterMode(RUN);"))
            solve_out = sa.RunScriptCommand(f"SolvePowerFlow({self.solve_method});")
            converged = not _err(solve_out)

            v_bus = _read_bus_voltages(sa, case.n_bus)
            send, recv = _read_branch_flows(sa, case, circuits)
        finally:
            try:
                sa.CloseCase()
            except Exception:
                pass
            if not self.keep_aux:
                try:
                    aux.unlink()
                    tmpdir.rmdir()
                except OSError:
                    pass

        flow_max = np.maximum(np.abs(send), np.abs(recv))
        losses = float(np.nansum(np.real(send + recv)))  # from+to sum ≈ series loss
        return ACBackendResult(
            backend=self.name,
            converged=bool(converged),
            v_bus=v_bus,
            flow_from_mva=send,
            flow_to_mva=recv,
            flow_max_mva=flow_max,
            losses_mw=losses,
        )

    # ------------------------------------------------------------- open mode
    def _solve_pwb(self, pwb_path: str) -> ACBackendResult:
        """Open an existing .PWB case, solve it, and return its full solution.
        Result arrays are sized to the opened case (its own bus/branch order)."""
        sa = self._dispatch()
        try:
            _check(sa.OpenCase(str(pwb_path)))
            solve_out = sa.RunScriptCommand(f"SolvePowerFlow({self.solve_method});")
            converged = not _err(solve_out)

            bus = sa.GetParametersMultipleElement("Bus", ["BusPUVolt", "BusAngle"], "")
            if _err(bus) or bus[1] is None:
                raise RuntimeError(f"could not read buses from {pwb_path}: {_err(bus)}")
            vm = np.array([float(x) for x in bus[1][0]])
            va = np.deg2rad([float(x) for x in bus[1][1]])
            v_bus = vm * np.exp(1j * va)

            br = sa.GetParametersMultipleElement(
                "Branch", ["LineMW", "LineMVR", "LineMW:1", "LineMVR:1"], ""
            )
            if _err(br) or br[1] is None:
                send = recv = np.zeros(0, dtype=complex)
            else:
                send = np.array([float(x) for x in br[1][0]]) + 1j * np.array(
                    [float(x) for x in br[1][1]]
                )
                recv = np.array([float(x) for x in br[1][2]]) + 1j * np.array(
                    [float(x) for x in br[1][3]]
                )
        finally:
            try:
                sa.CloseCase()
            except Exception:
                pass

        return ACBackendResult(
            backend=self.name,
            converged=bool(converged),
            v_bus=v_bus,
            flow_from_mva=send,
            flow_to_mva=recv,
            flow_max_mva=np.maximum(np.abs(send), np.abs(recv)),
            losses_mw=float(np.nansum(np.real(send + recv))),
        )


# --------------------------------------------------------------- SimAuto glue
def _err(result) -> str:
    """SimAuto returns (ErrorString, data...); non-empty ErrorString == failure."""
    if isinstance(result, (tuple, list)) and result:
        return str(result[0])
    return ""


def _check(result) -> None:
    msg = _err(result)
    if msg:
        raise RuntimeError(f"SimAuto error: {msg}")


def _read_bus_voltages(sa, n_bus: int) -> np.ndarray:
    out = sa.GetParametersMultipleElement("Bus", ["BusNum", "BusPUVolt", "BusAngle"], "")
    if _err(out) or out[1] is None:
        return np.full(n_bus, np.nan, dtype=complex)
    num = np.array([int(float(x)) for x in out[1][0]])
    vm = np.array([float(x) for x in out[1][1]])
    va = np.deg2rad([float(x) for x in out[1][2]])
    v = np.full(n_bus, np.nan, dtype=complex)
    v[num - 1] = vm * np.exp(1j * va)
    return v


def _read_branch_flows(sa, case: ACCase, circuits: np.ndarray):
    fields = ["BusNum", "BusNum:1", "LineCircuit", "LineMW", "LineMVR", "LineMW:1", "LineMVR:1"]
    out = sa.GetParametersMultipleElement("Branch", fields, "")
    send = np.full(case.n_branch, np.nan, dtype=complex)
    recv = np.full(case.n_branch, np.nan, dtype=complex)
    if _err(out) or out[1] is None:
        return send, recv
    cols = out[1]
    f = np.array([int(float(x)) for x in cols[0]]) - 1
    t = np.array([int(float(x)) for x in cols[1]]) - 1
    circ = np.array([int(str(x).strip()) for x in cols[2]])
    p_from = np.array([float(x) for x in cols[3]])
    q_from = np.array([float(x) for x in cols[4]])
    p_to = np.array([float(x) for x in cols[5]])
    q_to = np.array([float(x) for x in cols[6]])
    lookup = {(f[k], t[k], circ[k]): k for k in range(len(f))}
    for i in range(case.n_branch):
        k = lookup.get((int(case.from_bus[i]), int(case.to_bus[i]), int(circuits[i])))
        if k is not None:
            send[i] = p_from[k] + 1j * q_from[k]
            recv[i] = p_to[k] + 1j * q_to[k]
    return send, recv

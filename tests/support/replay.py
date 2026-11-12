"""A REAL RUN'S OUTPUT, SHOWN AS IT GREW (`process/testing.md` § 6).

The monitor and the viewers read a run while it runs, so a test of either
needs the run's output as it stood at a line and then a little later.  This copies a run made on the
road (`support/real_runs.py`) into a folder of its own, its output written up
to a line, with ``grow(to)`` writing on as the engine would.  What the monitor
itself writes -- its log and its samples -- and the timing tee, which grows
with the output, are left for the replay to write.

Used by ``tests/test_monitor_watches_a_live_run_e2e.py`` (a relaxation),
``tests/test_the_viewers_follow_a_real_run_e2e.py`` (a relaxation, as the
viewer reads it) and ``tests/test_transport_on_a_real_junction_e2e.py`` (a
transmission, a device).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable, List, Tuple


def output_lines(run: Path) -> List[str]:
    """The run's own output, line by line, its line ends kept -- the run
    door's (`runs.run_of`) engine output."""
    from molbuilder.runs import run_of
    return run_of(run).stdout.read_text(
        errors="replace").splitlines(keepends=True)


def replay(run: Path, tmp_path: Path, upto: int
           ) -> Tuple[Path, "object", Callable[[int], None]]:
    """``(folder, watched, grow)``: the run's folder copied under
    ``tmp_path`` with its output written to line ``upto``, the monitor's
    `WatchedRun` of it, and ``grow(to)``, which writes on to line ``to``."""
    from molbuilder import monitor
    from molbuilder.runs import run_of
    ran = run_of(run)
    there = tmp_path / run.name
    shutil.copytree(run, there, ignore=shutil.ignore_patterns(
        ran.stdout.name, "*.monitor.log", "*.util.csv", "*.scf-timing.log"))
    lines = output_lines(run)
    out = there / ran.stdout.name
    out.write_text("".join(lines[:upto]))
    at = {"n": upto}

    def grow(to: int) -> None:
        with out.open("a") as fh:
            fh.write("".join(lines[at["n"]:to]))
        at["n"] = to
    return there, monitor.WatchedRun(names=ran.names,
                                     directory=there), grow


class Output:
    """A SIESTA run's own output, and where its lines fall: the SCF rows,
    and the geometry moves SIESTA began (``Begin <kind> opt. move = N``) --
    where a test cuts a replay (:func:`replay`) to show a run as it stood."""

    def __init__(self, run: Path):
        self.lines = output_lines(run)
        self.rows = [i for i, ln in enumerate(self.lines)
                     if ln.lstrip().startswith("scf:")]
        self.moves = [i for i, ln in enumerate(self.lines)
                      if re.match(r"\s*Begin \w+ opt\. move", ln)]

    def rows_of(self, k: int):
        """The SCF rows of move ``k``."""
        end = (self.moves[k + 1] if k + 1 < len(self.moves)
               else len(self.lines))
        return [r for r in self.rows if self.moves[k] < r < end]

    def through(self, line: int) -> int:
        """The ``upto`` that writes line ``line`` and everything before it."""
        return line + 1

    def end_of_move(self, k: int) -> int:
        return self.through(self.rows_of(k)[-1])

    def criterion(self, name: str) -> float:
        """A criterion as the run's own ``redata:`` line states it --
        ``name`` a pattern for SIESTA's words before ``tolerance``."""
        for ln in self.lines:
            m = re.match(rf"\s*redata:\s*(?:{name})\s+tolerance\b[^=]*=\s*(\S+)",
                         ln)
            if m:
                return float(m.group(1))
        raise AssertionError(f"no redata line states the {name} tolerance")


def status_lines(log: Path) -> List[str]:
    """The [STATUS] lines a monitor wrote to its log."""
    return [ln for ln in log.read_text().splitlines() if "[STATUS]" in ln]


def statuses(watched) -> List[str]:
    """The [STATUS] lines the monitor wrote to the run's monitor log."""
    return status_lines(watched.path(".monitor.log"))

"""The one walk script empties a run's scratch folder as each round
begins and when the walk ends -- `engines/transport.md` § 2a.11, *What a
run's points share*: the emptying is a step of its own before the first
point of the round this walk runs, and one after the last step, a failed
member's included; it removes the file family alone and writes what it
did to the walk's log.

Driven as the launch drives it: the script rendered by `submit._walk_script`
with the steps a round-by-round walk gives it, run by bash over members
that do nothing, the log read back.  No engine file is made here -- what the
emptying removes is the real engine's, which the real-junction e2e shows
(`test_transport_on_a_real_junction_e2e`); this bounds WHERE the walk
empties.

PREVENTS: the emptying attached to the wrong step (a round's pair removed
under the round's later frames, or never), or skipped after a failed
member, which would leave the last round's pair behind.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _members(where: Path, names, fail=()):
    for n in names:
        d = where / "run-0" / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "go.sh").write_text("#!/bin/sh\n"
                                 + ("exit 1\n" if n in fail else "exit 0\n"))


def _log_lines(log: Path):
    """Each line's words after the walk's `[group] <stamp> ` head."""
    return [ln.split(" ", 2)[2] for ln in log.read_text().splitlines()
            if ln.startswith("[group] ")]


def test_the_folder_is_emptied_as_each_round_begins_and_at_the_end(tmp_path):
    from molbuilder.jobset.submit import _walk_script
    (tmp_path / "run-0" / "tsgf").mkdir(parents=True)
    _members(tmp_path, ["f000/v0", "f001/v0", "f000/v0.2", "f001/v0.2"],
             fail=("f001/v0",))
    steps = [(f"device@{n}", f"run-0/{n}", "go.sh", "", ())
             for n in ["f000/v0", "f001/v0", "f000/v0.2", "f001/v0.2"]]
    script = _walk_script("device-sweep", steps, where="the device's sweep",
                          log="walk.log", member_word="point",
                          needs=[None, None, 0, 1],
                          empties={0: "run-0/tsgf", 2: "run-0/tsgf"},
                          empty_at_end="run-0/tsgf")
    (tmp_path / "walk.sh").write_text(script)
    rc = subprocess.run(["bash", "walk.sh"], cwd=tmp_path,
                        capture_output=True, text=True).returncode
    assert rc == 1, "one member failed, so the walk exits 1"
    said = [ln for ln in _log_lines(tmp_path / "walk.log")
            if "tsgf/" in ln or " starts" in ln or "not run" in ln]
    # EACH ROUND BEGINS WITH THE EMPTYING, the walk ends with one, and the
    # failed frame's later point is not run while the round's emptying
    # still happens.
    assert said == [
        "tsgf/ empty",
        "-> device@f000/v0 starts",
        "-> device@f001/v0 starts",
        "device@f001/v0 failed -- what builds on it is not run",
        "tsgf/ empty",
        "-> device@f000/v0.2 starts",
        "device@f001/v0.2 not run -- what it builds on failed or did not run",
        "tsgf/ empty",
    ], said

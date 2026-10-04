"""The trajectory viewer follows THE RUN, by the state the server sends with
the file -- `web/results.md` § 4.1, `web/trajectory.md` § 5.

After every answer ``_settlePostLoad`` decides whether the viewer keeps
polling.  It decides by ``fileState.run`` -- ``{state, detail, live}``, the
one door's answer (`parse.dirs.run_answer`) -- and never by the file's own
ending: an output that states its end belongs to a job that may still be
deriving its result, and a run killed mid-step states nothing at all.  Until
2026-10-03 it read the file's ``run_state`` and waited for two ``ended``
ticks, so a run killed mid-step was polled every 15 s until the page closed
(plan W38 M2f).  A file that belongs to no run -- an upload -- has nothing to
follow, and its own ending says only whether it stopped.

THE CASES ARE ROWS, one runner below: the run the server sent (or ``None``,
no run), the file's own ending, and the transition the settle must make.
The function is lifted from the shipped module and run in node against a
fake ``state`` and a recording ``transition``.

Why not e2e: the decision is a pure function of those two inputs.  The
server half -- which run a file belongs to, and that the run's end is read
before the file's last read -- is ``tests/test_viewers_follow_the_run.py``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _slice(src: str, start_marker: str, end_marker: str) -> str:
    ix = src.index(start_marker)
    return src[ix:src.index(end_marker, ix)].rstrip()


def _node_or_skip() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    return node


def _run(state, live):
    return {"state": state, "detail": f"({state})", "live": live}


#: (case, the run the server sent, the file's own ending, the transition)
CASES = [
    ("a live run is followed",
     _run("running", True), "running", "WATCHING"),
    ("a queued run is followed -- it will start writing",
     _run("queued", True), None, "WATCHING"),
    ("an output that ended belongs to a job still deriving its result: followed",
     _run("running", True), "ended", "WATCHING"),
    ("a finished run is not followed",
     _run("finished", False), "ended", "LOADED"),
    ("a run that concluded is finished, whatever its output states",
     _run("finished", False), "running", "LOADED"),
    ("a run killed mid-step -- its output states nothing -- stops, as failed",
     _run("failed", False), "running", "ERROR"),
    ("a run whose output states a stop stops, as failed",
     _run("failed", False), "stopped", "ERROR"),
    ("a run never launched is not followed -- nothing arrives until it is",
     _run("pending", False), None, "LOADED"),
    ("an upload that stopped settles on its own ending",
     None, "stopped", "ERROR"),
    ("an upload out of memory settles on its own ending",
     None, "out_of_memory", "ERROR"),
    ("an upload that ended is loaded",
     None, "ended", "LOADED"),
    ("an upload is never followed, whatever its file says",
     None, "running", "LOADED"),
]


def _settle(run, run_state):
    """Run ``_settlePostLoad`` for one row; the transitions it made."""
    node = _node_or_skip()
    src = MODULE.read_text()
    # Lifted from the real source -- the RUN_STATE vocabulary, the settle
    # and the file-only check beside it -- so a rename breaks this test
    # instead of sliding past it.
    run_state_const = _slice(src, "const RUN_STATE = Object.freeze",
                             "});") + "});"
    fn_source = _slice(src, "function _settlePostLoad",
                       "function plottableFrames")
    data = None if run_state is None else {"run_state": run_state}
    harness = f"""
        {run_state_const}
        const _seen = [];
        function transition(name) {{ _seen.push(name); state.machine = name; }}
        const state = {{
            machine: "LOADING",
            fileState: {{ data: {json.dumps(data)}, run: {json.dumps(run)} }},
        }};
        {fn_source}
        _settlePostLoad();
        console.log(JSON.stringify(_seen));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case, run, run_state, want", CASES,
                         ids=[c[0] for c in CASES])
def test_the_viewer_follows_the_run(case, run, run_state, want):
    assert _settle(run, run_state) == [want], case


def test_the_files_own_vocabulary_is_the_parsers():
    """``RUN_STATE`` -- a FILE's ending, read only for a file with no run --
    is `model/parse.md` § 2b's closed vocabulary.  Evaluated, not grepped:
    node reads the file's own syntax.  A consumer holding a private copy of
    a vocabulary keeps compiling and silently stops matching (bug #12: a
    crashed run polled for as long as the tab stayed open)."""
    node = _node_or_skip()
    src = MODULE.read_text()
    const = _slice(src, "const RUN_STATE = Object.freeze", "});") + "});"
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e",
         const + "\nconsole.log(JSON.stringify(Object.values(RUN_STATE)));"],
        capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    got = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert got == {"running", "ended", "stopped", "out_of_memory", "unknown"}

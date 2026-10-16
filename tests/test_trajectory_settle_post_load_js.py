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

Why not e2e: the decision is a pure function of those two inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import lifted, run_node

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _run(state, live):
    return {"state": state, "detail": f"({state})", "live": live}


#: (case, the run the server sent, the file's own ending, the transition the
#: settle makes, the badge's kind).  The badge's words for its kinds, from
#: `web/trajectory.md` § 4: `finished` Finished, `stopped` Stopped, `queued`
#: Queued, `pending` Not launched, `unreadable` Record unreadable, `running`
#: Running.
CASES = [
    ("a live run is followed",
     _run("running", True), "running", "WATCHING", "running"),
    ("a queued run is followed -- it will start writing",
     _run("queued", True), None, "WATCHING", "queued"),
    ("an output that ended belongs to a job still deriving its result: followed",
     _run("running", True), "ended", "WATCHING", "running"),
    ("a finished run is not followed",
     _run("finished", False), "ended", "LOADED", "finished"),
    ("a run that concluded is finished, whatever its output states",
     _run("finished", False), "running", "LOADED", "finished"),
    ("a run killed mid-step -- its output states nothing -- stops, as failed",
     _run("failed", False), "running", "ERROR", "stopped"),
    ("a run whose output states a stop stops, as failed",
     _run("failed", False), "stopped", "ERROR", "stopped"),
    ("a run never launched is not followed -- nothing arrives until it is",
     _run("pending", False), None, "LOADED", "pending"),
    ("a run whose launch record does not read is never followed, and says so",
     _run("unreadable", False), "running", "LOADED", "unreadable"),
    ("an upload that stopped settles on its own ending",
     None, "stopped", "ERROR", "stopped"),
    ("an upload out of memory settles on its own ending",
     None, "out_of_memory", "ERROR", "stopped"),
    ("an upload that ended is loaded",
     None, "ended", "LOADED", "finished"),
    ("an upload is never followed, whatever its file says",
     None, "running", "LOADED", "running"),
    ("an upload whose file can state no ending is loaded",
     None, "unknown", "LOADED", "running"),
]


def _settle(run, run_state):
    """Run ``_settlePostLoad`` for one row; the transitions it made."""
    # Lifted from the real source -- the RUN_STATE vocabulary, the settle
    # and the file-only check beside it -- so a rename breaks this test
    # instead of sliding past it.
    run_state_const = lifted(MODULE, "const RUN_STATE = Object.freeze",
                             "});") + "});"
    fn_source = lifted(MODULE, "function _settlePostLoad",
                       "function plottableFrames")
    data = None if run_state is None else {"run_state": run_state}
    return run_node([], f"""
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
    """)


@pytest.mark.parametrize("case, run, run_state, want, badge", CASES,
                         ids=[c[0] for c in CASES])
def test_the_viewer_follows_the_run(case, run, run_state, want, badge):
    assert _settle(run, run_state) == [want], case


def _badge(run, run_state):
    """The REAL `_badgeKind` for one row -- the run's state, or for a file
    that belongs to no run, its own ending."""
    data = None if run_state is None else {"run_state": run_state}
    return run_node([], f"""
        {lifted(MODULE, "const RUN_STATE = Object.freeze", "});")}}});
        {lifted(MODULE, "function _badgeKind(run, data) {", "\n    function ")}
        console.log(JSON.stringify(_badgeKind({json.dumps(run)},
                                              {json.dumps(data)})));
    """)


@pytest.mark.parametrize("case, run, run_state, want, badge", CASES,
                         ids=[c[0] for c in CASES])
def test_the_badge_reads_the_run_or_the_files_own_ending(
        case, run, run_state, want, badge):
    """`web/trajectory.md` § 4: the badge reads THE RUN the file belongs to
    -- Finished, Stopped (failed), Queued, Not launched, Record unreadable,
    else Running -- and
    a file that belongs to no run by its own ending: `ended` Finished,
    `stopped` or `out_of_memory` Stopped, anything else Running.

    THE SILENT FAILURE: an uploaded output that ran out of memory, or ended,
    shown Running for as long as anyone looks -- the viewer spelling the
    parser's word its own way, with every follow decision still right.

    MUTATIONS THIS MUST FAIL AGAINST: the viewer's `ended` spelled off the
    parser's; out-of-memory read as running."""
    assert _badge(run, run_state) == badge, case


def test_every_ending_the_parser_concludes_with_has_a_row():
    """`model/parse.md` § 2b P-S4: *the run is over* has one answer, the
    parser's ``_run_ending.CONCLUDED``.  Every ending in it is a row above,
    run through the viewer's own settle -- so an ending the parser learns
    cannot reach the viewer unread, an upload that stopped shown as one
    still running.

    MUTATION THIS MUST FAIL AGAINST: a concluded ending the parser declares
    and no row answers."""
    from molbuilder.parse.engines._run_ending import CONCLUDED
    rows = {run_state for _, run, run_state, _, _ in CASES if run is None}
    assert set(CONCLUDED) <= rows, sorted(set(CONCLUDED) - rows)

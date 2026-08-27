"""The notify card writes a POLICY, and never a destination.

``docs/web/task-setup.md`` § 6.1 draws the line this card sits on: a queue
name and a wall may be written into a description because they are *"true
wherever the file is opened"*, while *"use 16 ranks"* may not.  **"Notify me
every six hours" passes that test and a webhook URL fails it** — the URL is
a fact about one machine, and a description travels.

So the card offers two checkboxes and a number, and the file it writes
carries exactly that.  These tests drive the REAL functions out of
``task-setup/viewer.js`` under Node, for the reason
``test_task_setup_cell_readers_js.py`` gives at length: the controller
cannot be imported without a DOM, so a test that does not run the source
can only check that names exist — and a stub returning the wrong thing
passes that.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "molbuilder/web/static/task-setup/viewer.js"


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)].rstrip()


def _run(controls: dict, task: dict | None, want: str):
    """Drive the real functions with a fake DOM and report what they did.

    ``controls`` is the state of the three inputs; ``task`` is the open
    description (``None`` for "the editor holds no object").  ``want`` names
    which answer to print.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    src = VIEWER.read_text()
    fns = "\n\n".join([
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
        _slice(src, "function readNotifyFromTask(task)", "/** One line saying"),
        _slice(src, "function paintNotifyNote()", "/** What a `prep` would"),
    ])

    harness = f"""
        const _els = {json.dumps(controls)};
        // The page's own accessor, faked: every control is an object with
        // the properties the real functions touch, and nothing else.
        function $(id) {{ return _els[id] || null; }}
        let _doc = {json.dumps(json.dumps(task) if task is not None else None)};
        const _cm = {{
            getValue: () => _doc,
            setValue: (t) => {{ _doc = t; }},
            getCursor: () => null,
            setCursor: () => {{}},
        }};
        {fns}
        const _out = {{}};
        _out.values = notifyValues();
        applyNotifyToDoc();
        paintNotifyNote();
        _out.doc = _doc;
        _out.note = (_els["ts-notify-note"] || {{}}).textContent || "";
        console.log(JSON.stringify(_out));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])[want]


def _controls(scf=False, periodic=False, hours="6"):
    return {
        "ts-notify-scf":      {"checked": scf},
        "ts-notify-periodic": {"checked": periodic},
        "ts-notify-hours":    {"value": hours},
        "ts-notify-note":     {"textContent": ""},
    }


_TASK = {"schema": "molbuilder/task@1", "engine": {"name": "siesta"},
         "run": {"name": "r", "id": "r"}}


# --------------------------------------------------------------------- #
#  what the ticks produce                                                #
# --------------------------------------------------------------------- #

def test_nothing_ticked_writes_no_key():
    """Absent-is-a-state, matching `task.Notify`: a description that
    reports on nothing must round-trip byte-identical, or every file
    changes the first time somebody opens this tab."""
    doc = json.loads(_run(_controls(), _TASK, "doc"))
    assert "notify" not in doc


def test_each_trigger_alone_is_a_valid_policy():
    """They combine with OR -- checkboxes, not a picker."""
    scf = json.loads(_run(_controls(scf=True), _TASK, "doc"))
    assert scf["notify"] == {"on_scf_converged": True}

    per = json.loads(_run(_controls(periodic=True, hours="4"), _TASK, "doc"))
    assert per["notify"] == {"every_hours": 4}


def test_both_together():
    doc = json.loads(_run(_controls(scf=True, periodic=True, hours="2.5"),
                          _TASK, "doc"))
    assert doc["notify"] == {"on_scf_converged": True, "every_hours": 2.5}


def test_the_hours_box_is_ignored_until_its_row_is_ticked():
    """A number sitting in an unticked row is not an answer.  Writing it
    anyway would turn the box's own offered default into a policy nobody
    chose -- the page inventing a cadence, which is exactly what the
    feature is designed not to do."""
    doc = json.loads(_run(_controls(periodic=False, hours="6"), _TASK, "doc"))
    assert "notify" not in doc


@pytest.mark.parametrize("bad", ["", "0", "-3", "abc"])
def test_a_period_that_is_not_a_positive_number_writes_nothing(bad):
    """`task.py` refuses these by name on the way in.  The page must not
    hand it one to refuse: a description written from this tab should be
    readable by the tool that wrote it."""
    doc = json.loads(_run(_controls(periodic=True, hours=bad), _TASK, "doc"))
    assert "notify" not in doc, f"{bad!r} became a policy"


def test_the_number_is_a_number_not_a_human_spelling():
    """HOURS as a JSON number on both sides.  `task.py` refuses `"6"` and
    `"6h"`, and it is right to -- a value that changes meaning crossing a
    boundary is how "4h" reached sbatch as `-t 4h`."""
    doc = json.loads(_run(_controls(periodic=True, hours="6"), _TASK, "doc"))
    assert doc["notify"]["every_hours"] == 6
    assert not isinstance(doc["notify"]["every_hours"], str)


# --------------------------------------------------------------------- #
#  what it must never write                                              #
# --------------------------------------------------------------------- #

def test_no_destination_or_credential_can_reach_the_description():
    """The card has no field for one and must never grow a key for one.

    A `task.json` travels -- to a cluster, into a handoff bundle, to
    whoever is handed the calculation.  `task.py`'s key allowlist refuses
    a `url` outright; this is the same rule one layer up, where the value
    would have to originate.
    """
    doc = json.loads(_run(_controls(scf=True, periodic=True), _TASK, "doc"))
    blob = json.dumps(doc).lower()
    for leak in ("url", "webhook", "token", "authorization", "slack", "bearer"):
        assert leak not in blob, f"the page wrote {leak!r} into the description"


# --------------------------------------------------------------------- #
#  reading a description back into the card                              #
# --------------------------------------------------------------------- #

def test_an_unparseable_document_loses_nothing():
    """Mid-edit the editor holds bytes that are not JSON.  The card must
    say nothing rather than overwrite the text somebody is typing."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = VIEWER.read_text()
    fns = "\n\n".join([
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
    ])
    harness = f"""
        const _els = {json.dumps(_controls(scf=True))};
        function $(id) {{ return _els[id] || null; }}
        let _doc = "{{ not json at all";
        const _cm = {{ getValue: () => _doc, setValue: (t) => {{ _doc = t; }},
                      getCursor: () => null, setCursor: () => {{}} }};
        {fns}
        applyNotifyToDoc();
        console.log(JSON.stringify({{doc: _doc}}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])["doc"] \
        == "{ not json at all", "the page rewrote text it could not parse"


def test_the_note_says_what_will_actually_be_sent():
    """Including the part that is not a choice: a run ending always
    reports, so the line has to say so or the card reads as "nothing"."""
    assert "only when it ends" in _run(_controls(), _TASK, "note")
    note = _run(_controls(scf=True, periodic=True, hours="3"), _TASK, "note")
    assert "each SCF convergence" in note
    assert "every 3 h" in note
    assert "when it ends" in note

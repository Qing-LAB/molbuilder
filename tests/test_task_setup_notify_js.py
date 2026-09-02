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
        # `notifyValues` asks it which channels are ticked, so a harness
        # without it would exercise a different function than the page runs.
        # `notifyValues` asks BOTH selectors, so the harness needs both or
        # it exercises a different function than the page runs.
        _slice(src, "const REPORT_ITEMS = [", "/** Paint one tick per report"),
        _slice(src, "function channelSelection()", "/** Paint one tick per channel"),
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        # `applyNotifyToDoc` calls it, so the harness needs it too: the
        # writer stopped moving the page on 2026-08-27 and does so through
        # a helper rather than inline.
        _slice(src, "function keepingPagePut(fn)", "/** Fill the card FROM"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
        _slice(src, "function readNotifyFromTask(task)", "/** One line saying"),
        _slice(src, "function paintNotifyNote()", "/** What a `prep` would"),
    ])

    harness = f"""
        const _els = {json.dumps(controls)};
        // The channel container answers ONE method, which is all
        // `channelSelection` asks it.  Built here rather than in the fixture
        // because a function does not survive JSON.
        _els["ts-notify-channels"] = {{
            querySelectorAll: () => _els["__ticks__"],
        }};
        // The page's own accessor, faked: every control is an object with
        // the properties the real functions touch, and nothing else.
        function $(id) {{ return _els[id] || null; }}
        // No DOM here.  `keepingPagePut` looks for the scrolling container
        // and must degrade to "nothing to restore" rather than throw --
        // which is also what it does on a page that has not mounted yet.
        const document = {{ querySelector: () => null }};
        const requestAnimationFrame = (f) => f();
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


def _controls(scf=False, periodic=False, hours="6", every=True, ticks=None):
    """The card's controls.

    ``every`` is the *every channel on the machine that runs it* box, and
    ``ticks`` the per-channel ones -- ``[(name, checked), ...]``.  The default
    is the state a page opens in, so every test written before channels
    existed still describes the same card.
    """
    return {
        "ts-notify-scf":      {"checked": scf},
        "ts-notify-periodic": {"checked": periodic},
        "ts-notify-hours":    {"value": hours},
        "ts-notify-note":     {"textContent": ""},
        "ts-notify-all":      {"checked": every},
        "__ticks__":          [{"value": n, "checked": c}
                               for n, c in (ticks or [])],
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

    A `task.json` travels -- to a cluster, into a composed copy, to
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
        # `notifyValues` asks it which channels are ticked, so a harness
        # without it would exercise a different function than the page runs.
        # `notifyValues` asks BOTH selectors, so the harness needs both or
        # it exercises a different function than the page runs.
        _slice(src, "const REPORT_ITEMS = [", "/** Paint one tick per report"),
        _slice(src, "function channelSelection()", "/** Paint one tick per channel"),
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        # `applyNotifyToDoc` calls it, so the harness needs it too: the
        # writer stopped moving the page on 2026-08-27 and does so through
        # a helper rather than inline.
        _slice(src, "function keepingPagePut(fn)", "/** Fill the card FROM"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
    ])
    harness = f"""
        const _els = {json.dumps(_controls(scf=True))};
        _els["ts-notify-channels"] = {{
            querySelectorAll: () => _els["__ticks__"],
        }};
        function $(id) {{ return _els[id] || null; }}
        // No DOM here.  `keepingPagePut` looks for the scrolling container
        // and must degrade to "nothing to restore" rather than throw --
        // which is also what it does on a page that has not mounted yet.
        const document = {{ querySelector: () => null }};
        const requestAnimationFrame = (f) => f();
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


# --------------------------------------------------------------------- #
#  which channels -- three states, and they are not two                  #
# --------------------------------------------------------------------- #

def test_every_channel_writes_no_key_at_all():
    """The default, and the reading of every description written before
    channels existed: use whatever is set up wherever this lands.  Writing
    the names out instead would freeze a travelling description to the
    machine it happened to be written on."""
    doc = json.loads(_run(_controls(scf=True, every=True,
                                    ticks=[("slack", True)]), _TASK, "doc"))
    assert doc["notify"] == {"on_scf_converged": True}


def test_a_subset_travels_as_names():
    """A name is a label the person chose.  It grants nothing, so it is safe
    to carry where an address is not (`run-reports.md` § 1)."""
    doc = json.loads(_run(
        _controls(scf=True, every=False,
                  ticks=[("slack", True), ("lab", False), ("phone", True)]),
        _TASK, "doc"))
    assert doc["notify"]["channels"] == ["slack", "phone"]


def test_ticking_nothing_writes_an_EMPTY_LIST_not_nothing():
    """**The one field written when falsy**, and the reason is the whole
    point of the control: an unticked list that dropped the key would mean
    *every channel*, so unticking them all would send reports to every
    channel the person had just turned off (`run-reports.md` § 3.0)."""
    doc = json.loads(_run(
        _controls(scf=True, every=False,
                  ticks=[("slack", False), ("lab", False)]), _TASK, "doc"))
    assert doc["notify"]["channels"] == []


def test_the_note_says_where_as_well_as_when():
    """*Reports every 6 h* with nothing ticked is a promise the run cannot
    keep, so the line that summarises the card carries both halves."""
    assert "every channel" in _run(_controls(), _TASK, "note")
    assert "to slack" in _run(
        _controls(scf=True, every=False, ticks=[("slack", True)]),
        _TASK, "note")
    assert "nothing" in _run(
        _controls(scf=True, every=False, ticks=[("slack", False)]),
        _TASK, "note").lower()


def test_the_page_hardcodes_NO_path_and_asks_for_it_instead():
    """**The stronger form of a defect found in the browser 2026-08-27.**

    The card used to state `~/.molbuilder/notify` while the monitor read
    `config_dir()/notify` — following the page put the file where nothing
    looks, and **absent means silently off**, so there was no notification,
    no error, and nothing to read.

    A hardcoded path is what drifts. The page now shows what
    `GET /api/notify/destination` reports, and that endpoint takes the path
    from `monitor.default_notify_path` — so the page, the API and the
    process that reads the file on a compute node all get it from one
    function and cannot disagree.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html = (root / "molbuilder/web/templates/task_setup.html").read_text()
    # a path in the MARKUP is the thing that went wrong; there must be none
    assert "~/.molbuilder/notify" not in html
    assert "~/.config/molbuilder/notify" not in html, \
        "the page hardcodes a path again -- ask the API instead"
    api = (root / "molbuilder/web/blueprints/notify_setup.py").read_text()
    assert "from ...monitor import default_notify_path" in api


def test_the_card_writes_ONE_file_and_has_no_control_for_a_secret():
    """**The rule that used to be held by a comment.**

    The address and key lived in this card from 2026-08-27 until 2026-08-31,
    under a template comment reminding whoever edited it that one card wrote
    two files -- `task.json`, which travels, and `config_dir()/notify`, which
    must not.  They are on the This-machine tab now, so the rule is held by
    there being nowhere on this page to type one.

    A comment is not a mechanism.  This is the mechanism.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html = (root / "molbuilder/web/templates/task_setup.html").read_text()
    assert 'id="ts-notify-card"' in html
    for gone in ('id="ts-reports-key"', 'id="ts-reports-url"',
                 'id="ts-reports-save"', 'type="password"'):
        assert gone not in html, f"{gone} came back to the page that travels"
    js = (root / "molbuilder/web/static/task-setup/viewer.js").read_text()
    for gone in ("/api/notify/destination", "saveDestination"):
        assert gone not in js, f"{gone} still writes a second file from here"
    # it may ASK for the names, and that route hands out nothing else
    assert "/api/notify/channels" in js


def test_the_names_are_all_that_reaches_the_description():
    """The tick list is painted from what the server reports, and what it
    reports is names.  The writer must not be able to reach anything else --
    the same rule `task.py`'s key allowlist enforces one layer down."""
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/task-setup/viewer.js").read_text()
    writer = js[js.index("function channelSelection()"):
                js.index("function readNotifyFromTask")]
    for leak in ("url", "key", "password", "has_key"):
        assert leak not in writer, \
            f"{leak} reached the function that writes task.json"


def test_the_card_says_KEY_not_token():
    """The secret stopped being a bearer token on 2026-08-27; it signs and
    never travels. A card still calling it a token teaches the old model."""
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1]
            / "molbuilder/web/templates/task_setup.html").read_text()
    card = html[html.index("ts-notify-card") - 900:]
    card = card[:card.index("ts-notify-opts")]
    assert "token" not in card.lower(), "the card still says token"


def test_no_two_elements_share_an_id_in_the_task_setup_page():
    """**Found in the browser, 2026-08-27, and it fails in silence.**

    A `ts-reports-card` was first written as `ts-dest-card` — a name
    already taken by the *Where this saves* card, where "dest" means the
    destination FOLDER. `getElementById` returns the first match, so the
    new card's JS would have read and written the wrong section entirely,
    with no error anywhere.

    Duplicate ids are invalid HTML and this is the failure mode: not a
    crash, but the wrong element quietly answering.
    """
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1]
            / "molbuilder/web/templates/task_setup.html").read_text()
    ids = re.findall(r'\bid="([^"]+)"', html)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate id(s) in task_setup.html: {dupes}"

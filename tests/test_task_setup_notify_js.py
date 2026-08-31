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
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        # `applyNotifyToDoc` calls it, so the harness needs it too: the
        # writer stopped moving the page on 2026-08-27 and does so through
        # a helper rather than inline.
        _slice(src, "function keepingPagePut(fn)", "/** Fill the card FROM"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
    ])
    harness = f"""
        const _els = {json.dumps(_controls(scf=True))};
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


def test_one_card_but_two_files():
    """The cards were merged on 2026-08-27 (user: *where the notification
    should be sent should be configurable in this card too*), and sharing a
    card is a UI decision that must not become a shared FILE.

    `run-reports.md` § 1 is about what travels: the ticks go into
    `task.json`, which does; the address and key go into
    `config_dir()/notify`, which never does. So the key input must sit
    below the policy inputs and nothing may carry it into the description.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html = (root / "molbuilder/web/templates/task_setup.html").read_text()
    assert 'id="ts-notify-card"' in html
    assert 'id="ts-reports-card"' not in html, "the split card came back"
    # the key belongs to the destination half, after the policy half
    assert html.index('id="ts-notify-scf"') < html.index('id="ts-reports-key"')
    js = (root / "molbuilder/web/static/task-setup/viewer.js").read_text()
    writer = js[js.index("function notifyValues()"):js.index("function readNotifyFromTask")]
    for leak in ("ts-reports-key", "ts-reports-url", "password"):
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




# ---------------------------------------------------------------------------
# The far-machine command must resolve the config directory the SAME way
# ---------------------------------------------------------------------------

def test_the_remote_command_implements_the_whole_config_dir_rule():
    """`configuration.md` § 2.1c, expressed in shell because it runs elsewhere.

    The card hands over a copy-paste block for a machine this server cannot
    write to, so the rule has to travel as shell rather than as an answer --
    and it has to be the WHOLE rule.  It implemented only the XDG branches
    until 2026-08-31: on any machine with ``MOLBUILDER_CONFIG_DIR`` set,
    following the card wrote the file where the monitor does not look, and
    silently, because an absent notify file just means "no notifier".

    This runs the emitted shell under all three environments and compares it
    against `config_dir()` itself, so the two cannot drift.  A comment saying
    "keep these in sync" is what failed here once already.
    """
    import os
    import re
    import subprocess
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / (
        "molbuilder/web/static/task-setup/viewer.js")
    line = next((l for l in src.read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("'cfg=")), None)
    assert line, "the command no longer starts by resolving a config dir"
    # the JS string literal, unescaped to the shell it emits
    shell = re.sub(r"^\s*'|\\n'\s*$", "", line.strip()).replace('\\"', '"')

    from molbuilder.config_dir import config_dir
    home = "/home/tester"
    cases = [
        ({"MOLBUILDER_CONFIG_DIR": "/scratch/me/mb"}, "the override, exact"),
        ({"XDG_CONFIG_HOME": "/tmp/xdg"}, "the XDG root, with our name under it"),
        ({}, "the default"),
    ]
    for extra, what in cases:
        env = {"HOME": home, **extra}
        got = subprocess.run(["bash", "-c", shell + '; echo "$cfg"'],
                             env=env, capture_output=True, text=True,
                             check=True).stdout.strip()
        saved = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(env)
            expected = str(config_dir())
        finally:
            os.environ.clear()
            os.environ.update(saved)
        assert got == expected, (
            f"{what}: the card's shell says {got!r}, config_dir() says "
            f"{expected!r} -- a file written by following the card would "
            f"land where nothing reads it")

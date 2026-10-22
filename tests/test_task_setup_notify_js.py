"""The notify card writes a POLICY, and never a destination.

``docs/web/task-setup.md`` § 6.1 draws the line this card sits on: a queue
name and a wall may be written into a description because they are *"true
wherever the file is opened"*, while *"use 16 ranks"* may not.  **"Notify me
every six hours" passes that test and a webhook URL fails it** — the URL is
a fact about one machine, and a description travels.

So the card offers two checkboxes and a number, and the file it writes
carries exactly that.  These tests drive the REAL functions out of
``task-setup/viewer.js`` under Node: the controller cannot be imported
without a DOM, so a test that does not run the source
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


#: A STAND-IN FOR THE PAGE'S FENCE that only runs the step it is given: the
#: page imports `lib/page-busy.js`; a slice cannot, so the harness declares
#: `pageBusy`, as it declares the page's `$`.  What these tests read is the
#: document the step writes; the fence's own claim and release are
#: `test_page_busy_js.py`'s, and the card's way through it the Task setup
#: fence table's (`test_task_setup_prep_e2e.py`).
_FENCE_IN_NODE = """
const pageBusy = { run: async (reason, cancelers, fn) => fn() };
"""


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)].rstrip()


def _run(controls: dict, request: dict | None, want: str):
    """Drive the real functions with a fake DOM and report what they did.

    ``controls`` is the state of the three inputs; ``request`` is the open
    job request (``None`` for "the editor holds no object").  ``want`` names
    which answer to print.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    src = VIEWER.read_text()
    fns = "\n\n".join([
        # `notifyValues` asks BOTH selectors, so the harness needs both or
        # it exercises a different function than the page runs.
        _slice(src, "//: ``null`` until the server has answered", "/** Paint one tick per report"),
        _slice(src, "function channelSelection()", "/** Paint one tick per channel"),
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        # `applyNotifyToDoc` calls it, so the harness needs it too.
        _FENCE_IN_NODE,
        # The buffers' accessor, the page's own.
        _slice(src, "function bufferText(which)", "/** Whether either buffer"),
        _slice(src, "function keepingPagePut(fn)", "/** Fill the card FROM"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
        _slice(src, "function readNotifyFromRequest(request, task)",
               "/** One line saying"),
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
        let _doc = {json.dumps(json.dumps(request) if request is not None else None)};
        // THE JOB REQUEST'S BUFFER, faked: what the card's writer reads
        // and writes (`patchDoc`).
        const _docs = {{ request: {{
            getValue: () => _doc,
            setValue: (t) => {{ _doc = t; }},
            getCursor: () => null,
            setCursor: () => {{}},
        }} }};
        let _request = null;
        let _parses = {{ task: true, request: true }};
        // The save bar is not this file's subject: its writer is called.
        function _setDirty() {{}}
        function edited() {{ return true; }}
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


def _tick(name, checked, extra=None):
    """One channel checkbox, as the page builds it.

    ``extra`` is what the server reported BESIDE the name --
    `/api/task-setup/channels` answers a listener with its url and signing
    key attached, so the element a person ticks is built from an object
    that holds them.  Tests that do not care pass 2-tuples and get the
    plain shape.
    """
    return dict({"value": name, "checked": checked}, **(extra or {}))


def _controls(scf=False, periodic=False, hours="6", every=True, ticks=None):
    """The card's controls.

    ``every`` is the *every channel on the machine that runs it* box, and
    ``ticks`` the per-channel ones -- ``[(name, checked), ...]``.  The default
    is the state a page opens in.
    """
    return {
        "ts-notify-scf":      {"checked": scf},
        "ts-notify-periodic": {"checked": periodic},
        "ts-notify-hours":    {"value": hours},
        "ts-notify-note":     {"textContent": ""},
        "ts-notify-all":      {"checked": every},
        # A tick may carry MORE than its name.  `/api/task-setup/channels`
        # answers what this machine has, and a listener's entry there has a
        # url and a signing key beside it -- so the checkbox a person sees
        # is built from an object that holds them.  Tests pass 2-tuples and
        # get the plain shape; a 3-tuple's dict is merged in, which is how
        # `test_a_channels_secret_never_reaches_the_description` plants one.
        "__ticks__":          [_tick(*t) for t in (ticks or [])],
    }


#: The open job request the card writes into (`job-request.md` § 3.5).
_REQUEST = {"schema": "molbuilder/job-request@1",
            "stages": {"coarse": {"group": 1}}}


# --------------------------------------------------------------------- #
#  what the ticks produce                                                #
# --------------------------------------------------------------------- #

def test_nothing_ticked_writes_no_key():
    """No block is no notification (`run-reports.md` § 2): nothing set up
    writes no `notify` key, so a description that reports on nothing
    round-trips byte-identical."""
    doc = json.loads(_run(_controls(), _REQUEST, "doc"))
    assert "notify" not in doc


def test_each_trigger_alone_is_a_valid_policy():
    """They combine with OR -- checkboxes, not a picker.  A block is written
    whole (`stages.md` § 6.9): the trigger not ticked as `false` or `null`
    (never), every channel and every field as `["*"]`."""
    scf = json.loads(_run(_controls(scf=True), _REQUEST, "doc"))
    assert scf["notify"] == {"on_scf_converged": True, "every_hours": None,
                             "channels": ["*"], "report": ["*"]}

    per = json.loads(_run(_controls(periodic=True, hours="4"), _REQUEST, "doc"))
    assert per["notify"] == {"on_scf_converged": False, "every_hours": 4,
                             "channels": ["*"], "report": ["*"]}


def test_both_together():
    doc = json.loads(_run(_controls(scf=True, periodic=True, hours="2.5"),
                          _REQUEST, "doc"))
    assert doc["notify"] == {"on_scf_converged": True, "every_hours": 2.5,
                             "channels": ["*"], "report": ["*"]}


def test_the_hours_box_is_ignored_until_its_row_is_ticked():
    """A number sitting in an unticked row is not an answer.  Writing it
    anyway would turn the box's own offered default into a policy nobody
    chose -- the page inventing a cadence, which is exactly what the
    feature is designed not to do."""
    doc = json.loads(_run(_controls(periodic=False, hours="6"), _REQUEST, "doc"))
    assert "notify" not in doc


@pytest.mark.parametrize("bad", ["", "0", "-3", "abc"])
def test_a_period_that_is_not_a_positive_number_writes_nothing(bad):
    """`job_request.py` refuses these by name on the way in.  The page must not
    hand it one to refuse: a description written from this tab should be
    readable by the tool that wrote it."""
    doc = json.loads(_run(_controls(periodic=True, hours=bad), _REQUEST, "doc"))
    assert "notify" not in doc, f"{bad!r} became a policy"


def test_the_number_is_a_number_not_a_human_spelling():
    """HOURS as a JSON number on both sides.  `job_request.py` refuses `"6"` and
    `"6h"`, and it is right to -- a value that changes meaning crossing a
    boundary is how "4h" reached sbatch as `-t 4h`."""
    doc = json.loads(_run(_controls(periodic=True, hours="6"), _REQUEST, "doc"))
    assert doc["notify"]["every_hours"] == 6
    assert not isinstance(doc["notify"]["every_hours"], str)


# --------------------------------------------------------------------- #
#  what it must never write                                              #
# --------------------------------------------------------------------- #

def test_no_destination_or_credential_can_reach_the_description():
    """The card has no field for one and must never grow a key for one.

    A `job-request.json` travels -- to a cluster, into a composed copy, to
    whoever is handed the calculation.  `job_request.py`'s key allowlist refuses
    a `url` outright; this is the same rule one layer up, where the value
    would have to originate.
    """
    doc = json.loads(_run(_controls(scf=True, periodic=True), _REQUEST, "doc"))
    blob = json.dumps(doc).lower()
    for leak in ("url", "webhook", "token", "authorization", "slack", "bearer"):
        assert leak not in blob, f"the page wrote {leak!r} into the description"


# --------------------------------------------------------------------- #
#  reading a description back into the card                              #
# --------------------------------------------------------------------- #

def test_an_unparseable_document_loses_nothing():
    """Mid-edit the editor holds bytes that are not JSON.  The card's write
    is refused and said, and the text somebody is typing is left as typed
    (`task-setup.md` § 9a)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = VIEWER.read_text()
    fns = "\n\n".join([
        # `notifyValues` asks BOTH selectors, so the harness needs both or
        # it exercises a different function than the page runs.
        _slice(src, "//: ``null`` until the server has answered", "/** Paint one tick per report"),
        _slice(src, "function channelSelection()", "/** Paint one tick per channel"),
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        # `applyNotifyToDoc` calls it, so the harness needs it too.
        _FENCE_IN_NODE,
        # The buffers' accessor, the page's own.
        _slice(src, "function bufferText(which)", "/** Whether either buffer"),
        _slice(src, "function keepingPagePut(fn)", "/** Fill the card FROM"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
        # `patchDoc` refuses through it, naming the records' tabs.
        "const RECORDS = ['task', 'request'];",
        _slice(src, "function refuseUnparsed()", "/** Whether record"),
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
        const _said = [];
        function setState(kind) {{ _said.push(kind); }}
        let _doc = "{{ not json at all";
        const _docs = {{ request: {{ getValue: () => _doc,
                                    setValue: (t) => {{ _doc = t; }},
                                    getCursor: () => null,
                                    setCursor: () => {{}} }} }};
        let _request = null;
        let _parses = {{ task: true, request: true }};
        // The save bar is not this file's subject: its writer is called.
        function _setDirty() {{}}
        function edited() {{ return true; }}
        // ...nor the cards a refused write puts back from the model.
        let _task = null;
        function readAsksFromRequest() {{}}
        function readNotifyFromRequest() {{}}
        function renderQueues() {{}}
        {fns}
        applyNotifyToDoc();
        console.log(JSON.stringify({{doc: _doc, said: _said}}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["doc"] == "{ not json at all", \
        "the page rewrote text it could not parse"
    assert got["said"] == ["refuse"], "the refused write was not said"


# --------------------------------------------------------------------- #
#  which channels -- three states, and they are not two                  #
# --------------------------------------------------------------------- #

def test_every_channel_is_written_every_never_as_this_machines_names():
    """The default: use whatever is set up wherever this lands, written
    `["*"]` (`run-reports.md` § 3.0).  Writing the ticked names out instead
    would freeze a travelling description to the machine it happened to be
    written on."""
    doc = json.loads(_run(_controls(scf=True, every=True,
                                    ticks=[("slack", True)]), _REQUEST, "doc"))
    assert doc["notify"]["channels"] == ["*"]


def test_a_subset_travels_as_names():
    """A name is a label the person chose.  It grants nothing, so it is safe
    to carry where an address is not (`run-reports.md` § 1)."""
    doc = json.loads(_run(
        _controls(scf=True, every=False,
                  ticks=[("slack", True), ("lab", False), ("phone", True)]),
        _REQUEST, "doc"))
    assert doc["notify"]["channels"] == ["slack", "phone"]


def test_ticking_nothing_writes_an_EMPTY_LIST_not_nothing():
    """None is `[]`, a state of its own: read as every channel, unticking
    them all would send reports to every channel the person had just turned
    off (`run-reports.md` § 3.0)."""
    doc = json.loads(_run(
        _controls(scf=True, every=False,
                  ticks=[("slack", False), ("lab", False)]), _REQUEST, "doc"))
    assert doc["notify"]["channels"] == []


def test_a_channels_secret_never_reaches_the_description():
    """A channel's URL and signing key must not travel in `job-request.json`.

    WHERE is yours and never travels with the description
    (`web/task-setup.md`; `job_request.py`'s key allowlist enforces the same rule
    one layer down).  So the checkbox is built from a server report that
    HAS a url and a key beside the name, and the writer must take the name
    and nothing else.

    This plants them and reads the document the real writer produced.
    """
    doc = json.loads(_run(
        _controls(scf=True, every=False,
                  ticks=[("lab-slack", True,
                          {"url": "https://hooks.example/T00/B00/XXXXXXXX",
                           "key": "sk-live-DEADBEEF",
                           "password": "hunter2",
                           "has_key": True})]),
        _REQUEST, "doc"))

    assert doc["notify"]["channels"] == ["lab-slack"], (
        f"the writer must carry the NAME and nothing else; it wrote "
        f"{doc['notify']['channels']!r}")
    blob = json.dumps(doc)
    for secret in ("hooks.example", "sk-live-DEADBEEF", "hunter2"):
        assert secret not in blob, (
            f"{secret!r} reached job-request.json -- a calculation travels to a "
            f"cluster and into a citation; the channel's credentials must "
            f"not go with it")


def test_a_card_write_moves_the_MODEL_too_not_only_the_buffer():
    """**The silent-loss race** (`task-setup.md` § 9a).

    `_request` is the model the cards are a view of; the buffer is what
    `save` sends.  Two writers patched the BUFFER ONLY -- the asks and the notify
    ticks -- while `syncFromModel` serialises `_request` OVER the buffer.  So:
    tick a channel, then touch anything that syncs (add a stage, edit a run
    row) and the stale `_request` was written over the top.  The tick was gone,
    silently, and the editor's own re-parse is debounced 400 ms -- exactly
    long enough to click the next control.

    `patchDoc` is the one door now, and it writes BOTH.  Asserted by doing
    what the race did: patch, then serialise the model, and check the patch
    survived."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = VIEWER.read_text(encoding="utf-8")
    fns = "\n\n".join([
        _slice(src, "//: ``null`` until the server has answered", "/** Paint one tick per report"),
        _slice(src, "function channelSelection()", "/** Paint one tick per channel"),
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        _FENCE_IN_NODE,
        # The buffers' accessor, the page's own.
        _slice(src, "function bufferText(which)", "/** Whether either buffer"),
        _slice(src, "function keepingPagePut(fn)", "/** Fill the card FROM"),
        _slice(src, "function applyNotifyToDoc()", "/** Fill the card FROM"),
    ])
    harness = f"""
        let _doc = JSON.stringify({{ schema: "molbuilder/job-request@1",
                                     stages: {{}} }}, null, 2);
        const _els = {{
            "ts-notify-scf":  {{ checked: true }},
            "ts-notify-all":  {{ checked: true }},
            "ts-report-all":  {{ checked: true }},
        }};
        const $ = (id) => _els[id] || null;
        _els["ts-notify-channels"] = {{ querySelectorAll: () => [] }};
        _els["ts-report-items"]    = {{ querySelectorAll: () => [] }};
        // No DOM: `keepingPagePut` must degrade to "nothing to restore".
        const document = {{ querySelector: () => null }};
        const requestAnimationFrame = (f) => f();
        // THE JOB REQUEST'S BUFFER, faked: what the card's writer reads
        // and writes (`patchDoc`).
        const _docs = {{ request: {{
            getValue: () => _doc,
            setValue: (t) => {{ _doc = t; }},
            getCursor: () => null,
            setCursor: () => {{}},
        }} }};
        let _request = null;
        let _parses = {{ task: true, request: true }};
        // The save bar is not this file's subject: its writer is called.
        function _setDirty() {{}}
        function edited() {{ return true; }}
        {fns}
        applyNotifyToDoc();                       // a card writes
        const afterPatch = JSON.parse(_doc).notify || null;
        // ...and now the thing that used to destroy it: the model is
        // serialised over the buffer.
        _doc = JSON.stringify(_request, null, 2) + "\\n";
        console.log(JSON.stringify({{
            afterPatch: afterPatch,
            afterSync: JSON.parse(_doc).notify || null,
        }}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["afterPatch"], "the card wrote nothing at all"
    assert out["afterSync"] == out["afterPatch"], (
        "serialising the model destroyed what the card had just written -- "
        "`patchDoc` did not move `_request` with the buffer")


def test_a_card_that_could_not_ask_keeps_the_descriptions_report_list():
    """`stages.md` § 6.9: which report fields a calculation can carry is the
    SERVER's answer (`GET /api/notify/report-fields`).  Until it has answered
    -- or when it could not -- the card does not know, and not knowing is not
    "none": it took the unanswered question for an empty list, so the next
    tick anywhere on the card rewrote a description's `notify.report` to
    `[]`, the summary line alone.  Driven through the real functions.

    MUTATION THIS MUST FAIL AGAINST: read an unknown field list as empty."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = VIEWER.read_text(encoding="utf-8")
    fns = "\n\n".join([
        _slice(src, "//: ``null`` until the server has answered",
               "/** Paint one tick per report"),
        _slice(src, "function channelSelection()", "/** Paint one tick per channel"),
        _slice(src, "function notifyValues()", "/** Write the policy INTO"),
        # The buffers' accessor, the page's own.
        _slice(src, "function bufferText(which)", "/** Whether either buffer"),
        _slice(src, "function currentRecord(which)", "function wireChannels()"),
    ])
    harness = f"""
        const _els = {{
            "ts-report-all":   {{ checked: false }},
            "ts-report-items": {{ querySelectorAll: () => [] }},
            "ts-notify-all":   {{ checked: true }},
            "ts-notify-scf":   {{ checked: true }},
        }};
        const $ = (id) => _els[id] || null;
        let _doc = JSON.stringify({{ notify: {{
            on_scf_converged: true, every_hours: null, channels: ["*"],
            report: ["energy", "n_iters"] }} }});
        const _docs = {{ request: {{ getValue: () => _doc }} }};
        {fns}
        console.log(JSON.stringify(notifyValues()));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got == {"on_scf_converged": True, "every_hours": None,
                   "channels": ["*"], "report": ["energy", "n_iters"]}, got

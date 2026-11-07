"""The notify card writes a POLICY, and never a destination.

``docs/web/task-setup.md`` § 6.1 draws the line this card sits on: a queue
name and a wall may be written into a description because they are *"true
wherever the file is opened"*, while *"use 16 ranks"* may not.  **"Notify me
every six hours" passes that test and a webhook URL fails it** — the URL is
a fact about one machine, and a description travels.

So the card offers two checkboxes and a number, and the file it writes
carries exactly that.  These tests drive the REAL functions out of
``task-setup/viewer.js`` under node (`_node_esm.lifted`,
`_node_esm.run_node`): the controller cannot be imported without a DOM, so a
test that does not run the source can only check that names exist — and a
stub returning the wrong thing passes that.  The document each card writes
into is a calculation's, as `molbuilder describe` wrote it; the card is
painted as the page paints it, from what the app's own routes answer its
requests (`_node_esm.served`) -- the channels set up on this machine among
them -- and then set as a person sets it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import lifted, run_node, served


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "molbuilder/web/static/task-setup/viewer.js"
PAGE_BUSY = ROOT / "molbuilder/web/static/lib/page-busy.js"


#: THE PAGE'S FENCE, imported as the viewer imports it: in node it holds its
#: state and paints nothing.  What these tests read is the document the step
#: writes; the fence's own claim and release are `test_page_busy_js.py`'s,
#: and the card's way through it the Task setup fence table's
#: (`test_task_setup_prep_e2e.py`).
_FENCE = f"const {{ pageBusy }} = await import({json.dumps(PAGE_BUSY.as_uri())});"


#: WHAT THE HARNESSES LIFT OUT OF THE VIEWER, each by `lifted`'s two anchors:
#: `notifyValues` asks BOTH selectors, so a harness needs both or it exercises
#: a different function than the page runs.
_CHOOSING = (
    ("//: ``null`` until the server has answered",
     "/** Paint one tick per report"),
    ("function channelSelection()", "/** Paint one tick per channel"),
    ("function notifyValues()", "/** Write the policy INTO"),
)
#: ...and what writes the choice into the document: `applyNotifyToDoc` runs
#: inside the fence, and asks the buffers' accessor, the page's own.
_WRITING = (
    ("function bufferText(which)", "/** Whether either buffer"),
    ("function keepingPagePut(fn)", "/** Fill the card FROM"),
    ("function applyNotifyToDoc()", "/** Fill the card FROM"),
)
_WRITER = (*_CHOOSING, _FENCE, *_WRITING)
#: THE CARD AS THE PAGE RUNS IT: what this machine has and how the card asks
#: for it (`loadChannelNames`), a tick painted per channel
#: (`paintChannelTicks`), its boxes wired (`wireChannels`) -- the report
#: fields' among them, which these tests leave as the page opens them -- the
#: card filled from the job request (`readNotifyFromRequest`), the policy,
#: and the writer.
_ON_THE_PAGE = (
    ("//: What this machine has, from `GET /api/notify/channels`.",
     "\nif (document.readyState"),
    ("function readNotifyFromRequest(", "/** What a `prep` would resolve"),
    ("function notifyValues()", "/** Write the policy INTO"),
    _FENCE, *_WRITING,
)


def _fns(*parts) -> str:
    """The parts in order: each a pair of `lifted`'s anchors, or a line of the
    harness's own."""
    return "\n\n".join(
        lifted(VIEWER, *part).rstrip() if isinstance(part, tuple) else part
        for part in parts)


#: The card's elements as `task_setup.html` ships them, and just the
#: element behaviour the card's functions use: a child list that setting the
#: text clears, attributes, one listener an event, and the checkboxes below
#: an element found.
_DOM = """
class El {
    constructor(tag, props) {
        this.tagName = tag; this.children = []; this.attrs = {};
        this._on = {}; this._text = ""; this.className = "";
        Object.assign(this, props || {});
    }
    get textContent() { return this._text; }
    set textContent(t) { this._text = t; this.children = []; }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    addEventListener(event, fn) { this._on[event] = fn; }
    appendChild(c) { this.children.push(c); return c; }
    querySelectorAll(_checkboxes) {
        const found = [];
        const walk = (e) => e.children.forEach((c) => {
            if (c.tagName === "input" && c.type === "checkbox") found.push(c);
            walk(c);
        });
        walk(this);
        return found;
    }
    // What a click on it leaves: the box changed, and its `change` heard.
    click() { this.checked = !this.checked; (this._on.change || (() => {}))(); }
}
const _els = {
    "ts-notify-scf":      new El("input", { type: "checkbox", checked: false }),
    "ts-notify-periodic": new El("input", { type: "checkbox", checked: false }),
    "ts-notify-hours":    new El("input", { type: "number", value: "6",
                                            defaultValue: "6" }),
    "ts-notify-all":      new El("input", { type: "checkbox", checked: true }),
    "ts-notify-channels": new El("div"),
};
function $(id) { return _els[id] || null; }
const document = { createElement: (tag) => new El(tag),
                   querySelector: () => null };
const requestAnimationFrame = (f) => f();
"""


def _written(app_at, records, *, scf=False, periodic=False, hours=None,
             every=True, untick=()) -> dict:
    """The job request after a person set the card -- the real functions,
    the card painted as the page paints it: its channels asked of the app
    serving at ``app_at``, then filled from ``records``, the calculation's
    two records as the editor's buffers hold them.  Then the person's acts:
    ``scf`` and ``periodic`` ticked, ``hours`` typed, the *every channel*
    box unticked when not ``every``, and each channel named in ``untick``
    unticked."""
    return json.loads(run_node([], f"""
        {_DOM}
        // THE RECORDS' BUFFERS, faked: what the card reads, and what its
        // writer reads and writes (`patchDoc`).
        const _texts = {json.dumps({"task": records["task"],
                                    "request": json.dumps(records["request"])})};
        const _buffer = (which) => ({{
            getValue: () => _texts[which],
            setValue: (t) => {{ _texts[which] = t; }},
            getCursor: () => null,
            setCursor: () => {{}},
        }});
        const _docs = {{ task: _buffer("task"), request: _buffer("request") }};
        let _request = null;
        let _parses = {{ task: true, request: true }};
        // The save bar is not this file's subject: its writer is called.
        function _setDirty() {{}}
        function edited() {{ return true; }}
        {_fns(*_ON_THE_PAGE)}
        // THE PAGE OPENS THE CARD: its boxes wired, this machine's channels
        // asked, and the card filled from the job request.
        wireChannels();
        await __quiet();
        // THE PERSON SETS IT.  The triggers and the hours, each written on
        // its own `change` (`watchAskControls`), as the write after them
        // here; then the channel boxes, each through its own listener.
        if ({json.dumps(scf)}) $("ts-notify-scf").click();
        if ({json.dumps(periodic)}) $("ts-notify-periodic").click();
        if ({json.dumps(hours)} !== null) $("ts-notify-hours").value = {json.dumps(hours)};
        applyNotifyToDoc();
        if (!{json.dumps(every)}) $("ts-notify-all").click();
        for (const name of {json.dumps(list(untick))}) {{
            $("ts-notify-channels").querySelectorAll("input[type=checkbox]")
                .find((box) => box.value === name).click();
        }}
        await __quiet();
        console.log(JSON.stringify(_texts.request));
    """, globals_js=served(app_at)))


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the card (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.fixture
def calculation(tmp_path, monkeypatch):
    """A calculation as `molbuilder describe` wrote it."""
    from support.road import describe_calculation
    return describe_calculation(tmp_path, monkeypatch)


@pytest.fixture
def request_doc(calculation):
    """The open job request the card writes into (`job-request.md` § 3.5)."""
    from support.road import read_request
    return read_request(calculation)


@pytest.fixture
def records(calculation, request_doc):
    """The calculation's two records, as the editor's buffers hold them: its
    description's text, and its job request."""
    from molbuilder.task import path_of
    return {"task": path_of(calculation).read_text(), "request": request_doc}


#: The channels a person set up on This machine -- each saved through its
#: route, with an address and, for one, a signing key.
_SET_UP = {"team": {"url": "https://hooks.slack.com/services/T00/B00/XXXX"},
           "lab": {"url": "https://lab.example.org/molbuilder/hook",
                   "key": "sk-live-DEADBEEF"},
           "phone": {"url": "https://ntfy.sh/a-topic-of-my-own"}}


@pytest.fixture
def channels():
    """The channels of :data:`_SET_UP`, saved through `PUT
    /api/notify/channels/<name>` as the This machine tab saves them: their
    names, in the order `/api/notify/channels` answers them -- the order the
    card paints its ticks in."""
    from molbuilder.web.app import create_app
    web = create_app(config={}).test_client()
    for name, spec in _SET_UP.items():
        r = web.put(f"/api/notify/channels/{name}", json=spec)
        assert r.status_code == 200, r.get_json()
    rows = web.get("/api/notify/channels").get_json()["channels"]
    assert sorted(r["name"] for r in rows) == sorted(_SET_UP), rows
    return [r["name"] for r in rows]


# --------------------------------------------------------------------- #
#  what the ticks produce                                                #
# --------------------------------------------------------------------- #

def test_nothing_ticked_writes_no_key(app_at, records):
    """No block is no notification (`run-reports.md` § 2): nothing set up
    writes no `notify` key, so a description that reports on nothing
    round-trips byte-identical -- the hours box holding its offered number
    all the while: a number in an unticked row is not an answer, and
    writing it would turn the box's own default into a cadence nobody
    chose."""
    doc = _written(app_at, records)
    assert "notify" not in doc


def test_each_trigger_alone_is_a_valid_policy(app_at, records):
    """They combine with OR -- checkboxes, not a picker.  A block is written
    whole (`stages.md` § 6.9): the trigger not ticked as `false` or `null`
    (never), every channel and every field as `["*"]`; the hours a JSON
    number -- `job_request.py` refuses `"6"` and `"6h"`, and a value that
    changes meaning crossing a boundary is how "4h" reached sbatch as
    `-t 4h`."""
    scf = _written(app_at, records, scf=True)
    assert scf["notify"] == {"on_scf_converged": True, "every_hours": None,
                             "channels": ["*"], "report": ["*"]}

    per = _written(app_at, records, periodic=True, hours="4")
    assert per["notify"] == {"on_scf_converged": False, "every_hours": 4,
                             "channels": ["*"], "report": ["*"]}


def test_both_together(app_at, records):
    doc = _written(app_at, records, scf=True, periodic=True, hours="2.5")
    assert doc["notify"] == {"on_scf_converged": True, "every_hours": 2.5,
                             "channels": ["*"], "report": ["*"]}


@pytest.mark.parametrize("bad", ["", "0", "-3"])
def test_a_period_that_is_not_a_positive_number_writes_nothing(bad, app_at,
                                                               records):
    """`job_request.py` refuses these by name on the way in.  The page must not
    hand it one to refuse: a description written from this tab should be
    readable by the tool that wrote it."""
    doc = _written(app_at, records, periodic=True, hours=bad)
    assert "notify" not in doc, f"{bad!r} became a policy"


# --------------------------------------------------------------------- #
#  what it must never write                                              #
# --------------------------------------------------------------------- #

def test_no_destination_or_credential_can_reach_the_description(
        app_at, records, channels):
    """The card has no field for one and must never grow a key for one.

    A `job-request.json` travels -- to a cluster, into a composed copy, to
    whoever is handed the calculation.  `job_request.py`'s key allowlist refuses
    a `url` outright; this is the same rule one layer up, where the value
    would have to originate.  Every channel this machine has, each ticked by
    name: the names travel, and none of the addresses or the key the
    channels were saved with.
    """
    doc = _written(app_at, records, scf=True, periodic=True, every=False)
    assert doc["notify"]["channels"] == channels
    blob = json.dumps(doc).lower()
    for leak in ("url", "webhook", "token", "authorization", "slack", "bearer",
                 *(v.lower() for spec in _SET_UP.values()
                   for v in spec.values())):
        assert leak not in blob, f"the page wrote {leak!r} into the description"


# --------------------------------------------------------------------- #
#  reading a description back into the card                              #
# --------------------------------------------------------------------- #

def test_an_unparseable_document_loses_nothing():
    """Mid-edit the editor holds bytes that are not JSON.  The card's write
    is refused and said, and the text somebody is typing is left as typed
    (`task-setup.md` § 9a)."""
    fns = _fns(*_WRITER,
               # `patchDoc` refuses through it, naming the records' tabs.
               "const RECORDS = ['task', 'request'];",
               ("function refuseUnparsed()", "/** Whether record"))
    got = run_node([], f"""
        // The card's SCF trigger ticked, every channel as the page opens it.
        const _els = {{ "ts-notify-scf": {{ checked: true }},
                       "ts-notify-all": {{ checked: true }} }};
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
    """)
    assert got["doc"] == "{ not json at all", \
        "the page rewrote text it could not parse"
    assert got["said"] == ["refuse"], "the refused write was not said"


# --------------------------------------------------------------------- #
#  which channels -- three states, and they are not two                  #
# --------------------------------------------------------------------- #

def test_every_channel_is_written_every_never_as_this_machines_names(
        app_at, records, channels):
    """The default: use whatever is set up wherever this lands, written
    `["*"]` (`run-reports.md` § 3.0) -- each of this machine's channels
    ticked beside it.  Writing the ticked names out instead would freeze a
    travelling description to the machine it happened to be written on."""
    doc = _written(app_at, records, scf=True)
    assert doc["notify"]["channels"] == ["*"]


def test_a_subset_travels_as_names(app_at, records, channels):
    """A name is a label the person chose.  It grants nothing, so it is safe
    to carry where an address is not (`run-reports.md` § 1)."""
    doc = _written(app_at, records, scf=True, every=False, untick=["lab"])
    assert doc["notify"]["channels"] == [n for n in channels if n != "lab"]


def test_ticking_nothing_writes_an_EMPTY_LIST_not_nothing(app_at, records,
                                                          channels):
    """None is `[]`, a state of its own: read as every channel, unticking
    them all would send reports to every channel the person had just turned
    off (`run-reports.md` § 3.0)."""
    doc = _written(app_at, records, scf=True, every=False, untick=channels)
    assert doc["notify"]["channels"] == []


def test_a_card_write_moves_the_MODEL_too_not_only_the_buffer(request_doc):
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
    out = run_node([], f"""
        let _doc = {json.dumps(json.dumps(request_doc, indent=2))};
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
        {_fns(*_WRITER)}
        applyNotifyToDoc();                       // a card writes
        const afterPatch = JSON.parse(_doc).notify || null;
        // ...and now the thing that used to destroy it: the model is
        // serialised over the buffer.
        _doc = JSON.stringify(_request, null, 2) + "\\n";
        console.log(JSON.stringify({{
            afterPatch: afterPatch,
            afterSync: JSON.parse(_doc).notify || null,
        }}));
    """)
    assert out["afterPatch"], "the card wrote nothing at all"
    assert out["afterSync"] == out["afterPatch"], (
        "serialising the model destroyed what the card had just written -- "
        "`patchDoc` did not move `_request` with the buffer")


def test_a_card_that_could_not_ask_keeps_the_descriptions_report_list(
        request_doc):
    """`stages.md` § 6.9: which report fields a calculation can carry is the
    SERVER's answer (`GET /api/notify/report-fields`).  Until it has answered
    -- or when it could not -- the card does not know, and not knowing is not
    "none": it took the unanswered question for an empty list, so the next
    tick anywhere on the card rewrote a description's `notify.report` to
    `[]`, the summary line alone.  Driven through the real functions, on a
    description whose report list a person chose, edited into its job
    request.

    MUTATION THIS MUST FAIL AGAINST: read an unknown field list as empty."""
    chosen = {**request_doc, "notify": {
        "on_scf_converged": True, "every_hours": None, "channels": ["*"],
        "report": ["energy", "n_iters"]}}
    fns = _fns(*_CHOOSING,
               ("function bufferText(which)", "/** Whether either buffer"),
               ("function currentRecord(which)", "function wireChannels()"))
    got = run_node([], f"""
        const _els = {{
            "ts-report-all":   {{ checked: false }},
            "ts-report-items": {{ querySelectorAll: () => [] }},
            "ts-notify-all":   {{ checked: true }},
            "ts-notify-scf":   {{ checked: true }},
        }};
        const $ = (id) => _els[id] || null;
        let _doc = {json.dumps(json.dumps(chosen))};
        const _docs = {{ request: {{ getValue: () => _doc }} }};
        {fns}
        console.log(JSON.stringify(notifyValues()));
    """)
    assert got == chosen["notify"], got

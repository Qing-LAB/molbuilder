"""The save bar says what is unsaved — `task-setup.md` § 7a, § 9a.1.

Every card of the page folds, the editor of `task.json` and
`job-request.json` among them, and Save sits in a bar pinned to the bottom of
the window. ONE rule is load-bearing:

    a buffer that differs from its file on disk is said, in the bar --
    either record's.

Getting that wrong is silent and expensive — you set up a calculation through
the cards, never press Save, and close the tab with nothing written. It was
wrong once already: "dirty" was measured against the last text *put into* the
editor, which the cards rewrite on every edit, so a card edit was never
unsaved. These drive the real functions rather than grepping for them, because
a source that merely CONTAINS `_setDirty` passes a substring check with the
comparison inverted.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "molbuilder/web/static/task-setup/viewer.js"


def _fns() -> str:
    src = VIEWER.read_text(encoding="utf-8")
    i = src.index("/** The records the editor holds")
    j = src.index("/* ---------- loading a folder ---------- */", i)
    body = src[i:j]
    # `ensureEditor` boots CodeMirror over the network; the harness supplies
    # an editor and a buffer per record of its own, which are the only parts
    # `setBuffers` touches.
    return body.replace("await ensureEditor()", "_cm")


def _doc() -> str:
    return """{ _v: "",
                setValue(v) { this._v = v; },
                getValue() { return this._v; },
                clearHistory() {} }"""


#: The bar's two elements, and the buffers.  `refreshSave` repaints why
#: Save is held and the cards' heads, and `saidSave` what the last save did
#: -- neither is what this file asks about.
HARNESS = f"""
    const _attrs = {{}};
    const _els = {{ "ts-dirty": {{ hidden: true }},
                   "ts-savebar": {{ setAttribute(k, v) {{ _attrs[k] = v; }} }} }};
    function $(id) {{ return _els[id] || null; }}
    function refreshSave() {{}}
    function saidSave() {{}}
    const _cm = {{ refresh() {{}} }};
    const _docs = {{ task: {_doc()}, request: {_doc()} }};
    let _disk = {{ task: "", request: "" }};
    let _parses = {{ task: true, request: true }};
    let _parsedText = {{ task: "", request: "" }};
"""


def _run(js: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    prog = HARNESS + _fns() + "\n(async () => {\n" + js + "\n})();"
    proc = subprocess.run([node, "--input-type=commonjs", "-e", prog],
                          capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _said(js_after: str) -> bool:
    """Whether the bar says *edited — not saved*: its words shown, and the
    bar marked so (the mark is what hides the "nothing unsaved" line)."""
    out = _run(js_after + """
        console.log(JSON.stringify({
            words: !_els["ts-dirty"].hidden,
            bar:   _attrs["data-dirty"] }));
    """)
    assert out["bar"] == ("true" if out["words"] else "false"), (
        "the bar's mark and its words disagree: " + json.dumps(out))
    return out["words"]


#: What the folder holds, read: both records' files.
_READ = ("await setBuffers({ task: '{\"a\":1}', request: '{\"r\":1}' },"
         " { fromDisk: true });")

#: (what happened, the buffers' moves, is it unsaved)
_ROWS = [
    ("what the folder holds is not an unsaved edit", _READ, False),
    ("a card edit of the description is unsaved",
     _READ + " await setBuffers({ task: '{\"a\":2}' });", True),
    ("a card edit of the job request alone is unsaved",
     _READ + " await setBuffers({ request: '{\"r\":2}' });", True),
    ("saving settles it -- the folder re-read carries the edit",
     _READ + " await setBuffers({ request: '{\"r\":2}' });"
     " await setBuffers({ task: '{\"a\":1}', request: '{\"r\":2}' },"
     " { fromDisk: true });", False),
    ("a folder with no description starts clean",
     "await setBuffers({ task: '', request: '' }, { fromDisk: true });",
     False),
]


@pytest.mark.parametrize("what,moves,unsaved", _ROWS,
                         ids=[r[0] for r in _ROWS])
def test_the_bar_says_what_is_not_on_disk(what, moves, unsaved):
    assert _said(moves) is unsaved, what


#: (what changed the buffers, each change's record and CodeMirror's word
#: for it, how many times the read-back draws the stage table)
_READ_BACK = [
    ("a hand edit in the editor is read back into the stage table",
     [("task", "+input")], 1),
    ("the page's own write is not drawn again after its fence",
     [("task", "setValue")], 0),
    ("a card's write inside a hand edit's pause leaves it read back",
     [("task", "+input"), ("request", "setValue")], 1),
]


@pytest.mark.parametrize("what,changes,drawn", _READ_BACK,
                         ids=[r[0] for r in _READ_BACK])
def test_the_stage_table_is_drawn_again_from_a_hand_edit_only(
        what, changes, drawn):
    """§ 7a: a card's write draws the cards once, under the fence; § 9a: a
    hand edit in the editor is read back into them, a card's write inside
    its pause too.  Through 2.68.2 the read-back drew the stage table again
    after the page's own write, after the fence's release, and a value typed
    in a stage cell and Tab left the person on the page's body.

    MUTATIONS THIS MUST FAIL AGAINST: the read-back drawing the stage table
    at every change of a buffer; the read-back judging by the last change
    of its pause alone."""
    out = _run("""
        globalThis._reparse = undefined;
        globalThis._handEdited = false;
        globalThis._mode = "description";
        globalThis._task = null;
        globalThis._request = null;
        let drawn = 0;
        globalThis.renderStages = () => { drawn++; };
        for (const f of ["readAsksFromRequest", "readNotifyFromRequest",
                         "renderQueues", "renderMachine", "renderNext",
                         "schedulePlan"]) {
            globalThis[f] = () => {};
        }
        READ
        for (const [which, origin] of CHANGES) {
            _docs[which].setValue(which === "task" ? '{"a":2}' : '{"r":2}');
            _bufferChanged(which, { origin });
        }
        await new Promise((r) => setTimeout(r, 450));
        console.log(JSON.stringify({ drawn }));
    """.replace("READ", _READ).replace("CHANGES", json.dumps(changes)))
    assert out == {"drawn": drawn}, what

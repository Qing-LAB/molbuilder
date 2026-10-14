"""The save bar says what is unsaved — `task-setup.md` § 7a, § 9a.1.

Every card of the page folds, the `task.json` editor among them, and Save
sits in a bar pinned to the bottom of the window. ONE rule is load-bearing:

    a buffer that differs from the file on disk is said, in the bar.

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
    i = src.index("/** Say whether the buffer differs from disk")
    j = src.index("/* ---------- loading a folder ---------- */", i)
    body = src[i:j]
    # `ensureEditor` boots CodeMirror over the network; the harness supplies a
    # buffer of its own, which is the only part `setEditorText` touches.
    return body.replace("await ensureEditor()", "_cm")


#: The bar's two elements, and the buffer.  `refreshSave` repaints why Save
#: is held and the cards' heads, and `saidSave` what the last save did --
#: neither is what this file asks about.
HARNESS = """
    const _attrs = {};
    const _els = { "ts-dirty": { hidden: true },
                   "ts-savebar": { setAttribute(k, v) { _attrs[k] = v; } } };
    function $(id) { return _els[id] || null; }
    function refreshSave() {}
    function saidSave() {}
    const _cm = { _v: "",
                  setValue(v) { this._v = v; },
                  getValue() { return this._v; },
                  clearHistory() {},
                  refresh() {} };
    let _diskText = "";
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


#: (what happened, the buffer's moves, is it unsaved)
_ROWS = [
    ("what the folder holds is not an unsaved edit",
     "await setEditorText('{\"a\":1}', { fromDisk: true });", False),
    ("a card edit is unsaved",
     "await setEditorText('{\"a\":1}', { fromDisk: true });"
     " await setEditorText('{\"a\":2}');", True),
    ("saving settles it -- the folder re-read carries the edit",
     "await setEditorText('{\"a\":1}', { fromDisk: true });"
     " await setEditorText('{\"a\":2}');"
     " await setEditorText('{\"a\":2}', { fromDisk: true });", False),
    ("a folder with no description starts clean",
     "await setEditorText('', { fromDisk: true });", False),
]


@pytest.mark.parametrize("what,moves,unsaved", _ROWS,
                         ids=[r[0] for r in _ROWS])
def test_the_bar_says_what_is_not_on_disk(what, moves, unsaved):
    assert _said(moves) is unsaved, what

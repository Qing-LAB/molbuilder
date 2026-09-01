"""The folded `task.json` editor — `task-setup.md` § 9a.1.

The editor sits in the main column, closed, so the page keeps two columns the
whole way down and the rail can travel with you. Folding it away puts the Save
button behind a click, which makes ONE rule load-bearing:

    a buffer that differs from the file on disk may never be hidden.

Getting that wrong is silent and expensive — you set up a calculation through
the cards, never open the fold, and close the tab with nothing written. It was
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


HARNESS = """
    const _els = { "ts-dirty": { hidden: true },
                   "ts-editor-card": { open: false } };
    function $(id) { return _els[id] || null; }
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


def _state(js_after: str) -> dict:
    return _run(js_after + """
        console.log(JSON.stringify({
            open:  _els["ts-editor-card"].open,
            dirty: !_els["ts-dirty"].hidden }));
    """)


def test_what_the_folder_holds_is_not_an_unsaved_edit():
    """Opening a folder must leave the fold closed — otherwise every folder
    opens with a wall of JSON in front of the cards, which is the state the
    fold exists to end."""
    out = _state('await setEditorText(\'{"a":1}\', { fromDisk: true });')
    assert out["dirty"] is False
    assert out["open"] is False


def test_a_card_edit_is_unsaved_and_shows_itself():
    """The failure this test exists for: a card writes the model into the
    buffer, and the page said nothing was pending because the buffer matched
    the last thing the page itself had written."""
    out = _state("""
        await setEditorText('{"a":1}', { fromDisk: true });
        await setEditorText('{"a":2}');          // a card changed the model
    """)
    assert out["dirty"] is True, "a card edit was not counted as unsaved"
    assert out["open"] is True, "the fold hid an unsaved edit"


def test_saving_settles_it():
    """`save()` ends by re-reading the folder, so the baseline moves and the
    flag clears. The fold is left OPEN — closing a card the user opened would
    be the page taking a decision back."""
    out = _state("""
        await setEditorText('{"a":1}', { fromDisk: true });
        await setEditorText('{"a":2}');
        await setEditorText('{"a":2}', { fromDisk: true });   // written, re-read
    """)
    assert out["dirty"] is False
    assert out["open"] is True


def test_a_folder_with_no_description_starts_clean():
    """An empty folder has nothing pending — "" IS what it holds."""
    out = _state('await setEditorText("", { fromDisk: true });')
    assert out["dirty"] is False
    assert out["open"] is False

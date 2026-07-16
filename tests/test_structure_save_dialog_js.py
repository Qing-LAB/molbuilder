"""Unit tests for the Save-panel name + overwrite confirm dialogs.

Pins the public API of ``molbuilder/web/static/lib/structure/
save-dialog.js`` — the two modals the Save panel routes a Save
click through:

  1. ``chooseSaveName(initialName)`` — confirm/edit the destination
     filename.  Resolves to the entered name (str) on Save, null
     on Cancel / ESC.
  2. ``confirmOverwrite(filename)`` — warn before clobbering an
     existing file.  Resolves true on Overwrite, false on Cancel.

Pattern + DOM stub mirror tests/test_structure_warning_modal_js.py
so the two dialogs stay in sync.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/structure/save-dialog.js"


_DOM_PRELUDE = r"""
function _mkEl(tag) {
    const el = {
        tagName:     String(tag).toUpperCase(),
        className:   "",
        id:          "",
        textContent: "",
        type:        "",
        value:       "",
        disabled:    false,
        _attrs:      new Map(),
        _children:   [],
        _listeners:  new Map(),
        _parent:     null,
        _open:       false,
        focused:     false,
        selected:    false,
        get parentNode() { return el._parent; },
        setAttribute: (k, v) => el._attrs.set(k, String(v)),
        getAttribute: (k) => el._attrs.has(k) ? el._attrs.get(k) : null,
        appendChild: (c) => {
            el._children.push(c);
            c._parent = el;
            return c;
        },
        removeChild: (c) => {
            const ix = el._children.indexOf(c);
            if (ix >= 0) el._children.splice(ix, 1);
            c._parent = null;
            return c;
        },
        addEventListener: (ev, fn) => {
            if (!el._listeners.has(ev)) el._listeners.set(ev, []);
            el._listeners.get(ev).push(fn);
        },
        dispatchEvent: (event) => {
            const lis = el._listeners.get(event.type) || [];
            for (const fn of lis.slice()) fn(event);
        },
        showModal: () => { el._open = true; },
        show:      () => { el._open = true; },
        close: () => {
            if (!el._open) return;
            el._open = false;
            el.dispatchEvent({ type: "close" });
        },
        focus:  () => { el.focused = true; },
        select: () => { el.selected = true; },
        querySelector: (sel) => {
            const a = sel.match(/^\[data-action="([^"]+)"\]$/);
            const r = sel.match(/^\[data-role="([^"]+)"\]$/);
            const key = a ? "data-action" : (r ? "data-role" : null);
            const want = (a || r) ? (a || r)[1] : null;
            if (!key) return null;
            function walk(n) {
                if (n.getAttribute && n.getAttribute(key) === want) {
                    return n;
                }
                for (const c of (n._children || [])) {
                    const x = walk(c);
                    if (x) return x;
                }
                return null;
            }
            return walk(el);
        },
    };
    return el;
}
const _body = _mkEl("body");
global.document = {
    createElement: (tag) => _mkEl(tag),
    body: _body,
};
global.window = global;
"""


def _run_node(snippet: str, *, prelude: str = "") -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        {_DOM_PRELUDE}
        {prelude}
        const dialog = require({json.dumps(str(module_path))});
        try {{
            (async () => {{
                {snippet}
            }})().catch(err => {{
                console.log(JSON.stringify({{
                    __test_unexpected_throw: true,
                    message: err && err.message ? err.message : String(err),
                }}));
            }});
        }} catch (err) {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
            }}));
        }}
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    last_line = proc.stdout.strip().splitlines()[-1]
    out = json.loads(last_line)
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, (
            "module threw: " + str(out)
        )
    return out


# ----- Surface presence + basic API ------------------------------ #


class TestSurface:

    def test_methods_callable(self):
        out = _run_node('''
            console.log(JSON.stringify({
                chooseSaveName:   typeof dialog.chooseSaveName,
                confirmOverwrite: typeof dialog.confirmOverwrite,
                isNameOpen:       typeof dialog.isNameOpen,
                isOverwriteOpen:  typeof dialog.isOverwriteOpen,
            }));
        ''')
        assert out == {
            "chooseSaveName":   "function",
            "confirmOverwrite": "function",
            "isNameOpen":       "function",
            "isOverwriteOpen":  "function",
        }


# ----- chooseSaveName behavior ----------------------------------- #


class TestChooseSaveName:

    def test_save_resolves_with_base_name_extension_stripped(self):
        """Clicking Save resolves with the trimmed BASE name -- the dialog owns
        no extension: a ".xyz" (or ".molstruct.json") the user typed out of habit
        is stripped, so the save can append the pair suffixes itself."""
        out = _run_node('''
            const p = dialog.chooseSaveName("water.xyz");
            // The newly-created dialog is the last child of body.
            const d = _body._children[_body._children.length - 1];
            const input = d.querySelector('[data-role="name-input"]');
            input.value = "renamed.xyz";
            d.querySelector('[data-action="save"]').dispatchEvent({type:"click"});
            const result = await p;
            console.log(JSON.stringify({result, open: dialog.isNameOpen()}));
        ''')
        assert out["result"] == "renamed"       # ".xyz" stripped -> base name
        assert out["open"] is False

    def test_preview_shows_both_output_files(self):
        """The dialog live-previews the PAIR the save writes (<name>.xyz +
        <name>.molstruct.json) so the user sees one name -> two files and never
        needs to type an extension."""
        out = _run_node('''
            dialog.chooseSaveName("");
            const d = _body._children[_body._children.length - 1];
            const input = d.querySelector('[data-role="name-input"]');
            const prev  = d.querySelector('[data-role="name-preview"]');
            input.value = "my_mol";
            input.dispatchEvent({type:"input"});
            console.log(JSON.stringify({ text: prev.textContent,
                                         hidden: !!prev.hidden }));
            dialog._reset();
        ''')
        assert out["hidden"] is False
        assert "my_mol.xyz" in out["text"]
        assert "my_mol.molstruct.json" in out["text"]

    def test_cancel_resolves_null(self):
        out = _run_node('''
            const p = dialog.chooseSaveName("water.xyz");
            const d = _body._children[_body._children.length - 1];
            d.querySelector('[data-action="cancel"]').dispatchEvent({type:"click"});
            const result = await p;
            console.log(JSON.stringify({result, open: dialog.isNameOpen()}));
        ''')
        assert out["result"] is None
        assert out["open"] is False

    def test_initial_name_prefills_input(self):
        out = _run_node('''
            dialog.chooseSaveName("foo.pdb");
            const d = _body._children[_body._children.length - 1];
            const input = d.querySelector('[data-role="name-input"]');
            console.log(JSON.stringify({value: input.value,
                                        focused: input.focused,
                                        selected: input.selected}));
            dialog._reset();
        ''')
        assert out["value"] == "foo.pdb"
        assert out["focused"] is True
        assert out["selected"] is True

    def test_empty_name_keeps_save_disabled(self):
        """Empty / whitespace-only filename must disable the Save
        button — silently saving to an empty path would clobber the
        directory entry."""
        out = _run_node('''
            dialog.chooseSaveName("");
            const d = _body._children[_body._children.length - 1];
            const save = d.querySelector('[data-action="save"]');
            const input = d.querySelector('[data-role="name-input"]');
            const initialDisabled = save.disabled;
            input.value = "  ";
            input.dispatchEvent({type:"input"});
            const whitespaceDisabled = save.disabled;
            input.value = "named.xyz";
            input.dispatchEvent({type:"input"});
            const namedDisabled = save.disabled;
            console.log(JSON.stringify({
                initialDisabled, whitespaceDisabled, namedDisabled,
            }));
            dialog._reset();
        ''')
        assert out["initialDisabled"] is True
        assert out["whitespaceDisabled"] is True
        assert out["namedDisabled"] is False

    def test_rejects_path_separators(self):
        """Path separators in the filename input MUST disable Save
        and surface an inline error.  Without this, a user typing
        ``../escape.xyz`` would smuggle a path through the dialog
        (server-side picker-root check would block it but the UX
        is clearer if we reject up front)."""
        out = _run_node('''
            dialog.chooseSaveName("water.xyz");
            const d = _body._children[_body._children.length - 1];
            const input = d.querySelector('[data-role="name-input"]');
            const save  = d.querySelector('[data-action="save"]');
            const err   = d.querySelector('[data-role="name-error"]');
            const results = [];
            for (const bad of ["../escape.xyz", "sub/file.xyz",
                                "back\\\\slash.xyz", ".", ".."]) {
                input.value = bad;
                input.dispatchEvent({type:"input"});
                results.push({bad, disabled: save.disabled,
                              hasErr: !err.hidden});
            }
            input.value = "ok.xyz";
            input.dispatchEvent({type:"input"});
            const okState = {disabled: save.disabled,
                             hasErr: !err.hidden};
            console.log(JSON.stringify({results, okState}));
            dialog._reset();
        ''')
        for entry in out["results"]:
            assert entry["disabled"] is True, (
                f"filename {entry['bad']!r} should disable Save"
            )
            assert entry["hasErr"] is True, (
                f"filename {entry['bad']!r} should show inline error"
            )
        assert out["okState"]["disabled"] is False
        assert out["okState"]["hasErr"] is False

    def test_esc_resolves_null(self):
        out = _run_node('''
            const p = dialog.chooseSaveName("water.xyz");
            const d = _body._children[_body._children.length - 1];
            d.close();  // native ESC -> close event
            const result = await p;
            console.log(JSON.stringify({result, open: dialog.isNameOpen()}));
        ''')
        assert out["result"] is None
        assert out["open"] is False

    def test_single_instance_reuses_promise(self):
        out = _run_node('''
            const p1 = dialog.chooseSaveName("water.xyz");
            const p2 = dialog.chooseSaveName("ethanol.xyz");
            const same = p1 === p2;
            const d = _body._children[_body._children.length - 1];
            d.querySelector('[data-action="cancel"]').dispatchEvent({type:"click"});
            await Promise.all([p1, p2]);
            console.log(JSON.stringify({same}));
        ''')
        assert out["same"] is True


# ----- confirmOverwrite behavior --------------------------------- #


class TestConfirmOverwrite:

    def test_overwrite_resolves_true(self):
        out = _run_node('''
            const p = dialog.confirmOverwrite("water.xyz");
            const d = _body._children[_body._children.length - 1];
            d.querySelector('[data-action="overwrite"]').dispatchEvent({type:"click"});
            const result = await p;
            console.log(JSON.stringify({result, open: dialog.isOverwriteOpen()}));
        ''')
        assert out["result"] is True
        assert out["open"] is False

    def test_cancel_resolves_false(self):
        out = _run_node('''
            const p = dialog.confirmOverwrite("water.xyz");
            const d = _body._children[_body._children.length - 1];
            d.querySelector('[data-action="cancel"]').dispatchEvent({type:"click"});
            const result = await p;
            console.log(JSON.stringify({result}));
        ''')
        assert out["result"] is False

    def test_cancel_is_default_focus(self):
        """Per the warning-modal pattern: destructive action (overwrite)
        requires explicit travel; Cancel gets the initial focus."""
        out = _run_node('''
            dialog.confirmOverwrite("water.xyz");
            const d = _body._children[_body._children.length - 1];
            const cancel = d.querySelector('[data-action="cancel"]');
            const overwrite = d.querySelector('[data-action="overwrite"]');
            console.log(JSON.stringify({
                cancelFocused: cancel.focused,
                overwriteFocused: overwrite.focused,
            }));
            dialog._reset();
        ''')
        assert out["cancelFocused"] is True
        assert out["overwriteFocused"] is False

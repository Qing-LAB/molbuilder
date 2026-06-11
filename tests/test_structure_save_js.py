"""Unit tests for the Save panel module.

Pins the public API of ``molbuilder/web/static/lib/structure/
save.js`` — the Sources-card panel that writes the workspace
structure back to disk.

Uses fake projects + structurePage + canvas surfaces so the
target-resolution + writeFile + markSavedTo flow can be exercised
without HTTP or sessionStorage.  The DOM wiring (``wirePanel``
against the actual button + readout) is exercised by an e2e test
in test_molbuilder_e2e.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/structure/save.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        function _mkFakeCanvas(initial) {{
            const state = Object.assign({{
                empty: true, dirty: false,
                structure: null,
                source: {{kind: "blank", file: null,
                         generator_input: null}},
                lastSaveTo: null,
            }}, initial || {{}});
            const calls = [];
            return {{
                isEmpty:        () => state.empty,
                isDirty:        () => state.dirty,
                getStructure:   () => state.structure,
                getSource:      () => state.source,
                getLastSavedTo: () => state.lastSaveTo,
                // Phase 10 — workspace-contract.md §2.1 — fake the
                // ws.* surface, not the legacy canvas.onChange.
                subscribe:      () => () => {{}},
                _calls:         () => calls.slice(),
            }};
        }}
        function _mkFakeProjects(writeImpl) {{
            const calls = [];
            return {{
                writeFile: (path, text) => {{
                    calls.push({{path, text}});
                    return Promise.resolve(writeImpl(path, text));
                }},
                _calls: () => calls.slice(),
            }};
        }}
        function _mkFakeStructurePage() {{
            const calls = [];
            return {{
                markSavedTo: (p) => calls.push(p),
                _calls:      () => calls.slice(),
            }};
        }}
        const save = require({json.dumps(str(module_path))});
        try {{
            (async () => {{
                {snippet}
            }})().catch(err => {{
                console.log(JSON.stringify({{
                    __test_unexpected_throw: true,
                    message: err && err.message ? err.message : String(err),
                    stack:   err && err.stack ? err.stack : null,
                }}));
            }});
        }} catch (err) {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
                stack:   err && err.stack ? err.stack : null,
            }}));
        }}
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    last_line = proc.stdout.strip().splitlines()[-1]
    out = json.loads(last_line)
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, (
            "module threw: " + str(out)
        )
    return out


# ----- Surface presence ------------------------------------------ #


class TestSurfacePresence:

    def test_methods_callable(self):
        out = _run_node('''
            console.log(JSON.stringify({
                configure:  typeof save.configure,
                save:       typeof save.save,
                targetPath: typeof save.targetPath,
                wirePanel:  typeof save.wirePanel,
            }));
        ''')
        for fn in ("configure", "save", "targetPath", "wirePanel"):
            assert out[fn] == "function"


# ----- targetPath resolution ------------------------------------- #


class TestTargetPath:

    def test_returns_last_save_to_when_set(self):
        """Once a Save has happened this session, the recorded path
        wins — even if source.file is different."""
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                source: {kind: "file", file: "/p/loaded.xyz"},
                lastSaveTo: "/p/saved.xyz",
            });
            save.configure({canvas: c, projects: {writeFile: () => {}},
                            structurePage: _mkFakeStructurePage()});
            console.log(JSON.stringify(save.targetPath()));
        ''')
        assert out == "/p/saved.xyz"

    def test_falls_back_to_source_file_when_kind_is_file(self):
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                source: {kind: "file", file: "/p/loaded.xyz"},
                lastSaveTo: null,
            });
            save.configure({canvas: c, projects: {writeFile: () => {}},
                            structurePage: _mkFakeStructurePage()});
            console.log(JSON.stringify(save.targetPath()));
        ''')
        assert out == "/p/loaded.xyz"

    def test_null_for_smiles_without_prior_save(self):
        """SMILES-generated structure with no last_save_to → null
        → Save button disables (Save-as comes later)."""
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                source: {kind: "smiles",
                         generator_input: {smiles: "CCO"}},
                lastSaveTo: null,
            });
            save.configure({canvas: c, projects: {writeFile: () => {}},
                            structurePage: _mkFakeStructurePage()});
            console.log(JSON.stringify(save.targetPath()));
        ''')
        assert out is None

    def test_null_on_empty_canvas(self):
        out = _run_node('''
            const c = _mkFakeCanvas();
            save.configure({canvas: c, projects: {writeFile: () => {}},
                            structurePage: _mkFakeStructurePage()});
            console.log(JSON.stringify(save.targetPath()));
        ''')
        assert out is None


# ----- save() flow ----------------------------------------------- #


class TestSaveFlow:

    def test_writes_to_source_file_and_marks_saved(self):
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz",
                            text: "3\\nH2O\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n"},
                source: {kind: "file", file: "/projects/p/water.xyz"},
                lastSaveTo: null,
            });
            const p = _mkFakeProjects(
                (path, text) => ({ok: true, path, size: text.length,
                                  mtime: 123}));
            const sp = _mkFakeStructurePage();
            save.configure({canvas: c, projects: p, structurePage: sp});
            const r = await save.save();
            console.log(JSON.stringify({
                envelope:        r,
                writeCalls:      p._calls(),
                markSavedCalls:  sp._calls(),
            }));
        ''')
        assert out["envelope"] == {
            "ok": True, "path": "/projects/p/water.xyz"}
        # Wrote canvas text to source file.
        assert out["writeCalls"] == [{
            "path": "/projects/p/water.xyz",
            "text": "3\nH2O\nO 0 0 0\nH 1 0 0\nH 0 1 0\n",
        }]
        # Orchestrator was told the save landed.
        assert out["markSavedCalls"] == ["/projects/p/water.xyz"]

    def test_uses_last_save_to_when_set(self):
        """A prior Save sets last_save_to; subsequent Save writes
        to the LATEST save path, not the original source.file."""
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "file", file: "/p/orig.xyz"},
                lastSaveTo: "/p/renamed.xyz",
            });
            const p = _mkFakeProjects(
                (path, text) => ({ok: true, path}));
            const sp = _mkFakeStructurePage();
            save.configure({canvas: c, projects: p, structurePage: sp});
            const r = await save.save();
            console.log(JSON.stringify({
                envelopePath:   r.path,
                writePath:      p._calls()[0].path,
                markSavedPath:  sp._calls()[0],
            }));
        ''')
        assert out["envelopePath"] == "/p/renamed.xyz"
        assert out["writePath"] == "/p/renamed.xyz"
        assert out["markSavedPath"] == "/p/renamed.xyz"


# ----- Refusal paths --------------------------------------------- #


class TestRefusals:

    def test_empty_canvas_refused(self):
        out = _run_node('''
            const c = _mkFakeCanvas({empty: true});
            save.configure({canvas: c, projects: {writeFile: () => {}},
                            structurePage: _mkFakeStructurePage()});
            const r = await save.save();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "No structure" in out["error"]

    def test_no_target_refused(self):
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "smiles",
                         generator_input: {smiles: "C"}},
                lastSaveTo: null,
            });
            let writeCalls = 0;
            const p = {writeFile: () => { writeCalls++;
                                          return Promise.resolve({ok:true});}};
            save.configure({canvas: c, projects: p,
                            structurePage: _mkFakeStructurePage()});
            const r = await save.save();
            console.log(JSON.stringify({envelope: r, writeCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "Save as" in out["envelope"]["error"]
        assert out["writeCalls"] == 0


# ----- Error paths ----------------------------------------------- #


class TestErrorPaths:

    def test_server_error_surfaces(self):
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "file", file: "/p/f.xyz"},
            });
            const p = _mkFakeProjects(
                () => ({ok: false, error: "permission denied"}));
            const sp = _mkFakeStructurePage();
            save.configure({canvas: c, projects: p, structurePage: sp});
            const r = await save.save();
            console.log(JSON.stringify({
                envelope: r,
                markSavedCalls: sp._calls(),
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert "permission denied" in out["envelope"]["error"]
        # MUST NOT mark saved when the write failed.
        assert out["markSavedCalls"] == []

    def test_network_throw_surfaces_envelope(self):
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "file", file: "/p/f.xyz"},
            });
            const p = {
                writeFile: () => Promise.reject(
                    new TypeError("Failed to fetch")),
            };
            const sp = _mkFakeStructurePage();
            save.configure({canvas: c, projects: p, structurePage: sp});
            const r = await save.save();
            console.log(JSON.stringify({
                envelope: r,
                markSavedCalls: sp._calls(),
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert "Failed to fetch" in out["envelope"]["error"]
        assert out["markSavedCalls"] == []


# ----- Configuration error surface ------------------------------- #


class TestConfigurationErrors:

    def test_save_rejects_when_canvas_unconfigured(self):
        out = _run_node('''
            const p = save.save();
            let rejected = false, msg = "";
            try { await p; }
            catch (e) { rejected = true; msg = e.message; }
            console.log(JSON.stringify({rejected, msg}));
        ''')
        assert out["rejected"] is True
        # Phase 10 (workspace-contract.md §1): renamed from
        # "canvas" to "workspace" — the legacy structureCanvas
        # store is no longer the save panel's dependency.
        assert "workspace" in out["msg"]

    def test_save_rejects_when_projects_unconfigured(self):
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "file", file: "/p/f.xyz"},
            });
            save.configure({canvas: c});
            const p = save.save();
            let rejected = false, msg = "";
            try { await p; }
            catch (e) { rejected = true; msg = e.message; }
            console.log(JSON.stringify({rejected, msg}));
        ''')
        assert out["rejected"] is True
        assert "projects" in out["msg"]

    def test_save_rejects_when_structurePage_unconfigured(self):
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "file", file: "/p/f.xyz"},
            });
            save.configure({
                canvas: c,
                projects: {writeFile: () => Promise.resolve({ok:true})},
                // structurePage MISSING
            });
            const p = save.save();
            let rejected = false, msg = "";
            try { await p; }
            catch (e) { rejected = true; msg = e.message; }
            console.log(JSON.stringify({rejected, msg}));
        ''')
        assert out["rejected"] is True
        assert "structurePage" in out["msg"]

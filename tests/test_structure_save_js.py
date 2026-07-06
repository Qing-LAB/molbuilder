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
                // 2026-06-09 (save dialog): save.js reads the current
                // sidebar dir as the destination root.  Mirrors
                // the workspace fixtures' ``/projects/p`` source dir
                // so the saved path round-trips to the original
                // location when the chosen filename matches the
                // source basename.
                getCurrentDir: () => "/projects/p",
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
        // save-flow.md §1: a Save now REQUIRES the name dialog (no default
        // filename).  Mount a fake that returns ``chosenName`` and records the
        // ``initial`` it was shown (must be blank) + overwrite confirmations.
        function _mountDialog(chosenName, opts) {{
            opts = opts || {{}};
            const calls = {{ chooseSaveName: [], confirmOverwrite: [] }};
            global.molbuilder = global.molbuilder || {{}};
            global.molbuilder.structureSaveDialog = {{
                chooseSaveName: (initial) => {{
                    calls.chooseSaveName.push(initial);
                    return Promise.resolve(chosenName);
                }},
                confirmOverwrite: (basename) => {{
                    calls.confirmOverwrite.push(basename);
                    return Promise.resolve(opts.confirmOverwrite !== false);
                }},
            }};
            return calls;
        }}
        // The unified save posts /api/workingcopy/save via global.fetch.  Mount a
        // fake returning {{status, json}} (or {{throw}}) + record the calls.
        function _mountFetch(responder) {{
            const calls = [];
            global.fetch = (url, opts) => {{
                let body = null;
                try {{ body = opts && opts.body ? JSON.parse(opts.body) : null; }}
                catch (_) {{ body = null; }}
                calls.push({{ url, body }});
                const res = responder ? responder(calls.length - 1, body)
                                      : {{ status: 200, json: {{ ok: true, saved: url }} }};
                if (res.throw) return Promise.reject(new TypeError(res.throw));
                return Promise.resolve({{
                    status: res.status,
                    json: () => Promise.resolve(res.json),
                }});
            }};
            return {{ calls: () => calls.slice() }};
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
            save.configure({canvas: c, projects: {writeFile: () => {}, getCurrentDir: () => "/projects/p"},
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
            save.configure({canvas: c, projects: {writeFile: () => {}, getCurrentDir: () => "/projects/p"},
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
            save.configure({canvas: c, projects: {writeFile: () => {}, getCurrentDir: () => "/projects/p"},
                            structurePage: _mkFakeStructurePage()});
            console.log(JSON.stringify(save.targetPath()));
        ''')
        assert out is None

    def test_null_on_empty_canvas(self):
        out = _run_node('''
            const c = _mkFakeCanvas();
            save.configure({canvas: c, projects: {writeFile: () => {}, getCurrentDir: () => "/projects/p"},
                            structurePage: _mkFakeStructurePage()});
            console.log(JSON.stringify(save.targetPath()));
        ''')
        assert out is None


# ----- save() flow ----------------------------------------------- #


class TestSaveFlow:

    def test_writes_named_file_via_one_workingcopy_save(self):
        """§4.0.1: the user names the file; the save makes ONE POST to
        /api/workingcopy/save carrying the whole dataset (xyz + sidecar) + target
        in the sidebar's current dir."""
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz",
                            text: "3\\nH2O\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n"},
                source: {kind: "file", file: "/projects/p/water.xyz"},
                lastSaveTo: null,
            });
            const sp = _mkFakeStructurePage();
            _mountDialog("water-modified.xyz");
            const f = _mountFetch(() => ({status: 200,
                json: {ok: true, saved: "/projects/p/water-modified.xyz"}}));
            save.configure({canvas: c, projects: _mkFakeProjects(() => ({ok:true})),
                            structurePage: sp});
            const r = await save.save();
            console.log(JSON.stringify({
                envelope:       r,
                fetch:          f.calls(),
                markSavedCalls: sp._calls(),
            }));
        ''')
        assert out["envelope"] == {
            "ok": True, "path": "/projects/p/water-modified.xyz"}
        assert len(out["fetch"]) == 1                 # ONE call, not two
        call = out["fetch"][0]
        assert call["url"] == "/api/workingcopy/save"
        assert call["body"]["target"] == "/projects/p/water-modified.xyz"
        assert call["body"]["data"]["xyz"] == "3\nH2O\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"
        assert call["body"]["overwrite"] is False     # first try is no-clobber
        assert out["markSavedCalls"] == ["/projects/p/water-modified.xyz"]

    def test_no_default_filename_dialog_gets_blank(self):
        """§1: NO default filename -- even with a prior last_save_to, the dialog
        opens BLANK (the Modify tab makes a modified version; we never pre-fill
        the loaded/last-saved name)."""
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "file", file: "/projects/p/orig.xyz"},
                lastSaveTo: "/projects/p/renamed.xyz",
            });
            const dlg = _mountDialog("chosen.xyz");
            const f = _mountFetch(() => ({status: 200,
                json: {ok: true, saved: "/projects/p/chosen.xyz"}}));
            save.configure({canvas: c, projects: _mkFakeProjects(() => ({ok:true})),
                            structurePage: _mkFakeStructurePage()});
            await save.save();
            console.log(JSON.stringify({
                initialShown: dlg.chooseSaveName,
                target:       f.calls()[0].body.target,
            }));
        ''')
        # Dialog was shown a BLANK initial -- no default, not the source name.
        assert out["initialShown"] == [""]
        # Saves to the sidebar dir + the user-chosen name.
        assert out["target"] == "/projects/p/chosen.xyz"


# ----- Refusal paths --------------------------------------------- #


class TestRefusals:

    def test_empty_canvas_refused(self):
        out = _run_node('''
            const c = _mkFakeCanvas({empty: true});
            save.configure({canvas: c, projects: {writeFile: () => {}, getCurrentDir: () => "/projects/p"},
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
                                          return Promise.resolve({ok:true});},
                       getCurrentDir: () => null};
            save.configure({canvas: c, projects: p,
                            structurePage: _mkFakeStructurePage()});
            const r = await save.save();
            console.log(JSON.stringify({envelope: r, writeCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        # 2026-06-09: generator workspaces now get a Save-as path if
        # the sidebar has a current directory.  The fake projects in
        # this test doesn't implement getCurrentDir, so save() falls
        # through to the "pick a project directory" refusal.
        msg = out["envelope"]["error"].lower()
        assert ("save as" in msg
                or "pick a project directory" in msg), (
            f"expected Save-as or pick-dir refusal; got {msg!r}"
        )
        assert out["writeCalls"] == 0


# ----- Error paths ----------------------------------------------- #


class TestErrorPaths:

    def test_server_error_surfaces(self):
        """A sidecar/write failure is SURFACED (returned), never swallowed --
        the whole point of the unification (no more silent lost .json)."""
        out = _run_node('''
            const c = _mkFakeCanvas({
                empty: false,
                structure: {source_format: "xyz", text: "x"},
                source: {kind: "file", file: "/p/f.xyz"},
            });
            const sp = _mkFakeStructurePage();
            _mountDialog("f.xyz");
            _mountFetch(() => ({status: 500,
                json: {ok: false, error: "permission denied"}}));
            save.configure({canvas: c, projects: _mkFakeProjects(() => ({ok:true})),
                            structurePage: sp});
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
            const sp = _mkFakeStructurePage();
            _mountDialog("f.xyz");
            _mountFetch(() => ({throw: "Failed to fetch"}));
            save.configure({canvas: c, projects: _mkFakeProjects(() => ({ok:true})),
                            structurePage: sp});
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
                projects: {getCurrentDir: () => "/projects/p"},
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

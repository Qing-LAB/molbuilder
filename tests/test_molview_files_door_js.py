"""The MolView files door (projects.md § 5; molview.md § 11.3-§ 11.4) — run.

The DOWNLOAD half executes here under Node against a minimal DOM, its
request answered by the app's own export route (`_node_esm.served`): the
claim that matters is BOTH FILES — the `.xyz` and the `.molstruct.json` —
reach the browser, because one without the other is a structure whose
labels were quietly dropped (§ 11.3), and 'the download does nothing' was
the user-reported bug this door closes.

The PROJECT half -- the save-where question, then the save route -- needs
a real DOM (this module's own dialogs) and is not run here: a browser saves
a run's structure through it in `test_results_export_e2e.py`
(`support.results_export.save_to_project`); the server's overwrite gate is
`test_structure_save_endpoint.py`; the menu-to-door handoff,
`test_molview_mount.py`.  No test runs the overwrite confirmation the door
asks between the two.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node, served

ROOT = Path(__file__).resolve().parents[1]
DOOR = ROOT / "molbuilder/web/static/lib/projects/molview-doors.js"

_DOM = r"""
globalThis.__downloads = [];
globalThis.URL = globalThis.URL || {};
URL.createObjectURL = () => "blob:fake";
URL.revokeObjectURL = () => {};
globalThis.document = {
    createElement: () => ({
        set download(v) { this._dl = v; },
        get download() { return this._dl; },
        click() { __downloads.push(this._dl); },
        remove() {},
    }),
    body: { appendChild() {} },
    head: { appendChild() {} },
};
globalThis.window = globalThis;
"""


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the door's export (`support.
    live_server.serve`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


def _run(app_at, snippet: str):
    prelude = (
        "const { molviewFiles } = await import("
        + json.dumps(DOOR.resolve().as_uri()) + ");\n"
    )
    return run_node([], prelude + snippet,
                    globals_js=_DOM + served(app_at))


def test_a_pair_downloads_as_one_archive_holding_both_files(app_at):
    """The pair leaves as ONE artifact (§ 11.3, 2026-08-19): two programmatic
    downloads is exactly what the browser's multiple-download policy
    swallows, and "the .json was missing" was the user's live report.  The
    structure carries a label, so the server's generator writes its
    metadata beside the coordinates; the archive is opened HERE with
    Python's own zipfile — the store-zip is a real zip or this test fails,
    not a blob with the right name — and holds the two files byte for byte
    as the server's answer carried them."""
    import base64
    import io
    import zipfile

    from support.envelope import envelope
    wire = envelope(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                    regions={"L-electrode": [0]})
    out = _run(app_at,
        """
        globalThis.__blobs = [];
        const RealBlob = Blob;
        globalThis.Blob = class extends RealBlob {
            constructor(parts, opts) { super(parts, opts); __blobs.push(this); }
        };
        const out = await molviewFiles.save("download", "wire",
            { structure: %s });
        const last = __blobs[__blobs.length - 1];
        const b64 = Buffer.from(await last.arrayBuffer()).toString("base64");
        console.log(JSON.stringify({
            out, downloads: __downloads, sent: __calls[0].body, b64,
            files: __answered[0].files,
        }));
        """ % json.dumps(wire)
    )
    assert out["out"]["ok"] is True
    assert out["downloads"] == ["wire.zip"], (
        f"a pair must leave as one archive: {out['downloads']}"
    )
    assert out["sent"]["name"] == "wire"
    written = {f["name"]: f["text"].encode() for f in out["files"]}
    assert sorted(written) == ["wire.molstruct.json", "wire.xyz"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(out["b64"]))) as z:
        assert sorted(z.namelist()) == sorted(written), (
            f"the archive lost a half: {z.namelist()}"
        )
        for name, blob in written.items():
            assert z.read(name) == blob, (
                f"{name}: the bytes did not survive the archive")


def test_the_servers_refusal_comes_back_as_the_doors_envelope(app_at):
    """The server's refusal -- an envelope with no atoms in it -- comes back
    as the door's envelope in the server's words, never a throw: the
    viewer's Export says what the door returns (`projects.md` § 3: the
    doors never throw, they return `{ok, ...}`)."""
    out = _run(app_at,
        """
        const failed = await molviewFiles.save("download", "x",
            { structure: {} });
        console.log(JSON.stringify({ failed, answered: __answered }));
        """
    )
    refused, = out["answered"]          # the door's one request: the export
    assert refused["ok"] is False
    assert out["failed"] == {"ok": False, "error": refused["error"]}


def test_the_gates_verdicts_ride_the_doors_result(app_at):
    """§ 11.3 / projects.md § 5 (2026-08-20): the server judges what leaves
    and sends `notices` beside the files — and the door carries them in its
    result instead of dropping them, which was the one place the whole
    chain went quiet.  The road's CO2 as a SIESTA vibration's decks hold it,
    in their order: the server puts it back in the user's order and says
    so."""
    from support.road import co2_in_engine_order
    co2 = co2_in_engine_order().to_dict()
    out = _run(app_at,
        "const r = await molviewFiles.save('download', 's',"
        " { structure: %s });\n"
        "console.log(JSON.stringify({ r, said: __answered[0].notices }));\n"
        % json.dumps(co2)
    )
    said = out["said"]
    assert said, "the export of a structure in an engine's order says nothing"
    assert out["r"]["ok"] is True
    assert out["r"]["notices"] == said


#: The save-where question's own dialogs, answered: the folder picker picks
#: ``P``, the name dialog answers ``globalThis.__typed``.
_QUESTION_ANSWERED = r"""
const { registerHooks } = await import("node:module");
const __stub = (src) => "data:text/javascript," + encodeURIComponent(src);
const __answers = {
    "../dialog.js": __stub("export async function askFields() "
                           + "{ return { name: globalThis.__typed }; }"),
    "../tree-picker.js": __stub("export async function pickPath() "
                                + "{ return 'P'; }"),
};
registerHooks({ resolve(spec, ctx, next) {
    return (spec in __answers
            && String(ctx.parentURL).endsWith("/lib/projects/dialogs.js"))
        ? { url: __answers[spec], shortCircuit: true } : next(spec, ctx);
} });
"""

DIALOGS = ROOT / "molbuilder/web/static/lib/projects/dialogs.js"


def test_a_name_typed_with_its_extension_is_saved_as_typed():
    """`projects.md` § 5: the save-where question asks the name without its
    extension and adds it once -- a name typed with it, in any case, is
    taken as typed: a second `.xyz` names a file nobody chose (V1.14, the
    2026-09-23 walk).  The question's own rule, so every door that asks it
    -- the viewer's export, the spectrum viewer's frame set -- names the
    file the same way."""
    out = run_node([], (
        "const { chooseSavePath } = await import("
        + json.dumps(DIALOGS.resolve().as_uri()) + ");\n"
        + r"""
        const saved = [];
        for (const typed of ["x", "x.xyz", "x.XYZ"]) {
            globalThis.__typed = typed;
            saved.push(await chooseSavePath({ ext: ".xyz" }));
        }
        console.log(JSON.stringify(saved));
    """), globals_js=_QUESTION_ANSWERED)
    assert out == ["P/x.xyz", "P/x.xyz", "P/x.XYZ"], out

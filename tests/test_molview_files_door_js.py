"""The MolView files door (projects.md § 5; molview.md § 11.3-§ 11.4) — run.

The DOWNLOAD half executes here under Node with a stubbed `fetch` and a
minimal DOM: the claim that matters is BOTH FILES — the `.xyz` and the
`.molstruct.json` — reach the browser, because one without the other is a
structure whose labels were quietly dropped (§ 11.3), and 'the download
does nothing' was the user-reported bug this door closes.

The PROJECT half's dialog flow needs a real DOM (this module's own dialogs)
and is covered at three other levels: the server route's overwrite contract
(test_web.py), the menu-to-door handoff (test_molview_mount.py), and the
live walk-through in the plan's done-when.  What the door makes of the
dialog's ANSWER is its own, and is run here with the dialog answered by a
stand-in (`_ANSWERED`).
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
DOOR = ROOT / "molbuilder/web/static/lib/projects/molview-doors.js"

_DOM = r"""
globalThis.__downloads = [];
globalThis.__calls = [];
globalThis.__replies = [];
globalThis.fetch = async (url, opts = {}) => {
    __calls.push({ url, body: opts.body ? JSON.parse(opts.body) : null });
    const hit = __replies.find((r) => url.includes(r.match)) || {};
    return { status: hit.http ?? 200, json: async () => hit.body ?? { ok: true } };
};
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


# The save-where dialog, answered: the door's `./dialogs.js` resolves to a
# module whose `chooseSavePath` answers `globalThis.__picked`.
_ANSWERED = r"""
const { registerHooks } = await import("node:module");
const __dialogs = "data:text/javascript," + encodeURIComponent(
    "export async function chooseSavePath() { return globalThis.__picked; }\n"
    + "export async function confirmDestructive() { return true; }\n");
registerHooks({ resolve(spec, ctx, next) {
    return spec === "./dialogs.js"
            && String(ctx.parentURL).endsWith("/lib/projects/molview-doors.js")
        ? { url: __dialogs, shortCircuit: true } : next(spec, ctx);
} });
"""


def _run(snippet: str, *, answered: bool = False):
    prelude = (
        "const { molviewFiles } = await import("
        + json.dumps(DOOR.resolve().as_uri()) + ");\n"
    )
    return run_node([], prelude + snippet,
                    globals_js=_DOM + (_ANSWERED if answered else ""))


def test_a_pair_downloads_as_one_archive_holding_both_files():
    """The pair leaves as ONE artifact (§ 11.3, 2026-08-19): two programmatic
    downloads is exactly what the browser's multiple-download policy
    swallows, and "the .json was missing" was the user's live report.  The
    archive is opened HERE with Python's own zipfile — the store-zip is a
    real zip or this test fails, not a blob with the right name."""
    import base64
    import io
    import zipfile

    out = _run(
        """
        globalThis.__replies = [{ match: "/api/structure/export", body: {
            ok: true,
            files: [{ name: "wire.xyz", text: "2..." },
                    { name: "wire.molstruct.json", text: '{"cell": 1}' }],
        }}];
        globalThis.__blobs = [];
        const RealBlob = Blob;
        globalThis.Blob = class extends RealBlob {
            constructor(parts, opts) { super(parts, opts); __blobs.push(this); }
        };
        const out = await molviewFiles.save("download", "wire",
            { structure: { elements: ["C"], positions: [[0,0,0]],
                           metadata: {} } });
        const last = __blobs[__blobs.length - 1];
        const b64 = Buffer.from(await last.arrayBuffer()).toString("base64");
        console.log(JSON.stringify({
            out, downloads: __downloads, sent: __calls[0].body, b64,
        }));
        """
    )
    assert out["out"]["ok"] is True
    assert out["downloads"] == ["wire.zip"], (
        f"a pair must leave as one archive: {out['downloads']}"
    )
    assert out["sent"]["name"] == "wire"
    assert "frames" not in out["sent"], "a one-frame export carries no frames"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(out["b64"]))) as z:
        assert sorted(z.namelist()) == ["wire.molstruct.json", "wire.xyz"], (
            f"the archive lost a half: {z.namelist()}"
        )
        assert z.read("wire.molstruct.json") == b'{"cell": 1}', (
            "the metadata bytes did not survive the archive"
        )


def test_a_single_file_downloads_bare():
    out = _run(
        """
        globalThis.__replies = [{ match: "/api/structure/export", body: {
            ok: true, files: [{ name: "plain.xyz", text: "2..." }],
        }}];
        const out = await molviewFiles.save("download", "plain",
            { structure: { elements: ["C"] } });
        console.log(JSON.stringify({ out, downloads: __downloads }));
        """
    )
    assert out["downloads"] == ["plain.xyz"], (
        "a metadata-less export needs no archive around one file"
    )


def test_failures_and_nonsense_come_back_as_envelopes_never_throws():
    out = _run(
        """
        globalThis.__replies = [{ match: "/api/structure/export",
                                  http: 500, body: { ok: false,
                                                     error: "no codec" } }];
        const failed = await molviewFiles.save("download", "x",
            { structure: {} });
        const empty = await molviewFiles.save("download", "x", null);
        const where = await molviewFiles.save("nowhere", "x",
            { structure: {} });
        const binEmpty = await molviewFiles.saveBinary("download", "x.png",
            null);
        console.log(JSON.stringify({ failed, empty, where, binEmpty }));
        """
    )
    assert out["failed"] == {"ok": False, "error": "no codec"}
    assert out["empty"]["ok"] is False
    assert "unknown destination" in out["where"]["error"]
    assert out["binEmpty"]["ok"] is False


def test_the_gates_verdicts_ride_the_doors_result():
    """§ 11.3 / projects.md § 5 (2026-08-20): the server judges what leaves
    (`checked_periodicity`) and sends `notices` beside the files — and the
    door carries them in its result instead of dropping them, which was the
    one place the whole chain went quiet."""
    out = _run(
        "__replies.push({ match: '/api/structure/export', body: { ok: true,"
        " files: [{ name: 's.xyz', text: '3\\n\\n' }],"
        " notices: [{ severity: 'warn', message: 'the origin is user-owned',"
        " about: 'cell.edit' }] } });\n"
        "const r = await molviewFiles.save('download', 's',"
        " { structure: { elements: ['O'] } });\n"
        "console.log(JSON.stringify(r));\n"
    )
    assert out["ok"] is True
    assert out["notices"] == [{"severity": "warn",
                               "message": "the origin is user-owned",
                               "about": "cell.edit"}]


#: The save-where question's own dialogs, answered: the folder picker picks
#: ``P``, the name dialog answers ``globalThis.__typed``, the page has a
#: project root.
_QUESTION_ANSWERED = r"""
const { registerHooks } = await import("node:module");
const __stub = (src) => "data:text/javascript," + encodeURIComponent(src);
const __answers = {
    "../dialog.js": __stub("export async function askFields() "
                           + "{ return { name: globalThis.__typed }; }"),
    "./state.js": __stub("export function getProjectsRoot() "
                         + "{ return '/root'; }"),
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

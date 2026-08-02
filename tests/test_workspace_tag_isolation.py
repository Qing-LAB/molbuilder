"""The tag — every test derived from ``docs/web/workspace.md``, never from the
source it checks.

The workspace keeps one copy of your work: a file on the server, under the
project directory. The **tag** says whose it is. § 4 promises two things, and
this file turns them from prose into something that fails when they stop being
true:

  1. One set of calls for saving and loading, with no way round it.
  2. What you save under your tag stays yours — another tag's save cannot change
     it, hide it or delete it, whichever of you saved first, and reading under
     one tag never gives you another's.

There used to be a second copy in the browser's own storage. Nothing ever
restored from it — every restore went to the server anyway — so it cost a write
on every edit and bought nothing. It is gone, and these tests are written against
what is left: the files.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "molbuilder" / "web" / "static" / "lib" / "workspace" / "dispatcher.js"
MODEL = REPO / "molbuilder" / "web" / "static" / "lib" / "molview" / "model.js"

# A stand-in server that keeps the files the real one would write: one per
# `<workspace_id>.<state_index>`, which is how the real one names them. Two tags
# landing in one file IS the bug, so it has to be visible here.
BROWSER = """
globalThis.window = globalThis;
globalThis.document = { addEventListener() {}, body: { dataset: {} } };
globalThis.addEventListener = () => {};
globalThis.dispatchEvent = () => {};
globalThis.CustomEvent = class { constructor(t, o) { this.detail = o && o.detail; } };

globalThis.files = {};          // "<workspace_id>.<state_index>" -> what was saved
globalThis.fetch = async (route, init) => {
    const body = JSON.parse(init.body);
    const name = body.workspace_id + "." + body.state_index;
    if (route.indexOf("write") !== -1) {
        globalThis.files[name] = body.data;
        return { ok: true, status: 200, json: async () => ({ ok: true }) };
    }
    if (route.indexOf("read") !== -1) {
        const there = name in globalThis.files;
        return { ok: true, status: 200,
                 json: async () => ({ ok: true,
                                      data: there ? globalThis.files[name] : null }) };
    }
    if (route.indexOf("prune") !== -1) {
        Object.keys(globalThis.files).forEach((k) => {
            const cut = k.lastIndexOf(".");
            if (k.slice(0, cut) === body.workspace_id
                    && Number(k.slice(cut + 1)) > body.above_index) {
                delete globalThis.files[k];
            }
        });
        return { ok: true, status: 200, json: async () => ({ ok: true }) };
    }
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
};
"""

# The load route answers with a one-atom molecule; the state-timeline routes go to
# the stand-in server above. Used by the two round-trip tests at the end.
WITH_A_SERVER = BROWSER + """
const _files = globalThis.fetch;
globalThis.fetch = async (route, init) => {
    if (route.indexOf("state-timeline") !== -1) return _files(route, init);
    return { ok: true, status: 200, json: async () => ({
        ok: true,
        atoms: [{ index: 0, element: "H", x: 0, y: 0, z: 0, regions: [] }],
    }) };
};
"""

PRELUDE = f"""
await import({json.dumps(WS.resolve().as_uri())});
const ws = globalThis.molbuilder.workspace;

// The write is sent without waiting, so give it a moment to arrive.
async function settle() {{ await new Promise((r) => setTimeout(r, 20)); }}
function fileNames() {{ return Object.keys(globalThis.files).sort(); }}
async function readBack(tag, step) {{
    return ws.readState({{ workspace_id: ws.workspaceId(tag), state_index: step }});
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=BROWSER)


# ---------------------------------------------------------------------------
# What you save under your tag stays yours
# ---------------------------------------------------------------------------

def test_one_tag_cannot_overwrite_another_in_either_order():
    """Two savers on one page, saving the same step number.

    Both orders are run because the failure this replaces was order-dependent:
    when one setting decided whose slot a save went into, whoever wrote last
    owned it — so a test that only saved A then B could pass while B then A lost
    everything.
    """
    out = _run(
        """
        ws.persist("modify", "A's molecule",
                   { workspace_id: ws.workspaceId("modify"), state_index: 1 });
        ws.persist("results:structure", "B's molecule",
                   { workspace_id: ws.workspaceId("results:structure"), state_index: 1 });
        await settle();
        const afterAB = { a: await readBack("modify", 1),
                          b: await readBack("results:structure", 1) };

        // The same thing in the other order, under two more tags.
        ws.persist("second:b", "B again",
                   { workspace_id: ws.workspaceId("second:b"), state_index: 1 });
        ws.persist("second:a", "A again",
                   { workspace_id: ws.workspaceId("second:a"), state_index: 1 });
        await settle();
        const afterBA = { a: await readBack("second:a", 1),
                          b: await readBack("second:b", 1) };

        console.log(JSON.stringify({ afterAB, afterBA, files: fileNames() }));
        """
    )
    assert out["afterAB"] == {"a": "A's molecule", "b": "B's molecule"}, (
        f"one tag's save changed another's: {out['afterAB']}"
    )
    assert out["afterBA"] == {"a": "A again", "b": "B again"}, (
        f"the same, in the other order: {out['afterBA']}"
    )
    assert len(out["files"]) == 4, (
        f"four tags saving step 1 did not make four files: {out['files']}"
    )


def test_reading_one_tag_never_returns_anothers():
    """A tag that has saved nothing reads as empty even when another tag is full
    — otherwise a page opening for the first time restores somebody else's
    molecule and believes it is its own.
    """
    out = _run(
        """
        ws.persist("modify", "only mine",
                   { workspace_id: ws.workspaceId("modify"), state_index: 0 });
        await settle();
        console.log(JSON.stringify({
            mine:     await readBack("modify", 0),
            stranger: await readBack("results:trajectory", 0),
        }));
        """
    )
    assert out["mine"] == "only mine"
    assert out["stranger"] is None, "a tag read back another tag's saved work"


def test_a_tags_id_is_stable_and_differs_from_the_next_tags():
    """The id names the file, so it has to be the same each time a tag comes back
    — otherwise a page reopening looks for its work under a name it has never
    used and finds nothing. And it has to differ between tags, or two savers
    write into one file.

    One remembered id passes the first half and fails the second, which is why
    both are here.
    """
    out = _run(
        """
        console.log(JSON.stringify({
            a1: ws.workspaceId("modify"),
            a2: ws.workspaceId("modify"),
            b1: ws.workspaceId("results:structure"),
            b2: ws.workspaceId("results:structure"),
        }));
        """
    )
    assert out["a1"] == out["a2"], "a tag's id changed between two asks"
    assert out["b1"] == out["b2"], "a tag's id changed between two asks"
    assert out["a1"] != out["b1"], (
        "two tags share an id, so their files are named the same and they write "
        "over each other's history"
    )
    assert ":" not in out["b1"], (
        f"the id has to work as a file name — the server takes only letters, "
        f"digits, _ and -: {out['b1']}"
    )


# ---------------------------------------------------------------------------
# One way in, and it does not read what it stores
# ---------------------------------------------------------------------------

def test_the_workspace_does_not_open_what_it_was_given():
    """It cannot tell a molecule from a shopping list.

    It once had two functions that opened your saved work to answer "is there a
    molecule in here?" and "which file did it come from?". One got it wrong: a
    molecule built from SMILES was never in a file, so it answered "nothing
    saved" and the tab wiped work that was there. Both questions belong to
    whoever wrote the bytes.
    """
    out = _run(
        """
        const notAMolecule = { v: 1, state: { shopping: ["milk", "bread"] } };
        ws.persist("some-tag", notAMolecule,
                   { workspace_id: ws.workspaceId("some-tag"), state_index: 0 });
        await settle();
        console.log(JSON.stringify({
            back:    await readBack("some-tag", 0),
            surface: Object.keys(ws).sort(),
        }));
        """
    )
    assert out["back"] == {"v": 1, "state": {"shopping": ["milk", "bread"]}}, (
        f"the workspace changed what it was handed: {out['back']}"
    )
    for gone in ("hasRestorableSnapshot", "mountRestoreTarget"):
        assert gone not in out["surface"], (
            f"{gone} is back — it opens your saved work to answer a question "
            f"that belongs to whoever wrote it"
        )


def test_there_is_no_second_way_to_save():
    """Everything that saves goes through this one set of calls. The
    browser-storage half used to be published beside it, so anything could write
    bytes carrying no tag — and the tag is the only thing keeping two savers
    apart. It is gone, along with the copy it wrote.
    """
    out = _run(
        """
        console.log(JSON.stringify({
            secondDoor: typeof globalThis.molbuilder.workspaceSnapshot,
            surface:    Object.keys(ws).sort(),
        }));
        """
    )
    assert out["secondDoor"] == "undefined", (
        "the browser-storage helper is published again — bytes can be written "
        "without a tag"
    )
    assert "useNamespace" not in out["surface"], (
        "the setter is back; the tag is an argument now"
    )
    assert "readPersistedSnapshot" not in out["surface"], (
        "the browser copy is back — nothing ever restored from it, and it cost "
        "a write on every edit"
    )


def test_a_call_without_a_tag_is_refused_rather_than_defaulted():
    """A default would be worse than an error: every untagged caller would land
    in one shared place that looks exactly like a private one.
    """
    out = _run(
        """
        function refused(fn) { try { fn(); return false; } catch (_) { return true; } }
        console.log(JSON.stringify({
            persist: refused(() => ws.persist(undefined, "x", {})),
            id:      refused(() => ws.workspaceId()),
            empty:   refused(() => ws.workspaceId("")),
        }));
        """
    )
    for call, was_refused in out.items():
        assert was_refused is True, f"{call} accepted a missing tag"


# ---------------------------------------------------------------------------
# The round trip: what MolView saves is what comes back
# ---------------------------------------------------------------------------

def test_what_molview_saves_can_be_read_back():
    """The real MolView model against the real workspace: open a molecule, press
    Save state, then ask for it back the way a reopening page does.

    Nothing like this existed, and that is why saving stayed broken — the write
    was checked against a fake that stored anything, the read against a fake that
    returned anything, and the two halves were never put together.
    """
    out = run_node(
        [],
        f"""
        const {{ createModel }} = await import({json.dumps(MODEL.resolve().as_uri())});
        await import({json.dumps(WS.resolve().as_uri())});
        const ws = globalThis.molbuilder.workspace;

        const m = createModel({{ workspace: ws, owner: "modify" }});
        await m.installMolecule({{ text: "1\\n\\nH 0 0 0\\n", filename: "/p/mine.xyz" }});
        await m.save(1);
        await new Promise((r) => setTimeout(r, 20));

        const saved = await ws.readState(
            {{ workspace_id: ws.workspaceId("modify"), state_index: 1 }});
        console.log(JSON.stringify({{
            gotSomethingBack: saved !== null,
            stamped:  saved && saved.v,
            atoms:    saved && saved.state && saved.state.structure
                      ? saved.state.structure.elements : null,
            // Nothing anywhere in what was saved may name the file it came from.
            everything: JSON.stringify(saved),
        }}));
        """,
        globals_js=WITH_A_SERVER,
    )
    assert out["gotSomethingBack"] is True, (
        "the page asked for what it saved and got nothing back"
    )
    assert out["stamped"] == 1, (
        "the saved file carries no version stamp — these files outlive the code "
        "that wrote them, and a later reader cannot tell an old layout apart "
        "from bytes it understands"
    )
    assert out["atoms"] == ["H"], (
        f"the structure did not survive the round trip: {out['atoms']}"
    )
    assert "/p/mine.xyz" not in out["everything"], (
        "the saved state names the file it was read from. MolView tracks "
        "contents; which file they came out of is a fact about an operation the "
        "TAB performed, and a viewer holding a path is a second answer to a "
        "question the tab already owns (molview.md § 6.7)"
    )


def test_an_edit_and_a_saved_point_go_to_different_files():
    """Press Save state, then edit again without saving.

    The edit is kept so a reload does not lose it — but it must not land on the
    point you saved, or Retract would take you back to something that had been
    quietly rewritten. Two file names, one directory.
    """
    out = run_node(
        [],
        f"""
        const {{ createModel }} = await import({json.dumps(MODEL.resolve().as_uri())});
        await import({json.dumps(WS.resolve().as_uri())});
        const ws = globalThis.molbuilder.workspace;

        const m = createModel({{ workspace: ws, owner: "modify" }});
        await m.installMolecule({{ text: "1\\n\\nH 0 0 0\\n", filename: "/p/mine.xyz" }});
        await m.save(1);
        await new Promise((r) => setTimeout(r, 20));
        const point = JSON.stringify(await ws.readState(
            {{ workspace_id: ws.workspaceId("modify"), state_index: 1 }}));

        m.selection.writeLabel("frozen_atoms", "add", [0]);   // an edit, no save
        await new Promise((r) => setTimeout(r, 20));

        console.log(JSON.stringify({{
            pointUnchanged: JSON.stringify(await ws.readState(
                {{ workspace_id: ws.workspaceId("modify"), state_index: 1 }})) === point,
            files: Object.keys(globalThis.files).sort(),
            badge: m.uncommitted,
        }}));
        """,
        globals_js=WITH_A_SERVER,
    )
    assert out["pointUnchanged"] is True, (
        "an unsaved edit overwrote the point you saved — Retract would take you "
        "back to something you never chose to keep"
    )
    assert any("draft" in f for f in out["files"]), (
        f"the edit was not kept anywhere, so a reload loses it: {out['files']}"
    )
    assert out["badge"] is True, (
        "the edit is kept against an accident, but it is not on the sequence, "
        "and the badge has to say so"
    )

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

These tests are written against the files the server keeps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from molbuilder.structure import Structure
from support.envelope import answer
from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "molbuilder" / "web" / "static" / "lib" / "workspace" / "dispatcher.js"
#: The page that hosts MolView, as a test plays it: it hands the viewer the
#: one reader of a server's answer (`support/molview_host.mjs`).
HOST = Path(__file__).resolve().parent / "support" / "molview_host.mjs"

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
// Each answer as a browser hands it, for the page's one reader (`lib/server.js`).
const answered = (body) => new Response(JSON.stringify(body), { status: 200 });
globalThis.fetch = async (route, init) => {
    const body = JSON.parse(init.body);
    const name = body.workspace_id + "." + body.state_index;
    if (route.indexOf("write") !== -1) {
        globalThis.files[name] = body.data;
        return answered({ ok: true });
    }
    if (route.indexOf("read") !== -1) {
        const there = name in globalThis.files;
        return answered({ ok: true,
                          data: there ? globalThis.files[name] : null });
    }
    if (route.indexOf("prune") !== -1) {
        // A slow disk, when a test asks for one: the delete lands late.
        await new Promise((r) => setTimeout(r, globalThis.pruneDelay || 0));
        Object.keys(globalThis.files).forEach((k) => {
            const cut = k.lastIndexOf(".");
            if (k.slice(0, cut) === body.workspace_id
                    && Number(k.slice(cut + 1)) > body.above_index) {
                delete globalThis.files[k];
            }
        });
        return answered({ ok: true });
    }
    return answered({ ok: true });
};
"""

# The load route answers with a one-atom molecule, as the server builds it
# (`support.envelope.answer`, the load's: one frame); the storage routes go
# to the stand-in server above. Used by the two round-trip tests at the end.
WITH_A_SERVER = BROWSER + """
const _files = globalThis.fetch;
globalThis.fetch = async (route, init) => {
    if (route.indexOf("workspace-storage") !== -1) return _files(route, init);
    return answered(""" + json.dumps(answer(
        Structure(elements=["H"], positions=[[0.0, 0.0, 0.0]]),
        n_frames=1)) + """);
};
"""

PRELUDE = f"""
await import({json.dumps(WS.resolve().as_uri())});
const ws = globalThis.molbuilder.workspace;

// The write is sent without waiting, so give it a moment to arrive.
async function settle() {{ await new Promise((r) => setTimeout(r, 20)); }}
function fileNames() {{ return Object.keys(globalThis.files).sort(); }}
async function readBack(tag, step) {{ return ws.readState(tag, step); }}
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
        ws.persist("modify", "A's molecule", 1);
        ws.persist("results:structure", "B's molecule", 1);
        await settle();
        const afterAB = { a: await readBack("modify", 1),
                          b: await readBack("results:structure", 1) };

        // The same thing in the other order, under two more tags.
        ws.persist("second:b", "B again", 1);
        ws.persist("second:a", "A again", 1);
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


@pytest.mark.parametrize("save, read", [
    ('ws.persist(TAG, "only mine", 0)', "ws.readState(TAG, 0)"),
    ('ws.persistDraft(TAG, "only mine")', "ws.readDraft(TAG)"),
], ids=["a point", "a draft"])
def test_reading_one_tag_never_returns_anothers(save, read):
    """A tag that has saved nothing reads as empty even when another tag is full
    — otherwise a page opening for the first time restores somebody else's
    molecule and believes it is its own.  A draft as a point: the caller names
    its tag, never a file (`workspace.md` § 5).
    """
    mine = read.replace("TAG", '"modify"')
    stranger = read.replace("TAG", '"results:trajectory"')
    out = _run(
        f"""
        {save.replace("TAG", '"modify"')};
        await settle();
        console.log(JSON.stringify({{
            mine:     await {mine},
            stranger: await {stranger},
        }}));
        """
    )
    assert out["mine"] == "only mine"
    assert out["stranger"] is None, "a tag read back another tag's saved work"


def test_a_tags_draft_is_its_own_file_beside_its_points():
    """`workspace.md` § 2: two kinds of file — numbered points and one draft,
    rewritten each time — and the draft is never a point: dropping every point
    leaves it.  THE SILENT FAILURES: a draft written as a point is erased when
    the tag's points are dropped, and one written as a new file each time
    leaves several drafts, none of them the one a reload reads."""
    out = _run(
        """
        ws.persist("modify", "point 0", 0);
        ws.persistDraft("modify", "first draft");
        ws.persistDraft("modify", "the draft now");
        await settle();
        await ws.pruneStatesAbove("modify", -1);
        console.log(JSON.stringify({
            draft: await ws.readDraft("modify"),
            point: await ws.readState("modify", 0),
            files: fileNames(),
        }));
        """
    )
    assert out["draft"] == "the draft now", "the draft is not one file rewritten"
    assert out["point"] is None, "dropping every point left one"
    assert len(out["files"]) == 1, (
        f"the draft and the points were not two kinds of file: {out['files']}")


def test_a_dropped_tail_is_gone_before_the_next_point_is_written():
    """`workspace.md` § 9.2, *clear the tail before re-anchoring*: a save on
    top of a retracted point drops the points above it, and that delete
    finishes before the next point is written.  THE SILENT FAILURE (to
    2.69.25): the delete was sent beside the write queue, so a slow one landed
    after the next point and erased it — Retract then stepped back past work
    that had been saved."""
    out = _run(
        """
        globalThis.pruneDelay = 30;
        ws.persist("modify", "one", 1);
        ws.persist("modify", "two", 2);
        ws.pruneStatesAbove("modify", 1);    // retracted to 1, then a save
        ws.persist("modify", "two again", 2);
        await new Promise((r) => setTimeout(r, 120));
        console.log(JSON.stringify({ one: await readBack("modify", 1),
                                     two: await readBack("modify", 2) }));
        """
    )
    assert out == {"one": "one", "two": "two again"}, (
        f"the tail delete landed after the next point and erased it: {out}")


@pytest.mark.parametrize("one, other", [
    ("modify", "results:structure"),
    ("results:structure", "results-structure"),
    ("a:b", "a::b"),
    ("a_b", "a-b"),
    ("a_d", "a-"),
    ("a.b", "a:b"),
], ids=["two pages", "a colon and a dash", "one colon and two",
        "an underscore and a dash", "an underscore spelled as a dash is",
        "a dot and a colon"])
def test_a_tags_file_is_stable_and_differs_from_the_next_tags(one, other):
    """The tag names the file, so a tag saving again writes the same file —
    otherwise a page reopening looks for its work under a name it has never
    used and finds nothing. And two tags write two files, or two savers
    write into one (`workspace.md` § 4: two tags are never one id).

    One remembered name passes the first half and fails the second, which is
    why both are here.  THE SILENT FAILURE (to 2.69.18): every character
    outside the server's set became a dash, so `a-b` and `a:b` shared one
    slot.
    """
    out = _run(
        f"""
        ws.persist({json.dumps(one)}, "one, first", 1);
        ws.persist({json.dumps(one)}, "one, again", 1);
        ws.persist({json.dumps(other)}, "the other", 1);
        await settle();
        console.log(JSON.stringify({{
            one: await readBack({json.dumps(one)}, 1),
            other: await readBack({json.dumps(other)}, 1),
            files: fileNames(),
        }}));
        """
    )
    assert out["one"] == "one, again" and out["other"] == "the other", out
    assert len(out["files"]) == 2, (
        f"a tag saving again made a second file, or two tags shared one: "
        f"{out['files']}")
    for got in out["files"]:
        assert re.fullmatch(r"[A-Za-z0-9_-]+\.1", got), (
            f"the file has to work as a file name — the server takes only "
            f"letters, digits, _ and - in an id: {got}")


def test_a_pages_saved_state_is_found_after_the_upgrade():
    """A tag of letters, digits and `:` keeps the id it had before 2.69.19
    (`workspace.md` § 4), so the points a page kept are where it looks for
    them."""
    out = _run("""
        ws.persist("results:structure", "kept", 3);
        await settle();
        console.log(JSON.stringify(fileNames()));""")
    assert out == ["ws-results-structure.3"], out


# ---------------------------------------------------------------------------
# One way in, and it does not read what it stores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["undefined", '""', '"modify:draft"'],
                         ids=["no tag", "an empty tag", "another tag's draft"])
def test_a_call_without_its_own_tag_is_refused_rather_than_defaulted(tag):
    """A default would be worse than an error: every untagged caller would land
    in one shared place that looks exactly like a private one.  And a tag
    ending in `:draft` names another tag's draft (`workspace.md` § 4), so it is
    refused as a missing one is — on every call.
    """
    out = _run(
        f"""
        function refused(fn) {{ try {{ fn(); return false; }} catch (_) {{ return true; }} }}
        console.log(JSON.stringify({{
            persist:      refused(() => ws.persist({tag}, "x", 1)),
            readState:    refused(() => ws.readState({tag}, 1)),
            prune:        refused(() => ws.pruneStatesAbove({tag}, 1)),
            persistDraft: refused(() => ws.persistDraft({tag}, "x")),
            readDraft:    refused(() => ws.readDraft({tag})),
        }}));
        """
    )
    for call, was_refused in out.items():
        assert was_refused is True, f"{call} accepted {tag} as its tag"


# ---------------------------------------------------------------------------
# The round trip: what MolView saves is what comes back
# ---------------------------------------------------------------------------

def test_what_molview_saves_can_be_read_back():
    """The real MolView model against the real workspace: open a molecule, press
    Save state, then ask for it back the way a reopening page does.  THE SILENT
    FAILURE: a workspace that opens what it stores to judge it -- one that once
    asked "which file did it come from?" answered "nothing saved" for a
    molecule built from SMILES, and the tab wiped work that was there.
    """
    out = run_node(
        [],
        f"""
        const {{ createModel }} = await import({json.dumps(HOST.as_uri())});
        await import({json.dumps(WS.resolve().as_uri())});
        const ws = globalThis.molbuilder.workspace;

        const m = createModel({{ workspace: ws, owner: "modify" }});
        await m.installMolecule({{ text: "1\\n\\nH 0 0 0\\n", filename: "/p/mine.xyz" }});
        await m.save(1);
        await new Promise((r) => setTimeout(r, 20));

        const saved = await ws.readState("modify", 1);
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
    """Press Save state, then edit again.

    THE INVARIANT: the edit must not land ON the point you saved, or Retract
    would take you back to something that had been quietly rewritten. Two file
    names, one directory.

    The edit also lays down a point of its own, at the next index (§ 11.2), so
    Retract steps back one edit rather than as far back as the last time
    somebody pressed the button.  The badge therefore goes back DOWN — the work
    is on the sequence.
    """
    out = run_node(
        [],
        f"""
        const {{ createModel }} = await import({json.dumps(HOST.as_uri())});
        await import({json.dumps(WS.resolve().as_uri())});
        const ws = globalThis.molbuilder.workspace;

        const m = createModel({{ workspace: ws, owner: "modify" }});
        await m.installMolecule({{ text: "1\\n\\nH 0 0 0\\n", filename: "/p/mine.xyz" }});
        await m.save(1);
        await new Promise((r) => setTimeout(r, 20));
        const point = JSON.stringify(await ws.readState("modify", 1));

        m.selection.writeLabel("frozen_atoms", "add", [0]);   // an edit, no save
        await new Promise((r) => setTimeout(r, 20));

        console.log(JSON.stringify({{
            pointUnchanged: JSON.stringify(
                await ws.readState("modify", 1)) === point,
            files: Object.keys(globalThis.files).sort(),
            badge: m.uncommitted,
            // AND THE EDIT'S OWN POINT, one index on from the saved one.
            newPoint: (await ws.readState("modify", 2)) !== null,
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
    assert out["badge"] is False, (
        "the badge still says there is work off the sequence, after the edit "
        "put itself on it (§ 11.2)"
    )
    assert out["newPoint"] is True, (
        "the edit laid down no point of its own, so Retract cannot step back "
        "one edit"
    )

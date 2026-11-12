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
#: What a page keeps in this browser tab -- where the tab's own id lives
#: (`runtime.md` § 3), loaded by the page header before any page's own script.
KEPT = REPO / "molbuilder" / "web" / "static" / "lib" / "page-prefs.js"
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

// The browser tab's own storage, as a page has it -- carried whole from an
// earlier run when a test plays that tab reloaded, empty for a new tab.
const _tabStore = new Map(Object.entries(globalThis.__tabStorage || {}));
globalThis.sessionStorage = {
    getItem: (k) => (_tabStore.has(k) ? _tabStore.get(k) : null),
    setItem: (k, v) => { _tabStore.set(k, String(v)); },
    removeItem: (k) => { _tabStore.delete(k); },
};
globalThis.tabStorage = () => Object.fromEntries(_tabStore);

// "<workspace_id>.<state_index>" -> what was saved: the server's folder,
// carried from an earlier run when a test plays a second visit to it.
globalThis.files = Object.assign({}, globalThis.__files || {});
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
await import({json.dumps(KEPT.resolve().as_uri())});
await import({json.dumps(WS.resolve().as_uri())});
const ws = globalThis.molbuilder.workspace;

// The write is sent without waiting, so give it a moment to arrive.
async function settle() {{ await new Promise((r) => setTimeout(r, 20)); }}
function fileNames() {{ return Object.keys(globalThis.files).sort(); }}
async function readBack(tag, step) {{ return ws.readState(tag, step); }}
"""


def _run(snippet: str, *, tab_storage=None, files=None):
    """``snippet`` on a page in one browser tab -- ``tab_storage`` what that
    tab kept in an earlier run (none: a tab opened fresh), ``files`` the
    server's folder as an earlier run left it (none: an empty one)."""
    seed = (f"globalThis.__tabStorage = {json.dumps(tab_storage or {})};\n"
            f"globalThis.__files = {json.dumps(files or {})};\n")
    return run_node([], PRELUDE + snippet, globals_js=seed + BROWSER)


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


def test_an_edits_draft_never_waits_behind_the_points_delete():
    """`workspace.md` § 9.2: the delete of abandoned points is ordered before
    the tag's next POINT -- and the draft, a file of its own, waits for
    nothing.  THE SILENT FAILURE (2.69.26-2.69.28): one queue for every write
    held an edit's draft behind its point and the delete, and a reload that
    came first found the draft from before the edit -- the Molbuilder tab
    showed the file, not the edit."""
    out = _run("""
        globalThis.pruneDelay = 300;
        ws.persist("modify", "point 1", 1);
        ws.pruneStatesAbove("modify", 1);
        ws.persistDraft("modify", "the edit");
        await new Promise((r) => setTimeout(r, 60));
        console.log(JSON.stringify({ draft: await ws.readDraft("modify") }));
    """)
    assert out == {"draft": "the edit"}, (
        f"the edit's draft was still queued behind the delete: {out}")


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


@pytest.mark.parametrize("same_tab, finds", [
    (True, "one's"), (False, None)],
    ids=["the same tab reloaded finds its work",
         "another tab finds nothing and leaves the first tab's alone"])
def test_each_browser_tab_has_its_own_saved_work(same_tab, finds):
    """`workspace.md` § 4 (the user, 2026-11-11: one set per tab): a tab
    keeps its own id, so a reload of it finds the work it saved, and another
    tab, opening on the same folder, finds none of it and changes none of
    it.  THE SILENT FAILURE (to 2.69.28): every tab read and wrote one file
    per page, so a second tab opened on the Molbuilder tab showed the first
    tab's unsaved molecule, and its saves overwrote the first tab's."""
    first = _run("""
        ws.persist("modify", "one's", 1);
        await settle();
        console.log(JSON.stringify({ files: globalThis.files,
                                     tab: tabStorage() }));""")
    second = _run("""
        const found = await readBack("modify", 1);
        ws.persist("modify", "two's", 1);
        await settle();
        console.log(JSON.stringify({ found, files: globalThis.files }));""",
        tab_storage=first["tab"] if same_tab else None,
        files=first["files"])
    assert second["found"] == finds, second
    if not same_tab:
        assert "one's" in second["files"].values(), (
            "the second tab's save changed the first tab's work: "
            f"{second['files']}")


# ---------------------------------------------------------------------------
# One way in: a call names its own tag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ['""', '"modify:draft"'],
                         ids=["no tag", "a tag ending in :draft"])
def test_a_tag_that_is_not_one_of_its_own_is_refused(tag):
    """`workspace.md` § 4: every call names its tag -- *a tag never ends in
    `:draft` -- refused, as a missing tag is*.  THE SILENT FAILURES: a call
    with no tag writing into one shared slot that looks exactly like a
    private one; a tag ending in `:draft` writing its points into the id of
    another tag's draft.  One example per clause of the one rule
    (`process/testing.md` § 3b)."""
    out = _run(
        f"""
        let refused = false;
        try {{ ws.persist({tag}, "x", 0); }} catch (_) {{ refused = true; }}
        console.log(JSON.stringify(refused));
        """
    )
    assert out is True, f"{tag} was taken as a tag of its own"


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
        await import({json.dumps(KEPT.resolve().as_uri())});
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
        await import({json.dumps(KEPT.resolve().as_uri())});
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

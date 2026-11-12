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

The page's requests are answered by the real routes (`_node_esm.served`;
`process/testing.md` § 4), each test's saves kept in its own store
(`conftest._isolated_workspace_store`), and what was saved is read back
through the page's own calls and counted in the server's folder.  Where a
test needs two requests in an order, it holds the real one (`__hold`), never
a time (§ 5).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node, served

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "molbuilder" / "web" / "static" / "lib" / "workspace" / "dispatcher.js"
#: What a page keeps in this browser tab -- where the tab's own id lives
#: (`runtime.md` § 3), loaded by the page header before any page's own script.
KEPT = REPO / "molbuilder" / "web" / "static" / "lib" / "page-prefs.js"
#: The page that hosts MolView, as a test plays it: it hands the viewer the
#: one reader of a server's answer (`support/molview_host.mjs`).
HOST = Path(__file__).resolve().parent / "support" / "molview_host.mjs"

# The browser around the page: its document and events, and the tab's own
# storage -- carried whole from an earlier run when a test plays that tab
# reloaded, empty for a new tab.
BROWSER = """
globalThis.window = globalThis;
globalThis.document = { addEventListener() {}, body: { dataset: {} } };
globalThis.addEventListener = () => {};
globalThis.dispatchEvent = () => {};
globalThis.CustomEvent = class { constructor(t, o) { this.detail = o && o.detail; } };

const _tabStore = new Map(Object.entries(globalThis.__tabStorage || {}));
globalThis.sessionStorage = {
    getItem: (k) => (_tabStore.has(k) ? _tabStore.get(k) : null),
    setItem: (k, v) => { _tabStore.set(k, String(v)); },
    removeItem: (k) => { _tabStore.delete(k); },
};
globalThis.tabStorage = () => Object.fromEntries(_tabStore);
"""

PRELUDE = f"""
await import({json.dumps(KEPT.resolve().as_uri())});
await import({json.dumps(WS.resolve().as_uri())});
const ws = globalThis.molbuilder.workspace;
async function readBack(tag, step) {{ return ws.readState(tag, step); }}
"""

#: The prune request, held where a test needs it in an order.
PRUNE = '(c) => c.url.endsWith("/api/workspace-storage/prune")'


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the workspace's storage requests and
    MolView's load (`support.live_server.serve`, standing on this module's
    projects tree; the saves go to each test's own store)."""
    from support.live_server import serve
    with serve() as base:
        yield base


def _kept():
    """The files the server keeps for this test, as it lays them in its
    folder (`workspace_storage._state_dir`, this test's own store)."""
    from molbuilder.web.blueprints.workspace_storage import _state_dir
    return sorted(p.name for p in _state_dir().glob("*.wc.json"))


def _run(snippet: str, *, app_at=None, tab_storage=None):
    """``snippet`` on a page in one browser tab -- ``tab_storage`` what that
    tab kept in an earlier run (none: a tab opened fresh) -- its requests
    answered by the app at ``app_at`` (none: a snippet that sends none)."""
    seed = f"globalThis.__tabStorage = {json.dumps(tab_storage or {})};\n"
    return run_node([], PRELUDE + snippet, globals_js=(
        seed + BROWSER + (served(app_at) if app_at else "")))


# ---------------------------------------------------------------------------
# What you save under your tag stays yours
# ---------------------------------------------------------------------------

def test_one_tag_cannot_overwrite_another_in_either_order(app_at):
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
        await __quiet();
        const afterAB = { a: await readBack("modify", 1),
                          b: await readBack("results:structure", 1) };

        // The same thing in the other order, under two more tags.
        ws.persist("second:b", "B again", 1);
        ws.persist("second:a", "A again", 1);
        await __quiet();
        const afterBA = { a: await readBack("second:a", 1),
                          b: await readBack("second:b", 1) };

        console.log(JSON.stringify({ afterAB, afterBA }));
        """, app_at=app_at)
    assert out["afterAB"] == {"a": "A's molecule", "b": "B's molecule"}, (
        f"one tag's save changed another's: {out['afterAB']}"
    )
    assert out["afterBA"] == {"a": "A again", "b": "B again"}, (
        f"the same, in the other order: {out['afterBA']}"
    )
    assert len(_kept()) == 4, (
        f"four tags saving step 1 did not make four files: {_kept()}")


@pytest.mark.parametrize("save, read", [
    ('ws.persist(TAG, "only mine", 0)', "ws.readState(TAG, 0)"),
    ('ws.persistDraft(TAG, "only mine")', "ws.readDraft(TAG)"),
], ids=["a point", "a draft"])
def test_reading_one_tag_never_returns_anothers(app_at, save, read):
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
        await __quiet();
        console.log(JSON.stringify({{
            mine:     await {mine},
            stranger: await {stranger},
        }}));
        """, app_at=app_at)
    assert out["mine"] == "only mine"
    assert out["stranger"] is None, "a tag read back another tag's saved work"


def test_a_tags_draft_is_its_own_file_beside_its_points(app_at):
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
        await __quiet();
        await ws.pruneStatesAbove("modify", -1);
        console.log(JSON.stringify({
            draft: await ws.readDraft("modify"),
            point: await ws.readState("modify", 0),
        }));
        """, app_at=app_at)
    assert out["draft"] == "the draft now", "the draft is not one file rewritten"
    assert out["point"] is None, "dropping every point left one"
    assert len(_kept()) == 1, (
        f"the draft and the points were not two kinds of file: {_kept()}")


def test_a_dropped_tail_is_gone_before_the_next_point_is_written(app_at):
    """`workspace.md` § 9.2, *clear the tail before re-anchoring*: a save on
    top of a retracted point drops the points above it, and that delete
    finishes before the next point is written.  THE SILENT FAILURE (to
    2.69.25): the delete was sent beside the write queue, so a slow one landed
    after the next point and erased it — Retract then stepped back past work
    that had been saved.  The delete is held until nothing else is on the
    wire, as a slow disk holds it."""
    out = _run(
        f"""
        __hold({PRUNE});
        ws.persist("modify", "one", 1);
        ws.persist("modify", "two", 2);
        ws.pruneStatesAbove("modify", 1);    // retracted to 1, then a save
        ws.persist("modify", "two again", 2);
        await __quiet();
        __release();
        await __quiet();
        console.log(JSON.stringify({{ one: await readBack("modify", 1),
                                      two: await readBack("modify", 2) }}));
        """, app_at=app_at)
    assert out == {"one": "one", "two": "two again"}, (
        f"the tail delete landed after the next point and erased it: {out}")


def test_an_edits_draft_never_waits_behind_the_points_delete(app_at):
    """`workspace.md` § 9.2: the delete of abandoned points is ordered before
    the tag's next POINT -- and the draft, a file of its own, waits for
    nothing.  THE SILENT FAILURE (2.69.26-2.69.28): one queue for every write
    held an edit's draft behind its point and the delete, and a reload that
    came first found the draft from before the edit -- the Molbuilder tab
    showed the file, not the edit.  The delete is held while the draft is
    read back, as a reload reads it."""
    out = _run(f"""
        __hold({PRUNE});
        ws.persist("modify", "point 1", 1);
        ws.pruneStatesAbove("modify", 1);
        ws.persistDraft("modify", "the edit");
        await __quiet();
        const draft = await ws.readDraft("modify");
        __release();
        await __quiet();
        console.log(JSON.stringify({{ draft }}));
    """, app_at=app_at)
    assert out == {"draft": "the edit"}, (
        f"the edit's draft was still queued behind the delete: {out}")


@pytest.mark.parametrize("save, read", [
    ('ws.persist("modify", BYTES, 1)', 'ws.readState("modify", 1)'),
    ('ws.persistDraft("modify", BYTES)', 'ws.readDraft("modify")'),
], ids=["a point", "the draft"])
def test_two_saves_to_one_file_land_in_the_order_made(app_at, save, read):
    """`workspace.md` § 9.2, *same-step saves stay in order*, of one id's
    files -- a point's, and the draft's, an id of its own: two writes to one
    file, and the later is what stays.  THE SILENT FAILURE: the two race on
    the threaded server and the stale one lands last -- Retract then
    restores a state that was abandoned, or a reload adopts the draft from
    before its position was written.  The first write is held until nothing
    else is on the wire."""
    out = _run(f"""
        __hold((c) => c.url.endsWith("/api/workspace-storage/write")
                      && c.body.data === "the abandoned");
        {save.replace("BYTES", '"the abandoned"')};
        {save.replace("BYTES", '"the one kept"')};
        await __quiet();
        __release();
        await __quiet();
        console.log(JSON.stringify(await {read}));
    """, app_at=app_at)
    assert out == "the one kept", (
        f"the earlier write to the file landed last: {out!r}")


@pytest.mark.parametrize("one, other", [
    ("results:structure", "results-structure"),
    ("a:b", "a::b"),
    ("a_b", "a-b"),
    ("a_d", "a-"),
    ("a.b", "a:b"),
], ids=["a colon and a dash", "one colon and two",
        "an underscore and a dash", "an underscore spelled as a dash is",
        "a dot and a colon"])
def test_a_tags_file_is_stable_and_differs_from_the_next_tags(app_at, one,
                                                               other):
    """The tag names the file, so a tag saving again writes the same file —
    otherwise a page reopening looks for its work under a name it has never
    used and finds nothing. And two tags write two files, or two savers
    write into one (`workspace.md` § 4: two tags are never one id) — each a
    name the server takes, or nothing is saved.

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
        await __quiet();
        console.log(JSON.stringify({{
            one: await readBack({json.dumps(one)}, 1),
            other: await readBack({json.dumps(other)}, 1),
        }}));
        """, app_at=app_at)
    assert out == {"one": "one, again", "other": "the other"}, out
    assert len(_kept()) == 2, (
        f"a tag saving again made a second file, or two tags shared one: "
        f"{_kept()}")


@pytest.mark.parametrize("same_tab, finds, first_tab_then", [
    (True, "one's", "two's"), (False, None, "one's")],
    ids=["the same tab reloaded finds its work",
         "another tab finds nothing and leaves the first tab's alone"])
def test_each_browser_tab_has_its_own_saved_work(app_at, same_tab, finds,
                                                 first_tab_then):
    """`workspace.md` § 4 (the user, 2026-11-11: one set per tab): a tab
    keeps its own id, so a reload of it finds the work it saved, and another
    tab, opening on the same folder, finds none of it and changes none of
    it -- the first tab, reloaded after, reads its own work, or the second
    save where the second was that tab.  THE SILENT FAILURE (to 2.69.28):
    every tab read and wrote one file per page, so a second tab opened on
    the Molbuilder tab showed the first tab's unsaved molecule, and its saves
    overwrote the first tab's."""
    first = _run("""
        ws.persist("modify", "one's", 1);
        await __quiet();
        console.log(JSON.stringify({ tab: tabStorage() }));""", app_at=app_at)
    second = _run("""
        const found = await readBack("modify", 1);
        ws.persist("modify", "two's", 1);
        await __quiet();
        console.log(JSON.stringify({ found }));""", app_at=app_at,
        tab_storage=first["tab"] if same_tab else None)
    again = _run("""
        console.log(JSON.stringify(await readBack("modify", 1)));""",
        app_at=app_at, tab_storage=first["tab"])
    assert (second["found"], again) == (finds, first_tab_then), (
        second, again)


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

def _with_molview(app_at, snippet: str):
    """``snippet`` with the real MolView model on the page, built as a page
    builds it, ``m``, its molecule loaded by the real load route."""
    return run_node([], f"""
        const {{ createModel }} = await import({json.dumps(HOST.as_uri())});
        await import({json.dumps(KEPT.resolve().as_uri())});
        await import({json.dumps(WS.resolve().as_uri())});
        const ws = globalThis.molbuilder.workspace;
        const m = createModel({{ workspace: ws, owner: "modify" }});
        await m.installMolecule({{ text: "1\\n\\nH 0 0 0\\n",
                                   filename: "/p/mine.xyz" }});
        {snippet}
        """, globals_js=BROWSER + served(app_at))


def test_what_molview_saves_can_be_read_back(app_at):
    """The real MolView model against the real workspace: open a molecule, press
    Save state, then ask for it back the way a reopening page does.  THE SILENT
    FAILURE: a workspace that opens what it stores to judge it -- one that once
    asked "which file did it come from?" answered "nothing saved" for a
    molecule built from SMILES, and the tab wiped work that was there.
    """
    out = _with_molview(app_at, """
        await m.save(1);
        await __quiet();
        const saved = await ws.readState("modify", 1);
        console.log(JSON.stringify({
            gotSomethingBack: saved !== null,
            stamped:  saved && saved.v,
            atoms:    saved && saved.state && saved.state.structure
                      ? saved.state.structure.elements : null,
            // Nothing anywhere in what was saved may name the file it came from.
            everything: JSON.stringify(saved),
        }));
        """)
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


def test_an_edit_and_a_saved_point_go_to_different_files(app_at):
    """Press Save state, then edit again.

    THE INVARIANT: the edit must not land ON the point you saved, or Retract
    would take you back to something that had been quietly rewritten. Two file
    names, one directory: the edit is in the draft, which a reload reads.

    The edit also lays down a point of its own, at the next index (§ 11.2), so
    Retract steps back one edit rather than as far back as the last time
    somebody pressed the button.  The badge therefore goes back DOWN — the work
    is on the sequence.
    """
    out = _with_molview(app_at, """
        await m.save(1);
        await __quiet();
        const point = await ws.readState("modify", 1);

        m.selection.writeLabel("frozen_atoms", "add", [0]);   // an edit, no save
        await __quiet();

        const draft = await ws.readDraft("modify");
        // THE EDIT'S OWN POINT, one index on from the saved one.
        const edited = await ws.readState("modify", 2);
        const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
        console.log(JSON.stringify({
            pointUnchanged: same(await ws.readState("modify", 1), point),
            newPoint: edited !== null && !same(edited.state, point.state),
            draftIsTheEdit: draft !== null && edited !== null
                && same(draft.state, edited.state),
            badge: m.uncommitted,
        }));
        """)
    assert out["pointUnchanged"] is True, (
        "an unsaved edit overwrote the point you saved — Retract would take you "
        "back to something you never chose to keep"
    )
    assert out["newPoint"] is True, (
        "the edit laid down no point of its own holding it, so Retract "
        "cannot step back one edit"
    )
    assert out["draftIsTheEdit"] is True, (
        "the draft does not hold the edit, so a reload loses it")
    assert out["badge"] is False, (
        "the badge still says there is work off the sequence, after the edit "
        "put itself on it (§ 11.2)"
    )

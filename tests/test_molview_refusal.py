"""When a change does not happen — every test derived from ``docs/web/molview.md``
§ 6.9, never from the source it checks (§ 13.1).

The rows of § 13.3 guarded here:

    § 6.9  a refused change throws, carrying the server's own sentence
    § 6.9  `null` and a refusal are different answers
    § 6.9  a failed change leaves the notices alone

§ 6.9 gives a door that changes the structure three outcomes and no fourth:
it worked (you get the thing), it was refused (**it throws**, carrying the
reason), or there was nothing to do (``null``, silently). The bug this replaces
is the one where the middle case answered ``null`` like the last one, so a
refused edit and an empty viewer were indistinguishable and the Update button
went dead with nothing on screen.

THE SERVER IS THE APP'S OWN (`_node_esm.served`, `molview.md` § 13.1): the
structure is a pair on disk, written as the road writes one, opened by
`/api/build/load`, edited by `/api/modify/*` and its cell changed by
`/api/structure/periodicity`; each refusal is one the route
really makes, and the sentence a caller is handed is checked against the one
the route answered (``__answered``) -- never a sentence typed here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from support import molview_page
from support.road import saved_by_name


@pytest.fixture(scope="module")
def pairs(isolated_projects_root_module):
    """Carbon monoxide in a typed 10 Å box, saved as the road saves a
    structure (`support.road.saved_by_name`) -- every test below starts from
    it loaded and editable, since a viewer with nothing in it answers `null`
    for a different reason (§ 6.9)."""
    return saved_by_name(isolated_projects_root_module, {"CO": {
        "elements": ["C", "O"], "positions": [[5.0, 5.0, 5.0],
                                              [5.0, 5.0, 6.13]],
        "cell": [10.0, 10.0, 10.0]}})


#: This file's own page script, after MolView's page (`support.
#: molview_page`).
OUTCOMES = """
/* Run a call and report which of § 6.9's three outcomes happened, so a test
 * asserts the OUTCOME rather than the shape of a try/catch -- and, for a
 * refusal, the sentence the route answered it with. */
async function outcome(fn) {
    globalThis.__answered.length = 0;
    try {
        const value = await fn();
        return { threw: false, isNull: value === null };
    } catch (err) {
        const said = globalThis.__answered.at(-1);
        return { threw: true, message: String(err && err.message),
                 answered: (said && said.error) || null };
    }
}
"""


def _run(app_at, pairs, snippet: str, **given):
    """The test's script on MolView's page (`support.molview_page`);
    ``given`` -- values worked out here, as constants of the same names."""
    return molview_page.run(app_at, pairs, OUTCOMES + snippet, given=given)


# ---------------------------------------------------------------------------
# § 6.9 — a refused change throws, carrying the server's own sentence
# ---------------------------------------------------------------------------

#: A REFUSAL AT EACH OF THE THREE DOORS THAT CHANGE THE STRUCTURE, each one a
#: person can reach and the route really makes: a file gone from disk; an
#: element the Add box was typed (`modify.html`'s free-text symbol) that is no
#: element; and the Cell page's Apply -- its one ``block`` -- with a cell typed
#: mirrored (`modify/periodicity.js`).
REFUSED = {
    "install": "m.installMolecule({ path: CO_GONE })",
    "op":      'm.applyOp("add_atom", { element: "Xx", offset: [0, 0, 1.5] })',
    "cell":    ('m.commitPeriodicityOp("block", { axis_kind: '
                '["periodic", "periodic", "periodic"], vacuum: null, '
                'cell: [[-4,0,0],[0,4,0],[0,0,4]], box_corner: null })'),
}


@pytest.mark.parametrize("door", list(REFUSED))
def test_a_refused_change_throws_the_servers_own_sentence(app_at, pairs, door):
    """§ 6.9: "it throws, carrying the reason", and "the words are the server's,
    unchanged. It is the side that decided, and its sentence carries the numbers
    — the clearance, the axis, what to do instead." One rule over every door
    that changes the structure, not a habit three of them happen to share.

    Verified against the old behaviour: a refused cell edit returned ``null``
    and the sentence never existed in the browser -- the response body was
    never read, and the status code that replaced it ("…: 400", nothing a
    reader can act on) was then swallowed by a bare ``catch``.
    ``installMolecule`` is a row deliberately: it was already the one door
    that let a throw through, so the rule could hold there for another reason.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO");
        console.log(JSON.stringify(await outcome(() => %s)));
        """ % REFUSED[door],
        CO_GONE=str(Path(pairs["CO"]).parent / "gone.xyz"))
    assert out["threw"] is True, f"{door} swallowed a refusal: {out}"
    assert out["answered"], f"{door}'s route did not refuse"
    assert out["message"] == out["answered"], (
        f"{door} did not pass on the server's sentence, whole: {out}")


# ---------------------------------------------------------------------------
# § 6.9 — `null` and a refusal are different answers
# ---------------------------------------------------------------------------

def test_nothing_to_do_answers_null_and_says_nothing(app_at, pairs):
    """§ 6.9: "the viewer is read-only · an op was asked for that the controls
    already rule out · … · the cell door with nothing loaded" → ``null``, and
    nothing else — "none of these needs explaining"; the interface has
    already said so.

    This is the half that makes the other half readable: if these threw too, a
    caller could not tell a refusal from an ordinary no-op and would show a
    message for both.
    """
    out = _run(app_at, pairs, """
        const empty  = createModel({});
        const frozen = await loaded("CO", { mode: "readonly" });
        const m      = await loaded("CO");

        console.log(JSON.stringify({
            // Nothing loaded, at the CELL door: a cell is a property OF a
            // structure, so with none there is no box to change (§ 6.9).
            // `applyOp` is deliberately NOT here -- an empty structure is a
            // structure, and an edit on one sends the request like any other,
            // which is what lets `slab` build the first thing on a blank
            // canvas (§ 6.9, corrected 2026-08-31).
            emptyCell: await outcome(() => empty.commitPeriodicityOp(
                "vacuum", [1,1,1])),
            // a viewer that does not edit (§ 9.4)
            frozenCell: await outcome(() => frozen.commitPeriodicityOp(
                "vacuum", [1,1,1])),
            // ops the controls rule out. NOT rotate or translate: § 11.1 gives
            // those `emptySelection: "all"`, so no selection means the whole
            // structure and they rightly go ahead. The two that refuse are the
            // one that needs a selection to have anything to act on, and the
            // one that needs an exact count.
            noSelection: await outcome(() => m.applyOp("delete", {})),
            // WRONG COUNT IS TWO, not zero.  `add_atom` takes one atom, and
            // with NONE picked it measures from the world origin instead
            // (§ 11.1, 2026-09-07) -- so zero is a legal request that goes to
            // the server, and the only count with no answer is "which of
            // these two would the offset be measured from?".
            wrongCount:  await outcome(() => {
                m.selection.add([0, 1]);
                return m.applyOp("add_atom", { element: "H" });
            }),
        }));
        """)
    for case, got in out.items():
        assert got["threw"] is False, f"{case} threw where § 6.9 says null"
        assert got["isNull"] is True, f"{case} did not answer null"


def test_nothing_to_do_never_reaches_the_server(app_at, pairs):
    """The same rule seen from the wire. A no-op that still sent a request could
    be refused, and would then produce a message for something the interface had
    already ruled out.
    """
    out = _run(app_at, pairs, """
        const frozen = await loaded("CO", { mode: "readonly" });
        await frozen.commitPeriodicityOp("vacuum", [1, 1, 1]);
        await frozen.applyOp("translate", { dx: 1 });
        console.log(JSON.stringify({ urls: __calls.map((c) => c.url) }));
        """)
    assert out["urls"] == [], "a read-only viewer sent an edit to the server"


def test_an_anchorless_add_reaches_the_server_with_no_anchor_key(app_at, pairs):
    """The other side of the same rule (§ 11.1): with nothing selected
    `add_atom` is NOT a no-op -- it is a request whose anchor is the world
    origin, and the way it says so on the wire is by OMITTING the key.

    Sending `anchor_index: null` would be a different claim (a value that is
    not an index) and sending `0` would silently mean atom 0, so what is
    checked here is the absence itself.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO");
        const got = await outcome(() => m.applyOp(
            "add_atom", { element: "H", offset: [0, 0, 1.5] }));
        const add = __calls.filter((c) => c.url.endsWith("/api/modify/add_atom"));
        console.log(JSON.stringify({
            got,
            sent: add.length,
            keys: add.map((c) => Object.keys(c.body).sort()),
        }));
        """)
    assert out["sent"] == 1, "an anchorless add never left the browser"
    assert out["got"]["threw"] is False, (
        f"the route refused an anchorless add: {out['got']}")
    assert "anchor_index" not in out["keys"][0], (
        f"the anchor key was sent anyway: {out['keys'][0]}")


# ---------------------------------------------------------------------------
# § 6.9 — a failed change leaves the notices alone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("door", ["op", "cell"])
def test_a_refused_change_leaves_the_earlier_notices_standing(app_at, pairs,
                                                              door):
    """§ 6.9: "A failed change clears nothing: nothing moved, so every condition
    that held a moment before still holds." -- an edit refused and a cell
    refused alike.

    This is why the reason is thrown rather than kept beside the notices. Stored
    there, a refusal would replace the set — wiping notices that are still true,
    on the strength of an edit that never took place.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO");
        const before = m.getNotices();
        const got = await outcome(() => %s);
        console.log(JSON.stringify({ before, got, after: m.getNotices() }));
        """ % REFUSED[door])
    assert out["before"], "the load answered no notice to keep"
    assert out["got"]["threw"] is True, out["got"]
    assert out["after"] == out["before"], (
        "a refused change altered the notices, which describe a structure "
        "that did not move")


def test_a_refused_edit_changes_no_data(app_at, pairs):
    """§ 11.1's "a failed edit changes nothing", for a failure that leaves by
    throwing. Throwing must not skip the part where nothing
    was written — and must not leave the door jammed against the next attempt,
    which is what an `running` flag cleared only on the success path would do.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO");
        const before = JSON.stringify(m.getStructure());
        const refused = await outcome(() => %s);
        const unchanged = JSON.stringify(m.getStructure()) === before;
        // The door must be usable again straight away.
        const second = await outcome(() => m.applyOp("translate", { dx: 5 }));
        console.log(JSON.stringify({ refused, unchanged, secondAttempt: second }));
        """ % REFUSED["op"])
    assert out["refused"]["threw"] is True, out["refused"]
    assert out["unchanged"] is True, "a refused edit changed the master copy"
    assert out["secondAttempt"]["threw"] is False, (
        "the door stayed jammed after a refusal")
    assert out["secondAttempt"]["isNull"] is False, (
        "the retry after a refusal did not go through")

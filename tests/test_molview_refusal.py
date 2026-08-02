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

The stand-in server obeys the document's account of the wire (§ 13.1): it
answers ``{ok: false, error}`` with a non-200 status when it refuses, which is
what ``web-api.md`` says every door sends.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
MODEL = MODULE_DIR / "model.js"

#: The gate's own words for a cell that cannot exist. The point of quoting a
#: REAL sentence rather than "nope" is that this one tells the reader what to do,
#: and that is the whole of what § 6.9 is protecting.
GATE_SENTENCE = (
    "A left-handed cell (determinant -64.00) cannot be used. "
    "Swap two lattice vectors or negate one."
)

SERVER = """
globalThis.__requests   = [];
globalThis.__refuse     = null;   // {status, body} -> the server says no
globalThis.__unreachable = false; // fetch itself rejects
globalThis.__nextPayload = null;

function atomRow(i, element, x) {
    return { index: i, element, x, y: 0, z: 0, regions: [] };
}

globalThis.fetch = async function (route, init) {
    globalThis.__requests.push({ route, body: JSON.parse(init.body) });
    if (globalThis.__unreachable) {
        throw new TypeError("Failed to fetch");
    }
    if (globalThis.__refuse) {
        const r = globalThis.__refuse;
        return {
            ok: false,
            status: r.status,
            json: async () => {
                if (r.notJson) throw new SyntaxError("Unexpected token <");
                return r.body;
            },
        };
    }
    const payload = globalThis.__nextPayload || {
        atoms: [atomRow(0, "C", 0), atomRow(1, "O", 1)],
    };
    return { ok: true, status: 200, json: async () => payload };
};
"""

PRELUDE = f"""
const {{ createModel }} = await import({json.dumps(MODEL.resolve().as_uri())});

/* A loaded, editable viewer — the state every test below starts from, since a
 * viewer with nothing in it answers `null` for a different reason (§ 6.9). */
async function loaded(opts) {{
    globalThis.__requests    = [];
    globalThis.__refuse      = null;
    globalThis.__unreachable = false;
    globalThis.__nextPayload = null;
    const m = createModel(opts || {{}});
    await m.installMolecule({{ text: "2\\n\\nC 0 0 0\\nO 1 0 0\\n",
                              filename: "x.xyz" }});
    return m;
}}

/* Run a call and report which of § 6.9's three outcomes happened, so a test
 * asserts the OUTCOME rather than the shape of a try/catch. */
async function outcome(fn) {{
    try {{
        const value = await fn();
        return {{ threw: false, isNull: value === null }};
    }} catch (err) {{
        return {{ threw: true, message: String(err && err.message) }};
    }}
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=SERVER)


# ---------------------------------------------------------------------------
# § 6.9 — a refused change throws, carrying the server's own sentence
# ---------------------------------------------------------------------------

def test_a_refused_cell_edit_throws_the_servers_own_sentence():
    """§ 6.9: "it throws, carrying the reason", and "the words are the server's,
    unchanged. It is the side that decided, and its sentence carries the numbers
    — the clearance, the axis, what to do instead."

    Verified against the old behaviour: this returned ``null`` and the sentence
    never existed in the browser, because the reason was thrown away twice — the
    response body was never read, and the status code that replaced it was then
    swallowed by a bare ``catch``.
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__refuse = { status: 400,
            body: { ok: false, error: %s } };
        const got = await outcome(() => m.commitPeriodicityOp(
            "cell", [[-4,0,0],[0,4,0],[0,0,4]]));
        console.log(JSON.stringify(got));
        """ % json.dumps(GATE_SENTENCE)
    )
    assert out["threw"] is True, "a refused cell edit did not throw"
    assert out["message"] == GATE_SENTENCE, (
        "the reason reaching the caller is not the sentence the server sent")


def test_the_status_code_is_not_the_message():
    """The specific regression. The message a caller receives has to be
    something a reader can act on, and "…: 400" is not — it is what the code
    produced when it built its error from the status line and never read the
    body the reason was in.
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__refuse = { status: 400,
            body: { ok: false, error: %s } };
        const got = await outcome(() => m.commitPeriodicityOp("vacuum", [1,1,1]));
        console.log(JSON.stringify(got));
        """ % json.dumps(GATE_SENTENCE)
    )
    assert "400" not in out["message"], (
        "the status code reached the caller instead of the reason")
    assert "swap two lattice vectors" in out["message"].lower()


def test_every_changing_door_reports_a_refusal_the_same_way():
    """§ 6.9 is one rule over the doors that change the structure, not a habit
    three of them happen to share. Asked of each: same refusal, same sentence.

    ``installMolecule`` is included deliberately — it was already the one door
    that let a throw through, so a test that skipped it would pass while the
    rule held for a different reason on that door than on the others.
    """
    out = _run(
        """
        const m = await loaded();
        const refuse = { status: 400, body: { ok: false, error: %s } };

        globalThis.__refuse = refuse;
        const install = await outcome(() => m.installMolecule(
            { text: "1\\n\\nH 0 0 0\\n", filename: "y.xyz" }));
        const op      = await outcome(() => m.applyOp("translate", { dx: 1 }));
        const cell    = await outcome(() => m.commitPeriodicityOp(
            "cell_origin", [1, 2, 3]));
        console.log(JSON.stringify({ install, op, cell }));
        """ % json.dumps(GATE_SENTENCE)
    )
    for door, got in out.items():
        assert got["threw"] is True, f"{door} swallowed a refusal"
        assert got["message"] == GATE_SENTENCE, (
            f"{door} did not pass on the server's sentence")


def test_a_refusal_with_no_reason_still_says_something_a_reader_can_use():
    """§ 6.9: "MolView writes a message itself only when there is none to quote."

    A crash page or a proxy answers with a body that is not the envelope, so
    there is no sentence to pass on. What must NOT happen is the failure path
    raising a second failure of its own — a parse error replacing the reason.
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__refuse = { status: 502, notJson: true };
        const got = await outcome(() => m.commitPeriodicityOp("vacuum", [1,1,1]));
        console.log(JSON.stringify(got));
        """
    )
    assert out["threw"] is True
    assert "Unexpected token" not in out["message"], (
        "the body parse error replaced the reason")
    assert out["message"].strip(), "a refusal arrived with an empty message"


def test_an_unreadable_success_does_not_leak_the_parser_at_the_reader():
    """The same rule on the other side of the status line. A 200 whose body is
    not JSON — a proxy or a cache answering in the server's stead — must not put
    the parser's own words in front of somebody: "Unexpected token <" is the kind
    of sentence § 6.9 exists to keep off the screen.

    Found by reading the change rather than by a failing test: the refusal path
    guarded its parse and the success path did not.
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.fetch = async () => ({
            ok: true, status: 200,
            json: async () => { throw new SyntaxError("Unexpected token <"); },
        });
        const got = await outcome(() => m.commitPeriodicityOp("vacuum", [1,1,1]));
        console.log(JSON.stringify(got));
        """
    )
    assert out["threw"] is True
    assert "Unexpected token" not in out["message"]
    assert "server" in out["message"].lower()


def test_a_request_that_never_arrived_says_so_in_plain_language():
    """§ 6.9's other written-by-MolView case. The browser's own wording is
    "Failed to fetch", which reads as a bug in the page rather than as something
    the reader can do anything about.
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__unreachable = true;
        const got = await outcome(() => m.commitPeriodicityOp("vacuum", [1,1,1]));
        console.log(JSON.stringify(got));
        """
    )
    assert out["threw"] is True
    assert "Failed to fetch" not in out["message"]
    assert "server" in out["message"].lower()


# ---------------------------------------------------------------------------
# § 6.9 — `null` and a refusal are different answers
# ---------------------------------------------------------------------------

def test_nothing_to_do_answers_null_and_says_nothing():
    """§ 6.9: "nothing is loaded · the viewer is read-only · an op was asked for
    that the controls already rule out" → ``null``, and nothing else. "None of
    these needs explaining" — the interface has already said so.

    This is the half that makes the other half readable: if these threw too, a
    caller could not tell a refusal from an ordinary no-op and would show a
    message for both.
    """
    out = _run(
        """
        const empty  = createModel({});
        const frozen = await loaded({ mode: "readonly" });
        const m      = await loaded();

        console.log(JSON.stringify({
            // nothing loaded
            emptyEdit: await outcome(() => empty.applyOp("translate", { dx: 1 })),
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
            wrongCount:  await outcome(() => m.applyOp("add_atom",
                                                       { element: "H" })),
        }));
        """
    )
    for case, got in out.items():
        assert got["threw"] is False, f"{case} threw where § 6.9 says null"
        assert got["isNull"] is True, f"{case} did not answer null"


def test_nothing_to_do_never_reaches_the_server():
    """The same rule seen from the wire. A no-op that still sent a request could
    be refused, and would then produce a message for something the interface had
    already ruled out.
    """
    out = _run(
        """
        const frozen = await loaded({ mode: "readonly" });
        globalThis.__requests = [];
        await frozen.commitPeriodicityOp("vacuum", [1, 1, 1]);
        await frozen.applyOp("translate", { dx: 1 });
        console.log(JSON.stringify({ routes: __requests.map((r) => r.route) }));
        """
    )
    assert out["routes"] == [], "a read-only viewer sent an edit to the server"


# ---------------------------------------------------------------------------
# § 6.9 — a failed change leaves the notices alone
# ---------------------------------------------------------------------------

def test_a_refused_edit_leaves_the_earlier_warnings_standing():
    """§ 6.9: "A failed change clears nothing: nothing moved, so every condition
    that held a moment before still holds."

    This is why the reason is thrown rather than kept beside the notices. Stored
    there, a refusal would replace the set — wiping warnings that are still true,
    on the strength of an edit that never took place.
    """
    out = _run(
        """
        const m = createModel({});
        globalThis.__nextPayload = {
            atoms: [{ index: 0, element: "C", x: 0, y: 0, z: 0, regions: [] }],
            notices: [{ level: "warn",
                        message: "the box does NOT contain the structure" }],
        };
        await m.installMolecule({ text: "1\\n\\nC 0 0 0\\n", filename: "x.xyz" });
        const before = m.getNotices();

        globalThis.__refuse = { status: 400,
            body: { ok: false, error: "refused" } };
        await outcome(() => m.commitPeriodicityOp("vacuum", [1, 1, 1]));

        console.log(JSON.stringify({ before, after: m.getNotices() }));
        """
    )
    assert out["before"]["list"], "the fixture never produced a notice to keep"
    assert out["after"] == out["before"], (
        "a refused edit changed the notices, which describe a structure that "
        "did not move")


def test_a_refused_edit_changes_no_data():
    """§ 11.1's "a failed edit changes nothing", re-asked now that a failure
    leaves by a different route. Throwing must not skip the part where nothing
    was written — and must not leave the door jammed against the next attempt,
    which is what an `running` flag cleared only on the success path would do.
    """
    out = _run(
        """
        const m = await loaded();
        const before = JSON.stringify(m.getStructure());

        globalThis.__refuse = { status: 400,
            body: { ok: false, error: "refused" } };
        await outcome(() => m.applyOp("translate", { dx: 5 }));

        // The door must be usable again straight away.
        globalThis.__refuse = null;
        const second = await outcome(() => m.applyOp("translate", { dx: 5 }));

        console.log(JSON.stringify({
            unchanged: JSON.stringify(m.getStructure()) === before,
            secondAttempt: second,
        }));
        """
    )
    assert out["unchanged"] is True, "a refused edit changed the master copy"
    assert out["secondAttempt"]["threw"] is False, (
        "the door stayed jammed after a refusal")
    assert out["secondAttempt"]["isNull"] is False, (
        "the retry after a refusal did not go through")

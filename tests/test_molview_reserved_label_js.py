"""MolView's end of a reserved label — one mechanism, one designated read.

Derived from `docs/web/molview.md` § 6.6 and § 9.3, never from the source (§ 13).

> A reserved meaning costs a **name** and **one accessor** — nothing else.

`frozen_atoms` is the first reserved label. Something downstream acts on it, so
MolView offers `getFrozen()` — the one place that answers "which atoms carry this
name". Everything else about it is an ordinary label: it arrives in the same
list, groups through the same walk, filters through the same rule, and leaves in
the same field.

WHAT THESE TESTS EXIST TO PREVENT. This module used to carry two translators,
one at each boundary, because the server kept the fact in a field of its own:

  * inbound  — an `is_frozen` flag on the atom was turned into a label, so that
               downstream code could have the one mechanism § 6.6 promises;
  * outbound — the label was pulled back out into a `frozen_atoms` field, so the
               server would recognise it.

Both are gone with the server's second store (2026-07-31). A translator that
comes back is a second storage that came back, so the tests below assert the
absence directly: the label crosses BOTH boundaries as a label, untouched.

THE STAND-IN (§ 13.1) speaks the server's names. Its atom rows carry `regions`
and nothing else — no `is_frozen` — because that is what `/api/selection/atoms`
and `/api/build/load` now send. A stand-in that still offered the flag would let
an inbound translator pass its own test forever.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
MODEL = MODULE_DIR / "model.js"
JOBS = MODULE_DIR / "model-jobs.js"

FROZEN = "frozen_atoms"          # the label's one name, on both sides

SERVER = """
globalThis.__requests = [];
globalThis.__nextPayload = null;

/* An atom row as the server sends it: the labels it carries, and nothing
 * beside them.  A reserved label is IN `regions`. */
globalThis.__atomRow = function (i, element, x, opts) {
    return Object.assign({ index: i, element, x, y: 0, z: 0, regions: [] },
                         opts || {});
};

globalThis.fetch = async function (route, init) {
    const body = JSON.parse(init.body);
    globalThis.__requests.push({ route, body });
    const payload = globalThis.__nextPayload || globalThis.__payload([
        globalThis.__atomRow(0, "C", 0), globalThis.__atomRow(1, "O", 1),
    ]);
    return { ok: true, status: 200, json: async () => payload };
};

globalThis.__payload = function (atoms, extra) {
    return Object.assign({ atoms }, extra || {});
};
"""

PRELUDE = f"""
const {{ createModel }} = await import({json.dumps(MODEL.resolve().as_uri())});

/* Two atoms: one ordinary label, one reserved. */
async function labelled() {{
    globalThis.__requests = [];
    globalThis.__nextPayload = globalThis.__payload([
        globalThis.__atomRow(0, "C", 0, {{ regions: ["L-electrode"] }}),
        globalThis.__atomRow(1, "O", 1, {{ regions: ["{FROZEN}"] }}),
    ]);
    const m = createModel({{}});
    await m.installMolecule({{ text: "x", filename: "x.xyz" }});
    globalThis.__requests = [];
    return m;
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=SERVER)


# ---------------------------------------------------------------------------
# It arrives as a label, and is one all the way down
# ---------------------------------------------------------------------------

def test_the_reserved_label_arrives_as_an_ordinary_label():
    """§ 6.2: an atom's facts are its labels and its residue — there is no
    frozen field. The reserved name sits in the same list as any other, so
    nothing downstream needs to know it is special."""
    out = _run(
        """
        const m = await labelled();
        console.log(JSON.stringify({
            labels:  m.getAtoms().map(a => a.labels),
            regions: m.getRegions(),
            keys:    Object.keys(m.getAtoms()[1]).sort(),
        }));
        """
    )
    assert out["labels"] == [["L-electrode"], [FROZEN]]
    assert out["regions"] == {"L-electrode": [0], FROZEN: [1]}
    assert not any("frozen" in k.lower() for k in out["keys"] if k != "labels"), (
        f"the atom grew a field of its own for the reserved label: {out['keys']}"
    )


def test_the_designated_read_is_a_cut_of_the_same_list():
    """§ 9.3: "exactly one is the main one and the rest are narrower cuts of it;
    a cut may disappear, but it must never grow into a rival."

    `getFrozen` is a cut of the labels — so it cannot disagree with `getRegions`,
    because there is nothing for it to disagree with.
    """
    out = _run(
        """
        const m = await labelled();
        console.log(JSON.stringify({
            frozen:      m.getFrozen(),
            fromRegions: m.getRegions()["%s"],
            fromLabels:  m.getAtoms()
                          .map((a, i) => a.labels.indexOf("%s") >= 0 ? i : -1)
                          .filter(i => i >= 0),
        }));
        """ % (FROZEN, FROZEN)
    )
    assert out["frozen"] == out["fromRegions"] == out["fromLabels"] == [1], (
        f"three ways of asking, three answers: {out}"
    )


def test_the_designated_read_cannot_be_used_to_write():
    """§ 9.4 / § 9.3: every read returns a copy, so changing what you were given
    can never change the viewer. That holds for the reserved label's read like
    any other — otherwise the one door that owns the name is also a way around
    it."""
    out = _run(
        """
        const m = await labelled();
        m.getFrozen().push(0);
        m.getRegions()["%s"].push(0);
        console.log(JSON.stringify({ frozen: m.getFrozen() }));
        """ % FROZEN
    )
    assert out["frozen"] == [1], "a read wrote through to the master copy"


def test_no_atom_carries_the_fact_twice():
    """The defect the server's second store caused, at this end: an atom with
    both a label and a flag renders its frozen state twice in the panel. One
    representation means the panel needs no rule about which to believe."""
    out = _run(
        """
        const m = await labelled();
        const a = m.getAtoms()[1];
        console.log(JSON.stringify({
            labels: a.labels,
            occurrences: a.labels.filter(l => l === "%s").length,
        }));
        """ % FROZEN
    )
    assert out["occurrences"] == 1, f"the fact appears twice on the atom: {out['labels']}"


# ---------------------------------------------------------------------------
# It leaves as a label too
# ---------------------------------------------------------------------------

def test_what_leaves_carries_the_reserved_label_in_the_label_field():
    """The outbound half. There is no split at this boundary any more: the
    server's shape IS the label store, so what leaves is what was held."""
    out = _run(
        """
        const m = await labelled();
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        const sent = globalThis.__requests[globalThis.__requests.length - 1].body;
        console.log(JSON.stringify({
            keys:    Object.keys(sent.structure).sort(),
            regions: sent.structure.regions,
        }));
        """
    )
    assert out["regions"] == {"L-electrode": [0], FROZEN: [1]}, (
        f"the reserved label did not leave in the label field: {out['regions']}"
    )
    assert FROZEN not in out["keys"], (
        f"a field of its own reappeared on the wire: {out['keys']}"
    )


def test_a_structure_that_goes_out_and_comes_back_is_unchanged():
    """The round trip is the real test of "no translation": if either boundary
    still renamed or moved the fact, one of the two directions would show it."""
    out = _run(
        """
        const m = await labelled();
        const before = m.getRegions();

        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 0, { regions: ["L-electrode"] }),
            globalThis.__atomRow(1, "O", 1, { regions: ["%s"] }),
        ]);
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });

        console.log(JSON.stringify({ before, after: m.getRegions(),
                                     frozen: m.getFrozen() }));
        """ % FROZEN
    )
    assert out["before"] == out["after"], f"the round trip changed the labels: {out}"
    assert out["frozen"] == [1]


# ---------------------------------------------------------------------------
# One name
# ---------------------------------------------------------------------------

def test_the_module_knows_the_reserved_name_in_exactly_one_place():
    """"A reserved meaning costs a NAME" — one. The module holds it as a single
    constant, so the name the panel offers as a filter row is by construction the
    name the server matches. Offering a name the server cannot match would give a
    row that always answers nothing."""
    sources = [p.read_text(encoding="utf-8") for p in MODULE_DIR.glob("*.js")]
    literal = sum(
        # the declaration itself excepted -- that IS the one place
        text.count('"%s"' % FROZEN) + text.count("'%s'" % FROZEN)
        for text in sources
    )
    assert literal <= 1, (
        f"the reserved name is written as a literal {literal} times across the "
        f"module; it must come from the one constant"
    )

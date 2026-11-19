"""MolView's end of a reserved label — one mechanism, one designated read.

Derived from `docs/web/molview.md` § 6.6 and § 9.3, never from the source (§ 13).

> A reserved meaning costs a **name** and **one accessor** — nothing else.

`frozen_atoms` is the first reserved label. Something downstream acts on it, so
MolView offers `getFrozen()` — the one place that answers "which atoms carry this
name". Everything else about it is an ordinary label: it arrives in the same
list, groups through the same walk, filters through the same rule, and leaves in
the same field.

WHAT THESE TESTS EXIST TO PREVENT: a translator at either boundary (an
`is_frozen` flag turned into a label inbound, the label pulled out into a
`frozen_atoms` field outbound) is a second storage, so the tests below assert
the absence directly: the label crosses BOTH boundaries as a label, untouched.

THE SERVER IS THE APP'S OWN (`_node_esm.served`): the structure is a pair on
disk, saved as the road saves one (`support.road.structure_saved`) with an ordinary label
on one atom and the reserved one on the other, opened by `/api/build/load` and
edited by `/api/modify/*`.  The structure they answer keeps every label in its
one label store (`metadata.regions`), the reserved one included, with no
`is_frozen` beside it — because that is what the routes send; asked of them,
not of a stand-in that could still offer the flag.
"""
from __future__ import annotations

import pytest

from support import molview_page
from support.road import saved_by_name

FROZEN = "frozen_atoms"          # the label's one name, on both sides


@pytest.fixture
def pairs(isolated_projects_root):
    """Two atoms saved as the road saves a structure (`support.road.
    saved_by_name`) -- one with an ordinary label, one with the reserved one -- in
    this test's own projects tree."""
    return saved_by_name(isolated_projects_root, {"CO labelled": {
        "elements": ["C", "O"],
        "positions": [[5.0, 5.0, 5.0], [5.0, 5.0, 6.13]],
        "regions": {"L-electrode": [0], FROZEN: [1]},
        "cell": [10.0, 10.0, 10.0]}})


def _run(app_at, pairs, snippet: str):
    """The test's script on MolView's page (`support.molview_page`), the
    reserved label's name given to it as ``FROZEN``."""
    return molview_page.run(app_at, pairs, snippet, given={"FROZEN": FROZEN})


# ---------------------------------------------------------------------------
# It arrives as a label, and is one all the way down
# ---------------------------------------------------------------------------

def test_the_reserved_label_arrives_as_an_ordinary_label(app_at, pairs):
    """§ 6.2: an atom's facts are its labels and its residue — there is no
    frozen field. The reserved name sits in the same list as any other, once,
    so nothing downstream needs to know it is special, and the panel has no
    rule about which of two representations to believe."""
    out = _run(app_at, pairs,
        """
        const m = await loaded("CO labelled");
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


def test_the_designated_read_is_a_cut_of_the_same_list(app_at, pairs):
    """§ 9.3: "exactly one is the main one and the rest are narrower cuts of it;
    a cut may disappear, but it must never grow into a rival."

    `getFrozen` is a cut of the labels — so it cannot disagree with `getRegions`,
    because there is nothing for it to disagree with.
    """
    out = _run(app_at, pairs,
        """
        const m = await loaded("CO labelled");
        console.log(JSON.stringify({
            frozen:      m.getFrozen(),
            fromRegions: m.getRegions()[FROZEN],
            fromLabels:  m.getAtoms()
                          .map((a, i) => a.labels.indexOf(FROZEN) >= 0 ? i : -1)
                          .filter(i => i >= 0),
        }));
        """
    )
    assert out["frozen"] == out["fromRegions"] == out["fromLabels"] == [1], (
        f"three ways of asking, three answers: {out}"
    )


def test_the_designated_read_cannot_be_used_to_write(app_at, pairs):
    """§ 9.4 / § 9.3: every read returns a copy, so changing what you were given
    can never change the viewer. That holds for the reserved label's read like
    any other — otherwise the one door that owns the name is also a way around
    it."""
    out = _run(app_at, pairs,
        """
        const m = await loaded("CO labelled");
        m.getFrozen().push(0);
        m.getRegions()[FROZEN].push(0);
        console.log(JSON.stringify({ frozen: m.getFrozen() }));
        """
    )
    assert out["frozen"] == [1], "a read wrote through to the master copy"


# ---------------------------------------------------------------------------
# It leaves as a label too
# ---------------------------------------------------------------------------

def test_what_leaves_carries_the_reserved_label_in_the_label_field(app_at, pairs):
    """The outbound half. There is no split at this boundary: the
    server's shape IS the label store, so what leaves is what was held -- in
    `metadata`, where the envelope keeps it (web-api.md § 1)."""
    out = _run(app_at, pairs,
        """
        const m = await loaded("CO labelled");
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        const sent = globalThis.__calls[globalThis.__calls.length - 1].body;
        console.log(JSON.stringify({
            keys:    Object.keys(sent.structure).sort(),
            regions: sent.structure.metadata.regions,
        }));
        """
    )
    assert out["regions"] == {"L-electrode": [0], FROZEN: [1]}, (
        f"the reserved label did not leave in the label field: {out['regions']}"
    )
    assert FROZEN not in out["keys"], (
        f"a field of its own reappeared on the wire: {out['keys']}"
    )


def test_a_structure_that_goes_out_and_comes_back_is_unchanged(app_at, pairs):
    """The round trip is the real test of "no translation": if either boundary
    still renamed or moved the fact, one of the two directions would show it."""
    out = _run(app_at, pairs,
        """
        const m = await loaded("CO labelled");
        const before = m.getRegions();
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });

        console.log(JSON.stringify({ before, after: m.getRegions(),
                                     frozen: m.getFrozen(),
                                     moved: globalThis.__calls.length }));
        """
    )
    assert out["moved"] == 1, out
    assert out["before"] == out["after"], f"the round trip changed the labels: {out}"
    assert out["frozen"] == [1]


# ---------------------------------------------------------------------------
# The edit actually reaches the route (§ 11.1)
# ---------------------------------------------------------------------------

def test_an_op_sends_its_arguments_where_the_route_reads_them(app_at, pairs):
    """§ 11.1: the table drives one generic piece of code. The route reads its
    own arguments off the BODY ROOT — `dx`, `indices`, `anchors` — so nesting
    them under a `params` object sends them where nothing looks. That shipped:
    `translate` answered 200 and moved the structure by (0, 0, 0)."""
    out = _run(app_at, pairs,
        """
        const m = await loaded("CO labelled");
        await m.applyOp("translate", { dx: 1.5, dy: 0, dz: 0 });
        const sent = globalThis.__calls[globalThis.__calls.length - 1];
        console.log(JSON.stringify({ route: sent.url, keys: Object.keys(sent.body).sort(),
                                     dx: sent.body.dx }));
        """
    )
    assert out["route"] == "/api/modify/translate"
    assert out["dx"] == 1.5, f"the op's arguments did not reach the body root: {out['keys']}"
    assert "params" not in out["keys"], (
        f"arguments nested where no route reads them: {out['keys']}"
    )



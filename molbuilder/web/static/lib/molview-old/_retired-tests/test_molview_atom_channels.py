"""MolView § 6.2 / § 9.5 — the data holds exactly what the filter enumerates.

Derived from ``docs/web/molview.md``, not from the source.  § 6.2, of the structure's
per-atom facts:

    Those three — element, labels, residue — are exactly what the filter enumerates from an
    atom (§ 9.5).  They are the same list, which is why filtering needs no case per property.

and § 9.5, on where the offered rows come from:

    Which rows are worth offering is read from the structure, not hard-coded ... That reading
    decides *what to offer*; the four rules decide *how to match*.  Keeping those apart is why
    a new label needs no panel change.

§ 13.3 turns that into: *every property the filter enumerates from an atom is a property the
structure actually carries; neither list can grow without the other.*  Both directions are
tested here — a channel with no field behind it, and a field with no channel in front of it,
are the two ways the lists come apart.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
ATOM = ROOT / "molbuilder/web/static/lib/molview/_atom.js"

_BOOT = f"""
    const atom = await import("file://{ATOM}");
"""


def _run(snippet: str) -> object:
    return run_node([], _BOOT + snippet)


def test_an_atom_enumerates_the_facts_it_carries_and_nothing_else():
    """§ 6.2 — element, labels and residue, each present only when the atom has one."""
    out = _run("""
        const full  = { element: "Au", labels: ["L-electrode"], residueName: "ALA" };
        const bare  = { element: "H" };                      // no labels, no residue
        const noRes = { element: "C", labels: ["bridge"] };   // a structure with no residues
        console.log(JSON.stringify({
            full:  Object.keys(atom.atomChannels(full)).sort(),
            bare:  Object.keys(atom.atomChannels(bare)).sort(),
            noRes: Object.keys(atom.atomChannels(noRes)).sort(),
            empty: Object.keys(atom.atomChannels(null)),
            elementIsOneValuePerAtom: atom.atomChannels(full).element.value,
            residueIsOneValuePerAtom: atom.atomChannels(full).residue.value,
        }));
    """)
    assert out["full"] == ["L-electrode", "element", "residue"]
    assert out["bare"] == ["element"], "a fact the atom does not carry offers no filter row"
    assert out["noRes"] == ["bridge", "element"]
    assert out["empty"] == [], "no atom, no channels — not a set of empty ones"
    assert out["elementIsOneValuePerAtom"] == "Au"
    assert out["residueIsOneValuePerAtom"] == "ALA"


def test_a_label_needs_no_case_of_its_own():
    """§ 9.5 / § 6.6 — a name nobody has ever seen filters like any other."""
    out = _run("""
        const invented = { element: "C", labels: ["something-nobody-planned-for"] };
        const reserved = { element: "C", labels: ["L-electrode"] };
        const ch = atom.atomChannels(invented);
        console.log(JSON.stringify({
            invented: Object.keys(ch).sort(),
            // A reserved name is stored, offered and matched exactly like any other label:
            // MolView interprets none of them (§ 6.6).
            sameShape: JSON.stringify(ch["something-nobody-planned-for"])
                    === JSON.stringify(atom.atomChannels(reserved)["L-electrode"]),
        }));
    """)
    assert out["invented"] == ["element", "something-nobody-planned-for"]
    assert out["sameShape"] is True, (
        "a reserved label is an ordinary label — if it is shaped differently here, "
        "MolView has started interpreting it")


def test_the_offered_rows_are_the_union_of_what_the_atoms_carry():
    """§ 9.5 — what to offer is READ from the structure, and read from all of it."""
    out = _run("""
        const atoms = [
            { element: "Au", labels: ["L-electrode"] },
            { element: "C",  labels: ["bridge"], residueName: "ALA" },
            { element: "H" },
        ];
        const offered = atom.channelKinds(atoms).map((c) => c.name);
        // Every offered row must be a row at least one atom actually answers to, and every
        // channel any atom carries must be offered.  Neither list may grow without the other.
        const carried = new Set();
        atoms.forEach((a) => Object.keys(atom.atomChannels(a)).forEach((n) => carried.add(n)));
        console.log(JSON.stringify({
            offered:      offered,
            offeredNotCarried: offered.filter((n) => !carried.has(n)),
            carriedNotOffered: [...carried].filter((n) => !offered.includes(n)),
        }));
    """)
    assert out["offeredNotCarried"] == [], "the panel offers a row no atom can match"
    assert out["carriedNotOffered"] == [], "an atom carries a fact the panel never offers"
    assert set(out["offered"]) == {"element", "residue", "L-electrode", "bridge"}


def test_no_atoms_means_no_rows():
    """§ 9.5 — with nothing loaded there is nothing to read the offered rows from."""
    out = _run("""
        console.log(JSON.stringify({
            none: atom.channelKinds([]),
            undef: atom.channelKinds(undefined),
        }));
    """)
    assert out["none"] == []
    assert out["undef"] == []

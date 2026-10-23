"""Single-point CONVERSION of an atom index — canonical 0-based ``Structure``
index ⇄ each engine's OWN numbering (both directions): *convert* changes where
counting starts, never which list the atom is in -- that is *map*, the atom
index map's (``model/overview.md`` § 2, *Map or convert*).

WHERE THE INDEX IS FIRST DEFINED (the fact).
    The canonical atom identity is the **0-based index into ``Structure``**
    (``elements[i]`` / ``positions[i]``).  It is established by the atom ORDER
    in the source file when it is parsed (``.xyz`` / ``.pdb`` / … → Structure);
    that order IS the identity.  Every layer references this identity, and all
    per-atom metadata (``frozen_atoms``, ``regions``, ``annotations``) are
    0-based indices into it.  See ``model/overview.md`` § 2 + § 2.1.

WHERE THE CONVERSION HAPPENS (this module).
    The engine-facing translators (``siesta/input.py``, ``pyscf/input.py``,
    ``transport/transiesta.py``) call THIS module — the single, explicit point
    where a 0-based identity becomes an engine's atom number (FORWARD), and
    where an engine's atom number in parsed OUTPUT becomes the 0-based identity
    again (INVERSE, :func:`convert_from_engine_numbering`).  Engines DIFFER, so each
    convention is one named, documented FACT function (forward) plus a base in
    the dispatch registry (both directions).  **Nothing else in the codebase
    may apply a bare ``i + 1`` OR ``n - 1`` to an atom index** — route it here
    so the convention is auditable in one place and bound by tests.

DISPLAY CONSISTENCY.
    The frontend displays 1-based (``lib/molview/_atom.js``), as SIESTA's
    ``.fdf`` and geomeTRIC's ``$freeze`` count -- so ``toDisplay(i)`` is a
    deck's number for the same atom where the deck lists the atoms as the
    view holds them; a sorted calculation's deck numbers its engine index
    + 1, while an editor shows the user index + 1 (``model/overview.md``
    § 2.1, § 2.2b F5).
"""
from __future__ import annotations


def siesta_numbering(i: int) -> int:
    """0-based ``Structure`` index → SIESTA ``.fdf`` atom number.

    FACT: SIESTA numbers atoms **1-based** in every index-bearing ``.fdf``
    block — ``AtomicCoordinatesAndAtomicSpecies`` order, ``%block
    Geometry.Constraints``, ``TS.Atoms`` (verified against SIESTA 5.4.2).
    """
    return i + 1


def siesta_numbering_from_end(i: int, n_atoms: int) -> int:
    """0-based ``Structure`` index → SIESTA's atom number counted from the
    LAST atom, of a deck of ``n_atoms``.

    FACT: a ``%block TS.Elec.<name>``'s ``elec-pos end`` names an atom from
    the end -- ``-1`` is the last atom (SIESTA 5.4 manual, TranSIESTA's
    electrode position) -- so atom ``i`` is ``-(n_atoms - i)``.
    """
    return -(n_atoms - i)


def geometric_numbering(i: int) -> int:
    """0-based ``Structure`` index → geomeTRIC constraint-file atom number.

    FACT: geomeTRIC ``$freeze`` / ``$set`` constraint files number atoms
    **1-based** (geomeTRIC's own convention) — even though PySCF's ``mol.atom``
    list (below) is 0-based.  This split is exactly why the translation must be
    engine-specific and not copied from SIESTA.
    """
    return i + 1


def pdb_numbering(i: int) -> int:
    """0-based ``Structure`` index → a PDB file's atom serial number.

    FACT: a PDB ``ATOM``/``HETATM`` record's serial (columns 7-11) counts
    **from 1** (wwPDB format 3.3).  A file format's numbering, not an
    engine's, kept here all the same: this module is the one place an index
    becomes a number counted from anything but 0.
    """
    return i + 1


def pyscf_numbering(i: int) -> int:
    """0-based ``Structure`` index → PySCF ``Mole.atom`` list index.

    FACT: PySCF's ``Mole.atom`` is a **0-based** Python list, so the identity
    maps through unchanged.  (Kept explicit so a reader sees the difference from
    :func:`geometric_numbering`; and so any future ``mol.atom`` index emission
    routes through here rather than assuming a base.)
    """
    return i


# ── The parametrized single-dispatch surface (both directions) ─────────────
#
# The three functions above are the per-engine FACTs (forward: internal →
# engine).  For code that handles the engine as a VARIABLE — a generic parser
# reading whatever engine's output, or the round-trip below — the dispatch
# entry points take the engine as an argument.  Each engine's convention is one
# BASE OFFSET (its lowest atom number): forward = ``i + base``, inverse =
# ``n - base``.  Adding an engine is one registry line, and its two directions
# can never disagree because they derive from the same base.  A test binds the
# named functions above to ``convert_to_engine_numbering`` so the two surfaces
# cannot drift.  THE ENGINE IS ALWAYS NAMED -- no default: a default of SIESTA
# made the dispatch read as SIESTA's own (the user, 2026-10-23).
_ENGINE_INDEX_BASE = {
    "siesta":    1,   # SIESTA .fdf atom numbers are 1-based
    "geometric": 1,   # geomeTRIC $freeze / $set are 1-based
    "pyscf":     0,   # PySCF Mole.atom is a 0-based list
    "pdb":       1,   # a PDB file's atom serial is 1-based
}


def _engine_base(engine: str) -> int:
    try:
        return _ENGINE_INDEX_BASE[engine]
    except KeyError:
        raise ValueError(
            f"unknown engine {engine!r}; "
            f"known engines: {sorted(_ENGINE_INDEX_BASE)}")


def convert_to_engine_numbering(i: int, engine: str) -> int:
    """0-based ``Structure`` index → ``engine``'s own atom number (FORWARD).

    The single, engine-parametrized entry point used when the target engine is
    a variable.  Equal to the per-engine FACT functions above
    (``convert_to_engine_numbering(i, "siesta") == siesta_numbering(i)``).  Unknown
    ``engine`` raises (fail-fast).
    """
    return i + _engine_base(engine)


def convert_from_engine_numbering(n: int, engine: str) -> int:
    """``engine``'s atom number → 0-based ``Structure`` index (INVERSE).

    The exact inverse of :func:`convert_to_engine_numbering`.  This is the RETURN LEG of
    the round-trip: when engine OUTPUT references an atom **by number** (a
    SIESTA constraints echo, a per-atom property keyed by number), translate it
    back to the canonical 0-based identity through HERE — never a bare
    ``n - 1``, which would be silently wrong for a 0-based engine (PySCF).

    (When engine output is order-preserved — coordinate / force blocks emit
    atoms in ``Structure`` order — the internal index is the row position and
    no number translation is needed; see ``model/overview.md`` § 2.)
    """
    return n - _engine_base(engine)

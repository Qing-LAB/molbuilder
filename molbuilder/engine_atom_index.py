"""Single-point atom-index translation — canonical 0-based ``Structure`` index
→ each engine's OWN atom-index convention.

WHERE THE INDEX IS FIRST DEFINED (the fact).
    The canonical atom identity is the **0-based index into ``Structure``**
    (``elements[i]`` / ``positions[i]``).  It is established by the atom ORDER
    in the source file when it is parsed (``.xyz`` / ``.pdb`` / … → Structure);
    that order IS the identity.  Every layer references this identity, and all
    per-atom metadata (``frozen_atoms``, ``regions``, ``annotations``) are
    0-based indices into it.  See ``data-vocabulary.md`` § 3.1–3.2.

WHERE THE TRANSLATION HAPPENS (this module).
    The engine-facing translators (``siesta/input.py``, ``pyscf/input.py``,
    ``transport/transiesta.py``) call THIS module — the single, explicit point
    where a 0-based identity becomes an engine's atom number.  Engines DIFFER,
    so each convention is one named, documented function.  **Nothing else in the
    codebase may apply a bare ``i + 1`` to an atom index** — route it here so
    the convention is auditable in one place and bound by tests.

DISPLAY CONSISTENCY.
    The frontend displays 1-based (``_atom-index.js``), chosen to equal the
    atom number the user reads in the generated files they cross-reference —
    SIESTA ``.fdf`` and geomeTRIC ``$freeze`` are both 1-based.  So
    ``siesta_atom_index(i)`` / ``geometric_atom_index(i)`` == the frontend's
    ``toDisplay(i)``; a test binds this equality.
"""
from __future__ import annotations


def siesta_atom_index(i: int) -> int:
    """0-based ``Structure`` index → SIESTA ``.fdf`` atom number.

    FACT: SIESTA numbers atoms **1-based** in every index-bearing ``.fdf``
    block — ``AtomicCoordinatesAndAtomicSpecies`` order, ``%block
    Geometry.Constraints``, ``TS.Atoms`` (verified against SIESTA 5.4.2).
    """
    return i + 1


def geometric_atom_index(i: int) -> int:
    """0-based ``Structure`` index → geomeTRIC constraint-file atom number.

    FACT: geomeTRIC ``$freeze`` / ``$set`` constraint files number atoms
    **1-based** (geomeTRIC's own convention) — even though PySCF's ``mol.atom``
    list (below) is 0-based.  This split is exactly why the translation must be
    engine-specific and not copied from SIESTA.
    """
    return i + 1


def pyscf_atom_index(i: int) -> int:
    """0-based ``Structure`` index → PySCF ``Mole.atom`` list index.

    FACT: PySCF's ``Mole.atom`` is a **0-based** Python list, so the identity
    maps through unchanged.  (Kept explicit so a reader sees the difference from
    :func:`geometric_atom_index`; and so any future ``mol.atom`` index emission
    routes through here rather than assuming a base.)
    """
    return i

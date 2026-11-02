"""Which atoms a PySCF deck gives a core potential -- PySCF's own rule.

``gto.M`` applies an effective core potential only when ``ecp`` names it
(PySCF 2.14 ``gto/mole.py``), so which elements get one is the person's
statement, resolved here once for both PySCF decks and PySCF's validator.
"""
from __future__ import annotations

import fnmatch
from typing import Dict, List, Optional, Sequence

from ..chemistry import resolve_element
from ..structure import Structure


def resolve_ecp(struct: Structure,
                ecp: str,
                ecp_atoms: "Sequence[str]") -> "Optional[Dict[str, str]]":
    """Which elements get which ECP -- the ONE place the rule lives.

    Called from BOTH the PySCF deck (``pyscf/input.py``) and the vibration
    deck (``pyscf/vibration_emitters.py::_emit_build_mol``), and by PySCF's
    validator's hint, so the two generators cannot drift.

    Inputs:
      * ``ecp``        -- the ECP name, e.g. ``"lanl2dz"``.  Empty = no ECP.
      * ``ecp_atoms``  -- element patterns naming which atoms get it:
                          ``[]`` none · ``["*"]`` every element present ·
                          ``["Au"]`` that element · ``["A*"]`` every symbol
                          starting with A · ``["Au", "Pt"]`` several.

    Returns ``{element: ecp}`` for the elements actually present in
    ``struct`` that match, or ``None`` when nothing does -- which is the
    signal to omit the ``gto.M(ecp=...)`` kwarg entirely.

    **Empty means empty, and nothing is chosen for the user** (user's
    ruling: *"there is no point to limit matching to heavy -- who defines
    heavy?  there is no clear reasoning or standard ... explicit is better
    than implicit."*).  An ECP named on a def2 basis is written as named:
    PySCF applies a core potential only when ``ecp`` names it, and a def2
    basis's own is named like any other: ``ecp = "def2-SVP"``.

    ``validation`` still HINTS when a structure looks like it wants an ECP
    and none is declared.  A hint is confirmed by a person; it is not this
    function quietly acting.
    """
    name = (ecp or "").strip()
    patterns = [p.strip() for p in (ecp_atoms or []) if p and p.strip()]
    if not name or not patterns:
        return None

    # Keyed by the LABEL as written, because that is what the emitted
    # ``atom=`` block says: an ``ecp={"Au": ...}`` never reaches an atom the
    # script calls ``Au1``.  A pattern matches either the label or the
    # element it names, so ``["Au"]`` still selects ``Au1``/``Au2`` while
    # ``["Au1"]`` selects just the one species.
    #
    # Matching is CASE-SENSITIVE, the same rule as everywhere else: folding
    # is what turns ``CA`` into calcium, and this function's own docstring
    # already says explicit beats implicit.  A pattern that matches nothing
    # yields no ECP, and validation's existing hint is what surfaces that.
    present: List[str] = []
    for el in struct.elements:
        label = str(el).strip()
        if label not in present:
            present.append(label)

    def _selects(label: str) -> bool:
        keys = [label]
        try:
            keys.append(resolve_element(label))
        except KeyError:
            pass                       # check_species_labels reports it
        return any(fnmatch.fnmatchcase(k, p) for k in keys for p in patterns)

    matched = {label: name for label in present if _selects(label)}
    return matched or None


__all__ = ["resolve_ecp"]

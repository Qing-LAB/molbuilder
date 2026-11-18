"""RDKit-based DNA / RNA builder.

Uses ``rdkit.Chem.MolFromSequence(seq, flavor=...)`` to construct a
chemically correct polymer (proper backbone, all atoms incl. H), then
embeds a 3-D conformer with ETKDG and cleans up with UFF.

Important caveat: RDKit's embedder is conformation-aware but not
*helix*-aware.  For oligos longer than ~6 residues the conformer is
typically a folded clump rather than a B-form helix.  Fine if you'll
run a thorough DFT optimisation anyway; not what you want for
production B-DNA / A-RNA structures.
"""

from __future__ import annotations

from typing import Optional

from ...chemistry import (INSTALL_TOOLS, TOOLS, embed_conformer,
                          rdkit_to_structure, require)
from ...structure import Structure
from . import Built

#: This backend builds a single strand only.
DUPLEX = False
#: How it is installed: a tool the host env's recipe lists
#: (`chemistry.INSTALL_TOOLS`) -- the host env's (``ENV``).
INSTALL = INSTALL_TOOLS
ENV = "host"
#: What RDKit is needed for here, as a missing-tool answer says it.
_NEEDED_FOR = "Building a nucleic acid on the RDKit backend"


#: EACH ASK'S rdkit.Chem.MolFromSequence FLAVOR, measured (RDKit 2026.03,
#: 2026-11-18): a flavor whose residues hold an O2' is RNA,
#: one without DNA; a 3'-phosphate is its own residue after the last
#: nucleotide.  Every terminal is built as asked.
#:   2 / 6 = RNA / DNA, 5'-OH and 3'-OH
#:   3 / 7 = RNA / DNA, 5'-phosphate
#:   4 / 8 = RNA / DNA, 3'-phosphate
#:   5 / 9 = RNA / DNA, both phosphates
_FLAVOR = {
    ("dna", "OH"): 6,  ("dna", "5P"): 7,
    ("dna", "3P"): 8,  ("dna", "PP"): 9,
    ("rna", "OH"): 2,  ("rna", "5P"): 3,
    ("rna", "3P"): 4,  ("rna", "PP"): 5,
}
#: The terminal state each flavor writes.
_FLAVOR_TERMINAL = {2: "OH", 3: "5P", 4: "3P", 5: "PP",
                    6: "OH", 7: "5P", 8: "3P", 9: "PP"}


def built(kind: str, form: str, terminal: str) -> Built:
    """What this backend builds of the ask (`backends.built`), read off the
    flavor its build takes (:data:`_FLAVOR`): an embedded conformer, no helix,
    and the terminal that flavor writes -- the one asked, every time."""
    return Built(
        form=None, terminal=_FLAVOR_TERMINAL[_FLAVOR[(kind, terminal)]],
        form_words="RDKit embeds a conformer (ETKDG), not a helix")


def is_available() -> bool:
    """RDKit here, by the module the tools table says proves it
    (`chemistry.TOOLS`) -- found without loading it, so ``auto`` asks
    cheaply what the build then imports."""
    import importlib.util
    try:
        return importlib.util.find_spec(TOOLS["rdkit"].module) is not None
    except ImportError:
        return False


def build(kind: str, sequence: str, form: str, terminal: str,
          title: Optional[str] = None, double_strand: bool = False,
          strand2: Optional[str] = None) -> Structure:
    """A single strand of ``kind`` from ``sequence`` -- read by the grammar
    already -- with its terminal as asked; no form is shaped here, the
    embedded conformer is whatever ETKDG makes (the check after the build
    says so, :func:`built`)."""
    Chem = require("rdkit", needed_for=_NEEDED_FOR)
    AllChem = require("rdkit", needed_for=_NEEDED_FOR,
                      module="rdkit.Chem.AllChem")
    flavor = _FLAVOR[(kind, terminal)]
    mol = Chem.MolFromSequence(sequence, flavor=flavor)
    if mol is None:
        # The letters were read by the grammar already: RDKit failing on
        # them is the engine's failure, never the request's.
        raise RuntimeError(
            f"RDKit could not build a {kind} polymer from sequence "
            f"{sequence!r} (flavor={flavor})")
    # Each hydrogen in the residue of the atom it bonds to (RDKit's own
    # ``addResidueInfo``): left out, all of them read as one residue `MOL`.
    mol = Chem.AddHs(mol, addResidueInfo=True)
    if not embed_conformer(mol):
        raise RuntimeError(
            f"RDKit ETKDG could not embed a 3-D conformer for "
            f"{kind} {sequence!r}")
    # MMFF doesn't have parameters for nucleic acids -- use UFF.
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # optimisation is best-effort
    return rdkit_to_structure(
        mol, title=title or f"{kind} {sequence} (rdkit, embedded conformer)")

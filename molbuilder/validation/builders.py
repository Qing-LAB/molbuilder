"""What a build says of what it built -- the check after every builder.

A builder returns the structure; this says what it found
(`science/validation.md` § 4.1 R5, *decided with the user, 2026-10-31,
(b)*): the structure's own, read off what was built -- a peptide's
C-terminus with no ``OXT``, a nucleic acid's steric clash, a SMILES's or a
name's molecules overlapping -- and what
the backend that built it does not build of what was asked, each
backend's own answer (`builders.backends.built`).  Findings are
``Issue``s, never Python warnings, so they reach a web user: the build route answers them as
notices, every builder panel draws them through the one renderer, and the
command line prints them as `validation.report` prints every finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..issues import Issue
from ..structure import Structure


@dataclass(frozen=True)
class Asked:
    """What the build asked, as the check reads it: the backend that ran
    (``auto`` resolved), the form and terminal it was built in (the kind's
    own when none was asked -- ``defaults`` names which), and whether clash
    relief was asked."""
    backend: Optional[str] = None
    form: Optional[str] = None
    terminal: Optional[str] = None
    relax_clashes: bool = False
    defaults: Tuple[str, ...] = ()

    def said(self, field: str) -> str:
        """How a finding names the form or terminal it was built in: asked,
        or the kind's default."""
        return "(the default)" if field in self.defaults else "asked"


def _aldehyde_end(struct: Structure, asked: Asked) -> List[Issue]:
    """A peptide whose C-terminus carries no ``OXT`` is an ALDEHYDE.

    PeptideBuilder does not write the C-terminal ``OXT``, so the chain ends
    one oxygen short of a free acid; `add_hydrogens` then saturates that
    dangling valence with an H and the result is a valid molecule -- a
    peptide aldehyde (``-CHO``), not a peptide (``-COOH``): a different
    compound, with no ionizable C-terminus.  **Molbuilder does not add the
    oxygen** (user ruling 2026-09-20): placing an OXT is a structural guess
    this tool has no business making on the person's behalf, so the limit is
    said.  Nothing else catches it -- the hydrogen count is identical either
    way.  Read off what was built: the day the kit writes OXT, this says
    nothing."""
    if "OXT" in struct.atom_names:
        return []
    return [Issue(
        "warn",
        "This peptide's C-terminus is an ALDEHYDE (-CHO), not the usual free "
        "acid (-COOH): the structure kit does not write the terminal OXT "
        "oxygen, and molbuilder does not add one for you. The molecule is "
        "one oxygen short of the peptide you named, so any energy, dipole or "
        "spectrum computed from it describes a peptide aldehyde -- a "
        "different compound, with no ionizable C-terminus. Add the OXT "
        "yourself if you want the free acid.",
        "build.peptide_aldehyde")]


def _steric_clash(struct: Structure, asked: Asked) -> List[Issue]:
    """Two atoms of different residues closer than a steric clash in a
    built structure (`chemistry.steric_clash`, `CLASH_CONTACT_A`,
    `NEAR_COINCIDENT_A`), the pair named -- on every build, said and never
    repaired unasked *(decided with the user, 2026-10-31, (b))*.  The danger
    for a later relaxation is the first SCF step from overlapping atoms.

    Two ways it is built, read off what was built: BETWEEN THE TWO STRANDS
    of a duplex -- a mismatched base pair, placed at the standard frame,
    interpenetrates (a canonical pair never does), relieved where it clashes
    when the build asked (``relax_clashes``, the builder's) and an overlap
    left said; or WITHIN A STRAND as built -- tleap's extended chain sets each link's backbone torsions so
    that C3' and the next residue's C5' are 2.66 A apart, and their
    hydrogens, as placed (tleap's on RNA, the methylene repair's on DNA),
    1.14 A apart (measured 2026-10-31)."""
    from ..builders.backends import BACKENDS
    from ..chemistry import (CLASH_CONTACT_A, NEAR_COINCIDENT_A, is_atom,
                             steric_clash)
    clash = steric_clash(struct)
    if clash is None:
        return []
    d, i, j = clash
    between_strands = struct.chain_ids[i] != struct.chain_ids[j]
    if between_strands and asked.relax_clashes:
        if d >= NEAR_COINCIDENT_A:
            return []
        return [Issue(
            "warn",
            f"Relieved clashes but a residual NEAR-COINCIDENT contact "
            f"remains ({d:.2f} A) -- an unusually dense run of mismatches; "
            f"inspect or minimize further before relaxing the geometry.",
            "build.steric_clash")]

    def _lab(k: int) -> str:
        # A HYDROGEN NAMED PLAIN ``H`` -- every one OpenBabel places -- BY
        # THE ATOM IT IS BONDED TO: the nearest heavy atom of its own
        # residue (a clash partner may sit closer than its bond).
        name = struct.atom_names[k]
        if name == "H":
            own = [m for m in range(struct.n_atoms)
                   if not is_atom(struct.elements[m], "H")
                   and struct.chain_ids[m] == struct.chain_ids[k]
                   and struct.residue_ids[m] == struct.residue_ids[k]]
            if own:
                d_k = np.linalg.norm(struct.positions[own]
                                     - struct.positions[k], axis=1)
                name = f"H on {struct.atom_names[own[int(np.argmin(d_k))]]}"
        return f"{struct.chain_ids[k]}{struct.residue_ids[k]}:{name}"

    danger = ("  These atoms are NEAR-COINCIDENT -- relaxing as-is risks an "
              "explosive first SCF step. " if d < NEAR_COINCIDENT_A else "  ")
    if between_strands:
        return [Issue(
            "warn",
            f"This duplex has a steric CLASH at {_lab(i)}<->{_lab(j)} "
            f"({d:.2f} A) -- a non-Watson-Crick (mismatched) base pair at the "
            f"standard frame." + danger + "Build it again with clash relief "
            "(Relieve clashes on the DNA panel) -- a short force-field "
            "minimization that removes "
            "the near-coincidence, SIESTA relaxing the rest -- or minimize "
            "externally before a geometry optimization.",
            "build.steric_clash")]
    return [Issue(
        "warn",
        f"This structure has a steric CLASH at {_lab(i)}<->{_lab(j)} "
        f"({d:.2f} A): atoms of two residues closer than {CLASH_CONTACT_A} A, "
        f"as built on {BACKENDS.get(asked.backend, asked.backend)}."
        + danger
        + "Minimize it with a force field before a geometry optimization, so "
        "the first SCF step does not start from overlapping atoms."
        + ("  Clash relief applies to an explicit duplex only."
           if asked.relax_clashes else ""),
        "build.steric_clash")]


def _molecules_overlap(struct: Structure, asked: Asked) -> List[Issue]:
    """Two molecules a SMILES or a name names -- each its own residue
    (`chemistry.molecules_to_structure`) -- placed by the embedder so close
    that two of their atoms sit nearer than half the sum of their van der
    Waals radii (`chemistry.van_der_waals_radius`): the molecules reach
    halfway into each other.  A hydrogen bond, and an ion pair at its own
    bond length, sit farther.  The
    pair named that overlaps most, by how far inside its reach.  Said, never
    moved: the molecules are moved apart on the Molbuilder tab (the user,
    2026-11-18: "Half the vdW sum")."""
    from ..chemistry import van_der_waals_radius
    if len(set(struct.residue_ids)) < 2:
        return []
    pos = np.asarray(struct.positions, dtype=float)
    radius = np.array([van_der_waals_radius(e) for e in struct.elements],
                      dtype=float)
    rid = np.asarray(struct.residue_ids)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    reach = (radius[:, None] + radius[None, :]) / 2
    inside = (rid[:, None] != rid[None, :]) & (d < reach)
    if not inside.any():
        return []
    depth = np.where(inside, d / reach, np.inf)
    i, j = np.unravel_index(np.argmin(depth), depth.shape)
    return [Issue(
        "warn",
        f"Two of the molecules overlap: {struct.atoms_named([i, j])} sit "
        f"{d[i, j]:.2f} A apart, nearer than half the sum of their van der "
        f"Waals radii ({reach[i, j]:.2f} A).  The molecules are where the "
        f"3-D embedding put them: move them apart on the Molbuilder tab "
        f"before a calculation.",
        "build.molecules_overlap")]


#: THE STRUCTURE'S OWN FINDINGS, by kind, read off what was built: a
#: peptide's C-terminus; a nucleic acid's steric clash; a SMILES's or a
#: name's molecules overlapping (`engines/builders.md` § 6).
_STRUCTURE_CHECKS: Dict[str, Tuple[Callable[[Structure, Asked],
                                            List[Issue]], ...]] = {
    "peptide": (_aldehyde_end,),
    "dna":     (_steric_clash,),
    "rna":     (_steric_clash,),
    "smiles":  (_molecules_overlap,),
    "name":    (_molecules_overlap,),
}


def _backend_limits(kind: str, asked: Asked) -> List[Issue]:
    """A form or a terminal asked that the backend that ran does not build --
    built all the same, as the backend makes it -- in its own words, of
    itself only (`builders.backends.built`): which other backend builds the
    whole ask is no single backend's to say, and the Backend menu lists each
    with its own words.  Nothing for a build that asks none."""
    if not asked.backend or asked.form is None or asked.terminal is None:
        return []
    from ..builders.backends import built
    made = built(asked.backend, kind, asked.form, asked.terminal)
    issues: List[Issue] = []
    if made.form != asked.form:
        how = (f"was built {made.form}-form" if made.form
               else "was not built")
        issues.append(Issue(
            "warn", f"The {asked.form}-form {asked.said('form')} {how}: "
                    f"{made.form_words}.",
            "build.form"))
    if made.terminal != asked.terminal:
        issues.append(Issue(
            "warn",
            f"The terminal {asked.terminal} {asked.said('terminal')} was "
            f"built {made.terminal}: {made.terminal_words}.",
            "build.terminal"))
    return issues


def check_built(struct: Structure, *, kind: str, asked: Asked) -> List[Issue]:
    """What a build of ``kind`` says of ``struct``, the structure it built:
    its own findings (:data:`_STRUCTURE_CHECKS`), then what the backend that
    ran does not build of what was ``asked``.  ``[]`` for a clean build."""
    issues: List[Issue] = []
    for check in _STRUCTURE_CHECKS.get(kind, ()):
        issues += check(struct, asked)
    issues += _backend_limits(kind, asked)
    return issues

"""Cross-engine chemistry helpers.

L1 module shared by SIESTA + PySCF + Spectra emitters and the
validation pass: a small library of inferences that work on a bare
``Structure`` (no explicit bond orders required) using heavy-atom
adjacency.

Its public surface -- the charge and spin facts, the hydrogens, and the
builders' shared steps (the tools they import, the one embedding and the
one conversion of a built molecule, the atoms they add, the checks' radii)
-- is listed once, in `model/chemistry.md` § 4.  The DECISION -- which
charge and spin a calculation runs with -- is `electronic_state.py`'s
(science/chemistry-correctness.md § 2a).

The chemistry rule encoded in the phosphate helpers is the standard
interpretation: one P=O double bond, the other oxygens single-bonded
to H, R-O-, or O- depending on protonation.  "Non-bridging" is
inferred purely from heavy-atom adjacency (only P as a heavy
neighbour).  Other charged groups (carboxylates, protonated amines,
quaternary ammonium nitrogens) are NOT counted.
"""

from __future__ import annotations


import math
from dataclasses import dataclass
from typing import (Dict, Iterable, List, Optional, Tuple)

import numpy as np

from .structure import Structure, split_species_label


class BackendUnavailable(RuntimeError):
    """What a build needs is not here: a tool this install lacks -- a
    nucleic-acid backend (`builders.backends.BACKENDS`), a Python package a
    builder imports (:data:`TOOLS`), the hydrogen engines -- or PubChem out
    of reach.  The one type every builder raises for it
    (`engines/builders.md` § 1); ``missing`` names it, when one is the
    answer, for a surface that acts on it -- the Build page answers this as
    advice, not as a server fault (`web-api.md` § 1) -- and ``env`` the env
    that holds it, where the words cannot name the command that installs it
    (``"host"``, the host env's packages, from here, below the env
    framework): the build door adds it (`builders.backends.install_command`).

    DEFINED HERE, the lowest layer that raises it (`chemistry` is L1,
    `architecture.md` § 3): `builders/` (L2) raises it for its backends and
    reaches it downward, and :func:`add_hydrogens` raises it for its engines.
    """

    def __init__(self, message: str, *, missing: Optional[str] = None,
                 env: Optional[str] = None):
        super().__init__(message)
        self.missing = missing
        self.env = env


class BuildRefused(ValueError):
    """What was asked of a builder cannot be built -- a letter its grammar
    does not take, a form no nucleic acid is built in, a Z-form sequence
    3DNA does not build, a name PubChem does not know -- said in the
    builder's own words (`engines/builders.md` § 1).  The one type every
    builder refuses with, so every surface says it one way: the build route
    as a bad request (it is a ``ValueError``), the command line in one line
    (`cli.main`'s door) -- and a fault of ours, which is none of these, is
    never mistaken for it.  Defined here beside :class:`BackendUnavailable`,
    the lowest layer the builders share."""


@dataclass(frozen=True)
class Tool:
    """A Python package a builder imports: the name a person knows it by,
    and the module that proves it is here."""
    name: str
    module: str


#: THE TOOLS A BUILDER IMPORTS, by the ``missing`` name a surface acts on
#: (`engines/builders.md` § 1): one home for each one's words, read by
#: :func:`require` and by every message naming one.  How a tool is installed
#: is the host env's recipe's, never repeated here (`envs/host-env.txt`,
#: "one list, written here and nowhere else"), and so is how a command is
#: spelled (`envs.hints`, which this module, below it, cannot reach): every
#: message says the tool is one of the host env's packages
#: (:data:`INSTALL_TOOLS`), raised with ``env="host"``, and the build door,
#: above, adds the env framework's command that installs it
#: (`builders.backends.install_command`).
TOOLS = {
    "rdkit": Tool("RDKit", "rdkit.Chem"),
    "openbabel": Tool("OpenBabel", "openbabel.openbabel"),
    "peptidebuilder": Tool("PeptideBuilder", "PeptideBuilder"),
    "pubchempy": Tool("PubChemPy", "pubchempy"),
}

#: HOW ANY OF :data:`TOOLS` IS INSTALLED, in words: one of the host env's
#: packages, which the env framework's repair installs
#: (`ops/env-framework.md` § 7).
INSTALL_TOOLS = "it is one of the host env's packages"


def require(missing: str, *, needed_for: str, module: Optional[str] = None):
    """The tool :data:`TOOLS` names ``missing``, imported -- its own module,
    or ``module`` of it -- or :class:`BackendUnavailable` naming it, what it
    is ``needed_for`` and how it is installed (:data:`INSTALL_TOOLS`): THE
    one door for a Python package a builder needs (`engines/builders.md`
    § 1)."""
    import importlib
    try:
        return importlib.import_module(module or TOOLS[missing].module)
    except ImportError as exc:
        raise BackendUnavailable(
            f"{needed_for} needs {TOOLS[missing].name}, which this install "
            f"lacks: {INSTALL_TOOLS}.",
            missing=missing, env="host") from exc


#: THE ETKDG SEED every RDKit embedding takes -- a SMILES's and an RDKit
#: nucleic acid's -- pinned for reproducibility.
ETKDG_SEED = 0xF00D


def embed_conformer(mol, seed: int = ETKDG_SEED) -> bool:
    """Embed a 3-D conformer in the RDKit molecule ``mol`` -- ETKDGv3, then a
    random-coordinates retry -- THE one embedding recipe, a SMILES's and an
    RDKit nucleic acid's.  True iff a conformer was embedded.  An RDKit
    molecule in hand is RDKit installed: its caller asked for it already
    (:func:`require`)."""
    from rdkit.Chem import AllChem
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != -1:
        return True
    params.useRandomCoords = True
    return AllChem.EmbedMolecule(mol, params) != -1


def molecules_to_structure(elements, positions, molecules, *, pdb=None,
                           title: Optional[str] = None) -> Structure:
    """An embedded molecule as a :class:`Structure` -- THE one conversion,
    an RDKit build's (:func:`rdkit_to_structure`) and an OpenBabel one's
    (`smiles._openbabel_build`): each atom its PDB record where the engine
    carries one (``pdb[i]``: name, residue number, residue name, chain -- a
    nucleotide's), else its element and number (``C1``) in residue ``MOL``
    of chain ``A``, one residue for each molecule (``molecules``: each one's
    atom indices, from 0, the engine's own split of its bonds) -- so a
    SMILES's several molecules are told apart (`engines/builders.md` § 6)."""
    from .engine_atom_index import pdb_numbering
    molecule_of = {i: k for k, atoms in enumerate(molecules) for i in atoms}
    names, rids, rnames, chains = [], [], [], []
    for i, (element, record) in enumerate(
            zip(elements, pdb or [None] * len(elements))):
        name, rid, rname, chain = record or ("", 0, "", "")
        names.append(name or f"{element}{pdb_numbering(i)}")
        rids.append(rid or molecule_of[i] + 1)
        rnames.append(rname or "MOL")
        chains.append(chain or "A")
    return Structure(elements=list(elements),
                     positions=np.asarray(positions, dtype=float),
                     atom_names=names, residue_ids=rids, residue_names=rnames,
                     chain_ids=chains, title=title or "")


def rdkit_to_structure(mol, *, title: Optional[str] = None) -> Structure:
    """An RDKit molecule's embedded conformer as a :class:`Structure`, by
    :func:`molecules_to_structure`: RDKit's own split into molecules
    (``GetMolFrags``) and each atom's PDB record where it carries one."""
    from rdkit import Chem
    conf = mol.GetConformer()
    atoms = list(mol.GetAtoms())

    def record(atom):
        info = atom.GetPDBResidueInfo()
        return ((info.GetName().strip(), info.GetResidueNumber(),
                 info.GetResidueName().strip(), info.GetChainId().strip())
                if info else None)
    return molecules_to_structure(
        [atom.GetSymbol() for atom in atoms],
        [[p.x, p.y, p.z] for p in map(conf.GetAtomPosition,
                                      range(len(atoms)))],
        Chem.GetMolFrags(mol), pdb=[record(atom) for atom in atoms],
        title=title)


# --------------------------------------------------------------------- #
#  The periodic table, Z-indexed                                        #
# --------------------------------------------------------------------- #
#
# ONE home, here: SIESTA's outputs (``parse/engines/siesta_mdnc.py``,
# ``parse/coords/siesta_xv.py``) store ``iza`` (atomic numbers), never
# symbols, so a reader must name the elements itself.
#
# ``chemistry`` is the right layer for it: it imports only
# ``structure``, so nothing gains a dependency by reaching for this.
#
# Index IS the atomic number -- position 0 is the placeholder that
# makes that true.
PERIODIC_TABLE: List[str] = [
    "",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
]

def symbol_for_z(z: int) -> str:
    """The element symbol for atomic number *z*.

    Raises rather than returning a placeholder: a Z this table does not
    cover means the file names an element molbuilder cannot represent,
    and a silent ``"X"`` would flow into a Structure and be drawn,
    measured and written out as if it were real.
    """
    if not isinstance(z, (int, np.integer)) or z < 1 or z >= len(PERIODIC_TABLE):
        raise ValueError(
            f"atomic number {z!r} is outside the table this build carries "
            f"(1..{len(PERIODIC_TABLE) - 1}).  A file naming it describes an "
            f"element molbuilder has no symbol for; extend PERIODIC_TABLE "
            f"rather than letting an unnamed atom through.")
    return PERIODIC_TABLE[int(z)]


# Charged amino-acid side chains at physiological pH (7.4).  Used by
# `expected_pH7_peptide_charge()` to estimate the net charge of a
# peptide for the validator's "you built a neutral peptide but it
# would be charged at pH 7" hint.
#
# Histidine is intentionally skipped: pKa ~6 means it's roughly half-
# protonated at pH 7, and its protonation state is sequence- and
# environment-dependent (usually neutral N-tau-H tautomer in
# proteins).  Cys and Tyr have side-chain pKa > 8, so they're
# neutral at pH 7 -- not counted.
_CHARGED_RESIDUES_PH7 = {
    "ASP": -1, "GLU": -1,                       # acidic
    "LYS":  1, "ARG":  1,                       # basic
    # The methylated lysines stay cationic: the mono- and dimethyl amines
    # protonated as lysine's is, the trimethyl a quaternary ammonium, +1 at
    # any pH (`model/chemistry.md` § 1).
    "MLZ":  1, "MLY":  1, "M3L":  1,
}


# Open-shell transition metals + lanthanides.  Their ground-state
# configurations have unpaired electrons in d (or f) shells in every
# common oxidation state, so running them through a closed-shell
# (restricted) SCF typically converges to a fictitious state with
# garbage forces.  A blank spin on a structure holding one is decided
# unrestricted, at the metal's usual count (`electronic_state.recommend`),
# and a stated restricted is warned (ES9, `science/chemistry-correctness.md`
# § 2a).
#
# Source: ground-state electron configurations from NIST atomic
# spectra database.
#
# Three physical categories:
#
#   1. OPEN_D_TRANSITION_METALS — incomplete d-shell in the atomic
#      ground state AND extended phases.  Stoner criterion satisfied
#      for the 3d ferromagnets; itinerant moments for the 4d / 5d
#      analogues.  Open-shell DFT is the default expectation.
#
#   2. NOBLE_METALS_S1 — Cu, Ag, Au.  Atomic ground state is
#      nd¹⁰ (n+1)s¹ — single unpaired s electron — but in any
#      extended metallic context (cluster ≥ 4 atoms, surface, bulk,
#      junction) the s-band delocalizes and the system is closed-
#      shell singlet for even total electron count.  Stoner criterion
#      fails for noble metals: I·N(E_F) < 1, no spontaneous magnetism
#      in bulk.  Standard treatment for Au transport junctions is
#      spin-restricted DFT.
#      Refs:
#        * Taylor, Brandbyge, Stokbro, PRB 63 (2001) 245407 — the
#          original TranSIESTA Au-BDT-Au paper, spin-restricted.
#        * Ke, Baranger, Yang, JCP 122 (2005) 074704 — Au-BDT-Au NEGF.
#        * Verzijl, Thijssen, JPCC 116 (2012) 24811 — DFT+Σ Au-
#          alkanedithiol-Au benchmark, spin-restricted.
#        * Marder, "Condensed Matter Physics" Ch. 17 — Stoner
#          criterion derivation; Cu/Ag/Au listed as non-magnetic.
#      When open-shell IS the right call for noble-metal systems:
#      sub-4-atom clusters (shell-closing incomplete; Au_2 / Au_4),
#      single Au atom on insulator (Au/CeO2, Au/MgO catalysis lit.),
#      Au with magnetic 3d co-adsorbate (Au-Co, Au-Fe alloys), or
#      explicit Kondo / spin-orbit studies.  Users in those regimes
#      override via the form.
#
#   3. CLOSED_D10_METALS — Zn, Cd, Hg (always nd¹⁰ (n+1)s² in
#      common oxidation states) PLUS Pd (4d¹⁰ 5s⁰ atomic ground state
#      per NIST) AND Pt (5d⁹ 6s¹ atom but 5d¹⁰-like in metallic
#      bonding; same logic as the noble metals but conventionally
#      treated as closed-shell in catalysis surface DFT).
OPEN_D_TRANSITION_METALS = frozenset({
    # First-row 3d (incomplete d-shell)
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    # Second-row 4d (Pd excluded — closed-shell atomic ground state)
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    # Third-row 5d (Pt + Au excluded — handled below)
    "Hf", "Ta", "W",  "Re", "Os", "Ir",
    # Lanthanides (4f incomplete)
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Common actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu",
})

NOBLE_METALS_S1 = frozenset({"Cu", "Ag", "Au"})

CLOSED_D10_METALS = frozenset({
    "Zn", "Cd", "Hg",          # ns² nd¹⁰ — always closed-shell
    "Pd",                       # 4d¹⁰ 5s⁰ atomic ground state (NIST)
    "Pt",                       # 5d⁹ 6s¹ atom; metallic Pt is
                                # conventionally closed-shell in surface
                                # DFT.  Catalysis lit. treats Pt(111)
                                # as RKS unless studying magnetism.
})

#: THE NON-METALS, with the metalloids molecules are built from -- the heavy
#: atoms an organic molecule's hydrogens are counted against.  The H/heavy
#: note's reference range is organic molecules', so a metal's atoms -- a gold
#: lead, a junction's electrodes -- are no part of its count
#: (`science/overview.md` § 4; 2026-10-13, a lead read 0/75).
NONMETALS = frozenset({
    "H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
    "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn",
})

def resolve_element(label: str) -> str:
    """The element a species LABEL names.

    **A species label is the user's; the element is ours to derive.**
    SIESTA's ``%block ChemicalSpeciesLabel`` is ``index Z label`` and PySCF
    takes labelled atoms, so ``Au1`` / ``Au2`` -- two gold species carrying
    different basis or pseudopotential -- is ordinary input, not a typo.  We
    do not police the name.  What has to be right is the Z beside it, and
    that is what this resolves.

    **Explicit, not clever: the name is looked up VERBATIM.**  Split a
    trailing index off the label, then use what is left as the key exactly
    as written:

        ``Au1`` -> ``Au`` + index 1 -> found
        ``Au``  -> ``Au``           -> found
        ``au``  -> ``au``           -> KeyError
        ``FE``  -> ``FE``           -> KeyError

    No case correction (user ruling, 2026-09-09: *"there are so many
    misdetections associated with this"*).  Case-folding a two-letter symbol
    is how ``CA`` becomes calcium and ``CO`` becomes cobalt, and the guesses
    that follow are unfixable because nothing records that a guess happened.
    A label we cannot look up is a question for the user, not something to
    resolve on their behalf.

    This costs nothing on file input, because **the format readers have
    already decoded**: ``from_xyz`` is ``ase.io.read``, whose
    ``get_chemical_symbols()`` is canonical, and the PDB reader decodes
    columns 77-78, whose convention IS uppercase (``MG``, ``NA``, ``CL``).
    Decoding a format is not correcting a user.  What reaches here
    un-canonical is what a person typed: a UI field, or a species label they
    chose -- and there an error naming the label is the useful answer.

    Raises KeyError naming the label when the name is not an element.  **An
    element denotes Z >= 1**: ASE's table also lists ``X``, its dummy atom at
    Z = 0, which is a placeholder and not an element -- in an engine input it
    is a ghost species.

    **Callers emitting an engine input MUST let that propagate.**  A species
    with no element is not something a calculation can run, and the
    alternative -- quietly writing ``Z=0`` into ``ChemicalSpeciesLabel`` --
    is the defect this function exists to end.

    NOT for PDB atom names or residue names.  ``CA`` is an alpha carbon in
    the atom-name namespace and calcium in this one; the two share strings
    and share nothing else.  Those namespaces are carried verbatim
    (``Structure.atom_names``, ``Structure.residue_names``) and never
    reach this door.
    """
    from ase.data import atomic_numbers as _Z
    name, _index = split_species_label(label)
    if _Z.get(name, 0) >= 1:
        return name
    hint = (f"  Did you mean {name.capitalize()!r}?  Case is not corrected -- "
            f"folding two letters is how 'CA' becomes calcium."
            if name.capitalize() in _Z else "")
    origin = (f" (from species label {label!r})"
              if name != str(label).strip() else "")
    if name in _Z:
        raise KeyError(
            f"{name!r}{origin} is ASE's dummy atom (Z = 0), not an element: "
            f"a calculation needs the element behind every atom.")
    raise KeyError(f"unknown element symbol {name!r}{origin}.{hint}")


def is_atom(label: str, symbol: str) -> bool:
    """Is this species label an atom of ``symbol``?

    **The one way to ask.**  ``el == "P"`` is the wrong question the moment a
    user labels a species: ``Au1`` is gold, and a bare ``==`` says it is not.
    Comparing labels also invites the fold this module refuses -- once you
    write ``el.capitalize() == sym`` you have made ``CA`` calcium again.

        is_atom("Au1", "Au")  -> True
        is_atom("Au",  "Au")  -> True
        is_atom("au",  "Au")  -> False    (a name we cannot look up)
        is_atom("Xx",  "Au")  -> False

    ``symbol`` is an element symbol, not a label: asking whether an atom is
    ``"Au1"`` is asking about the user's naming, which is not a question this
    module answers.  A label that names no element is simply not that atom --
    ``validation.chemistry.check_species_labels`` is what reports it.
    """
    try:
        return resolve_element(label) == symbol
    except KeyError:
        return False


def atomic_number(label: str) -> int:
    """Z for a species label.  :func:`resolve_element`, then one lookup.

    The number every engine input needs beside the label the user chose.
    """
    from ase.data import atomic_numbers as _Z
    return int(_Z[resolve_element(label)])


#: Group VI (chalcogens) and group VII (halogens).  Hydrogen is written
#: BEFORE these and after everything else -- H2O, HF, HCl, H2S against
#: B2H6, SiH4, NH3 -- which is the electronegativity ordering chemical
#: formulae have always used.  `model/chemistry.md` § 3a.
_H_GOES_FIRST_WITH = frozenset(("O", "S", "Se", "Te", "Po",      # VI
                                "F", "Cl", "Br", "I", "At"))     # VII


def species_order(elements: "Iterable[str]",
                  override: "Optional[Iterable[str]]" = None) -> List[str]:
    """**THE ONE ENTRY** for "which species, in what order".

    Every emitter, every pseudopotential copy, every list of a structure's
    species asks THIS.  Nothing re-derives it: the order fixes each
    element's index in ``%block ChemicalSpeciesLabel``, that index fixes the
    orbital ordering inside ``.DM`` and ``.TSHS``, and two answers to one
    question is how files stop being readable by the next stage
    (`engines/transport.md` § 2a.13).

    *override* is the person's own answer -- the ``species_order`` catalogue
    row.  Non-empty, it is honoured VERBATIM and the rule below does not
    run; an override that omits a species the structure uses is refused
    HERE, naming it, before any deck exists -- a ``ValueError``.  Resolving
    the override here rather than at each call site is the point.

    THE DEFAULT, when nobody has said otherwise.

    THE ONE RULE, every engine (`model/chemistry.md` § 3a, decided
    2026-09-23).  The order fixes each element's index in
    ``%block ChemicalSpeciesLabel``, and that index fixes the orbital
    ordering inside SIESTA's ``.DM`` and ``.TSHS`` -- two runs that order
    species differently write files the next stage cannot read correctly
    (`engines/transport.md` § 2a.13).

    Start from atomic number, then place hydrogen the way a chemist writes
    it.  Three cases, in this order:

    1. **Carbon present** -> carbon first, then hydrogen, then the rest.
       This is the organic convention and it WINS outright: methanol is
       CH4O and acetic acid C2H4O2, never HC...
    2. **No carbon, but a group VI or VII element present** -> hydrogen
       FIRST, where its atomic number already puts it.  H2O, HF, HCl, H2S,
       and HNO3 -- hydrogen is the electropositive partner and is written
       first.
    3. **Neither** -> hydrogen goes LAST of the light elements, immediately
       after its anchor: nitrogen if present, else the lowest-Z species that
       is not hydrogen.  NH3, B2H6, SiH4.

    Ties inside the atomic-number sort keep first-seen order, so ``Au1`` and
    ``Au2`` stay distinct and stay as the file listed them.  Matching is on
    the ELEMENT a label names, never the label text, so ``H1``/``H2`` are
    both hydrogen and move together -- and they land after the LAST anchor
    label, because hydrogen belongs after the carbons as a group rather than
    between ``C1`` and ``C2``.

        {Au,C,H,S} -> C, H, S, Au        {H,O}   -> H, O
        {C,H,N,O}  -> C, H, N, O         {H,N,O} -> H, N, O
        {H,N}      -> N, H               {B,H}   -> B, H

    A person's ``species_order`` overrides this entirely; this is the
    default beneath it.
    """
    if override:
        order = list(override)
        missing = [s for s in dict.fromkeys(elements) if s not in order]
        if missing:
            raise ValueError(
                f"species_order does not list {', '.join(missing)}, which "
                f"this structure contains.  The order is honoured verbatim, "
                f"so every species has to appear in it -- or leave it unset "
                f"and the default rule orders them")
        return order
    seen: List[str] = []
    for s in elements:
        if s not in seen:
            seen.append(s)
    ordered = sorted(seen, key=atomic_number)

    hydrogens = [s for s in ordered if resolve_element(s) == "H"]
    if not hydrogens:
        return ordered
    symbols = {resolve_element(s) for s in ordered}

    if "C" in symbols:
        anchor = "C"                       # case 1 -- organic, and it wins
    elif symbols & _H_GOES_FIRST_WITH:
        return ordered                     # case 2 -- H already first by Z
    elif "N" in symbols:
        anchor = "N"                       # case 3 -- NH3
    else:
        rest = [s for s in ordered if s not in hydrogens]
        if not rest:
            return ordered                 # hydrogen alone
        anchor = resolve_element(rest[0])  # case 3 -- B2H6, SiH4

    rest = [s for s in ordered if s not in hydrogens]
    last = max(i for i, s in enumerate(rest) if resolve_element(s) == anchor)
    return rest[:last + 1] + hydrogens + rest[last + 1:]


def total_electrons(struct: Structure, charge: int = 0) -> int:
    """Sum of atomic numbers minus charge.  Used by parity checks
    (closed-shell spin=0 requires an even count; an odd total means
    spin must be at least 1).  Raises KeyError on a label that names no
    element (catches typos before PySCF does the same).
    """
    return sum(atomic_number(el) for el in struct.elements) - int(charge)


def every_label_resolves(struct: Structure) -> bool:
    """Can an electron count be taken at all?

    A check whose question needs Z asks this first and STANDS DOWN when
    the answer is no -- it does not report the bad label, because
    ``validation.chemistry.check_species_labels`` owns that finding and
    a second copy would be the same fact twice.  Standing down is not
    the same as passing: the count is unanswerable, so the check says
    nothing rather than saying nothing is wrong.

    Unlike the per-atom skip an ECP check can do, an electron count is
    over the WHOLE structure -- one unresolvable label leaves no partial
    answer -- so this is all-or-nothing by nature.
    """
    return all(names_an_element(str(el).strip()) for el in struct.elements)


def names_an_element(label: str) -> bool:
    """Does this species label name an element (:func:`resolve_element`)?"""
    try:
        resolve_element(label)
    except KeyError:
        return False
    return True


#: WHICH MASSES :func:`atomic_mass` GIVES, as a record says it -- a mode's
#: frame set states it beside the masses its coordinate is measured with
#: (`model/structure.md` § 2.2f, ``mass_source``).
MASS_SOURCE = "standard atomic weights"


def atomic_mass(element: str) -> float:
    """Standard atomic weight of ``element``, in amu.

    Reads through :func:`resolve_element`, so a species label carries:
    ``atomic_mass("Au1")`` is gold's weight, which is what a mass-weighted
    quantity over a labelled species needs.

    The same lookup :func:`total_electrons` does for atomic number, for
    the other number every engine needs: ASE ships the IUPAC standard
    atomic weights, so this is a name for a table rather than a copy of
    one.  Typing 118 masses into this file would be a second source of
    truth that no test would ever notice going stale.

    WHO NEEDS IT.  Any mass-weighted quantity: which atoms carry a
    vibrational mode (``vibration.motion_shares_of``, the readers' masses),
    reduced masses, centre of mass.  Mass is a chemistry fact and lives
    beside the other chemistry facts, not in the module that happens to
    want it first.

    NOT the isotope-resolved mass -- the standard weight is the natural
    isotopic average, which is what a Hessian is built with unless a run
    deliberately substitutes (deuteration), and PySCF's own default
    table is the same convention.  Raises KeyError on an unknown symbol,
    with the same message shape as :func:`total_electrons`.
    """
    from ase.data import atomic_masses as _M
    return float(_M[atomic_number(element)])


def check_spin_charge_parity(struct: Structure, charge: int,
                             unpaired: int) -> Optional[str]:
    """Return a human-readable error string when the (charge, 2S) pair is
    impossible for ``struct``, else None.

    Rule: 2S counts UNPAIRED electrons (= n_alpha - n_beta), so its parity
    must match the electron count's (ΣZ - charge): a closed shell (0) needs
    an even count, a doublet (1) an odd one.  PySCF raises ``RuntimeError
    ("Mol.nelectron N is odd, but spin = 0")`` at runtime on a mismatch;
    catching it before the deck is written gives a clearer message before
    anybody spends minutes on a doomed SCF.

    Engine-independent, and asked by exactly one caller: the electronic
    state's check (``validation/chemistry.check_electronic_state``), which
    applies it to a FINITE system only (ES3) -- a repeating cell's count per
    cell is not a spin.
    """
    # No type or sign check: the count arrives from the item's enum (0-10),
    # validated where the template is read -- a float or a negative 2S has no
    # way in, and a branch refusing one would be unreachable.
    # Parity IS an electron count, so it stands down on a label naming
    # no element rather than reporting it -- see `every_label_resolves`.
    if not every_label_resolves(struct):
        return None
    n_elec = total_electrons(struct, charge)
    # Electron-count sanity: over-ionised past the nucleus is impossible.
    if n_elec < 0:
        return (
            f"a charge of {charge:+d} removes more electrons than exist "
            f"(ΣZ - charge = {n_elec} < 0) -- the system has no electrons "
            f"left.  Reduce the positive charge."
        )
    # Upper bound (EXACT): 2S = n_alpha - n_beta with n_alpha + n_beta =
    # n_elec, so n_beta = (n_elec - 2S)/2 >= 0 requires 2S <= n_elec.
    # (Parity alone let 2S = 10 on H2 pass.)
    if unpaired > n_elec:
        return (
            f"the unpaired count {unpaired} exceeds the electron count "
            f"({n_elec}): use at most {n_elec}."
        )
    if (n_elec % 2) != (unpaired % 2):
        parity = "even" if n_elec % 2 == 0 else "odd"
        needs = "even" if unpaired % 2 == 0 else "odd"
        right = " or ".join(str(c) for c in (unpaired - 1, unpaired + 1)
                            if c >= 0)
        return (
            f"the electron count and the unpaired count disagree: the "
            f"electron count ({n_elec}) is {parity}, and the unpaired count "
            f"{unpaired} needs an {needs} one.  Change the charge by 1, or "
            f"the unpaired count to {right}."
        )
    return None


# Common spin-state -> (oxidation state, name) mapping for first-row
# transition metals.  Keyed by (element, spin = 2S).  Used to explain
# to the user what their (charge, spin) input
# IMPLIES about the metal centre.
#
# Source: standard ligand-field theory; ground-state d-electron counts
# from CRC handbook.  Entries are intentionally restricted to the
# COMMON oxidation states a user is likely to encounter in
# biological / coordination chemistry; rare ones (e.g. Fe(0), Fe(VI))
# are omitted to avoid noisy guesses.
_METAL_SPIN_HINTS: dict = {
    # Fe: d⁶ for Fe(II), d⁵ for Fe(III)
    ("Fe", 0): "Fe(II), low-spin (S=0, 0 unpaired) -- e.g. CO- or CN⁻-bound heme",
    ("Fe", 2): "Fe(II), intermediate-spin (S=1, 2 unpaired) -- e.g. four-coordinate porphyrins and phthalocyanines (FeTPP, FePc); uncommon elsewhere",
    ("Fe", 4): "Fe(II), high-spin (S=2, 4 unpaired) -- e.g. deoxy-heme, bis-thiolate",
    ("Fe", 1): "Fe(III), low-spin (S=1/2, 1 unpaired) -- e.g. bis-imidazole heme",
    ("Fe", 3): "Fe(III), intermediate-spin (S=3/2, 3 unpaired) -- e.g. quantum-admixed S=3/2 5-coord Fe(III) porphyrins (oxoferryl is Fe(IV), not this)",
    ("Fe", 5): "Fe(III), high-spin (S=5/2, 5 unpaired) -- e.g. met-myoglobin",
    # Mn: d⁵ for Mn(II), d⁴ for Mn(III)
    # (d⁵ is ODD -> minimum one unpaired electron; Mn(II) has NO 2S=0 state.
    #  Low-spin Mn(II) is S=1/2 = 2S=1, not 0.)
    ("Mn", 1): "Mn(II), low-spin (S=1/2, 1 unpaired) -- rare for Mn²⁺",
    ("Mn", 5): "Mn(II), high-spin (S=5/2, 5 unpaired) -- common for free Mn²⁺",
    ("Mn", 4): "Mn(III), high-spin (S=2, 4 unpaired)",
    # Co: d⁷ for Co(II), d⁶ for Co(III)
    ("Co", 1): "Co(II), low-spin (S=1/2, 1 unpaired)",
    ("Co", 3): "Co(II), high-spin (S=3/2, 3 unpaired)",
    ("Co", 0): "Co(III), low-spin (S=0, 0 unpaired) -- e.g. cobalamin/B12",
    # Cu: d⁹ for Cu(II), d¹⁰ for Cu(I)
    ("Cu", 1): "Cu(II) (S=1/2, 1 unpaired)",
    ("Cu", 0): "Cu(I), d¹⁰ closed-shell",
    # Ni: d⁸ for Ni(II)
    ("Ni", 0): "Ni(II), low-spin square-planar (S=0)",
    ("Ni", 2): "Ni(II), high-spin tetrahedral / octahedral (S=1, 2 unpaired)",
}


def explain_metal_spin(element: str, spin: int) -> Optional[str]:
    """Return a one-line description of what (element, spin) implies
    for a transition-metal centre (oxidation state + spin-state name).
    None if no hint is registered for this combination.
    """
    try:
        sym = resolve_element(element)
    except KeyError:
        return None                    # no hint for a label we cannot read
    return _METAL_SPIN_HINTS.get((sym, int(spin)))


# The full d-block (+ f-block) metal set for basis-adequacy checks: d-orbital
# coverage matters for CLOSED-shell metals (Zn/Cd/Hg d10, Pd/Pt) too, not only
# open-shell ones -- the concern is orbital coverage, orthogonal to spin state.
_ALL_TRANSITION_METALS = (OPEN_D_TRANSITION_METALS
                          | CLOSED_D10_METALS | NOBLE_METALS_S1)


def detect_transition_metals(struct: Structure) -> List[str]:
    """Every transition / f-block metal present (open- AND closed-shell), in
    first-appearance order.  For basis-adequacy: Zn/Cd/Hg/Pd/Pt need proper
    d/polarisation coverage although they carry no moment -- the concern is
    orbital coverage, orthogonal to the spin state."""
    seen: List[str] = []
    seen_set: set = set()
    for el in struct.elements:
        # The ELEMENT a label names -- so a deliberate ``Au1``/``Au2`` is
        # still gold here.  An unresolvable label is skipped, not folded:
        # ``validation.chemistry.check_species_labels`` is what reports it.
        try:
            key = resolve_element(el)
        except KeyError:
            continue
        if key in _ALL_TRANSITION_METALS and key not in seen_set:
            seen.append(key)
            seen_set.add(key)
    return seen


# --------------------------------------------------------------------- #
#  L2 — the structure's chemistry FACTS                                  #
#                                                                       #
#  What a structure is, chemically: its elements, its electron count,   #
#  the metals that bear on its spin and their usual spin states.  The   #
#  DECISION -- which charge, which spin -- is the electronic-state       #
#  class's (`molbuilder/electronic_state.py`,                            #
#  `science/chemistry-correctness.md` § 2a), which reads these facts at #
#  the calculation's own charge and periodicity.                        #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpinChoice:
    """One spin state a transition metal commonly takes.

    ``spin`` is 2S, the number of unpaired electrons -- the convention of the
    ``unpaired_electrons`` item (`chemistry-correctness.md` § 2a.1).
    """
    spin:  int
    label: str   # e.g. "Fe(II), intermediate (4-coord porphyrin)"


@dataclass(frozen=True)
class MetalHint:
    """The common spin states of one metal present.

    ``common_spins`` are ordered low-spin → high-spin so a UI can show
    a sweep from "most-paired" to "most-unpaired" without sorting.
    """
    element:      str
    common_spins: List[SpinChoice]


@dataclass(frozen=True)
class ChemistryAnalysis:
    """What a structure IS, chemically -- facts, not a decision.

    The electronic-state class (`electronic_state.py`) decides the charge and
    the spin from these, at the calculation's own charge and periodicity; the
    chemistry card shows the metal hints beside its answer.
    """
    n_atoms:              int
    elements:             List[str]      # unique element symbols, sorted
    #: The metals that bear on the spin, each list sorted: the open-d
    #: transition metals (``OPEN_D_TRANSITION_METALS``) and the noble s1
    #: metals (Cu, Ag, Au), split once here so no reader filters again.
    open_d_metals:        List[str]
    noble_metals:         List[str]
    #: Both, open-d first.  [] for organics.
    metals:               List[str]
    metal_hints:          List[MetalHint]


def _metal_hint(element: str) -> MetalHint:
    """Build a MetalHint by walking spin = 0..6 and collecting every
    spin for which ``explain_metal_spin`` registers a label."""
    spins: List[SpinChoice] = []
    for s in range(0, 7):
        label = explain_metal_spin(element, s)
        if label:
            spins.append(SpinChoice(spin=s, label=label))
    return MetalHint(element=element, common_spins=spins)


def count_element(struct: Structure, symbol: str) -> int:
    """Number of atoms of the element ``symbol`` in struct.

    Counts by ELEMENT, so two gold species labelled ``Au1``/``Au2`` are two
    gold atoms, not zero.
    """
    return sum(1 for el in struct.elements if is_atom(el, symbol))


def analyze_structure(struct: Structure) -> ChemistryAnalysis:
    """The chemistry facts of ``struct``.  Pure function — no I/O, no engine
    dependence, no global state.

    EVERY LABEL MUST RESOLVE: ``resolve_element`` raises ``KeyError`` on a
    label naming no element -- by design.  Every reader of these facts goes
    on to count electrons, and a count with an unreadable atom left out is
    not a smaller answer, it is a wrong one.

    The metals come from the structure's SORTED element list, never a
    frozenset's iteration order (which CPython randomises per process), so the
    metal that decides a multi-metal structure is the same on every run.
    """
    elements_sorted = sorted({resolve_element(el) for el in struct.elements})
    open_d = [m for m in elements_sorted if m in OPEN_D_TRANSITION_METALS]
    nobles = [m for m in elements_sorted if m in NOBLE_METALS_S1]
    metals = open_d + nobles
    return ChemistryAnalysis(
        n_atoms       = struct.n_atoms,
        elements      = elements_sorted,
        open_d_metals = open_d,
        noble_metals  = nobles,
        metals        = metals,
        metal_hints   = [_metal_hint(m) for m in metals],
    )


def expected_pH7_peptide_charge(struct: Structure) -> Optional[int]:
    """Estimate net peptide charge at physiological pH (7.4).

    Returns
    -------
    int    Estimated charge from charged side chains.  Asp/Glu
           contribute -1 each; Lys/Arg and the methylated lysines (MLZ,
           MLY, M3L) +1 each.  N- and
           C-termini cancel each other for a free peptide.  His /
           Cys / Tyr are intentionally skipped (His ambiguous; the
           others have pKa > 8).
    None   The structure doesn't look like a peptide (no recognised
           amino-acid residue names).  For nucleic acids use
           ``formal_charge_from_phosphates`` instead.

    Used by validators to surface the gap between the charge the
    calculation carries (a peptide's blank charge resolves to 0: the builder
    makes the gas-phase neutral form) and the physiological charge state the
    user often actually wants.  Never raises; never
    silently mutates the input.
    """
    # Collect one residue-name per residue id (atoms in the same
    # residue contribute to the count once, not per atom).
    seen: dict[int, str] = {}
    for rid, rname in zip(struct.residue_ids, struct.residue_names):
        if rid not in seen:
            seen[rid] = rname
    # Confirm we're looking at a peptide: at least one amino-acid residue
    # name -- a standard one, or a modified one the builder's grammar takes
    # (`residues.MODIFIED_RESIDUES`) -- must appear.
    from .residues import AA_THREE_TO_ONE, MODIFIED_RESIDUES
    aa_present = sum(1 for n in seen.values()
                     if n in AA_THREE_TO_ONE or n in MODIFIED_RESIDUES)
    if aa_present == 0:
        return None
    # Sum the side-chain contributions.
    charge = 0
    for name in seen.values():
        charge += _CHARGED_RESIDUES_PH7.get(name, 0)
    return charge


# Bond cutoffs used for proximity-based adjacency.  Wide enough to
# catch slightly-distorted bonds, narrow enough not to misclassify
# 1,3 contacts as bonds.
_HX_CUT = 1.30   # X-H bond cutoff
_XX_CUT = 1.95   # heavy-heavy bond cutoff


def _adjacency(elements: List[str], positions: np.ndarray
               ) -> Tuple[List[List[int]], List[List[int]]]:
    """Return per-atom heavy-neighbour and H-neighbour lists."""
    n = len(elements)
    nb_heavy: List[List[int]] = [[] for _ in range(n)]
    nb_h:     List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        ei = elements[i]
        for j in range(i + 1, n):
            ej = elements[j]
            d = float(np.linalg.norm(positions[i] - positions[j]))
            hi, hj = is_atom(ei, "H"), is_atom(ej, "H")
            cutoff = _HX_CUT if (hi or hj) else _XX_CUT
            if d > cutoff:
                continue
            if hj: nb_h[i].append(j)
            else:  nb_heavy[i].append(j)
            if hi: nb_h[j].append(i)
            else:  nb_heavy[j].append(i)
    return nb_heavy, nb_h


def formal_charge_from_phosphates(struct: Structure) -> int:
    """Estimate formal charge from phosphate protonation state.

    For each P atom:
        n_nb  = number of non-bridging O neighbours
        n_h   = number of those Os that already carry an H
        Of the n_nb non-bridging Os, one is the implicit P=O (no H);
        the remaining (n_nb - 1) should be -OH for a neutral phosphate.
        Missing Hs there each contribute -1 to the molecular charge:
            charge_contribution_per_P = -max(0, n_nb - 1 - n_h)

    Returns the sum across all P atoms.  Returns 0 for structures with
    no phosphates.
    """
    elements  = struct.elements
    positions = np.asarray(struct.positions, dtype=float)
    nb_heavy, nb_h = _adjacency(elements, positions)

    charge = 0
    for i, el in enumerate(elements):
        if not is_atom(el, "P"):
            continue
        # Non-bridging O = O whose only heavy neighbour is this P
        non_bridging = [
            j for j in nb_heavy[i]
            if is_atom(elements[j], "O") and len(nb_heavy[j]) == 1
        ]
        if len(non_bridging) < 1:
            continue
        n_nb = len(non_bridging)
        n_h  = sum(1 for j in non_bridging if len(nb_h[j]) > 0)
        # Implicit P=O on the first one; (n_nb - 1) should be OH
        missing = max(0, (n_nb - 1) - n_h)
        charge -= missing
    return charge


def van_der_waals_radius(label: str) -> Optional[float]:
    """The van der Waals radius of the element ``label`` names, in Angstrom:
    ASE's table (``ase.data.vdw_radii``: Bondi 1964 for most of the main
    group, Mantina et al. 2009 for the rest of it -- `Bondi1964`,
    `Mantina2009`), and for an element it holds no radius for -- the
    transition metals, the lanthanides -- RDKit's (its periodic table, from
    the Blue Obelisk Data Repository, as RDKit's source states) (the user,
    2026-11-18: "RDKit's radii for the gap").  Each element one source.
    ``None`` for a label that names no element."""
    from ase.data import atomic_numbers, vdw_radii
    try:
        z = atomic_numbers[resolve_element(label)]
    except KeyError:
        return None
    r = float(vdw_radii[z]) if z < len(vdw_radii) else math.nan
    if math.isnan(r):
        from rdkit import Chem
        r = float(Chem.GetPeriodicTable().GetRvdw(z))
    return r


def quaternary_ammonium(struct: Structure) -> List[int]:
    """The nitrogens bonded to four carbons -- each a QUATERNARY AMMONIUM,
    +1 whatever the pH (a trimethyl-lysine's NZ, a choline's N), which a
    structure, holding no charge of its own, carries as a neutral atom.
    Bonded by the phosphate rule's cutoff for two heavy atoms (``_XX_CUT``)
    and its element test (:func:`is_atom`), measured from each nitrogen only,
    so a large structure pays nothing for its other atoms.  A blank charge does not count them --
    the net charge is theirs with any other charged group the structure
    holds -- and the electronic state's checks warn of them
    (`model/chemistry.md` § 1)."""
    elements = struct.elements
    pos = np.asarray(struct.positions, dtype=float)
    heavy = np.array([not is_atom(el, "H") for el in elements], dtype=bool)
    found = []
    for n, el in enumerate(elements):
        if not is_atom(el, "N"):
            continue
        d = np.linalg.norm(pos - pos[n], axis=1)
        near = np.flatnonzero(heavy & (d <= _XX_CUT) & (d > 0.0))
        if len(near) == 4 and all(is_atom(elements[j], "C") for j in near):
            found.append(n)
    return found


def with_atoms(struct: Structure, added) -> Structure:
    """``struct`` with the atoms ``added`` -- each ``(i, element, position,
    name)``, an atom in the residue and chain of ``struct``'s atom ``i`` --
    each placed after the last atom of that residue, in the order given:
    what a builder adds to what it built (a hydroxyl's hydrogen, 3DNA's
    third phosphate oxygen), joined through the Structure's own doors
    (:meth:`Structure.concat`, :meth:`Structure.take`), which carry the
    structure's other facts."""
    if not added:
        return struct
    rid, chain = list(struct.residue_ids), list(struct.chain_ids)
    last = {(rid[k], chain[k]): k for k in range(struct.n_atoms)}
    after: Dict[int, List[int]] = {}
    for n, (i, _element, _position, _name) in enumerate(added):
        after.setdefault(last[(rid[i], chain[i])], []).append(
            struct.n_atoms + n)
    order: List[int] = []
    for k in range(struct.n_atoms):
        order += [k, *after.get(k, ())]
    return Structure.concat(
        [struct, *(Structure(elements=[element],
                             positions=np.asarray([position], dtype=float),
                             atom_names=[name], residue_ids=[rid[i]],
                             residue_names=[struct.residue_names[i]],
                             chain_ids=[chain[i]])
                   for i, element, position, name in added)],
        renumber_residues=False, title=struct.title).take(order)


def protonate_phosphate_oxygens(struct: Structure) -> Tuple[Structure, int]:
    """Neutralise the molecule by adding Hs to deprotonated phosphate Os.

    Per P atom: leaves one non-bridging O alone (the implicit P=O) and
    adds an H to every other non-bridging O that doesn't already have
    one.

    H placement: 0.96 A from O at 109.47 deg from the P-O direction,
    rotated in the plane formed by P-O and the "outward" direction
    (away from the centroid of the rest of the molecule).  This puts
    the new -OH cleanly on the outside of the structure, matching what
    a force-field optimisation would settle on.

    Returns (new_structure, n_hs_added): each H in its oxygen's residue,
    after its last atom (:func:`with_atoms`).  If no protonation
    is needed the original Structure is returned unchanged (n_hs_added = 0).
    """
    elements      = list(struct.elements)
    positions     = np.asarray(struct.positions, dtype=float)
    atom_names    = list(struct.atom_names)

    if not elements:
        return struct, 0

    nb_heavy, nb_h = _adjacency(elements, positions)

    BOND_OH = 0.96
    COS_SP3 = -1.0 / 3.0
    SIN_SP3 = math.sqrt(8.0) / 3.0    # = sin(109.47 deg)

    added: List[tuple] = []

    for p_idx, el in enumerate(elements):
        if not is_atom(el, "P"):
            continue
        non_bridging = [
            j for j in nb_heavy[p_idx]
            if is_atom(elements[j], "O") and len(nb_heavy[j]) == 1
        ]
        if len(non_bridging) < 2:
            continue   # nothing to do (need >= 1 P=O + >= 1 OH)
        # Choosing the implicit P=O:
        #   * An O that already carries an H must stay as -OH; protonating
        #     it would over-saturate the phosphate.
        #   * Among the bare Os we sort by atom name for deterministic
        #     output across runs.
        # So: take the bare-O list, sorted; the first is P=O, the rest
        # become -OH.  Os that already have H are never re-touched.
        bare = sorted([j for j in non_bridging if len(nb_h[j]) == 0],
                      key=lambda j: atom_names[j])
        if not bare:
            continue   # everything already protonated -- nothing to add
        # Reserve the first bare O as the implicit P=O and protonate the rest.
        targets = bare[1:]
        if not targets:
            continue
        for o_idx in targets:
            o_pos = positions[o_idx]
            p_pos = positions[p_idx]

            u_op = p_pos - o_pos
            n_op = float(np.linalg.norm(u_op))
            if n_op < 1e-9:
                continue
            u_op_norm = u_op / n_op

            # Outward direction = (this O) minus (centroid of P's other
            # heavy neighbours).  Project away from u_op to keep the
            # tilt-plane perpendicular to the P-O bond.
            other = [k for k in nb_heavy[p_idx] if k != o_idx]
            if other:
                centroid = positions[other].mean(axis=0)
                outward  = o_pos - centroid
                perp     = outward - np.dot(outward, u_op_norm) * u_op_norm
            else:
                perp = np.zeros(3)

            n_perp = float(np.linalg.norm(perp))
            if n_perp < 1e-6:
                # Pick any unit vector perpendicular to u_op_norm.
                seed = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(seed, u_op_norm)) > 0.9:
                    seed = np.array([0.0, 1.0, 0.0])
                perp = seed - np.dot(seed, u_op_norm) * u_op_norm
                n_perp = float(np.linalg.norm(perp))
            perp_norm = perp / n_perp

            h_pos = o_pos + BOND_OH * (COS_SP3 * u_op_norm + SIN_SP3 * perp_norm)

            # H name -- match common PDB conventions: HOP1, HOP2, ...
            added.append((o_idx, "H", h_pos, ("H" + atom_names[o_idx])[:4]))

    if not added:
        return struct, 0
    return with_atoms(struct, added), len(added)


# --------------------------------------------------------------------- #
#  Hydrogen addition: tool comparison + design rationale                #
#  ===================================================                  #
#                                                                       #
#  Single source of truth for adding explicit H atoms to a heavy-atom   #
#  skeleton.  Used by:                                                  #
#    - peptide.build_peptide        (PeptideBuilder emits heavy-only)   #
#    - nucleic.build_dna/build_rna  (X3DNA's `fiber` is heavy-only;     #
#                                    amber/rdkit produce H themselves   #
#                                    and skip this via the H/heavy>=0.5 #
#                                    gate in nucleic._maybe_add_hydrogens)
#                                                                       #
#  Why two engines, in this order                                       #
#  -------------------------------                                      #
#  Both OpenBabel and RDKit place H along correct sp3-tetrahedral /     #
#  sp2-planar / sp-linear vectors based on perceived bond orders.       #
#  They differ in how they handle ambiguous-valence sites (typically    #
#  exocyclic NH2 amines on nucleic acid bases and -NH3+ at peptide      #
#  N-termini):                                                          #
#                                                                       #
#  OpenBabel `OBMol.AddHydrogens()` (preferred):                        #
#    - Geometric H placement directly from each parent's hybridization  #
#      and existing neighbours.  No "give up" failure mode.             #
#    - The 20 amino acids by OpenBabel's residue templates; a           #
#      nucleobase's bond orders set first from its chemistry            #
#      (`residues.NUCLEOBASE_BONDS`): perceived from a 3DNA skeleton's  #
#      geometry, every cytosine came out imino (measured 2026-10-31).   #
#    - Doesn't reorder atoms, so PDB indices are preserved.             #
#    - Verified: X3DNA fiber heavy-skeleton -> AddHydrogens produces    #
#      the canonical 5 O-H + 37 C-H + 8 N-H breakdown for ATGC,         #
#      matching Amber-tleap and RDKit-via-SMILES exactly.               #
#                                                                       #
#  RDKit `Chem.AddHs(mol, addCoords=True)` (fallback):                  #
#    - Bond orders from RDKit's PDB residue templates: a nucleobase's   #
#      canonical orders (measured 2026-10-31 on a 3DNA skeleton).  The  #
#      fallback's limit is the ghost-H drop below -- cytosine N4 and    #
#      guanine N2 left one H each.  Nucleic acids want OpenBabel.       #
#    - BUT for sites where the heavy-atom geometry doesn't constrain    #
#      H placement uniquely -- exocyclic -NH2 on bases (A.N6, G.N2,     #
#      C.N4), peptide N-terminal -NH3+ -- the addCoords=True flag       #
#      sometimes leaves H atoms AT THEIR PARENT'S COORDINATES (zero    #
#      distance "ghost H").  This is a known RDKit limitation when     #
#      placing H from a heavy-atom-only PDB.                            #
#    - For DNA bases this strips Watson-Crick H-bond donors (4 H short  #
#      out of 50 on an ATGC chain pre-OpenBabel -- the bug that         #
#      motivated the fallback ordering).                                #
#    - SMILES-construct path doesn't have this issue; only PDB-parse    #
#      then AddHs has it.  The rdkit nucleic backend reaches the SMILES #
#      path; build_peptide and the X3DNA path land here.                #
#                                                                       #
#  Why not AmberTools `reduce`                                          #
#    - It's the gold standard for protein protonation (His tautomers,   #
#      Asn/Gln flips), but for DNA it's not better than OpenBabel and   #
#      requires shelling out + a temp-file round trip.  We already      #
#      have AmberTools as a transitive dep for the amber-tleap          #
#      backend; using `reduce` here wouldn't add a new dep but would    #
#      make this code path harder to reason about (subprocess vs        #
#      in-process).  Sticking with OpenBabel keeps the H-placement      #
#      logic uniform across peptide and nucleic flows.                  #
#                                                                       #
#  Why _drop_overlapping_hydrogens after each engine                    #
#    - Both engines (different reasons) can produce H at zero distance  #
#      from an anchor.  RDKit: addCoords ghost-H artifact above.        #
#      OpenBabel: rare, but multiple H written at the same coord for    #
#      tautomeric or ill-defined sites.  Keeping the post-pass means    #
#      the caller never sees the broken structure; downstream           #
#      validators don't have to special-case zero-distance pairs.       #
#    - Trade-off: for the RDKit path on nucleic acid bases, the drop    #
#      ALSO removes the legitimate-but-poorly-placed Watson-Crick H,    #
#      which is why OpenBabel is preferred-first.  Re-PLACING the       #
#      ghost H (rather than dropping) would be a smarter remediation    #
#      but is substantial new code and unnecessary as long as           #
#      OpenBabel is the primary engine.                                 #
# --------------------------------------------------------------------- #


def add_hydrogens(struct: Structure) -> Structure:
    """Add explicit H atoms to ``struct`` with correct sp3 / sp2 / sp
    geometry.

    Detection chain (first installed engine wins):
      1. OpenBabel ``OBMol.AddHydrogens()`` -- preferred.  Geometric H
         placement; doesn't fail on ambiguous-valence amine sites.
      2. RDKit ``Chem.AddHs(mol, addCoords=True)`` -- fallback.  Works
         well for SMILES-constructed molecules; for PDB-parsed inputs
         (heavy-atom only) it can leave exocyclic -NH2 H at parent
         coordinates.  See module-header comment for the full caveat.
      3. Neither -- or RDKit cannot read the structure, with OpenBabel
         absent: REFUSED, with ``BackendUnavailable`` -- the door every
         builder already uses for a dependency this install lacks, which the
         Build page answers as advice rather than a server fault.  A
         warning reaches no web user (`science/validation.md` § 4.1 R5), and
         a structure with no hydrogens gives DFT the wrong electron count.

    Both engines emit a final pass through ``_drop_overlapping_hydrogens``
    to strip any H that ended up sitting on another atom (the addCoords
    ghost-H artifact and rare OpenBabel duplicates).
    """
    # ---- try OpenBabel first (no ghost-coord failure mode) ----------
    try:
        from openbabel import openbabel as ob
    except ImportError:
        ob = None

    if ob is not None:
        return _protonate_openbabel(struct, ob)

    # ---- fall back to RDKit ------------------------------------------
    try:
        from rdkit import Chem
    except ImportError:
        Chem = None  # type: ignore

    if Chem is not None:
        return _protonate_rdkit(struct, Chem)

    raise BackendUnavailable(
        f"Cannot add hydrogens: neither {TOOLS['openbabel'].name} nor "
        f"{TOOLS['rdkit'].name} is installed -- each is one of the host "
        "env's packages -- and a "
        "structure without them would give DFT the wrong electron count.  "
        "Install OpenBabel for canonical "
        "biomolecule protonation; RDKit also works for SMILES-constructed "
        "inputs but has a known ambiguous-valence ghost-coord artifact for "
        "PDB-parsed nucleic-acid bases.", missing="hydrogens", env="host")


def _protonate_openbabel(struct: Structure, ob) -> Structure:
    """Geometric H placement via OBMol.AddHydrogens().

    OpenBabel's path: PDB -> OBMol (bond orders perceived from the
    geometry, a nucleobase's then set from its chemistry,
    :func:`_nucleobase_bond_orders`) -> AddHydrogens (geometric placement
    using sp3/sp2/sp vectors and existing neighbours) -> PDB.  Round-trip preserves
    atom order; placement is robust on ambiguous-valence amines that
    bite RDKit's PDB-then-AddHs path.
    """
    obconv = ob.OBConversion()
    obconv.SetInAndOutFormats("pdb", "pdb")
    mol = ob.OBMol()
    obconv.ReadString(mol, struct.to_pdb())
    _nucleobase_bond_orders(mol, ob)
    mol.AddHydrogens()
    out = obconv.WriteString(mol)
    return _drop_overlapping_hydrogens(
        Structure.from_pdb(out, title=struct.title)
    )


def _nucleobase_bond_orders(mol, ob) -> None:
    """Each nucleobase's bond orders from its chemistry
    (`residues.NUCLEOBASE_BONDS`), set on an OpenBabel molecule read from a
    PDB, and its atoms' implicit hydrogens counted again from them -- so
    ``AddHydrogens`` places the canonical tautomer's.  OpenBabel perceives
    the orders from the geometry, which a heavy-atom skeleton does not fix
    (the table's note); no other residue's bonds are touched, but a bond
    OpenBabel perceives from a base's atom to another residue is deleted."""
    from .residues import NUCLEOBASE_BONDS, nucleobase

    def base_atom(atom) -> bool:
        res = atom.GetResidue()
        base = nucleobase(res.GetName()) if res is not None else None
        return (base is not None
                and res.GetAtomID(atom).strip() in NUCLEOBASE_BONDS[base][0])

    # A BASE'S ATOMS BOND ONLY WITHIN ITS NUCLEOTIDE: a bond OpenBabel
    # perceives from one to another residue is two bases too close -- a
    # mismatched pair at the standard frame, its G's N1 1.17 A from its
    # A's N1 (measured 2026-11-18) -- and, kept, it takes the place of the
    # hydrogens there (AAAA,TTTG built 3 short, an odd electron count).
    false = [bond for bond in ob.OBMolBondIter(mol)
             if bond.GetBeginAtom().GetResidue() is not None
             and bond.GetEndAtom().GetResidue() is not None
             and (bond.GetBeginAtom().GetResidue().GetIdx()
                  != bond.GetEndAtom().GetResidue().GetIdx())
             and (base_atom(bond.GetBeginAtom())
                  or base_atom(bond.GetEndAtom()))]
    touched = set()
    for bond in false:
        touched.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        mol.DeleteBond(bond)
    # A deleted bond clears OpenBabel's "chains perceived" mark, and the next
    # residue asked would perceive them anew -- renaming every residue by its
    # own template (DA -> A) and renumbering the second strand.  The residues
    # are the file's, read as written.
    mol.SetChainsPerceived()
    for bond in ob.OBMolBondIter(mol):
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        res, other = a.GetResidue(), b.GetResidue()
        if res is None or other is None or res.GetIdx() != other.GetIdx():
            continue
        base = nucleobase(res.GetName())
        if base is None:
            continue
        atoms, doubles = NUCLEOBASE_BONDS[base]
        pair = (res.GetAtomID(a).strip(), res.GetAtomID(b).strip())
        if pair[0] not in atoms or pair[1] not in atoms:
            continue
        bond.SetBondOrder(2 if pair in doubles or pair[::-1] in doubles
                          else 1)
        bond.SetAromatic(False)
        touched.update((a.GetIdx(), b.GetIdx()))
    for idx in touched:
        atom = mol.GetAtom(idx)
        atom.SetAromatic(False)
        ob.OBAtomAssignTypicalImplicitHydrogens(atom)


def _protonate_rdkit(struct: Structure, Chem) -> Structure:
    """Fallback H placement via Chem.AddHs(mol, addCoords=True).

    Caveat (see module-header comment): for PDB-parsed inputs with
    ambiguous-valence sites (exocyclic -NH2 on bases, peptide
    N-terminal -NH3+), AddHs(addCoords=True) can leave H at parent
    coordinates.  ``_drop_overlapping_hydrogens`` removes those ghosts, and
    with them real hydrogens: a peptide's and a nucleic acid's count comes
    out short of OpenBabel's (measured 2026-11-04).  The environment ships
    OpenBabel, so the road we support does not land here.
    """
    mol = Chem.MolFromPDBBlock(struct.to_pdb(), removeHs=False, sanitize=False)
    if mol is None:
        # RDKit can choke on partial / unusual PDBs, and OpenBabel -- which
        # does not -- is not installed (it would have run first): no engine
        # here can place the hydrogens, the third case of `add_hydrogens`,
        # refused as it is.
        raise BackendUnavailable(
            "Cannot add hydrogens: RDKit could not read this structure to "
            "place them, and OpenBabel -- which can -- is not installed "
            f"({INSTALL_TOOLS}).  A structure "
            "without them would give DFT the wrong electron count.",
            missing="hydrogens", env="host")
    mol = Chem.AddHs(mol, addCoords=True)
    pdb_out = Chem.MolToPDBBlock(mol)
    return _drop_overlapping_hydrogens(
        Structure.from_pdb(pdb_out, title=struct.title)
    )


def _drop_overlapping_hydrogens(struct: Structure) -> Structure:
    """Remove H atoms that overlap (< 0.05 Å) with any other atom.

    Why 0.05 Å: a real X-H bond is always > 0.9 Å (the shortest
    physical X-H bond, H-F, is ~0.92 Å; C-H ~1.1 Å; N-H/O-H ~1.0 Å).
    A H within 0.05 Å of another atom is unambiguously a placement
    artifact -- typically a "ghost H" written at its parent atom's
    coordinates because the engine couldn't compute a real position.

    What this catches:
      * RDKit ``AddHs(addCoords=True)`` ghost H at ambiguous-valence
        sites (exocyclic -NH2 on nucleic-acid bases, peptide N-terminal
        -NH3+ extras).  The DEFINING failure mode of the RDKit fallback
        path; OpenBabel doesn't produce these.
      * Rare OpenBabel duplicates at tautomeric sites.

    What this does NOT do (and why):
      * Re-PLACE the ghost H at a sensible position.  That's the
        smarter remediation, but it requires hybridization perception
        (already in `_adjacency`) plus open-valence vector computation
        (new code).  Worth doing only if RDKit becomes the primary
        engine; with OpenBabel preferred, the drop is a safety net,
        not a load-bearing path.
      * Touch heavy atoms.  Only H-element atoms are candidates for
        removal; a heavy atom < 0.05 Å from another heavy atom is a
        broken structure that the validator should error on, not
        something we silently fix.

    Heavy atoms are never removed.

    H-H ghost pair handling: when two H atoms land at identical
    coordinates (rare, but possible for tautomer-ambiguous sites),
    a naive symmetric pass would mark BOTH as overlapping and drop
    them both -- removing real protons.  We track ``already_dropped``
    so once an H is flagged, it can't cause its peer to be flagged
    too.  Net effect on an H-H ghost pair: drop one, keep the other.
    """
    pos      = struct.positions
    elements = struct.elements
    n        = len(pos)
    keep     = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_atom(elements[i], "H") or not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if float(np.linalg.norm(pos[i] - pos[j])) < 0.05:
                keep[i] = False
                break
    if keep.all():
        return struct
    return struct.without(np.flatnonzero(~keep))


# --------------------------------------------------------------------- #
#  Steric-clash detection + relief                                      #
#                                                                        #
#  Building a duplex with a non-Watson-Crick base pair (a mismatch)     #
#  places the two bases at the standard WC frame, where they can        #
#  interpenetrate -- a topologically correct but physically unusable    #
#  starting structure (huge forces / SCF trouble if relaxed as-is).     #
#  `min_nonbonded_contact` DETECTS the overlap; `relieve_clashes`       #
#  optionally REMOVES it with a short force-field minimization.         #
# --------------------------------------------------------------------- #


def min_nonbonded_contact(struct: Structure, search_radius: float = 2.5):
    """Closest approach between atoms in DIFFERENT residues -- a steric-clash probe.

    Returns ``(distance, i, j)`` for the closest inter-residue atom pair within
    ``search_radius`` Angstrom, or ``(None, None, None)`` when the structure has no
    residue labels or no inter-residue pair inside the radius.

    Inter-residue only: within a residue, close contacts are covalent bonds (not
    clashes).  Between residues in DNA the only covalent link is the O3'-P
    backbone (~1.6 A), so any inter-residue pair well below that is an overlap --
    e.g. a mismatched base pair whose two bases interpenetrate.
    """
    P = np.asarray(struct.positions, dtype=float)
    if P.shape[0] < 2:
        return (None, None, None)
    keys = list(zip(struct.chain_ids, struct.residue_ids))
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    best = (None, None, None)
    for i, j in tree.query_pairs(r=search_radius):
        if keys[i] == keys[j]:
            continue
        d = float(np.linalg.norm(P[i] - P[j]))
        if best[0] is None or d < best[0]:
            best = (d, int(i), int(j))
    return best


# Two thresholds (Angstrom), keyed to what actually harms a downstream DFT/SIESTA
# relaxation -- not to reaching an ideal geometry (SIESTA does that):
#   CLASH_CONTACT_A  -- an inter-residue contact this close is a real steric
#     overlap (a mismatched base pair at the WC frame).  Below the tightest
#     legitimate canonical contact (~1.5 A adjacent-residue H-H) and a WC
#     H-bond (~1.8 A).
#   NEAR_COINCIDENT_A -- NEAR-COINCIDENT atoms (nuclei almost on top of each
#     other): THIS is what makes the first SCF step explode.  The opt-in relief
#     only needs to clear this line; a merely-tight contact above it, SIESTA
#     relaxes fine (user guidance 2026-07-14).
CLASH_CONTACT_A   = 1.3
NEAR_COINCIDENT_A = 1.0

#: THE O3'-P BRIDGE, one rule (`engines/builders.md` § 2): a nucleotide's
#: phosphorus is bonded to the O3' of the nucleotide before it in its chain
#: when within this distance (Angstrom) -- a P-O single bond is near 1.6 A,
#: and 1.80 leaves room for an unrelaxed starting geometry.
O3P_BRIDGE_A = 1.80

#: The 3'-oxygen's name, as the PDB spells it now and as older files did.
_O3_NAMES = ("O3'", "O3*")


@dataclass(frozen=True)
class BackboneLink:
    """One nucleotide's link to the one before it in its chain: the atom
    index of residue *r*'s P and of residue *r* - 1's O3', their distance,
    and whether they are bridged -- the one comparison with
    :data:`O3P_BRIDGE_A`, made here and nowhere else."""
    p: int
    o3: int
    distance: float

    @property
    def bridged(self) -> bool:
        return self.distance <= O3P_BRIDGE_A


def backbone_links(struct: Structure) -> "dict":
    """Each nucleotide's link to the one before it in its chain, ``(chain,
    r) -> BackboneLink``, for every NUCLEOTIDE residue *r*
    (`residues.nucleobase`) holding a P whose chain holds a nucleotide
    *r* - 1 with an O3'.  The one rule every reader of the backbone asks --
    the check of a polymer's orientation (`validation.geometry`), the self-
    check 3DNA's and AmberTools' builds run (`builders.backends._common`),
    clash relief's bond clean-up (:func:`relieve_clashes`) -- reading
    ``bridged``, never comparing a distance of its own; a phosphoserine's
    or any other residue's P is no backbone."""
    from .residues import nucleobase
    p_at: dict = {}
    o3_at: dict = {}
    for i in range(struct.n_atoms):
        if nucleobase(struct.residue_names[i]) is None:
            continue
        where = (struct.chain_ids[i], struct.residue_ids[i])
        if struct.atom_names[i] == "P":
            p_at[where] = i
        elif struct.atom_names[i] in _O3_NAMES:
            o3_at[where] = i
    return {(c, r): BackboneLink(
                p_at[c, r], o3_at[c, r - 1],
                float(np.linalg.norm(struct.positions[p_at[c, r]]
                                     - struct.positions[o3_at[c, r - 1]])))
            for (c, r) in p_at if (c, r - 1) in o3_at}


def steric_clash(struct: Structure) -> "Optional[Tuple[float, int, int]]":
    """The closest inter-residue contact (:func:`min_nonbonded_contact`) when
    it is a steric clash -- below :data:`CLASH_CONTACT_A` -- else ``None``:
    what the duplex builder relieves on request and the check after the build
    says (`validation.builders`)."""
    d, i, j = min_nonbonded_contact(struct)
    return None if d is None or d >= CLASH_CONTACT_A else (d, i, j)


def relieve_clashes(struct: Structure, steps: int = 1000) -> Structure:
    """Push apart steric overlaps with a short OpenBabel UFF minimization.

    Returns a new Structure (same atom order + count).  The GOAL is not an ideal
    geometry -- a subsequent DFT/SIESTA relaxation does that -- but to remove
    NEAR-COINCIDENT atoms that would otherwise make the first SCF step explode.

    Two robustness measures make this GENERAL (any sequence / mismatch), not tuned
    to one pair:

      1. **Bond cleanup.**  Our PDB carries no CONECT, so OpenBabel perceives bonds
         from geometry -- and perceives an OVERLAPPING pair as BONDED, which would
         pin it together and defeat the minimization.  We first DELETE every
         perceived inter-residue bond except the real O3'-P backbone link, so UFF
         sees the clashing atoms as non-bonded and repels them.  (This is also why
         the result is safe to hand downstream: no atoms left "wrongly bonded".)
      2. **Steepest-descent warm-up** then conjugate gradients: SD clears the hard
         short-range overlaps that CG alone can get stuck on.

    Raises :class:`BackendUnavailable` if OpenBabel isn't installed, and
    RuntimeError if its force field won't set up -- the caller (build_dna)
    surfaces either rather than silently no-op'ing.
    """
    ob = require("openbabel", needed_for="Clash relief")

    conv = ob.OBConversion()
    conv.SetInAndOutFormats("pdb", "pdb")
    mol = ob.OBMol()
    conv.ReadString(mol, struct.to_pdb())

    def _res_key(atom):
        r = atom.GetResidue()
        return (r.GetChain(), r.GetNum()) if r else None

    # Drop perceived inter-residue bonds except the O3'-P backbone links --
    # the bridges THE ONE RULE names (:func:`backbone_links`), by atom; the
    # mis-perceived clash "bond" is what pins overlapping atoms together.
    # OpenBabel numbers the atoms of the PDB it read from 1, in its order.
    bridges = {frozenset((link.p, link.o3))
               for link in backbone_links(struct).values() if link.bridged}
    doomed = []
    for bond in ob.OBMolBondIter(mol):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if (_res_key(a1) != _res_key(a2)
                and frozenset((a1.GetIdx() - 1, a2.GetIdx() - 1))
                not in bridges):
            doomed.append(bond)
    for bond in doomed:
        mol.DeleteBond(bond)

    ff = ob.OBForceField.FindForceField("UFF")
    if ff is None or not ff.Setup(mol):
        raise RuntimeError(
            "OpenBabel UFF force field could not be set up for clash relief.")
    ff.SteepestDescent(max(1, steps // 2))   # clear hard short-range overlaps
    ff.ConjugateGradients(steps)             # then settle
    ff.GetCoordinates(mol)
    # THE ATOMS MOVE AND NOTHING ELSE: same order, same count, every name,
    # residue and chain the build gave them.
    return struct.replace(positions=np.array(
        [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(mol)]))


# --------------------------------------------------------------------- #
#  Pauling electronegativity table + heuristic partial-charge estimate #
#                                                                        #
#  This is a heuristic, not a QM result.  Used by the validation pass  #
#  to estimate molecular dipoles for the "polar molecule in vacuum"    #
#  warning -- the goal is "is this molecule meaningfully polar?",     #
#  not a research-grade dipole.                                        #
# --------------------------------------------------------------------- #


# Pauling electronegativities for elements common in molbuilder targets.
# Source: standard chemistry references (e.g. Cotton & Wilkinson).
# Elements not in the table fall back to 2.20 (carbon-ish) -- the
# heuristic is forgiving about exact values; what matters is that
# polar bonds (>0.4 difference) tilt charge in the right direction.
_PAULING_EN = {
    "H":  2.20, "Li": 0.98, "Be": 1.57, "B":  2.04, "C":  2.55,
    "N":  3.04, "O":  3.44, "F":  3.98, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P":  2.19, "S":  2.58, "Cl": 3.16,
    "K":  0.82, "Ca": 1.00, "Br": 2.96, "I":  2.66,
}
_DEFAULT_EN = 2.20


def estimate_partial_charges(struct: Structure,
                             total_charge: float = 0.0,
                             *,
                             bond_cutoff: float = 1.95,
                             hx_cutoff:   float = 1.30) -> np.ndarray:
    """Heuristic per-atom partial charges from electronegativity gaps.

    For each bonded pair (heavy-heavy if d < bond_cutoff Å, X-H if
    d < hx_cutoff Å), Pauling's ionic-character formula gives the
    fractional charge transfer:

        ionic_fraction = 1 - exp(-0.25 * (Δχ)²)

    where Δχ is the Pauling EN difference.  The more electronegative
    atom receives a partial charge of -ionic_fraction; its partner
    receives +ionic_fraction.  Each per-bond shift is then capped at
    ±0.5 e to avoid outsize numbers on extreme pairs (e.g. F-Na).

    The result is shifted uniformly so that ``sum(q) == total_charge``,
    absorbing rounding error from any missed bonds at the edges.

    Cross-checks (Pauling formula; agreement within ~10% of reality):
        H2O    -> 1.8 D   (vs 1.85 D experimental)
        HF     -> 2.4 D   (vs 1.83 D)
        N2     -> 0.0 D   (vs 0)
        CO2    -> 0.0 D   (vs 0)
        CH3OH  -> 1.5 D   (vs 1.69 D)

    Not a substitute for QM partial charges.  Used by validation
    for the "polar molecule in vacuum" dipole warning, where the
    question is "is the dipole 0.5 D or 5 D?", not precise-to-decimal.
    """
    n         = struct.n_atoms
    elements  = list(struct.elements)
    positions = struct.positions
    q         = np.zeros(n, dtype=float)

    for i in range(n):
        ei   = elements[i]
        en_i = _PAULING_EN.get(ei, _DEFAULT_EN)
        for j in range(i + 1, n):
            ej = elements[j]
            d  = float(np.linalg.norm(positions[i] - positions[j]))
            if "H" in (ei, ej):
                if d > hx_cutoff:
                    continue
            elif d > bond_cutoff:
                continue
            en_j      = _PAULING_EN.get(ej, _DEFAULT_EN)
            delta_en  = en_i - en_j     # positive when i is more EN
            ionic     = 1.0 - math.exp(-0.25 * delta_en * delta_en)
            # Sign: more-EN atom gets negative.  ionic is always >= 0;
            # apply the sign of delta_en so i (more EN) ends up
            # negative when delta_en > 0.
            shift     = ionic if delta_en > 0 else -ionic
            shift     = max(-0.5, min(0.5, shift))
            q[i] -= shift          # more-EN atom: more negative
            q[j] += shift          # less-EN atom: more positive

    excess = q.sum() - total_charge
    if n > 0:
        q -= excess / n
    return q


def estimate_dipole_moment_debye(struct: Structure,
                                 total_charge: float = 0.0) -> float:
    """Magnitude of the heuristic molecular dipole, in Debye.

    Uses :func:`estimate_partial_charges` and computes
    ``|sum(q_i * (r_i - r_com))|`` with positions in Å and charges in
    elementary units, converting via `constants.DEBYE_E_ANGSTROM`.

    ABOUT THE CENTRE OF MASS (plan § 5q D8): an ion's dipole depends on the
    origin (`science/normal-modes.md` § 4a.5), and a deck hands the validators
    the molecule placed in its box, half a box from where it was drawn -- so
    ``sum(q_i * r_i)`` measured the box, not the molecule (formate, charge
    -1 in a 10 Å vacuum: 88.8 D placed, 1.4 D as drawn).  For a neutral
    molecule the origin cancels and this is the same number.
    """
    if struct.n_atoms == 0:
        return 0.0
    q = estimate_partial_charges(struct, total_charge)
    pos = struct.positions.astype(float)
    m = np.array([atomic_mass(e) for e in struct.elements], dtype=float)
    com = (m[:, None] * pos).sum(axis=0) / m.sum()
    p = (q[:, None] * (pos - com)).sum(axis=0)   # e·Å
    from .constants import DEBYE_E_ANGSTROM
    return float(np.linalg.norm(p) * DEBYE_E_ANGSTROM)   # -> Debye

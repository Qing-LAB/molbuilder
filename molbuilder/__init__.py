"""molbuilder -- build 3-D molecules from sequences / SMILES / names.

Public API:

    >>> import molbuilder
    >>> s = molbuilder.build_peptide("ARNDC")               # 1-letter
    >>> s = molbuilder.build_peptide("AR[SEP]C")            # phospho-Ser
    >>> s = molbuilder.build_dna("ATGCATGCAT")
    >>> s = molbuilder.build_rna("AUGCAUGCAU")
    >>> s = molbuilder.build_from_smiles("Sc1ccc(S)cc1")    # 1,4-BDT
    >>> s = molbuilder.build_from_name("benzene")           # PubChem lookup

    # Load existing geometry from disk (auto-detects format):
    >>> from molbuilder.workingcopy_structure import StructureCodec
    >>> s = StructureCodec().load("structure.xyz")   # the pair

    # Save geometry to disk -- always the PAIR, never a lone file:
    >>> StructureCodec().write(s, "out.xyz")   # + out.molstruct.json
    >>> print(s.to_xyz())                      # the TEXT, for a deck
    >>> print(s.to_pyscf(as_string=True))
    >>> atoms = s.to_ase()

    # SIESTA input file:
    >>> from molbuilder.siesta import SiestaConfig, render_fdf
    >>> print(render_fdf(s, SiestaConfig(system_label="bdt", kgrid=(1,1,1))))

    # Browser UI for interactive building + SIESTA input generation:
    $ molbuilder serve
"""

from pathlib import Path
from typing import Union

from .structure import Structure
from .peptide import build_peptide
from .nucleic import build_dna, build_rna

__version__ = "1.1.0"

__all__ = [
    "Structure",
    "build_peptide",
    "build_dna",
    "build_rna",
    "build_from_smiles",
    "build_from_name",
    "__version__",
]


# --------------------------------------------------------------------- #
#  repo_root() -- where this checkout is                                #
# --------------------------------------------------------------------- #


def repo_root() -> Path:
    """The directory that CONTAINS the ``molbuilder`` package.

    **Architecture rule A11: one home per root.**  The alternative is each
    caller climbing its own parent chain to this same place --
    ``references.py`` for ``docs/science/references.bib``,
    ``web/blueprints/docs.py`` for ``docs/``, ``runwrap.py`` and
    ``script_emit.py`` for the checkout a generated script must activate
    against, ``builders/backends/_threedna.py`` for the ``x3dna*/`` unpack
    directory (`ops/installation.md` § "Option A").  Every spelling of that
    climb carries a level count, and the count differs by where the file sits:
    ``_threedna`` is four levels down where the others are two.  A count is a
    fact about a file's depth in the tree, and it is wrong the moment the file
    moves.

    **Why the package's own module answers this.**  Only ``molbuilder`` knows
    where ``molbuilder`` is, and a caller deriving it from its own ``__file__``
    is reading this package's self-knowledge from outside.  Asking here means
    one answer, and it stays right when a caller is moved to a different
    depth.

    **What it is, precisely:** ``Path(__file__).resolve().parent.parent`` --
    for the supported deployment (a source checkout, run in place) that is the
    checkout root, the directory holding ``pyproject.toml``, ``docs/`` and any
    unpacked ``x3dna*/``.  It is not a search: nothing is probed and nothing
    falls back, so a caller that needs a file under it checks for that file.

    **Callers inside the import chain must import it lazily.**  ``__init__``
    imports ``structure`` -> ... -> ``builders.backends._threedna`` and
    ``projects``, so a module-level ``from molbuilder import repo_root`` in
    any of those is a cycle.  Import it inside the function instead; modules
    outside the chain (``references.py``) may import it at module level.
    """
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- #
#  load() -- read existing XYZ or PDB into a Structure                  #
# --------------------------------------------------------------------- #


# THERE IS NO `load()` HERE, and that is the rule, not an omission.  Reading a
# geometry file without the `.molstruct.json` beside it hands the caller a
# structure quietly smaller than what is on disk -- no regions, no frozen
# atoms, no cell.
#
# THE DOOR IS `StructureCodec`:
#
#     from molbuilder.workingcopy_structure import StructureCodec
#     s = StructureCodec().load("structure.xyz")
#
# which reads the pair, applies the sidecar through the one applier, and
# refuses an extension it does not know.  A second name for it is what lets
# the geometry and its sidecar drift apart.

# --------------------------------------------------------------------- #
#  Optional builders -- imported lazily so users without RDKit /        #
#  PubChemPy don't pay the import cost.                                 #
# --------------------------------------------------------------------- #


def build_from_smiles(smiles: str, **kwargs):
    """Build a Structure from a SMILES string (RDKit + ETKDG + MMFF)."""
    from .smiles import build_from_smiles as _impl
    return _impl(smiles, **kwargs)


def build_from_name(name: str, **kwargs):
    """Build a Structure from a common/IUPAC name (PubChem lookup + RDKit)."""
    from .pubchem import build_from_name as _impl
    return _impl(name, **kwargs)

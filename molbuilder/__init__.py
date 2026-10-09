"""molbuilder -- build 3-D molecules from sequences / SMILES / names.

Public API -- each builder from its own module:

    >>> from molbuilder.peptide import build_peptide
    >>> s = build_peptide("ARNDC")                          # 1-letter
    >>> s = build_peptide("AR[SEP]C")                       # phospho-Ser
    >>> from molbuilder.nucleic import build_dna, build_rna
    >>> s = build_dna("ATGCATGCAT")
    >>> s = build_rna("AUGCAUGCAU")
    >>> from molbuilder.smiles import build_from_smiles
    >>> s = build_from_smiles("Sc1ccc(S)cc1")               # 1,4-BDT
    >>> from molbuilder.pubchem import build_from_name
    >>> s = build_from_name("benzene")                      # PubChem lookup

    # Load existing geometry from disk (auto-detects format):
    >>> from molbuilder.workingcopy_structure import StructureCodec
    >>> s = StructureCodec().load("structure.xyz")   # the pair

    # Save geometry to disk -- always the PAIR, never a lone file:
    >>> StructureCodec().write(s, "out.xyz")   # + out.molstruct.json
    >>> print(s.to_xyz())                      # the TEXT, for a deck
    >>> print(s.to_pyscf(as_string=True))
    >>> atoms = s.to_ase()

    # A SIESTA or PySCF deck is written by `molbuilder jobset prep`, from a
    # calculation's description (docs/execution/running-a-job.md).

    # Browser UI for interactive building:
    $ molbuilder serve
"""

from pathlib import Path
from typing import Union

from .structure import Structure

__version__ = "1.1.0"

#: THE BUILDERS ARE NOT RE-EXPORTED HERE.  Python loads this module before
#: any module inside the package, so whatever it imports loads under every
#: one of them, and the domain layer would load under every core module
#: (`architecture.md` § 3).  Each caller imports the builder it uses from its
#: own module.
__all__ = [
    "Structure",
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
    ``web/blueprints/docs.py`` for ``docs/``, ``script_emit.py`` for the
    checkout whose git revision a generated deck records, ``projects.py``
    for the default projects root, ``builders/backends/_threedna.py`` for
    the ``x3dna*/`` unpack directory (`ops/installation.md` § "Option A").  Every spelling of that
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
    imports ``structure`` and nothing else, so a module-level
    ``from molbuilder import repo_root`` in a module ``structure`` imports
    would be a cycle.  Import it inside the function there; a module outside
    that chain (``references.py``) may import it at module level.
    """
    return Path(__file__).resolve().parent.parent


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

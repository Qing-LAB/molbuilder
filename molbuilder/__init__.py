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
    >>> s = molbuilder.load("structure.xyz")
    >>> s = molbuilder.load("structure.pdb")

    >>> s.to_xyz("out.xyz")
    >>> s.to_pdb("out.pdb")
    >>> print(s.to_pyscf(as_string=True))
    >>> atoms = s.to_ase()

    # SIESTA input file:
    >>> from molbuilder.siesta import Config, render_fdf
    >>> print(render_fdf(s, Config(system_label="bdt", kgrid=(1,1,1))))

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
    "load",
    "__version__",
]


# --------------------------------------------------------------------- #
#  repo_root() -- where this checkout is                                #
# --------------------------------------------------------------------- #


def repo_root() -> Path:
    """The directory that CONTAINS the ``molbuilder`` package.

    **Architecture rule A11: one home per root.**  Five modules used to climb
    a parent chain to this same place -- ``references.py`` for
    ``docs/science/references.bib``, ``web/blueprints/docs.py`` for ``docs/``,
    ``runwrap.py`` and ``script_emit.py`` for the checkout a generated script
    must activate against, and ``builders/backends/_threedna.py`` (twice) for
    the ``x3dna*/`` unpack directory (`ops/installation.md` § "Option A").
    Each spelled the climb itself, and ``_threedna`` had to count four levels
    instead of two because of where it sits.  A count is a fact about a file's
    depth in the tree, and it is wrong the moment the file moves.

    **Why the package's own module answers this.**  Only ``molbuilder`` knows
    where ``molbuilder`` is; every one of those five derived it from a
    ``__file__``, which is this package's self-knowledge read from outside.
    Asking here means one answer, and it stays right when a caller is moved to
    a different depth.

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


def load(path: Union[str, Path], *, format: str = "auto",
         title: str = None) -> Structure:
    """Load a 3-D structure from a file on disk.

    Parameters
    ----------
    path
        Path to the file.  ``.xyz`` and ``.pdb`` extensions are
        recognised; pass ``format`` explicitly to override.
    format
        ``"auto"`` (default) -- inferred from the path extension.
        ``"xyz"`` or ``"pdb"`` to force.
    title
        Optional override for the structure's title.

    Returns
    -------
    :class:`molbuilder.structure.Structure`

    Examples
    --------
    >>> s = molbuilder.load("polymer.pdb")
    >>> s.n_atoms
    87
    >>> from molbuilder.siesta import Config, render_fdf
    >>> print(render_fdf(s, Config(system_label="loaded", kgrid=(1, 1, 1))))
    """
    p = Path(path)
    fmt = format.lower()
    if fmt == "auto":
        ext = p.suffix.lower().lstrip(".")
        if ext in ("xyz", "pdb"):
            fmt = ext
        else:
            raise ValueError(
                f"can't infer format from extension {p.suffix!r}; "
                f"pass format='xyz' or format='pdb' explicitly"
            )
    if fmt == "xyz":
        return Structure.from_xyz(p, title=title)
    if fmt == "pdb":
        return Structure.from_pdb(p, title=title)
    raise ValueError(f"unknown format {format!r}; expected 'xyz' or 'pdb'")


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

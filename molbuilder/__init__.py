"""molbuilder -- build 3-D molecules from sequences / SMILES / names.

Public API:

    >>> import molbuilder
    >>> s = molbuilder.build_peptide("ARNDC")               # 1-letter
    >>> s = molbuilder.build_peptide("AR[SEP]C")            # phospho-Ser
    >>> s = molbuilder.build_dna("ATGCATGCAT")
    >>> s = molbuilder.build_rna("AUGCAUGCAU")
    >>> s = molbuilder.build_from_smiles("Sc1ccc(S)cc1")    # 1,4-BDT
    >>> s = molbuilder.build_from_name("benzene")           # PubChem lookup

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

from .structure import Structure
from .peptide import build_peptide
from .nucleic import build_dna, build_rna

__version__ = "0.2.0"

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

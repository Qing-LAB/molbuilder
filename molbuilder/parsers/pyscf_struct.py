"""Structure-side PySCF readers: ``<JOB>_optimized.xyz`` (final
converged coords from a molbuilder-generated PySCF run) and ``.py``
initial coordinates from ``mol = gto.M(atom = '''…''')``.

Used by :mod:`molbuilder.script_bundle` to recover the final
structure for handoff into the next workflow stage.  Separate from
:mod:`molbuilder.parsers.pyscf` (which parses pyscf logs for the
trajectory inspector) because the consumers are different:

* ``<JOB>_optimized.xyz`` is a plain XYZ written by
  ``molbuilder/pyscf/input.py::_emit_optimized_save`` on every
  successful geometry-opt completion.  Plain XYZ -> reuse
  :func:`Structure.from_xyz`.
* ``.py`` initial coords come from the ``atom = '''…'''`` block
  inside ``mol = gto.M(...)``.  Only the molbuilder generator
  emit-format is supported (whitespace-delimited rows of
  ``element x y z``); hand-written PySCF scripts may use a
  different format (e.g. ``[["H", (0,0,0)], ...]``) that this
  reader doesn't accept.

Bundle-contract reference: ``docs/protocols/bundle-contract.md § 4.1``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Errors                                                               #
# --------------------------------------------------------------------- #


class PyscfStructureError(ValueError):
    """Raised when a ``.py`` lacks the gto.M atom block, or the block
    can't be parsed back to a Structure."""


# --------------------------------------------------------------------- #
#  JOB literal extraction                                               #
# --------------------------------------------------------------------- #


# Match the canonical line the generator emits:
#     JOB = "<label>"
# (input.py § 326).  We accept single OR double quotes since the
# generator could conceivably switch; keeps the parser tolerant.
_JOB_RE = re.compile(
    r"""^\s*JOB\s*=\s*(['"])(?P<label>[^'"]+)\1\s*$""",
    re.MULTILINE,
)


def extract_pyscf_job(text: str) -> Optional[str]:
    """Return the ``JOB`` literal from a molbuilder-generated PySCF
    script, or ``None`` when no recognisable ``JOB = "..."`` line is
    present.

    Used by the bundle layer to derive ``<JOB>_optimized.xyz`` as
    the preferred final-coords source.
    """
    m = _JOB_RE.search(text)
    return m.group("label") if m else None


# --------------------------------------------------------------------- #
#  .py initial-coords reader                                            #
# --------------------------------------------------------------------- #


# Match the molbuilder atom-block emit format (input.py §§ 381-384):
#     mol = gto.M(
#         atom = '''
#         <El>  x  y  z
#         ...
#         ''',
#         basis = "...",
#         ...
#     )
#
# We only need the inner block between the triple quotes.  The
# whitespace and the assignment style are determined by
# ``_atoms_block`` (input.py § 59) which writes lines of
# ``"{indent}{el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}"``.
_ATOM_BLOCK_RE = re.compile(
    r"atom\s*=\s*'''(?P<block>.*?)'''",
    re.DOTALL,
)


def read_py_initial_coords(text_or_path: Union[str, Path]) -> Structure:
    """Parse the ``atom = '''…'''`` block of a molbuilder-generated
    PySCF script and return a :class:`Structure`.

    Used as a last-resort fallback when ``<JOB>_optimized.xyz`` is
    missing.  Coords come back in Å (the generator emits Å per
    ``unit = 'Ang'`` in the gto.M call).

    Caller is responsible for recording ``RunBundle.final_coords_from
    = "py-initial"`` so audit consumers can tell this isn't the
    converged geometry.

    Only the molbuilder-generated emit format is recognised:
    whitespace-delimited rows of ``element x y z``.  Hand-written
    PySCF scripts using list-of-tuple or other formats will raise
    :class:`PyscfStructureError` and the bundle layer will surface
    it as an actionable error.
    """
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = str(text_or_path)

    m = _ATOM_BLOCK_RE.search(text)
    if m is None:
        raise PyscfStructureError(
            "no triple-quoted ``atom = '''…'''`` block in the .py.  "
            "Only molbuilder-generated PySCF scripts are supported "
            "for bundle assembly; hand-written scripts using list-of-"
            "tuple atom specs need to be re-rendered through "
            "molbuilder first."
        )
    elements: List[str] = []
    positions: List[List[float]] = []
    for raw in m.group("block").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            x = float(toks[1])
            y = float(toks[2])
            z = float(toks[3])
        except ValueError:
            continue
        # Canonicalise the element symbol the same way Structure.from_xyz
        # does: "FE" or "fe" become "Fe" so downstream consumers
        # (validation, ase) all see the canonical form.
        elements.append(toks[0].capitalize())
        positions.append([x, y, z])

    if not elements:
        raise PyscfStructureError(
            "the atom block in the .py is empty after parsing; the "
            "rows may not be whitespace-delimited (`El x y z`) as "
            "the molbuilder generator emits them."
        )
    return Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
    )


# --------------------------------------------------------------------- #
#  .opt-xyz reader (thin wrapper over Structure.from_xyz)               #
# --------------------------------------------------------------------- #


def read_optimized_xyz(path: Union[str, Path]) -> Structure:
    """Read a ``<JOB>_optimized.xyz`` file and return a Structure.

    Thin wrapper over :func:`Structure.from_xyz` so the bundle
    layer has a single entry point for "final coords from the PySCF
    run".  Errors surface as :class:`ValueError` from from_xyz.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    s = Structure.from_xyz(text)
    # XYZ comment line typically reads "Optimized geometry (PySCF)" --
    # mirror the file stem onto Structure.title so downstream
    # re-render can label it.
    if not s.title:
        s.title = p.stem
    return s


__all__ = [
    "PyscfStructureError",
    "extract_pyscf_job",
    "read_optimized_xyz",
    "read_py_initial_coords",
]

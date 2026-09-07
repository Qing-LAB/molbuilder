"""L1 tests for ``molbuilder.parse.coords.pyscf_geom``.

Pins the ``<JOB>_optimized.xyz`` reader and the ``.py`` initial-coords
parser.  See ``docs/execution/job-contracts.md`` for source
priority + § 5 for the API surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.parse.coords.pyscf_geom import (
    read_optimized_xyz,
)


# --------------------------------------------------------------------- #
#  JOB literal extraction                                               #
# --------------------------------------------------------------------- #










# --------------------------------------------------------------------- #
#  .py initial-coords reader                                            #
# --------------------------------------------------------------------- #


def _h2_py(*, with_atom_block: bool = True) -> str:
    """Minimal molbuilder-style PySCF script with the canonical
    triple-quoted atom block."""
    if with_atom_block:
        atom = (
            "mol = gto.M(\n"
            "    atom = '''\n"
            "    H    0.00000000    0.00000000    0.00000000\n"
            "    H    0.74000000    0.00000000    0.00000000\n"
            "    ''',\n"
            "    basis = 'def2-SVP',\n"
            ")\n"
        )
    else:
        atom = "# no gto.M call here\n"
    return (
        'JOB = "h2-test"\n'
        "import pyscf\n"
        "from pyscf import gto\n"
        + atom
    )












# --------------------------------------------------------------------- #
#  <JOB>_optimized.xyz reader                                           #
# --------------------------------------------------------------------- #


def test_read_optimized_xyz_round_trips_basic_xyz(tmp_path):
    p = tmp_path / "h2-test_optimized.xyz"
    p.write_text(
        "2\nOptimized geometry (PySCF)\n"
        "H 0.0 0.0 0.0\n"
        "H 0.74 0.0 0.0\n"
    )
    s = read_optimized_xyz(p)
    assert s.elements == ["H", "H"]
    np.testing.assert_allclose(s.positions[1], [0.74, 0.0, 0.0])
    # When the XYZ comment has content, Structure.from_xyz uses it as
    # the title; only fall back to the stem when the comment is empty.
    assert "Optimized geometry" in s.title


def test_read_optimized_xyz_assigns_stem_title_when_xyz_comment_blank(tmp_path):
    p = tmp_path / "no-comment_optimized.xyz"
    p.write_text("1\n\nH 0 0 0\n")
    s = read_optimized_xyz(p)
    assert s.title == "no-comment_optimized"

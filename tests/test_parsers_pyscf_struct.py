"""L1 tests for molbuilder.parsers.pyscf_struct.

Pins the ``<JOB>_optimized.xyz`` reader and the ``.py`` initial-coords
parser.  See ``docs/execution/job-contracts.md`` for source
priority + § 5 for the API surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.parse.coords.pyscf_geom import (
    PyscfStructureError,
    extract_pyscf_job,
    read_optimized_xyz,
    read_py_initial_coords,
)


# --------------------------------------------------------------------- #
#  JOB literal extraction                                               #
# --------------------------------------------------------------------- #


def test_extract_pyscf_job_finds_double_quoted():
    assert extract_pyscf_job('JOB = "pyscf-stage1"\n') == "pyscf-stage1"


def test_extract_pyscf_job_finds_single_quoted():
    assert extract_pyscf_job("JOB = 'pyscf-stage1'\n") == "pyscf-stage1"


def test_extract_pyscf_job_returns_none_when_missing():
    assert extract_pyscf_job("# no JOB literal here\n") is None


def test_extract_pyscf_job_ignores_quoted_string_in_comment():
    """Comment-line JOB= should be ignored — match anchors to ^."""
    text = '# JOB = "this is just a comment"\nJOB = "real"\n'
    assert extract_pyscf_job(text) == "real"


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


def test_read_py_initial_coords_parses_minimal_h2():
    s = read_py_initial_coords(_h2_py())
    assert s.elements == ["H", "H"]
    np.testing.assert_allclose(s.positions[1], [0.74, 0.0, 0.0])


def test_read_py_initial_coords_canonicalises_element_case():
    """Lower-case ``fe`` -> ``Fe`` so downstream (ase, validation)
    sees the canonical symbol."""
    text = (
        "mol = gto.M(\n"
        "    atom = '''\n"
        "    fe   0.0 0.0 0.0\n"
        "    fe   2.5 0.0 0.0\n"
        "    ''',\n"
        ")\n"
    )
    s = read_py_initial_coords(text)
    assert s.elements == ["Fe", "Fe"]


def test_read_py_initial_coords_raises_when_no_block():
    with pytest.raises(PyscfStructureError) as exc:
        read_py_initial_coords(_h2_py(with_atom_block=False))
    assert "triple-quoted" in str(exc.value)


def test_read_py_initial_coords_raises_when_block_is_empty():
    text = "mol = gto.M(\n    atom = '''\n    \n    ''',\n)\n"
    with pytest.raises(PyscfStructureError) as exc:
        read_py_initial_coords(text)
    assert "empty" in str(exc.value)


def test_read_py_initial_coords_accepts_path(tmp_path):
    from pathlib import Path
    p = tmp_path / "h2.py"
    p.write_text(_h2_py())
    s = read_py_initial_coords(p)
    assert s.elements == ["H", "H"]


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

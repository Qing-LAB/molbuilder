"""Structure + sidecar codec (``StructureCodec``) — load, edit, write, scratch
round-trip of the ``.xyz`` + ``.molstruct.json`` pair.

The codec is the LIVE remnant of the retired working-copy stack: it is used by
``/api/structure/resolve-cell`` (build.py) to rebuild a Structure from a scratch
blob.  These tests exercise its public surface directly (``load`` / ``files`` /
``scratch_blob`` / ``from_scratch``) — no ``WorkingCopy`` wrapper (that class +
the ``/api/workingcopy/*`` door were removed).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from molbuilder.workingcopy_structure import StructureCodec
from molbuilder.structure import Structure, AtomChannel

CODEC = StructureCodec()


@pytest.fixture
def project(tmp_path):
    s = Structure(elements=["H", "C", "N", "O", "F"],
                  positions=np.array([[float(i), 0.0, 0.0] for i in range(5)]),
                  cell=np.diag([40.0, 40.0, 40.0]), pbc=[True, True, True])
    (tmp_path / "mol.xyz").write_text(s.to_xyz())
    return tmp_path


def _write(files):
    for path, blob in files:
        Path(path).write_bytes(blob)


def test_codec_writes_xyz_and_sidecar(project):
    s = CODEC.load(project / "mol.xyz")
    assert s.n_atoms == 5
    s.frozen_atoms = [1]
    s.regions = {"bridge": [2, 3]}
    s.set_channel("charge", AtomChannel("value", {0: -1.0}))
    assert not (project / "mol.molstruct.json").exists()   # not written yet
    _write(CODEC.files(s, project / "mol.xyz"))
    sidecar = json.loads((project / "mol.molstruct.json").read_text())
    assert sidecar["frozen_atoms"] == [1]
    assert sidecar["regions"] == {"bridge": [2, 3]}
    assert sidecar["annotations"]["charge"]["kind"] == "value"


def test_labels_roundtrip_through_load(project):
    """``files`` writes the pair; a fresh ``load`` reads the ``.xyz`` AND applies
    the companion ``.molstruct.json`` sidecar."""
    s = CODEC.load(project / "mol.xyz")
    s.regions = {"L-electrode": [0, 1]}
    _write(CODEC.files(s, project / "mol.xyz"))
    s2 = CODEC.load(project / "mol.xyz")           # re-reads .xyz + sidecar
    assert s2.regions == {"L-electrode": [0, 1]}


def test_scratch_blob_roundtrip(project):
    """``scratch_blob`` -> ``from_scratch`` restores structure + labels +
    annotations (the exact round-trip ``/api/structure/resolve-cell`` relies on)."""
    s = CODEC.load(project / "mol.xyz")
    s.frozen_atoms = [4]
    s.set_channel("spin", AtomChannel("value", {2: 0.5}))
    blob = CODEC.scratch_blob(s)
    s2 = CODEC.from_scratch(blob)
    assert s2.n_atoms == 5
    assert s2.frozen_atoms == [4]
    assert s2.get_channel("spin").data == {2: 0.5}


def test_load_parses_a_pdb_source_not_as_xyz(tmp_path):
    """The file picker accepts .pdb, so the codec must parse a .pdb SOURCE via
    from_pdb.  Regression: load() called from_xyz unconditionally, so a .pdb's
    "HEADER ..." first line raised "must be an integer atom count"."""
    pdb = tmp_path / "water.pdb"
    pdb.write_text(
        "HEADER    WATER\n"
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      2  H   HOH A   1       0.757   0.586   0.300  1.00  0.00           H\n"
        "ATOM      3  H   HOH A   1      -0.757   0.586  -0.300  1.00  0.00           H\n"
        "END\n")
    struct = CODEC.load(pdb)
    assert struct.n_atoms == 3
    assert struct.elements == ["O", "H", "H"]


def test_load_rejects_unknown_format_explicitly(tmp_path):
    """An unknown extension is an EXPLICIT error, never a silent from_xyz
    attempt that fails with a confusing 'atom count' message."""
    weird = tmp_path / "mol.cif"
    weird.write_text("data_x\n_cell_length_a 10.0\n")
    with pytest.raises(ValueError, match="unsupported structure format"):
        CODEC.load(weird)

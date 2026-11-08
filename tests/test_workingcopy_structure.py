"""Structure + sidecar codec (``StructureCodec``) — load, edit, write, read of
the ``.xyz`` + ``.molstruct.json`` pair.

The codec is the one place a Structure becomes bytes for every surface that
writes one: ``write`` backs ``/api/structure/save``, ``files`` backs
``/api/structure/export``, and ``read`` backs ``/api/build/load``.  These tests
exercise that surface directly (``load`` / ``pair`` / ``files`` / ``write`` /
``read``).
"""
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
                  cell=np.diag([40.0, 40.0, 40.0]))
    (tmp_path / "mol.xyz").write_text(s.to_xyz())
    return tmp_path


def _write(files):
    for path, blob in files:
        Path(path).write_bytes(blob)


def test_labels_roundtrip_through_load(project):
    """``files`` writes the pair; a fresh ``load`` reads the ``.xyz`` AND applies
    the companion ``.molstruct.json`` sidecar."""
    s = CODEC.load(project / "mol.xyz")
    s.regions = {"L-electrode": [0, 1]}
    _write(CODEC.files(s, project / "mol.xyz"))
    s2 = CODEC.load(project / "mol.xyz")           # re-reads .xyz + sidecar
    assert s2.regions == {"L-electrode": [0, 1]}


def test_write_then_read_restores_structure_labels_and_annotations(project):
    """Out through the file door and back, with everything the sidecar carries
    still attached."""
    s = CODEC.load(project / "mol.xyz")
    s.frozen_atoms = [4]
    s.set_channel("spin", AtomChannel("value", {2: 0.5}))
    CODEC.write(s, project / "mol.xyz")
    s2 = CODEC.load(project / "mol.xyz")
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


def test_it_writes_utf8_regardless_of_the_platform_locale(tmp_path):
    """An explicit encoding, never the platform's, or a non-ASCII title is
    silently corrupted on cp1252 -- and the codec's own `load` reads
    `utf-8-sig`, so a locale-encoded write would not even round-trip here.
    """
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    s = Structure.from_xyz("3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
    s.title = "wasser — Ångström"
    path = tmp_path / "t.xyz"
    StructureCodec().write(s.with_frames([s.positions, s.positions]), path)

    assert "Ångström" in path.read_text(encoding="utf-8")
    assert StructureCodec().load(path).title == "wasser — Ångström"


def test_a_sidecar_that_cannot_be_written_leaves_the_pair_as_it_was(tmp_path):
    """`model/structure.md` § 2.4, both or neither: the codec renders both
    halves before it writes either, so a sidecar its serialiser refuses -- a
    NaN recorded in `info` (a divergent run's energy), which `set_info`
    takes and no JSON file holds -- stops the write with the pair on disk
    byte for byte as it was, and no temp file beside it.  The silent
    failure (plan V1.10): the new geometry put in place, then the sidecar
    refused, the new coordinates left beside the old sidecar."""
    s = Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
    target = tmp_path / "h2.xyz"
    CODEC.write(s, target)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    moved = s.replace(positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]]))
    moved.set_info("relaxation", {"energy_ev": float("nan")})
    with pytest.raises(ValueError):
        CODEC.write(moved, target)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_a_half_that_cannot_be_staged_leaves_the_pair_as_it_was(
        tmp_path, monkeypatch):
    """`model/structure.md` § 2.4, both or neither, the disk's half: both are
    staged before either lands, so a full disk or a quota at the second leaves
    the old pair byte for byte, and neither temp file beside it."""
    import molbuilder.workingcopy_structure as W
    s = Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
    target = tmp_path / "h2.xyz"
    CODEC.write(s, target)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    stage = W.stage_bytes

    def _second_fails(path, data, **kw):
        if path != target:
            raise OSError("no space left on device")
        return stage(path, data, **kw)

    monkeypatch.setattr(W, "stage_bytes", _second_fails)
    moved = s.replace(positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]]))
    with pytest.raises(OSError):
        CODEC.write(moved, target)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

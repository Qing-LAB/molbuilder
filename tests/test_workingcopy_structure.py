"""Structure codec — load, edit, save the .xyz + .molstruct.json pair."""
import json

import numpy as np
import pytest

from molbuilder import workingcopy as wc
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


def _open(project, session="s1"):
    return wc.WorkingCopy.open(project / "mol.xyz", CODEC,
                               session=session, project_dir=project)


def test_open_edit_save_writes_xyz_and_sidecar(project):
    w = _open(project)
    s = w.data
    assert s.n_atoms == 5
    s.frozen_atoms = [1]
    s.regions = {"bridge": [2, 3]}
    s.set_channel("charge", AtomChannel("value", {0: -1.0}))
    w.update(s)
    assert not (project / "mol.molstruct.json").exists()   # not saved yet
    w.save(project / "mol.xyz")
    sidecar = json.loads((project / "mol.molstruct.json").read_text())
    assert sidecar["frozen_atoms"] == [1]
    assert sidecar["regions"] == {"bridge": [2, 3]}
    assert sidecar["annotations"]["charge"]["kind"] == "value"


def test_save_as_and_labels_roundtrip(project):
    w = _open(project)
    s = w.data
    s.regions = {"L-electrode": [0, 1]}
    w.update(s)
    w.save(project / "mol.xyz")
    w2 = _open(project, session="s2")           # a fresh open reads them back
    assert w2.data.regions == {"L-electrode": [0, 1]}


def test_reload_restores_structure_and_annotations(project):
    w = _open(project)
    s = w.data
    s.frozen_atoms = [4]
    s.set_channel("spin", AtomChannel("value", {2: 0.5}))
    w.update(s)
    orphans = wc.list_orphans(project, live_sessions=[])
    w2 = wc.WorkingCopy.recover(orphans[0], CODEC, project_dir=project)
    assert w2.data.n_atoms == 5
    assert w2.data.frozen_atoms == [4]
    assert w2.data.get_channel("spin").data == {2: 0.5}

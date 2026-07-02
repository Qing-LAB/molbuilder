"""Structure codec (molbuilder/workingcopy_structure.py) driven through the
working-copy core — the real .xyz + .molstruct.json application (Phase 2)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from molbuilder import workingcopy as wc
from molbuilder.workingcopy_structure import StructureCodec
from molbuilder.structure import Structure, AtomChannel
from molbuilder.sidecars import molstruct

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


def test_open_edit_commit_writes_xyz_and_sidecar_with_matching_hash(project):
    w = _open(project)
    struct = w.data
    assert struct.n_atoms == 5
    # Edit: freeze atom 1, add a region + a value channel.
    struct.frozen_atoms = [1]
    struct.regions = {"bridge": [2, 3]}
    struct.set_channel("charge", AtomChannel("value", {0: -1.0}))
    w.update(struct)
    assert w._scratch_file().exists()
    assert not (project / "mol.molstruct.json").exists()   # not durable yet

    r = w.commit(project / "mol.xyz")
    assert r.ok
    # Both durable files written.
    xyz_bytes = (project / "mol.xyz").read_bytes()
    sidecar = json.loads((project / "mol.molstruct.json").read_text())
    # Sidecar carries the edits + annotations.
    assert sidecar["frozen_atoms"] == [1]
    assert sidecar["regions"] == {"bridge": [2, 3]}
    assert sidecar["annotations"]["charge"]["kind"] == "value"
    # ATOM-IDENTITY INVARIANT: the sidecar's structure_hash == sha256(the .xyz).
    assert sidecar["structure_hash"] == hashlib.sha256(xyz_bytes).hexdigest()
    assert not w._scratch_file().exists()


def test_reload_from_scratch_restores_structure_and_annotations(project):
    w = _open(project)
    s = w.data
    s.frozen_atoms = [4]
    s.set_channel("spin", AtomChannel("value", {2: 0.5}))
    w.update(s)
    # Recover from scratch (simulated reload/crash).
    orphans = wc.list_orphans(project, live_sessions=[])
    assert len(orphans) == 1
    w2 = wc.WorkingCopy.recover(orphans[0], CODEC, project_dir=project)
    assert w2.data.n_atoms == 5
    assert w2.data.frozen_atoms == [4]
    assert w2.data.get_channel("spin").data == {2: 0.5}


def test_commit_refuses_when_xyz_changed_underneath(project):
    w = _open(project)
    s = w.data
    s.frozen_atoms = [0]
    w.update(s)
    # The .xyz is edited by another tool (atoms reordered) -> hash changes.
    reordered = Structure(elements=["F", "O", "N", "C", "H"],
                          positions=np.array([[float(i), 1.0, 0.0] for i in range(5)]),
                          cell=np.diag([40.0, 40.0, 40.0]), pbc=[True, True, True])
    (project / "mol.xyz").write_text(reordered.to_xyz())
    r = w.commit(project / "mol.xyz")
    assert not r.ok and r.reason == "mismatch"
    # No sidecar written (no stale-index laundering).
    assert not (project / "mol.molstruct.json").exists()


def test_roundtrip_sidecar_applies_back_via_load(project):
    # Commit once, then a fresh open() must read the labels back.
    w = _open(project)
    s = w.data
    s.regions = {"L-electrode": [0, 1]}
    w.update(s)
    assert w.commit(project / "mol.xyz").ok
    w2 = _open(project, session="s2")
    assert w2.data.regions == {"L-electrode": [0, 1]}

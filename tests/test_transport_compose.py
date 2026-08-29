"""P4a — prep composes from the citation (`plans/transport-design.md`
§ 4.1–4.2, `transport/compose.py`).

Properties under guard, each named for its failure:

* the happy path: citation → relaxed geometry FROM THE .XV (never a
  file copy), sorted canonical, electrode models extracted from the
  sorted blocks, provenance with content hashes;
* strict composition (ruling Q2): a missing attempt names the commands
  to run first; an unconcluded attempt refuses without deciding;
* frozen means unmoved (ruling Q3): a drifted electrode atom is
  refused naming the atom, its label, and the distance;
* the .XV must describe the cited source (element mismatch refused);
* prep refuses a transport task with the honest under-construction
  message instead of crashing on the structure it does not carry.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.transport import (REGION_BRIDGE,
                                         REGION_LEFT_ELECTRODE,
                                         REGION_RIGHT_ELECTRODE)
from molbuilder.structure import Structure
from molbuilder.transport.compose import (FROZEN_TOL_ANG, ComposeError,
                                          compose_junction, read_xv,
                                          write_compose_record)

_ANG_BOHR = 1.0 / 0.529177
_Z = {"Au": 79, "S": 16, "C": 6, "N": 7}

#: six 2.5 Å layers a side (12.5 Å span — over the wizard's 12 Å lead
#: floor), the molecule between.
_LAYERS_L = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
_BRIDGE = [("S", 15.0), ("C", 16.4), ("C", 17.8), ("S", 19.2)]
_LAYERS_R = [22.0, 24.5, 27.0, 29.5, 32.0, 34.5]


def _junction_struct():
    elements, zs, labels = [], [], []
    for z in _LAYERS_L:
        elements.append("Au"); zs.append(z)
        labels.append(REGION_LEFT_ELECTRODE)
    for el, z in _BRIDGE:
        elements.append(el); zs.append(z); labels.append(REGION_BRIDGE)
    for z in _LAYERS_R:
        elements.append("Au"); zs.append(z)
        labels.append(REGION_RIGHT_ELECTRODE)
    positions = np.array([[1.0, 1.0, z] for z in zs])
    regions: dict = {}
    for i, lab in enumerate(labels):
        regions.setdefault(lab, []).append(i)
    frozen = [i for i, lab in enumerate(labels) if lab != REGION_BRIDGE]
    return Structure(elements=elements, positions=positions,
                     regions=regions, frozen_atoms=frozen,
                     cell=np.diag([8.0, 8.0, 40.0]))


def _write_xv(path, struct, *, perturb_bridge=0.3, perturb_electrode=None):
    """A SIESTA-shaped .XV: cell rows then per-atom rows, in Bohr.
    Bridge atoms shift by ``perturb_bridge`` A (the 'relaxation');
    ``perturb_electrode=(index, dist)`` breaks one frozen atom."""
    pos = np.asarray(struct.positions, dtype=float).copy()
    for i in struct.regions.get(REGION_BRIDGE, ()):
        pos[i, 0] += perturb_bridge
    if perturb_electrode:
        i, d = perturb_electrode
        pos[i, 2] += d
    lines = []
    for row in np.asarray(struct.cell):
        v = " ".join(f"{x * _ANG_BOHR:.9f}" for x in row)
        lines.append(f"  {v}  0.0 0.0 0.0")
    lines.append(f"  {len(struct.elements)}")
    for i, el in enumerate(struct.elements):
        xyz = " ".join(f"{x * _ANG_BOHR:.9f}" for x in pos[i])
        lines.append(f"  1  {_Z[el]}  {xyz}  0.0 0.0 0.0")
    path.write_text("\n".join(lines) + "\n")
    return pos


@pytest.fixture
def tree(tmp_path):
    """A projects tree holding one concluded junction relaxation."""
    from molbuilder.task import (Stage, StructureRef, Task, derive_run,
                                 write_task)
    from molbuilder.workingcopy_structure import StructureCodec

    root = tmp_path / "projects"
    calc = root / "J" / "optimization" / "Relax"
    attempt = calc / "01_coarse" / "run-0"
    attempt.mkdir(parents=True)

    struct = _junction_struct()
    StructureCodec().write(struct, calc / "j.source.xyz")
    formula = "C2S2Au12"
    task = Task(engine="siesta", shape="hierarchical",
                run=derive_run("Relax", formula, stage_names=("coarse",)),
                structure=StructureRef(source="j.source.xyz",
                                       formula=formula,
                                       atoms=len(struct.elements)),
                varies=(), stages=(Stage(name="coarse", enabled=True,
                                         overrides={}),))
    write_task(calc / "task.json", task)

    (attempt / "Relax_01_coarse.fdf").write_text(
        "SystemLabel Relax\nMeshCutoff 300.0 Ry\nXC.functional GGA\n"
        "XC.authors PBE\nPAO.BasisSize DZP\n")
    (attempt / "Relax_01_coarse-run0.concluded").write_text("rc=0\n")
    relaxed_pos = _write_xv(attempt / "Relax.XV", struct)
    return root, struct, relaxed_pos


_CITE = "J/optimization/Relax@01_coarse/run-0"


class TestHappyPath:

    def test_composes_sorted_with_the_relaxed_geometry(self, tree):
        root, src, relaxed_pos = tree
        out = compose_junction(_CITE, tree_root=root)
        dev = out.sorted.structure
        n = len(src.elements)
        # canonical layout: 6 Au, 4 bridge, 6 Au
        assert dev.elements[:6] == ["Au"] * 6
        assert dev.elements[6:10] == ["S", "C", "C", "S"]
        assert dev.elements[10:] == ["Au"] * 6
        # the geometry is the .XV's (bridge x shifted by 0.3), in A
        assert dev.positions[6, 0] == pytest.approx(1.3, abs=1e-6), (
            "the relaxed positions must come from the .XV, parsed -- "
            "not the source's")
        assert np.allclose(np.asarray(dev.cell),
                           np.diag([8.0, 8.0, 40.0]), atol=1e-6)
        # electrode models extracted from the sorted blocks
        assert len(out.electrode_left.elements) == 6
        assert len(out.electrode_right.elements) == 6
        # the fdf snapshot is the attempt's own deck
        assert out.fdf_params.mesh_cutoff_ry == pytest.approx(300.0)
        assert "PBE" in out.fdf_params.xc
        # provenance carries the hashes
        assert out.provenance["citation"] == _CITE
        assert len(out.provenance["xv_sha256"]) == 64
        assert out.provenance["concluded"].startswith("rc=")

    def test_the_record_writes_both_sidecars(self, tree, tmp_path):
        root, _, _ = tree
        out = compose_junction(_CITE, tree_root=root)
        dest = tmp_path / "transport-calc"
        dest.mkdir()
        names = write_compose_record(dest, out)
        assert sorted(names) == ["atom-permutation.json",
                                 "slot-provenance.json"]
        import json
        perm = json.loads((dest / "atom-permutation.json").read_text())
        assert perm["schema"] == "molbuilder/atom-permutation@1"
        assert sorted(perm["sorted_to_original"]) == list(range(16))

    def test_read_xv_round_trips_units(self, tree):
        root, src, relaxed_pos = tree
        cell, elements, pos = read_xv(
            root / "J/optimization/Relax/01_coarse/run-0/Relax.XV")
        assert elements == list(src.elements)
        assert np.allclose(pos, relaxed_pos, atol=1e-6)
        assert np.allclose(cell, np.diag([8.0, 8.0, 40.0]), atol=1e-6)


class TestStrictComposition:

    def test_a_missing_attempt_names_the_commands_to_run(self, tree):
        root, _, _ = tree
        with pytest.raises(ComposeError) as e:
            compose_junction("J/optimization/Relax@01_coarse/run-9",
                             tree_root=root)
        msg = str(e.value)
        assert "jobset prep run" in msg and "jobset launch run" in msg, (
            "strict composition refuses by naming what to run FIRST")

    def test_an_unconcluded_attempt_refuses_without_deciding(self, tree):
        root, _, _ = tree
        (root / "J/optimization/Relax/01_coarse/run-0/"
                "Relax_01_coarse-run0.concluded").unlink()
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        msg = str(e.value)
        assert "CONCLUDED" in msg
        assert "still running" in msg and "force-stopped" in msg, (
            "the two states are indistinguishable on disk -- the refusal "
            "must name both, never decide")

    def test_a_missing_calculation_is_refused(self, tree):
        root, _, _ = tree
        with pytest.raises(ComposeError) as e:
            compose_junction("Nope/optimization/Gone@run-0",
                             tree_root=root)
        assert "task.json" in str(e.value)


class TestTheGates:

    def test_a_moved_frozen_atom_is_refused_by_name(self, tree):
        root, src, _ = tree
        _write_xv(root / "J/optimization/Relax/01_coarse/run-0/Relax.XV",
                  src, perturb_electrode=(0, 0.05))
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        msg = str(e.value)
        assert "atom 0 (Au" in msg and "MOVED" in msg
        assert REGION_LEFT_ELECTRODE in msg

    def test_arithmetic_dust_is_not_a_move(self, tree):
        root, src, _ = tree
        _write_xv(root / "J/optimization/Relax/01_coarse/run-0/Relax.XV",
                  src, perturb_electrode=(0, FROZEN_TOL_ANG / 10))
        compose_junction(_CITE, tree_root=root)     # must not raise

    def test_an_xv_of_a_different_structure_is_refused(self, tree):
        root, src, _ = tree
        other = _junction_struct()
        other.elements[7] = "N"                     # a bridge C -> N
        _write_xv(root / "J/optimization/Relax/01_coarse/run-0/Relax.XV",
                  other)
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        assert "elements differ" in str(e.value)


def test_prep_refuses_transport_honestly(tmp_path, monkeypatch):
    from molbuilder.jobset.prep import PrepError, prep_calculation
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.task import Stage, Task, derive_run, write_task
    root = tmp_path / "projects"
    (root / "J" / "optimization" / "Relax").mkdir(parents=True)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
    dest = root / "J" / "transport" / "T"
    dest.mkdir(parents=True)
    stages = tuple(Stage(name=n, enabled=True, overrides={})
                   for n in ("seed", "electrode_L", "electrode_R",
                             "device", "transmission"))
    write_task(dest / "task.json", Task(
        engine="siesta", shape="hierarchical",
        run=derive_run("T", _CITE,
                       stage_names=tuple(s.name for s in stages)),
        structure=None, calculation="transport",
        slots={"junction": _CITE}, varies=(), stages=stages))
    with pytest.raises(PrepError) as e:
        prep_calculation(dest)
    assert "P4b" in str(e.value), (
        "prep must refuse the composite honestly, not crash on the "
        "structure it deliberately does not carry")

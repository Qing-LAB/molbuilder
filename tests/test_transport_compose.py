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
* the § 3 lead gates: a thin block (under the principal-layer floor)
  and a block that does not tile are refused naming the numbers;
* the .XV must describe the cited source (element mismatch refused);
* the composed record travels: written whole (sorted pair + the
  attempt's own deck + sidecars), loaded back without the cited tree,
  and a re-pointed or incomplete record reads as NO record.

(The prep arm itself — stage decks, wrappers, run dirs — is
`test_transport_prep.py`'s subject.)
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


def _junction_struct(layers_l=_LAYERS_L, layers_r=_LAYERS_R):
    elements, zs, labels = [], [], []
    for z in layers_l:
        elements.append("Au"); zs.append(z)
        labels.append(REGION_LEFT_ELECTRODE)
    for el, z in _BRIDGE:
        elements.append(el); zs.append(z); labels.append(REGION_BRIDGE)
    for z in layers_r:
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


def _write_tree(root, struct):
    """One concluded junction relaxation under *root* — the shared
    scaffold, parameterised so the gate tests can cite structures the
    happy path never builds (thin blocks, broken tiling)."""
    from molbuilder.task import (Stage, StructureRef, Task, derive_run,
                                 write_task)
    from molbuilder.workingcopy_structure import StructureCodec

    calc = root / "J" / "optimization" / "Relax"
    attempt = calc / "01_coarse" / "run-0"
    attempt.mkdir(parents=True)

    StructureCodec().write(struct, calc / "j.source.xyz")
    task = Task(engine="siesta", shape="hierarchical",
                run=derive_run("Relax", struct.formula,
                               stage_names=("coarse",)),
                structure=StructureRef(source="j.source.xyz",
                                       formula=struct.formula,
                                       atoms=len(struct.elements)),
                varies=(), stages=(Stage(name="coarse", enabled=True,
                                         overrides={}),))
    write_task(calc / "task.json", task)

    # Self-describing deck (4.1b form A): its coordinate block is the
    # frozen gate's baseline; the in-body ATOM-METADATA block carries
    # the labels (emitted through the real emitter).
    from molbuilder.script_emit import emit_atom_metadata
    coords = "\n".join(
        f"  {q[0]:.6f}  {q[1]:.6f}  {q[2]:.6f}  1"
        for q in struct.positions)
    label_store = {k: list(v) for k, v in struct.regions.items()}
    if struct.frozen_atoms:
        label_store["frozen_atoms"] = list(struct.frozen_atoms)
    block = emit_atom_metadata(regions=label_store,
                               n_atoms_total=len(struct.elements)) or ""
    (attempt / "Relax_01_coarse.fdf").write_text(
        "SystemLabel Relax\nMeshCutoff 300.0 Ry\nXC.functional GGA\n"
        "XC.authors PBE\nPAO.BasisSize DZP\n"
        "AtomicCoordinatesFormat Ang\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        + coords + "\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n\n"
        + block + "\n")
    (attempt / "Relax_01_coarse-run0.concluded").write_text("rc=0\n")
    return _write_xv(attempt / "Relax.XV", struct)


@pytest.fixture
def tree(tmp_path):
    """A projects tree holding one concluded junction relaxation."""
    root = tmp_path / "projects"
    struct = _junction_struct()
    relaxed_pos = _write_tree(root, struct)
    return root, struct, relaxed_pos


_CITE = "J/optimization/Relax/01_coarse/run-0"


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
        assert len(out.provenance["files"]["Relax.XV"]) == 64
        assert out.provenance["evidence"].startswith("rc=")
        assert out.provenance["form"] == "relaxation"
        assert "Relax.XV" in out.provenance["files"]

    def test_the_record_is_the_whole_travelling_copy(self, tree, tmp_path):
        """§ 4.1: the cited structure is COPIED in with provenance --
        the SORTED pair, the attempt's own deck verbatim, and the two
        sidecars, so the folder answers without the cited tree."""
        root, _, _ = tree
        out = compose_junction(_CITE, tree_root=root)
        dest = tmp_path / "transport-calc"
        dest.mkdir()
        names = write_compose_record(dest, out)
        assert sorted(names) == ["atom-permutation.json",
                                 "junction.cited.fdf", "junction.xyz",
                                 "slot-provenance.json"]
        import json
        perm = json.loads((dest / "atom-permutation.json").read_text())
        assert perm["schema"] == "molbuilder/atom-permutation@1"
        assert sorted(perm["sorted_to_original"]) == list(range(16))
        # the deck is the attempt's own, byte for byte
        assert "MeshCutoff 300.0 Ry" in (dest / "junction.cited.fdf"
                                         ).read_text()
        # the geometry copy is the SORTED junction
        from molbuilder.workingcopy_structure import StructureCodec
        dev = StructureCodec().load(dest / "junction.xyz")
        assert dev.elements[:6] == ["Au"] * 6
        assert dev.elements[6:10] == ["S", "C", "C", "S"]

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
        assert ".fdf" in msg and ".XV" in msg and ".molstruct.json" in msg, (
            "the refusal states the WHOLE 4.1b condition -- both "
            "admissible forms, so the user learns what a citable "
            "directory holds")

    def test_an_unconcluded_attempt_refuses_without_deciding(self, tree):
        """A run RECORD that does not conclude refuses (mid-run or
        force-stopped -- indistinguishable, never decided).  The record
        here is run.json: the marker molbuilder's launch writes at
        process start."""
        root, _, _ = tree
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        (attempt / "Relax_01_coarse-run0.concluded").unlink()
        (attempt / "run.json").write_text("{}")
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        msg = str(e.value)
        assert "CONCLUDED" in msg
        assert "still running" in msg and "force-stopped" in msg, (
            "the two states are indistinguishable on disk -- the refusal "
            "must name both, never decide")

    def test_no_record_at_all_composes_and_says_so(self, tree):
        """4.1b: a directory with NO run record is a plain finished
        relaxation from anywhere -- the .XV is taken as the final
        geometry, and the provenance says the evidence honestly."""
        root, _, _ = tree
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        (attempt / "Relax_01_coarse-run0.concluded").unlink()
        out = compose_junction(_CITE, tree_root=root)
        assert out.provenance["evidence"] == "no-record"

    def test_a_missing_directory_is_refused(self, tree):
        root, _, _ = tree
        with pytest.raises(ComposeError) as e:
            compose_junction("Nope/optimization/Gone/run-0",
                             tree_root=root)
        assert "not a directory" in str(e.value)


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
        """The deck and the .XV must describe the SAME relaxation --
        the detectable disagreement under 4.1b (elements come FROM the
        .XV now) is the atom count."""
        import numpy as np
        root, src, _ = tree
        other = Structure(elements=list(src.elements) + ["Au"],
                          positions=np.vstack([src.positions,
                                               [[1.0, 1.0, 99.0]]]),
                          cell=src.cell)
        _write_xv(root / "J/optimization/Relax/01_coarse/run-0/Relax.XV",
                  other, perturb_bridge=0.0)
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        assert "do not describe the same relaxation" in str(e.value)

    def test_a_thin_electrode_block_is_refused(self, tmp_path):
        """§ 3: the principal-layer condition.  Five 2.5 A layers span
        10 A -- under the ~12 A floor -- and the refusal names the
        block, the span and the fix."""
        root = tmp_path / "projects"
        thin = _junction_struct(layers_l=[0.0, 2.5, 5.0, 7.5, 10.0])
        _write_tree(root, thin)
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        msg = str(e.value)
        assert REGION_LEFT_ELECTRODE in msg
        assert "principal-layer" in msg and "10.00 A" in msg

    def test_a_block_that_does_not_tile_is_refused(self, tmp_path):
        """§ 3: repeating the block must reproduce a bulk lead.  A 3.6 A
        gap among 2.5 A layers means the label boundary cut a partial
        layer -- refused naming the spacings."""
        root = tmp_path / "projects"
        broken = _junction_struct(
            layers_l=[0.0, 2.5, 5.0, 7.5, 10.0, 13.6])
        _write_tree(root, broken)
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        msg = str(e.value)
        assert "TILE" in msg and "3.600" in msg


class TestTravelCopy:
    """`load_compose_record` — the § 4.1 promise that the folder,
    once composed, answers WITHOUT the cited tree."""

    def _record(self, tree, tmp_path):
        root, _, _ = tree
        out = compose_junction(_CITE, tree_root=root)
        dest = tmp_path / "transport-calc"
        dest.mkdir()
        write_compose_record(dest, out)
        return dest, out

    def test_the_copy_loads_back_whole(self, tree, tmp_path):
        from molbuilder.transport.compose import load_compose_record
        dest, out = self._record(tree, tmp_path)
        back = load_compose_record(dest, citation=_CITE)
        assert back is not None
        dev = back.sorted.structure
        assert dev.elements == out.sorted.structure.elements
        assert np.allclose(dev.positions, out.sorted.structure.positions,
                           atol=1e-6)
        assert back.sorted.sorted_to_original == \
            out.sorted.sorted_to_original
        assert back.fdf_params.mesh_cutoff_ry == pytest.approx(300.0)
        assert back.provenance["citation"] == _CITE
        assert len(back.electrode_left.elements) == 6

    def test_a_repointed_citation_reads_as_no_record(self, tree, tmp_path):
        """A task.json re-cited to a different attempt must NOT keep
        serving the old copy -- prep composes fresh instead."""
        from molbuilder.transport.compose import load_compose_record
        dest, _ = self._record(tree, tmp_path)
        assert load_compose_record(
            dest, citation="J/optimization/Relax/01_coarse/run-1") is None

    def test_an_incomplete_copy_reads_as_no_record(self, tree, tmp_path):
        from molbuilder.transport.compose import load_compose_record
        dest, _ = self._record(tree, tmp_path)
        (dest / "junction.cited.fdf").unlink()
        assert load_compose_record(dest, citation=_CITE) is None


class TestFormB:
    """4.1b form B: a labeled .xyz+.molstruct.json pair, anywhere."""

    def _pair_dir(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "anything" / "at all"
        d.mkdir(parents=True)
        StructureCodec().write(_junction_struct(), d / "junction.xyz")
        return root, "anything/at all"

    def test_a_labeled_pair_composes_without_any_layout(self, tmp_path):
        root, cite = self._pair_dir(tmp_path)
        out = compose_junction(cite, tree_root=root)
        assert out.form == "structure"
        assert out.deck_text is None and out.fdf_params is None
        assert out.provenance["evidence"] == "given"
        assert len(out.electrode_left.elements) == 6

    def test_a_wins_over_b_when_both_coexist(self, tree, tmp_path):
        """The deck carries the contract -- more information never
        loses to less."""
        from molbuilder.workingcopy_structure import StructureCodec
        root, src, _ = tree
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        StructureCodec().write(src, attempt / "junction.xyz")
        out = compose_junction(_CITE, tree_root=root)
        assert out.form == "relaxation"
        assert out.deck_text is not None

    def test_two_decks_are_refused_as_ambiguous(self, tree):
        root, _, _ = tree
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        (attempt / "second.fdf").write_text("SystemLabel x\n")
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        assert "unambiguously" in str(e.value)

    def test_a_pair_without_a_cell_is_refused(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "loose"
        d.mkdir(parents=True)
        s2 = _junction_struct()
        s2.cell = None
        StructureCodec().write(s2, d / "junction.xyz")
        with pytest.raises(ComposeError) as e:
            compose_junction("loose", tree_root=root)
        assert "no cell" in str(e.value)

    def test_a_bare_xyz_names_the_missing_sidecar(self, tmp_path):
        root = tmp_path / "projects"
        d = root / "loose"
        d.mkdir(parents=True)
        (d / "junction.xyz").write_text("1\n\nC 0 0 0\n")
        with pytest.raises(ComposeError) as e:
            compose_junction("loose", tree_root=root)
        msg = str(e.value)
        assert ".molstruct.json" in msg and ".fdf" in msg, (
            "the refusal states the whole condition")


class TestTheRecordedContract:
    """4.1b's third shade (structure-info-plan.md I6): a pair whose
    sidecar carries `info.calculation` seals like a cited deck."""

    def _recorded_pair(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "exported"
        d.mkdir(parents=True)
        s2 = _junction_struct()
        s2.info = {"calculation": {
            "engine": "siesta",
            "contract": {"basis_size": "TZP",
                         "siesta_mesh_cutoff_ry": 275,
                         "xc_functional": "GGA", "xc_authors": "revPBE",
                         "k_mesh_transverse": [3, 3, 2],
                         "electronic_temperature_k": 150.0},
            "source": "Relax.fdf", "source_sha256": "c" * 64}}
        StructureCodec().write(s2, d / "junction.xyz")
        return root, "exported"

    def test_the_record_rides_the_composition(self, tmp_path):
        root, cite = self._recorded_pair(tmp_path)
        out = compose_junction(cite, tree_root=root)
        assert out.form == "structure"
        assert out.recorded_contract is not None
        assert out.recorded_contract["contract"]["basis_size"] == "TZP"
        assert out.provenance["recorded_contract"]["source"] == "Relax.fdf"

    def test_the_record_fills_the_config_and_forces_kz(self, tmp_path):
        from molbuilder.transport.stages import config_for
        from molbuilder.task import Stage, Task, derive_run
        root, cite = self._recorded_pair(tmp_path)
        out = compose_junction(cite, tree_root=root)
        task = Task(engine="siesta", shape="hierarchical",
                    run=derive_run("T", cite, stage_names=("seed",)),
                    structure=None, calculation="transport",
                    slots={"junction": cite}, bias=(0.0,), varies=(),
                    stages=(Stage(name="seed", enabled=True,
                                  overrides={}),))
        cfg = config_for(task, out)
        assert cfg.basis_size == "TZP"
        assert cfg.siesta_mesh_cutoff_ry == 275
        assert cfg.xc_authors == "revPBE"
        assert cfg.k_mesh_transverse == (3, 3, 1), "kz forced 1, always"
        assert cfg.electronic_temperature_k == 150.0

    def test_the_record_seals_the_contract_fields(self, tmp_path):
        from molbuilder.transport.stages import StageError, config_for
        from molbuilder.task import Stage, Task, derive_run
        root, cite = self._recorded_pair(tmp_path)
        out = compose_junction(cite, tree_root=root)
        task = Task(engine="siesta", shape="hierarchical",
                    run=derive_run("T", cite, stage_names=("seed",)),
                    structure=None, calculation="transport",
                    slots={"junction": cite}, bias=(0.0,),
                    varies=("basis_size",),
                    stages=(Stage(name="seed", enabled=True,
                                  overrides={"basis_size": "SZ"}),))
        with pytest.raises(StageError) as e:
            config_for(task, out)
        assert "RECORDED" in str(e.value)

    def test_a_plain_pair_stays_open(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "plain"
        d.mkdir(parents=True)
        StructureCodec().write(_junction_struct(), d / "junction.xyz")
        out = compose_junction("plain", tree_root=root)
        assert out.recorded_contract is None

    def test_the_travel_copy_keeps_the_record(self, tmp_path):
        from molbuilder.transport.compose import (load_compose_record,
                                                  write_compose_record)
        root, cite = self._recorded_pair(tmp_path)
        out = compose_junction(cite, tree_root=root)
        dest = tmp_path / "travelled"
        dest.mkdir()
        write_compose_record(dest, out)
        back = load_compose_record(dest, citation=cite)
        assert back is not None
        assert back.recorded_contract["contract"]["basis_size"] == "TZP"

"""P4a — prep composes from the citation (`archive/2026-09-01-transport-design.md`
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


def _write_ion(path, rc_bohr):
    """A minimal .ion: one orbital block, cutoff *rc_bohr* -- the shape
    parse/ion.py anchors on (#orbital header, then npts/delta/cutoff)."""
    path.write_text(
        "  0  6  1  0  1.000000   #orbital l, n, z, is_polarized, "
        "population\n"
        f" 500    0.4883E-02     {rc_bohr:.6f}     # npts, delta, "
        "cutoff\n")


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
        the SORTED PAIR (geometry AND the file carrying its region
        labels), the attempt's own deck verbatim, and the two sidecars,
        so the folder answers without the cited tree.

        The label file is named explicitly here because it was missing
        from this list for a while (found by reading, 2026-08-29): the
        codec writes the geometry as a pair, so it landed on disk, but
        nothing declared it part of the record -- and the load check
        did not require it either."""
        root, _, _ = tree
        out = compose_junction(_CITE, tree_root=root)
        dest = tmp_path / "transport-calc"
        dest.mkdir()
        names = write_compose_record(dest, out)
        assert sorted(names) == ["atom-permutation.json",
                                 "junction.cited.fdf",
                                 "junction.molstruct.json", "junction.xyz",
                                 "slot-provenance.json"]
        assert (dest / "junction.molstruct.json").is_file()
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

    def test_siestas_own_exit_marker_counts_as_concluded(self, tree):
        """4.1b: evidence is FILES, never our marker spelling.  A run
        that carries SIESTA's 0_NORMAL_EXIT ran to its own end whatever
        launched it -- run.json present, no molbuilder marker, and it
        still composes, saying which file answered."""
        root, _, _ = tree
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        (attempt / "Relax_01_coarse-run0.concluded").unlink()
        (attempt / "run.json").write_text("{}")
        (attempt / "0_NORMAL_EXIT").write_text("")
        out = compose_junction(_CITE, tree_root=root)
        assert "0_NORMAL_EXIT" in out.provenance["evidence"]

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

    def test_the_orbital_range_gate_refuses_when_the_ion_says_so(
            self, tmp_path):
        """§ 3: the principal-layer condition compares the orbital
        interaction range (READ from the citation's .ion files) against
        the lead's period.  A basis whose reach exceeds period +
        interlayer couples beyond adjacent cells -- refused naming the
        numbers and the source file."""
        root = tmp_path / "projects"
        thin = _junction_struct(layers_l=[0.0, 2.5, 5.0, 7.5, 10.0])
        _write_tree(root, thin)
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        # rc 15 Bohr = 7.94 A: reach 15.88 A > the L lead's
        # period 12.5 + interlayer 2.5 = 15.0 A.
        _write_ion(attempt / "Au.ion", 15.0)
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        msg = str(e.value)
        assert REGION_LEFT_ELECTRODE in msg
        assert "orbital interaction range" in msg and "Au.ion" in msg

    def test_a_short_reach_basis_passes_the_gate(self, tmp_path):
        """The same thin block with a realistic basis (rc 6.13 Bohr,
        Au DZP) composes -- reach 6.49 A fits the 15.0 A gap.  Kills a
        mutation that inverts the comparison.

        And the MEASURED verdict replaces the wizard's ~12 A guess:
        this 10 A block would otherwise carry "may be thinner than the
        electronic principal layer" beside the numbers proving it is
        not (found by reading, 2026-08-29)."""
        root = tmp_path / "projects"
        thin = _junction_struct(layers_l=[0.0, 2.5, 5.0, 7.5, 10.0])
        _write_tree(root, thin)
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        _write_ion(attempt / "Au.ion", 6.13)
        out = compose_junction(_CITE, tree_root=root)
        notes = out.electrode_left.notes
        assert not any("UNVERIFIED" in n for n in notes)
        assert any("MEASURED" in n for n in notes), (
            "a measured lead does not say so")
        assert not any("may be thinner" in n for n in notes), (
            "the heuristic floor still contradicts the measurement "
            "that superseded it")

    def test_the_refusal_boundary_is_the_next_nearest_cell_gap(
            self, tmp_path):
        """The gate fires exactly when the orbital reach exceeds the
        separation of NEXT-NEAREST lead cells -- pinned at the boundary
        from both sides.

        The gap is written ``2*period - span``.  That is the general
        expression; in every path compose takes it equals
        ``period + interlayer``, because compose always lets the period
        be DERIVED as span + interlayer (only the wizard CLI overrides
        it).  So this test cannot tell the two spellings apart, and
        does not claim to -- it pins the NUMBER and the threshold."""
        root = tmp_path / "projects"
        # 6 layers, 2.5 A apart: span 12.5, interlayer 2.5,
        # period 15.0 -> gap = 2*15.0 - 12.5 = 17.5 A.
        _write_tree(root, _junction_struct())
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        # rc 16.5 Bohr = 8.732 A -> reach 17.46 A, just under 17.5.
        for el in ("Au", "S", "C"):
            _write_ion(attempt / f"{el}.ion", 0.1)
        _write_ion(attempt / "Au.ion", 16.5)
        out = compose_junction(_CITE, tree_root=root)
        m = out.electrode_left
        # 1e-4, not 1e-6: the fixture's geometry arrives through a
        # Bohr->Angstrom conversion, so 17.5 is exact only to the
        # precision of that constant.  What is pinned here is a
        # THRESHOLD at 17.5 A, and 7e-6 A is not a threshold question.
        assert abs((2.0 * m.z_period - m.z_span) - 17.5) < 1e-4
        assert not any("UNVERIFIED" in n for n in m.notes)
        # 16.6 Bohr = 8.785 A -> reach 17.57 A, just over: refused.
        _write_ion(attempt / "Au.ion", 16.6)
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        assert "next-nearest" in str(e.value)

    def test_no_ion_files_leave_the_condition_unverified(self, tmp_path):
        """No .ion beside the citation -> nothing measured, nothing
        refused: the condition is recorded as UNVERIFIED on the model
        (TranSIESTA verifies lead connectivity itself at run time)."""
        root = tmp_path / "projects"
        thin = _junction_struct(layers_l=[0.0, 2.5, 5.0, 7.5, 10.0])
        _write_tree(root, thin)
        out = compose_junction(_CITE, tree_root=root)
        notes = " ".join(out.electrode_left.notes)
        assert "UNVERIFIED" in notes and "Au.ion" in notes

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

    def test_a_block_that_does_not_tile_is_told_THAT_first(self, tmp_path):
        """The order of these two gates is a DEPENDENCY, not taste.
        Every number the orbital-range condition uses comes from the
        layer spacing -- a block that does not tile has a meaningless
        median, hence a meaningless period and gap.  Checked second, it
        was refused for "orbital range exceeds the gap": true about
        invented numbers, and the wrong thing to go and fix."""
        root = tmp_path / "projects"
        broken = _junction_struct(
            layers_l=[0.0, 2.5, 5.0, 7.5, 10.0, 13.6])
        _write_tree(root, broken)
        # A basis wide enough that the orbital gate would ALSO fire.
        _write_ion(root / "J/optimization/Relax/01_coarse/run-0/Au.ion",
                   30.0)
        with pytest.raises(ComposeError) as e:
            compose_junction(_CITE, tree_root=root)
        msg = str(e.value)
        assert "TILE" in msg, (
            "the block's real defect is that it does not tile; the "
            "refusal talks about something derived from it instead")
        assert "orbital interaction range" not in msg


class TestOrientation:
    """§ 4.1a, user ruling 2026-08-29: CHECK z, WARN, the author
    decides.  The usual convention is L-electrode low / R high, but a
    junction labeled the other way round composes and runs -- it biases
    the other end.  What is NOT negotiable is that the LOWER block
    leads the atom list (it is the -A3 lead) and that the two blocks
    do not interleave."""

    def test_the_conventional_junction_sorts_l_first(self, tree):
        root, _, _ = tree
        srt = compose_junction(_CITE, tree_root=root).sorted.structure
        n_l = len(srt.regions[REGION_LEFT_ELECTRODE])
        assert sorted(srt.regions[REGION_LEFT_ELECTRODE]) == list(
            range(n_l)), "the L block leads the canonical order"

    def test_an_inverted_and_interleaved_pair_is_not_offered_a_swap(
            self, tmp_path):
        """Found by reading, not by running (2026-08-29): a junction
        that is BOTH named the wrong way round and interleaved is not
        fixable by a swap -- classifying it `inverted` would offer a
        relabel that leaves it just as unusable, and the person would
        have edited their run for nothing.  Interleaving is decided
        first."""
        from molbuilder.transport.sort import (ORDER_INTERLEAVED,
                                               electrode_orientation)
        both = _junction_struct(
            layers_l=[1.25, 3.75, 6.25, 8.75, 11.25, 13.75],
            layers_r=[0.0, 2.5, 5.0, 7.5, 10.0, 12.5])
        assert electrode_orientation(both) == ORDER_INTERLEAVED, (
            "an unfixable junction was classified as a naming mistake")
        root = tmp_path / "projects"
        _write_tree(root, both)
        from molbuilder.transport.sort import SortError
        with pytest.raises(SortError) as e:
            compose_junction(_CITE, tree_root=root)
        assert "INTERLEAVE" in str(e.value)
        assert "SWAP THE TWO LABELS" not in str(e.value)

    def test_l_on_top_composes_and_says_what_it_means(self, tmp_path):
        """It COMPOSES (no refusal), the LOWER block leads the atom
        list whatever it is named, and the note states the consequence
        the author has to accept or fix: the high-z lead is the one at
        +V/2."""
        root = tmp_path / "projects"
        flipped = _junction_struct(layers_l=_LAYERS_R,
                                   layers_r=_LAYERS_L)
        _write_tree(root, flipped)
        composed = compose_junction(_CITE, tree_root=root)
        srt = composed.sorted.structure
        n_r = len(srt.regions[REGION_RIGHT_ELECTRODE])
        assert sorted(srt.regions[REGION_RIGHT_ELECTRODE]) == list(
            range(n_r)), "the LOWER block leads, whatever it is called"
        assert composed.sorted.notes, "the inversion must be reported"
        note = composed.sorted.notes[0]
        assert "HIGH-z" in note and "+V/2" in note

    def test_interleaved_blocks_are_refused(self, tmp_path):
        """What geometry MUST supply is two distinguishable ends: the
        blocks' z-ranges may not interleave."""
        root = tmp_path / "projects"
        tangled = _junction_struct(
            layers_l=[0.0, 2.5, 5.0, 7.5, 10.0, 12.5],
            layers_r=[1.25, 3.75, 6.25, 8.75, 11.25, 13.75])
        _write_tree(root, tangled)
        from molbuilder.transport.sort import SortError
        # SortError propagates as its own type through compose (the
        # documented 4.1a seam; prep and the web door catch both).
        with pytest.raises(SortError) as e:
            compose_junction(_CITE, tree_root=root)
        assert "INTERLEAVE" in str(e.value)


class TestTheSwap:
    """The one-click fix the tab offers (user, 2026-08-29): swapping
    two labels on the cited files, and nothing else."""

    @staticmethod
    def _cited(root):
        from molbuilder.transport.compose import resolve_citation
        return resolve_citation(_CITE, root)[1]

    def test_the_swap_makes_an_inverted_junction_compose(self, tmp_path):
        from molbuilder.transport.compose import swap_electrode_labels
        root = tmp_path / "projects"
        _write_tree(root, _junction_struct(layers_l=_LAYERS_R,
                                           layers_r=_LAYERS_L))
        changed = swap_electrode_labels(self._cited(root))
        assert changed.endswith(".fdf")
        srt = compose_junction(_CITE, tree_root=root).sorted.structure
        n_l = len(srt.regions[REGION_LEFT_ELECTRODE])
        assert sorted(srt.regions[REGION_LEFT_ELECTRODE]) == list(
            range(n_l)), "after the swap L must lead the canonical order"

    def test_the_swap_keeps_everything_it_did_not_come_to_change(
            self, tmp_path):
        """Found by reading (2026-08-29): the rewrite re-emits the
        whole block, so anything not threaded through is DELETED.  The
        atom count, the creation time and the other labels must
        survive, and the write must say a swap happened."""
        from molbuilder.parse.scripts.atom_metadata import (
            _extract_atom_metadata_dict,
        )
        from molbuilder.transport.compose import swap_electrode_labels
        root = tmp_path / "projects"
        _write_tree(root, _junction_struct(layers_l=_LAYERS_R,
                                           layers_r=_LAYERS_L))
        deck = root / "J/optimization/Relax/01_coarse/run-0/Relax_01_coarse.fdf"
        before = _extract_atom_metadata_dict(deck.read_text())
        swap_electrode_labels(self._cited(root))
        after = _extract_atom_metadata_dict(deck.read_text())
        assert after["n_atoms_total"] == before["n_atoms_total"]
        assert after.get("created_at") == before.get("created_at"), (
            "the block's creation time was dropped by the rewrite")
        assert "swapped" in str(after.get("created_by", "")).lower(), (
            "the file does not record that its labels were swapped")
        assert (after["regions"][REGION_BRIDGE]
                == before["regions"][REGION_BRIDGE]), (
            "a label the swap never touches came back different")
        assert (after["regions"][REGION_LEFT_ELECTRODE]
                == before["regions"][REGION_RIGHT_ELECTRODE])
        # The deck's own physics is untouched -- only the fenced block.
        for keyword in ("MeshCutoff 300.0 Ry", "PAO.BasisSize DZP",
                        "%block AtomicCoordinatesAndAtomicSpecies"):
            assert keyword in deck.read_text(), (
                f"the rewrite disturbed {keyword!r} outside the fence")

    def test_the_swap_consults_no_geometry(self, tree):
        """A RENAME IS A RENAME (user ruling, 2026-08-29).  An earlier
        draft read the coordinates first and refused to run unless they
        said the labels were inverted -- the tool second-guessing a
        decision that is the author's.  Pressed on an already-canonical
        junction it must simply do it, leaving the pair the other way
        round."""
        from molbuilder.transport.compose import swap_electrode_labels
        root, _, _ = tree
        before = compose_junction(_CITE, tree_root=root)
        assert not before.sorted.notes, "fixture is not canonical"

        swap_electrode_labels(self._cited(root))            # no refusal

        after = compose_junction(_CITE, tree_root=root)
        assert after.sorted.notes, (
            "the rename did nothing -- L should now be the high-z block")

    def test_the_file_that_carries_the_labels_is_the_file_that_changes(
            self, tmp_path):
        """Found by reading (2026-08-29): form A accepts labels from
        the deck's in-body block OR from one .molstruct.json beside it.
        The swap must rewrite whichever the composition READS -- an
        earlier draft always edited the deck, so a sidecar-labeled
        citation would have been told it had no labels to swap."""
        from molbuilder.transport.compose import swap_electrode_labels
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        struct = _junction_struct(layers_l=_LAYERS_R, layers_r=_LAYERS_L)
        _write_tree(root, struct)
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        deck = attempt / "Relax_01_coarse.fdf"
        # Strip the in-body block: the labels now live ONLY in the
        # sidecar beside the deck -- written through the codec, which
        # is the pair's one writer.
        text = deck.read_text()
        cut = text.index("# === molbuilder atom-metadata BEGIN ===")
        deck.write_text(text[:cut])
        StructureCodec().write(struct, attempt / "Relax.source.xyz")
        changed = swap_electrode_labels(self._cited(root))
        assert changed.endswith(".molstruct.json"), (
            f"the swap rewrote {changed}, not the file the labels are in")

    def test_the_provenance_records_the_file_the_labels_came_from(
            self, tmp_path):
        """A junction's electrode REGIONS are a fact about it as much
        as its coordinates are.  When they live in a .molstruct.json
        beside the deck rather than in the deck itself, that file is in
        none of the other provenance slots -- so the record of "which
        bytes built this" was missing the file that decided which atoms
        are leads, and which the rename endpoint rewrites."""
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        struct = _junction_struct()
        _write_tree(root, struct)
        attempt = root / "J/optimization/Relax/01_coarse/run-0"
        deck = attempt / "Relax_01_coarse.fdf"
        text = deck.read_text()
        deck.write_text(
            text[:text.index("# === molbuilder atom-metadata BEGIN ===")])
        StructureCodec().write(struct, attempt / "Relax.source.xyz")

        files = compose_junction(_CITE, tree_root=root).provenance["files"]
        assert "Relax.source.molstruct.json" in files, (
            f"the labels' own file is unrecorded; provenance names "
            f"{sorted(files)}")


class TestTheReloadGate:
    """The travelled record re-runs the § 3 lead gates -- and the
    principal-layer half needs the CITED directory's .ion files, which
    the travelled folder does not carry."""

    def test_the_reload_reads_the_citations_ion_files(self, tree,
                                                       tmp_path):
        from molbuilder.transport.compose import (load_compose_record,
                                                  write_compose_record)
        root, _, _ = tree
        _write_ion(root / "J/optimization/Relax/01_coarse/run-0/Au.ion",
                   6.13)
        out = compose_junction(_CITE, tree_root=root)
        base = tmp_path / "calc"
        base.mkdir()
        write_compose_record(base, out)
        back = load_compose_record(base, citation=_CITE, tree_root=root)
        assert back is not None
        assert not any("UNVERIFIED" in n
                       for n in back.electrode_left.notes), (
            "the reload did not read the citation's .ion files, so the "
            "principal-layer gate silently stopped running")

    def test_without_the_root_the_reload_says_unverified(self, tree,
                                                          tmp_path):
        from molbuilder.transport.compose import (load_compose_record,
                                                  write_compose_record)
        root, _, _ = tree
        out = compose_junction(_CITE, tree_root=root)
        base = tmp_path / "calc"
        base.mkdir()
        write_compose_record(base, out)
        back = load_compose_record(base, citation=_CITE)
        assert any("UNVERIFIED" in n
                   for n in back.electrode_left.notes)


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

    def test_a_copy_without_its_LABELS_reads_as_no_record(self, tree,
                                                           tmp_path):
        """Found by reading (2026-08-29).  The geometry travels as a
        PAIR and the second half carries the electrode regions, but the
        completeness check named only the first -- so a record whose
        labels had been deleted passed it, loaded a junction with no
        electrodes, and died deep inside the lead gates with a message
        about regions rather than answering "incomplete, compose
        again"."""
        from molbuilder.transport.compose import load_compose_record
        dest, _ = self._record(tree, tmp_path)
        (dest / "junction.molstruct.json").unlink()
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
    """4.1b's third shade (archive/2026-09-01-structure-info-plan.md I6): a pair whose
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

"""P4b — prep renders the transport composite's five stages
(`engines/transport.md` § 1 + § 3, the arm in `jobset/prep.py` +
`transport/stages.py`).

A fixture junction preps end-to-end; each gate refuses its mutation; the
emitter's own order-preflight never fires, because prep sorted first
(§ 4, *"in the composite it cannot fire in anger"*).

Properties under guard, each named for its failure:

* each stage's deck is born in its own stage directory, wrapper beside
  it, through the SHARED prep tail (job-set merge, STAGE-PLAN, run
  dirs) — no forked machinery;
* the electronic contract (basis · XC · energy shift · mesh · k ·
  electronic T) is read from the CITED attempt's own deck and lands in
  every stage's deck — one template governs (§ 5's invariant set, baked
  identically into all three fdfs; `stages.config_for`, fdf-is-truth);
* the electrode deck's SystemLabel IS the ``.TSHS`` stem the device
  deck references — one spelling, both writers;
* the emitter's order-preflight never fires: a source whose atom order
  would trip it preps clean, because prep sorted first (§ 4);
* buffer atoms emit ``TS.Atoms.Buffer`` + explicit electrode positions
  (§ 4, the ``buffer`` label: with padding outermost, TranSIESTA's default
  first-N/last-N placement no longer holds);
* the composed record is written once, reused by later stages, travels
  with the folder, and a re-pointed citation recomposes;
* refusals: unnamed/unknown/disabled stage, a sweep, a moved frozen
  atom, a pseudopotential the citation cannot supply (§ 3.1 — a citation
  names a directory, so the directory must hold what the stage consumes).
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from conftest import write_pseudos
from molbuilder.config.transport import (REGION_BRIDGE, REGION_BUFFER,
                                         REGION_LEFT_ELECTRODE,
                                         REGION_RIGHT_ELECTRODE)
from molbuilder.jobset.prep import PrepError, prep_calculation
from molbuilder.structure import Structure
from test_transport_compose import _BRIDGE, _LAYERS_L, _LAYERS_R, _write_xv

_CITE = "J/optimization/Relax/01_coarse/run-0"
_STAGES = ("seed", "electrode_L", "electrode_R", "device", "transmission")

#: The cited deck carries DISTINCTIVE values, so every assertion below
#: that finds one in a rendered stage deck proves the contract flowed
#: from the citation rather than from a default that happens to agree.
_CITED_DECK = """SystemLabel Relax
MeshCutoff 250.0 Ry
PAO.BasisSize TZP
PAO.EnergyShift 0.02 Ry
XC.functional GGA
XC.authors revPBE
ElectronicTemperature 200.0 K
%block kgrid_Monkhorst_Pack
  4 0 0 0.0
  0 4 0 0.0
  0 0 2 0.0
%endblock kgrid_Monkhorst_Pack
"""


def _junction_struct(*, order="canonical", buffers=False):
    """The BDT-ish fixture sandwich; ``order="scrambled"`` writes the
    same geometry with the bridge FIRST and the leads swapped after it
    — exactly the order the emitter's preflight refuses."""
    rows = []       # (element, z, label)
    for z in _LAYERS_L:
        rows.append(("Au", z, REGION_LEFT_ELECTRODE))
    for el, z in _BRIDGE:
        rows.append((el, z, REGION_BRIDGE))
    for z in _LAYERS_R:
        rows.append(("Au", z, REGION_RIGHT_ELECTRODE))
    if buffers:
        for z in (-5.0, -2.5, 37.0, 39.5):
            rows.append(("Au", z, REGION_BUFFER))
    if order == "scrambled":
        rows = ([r for r in rows if r[2] == REGION_BRIDGE]
                + [r for r in rows if r[2] == REGION_RIGHT_ELECTRODE]
                + [r for r in rows if r[2] == REGION_LEFT_ELECTRODE]
                + [r for r in rows if r[2] == REGION_BUFFER])
    elements = [r[0] for r in rows]
    positions = np.array([[1.0, 1.0, r[1]] for r in rows])
    regions: dict = {}
    for i, r in enumerate(rows):
        regions.setdefault(r[2], []).append(i)
    frozen = [i for i, r in enumerate(rows)
              if r[2] in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE)]
    return Structure(elements=elements, positions=positions,
                     regions=regions, frozen_atoms=frozen,
                     cell=np.diag([8.0, 8.0, 40.0]))


def _write_junction(root, struct):
    """One concluded junction relaxation with the distinctive deck."""
    from molbuilder.task import (Stage, StructureRef, Task, derive_run,
                                 write_task)
    from molbuilder.workingcopy_structure import StructureCodec
    calc = root / "J" / "optimization" / "Relax"
    attempt = calc / "01_coarse" / "run-0"
    attempt.mkdir(parents=True)
    StructureCodec().write(struct, calc / "j.source.xyz")
    write_task(calc / "task.json", Task(
        engine="siesta", shape="hierarchical",
        run=derive_run("Relax", struct.formula, stage_names=("coarse",)),
        structure=StructureRef(source="j.source.xyz",
                               formula=struct.formula,
                               atoms=len(struct.elements)),
        varies=(), stages=(Stage(name="coarse", enabled=True,
                                 overrides={}),)))
    # The deck is SELF-DESCRIBING (4.1b form A): its own coordinate
    # block is the frozen gate's baseline, and the in-body
    # ATOM-METADATA block carries the labels -- emitted through the
    # real emitter, never hand-spelled.
    from molbuilder.script_emit import emit_atom_metadata
    coords = "\n".join(
        f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}  1"
        for p in struct.positions)
    label_store = {k: list(v) for k, v in struct.regions.items()}
    if struct.frozen_atoms:
        label_store["frozen_atoms"] = list(struct.frozen_atoms)
    block = emit_atom_metadata(regions=label_store,
                               n_atoms_total=len(struct.elements)) or ""
    deck_text = (_CITED_DECK
                 + "AtomicCoordinatesFormat Ang\n"
                 + "%block AtomicCoordinatesAndAtomicSpecies\n"
                 + coords + "\n"
                 + "%endblock AtomicCoordinatesAndAtomicSpecies\n\n"
                 + block + "\n")
    (attempt / "Relax_01_coarse.fdf").write_text(deck_text)
    (attempt / "Relax_01_coarse-run0.concluded").write_text("rc=0\n")
    _write_xv(attempt / "Relax.XV", struct)
    # Pseudos live IN the cited directory (4.1b: same-directory rule).
    write_pseudos(attempt, ["Au", "S", "C"])
    return calc


def _describe_transport(root, *, cite=_CITE, bias=(0.0, 0.2)):
    from molbuilder.task import Stage, Task, derive_run, write_task
    dest = root / "J" / "transport" / "T"
    dest.mkdir(parents=True, exist_ok=True)
    write_task(dest / "task.json", Task(
        engine="siesta", shape="hierarchical",
        run=derive_run("T", cite, stage_names=_STAGES),
        structure=None, calculation="transport",
        slots={"junction": cite}, bias=bias, varies=(),
        stages=tuple(Stage(name=n, enabled=True, overrides={})
                     for n in _STAGES)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate"}}))
    return dest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """Same sandbox as test_prep_calculation: the wrapper writer must
    read the fixture's bundle-scoped config, never this repo's."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # ...and the box is probed.  Moving HOME moves the machine scope
    # out from under conftest's own record, and prep refuses without
    # one (`running-a-job.md` § 3.1).
    from conftest import write_machine_record
    write_machine_record()
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture
def calc(tmp_path):
    """A projects tree with a concluded junction + a described
    transport composite, exactly as `jobset init` leaves them."""
    root = tmp_path / "projects"
    _write_junction(root, _junction_struct())
    return _describe_transport(root)


class TestTheLadderPreps:

    def test_seed_preps_end_to_end(self, calc):
        """SCIENCE + the shared-tail lint. The seed stage preps whole: a `diagon` deck
        in its own stage directory, sharing the task's `SystemLabel`, with the
        wrapper, the composed record, the citation's pseudopotentials and the
        job-set row beside it.

        Catches two failures with one run. (1) The seed is a plain SIESTA
        single-point whose job is to produce the `.DM` the device warm-starts from
        -- so it must be `SolutionMethod diagon`, and its `SystemLabel` must be the
        task's, or the density it writes is named something the device stage does
        not look for and the expensive TranSIESTA run starts cold. (2) The transport
        arm going around the shared prep tail: the wrapper, `job-set.json` and
        `STAGE-PLAN.md` are what every other calculation gets, and a forked
        transport path that produced a deck and none of the rest would look correct
        in the directory listing and be unlaunchable.

        Contract: `engines/transport.md` § 1 (one citation, five derived stages)
        + § 3.1 (a citation names a directory, and the directory must hold what the
        stage consumes).
        """
        dirs = prep_calculation(calc, "seed")
        assert dirs
        deck = calc / "01_seed" / "T_01_seed.fdf"
        assert deck.is_file(), "the deck is born in its stage directory"
        text = deck.read_text()
        assert "SolutionMethod         diagon" in text
        assert "SystemLabel            T" in text, (
            "the seed shares the task label so its .DM is what the "
            "device stage will read")
        assert (calc / "01_seed" / "T_01_seed.run.sh").is_file(), (
            "the wrapper renders beside the deck -- the shared tail")
        # the composed record landed beside task.json
        for name in ("junction.xyz", "junction.cited.fdf",
                     "slot-provenance.json", "atom-permutation.json"):
            assert (calc / name).is_file(), name
        # pseudos arrived from the citation, grouped
        assert (calc / "pseudos" / "Au.psml").is_file()
        # the run plan carries the rung
        js = json.loads((calc / "job-set.json").read_text())
        assert [j["name"] for j in js["jobs"]] == ["seed"]
        assert (calc / "STAGE-PLAN.md").is_file()

    def test_the_RUNS_CHOSEN_SHAPE_reaches_a_transport_stage(self, calc):
        """**The run's launch shape travels into the transport arm.**

        `prep_calculation` takes `chosen` -- what the person decided on the
        run card, or `--np` on the command line -- and the transport
        hand-off DROPPED it: every other calculation honoured the decision
        and a transport run silently fell back to the target's own width.
        Both sides had always declared the parameter; only the forwarding
        was missing, which is why no signature check and no type checker
        could see it.

        Read off the WRAPPER, which is the artifact the cluster runs, rather
        than off an intermediate: a value that reaches an inner function and
        not the header is the failure this whole class of test exists for.
        """
        prep_calculation(calc, "seed", chosen={"mpi_np": 7})
        text = (calc / "01_seed" / "T_01_seed.run.sh").read_text()
        assert "_mpi_np_default=7" in text, (
            "the run card asked for 7 ranks and the wrapper was rendered "
            "for something else:\n"
            + "\n".join(ln for ln in text.splitlines()
                         if "mpi_np" in ln or "np_default" in ln))

    def test_the_electrode_deck_is_the_tshs_the_device_asks_for(self, calc):
        """SCIENCE. The electrode deck's `SystemLabel` IS the `.TSHS` stem the device
        deck names -- one spelling written by two different writers -- and the
        electrode saves its H/S at all.

        Catches the lead self-energy never being built from the lead. Sigma_L and
        Sigma_R come from a SEPARATE pristine bulk run (`engines/transport.md` § 2:
        hence three runs), and the only thing joining that run to the device is the
        filename. Two writers each choose it: a drift between them means the device
        stage looks for a `.TSHS` nobody wrote, or -- worse, once the gather is
        permissive -- picks up a stale one. `TS.HS.Save true` is the other half:
        without it the electrode run converges happily and writes nothing.

        Contract: `engines/transport.md` § 2 (Sigma from a separate bulk-lead run)
        + § 5 (the consistency contract).
        """
        prep_calculation(calc, "electrode_L")
        prep_calculation(calc, "device")
        elec = (calc / "02_electrode_L" / "T_02_electrode_L.fdf"
                ).read_text()
        dev = (calc / "04_device" / "T_04_device.fdf").read_text()
        assert "SystemLabel            T_L-electrode" in elec, (
            "the electrode's SystemLabel IS the .TSHS stem")
        assert "T_L-electrode.TSHS" in dev, (
            "the device references exactly the file the electrode "
            "stage will write -- one spelling, both writers")
        assert "TS.HS.Save             true" in elec

    def test_the_device_deck_is_transiesta_on_the_sorted_junction(self, calc):
        """SCIENCE. The device deck is `SolutionMethod transiesta` with a `TS.Elecs`
        block, and its coordinate block is CATEGORICALLY SORTED -- six Au rows, then
        the four bridge rows as S C C S.

        Catches the two ways the device stage stops being an NEGF calculation.
        Without `transiesta` + `TS.Elecs` it is an ordinary closed-boundary
        single-point that converges and means nothing. And TranSIESTA identifies
        each electrode by a CONTIGUOUS atom range, so the order of the coordinate
        block is not cosmetic: an unsorted junction makes the lead ranges name the
        wrong atoms, and the run computes transmission through a region that is not
        the molecule. Asserting the species column, not just the row count, is what
        distinguishes "sorted" from "reordered into a different wrong order".

        Contract: `engines/transport.md` § 4 (region labels drive the partition;
        the categorical sort) + § 2 (the L | bridge | R partition is one cell).
        """
        prep_calculation(calc, "device")
        text = (calc / "04_device" / "T_04_device.fdf").read_text()
        assert "SolutionMethod         transiesta" in text
        assert "%block TS.Elecs" in text
        # sorted: the first six coordinate rows are Au (species 1 --
        # alphabetical Au/C/S), the next four the bridge (S C C S)
        block = text.split("%block AtomicCoordinatesAndAtomicSpecies")[1]
        rows = [ln.split() for ln in block.splitlines()
                if ln.strip() and not ln.startswith("%")]
        assert [r[3] for r in rows[:6]] == ["1"] * 6
        assert [r[3] for r in rows[6:10]] == ["3", "2", "2", "3"]

    def test_the_transmission_deck_carries_the_tbt_window(self, calc):
        """SCIENCE. The transmission deck carries the tbtrans energy window
        (`TS.TBT.NumE`, `TS.TBT.Emin`).

        Catches the deliverable being computed over no energy range. T(E) is
        evaluated on a grid the deck specifies; with the window keywords missing,
        tbtrans falls back to its own defaults and the transmission curve -- the one
        number this whole five-stage ladder exists to produce -- is reported over an
        interval nobody chose and that need not contain E_F.

        Contract: `engines/transport.md` § 2 (T(E) = Tr[Gamma_L G Gamma_R G+])
        + § 1 (the transmission stage's product).

        THIN: it asserts the keywords are PRESENT, not that their values bracket
        E_F, which is what makes the window right or wrong.
        """
        prep_calculation(calc, "transmission")
        text = (calc / "05_transmission" / "T_05_transmission.fdf"
                ).read_text()
        assert "TS.TBT.NumE" in text and "TS.TBT.Emin" in text

    def test_the_electronic_contract_is_the_citations(self, calc):
        """fdf-is-truth: the distinctive values in the cited deck land
        in every stage's deck -- none of them is a default."""
        prep_calculation(calc, "seed")
        prep_calculation(calc, "electrode_R")
        prep_calculation(calc, "device")
        seed = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        elec = (calc / "03_electrode_R" / "T_03_electrode_R.fdf"
                ).read_text()
        dev = (calc / "04_device" / "T_04_device.fdf").read_text()
        for text, who in ((seed, "seed"), (elec, "electrode"),
                          (dev, "device")):
            assert "PAO.BasisSize          TZP" in text, who
            assert "XC.authors             revPBE" in text, who
            assert "MeshCutoff             250 Ry" in text, who
            assert "PAO.EnergyShift        0.02 Ry" in text, who
            assert "ElectronicTemperature  200.0 K" in text, who
        # transverse k = the relaxation's (4, 4), transport axis 1
        assert "    0    0    1      0.0" in dev, (
            "the device kz is forced to 1 (open boundary)")
        assert "  4    0    0" in dev and "  4    0    0" in seed

    def test_the_emitters_order_preflight_never_fires(self, tmp_path):
        """THE P4 gate: a source whose atom order would trip the
        emitter's ordering error preps clean, because prep sorted."""
        from molbuilder.transport.stages import config_for  # noqa: F401
        from molbuilder.transport.transiesta import TransiestaEngine
        root = tmp_path / "projects"
        scrambled = _junction_struct(order="scrambled")
        _write_junction(root, scrambled)
        dest = _describe_transport(root)
        # the fixture genuinely trips the preflight when unsorted --
        # without this half, the test would pass on a tame fixture
        from molbuilder.config.transport import TransportConfig
        raw = TransiestaEngine.preflight(scrambled, TransportConfig())
        assert any("ordered" in i.message for i in raw
                   if i.severity == "error"), (
            "the scrambled fixture must be one the emitter refuses raw")
        prep_calculation(dest, "device")     # must not raise
        text = (dest / "04_device" / "T_04_device.fdf").read_text()
        assert "SolutionMethod         transiesta" in text

    def test_buffer_atoms_emit_ts_atoms_buffer(self, tmp_path):
        """SCIENCE. A junction carrying `buffer` atoms emits `TS.Atoms.Buffer` with the
        right ranges AND explicit electrode positions.

        Catches the second half being forgotten, which is the silent one.
        TranSIESTA's default electrode placement is "the first N and the last N
        atoms"; the categorical sort puts BUFFER atoms outermost, so that default
        now names buffer padding as the leads. The deck must therefore state
        `elec-pos begin 3` / `elec-pos end -3` explicitly. Emit the buffer block and
        not the positions and the run completes, having built the lead self-energies
        from atoms deliberately excluded from the NEGF region.

        Contract: `engines/transport.md` § 4 (the `buffer` label: atoms excluded
        from the NEGF region, placed outermost by the categorical sort).
        """
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct(buffers=True))
        dest = _describe_transport(root)
        prep_calculation(dest, "device")
        text = (dest / "04_device" / "T_04_device.fdf").read_text()
        assert "%block TS.Atoms.Buffer" in text
        # sorted layout: 2 buffers, 6 Au, 4 bridge, 6 Au, 2 buffers
        assert "atom [ 1 -- 2 ]" in text
        assert "atom [ 19 -- 20 ]" in text
        assert "elec-pos begin     3" in text
        assert "elec-pos end       -3" in text


class TestTheRecord:

    def test_written_once_and_reused_by_later_stages(self, calc):
        """The composed record is written ONCE: a later stage loads it rather than
        recomposing (asserted by mtime).

        Catches every stage re-deriving the junction from the citation. Five stages
        that each compose their own geometry can disagree -- a re-read of a live
        `.XV`, a re-sort, a re-numbering -- and then the electrode ranges the device
        deck names no longer address the atoms the electrode deck computed. One
        composition, reused, is what makes the five decks describe one system.

        Contract: `engines/transport.md` § 1 (one citation -> five derived stages)
        + § 5 (the consistency contract).
        """
        prep_calculation(calc, "seed")
        stamp = (calc / "junction.xyz").stat().st_mtime_ns
        prep_calculation(calc, "electrode_L")
        assert (calc / "junction.xyz").stat().st_mtime_ns == stamp, (
            "a later stage loads the record instead of recomposing")

    def test_a_repointed_citation_recomposes(self, calc, tmp_path):
        """The exception to "written once": re-pointing `task.json` at a DIFFERENT
        concluded attempt recomposes, and `slot-provenance.json` names the new one.

        Catches the cached record outliving the citation it came from. The two rules
        pull opposite ways -- reuse the record, but not when it answers a question
        nobody is asking any more -- and getting this half wrong is invisible: the
        user re-cites a better relaxation, prep succeeds, and every stage is built
        from the old geometry. The provenance file is what makes the answer auditable
        afterwards.

        Contract: `engines/transport.md` § 3.1 (what a citation names) + § 5.
        """
        prep_calculation(calc, "seed")
        # a second concluded attempt with a different relaxed geometry
        root = tmp_path / "projects"
        attempt = root / "J" / "optimization" / "Relax" / "01_coarse" \
            / "run-1"
        attempt.mkdir()
        from molbuilder.script_emit import emit_atom_metadata
        s2 = _junction_struct()
        coords = "\n".join(
            f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}  1"
            for p in s2.positions)
        store = {k: list(v) for k, v in s2.regions.items()}
        store["frozen_atoms"] = list(s2.frozen_atoms)
        blk = emit_atom_metadata(regions=store,
                                 n_atoms_total=len(s2.elements)) or ""
        (attempt / "Relax_01_coarse.fdf").write_text(
            _CITED_DECK
            + "AtomicCoordinatesFormat Ang\n"
            + "%block AtomicCoordinatesAndAtomicSpecies\n"
            + coords + "\n%endblock AtomicCoordinatesAndAtomicSpecies\n\n"
            + blk + "\n")
        (attempt / "Relax_01_coarse-run1.concluded").write_text("rc=0\n")
        _write_xv(attempt / "Relax.XV", s2, perturb_bridge=0.4)
        cite2 = "J/optimization/Relax/01_coarse/run-1"
        _describe_transport(root, cite=cite2)
        prep_calculation(calc, "seed")
        prov = json.loads((calc / "slot-provenance.json").read_text())
        assert prov["citation"] == cite2, (
            "task.json re-cited -> the old copy must not keep serving")

    def test_the_folder_travels(self, calc, tmp_path):
        """Prep once in the tree, move the folder to a tree WITHOUT the
        cited junction: the next stage preps from the record."""
        prep_calculation(calc, "seed")
        new_home = tmp_path / "elsewhere" / "projects" / "J" \
            / "transport" / "T"
        new_home.parent.mkdir(parents=True)
        shutil.move(str(calc), str(new_home))
        prep_calculation(new_home, "electrode_L")
        assert (new_home / "02_electrode_L" / "T_02_electrode_L.fdf"
                ).is_file()


class TestRefusals:

    def test_an_unnamed_stage_is_refused_naming_the_ladder(self, calc):
        """Prep with no stage named is refused, and the refusal LISTS the ladder.

        Catches a default. The composite has five stages that must run in order; if
        `prep_calculation` picked one (the first, the only enabled one) the user
        would get a prepped stage they did not ask for and would not know which. The
        message naming both ends of the ladder is what turns the refusal into the
        answer.

        Contract: `engines/transport.md` § 1 + § 3 (prep + launch the ladder, stage
        by stage).
        """
        with pytest.raises(PrepError) as e:
            prep_calculation(calc)
        msg = str(e.value)
        assert "seed" in msg and "transmission" in msg

    def test_an_unknown_stage_is_refused_by_name(self, calc):
        """A stage name that is not on this ladder is refused, quoting what was asked
        for.

        Catches a typo silently prepping something else -- "coarse" is a relaxation
        stage name, exactly the kind of name a user carries over from the
        calculation they cited. Quoting the rejected string is what tells them it was
        their word and not the ladder that was wrong.

        Contract: `engines/transport.md` § 1.
        """
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "coarse")
        assert "'coarse'" in str(e.value).replace('"', "'")

    def test_a_disabled_seed_refuses_with_the_skip_rule(self, calc):
        """Prepping a stage the description has DISABLED is refused, and the refusal
        cites the ruling that makes the seed skippable.

        Catches a disabled stage prepping anyway. The seed is the one stage that may
        legitimately be turned off (ruling Q4 -- the device can start cold), so
        `enabled: false` is a real choice a user makes; prepping it regardless would
        put a deck and a job-set row in the tree for a stage the DAG does not expect,
        and the gather then looks for a `.DM` that will never be produced.

        Contract: `engines/transport.md` § 1 + ruling Q4 (the seed is skippable),
        whose other half is `test_a_disabled_seed_drops_its_row`.
        """
        from molbuilder.task import (Stage, Task, derive_run, read_task,
                                     write_task)
        t = read_task(calc / "task.json")
        stages = tuple(Stage(name=s.name, enabled=(s.name != "seed"),
                             overrides={}) for s in t.stages)
        write_task(calc / "task.json", Task(
            engine=t.engine, shape=t.shape, run=t.run, structure=None,
            calculation="transport", slots=dict(t.slots), bias=t.bias,
            varies=(), stages=stages))
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed")
        assert "disabled" in str(e.value) and "Q4" in str(e.value)

    def test_a_sweep_is_refused_naming_the_bias_axis(self, calc):
        """A generic `sweep=` on a transport prep is refused, naming bias.

        Catches two axes of variation existing at once. Transport already has ONE
        axis -- the bias points -- with its own layout (a v-dir per point) and its
        own chained launch that hands `.TSDE` forward. A second, generic sweep would
        multiply against it into a directory shape nothing walks, and the chain
        would carry a converged density between points that differ in something
        other than voltage.

        Contract: `engines/transport.md` § 1; the bias axis is
        `archive/2026-09-01-transport-design.md` § 4.3.
        """
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed", sweep={"x": [1, 2]})
        assert "bias" in str(e.value)

    def test_a_moved_frozen_atom_stops_prep(self, calc, tmp_path):
        """SCIENCE. If the cited relaxation MOVED an atom that was declared frozen,
        prep refuses.

        Catches the geometric premise of the whole composite being false. The
        electrode atoms are frozen because they must stay a pristine bulk slab --
        that is what lets the same lattice be used for the separate bulk-lead run
        the self-energies come from. If the relaxation moved one, the "bulk"
        electrode in the device is no longer the bulk the lead run computes Sigma
        for, so the leads are matched to a material that is not there. Nothing
        downstream can see this: the deck renders, the run converges, and the
        transmission is wrong.

        Contract: `engines/transport.md` § 2 (Sigma comes from a separate pristine
        bulk-lead run; frozen is the geometry constraint that keeps the two the same)
        + § 4 (the electrode regions are BULK metal only).
        """
        root = tmp_path / "projects"
        _write_xv(root / "J/optimization/Relax/01_coarse/run-0/Relax.XV",
                  _junction_struct(), perturb_electrode=(0, 0.05))
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed")
        assert "MOVED" in str(e.value)

    def test_a_pseudo_the_citation_cannot_supply_is_named(self, calc,
                                                          tmp_path):
        """SCIENCE. A pseudopotential missing from the cited directory stops prep, and
        the refusal names the file.

        Catches the five stages being built on a different pseudopotential from the
        relaxation they cite. The pseudopotential defines the effective nuclear
        potential and the valence partitioning -- change it and the energies are not
        comparable to the geometry that was relaxed with it. § 3.1's rule is that a
        citation names a DIRECTORY and the directory must hold what the stage
        consumes, so the failure mode without this gate is a silent fallback to a
        system-wide `.psml` that happens to be on the path.

        Contract: `engines/transport.md` § 3.1 (files, not layout) +
        `science/pseudopotentials.md`.
        """
        (tmp_path / "projects" / "J" / "optimization" / "Relax"
         / "01_coarse" / "run-0" / "Au.psml").unlink()
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed")
        assert "Au.psml" in str(e.value)


# --------------------------------------------------------------------- #
#  P5a — the launch side: the DAG gather, the warm rows, the binary      #
# --------------------------------------------------------------------- #

_TOKEN = {"seed": "01_seed", "electrode_L": "02_electrode_L",
          "electrode_R": "03_electrode_R", "device": "04_device",
          "transmission": "05_transmission"}


def _conclude(calc, stage, files, *, deck_text=None, point=None):
    """Simulate a concluded run-0 of *stage*: the attempt holds the
    stage's own rendered deck (or ``deck_text`` to fake a STALE one),
    the conclusion marker, and the named product files.  ``point``
    targets one bias point's v-dir (the scan layout)."""
    token = _TOKEN[stage]
    stem = f"T_{token}"
    stage_dir = calc / token if point is None else calc / token / point
    attempt = stage_dir / "run-0"
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / f"{stem}.fdf").write_text(
        deck_text if deck_text is not None
        else (stage_dir / f"{stem}.fdf").read_text())
    (attempt / f"{stem}-run0.concluded").write_text("rc=0\n")
    for name in files:
        (attempt / name).write_bytes(b"\0binary\0")
    return attempt


class TestTheGather:
    """`gather_transport_inputs` — the § 4.2 DAG's inputs, copied in at
    prep with three gates per input (P5)."""

    def _task(self, calc):
        from molbuilder.task import read_task
        return read_task(calc / "task.json")

    def test_device_gathers_dm_and_both_tshs(self, calc, tmp_path):
        """SCIENCE. The device gather copies in exactly three inputs -- the seed's
        `T.DM` and BOTH electrodes' `.TSHS` -- and records where each came from.

        Catches a device run starting without one of its leads. The `.TSHS` files
        are the pristine bulk H/S the self-energies Sigma_L and Sigma_R are built
        from; gather one and not the other and TranSIESTA is asked for a
        two-terminal calculation with one terminal. `.gathered-from` is the audit
        trail: without it, a device attempt cannot afterwards be traced to the
        electrode runs whose numbers it depends on.

        Contract: `engines/transport.md` § 2 (three runs; Sigma from the separate
        bulk runs) + § 6 (the pieces and data flow).
        """
        from molbuilder.jobset.prep import gather_transport_inputs
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        dest = tmp_path / "device-attempt"
        dest.mkdir()
        got = gather_transport_inputs(calc, self._task(calc), "device",
                                      dest)
        names = sorted(fn for _src, fn in got)
        assert names == ["T.DM", "T_L-electrode.TSHS",
                         "T_R-electrode.TSHS"]
        for n in names:
            assert (dest / n).is_file()
        record = (dest / ".gathered-from").read_text()
        assert "T_L-electrode.TSHS <- 02_electrode_L/run-0" in record

    def test_device_before_electrodes_conclude_is_refused_by_name(
            self, calc, tmp_path):
        """THE P5 gate row: device before the electrodes conclude is a
        named refusal, never a wait and never an auto-run."""
        from molbuilder.jobset.prep import gather_transport_inputs
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])
        # electrode_L has an OPEN attempt -- deck in place, product even
        # written, but no conclusion marker: launched-and-still-running
        # (or force-stopped) must refuse exactly like never-launched.
        stage_dir = calc / "02_electrode_L"
        run0 = stage_dir / "run-0"
        run0.mkdir()
        shutil.copy2(stage_dir / "T_02_electrode_L.fdf",
                     run0 / "T_02_electrode_L.fdf")
        (run0 / "T_L-electrode.TSHS").write_bytes(b"\0half-written\0")
        dest = tmp_path / "d"
        dest.mkdir()
        with pytest.raises(PrepError) as e:
            gather_transport_inputs(calc, self._task(calc), "device", dest)
        msg = str(e.value)
        assert "electrode_L" in msg and "CONCLUDED" in msg
        assert "launch run electrode_L" in msg, (
            "strict composition: the refusal names what to run first")

    def test_an_unprepped_upstream_is_refused_by_name(self, calc,
                                                      tmp_path):
        """Gathering for the device when an upstream stage was never prepped is a named
        refusal.

        Catches the gather silently producing an empty input set. "Never prepped" and
        "prepped but still running" are different states with different advice, and
        the one that must never happen is either of them becoming "gathered nothing
        and carried on" -- a device attempt with no `.TSHS` in it launches and dies
        on the cluster hours later.

        Contract: `engines/transport.md` § 3 (each prep gathers what the stage
        consumes from the CONCLUDED stages before it).
        """
        from molbuilder.jobset.prep import gather_transport_inputs
        dest = tmp_path / "d"
        dest.mkdir()
        with pytest.raises(PrepError) as e:
            gather_transport_inputs(calc, self._task(calc), "device", dest)
        assert "has not been prepped" in str(e.value)

    def test_a_stale_upstream_attempt_is_refused(self, calc, tmp_path):
        """A concluded attempt of a DIFFERENT deck answers a different
        calculation -- the gather must skip it and say why."""
        from molbuilder.jobset.prep import gather_transport_inputs
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"],
                  deck_text="SystemLabel T_L-electrode\n# an OLD render\n")
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        dest = tmp_path / "d"
        dest.mkdir()
        with pytest.raises(PrepError) as e:
            gather_transport_inputs(calc, self._task(calc), "device", dest)
        assert "none ran the deck this composition renders" in str(e.value)

    def test_a_concluded_attempt_missing_its_product_is_refused(
            self, calc, tmp_path):
        """A stage that concluded but did not write its named product is refused,
        saying which file is missing.

        Catches "it finished" being mistaken for "it produced". The conclusion marker
        records that the wrapper exited, not that SIESTA wrote a density -- an SCF
        that hit its iteration limit, or a run killed after the last write, leaves a
        concluded attempt with no `T.DM`. Without this gate the device gathers
        nothing from the seed and warm-starts from a file that is not there.

        Contract: `engines/transport.md` § 3 (gather from the concluded stages
        before it); the sibling gates are stale-deck and not-yet-concluded.
        """
        from molbuilder.jobset.prep import gather_transport_inputs
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", [])                    # no T.DM written
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        dest = tmp_path / "d"
        dest.mkdir()
        with pytest.raises(PrepError) as e:
            gather_transport_inputs(calc, self._task(calc), "device", dest)
        assert "did not write T.DM" in str(e.value)

    def test_a_disabled_seed_drops_its_row(self, calc, tmp_path):
        """Ruling Q4: the seed is skippable -- a disabled seed is not a
        missing dependency."""
        from molbuilder.jobset.prep import gather_transport_inputs
        from molbuilder.task import (Stage, Task, derive_run, read_task,
                                     write_task)
        for st in ("electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        t = read_task(calc / "task.json")
        stages = tuple(Stage(name=s.name, enabled=(s.name != "seed"),
                             overrides={}) for s in t.stages)
        task2 = Task(engine=t.engine, shape=t.shape, run=t.run,
                     structure=None, calculation="transport",
                     slots=dict(t.slots), bias=t.bias, varies=(),
                     stages=stages)
        dest = tmp_path / "d"
        dest.mkdir()
        got = gather_transport_inputs(calc, task2, "device", dest)
        assert sorted(fn for _s, fn in got) == [
            "T_L-electrode.TSHS", "T_R-electrode.TSHS"]

    def test_transmission_gathers_the_device_products(self, calc,
                                                      tmp_path):
        """SCIENCE. The transmission gather takes the device's `T.TS.HSX` plus both
        electrode `.TSHS` -- and NOT the `.TSDE`.

        Catches tbtrans being fed the wrong file. SIESTA 5.x writes the converged
        device Hamiltonian as `TS.HSX`; the `.TSDE` is the density matrix used to
        warm-start the next bias point, not the H/S tbtrans evaluates T(E) from. The
        distinction was measured live on 2026-08-29, and the failure it prevents is
        a transmission computed from the wrong matrix rather than an error.

        Contract: `engines/transport.md` § 2 (T(E) is built from the device G and
        the leads' Gamma) + § 6.
        """
        from molbuilder.jobset.prep import gather_transport_inputs
        for st in ("electrode_L", "electrode_R", "device"):
            prep_calculation(calc, st)
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        _conclude(calc, "device", ["T.TS.HSX"])
        dest = tmp_path / "t"
        dest.mkdir()
        got = gather_transport_inputs(calc, self._task(calc),
                                      "transmission", dest)
        assert sorted(fn for _s, fn in got) == [
            "T.TS.HSX", "T_L-electrode.TSHS",
            "T_R-electrode.TSHS"], (
            "SIESTA 5.x writes the device H as TS.HSX; tbtrans consumes "
            "it plus the electrode .TSHS -- never the .TSDE (measured "
            "live 2026-08-29)")


class TestTheLaunchSide:

    def test_the_transmission_wrapper_launches_tbtrans(self, calc):
        """The same deck text, a different program: the binary rides
        Resources.program into the wrapper (P5)."""
        prep_calculation(calc, "transmission")
        prep_calculation(calc, "seed")
        trans = (calc / "05_transmission" / "T_05_transmission.run.sh"
                 ).read_text()
        seed = (calc / "01_seed" / "T_01_seed.run.sh").read_text()
        assert '_siesta_target="tbtrans"' in trans
        assert "command -v tbtrans" in trans
        assert "tbtrans" not in seed, (
            "only the transmission stage routes to tbtrans")

    def test_the_device_declares_its_tsde_warm_row(self, calc):
        """The device job declares `T.TSDE` and `T.DM` as warm-start rows; an electrode
        single-point declares none.

        Catches the warm-start bookkeeping being applied uniformly. The device is the
        expensive stage and the only one where resuming from a previous density pays;
        an electrode is a cheap single-point where re-running beats reasoning about
        whether a half-written copy is trustworthy. Declaring warm rows for a stage
        that should start clean is how a corrupt density silently seeds a run.

        Contract: `engines/transport.md` § 1 + § 6 (the data flow between stages).
        """
        prep_calculation(calc, "device")
        prep_calculation(calc, "electrode_L")
        js = json.loads((calc / "job-set.json").read_text())
        rows = {j["name"]: j for j in js["jobs"]}
        device_warm = [w["name"] for w in rows["device"].get("warm", [])]
        assert "T.TSDE" in device_warm and "T.DM" in device_warm
        assert rows["electrode_L"].get("warm", []) == [], (
            "an electrode single-point declares nothing -- re-running "
            "is cheaper than reasoning about a half-finished copy")

    def test_the_device_deck_honours_the_seed_dm(self, calc):
        """SCIENCE. The device deck sets `DM.UseSaveDM true`.

        Catches the seed stage being pointless. SIESTA's default is FALSE, so
        gathering `T.DM` into the device attempt puts the file in place and leaves
        it unread -- the TranSIESTA SCF starts from scratch, the run costs what the
        seed was meant to save, and nothing anywhere reports that the warm start did
        not happen. The whole seed rung's justification is this one keyword.

        Contract: `engines/transport.md` § 1 (the seed produces the density the
        device starts from) + § 6.
        """
        prep_calculation(calc, "device")
        text = (calc / "04_device" / "T_04_device.fdf").read_text()
        assert "DM.UseSaveDM           true" in text, (
            "SIESTA's default is false -- without the keyword the "
            "seed's density would sit present but not honoured")


class TestTheCliRoute:
    """`molbuilder jobset prep run <stage>` on a transport calc — the
    template gate opens for task.json alone, and the tail gathers."""

    def _invoke(self, args, root, monkeypatch):
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        from molbuilder.projects import PROJECTS_ROOT_ENV
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        return CliRunner().invoke(jobset_group, args)

    def test_prep_routes_without_a_template(self, calc, tmp_path,
                                            monkeypatch):
        """The CLI preps a transport stage from `task.json` ALONE -- no template -- and
        opens the attempt directory like any other rung.

        Catches the template gate refusing the composite. Every other calculation
        preps through a template; transport has none by design (the electronic
        contract comes from the cited deck, § 5), so the gate has to open for a
        task.json with no template beside it. Get that wrong and the arm is
        unreachable from the command line while every direct-call test in this file
        still passes.

        Contract: `engines/transport.md` § 3 (the CLI) + § 5 (one template governs,
        and it is the citation's).
        """
        r = self._invoke(["prep", "run", "seed", "--bundle",
                          "J/transport/T"], tmp_path / "projects",
                         monkeypatch)
        assert r.exit_code == 0, r.output
        assert (calc / "01_seed" / "run-0" / "T_01_seed.fdf").is_file(), (
            "the CLI tail opens the attempt like any ladder rung")

    def test_prep_device_gathers_through_the_cli(self, calc, tmp_path,
                                                 monkeypatch):
        """The CLI route gathers too -- and for a BIAS SCAN it opens one attempt ladder
        per bias point, each with its own deck, wrapper and gathered inputs.

        Catches the gather living only in the Python door. The CLI is what a person
        actually runs; a route that renders the decks and skips the gather produces
        per-point attempts that look complete and contain no `.TSHS`. The per-point
        assertion is the sharper half: a single shared attempt for a two-point scan
        would pass any "did it gather" check and then run both voltages in one
        directory, overwriting the first point's results with the second's.

        Contract: `engines/transport.md` § 3 (a bias scan launches as one chain job
        walking the points); the layout is `archive/2026-09-01-transport-design.md`
        § 4.3.
        """
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        r = self._invoke(["prep", "run", "device", "--bundle",
                          "J/transport/T"], tmp_path / "projects",
                         monkeypatch)
        assert r.exit_code == 0, r.output
        assert "gathered: T_L-electrode.TSHS" in r.output
        # bias=(0.0, 0.2) is a SCAN, so the CLI opens one attempt ladder
        # PER POINT (layout ruled 2026-08-29) and gathers into each.
        for point in ("v0", "v0.2"):
            run0 = calc / "04_device" / point / "run-0"
            for n in ("T.DM", "T_L-electrode.TSHS",
                      "T_R-electrode.TSHS", "T_04_device.fdf",
                      "T_04_device.run.sh"):
                assert (run0 / n).is_file(), f"{point}/{n}"
        assert "prepared device @ v0.2" in r.output

    def test_prep_device_refuses_through_the_cli_too(self, calc,
                                                     tmp_path,
                                                     monkeypatch):
        """The CLI refuses an unready device with the same named refusal the Python door
        gives, and a non-zero exit code.

        Catches the refusal being swallowed at the CLI boundary. A door that raises
        and a command that prints the reason and exits 0 are different things: a
        script driving the ladder reads the exit code, so a zero here means the
        caller proceeds to launch a device attempt that was never prepared.

        Contract: `engines/transport.md` § 3 (prep + launch stage by stage).
        """
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])      # electrodes stay unconcluded
        r = self._invoke(["prep", "run", "device", "--bundle",
                          "J/transport/T"], tmp_path / "projects",
                         monkeypatch)
        assert r.exit_code != 0
        assert "electrode_L" in r.output and "CONCLUDED" in r.output

    def test_prep_bench_on_transport_is_refused(self, calc, tmp_path,
                                                monkeypatch):
        """`prep bench` on a transport calculation is refused.

        Catches a benchmark ladder being built for the composite. Benchmarking sizes
        a run by scaling one job across widths; the transport stages are five
        DIFFERENT calculations with a dependency order and a chained bias walk, so
        "benchmark it" has no meaning here -- and silently producing something would
        put bench rungs in a job-set the transport launcher then tries to walk.

        Contract: `engines/transport.md` § 1 (the composite's shape).
        """
        r = self._invoke(["prep", "bench", "device", "--bundle",
                          "J/transport/T"], tmp_path / "projects",
                         monkeypatch)
        assert r.exit_code != 0
        assert "no benchmark" in r.output

    def test_init_refuses_the_flat_shape(self, calc, tmp_path,
                                         monkeypatch):
        """`jobset init --calculation transport --shape flat` is refused, naming
        hierarchical.

        Catches the composite being described in a layout that cannot hold it. Flat
        puts every stage's files in ONE directory distinguished by a filename token;
        the five transport stages each need their own directory (and the device
        needs a v-dir per bias point beneath it), so a flat transport description
        would collide five decks and their attempts in one place. Refusing at
        `init` is the only cheap moment -- after that there is a tree to unpick.

        Contract: `engines/transport.md` § 1 + `execution/project-layout.md`
        (flat vs hierarchical).
        """
        r = self._invoke(["init", "--calculation", "transport",
                          "--shape", "flat",
                          "--bundle", "J/transport/T2",
                          "--slot", f"junction={_CITE}"],
                         tmp_path / "projects", monkeypatch)
        assert r.exit_code != 0
        assert "hierarchical" in r.output


class TestTheBiasScan:
    """P5b — the bias chain (archive/2026-09-01-transport-design.md § 4.3; layout ruled
    2026-08-29: plain v-dirs, one attempt ladder per point, one
    submission walking them with the .TSDE handed forward)."""

    def _ready(self, calc, tmp_path, monkeypatch, *, bias=None):
        """Upstreams concluded, device prepped through the CLI (per-point
        attempts open + gathered).  Returns (task, jobset)."""
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        from molbuilder.jobset.model import JobSet
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.task import read_task
        if bias is not None:
            root = tmp_path / "projects"
            _describe_transport(root, bias=bias)
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "projects"))
        r = CliRunner().invoke(jobset_group,
                               ["prep", "run", "device", "--bundle",
                                "J/transport/T"])
        assert r.exit_code == 0, r.output
        return read_task(calc / "task.json"), JobSet.load(
            calc / "job-set.json")

    def _stub(self, calc, point):
        """Replace one point's wrapper with a stub that records the run
        order and the density it STARTED with, then writes its own."""
        att = calc / "04_device" / point / "run-0"
        (att / "T_04_device.run.sh").write_text(
            "#!/bin/bash\n"
            'echo "$(basename $(dirname $(dirname $PWD)))'
            f'/{point}" >> ../../chain-order.log\n'
            "if [ -f T.TSDE ]; then cp T.TSDE TSDE-at-start; fi\n"
            f'echo "density-from-{point}" > T.TSDE\n')

    def test_the_points_render_their_own_decks(self, calc):
        """SCIENCE. Each bias point renders its OWN deck carrying its OWN
        `TS.Voltage`, and the stage-directory deck is the equilibrium point's.

        Catches every point running at the same voltage. The bias is the independent
        variable of the whole scan -- the difference in the leads' chemical
        potentials, mu_L - mu_R -- so a per-point directory whose deck still says
        0.0 eV produces a set of identical equilibrium results labelled as an I-V
        curve. Nothing about the output would look wrong. The stage-directory copy
        being the equilibrium point's matters too: it is what a person opens to read
        the deck, and it must not be an arbitrary point's.

        Contract: `engines/transport.md` § 2 (each lead carries a chemical
        potential); the layout is `archive/2026-09-01-transport-design.md` § 4.3.
        """
        prep_calculation(calc, "device")
        v0 = (calc / "04_device" / "v0" / "T_04_device.fdf").read_text()
        v2 = (calc / "04_device" / "v0.2" / "T_04_device.fdf").read_text()
        top = (calc / "04_device" / "T_04_device.fdf").read_text()
        assert "TS.Voltage             0.0000 eV" in v0
        assert "TS.Voltage             0.2000 eV" in v2
        assert "TS.Voltage             0.0000 eV" in top, (
            "the stage-dir deck is the equilibrium point's")
        assert (calc / "04_device" / "v0.2" / "T_04_device.run.sh"
                ).is_file(), "each point carries its own wrapper"

    def test_a_single_point_keeps_the_plain_layout(self, tmp_path):
        """A single-point (equilibrium-only) description does NOT grow a v-dir layer.

        Catches the per-point directory becoming unconditional. Most transport runs
        are one voltage; wrapping them in a `v0/` subdirectory would change the path
        of every deck and product for the common case, and every downstream reader
        -- the gather, the results presenter, a person -- would have to know which
        shape it was looking at. The v-dir layer exists for the AXIS, not for every
        run.

        Contract: `archive/2026-09-01-transport-design.md` § 4.3 (layout ruled
        2026-08-29).
        """
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        dest = _describe_transport(root, bias=(0.0,))
        prep_calculation(dest, "device")
        assert (dest / "04_device" / "T_04_device.fdf").is_file()
        assert not (dest / "04_device" / "v0").exists(), (
            "the v-dir layer exists for the axis, not for every run")

    def test_the_chain_warm_chains(self, calc, tmp_path, monkeypatch):
        """THE P5 gate: a two-point bias fixture warm-chains -- the
        second point STARTS with the first point's .TSDE."""
        from molbuilder.jobset.submit import submit_transport_chain
        task, js = self._ready(calc, tmp_path, monkeypatch)
        self._stub(calc, "v0")
        self._stub(calc, "v0.2")
        results = submit_transport_chain(js, calc, task, mode="direct")
        assert results[0].status == "ran", results
        order = (calc / "04_device" / "chain-order.log"
                 ).read_text().splitlines()
        assert [o.split("/")[-1] for o in order] == ["v0", "v0.2"], (
            "the chain walks the points in the description's order")
        seen = (calc / "04_device" / "v0.2" / "run-0" / "TSDE-at-start")
        assert seen.is_file(), "point 2 must START with a .TSDE"
        assert seen.read_text().strip() == "density-from-v0"
        for point in ("v0", "v0.2"):
            assert (calc / "04_device" / point / "run-0" / "run.json"
                    ).is_file(), "every point is launched by this command"

    def test_the_chain_stops_on_a_failed_point(self, calc, tmp_path,
                                               monkeypatch):
        """SCIENCE. When a bias point FAILS, the chain stops -- the later points are not
        walked -- and the failure's return code is reported.

        Catches the walk continuing past a failure. Each point warm-starts from the
        PREVIOUS point's `.TSDE`, so point 3 after a failed point 2 would either
        start from point 1's density (a voltage jump the continuation was designed to
        avoid) or from whatever half-written file the failed run left behind. Both
        produce a converged-looking answer at the wrong bias. The three-point fixture
        is what makes "stopped" distinguishable from "there was nothing left to do":
        the log must hold exactly one entry.

        Contract: `archive/2026-09-01-transport-design.md` § 4.3 (one submission
        walking the points with the .TSDE handed forward).
        """
        from molbuilder.jobset.submit import submit_transport_chain
        from molbuilder.task import read_task
        # a three-point scan: rewrite the description (the id derives
        # from the citation, so the bias edit keeps it)
        _describe_transport(tmp_path / "projects", bias=(0.0, 0.2, 0.4))
        task, js = self._ready(calc, tmp_path, monkeypatch)
        self._stub(calc, "v0")
        att = calc / "04_device" / "v0.2" / "run-0"
        (att / "T_04_device.run.sh").write_text(
            "#!/bin/bash\nexit 7\n")
        self._stub(calc, "v0.4")
        results = submit_transport_chain(js, calc, task, mode="direct")
        assert results[0].status == "failed"
        assert results[0].returncode == 7
        order = (calc / "04_device" / "chain-order.log"
                 ).read_text().splitlines()
        assert len(order) == 1, (
            "later points chain their density from the failed one -- "
            "the walk must stop, not continue")

    def test_an_unprepped_point_refuses_the_chain(self, calc, tmp_path,
                                                  monkeypatch):
        """A chain launch with one point unprepped is refused, naming the point and the
        command that fixes it.

        Catches the chain launching a partial scan. The submission walks every point
        in one job; a missing attempt directory discovered mid-walk would mean the
        earlier points have already run and the user has a half-finished scan with no
        single command to resume it. Refusing before anything runs is the difference.

        Contract: `archive/2026-09-01-transport-design.md` § 4.3.
        """
        from molbuilder.jobset.submit import (SubmitError,
                                              submit_transport_chain)
        task, js = self._ready(calc, tmp_path, monkeypatch)
        shutil.rmtree(calc / "04_device" / "v0.2" / "run-0")
        with pytest.raises(SubmitError) as e:
            submit_transport_chain(js, calc, task, mode="direct")
        assert "v0.2" in str(e.value) and "prep run device" in str(e.value)

    def test_a_launched_point_refuses_relaunch(self, calc, tmp_path,
                                               monkeypatch):
        """A point whose attempt already carries a `run.json` refuses relaunch, citing
        immutability.

        Catches results being overwritten in place. An attempt directory is immutable
        once launched -- that is what makes a result traceable to the deck that
        produced it -- and a chain that re-walked a launched point would write new
        output over old in the same directory, leaving a run.json and a set of
        products that came from two different launches.

        Contract: `execution/running-a-job.md` (the attempt is immutable once
        launched) + `archive/2026-09-01-transport-design.md` § 4.3.
        """
        from molbuilder.jobset.submit import (SubmitError,
                                              submit_transport_chain)
        task, js = self._ready(calc, tmp_path, monkeypatch)
        (calc / "04_device" / "v0" / "run-0" / "run.json").write_text("{}")
        with pytest.raises(SubmitError) as e:
            submit_transport_chain(js, calc, task, mode="direct")
        assert "immutable" in str(e.value)

    def test_transmission_gathers_the_matching_point(self, calc,
                                                     tmp_path,
                                                     monkeypatch):
        """The transmission at v reads the DEVICE at v -- never another
        point's converged state."""
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        self._ready(calc, tmp_path, monkeypatch)
        _conclude(calc, "device", ["T.TS.HSX"], point="v0")
        _conclude(calc, "device", ["T.TS.HSX"], point="v0.2")
        r = CliRunner().invoke(jobset_group,
                               ["prep", "run", "transmission",
                                "--bundle", "J/transport/T"])
        assert r.exit_code == 0, r.output
        rec = (calc / "05_transmission" / "v0.2" / "run-0"
               / ".gathered-from").read_text()
        assert "T.TS.HSX <- 04_device/v0.2/run-0" in rec
        assert (calc / "05_transmission" / "v0.2" / "run-0" / "T.TS.HSX"
                ).is_file()


class TestTheOverrideLane:
    """Transport-only knobs travel as stage overrides (P7b): the
    composite has no template, so the stages' own bags are the
    description's one place for the transmission window, the contour --
    everything that is NOT the citation's to say."""

    def _with_override(self, calc, stage_name, overrides):
        from molbuilder.task import (Stage, Task, derive_run, read_task,
                                     write_task)
        t = read_task(calc / "task.json")
        stages = tuple(Stage(name=s.name, enabled=s.enabled,
                             overrides=(overrides if s.name == stage_name
                                        else {}))
                       for s in t.stages)
        write_task(calc / "task.json", Task(
            engine=t.engine, shape=t.shape, run=t.run, structure=None,
            calculation="transport", slots=dict(t.slots), bias=t.bias,
            varies=tuple(sorted(overrides)), stages=stages))

    def test_a_knob_override_lands_in_the_deck(self, calc):
        """A transport-only knob set as a stage override reaches the rendered deck.

        Catches the override lane being inert. The composite has no template, so a
        stage's `overrides` bag is the description's ONLY place to say anything the
        citation does not own -- the transmission window, the contour. An override
        that is accepted, written into task.json, and then not rendered gives the
        user a description that reads as configured and a deck that is at defaults.

        Contract: `engines/transport.md` § 5 (the invariant set is the citation's;
        everything else is the description's).
        """
        self._with_override(calc, "device",
                            {"transmission_n_points": 101})
        prep_calculation(calc, "device")
        text = (calc / "04_device" / "T_04_device.fdf").read_text()
        assert "TS.TBT.NumE            101" in text

    def test_a_contract_field_is_sealed(self, calc):
        """SCIENCE. A CONTRACT field -- here `basis_size` -- cannot be overridden on a
        stage when the citation is a concluded relaxation; the refusal says it is
        the citation's to say.

        Catches the consistency contract being broken one keyword at a time. The
        invariant set (basis, XC, energy shift, mesh, k, electronic temperature) must
        be IDENTICAL across the seed, both electrodes and the device, because the
        electrode `.TSHS` and the device Hamiltonian are matched matrices: a device
        computed in TZP against leads computed in DZP does not just lose accuracy, it
        mismatches the basis the self-energies are expressed in. Sealing the field is
        what makes "one template governs" enforceable rather than advisory.

        Contract: `engines/transport.md` § 5 (the consistency contract -- the
        invariant set) + § 3.1 (fdf-is-truth: the contract is read from the cited
        deck).
        """
        self._with_override(calc, "device", {"basis_size": "DZP"})
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "device")
        assert "citation's to say" in str(e.value)

    def test_an_unknown_knob_is_refused_by_name(self, calc):
        """A misspelled override key is refused, quoting the key.

        Catches a typo being silently ignored. `n_pionts` for `n_points` is accepted
        by any dict, written into task.json, and then does nothing -- the user
        believes they set the transmission grid and the deck renders at the default.
        Quoting the offending key is what turns the refusal into a fix.

        Contract: `engines/transport.md` § 5.
        """
        self._with_override(calc, "transmission", {"n_pionts": 7})
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "transmission")
        assert "'n_pionts'" in str(e.value).replace('"', "'")


class TestFormBContract:
    """4.1b: a labeled-pair citation has no deck, so the electronic
    contract is the description's own -- CONTRACT_FIELDS become
    ordinary overrides and land in the rendered deck."""

    def test_an_open_contract_field_reaches_the_deck(self, tmp_path):
        """SCIENCE, and the other half of the seal. When the citation is a labeled
        PAIR (form B) rather than a concluded run, `basis_size` is an ORDINARY
        override and must reach the deck.

        Catches the seal being applied unconditionally. A pair has no deck, so there
        is no cited electronic contract to defer to -- if the field stayed sealed
        there would be no way to state the basis at all, and every form-B transport
        calculation would render at the default basis with the user unable to change
        it. The rule is not "these fields are frozen", it is "these fields belong to
        the citation when the citation HAS them".

        Contract: `engines/transport.md` § 3.1 (what makes a directory citable --
        form A vs form B) + § 5 (the invariant set).
        """
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        pair = root / "J" / "structure" / "junc"
        pair.mkdir(parents=True)
        StructureCodec().write(_junction_struct(), pair / "junction.xyz")
        write_pseudos(pair, ["Au", "S", "C"])
        dest = _describe_transport(root, cite="J/structure/junc")
        # basis_size is CONTRACT -- sealed for form A, open here.
        import json as _json
        t = _json.loads((dest / "task.json").read_text())
        for st in t["stages"]:
            if st["name"] == "seed":
                st["overrides"] = {"basis_size": "TZP"}
        t["varies"] = ["basis_size"]
        (dest / "task.json").write_text(_json.dumps(t, indent=2) + "\n")
        prep_calculation(dest, "seed")
        deck = (dest / "01_seed" / "T_01_seed.fdf").read_text()
        assert "TZP" in deck, "the open contract field must reach the deck"

    def test_the_same_field_stays_sealed_for_a_relaxation(self, calc):
        """SCIENCE. The paired negative: the SAME field, on a form-A citation, is still
        sealed.

        Catches the form-B opening being implemented as "stop sealing". The two
        tests are only meaningful together -- either alone is satisfied by a
        constant answer -- and the failure this one catches is the dangerous
        direction: a description silently re-specifying the basis the cited
        relaxation was run at, so the device Hamiltonian and the geometry it uses
        come from different electronic structures.

        Contract: `engines/transport.md` § 5 + § 3.1.
        """
        import json as _json
        t = _json.loads((calc / "task.json").read_text())
        for st in t["stages"]:
            if st["name"] == "seed":
                st["overrides"] = {"basis_size": "SZ"}
        t["varies"] = ["basis_size"]
        (calc / "task.json").write_text(_json.dumps(t, indent=2) + "\n")
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed")
        assert "citation's to say" in str(e.value)

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
import re
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


def _says(text: str, keyword: str, value: str) -> bool:
    """Does *text* set *keyword* to *value*?  **Whitespace-insensitive.**

    The column alignment of a deck line belongs to whoever wrote it: the
    hand-written transport emitters padded to a fixed column, the framework's
    syntax door (`siesta/layout.py::line`) does not, and libfdf cares about
    neither.  Asserting the PAIR rather than the spacing is what lets these
    tests mean the same thing before and after a rung moves onto the seam
    (`engines/transport.md` § 3.6) -- otherwise every migrated rung breaks a
    science assertion for a reason that has nothing to do with science.
    """
    want = (keyword + " " + value).split()
    head = re.escape(want[0])
    for line in text.splitlines():
        got = line.split()
        if not got or not re.fullmatch(head, got[0]) or len(got) != len(want):
            continue
        if all(_same_token(a, b) for a, b in zip(got[1:], want[1:])):
            return True
    return False


def _has_row(text: str, row: str) -> bool:
    """Does *text* hold the block row *row*?  **Whitespace-insensitive**, as
    `_says` is for a keyword line: a k block's columns are its writer's
    (`kmesh.write`), and the counts and the offset are what a test means."""
    return any(" ".join(ln.split()) == row for ln in text.splitlines())


def _same_token(got: str, want: str) -> bool:
    """One token of a deck line, compared by VALUE where it is a number.

    `250` and `250.0` are the same mesh cutoff.  They differ because the
    framework's syntax door formats from the item's DECLARED type (a float)
    while the hand-written emitters it is replacing formatted the Python
    value they happened to hold (an int) -- so during the migration one
    rung says one and another says the other, for a value they agree on.

    Asserting the spelling would make a test fail for a reason that has
    nothing to do with the science, which is the same trap the column
    alignment set (see `_says`).
    """
    try:
        return float(got) == float(want)
    except ValueError:
        return got == want


#: The fixture's leads are a CHAIN, one gold atom per layer, so they are
#: ISOLATED across the transport axis in the 8 Å box they sit in -- what the
#: relaxation deck records and `compose` carries (`engines/transport.md`
#: § 6.1c).  One layer spacing is the room the transport boundary leaves.
_ACROSS = ("isolated", "isolated")
_SPACING = _LAYERS_L[1] - _LAYERS_L[0]


def _junction_struct(*, order="canonical", buffers=False, across=_ACROSS,
                     width=8.0, room=None):
    """The BDT-ish fixture sandwich; ``order="scrambled"`` writes the
    same geometry with the bridge FIRST and the leads swapped after it
    — exactly the order the emitter's preflight refuses.

    *across* and *width* are the transverse axes' kinds and length: the
    chain in an 8 Å box is a wire, isolated across; ``across=("periodic",
    "periodic"), width=_SPACING`` is the same chain as a lattice it tiles --
    the reading a relaxation deck from before the placement record gets
    (`compose._junction_axis_kind`).  *room* is what the transport boundary
    leaves, one layer spacing unless a test opens it."""
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
    # THE CELL MUST CONTAIN THE ATOMS, and with buffers it did not: the
    # buffer padding sits at z = -5 .. 39.5, a 44.5 A span in a 40 A box, so
    # atoms overlapped their own periodic images along the transport axis.
    # It went unnoticed because the device rung had NO settings gate until
    # TR5b put it on the seam -- the first time anything looked.  Sized from
    # the geometry so it cannot drift again.
    #
    # AND IT IS THE STRUCTURE A TRANSPORT RUN CAN USE (M5 step 2, § 6.1c).
    # The leads continue through the transport boundary into the image, so
    # the room there is ONE of the lead's layer spacings -- this fixture left
    # 5.5 A against a 2.5 A spacing, a missing layer the gate now refuses.
    # And the leads are a CHAIN, one gold atom per layer, in an 8 A box: a
    # wire, isolated across transport, which is what it states -- periodic
    # there would say the chain tiles a plane it does not.
    zs = positions[:, 2]
    c = float(zs.max() - zs.min()) + (_SPACING if room is None else room)
    return Structure(elements=elements, positions=positions,
                     regions=regions, frozen_atoms=frozen,
                     cell=np.diag([width, width, c]),
                     axis_kind=(*across, "transport"))


def _write_junction(root, struct, *, record=True):
    """One concluded junction relaxation with the distinctive deck.

    ``record`` -- the deck carries its placement record, as every deck
    prepped today does; ``False`` is a relaxation prepped before the rule,
    whose junction the rule centres (plan § 5q D7)."""
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
    deck_text = _cited_deck_text(struct, record=record)
    (attempt / "Relax_01_coarse.fdf").write_text(deck_text)
    (attempt / "Relax_01_coarse-run0.concluded").write_text("rc=0\n")
    _write_xv(attempt / "Relax.XV", struct)
    # Pseudos live IN the cited directory (4.1b: same-directory rule).
    write_pseudos(attempt, ["Au", "S", "C"])
    return calc


def _cited_deck_text(struct, *, record=True):
    """The cited relaxation's deck.  SELF-DESCRIBING (4.1b form A): its own
    coordinate block is the frozen gate's baseline, and the in-body
    ATOM-METADATA block carries the labels -- emitted through the real
    emitter, never hand-spelled -- and, with *record*, the ENGINE-OFFSET
    block whose axis kinds `compose` reads across the transport axis."""
    from molbuilder.script_emit import emit_atom_metadata
    coords = "\n".join(
        f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}  1"
        for p in struct.positions)
    label_store = {k: list(v) for k, v in struct.regions.items()}
    if struct.frozen_atoms:
        label_store["frozen_atoms"] = list(struct.frozen_atoms)
    block = emit_atom_metadata(regions=label_store,
                               n_atoms_total=len(struct.elements)) or ""
    if record:
        from molbuilder.cell import to_engine
        from molbuilder.script_emit import emit_engine_offset
        kinds = tuple(struct.axis_kind)
        block += "\n" + emit_engine_offset(
            to_engine(struct.replace(engine_offset=np.zeros(3))), kinds)
    return (_CITED_DECK
            + "AtomicCoordinatesFormat Ang\n"
            + "%block AtomicCoordinatesAndAtomicSpecies\n"
            + coords + "\n"
            + "%endblock AtomicCoordinatesAndAtomicSpecies\n\n"
            + block + "\n")


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
    # THE TEMPLATE, through the product's own doors -- `jobset init` writes
    # one for a transport description since 2026-09-16 (TR1), and a fixture
    # that skipped it would stop matching what this claims to reproduce.
    from molbuilder.transport.citation_defaults import (
        transport_template_text)
    (dest / "T.template.toml").write_text(
        transport_template_text(root / cite, label="T"))
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


#: The SIESTA keywords `engines/transport.md` § 3.2 measured as reaching NO
#: transport deck -- the ones a hand-written emitter's fixed list left out.
#: `MaxSCFIterations` and `DM.Tolerance` are the two that cost a real run: a
#: seed ran to 1000 iterations and died SCF_NOT_CONV because neither was in
#: the file, so SIESTA used its own defaults and nobody could ask otherwise.
UNREACHABLE_BEFORE_THE_SEAM = (
    "MaxSCFIterations", "DM.Tolerance", "DM.EnergyTolerance",
    "SCF.Mixer.Weight", "SCF.Mixer.History", "SCF.FreeE.Converge",
    "WriteForces", "WriteCoorStep", "WriteCoorXmol", "Diag.ParallelOverK",
)


class TestAnOverrideReachesTheRungThatOwnsIt:
    """TR8 — **the defect this whole programme started from.**

    Every override went onto the `device` rung, whatever it was.  So a person
    who set the transmission's energy window had it written into the deck
    `siesta` runs -- where `TBT.*` keywords are inert -- and NOT into the deck
    `tbtrans` runs, which is the one that computes T(E).  No error and no
    warning: you asked for ±3 eV and got the default.

    Routing by the `stages` declaration (`engines/template.md` § 6.4) is what
    makes that structurally impossible rather than something to remember.
    """

    def _describe_with(self, root, bags):
        """A description built the way the describe door builds one: per-rung
        bags, the shape `task.stages` carries (`engines/transport.md`
        § 3.8.2a)."""
        from molbuilder.task import Task, derive_run, write_task
        from molbuilder.transport.stages import stages_for_transport
        from molbuilder import template as _T
        from molbuilder.transport.citation_defaults import (
            siesta_config_from_citation)
        dest = root / "J" / "transport" / "T"
        dest.mkdir(parents=True, exist_ok=True)
        write_task(dest / "task.json", Task(
            engine="siesta", shape="hierarchical",
            run=derive_run("T", _CITE, stage_names=_STAGES),
            structure=None, calculation="transport",
            slots={"junction": _CITE}, bias=(),
            varies=tuple(sorted({n for b in bags.values() for n in b})),
            stages=tuple(stages_for_transport(bags))))
        (dest / ".molbuilder.json").write_text(json.dumps(
            {"script_generation": {"activation": "conda activate"}}))
        (dest / "T.template.toml").write_text(_T.template_with_values(
            siesta_config_from_citation(root / _CITE, label="T"),
            engine="siesta", calculation="transport"))
        return dest

    def test_the_TE_window_reaches_the_deck_tbtrans_runs(self, tmp_path):
        """THE ORIGINAL DEFECT, as something that can fail."""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        dest = self._describe_with(
            root, {"transmission": {"transmission_emin_ev": -3.0}})
        prep_calculation(dest, "transmission")
        deck = (dest / "05_transmission"
                / "T_05_transmission.fdf").read_text()
        from _deck import fdf_energy_window
        lo, hi = fdf_energy_window(deck, "TBT.Contour.window")
        assert lo == pytest.approx(-3.0), (
            f"the person's energy window must reach the TRANSMISSION deck "
            f"-- it is the one tbtrans runs, and the only place the window "
            f"has any effect; got from={lo}")
        assert hi == pytest.approx(2.0), (
            f"and the upper bound is the default, untouched: got to={hi}")

    def test_a_lead_parameter_reaches_BOTH_leads(self, tmp_path):
        """`electrode_kz` declares two owning rungs, and a junction has two
        leads.  Each lead's tab sets it for that lead (§ 3.8.2a: the rung is
        the person's answer, nothing routes), and each lead's deck carries
        it -- one left at the default would build its self-energy on a
        different Fermi-level resolution."""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        dest = self._describe_with(root, {"electrode_L": {"electrode_kz": 80},
                                          "electrode_R": {"electrode_kz": 80}})
        for rung, tok in (("electrode_L", "02_electrode_L"),
                          ("electrode_R", "03_electrode_R")):
            prep_calculation(dest, rung)
            deck = (dest / tok / f"T_{tok}.fdf").read_text()
            assert _has_row(deck, "0 0 80 0.0"), (
                f"{rung} must carry the person's lead k-density")

    def test_the_device_does_not_get_what_it_does_not_own(self, tmp_path):
        """THE OTHER HALF, and the one that makes this a routing test rather
        than a delivery test: a transmission parameter must not ALSO land on
        the device.  It was inert there, which is exactly why nobody noticed
        it was the only place it landed."""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        dest = self._describe_with(
            root, {"transmission": {"transmission_emin_ev": -3.0}})
        from molbuilder.task import read_task
        stages = {s.name: dict(s.overrides)
                  for s in read_task(dest / "task.json").stages}
        assert stages["transmission"] == {"transmission_emin_ev": -3.0}
        assert stages["device"] == {}, (
            "the device rung must not carry a parameter it does not own")


class TestTheBiasAxisIsTheParametersValues:
    """`engines/transport.md` § 2a.10: *single bias is the degenerate case of
    the bias axis — one point, at zero, where every list starts* — and the
    list is the bias's only home (plan § 5w K1, 2026-09-29).

    ONE mechanism, not two.  `bias_voltage_v` declares the parameter — its
    range, its unit, its help — and the rung fixes its value: each point of
    the description's list is one device run, written at that point.  The
    template's value answered a calculation with no list until 2026-09-29,
    a second home beside the list; that refusal is
    `TestTheRungFixesItsOwn`'s, through the CLI.
    """

    def test_each_point_gets_its_own_value(self, calc):
        prep_calculation(calc, "device")
        got = {}
        for d in (calc / "04_device").rglob("T_04_device.fdf"):
            import re as _re
            m = _re.search(r"^TS\.Voltage\s+(\S+)", d.read_text(), _re.M)
            got[d.parent.name] = float(m.group(1))
        assert got["v0"] == 0.0 and got["v0.2"] == 0.2


class TestTheElectrodeRungIsOnTheSeam:
    """TR5a — a lead rung renders through the framework, from the lead's own
    structure.

    The seam is `spec_for(struct, cfg, ...)` and its premise is *a deck
    describes a structure*.  An electrode rung describes the lead **taken out
    of the cited junction by its region label** -- same atoms, same
    relaxation, a subset rather than a geometry derived from elsewhere.  So
    the seam needed no widening: two rungs, two structures, one file.
    """

    def test_the_lead_deck_carries_the_template_items(self, calc):
        prep_calculation(calc, "electrode_L")
        deck = (calc / "02_electrode_L" / "T_02_electrode_L.fdf").read_text()
        missing = [k for k in UNREACHABLE_BEFORE_THE_SEAM
                   if not re.search(r"^" + re.escape(k) + r"\s", deck, re.M)]
        assert not missing, (
            f"the lead rung renders through the framework now, so these "
            f"should be in its deck: {missing}")

    def test_the_lead_gets_a_validation_report_and_a_check_gate(self, calc):
        """What a hand-written `write_text` could not give it."""
        prep_calculation(calc, "electrode_L")
        assert (calc / "02_electrode_L"
                / "T_02_electrode_L.validation.txt").is_file()

    def test_the_transport_axis_is_DENSE_here_and_1_on_the_seed(self, calc):
        """THE CONTRAST, and it is the physics rather than a formatting
        difference.

        This replaces the deleted `test_the_un_migrated_rungs_still_lack_them`:
        instead of *the lead lacks what the seed has*, it asserts the one
        thing the lead must have that the seed must not.  A lead is a
        genuinely periodic bulk crystal along transport and its Fermi level
        is the reference energy everything downstream is measured against; a
        device is an open boundary there and is not sampled at all.
        """
        prep_calculation(calc, "seed")
        prep_calculation(calc, "electrode_L")
        seed = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        lead = (calc / "02_electrode_L" / "T_02_electrode_L.fdf").read_text()
        assert _has_row(lead, "0 0 40 0.0"), (
            "the lead's transport axis must be DENSE -- that density is what "
            "resolves its Fermi level")
        assert _has_row(seed, "0 0 1 0.0"), (
            "...and the seed's must be 1: no transport-axis sampling")

    def test_the_lead_is_labelled_as_the_TSHS_the_device_will_name(self, calc):
        """One spelling, `electrode_hs_stem`, read by both writers.

        The device deck's `TS.Elec` block names a file; the lead run must
        write exactly that file, and it does so by taking the stem as its
        SystemLabel.
        """
        prep_calculation(calc, "electrode_L")
        prep_calculation(calc, "device")
        lead = (calc / "02_electrode_L" / "T_02_electrode_L.fdf").read_text()
        dev = (calc / "04_device" / "T_04_device.fdf").read_text()
        assert _says(lead, "SystemLabel", "T_L-electrode")
        assert "T_L-electrode.TSHS" in dev

    def test_the_lead_deck_is_the_LEAD_not_the_junction(self, calc):
        """The structure is the extracted subset, so the deck is smaller.

        Without this the tests above would pass on a lead deck that had
        quietly rendered the whole junction.
        """
        from molbuilder.transport.compose import load_compose_record
        from molbuilder.task import read_task
        prep_calculation(calc, "electrode_L")
        deck = (calc / "02_electrode_L" / "T_02_electrode_L.fdf").read_text()
        task = read_task(calc / "task.json")
        composed = load_compose_record(calc, citation=task.slots["junction"])
        n_lead = composed.electrode_left.n_atoms
        n_dev = composed.sorted.structure.n_atoms
        assert n_lead < n_dev, "the fixture must have a lead smaller than the device"
        assert _says(deck, "NumberOfAtoms", str(n_lead)), (
            f"the lead deck must describe the {n_lead}-atom lead, not the "
            f"{n_dev}-atom junction")


class TestTheTransportArmResolves:
    """TR4 — `prep`'s transport arm hands its deciding to `resolve`.

    `engines/transport.md` § 3.2 measured the arm as *"a second conductor
    that decides"*: it composed, gated, extracted and rendered, so there was
    no `ParameterSet`, no provenance, and `--pipeline-log` was a documented
    no-op.  TR1 is what made the fix reachable -- `resolve` reads a template
    and a transport calculation did not have one.
    """

    def _log(self, calc):
        return next(calc.rglob("*.pipeline.log"), None)

    def test_the_pipeline_log_is_written_and_names_the_resolve_step(self, calc):
        prep_calculation(calc, "seed", pipeline_log=True)
        log = self._log(calc)
        assert log is not None, (
            "--log printed 'not wired for the transport arm yet' until "
            "2026-09-16; it must write a file now")
        text = log.read_text()
        assert "STEP 2 · RESOLVE" in text

    def test_every_value_names_the_source_that_set_it(self, calc):
        """PROVENANCE -- what the arm had nothing to print.

        `project-layout.md` M3's *"the numbers were wrong"* is answerable
        only if each value says where it came from.
        """
        prep_calculation(calc, "seed", pipeline_log=True)
        text = self._log(calc).read_text()
        for name in ("basis_size", "dm_tolerance", "max_scf_iter",
                     "electronic_temperature"):
            assert re.search(r"^\s*⊕\s+" + name + r"\s+\S.*<- \w+$",
                             text, re.M), (
                f"{name} reached the deck with no recorded source")

    def test_a_description_with_no_template_is_refused_by_name(self, calc):
        """The honest failure for a description written before TR1.

        Refused with what to do, rather than falling back to re-reading the
        citation -- which would be the sealed behaviour returning by the
        back door, silently.
        """
        from molbuilder.template import find_template
        find_template(calc).unlink()
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed")
        assert "template" in str(e.value) and "TR1" in str(e.value)


class TestTheTransportAxisIsNotSettable:
    """TR3 — the k-grid's three components fall in three classes, and the
    third is not the person's.

    `engines/transport.md` § 2a.13. The transverse pair is shared by the
    leads and the device; the LEAD's transport-axis density is each
    electrode's own (`electrode_kz`); the DEVICE's transport axis is fixed
    at 1, because that axis is the open boundary and is not
    Brillouin-zone sampled at all.

    Before this the third component could be set and the renderer wrote 1
    anyway -- a control that appears to do something and does not.  Its
    refusal, on every door, is `test_k_point_mesh_e2e.py`'s (the k-point
    mesh, `engines/siesta.md` § 6.1); a copy of its prep half stood here
    until the K3 review.
    """

    def test_the_transverse_pair_is_still_the_persons(self, calc):
        """The half that keeps the refusal from being a ban on the row.

        Without it, deleting the k-grid control entirely would pass the
        test above.
        """
        from molbuilder.template import _emit, find_template, read_template
        import dataclasses
        tmpl = find_template(calc)
        parsed = read_template(tmpl.read_text())
        tmpl.write_text(_emit(
            [dataclasses.replace(i, value=(6, 6, 1)) if i.name == "kgrid"
             else i for i in parsed.items], engines=("siesta",)))
        prep_calculation(calc, "seed")          # must not raise
        deck = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        assert _has_row(deck, "6 0 0 0.0"), (
            "the person's transverse grid must reach the deck")


class TestTheTemplateIsTheSharedBaseline:
    """TR1 — transport has a template, and it is what a deck renders from.

    `engines/transport.md` § 2a.7, ruling 1: the cited relaxation **defaults**
    the shared electronic description; it does not seal it.  The person may
    change any of it afterwards, and a change applies to every stage at once
    because there is one template.

    A transport folder carried no template at all until 2026-09-16, so there
    was nowhere for that to be true.
    """

    def _template(self, calc):
        from molbuilder.template import find_template
        return find_template(calc)

    def test_editing_the_template_changes_the_deck(self, calc):
        """THE RULING, as something that can fail.

        Without this the template is a file nothing reads — the defect
        `electrode_kz` shipped with, one directory up.
        """
        from molbuilder.template import read_template, _emit
        import dataclasses
        tmpl = self._template(calc)
        assert tmpl is not None, (
            "TR1: a transport description must carry a template")

        prep_calculation(calc, "seed")
        before = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        assert _says(before, "PAO.BasisSize", "TZP"), (
            "the template's value, defaulted from the cited run, must reach "
            "the deck")
        # The citation says TZP too, so the line above cannot tell the two
        # sources apart -- which is the whole question.  The edit below is
        # to a value the CITATION does not hold, so only the template can
        # be the source of what lands.

        # ...now the person changes it, which is exactly what the ruling
        # exists to permit: relax cheap, transport accurate.
        parsed = read_template(tmpl.read_text())
        edited = [dataclasses.replace(i, value="DZP")
                  if i.name == "basis_size" else i for i in parsed.items]
        tmpl.write_text(_emit(edited, engines=("siesta",)))

        prep_calculation(calc, "seed")
        after = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        assert _says(after, "PAO.BasisSize", "DZP"), (
            "the edit did not reach the deck -- the citation is being "
            "re-read at prep, which is the SEALED behaviour the ruling "
            "reversed")

    def test_the_cited_run_fills_it(self, calc):
        """DEFAULTED, not invented: the values are the citation's."""
        from molbuilder.template import read_template
        got = {i.name: i.value
               for i in read_template(self._template(calc).read_text()).items}
        assert got["basis_size"] == "TZP"
        assert got["xc_authors"] == "revPBE"
        assert got["mesh_cutoff"] == 250.0
        assert got["kgrid"] == (4, 4, 1), (
            "the transverse pair carries over and the transport axis is "
            "forced to 1 -- that axis is the open boundary and is not "
            "sampled at all")

    def test_a_role_item_is_not_answered_by_the_template(self, calc):
        """`solution_method` is the rung's, so the file must not claim it."""
        from molbuilder.template import read_template
        got = {i.name: i.value
               for i in read_template(self._template(calc).read_text()).items}
        assert got["solution_method"] is None


class TestTheSeedIsOnTheSeam:
    """§ 3.6 items 1-4: the seed rung renders through `spec_for` ->
    `prepare_deck`, so the template's items reach it.

    What this pins is the MIGRATION, not one keyword.

    It had a second half -- `test_the_un_migrated_rungs_still_lack_them` --
    asserting the SAME list was ABSENT from the electrode rung, so the
    contrast meant *this rung is on the seam* rather than *SIESTA decks tend
    to have SCF settings*.  That test carried an instruction to delete it
    when the electrode rung was tabled, *"at which point it should fail, and
    that failure is the migration being done"*.  It failed on 2026-09-16 and
    was deleted.  The contrast now lives in
    :class:`TestTheElectrodeRungIsOnTheSeam`, which asserts what the lead
    deck HAS -- and what it has that the seed does not.
    """

    def test_the_template_items_reach_the_seed_deck(self, calc):
        prep_calculation(calc, "seed")
        seed = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        missing = [k for k in UNREACHABLE_BEFORE_THE_SEAM
                   if not re.search(r"^" + re.escape(k) + r"\s", seed, re.M)]
        assert not missing, (
            f"the seed renders through the framework now, so these should "
            f"be in its deck: {missing}")

    def test_every_value_arrives_with_its_reason(self, calc):
        """The note-with-the-value rule, which a literal f-string cannot
        keep: each item is written through its declaration, so the deck
        explains itself."""
        prep_calculation(calc, "seed")
        seed = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        for kw in ("MaxSCFIterations", "DM.Tolerance"):
            assert re.search(r"^#\s*" + re.escape(kw) + r"\s*$", seed, re.M), (
                f"{kw} should be introduced by its own note line")


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
        assert _says(text, "SolutionMethod", "diagon")
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

    def test_no_rung_says_its_frozen_atoms_are_held_in_a_relaxation(self,
                                                                    calc):
        """A transport rung relaxes nothing, so its report says nothing about
        holding atoms during a relaxation.

        GOAL: every junction rung of the 2026-09-25 ladder reported "108
        atom(s) held fixed during SIESTA relaxation", reasoned from
        `relax_type`'s catalogue default on a deck with no MD block.
        CONTRACT: `science/validation.md` § 7 (one fact, one finding: a
        family reasoned from the wrong calculation is not fired) and the seed
        deck's own note (`transport/deck.py`): no MD block, a single point at
        the relaxed geometry.
        """
        prep_calculation(calc, "seed")
        report = (calc / "01_seed" / "T_01_seed.validation.txt").read_text()
        assert "[config.frozen_atoms]" not in report, report

    def test_no_rung_says_its_pseudopotentials_are_unconfigured(self, calc):
        """The pseudopotentials come with the citation, and the report says
        nothing about a library transport never uses.

        GOAL: every rung of the 2026-09-25 ladder reported "cfg.psml_lib is
        not set -- SIESTA ... will refuse to start", beside a deck whose
        pseudopotentials prep had just copied from the cited run.  CONTRACT:
        `engines/transport.md` (the pseudopotentials travel with the
        citation) and `pseudos.psml_sources` (the gate reads the calculation's
        folder first, as prep does).
        """
        prep_calculation(calc, "seed")
        report = (calc / "01_seed" / "T_01_seed.validation.txt").read_text()
        assert "[config.psml_lib" not in report, report

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
        assert _says(elec, "TS.HS.Save", ".true.")

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
        assert _says(text, "SolutionMethod", "transiesta")
        assert "%block TS.Elecs" in text
        # Sorted: the first six coordinate rows are the lower Au electrode,
        # the next four the bridge (S C C S).
        #
        # THE SPECIES INDEX IS ASKED, NEVER SPELLED.  What this test is about
        # is the ORDER OF THE ATOMS -- that the sort put a contiguous lead
        # first -- and the species column is how that is read off the deck.
        # The column's numbering is a different contract
        # (`model/chemistry.md` § 3a) with its own test, so spelling `1` here
        # pinned that rule as a side effect: it said Au=1 from the
        # alphabetical order this emitter used until 2026-09-23, and went red
        # when carbon moved to the front for a reason this test has no
        # opinion about.
        from molbuilder.chemistry import species_order
        idx = {el: str(i + 1) for i, el
               in enumerate(species_order(["Au", "C", "S"]))}
        block = text.split("%block AtomicCoordinatesAndAtomicSpecies")[1]
        rows = [ln.split() for ln in block.splitlines()
                if ln.strip() and not ln.startswith("%")]
        assert [r[3] for r in rows[:6]] == [idx["Au"]] * 6
        assert [r[3] for r in rows[6:10]] == [idx[e] for e in "SCCS"]

    @pytest.mark.parametrize("recorded", [True, False],
                             ids=["a-deck-with-its-record",
                                  "a-deck-from-before-the-rule"])
    def test_every_rung_hands_the_engine_placed_coordinates(self, tmp_path,
                                                            recorded):
        """T1 for the transport rungs (plan § 5q.4): each deck writes the
        coordinates its ENGINE-OFFSET record accounts for, every atom inside
        the cell.

        The seed and the device are the cited relaxation's `.XV` -- the
        engine's own coordinates.  When the cited deck recorded its placement
        they go in verbatim and the record states 0: re-centring the device
        against its leads is the failure W33 was opened for.  A deck from
        before the rule has no record, so the rule centres the junction -- one
        rigid shift, the same for the seed and the device (D7).  The leads are
        cut from that junction and placed by the rule, centred in the cell the
        device gives them.

        Contract: `model/structure-periodicity.md` § 6.0 (the invariant; *A
        stated offset*: a transport rung from the cited `.XV`, D7; check 3).
        """
        from molbuilder.parse.coords.siesta_xv import read_xv_with_cell
        from molbuilder.deck_record import extract_engine_offset
        root = tmp_path / "projects"
        # A deck from before the rule records no axis kinds either, and its
        # junction is read periodic across -- so the chain it cites is the
        # lattice it tiles, or the settings gate refuses its vacuum.
        _write_junction(root, _junction_struct() if recorded else
                        _junction_struct(across=("periodic", "periodic"),
                                         width=_SPACING), record=recorded)
        calc = _describe_transport(root)
        xv, _ = read_xv_with_cell(root / _CITE / "Relax.XV")

        def deck(stage):
            prep_calculation(calc, stage)
            tok = _TOKEN[stage]
            text = (calc / tok / f"T_{tok}.fdf").read_text()
            block = (text.split("%block AtomicCoordinatesAndAtomicSpecies")[1]
                     .split("%endblock")[0])
            written = np.array([[float(v) for v in ln.split()[:3]]
                                for ln in block.splitlines() if ln.strip()])
            record = extract_engine_offset(text)
            assert record is not None, f"the {stage} deck carries no record"
            cell = np.asarray(record["cell"], dtype=float)
            frac = np.linalg.solve(cell.T, written.T).T
            lens = np.linalg.norm(cell, axis=1)
            near, far = frac.min(axis=0) * lens, (1.0 - frac.max(axis=0)) * lens
            assert np.all(near >= -1e-6) and np.all(far >= -1e-6), (stage,
                                                                    near, far)
            return written, record, near, far

        for stage in ("seed", "device"):
            written, record, near, far = deck(stage)
            assert record["stated"] is recorded, (stage, record)
            if recorded:
                np.testing.assert_allclose(record["applied_offset"], 0.0,
                                           atol=0.0)
            else:
                np.testing.assert_allclose(near, far, atol=1e-6)
            np.testing.assert_allclose(
                written, xv.positions + np.asarray(record["applied_offset"]),
                atol=1e-6)
        for stage in ("electrode_L", "electrode_R"):
            _written, record, near, far = deck(stage)
            assert record["stated"] is False, (stage, record)
            np.testing.assert_allclose(near, far, atol=1e-6)

    def test_the_transmission_deck_carries_the_tbt_window(self, calc):
        """SCIENCE. The transmission deck carries the tbtrans energy window
        -- as the contour block tbtrans actually reads.

        Catches the deliverable being computed over no energy range. T(E) is
        evaluated on a grid the deck specifies; with the window keywords missing,
        tbtrans falls back to its own defaults and the transmission curve -- the one
        number this whole five-stage ladder exists to produce -- is reported over an
        interval nobody chose and that need not contain E_F.

        Contract: `engines/transport.md` § 2 (T(E) = Tr[Gamma_L G Gamma_R G+])
        + § 1 (the transmission stage's product).

        THIN: it asserts the window is PRESENT, not that its values bracket
        E_F, which is what makes the window right or wrong.

        **It asserted `TS.TBT.NumE` / `TS.TBT.Emin` until 2026-09-15** --
        keywords the installed 5.4.2 tbtrans cannot read, so it was pinning
        precisely the failure its own docstring describes: tbtrans falling
        back to its own defaults over an interval nobody chose
        (`plan.md` § 5o).  A test written to catch that was satisfied by it.
        """
        prep_calculation(calc, "transmission")
        text = (calc / "05_transmission" / "T_05_transmission.fdf"
                ).read_text()
        assert "%block TBT.Contours" in text
        assert "%block TBT.Contour." in text and "part line" in text

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
            assert _says(text, "PAO.BasisSize", "TZP"), who
            assert _says(text, "XC.authors", "revPBE"), who
            assert _says(text, "MeshCutoff", "250 Ry"), who
            assert _says(text, "PAO.EnergyShift", "0.02 Ry"), who
            assert _says(text, "ElectronicTemperature", "200.0 K"), who
        # transverse k = the relaxation's (4, 4), transport axis 1
        assert _has_row(dev, "0 0 1 0.0"), (
            "the device kz is forced to 1 (open boundary)")
        assert _has_row(dev, "4 0 0 0.0") and _has_row(seed, "4 0 0 0.0")

    def test_a_scrambled_source_preps_clean_because_prep_sorted(self, tmp_path):
        """THE P4 gate: a source whose atom order TranSIESTA would
        misread preps clean, because prep sorts before any deck exists.

        TranSIESTA reads ``TS.NumUsedAtomsLeft = N`` as *"the first N atoms
        in the coordinates block are the left electrode"*, so an order other
        than ``[lower][bridge][upper]`` builds the lead self-energies from
        the wrong atoms -- and converges while doing it.

        *(The discriminating half drove `TransiestaEngine.preflight` until
        2026-09-17.  That class is deleted: it dispatched for nothing, since
        every rung resolves a `SiestaConfig`.  The live refusal is
        `sort.categorical_sort`, which is also what makes the claim TRUE --
        the order is held by construction, not by a gate.)*
        """
        from molbuilder.transport.sort import categorical_sort
        root = tmp_path / "projects"
        scrambled = _junction_struct(order="scrambled")
        _write_junction(root, scrambled)
        dest = _describe_transport(root)

        # The discriminating half: this fixture is genuinely out of order,
        # so the test cannot pass on a tame one.  `categorical_sort` returns
        # the permutation it applied; identity would mean nothing moved.
        moved = categorical_sort(scrambled)
        assert list(moved.original_to_sorted) != list(
                range(len(moved.original_to_sorted))), (
            "the scrambled fixture is already in canonical order -- this "
            "test would pass without prep sorting anything")

        prep_calculation(dest, "device")     # must not raise
        text = (dest / "04_device" / "T_04_device.fdf").read_text()
        assert _says(text, "SolutionMethod", "transiesta")

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
        # A relaxation from before the rule: its buffer layers overhang the
        # cell as authored (z = -5 and 39.5 in c = 47), and with no placement
        # record the rule centres the junction (plan § 5q D7) -- and reads it
        # periodic across, so the chain is the lattice it tiles.
        _write_junction(root, _junction_struct(
            buffers=True, across=("periodic", "periodic"), width=_SPACING),
            record=False)
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
        s2 = _junction_struct()
        (attempt / "Relax_01_coarse.fdf").write_text(_cited_deck_text(s2))
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

    def test_a_re_prepped_seed_still_satisfies_the_device(self, calc,
                                                          monkeypatch):
        """THE DAG GATE ASKS "same calculation", NOT "same bytes".

        A deck that renders through the framework carries a `generated-at`
        timestamp and the generator's git sha.  The gate compared full text,
        so once the seed rung joined the render pipeline (2026-09-15) two
        ordinary things broke the device's gather: re-prepping a concluded
        seed (a different allocation, a different `--target`, or just running
        the command twice), and committing between the two preps, which moves
        the sha.  It refused with *"the junction citation or its contract
        changed"* -- false, and it pointed the reader at the science.

        Mutation check: revert the gate to `read_text() == read_text()` and
        this fails, because the re-prep genuinely rewrites the timestamp.

        Contract: `engines/transport.md` § 4.2 (the DAG) + `script_emit.
        same_calculation`.
        """
        # THE CLOCK TICKS BETWEEN THE TWO PREPS, by construction.  The stamp
        # has one-second resolution, so two preps inside one second wrote
        # byte-identical decks and the vacuity guard below failed on a fast
        # run -- read off `script_emit.generated_at_now`, 2026-09-25.  The
        # premise is a REWRITTEN RECORD, so the test states one.
        import itertools
        from molbuilder import script_emit as _se
        _tick = itertools.count()
        monkeypatch.setattr(
            _se, "generated_at_now",
            lambda: f"2026-09-25T00:00:{next(_tick) % 60:02d}-07:00")
        prep_calculation(calc, "seed")
        prep_calculation(calc, "electrode_L")
        prep_calculation(calc, "electrode_R")
        before = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        _conclude(calc, "seed", ["T.DM"])
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])

        # ...and now the seed is prepped again, which is what a person does
        # when they change an allocation.  Only the record moves.
        prep_calculation(calc, "seed")
        after = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        assert after != before, (
            "this test is vacuous unless the re-prep really did rewrite the "
            "deck -- if the deck became deterministic, delete the test")
        from molbuilder.script_emit import same_calculation
        assert same_calculation(before, after), (
            "a re-prep changed something other than the record; that is a "
            "different bug from the one this test is about")

        from molbuilder.jobset.prep import gather_transport_inputs
        dest = calc / "device-attempt"
        dest.mkdir()
        got = gather_transport_inputs(calc, self._task(calc), "device", dest)
        assert sorted(fn for _s, fn in got) == [
            "T.DM", "T_L-electrode.TSHS", "T_R-electrode.TSHS"], (
            "the re-prepped seed's .DM must still be gathered: only its "
            "record moved, so it answers the same calculation")


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
        #
        # PREP OPENS THE ATTEMPT AND LAYS THE DECK IN, since 2026-09-16 --
        # it did not until then, on the transport arm only, so this test
        # built `run-0` by hand.  Only the half-written product is the
        # test's own now; the rest is what prep really produces.
        run0 = calc / "02_electrode_L" / "run-0"
        assert (run0 / "T_02_electrode_L.fdf").is_file(), (
            "prep must open the rung's attempt and bring its deck in -- "
            "a folder with no attempt is one `launch` refuses")
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
        """The transmission's own deck, a different program: the binary
        rides Resources.program into the wrapper (P5)."""
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


class TestTheBrowserRoute:
    """`POST /api/task-setup/prep` on a transport rung — the SAME steps.

    The browser's Prep button calls `prep_calculation` directly, so every step
    the CLI adds afterwards is a step this road did not take. Until 2026-09-16
    the DAG gather was one of them, and it was survivable only by accident: the
    transport arm opened no attempt at all, so `launch` refused the folder by
    name. Opening the attempt -- the same day, for a different reason -- removed
    that refusal and left the gap standing, which is how one fix makes another
    bug reachable.
    """

    def _prep(self, calc, root, monkeypatch, stage):
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.scheduler.record import LOCAL_TARGET
        from molbuilder.web.app import create_app
        # The app serves the developer's real `projects/` by default, and the
        # door's own path fence refuses anything outside it -- correctly.
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        client = create_app(config={}).test_client()
        return client.post("/api/task-setup/prep", json={
            "dest": str(calc), "kind": "run", "stage": stage,
            "target": LOCAL_TARGET})

    def test_the_browser_gathers_what_the_device_consumes(
            self, calc, tmp_path, monkeypatch):
        """The leads' `.TSHS` and the seed's `.DM` land in the attempt.

        Without this the browser reports success and hands back an attempt
        holding a deck, a wrapper and nothing to read -- so the job reaches the
        node and dies for want of an electrode Hamiltonian, after the queue
        wait. The terminal road has always carried them; this asserts the two
        roads produce the same directory rather than the same message.
        """
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])

        r = self._prep(calc, tmp_path / "projects", monkeypatch, "device")
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        assert body["ok"] is True
        # bias=(0.0, 0.2) is a scan: one attempt ladder per point, each
        # gathered against ITS OWN voltage.
        for point in ("v0", "v0.2"):
            run0 = calc / "04_device" / point / "run-0"
            for n in ("T.DM", "T_L-electrode.TSHS", "T_R-electrode.TSHS"):
                assert (run0 / n).is_file(), f"{point}/{n} was not carried in"
        carried = {p["attempt"] for p in body["points"]}
        assert any("v0.2" in c for c in carried), (
            "the response must say what landed where -- a person who cannot "
            "see the carry cannot tell this apart from the old silence")

    def test_an_unready_device_is_refused_here_too(self, calc, tmp_path,
                                                   monkeypatch):
        """The same three gates, in the browser, before the queue is spent.

        The half that makes the test above mean something: a road that
        gathered when it could and shrugged when it could not would pass it.
        `gather_transport_inputs` refuses an upstream that has not CONCLUDED,
        and that refusal has to reach the browser as a refusal.
        """
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])      # the electrodes stay unconcluded
        r = self._prep(calc, tmp_path / "projects", monkeypatch, "device")
        assert r.status_code == 400
        msg = r.get_json()["error"]
        assert "electrode_L" in msg and "CONCLUDED" in msg, msg

    def test_a_rung_with_no_upstream_still_preps(self, calc, tmp_path,
                                                 monkeypatch):
        """The seed consumes nothing, so the gather must be a no-op for it.

        Without this half the gather could refuse everything and the test above
        would still pass.
        """
        r = self._prep(calc, tmp_path / "projects", monkeypatch, "seed")
        assert r.status_code == 200, r.get_json()
        assert (calc / "01_seed" / "run-0" / "T_01_seed.fdf").is_file()


class TestTheCliRouteRefusal:

    def _invoke(self, args, root, monkeypatch):
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        from molbuilder.projects import PROJECTS_ROOT_ENV
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        return CliRunner().invoke(jobset_group, args)

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
    """Transport-only knobs travel as stage overrides (P7b): the shared
    electronic description lives in the template (the transport tab's
    shared panel writes it), and the stages' own bags are the description's
    place for what a rung owns alone -- the transmission window, the
    contour (`engines/transport.md` § 3.8.2)."""

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

        Catches the override lane being inert. A stage's `overrides` bag is the
        description's place for what that rung owns alone -- the transmission
        window, the contour (the shared description is the template's). An override
        that is accepted, written into task.json, and then not rendered gives the
        user a description that reads as configured and a deck that is at defaults.

        Contract: `engines/transport.md` § 5 (the invariant set is the citation's;
        everything else is the description's).

        **The evidence changed on 2026-09-15, the property did not.** This
        asserted `TS.TBT.NumE 101` -- a SIESTA-3.x keyword the installed
        5.4.2 tbtrans cannot read (`plan.md` § 5o), so the test was pinning
        the bug: the override did land in the deck and the deck could not be
        read, which is the one arrangement that satisfies "reaches the
        rendered deck" while the setting still does nothing.  It now asserts
        the contour block's own `points`, and
        `test_transport_keywords_exist_in_the_binary.py` is what stops a
        dead keyword satisfying this test again.

        **And the rung changed on 2026-09-24, the property did not.** This
        placed the TRANSMISSION's point count on the DEVICE rung and read it
        back from the device deck -- the pre-TR8 placement, which `prep`
        now refuses by name (`engines/template.md` § 6.4, the `stages`
        declaration; the test below).  The knob is now one the device OWNS,
        the equilibrium contour's pole energy, read back from the device
        deck through the keyword the binary reads.
        """
        self._with_override(calc, "device", {"negf_eq_pole_ev": 2.5})
        prep_calculation(calc, "device")
        text = (calc / "04_device" / "T_04_device.fdf").read_text()
        eq = [ln for ln in text.splitlines()
              if ln.strip().startswith("TS.Contours.Eq.Pole")]
        assert eq, "the device deck carries no TS.Contours.Eq.Pole line"
        assert "2.5" in eq[0], (
            f"the override did not reach the deck line: {eq[0]!r}")

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
        msg = str(e.value)
        # AND a shared value no run answers (`species_order` binds the leads'
        # orbital ordering to the device's; `engines/transport.md` § 3.8.6):
        # the refusal read the `citation` marker alone until 2026-09-24 and
        # let this one through.
        self._with_override(calc, "device", {"species_order": ["S", "C", "Au"]})
        with pytest.raises(PrepError, match="(?i)shared by every stage"):
            prep_calculation(calc, "device")
        assert "SHARED" in msg or "shared" in msg, (
            f"the refusal must say WHY -- the value is shared by every "
            f"stage, so one rung cannot have its own.  It said \"the "
            f"citation's to say\" until 2026-09-16, which § 2a.7 reversed: "
            f"the cited run DEFAULTS these values rather than owning them, "
            f"so that message sent a person to redo a relaxation when they "
            f"could edit one line of the template: {msg}")
        assert "template" in msg, (
            f"...and it must name where the value IS changed: {msg}")

    def test_a_rungs_own_value_on_another_rung_is_refused_by_name(self, calc):
        """`engines/template.md` § 6.4, the `stages` declaration: *"only these
        rungs may; it is not that rung's business anywhere else."*

        The describe door ROUTES an override to the rung that owns it (TR8).
        A description written by any other road -- the CLI, the stage table,
        a hand edit -- reached `resolve`, which knows no ownership, so a
        transmission window placed on the seed was written into the seed's
        deck, where `tbtrans` never reads it, and the transmission ran on
        the default.  Silently, on every road TR8 did not cover, until
        2026-09-24.  The refusal names the owning rung.
        """
        self._with_override(calc, "seed", {"transmission_n_points": 101})
        with pytest.raises(PrepError, match="(?i)not this rung") as e:
            prep_calculation(calc, "seed")
        assert "transmission" in str(e.value) and "'seed'" in str(e.value)

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
        """SCIENCE. A form-B citation carries no deck, so nothing dictates its
        electronic description — and the person must still be able to state it.

        **The mechanism changed on 2026-09-16 and the concern did not.** This
        test used to set `basis_size` as a STAGE OVERRIDE, because that was the
        only way in: the contract fields were sealed for form A and opened for
        form B, so the override lane was where a pair's basis could be said.
        Its worry was exact — *"if the field stayed sealed there would be no
        way to state the basis at all"*.

        `engines/transport.md` § 2a.7 answered it better. Every transport
        calculation now carries a TEMPLATE: for form A it is filled from the
        cited deck, for form B from the catalogue's own defaults, and either
        way the person may change it. So the basis is stateable for a pair —
        in the place where it applies to all five rungs at once, rather than
        on one rung where it could make the device disagree with its leads.

        Contract: `engines/transport.md` § 3.1 (form A vs form B) + § 2a.7.
        """
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        pair = root / "J" / "structure" / "junc"
        pair.mkdir(parents=True)
        StructureCodec().write(_junction_struct(), pair / "junction.xyz")
        write_pseudos(pair, ["Au", "S", "C"])
        dest = _describe_transport(root, cite="J/structure/junc")
        # A pair has no deck, so the template starts from the catalogue's
        # defaults -- and the person changes it there.
        import dataclasses
        from molbuilder.template import _emit, find_template, read_template
        tmpl = find_template(dest)
        assert tmpl is not None, (
            "a form-B transport calculation carries a template too -- "
            "without one there would be nowhere to state the basis")
        parsed = read_template(tmpl.read_text())
        tmpl.write_text(_emit(
            [dataclasses.replace(i, value="TZP") if i.name == "basis_size"
             else i for i in parsed.items], engines=("siesta",)))
        prep_calculation(dest, "seed")
        deck = (dest / "01_seed" / "T_01_seed.fdf").read_text()
        assert _says(deck, "PAO.BasisSize", "TZP"), (
            "a pair's electronic description is the person's to state, and "
            "the template is where they state it")

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
        msg = str(e.value)
        assert "SHARED" in msg or "shared" in msg, (
            f"the refusal must say WHY -- the value is shared by every "
            f"stage, so one rung cannot have its own.  It said \"the "
            f"citation's to say\" until 2026-09-16, which § 2a.7 reversed: "
            f"the cited run DEFAULTS these values rather than owning them, "
            f"so that message sent a person to redo a relaxation when they "
            f"could edit one line of the template: {msg}")
        assert "template" in msg, (
            f"...and it must name where the value IS changed: {msg}")


def test_a_blank_spin_is_decided_once_on_the_junction_and_said_on_every_rung(
        tmp_path):
    """ES1 on the ladder (`science/chemistry-correctness.md` § 2a, the
    transport rules): a spin the template leaves blank is decided ONCE, on
    the whole junction, and every rung -- a gold lead included -- carries
    that answer and says where it came from.  TranSIESTA joins the leads'
    self-energies to the device, so decided on the lead's own atoms (gold in
    a repeating cell: restricted) it would sit beside a polarized device.

    The bridge carries an iron centre, and the citation's spin is blanked on
    the template: iron in a repeating cell floats its moment."""
    root = tmp_path / "projects"
    struct = _junction_struct()
    elements = list(struct.elements)
    elements[elements.index("C")] = "Fe"
    struct = struct.replace(elements=elements)
    _write_junction(root, struct)
    write_pseudos(root / _CITE, ["Fe"])
    calc = _describe_transport(root)
    tmpl = calc / "T.template.toml"
    text = tmpl.read_text()
    for item in ("spin_treatment", "unpaired_electrons"):
        i = text.index(f"[item.{item}]")
        j = text.find("[item.", i + 1)
        j = len(text) if j < 0 else j
        block = re.sub(r"(?m)^value = .*\n", "", text[i:j])
        text = text[:i] + block + text[j:]
    tmpl.write_text(text)

    prep_calculation(calc, "electrode_L")
    lead = (calc / "02_electrode_L" / "T_02_electrode_L.fdf").read_text()
    assert _says(lead, "Spin", "polarized"), lead
    assert "Spin.Fix" not in lead, "a floating moment pins nothing"
    assert ("# Spin: unrestricted (detected: Fe is an open-d metal in a "
            "repeating cell") in lead, lead
    assert "Decided ONCE, on the whole junction" in lead
    assert "# NetCharge: not written -- +0 (rule:" in lead


@pytest.mark.parametrize("spelled", ["Spin polarized", "Spin COLLINEAR",
                                     "SpinPolarized .true."])
def test_a_cited_decks_spin_is_read_in_any_word_siesta_accepts(
        tmp_path, spelled):
    """ES7 from a DECK (`science/chemistry-correctness.md` § 2a): the cited
    run's spin is written into the transport template -- read in every word
    SIESTA 5.4.2 accepts for it (`spin_subs.F90`, case-blind) and from the
    retired flag SIESTA still honours.  A polarized run with no ``Spin.Fix``
    let its moment float.  (The reader knew only the writer's own spelling
    until the M6 review, and read ``collinear`` as restricted.)"""
    from molbuilder.template import one, read_template
    root = tmp_path / "projects"
    _write_junction(root, _junction_struct())
    deck = root / _CITE / "Relax_01_coarse.fdf"
    deck.write_text(deck.read_text().replace(
        "SystemLabel Relax\n", f"SystemLabel Relax\n{spelled}\n", 1))
    tmpl = read_template(
        (_describe_transport(root) / "T.template.toml").read_text())
    assert one(tmpl, "spin_treatment").value == "unrestricted"
    assert one(tmpl, "unpaired_electrons").value == "free"


class _LadderThroughTheCli:
    """A ladder prepped through `molbuilder jobset prep`, rung by rung --
    the road the classes below drive.  No tests of its own."""

    def _cli(self, args, root, monkeypatch):
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        from molbuilder.projects import PROJECTS_ROOT_ENV
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        return CliRunner().invoke(jobset_group, args)

    def _ladder(self, tmp_path, monkeypatch, *, edit=None):
        """A single-bias ladder prepped rung by rung through the CLI, each
        upstream rung concluded as a finished run leaves it; the device and
        transmission decks."""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        calc = _describe_transport(root, bias=(0.0,))
        if edit is not None:
            edit(calc)
        products = {"seed": ["T.DM"],
                    "electrode_L": ["T_L-electrode.TSHS"],
                    "electrode_R": ["T_R-electrode.TSHS"],
                    "device": ["T.TS.HSX"]}
        for stage in _STAGES:
            r = self._cli(["prep", "run", stage, "--bundle", "J/transport/T"],
                          root, monkeypatch)
            assert r.exit_code == 0, (stage, r.output)
            if stage in products:
                _conclude(calc, stage, products[stage])
        return ((calc / "04_device" / "T_04_device.fdf").read_text(),
                (calc / "05_transmission" / "T_05_transmission.fdf").read_text())

    @staticmethod
    def _settings(deck):
        """The deck's non-comment lines -- what a program reads."""
        return [ln.strip() for ln in deck.splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]


class TestEachDeckCarriesWhatItsProgramReads(_LadderThroughTheCli):
    """`engines/transport.md` § 6.1b, through `molbuilder jobset prep`:
    two programs read the NEGF rungs' decks, and each deck carries what its
    own program reads, every value with its note.

    Read in the engine's source (SIESTA 5.4.2): `siesta` holds no `TBT.*`
    label, so the device deck carries none; `tbtrans` reads the `TS.*`
    junction description and takes `TS.Voltage` and `TS.Elecs.Bulk` as the
    defaults of its own settings, so the transmission deck carries them; and
    `tbtrans` reads `TBT.k` only as a bracketed list or a block, so the bare
    triple the deck wrote until 2026-09-29 was skipped for the SCF's grid.
    """

    def test_each_deck_carries_what_its_own_program_reads(self, tmp_path,
                                                         monkeypatch):
        device, transmission = self._ladder(tmp_path, monkeypatch)
        dev = self._settings(device)
        assert not [ln for ln in dev if ln.upper().startswith(("TBT.",
                                                            "%BLOCK TBT"))], (
            "siesta reads no TBT.* keyword, so the device deck carries none")
        assert _says(device, "SolutionMethod", "transiesta")
        assert any(ln.startswith("TS.Elecs.Bulk") for ln in dev)

        tr = self._settings(transmission)
        # the junction as tbtrans reads it, and the bias it reads it at
        for needed in ("%block TS.Elecs", "%block TS.ChemPots",
                       "TBT.HS                 T.TS.HSX"):
            assert needed in tr, needed
        assert any(ln.startswith("TS.Voltage") for ln in tr)
        assert any(ln.startswith("TS.Elecs.Bulk") for ln in tr)
        assert not any(ln.startswith("SolutionMethod") for ln in tr), (
            "tbtrans runs no SCF")
        # ...nor any of the output group, which tbtrans compiles no reader
        # of: each line would be a setting nothing reads (§ 6.1b's audit).
        assert not [ln for ln in tr if ln.split()[0] in (
            "WriteForces", "WriteCoorStep", "WriteCoorXmol",
            "WriteMDhistory", "WriteMDXmol", "SaveHS")], tr
        # TBT.k is the block tbtrans reads, which carries the offset -- the
        # cited run's transverse grid, which `jobset init` put in the
        # template, and one point along transport (`engines/siesta.md` § 6.1)
        from molbuilder.template import read_template, find_template
        k = next(i.value for i in read_template(
            find_template(tmp_path / "projects" / "J" / "transport" / "T")
            .read_text()).items if i.name == "tbt_k_grid")
        block = transmission[transmission.index("%block TBT.k"):
                             transmission.index("%endblock TBT.k")]
        rows = [" ".join(ln.split()) for ln in block.splitlines()[1:4]]
        assert rows == [f"{k[0]} 0 0 0.0", f"0 {k[1]} 0 0.0",
                        "0 0 1 0.0"], rows
        assert "TBT.Verbosity          5" in tr
        # every TBT.* value is written with its note above it: the note is
        # headed by the keyword it explains
        lines = transmission.splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("TBT.") and not ln.startswith("TBT.HS"):
                key = ln.split()[0]
                above = "\n".join(lines[max(0, i - 25):i])
                assert f"# {key}" in above, f"{key} has no note above it"

    def test_the_leads_bulk_treatment_is_one_value_for_both_decks(
            self, tmp_path, monkeypatch):
        """`electrodes_bulk` is shared: set on the template it reaches both
        NEGF decks, and a single rung may not carry its own -- the device and
        the transmission would describe two different junctions.

        The road is `jobset prep`; the rung's own value is injected into
        `task.json` with `write_task`, the way a hand edit would put it there,
        because no door writes a shared item into a rung's bag."""
        from molbuilder.template import _emit, find_template, read_template
        import dataclasses

        def bulk_off(calc):
            tmpl = find_template(calc)
            items = [dataclasses.replace(i, value=False)
                     if i.name == "electrodes_bulk" else i
                     for i in read_template(tmpl.read_text()).items]
            tmpl.write_text(_emit(items, engines=("siesta",)))

        device, transmission = self._ladder(tmp_path, monkeypatch,
                                            edit=bulk_off)
        for deck in (device, transmission):
            assert "TS.Elecs.Bulk          .false." in self._settings(deck)

        # ...and one rung's own value is refused rather than taken
        from molbuilder.task import read_task, write_task
        calc = tmp_path / "projects" / "J" / "transport" / "T"
        task = read_task(calc / "task.json")
        write_task(calc / "task.json", dataclasses.replace(
            task, varies=("electrodes_bulk",),
            stages=tuple(dataclasses.replace(
                s, overrides={"electrodes_bulk": True})
                if s.name == "device" else s for s in task.stages)))
        r = self._cli(["prep", "run", "device", "--bundle", "J/transport/T"],
                      tmp_path / "projects", monkeypatch)
        assert r.exit_code != 0 and "electrodes_bulk" in r.output, r.output

    def test_a_template_naming_the_old_spelling_is_told_the_new_one(
            self, tmp_path, monkeypatch):
        """`elecs_bulk` became `electrodes_bulk` on 2026-09-29.  A template
        written before is refused -- never read as the new name -- and the
        refusal says what to rename the line to.  (The old template is made
        by renaming the row in a new one: no door writes the old name.)"""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        calc = _describe_transport(root, bias=(0.0,))
        from molbuilder.template import find_template
        tmpl = find_template(calc)
        tmpl.write_text(tmpl.read_text().replace("[item.electrodes_bulk]",
                                                 "[item.elecs_bulk]"))
        r = self._cli(["prep", "run", "seed", "--bundle", "J/transport/T"],
                      root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert "'elecs_bulk' is now 'electrodes_bulk'" in r.output, r.output

    def test_a_zero_that_means_the_programs_own_rule_writes_nothing(
            self, tmp_path, monkeypatch):
        """Two defaults are FORMULAS, and a 0 written in their place replaces
        the formula: an explicit 0 broadening overrides min(eta)/10.  So at 0
        nothing is written.  `TBT.Spin`'s default IS a number -- 0, every
        channel (`m_tbt_hs.F90`) -- so it is written.  (The pole energy was
        the third until M5 step 2: it is always written now, § 6.1c.)"""
        device, transmission = self._ladder(tmp_path, monkeypatch)
        dev, tr = self._settings(device), self._settings(transmission)
        assert not any(ln.startswith("TS.Contours.nEq.Eta") for ln in dev)
        assert not any(ln.startswith("TBT.Contours.Eta") for ln in tr)
        assert "TBT.Spin               0" in tr

    def test_each_bias_points_transmission_reads_that_points_voltage(
            self, tmp_path, monkeypatch):
        """tbtrans takes `TS.Voltage` as the default of its own voltage, and
        a transmission point must read its own point's converged device: its
        deck carries that point's voltage.  (tbtrans only WARNS when the
        voltage disagrees with the Hamiltonian it reads, `m_tbt_contour.F90`.)"""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        calc = _describe_transport(root, bias=(0.0, 0.2))
        for stage, files in (("seed", ["T.DM"]),
                             ("electrode_L", ["T_L-electrode.TSHS"]),
                             ("electrode_R", ["T_R-electrode.TSHS"])):
            r = self._cli(["prep", "run", stage, "--bundle", "J/transport/T"],
                          root, monkeypatch)
            assert r.exit_code == 0, (stage, r.output)
            _conclude(calc, stage, files)
        r = self._cli(["prep", "run", "device", "--bundle", "J/transport/T"],
                      root, monkeypatch)
        assert r.exit_code == 0, r.output
        for point in ("v0", "v0.2"):
            _conclude(calc, "device", ["T.TS.HSX"], point=point)
        r = self._cli(["prep", "run", "transmission", "--bundle",
                       "J/transport/T"], root, monkeypatch)
        assert r.exit_code == 0, r.output
        for point, volts in (("v0", "0.0000"), ("v0.2", "0.2000")):
            dev = (calc / "04_device" / point / "T_04_device.fdf").read_text()
            tr = (calc / "05_transmission" / point
                  / "T_05_transmission.fdf").read_text()
            for deck in (dev, tr):
                assert f"TS.Voltage             {volts} eV" in (
                    self._settings(deck)), (point, volts)

    def test_a_migration_keeps_a_renamed_items_value(self, tmp_path,
                                                      monkeypatch):
        """`jobset migrate` rewrites a template written before the electronic
        state; one that also names `elecs_bulk` keeps its value under
        `electrodes_bulk` and says so -- a migration keeps what the run was,
        and dropping the value would write the default in its place.
        (The pre-2026-09-28 template is made by editing a new one: no door
        writes the old spellings.)"""
        import re as _re
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        calc = _describe_transport(root, bias=(0.0,))
        from molbuilder.template import find_template
        tmpl = find_template(calc)
        def row_span(text, name):
            # a row runs from its header to the next item's header
            start = text.index(f"[item.{name}]")
            end = text.find("\n[item.", start + 1)
            return start, (len(text) if end < 0 else end)

        text = tmpl.read_text().replace("[item.electrodes_bulk]",
                                        "[item.elecs_bulk]")
        s, e = row_span(text, "elecs_bulk")
        text = text[:s] + _re.sub(r"^value = .*$", "value = false",
                                  text[s:e], count=1, flags=_re.M) + text[e:]
        # the spin row as SIESTA's own vocabulary wrote it before 2026-09-28,
        # declaration and all -- what `jobset migrate` exists to rewrite
        s, e = row_span(text, "spin_treatment")
        text = text[:s] + (
            '[item.spin_treatment]\nkind = "engine"\ncategory = ["system"]\n'
            'anchor = "Spin"\nengine_key = "Spin"\ntype = "enum"\n'
            'choices = ["non-polarized", "polarized", "non-colinear", '
            '"spin-orbit"]\nvalue = "non-polarized"\ngroup = "profile"\n'
            'help = "old"\n') + text[e:]
        tmpl.write_text(text)
        assert "value = false" in text and '"non-polarized"' in text
        r = self._cli(["migrate", "--bundle", "J/transport/T"], root,
                      monkeypatch)
        assert r.exit_code == 0, r.output
        assert "elecs_bulk = False -> electrodes_bulk (renamed)" in r.output, (
            r.output)
        from molbuilder.template import read_template
        got = {i.name: i.value for i in read_template(tmpl.read_text()).items}
        assert got["electrodes_bulk"] is False, got.get("electrodes_bulk")


class TestBeforeAnyDeviceRuns(_LadderThroughTheCli):
    """`engines/transport.md` § 6.1c (M5 step 2), through `molbuilder jobset
    prep`: the device's equilibrium contour stated with the count it gives,
    vacuum refused where the crystal continues and kept where a wire is
    isolated, and `tbtrans` asked for what the Results tab draws."""

    def _prep_seed(self, root, monkeypatch):
        return self._cli(["prep", "run", "seed", "--bundle", "J/transport/T"],
                         root, monkeypatch)

    @staticmethod
    def _described(tmp_path, **values):
        """A described ladder whose template states *values* -- set the way
        a person edits the file, through its own reader and writer."""
        from molbuilder.template import _emit, find_template, read_template
        import dataclasses
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        calc = _describe_transport(root, bias=(0.0,))
        tmpl = find_template(calc)
        tmpl.write_text(_emit(
            [dataclasses.replace(i, value=values[i.name])
             if i.name in values else i
             for i in read_template(tmpl.read_text()).items],
            engines=("siesta",)))
        return root, calc

    def test_the_pole_energy_is_written_with_the_count_it_gives(
            self, tmp_path, monkeypatch):
        """10 eV, always written, and beside it the count TranSIESTA takes at
        the run's own temperature -- the cited relaxation's 200 K here, where
        its rule, N = int(E / (pi k_B T)), gives 184 (123 at 300 K).  The
        count is a comment: libfdf ends a line's tokens at `#`, so `siesta`
        reads the energy alone."""
        device, _ = self._ladder(tmp_path, monkeypatch)
        assert ("TS.Contours.Eq.Pole    10.0000 eV   # 184 poles at 200 K"
                in self._settings(device)), [
            ln for ln in device.splitlines() if "Eq.Pole" in ln]

    @pytest.mark.parametrize("energy", [0.0, 1.0],
                             ids=["zero", "eighteen-poles"])
    def test_a_pole_energy_under_twenty_poles_is_refused_before_any_rung(
            self, tmp_path, monkeypatch, energy):
        """TranSIESTA stops a device run under 20 poles, after the queue
        wait; the settings gate refuses it at the first rung prepped, and
        names the least energy it would take.  0 is refused too -- it no
        longer leaves the choice to TranSIESTA, whose own 42 lost the charge
        on a real device, and TranSIESTA does not take it as an energy.  (1 eV
        is 18 poles at the fixture's 200 K.)"""
        root, calc = self._described(tmp_path, negf_eq_pole_ev=energy)
        r = self._prep_seed(root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert "TranSIESTA needs at least 20" in r.output, r.output
        assert "the least energy is 1.09 eV" in r.output, r.output
        assert not (calc / "01_seed" / "T_01_seed.fdf").exists()

    def test_the_floor_is_the_runs_own_temperature(self, tmp_path,
                                                  monkeypatch):
        """The count follows the temperature, so the floor does: 1.2 eV is
        22 poles at the fixture's 200 K and preps -- the same energy is 14 at
        300 K, so a gate that assumed 300 K would refuse it."""
        root, _ = self._described(tmp_path, negf_eq_pole_ev=1.2)
        r = self._prep_seed(root, monkeypatch)
        assert r.exit_code == 0, r.output

    def test_an_electronic_temperature_under_ten_kelvin_is_refused(
            self, tmp_path, monkeypatch):
        """TranSIESTA stops below 10 K before it counts a pole
        (`m_ts_options.F90`); refused at the first rung, naming the floor."""
        root, _ = self._described(tmp_path, electronic_temperature=5.0)
        r = self._prep_seed(root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert "below 10 K" in r.output, r.output

    def test_tbtrans_is_asked_for_the_dos_and_the_eigenchannels(
            self, tmp_path, monkeypatch):
        """For two electrodes `tbtrans` writes T(E) alone unless asked
        (`m_tbt_options.F90`); W35 decision 7 asks for the device DOS, the
        spectral DOS from the electrodes, the leads' bulk DOS and
        transmission, and four eigenchannels."""
        _, transmission = self._ladder(tmp_path, monkeypatch)
        for keyword, value in (("TBT.DOS.Gf", ".true."),
                               ("TBT.DOS.A", ".true."),
                               ("TBT.DOS.Elecs", ".true."),
                               ("TBT.T.Bulk", ".true."),
                               ("TBT.T.Eig", "4")):
            assert _says(transmission, keyword, value), keyword

    @pytest.mark.parametrize("room,refused", [(3.5, False), (4.0, True)],
                             ids=["1.4-spacings", "1.6-spacings"])
    def test_the_room_along_transport_is_measured_against_the_lead(
            self, tmp_path, monkeypatch, room, refused):
        """The leads continue through the transport boundary into the
        periodic image, so the room there is one of the lead's layer
        spacings; above 1.5 of them it is vacuum, refused before any rung
        runs and naming both numbers.  Against the chain's 2.5 Å: 3.5 Å
        preps -- which the fixed 3.0 Å rule this replaced would have flagged
        -- and 4.0 Å is refused."""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct(room=room))
        _describe_transport(root, bias=(0.0,))
        r = self._prep_seed(root, monkeypatch)
        if not refused:
            assert r.exit_code == 0, r.output
            return
        assert r.exit_code != 0, r.output
        assert "leaves 4.00 Å at the transport boundary" in r.output, r.output
        assert "2.50 Å layer spacings" in r.output, r.output

    def test_vacuum_on_a_periodic_transverse_axis_is_refused(
            self, tmp_path, monkeypatch):
        """Periodic says the crystal continues across the boundary.  The
        chain in its 8 Å box, declared periodic, reaches 8 Å across it
        against its own 2.5 Å bond: refused, naming the axis."""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct(across=("periodic",
                                                       "periodic")))
        _describe_transport(root, bias=(0.0,))
        r = self._prep_seed(root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert "axis a is declared periodic" in r.output, r.output
        assert "axis b is declared periodic" in r.output, r.output
        # each lead on its own, and named
        assert "nearest atom of L-electrode" in r.output, r.output
        assert "nearest atom of R-electrode" in r.output, r.output

    def test_an_isolated_wire_keeps_its_vacuum_through_the_ladder(
            self, tmp_path, monkeypatch):
        """A wire is isolated across transport, and its vacuum is what
        isolates it.  The declaration the relaxation's deck recorded reaches
        the composed junction -- `compose` stated every junction periodic
        across until 2026-09-29 -- and the lead cut from it, and every rung
        preps."""
        device, _ = self._ladder(tmp_path, monkeypatch)
        calc = tmp_path / "projects" / "J" / "transport" / "T"
        from molbuilder.sidecars.molstruct import load
        assert list(load(calc / "junction.molstruct.json")["axis_kind"]) == [
            "isolated", "isolated", "transport"]
        # each deck's own placement record, read by its one reader -- the
        # device open along transport, the lead cut from it periodic there
        from molbuilder.deck_record import extract_engine_offset
        lead = (calc / "02_electrode_L" / "T_02_electrode_L.fdf").read_text()
        for deck, along in ((device, "transport"), (lead, "periodic")):
            assert extract_engine_offset(deck)["axis_kind"] == [
                "isolated", "isolated", along], along



class TestTheRungFixesItsOwn(_LadderThroughTheCli):
    """What a transport rung fixes (`role`, `engines/template.md` § 6.4) is
    set by no door but the rung, through `molbuilder jobset prep`: the bias is
    the description's list, the device's solver NEGF.  Until 2026-09-29 a
    device override of the bias ran the device at one voltage while the
    transmission, which reads the device's Hamiltonian, ran at another
    (T-F5), and a template value was a single-bias calculation's voltage."""

    def _prep(self, stage, root, monkeypatch):
        return self._cli(["prep", "run", stage, "--bundle", "J/transport/T"],
                         root, monkeypatch)

    def _described(self, root, bias=()):
        _write_junction(root, _junction_struct())
        return _describe_transport(root, bias=bias)

    @pytest.mark.parametrize("rung", ["device", "seed"])
    def test_a_rung_cannot_carry_its_own_bias(self, tmp_path, monkeypatch,
                                              rung):
        """T-F5 -- on the rung that writes the bias and on one that does
        not: both meet the one refusal, never *"move it to the device"*,
        which the device would refuse in turn.  MUTATIONS THIS MUST FAIL
        AGAINST: the stage-override door left open for a fixed item (the
        override is laid over by the answer and the prep says nothing); a
        fixed item counted as a rung's to own (the seed is sent to the
        device)."""
        root = tmp_path / "projects"
        calc = self._described(root)
        t = json.loads((calc / "task.json").read_text())
        for st in t["stages"]:
            if st["name"] == rung:
                st["overrides"] = {"bias_voltage_v": 0.5}
        t["varies"] = ["bias_voltage_v"]
        (calc / "task.json").write_text(json.dumps(t, indent=2) + "\n")
        r = self._prep(rung, root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert "'bias_voltage_v'" in r.output, r.output
        assert "the rung fixes" in r.output and "bias list" in r.output, (
            r.output)
        assert "belongs to" not in r.output, r.output

    @pytest.mark.parametrize("bias, stated", [((), 0.3), ((0.0, 0.2), 0.3),
                                              ((0.0, 0.2), 0.0)])
    def test_a_template_bias_is_refused_naming_the_list(self, tmp_path,
                                                       monkeypatch, bias,
                                                       stated):
        """THE SECOND HOME, closed -- with a list and without one, and at
        0 V too: a rung answers the bias point by point, so no one value in
        a calculation-wide file states it.  MUTATIONS THIS MUST FAIL
        AGAINST: the template door left open (the rung lays 0 V over the
        template's 0.3 without a word); a template value accepted when it
        equals the catalogue's 0 V (the v0.2 deck then runs at 0.2 beside
        a template that says 0)."""
        import dataclasses
        from molbuilder.template import _emit, find_template, read_template
        root = tmp_path / "projects"
        calc = self._described(root, bias=bias)
        tmpl = find_template(calc)
        parsed = read_template(tmpl.read_text())
        tmpl.write_text(_emit(
            [dataclasses.replace(i, value=stated)
             if i.name == "bias_voltage_v" else i for i in parsed.items],
            engines=parsed.engines))
        r = self._prep("device", root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert "'bias_voltage_v'" in r.output and "bias list" in r.output, (
            r.output)

    def test_each_rung_writes_what_it_fixes_from_its_config(
            self, tmp_path, monkeypatch):
        """ONE ROAD: a rung's answer -- the device's own (`role_values`),
        the leads' the item's value -- is laid on its config by `resolve`,
        and the section walk writes it, noting it is fixed.  A block typed
        both lines until 2026-09-29, while the device's config said
        `diagon` to the gate and the record.  MUTATIONS THIS MUST FAIL
        AGAINST: the answer not laid on (the device's section says
        `diagon`); the walk skipping a fixed item (no line at all)."""
        import re as _re
        device, _transmission = self._ladder(tmp_path, monkeypatch)
        calc = tmp_path / "projects" / "J" / "transport" / "T"
        lead = (calc / "02_electrode_L" / "T_02_electrode_L.fdf").read_text()
        for deck, keyword, value in ((device, "SolutionMethod", "transiesta"),
                                     (lead, "TS.HS.Save", ".true."),
                                     (lead, "SolutionMethod", "diagon")):
            lines = deck.splitlines()
            at = [k for k, ln in enumerate(lines)
                  if _re.match(rf"{_re.escape(keyword)}\s", ln)]
            assert len(at) == 1, (keyword, [lines[k] for k in at])
            assert lines[at[0]].split()[1:] == [value], lines[at[0]]
            assert any("Fixed by this rung" in ln
                       for ln in lines[max(0, at[0] - 3):at[0]]), (
                lines[max(0, at[0] - 3):at[0] + 1])


class TestTheSpinTranSIESTARuns(_LadderThroughTheCli):
    """T-F20, through `molbuilder jobset prep`: TranSIESTA runs two spin
    treatments and never a fixed total spin (`engines/transport.md` § 3.1's
    spin note; `science/chemistry-correctness.md` § 2a.3, ES6).  Until
    2026-09-30 a transport calculation was offered all four treatments and
    any count, and the device died on the node."""

    def _template_says(self, calc, **values):
        import dataclasses
        from molbuilder.template import _emit, find_template, read_template
        tmpl = find_template(calc)
        parsed = read_template(tmpl.read_text())
        tmpl.write_text(_emit(
            [dataclasses.replace(i, value=values[i.name])
             if i.name in values else i for i in parsed.items],
            engines=parsed.engines))

    def _seed(self, root, monkeypatch):
        return self._cli(["prep", "run", "seed", "--bundle", "J/transport/T"],
                         root, monkeypatch)

    @pytest.mark.parametrize("values, words", [
        ({"spin_treatment": "non-collinear"},
         "more than two spin components"),
        ({"spin_treatment": "unrestricted", "unpaired_electrons": 2},
         "cannot hold a fixed total spin"),
        ({"spin_treatment": "unrestricted", "unpaired_electrons": 0},
         "cannot be held here"),
        # ES5 first, and its way out the one TranSIESTA can take
        ({"spin_treatment": "restricted", "unpaired_electrons": 2},
         "its count floats here"),
    ], ids=["non-collinear", "a-count", "a-zero-under-unrestricted",
            "restricted-with-a-count"])
    def test_what_transiesta_cannot_run_is_refused(self, tmp_path,
                                                   monkeypatch, values,
                                                   words):
        """MUTATIONS THIS MUST FAIL AGAINST: the transport kind's treatment
        set undeclared (non-collinear preps); the transport clause of the
        float-only rule removed (the zero preps, writing `Spin.Fix`).  (The
        count's set is held by the float-only rule as well, so the form
        test in `test_what_a_kind_offers_e2e.py` is its catch.)"""
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        calc = _describe_transport(root, bias=(0.0,))
        self._template_says(calc, **values)
        r = self._seed(root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert words in r.output and "TranSIESTA" in r.output, r.output

    def test_a_cited_fixed_count_floats(self, tmp_path, monkeypatch):
        """The relaxation a transport cites ran polarized at a fixed 2S = 2.
        Its number is not a value TranSIESTA can take, so the template
        leaves the count blank and it floats by the rule, which the deck
        says.  MUTATION THIS MUST FAIL AGAINST: the citation carrying the
        number (every prep refused until the person edits the template)."""
        root = tmp_path / "projects"
        calc_relax = _write_junction(root, _junction_struct())
        deck = calc_relax / "01_coarse" / "run-0" / "Relax_01_coarse.fdf"
        deck.write_text(deck.read_text().replace(
            "ElectronicTemperature 200.0 K\n",
            "ElectronicTemperature 200.0 K\nSpin polarized\n"
            "Spin.Fix .true.\nSpin.Total 2.0\n"))
        calc = _describe_transport(root, bias=(0.0,))
        from molbuilder.template import find_template, one, read_template
        tmpl = read_template(find_template(calc).read_text())
        assert one(tmpl, "spin_treatment").value == "unrestricted"
        assert one(tmpl, "unpaired_electrons").value is None
        r = self._seed(root, monkeypatch)
        assert r.exit_code == 0, r.output
        seed = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        settings = self._settings(seed)
        assert "Spin polarized" in [" ".join(ln.split()) for ln in settings]
        assert not [ln for ln in settings
                    if ln.split()[0] in ("Spin.Fix", "Spin.Total")], settings
        assert "TranSIESTA cannot hold a fixed total spin" in seed

    def test_a_cited_treatment_transiesta_cannot_run_is_worked_out(
            self, tmp_path, monkeypatch):
        """The relaxation a transport cites ran non-collinear, which
        TranSIESTA cannot: the template leaves the treatment and its count
        blank, and the junction's own is worked out and said in the deck.
        MUTATION THIS MUST FAIL AGAINST: the citation carrying the
        treatment (every prep refused until the person edits the template,
        while the shared panel shows no such choice)."""
        root = tmp_path / "projects"
        calc_relax = _write_junction(root, _junction_struct())
        deck = calc_relax / "01_coarse" / "run-0" / "Relax_01_coarse.fdf"
        deck.write_text(deck.read_text().replace(
            "ElectronicTemperature 200.0 K\n",
            "ElectronicTemperature 200.0 K\nSpin non-colinear\n"))
        calc = _describe_transport(root, bias=(0.0,))
        from molbuilder.template import find_template, one, read_template
        tmpl = read_template(find_template(calc).read_text())
        assert one(tmpl, "spin_treatment").value is None
        assert one(tmpl, "unpaired_electrons").value is None
        r = self._seed(root, monkeypatch)
        assert r.exit_code == 0, r.output
        seed = (calc / "01_seed" / "T_01_seed.fdf").read_text()
        assert "Spin non-polarized" in [" ".join(ln.split())
                                        for ln in self._settings(seed)]

    def test_a_recorded_treatment_is_refused_saying_it_was_recorded(
            self, tmp_path, monkeypatch):
        """A pair whose record says it ran non-collinear, cited with the
        spin left blank: the junction takes the recorded treatment (ES7),
        which TranSIESTA cannot run -- refused on the junction, saying the
        value was RECORDED, where every rung's gate would have called the
        folded value *stated*.  MUTATION THIS MUST FAIL AGAINST: the
        junction's own check removed (the refusal says "stated")."""
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        pair = root / "J" / "structure" / "junc"
        pair.mkdir(parents=True)
        s = _junction_struct()
        s.info = dict(s.info or {}, calculation={
            "engine": "siesta", "source": "Relax.fdf",
            "contract": {"net_charge": 0, "spin_treatment": "non-collinear",
                         "unpaired_electrons": "free"}})
        StructureCodec().write(s, pair / "junction.xyz")
        write_pseudos(pair, ["Au", "S", "C"])
        calc = _describe_transport(root, cite="J/structure/junc",
                                   bias=(0.0,))
        r = self._seed(root, monkeypatch)
        assert r.exit_code != 0, r.output
        assert "spin_treatment = non-collinear (recorded" in r.output, (
            r.output)

"""P4b — prep renders the transport composite's five stages
(`plans/transport-design.md` § 4.2, the arm in `jobset/prep.py` +
`transport/stages.py`).

The design's own gate row, verbatim: *"A fixture junction preps
end-to-end; each gate refuses its mutation; the emitter's own
order-preflight never fires (prep sorted first)."*

Properties under guard, each named for its failure:

* each stage's deck is born in its own stage directory, wrapper beside
  it, through the SHARED prep tail (job-set merge, STAGE-PLAN, run
  dirs) — no forked machinery;
* the electronic contract (basis · XC · energy shift · mesh · k ·
  electronic T) is read from the CITED attempt's own deck and lands in
  every stage's deck — one template governs (ruling Q5, fdf-is-truth);
* the electrode deck's SystemLabel IS the ``.TSHS`` stem the device
  deck references — one spelling, both writers;
* the emitter's order-preflight never fires: a source whose atom order
  would trip it preps clean, because prep sorted first (§ 4.1a);
* buffer atoms emit ``TS.Atoms.Buffer`` + explicit electrode positions
  (§ 3, buffer sanity);
* the composed record is written once, reused by later stages, travels
  with the folder, and a re-pointed citation recomposes;
* refusals: unnamed/unknown/disabled stage, a sweep, a moved frozen
  atom, a pseudopotential the citation cannot supply.
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

_CITE = "J/optimization/Relax@01_coarse/run-0"
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
    (attempt / "Relax_01_coarse.fdf").write_text(_CITED_DECK)
    (attempt / "Relax_01_coarse-run0.concluded").write_text("rc=0\n")
    _write_xv(attempt / "Relax.XV", struct)
    write_pseudos(calc, ["Au", "S", "C"])
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

    def test_the_electrode_deck_is_the_tshs_the_device_asks_for(self, calc):
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
        prep_calculation(calc, "seed")
        stamp = (calc / "junction.xyz").stat().st_mtime_ns
        prep_calculation(calc, "electrode_L")
        assert (calc / "junction.xyz").stat().st_mtime_ns == stamp, (
            "a later stage loads the record instead of recomposing")

    def test_a_repointed_citation_recomposes(self, calc, tmp_path):
        prep_calculation(calc, "seed")
        # a second concluded attempt with a different relaxed geometry
        root = tmp_path / "projects"
        attempt = root / "J" / "optimization" / "Relax" / "01_coarse" \
            / "run-1"
        attempt.mkdir()
        (attempt / "Relax_01_coarse.fdf").write_text(_CITED_DECK)
        (attempt / "Relax_01_coarse-run1.concluded").write_text("rc=0\n")
        _write_xv(attempt / "Relax.XV", _junction_struct(),
                  perturb_bridge=0.4)
        cite2 = "J/optimization/Relax@01_coarse/run-1"
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
        with pytest.raises(PrepError) as e:
            prep_calculation(calc)
        msg = str(e.value)
        assert "seed" in msg and "transmission" in msg

    def test_an_unknown_stage_is_refused_by_name(self, calc):
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "coarse")
        assert "'coarse'" in str(e.value).replace('"', "'")

    def test_a_disabled_seed_refuses_with_the_skip_rule(self, calc):
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
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed", sweep={"x": [1, 2]})
        assert "bias" in str(e.value)

    def test_a_moved_frozen_atom_stops_prep(self, calc, tmp_path):
        root = tmp_path / "projects"
        _write_xv(root / "J/optimization/Relax/01_coarse/run-0/Relax.XV",
                  _junction_struct(), perturb_electrode=(0, 0.05))
        with pytest.raises(PrepError) as e:
            prep_calculation(calc, "seed")
        assert "MOVED" in str(e.value)

    def test_a_pseudo_the_citation_cannot_supply_is_named(self, calc,
                                                          tmp_path):
        (tmp_path / "projects" / "J" / "optimization" / "Relax"
         / "Au.psml").unlink()
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
        r = self._invoke(["prep", "run", "seed", "--bundle",
                          "J/transport/T"], tmp_path / "projects",
                         monkeypatch)
        assert r.exit_code == 0, r.output
        assert (calc / "01_seed" / "run-0" / "T_01_seed.fdf").is_file(), (
            "the CLI tail opens the attempt like any ladder rung")

    def test_prep_device_gathers_through_the_cli(self, calc, tmp_path,
                                                 monkeypatch):
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
        r = self._invoke(["prep", "bench", "device", "--bundle",
                          "J/transport/T"], tmp_path / "projects",
                         monkeypatch)
        assert r.exit_code != 0
        assert "no benchmark" in r.output

    def test_init_refuses_the_flat_shape(self, calc, tmp_path,
                                         monkeypatch):
        r = self._invoke(["init", "--calculation", "transport",
                          "--shape", "flat",
                          "--bundle", "J/transport/T2",
                          "--slot", f"junction={_CITE}"],
                         tmp_path / "projects", monkeypatch)
        assert r.exit_code != 0
        assert "hierarchical" in r.output


class TestTheBiasScan:
    """P5b — the bias chain (transport-design.md § 4.3; layout ruled
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
        from molbuilder.jobset.submit import (SubmitError,
                                              submit_transport_chain)
        task, js = self._ready(calc, tmp_path, monkeypatch)
        shutil.rmtree(calc / "04_device" / "v0.2" / "run-0")
        with pytest.raises(SubmitError) as e:
            submit_transport_chain(js, calc, task, mode="direct")
        assert "v0.2" in str(e.value) and "prep run device" in str(e.value)

    def test_a_launched_point_refuses_relaunch(self, calc, tmp_path,
                                               monkeypatch):
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

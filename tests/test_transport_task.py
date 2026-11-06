"""The transport calculation KIND -- `engines/transport.md` § 1 and § 3.1: one
slot, one explicitly named directory, nothing picked for you -- asked where a
person meets it, `molbuilder describe`.

* `init --calculation transport` refuses: no junction slot or a second one, a
  citation that is not tree-relative, and every option whose answer arrives
  via the citation (--structure / --vacuum / --stage-strategy); beside
  `--reuses`, every option naming what the extension keeps (the kind's
  `describe.not_taken`), a window flag spelled wrong, and a folder that is
  not a transport calculation; and `--slot` on any other kind.  Each refusal is the door's own -- the
  codec's where the codec owns the rule (`task.py`), surfaced by name.  The
  refusal that comes after a citation is read -- a bias list that does not
  start at 0.0 -- needs a finished relaxation to cite, so it is asked on the
  junction the end-to-end pass relaxes
  (`tests/test_transport_on_a_real_junction_e2e.py`, plan § 5y).
* the codec round-trips slots + bias and carries NO structure block for the
  kind (its structure IS the citation): API-level, since the file is the one
  door every road reads a description through (`template.md` § 5.3) and the
  round trip is the codec's own property.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from molbuilder.identity import run_id
from molbuilder.task import Run, Stage, Task
from molbuilder.jobset._cli import describe_cmd

_CITE = "BDT-Au/optimization/JunctionRelax/01_coarse/run-2"
_STAGES = ("seed", "electrode_L", "electrode_R", "device", "transmission")


def _stages():
    return tuple(Stage(name=n, overrides={}) for n in _STAGES)


def test_round_trip_and_no_structure_block():
    """The codec's own property (`engines/transport.md` § 2a.14): a transport
    description carries its slot and its bias list, and no structure block --
    its structure IS the citation."""
    t = Task(engine="siesta", run=Run(name="T", id=run_id("T", _CITE, stage_names=_STAGES)),
             structure=None, calculation="transport",
             slots={"junction": _CITE}, bias=(0.0, 0.2),
             low_bias_approximation=False, varies=(), stages=_stages())
    d = t.to_dict()
    assert "structure" not in d, (
        "a transport description's structure IS its citation")
    assert d["slots"] == {"junction": _CITE}
    assert d["bias"] == {"voltages_v": [0.0, 0.2],
                         "low_bias_approximation": False}
    t2 = Task.from_dict(d)
    assert t2.slots == t.slots and t2.bias == (0.0, 0.2)
    assert t2.structure is None
    assert [s.name for s in t2.stages] == list(_STAGES)


@pytest.mark.parametrize("names", [
    ("electrode_L", "electrode_R", "device", "transmission"),
    ("electrode_L", "seed", "electrode_R", "device", "transmission"),
], ids=["without-the-seed", "out-of-order"])
def test_a_ladder_is_its_five_rungs_every_one(names):
    """`engines/transport.md` § 3.8.2a, the kind table (user,
    2026-10-12: "Always keep the seed"): a transport description's stages
    are the five rungs, every one, in order -- the reader refuses any other
    list, naming the five.  Without the seed the device's SCF would start
    from atomic densities, and nothing downstream would say so; the codec is
    the one door every road reads a description through.

    MUTATION THIS MUST FAIL AGAINST: the reader letting the seed be left out
    (ruling Q4's skip, withdrawn)."""
    stages = tuple(Stage(name=n, overrides={}) for n in names)
    with pytest.raises(ValueError) as got:
        Task(engine="siesta", run=Run(name="T", id=run_id("T", _CITE, stage_names=names)),
             structure=None, calculation="transport",
             slots={"junction": _CITE}, bias=(0.0, 0.2),
             low_bias_approximation=False, varies=(), stages=stages)
    assert ("stages are its rungs, every one, in order -- seed, "
            "electrode_L, electrode_R, device, transmission") in str(got.value)


def test_a_kind_on_an_engine_it_does_not_run_on_is_refused_by_the_reader():
    """`execution/architecture.md` § 3.3: a kind asked on an engine it has
    no part for is refused by the one door, naming the engines it runs on
    -- the reader every road reads a description through, so a hand-written
    transport description on PySCF stops here and not as an error deep in
    resolve.

    MUTATION THIS MUST FAIL AGAINST: the reader asking the lenient
    ``kind_table.pair`` (None for the pair) instead of the kind's part."""
    with pytest.raises(ValueError) as got:
        Task(engine="pyscf", run=Run(name="T", id=run_id("T", _CITE, stage_names=_STAGES)),
             structure=None, calculation="transport",
             slots={"junction": _CITE}, bias=(0.0,), varies=(),
             stages=_stages())
    assert ("a transport calculation does not run on 'pyscf' -- it runs on "
            "siesta") in str(got.value)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    from molbuilder.projects import PROJECTS_ROOT_ENV
    root = tmp_path / "projects"
    attempt = (root / "BDT-Au" / "optimization" / "JunctionRelax"
               / "01_coarse" / "run-2")
    attempt.mkdir(parents=True)
    # The refusals below fire before a citation is read.
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
    return root


def _init(args):
    return CliRunner().invoke(describe_cmd,
                              ["--engine", "siesta"] + args)


_TRANSPORT = ["--calculation", "transport", "--bundle", "BDT-Au/transport/T"]


def _cited_transport(tree, *, frames, bias):
    """`molbuilder describe` of a transport calculation citing a structure
    pair of ``frames`` frames, each moving its bridge as a frame set's must, its
    electrodes held.  No run made this pair, a run being the engine's: it
    states a level of theory through the door that names one, which is all a
    citation asks of it, and the subject here is what comes after."""
    from conftest import write_pseudos
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.parse.contract import contract_fields_of
    from molbuilder.transport.sort import REGION_BRIDGE
    from molbuilder.workingcopy_structure import StructureCodec
    from support.junction import _junction_struct
    built = _junction_struct()
    built.set_info("calculation", {
        "engine": "siesta", "source": "J.fdf",
        "contract": contract_fields_of(SiestaConfig(system_label="J"),
                                       kind="optimization")})
    if frames > 1:
        moved = [built.positions.copy() for _ in range(frames)]
        for i, at in enumerate(moved):
            at[built.regions[REGION_BRIDGE]] += 0.01 * i
        built = built.with_frames(moved)
    (tree / "BDT-Au" / "structure").mkdir(parents=True)
    StructureCodec().write(built, tree / "BDT-Au" / "structure" / "j.xyz")
    (tree / "pseudopotential").mkdir()
    write_pseudos(tree / "pseudopotential", sorted(set(built.elements)))
    r = _init(_TRANSPORT + ["--slot", "junction=BDT-Au/structure/j.xyz",
                            "--psml-lib", "pseudopotential", "--bias", bias]
              + (["--no-low-bias-approximation"] if "," in bias else []))
    assert r.exit_code == 0, r.output
    return tree / "BDT-Au" / "transport" / "T"


@pytest.mark.parametrize("frames, bias, walk", [
    (2, "0,0.2", "one job walking its 4 points (each point's wall, summed)"),
    (2, "0", "one job walking its 2 points (each point's wall, summed)"),
    (1, "0,0.2", "one job walking its 2 points (each point's wall, summed)"),
    (1, "0", None),
], ids=["two frames, two voltages", "two frames, one voltage",
        "one frame, two voltages", "one frame, one voltage"])
def test_a_run_of_points_is_told_from_its_citation_before_any_prep(
        tree, frames, bias, walk):
    """`execution/runs-of-points.md` § 1.2: its wall is typed before its first
    prep, so the run card says what the job walks then, and the count is what
    the cited frame set holds.

    THE SILENT FAILURE: the count read from the record prep writes -- *its 2
    points* said of a run of 4, and of a frame set with one voltage nothing, so
    its wall is typed for one run rather than each point's."""
    from support.road import _road_points_before_prep
    _road_points_before_prep({"device": walk},
                             _cited_transport(tree, frames=frames, bias=bias))


def test_the_device_s_scratch_folder_is_said_on_every_page(
        tree, web_client):
    """`engines/transport.md` § 2a.11: the device stage's row on `jobset status`,
    Task setup and the Results ladder carries one line about the files its run
    keeps, read by one door.

    THE SILENT FAILURE: the person choosing a queue on Task setup is not told a
    run will want tens of GiB of scratch, which only the terminal said."""
    from support.road import jobset
    bundle = _cited_transport(tree, frames=1, bias="0,0.2")
    r = jobset("prep", "task", "--group", "1", "--group", "2", "--group", "3",
               "--bundle", bundle, "--target", "this", "--np", "2",
               "--cpus-per-task", "1")
    assert r.exit_code == 0, r.output
    status = jobset("status", "--bundle", bundle)
    rows = web_client.get(f"/api/task-setup/folder?dir={bundle}").get_json()[
        "ladder"]["stages"]
    said = {row["stage"]: row["scratch"] for row in rows}
    assert said["device"] and said["device"].startswith("tsgf/: ")
    assert said["device"] in status.output, status.output
    assert said["transmission"] is None
    # ...AND THE FOLDER ITSELF SAYS WHAT IT IS on the Results tab, where nothing
    # in it opens (`web/results.md` § 2): the place note is drawn from this.
    place = web_client.get(
        "/api/results/dir?path=" + str(bundle / "04_device" / "run-0" / "tsgf")
    ).get_json()["place"]
    assert place["role"] == "scratch" and "the engine's scratch" in place["says"]
    run = web_client.get(
        "/api/results/dir?path=" + str(bundle / "04_device" / "run-0")
    ).get_json()["place"]
    assert "the engine's scratch" not in run["says"]


class TestInitCLI:

    def test_an_engine_that_runs_no_transport_is_refused_in_the_tabs_words(
            self, tree):
        """`molbuilder describe` refuses an engine the kind runs no part on
        in the one sentence the engine tab and the web doors say -- the
        kind's own (`CalculationKind.why_not_on`, `engines/transport.md`
        § 3.8.8) -- before any citation is read."""
        from molbuilder.kind_table import kind
        r = CliRunner().invoke(describe_cmd, ["--engine", "pyscf"] + _TRANSPORT
                               + ["--slot", f"junction={_CITE}"])
        assert r.exit_code != 0
        assert kind("transport").why_not_on("pyscf") in r.output, r.output
        assert "no PySCF transport backend is built yet" in r.output

    def test_the_junction_slot_is_required(self, tree):
        """One slot, `junction`, and nothing else (§ 3.1): none and two are
        refused by name before any citation is read."""
        r = _init(_TRANSPORT)
        assert r.exit_code != 0 and "junction" in r.output
        r = _init(_TRANSPORT + ["--slot", f"junction={_CITE}",
                                "--slot", f"other={_CITE}"])
        assert r.exit_code != 0 and "junction" in r.output

    def test_a_traversing_citation_is_refused(self, tree):
        """A citation is a tree-relative directory: no leading '/', no '..'
        (`task.py`, the codec's form; the tree fence, § 2.5b)."""
        for bad in ("/abs/path", "a/../b"):
            r = _init(_TRANSPORT + ["--slot", f"junction={bad}"])
            assert r.exit_code != 0, bad
            assert bad in r.output or "tree" in r.output, r.output

    def test_the_cited_calculation_must_exist_in_the_tree(self, tree):
        r = _init(_TRANSPORT + ["--slot", "junction=Nope/optimization/Gone/run-0"])
        assert r.exit_code != 0
        assert "Nope/optimization/Gone" in r.output

    def test_options_answered_by_the_citation_are_refused(self, tree):
        (tree / "BDT-Au" / "structure").mkdir(parents=True, exist_ok=True)
        (tree / "BDT-Au" / "structure" / "j.xyz").write_text(
            "1\n\nH 0 0 0\n")
        for extra in (["--structure", "BDT-Au/structure/j.xyz"],
                      ["--vacuum", "8"],
                      ["--stage-strategy", "publishable"]):
            r = _init(_TRANSPORT + ["--slot", f"junction={_CITE}"] + extra)
            assert r.exit_code != 0, f"{extra} must be refused"
            assert f"{extra[0]} does not apply to" in r.output, r.output

    @pytest.mark.parametrize("extra, words", [
        (["--name", "R"], "--name does not apply beside --reuses"),
        (["--psml-lib", "pseudopotential"],
         "--psml-lib does not apply beside --reuses"),
        (["--structure", "<the road's structure>"],
         "--structure does not apply beside --reuses"),
        (["--vacuum", "8"], "--vacuum does not apply beside --reuses"),
        (["--stage-strategy", "publishable"],
         "--stage-strategy does not apply beside --reuses"),
        (["--finer", "3"], "--finer cuts the window fitted by --window"),
        (["--window", "lo"], "--window lo: two energies in eV"),
        ([], "is not a transport calculation"),
    ], ids=["name", "psml-lib", "structure", "vacuum", "stage-strategy",
            "finer-alone", "window-spelled-wrong", "not-transport"])
    def test_what_an_extension_does_not_take_is_refused(self, tmp_path,
                                                        monkeypatch, extra,
                                                        words):
        """`engines/transport.md` § 2a.16: beside `--reuses`, an option
        naming what the extension keeps is refused by name, and a folder
        that is not a transport calculation's is refused in its
        description's words -- before anything is written.  THE SILENT
        FAILURE: an option taken and dropped, the extension written as if
        it had been read.  The folder reused is an optimization described
        by the road (`support.road.describe_calculation`)."""
        from support.road import STRUCTURE, describe_calculation
        finished = describe_calculation(tmp_path, monkeypatch)
        extra = [STRUCTURE if a == "<the road's structure>" else a
                 for a in extra]
        rel = finished.relative_to(tmp_path / "projects").as_posix()
        r = _init(["--calculation", "transport", "--bundle",
                   "P/transport/R", "--reuses", rel] + extra)
        assert r.exit_code != 0, r.output
        assert words in r.output, r.output
        assert not (tmp_path / "projects" / "P" / "transport").exists()

    def test_slot_on_a_non_transport_init_is_refused(self, tree):
        (tree / "BDT-Au" / "structure").mkdir(parents=True, exist_ok=True)
        (tree / "BDT-Au" / "structure" / "j.xyz").write_text(
            "1\n\nH 0 0 0\n")
        r = _init([
            "--calculation", "optimization", "--structure", "BDT-Au/structure/j.xyz",
            "--bundle", "BDT-Au/optimization/X",
            "--slot", f"junction={_CITE}"])
        assert r.exit_code != 0
        assert "transport" in r.output

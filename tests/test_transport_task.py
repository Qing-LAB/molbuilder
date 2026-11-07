"""The transport calculation KIND -- `engines/transport.md` § 1 and § 3.1: one
slot, one explicitly named directory, nothing picked for you -- asked where a
person meets it, `molbuilder describe`.

* `init --calculation transport` refuses: no junction slot or a second one, a
  citation that is not tree-relative, and an option whose answer arrives
  via the citation; beside
  `--reuses`, an option naming what the extension keeps (the kind's
  `describe.not_taken`), a window flag spelled wrong, and a folder that is
  not a transport calculation; and `--slot` on any other kind.  Each refusal is
  the door's own -- the codec's where the codec owns the rule (`task.py`),
  surfaced by name.
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


def test_the_k_points_are_counted_as_transiesta_places_them():
    """`find_kgrid.F`: an even grid holds Gamma and the zone boundary unshifted
    -- 4 x 4 x 1 is 10 points, 8 with the half-step shift.  THE SILENT FAILURE:
    a scratch size counted on the grid's 25 where the engine holds 13."""
    from molbuilder.transport.gf_files import k_points_time_reversed
    assert k_points_time_reversed([(4, 0.0), (4, 0.0), (1, 0.0)]) == 10
    assert k_points_time_reversed([(4, 0.5), (4, 0.5), (1, 0.0)]) == 8


def test_two_points_in_one_folder_are_refused_by_the_reader():
    """`engines/template.md` § 5.3: each voltage is one device run in a folder
    of its own, so a repeated one is refused by the door every road reads a
    description through.  THE SILENT FAILURE: two points overwriting one
    folder."""
    with pytest.raises(ValueError) as got:
        Task(engine="siesta",
             run=Run(name="T", id=run_id("T", _CITE, stage_names=_STAGES)),
             structure=None, calculation="transport",
             slots={"junction": _CITE}, bias=(0.0, 0.2, 0.2),
             low_bias_approximation=False, varies=(), stages=_stages())
    assert "the bias list repeats 0.2 V" in str(got.value)


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
    root.mkdir()
    # The refusals below fire before a citation is read, so no run is made
    # for one: `_CITE` names a folder nothing ran in, and none is there.
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
    return root


def _init(args):
    return CliRunner().invoke(describe_cmd,
                              ["--engine", "siesta"] + args)


_TRANSPORT = ["--calculation", "transport", "--bundle", "BDT-Au/transport/T"]


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
        r = _init(_TRANSPORT + ["--slot", f"junction={_CITE}", "--vacuum", "8"])
        assert r.exit_code != 0
        assert "--vacuum does not apply to" in r.output, r.output

    @pytest.mark.parametrize("extra, words", [
        (["--name", "R"], "--name does not apply beside --reuses"),
        (["--finer", "3"], "--finer cuts the window fitted by --window"),
        (["--window", "lo"], "--window lo: two energies in eV"),
        ([], "is not a transport calculation"),
    ], ids=["name", "finer-alone", "window-spelled-wrong", "not-transport"])
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
        from support.road import describe_calculation
        finished = describe_calculation(tmp_path, monkeypatch)
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

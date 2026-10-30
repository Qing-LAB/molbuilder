"""The transport calculation KIND -- `engines/transport.md` § 1 and § 3.1: one
slot, one explicitly named directory, nothing picked for you -- asked where a
person meets it, `molbuilder describe`.

* `init --calculation transport` refuses: no junction slot or a second one, a
  citation that is not tree-relative, and every option whose answer arrives
  via the citation (--structure / --psml-lib / --vacuum / --stage-strategy);
  and `--slot` on any other kind.  Each refusal is the door's own -- the
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


class TestInitCLI:

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
            assert "transport" in r.output

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

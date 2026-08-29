"""The transport calculation KIND — `plans/transport-design.md` § 4.1
(build P3): one slot by explicit citation, the bias axis, floor 2 =
task.json alone.

Properties under guard, each named for its failure:

* the codec round-trips slots + bias, and a transport description
  carries NO structure block (its structure IS the citation);
* the rulings are refusals, not conventions: junction required, @run-N
  mandatory (Q1), bias starts at 0.0 (the .TSDE chain starts from
  equilibrium), slots/bias on a non-transport task refused;
* `init --calculation transport` writes the five-stage task.json and
  refuses every option whose answer arrives via the citation
  (--structure / --psml-lib / --vacuum / --stage-strategy);
* the identity derives from the citation — two transports of two
  junctions can never share an id.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from molbuilder.identity import run_id
from molbuilder.task import Run, Stage, Task

_CITE = "BDT-Au/optimization/JunctionRelax@run-2"
_STAGES = ("seed", "electrode_L", "electrode_R", "device", "transmission")


def _stages():
    return tuple(Stage(name=n, enabled=True, overrides={}) for n in _STAGES)


def _task(**over):
    kw = dict(engine="siesta", shape="hierarchical",
              run=Run(name="T", id=run_id("T", _CITE,
                                          stage_names=_STAGES)),
              structure=None, calculation="transport",
              slots={"junction": _CITE}, bias=(0.0, 0.2),
              varies=(), stages=_stages())
    kw.update(over)
    return Task(**kw)


class TestCodec:

    def test_round_trip_and_no_structure_block(self):
        t = _task()
        d = t.to_dict()
        assert "structure" not in d, (
            "a transport description's structure IS its citation")
        assert d["slots"] == {"junction": _CITE}
        assert d["bias"] == {"voltages_v": [0.0, 0.2]}
        t2 = Task.from_dict(d)
        assert t2.slots == t.slots and t2.bias == (0.0, 0.2)
        assert t2.structure is None
        assert [s.name for s in t2.stages] == list(_STAGES)

    def test_a_structure_block_on_transport_is_refused(self):
        d = _task().to_dict()
        d["structure"] = {"source": "x.xyz", "formula": "H2", "atoms": 2}
        with pytest.raises(ValueError) as e:
            Task.from_dict(d)
        assert "junction citation" in str(e.value)

    def test_the_junction_slot_is_required(self):
        with pytest.raises(ValueError) as e:
            _task(slots={})
        assert "junction" in str(e.value)

    def test_a_citation_without_run_N_is_refused(self):
        """Ruling Q1: the attempt is named explicitly, never picked."""
        with pytest.raises(ValueError) as e:
            _task(slots={"junction": "BDT-Au/optimization/JunctionRelax"})
        assert "@run-N" in str(e.value)

    def test_bias_must_start_from_equilibrium(self):
        with pytest.raises(ValueError) as e:
            _task(bias=(0.2, 0.4))
        assert "0.0" in str(e.value) and "TSDE" in str(e.value)

    def test_slots_and_bias_belong_to_transport_alone(self):
        from molbuilder.task import StructureRef
        base = dict(engine="siesta", shape="hierarchical",
                    run=Run(name="R", id=run_id("R", "H2")),
                    structure=StructureRef(source="x.xyz", formula="H2",
                                           atoms=2),
                    varies=None, stages=None)
        with pytest.raises(ValueError):
            Task(slots={"junction": _CITE}, **base)
        with pytest.raises(ValueError):
            Task(bias=(0.0,), **base)

    def test_the_identity_derives_from_the_citation(self):
        a = _task()
        b_cite = "Other/optimization/Relax@run-0"
        b = _task(slots={"junction": b_cite},
                  run=Run(name="T", id=run_id("T", b_cite,
                                              stage_names=_STAGES)))
        assert a.run.id != b.run.id, (
            "two transports of two junctions must never share an id")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    from molbuilder.projects import PROJECTS_ROOT_ENV
    root = tmp_path / "projects"
    (root / "BDT-Au" / "optimization" / "JunctionRelax" / "run-2"
     ).mkdir(parents=True)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
    return root


class TestInitCLI:

    def _invoke(self, args):
        from molbuilder.jobset._cli import jobset_group
        return CliRunner().invoke(jobset_group, ["init"] + args)

    def test_happy_path_writes_the_five_stage_task(self, tree):
        r = self._invoke([
            "--calculation", "transport", "--shape", "hierarchical",
            "--bundle", "BDT-Au/transport/BDTTrans",
            "--slot", f"junction={_CITE}", "--bias", "0.0,0.2"])
        assert r.exit_code == 0, r.output
        raw = json.loads((tree / "BDT-Au" / "transport" / "BDTTrans"
                          / "task.json").read_text())
        assert raw["calculation"] == "transport"
        assert raw["slots"] == {"junction": _CITE}
        assert [s["name"] for s in raw["stages"]] == list(_STAGES)
        assert "structure" not in raw
        # floor 2 is task.json ALONE
        written = sorted(p.name for p in
                         (tree / "BDT-Au" / "transport" / "BDTTrans"
                          ).iterdir())
        assert written == ["task.json"]

    def test_the_cited_calculation_must_exist_in_the_tree(self, tree):
        r = self._invoke([
            "--calculation", "transport", "--shape", "hierarchical",
            "--bundle", "BDT-Au/transport/T",
            "--slot", "junction=Nope/optimization/Gone@run-0"])
        assert r.exit_code != 0
        assert "Nope/optimization/Gone" in r.output

    def test_a_slot_without_the_attempt_is_refused(self, tree):
        r = self._invoke([
            "--calculation", "transport", "--shape", "hierarchical",
            "--bundle", "BDT-Au/transport/T",
            "--slot", "junction=BDT-Au/optimization/JunctionRelax"])
        assert r.exit_code != 0
        assert "@run-N" in r.output

    def test_options_answered_by_the_citation_are_refused(self, tree):
        (tree / "BDT-Au" / "structure").mkdir(parents=True, exist_ok=True)
        (tree / "BDT-Au" / "structure" / "j.xyz").write_text(
            "1\n\nH 0 0 0\n")
        for extra in (["--structure", "BDT-Au/structure/j.xyz"],
                      ["--vacuum", "8"],
                      ["--stage-strategy", "publishable"]):
            r = self._invoke([
                "--calculation", "transport", "--shape", "hierarchical",
                "--bundle", "BDT-Au/transport/T",
                "--slot", f"junction={_CITE}"] + extra)
            assert r.exit_code != 0, f"{extra} must be refused"
            assert "transport" in r.output

    def test_slot_on_a_non_transport_init_is_refused(self, tree):
        (tree / "BDT-Au" / "structure").mkdir(parents=True, exist_ok=True)
        (tree / "BDT-Au" / "structure" / "j.xyz").write_text(
            "1\n\nH 0 0 0\n")
        r = self._invoke([
            "--shape", "hierarchical",
            "--structure", "BDT-Au/structure/j.xyz",
            "--bundle", "BDT-Au/optimization/X",
            "--slot", f"junction={_CITE}"])
        assert r.exit_code != 0
        assert "transport" in r.output

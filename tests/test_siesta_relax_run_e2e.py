"""A vibration's relaxation, run with the real SIESTA, and what it hands on:
the relaxation record (`model/parse.md` § 5b.1), the two doors that answer
it, the stage that builds on it (`engines/vibration.md` § 5.2a), and the
vibration gate's verdicts on a structure that carries it (§ 2.2, the record
table).

``jobset init --calculation vibration`` (hierarchical) -> ``prep task --group 1``
-> ``launch task --stage relax --mode direct``, on the road's CO2 in a 10 Å
box, isolated, listed O O C with its carbon held: the `relax` stage of the
kind's own ladder, at the kind's own tight tolerance, its deck written in the
engine's held-first order (`engines/vibration.md` § 5.2) -- and what the
run's order means for every door that shows or writes it
(`model/overview.md` § 2.2a).

Every expectation is what the run was given -- the structure written here,
the deck prep wrote -- or the rule of the record applied to the run's own
output; never a number a reader printed.  The relaxation is made with the
engine in this pass, once for every module that reads it
(`support/real_runs.py`; user, 2026-10-06: "when a test need siesta's
output why is it not part of a e2e test?").
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from _road import (CO2_ELEMENTS, CO2_ENGINE_ELEMENTS, CO2_HELD,
                   CO2_HELD_FIRST, conda_hook, env_available)

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

@pytest.fixture(scope="module")
def relaxed(real_relaxed_for_vibration):
    """The calculation, its `relax` stage run (`support/real_runs.RealRun`)."""
    return real_relaxed_for_vibration


def _run(calc) -> Path:
    """The `relax` stage's run -- its folder, the handle the run door
    takes."""
    return calc.attempt("relax")


def _deck(calc) -> Path:
    """The deck the run read (`runs.Run.deck`)."""
    from molbuilder.runs import run_of
    return run_of(_run(calc)).deck


def _out(calc) -> Path:
    """The run's output the Results tab opens (`runs.openable`)."""
    from molbuilder.runs import openable
    return Path(openable(_run(calc))[0])


def _tolerance(calc) -> float:
    """The force tolerance the deck states -- what SIESTA was told, read by
    fdf's own rule (`parse.fdf.read_fdf`)."""
    from molbuilder.parse.fdf import fdf_key, read_fdf
    told = read_fdf(_deck(calc).read_text())[0][fdf_key("MD.MaxForceTol")]
    return float(told[0])


def _as_its_deck_states(attempt: Path):
    """The structure a run's deck hands its engine, in the engine's order,
    with what the run declared about it (`runs.view_of`: its labels, its
    order record) -- the species and coordinates through the deck's own
    readers (`parse.fdf.deck_atoms`, ``parse_fdf_params(...).coords_ang``)."""
    from molbuilder.parse.fdf import deck_atoms, parse_fdf_params
    from molbuilder.runs import run_of, view_of
    ran = run_of(attempt)
    text = ran.deck.read_text()
    return view_of(ran).structure(deck_atoms(text).labels,
                                  parse_fdf_params(text).coords_ang)


def _final_geometry(calc):
    """THE RECORD'S INDEPENDENT ORACLE: the coordinates the output printed
    last, under ``outcoor: Relaxed atomic coordinates (Ang)``, read off the
    text here rather than through the parser the record is built on
    (`parse.contract.relaxation_of`) -- the geometry the record is about, in
    the engine's order."""
    from molbuilder.structure import Structure
    lines = _out(calc).read_text(errors="replace").splitlines()
    at = max(i for i, ln in enumerate(lines)
             if "outcoor: Relaxed atomic coordinates" in ln)
    pos = []
    for ln in lines[at + 1:]:
        words = ln.split()
        if len(words) < 3:
            break
        pos.append([float(w) for w in words[:3]])
    return Structure(elements=list(CO2_ENGINE_ELEMENTS),
                     positions=np.array(pos))


def _moves(calc) -> int:
    """THE RECORD'S INDEPENDENT ORACLE of its step count: how many geometry
    moves the output began, counted off its text."""
    text = _out(calc).read_text(errors="replace")
    return len(re.findall(r"^\s*Begin \w+ opt\. move", text, re.M))


# --------------------------------------------------------------------- #
#  The record (`model/parse.md` § 5b.1)                                   #
# --------------------------------------------------------------------- #

def test_a_finished_relaxation_answers_its_record(relaxed):
    """The run's own tolerance; the last reported forces over every atom and
    over the moved one, judged as SIESTA judges them; the held set and its
    atoms' lines; the verdict; and a fingerprint of the FINAL geometry,
    whatever order the atoms are listed in."""
    from molbuilder.parse.contract import relaxation_of
    from molbuilder.structure import Structure
    rec = relaxation_of(_out(relaxed))
    assert rec is not None
    assert (rec["engine"], rec["source"]) == ("siesta", _out(relaxed).name)
    tol = _tolerance(relaxed)
    assert rec["force_tolerance_ev_ang"] == pytest.approx(tol)
    assert rec["n_steps"] == _moves(relaxed)
    deck = _as_its_deck_states(_run(relaxed))
    lines = deck.geometry_lines()
    assert rec["held_atom_keys"] == [lines[i] for i in deck.frozen_atoms]
    assert rec["converged"] is True and rec["run_state"] == "ended"
    assert rec["max_force_free_ev_ang"] <= tol
    assert rec["max_force_ev_ang"] >= rec["max_force_free_ev_ang"]
    final = _final_geometry(relaxed)
    assert rec["geometry_sha256"] == final.geometry_fingerprint()
    swapped = Structure(elements=final.elements[::-1],
                        positions=final.positions[::-1])
    assert swapped.geometry_fingerprint() == rec["geometry_sha256"]


def test_both_doors_answer_one_record_of_the_run(relaxed, monkeypatch):
    """The trajectory load, of the file it opened, and the structure
    inspector's door, of the file the Results tab opens in the structure's
    folder (`runs.openable`): one record, and the level of theory its own
    deck states (`model/parse.md` § 5b)."""
    from molbuilder.parse.contract import contract_of
    client = _client(relaxed, monkeypatch)
    run = _run(relaxed)
    loaded = client.post("/api/watch/load",
                         json={"path": str(run)}).get_json()
    assert loaded["ok"] is True, loaded
    asked = client.get("/api/results/contract",
                       query_string={"path": str(run / "CO2.XV")}).get_json()
    assert asked["ok"] is True, asked
    record = loaded["info"]["relaxation"]
    assert record["source"] == Path(loaded["path"]).name, record
    assert record["converged"] is True
    assert asked["relaxation"] == record
    assert (asked["calculation"] == loaded["info"]["calculation"]
            == contract_of(_deck(relaxed)))


def test_the_force_constants_build_on_the_relaxation(relaxed):
    """`freq` builds on its `relax` run -- the newest, which finished -- and
    is offered every run of `relax` and no start from the structure
    (`engines/vibration.md` § 5.2a's table, W38 F9)."""
    from molbuilder.jobset.continuation import (continuation_answer,
                                                continue_from_choices)
    from molbuilder.task import read_task
    task = read_task(relaxed.bundle / "task.json")
    got, refused = continuation_answer(relaxed.bundle, task, "freq")
    assert refused is None, refused
    assert (got.stage, got.source, got.by_default, got.linked) == (
        "relax", "01_relax/run-0", True, True), got
    assert "(the relaxation it builds on; concluded rc=0" in got.line()
    named, refused = continuation_answer(relaxed.bundle, task, "freq",
                                         from_attempt="01_relax/run-0")
    assert refused is None and not named.by_default and named.linked, named
    offered = continue_from_choices(relaxed.bundle, task, "freq")
    assert offered["from_stage"] == "relax" and offered["cold"] is False
    assert [r["source"] for r in offered["runs"]] == ["01_relax/run-0"]


# --------------------------------------------------------------------- #
#  The gate on a structure carrying the record (vibration.md § 2.2)      #
# --------------------------------------------------------------------- #

def _carrying_the_record(relaxed, *, shift_z: float = 0.0):
    """The geometry the relaxation left, in your order, with what the run
    says about it -- its labels, its box, its record -- through the run
    door's one read (`runs.steps_of`): what the Results tab's export writes
    (`model/overview.md` § 2.2a, I1); its second oxygen moved by
    ``shift_z``."""
    from molbuilder.runs import run_of, steps_of
    s = steps_of(run_of(_run(relaxed)), _out(relaxed)).last
    pos = np.asarray(s.positions, dtype=float).copy()
    pos[1, 2] += shift_z
    return s.replace(positions=pos)


def _record_findings(struct, cfg):
    """What the gate says on the box's card, beside the statement's own
    finding (the ticked warning, the unticked info of § 5.8)."""
    from molbuilder.validation import validate
    return [i for i in validate(struct, cfg, calculation="vibration")
            if i.where == "config.already_relaxed"
            and not i.message.startswith(("You stated",
                                          "The structure is not stated"))]


def _ev(x: float) -> str:
    """A force as the gate words it: four decimals down to half a
    milli-eV/Å, an exponent below that (`validation/sidecar._ev`)."""
    return f"{x:.4f}" if x >= 5e-4 else f"{x:.1e}"


def _cfg(**over):
    from molbuilder.config.siesta import SiestaConfig
    return SiestaConfig(system_label="co2", psml_lib="pseudopotential", **over)


def test_a_matching_record_answers_the_ticked_box_with_its_numbers(relaxed):
    """Ticked, the record for these coordinates, within this calculation's
    tolerance at the same level and held set: one info line with the
    engine, the run's tolerance and the largest force it left; and a record
    edited by hand says whose word its values are, first."""
    s = _carrying_the_record(relaxed)
    rec = s.info["relaxation"]
    tol = _tolerance(relaxed)
    found = _record_findings(s, _cfg(already_relaxed=True,
                                     relax_force_tol=tol))
    assert len(found) == 1 and found[0].severity == "info", found
    assert f"relaxed on siesta to {tol:g} eV/Å" in found[0].message
    assert (f"{_ev(rec['max_force_free_ev_ang'])} eV/Å, within this "
            f"calculation's tolerance of {tol:g}") in found[0].message
    s.info["relaxation"] = dict(rec, edited_by_hand="2026-10-03T12:00:00Z")
    edited = _record_findings(s, _cfg(already_relaxed=True,
                                      relax_force_tol=tol))
    assert [i.severity for i in edited] == ["info", "info"], edited
    assert edited[0].message.startswith(
        "This structure's relaxation record was edited by hand"), edited[0]
    assert edited[1].message == found[0].message


def test_a_record_looser_than_this_calculation_warns_when_ticked(relaxed):
    """The record's largest force above THIS calculation's tolerance: a
    warning when ticked, information when unticked (rows 3 and 5)."""
    s = _carrying_the_record(relaxed)
    left = s.info["relaxation"]["max_force_free_ev_ang"]
    assert left > 0
    tighter = 10 ** np.floor(np.log10(left)) / 2       # below what it left
    found = _record_findings(s, _cfg(already_relaxed=True,
                                     relax_force_tol=tighter))
    assert len(found) == 1 and found[0].severity == "warn", found
    assert f"above this calculation's tolerance of {tighter:g}" \
        in found[0].message
    found2 = _record_findings(s, _cfg(already_relaxed=False,
                                      relax_force_tol=tighter))
    assert len(found2) == 1 and found2[0].severity == "info", found2


def test_a_record_for_another_geometry_does_not_vouch(relaxed):
    """Another frame of the run, or an edit since: the fingerprint differs
    and the record is set aside with a warning (ticked)."""
    s = _carrying_the_record(relaxed, shift_z=0.01)
    found = _record_findings(s, _cfg(already_relaxed=True))
    assert len(found) == 1 and found[0].severity == "warn", found
    assert "different geometry" in found[0].message


def test_a_record_at_another_level_of_theory_warns(relaxed):
    """The recorded contract against a form that runs another basis: a
    different level of theory, named field by field -- and another engine is
    the same fact at the engine level."""
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.parse.contract import contract_of
    s = _carrying_the_record(relaxed)
    ran = contract_of(_deck(relaxed))["contract"]["basis_size"]
    other = "SZ" if ran != "SZ" else "DZP"
    found = [i for i in _record_findings(
        s, _cfg(already_relaxed=True, basis_size=other))
        if i.severity == "warn"]
    assert len(found) == 1, found
    assert f"basis_size: relaxed with {ran}, this run {other}" \
        in found[0].message
    pfound = _record_findings(s, PySCFConfig(job_name="co2",
                                             already_relaxed=True))
    assert any(i.severity == "warn"
               and "relaxed on siesta; this is a pyscf calculation"
               in i.message for i in pfound), pfound


def test_a_record_at_another_charge_warns(relaxed):
    """ES7 (`science/chemistry-correctness.md` § 2a): the electronic state is
    part of the level of theory the record is compared against, resolved --
    CO2 at -2 keeps the electron count even, so the parity rule does not
    answer first."""
    s = _carrying_the_record(relaxed)
    found = [i for i in _record_findings(
        s, _cfg(already_relaxed=True, net_charge=-2)) if i.severity == "warn"]
    assert any("net_charge: relaxed with 0, this run -2" in i.message
               for i in found), found


def test_a_good_record_under_an_unticked_box_offers_the_skip(relaxed):
    """Unticked, with a record that already meets the criterion: the info
    line offers the skip (row 4)."""
    s = _carrying_the_record(relaxed)
    found = _record_findings(s, _cfg(already_relaxed=False,
                                     relax_force_tol=_tolerance(relaxed)))
    assert len(found) == 1 and found[0].severity == "info", found
    assert "the box may be ticked and the relaxation skipped" \
        in found[0].message


def test_no_record_under_a_ticked_box_is_accepted_with_a_hint(relaxed):
    """Ticked with no record: accepted, with the hint that the statement
    stands on its own (row 1)."""
    s = _carrying_the_record(relaxed)
    s.apply_info_dict(None)
    found = _record_findings(s, _cfg(already_relaxed=True))
    assert len(found) == 1 and found[0].severity == "info", found
    assert "No relaxation record travels with this structure" \
        in found[0].message


# --------------------------------------------------------------------- #
#  The order the run's atoms are in (`model/overview.md` § 2.2a)         #
# --------------------------------------------------------------------- #
#
# The run's deck is written in the engine's held-first order: the held
# carbon, then the oxygens as listed (`engines/vibration.md` § 5.2).  The
# silent failure every row guards: a run in its engine's order shown, saved or
# converted as if it were in yours -- the carbon taken for the first oxygen --
# with nothing to say so.

#: The viewer's mark of a structure in this order (`web/molview.md` § 6.2).
_SAID = "in the engine's order (held atoms first)"


def _says_it_was_restored(message: str, name: str) -> None:
    """What a door that put the structure back in your order says of it
    (`model/overview.md` § 2.2a, I8): the file, and the order it was in, in
    its words."""
    assert name in message and "held atoms first" in message, message

_ORDER_ROWS = [
    dict(name="the deck is written in the engine's order and states it; "
              "the run door reads it", door="deck"),
    dict(name="the Results tab's load answers the run in the engine's "
              "order, its labels and order record on it, and the viewer "
              "marks it", door="watch_load"),
    dict(name="SIESTA's own CO2.xyz opens through its run's view, in the "
              "engine's order, marked", door="build_load"),
    dict(name="...and in your order when an editor opens it, saying so",
         door="build_load", body={"restore": True}),
    dict(name="a bare xv2xyz of the run's .XV is refused, naming "
              "--from-run", door="xv2xyz"),
    dict(name="xv2xyz --from-run writes the pair in your order, and says so",
         door="xv2xyz", from_run=True),
    dict(name="a bare file reader reads the run's output as the engine wrote "
              "it, and says so", door="watch_parse"),
    dict(name="SIESTA's own CO2.xyz is not a pair a transport calculation "
              "cites: refused by name -- cite the run", door="cite"),
]


def _client(relaxed, monkeypatch):
    """The app, its file picker rooted at the run's projects tree."""
    from molbuilder import diagnostics
    from molbuilder.web.app import create_app
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((relaxed.tree.resolve(), "projects"),))
    return create_app(config={}).test_client()


def _in_your_order(struct) -> None:
    """The road's CO2 as it was written: O O C, the carbon held, no order
    record."""
    assert (struct.elements, struct.frozen_atoms, struct.atom_order) == (
        CO2_ELEMENTS, CO2_HELD, None), struct


@pytest.mark.parametrize("row", _ORDER_ROWS,
                         ids=[r["name"] for r in _ORDER_ROWS])
def test_the_run_is_in_the_engines_order_and_says_so(row, relaxed,
                                                     monkeypatch, tmp_path):
    from molbuilder.structure import Structure
    run = _run(relaxed)
    if row["door"] == "deck":
        from molbuilder.runs import declared, run_of
        assert _as_its_deck_states(run).elements == CO2_ENGINE_ELEMENTS
        assert declared(run_of(run)).atom_order.to_dict() == CO2_HELD_FIRST
        return
    if row["door"] == "cite":
        from molbuilder.transport.compose import ComposeError, resolve_citation
        with pytest.raises(ComposeError) as refused:
            resolve_citation(str((run / "CO2.xyz").relative_to(relaxed.tree)),
                             relaxed.tree)
        assert "it has no .molstruct.json beside it" in str(refused.value)
        assert "cite that run" in str(refused.value)
        return
    if row["door"] == "watch_parse":
        from click.testing import CliRunner
        from molbuilder import cli
        said = CliRunner().invoke(cli.cli, ["watch", "parse", str(_out(relaxed)),
                                            "--frames-only"])
        assert said.exit_code == 0, said.output
        assert f"note: {_out(relaxed).name} is {_SAID}" in said.stderr, (
            said.stderr)
        return
    if row["door"] == "xv2xyz":
        from click.testing import CliRunner
        from molbuilder import cli
        from molbuilder.workingcopy_structure import StructureCodec
        out = tmp_path / "co2.xyz"
        said = CliRunner().invoke(cli.cli, [
            "xv2xyz", str(run / "CO2.XV"), str(out)]
            + (["--from-run"] if row.get("from_run") else []))
        if not row.get("from_run"):
            assert said.exit_code != 0, said.output
            assert "--from-run" in said.output and _SAID in said.output
            assert not out.exists()
            return
        assert said.exit_code == 0, said.output
        _says_it_was_restored(said.output, "CO2.XV")
        _in_your_order(StructureCodec().load(out))
        return
    client = _client(relaxed, monkeypatch)
    if row["door"] == "watch_load":
        loaded = client.post("/api/watch/load",
                             json={"path": str(run)}).get_json()
        assert loaded["ok"] is True, loaded
        env = loaded["structure"]
        assert env["metadata"]["atom_order"] == CO2_HELD_FIRST
        assert env["elements"] == CO2_ENGINE_ELEMENTS
        assert env["metadata"]["regions"]["frozen_atoms"] == [0]
        # ...and installed in the viewer, as its frame 0 is: marked.
        shown = client.post("/api/build/load",
                            json={"structure": env}).get_json()
        assert shown["ok"] is True and shown["atom_order_said"] == _SAID
        return
    answer = client.post("/api/build/load", json={
        "path": str(run / "CO2.xyz"), **row.get("body", {})}).get_json()
    assert answer["ok"] is True, answer
    got = Structure.from_dict(answer["structure"])
    restored = [n for n in answer.get("notices", [])
                if n["where"] == "structure.restored"]
    if row.get("body", {}).get("restore"):
        _in_your_order(got)
        assert [n["severity"] for n in restored] == ["info"], restored
        _says_it_was_restored(restored[0]["message"], "CO2.xyz")
        assert answer["atom_order_said"] is None
        return
    assert got.atom_order.to_dict() == CO2_HELD_FIRST and restored == []
    assert (got.elements, got.frozen_atoms) == (CO2_ENGINE_ELEMENTS, [0])
    assert answer["atom_order_said"] == _SAID


# --------------------------------------------------------------------- #
#  A deck written before the order was recorded (I9)                     #
# --------------------------------------------------------------------- #
#
# The same run, its deck as molbuilder 2.9.6 wrote it -- that version's
# stamp, and its ATOM-METADATA block at schema 13 with no order record -- as
# the pin table rewrites a sidecar as an older molbuilder wrote it.  The
# silent failure: a run of a kind that reorders, made before 2.10.0, shown or
# converted as if its atoms were in your order.  Every door that builds
# refuses it by name; every door that shows it says so -- once.


def _says_unstated(text: str) -> None:
    """The run door's one sentence of such a run, said once: who prepared
    it, the release that began recording the order, and the way out
    (`support/real_runs.PREPARED_BY_2_9_6`)."""
    from support.real_runs import PREPARED_BY_2_9_6
    assert all(part in text for part in PREPARED_BY_2_9_6), text
    assert text.count(PREPARED_BY_2_9_6[1]) == 1, text

_PRE_RULE_ROWS = [
    dict(name="xv2xyz --from-run refuses it by name", door="xv2xyz",
         from_run=True),
    dict(name="a bare xv2xyz refuses it too", door="xv2xyz"),
    dict(name="the Results tab's load says so", door="watch_load"),
    dict(name="SIESTA's own CO2.xyz opens saying so", door="build_load"),
    dict(name="a bare file reader says so", door="watch_parse"),
    dict(name="jobset init refuses to build a calculation from its CO2.xyz",
         door="init"),
    dict(name="an editor's open refuses it -- an editor holds only your order",
         door="build_load", body={"restore": True}),
    dict(name="molbuilder modify refuses it", door="modify"),
]


@pytest.fixture(scope="module")
def before_the_rule(relaxed):
    """A copy of the run's calculation beside it, every deck of its `relax`
    stage as 2.9.6 wrote it (`support/real_runs.as_prepared_by_2_9_6`)."""
    from support.real_runs import as_prepared_by_2_9_6
    return as_prepared_by_2_9_6(relaxed)


@pytest.mark.parametrize("row", _PRE_RULE_ROWS,
                         ids=[r["name"] for r in _PRE_RULE_ROWS])
def test_a_deck_from_before_the_rule_is_refused_or_said(row, before_the_rule,
                                                        monkeypatch, tmp_path):
    run = _run(before_the_rule)
    if row["door"] in ("init", "modify"):
        from click.testing import CliRunner
        from molbuilder import cli
        out = tmp_path / "co2.xyz"
        again = before_the_rule.tree / "P" / "init-refused"
        if row["door"] == "init":
            from conftest import _point_the_one_door_at
            from support.road import jobset
            _point_the_one_door_at(before_the_rule.tree.parent,
                                   monkeypatch.setenv)
            said = jobset("init", "--structure", str(run / "CO2.xyz"),
                          "--bundle", str(again), "--shape", "hierarchical",
                          "--engine", "siesta", "--calculation", "optimization",
                          "--name", "CO2", "--psml-lib", "pseudopotential")
        else:
            said = CliRunner().invoke(cli.cli, [
                "modify", str(run / "CO2.xyz"), str(out), "--rotate", "z:90"])
        assert said.exit_code != 0, said.output
        _says_unstated(said.output)
        assert not (again / "task.json").exists() if row["door"] == "init" \
            else not out.exists()
        return
    if row["door"] in ("xv2xyz", "watch_parse"):
        from click.testing import CliRunner
        from molbuilder import cli
        out = tmp_path / "co2.xyz"
        words = (["watch", "parse", str(_out(before_the_rule)), "--frames-only"]
                 if row["door"] == "watch_parse" else
                 ["xv2xyz", str(run / "CO2.XV"), str(out)]
                 + (["--from-run"] if row.get("from_run") else []))
        said = CliRunner().invoke(cli.cli, words)
        if row["door"] == "watch_parse":
            assert said.exit_code == 0, said.output
            _says_unstated(said.stderr)
            return
        assert said.exit_code != 0, said.output
        _says_unstated(said.output)
        assert not out.exists(), said.output
        return
    client = _client(before_the_rule, monkeypatch)
    answer = (client.post("/api/watch/load", json={"path": str(run)})
              if row["door"] == "watch_load" else
              client.post("/api/build/load", json={
                  "path": str(run / "CO2.xyz"),
                  **row.get("body", {})})).get_json()
    if row.get("body", {}).get("restore"):
        assert answer["ok"] is False, answer
        _says_unstated(answer["error"])
        return
    assert answer["ok"] is True, answer
    said = [n for n in answer.get("notices", [])
            if n["where"] == "structure.order_unstated"]
    assert [n["severity"] for n in said] == ["warn"], answer.get("notices")
    _says_unstated(said[0]["message"])


# --------------------------------------------------------------------- #
#  An order record that cannot be read (`job-contracts.md` § 3.1)        #
# --------------------------------------------------------------------- #
#
# The same run, its deck's order record broken by hand -- an atom named
# twice.  The silent failure: read as a deck with no order, the run shown in
# your order though it is not.  Every surface refuses it, naming the deck
# and the way out.

_BROKEN_ROWS = [
    dict(name="the Results tab's load refuses it with its sentence",
         door="watch_load"),
    dict(name="xv2xyz --from-run refuses it", door="xv2xyz"),
    dict(name="molbuilder validate refuses SIESTA's own CO2.xyz with it",
         door="validate"),
    dict(name="molbuilder modify refuses it too", door="modify"),
]


@pytest.fixture(scope="module")
def broken_order(relaxed):
    """A copy of the run's calculation (`support/real_runs.copy_of`), its
    `relax` decks' order record naming one atom twice."""
    from support.real_runs import copy_of, relax_decks, with_atom_metadata

    def name_one_atom_twice(block):
        assert block["atom_order"] == CO2_HELD_FIRST, block
        block["atom_order"] = dict(CO2_HELD_FIRST,
                                   sorted_to_original=[2, 0, 0])

    def break_it(copy):
        for deck in relax_decks(copy):
            with_atom_metadata(deck, name_one_atom_twice)
    return copy_of(relaxed, "broken-order", break_it)


@pytest.mark.parametrize("row", _BROKEN_ROWS,
                         ids=[r["name"] for r in _BROKEN_ROWS])
def test_an_order_record_that_cannot_be_read_is_refused(row, broken_order,
                                                       monkeypatch, tmp_path):
    run = _run(broken_order)
    words = "holds an order record that cannot be read"

    def names_the_deck_and_the_way_out(text):
        assert words in text and _deck(broken_order).name in text, text
        assert "checkpoint restore" in text, text
    if row["door"] in ("xv2xyz", "validate", "modify"):
        from click.testing import CliRunner
        from molbuilder import cli
        said = CliRunner().invoke(cli.cli, {
            "xv2xyz": ["xv2xyz", str(run / "CO2.XV"),
                       str(tmp_path / "co2.xyz"), "--from-run"],
            "validate": ["validate", str(run / "CO2.xyz")],
            "modify": ["modify", str(run / "CO2.xyz"),
                       str(tmp_path / "co2.xyz"), "--rotate", "z:90"],
        }[row["door"]])
        assert said.exit_code != 0, said.output
        names_the_deck_and_the_way_out(said.output)
        assert said.exception is None or isinstance(said.exception,
                                                    SystemExit), said.exception
        return
    answer = _client(broken_order, monkeypatch).post(
        "/api/watch/load", json={"path": str(run)})
    assert answer.status_code == 400, answer.get_data(as_text=True)[:300]
    names_the_deck_and_the_way_out(answer.get_json()["error"])


# --------------------------------------------------------------------- #
#  The force-constant stage builds on the relaxation (plan V1.41)       #
# --------------------------------------------------------------------- #
#
# `engines/vibration.md` § 2.3a: within one ladder the held set may grow.
# The same calculation, its structure's held set grown after `relax` ran --
# the carbon and the first oxygen held, where `relax` held the carbon alone
# -- and `freq` prepared: its deck is written in ITS OWN held-first order
# (O C O, the order record [0, 2, 1]), and its coordinates are `relax`'s
# relaxed ones, atom for atom, through both decks' order records.  The
# silent failure: `relax`'s coordinates copied row by row into another
# order, each atom given another atom's place.

#: `freq`'s order record: the rule applied to the grown held set
#: (`support/real_runs.HELD_GROWN`, the first oxygen and the carbon) -- those
#: two first in their own order, then the free oxygen: O C O.
_FREQ_ORDER = {"key": "held-first", "sorted_to_original": [0, 2, 1],
               "original_to_sorted": [0, 2, 1]}


@pytest.fixture(scope="module")
def held_grown(relaxed, tmp_path_factory):
    """The calculation, its held set grown and its `freq` stage prepared on
    the road (`support/real_runs.held_grown`)."""
    from support.real_runs import held_grown as grown
    return grown(relaxed, tmp_path_factory)


def test_the_force_constant_deck_takes_the_relaxed_geometry_in_its_own_order(
        relaxed, held_grown):
    from molbuilder.runs import declared, run_of, view_of
    freq = _as_its_deck_states(held_grown.attempt("freq"))
    assert freq.atom_order.to_dict() == _FREQ_ORDER
    assert freq.elements == ["O", "C", "O"]
    relax = run_of(_run(relaxed))
    assert declared(relax).atom_order.to_dict() == CO2_HELD_FIRST
    mine = freq.restored()
    # WHERE THE RELAXATION ENDED, read off its output's text (the record's
    # oracle) and put in your order through `relax`'s own order record.
    theirs = view_of(relax).onto(_final_geometry(relaxed)).restored()
    assert mine.elements == theirs.elements == CO2_ELEMENTS
    assert np.allclose(mine.positions, theirs.positions, atol=1e-5), (
        mine.positions, theirs.positions)

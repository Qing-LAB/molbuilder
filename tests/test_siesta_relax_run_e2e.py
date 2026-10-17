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

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pytest

from _road import conda_hook, env_available
from support.road import CO2, CO2_ENGINE_ELEMENTS, CO2_HELD_FIRST

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


def test_the_run_record_states_the_forces_the_relaxation_judged(
        relaxed, monkeypatch):
    """The run record's verdict states the last step's largest force on the
    moved atoms and -- the held carbon's being the larger -- on the held
    atoms (`model/parse.md` § 5d.6): the two the relaxation record judged
    from that step's per-atom forces (§ 5b.1); and the stress, SIESTA's six
    numbers.  The silent failure: the held atoms' force read and dropped,
    a run whose held atoms carry the larger force showing the moved one's
    alone."""
    from molbuilder.parse.contract import relaxation_of
    from molbuilder.runs import folder_answer
    _at_the_runs_tree(relaxed, monkeypatch)
    rec = relaxation_of(_out(relaxed))
    assert rec["max_force_ev_ang"] > rec["max_force_free_ev_ang"], (
        "the held carbon carries no larger force: the held part goes "
        "unasked", rec)
    verdict = folder_answer(_run(relaxed))["record"]["verdict"]
    assert verdict["max_force_free_ev_ang"] == pytest.approx(
        rec["max_force_free_ev_ang"]), verdict
    assert verdict["max_force_held_ev_ang"] == pytest.approx(
        rec["max_force_ev_ang"]), verdict
    assert len(verdict["stress_voigt_kbar"]) == 6, verdict


def test_both_doors_answer_one_record_of_the_run(relaxed, web_client,
                                                 monkeypatch):
    """The trajectory load, of the file it opened in the run's folder
    (`runs.openable`), and the structure inspector's door, of the run's
    `.XV` a person opens there: one record, and the level of theory its own
    deck states (`model/parse.md` § 5b)."""
    from molbuilder.parse.contract import contract_of
    _at_the_runs_tree(relaxed, monkeypatch)
    run = _run(relaxed)
    client = web_client
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
    # THE K-POINT GRID, as this run's deck will write it (`model/parse.md`
    # § 5b): the silent failure, the 2026-10-10 walk's slab relaxed at one
    # k-point and its vibration read at its own grid unsaid.
    grid = contract_of(_deck(relaxed))["contract"]["kgrid"]
    more = [n + 1 for n in grid]
    found = [i for i in _record_findings(
        s, _cfg(already_relaxed=True, kgrid=more)) if i.severity == "warn"]
    assert len(found) == 1, found
    assert f"kgrid: relaxed with {grid}, this run {more}" in found[0].message
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
# carbon, then the oxygens as listed (`engines/vibration.md` § 5.2).  Every
# door that shows, writes or builds from the run is asked of it -- the silent
# failure: a run in its engine's order shown, saved or converted as if it
# were in yours, the carbon taken for the first oxygen, with nothing to say
# so.

#: The viewer's mark of a structure in this order (`web/molview.md` § 6.2).
_SAID = "in the engine's order (held atoms first)"

#: The notices a row reads, of all a route says (`model/overview.md`
#: § 2.2a, I8, I9).
_ORDER_NOTICES = ("structure.restored", "structure.order_unstated")


def _says_restored(got, row) -> None:
    """A door that put the structure back in your order says so (§ 2.2a,
    I8): the file, and the order it was in, in its words."""
    assert got.named in got.text and "held atoms first" in got.text, (
        got.text)


def _says_engine_order(got, row) -> None:
    """A bare reader, or a refused bare `xv2xyz`, says the file is in its
    engine's order (`model/parse.md` § 5.3), in the viewer's words; the
    refusal names its way out."""
    assert got.named in got.text and _SAID in got.text, got.text
    if not got.ok:
        assert "--from-run" in got.text, got.text


def _says_cite_the_run(got, row) -> None:
    """SIESTA's own structure file is not a pair: it has no .molstruct.json,
    and the run is what is cited (`engines/transport.md` § 3.1)."""
    assert ".molstruct.json" in got.text and "cite that run" in got.text, (
        got.text)


_SAYS = {"restored": _says_restored, "engine order": _says_engine_order,
         "cite the run": _says_cite_the_run}


def _in_the_engines_order(struct) -> None:
    """The run as the engine holds it: C O O, the carbon held first, its
    order record on it."""
    assert (struct.elements, struct.frozen_atoms) == (CO2_ENGINE_ELEMENTS,
                                                      [0]), struct
    assert struct.atom_order.to_dict() == CO2_HELD_FIRST


def _in_your_order(struct) -> None:
    """The road's CO2 as it was written: O O C, the carbon held, no order
    record -- and each atom in its own place, the first oxygen above the
    carbon and the second below it, as `support.road.CO2` put them."""
    assert (struct.elements, struct.frozen_atoms, struct.atom_order) == (
        CO2["elements"], CO2["regions"]["frozen_atoms"], None), struct
    z = np.asarray(struct.positions, dtype=float)[:, 2]
    assert z[0] > z[2] > z[1], z


_STRUCTURES = {"engine": _in_the_engines_order, "yours": _in_your_order}

#: Each row: the door (:func:`_ask`) asked of the run and what the door is
#: given (``file``, a file of the run a person names; ``from_run``;
#: ``restore``, an editor's open),
#: and what it answers -- ``ok``, and where stated: ``says``, the contract's
#: check of what it said (:data:`_SAYS`); ``notices``, the route's notices
#: of :data:`_ORDER_NOTICES`, by name and level; ``structure``, the order it
#: answered or wrote (:data:`_STRUCTURES`); ``marked``, the viewer's mark;
#: ``wrote``, whether a file was written.
_ROWS = [
    dict(name="the deck is written in the engine's order and states it; "
              "the run door reads it", door="deck", ok=True,
         structure="engine"),
    dict(name="the Results tab's load answers the run in the engine's "
              "order, its labels and order record on it, and the viewer "
              "marks it", door="watch_load", ok=True,
         notices=[], structure="engine", marked=True),
    dict(name="SIESTA's own CO2.xyz opens through its run's view, in the "
              "engine's order, marked", door="build_load",
         file="CO2.xyz", ok=True, notices=[], structure="engine",
         marked=True),
    dict(name="...and in your order when an editor opens it, saying so",
         door="build_load", file="CO2.xyz", restore=True,
         ok=True, notices=[("structure.restored", "info")], says="restored",
         structure="yours", marked=False),
    dict(name="a bare xv2xyz of the run's .XV is refused, naming "
              "--from-run", door="xv2xyz", file="CO2.XV",
         ok=False, says="engine order", wrote=False),
    dict(name="xv2xyz --from-run writes the pair in your order, and says so",
         door="xv2xyz", file="CO2.XV", from_run=True,
         ok=True, says="restored", structure="yours", wrote=True),
    dict(name="a bare file reader reads the run's output as the engine wrote "
              "it, and says so", door="watch_parse",
         ok=True, says="engine order"),
    dict(name="SIESTA's own CO2.xyz is not a pair a transport calculation "
              "cites: refused by name -- cite the run",
         door="cite", file="CO2.xyz", ok=False, says="cite the run"),
]


@dataclass
class _Answer:
    """What a door answered, in one shape: the name of the file it was given,
    whether it went through, what it said (its output or error, its notices'
    sentences), its notices by name and level, the structure it answered or
    wrote, the viewer's mark of it, and whether it wrote a file."""
    named: str
    ok: bool
    text: str = ""
    notices: Tuple[Tuple[str, str], ...] = ()
    structure: Any = None
    mark: Optional[str] = None
    wrote: Optional[bool] = None


def _at_the_runs_tree(calc, monkeypatch) -> None:
    """The one door pointed at the run's projects tree -- the tree every
    route and command of the test reads (`conftest._point_the_one_door_at`)."""
    from conftest import _point_the_one_door_at
    _point_the_one_door_at(calc.tree.parent, monkeypatch.setenv)


def _ask(row, calc, client, tmp_path, monkeypatch) -> _Answer:
    """THE ONE RUNNER: the row's door asked of ``calc``, as a person asks it
    -- the run door, the CLI, a route, the citation."""
    from click.testing import CliRunner

    from molbuilder import cli
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    run, door = _run(calc), row["door"]
    named = (run / row["file"] if "file" in row
             else _out(calc) if door == "watch_parse" else run)
    out = tmp_path / "co2.xyz"
    answered = functools.partial(_Answer, named.name)
    if door == "deck":
        return answered(ok=True, structure=_as_its_deck_states(run))
    if door == "cite":
        from molbuilder.transport.compose import ComposeError, compose_junction
        try:
            compose_junction(str(named.relative_to(calc.tree)),
                             tree_root=calc.tree)
        except ComposeError as exc:
            return answered(ok=False, text=str(exc))
        return answered(ok=True)
    _at_the_runs_tree(calc, monkeypatch)
    if door in ("xv2xyz", "watch_parse"):
        words = {"xv2xyz": ["xv2xyz", named, out]
                 + (["--from-run"] if row.get("from_run") else []),
                 "watch_parse": ["watch", "parse", named,
                                 "--frames-only"]}[door]
        said = CliRunner().invoke(cli.cli, [str(w) for w in words])
        assert said.exception is None or isinstance(
            said.exception, SystemExit), said.exception
        wrote = out.exists()
        return answered(
            ok=said.exit_code == 0,
            text=said.stderr if door == "watch_parse" else said.output,
            structure=StructureCodec().load(out) if wrote else None,
            wrote=wrote)
    answer = (client.post("/api/watch/load", json={"path": str(run)})
              if door == "watch_load" else
              client.post("/api/build/load", json={
                  "path": str(named),
                  **({"restore": True} if row.get("restore") else {})})
              ).get_json()
    if not answer.get("ok"):
        return answered(ok=False, text=answer.get("error", ""))
    said = [n for n in answer.get("notices", [])
            if n["where"] in _ORDER_NOTICES]
    env = answer["structure"]
    # THE VIEWER'S MARK: the build load's answer says it; the Results tab
    # installs its frame 0 through the same door, which says it then.
    mark = (answer if door == "build_load" else client.post(
        "/api/build/load", json={"structure": env}).get_json()
            )["atom_order_said"]
    return answered(ok=True, text="  ".join(n["message"] for n in said),
                    notices=tuple((n["where"], n["severity"]) for n in said),
                    structure=Structure.from_dict(env), mark=mark)


@pytest.mark.parametrize("row", _ROWS, ids=[r["name"] for r in _ROWS])
def test_every_door_asked_of_the_run_says_its_order(
        row, relaxed, web_client, tmp_path, monkeypatch):
    got = _ask(row, relaxed, web_client, tmp_path, monkeypatch)
    assert got.ok is row["ok"], (row["name"], got)
    if "says" in row:
        _SAYS[row["says"]](got, row)
    if "notices" in row:
        assert list(got.notices) == row["notices"], got
    if "structure" in row:
        _STRUCTURES[row["structure"]](got.structure)
    if "marked" in row:
        assert got.mark == (_SAID if row["marked"] else None), got.mark
    if "wrote" in row:
        assert got.wrote is row["wrote"], got

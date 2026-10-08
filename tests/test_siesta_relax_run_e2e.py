"""A vibration's relaxation, run with the real SIESTA, and what it hands on:
the relaxation record (`model/parse.md` § 5b.1), the two doors that answer
it, the stage that builds on it (`engines/vibration.md` § 5.2a), and the
vibration gate's verdicts on a structure that carries it (§ 2.2, the record
table).

``jobset init --calculation vibration`` (hierarchical) -> ``prep task --stage relax``
-> ``launch task --stage relax --mode direct``, on an H2 in a 10 Å box, isolated, its
first atom held: the `relax` stage of the kind's own ladder, at the kind's
own tight tolerance.

Every expectation is what the run was given -- the structure written here,
the deck prep wrote -- or the rule of the record applied to the run's own
output; never a number a reader printed.  The relaxation is made with the
engine here (user, 2026-10-06:
"when a test need siesta's output why is it not part of a e2e test?").
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from _road import (conda_hook, env_available,
                   h2_relaxed_for_vibration, live_siesta)

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

_BOX = 10.0


@pytest.fixture(scope="module")
def relaxed(isolated_projects_root_module, tmp_path_factory):
    """The calculation, its `relax` stage run: the bundle."""
    tree = isolated_projects_root_module
    with live_siesta(tree, tmp_path_factory):
        yield h2_relaxed_for_vibration(tree)


def _run(bundle: Path) -> Path:
    return bundle / "01_relax" / "run-0"


def _out(bundle: Path) -> Path:
    return _run(bundle) / "H2_01_relax-run0.out"


def _deck(bundle: Path) -> Path:
    return _run(bundle) / "H2_01_relax.fdf"


def _tolerance(bundle: Path) -> float:
    """The force tolerance the deck states -- what SIESTA was told."""
    for ln in _deck(bundle).read_text().splitlines():
        words = ln.split()
        if words and words[0] == "MD.MaxForceTol":
            return float(words[1])
    raise AssertionError("the relax deck states no MD.MaxForceTol")


def _deck_geometry(bundle: Path):
    """The structure as the deck hands it to SIESTA -- its own coordinate
    block, read by fdf's label rule."""
    from molbuilder.parse.fdf import _norm, _parse_fdf
    from molbuilder.structure import Structure
    rows = _parse_fdf(_deck(bundle).read_text())[1][
        _norm("AtomicCoordinatesAndAtomicSpecies")]
    return Structure(elements=["H", "H"],
                     positions=np.array([[float(x) for x in r[:3]]
                                         for r in rows]))


def _final_geometry(bundle: Path):
    """The coordinates the output printed last, under ``outcoor: Relaxed
    atomic coordinates (Ang)`` -- the geometry the record is about."""
    from molbuilder.structure import Structure
    lines = _out(bundle).read_text(errors="replace").splitlines()
    at = max(i for i, ln in enumerate(lines)
             if "outcoor: Relaxed atomic coordinates" in ln)
    pos = []
    for ln in lines[at + 1:]:
        words = ln.split()
        if len(words) < 3:
            break
        pos.append([float(w) for w in words[:3]])
    return Structure(elements=["H", "H"], positions=np.array(pos))


def _moves(bundle: Path) -> int:
    """How many geometry moves the output began."""
    text = _out(bundle).read_text(errors="replace")
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
    assert rec["held_atom_idxs"] == [0]
    held = _deck_geometry(relaxed).geometry_lines()[0]
    assert rec["held_atom_keys"] == [held]
    assert rec["converged"] is True and rec["run_state"] == "ended"
    assert rec["max_force_free_ev_ang"] <= tol
    assert rec["max_force_ev_ang"] >= rec["max_force_free_ev_ang"]
    final = _final_geometry(relaxed)
    assert rec["geometry_sha256"] == final.geometry_fingerprint()
    swapped = Structure(elements=["H", "H"], positions=final.positions[::-1])
    assert swapped.geometry_fingerprint() == rec["geometry_sha256"]


def test_both_doors_answer_one_record_of_the_run(relaxed, monkeypatch):
    """The trajectory load, of the file it opened, and the structure
    inspector's door, of the file the Results tab opens in the structure's
    folder (`runs.openable`): one record, and the level of theory its own
    deck states (`model/parse.md` § 5b)."""
    from molbuilder import diagnostics
    from molbuilder.parse.contract import contract_of
    from molbuilder.web.app import create_app
    root = next(p for p in relaxed.parents if p.name == "projects")
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((root.resolve(), "projects"),))
    client = create_app(config={}).test_client()
    run = _run(relaxed)
    loaded = client.post("/api/watch/load",
                         json={"path": str(run)}).get_json()
    assert loaded["ok"] is True, loaded
    asked = client.get("/api/results/contract",
                       query_string={"path": str(run / "H2.XV")}).get_json()
    assert asked["ok"] is True, asked
    record = loaded["info"]["relaxation"]
    assert record["source"] == Path(loaded["path"]).name, record
    assert record["converged"] is True and record["held_atom_idxs"] == [0]
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
    task = read_task(relaxed / "task.json")
    got, refused = continuation_answer(relaxed, task, "freq")
    assert refused is None, refused
    assert (got.stage, got.source, got.by_default, got.linked) == (
        "relax", "01_relax/run-0", True, True), got
    assert "(the relaxation it builds on; concluded rc=0" in got.line()
    named, refused = continuation_answer(relaxed, task, "freq",
                                         from_attempt="01_relax/run-0")
    assert refused is None and not named.by_default and named.linked, named
    offered = continue_from_choices(relaxed, task, "freq")
    assert offered["from_stage"] == "relax" and offered["cold"] is False
    assert [r["source"] for r in offered["runs"]] == ["01_relax/run-0"]


# --------------------------------------------------------------------- #
#  The gate on a structure carrying the record (vibration.md § 2.2)      #
# --------------------------------------------------------------------- #

def _carrying_the_record(relaxed, *, shift_z: float = 0.0):
    """The geometry the relaxation left, held atom first as the run had it,
    with the record the run directory answers for itself -- the pair the
    Results tab exports, through the composer it uses."""
    from molbuilder.parse.dirs.run_info import run_info
    from molbuilder.runs import declared, openable, run_of
    from molbuilder.structure import Structure
    run = _run(relaxed)
    pos = _final_geometry(relaxed).positions.copy()
    pos[1, 2] += shift_z
    s = Structure(elements=["H", "H"], positions=pos,
                  regions={"frozen_atoms": [0]},
                  cell=np.diag([_BOX] * 3), axis_kind=("isolated",) * 3)
    s.apply_info_dict(run_info(deck=declared(run_of(run)).deck,
                               output=openable(run)[0]))
    return s


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
    return SiestaConfig(system_label="h2", psml_lib="pseudopotential", **over)


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
    pfound = _record_findings(s, PySCFConfig(job_name="h2",
                                             already_relaxed=True))
    assert any(i.severity == "warn"
               and "relaxed on siesta; this is a pyscf calculation"
               in i.message for i in pfound), pfound


def test_a_record_at_another_charge_warns(relaxed):
    """ES7 (`science/chemistry-correctness.md` § 2a): the electronic state is
    part of the level of theory the record is compared against, resolved --
    H2 at -2 keeps the electron count even, so the parity rule does not
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

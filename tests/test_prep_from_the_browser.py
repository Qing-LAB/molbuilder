"""`prep` from the browser — the same verb, for the machine you name.

**Why a browser may trigger it.**  `project-layout.md` § 2.2 says the deck
cannot be finished in the browser, and its argument is about WHOSE FACTS the
deck is rendered from: `prep` needs four inputs, two portable (the template,
the description) and two the target machine's.  A named record supplies the
machine half -- that is what `environments/<name>.json` IS -- so preparing
FOR a cluster FROM here is the case `preparing-for-another-machine.md` exists
for.  The section constrains the FACTS, not the surface.

**Prep, never launch** (user, 2026-08-24): prep writes files and can be run
again; launch spends a queue slot and refuses batch submission by design.

**The reserved local name.**  `known_machines` displays `(this machine)`,
which is a label nobody can type -- and with any named record on file,
omitting `--target` raised `AmbiguousTarget`, whose own message said *"omit
--target only when this machine is the one"*.  The instruction the refusal
gave was the action that produced it, so preparing for the box you are
sitting at became impossible the moment you saved one cluster record.
`LOCAL_TARGET` is the typeable name; C1 still refuses SILENCE.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from molbuilder.scheduler.record import LOCAL_TARGET


@pytest.fixture
def described(isolated_projects_root, web_client):
    """A described calculation inside the projects tree the app serves, plus
    one NAMED record so the machine question is genuinely ambiguous.

    Built in an ISOLATED tree: these tests write, and the app's default root
    is the developer's real `projects/`.
    """
    from molbuilder.task import FILENAME as TASK_FILENAME
    calc = isolated_projects_root / "calc"
    calc.mkdir(parents=True, exist_ok=True)
    (calc / TASK_FILENAME).write_text(json.dumps({
        "schema": "molbuilder/task@1",
        "engine": {"name": "siesta"}, "shape": "hierarchical",
        "run": {"name": "JOB", "id": "JOB_H2"},
        "structure": {"source": "h2.xyz", "formula": "H2", "atoms": 2},
        "varies": [],
        "stages": [{"name": "coarse", "enabled": True, "overrides": {}}],
        # THE RUN CARD STATES THE LAUNCH SHAPE (`architecture.md` § 5.2) --
        # a run whose shape is stated nowhere is refused at prep.
        "execution": {"mpi_np": 2, "omp_threads": 1},
    }))
    from molbuilder.scheduler import environments_dir
    d = environments_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "faraway.json").write_text(json.dumps({
        "schema": "molbuilder/environment@2",
        "detected_at": "2026-08-01T00:00:00+00:00", "scheduler": "slurm",
        "topology": {"sockets": 2, "cores_per_socket": 24,
                     "threads_per_core": 1, "numa_per_socket": None,
                     "gpus_per_node": 0, "gpu_type": None,
                     "mem_total_gb": 500.0},
        "site": {"partition": "htc", "qos": "public"},
        "domains": [],
    }))
    return str(calc)


def _make_preppable(calc):
    """Finish *calc* into a folder `prep` will actually act on.

    The `described` fixture stops at `task.json`, which is all its refusal
    tests need.  A portable folder is a template PLUS a description
    (`project-layout.md` § 2.1), and the structure the description names has
    to be there, so `prep` has something to render from.  Written through
    the product's own doors rather than spelled, so the fixture cannot drift
    away from what `jobset init` leaves.
    """
    from molbuilder import template as _T
    from molbuilder.config.siesta import SiestaConfig
    (calc / "h2.xyz").write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    # The pseudopotential the description's species needs, in the folder --
    # `prep` refuses without one (`project-layout.md` § 2.6) and then runs
    # the screening over it (`science/pseudopotentials.md` § 1), so a
    # touch-file will not do.  `write_pseudos` is the suite's one home for
    # a PSML that parses.
    from conftest import write_pseudos
    write_pseudos(calc, ["H"])
    cfg = SiestaConfig(system_label="JOB")
    (calc / "JOB.template.toml").write_text(
        _T.template_with_values(cfg, engine="siesta"))


def _post(client, **body):
    r = client.post("/api/task-setup/prep", json=body)
    return r.status_code, (r.get_json() or {})


def test_it_refuses_a_folder_with_no_description(web_client, described,
                                                 isolated_projects_root):
    bare = isolated_projects_root / "bare"
    bare.mkdir()
    st, j = _post(web_client, dest=str(bare), kind="run", stage="coarse",
                  target=LOCAL_TARGET, plan=True)
    assert st == 400 and "task.json" in j["error"]


def test_silence_about_the_machine_is_still_refused(web_client, described):
    """C1 unchanged: the browser cannot offer a default the CLI rejects."""
    st, j = _post(web_client, dest=described, kind="run", stage="coarse",
                  plan=True)
    assert st == 400
    assert "none was named" in j["error"] or "task.json" in j["error"]


def test_the_local_machine_can_be_NAMED(web_client, described):
    """The regression this closes: with a named record on file there was no
    spelling for "the box I am on" at all."""
    from molbuilder.scheduler.record import machine_for
    # (`assert ... is None or True` stood here, unfailable and therefore
    #  saying nothing.  The important half is below and always was.)
    # the important half -- it does not raise the ambiguity refusal
    from molbuilder.scheduler.record import AmbiguousTarget
    try:
        machine_for(target=LOCAL_TARGET)
    except AmbiguousTarget:
        pytest.fail("naming this machine still reads as silence")


def test_a_prep_from_here_is_recorded_in_the_bundle(web_client, described):
    """This surface acted, so this surface appends (`jobset/ledger.py`).

    It did not.  The only importer of `ledger.record` in the package was the
    CLI, while the Task Setup **bundle card** listed `jobset-decisions.log`
    as *"every decision prep made, one line each"* — so a calculation
    prepped only from the browser had no such file, and one prepped from
    both told a false story by holding the CLI's lines alone.  Measured
    2026-09-19: `grep -rn ledger molbuilder/web/` returned one line, the
    import of the file's NAME for that card.

    The line's CONTENTS are `ledger.prepped`'s and are not re-asserted here
    — what this pins is that the surface calls it at all.
    """
    import json as _json
    from molbuilder.jobset.ledger import LEDGER_FILE

    _make_preppable(Path(described))
    st, j = _post(web_client, dest=described, kind="run", stage="coarse",
                  target=LOCAL_TARGET, save=False)
    assert st == 200, j

    log = Path(described) / LEDGER_FILE
    assert log.is_file(), (
        "prep from the browser wrote no ledger line; the bundle card "
        "promises one")
    lines = [_json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert ("prep", "prepped") in [(e["verb"], e["decision"]) for e in lines]


def test_the_refusal_names_a_spelling_that_WORKS(web_client, described):
    """Its own instruction used to be the action that caused it."""
    from molbuilder.scheduler.record import AmbiguousTarget, machine_for
    with pytest.raises(AmbiguousTarget) as e:
        machine_for(target=None)
    assert f"--target {LOCAL_TARGET}" in str(e.value)
    assert "omit --target only when" not in str(e.value)


def test_the_reserved_name_cannot_be_taken_by_a_record():
    """A record called `this` could never be prepped for, so writing one is
    refused rather than allowed and then shadowed."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group,
                           ["probe", "--write", "--yes", "--name",
                            LOCAL_TARGET])
    assert r.exit_code != 0
    assert "reserved" in r.output


def test_an_unknown_machine_is_refused_by_name(web_client, described):
    st, j = _post(web_client, dest=described, kind="run", stage="coarse",
                  target="no-such-box", plan=True)
    assert st == 400 and "no-such-box" in j["error"]


def test_kind_must_be_run_or_bench(web_client, described):
    st, j = _post(web_client, dest=described, kind="launch",
                  stage="coarse", target=LOCAL_TARGET, plan=True)
    assert st == 400 and "run" in j["error"] and "bench" in j["error"]


def test_there_is_no_launch_door_here(web_client):
    """Prep writes files; launch spends a queue slot.  Only the first is
    exposed, and its absence should be visible rather than assumed."""
    from molbuilder.web.blueprints import build as _b
    routes = {r for r in dir(_b) if r.startswith("api_task_setup")}
    assert "api_task_setup_prep" in routes
    assert not any("launch" in r or "submit" in r for r in routes), (
        "a submit door appeared on the task-setup blueprint")


# --------------------------------------------------------------------- #
#  ONE prep entry, two doors (`job-system.md` § 5.3; plan W38 F7)       #
# --------------------------------------------------------------------- #
#
# Until 2026-09-29 this door called the five steps alone: it skipped the
# preflight, the question prep asks (*already under way*, until 2026-10-02;
# the save since), the launch agreement and their ledger lines, refused an
# axis-less bench, and showed only the folders.  Each test below runs the SAME prep through both doors -- the
# command line and this route -- and compares what each says and records.

def _twin(calc, name):
    """A second, identical calculation beside *calc*, for the other door."""
    twin = Path(calc).parent / name
    shutil.copytree(calc, twin)
    return twin


def _ledger_decisions(calc):
    from molbuilder.jobset.ledger import LEDGER_FILE
    log = Path(calc) / LEDGER_FILE
    return [(e["verb"], e["decision"]) for e in
            (json.loads(x) for x in log.read_text().splitlines() if x.strip())]


def _cli(*args, input=None):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, list(args), input=input)


def _a_ladder_the_preflight_warns_about(calc):
    """`tight` resolving to `coarse`'s settings and starting clean -- the
    description's own preflight warns (`engines/stages.md` § 6.6a)."""
    from molbuilder.task import FILENAME as TASK_FILENAME
    desc = Path(calc) / TASK_FILENAME
    d = json.loads(desc.read_text())
    d["varies"] = []
    d["stages"] = [{"name": "coarse", "enabled": True, "overrides": {}},
                   {"name": "tight", "enabled": True, "overrides": {},
                    "execution": {"restart": "clean"}}]
    desc.write_text(json.dumps(d))


def test_both_doors_give_the_same_answer_and_record_the_same_decisions(
        web_client, described):
    """The tab's answer carries what the command line prints -- the
    preflight's note, the attempt, the deck's agreement -- and the two
    calculations' ledgers hold the same decisions."""
    calc = Path(described)
    _make_preppable(calc)
    _a_ladder_the_preflight_warns_about(calc)
    twin = _twin(calc, "calc-cli")

    # the save, answered up front: no, as the terminal's silence is
    st, j = _post(web_client, dest=str(calc), kind="run", stage="coarse",
                  target=LOCAL_TARGET, save=False)
    assert st == 200, j
    r = _cli("prep", "run", "coarse", "--bundle", str(twin),
             "--target", LOCAL_TARGET)
    assert r.exit_code == 0, r.output

    notes = [f["message"] for f in j["findings"]]
    assert notes and any("starts clean" in n for n in notes), j["findings"]
    for n in notes:
        assert f"note: {n}" in r.output, n
    assert j["attempt"]["dir"] == "01_coarse/run-0"
    assert "prepared coarse: 01_coarse/run-0" in r.output
    assert j["agreement"]["verdict"] == "agrees", j["agreement"]
    assert (f"{j['deck']}: rendered for mpi_np "
            f"{j['agreement']['rendered_for']} -- agrees") in r.output
    assert _ledger_decisions(calc) == _ledger_decisions(twin)
    assert ("prep", "preflight-report") in _ledger_decisions(calc)
    assert ("prep", "launch-agreement") in _ledger_decisions(calc)
    # What each deck's checks said reaches the tab too -- the terminal read
    # it on stderr as the deck rendered.
    assert j["deck_findings"], "the tab was not told what the deck's checks said"
    for f in j["deck_findings"]:
        assert f["message"] in r.output, f["message"]
    # ...and the terminal says what was found BEFORE the decks are written,
    # as it always has (`prep_stage`'s ``on_found``).
    assert r.output.index(f"note: {notes[0]}") < min(
        r.output.index(f["message"]) for f in j["deck_findings"]), r.output


def test_the_save_is_offered_on_both_doors_and_nothing_is_written_first(
        web_client, described):
    """`checkpointing.md` § 9: before prep writes into a folder whose state
    is not saved, it offers the save, its note drafted.  The tab gets the
    offer with nothing written, and its answer saves the folder with the
    note it sent; the terminal asks, and a yes saves it with the draft --
    each recorded in its calculation's ledger.  (It replaced the tests of
    *already under way here*, 2026-10-02: a prepped stage is refused now,
    `job-system.md` § 5.0.)"""
    from molbuilder.checkpoint import Repo
    calc = Path(described)
    _make_preppable(calc)
    twin = _twin(calc, "calc-cli")

    st, j = _post(web_client, dest=str(calc), kind="run", stage="coarse",
                  target=LOCAL_TARGET)
    assert st == 200 and j["offer"] is not None, j
    assert j["offer"]["note"] == "before prep run coarse", j["offer"]
    assert j["offer"]["standing_at"] is None, j["offer"]   # none saved yet
    assert j["dirs"] == [] and not (calc / "01_coarse").exists(), (
        "an offer unanswered wrote something")
    st, j = _post(web_client, dest=str(calc), kind="run", stage="coarse",
                  target=LOCAL_TARGET, save=True, note="before coarse, here")
    assert st == 200 and j["offer"] is None and j["dirs"], j
    assert [x.note for x in Repo(str(calc)).states()] == [
        "before coarse, here"]

    r = _cli("prep", "run", "coarse", "--bundle", str(twin),
             "--target", LOCAL_TARGET, input="y\n\n")
    assert r.exit_code == 0, r.output
    assert [x.note for x in Repo(str(twin)).states()] == [
        "before prep run coarse"]

    answers = [json.loads(x)["answer"] for d in (calc, twin)
               for x in (d / "jobset-decisions.log").read_text().splitlines()
               if '"save-offer"' in x]
    assert [a.startswith("saved as ") for a in answers] == [True, True], (
        answers)


def test_a_bench_with_no_axes_is_the_machines_proposal_on_both_doors(
        web_client, described):
    """`generator.md` § 4.3a: an absent declaration keeps the machine's
    enumeration.  The tab refused it -- "declares nothing to measure" --
    while the command line prepped it."""
    calc = Path(described)
    _make_preppable(calc)
    twin = _twin(calc, "calc-cli")
    st, j = _post(web_client, dest=str(calc), kind="bench", stage="coarse",
                  target=LOCAL_TARGET, save=False)
    assert st == 200, j
    r = _cli("prep", "bench", "coarse", "--bundle", str(twin),
             "--target", LOCAL_TARGET)
    assert r.exit_code == 0, r.output
    assert j["dirs"] and f"prepped {len(j['dirs'])} trial dir(s)" in r.output
    assert any("combination(s) enumerated" in n for n in j["notes"]), j["notes"]


def test_a_refusal_shows_what_it_points_at_on_both_doors(
        web_client, described):
    """A bench whose every declared point is crossed out refuses with "see
    the crossed-out list above" -- and each door shows that list with the
    sentence: the terminal before it, the tab beside it.  The entry
    assembled the list before refusing; a refusal that dropped it would be
    pointing at nothing."""
    from molbuilder.task import FILENAME as TASK_FILENAME
    calc = Path(described)
    _make_preppable(calc)
    desc = calc / TASK_FILENAME
    d = json.loads(desc.read_text())
    d["bench"] = {"mpi_np": [4096]}
    desc.write_text(json.dumps(d))
    twin = _twin(calc, "calc-cli")

    st, j = _post(web_client, dest=str(calc), kind="bench", stage="coarse",
                  target=LOCAL_TARGET)
    assert st == 400 and "crossed-out list" in j["error"], j
    assert any("crossed out (1)" in n for n in j["notes"]), j["notes"]
    r = _cli("prep", "bench", "coarse", "--bundle", str(twin),
             "--target", LOCAL_TARGET)
    assert r.exit_code != 0, r.output
    assert r.output.index("crossed out (1)") < r.output.index(
        "crossed-out list"), r.output


# `test_a_confirm_answers_only_the_evidence_it_was_shown` retired 2026-10-02 with
# the question it pinned: a prepped stage is refused now (`job-system.md` § 5.0).


def _pyscf_calc(root, name):
    """A described PySCF optimization -- its deck makes no claim about the
    launch it was rendered for (no BENCH-MARKS block)."""
    from molbuilder import template as _T
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.task import FILENAME as TASK_FILENAME
    calc = Path(root) / name
    calc.mkdir(parents=True)
    (calc / TASK_FILENAME).write_text(json.dumps({
        "schema": "molbuilder/task@1",
        "engine": {"name": "pyscf"}, "shape": "hierarchical",
        "run": {"name": "JOB", "id": "JOB_H2"},
        "structure": {"source": "h2.xyz", "formula": "H2", "atoms": 2},
        "varies": [],
        "stages": [{"name": "coarse", "enabled": True, "overrides": {}}],
        "execution": {"threads": 1},
    }))
    (calc / "h2.xyz").write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    (calc / "JOB.template.toml").write_text(
        _T.template_with_values(PySCFConfig(), engine="pyscf"))
    return calc


def test_a_deck_that_makes_no_claim_gets_no_agreement_on_either_door(
        web_client, described, isolated_projects_root):
    """`launch_agreement` answers *silent* for a deck with no launch claim,
    and saying *agrees* would be a claim nobody made: no agreement in the
    answer, no line printed, no `launch-agreement` in the ledger -- on
    both doors.  A PySCF deck is such a deck."""
    web = _pyscf_calc(isolated_projects_root, "py-web")
    cli = _pyscf_calc(isolated_projects_root, "py-cli")
    st, j = _post(web_client, dest=str(web), kind="run", stage="coarse",
                  target=LOCAL_TARGET, save=False)
    assert st == 200, j
    assert j["agreement"] is None and j["attempt"], j
    r = _cli("prep", "run", "coarse", "--bundle", str(cli),
             "--target", LOCAL_TARGET)
    assert r.exit_code == 0, r.output
    assert "rendered for mpi_np" not in r.output
    for calc in (web, cli):
        assert ("prep", "launch-agreement") not in _ledger_decisions(calc)



# `test_the_terminal_is_asked_again_when_the_folder_changed_meanwhile` retired
# 2026-10-02 with the question it pinned (`job-system.md` § 5.0).


def test_a_preflight_refusal_keeps_its_notes_on_both_doors(
        web_client, described):
    """A description that fails its own preflight -- a physics value in the
    run's `execution` block -- while its ladder also earns a note: each door
    refuses with the note beside the sentence, and the ledger holds it, as
    for every other refusal (HEAD's terminal did; the one entry dropped it
    until review, 2026-09-29)."""
    from molbuilder.task import FILENAME as TASK_FILENAME
    calc = Path(described)
    _make_preppable(calc)
    _a_ladder_the_preflight_warns_about(calc)
    desc = calc / TASK_FILENAME
    d = json.loads(desc.read_text())
    d["execution"] = {"basis_size": "SZ"}
    desc.write_text(json.dumps(d))
    twin = _twin(calc, "calc-cli")

    st, j = _post(web_client, dest=str(calc), kind="run", stage="coarse",
                  target=LOCAL_TARGET)
    assert st == 400 and "fails its own preflight" in j["error"], j
    assert any("starts clean" in f["message"] for f in j["findings"]), j
    r = _cli("prep", "run", "coarse", "--bundle", str(twin),
             "--target", LOCAL_TARGET)
    assert r.exit_code != 0, r.output
    assert r.output.index("starts clean") < r.output.index(
        "fails its own preflight"), r.output
    # The notes, then the refusal itself -- a decision too (W52).
    assert (_ledger_decisions(calc) == _ledger_decisions(twin)
            == [("prep", "preflight-report"), ("prep", "refused")])


def test_a_folder_holding_two_templates_is_refused_in_words_on_both_doors(
        web_client, described):
    """`find_template` refuses a folder with two templates -- a leftover the
    person removes -- and the entry says so on both doors: a 400 with the
    sentence, and the terminal's `Error:`, never a 500 or a traceback."""
    calc = Path(described)
    _make_preppable(calc)
    shutil.copy(calc / "JOB.template.toml", calc / "OLD.template.toml")
    twin = _twin(calc, "calc-cli")
    st, j = _post(web_client, dest=str(calc), kind="run", stage="coarse",
                  target=LOCAL_TARGET)
    assert st == 400 and "holds 2 templates" in j["error"], j
    r = _cli("prep", "run", "coarse", "--bundle", str(twin),
             "--target", LOCAL_TARGET)
    assert isinstance(r.exception, SystemExit), r.exception
    assert r.exit_code == 1 and "holds 2 templates" in r.output, r.output


def test_the_page_is_offered_a_bench_only_where_prep_takes_one(
        web_client, described, isolated_projects_root):
    """The folder answer carries the entry's own reason a description takes
    no bench (`bench_refusal`), so the tab offers the Measure step exactly
    where `prep bench` would take it -- and a bench preview is refused in
    the same words as the write (found by review, 2026-09-29: the page
    offered a bench on every PySCF and transport calculation)."""
    calc = Path(described)
    _make_preppable(calc)
    py = _pyscf_calc(isolated_projects_root, "py")

    def _folder(d):
        return web_client.get("/api/task-setup/folder",
                              query_string={"dir": str(d)}).get_json()

    assert _folder(calc)["bench_refusal"] is None
    why = _folder(py)["bench_refusal"]
    assert why and "only speaks SIESTA" in why, why
    for plan in (True, False):
        st, j = _post(web_client, dest=str(py), kind="bench", stage="coarse",
                      target=LOCAL_TARGET, plan=plan)
        assert st == 400 and j["error"] == why, (plan, j)

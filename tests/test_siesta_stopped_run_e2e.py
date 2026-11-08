"""A SIESTA run that SIESTA stops, made on the road, and what each reader says
about it.

The pass's own stopped run (`support.real_runs.stopped`,
``real_h2_stopped``): ``molbuilder describe`` -> ``prep task --group 1`` ->
``launch task --mode direct`` on the road's H2 relaxation, its SCF unable to
converge within its cap: ten cycles of plain linear mixing at weight 0.001
against a DM tolerance of 1e-8, set in the calculation's template, with
``SCF.MustConverge`` left at SIESTA's default -- abort.  SIESTA prints
``SCF_NOT_CONV: ... (required)`` and ``die``s, and ``die`` writes its own
cascade after it; the run ends there, failed, and the run script says what a
launch again will do (`running-a-job.md` § 3.5: to continue is to launch
again).

What each reader must say (`running-a-job.md` § 4.2, `model/parse.md` § 2b):
the folder is ``failed``, its detail quoting the line that stopped it -- the
SCF's, not the cascade's; the viewer's stop reason is that cause in the
table's words; and the run's session log is found as the run's.  The Results
tab's Run panel (`web/results.md` § 3a) says the same, from the run's record,
in a browser.  And what molbuilder does with a run that failed
(`job-system.md` § 5.4): the stage after it is prepared -- prep takes
nothing from another run -- and its launch refused, saying so, and `status`
says what launching it again does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _road import conda_hook, env_available

CONDA_SH = conda_hook()

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (CONDA_SH.is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


@pytest.fixture(scope="module")
def stopped(real_h2_stopped):
    """The run SIESTA stopped: `coarse`'s first run's folder
    (:meth:`support.real_runs.RealRun.attempt`), its ladder the shipped
    `publishable` one, its deck stating the cap the person set, and its
    output there."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runs import run_of
    from molbuilder.task import of as task_of
    bundle = real_h2_stopped.bundle
    assert [s.name for s in task_of(bundle).stages] == [
        "coarse", "medium"], task_of(bundle).stages
    attempt = real_h2_stopped.attempt("coarse")
    home = stage_home(bundle, task_of(bundle), "coarse")
    deck = (attempt / home.names.name(".fdf")).read_text()
    assert any(ln.split()[:2] == ["MaxSCFIterations", "10"]
               for ln in deck.splitlines()), deck[-2000:]
    assert run_of(attempt).stdout is not None, \
        sorted(p.name for p in attempt.iterdir())
    return attempt


def test_the_folder_is_failed_and_says_what_stopped_it(stopped,
                                                        real_h2_stopped):
    """The folder's status quotes the line that stopped the run -- the SCF's,
    which SIESTA states as fatal, and not the ``die`` lines after it -- and
    the launch that waited on it said the same, in the same words, and
    exited 1 (`job-system.md`, *A run here, waited on*): a script trusting
    the exit status is told the run failed -- and it ended with the next
    step (*The task*, *The next step*), the stage that failed and its launch
    again.  The silent failure: a launch that leaves you with no follow-up,
    or one naming another group's line."""
    from molbuilder.parse.dirs import run_status
    from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER

    st = run_status(stopped, "H2_01_coarse")    # the run's stem
    assert st.state == "failed", st
    assert st.detail.startswith("stopped before its end: SCF_NOT_CONV"), st
    # the run's one output speaks, stopped on the SCF
    assert st.active_source == "H2_01_coarse.out", st
    end = st.endings["H2_01_coarse.out"]
    assert (end.run_state, end.cause) == ("stopped", SCF_NOT_CONV_MARKER), end

    said = real_h2_stopped.said["launch coarse"]
    assert said.exit_code == 1, said.output
    line = next(ln for ln in said.output.splitlines()
                if ln.strip().startswith("FAILED"))
    assert f"coarse: failed -- {st.detail}" in line, line
    steps = said.output.split("each group, and what is next:\n", 1)[1]
    assert "failed: coarse -- launch it again" in steps, steps
    assert "launch task --stage coarse" in steps, steps


def test_a_run_that_failed_is_never_built_on(real_h2_stopped):
    """The stage after it is prepared -- prep takes nothing from another
    run -- and its first launch is refused before anything is sent, naming
    the run and how it ended (`job-system.md` § 5.4: the newest run, which
    must have finished; *Every refusal*, R-L11: what it builds on is not
    usable)."""
    said = real_h2_stopped.said
    assert said["prep medium"].exit_code == 0, said["prep medium"].output
    r = said["launch medium"]
    assert r.exit_code != 0, r.output
    assert "01_coarse/run-0, which failed -- " in r.output, r.output


def test_status_says_what_launching_it_again_does(real_h2_stopped):
    """However it ended, a prepared stage is launched again -- warm, from
    its own latest run, for a relaxation -- and `status` says so."""
    r = real_h2_stopped.said["status"]
    assert r.exit_code == 0, r.output
    for words in ("failed: coarse",
                  "launch it again -- it continues from its own latest run"):
        assert words in r.output, (words, r.output)


def test_the_viewer_says_why_in_the_ending_readers_words(stopped,
                                                         real_h2_stopped,
                                                         monkeypatch):
    """The viewer's "Reason:" line is the server's: the file's cause as the
    one ending reader states it, in the SIESTA family's words
    (`siesta_grammar.CAUSE_WORDS`) -- the browser keeps no copy of the
    markers.

    MUTATION THIS MUST FAIL AGAINST: take the LAST fatal line as the cause.
    """
    from conftest import point_the_one_door_at
    from molbuilder.parse.engines.siesta_grammar import (CAUSE_WORDS,
                                                         SCF_NOT_CONV_MARKER)
    from molbuilder.web.app import create_app

    # THE PROJECTS TREE THE RUN WAS MADE IN, the one door pointed at it, so
    # the route reads it.
    point_the_one_door_at(real_h2_stopped.tree.parent, monkeypatch.setenv)
    app = create_app(config={})
    from molbuilder.runs import run_of
    out = run_of(stopped).stdout
    body = app.test_client().post("/api/watch/load",
                                  json={"path": str(out)}).get_json()
    assert body["ok"] is True, body
    assert body["data"]["run_state"] == "stopped"
    assert body["data"]["stop_reason"] == CAUSE_WORDS[SCF_NOT_CONV_MARKER]


def test_a_step_siesta_stopped_inside_is_no_step(stopped):
    """`runs.StepsTable`: the run SIESTA stopped inside its first SCF holds no
    step it finished -- its last frame read as cut short, and its steps
    refused by name.  The silent failure: the energy of an SCF row read as
    the run's final energy, about 1000 eV off on a real 52-atom relaxation
    stopped inside a move.  MUTATION THIS MUST FAIL AGAINST: a frame the run
    stopped inside counted among its steps (the ``cut_short`` clause of
    `runs.steps_table`'s filter)."""
    from molbuilder.parse import detect
    from molbuilder.postprocess import RunStepsError, read_steps
    from molbuilder.runs import run_of
    out = run_of(stopped).stdout
    assert detect(out).parse(str(out)).frames[-1].cut_short
    with pytest.raises(RunStepsError, match="holds no step its run finished"):
        read_steps(stopped)


def test_the_run_is_one_start_that_says_what_a_launch_again_does(stopped):
    """A run is one start (`project-layout.md` § 1.6.1): its one session
    log, named on the stage's stem, found as the run's -- the log the run
    record and ``run_status`` read (`wrapper_log.log_of_run`) -- holds that
    start alone; and, the SCF having stopped it, the run script ends it
    saying what a launch again will do: continue warm from the files this
    run saved (`running-a-job.md` § 3.5).

    THE SILENT FAILURES: the run script starting the engine a second time
    in the folder; a failed run that leaves the person no word of how to
    continue it."""
    from molbuilder.wrapper_log import WRAPPER_LOG_START, log_of_run

    log = log_of_run(stopped, "H2", "01_coarse")
    assert log is not None and log.name == "H2_01_coarse.runwrap.log", (
        sorted(p.name for p in stopped.iterdir()))
    text = log.read_text(errors="replace")
    assert text.count(WRAPPER_LOG_START) == 1, text
    assert "warm from the files this run saved" in text, text


def test_the_setup_tells_a_key_the_engine_read_alone_from_the_decks(stopped):
    """`engine_only` names what the engine read and no catalogue item
    writes (`model/parse.md` § 5d.3), as the Results tab's folder door
    answers it.  SIESTA reads some keys twice, each time its own default --
    `DM.NumberPulay` and `MD.FinalTimeStep` in this run's fdf log -- and
    such a key is not the deck's: no item writes `MD.FinalTimeStep`, which
    bounds the molecular dynamics molbuilder does not run (`engines/siesta.md`
    § 3, row 14).  A key a catalogue item writes is a row of the setup, never
    one of these: `MeshCutoff` (`mesh_cutoff`).

    MUTATION THIS MUST FAIL AGAINST: `in_deck` from a top-level `default`
    alone -- a key read several times has none.
    """
    from molbuilder.runs import folder_answer
    setup = folder_answer(stopped)["record"]["setup"]
    rows = {r["key"].lower(): r for r in setup["engine_only"]}
    for key in ("dm.numberpulay", "md.finaltimestep"):
        row = rows[key]
        assert len(row["readings"]) == 2 and row["in_deck"] is False, row
    written = {k.lower() for r in setup["rows"] for k in r.get("keys", [])}
    assert "meshcutoff" in written and "meshcutoff" not in rows, written


def test_the_run_panel_says_what_ran_and_why_it_stopped(stopped, page,
                                                         real_h2_stopped,
                                                         monkeypatch):
    """The Results tab's Run panel (`web/results.md` § 3a) reads this run's
    record: closed, one line -- the engine, the ranks, how it ended (its one
    ending: a run is one start, `project-layout.md` § 1.6.1); open, the
    verdict, the setup's three columns, the computation and the deck, which
    opens in the sidebar's viewer.  It is hidden for a
    container, and from the moment the panel is bound to another folder --
    a scan that fails announces nothing, so the old record must not stay.

    MUTATION THIS MUST FAIL AGAINST: the picker not carrying ``record`` in
    its selection event; the panel not hiding when it is re-bound; the file
    card following a sidebar click by the scope's folder rather than the
    folder its files are from.
    """
    import shutil

    from conftest import point_the_one_door_at
    from support.live_server import serve
    from support.real_runs import CANNOT_CONVERGE
    from support.sidebar import reload_from, results_on, walk_to

    # THE PROJECTS TREE THE RUN WAS MADE IN, the one door pointed at it, and
    # the server's (`support.live_server.serve`).
    root = real_h2_stopped.tree
    bundle = real_h2_stopped.bundle
    point_the_one_door_at(root.parent, monkeypatch.setenv)
    with serve() as base:
        results_on(page, base, stopped)
        panel = page.locator("#results-run-panel")
        panel.locator(".rp-summary").wait_for(timeout=20000)
        record = page.evaluate(
            "(d) => fetch('/api/results/dir?path=' + encodeURIComponent(d))"
            ".then(r => r.json()).then(b => b.record)", str(stopped))

        line = panel.locator(".rp-summary").inner_text()
        engine = record["computation"]["engine"]
        assert f"{engine['program']} {engine['version']}" in line, line
        assert "1 rank" in line and "stopped" in line, line
        assert "FAILED" in line.upper(), line

        panel.locator(".rp-toggle").click()
        sections = {s.locator(".rp-section-title").inner_text():
                    s.inner_text() for s in panel.locator(".rp-section").all()}
        assert list(sections) == ["Verdict", "Setup", "Computation", "Deck"]
        assert "stopped before its end: SCF_NOT_CONV" in sections["Verdict"]
        assert (record["computation"]["host"]["hostname"]
                in sections["Computation"])

        # THE SETUP'S THREE COLUMNS, as the run recorded them: the catalogue
        # default from the wrapper's block, what the deck asked, what SIESTA
        # read.
        rows = page.evaluate(
            "() => [...document.querySelectorAll("
            "  '#results-run-panel .rp-setup tbody tr')].map(tr => ["
            "    tr.querySelector('.rp-item').textContent,"
            "    ...[...tr.children].slice(1).map(td => td.textContent)])")
        by_item = {r[0]: r[1:] for r in rows}
        stated = {r["item"]: r for r in record["setup"]["rows"]}
        default, asked, used = by_item["max_scf_iter"]
        assert (default, asked, used) == (
            str(stated["max_scf_iter"]["default"]), "10", "10"), \
            by_item["max_scf_iter"]
        for item, value in CANNOT_CONVERGE.items():
            _d, asked, used = by_item[item]
            assert float(asked) == float(value) == float(used.split()[0]), (
                item, by_item[item])
        unset = [r["item"] for r in record["setup"]["rows"]
                 if "asked" not in r]
        assert unset and all(by_item[i][1] == "engine default" for i in unset)
        assert (f"({len(record['setup']['engine_only'])})"
                in panel.locator(".rp-fold > summary").inner_text())
        assert "H.psml" in sections["Setup"]

        panel.locator(".rp-view").click()
        preview = page.locator("#ps-preview-modal")
        preview.wait_for(state="visible", timeout=10000)
        # THE DECK THE RECORD NAMES, whole, opened as it is (`web/results.md`
        # § 0.3) -- the viewer shows its name.
        assert Path(record["deck"]["path"]).name in preview.inner_text()
        page.keyboard.press("Escape")

        # A FOLDER WHOSE SCAN FAILS: one of the person's, the sidebar walked
        # to it, then deleted from the disk elsewhere before Reload -- nothing
        # is announced, and the panel must not keep the run above under it.
        # Two levels gone: the door answers a path that is not a directory
        # for its parent, so one missing level would be answered as its
        # parent.
        gone = root / "P" / "gone"
        (gone / "attempt").mkdir(parents=True)
        walk_to(page, gone / "attempt")
        assert panel.locator(".rp-summary").is_visible(), (
            "browsing moved the panel off the run")
        shutil.rmtree(gone)
        page.get_by_role("button",
                         name="Reload from current project dir").click()
        panel.wait_for(state="hidden", timeout=10000)
        reload_from(page, stopped)
        panel.locator(".rp-summary").wait_for(timeout=20000)
        # The calculation root is a container: it has a ladder, not a record.
        reload_from(page, bundle)
        page.locator(".results-ladder").wait_for(state="visible",
                                                 timeout=20000)
        assert panel.is_hidden()
        # ...WITH THE NEXT STEP under its rows (`web/results.md` § 2.4): the
        # ladder door's own answer, the next group's words and lines, and when
        # it was read.
        from molbuilder.calculation import open as open_calculation
        from molbuilder.jobset.runstatus import jobset_status
        step = next(g for g in jobset_status(open_calculation(bundle)).steps
                    if g.next)
        shown = page.locator(
            ".results-ladder-steps-list li.is-next").inner_text()
        # ...ITS WORDS AS THE PAGE SETS THEM: a `quoted` command drawn as
        # code, its quotes the marks that say so (`dom.codeWords`, 2.67.2).
        assert step.words.replace("`", "") in shown and step.lines, (
            step, shown)
        assert all(line in shown for line in step.lines), (step, shown)
        assert "read at" in page.locator(
            ".results-ladder-steps-head").inner_text()

        # A RUN PICKED OFF THE LADDER: the file card then holds THAT run's
        # files (`web/results.md` § 2.4) while the sidebar stays on the root
        # -- so a single click on a root file is not looked up among the
        # run's files, and the card keeps naming the run's; a click on
        # another file of the run's own folder is followed (R2-15,
        # `archive/2026-10-13-plan-rows.md`).
        menu = page.locator("#results-file-picker-select")
        chosen = "(s) => s.options[s.selectedIndex].textContent"
        at_root = menu.evaluate(chosen)
        page.locator(".results-ladder-open[aria-pressed]").first.click()
        page.locator(".results-ladder-open[aria-pressed='true']").wait_for(
            timeout=20000)
        card = page.locator("#results-file-card .fc-name")
        card.wait_for(timeout=20000)
        picked = card.inner_text()
        walk_to(page, bundle / "task.json")
        page.wait_for_timeout(500)
        assert card.inner_text() == picked, (
            "a click on a root file was taken for one of the picked run's "
            "files")
        # THE TITLE NAMES THE FILE FROM THE PROJECTS FOLDER, as the sidebar
        # names a path it only shows (`web/results.md` \u00a7 0.3).
        opened = page.locator(".results-ladder-open[aria-pressed='true']") \
            .first.get_attribute("title").split(" \u2014 ")[0]
        run_dir = root / Path(opened).parent
        other = next(f for f in sorted(run_dir.iterdir())
                     if f.is_file() and f.name != picked)
        walk_to(page, other)
        page.wait_for_function(
            "(n) => document.querySelector('#results-file-card .fc-name')"
            "         .textContent === n", arg=other.name, timeout=10000)

        # ...AND THE MENU NAMES THE RUN SHOWN (`web/results.md` § 2.2,
        # § 2.4): the root's menu row names no file of the picked run, so the
        # menu's first row says which run is shown; *back to the calculation*
        # names the root's choice again.
        assert menu.evaluate(chosen) == (
            f"(showing {Path(opened).name}, a run off the ladder)")
        page.locator(".results-ladder-back").click()
        page.wait_for_function(
            "(t) => { const s = document.querySelector("
            "'#results-file-picker-select');"
            " return s.selectedIndex >= 0"
            "   && s.options[s.selectedIndex].textContent === t; }",
            arg=at_root, timeout=10000)
        assert menu.locator("option[data-picked-run]").count() == 0

        # RELOAD KEEPS THE PICK; A CHOICE IN THE MENU LEAVES IT
        # (`web/results.md` § 2.4).  THE SILENT FAILURE: the run a person
        # opened gone from the screen on every Reload, or a file chosen in the
        # menu not shown at all, the picked run put back over it.
        # THE SIDEBAR BACK ON THE CALCULATION, where the pick lives (§ 2.4:
        # "while the sidebar stays on the calculation"): Reload binds the
        # panel to the sidebar's folder, and the click above left it on the
        # run's.
        walk_to(page, bundle)
        page.locator(
            ".results-ladder-open[aria-pressed='false']").first.click()
        page.wait_for_selector(".results-ladder-back", timeout=10000)
        showing = menu.evaluate(chosen)
        page.get_by_role("button",
                         name="Reload from current project dir").click()
        page.wait_for_function(
            "(t) => { const s = document.querySelector("
            "'#results-file-picker-select');"
            " return s.selectedIndex >= 0"
            "   && s.options[s.selectedIndex].textContent === t; }",
            arg=showing, timeout=10000)
        assert page.locator(".results-ladder-back").count() == 1
        a_file = next(v for v in menu.evaluate(
            "(s) => Array.from(s.options).map((o) => o.value)") if v)
        menu.select_option(a_file)
        page.wait_for_function(
            "() => !document.querySelector('.results-ladder-back')",
            timeout=10000)
        assert menu.input_value() == a_file
        assert menu.locator("option[data-picked-run]").count() == 0

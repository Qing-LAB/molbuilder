"""A SIESTA run that SIESTA stops, made on the road, and what each reader says
about it.

``jobset init`` -> ``prep task --stage coarse`` -> ``launch task --mode direct`` on an
H2 relaxation whose SCF cannot converge within its cap: ten cycles of plain
linear mixing at weight 0.001 against a DM tolerance of 1e-8, set in the
calculation's template, with ``SCF.MustConverge`` left at SIESTA's default --
abort.  SIESTA prints
``SCF_NOT_CONV: ... (required)`` and ``die``s, and ``die`` writes its own
cascade after it; the wrapper retries once, warm, as run 1 (`job-contracts.md`
§ 2.6), and run 1 stops the same way.

What each reader must say (`running-a-job.md` § 4.2, `model/parse.md` § 2b):
the folder is ``failed``, its detail quoting the line that stopped it -- the
SCF's, not the cascade's; the viewer's stop reason is that cause in the
table's words; and the run's session log is found as the run's.  The Results
tab's Run panel (`web/results.md` § 3a) says the same, from the run's record,
in a browser.  And what molbuilder does with a run that failed
(`job-system.md` § 5.4): the stage after it is refused, saying so, and
`status` says what launching it again does.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, env_bin

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONDA_SH = conda_hook()

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (CONDA_SH.is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

#: The calculation's own values: an SCF that cannot reach 1e-8 in ten cycles
#: of plain linear mixing at 0.001 -- no Pulay history, which otherwise
#: extrapolates past the small weight and converges H2 in nine (measured) --
#: each inside its item's declared range.
_CANNOT_CONVERGE = {"max_scf_iter": "10", "mixing_weight": "0.001",
                    "pulay_history": "0", "dm_tolerance": "1e-08"}


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, list(args))


def _set_item(template: Path, name: str, value: str) -> None:
    """``[item.<name>]``'s ``value``, as a person sets it in the template."""
    text = template.read_text()
    head, sep, tail = text.partition(f"[item.{name}]")
    assert sep, f"no [item.{name}] in {template.name}"
    body, nxt, rest = tail.partition("\n[item.")
    lines = body.split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith("value = "))
    lines[at] = f"value = {value}"
    template.write_text(head + sep + "\n".join(lines) + nxt + rest)


@pytest.fixture(scope="module")
def stopped(isolated_projects_root_module, tmp_path_factory):
    """The attempt directory of the stopped run, made on the road."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    tree = isolated_projects_root_module
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    shutil.copy(FIXTURES / "psml" / "H.psml",
                tree / "pseudopotential" / "H.psml")
    StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]]),
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3),
        tree / "P" / "structure" / "h2.xyz")

    mp = pytest.MonkeyPatch()
    try:
        # THE SUITE'S CONFIG RULE, which its autouse fixture applies per test
        # and so not to a module's fixture: no read of the developer's
        # config directory (`conftest.config_root_is_never_the_developers`).
        mp.delenv("MOLBUILDER_CONFIG_DIR", raising=False)
        mp.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
        # ...and the box probed, as a real one is before its first prep.
        # ...its record saying how a shell enters conda here -- the
        # activation the generator reads (`configuration.md` § 4).
        from conftest import write_machine_record
        write_machine_record(env_init={
            "activation": "conda activate", "preamble": f"source {CONDA_SH}"})
        mp.chdir(tree.parent)
        # The wrapper finds the engine BY NAME after activating the env; the
        # env's own bin goes ahead of the suite's stub toolchain so the real
        # engine is the one on the road.
        bin_ = env_bin("molbuilder-siesta")
        assert (bin_ / "siesta").is_file(), bin_
        mp.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")

        r = _jobset("init", "--structure", "P/structure/h2.xyz",
                    "--bundle", "P/opt/R", "--engine", "siesta",
                    "--shape", "hierarchical", "--calculation", "optimization",
                    "--name", "H2", "--psml-lib", "pseudopotential",
                    "--stage-strategy", "publishable")
        assert r.exit_code == 0, r.output
        bundle = tree / "P" / "opt" / "R"
        task = json.loads((bundle / "task.json").read_text())
        task["execution"] = {**task.get("execution", {}), "mpi_np": 1,
                             "omp_threads": 1}
        assert [s["name"] for s in task["stages"]] == ["coarse", "medium"], task
        (bundle / "task.json").write_text(json.dumps(task, indent=2))
        for name, value in _CANNOT_CONVERGE.items():
            _set_item(bundle / "H2.template.toml", name, value)

        r = _jobset("prep", "task", "--stage", "coarse", "--bundle", str(bundle),
                    "--target", "this")
        assert r.exit_code == 0, r.output
        attempt = bundle / "01_coarse" / "run-0"
        deck = (attempt / "H2_01_coarse.fdf").read_text()
        assert any(ln.split()[:2] == ["MaxSCFIterations", "10"]
                   for ln in deck.splitlines()), deck[-2000:]
        # The run fails -- that is its point -- so the verb's exit is not
        # asserted; what it left is.
        _jobset("launch", "task", "--stage", "coarse", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
        assert (attempt / "H2_01_coarse-run0.out").is_file(), \
            sorted(p.name for p in attempt.iterdir())
        yield attempt
    finally:
        mp.undo()


def test_the_folder_is_failed_and_says_what_stopped_it(stopped):
    """The folder's status quotes the line that stopped the run -- the SCF's,
    which SIESTA states as fatal, and not the ``die`` lines after it."""
    from molbuilder.parse.dirs import run_status
    from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER

    st = run_status(stopped, "H2_01_coarse")    # the run's stem
    assert st.state == "failed", st
    assert st.detail.startswith("stopped before its end: SCF_NOT_CONV"), st
    # the latest run speaks -- the warm retry's -- and the one before it
    # stopped the same way
    assert st.active_source == "H2_01_coarse-run1.out", st
    for name in ("H2_01_coarse-run0.out", "H2_01_coarse-run1.out"):
        end = st.endings[name]
        assert (end.run_state, end.cause) == ("stopped",
                                              SCF_NOT_CONV_MARKER), end


@pytest.fixture(scope="module")
def after_the_stop(stopped):
    """What molbuilder answers about the stopped calculation: the next
    stage's prep, and `status`."""
    bundle = str(stopped.parent.parent)
    return {"prep medium": _jobset("prep", "task", "--stage", "medium",
                                   "--bundle", bundle, "--target", "this"),
            "status": _jobset("status", "--bundle", bundle)}


def test_a_run_that_failed_is_never_built_on(after_the_stop):
    """The stage after it is refused, naming the run and how it ended
    (`job-system.md` § 5.4: the newest run, which must have finished)."""
    r = after_the_stop["prep medium"]
    assert r.exit_code != 0, r.output
    assert "01_coarse/run-0, which failed -- " in r.output, r.output


def test_status_says_what_launching_it_again_does(after_the_stop):
    """However it ended, a prepared stage is launched again -- warm, from
    its own latest run, for a relaxation -- and `status` says so."""
    r = after_the_stop["status"]
    assert r.exit_code == 0, r.output
    for words in ("coarse, failed",
                  "launch it again -- it continues from its own latest run"):
        assert words in r.output, (words, r.output)


def test_the_retry_hears_an_scf_stop_and_no_capped_relaxation(stopped):
    """What the wrapper's two retries ask of a run (`running-a-job.md`
    § 3.5), answered by the ending reader on this run's own outputs: each
    stopped on the SCF -- the retriable case on the non-zero branch -- and
    neither is a relaxation out of moves."""
    from molbuilder.parse.engines._run_ending import QUESTIONS, ending_of
    from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER
    for name in ("H2_01_coarse-run0.out", "H2_01_coarse-run1.out"):
        end = ending_of(stopped / name)
        assert QUESTIONS["stopped-by"](end, SCF_NOT_CONV_MARKER), (name, end)
        assert not QUESTIONS["relaxation-capped"](end), (name, end)


def test_the_viewer_says_why_in_the_ending_readers_words(stopped,
                                                         monkeypatch):
    """The viewer's "Reason:" line is the server's: the file's cause as the
    one ending reader states it, in the SIESTA family's words
    (`siesta_grammar.CAUSE_WORDS`) -- the browser keeps no copy of the
    markers.

    MUTATION THIS MUST FAIL AGAINST: take the LAST fatal line as the cause.
    """
    from molbuilder import diagnostics
    from molbuilder.diagnostics import Capabilities
    from molbuilder.parse.engines.siesta_grammar import (CAUSE_WORDS,
                                                         SCF_NOT_CONV_MARKER)
    from molbuilder.web.app import create_app

    root = next(p for p in stopped.parents if p.name == "projects")

    class _Root(Capabilities):
        def file_picker_roots(self):  # type: ignore[override]
            return ((root.resolve(), "road"),)

    app = create_app(config={})
    diagnostics.set_capabilities(_Root())
    out = stopped / "H2_01_coarse-run0.out"
    body = app.test_client().post("/api/watch/load",
                                  json={"path": str(out)}).get_json()
    assert body["ok"] is True, body
    assert body["data"]["run_state"] == "stopped"
    assert body["data"]["stop_reason"] == CAUSE_WORDS[SCF_NOT_CONV_MARKER]


def test_each_run_is_paired_with_its_own_session_log(stopped):
    """The warm retry re-execs the wrapper with its output still going to
    the first log and opens a log of its own, so the first log holds run 0
    AND run 1's section.  A run's log is the one whose FIRST section is that
    run -- the pairing the run record and ``run_status`` both use
    (`wrapper_log.logs_by_run`)."""
    from molbuilder.wrapper_log import first_run_index, log_of_run

    first = log_of_run(stopped, "H2", 0, "01_coarse")
    second = log_of_run(stopped, "H2", 1, "01_coarse")
    assert first is not None and second is not None and first != second, (
        sorted(p.name for p in stopped.iterdir()))
    assert (first_run_index(first), first_run_index(second)) == (0, 1)
    assert "run index: 1" in first.read_text(errors="replace"), (
        "the retry's section did not reach the first log, so this run does "
        "not show why the FIRST section decides")
    assert log_of_run(stopped, "H2", 2, "01_coarse") is None
    assert log_of_run(stopped, "H2", 0, None) is None


def test_the_setup_tells_a_key_the_engine_read_alone_from_the_decks(stopped):
    """`engine_only` names what the engine read and nobody wrote
    (`model/parse.md` § 5d.3).  SIESTA reads some keys twice, each time its
    own default -- `MD.FinalTimeStep` and `DM.NumberPulay` in this run's fdf
    log -- and such a key is not the deck's; `MeshCutoff`, which the deck
    sets, is.

    MUTATION THIS MUST FAIL AGAINST: `in_deck` from a top-level `default`
    alone -- a key read several times has none.
    """
    from molbuilder.parse import detect
    from molbuilder.parse.dirs.setup import _engine_only

    log = sorted(stopped.glob("fdf.*.log"))[0]
    res = detect(log).parse(log)
    rows = {r["key"].lower(): r for r in _engine_only(res.params, set())}
    for key in ("md.finaltimestep", "dm.numberpulay"):
        assert len(rows[key]["readings"]) == 2, rows[key]
        assert rows[key]["in_deck"] is False, rows[key]
    assert rows["meshcutoff"]["in_deck"] is True, rows["meshcutoff"]


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def test_the_run_panel_says_what_ran_and_why_it_stopped(stopped, page,
                                                         flask_server,
                                                         monkeypatch):
    """The Results tab's Run panel (`web/results.md` § 3a) reads this run's
    record: closed, one line -- the engine, the ranks, how the latest run
    ended; open, the verdict, the setup's three columns, the computation and
    the deck, which opens in the sidebar's viewer.  It is hidden for a
    container, and from the moment the panel is bound to another folder --
    a scan that fails announces nothing, so the old record must not stay.

    MUTATION THIS MUST FAIL AGAINST: the picker not carrying ``record`` in
    its selection event; the panel not hiding when it is re-bound.
    """
    from molbuilder import diagnostics

    root = next(p for p in stopped.parents if p.name == "projects")
    bundle = stopped.parent.parent
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((root.resolve(), "road"),))
    page.add_init_script(
        "try { sessionStorage.setItem('molbuilder.current_dir', "
        f"{json.dumps(str(stopped))}); }} catch (_) {{}}")
    page.goto(f"{flask_server}/results")
    panel = page.locator("#results-run-panel")
    panel.locator(".rp-summary").wait_for(timeout=20000)
    record = page.evaluate(
        "(d) => fetch('/api/results/dir?path=' + encodeURIComponent(d))"
        ".then(r => r.json()).then(b => b.record)", str(stopped))

    line = panel.locator(".rp-summary").inner_text()
    engine = record["computation"]["engine"]
    assert f"{engine['program']} {engine['version']}" in line, line
    assert "1 rank" in line and "run 1 stopped" in line, line
    assert "FAILED" in line.upper(), line

    panel.locator(".rp-toggle").click()
    sections = {s.locator(".rp-section-title").inner_text(): s.inner_text()
                for s in panel.locator(".rp-section").all()}
    assert list(sections) == ["Verdict", "Setup", "Computation", "Deck"]
    assert "stopped before its end: SCF_NOT_CONV" in sections["Verdict"]
    assert "run 0: stopped" in sections["Verdict"]
    assert record["computation"]["host"]["hostname"] in sections["Computation"]

    # THE SETUP'S THREE COLUMNS, as the run recorded them: the catalogue
    # default from the wrapper's block, what the deck asked, what SIESTA read.
    rows = page.evaluate(
        "() => [...document.querySelectorAll("
        "  '#results-run-panel .rp-setup tbody tr')].map(tr => ["
        "    tr.querySelector('.rp-item').textContent,"
        "    ...[...tr.children].slice(1).map(td => td.textContent)])")
    by_item = {r[0]: r[1:] for r in rows}
    stated = {r["item"]: r for r in record["setup"]["rows"]}
    default, asked, used = by_item["max_scf_iter"]
    assert (default, asked, used) == (str(stated["max_scf_iter"]["default"]),
                                      "10", "10"), by_item["max_scf_iter"]
    for item, value in _CANNOT_CONVERGE.items():
        _d, asked, used = by_item[item]
        assert float(asked) == float(value) == float(used.split()[0]), (
            item, by_item[item])
    unset = [r["item"] for r in record["setup"]["rows"] if "asked" not in r]
    assert unset and all(by_item[i][1] == "engine default" for i in unset)
    assert (f"({len(record['setup']['engine_only'])})"
            in panel.locator(".rp-fold > summary").inner_text())
    assert "H.psml" in sections["Setup"]

    panel.locator(".rp-view").click()
    preview = page.locator("#ps-preview-modal")
    preview.wait_for(state="visible", timeout=10000)
    assert record["deck"]["path"] in preview.inner_text()
    page.keyboard.press("Escape")

    def rebind(folder):
        page.evaluate("(d) => window.molbuilder.projects.setShared(d, '')",
                      str(folder))
        page.click("#results-file-picker-refresh")

    # A folder whose scan fails: nothing is announced, and the panel must
    # not keep the run above under it.  Two levels that do not exist: the
    # door answers a path that is not a directory for its parent, so one
    # missing level would be answered as the stage container.
    rebind(stopped.parent / "no-such" / "attempt")
    panel.wait_for(state="hidden", timeout=10000)
    rebind(stopped)
    panel.locator(".rp-summary").wait_for(timeout=20000)
    # The calculation root is a container: it has a ladder, not a record.
    rebind(bundle)
    page.locator(".results-ladder").wait_for(state="visible", timeout=20000)
    assert panel.is_hidden()

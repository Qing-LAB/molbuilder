"""The inspector registry (`web/presenters.md`) on a live ``/results``
page, where alone it exists -- in JavaScript, with the presenters the page
registered, in the order it registered them:

  * which viewer opens a file, which lists it in the dropdown, and under
    which heading -- each row of ``tests/data/presenters.toml``;
  * opening a viewer from the Result file menu again and again leaves on
    the page what once left -- a spectrum put away while its mode box is
    still being built among them;
  * a viewer whose file cannot load says why, and lets the tab's cover and
    the menu's busy line go.
"""
from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from _road import conda_hook, env_available
from support.sidebar import WAIT_MS

#: THE ROWS ARE ASKED OF REAL RUNS (`support/real_runs.py`): SIESTA's and
#: PySCF's.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")
             and env_available("molbuilder-pySCF")),
        reason="needs the molbuilder-siesta and molbuilder-pySCF envs + a "
               "detectable conda hook"),
]

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
PRESENTERS = tomllib.loads(
    (ROOT / "tests" / "data" / "presenters.toml").read_text())


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


# --------------------------------------------------------------------- #
#  Which viewer opens a file -- in the order the page registered them    #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def described_transport(real_bridged_chain, tmp_path_factory):
    """A transport calculation, described on the road (`support.road`) on
    the bridged chain a real relaxation saved as a pair -- what a transport
    calculation cites (`engines/transport.md` § 3.1): its description is
    what the report rows ask about.  No transport ladder runs in a session
    fixture, so its record (`.transport.json`, a finished ladder's) is not
    here."""
    from support.real_runs import RealRun
    from support.road import describe_calculation
    with pytest.MonkeyPatch.context() as local:
        bundle = describe_calculation(tmp_path_factory.mktemp("transport"),
                                      local, calculation="transport",
                                      structure=real_bridged_chain.pair)
    return RealRun(tree=bundle.parents[2], bundle=bundle)


@pytest.fixture(scope="module")
def real_runs(real_h2_layered, real_h2_pyscf, real_co2_vibration,
              real_h2_sweep, real_h2_bench, described_transport):
    """THE RUNS A ROW'S ANSWER IS ASKED OF, by the name its ``answer``
    gives: each the session's own, made once a pass whoever asks first
    (`tests/conftest.py`, `support/real_runs.py`)."""
    return {"siesta": real_h2_layered, "pyscf": real_h2_pyscf,
            "vibration": real_co2_vibration, "sweep": real_h2_sweep,
            "bench": real_h2_bench, "transport": described_transport}


def _folders(run):
    """``run``'s calculation, every folder in it, and its tree's structure
    folder -- the person's folders: the checkpoint's own (`.git`,
    `.binsnapshots`) are its machinery."""
    return sorted({run.bundle, run.tree / "P" / "structure",
                   *(d for d in run.bundle.rglob("*") if d.is_dir()
                     and not any(part.startswith(".") for part in
                                 d.relative_to(run.bundle).parts))})


def _answers(page, tree, folders):
    """What the server answers of every file in ``folders`` -- each asked
    of the route the Results tab asks (`/api/results/dir`), on the live
    server, the projects root pointed at ``tree``; each file's record
    composed by the page's own picker as the dropdown composes it
    (`filterToResultFiles`, `metaOf`): ``[[folder, file, meta]]``, the page
    on `/results`."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    with pytest.MonkeyPatch.context() as local:
        local.setenv(PROJECTS_ROOT_ENV, str(tree))
        got = page.evaluate("""async (folders) => {
            const fp = window.molbuilder.resultsFilePicker;
            const out = [], refused = [];
            for (const d of folders) {
                const r = await fetch("/api/results/dir?path="
                                      + encodeURIComponent(d));
                const body = await r.json();
                if (!body.ok) { refused.push([d, body.error]); continue; }
                for (const rec of fp.filterToResultFiles(body.files, body,
                                                         null))
                    out.push([d, rec.path, fp.metaOf(rec)]);
            }
            return {out, refused}; }""", [str(d) for d in folders])
    assert not got["refused"], got["refused"]
    return got["out"]


def _answer(answers, run, want):
    """The first file of ``answers`` (:func:`_answers`) the row's
    ``answer`` names -- by its ``role``, else its ``parser``, in its
    ``folder`` (from the run's tree) when it names one: ``(file, meta)``."""
    folder = str(run.tree / want["folder"]) if "folder" in want else None
    for d, path, meta in answers:
        if folder is not None and d != folder:
            continue
        if ("role" in want and meta["role"] == want["role"]) or (
                "role" not in want and not meta["role"]
                and meta["parser"] == want["parser"]):
            return path, meta
    raise AssertionError(f"no file of the {run.bundle.name} run is answered "
                         f"as {want}")


def _copied(page, file, where, tmp_path_factory):
    """``file`` copied, as a person copies one, into ``where`` -- a folder
    of a projects tree of its own that no calculation claims -- and what the
    server answers of the copy there (:func:`_answers`): ``(file, meta)``."""
    tree = tmp_path_factory.mktemp("copied") / "projects"
    folder = tree / where
    folder.mkdir(parents=True)
    copy = Path(shutil.copy2(file, folder))
    (got,) = [(f, meta) for _d, f, meta in _answers(page, tree, [folder])
              if f == str(copy)]
    return got


@pytest.fixture(scope="module")
def registry_answers(browser, flask_server, real_runs, tmp_path_factory):
    """Every row of `tests/data/presenters.toml`, asked of the registry on
    the live `/results` page in one visit, once every presenter the rows
    name has registered: ``{table: {row name: answer}}``.

    A row naming an ``answer`` is asked of a real file with what the
    server says of it -- the route's answer for its folder, composed by the
    page as the dropdown passes it (:func:`_answers`), or for a copy of it
    in a folder of the person's own (:func:`_copied`); a row naming none is
    asked with no answer, as a file no folder answers for is.  Two rows
    alone carry a typed ``meta``, each field of it checked to be one the
    server can give (`test_presenters_rows.py`): the transport record's,
    `summarize task`'s on a finished transport ladder, which no session
    fixture runs (the junction module,
    `test_transport_on_a_real_junction_e2e.py`, runs one for itself, some
    800 s)."""
    page = browser.new_page()
    try:
        page.goto(f"{flask_server}/results")
        named = sorted({r[k] for tbl, k in (("pick", "opens"),
                                             ("result", "lists"))
                        for r in PRESENTERS[tbl] if k in r})
        page.wait_for_function(
            "(names) => { const reg = (window.molbuilder || {}).inspectors;"
            "  return !!reg && names.every(n =>"
            "    reg.list().some(i => i.name === n))"
            "  && !!(window.molbuilder.resultsFilePicker || {}).metaOf; }",
            arg=named, timeout=10000)
        answers = {}
        asked = {}
        for tbl in ("pick", "result", "heading"):
            asked[tbl] = []
            for r in PRESENTERS[tbl]:
                file, meta = r.get("file", ""), r.get("meta")
                if "answer" in r:
                    want = r["answer"]
                    run = real_runs[want["run"]]
                    if want["run"] not in answers:
                        answers[want["run"]] = _answers(page, run.tree,
                                                        _folders(run))
                    file, meta = _answer(answers[want["run"]], run, want)
                    if "copied_to" in want:
                        file, meta = _copied(page, file, want["copied_to"],
                                             tmp_path_factory)
                asked[tbl].append({"name": r["name"], "file": file,
                                   "meta": meta})
        return page.evaluate("""(t) => {
            const reg = window.molbuilder.inspectors;
            const name = (p) => p ? p.name : null;
            const out = { pick: {}, result: {}, heading: {} };
            for (const r of t.pick)
                out.pick[r.name] = name(reg.pick(r.file, r.meta || null));
            for (const r of t.result)
                out.result[r.name] = name(reg.pickResult(r.file, r.meta || null));
            for (const r of t.heading) {
                const p = reg.pickResult(r.file, r.meta || null);
                out.heading[r.name] = p ? p.resultCategory(r.file, r.meta || null)
                                        : null;
            }
            return out; }""", asked)
    finally:
        page.close()


def _rows(table):
    return pytest.mark.parametrize(
        "row", PRESENTERS[table], ids=[r["name"] for r in PRESENTERS[table]])


@_rows("pick")
def test_a_file_opens_in_the_viewer_its_kind_names(registry_answers, row):
    """`presenters.md` § 1, § 2: the first presenter to claim a file, in
    the order the PAGE registered them, opens it.

    THE SILENT FAILURE (2026-08-05): one presenter script in another queue
    -- a module among plain scripts -- ran after the text pane, and
    `.spectra.json` files opened as text; every presenter still matched and
    mounted correctly on its own.

    MUTATION THIS MUST FAIL AGAINST: the text pane's script left out of
    the deferred queue, so it registers first."""
    assert registry_answers["pick"][row["name"]] == row.get("opens")


@_rows("result")
def test_the_dropdown_lists_a_file_by_the_viewer_its_kind_names(
        registry_answers, row):
    """`presenters.md` § 1: the dropdown lists what a RESULT presenter
    claims -- never the inputs and logs the text pane takes."""
    assert registry_answers["result"][row["name"]] == row.get("lists")


@_rows("heading")
def test_the_dropdown_files_it_under_its_heading(registry_answers, row):
    """`presenters.md` § 1-§ 2 (2.65.0): a run's heading is the folder
    answer's words -- the engine's and the calculation's noun, *SIESTA
    optimization*, *SIESTA vibration* -- never guessed from a suffix; a
    run's file copied into a folder of the person's own is no run's there,
    so no run viewer lists it under any heading; the spelling is what the
    person reads.

    THE SILENT FAILURES: a heading from the page's own guess -- SIESTA runs
    filed under *PySCF optimization* while the plot title said SIESTA
    (2026-09-04); every run read *... optimization* until 2.65.0.

    MUTATION THIS MUST FAIL AGAINST: the trajectory presenter's
    `resultCategory` reading ``meta.engine`` or a fixed noun instead of the
    served ``engine_words`` and ``noun``."""
    assert registry_answers["heading"][row["name"]] == row.get("heading")




# --------------------------------------------------------------------- #
#  What a viewer leaves on the page, and how a failed load ends         #
# --------------------------------------------------------------------- #

#: The Result file menu, and the bar's line, busy through a scan and the
#: load it starts (`file-picker.js`).
_MENU = "#results-file-picker-select"
_BUSY = "#results-file-picker-meta.is-busy"


def _bound(page, base, folder):
    """The Results tab bound to ``folder`` as a person binds it -- the
    sidebar walked there, *Reload from current project dir* pressed
    (`support.sidebar.results_on`, `web/results.md` § 2.1) -- and the scan
    ended, with any load it started: the menu lists the folder's files, the
    bar's line no longer busy."""
    from support.sidebar import results_on
    results_on(page, base, folder)
    page.wait_for_function(
        "(a) => [...document.querySelectorAll(a.menu + ' option')]"
        "        .some(o => o.value.startsWith(a.dir + '/'))",
        arg={"menu": _MENU, "dir": str(folder)}, timeout=WAIT_MS)
    page.wait_for_selector(_BUSY, state="detached", timeout=WAIT_MS)


def _listed(page):
    """The files the menu lists, in its order."""
    return page.eval_on_selector_all(
        f"{_MENU} option", "os => os.map(o => o.value).filter(Boolean)")


def _opened(page, path, viewer, drawn=None):
    """``path`` opened from the menu as a person opens it -- picked, or,
    when it is the file on screen, opened again by Reload -- and the load it
    starts waited out: ``viewer``'s word that its load ended, carrying no
    error, read from ``window.__ended`` (the page's words, kept by the
    test); the bar's line no longer busy; and ``drawn``, when named, on the
    page."""
    ended = page.evaluate("() => window.__ended.length")
    if page.locator(_MENU).input_value() == path:
        page.get_by_role("button",
                         name="Reload from current project dir").click()
    else:
        page.select_option(_MENU, path)
    page.wait_for_function(
        "(a) => window.__ended.slice(a.n).some(d =>"
        "  d.inspector === a.viewer || d.error || d.unchanged)",
        arg={"n": ended, "viewer": viewer}, timeout=WAIT_MS)
    said = page.evaluate("(n) => window.__ended.slice(n)", ended)
    assert any(d.get("inspector") == viewer for d in said) and not any(
        d.get("error") or d.get("unchanged") for d in said), (path, said)
    page.wait_for_selector(_BUSY, state="detached", timeout=WAIT_MS)
    if drawn:
        page.wait_for_selector(drawn, state="attached", timeout=WAIT_MS)


def _ended_kept(page):
    """The test's own count of the viewers' word that a load ended
    (``window.__ended``, read by :func:`_opened`), kept from now on."""
    page.evaluate("""() => {
        window.__ended = [];
        document.addEventListener(
            window.molbuilder.constants.EVENT_INSPECTOR_READY,
            (e) => window.__ended.push((e && e.detail) || {}));
    }""")


def _again_and_again(page, base, tree, folder, files, viewer, drawn=None):
    """What opening ``files`` in ``viewer`` five rounds left on the page
    that one round did not -- the listeners on the window, the document or
    an element still on the page, paired add to remove as the browser pairs
    them, and the live GL contexts (`support.leftovers.what_grew`) --
    where a file opened puts the viewer showing the last one away, its
    timers, listeners and drawing, BEFORE it mounts the next
    (`presenters.md` § 3, step 2; `molview.md` § 8, § 9.9a).

    ``folder``, in the projects ``tree``, bound as a person binds it
    (:func:`_bound`), and ``drawn``, what the viewer draws after its word,
    when named, on the page; a round opens each of ``files``, the one on
    screen, if any, last -- each picked in the menu, and the one on screen,
    where no other is named, opened again by Reload, which is the clean
    reload a file switch runs (`web/results.md` § 4) -- each open waited out
    by the viewer's own word that its load ended, carrying no error, and
    then by ``drawn``.

    THE SILENT FAILURES: a listener a mount adds and its dispose leaves,
    firing against a dead panel on every later mount (2026-08-23: the
    shared scope took the registrations while each core drained an old
    array of its own, 19 left a mount); a drawing window made for every
    file opened, each left alive with its listeners for the page's life
    (2026-10-15: twenty of twenty); the tab mounting the next viewer without
    putting the last one away.

    MUTATIONS EACH CALLER MUST FAIL AGAINST: a listener its viewer adds
    outside the viewers' shared scope; a drawing window made for every
    mount; the tab mounting the next viewer without putting the last one
    away."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.leftovers import what_grew
    with pytest.MonkeyPatch.context() as local:
        local.setenv(PROJECTS_ROOT_ENV, str(tree))
        _bound(page, base, folder)
        if drawn:
            page.wait_for_selector(drawn, state="attached", timeout=WAIT_MS)
        shown = page.locator(_MENU).input_value()
        rounds = sorted(map(str, files), key=lambda f: f == shown)
        _ended_kept(page)

        def a_round():
            for path in rounds:
                _opened(page, path, viewer, drawn)

        return what_grew(page, a_round)


def test_opening_the_structure_viewer_again_and_again_leaves_what_once_left(
        page, flask_server, isolated_projects_root):
    """Opening the structure viewer again and again leaves on the page what
    once left (`presenters.md` § 3, step 2; :func:`_again_and_again`): two
    structures in a folder of the person's own -- a geometry format a
    person brings, which no run makes -- opened in turn."""
    tree = isolated_projects_root.resolve()
    folder = tree / "P" / "two"
    folder.mkdir(parents=True)
    (folder / "water.xyz").write_text(
        "3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
    (folder / "co2.xyz").write_text(
        "3\nco2\nC 0 0 0\nO 1.16 0 0\nO -1.16 0 0\n")
    grew = _again_and_again(page, flask_server, tree, folder,
                            [folder / "water.xyz", folder / "co2.xyz"],
                            "structure")
    assert not grew, ("opening the structure viewer five rounds left more "
                      "on the page than one:\n  " + "\n  ".join(grew))


def test_opening_the_trajectory_viewer_again_and_again_leaves_what_once_left(
        page, flask_server, real_h2_layered):
    """Opening the trajectory viewer again and again leaves on the page
    what once left (`presenters.md` § 3, step 2; :func:`_again_and_again`):
    the layered H2 relaxation's first `coarse` run, made on the road
    (`support.real_runs.layered`) -- its progress log and SIESTA's output,
    each the trajectory viewer's, named by their roles -- opened in turn."""
    from molbuilder.runfiles import ROLE_MOLWATCH
    from molbuilder.runs import run_of
    folder = real_h2_layered.attempt("coarse")
    run = run_of(folder)
    grew = _again_and_again(page, flask_server, real_h2_layered.tree, folder,
                            [run.file(ROLE_MOLWATCH), run.stdout],
                            "trajectory")
    assert not grew, ("opening the trajectory viewer five rounds left more "
                      "on the page than one:\n  " + "\n  ".join(grew))


def test_opening_the_spectra_viewer_again_and_again_leaves_what_once_left(
        page, flask_server, real_co2_vibration):
    """Opening the spectra viewer again and again leaves on the page what
    once left (`presenters.md` § 3, step 2; :func:`_again_and_again`): the
    CO2 vibration's run (`support.real_runs.vibration`) -- its one result,
    the spectrum -- opened again by Reload.  The spectrum's mode viewer is
    built once the spectrum is drawn (`spectra/core.js` `_showMode`), and is
    done when it shows its mode, captioned."""
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    folder = real_co2_vibration.attempt("freq")
    grew = _again_and_again(page, flask_server, real_co2_vibration.tree,
                            folder, [on_label_at(folder, ROLE_SPECTRA)],
                            "spectra",
                            "#mode-viewer .vibview-caption:not([hidden])")
    assert not grew, ("opening the spectra viewer five rounds left more on "
                      "the page than one:\n  " + "\n  ".join(grew))


#: The request the first mode box of a page waits on: its stylesheet,
#: which the box links and awaits before it makes its drawing window
#: (`vibrationview/_seal.js` `ensureStylesheet`) -- fetched once a page.
_BOX_SHEET = "**/static/lib/vibrationview/_style.css"


def test_a_spectrum_put_away_while_its_mode_box_is_built_leaves_nothing(
        page, flask_server, real_water_vibration, monkeypatch):
    """A spectrum put away while its mode box is still being built leaves on
    the page what one put away after its box was built left: the box,
    arriving once its panel has gone, is released as it arrives, its drawing
    window back to the pool, never kept (`web/spectra.md` § 4, *The mode box
    belongs to one result*; `vibrationview.md` § 8).

    THE SILENT FAILURE: the box kept by a panel that is gone -- its drawing
    window never given back, its animation running, and the next spectrum
    building a window of its own.

    THE ORDER IS HELD (`process/testing.md` § 5, *An order is held*): the
    water vibration's run (`support.real_runs.water`), its spectrum and its
    relaxation's progress log in one folder, bound as a person binds it
    (:func:`_bound`), the spectrum shown -- and the one request its mode box
    waits on, the box's stylesheet (:data:`_BOX_SHEET`), held at the
    browser.  The log is picked in the menu while the box waits; once the
    log's viewer says its load ended, the sheet is let go, the server's own
    answer, and the box arrives with its panel gone -- its drawing window
    made, which the spy counts.  Then the spectrum and the log picked in
    turn, four rounds, each box built and put away; what the page holds
    after them is what it held after the first (`support.leftovers`).

    MUTATION THIS MUST FAIL AGAINST: `_showMode` keeping a box that arrives
    once its panel is put away (`spectra/core.js`, the release at
    ``IDLE``)."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.runfiles import ROLE_MOLWATCH, ROLE_SPECTRA
    from molbuilder.runs import on_label_at, run_of
    from support.leftovers import what_grew
    folder = real_water_vibration.attempt("freq")
    spectrum = str(on_label_at(folder, ROLE_SPECTRA))
    log = str(run_of(folder).file(ROLE_MOLWATCH))
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(real_water_vibration.tree))
    held = []
    page.route(_BOX_SHEET, lambda route: held.append(route))
    _bound(page, flask_server, folder)
    assert page.locator(_MENU).input_value() == spectrum, _listed(page)
    _ended_kept(page)

    def put_away_while_built():
        windows = page.evaluate("() => window.__spy.state().contexts")
        _opened(page, log, "trajectory")
        assert len(held) == 1, f"the box's stylesheet asked {len(held)} times"
        held.pop().continue_()
        page.unroute(_BOX_SHEET)
        page.wait_for_function(
            "(n) => window.__spy.state().contexts > n", arg=windows,
            timeout=WAIT_MS)

    def a_round():
        _opened(page, spectrum, "spectra")
        _opened(page, log, "trajectory")

    grew = what_grew(page, a_round, first=put_away_while_built)
    assert not grew, (
        "a spectrum put away five times, the first while its mode box was "
        "built, left more on the page than once:\n  " + "\n  ".join(grew))


def test_a_viewer_whose_file_cannot_load_says_why_and_lets_go(
        page, flask_server, real_water_vibration, monkeypatch):
    """A load that fails has ENDED, and says so twice: the viewer's own
    status line carries the reason, and the ready signal goes, carrying it
    -- so the tab's cover and the bar's busy *Parsing…* line lift at once
    instead of sitting over the error for their safety timers, 15 s and 8 s
    (`web/results.md` § 3; the Results-tab review, 2026-09-28).  The
    trajectory panel's line keeps its own layout class through the write.

    THE FILES ARE A REAL RUN'S, GONE AFTER THE MENU LISTED THEM: a copy of
    the water vibration (`support.real_runs.copy_of`), whose run holds its
    spectrum and its relaxation's progress log in one folder -- the folder
    bound as a person binds it (:func:`_bound`), both files deleted once the
    menu lists them, each then picked in the menu.

    MUTATIONS THIS MUST FAIL AGAINST: spectra's refused load with no
    signal; the one status writer replacing the line's whole class list (it
    stripped `trajectory-status` on the first write); the cover or the
    busy line not lifting on a signal that carries an error."""
    from playwright.sync_api import expect

    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.runfiles import ROLE_MOLWATCH, ROLE_SPECTRA
    from molbuilder.runs import on_label_at, run_of
    from support.real_runs import copy_of
    gone = copy_of(real_water_vibration, "vanished", lambda copy: None)
    folder = gone.attempt("freq")
    files = {"trajectory": run_of(folder).file(ROLE_MOLWATCH),
             "spectra": on_label_at(folder, ROLE_SPECTRA)}
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(gone.tree))
    _bound(page, flask_server, folder)
    assert {str(f) for f in files.values()} <= set(_listed(page)), (
        files, _listed(page))
    for path in files.values():
        path.unlink()
    _ended_kept(page)
    lines = {"trajectory": "#trajectory-status", "spectra": "#watch-status"}
    # The one on screen picked last: a pick of it would change nothing.
    shown = page.locator(_MENU).input_value()
    for name, path in sorted(files.items(), key=lambda f: str(f[1]) == shown):
        n = page.evaluate("() => window.__ended.length")
        page.select_option(_MENU, str(path))
        page.wait_for_function(
            "(a) => window.__ended.slice(a.n).some(d => d.inspector === a.name)",
            arg={"n": n, "name": name}, timeout=15000)
        said = next(d for d in page.evaluate("() => window.__ended")[n:]
                    if d.get("inspector") == name)
        assert said.get("error"), (name, said)
        line = page.locator(lines[name])
        assert line.inner_text(), (name, said)
        assert "error" in line.get_attribute("class").split(), (
            name, line.get_attribute("class"))
        if name == "trajectory":
            assert "trajectory-status" in line.get_attribute("class").split()
        expect(page.locator(".results-loading-overlay")).to_be_hidden(
            timeout=1000)
        expect(page.locator(_BUSY)).to_have_count(0, timeout=1000)

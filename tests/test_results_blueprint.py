"""Tests for the ``/results`` blueprint -- the post-merge unified
inspector page.

Architecture (``docs/web/results.md`` § 0, ``docs/web/presenters.md`` § 3):

  * The page has ONE mount point (``#inspector-host``).
  * Each file type is owned by its own inspector module under
    ``static/lib/inspectors/`` that self-registers on script load.
  * ``static/results/viewer.js`` mounts the file the picker announces:
    it asks the registry for a match and lets the inspector own the host.

These tests pin the contract that lets new inspectors slot in
without touching this file: route + sidebar + the partials + the
contract endpoint.  The modules the page names are asked of the server by
``test_wheel_ships_the_front_end.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

#: Whatever the inspector reaches for besides the elements it asks its root
#: for -- the document, the window's listeners, the network -- answered by a
#: stand-in that takes every call and says nothing, so the mount runs to its
#: end: the drawing is not what is asked here.
_ANY_DOM = """
const __any = () => new Proxy(function () {}, {
  get: (t, k) => (k === "then" ? undefined
                  : k === Symbol.toPrimitive ? (() => "")
                  : k === Symbol.iterator ? (function* () {})
                  : __any()),
  apply: () => __any(), construct: () => __any(), set: () => true,
});
globalThis.document = __any();
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.fetch = () => new Promise(() => {});
"""


# --------------------------------------------------------------------- #
#  Route + page scaffolding                                             #
# --------------------------------------------------------------------- #


class TestResultsRoute:

    def test_page_has_single_inspector_host(self, web_client):
        """The registry-driven architecture mounts every inspector
        inside ONE host element.  An id declared twice is
        `test_page_ids_unique.py`'s."""
        from support.pages import ids_of
        r = web_client.get("/results")
        assert r.status_code == 200, r.status_code
        ids = ids_of(r.get_data(as_text=True))
        assert "inspector-host" in ids
        # The fallback content lives inside the host until the
        # dispatch detaches it on first selection.
        assert "results-fallback" in ids


# --------------------------------------------------------------------- #
#  Server-rendered partials                                             #
# --------------------------------------------------------------------- #
#
# The trajectory and spectra inspectors (lib/inspectors/trajectory.js,
# spectra.js) fetch their markup from GET /partials/<name> and put it in
# the registry-supplied host (`_partial_inspector_factory.js`): if the
# endpoint fails, every click on such a file on /results fails to mount.


class TestThePartials:

    @pytest.mark.parametrize("partial", ["/partials/trajectory-inspector",
                                         "/partials/spectra-inspector"])
    def test_a_partial_is_a_fragment_the_browser_asks_about_again(
            self, web_client, partial):
        """Each partial answers, and as what the host takes:

          * ``Cache-Control: private, no-cache`` -- the browser asks before
            reusing its copy, as it does for every script: the partial and
            the scripts that query it change together, and a copy kept
            without asking put a release's scripts under the last release's
            controls (2.49.0 on Sol: the removed atoms select still on
            screen).  ``private`` so intermediaries don't keep it.
          * a FRAGMENT, not a page: no top-level ``<html>``/``<head>``/
            ``<body>`` or doctype, which would nest inside the host's
            document.  Matched on the tag (``<head`` is a prefix of the
            partials' own ``<header>``)."""
        import re
        r = web_client.get(partial)
        assert r.status_code == 200, (
            f"{partial} returned {r.status_code}; the inspector on /results "
            f"mounts on what it answers")
        cc = r.headers.get("Cache-Control", "")
        assert "private" in cc, f"missing 'private' (got {cc!r})"
        assert "no-cache" in cc and "max-age" not in cc, (
            f"expected no-cache (got {cc!r}); a copy reused without asking "
            f"outlives a release of the scripts that query it")
        body = r.get_data(as_text=True)
        for tag in ("html", "head", "body"):
            assert not re.search(rf"<{tag}[\s>/]", body, re.IGNORECASE), (
                f"{partial} carries a top-level <{tag}>; it must be a DOM "
                f"fragment the inspector puts in its host")
        assert "<!doctype" not in body.lower(), (
            f"{partial} carries a <!DOCTYPE> declaration")

    def test_every_element_the_inspector_asks_for_is_served(self, web_client):
        """THE INSPECTOR, MOUNTED ON WHAT IS SERVED, FINDS EVERY ELEMENT IT
        ASKS FOR.  `lib/spectra/core.js` mounts on two pages: its results side
        on the spectra partial (`/results`), its form side on the spectrum
        calculation page; each element it binds by id is asked of the root
        it is mounted in.  Run here under node against the partial as
        served, recording each id the mount asks for: every one is in the
        partial, or on the spectrum calculation page as served.

        THE SILENT FAILURE: an id renamed on one side only -- the inspector
        binds null, and its guarded uses (the export controls, the methods
        block, the phase dots) drop that control from the page without an
        error.  The ids are the inspector's own, asked as it mounts; this
        test holds no list of them."""
        from support.pages import ids_of
        from tests._node_esm import run_node
        lib = Path(__file__).resolve().parents[1] / "molbuilder/web/static/lib"
        partial = set(ids_of(web_client.get("/partials/spectra-inspector")
                             .get_data(as_text=True)))
        page = set(ids_of(web_client.get("/spectrum-calculation")
                          .get_data(as_text=True)))
        asked = run_node(
            [lib / "inspectors" / "lifecycle.js", lib / "spectra" / "core.js"],
            """
            const held = new Set(%s);
            const asked = [];
            const rootEl = new Proxy({}, { get: (t, k) => {
              if (k === "querySelector") return (sel) => {
                if (!sel.startsWith("#")) return __any();
                asked.push(sel.slice(1));
                return held.has(sel.slice(1)) ? __any() : null;
              };
              if (k === "querySelectorAll") return () => [];
              return k === "then" ? undefined : __any();
            }});
            window.molbuilder.spectraInspector.mount(rootEl, {});
            console.log(JSON.stringify(asked));
            """ % json.dumps(sorted(partial)), globals_js=_ANY_DOM)
        assert len(asked) > len(set(asked) - partial) > 0, asked
        assert sorted(set(asked) - partial - page) == []


@pytest.fixture
def relaxation_and_its_bench(tmp_path, monkeypatch):
    """An H2 relaxation through the road, no engine run: `molbuilder
    describe`, `prep task` of its first group, and `prep bench` of its first
    stage's benchmark -- every folder prep makes for it."""
    from support.road import (declare_bench, described, jobset, prep_groups,
                              road_tree)
    bundle = described(road_tree(tmp_path, monkeypatch))
    prep_groups(bundle, "1")
    declare_bench(bundle, {"mpi_np": [1, 2], "omp_threads": [1]})
    got = jobset("prep", "bench", "coarse", "--bundle", bundle,
                 "--target", "this")
    assert got.exit_code == 0, got.output
    return bundle


class TestTheContractEndpoint:
    """/api/results/contract: the
    contract a run's own deck states (`model/parse.md` § 5b).  Isolated onto
    a tmp projects root -- tests never touch the real tree."""

    def test_a_directory_with_no_run_reports_no_run_state(
            self, isolated_projects_root, web_client, relaxation_and_its_bench):
        """`run_status` cannot say *there is no run here*.

        Asked without a launch record, an absence of evidence comes back
        as **running, no result file yet** —
        and every consumer that asks it about a directory that is not a run
        gets that.  MEASURED 2026-09-19 on the regenerated tree: TEN of
        nineteen directories under one project reported *running*, among
        them `scan/`, `structure/`, `transport/` and the project root, none
        of which has ever held a run.

        The container half of this was closed in 6ddc551a; this is the
        other door, an UNMARKED directory (`project-layout.md` § 1.4a:
        absence narrows the answer).  Such a directory is still read alone
        — its files listed, each as the registry reads it — but nothing in
        it is claimed as a run's (`model/parse.md` § 5.0): no run state,
        and no file opened for the person, who picks one.  Its file is one
        a run of ours writes, the progress log prep seeds, copied out of a
        prepared run (:func:`relaxation_and_its_bench`) as a person copies
        one.

        THE SILENT FAILURE: the door offering the copy as the folder's
        result by its role while the copy's own row carries none, so no
        viewer claimed it and the picker dropped the door's pick without a
        word.
        """
        import shutil
        from molbuilder.runs import run_of
        run = run_of(relaxation_and_its_bench / "01_coarse" / "run-0")
        bare = isolated_projects_root / "topic"   # a folder nobody described
        bare.mkdir()
        copy = Path(shutil.copy2(run.file(".molwatch.log"), bare))
        body = web_client.get("/api/results/dir?path=" + str(bare)).get_json()
        assert body["ok"] is True
        assert body["place"]["role"] is None, "nothing marks this directory"
        assert body["status"] is None, (
            "an unmarked directory with no run must not report one; got "
            + repr(body["status"]))
        # ...and it is still READ: the listing is the part absence keeps,
        # each file as the registry reads it, none claimed as a run's.
        assert [(f["name"], f["parser"], f["role"])
                for f in body["files"]] == [(copy.name, "molwatch", None)]
        assert body["openable"] is None, (
            "a folder no calculation claims opens nothing on its own; got "
            + repr(body["openable"]))


    def test_a_saved_pair_in_a_folder_nobody_described_is_molbuilders(
            self, isolated_projects_root, web_client):
        """`web/results.md` § 3b: the file card says what each half of a
        saved structure pair holds and who saves it; a file beside it that
        molbuilder did not write stays *not written by molbuilder*.  Silent
        before 2026-10-09: a pair the Molbuilder tab saved read as not
        molbuilder's (plan § 5z.1)."""
        import numpy as np
        from molbuilder.runfiles import SAVED_PAIR_WHAT
        from molbuilder.structure import Structure
        from molbuilder.workingcopy_structure import StructureCodec
        shelf = isolated_projects_root / "P" / "structure"
        shelf.mkdir(parents=True)
        StructureCodec().write(
            Structure(elements=["H", "H"],
                      positions=np.array([[0, 0, 0], [0, 0, 0.74]], float),
                      regions={"bridge": [0, 1]},
                      info={"note": "a pair with its labels and record"}),
            shelf / "h2.xyz")
        (shelf / "notes.txt").write_text("mine\n")
        body = web_client.get("/api/results/dir?path=" + str(shelf)).get_json()
        about = {f["name"]: f["about"] for f in body["files"]}
        assert about["h2.xyz"]["ours"] is True, about
        assert about["h2.xyz"]["what"] == SAVED_PAIR_WHAT["structure"]
        assert about["h2.molstruct.json"]["what"] == SAVED_PAIR_WHAT["sidecar"]
        assert about["notes.txt"] == {"ours": False}, about

    def test_a_calculations_folders_are_answered_by_the_run_door(
            self, web_client, relaxation_and_its_bench):
        """Through the road (:func:`relaxation_and_its_bench`), the run door
        answers each folder as what it IS
        (`project-layout.md` § 1.4a; `execution/architecture.md` § 3.2):

        * the calculation's ROOT is a container: no run state, no record --
          a container is not a run, and inventing one is what made a
          `pseudos/` folder report *running* (6ddc551a) -- and its ladder;
        * a PREPPED stage's attempt is a run before it has written a byte:
          ``pending``, prepared and never launched, as its missing launch
          record says (§ 1.6) -- not *not mine*, which would make every
          consumer asking about a rung before it runs raise;
        * each file says what it is: the attempt's deck is the catalogue's,
          its launch record not there yet;
        * a BENCHMARK TRIAL's folder is a run of ours too, pending: prep
          marks it as it marks a stage's attempt (§ 1.4a).

        *(The root's own PRODUCT -- a transport calculation's I–V record --
        is transport's own case, on the minimal junction (plan Q5–Q7).)*
        """
        bundle = relaxation_and_its_bench
        body = web_client.get(
            "/api/results/dir?path=" + str(bundle)).get_json()
        assert body["status"] is None and body["record"] is None, (
            "a container is not a run")
        assert body["ladder"] is not None, "a calculation root has a ladder"
        # ...and under each rung every run, with what it read and what was
        # checked before it was written (`web/results.md` § 2.4): the deck in
        # the run's folder, the `.validation.txt` beside the deck prep wrote,
        # in the stage's folder -- the run's with its `run-<n>` level out --
        # each path whole, as the server resolves the folder (§ 0.3).
        coarse = next(s for s in body["ladder"]["stages"]
                      if s["name"] == "coarse")
        whole = bundle.resolve()
        assert [(r["deck"], r["validation"]) for r in coarse["runs"]] == [
            (str(whole / "01_coarse/run-0/H2_01_coarse.fdf"),
             str(whole / "01_coarse/H2_01_coarse.validation.txt"))], (
            coarse["runs"])

        attempt = bundle / "01_coarse" / "run-0"
        body = web_client.get(
            "/api/results/dir?path=" + str(attempt)).get_json()
        assert (body["status"]["state"], body["status"]["detail"]) == (
            "pending", "prepared, not launched (no launch record)"), body["status"]
        about = {f["name"]: f["about"] for f in body["files"]}
        assert about["H2_01_coarse.fdf"]["ours"] is True, about
        assert about["calcdir.json"]["ours"] is True, about

        trial = sorted((bundle / "01_coarse" / "bench").glob("bench-*"))[0]
        body = web_client.get("/api/results/dir?path="
                              + str(trial / "run-0")).get_json()
        assert body["status"]["state"] == "pending", body["status"]

    #: WHERE EACH FOLDER PREP MAKES IS (`project-layout.md` § 1.4a, *Where am
    #: I*; invariant 6b; `web/results.md` § 2.3, `place`) -- an H2
    #: relaxation's first group and its first stage's benchmark, prepared:
    #: ``(folder, what, role, stage, run, trial)``, the folder relative to
    #: the calculation, ``bench-*`` a trial and ``*`` its point.  Every
    #: folder prep made answers one row, and every row a folder.
    EVERY_FOLDER = [
        ("", "calculation", "container", None, None, None),
        ("pseudos", "support", "container", None, None, None),
        ("01_coarse", "stage", "container", "01_coarse", None, None),
        ("01_coarse/run-0", "attempt", "run", "01_coarse", 0, None),
        ("01_coarse/bench", "benchmark", "container", "01_coarse", None,
         None),
        ("01_coarse/bench/bench-*", "trial", "container", "01_coarse", None,
         "*"),
        ("01_coarse/bench/bench-*/run-0", "attempt", "run", "01_coarse", 0,
         "*"),
    ]

    def test_every_folder_prep_makes_says_where_it_is(
            self, web_client, relaxation_and_its_bench):
        """Through the road (:func:`relaxation_and_its_bench`), every folder
        prep made answers
        the Results tab's door with where it is: its calculation, what it
        is, its role, and the stage, run and trial it sits in
        (`project-layout.md` § 1.4a, *Where am I*; `web/results.md` § 2.3).
        The silent failures: a folder read as another stage's or trial's has
        its files read on the wrong label, none of them molbuilder's; a
        benchmark's folder not read as one, its summary refused; a folder
        prep made with no `calcdir.json` (invariant 6b), read as *not
        marked*, its run never asked.

        MUTATION THIS MUST FAIL AGAINST: a trial read as support (the
        benchmark's child rule), an attempt's index not read, a folder an
        opener left unstamped."""
        from fnmatch import fnmatch
        bundle = relaxation_and_its_bench
        rows = self.EVERY_FOLDER
        # The folders prep made -- not the saved history's own.
        folders = [bundle] + sorted(
            p for p in bundle.rglob("*") if p.is_dir() and not any(
                part.startswith(".") for part in p.relative_to(bundle).parts))
        answered = set()
        for folder in folders:
            rel = "" if folder == bundle else folder.relative_to(
                bundle).as_posix()
            row = next((r for r in rows if fnmatch(rel, r[0])
                        and rel.count("/") == r[0].count("/")), None)
            assert row is not None, f"prep made {rel!r}, and no row says it"
            answered.add(row[0])
            pattern, what, role, stage, run, trial = row
            if trial == "*":
                trial = next(part for part in rel.split("/")
                             if part.startswith("bench-"))[len("bench-"):]
            place = web_client.get(
                "/api/results/dir?path=" + str(folder)).get_json()["place"]
            assert Path(place["calculation"]).resolve() == bundle.resolve(), (
                rel, place)
            assert {k: place[k] for k in ("what", "role", "stage", "run",
                                          "trial", "point")} == {
                "what": what, "role": role, "stage": stage, "run": run,
                "trial": trial, "point": None}, (rel, place)
        assert answered == {r[0] for r in rows}, (
            "a row no folder answered", sorted({r[0] for r in rows} - answered))


    def test_no_deck_answers_null(self, isolated_projects_root, web_client):
        d = isolated_projects_root / "bare"
        d.mkdir()
        (d / "chain.xyz").write_text("1\n\nC 0 0 0\n")
        r = web_client.get("/api/results/contract?path="
                           + str(d / "chain.xyz"))
        out = r.get_json()
        assert out["ok"] is True and out["calculation"] is None

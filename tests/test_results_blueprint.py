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


import pytest


@pytest.fixture
def web():
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


# --------------------------------------------------------------------- #
#  Route + page scaffolding                                             #
# --------------------------------------------------------------------- #


class TestResultsRoute:

    def test_route_returns_200(self, web):
        r = web.get("/results")
        assert r.status_code == 200

    def test_page_includes_projects_sidebar(self, web):
        """The sidebar is the dispatch mechanism: without it the
        user has no way to pick a file."""
        body = web.get("/results").get_data(as_text=True)
        assert 'class="projects-sidebar dock-panel"' in body
        assert 'id="projects-sidebar"' in body

    def test_page_has_single_inspector_host(self, web):
        """The registry-driven architecture mounts every inspector
        inside ONE host element."""
        body = web.get("/results").get_data(as_text=True)
        assert 'id="inspector-host"' in body
        # The fallback content lives inside the host until the
        # dispatch detaches it on first selection.
        assert 'id="results-fallback"' in body

    def test_page_has_status_header(self, web):
        body = web.get("/results").get_data(as_text=True)
        assert 'id="results-current-file"' in body
        assert 'id="results-current-kind"' in body


# --------------------------------------------------------------------- #
#  Tab nav                                                              #
# --------------------------------------------------------------------- #


class TestResultsInTabNav:
    """Results must appear in the shared app-tabs nav on every page,
    AND the /results page itself must mark its own tab active."""

    @pytest.mark.parametrize("path", ["/molbuilder",
                                       "/structure-optimization",
                                       "/spectrum-calculation",
                                       "/transport-calculation",
                                       "/results"])
    def test_results_link_present_on_every_page(self, web, path):
        body = web.get(path).get_data(as_text=True)
        assert 'href="/results"' in body

    def test_results_page_marks_itself_active(self, web):
        body = web.get("/results").get_data(as_text=True)
        import re
        m = re.search(
            r'<a[^>]*href="/results"[^>]*class="[^"]*is-active[^"]*"',
            body,
        )
        assert m, "Results tab link on /results is missing is-active"

    def test_other_pages_do_not_mark_results_active(self, web):
        import re
        for path in ("/molbuilder", "/structure-optimization",
                     "/spectrum-calculation", "/transport-calculation"):
            body = web.get(path).get_data(as_text=True)
            m = re.search(
                r'<a[^>]*href="/results"[^>]*class="[^"]*is-active[^"]*"',
                body,
            )
            assert not m, f"{path!r} incorrectly marks /results active"


# --------------------------------------------------------------------- #
#  Server-rendered partials                                             #
# --------------------------------------------------------------------- #
#
# The trajectory inspector wrapper (lib/inspectors/trajectory.js)
# fetches its DOM markup from GET /partials/trajectory-inspector and
# assigns it to the registry-supplied host's innerHTML.  That endpoint
# is now load-bearing -- if it 404s or returns the wrong shape, every
# .molwatch.log click on /results fails to mount.  These tests pin the
# wire contract.


class TestPartialTrajectoryInspectorEndpoint:
    """Pin the GET /partials/trajectory-inspector contract.

    Why each pin matters:
      * status 200 + text/html content-type: the inspector's fetch
        path assumes successful HTML; anything else stops the mount
        cold without a meaningful client-side error.
      * Cache-Control: private, no-cache -- the browser asks before
        reusing its copy, as it does for every script: the partial and
        the scripts that query it change together, and a copy kept
        without asking put a release's scripts under the last release's
        controls (2.49.0 on Sol: the removed atoms select still on
        screen).  ``private`` so intermediaries don't cache.

    Every element the trajectory viewer looks up on the panel answers
    when the panel is mounted on a real run
    (`test_siesta_run_in_the_viewer_e2e.py`).
    """

    def test_endpoint_returns_html_200(self, web):
        r = web.get("/partials/trajectory-inspector")
        assert r.status_code == 200, (
            f"/partials/trajectory-inspector returned "
            f"{r.status_code}; the trajectory inspector on /results "
            f"depends on this endpoint for its DOM"
        )

    def test_content_type_is_html(self, web):
        r = web.get("/partials/trajectory-inspector")
        ctype = r.headers.get("Content-Type", "")
        assert "text/html" in ctype, (
            f"expected text/html, got {ctype!r}; the inspector's "
            f"fetch path assumes HTML"
        )
        assert "charset=utf-8" in ctype.lower(), (
            "missing charset; non-ASCII content in the partial "
            "(none today, but defensive) could be mis-decoded"
        )

    def test_cache_control_is_private_and_revalidated(self, web):
        r = web.get("/partials/trajectory-inspector")
        cc = r.headers.get("Cache-Control", "")
        assert "private" in cc, (
            f"missing 'private' directive (got {cc!r}); the body "
            f"shouldn't be cached by intermediaries"
        )
        assert "no-cache" in cc and "max-age" not in cc, (
            f"expected no-cache (got {cc!r}); a copy reused without "
            f"asking outlives a release of the scripts that query it"
        )

    def test_partial_does_not_carry_page_chrome(self, web):
        """The partial is a FRAGMENT, not a full page.  It must not
        wrap the inspector in <html>/<head>/<body> (those would
        nest inside the host's existing document, which browsers
        treat as malformed).

        Match on the TAG, not a substring -- otherwise the regex
        catches semantic HTML5 elements like <header> as false
        positives (``<head`` is a prefix of ``<header``).  The
        match pattern is ``<tag>`` or ``<tag `` (open tag with
        attributes) -- both legal forms of the forbidden top-level
        wrappers."""
        import re
        r = web.get("/partials/trajectory-inspector")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        for forbidden_tag in ("html", "head", "body"):
            pat = re.compile(rf"<{forbidden_tag}[\s>/]", re.IGNORECASE)
            assert not pat.search(body), (
                f"partial response contains a top-level <{forbidden_tag}> "
                f"element; it must be a DOM fragment (the trajectory "
                f"inspector wrapper injects it into a registry-supplied "
                f"host)"
            )
        # Doctype is its own case (not a tag).
        assert "<!doctype" not in body.lower(), (
            "partial response carries a <!DOCTYPE> declaration"
        )

    def test_endpoint_does_NOT_require_auth_gate_for_the_test_client(self, web):
        """Sanity: the test client (created via create_app(config={}))
        has NO auth configured, so every endpoint is public.  This
        test is mostly a guardrail: if a future change accidentally
        makes /partials/* fail without auth, the registry inspector's
        fetch path breaks across the board."""
        r = web.get("/partials/trajectory-inspector")
        # No redirect to /login in the no-auth test config.
        assert r.status_code != 302, (
            "endpoint is redirecting in the no-auth test fixture; "
            "either the test fixture leaked auth config or the "
            "endpoint sprouted a per-route gate"
        )


class TestPartialSpectraInspectorEndpoint:
    """Pin the GET /partials/spectra-inspector contract.

    Mirror of TestPartialTrajectoryInspectorEndpoint -- the two
    partial endpoints share a wire contract by design so the
    registry-side wrappers can be near-identical adapters with the
    same trust boundary + caching semantics.

    Why each pin matters:
      * status 200 + text/html content-type: the spectra inspector's
        fetch path assumes successful HTML; anything else stops the
        mount cold without a meaningful client-side error.
      * Cache-Control: private, no-cache -- same rationale as the
        trajectory partial.
      * Partial-id parity: the body MUST contain every id the
        spectra inspector JS queries via $().  A drift between this
        partial and the inspector's selectors silently breaks the
        inspector on /results.  We sample the most-load-bearing ids
        from the inspect-side functions in lib/spectra/core.js.
    """

    # Hand-picked from the inspect-side surface of
    # lib/spectra/core.js -- a representative slice covering
    # every functional area (load controls, results summary, chart,
    # modes table + filter + CSV, mode viewer + animation, ES bar
    # diagram).
    REQUIRED_IDS = (
        # The run's progress: the status line and the phase dots (the
        # dropdown is the one route to a file, web/spectra.md § 7).
        "watch-status",
        "phase-indicator",
        # Thermochemistry tab (v5 `thermo`; spectra-migration-plan § 2b):
        # the tab button + panel + the two Plotly boxes + the words.
        "mode-tabbtn-thermo",
        "mode-tab-thermo",
        "thermo-note",
        "thermo-curves",
        "thermo-decomp",
        # Results summary + chart
        "results-summary",
        "results-summary-list",
        "spectrum-chart",
        "broadening-fwhm",
        # The spectrum where a strength was computed, the mode positions
        # where none was (web/spectra.md § 2): the section and its heading,
        # the controls the core hides when there are no heights, and the
        # sentence saying why.
        "spectrum-section",
        "spectrum-heading",
        "spectrum-controls",
        "spectrum-absent",
        # The intensity floor, and the methods block the spectrum is
        # reported with.  All six are live: `lib/spectra/core.js` binds
        # every one through `$()`.
        "display-floor",
        "display-floor-out",
        "methods-block",
        "methods-text",
        "methods-copy",
        "methods-copy-note",
        # Modes table
        "modes-table",
        "modes-tbody",
        "modes-thead-row",
        "modes-filter",
        "modes-filter-count",
        "modes-csv-btn",
        # The three views of one selection, as tabs under the spectrum
        # (2026-08-05).  Both halves of each tab are pinned: the button the
        # user clicks and the panel it controls, because `aria-controls` ties
        # them by id and a rename of either alone breaks the pairing silently.
        "mode-tabs",
        "mode-tabbtn-table",
        "mode-tab-table",
        "mode-tabbtn-viewer",
        "mode-tab-viewer",
        "mode-tabbtn-es",
        "mode-tab-es",
        # Mode viewer (3D animation)
        "mode-viewer-wrap",
        "mode-viewer",
        "viewer-status",
        "anim-amplitude",
        "anim-amplitude-val",
        "anim-speed",
        "anim-speed-val",
        "anim-toggle",
        # How big to draw the motion (docs/web/spectra.md § 4.1).  The two
        # ``-row`` wrappers are the seam, not decoration: the core shows and
        # hides them per amplitude mode, so a missing wrapper leaves the
        # temperature box on screen under "exaggerated", where it means nothing.
        "anim-amplitude-mode",
        "anim-amplitude-row",
        "anim-temperature",
        "anim-temperature-row",
        # Getting the animation out (vibrationview.md § 12).
        "anim-export-format",
        "anim-export-width",
        "anim-export-height",
        "anim-export-background",
        "anim-export-cycles",
        "anim-export-btn",
        "anim-export-cancel",
        "anim-export-status",
        # ES bar diagram
        "es-panel",
        "es-bar-diagram",
        "es-mode-idx",
        "es-mode-freq",
        "es-summary",
    )

    def test_endpoint_returns_html_200(self, web):
        r = web.get("/partials/spectra-inspector")
        assert r.status_code == 200, (
            f"/partials/spectra-inspector returned {r.status_code}; "
            f"the spectra inspector on /results depends on this "
            f"endpoint for its DOM"
        )

    def test_content_type_is_html(self, web):
        r = web.get("/partials/spectra-inspector")
        ctype = r.headers.get("Content-Type", "")
        assert "text/html" in ctype, f"expected text/html, got {ctype!r}"
        assert "charset=utf-8" in ctype.lower(), (
            f"missing charset; got {ctype!r}"
        )

    def test_cache_control_is_private_and_revalidated(self, web):
        r = web.get("/partials/spectra-inspector")
        cc = r.headers.get("Cache-Control", "")
        assert "private" in cc, f"missing 'private' (got {cc!r})"
        assert "no-cache" in cc and "max-age" not in cc, (
            f"expected no-cache (got {cc!r})")

    @pytest.mark.parametrize("element_id", REQUIRED_IDS)
    def test_body_carries_required_inspector_ids(self, web, element_id):
        """Every id the spectra inspector core scopes against (via $()
        or els.<key> = document.getElementById(...)) must be present
        in the partial response.  A missing id silently breaks the
        inspector on /results."""
        body = web.get("/partials/spectra-inspector").get_data(as_text=True)
        needle = f'id="{element_id}"'
        assert needle in body, (
            f"partial response is missing {needle!r}; the spectra "
            f"inspector queries this id and would crash on /results"
        )

    def test_partial_does_not_carry_page_chrome(self, web):
        """Fragment, not a full page -- no top-level <html>/<head>/
        <body>.  Match on tags, not substrings (``<head`` is a
        prefix of the legitimate HTML5 ``<header>`` element used
        for section headers within the partial)."""
        import re
        r = web.get("/partials/spectra-inspector")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        for forbidden_tag in ("html", "head", "body"):
            pat = re.compile(rf"<{forbidden_tag}[\s>/]", re.IGNORECASE)
            assert not pat.search(body), (
                f"partial response contains a top-level "
                f"<{forbidden_tag}> element"
            )
        assert "<!doctype" not in body.lower(), (
            "partial response carries a <!DOCTYPE> declaration"
        )

    def test_partial_does_NOT_carry_generate_side_ids(self, web):
        """SEMANTIC pin of the generate-vs-inspect split: the partial
        must contain ONLY inspect-side ids.  Generate-side surfaces
        (form, methods modal, generate-script buttons, script
        preview, generate-time issues) belong in spectra.html, not
        in this partial.  A regression that pulls a generate-side
        id in here would mean /results renders a non-functional
        generate UI on every trajectory click."""
        r = web.get("/partials/spectra-inspector")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        for generate_side_id in (
            "spectra-form-container",   # the form itself
            "generate-btn",             # script generation button
            "script-preview",           # script-preview <pre>
            "methods-modal",            # methods help modal
            "issues-panel",             # generate-time validation
        ):
            needle = f'id="{generate_side_id}"'
            assert needle not in body, (
                f"partial response contains {needle!r} which is a "
                f"generate-side id -- it belongs in spectra.html, "
                f"not in the inspector partial.  See § 2.3 of "
                f"docs/web/results.md for the split."
            )


class TestTheContractEndpoint:
    """/api/results/contract: the
    contract a run's own deck states (`model/parse.md` § 5b).  Isolated onto
    a tmp projects root -- tests never touch the real tree."""

    @pytest.fixture
    def isolated(self, tmp_path, monkeypatch):
        from molbuilder.projects import PROJECTS_ROOT_ENV
        root = tmp_path / "projects"
        root.mkdir()
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        from molbuilder.web.app import create_app
        return root, create_app(config={}).test_client()


    def test_a_directory_with_no_run_reports_no_run_state(self, isolated):
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
        — its files listed, one of them opened — but a run state has to be
        grounded on something: the record saying RUN, or the door finding
        this directory's product.
        """
        root, client = isolated
        bare = root / "topic"          # a folder nobody described
        bare.mkdir()
        (bare / "README.md").write_text("notes\n")
        body = client.get("/api/results/dir?path=" + str(bare)).get_json()
        assert body["ok"] is True
        assert body["place"]["role"] is None, "nothing marks this directory"
        assert body["status"] is None, (
            "an unmarked directory with no run must not report one; got "
            + repr(body["status"]))
        # ...and it is still READ: the listing is the part absence keeps.
        assert [f["name"] for f in body["files"]] == ["README.md"]


    def test_a_saved_pair_in_a_folder_nobody_described_is_molbuilders(
            self, isolated):
        """`web/results.md` § 3b: the file card says what each half of a
        saved structure pair holds and who saves it; a file beside it that
        molbuilder did not write stays *not written by molbuilder*.  Silent
        before 2026-10-09: a pair the Molbuilder tab saved read as not
        molbuilder's (plan § 5z.1)."""
        import numpy as np
        from molbuilder.runfiles import SAVED_PAIR_WHAT
        from molbuilder.structure import Structure
        from molbuilder.workingcopy_structure import StructureCodec
        root, client = isolated
        shelf = root / "P" / "structure"
        shelf.mkdir(parents=True)
        StructureCodec().write(
            Structure(elements=["H", "H"],
                      positions=np.array([[0, 0, 0], [0, 0, 0.74]], float),
                      regions={"bridge": [0, 1]},
                      info={"note": "a pair with its labels and record"}),
            shelf / "h2.xyz")
        (shelf / "notes.txt").write_text("mine\n")
        body = client.get("/api/results/dir?path=" + str(shelf)).get_json()
        about = {f["name"]: f["about"] for f in body["files"]}
        assert about["h2.xyz"]["ours"] is True, about
        assert about["h2.xyz"]["what"] == SAVED_PAIR_WHAT["structure"]
        assert about["h2.molstruct.json"]["what"] == SAVED_PAIR_WHAT["sidecar"]
        assert about["notes.txt"] == {"ours": False}, about

    def test_a_calculations_folders_are_answered_by_the_run_door(
            self, isolated, tmp_path, monkeypatch):
        """Through the road -- `molbuilder describe` and `prep` of an H2 relaxation,
        no engine run -- the run door answers each folder as what it IS
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
        from conftest import write_machine_record
        from support.road import describe_calculation, jobset
        root, client = isolated
        write_machine_record()
        bundle = describe_calculation(tmp_path, monkeypatch)
        got = jobset("prep", "task", "--group", "1", "--bundle", bundle,
                     "--target", "this")
        assert got.exit_code == 0, got.output

        body = client.get("/api/results/dir?path=" + str(bundle)).get_json()
        assert body["status"] is None and body["record"] is None, (
            "a container is not a run")
        assert body["ladder"] is not None, "a calculation root has a ladder"
        # ...and under each rung every run, with what it read and what was
        # checked before it was written (`web/results.md` § 2.4): the deck in
        # the run's folder, the `.validation.txt` beside the deck prep wrote,
        # in the stage's folder -- the run's with its `run-<n>` level out.
        coarse = next(s for s in body["ladder"]["stages"]
                      if s["name"] == "coarse")
        assert [(r["deck"], r["validation"]) for r in coarse["runs"]] == [
            ("01_coarse/run-0/H2_01_coarse.fdf",
             "01_coarse/H2_01_coarse.validation.txt")], coarse["runs"]

        attempt = bundle / "01_coarse" / "run-0"
        body = client.get("/api/results/dir?path=" + str(attempt)).get_json()
        assert (body["status"]["state"], body["status"]["detail"]) == (
            "pending", "prepared, not launched (no launch record)"), body["status"]
        about = {f["name"]: f["about"] for f in body["files"]}
        assert about["H2_01_coarse.fdf"]["ours"] is True, about
        assert about["calcdir.json"]["ours"] is True, about

        got = jobset("prep", "bench", "coarse", "--bundle", bundle,
                     "--target", "this")
        assert got.exit_code == 0, got.output
        trial = sorted((bundle / "01_coarse" / "bench").glob("bench-*"))[0]
        body = client.get("/api/results/dir?path="
                          + str(trial / "run-0")).get_json()
        assert body["status"]["state"] == "pending", body["status"]

    #: WHERE EACH FOLDER PREP MAKES IS (`project-layout.md` § 1.4a, *Where am
    #: I*; invariant 6b; `web/results.md` § 2.3, `place`) -- an H2
    #: relaxation's first group and its first stage's benchmark, prepared in
    #: each shape: ``(shape, folder, what, role, stage, run, trial)``, the
    #: folder relative to the calculation, ``bench-*`` a trial and ``*`` its
    #: point.  Every folder prep made answers one row, and every row a folder.
    EVERY_FOLDER = [
        ("hierarchical", "", "calculation", "container", None, None, None),
        ("hierarchical", "pseudos", "support", "container", None, None, None),
        ("hierarchical", "01_coarse", "stage", "container", "01_coarse",
         None, None),
        ("hierarchical", "01_coarse/run-0", "attempt", "run", "01_coarse",
         0, None),
        ("hierarchical", "01_coarse/bench", "benchmark", "container",
         "01_coarse", None, None),
        ("hierarchical", "01_coarse/bench/bench-*", "trial", "container",
         "01_coarse", None, "*"),
        ("hierarchical", "01_coarse/bench/bench-*/run-0", "attempt", "run",
         "01_coarse", 0, "*"),
        ("flat", "", "calculation", "run", None, None, None),
        ("flat", "pseudos", "support", "container", None, None, None),
        ("flat", "bench_01_coarse", "benchmark", "container", "01_coarse",
         None, None),
        ("flat", "bench_01_coarse/bench-*", "trial", "run", "01_coarse",
         None, "*"),
    ]

    @pytest.mark.parametrize("shape", ["hierarchical", "flat"])
    def test_every_folder_prep_makes_says_where_it_is(
            self, isolated, tmp_path, monkeypatch, shape):
        """Through the road -- `molbuilder describe`, `prep task` and `prep bench`
        of an H2 relaxation, no engine run -- every folder prep made answers
        the Results tab's door with where it is: its calculation, what it
        is, its role, and the stage, run and trial it sits in
        (`project-layout.md` § 1.4a, *Where am I*; `web/results.md` § 2.3).
        The silent failures: a folder read as another stage's or trial's has
        its files read on the wrong label, none of them molbuilder's; a
        benchmark's folder not read as one, its summary refused; a folder
        prep made with no `calcdir.json` (invariant 6b), read as *not
        marked*, its run never asked.

        MUTATION THIS MUST FAIL AGAINST: a trial read as support (the
        benchmark's child rule), an attempt's index not read, the flat
        benchmark not read at the root, a folder an opener left unstamped."""
        from fnmatch import fnmatch
        from pathlib import Path
        from conftest import write_machine_record
        from support.road import describe_calculation, jobset
        _root, client = isolated
        write_machine_record()
        bundle = describe_calculation(tmp_path, monkeypatch, shape=shape)
        for verb in (("task", "--group", "1"), ("bench", "coarse")):
            got = jobset("prep", *verb, "--bundle", bundle, "--target", "this")
            assert got.exit_code == 0, got.output
        rows = [row[1:] for row in self.EVERY_FOLDER if row[0] == shape]
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
            place = client.get(
                "/api/results/dir?path=" + str(folder)).get_json()["place"]
            assert Path(place["calculation"]).resolve() == bundle.resolve(), (
                rel, place)
            assert {k: place[k] for k in ("what", "role", "stage", "run",
                                          "trial", "point")} == {
                "what": what, "role": role, "stage": stage, "run": run,
                "trial": trial, "point": None}, (rel, place)
        assert answered == {r[0] for r in rows}, (
            "a row no folder answered", sorted({r[0] for r in rows} - answered))


    def test_no_deck_answers_null(self, isolated):
        root, client = isolated
        d = root / "bare"
        d.mkdir()
        (d / "chain.xyz").write_text("1\n\nC 0 0 0\n")
        r = client.get("/api/results/contract?path="
                       + str(d / "chain.xyz"))
        out = r.get_json()
        assert out["ok"] is True and out["calculation"] is None

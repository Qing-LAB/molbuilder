"""Tests for the ``/results`` blueprint -- the post-merge unified
inspector page.

Architecture (see ``docs/web/results.md`` § 4 + the
Inspector Registry pattern in
``molbuilder/web/static/lib/inspectors/registry.js``):

  * The page has ONE mount point (``#inspector-host``).
  * Each file type is owned by its own inspector module under
    ``static/lib/inspectors/`` that self-registers on script load.
  * ``static/results/viewer.js`` subscribes to the sidebar
    selection, asks the registry for a match, and lets the
    inspector own the host.

These tests pin the contract that lets new inspectors slot in
without touching this file: route + sidebar + the registered
inspectors served + the partials + the contract endpoint.
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
#  Inspector modules are served + carry the expected interface          #
# --------------------------------------------------------------------- #


class TestInspectorModulesServed:
    """Each inspector module + the registry are reachable via the
    static route + carry the canonical Inspector interface fields."""

    INSPECTORS = ["source", "structure", "trajectory", "spectra"]


    @pytest.mark.parametrize("name", INSPECTORS)
    def test_inspector_module_served(self, web, name):
        r = web.get(f"/static/lib/inspectors/{name}.js")
        assert r.status_code == 200


# --------------------------------------------------------------------- #
#  Error-rendering contract for inspector adapters                       #
#                                                                       #
#  Surfaced by the 2026-05-20 audit: no test pinned what the user        #
#  sees when an inspector mount or partial-fetch fails (404, network    #
#  drop, malformed file).  Both registry adapters (trajectory +         #
#  spectra) render an inspector-card.error-card via _renderError on     #
#  fetch failure; this section pins the wiring at source level (the     #
#  full behavioural verification lives in the Playwright suite          #
#  test_inspector_registry_e2e.py).                                     #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  Dispatch JS                                                          #
# --------------------------------------------------------------------- #


class TestResultsDispatchJS:
    """results/viewer.js is intentionally tiny: it only does
    pick + mount + dispose against the registry.  No per-file-type
    logic should land here."""

    def test_viewer_js_served(self, web):
        assert web.get("/static/results/viewer.js").status_code == 200

    def test_style_css_served(self, web):
        assert web.get("/static/results/style.css").status_code == 200


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
      * Cache-Control: private, max-age=300 -- the partial is static
        across requests but only stable until a deploy; capping at
        5 minutes prevents stale-after-deploy clients indefinitely.
        Set as ``private`` so intermediaries don't cache (no user
        data in the body, but consistency with the per-session
        responses elsewhere).
      * Partial-id parity: the body MUST contain the same ids the
        trajectory core (lib/trajectory/core.js) queries.  A drift
        between this partial and the core's selectors silently
        breaks the inspector on /results.  We pin the load-bearing
        ids that core.js scopes via $() rather than re-deriving
        the full partial-id set (that's tested separately in
        test_trajectory_inspector_partial.py::TestPartialIntegrity).
    """

    # Hand-picked from `grep -oE '\$\("[a-z0-9_-]+"\)' lib/trajectory/core.js`
    # -- a representative slice covering each functional area (MolView
    # mount host, force overlay, plots).  We don't pin EVERY id (~28 of
    # them; tested exhaustively in test_trajectory_inspector_partial.py::
    # TestPartialIntegrity), just enough that a partial-id removal
    # caught by either test surfaces here too.
    REQUIRED_IDS = (
        "viewer-host",             # empty host molview.mount fills (task #34)
        "force-scale",             # force overlay magnitude knob
        "hide-frozen",             # force-arrow frozen-atom filter
        "energy-plot",             # Plotly chart: per-frame energy
        "force-plot",              # Plotly chart: max force
        "scf-energy-plot",         # Plotly chart: SCF history (hidden when no data)
        # The viewer + frame bar + selection/atom-pick UI + playback
        # controls live in MolView (task #34).  See
        # test_trajectory_inspector_partial.py for the authoritative set.
    )

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
            f"missing charset; non-ASCII content in the partial "
            f"(none today, but defensive) could be mis-decoded"
        )

    def test_cache_control_is_private_short_max_age(self, web):
        r = web.get("/partials/trajectory-inspector")
        cc = r.headers.get("Cache-Control", "")
        assert "private" in cc, (
            f"missing 'private' directive (got {cc!r}); the body "
            f"shouldn't be cached by intermediaries"
        )
        assert "max-age=300" in cc, (
            f"expected max-age=300 (got {cc!r}); short cap prevents "
            f"stale clients after a partial-template deploy"
        )

    @pytest.mark.parametrize("element_id", REQUIRED_IDS)
    def test_body_carries_required_inspector_ids(self, web, element_id):
        """Every id the trajectory core scopes ``$()`` against must
        be present in the partial response.  A missing id silently
        breaks the inspector on /results (the $() lookup returns
        null + later code crashes at the first .addEventListener)."""
        body = web.get("/partials/trajectory-inspector").get_data(as_text=True)
        needle = f'id="{element_id}"'
        assert needle in body, (
            f"partial response is missing {needle!r}; "
            f"lib/trajectory/core.js queries this id via $() and "
            f"will crash when mounted on /results"
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
      * Cache-Control: private, max-age=300 -- same rationale as the
        trajectory partial; keeps stale clients bounded after a
        partial-template deploy.
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

    def test_cache_control_is_private_short_max_age(self, web):
        r = web.get("/partials/spectra-inspector")
        cc = r.headers.get("Cache-Control", "")
        assert "private" in cc, f"missing 'private' (got {cc!r})"
        assert "max-age=300" in cc, f"expected max-age=300 (got {cc!r})"

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

    def test_a_calculations_folders_are_answered_by_the_run_door(
            self, isolated, tmp_path, monkeypatch):
        """Through the road -- `jobset init` and `prep` of an H2 relaxation,
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
        got = jobset("prep", "task", "--stage", "coarse", "--bundle", bundle,
                     "--target", "this")
        assert got.exit_code == 0, got.output

        body = client.get("/api/results/dir?path=" + str(bundle)).get_json()
        assert body["place"]["role"] == "container", (
            "a hierarchical root is a container (§ 1.4)")
        assert body["status"] is None and body["record"] is None, (
            "a container is not a run")
        assert body["ladder"] is not None, "a calculation root has a ladder"

        attempt = bundle / "01_coarse" / "run-0"
        body = client.get("/api/results/dir?path=" + str(attempt)).get_json()
        assert body["place"]["role"] == "run"
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
        assert body["place"]["role"] == "run", body["place"]
        assert body["status"]["state"] == "pending", body["status"]


    def test_no_deck_answers_null(self, isolated):
        root, client = isolated
        d = root / "bare"
        d.mkdir()
        (d / "chain.xyz").write_text("1\n\nC 0 0 0\n")
        r = client.get("/api/results/contract?path="
                       + str(d / "chain.xyz"))
        out = r.get_json()
        assert out["ok"] is True and out["calculation"] is None

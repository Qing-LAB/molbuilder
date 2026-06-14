"""Tests for the ``/results`` blueprint -- the post-merge unified
inspector page.

Architecture (see ``docs/protocols/results-tab.md`` § 4 + the
Inspector Registry pattern in
``molbuilder/web/static/lib/inspectors/registry.js``):

  * The page has ONE mount point (``#inspector-host``).
  * Each file type is owned by its own inspector module under
    ``static/lib/inspectors/`` that self-registers on script load.
  * ``static/results/viewer.js`` subscribes to the sidebar
    selection, asks the registry for a match, and lets the
    inspector own the host.

These tests pin the contract that lets new inspectors slot in
without touching this file: route + sidebar + registry script
order + the four currently-registered inspectors + dispatch JS.
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
        assert 'class="has-projects-sidebar"' in body
        assert 'id="projects-sidebar"' in body

    def test_page_has_single_inspector_host(self, web):
        """The registry-driven architecture mounts every inspector
        inside ONE host element.  The previous 5-panel layout was
        a different (now-retired) design."""
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
#  Registry + inspector script wiring                                   #
# --------------------------------------------------------------------- #


class TestResultsRegistryScripts:
    """Script load order matters: registry must precede every
    inspector module (each self-registers at load time), and the
    page-level viewer.js must come last so its init runs after
    all inspectors have registered."""

    REQUIRED_SCRIPTS = [
        "lib/mol-viewer.js",
        "lib/xyz-io.js",
        "lib/path-utils.js",
        "lib/inspectors/registry.js",
        # Shared scaffolding for partial-fetch inspectors (task #308).
        "lib/inspectors/_partial_inspector_factory.js",
        "lib/inspectors/source.js",
        "lib/inspectors/structure.js",
        "lib/inspectors/trajectory.js",
        "lib/inspectors/spectra.js",
        "results/viewer.js",
    ]

    def test_all_required_scripts_present(self, web):
        body = web.get("/results").get_data(as_text=True)
        for script in self.REQUIRED_SCRIPTS:
            assert script in body, f"missing script tag: {script}"

    def test_registry_loads_before_inspectors(self, web):
        body = web.get("/results").get_data(as_text=True)
        reg_ix = body.find("lib/inspectors/registry.js")
        assert reg_ix > 0, "registry.js script tag missing"
        for inspector_name in ("source", "structure", "trajectory", "spectra"):
            insp_ix = body.find(f"lib/inspectors/{inspector_name}.js")
            assert insp_ix > reg_ix, (
                f"inspector {inspector_name}.js loads BEFORE registry.js; "
                f"self-registration would fail"
            )

    def test_inspectors_load_before_viewer(self, web):
        """results/viewer.js's init reads from the registry, so it
        must come AFTER all inspector self-registration."""
        body = web.get("/results").get_data(as_text=True)
        viewer_ix = body.find("results/viewer.js")
        for inspector_name in ("source", "structure", "trajectory", "spectra"):
            insp_ix = body.find(f"lib/inspectors/{inspector_name}.js")
            assert insp_ix < viewer_ix, (
                f"viewer.js loads BEFORE {inspector_name}.js; "
                f"init may run before the inspector registered"
            )


# --------------------------------------------------------------------- #
#  Inspector modules are served + carry the expected interface          #
# --------------------------------------------------------------------- #


class TestInspectorModulesServed:
    """Each inspector module + the registry are reachable via the
    static route + carry the canonical Inspector interface fields."""

    INSPECTORS = ["source", "structure", "trajectory", "spectra"]

    def test_registry_served(self, web):
        r = web.get("/static/lib/inspectors/registry.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        for sym in ("register", "pick", "mount", "list",
                    "createDefaultContext", "_clear"):
            assert sym in body, f"registry export missing: {sym}"

    @pytest.mark.parametrize("name", INSPECTORS)
    def test_inspector_module_served(self, web, name):
        r = web.get(f"/static/lib/inspectors/{name}.js")
        assert r.status_code == 200

    @pytest.mark.parametrize("name", INSPECTORS)
    def test_inspector_module_declares_canonical_interface(self, web, name):
        """Every inspector module declares the four required
        interface fields (name, displayName, match, mount) and
        self-registers via window.molbuilder.inspectors.register.

        Two acceptable shapes after task #308:
          * direct: an inspector literal with ``mount(...)`` and the
            other fields inline (source, structure).
          * factory-built: ``makePartialInspector({...})`` with
            ``mount`` provided by the factory (trajectory, spectra).
            The factory itself MUST live in the partial-inspector-
            factory module the load-order test below pins.
        """
        body = web.get(f"/static/lib/inspectors/{name}.js").get_data(as_text=True)
        for key in ("name:", "displayName:", "match:"):
            assert key in body, (
                f"inspector {name!r} module is missing interface key {key!r}"
            )
        # Mount comes either as a method literal or via the factory.
        has_mount = ("mount(" in body
                     or "makePartialInspector(" in body)
        assert has_mount, (
            f"inspector {name!r} module is missing mount() — neither "
            f"an inline mount(...) literal nor a "
            f"makePartialInspector({{...}}) call was found"
        )
        # Self-registration call.
        assert "inspectors.register(inspector)" in body, (
            f"inspector {name!r} module does not self-register"
        )

    def test_source_inspector_matches_text_file_extensions(self, web):
        """The source inspector is the catch-all for human-readable
        text formats; pin its extension list against the spec.

        Note: ``.out`` is INTENTIONALLY absent -- it routes to the
        trajectory inspector (SIESTA's redirected stdout is parsed
        into a frame trajectory; raw text view is hostile UX for
        multi-MB outputs).  See
        ``test_trajectory_inspector_claims_dot_out`` below.
        """
        r = web.get("/static/lib/inspectors/source.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        for ext in (".fdf", ".py", ".log", ".json", ".txt", ".md"):
            assert ext in body, (
                f"source inspector dispatch missing extension {ext!r}"
            )
        # Negative pin: .out moved to trajectory.
        # Use a stricter match -- the docstring/comment mentions
        # ``.out`` so just checking absence of the substring would
        # produce false positives.  Match the actual match() clause.
        assert 'endsWith(".out")' not in body, (
            "source inspector still matches .out -- it should route "
            "to trajectory (see source.js match function comment)"
        )

    def test_structure_inspector_matches_xyz_pdb(self, web):
        body = web.get("/static/lib/inspectors/structure.js").get_data(as_text=True)
        assert ".xyz" in body
        assert ".pdb" in body

    def test_trajectory_inspector_matches_compound_extension(self, web):
        """``.molwatch.log`` matched as a compound extension so it
        wins over plain ``.log`` (which the source inspector also
        claims)."""
        body = web.get("/static/lib/inspectors/trajectory.js").get_data(as_text=True)
        assert ".molwatch.log" in body

    def test_trajectory_inspector_claims_dot_out(self, web):
        """SIESTA's redirected stdout (.out) is a trajectory in
        practice -- the backend SiestaParser extracts a frame
        sequence + per-step energies + force vectors.  The
        trajectory inspector renders that via /api/watch/load,
        which uses detect_parser() to pick the right backend
        parser.  Pin the routing here so a future refactor that
        narrows trajectory's match back to just .molwatch.log
        lands as a clear failure.
        """
        body = web.get("/static/lib/inspectors/trajectory.js").get_data(as_text=True)
        assert 'endsWith(".out")' in body, (
            "trajectory inspector no longer claims .out -- SIESTA "
            "output would fall through to the source inspector and "
            "render as raw text (the 2026-05-18 user-report bug)"
        )

    def test_trajectory_inspector_does_NOT_claim_pyscf_log(self, web):
        """Phase C (2026-06-07): PySCF wrapper output renamed from
        ``.out`` to ``.pyscf.log`` so the dispatcher can tell PySCF
        apart from SIESTA.  The trajectory inspector deliberately
        does NOT claim ``.pyscf.log`` -- the file is plain PySCF
        stdout, not a trajectory format; a dedicated pyscf-log
        inspector is on the roadmap.  Until then ``.pyscf.log``
        falls through to the source inspector (text viewer)."""
        body = web.get("/static/lib/inspectors/trajectory.js").get_data(as_text=True)
        # The match function MUST NOT include .pyscf.log.  The
        # documentation comment may mention it (noting why it's
        # excluded), but the actual match clause must not.  Find
        # the match function body and inspect.
        import re
        m = re.search(r'match:\s*\(file\)\s*=>\s*\{(.+?)\},',
                       body, re.DOTALL)
        assert m, "trajectory match() function not found"
        match_body = m.group(1)
        assert '.pyscf.log' not in match_body, (
            "trajectory inspector match() claims .pyscf.log; it "
            "should fall through to source.js until a dedicated "
            "pyscf-log inspector ships."
        )

    def test_spectra_inspector_matches_compound_extension(self, web):
        body = web.get("/static/lib/inspectors/spectra.js").get_data(as_text=True)
        assert ".spectra.json" in body

    def test_source_inspector_requests_max_read_budget(self, web):
        """SIESTA's runtime output (.out) routinely exceeds the
        ``/api/files/read`` 1 MB default cap, returning 413 + a
        size-exceeds error rendered as "Error: file is N bytes;
        exceeds max_bytes = 1048576".  The source inspector MUST
        pass ``maxBytes`` to ``ctx.readFile`` so the request hits
        the server's hard ceiling (16 MB today) instead of the
        default.  Pin the call shape so a future refactor that
        drops the maxBytes argument lands as a clear failure (and
        not as "siesta .out file silently shows an error").
        """
        body = web.get("/static/lib/inspectors/source.js").get_data(as_text=True)
        # The readFile call must pass an opts object with a
        # maxBytes field.  Accept any of: ``maxBytes: 16 * 1024 * 1024``,
        # ``maxBytes: 16777216``, ``maxBytes: _MAX_READ_BYTES``.  Pin
        # SHAPE -- the literal "maxBytes" key inside a readFile call.
        import re
        match = re.search(
            r"readFile\s*\([^)]*\{\s*[^}]*maxBytes\s*:",
            body,
        )
        assert match, (
            "lib/inspectors/source.js calls ctx.readFile() without a "
            "maxBytes override.  Real SIESTA .out files exceed the "
            "server-side 1 MB default and the inspector silently shows "
            "a 413 error.  Pass {maxBytes: 16 * 1024 * 1024} (or the "
            "server's _MAX_READ_BYTES) to ctx.readFile."
        )

    def test_registry_readfile_helper_accepts_maxbytes_opt(self, web):
        """createDefaultContext's readFile must accept an ``opts``
        argument and thread ``opts.maxBytes`` into the URL.  Without
        this seam, the per-inspector budget override (e.g., source's
        16 MB request) can't reach the backend.  Pin both: the
        signature accepts opts AND the URL append happens.
        """
        body = web.get(
            "/static/lib/inspectors/registry.js").get_data(as_text=True)
        # Signature accepts opts.
        assert "readFile(file, opts)" in body or \
               "readFile(file,opts)" in body, (
            "createDefaultContext.readFile does not accept opts -- "
            "per-inspector maxBytes overrides have no seam to reach "
            "the backend"
        )
        # opts.maxBytes flows into the URL as ``max_bytes=``.
        assert "max_bytes=" in body, (
            "readFile no longer threads maxBytes into the request URL "
            "-- the opts.maxBytes override is silently ignored"
        )


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


class TestInspectorErrorRendering:
    """Pin the error-card UX wiring in each adapter.  Source-level
    pins because the runtime path needs Playwright to exercise."""

    # ``trajectory`` + ``spectra`` are factory-derived inspectors
    # (task #308) — the error-rendering code lives in
    # _partial_inspector_factory.js, not in the per-engine wrapper.
    # Pin the contract on the factory instead.  ``structure`` and
    # ``source`` aren't factory-derived; their error paths are
    # exercised elsewhere (test_source_inspector_e2e.py renders the
    # source viewer's error path; structure's empty-error path is
    # covered by Playwright's test_inspector_registry_e2e.py).
    PARTIAL_INSPECTOR_FACTORY = "_partial_inspector_factory"

    def test_factory_renders_error_card_on_fetch_failure(self, web):
        """The partial-inspector factory MUST call _renderError()
        inside its .catch() handler when the partial fetch fails.
        Without this, a 404 from GET /partials/<name>-inspector
        would leave the user staring at a blank inspector host with
        no signal that the mount failed.

        Pin SHAPE not exact-text -- ``.catch(`` somewhere in the
        file, ``_renderError(`` somewhere AFTER it.  Anything more
        precise (e.g., requiring _renderError inside the exact
        ``.catch(arrow => { ... })`` body) breaks on every
        refactor of the multi-line chain.
        """
        body = web.get(
            f"/static/lib/inspectors/"
            f"{self.PARTIAL_INSPECTOR_FACTORY}.js").get_data(as_text=True)
        catch_pos = body.find(".catch(")
        render_after_catch = body.find("_renderError(", catch_pos)
        assert catch_pos > 0, (
            "lib/inspectors/_partial_inspector_factory.js doesn't "
            "have a .catch() handler on the partial-fetch promise "
            "chain -- partial-fetch failures fall through unhandled"
        )
        assert render_after_catch > catch_pos, (
            "_partial_inspector_factory doesn't call _renderError(...) "
            "AFTER the .catch() handler -- the user sees a blank host "
            "when the partial fetch fails"
        )
        # The error card itself must use the .inspector-card.error-card
        # class hook (CSS in results/style.css renders this with a
        # red-tinted border + clear failure styling).
        assert "inspector-card error-card" in body, (
            "_partial_inspector_factory doesn't use the "
            ".inspector-card.error-card class for failed mounts -- "
            "the CSS in results/style.css would lose its hook"
        )
        # XSS-safe DOM construction (no innerHTML string-concat in
        # _renderError; pinned in the global XSS audit too).
        assert "createElement" in body

    def test_factory_ignores_aborterror_in_catch(self, web):
        """A user picking another file mid-fetch triggers the
        AbortController's AbortError.  That's NOT an error to
        surface to the user -- it's the expected superseding
        action.  The factory guards via ``err.name === "AbortError"``
        and bails before _renderError.  Pinned so a future refactor
        that drops the abort guard surfaces with a clear failure
        (else: every rapid file switch shows a spurious error card).
        """
        body = web.get(
            f"/static/lib/inspectors/"
            f"{self.PARTIAL_INSPECTOR_FACTORY}.js").get_data(as_text=True)
        assert 'err.name === "AbortError"' in body, (
            "_partial_inspector_factory doesn't guard against "
            "AbortError in its .catch() -- rapid file-switching "
            "will show a spurious 'mount failed' card after each swap"
        )

    def test_inspector_error_css_classes_are_styled(self, web):
        """The error-card classes the adapters create must have CSS
        rules in results/style.css; without them the card renders
        unstyled (white text on dark, no border, looks broken)."""
        css = web.get(
            "/static/results/style.css").get_data(as_text=True)
        assert ".inspector-error" in css, (
            "results/style.css missing .inspector-error rule -- "
            "block-level error mounts render unstyled"
        )
        assert ".inspector-card" in css, (
            "results/style.css missing .inspector-card rule -- the "
            "error card's container is unstyled"
        )


# --------------------------------------------------------------------- #
#  Dispatch JS                                                          #
# --------------------------------------------------------------------- #


class TestResultsDispatchJS:
    """results/viewer.js is intentionally tiny: it only does
    pick + mount + dispose against the registry.  No per-file-type
    logic should land here."""

    def test_viewer_js_served(self, web):
        r = web.get("/static/results/viewer.js")
        assert r.status_code == 200
        assert b"inspector-host" in r.data

    def test_style_css_served(self, web):
        r = web.get("/static/results/style.css")
        assert r.status_code == 200
        assert b".results-inspector-host" in r.data
        assert b".inspector-card" in r.data

    def test_dispatch_uses_registry_pick_and_mount(self, web):
        """The dispatch is implemented in terms of the registry --
        not by hand-coded match rules.  Adding a new inspector
        must not require editing viewer.js."""
        body = web.get("/static/results/viewer.js").get_data(as_text=True)
        for needle in ("inspectors", "pick(", "mount(",
                       "currentHandle", "dispose"):
            assert needle in body, (
                f"viewer.js dispatch is missing: {needle!r}"
            )

    def test_dispatch_disposes_before_remount(self, web):
        """A new file selection must dispose the PREVIOUS inspector's
        handle BEFORE mounting the new one -- timers / listeners /
        3Dmol viewers leak otherwise."""
        body = web.get("/static/results/viewer.js").get_data(as_text=True)
        # Pin the dispose-before-mount comment so a future refactor
        # that drops the cleanup lands with a clear test failure.
        assert "Dispose the previous inspector BEFORE" in body, (
            "the dispose-before-mount invariant lost its anchoring "
            "comment in viewer.js"
        )

    def test_dispatch_builds_mount_context_once_via_registry_helper(
            self, web):
        """The 2026-05-17 architectural cleanup moved mount-context
        construction OUT of registry.mount() and INTO the dispatcher.
        Pin that the dispatcher uses ``createDefaultContext(host)``
        once at init + reuses the same context for every mount, so a
        future dispatcher that wants a custom ctx (cached readFile,
        custom error UI) has a clean injection point."""
        body = web.get("/static/results/viewer.js").get_data(as_text=True)
        assert "createDefaultContext" in body, (
            "dispatcher no longer asks the registry to build the "
            "mount context; either the registry's helper was renamed "
            "or the dispatcher reverted to letting registry.mount() "
            "build a default (less customisable)."
        )
        # Built ONCE and reused (a "let mountContext = null;" hoist
        # at top + assignment inside init).
        assert "let mountContext" in body, (
            "mount context should be a closure variable built once "
            "at init, not re-built per dispatch call"
        )

    def test_dispatch_validates_registry_populated_at_init(self, web):
        """The dispatcher logs a loud console.error when the registry
        is empty at init -- catches "inspector script tag failed to
        load" before the user clicks something and gets a blank
        panel."""
        body = web.get("/static/results/viewer.js").get_data(as_text=True)
        assert "registry empty at init" in body, (
            "dispatcher init no longer validates that inspectors "
            "have registered before the page goes live"
        )


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
    # -- a representative slice covering each functional area
    # (viewer host, frame controls, playback buttons, force overlay,
    # atom-pick UI, plots).  We don't pin EVERY id (~30 of them; tested
    # exhaustively in test_trajectory_inspector_partial.py::
    # TestPartialIntegrity), just enough that a partial-id removal
    # caught by either test surfaces here too.
    REQUIRED_IDS = (
        "viewer",                  # 3Dmol viewport container
        "force-scale",             # force overlay magnitude knob
        "inspect-atom-list-body",  # atom-pick table body
        "energy-plot",             # Plotly chart: per-frame energy
        "force-plot",              # Plotly chart: max force
        "scf-energy-plot",         # Plotly chart: SCF history (hidden when no data)
        # Frame-strip controls (prev/play/pause/next/frame-slider/
        # frame-idx) moved to the embed's auto-mounted frame strip
        # in #246 A1 — they are intentionally NOT in the partial
        # anymore.  See test_trajectory_inspector_partial.py
        # EMBED_OWNED_IDS for the authoritative split.
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


class TestInspectorRegistrationOrder:
    """The inspector registry resolves overlapping match predicates
    by REGISTRATION ORDER (``pick()`` returns the first match).
    The registry's own docstring states this contract:

        Inspectors with more specific predicates (compound extensions
        like ``.molwatch.log``) MUST register before more general
        ones (``.log``) to win the dispatch.

    The ``source`` inspector matches generic ``.log`` and ``.json``,
    which would silently steal ``.molwatch.log`` (trajectory's compound
    extension) and ``.spectra.json`` (spectra's compound extension)
    if it loaded first.  This test pins the load order in
    ``results.html`` so a future refactor (e.g. an alphabetical-sort
    accident in the script-tag list) can't reintroduce the bug.

    The bug WAS in production (user-reported 2026-05-18): selecting
    a ``.molwatch.log`` rendered the source-inspector text view
    instead of the trajectory viewer.  Order was alphabetical
    (source → structure → trajectory → spectra).  Fix: order by
    match-specificity (specific → general).

    BEHAVIOUR-LEVEL COUNTERPART:
    ``tests/test_inspector_registry_e2e.py`` evaluates the real
    ``window.molbuilder.inspectors.pick(...)`` in a Playwright-driven
    browser and asserts:
      * pick('foo.molwatch.log') -> 'trajectory'
      * pick('foo.spectra.json') -> 'spectra'
      * pick('foo.json')         -> 'source'
    Those tests are skipped when playwright isn't installed (the
    default in CI today) -- when they DO run, they prove the load
    order pinned below actually translates to the right dispatch
    decision at runtime.  This static test catches the regression
    earlier (in unit-test phase, no browser needed) while the e2e
    test catches it more strictly (real registry, real DOM).
    """

    EXPECTED_ORDER = (
        # The registry must load before any inspector that
        # self-registers against it.
        "lib/inspectors/registry.js",
        # Most-specific (compound extensions) first.
        "lib/inspectors/trajectory.js",   # .molwatch.log
        "lib/inspectors/spectra.js",      # .spectra.json
        # Then specific-but-single-extension (.xyz / .pdb).
        "lib/inspectors/structure.js",
        # Then the catch-all source inspector (.log / .json / .py /
        # .fdf / .out / .txt / .md).  Always last.
        "lib/inspectors/source.js",
    )

    def test_inspector_script_tags_appear_in_specificity_order(self, web):
        body = web.get("/results").get_data(as_text=True)
        # Find the byte index of each expected script tag.  We don't
        # match the FULL <script src="..."> form because the template
        # uses Jinja's url_for which may emit slightly different
        # quoting in future Flask versions; matching on the path
        # suffix is robust to that and still strict on order.
        positions = []
        for needle in self.EXPECTED_ORDER:
            ix = body.find(needle)
            assert ix != -1, (
                f"results.html no longer includes {needle!r}; the "
                f"inspector registration was probably renamed or "
                f"removed without updating this test"
            )
            positions.append((needle, ix))
        sorted_by_position = sorted(positions, key=lambda np: np[1])
        actual_order = tuple(np[0] for np in sorted_by_position)
        assert actual_order == self.EXPECTED_ORDER, (
            f"inspector script tags are not in specificity order.  "
            f"Expected:\n  " + "\n  ".join(self.EXPECTED_ORDER) +
            f"\nActual (by appearance in results.html):\n  " +
            "\n  ".join(actual_order) +
            f"\nThe registry's pick() returns the FIRST match.  "
            f"Loading the source inspector (matches .log / .json) "
            f"before the trajectory/spectra inspectors (match the "
            f"compound .molwatch.log / .spectra.json) makes source "
            f"silently steal those files.  See registry.js's "
            f"`Ordering` section."
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
        from the inspect-side functions in static/spectra/viewer.js
        (and, after step 2.4 lands, lib/spectra/core.js).
    """

    # Hand-picked from the inspect-side surface of
    # static/spectra/viewer.js -- a representative slice covering
    # every functional area (load controls, results summary, chart,
    # modes table + filter + CSV, mode viewer + animation, ES bar
    # diagram).  Each id is queried by ``els.<key> = document.
    # getElementById(...)`` near the top of the IIFE.
    REQUIRED_IDS = (
        # Load controls
        "watch-path",
        "load-path-btn",
        "watch-btn",
        "watch-stop-btn",
        "watch-status",
        "phase-indicator",
        # Results summary + chart
        "results-summary",
        "results-summary-list",
        "spectrum-chart",
        "broadening-fwhm",
        # Modes table
        "modes-table",
        "modes-tbody",
        "modes-thead-row",
        "modes-filter",
        "modes-filter-count",
        "modes-csv-btn",
        # Mode viewer (3D animation)
        "mode-viewer-wrap",
        "mode-viewer",
        "viewer-status",
        "anim-amplitude",
        "anim-amplitude-val",
        "anim-speed",
        "anim-speed-val",
        "anim-toggle",
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

    def test_partial_has_no_undocumented_ids(self, web):
        """Bidirectional contract pin: every id in the partial body
        must be in ``REQUIRED_IDS``.  Pre-2026-06-13 only the one-way
        check existed; new template additions could drift in without
        the test set being updated, defeating the "explicit contract"
        invariant the trajectory-partial test pins.  Per
        ``docs/protocols/test-strategy.md`` § 8.8 the partial-pin tests
        should mirror ``TestPartialIntegrity::test_partial_declares_*``
        in both directions."""
        import re
        body = web.get("/partials/spectra-inspector").get_data(as_text=True)
        actual = set(re.findall(r'id="([^"]+)"', body))
        extra = actual - set(self.REQUIRED_IDS)
        assert not extra, (
            f"spectra inspector partial has IDs not in the explicit "
            f"contract: {sorted(extra)}.  Add them to REQUIRED_IDS "
            f"if intentional, or remove them from the template."
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
                f"docs/protocols/results-tab.md for the split."
            )


# --------------------------------------------------------------------- #
#  /partials/selection-panel                                            #
# --------------------------------------------------------------------- #


class TestPartialSelectionPanelEndpoint:
    """Pin the GET /partials/selection-panel contract.

    The selection panel (lib/selection-panel.js) fetches this partial
    on /modify (and later /spectra) and assigns it to a host
    container's innerHTML.  Same shape + cache contract as the
    inspector partials -- one HTTP contract across all partials so
    a deploy hiccup affects every partial consistently rather than
    silently breaking one.
    """

    REQUIRED_IDS = (
        # Outer wrapper (load-bearing for selection-panel.css layout).
        "selection-card",
        "selection-mode-click",
        "selection-mode-filter",
        "selection-click-section",
        "selection-filter-section",
        "selection-filter-rows",
        "selection-combinator",
        "selection-add-filter",
        "selection-apply-filter",
        "selection-clear-btn",
        "selection-count",
        "selection-loading",
        "selection-atom-list",
        "selection-list-empty",
        "selection-select-all",
        "selection-assign-target",
        "selection-assign-new-label",
        "selection-assign-btn",
        "selection-add-btn",
        "selection-remove-btn",
        "selection-error",
        # Isolate-mode toggle + Invert button (2026-06-12 selection
        # panel v3; lib/selection-panel.js queries both by id).
        "selection-invert-btn",
        "selection-isolate-checkbox",
    )

    def test_endpoint_returns_html_200(self, web):
        r = web.get("/partials/selection-panel")
        assert r.status_code == 200

    def test_content_type_is_html(self, web):
        r = web.get("/partials/selection-panel")
        ctype = r.headers.get("Content-Type", "")
        assert "text/html" in ctype
        assert "charset=utf-8" in ctype.lower()

    def test_cache_control_is_private_short_max_age(self, web):
        r = web.get("/partials/selection-panel")
        cc = r.headers.get("Cache-Control", "")
        assert "private" in cc
        assert "max-age=300" in cc

    @pytest.mark.parametrize("element_id", REQUIRED_IDS)
    def test_body_carries_required_panel_ids(self, web, element_id):
        body = web.get("/partials/selection-panel").get_data(as_text=True)
        assert f'id="{element_id}"' in body, (
            f"selection partial is missing {element_id!r}; "
            f"lib/selection-panel.js will fail on $() lookup."
        )

    def test_partial_has_no_undocumented_ids(self, web):
        """Bidirectional contract pin (added 2026-06-13).  Catches new
        template IDs that aren't yet in ``REQUIRED_IDS`` — the gap
        that let ``selection-invert-btn`` + ``selection-isolate-checkbox``
        sit unpinned for a month after the 2026-06-12 selection-panel
        v3 work."""
        import re
        body = web.get("/partials/selection-panel").get_data(as_text=True)
        actual = set(re.findall(r'id="([^"]+)"', body))
        extra = actual - set(self.REQUIRED_IDS)
        assert not extra, (
            f"selection panel partial has IDs not in the explicit "
            f"contract: {sorted(extra)}.  Add them to REQUIRED_IDS "
            f"if intentional, or remove them from the template."
        )


class TestSpectraIssuesPanelSeverityCoverage:
    """Task #304 regression: ``lib/spectra/core.js`` previously
    filtered the issues array into ``errs + warns`` and silently
    dropped ``info`` severity, so the Pattern-B regions notice
    (and any future info-level message) would be invisible on the
    Spectra results panel even though the API delivered it.

    Static asserts on the source so a regression would surface in
    code review rather than waiting for the right-shaped fixture
    to flow through an e2e."""

    def test_spectra_renderIssues_includes_info_severity(self):
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] \
            / "molbuilder/web/static/lib/spectra/core.js"
        text = src.read_text()
        # The function must filter info alongside error + warn.
        assert 'i.severity === "info"' in text, (
            "spectra/core.js renderIssues must filter info-severity "
            "issues into the rendered list; pre-#304 it dropped them "
            "silently.  See docs/protocols/three-stage-contract.md."
        )
        # ...and must concatenate info into the rendered list (the
        # iteration step, not just the filter expression).  Catches a
        # regression where someone filters info-severity issues but
        # forgets to include them in errs.concat(warns).concat(infos).
        assert ".concat(infos)" in text or "concat(infos)" in text, (
            "spectra/core.js renderIssues filters info but never "
            "concatenates them into the rendered list."
        )

    def test_spectra_style_has_info_severity_rules(self):
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] \
            / "molbuilder/web/static/spectra/style.css"
        text = src.read_text()
        assert ".issue.info" in text, (
            "spectra/style.css is missing .issue.info colour rules; "
            "the renderer emits class 'issue info' but the panel "
            "would render unstyled blocks without these rules."
        )
        assert ".issue.info  .badge" in text \
               or ".issue.info .badge" in text, (
            "spectra/style.css needs a badge colour for the info "
            "severity, matching error + warn."
        )

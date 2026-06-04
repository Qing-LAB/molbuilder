"""Tests for ``templates/_trajectory_inspector.html`` -- the shared
DOM scaffolding the trajectory inspector reaches via getElementById.

The partial is the ONE source of the inspector ids.  /results is
the sole consumer today (fetches the partial via
``GET /partials/trajectory-inspector``); /watch was retired
2026-05-19 so the legacy ``{% include %}`` path is gone.

Invariants pinned here:

  * the partial declares the canonical id set the trajectory core
    (lib/trajectory/core.js) reaches via $();
  * the partial carries NO page chrome (no <html>/<body>/<head>) so
    its innerHTML-injection on /results doesn't double-up;
  * the partial does NOT carry the deleted /watch loader-bar +
    workflow-guide ids -- those were /watch.html-only and would be
    UX noise if they crept into /results' inspector body.

If any of these break, the trajectory inspector silently stops
working in a way that the smoke tests wouldn't notice -- this
file makes the failure loud.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def web():
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


@pytest.fixture(scope="module")
def partial_path() -> Path:
    p = (Path(__file__).resolve().parent.parent
         / "molbuilder" / "web" / "templates"
         / "_trajectory_inspector.html")
    assert p.is_file(), f"partial not found: {p}"
    return p


@pytest.fixture(scope="module")
def partial_ids(partial_path) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', partial_path.read_text()))


# --------------------------------------------------------------------- #
#  Partial integrity                                                    #
# --------------------------------------------------------------------- #


class TestPartialIntegrity:

    # ------------------------------------------------------------- #
    # Partial-vs-embed boundary (post-#205 architecture).
    #
    # When the standard knob bar moved into the embedded MolViewer
    # in #205, the trajectory partial stopped owning chrome controls
    # (style.rep, style.radiusScale, style.colorScheme,
    # style.background — formerly the ``rep``, ``radius``,
    # ``colorscheme``, ``bg`` IDs).  The embed's knob bar (see
    # docs/protocols/embedded-viewer.md § 6) now owns those, and
    # they're tested separately via test_mol_viewer_embed_e2e.py's
    # knob-bar contract.
    #
    # The boundary is encoded below as two explicit sets so a future
    # accidental re-introduction of chrome IDs into the partial
    # fails the build, AND a missing trajectory-specific ID also
    # fails.  This replaces the pre-#205 EXPECTED set that mixed
    # both responsibilities.
    # ------------------------------------------------------------- #

    # IDs the trajectory partial MUST declare.  These are
    # trajectory-domain UI (data display, parse warnings, run-
    # state, force overlay toggles, inspect panel, plots) — things
    # the embed has no concept of and that have no equivalent on
    # /build, /modify, or the spectra inspector.
    TRAJECTORY_PARTIAL_IDS = {
        # Embed mount + frame strip wrapper (the strip itself lives
        # inside the embed per § 6.3; the OUTER controls are still
        # the partial's responsibility).
        "viewer",
        "frame-idx", "frame-tot", "frame-slider",
        "prev", "play", "pause", "next",
        # Run-state badge + compact runtime-info one-liner.
        "run-state-badge", "run-state-label", "run-state-detail",
        "runtime-summary",
        # Parse-warnings panel (Level-3 parser, 2026-05-28).  Hidden
        # by default; shows when the parser hit non-fatal issues.
        "parse-warnings",
        "parse-warnings-count",
        "parse-warnings-list",
        # Trajectory-specific overlay toggles.  show-cell + show-
        # indices wire to handle.setCell / handle.setLabels; show-
        # forces drives the trajectory's arrowsPerFrame (DFT force
        # vectors — the embed knows nothing about forces).
        "show-cell", "show-indices",
        "show-forces", "forces-status",
        "force-scale", "force-scale-val", "force-min",
        "highlight-max",
        # Inspect panel (atom list + distance pick readout).
        "inspect-hint", "inspect-atom-list-body",
        "inspect-table", "inspect-a", "inspect-b", "inspect-d",
        "inspect-clear",
        # Trajectory playback knobs.
        "speed", "loop", "save-frame",
        # SCF banner + plots.
        "scf-section", "scf-title", "scf-status",
        "energy-plot", "force-plot",
        "scf-energy-plot", "scf-gnorm-plot",
    }

    # IDs the partial MUST NOT declare — they belong to the embed's
    # standard knob bar (style.rep / radiusScale / colorScheme /
    # background per § 3.3 + § 6).  A re-introduction here means a
    # consumer hand-rolled chrome that should go through setStyle /
    # setBackground / the knob bar.
    EMBED_OWNED_IDS = {
        "rep", "radius", "colorscheme", "bg",
    }

    def test_partial_declares_trajectory_specific_ids(self, partial_ids):
        """The partial owns trajectory-domain UI; deleting any of
        these IDs without an explicit JS-side change is a silent
        regression.  Updating the set is a deliberate API change."""
        missing = self.TRAJECTORY_PARTIAL_IDS - partial_ids
        assert not missing, (
            f"partial is missing trajectory-specific IDs: "
            f"{sorted(missing)}"
        )
        # Extras flag IDs not covered by the explicit contract.
        # Exclude embed-owned IDs from this check — they're handled
        # by the next test as a separate boundary violation.
        extra = (partial_ids
                 - self.TRAJECTORY_PARTIAL_IDS
                 - self.EMBED_OWNED_IDS)
        assert not extra, (
            f"partial has IDs not in the explicit contract: "
            f"{sorted(extra)}.  Add them to "
            f"TRAJECTORY_PARTIAL_IDS if intentional."
        )

    def test_partial_does_not_redeclare_embed_chrome_ids(self, partial_ids):
        """Post-#205 the embed's standard knob bar owns the style /
        background controls.  Re-introducing ``rep`` / ``radius`` /
        ``colorscheme`` / ``bg`` in the trajectory partial would
        be a double-UI bug: the partial's input would fight (or
        silently shadow) the knob bar's input.  Hosts that need
        custom chrome should configure ``opts.knobs`` per § 3.10
        instead of hand-rolling DOM."""
        leaked = self.EMBED_OWNED_IDS & partial_ids
        assert not leaked, (
            f"partial re-declares embed-owned chrome IDs: "
            f"{sorted(leaked)}.  These moved to the embed's knob "
            f"bar in #205 (see docs/protocols/embedded-viewer.md "
            f"§ 6).  If the partial needs a custom chrome control, "
            f"configure opts.knobs at mount instead."
        )

    def test_partial_has_no_page_specific_markup(self, partial_path):
        """Page-specific bits (the path-input loader bar, the workflow
        guide, the per-page <head> / <body>) must NOT live in the
        partial.  The partial is reused by /results in step 4 and
        carrying /watch-specific markup would break that.

        Comments inside the partial may legitimately mention these
        names (the partial's docstring explains what's intentionally
        NOT in it), so we strip Jinja + HTML comments before scanning.
        """
        text = partial_path.read_text()
        # Strip Jinja comments {# ... #} (including {#- ... -#}).
        text = re.sub(r'\{#-?.*?-?#\}', '', text, flags=re.S)
        # Strip HTML comments <!-- ... -->.
        text = re.sub(r'<!--.*?-->', '', text, flags=re.S)

        forbidden = (
            ("<html", "whole-page markup"),
            ("<body", "whole-page markup"),
            ("<head>", "whole-page markup"),
            ('id="path-input"', "loader-bar id (/watch-only)"),
            ('id="load-btn"',   "loader-bar id (/watch-only)"),
            ('id="load-from-selection-btn"', "sidebar-load button"),
            ('id="workflow-guide"', "/watch tutorial content"),
            ('id="status"',     "page-level status banner"),
        )
        for needle, why in forbidden:
            assert needle not in text, (
                f"partial leaked page-specific markup {needle!r} "
                f"({why}); the partial must stay reusable for /results"
            )


# --------------------------------------------------------------------- #
#  watch.html include                                                   #
# --------------------------------------------------------------------- #


class TestRenderedPartial:
    """Render the partial via the canonical endpoint
    ``GET /partials/trajectory-inspector`` and verify the rendered
    HTML carries every id the partial declares + no duplicates.
    Replaces the legacy ``TestWatchHtmlUsesPartial`` +
    ``TestRenderedWatchPage`` suites that rendered ``/watch``
    directly; /watch was removed 2026-05-19."""

    def test_partial_endpoint_renders(self, web):
        r = web.get("/partials/trajectory-inspector")
        assert r.status_code == 200, (
            "/partials/trajectory-inspector must respond 200; /results' "
            "registry adapter fetches it on every .molwatch.log / .out "
            "selection"
        )

    def test_every_partial_id_present_in_rendered_html(self, web, partial_ids):
        html = web.get("/partials/trajectory-inspector").get_data(as_text=True)
        for pid in sorted(partial_ids):
            assert f'id="{pid}"' in html, (
                f"id {pid!r} missing from rendered partial HTML "
                f"(the partial declares it but the render path "
                f"dropped it)"
            )

    def test_no_duplicate_ids_in_rendered_html(self, web):
        html = web.get("/partials/trajectory-inspector").get_data(as_text=True)
        all_ids = re.findall(r'\bid="([^"]+)"', html)
        seen = set()
        dupes = []
        for the_id in all_ids:
            if the_id in seen:
                dupes.append(the_id)
            seen.add(the_id)
        assert not dupes, (
            f"duplicate ids in rendered partial: {dupes}.  Mounting "
            f"this HTML into /results' inspector-host would create "
            f"id collisions; getElementById would pick one arbitrarily."
        )


# --------------------------------------------------------------------- #
#  Stage 1B: viewer.js DOM queries are scoped to a rootEl argument      #
# --------------------------------------------------------------------- #


class TestViewerJsRootScoping:
    """Post-lift (task #76 / docs/protocols/results-tab.md § 4): the
    inspector body lives in ``static/lib/trajectory/core.js`` (the
    shared core that both /watch and /results call into).  Every
    DOM lookup inside the inspector body must be either (a) scoped
    to ``rootEl`` (the partial-resident ids -- 38 of them) or
    (b) document-wide via ``$doc()`` (the page-level loader-bar
    ids -- 4 of them, visible only on /watch).

    Pinning these invariants in the SHARED module catches a
    regression the moment someone accidentally bypasses the
    scoping helpers and goes direct to ``document.getElementById``
    again; the bug would break /results-side mounts where the
    document is not the inspector's root."""

    @pytest.fixture(scope="class")
    def viewer_js_path(self):
        # Post-stage-1.2: the inspector body lives in the shared
        # core, NOT in watch/viewer.js (which is now just a 48-line
        # /watch-page bootstrap).  The watch/viewer.js shape is
        # pinned separately in test_web.py::test_watch_viewer_js_is_only_the_bootstrap.
        p = (Path(__file__).resolve().parent.parent
             / "molbuilder" / "web" / "static" / "lib" / "trajectory" / "core.js")
        assert p.is_file(), f"missing {p}"
        return p

    @pytest.fixture(scope="class")
    def viewer_js(self, viewer_js_path):
        return viewer_js_path.read_text()

    def test_mountInspector_function_exists(self, viewer_js):
        """The inspector wrapper function must remain.  Signature
        may grow extra arguments (currently ``(rootEl, opts)``), so
        we match ``mountInspector(rootEl`` with optional trailing
        parameters."""
        assert re.search(
            r'\bfunction\s+mountInspector\s*\(\s*rootEl\b', viewer_js
        ), (
            "lib/trajectory/core.js no longer wraps its body in "
            "mountInspector(rootEl, ...); the root-scoped wrapper "
            "is load-bearing for /results-side mounts"
        )

    def test_core_module_does_NOT_auto_bootstrap(self, viewer_js):
        """POSITIVE PIN of the post-stage-1.2 design: the SHARED
        core module must NOT call ``mountInspector(document)`` (or
        register a DOMContentLoaded handler that does so) on its
        own.

        Why this is a feature, not a missing one: ``core.js`` is
        loaded by /watch AND /results.  If it self-mounted on page
        load, /results would try to mount the inspector against
        the ``document`` -- finding loader-bar ids (path-input,
        load-btn) that don't exist on /results, and racing the
        registry-side mount that's about to inject the partial
        into ``#inspector-host``.  The mount trigger belongs in
        the per-consumer bootstrap:

          * /watch:    static/watch/viewer.js (the 48-line
                       bootstrap file; tested in test_web.py)
          * /results:  static/lib/inspectors/trajectory.js (the
                       registry adapter; mounts on file-pick, not
                       on page load)

        A regression that re-introduces an auto-bootstrap here
        would silently break /results.  Pinning the negative
        keeps the boundary visible."""
        # The previous design had `mountInspector(document)` and
        # a DOMContentLoaded handler around it.  Both must be absent
        # from core.js after the lift.
        for forbidden in (
            r'\bmountInspector\s*\(\s*document\s*\)',
            r'addEventListener\s*\(\s*["\']DOMContentLoaded',
        ):
            sites = re.findall(forbidden, viewer_js)
            assert not sites, (
                f"lib/trajectory/core.js contains forbidden "
                f"pattern {forbidden!r} ({len(sites)} site(s)); "
                f"the core must NOT auto-bootstrap.  Move the "
                f"mount trigger to the appropriate per-consumer "
                f"bootstrap file (watch/viewer.js for /watch, "
                f"lib/inspectors/trajectory.js for /results)."
            )

    def test_scoped_dollar_uses_rootEl(self, viewer_js):
        """The scoped $() helper must close over rootEl, not
        document."""
        # The $() definition uses rootEl + querySelector.
        m = re.search(r'const\s+\$\s*=\s*\(id\)\s*=>\s*([^;]+);', viewer_js)
        assert m, "$ helper definition not found"
        body = m.group(1)
        assert "rootEl" in body, (
            f"$ helper does not use rootEl (got: {body!r}); "
            f"scoping is broken"
        )

    def test_no_direct_getElementById_inside_inspector(self, viewer_js):
        """Any ``document.getElementById`` call inside the inspector
        body bypasses the scoping helpers.  The ONLY allowed
        ``document.getElementById`` is inside the ``$doc`` helper
        itself."""
        # Find every document.getElementById call site.
        sites = re.findall(r'document\.getElementById\([^)]*\)', viewer_js)
        # Allow exactly one occurrence: the $doc helper's body.
        # Stage 1C may also add one inside _trajectory_core's
        # default $doc.  Either way, more than 1 means a leak.
        assert len(sites) == 1, (
            f"viewer.js has {len(sites)} document.getElementById "
            f"call(s); expected exactly one (the $doc helper "
            f"definition).  Sites: {sites}"
        )

    def test_scoped_ids_match_partial(self, viewer_js, partial_ids):
        """Every $("foo") call site must reference an id that's in
        the trajectory-inspector partial.  An id outside the partial
        means /results-side mounts will silently fail to find it."""
        called = set(re.findall(r'\$\("([a-zA-Z][\w-]*)"\)', viewer_js))
        # Names like '$3Dmol' should NOT match (they don't have the
        # leading `$(` syntax) -- the regex anchors on `$(`.
        outside = sorted(called - partial_ids)
        assert not outside, (
            f"$() called on ids that aren't in the partial: "
            f"{outside}.  These must use $doc() (page-level) or "
            f"be added to the partial."
        )

    def test_page_level_dollar_doc_helper_exists(self, viewer_js):
        """The page-level $doc helper must be defined and stay
        document-scoped (rootEl-independent) so the loader-bar
        handlers find their elements regardless of which page is
        hosting the inspector."""
        m = re.search(
            r'const\s+\$doc\s*=\s*\(id\)\s*=>\s*document\.getElementById\(id\)',
            viewer_js,
        )
        assert m, "$doc helper missing or no longer document-scoped"

    def test_page_level_ids_use_dollar_doc(self, viewer_js):
        """The four loader-bar ids (path-input + load-btn + status
        + file-picker) must go through $doc, never the scoped $."""
        for the_id in ("path-input", "load-btn", "status",
                       "file-picker"):
            scoped = re.search(
                r'\$\("' + re.escape(the_id) + r'"\)', viewer_js
            )
            assert scoped is None, (
                f"page-level id {the_id!r} is still being looked "
                f"up via the scoped $() (would miss it when the "
                f"inspector mounts inside a smaller rootEl)"
            )

    def test_no_unguarded_dollar_doc_dereference(self, viewer_js):
        """Stage 1D readiness: when /results' dispatcher calls
        ``mountInspector(panel)`` against a host that doesn't carry
        /watch's loader bar, every $doc lookup must return null
        without crashing.  The required pattern is:

            const _foo = $doc("foo");
            if (_foo) _foo.addEventListener(...);

        ANY direct ``$doc("X").<member>`` is an NPE waiting to
        happen.  This test fails the moment one creeps back in.
        """
        # Match the dangerous pattern: $doc("id").something
        # without a preceding `if (` on the same line (single-line
        # guards like `if (_foo) _foo.bar` ARE safe but use the
        # captured variable, not the literal $doc call).
        sites = re.findall(
            r'\$doc\("[\w-]+"\)\s*\.', viewer_js
        )
        assert sites == [], (
            f"viewer.js has {len(sites)} unguarded $doc dereference "
            f"site(s); each is an NPE risk when the inspector mounts "
            f"in a host without the loader bar.  Capture to a const "
            f"first + guard with `if (el)`."
        )


# --------------------------------------------------------------------- #
#  Public contract of lib/trajectory/core.js's mount() function.        #
#  Changes here require updating BOTH consumers (watch/viewer.js for    #
#  /watch and lib/inspectors/trajectory.js for /results).               #
# --------------------------------------------------------------------- #


class TestTrajectoryCoreMountContract:
    """The shared core's ``mount`` is the public API both /watch
    and /results depend on -- a change here requires updating
    BOTH consumers + the inspector registry's expectations.

      * signature ``mountInspector(rootEl, opts={file?})``
      * returns ``{dispose(), load(path)}``
      * dispose() tears down every long-lived resource (poll +
        playback timers, window resize listener, 3Dmol viewer
        contents) so rapid mount->dispose->mount cycles in the
        /results dispatcher don't leak setInterval handles, WebGL
        contexts, or window listeners.
      * ``window.molbuilder.trajectoryInspector.mount`` is THE
        exported entry point; consumers call it directly (no
        auto-bootstrap, see TestTrajectoryCoreRootScoping's
        ``test_core_module_does_NOT_auto_bootstrap``).

    These are static / string-pin checks; the runtime behaviour
    is exercised by Playwright E2E tests in
    ``tests/test_inspector_registry_e2e.py`` (mount/dispose
    cycle through the registry).
    """

    @pytest.fixture(scope="class")
    def viewer_js(self):
        p = (Path(__file__).resolve().parent.parent
             / "molbuilder" / "web" / "static" / "lib" / "trajectory" / "core.js")
        return p.read_text()

    def test_mountInspector_signature_takes_opts(self, viewer_js):
        assert "function mountInspector(rootEl, opts)" in viewer_js, (
            "mountInspector must take (rootEl, opts) so the "
            "registry-side trajectory inspector can pass "
            "{file: ...} to auto-load on mount"
        )

    def test_mountInspector_handles_opts_dot_file(self, viewer_js):
        assert "if (opts.file)" in viewer_js, (
            "mountInspector must auto-load opts.file when provided"
        )

    def test_mountInspector_returns_handle_with_dispose_and_load(
            self, viewer_js):
        # The handle is the API for /results' dispose-before-mount
        # contract.  Pin both required methods.
        assert "dispose()" in viewer_js
        assert "load(path) { return loadByPath(path); }" in viewer_js, (
            "the handle must expose load(path) so the registry-side "
            "inspector can swap files without a full re-mount"
        )

    def _dispose_body(self, viewer_js):
        """Return the full text of dispose()'s body, from ``dispose() {``
        through its matching closing ``},``.  Used by the contract
        tests below so the assertions are scoped to the dispose
        block without being pinned to a fixed window size that
        breaks when the body grows (today: the _cleanups walker
        comment adds ~10 lines)."""
        ix = viewer_js.find("dispose() {")
        assert ix > 0, "dispose() definition not found"
        # Scan forward + count braces.  Acceptable because dispose
        # body contains only balanced JS.  Stop at the closing },
        # of the dispose method (it's the immediate-next "},\n" at
        # depth 0 relative to dispose() {).
        depth = 0
        i = ix
        while i < len(viewer_js):
            ch = viewer_js[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # i points at the closing }; +1 to include it.
                    return viewer_js[ix:i + 1]
            i += 1
        raise AssertionError("dispose() body has unbalanced braces")

    def test_dispose_clears_poll_timer(self, viewer_js):
        """``stopPolling()`` is the canonical entry; dispose must call
        it or the 15 s background poll loop survives every file swap
        on /results."""
        body = self._dispose_body(viewer_js)
        assert "stopPolling()" in body, (
            "dispose() doesn't call stopPolling -- the 15 s poll "
            "loop would survive every file swap on /results"
        )

    def test_dispose_clears_play_timer(self, viewer_js):
        body = self._dispose_body(viewer_js)
        assert "clearInterval(state.playTimer)" in body, (
            "dispose() doesn't clear the playback timer -- the "
            "frame-step interval would survive every file swap"
        )

    def test_dispose_tears_down_listeners_via_cleanups(self, viewer_js):
        """The trajectory core uses an ``_on()`` helper that captures
        a teardown closure into ``_cleanups``; dispose() walks the
        array in reverse.  This covers the window.resize listener
        AND every element-level listener attached during mount --
        a stronger contract than the old single-line
        ``window.removeEventListener("resize", _onResize)``.

        Pin the contract pieces: _cleanups array exists, _on()
        helper exists, dispose() walks _cleanups, and the window
        resize listener is registered THROUGH _on (not raw)."""
        body = self._dispose_body(viewer_js)
        # dispose walks the cleanups array.
        assert "_cleanups.pop()" in body, (
            "dispose() doesn't walk _cleanups -- listeners attached "
            "during mount won't be removed.  See the spectra core's "
            "dispose for the reference pattern."
        )
        # Backbone exists at module scope.
        assert "const _cleanups = []" in viewer_js, (
            "_cleanups array missing from lib/trajectory/core.js"
        )
        assert "function _on(target, event, handler" in viewer_js, (
            "_on() helper missing from lib/trajectory/core.js"
        )
        # #236: the embed installs its own ResizeObserver on the
        # canvas host so the window resize listener went away with
        # the raw-viewer escape hatch.  Pin that the legacy wiring
        # is gone (no _on(window, "resize"...) registration).
        assert '_on(window, "resize"' not in viewer_js, (
            "window resize listener resurfaced -- the embed already "
            "owns canvas resize via ResizeObserver; remove the "
            "duplicate wiring"
        )

    def test_dispose_tears_down_3Dmol_viewer(self, viewer_js):
        """dispose() must tear down the 3Dmol WebGL state so one
        mount doesn't leak into the next.  Post-#236 the trajectory
        no longer holds a raw viewer reference; the embed handle
        carries the dispose responsibility and ``_handle.dispose()``
        is the single call that drops models + shapes + labels +
        ResizeObserver + animation loop in one go (matches the
        spectra and modify dispose paths)."""
        body = self._dispose_body(viewer_js)
        assert "_handle.dispose()" in body, (
            "dispose() doesn't call _handle.dispose() -- the embed's "
            "3Dmol bookkeeping (models + shapes + labels) leaks "
            "across mounts.  Spectra core uses the same pattern."
        )

    def test_module_exposes_trajectoryInspector_mount(self, viewer_js):
        # The Stage-1C lift requires the registry-side trajectory
        # inspector to delegate to /watch's mountInspector via the
        # exposed global.  Without this export the lift can't
        # happen without duplicating the code.
        assert "root.molbuilder.trajectoryInspector = {" in viewer_js
        assert "mount: mountInspector" in viewer_js, (
            "trajectoryInspector.mount export missing; the registry-"
            "side trajectory inspector module has no clean way to "
            "call this implementation"
        )

    def test_iife_invocation_passes_window(self, viewer_js):
        # The IIFE now takes ``root`` and is invoked with window
        # (or this in non-browser contexts).  Without this the
        # ``root.molbuilder = ...`` export would throw.
        assert "(function (root) {" in viewer_js
        assert 'typeof window !== "undefined" ? window : this' in viewer_js


# --------------------------------------------------------------------- #
#  Inspector placeholder XSS safety                                     #
# --------------------------------------------------------------------- #


class TestRegistryInspectorsNoStringConcatInInnerHTML:
    """The registry-side inspector modules MUST NOT build
    ``innerHTML`` via string concat or template-literal
    interpolation of the file path.  Why this remains a guard
    even after the trajectory inspector stopped being a
    placeholder: it now does ``host.innerHTML = partialHtml``
    where ``partialHtml`` is the trusted response of
    ``GET /partials/trajectory-inspector`` (a same-origin Jinja
    render).  That single trusted assignment is fine; what we
    forbid is ANY ``innerHTML = "x" + path + "y"`` pattern,
    which would re-introduce DOM XSS if a bad file path slipped
    past upstream validation.

    Spectra is still the placeholder shape and is also covered
    here.  Both inspectors get the same invariant pinned.
    """

    @pytest.fixture(scope="class")
    def inspectors_dir(self):
        return (Path(__file__).resolve().parent.parent
                / "molbuilder" / "web" / "static" / "lib" / "inspectors")

    @pytest.mark.parametrize("name", ["trajectory", "spectra"])
    def test_no_innerHTML_string_concat(self, inspectors_dir, name):
        body = (inspectors_dir / f"{name}.js").read_text()
        # Any innerHTML assignment that contains ``+`` (string concat)
        # OR a backtick (template literal interpolation) is the
        # forbidden pattern.  ``innerHTML = ""``, ``innerHTML = "literal"``,
        # and ``innerHTML = variable`` are all permitted -- only the
        # CONCAT / TEMPLATE patterns can interpolate untrusted data.
        sites = re.findall(
            r'\.innerHTML\s*=\s*[^"\';]*[+`][^;]*;',
            body,
        )
        bad = [s for s in sites if "+" in s or "`" in s]
        assert bad == [], (
            f"{name}.js has innerHTML assignments with string "
            f"concatenation or template-literal interpolation: "
            f"{bad}.  Use textContent + createElement (or a single "
            f"trusted server-rendered string, see "
            f"lib/inspectors/trajectory.js)."
        )

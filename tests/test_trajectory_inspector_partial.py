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
    # docs/web/molview.md) now owns those, and
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
        # MolView mount target (task #34): an EMPTY host div that
        # molview.mount fills with the whole fused card (viewer +
        # selection/Cell panel + view toggles + frame bar).  The
        # trajectory inspector no longer declares a #viewer div, a
        # frame strip, playback knobs (#speed / #loop), a unit-cell
        # toggle (#show-cell), an atom-list Inspect panel, or a
        # per-frame Save-XYZ button — MolView owns all of that.  The
        # ONLY trajectory-specific control left is the force-vector
        # producer (below).
        "viewer-host",
        # Run-state badge + compact runtime-info one-liner.
        "run-state-badge", "run-state-label", "run-state-detail",
        "runtime-summary",
        # Pre-data empty-state banner (batch A, 2026-06-14): rendered
        # by lib/trajectory/core.js::_renderEmptyStatus when the
        # parser surfaces zero frames or fails to attach an energy
        # series.  Cleared once state.data is populated.  See the
        # _lastEmptyStatus cache in core.js for the poll-tick guard.
        "trajectory-empty-status",
        # Parse-warnings panel (Level-3 parser, 2026-05-28).  Hidden
        # by default; shows when the parser hit non-fatal issues.
        "parse-warnings",
        "parse-warnings-count",
        "parse-warnings-list",
        # Force-vector PRODUCER PARAMETERS — the trajectory-specific
        # controls (task #34).  The inspector hands the ENGINE filtered
        # raw per-frame forces + drives the forceScale flag; the engine
        # builds + styles the arrows (gold max-highlight + magnitude
        # ramp, process.js §2.4).  Whether they're DRAWN is MolView's
        # "show overlay" view-toggle, so there is no show-forces /
        # highlight-max / status-readout control here — only the
        # scale + filter knobs.  #hide-frozen is a PURE force filter
        # (atom hiding in the viewer is MolView's selection/isolate
        # job); it zeroes frozen-atom forces so the engine draws no
        # arrow for them.
        "force-scale", "force-scale-val", "force-min",
        "hide-frozen",
        # SCF banner + plots.
        "scf-section", "scf-title", "scf-status",
        "energy-plot", "force-plot",
        "scf-energy-plot", "scf-gnorm-plot",
        # Convergence-targets summary band (task #362, 2026-06-12).
        # Populated by lib/trajectory/core.js::_renderConvergenceSummary
        # from Trajectory.runtime_info.convergence_targets — the
        # parser captures it from the SIESTA input echo / molwatch
        # header / PySCF script's _CONVERGENCE_TARGETS dict.  Hidden
        # by default; shows when the trajectory carries targets.
        "convergence-summary",
        "convergence-summary-source",
        "convergence-summary-targets",
        "convergence-summary-current",
        "convergence-summary-hint",
        # Per-trajectory CSV export button (compact legend + Export
        # all CSV row, 2026-06-12 result-plot polish).
        "trajectory-export-csv-btn",
    }

    # IDs the partial MUST NOT declare — they belong to the embed.
    # Chrome IDs (rep / radius / colorscheme / bg) moved to the
    # standard knob bar in #205; frame-strip IDs (prev / play /
    # pause / next / frame-slider / frame-idx / frame-tot) moved
    # to the embed's auto-mounted frame strip in #246 A1.  A
    # re-introduction here means a consumer hand-rolled chrome
    # or duplicated frame-strip UI that should go through the
    # embed APIs (setStyle / setBackground / setAnimation +
    # frame-strip auto-mount per § 6.3).
    EMBED_OWNED_IDS = {
        "rep", "radius", "colorscheme", "bg",
        "prev", "play", "pause", "next",
        "frame-slider", "frame-idx", "frame-tot",
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
            f"bar in #205 (see docs/web/molview.md "
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
    """Post-lift (task #76 / docs/web/results.md): the
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

    # Phase 5i retired the ``$doc`` helper.  At the time it existed
    # to keep page-level loader-bar ids (path-input / load-btn /
    # status / file-picker) addressable from inside the inspector
    # mount via document-scoped lookups, while the inspector's own
    # ids went through the rootEl-scoped ``$``.  After the /watch
    # retirement and the structure-inspector cleanup, the only
    # page-level lookup left was the ``status`` banner, and that
    # single call is now inlined as ``document.getElementById`` in
    # ``setStatus`` (see ``lib/trajectory/core.js``).  The three
    # tests this block used to host (``$doc`` helper exists,
    # page-level ids use ``$doc``, no-unguarded-deref) all enforced
    # an invariant the code no longer needs.  Removed in the same
    # commit that lands the transport + Makov-Payne items, where
    # the test sweep first surfaced the stale assertion.
    pass


# --------------------------------------------------------------------- #
#  Public contract of lib/trajectory/core.js's mount() function.        #
#  Changes here require updating BOTH consumers (watch/viewer.js for    #
#  /watch and lib/inspectors/trajectory.js for /results).               #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  RETIRED 2026-09-03 — TestTrajectoryCoreMountContract (9 tests)       #
#                                                                       #
#  Its own docstring said "These are static / string-pin checks; the    #
#  runtime behaviour is exercised by Playwright E2E tests in            #
#  tests/test_inspector_registry_e2e".  Per testing.md § 3a.1 that      #
#  admission is the verdict: the author knew what would verify the      #
#  contract and wrote something else.                                   #
#                                                                       #
#  Checked before deleting, because a cited replacement is a claim:     #
#    * handle shape, dispose clearing the host, listener add/remove     #
#      balance   -> already covered there, so those pins were pure      #
#      duplicates;                                                      #
#    * TIMER teardown -> NOT covered.  The listener spy watches         #
#      EventTarget, which a setInterval never touches, so the one       #
#      resource that leaks silently was the one nothing watched.        #
#      test_no_trajectory_poll_survives_dispose now drives it: mount an #
#      ongoing run, confirm the poll is live, dispose, assert cleared.  #
#      Mutation-verified against stopPolling().                         #
#                                                                       #
#  Two pins had no behaviour to move at all: one asserted the IIFE's    #
#  exact formatting, one asserted a deleted feature was still deleted.  #
# --------------------------------------------------------------------- #
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

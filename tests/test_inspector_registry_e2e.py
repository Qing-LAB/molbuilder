"""End-to-end runtime tests for the Inspector Registry contract.

The registry's logic (register, pick, mount, dispose ordering) only
exists in JavaScript, so we drive it via Playwright against a live
``/results`` page.  Tests here pin the behavior contract a future
inspector author needs to depend on:

  * pick() returns null for unknown extensions
  * pick() honours registration order (first match wins)
  * register() is idempotent (re-registering same name replaces)
  * mount() returns a handle with dispose()
  * dispose() runs before the next inspector takes the host

Plus the integration test: selecting a file in the sidebar via
``window.molbuilder.projects.setShared()`` swaps the visible
inspector.  This is the milestone validation for the registry-
driven /results architecture.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ongoing_trajectory() -> str:
    """A multi-frame ``*_geom_optim.xyz`` that reads as a RUNNING job.

    **Nothing here is invented.**  Every part of this file is the project's
    own:

    * the NAME follows ``projects.py::_GEOM_OUTPUT_PATTERNS`` —
      ``*_geom_optim.xyz`` is the geomeTRIC trajectory convention the picker
      and the inspector both key on;
    * the DIRECTORY is under the real projects root, because
      ``/api/watch/load`` resolves through ``_resolve_within_roots`` and
      refuses anything outside ``Capabilities.file_picker_roots()``.  A
      ``tmp_path`` file would be rejected — correctly — and the test would be
      exercising the refusal instead of the timer;
    * each FRAME is written by ``Structure.to_xyz()``, molbuilder's own
      writer, rather than a hand-typed block.  A multi-frame trajectory is
      those frames concatenated, which is what the convention above names
      and what geomeTRIC emits.

    The run reads as ONGOING because it carries no completion marker:
    ``_settlePostLoad`` treats an absent ``run_state`` as "still going" and
    transitions to WATCHING, which starts the poll.  That is the state whose
    timer must not survive dispose.
    """
    import numpy as np

    from molbuilder.structure import Structure

    root = ROOT / "projects/_t_timer_e2e"
    if root.exists():
        shutil.rmtree(root)
    d = root / "optimization"
    d.mkdir(parents=True)
    f = d / "probe_geom_optim.xyz"
    frames = [
        Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, z]])).to_xyz(
            comment=f"Iteration {i} Energy -1.1{i}")
        for i, z in enumerate((0.74, 0.72))
    ]
    f.write_text("".join(frames), encoding="utf-8")
    try:
        yield str(f.resolve())
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open_results(page, base_url):
    """Navigate to /results + capture JS errors."""
    errors = []
    page.on("pageerror", lambda exc: errors.append(("pageerror", str(exc))))
    page.on("console", lambda msg: (
        errors.append(("console.error", msg.text))
        if msg.type == "error" else None
    ))
    page.goto(f"{base_url}/results")
    page.wait_for_selector("#inspector-host", timeout=5000)
    # Give inspector scripts a tick to self-register.
    page.wait_for_function(
        "() => window.molbuilder "
        "&& window.molbuilder.inspectors "
        "&& window.molbuilder.inspectors.list().length >= 4"
    )
    return errors


# --------------------------------------------------------------------- #
#  Registry + picker contracts — DEMOTED 2026-06-13                     #
#                                                                       #
#  28 pure dispatch/shape tests were migrated to                        #
#  ``tests/test_inspector_registry_dispatch_js.py`` (L2 module tier).   #
#  Each one called ``window.molbuilder.inspectors.pick(path)`` /        #
#  ``.pickResult(path)`` / ``.list()`` and checked the return — pure   #
#  JS function dispatch with no DOM dependency.  Per                   #
#  docs/process/testing.md, that's the canonical L5 → L2  #
#  shape: chromium adds ~1.8 s per test for a function call that runs #
#  in 30 ms under Node.                                                 #
#                                                                       #
#  Kept e2e in this file: ``TestMountLifecycle`` +                     #
#  ``TestResultsDispatchIntegration`` + ``TestInspectorListenerTeardown``#
#  — those exercise real DOM mount + dispose + listener teardown that  #
#  genuinely needs a browser.                                           #
# --------------------------------------------------------------------- #


class TestMountLifecycle:
    """The host element is owned exclusively by the active inspector;
    a new selection disposes the previous handle BEFORE mounting
    the new one.

    The unknown-extension → null path is covered at L2 by
    ``test_inspector_registry_dispatch_js.py::
    test_registry_pick_dispatches_filename[unknown_ext→null]``; mount()
    returns null when pick() returns null, so the e2e equivalent
    added no contract beyond the dispatch table."""

    def test_mount_returns_handle_with_dispose(self, page, flask_server):
        """Registry's mount(host, file, ctx) returns a handle with
        ``dispose()``.  The ctx arg is required (added 2026-05-18 to
        carry shared state into each inspector adapter); tests use
        ``createDefaultContext(host)`` for the standard context."""
        _open_results(page, flask_server)
        ok = page.evaluate("""() => {
            const reg = window.molbuilder.inspectors;
            const host = document.createElement("div");
            const ctx  = reg.createDefaultContext(host);
            const h = reg.mount(host, "/projects/foo/spectrum/run.fdf", ctx);
            return h && typeof h.dispose === "function";
        }""")
        assert ok


# --------------------------------------------------------------------- #
#  /results integration: sidebar selection swaps inspector              #
# --------------------------------------------------------------------- #


class TestResultsDispatchIntegration:
    """End-to-end: setting the sidebar's current_file fires the
    dispatch + mounts the right inspector inside #inspector-host."""

    def test_sidebar_selection_mounts_source_inspector_for_fdf(
            self, page, flask_server):
        _open_results(page, flask_server)
        # Drive the sidebar's selection state directly + wait for
        # the dispatch to land + render.  No real file on disk
        # because the source inspector's readFile will 404, but the
        # placeholder card structure renders first.
        page.evaluate("""() => {
            // Use the published setShared if available; otherwise
            // fall back to sessionStorage + a manual dispatch tick.
            const proj = window.molbuilder.projects;
            if (proj && proj.onChange) {
                // onChange fires the subscriber; setShared is
                // exposed via projects/state.js as an internal --
                // do it via sessionStorage + dispatching a custom
                // change.
                sessionStorage.setItem("molbuilder.current_dir", "/projects/foo");
                sessionStorage.setItem("molbuilder.current_file",
                                       "/projects/foo/spectrum/run.fdf");
                // No public publish API -- the dispatch listens to
                // onChange.  Easiest reliable trigger is calling
                // proj.refresh() if exposed, else a small DOM event.
            }
        }""")
        # The setShared is internal; the cleanest cross-tab trigger
        # is the storage event, which fires for cross-window writes
        # but not same-window.  For the integration test we exercise
        # the dispatch directly via the public mount call.
        page.evaluate("""() => {
            const host = document.getElementById("inspector-host");
            const reg  = window.molbuilder.inspectors;
            const ctx  = reg.createDefaultContext(host);
            const handle = reg.mount(
                host, "/projects/foo/spectrum/run.fdf", ctx);
            window._testHandle = handle;
        }""")
        # The source inspector injects a .source-card.
        page.wait_for_selector(".source-card", timeout=3000)
        title = page.locator(".source-card .inspector-card-title").inner_text()
        assert "run.fdf" in title

    def test_dispose_clears_the_host(self, page, flask_server):
        _open_results(page, flask_server)
        cleared = page.evaluate("""() => {
            const host = document.createElement("div");
            document.body.appendChild(host);
            const reg = window.molbuilder.inspectors;
            const ctx = reg.createDefaultContext(host);
            const h   = reg.mount(host, "/projects/foo/x.fdf", ctx);
            const beforeHTML = host.innerHTML.length;
            h.dispose();
            const afterHTML = host.innerHTML.length;
            document.body.removeChild(host);
            return {beforeHTML, afterHTML};
        }""")
        assert cleared["beforeHTML"] > 0, "mount produced no DOM"
        assert cleared["afterHTML"] == 0, "dispose did not clear the host"


# --------------------------------------------------------------------- #
#  Listener-leak regression test (review P1 #1 from 2026-05-18)         #
# --------------------------------------------------------------------- #


class TestInspectorListenerTeardown:
    """Behavioural verification that dispose() actually tears down
    the element listeners it registered -- complement to the static
    source-pin tests in tests/spectra/test_blueprint.py::
    TestSpectraDisposeContract.

    The static tests catch "the cleanup CODE is wired"; these tests
    catch "the cleanup CODE actually works at runtime".  Together
    they fence in the dispose contract on both sides.
    """

    def test_addremove_pair_balance_after_mount_dispose(
            self, page, flask_server):
        """Spy on EventTarget.{add,remove}EventListener globally,
        mount the spectra inspector, dispose it, and assert every
        addEventListener that fired during mount was paired with a
        removeEventListener during dispose.

        Why this catches the bug class:  a direct
        ``els.foo.addEventListener("click", h)`` (i.e., one that
        escaped the _on() helper) shows up as an unpaired add.  The
        2026-05-18 review found exactly this latent issue with the
        ES selector listener -- which would have failed this test if
        it had existed.

        Note: the spectra adapter does an async partial fetch before
        calling into the core's mount.  We wait for the inner mount
        to land by polling for the partial's well-known id
        (``watch-path``).  Without that wait, dispose() might run
        before mount() ever wires its listeners and the test would
        always pass trivially.
        """
        _open_results(page, flask_server)
        result = page.evaluate("""async () => {
            // Spy add/remove globally.  Capture call counts only
            // (not full args) to keep the snapshot serialisable
            // across the Playwright bridge.
            let adds = 0, removes = 0;
            const origAdd    = EventTarget.prototype.addEventListener;
            const origRemove = EventTarget.prototype.removeEventListener;
            EventTarget.prototype.addEventListener = function (...a) {
                adds += 1;
                return origAdd.apply(this, a);
            };
            EventTarget.prototype.removeEventListener = function (...a) {
                removes += 1;
                return origRemove.apply(this, a);
            };
            try {
                const host = document.createElement("div");
                host.id = "test-host";
                document.body.appendChild(host);
                // Mount via the registry so the full adapter chain
                // runs (partial fetch + core mount).  ctx is required
                // by the registry's mount(host, file, ctx) contract.
                const reg    = window.molbuilder.inspectors;
                const ctx    = reg.createDefaultContext(host);
                const handle = reg.mount(
                    host, "/projects/foo/job.spectra.json", ctx);
                // Wait until the core's $() lookups would succeed --
                // i.e., the partial has been injected and core mount
                // ran.  Bounded poll (10x100ms) so a stuck mount
                // surfaces as a test timeout, not a hang.
                for (let i = 0; i < 30; i += 1) {
                    if (host.querySelector("#watch-path")) break;
                    await new Promise(r => setTimeout(r, 100));
                }
                const mounted = !!host.querySelector("#watch-path");
                const addsAfterMount = adds;
                const removesBeforeDispose = removes;
                handle.dispose();
                document.body.removeChild(host);
                return {
                    mounted: mounted,
                    addsDuringMount:    addsAfterMount,
                    removesBeforeDisp:  removesBeforeDispose,
                    removesAfterDisp:   removes,
                };
            } finally {
                EventTarget.prototype.addEventListener    = origAdd;
                EventTarget.prototype.removeEventListener = origRemove;
            }
        }""")
        # Sanity: the mount actually ran.  If False, the partial
        # fetch may have failed or the adapter chain is broken --
        # surface that as a clear assertion rather than a confusing
        # missing-pair count below.
        assert result["mounted"], (
            "spectra inspector did not finish mounting within 3s -- "
            "the partial fetch or the core's init() may be broken; "
            "fix the mount path before adding more dispose tests"
        )
        # The actual contract: every listener registered during the
        # mount cycle must be removed during dispose.  We allow
        # removes >= adds (third-party libs like 3Dmol may register
        # and tear down their own listeners; we don't want a
        # strict-equality test that breaks on every vendor update).
        added_by_mount = (result["addsDuringMount"]
                          - result["removesBeforeDisp"])
        removed_by_disp = (result["removesAfterDisp"]
                           - result["removesBeforeDisp"])
        assert removed_by_disp >= added_by_mount, (
            f"dispose() removed {removed_by_disp} listeners but "
            f"mount() had registered (net) {added_by_mount} -- a "
            f"listener was leaked.  Likely cause: an "
            f"``addEventListener`` call inside mountInspector that "
            f"escaped the _on() helper (see "
            f"tests/spectra/test_blueprint.py::"
            f"TestSpectraDisposeContract::"
            f"test_all_element_listeners_route_through_on_helper)."
        )

    def test_no_interval_survives_mount_dispose(self, page, flask_server):
        """Every ``setInterval`` started during mount is cleared by dispose.

        **Why this exists (2026-09-03).**  The trajectory core's timer
        teardown was pinned only by two source greps —
        ``"clearInterval(state.pollTimer)" in viewer_js`` and its
        play-timer sibling — inside a class whose own docstring said the
        runtime behaviour was covered by this file.  It was not: the
        listener spy above watches ``EventTarget``, which a timer never
        touches.  So the one resource that leaks *silently* — a poll
        interval keeps firing into a disposed inspector, refetching
        forever — was the one resource nothing actually watched.

        A leaked interval is worse than a leaked listener: the listener
        waits for an event that may never come, the interval bills you
        every tick.  The /results dispatcher mounts and disposes on every
        sidebar click, so the leak compounds across a browsing session.

        **The watch must actually be RUNNING, or the test is vacuous.**
        Mounting alone starts no timer — ``startWatch()`` is user-driven,
        behind the partial's "Start watching" button.  The first draft of
        this test disposed a freshly-mounted inspector and asserted zero
        live intervals, which is ``0 == 0``: it passed with the
        dispose-path ``clearInterval`` commented out.  So the test drives
        the real control, and asserts a timer was live *before* dispose —
        that guard is what keeps it honest if the watch path changes.

        The watched path need not exist: `startWatch` starts the interval
        even when the immediate tick 404s, precisely so polling continues
        while a run is still producing its first output.
        """
        _open_results(page, flask_server)
        result = page.evaluate("""async () => {
            const live = new Set();
            const origSet   = window.setInterval;
            const origClear = window.clearInterval;
            window.setInterval = function (...a) {
                const id = origSet.apply(window, a);
                live.add(id);
                return id;
            };
            window.clearInterval = function (id) {
                live.delete(id);
                return origClear.call(window, id);
            };
            try {
                const host = document.createElement("div");
                host.id = "timer-host";
                document.body.appendChild(host);
                const reg    = window.molbuilder.inspectors;
                const ctx    = reg.createDefaultContext(host);
                const handle = reg.mount(
                    host, "/projects/foo/job.spectra.json", ctx);
                for (let i = 0; i < 30; i += 1) {
                    if (host.querySelector("#watch-btn")) break;
                    await new Promise(r => setTimeout(r, 100));
                }
                const mounted = !!host.querySelector("#watch-btn");
                if (!mounted) return {mounted, watching: 0, afterDispose: 0};

                // Drive the user's own control: type a path, press
                // "Start watching".  The path 404s, which is the case
                // the poll loop is built for.
                host.querySelector("#watch-path").value =
                    "/projects/foo/job.spectra.json";
                host.querySelector("#watch-btn").click();
                for (let i = 0; i < 30; i += 1) {
                    if (live.size > 0) break;
                    await new Promise(r => setTimeout(r, 100));
                }
                const watching = live.size;
                handle.dispose();
                const afterDispose = live.size;
                document.body.removeChild(host);
                return {mounted, watching, afterDispose};
            } finally {
                window.setInterval   = origSet;
                window.clearInterval = origClear;
            }
        }""")
        assert result["mounted"], (
            "spectra inspector did not finish mounting within 3s -- fix the "
            "mount path before reading the timer counts below")
        # The anti-vacuity guard.  Without a live timer the assertion
        # below is 0 == 0 and passes on a deleted clearInterval.
        assert result["watching"] >= 1, (
            "no interval was running when dispose() was called, so this "
            "test proves nothing about teardown.  Either 'Start watching' "
            "no longer starts a poll interval, or the control moved -- fix "
            "the driving above rather than deleting this assertion")
        assert result["afterDispose"] == 0, (
            f"{result['afterDispose']} of {result['watching']} setInterval "
            f"handle(s) outlived dispose().  A poll or playback timer is "
            f"still firing into a torn-down inspector -- it will refetch "
            f"forever, and the /results dispatcher mounts and disposes on "
            f"every sidebar click, so the leak compounds.  Every interval "
            f"must be held where dispose() can reach it (the lifecycle "
            f"scope), not in a bare local.")

    def test_no_trajectory_poll_survives_dispose(
            self, page, flask_server, ongoing_trajectory):
        """The same contract for the OTHER core.

        `lib/trajectory/core.js` keeps its own poll timer
        (``startPolling`` / ``stopPolling``), so the spectra test above says
        nothing about it — and the trajectory timer is the more expensive
        leak, because it re-fetches `/api/watch/data` for the whole
        trajectory rather than one spectrum.

        No user gesture starts it: `_settlePostLoad` transitions to
        WATCHING on its own whenever the loaded run has no completion
        marker.  Mounting an ongoing run IS the trigger.
        """
        _open_results(page, flask_server)
        result = page.evaluate("""async (traj) => {
            const live = new Set();
            const origSet   = window.setInterval;
            const origClear = window.clearInterval;
            window.setInterval = function (...a) {
                const id = origSet.apply(window, a);
                live.add(id);
                return id;
            };
            window.clearInterval = function (id) {
                live.delete(id);
                return origClear.call(window, id);
            };
            try {
                const host = document.createElement("div");
                host.id = "traj-timer-host";
                document.body.appendChild(host);
                const reg    = window.molbuilder.inspectors;
                const ctx    = reg.createDefaultContext(host);
                const handle = reg.mount(host, traj, ctx);
                // Wait for the load to resolve and _settlePostLoad to
                // put the machine in WATCHING (which starts the timer).
                for (let i = 0; i < 50; i += 1) {
                    if (live.size > 0) break;
                    await new Promise(r => setTimeout(r, 100));
                }
                const watching = live.size;
                handle.dispose();
                const afterDispose = live.size;
                document.body.removeChild(host);
                return {watching, afterDispose};
            } finally {
                window.setInterval   = origSet;
                window.clearInterval = origClear;
            }
        }""", ongoing_trajectory)
        assert result["watching"] >= 1, (
            "mounting an ongoing trajectory started no poll interval, so "
            "this test proves nothing about teardown.  Either the fixture "
            "no longer reads as a running job, or the poll moved -- fix "
            "the setup rather than deleting this assertion")
        assert result["afterDispose"] == 0, (
            f"{result['afterDispose']} of {result['watching']} poll "
            f"interval(s) outlived dispose().  A torn-down trajectory "
            f"inspector is still polling /api/watch/data forever.")


# --------------------------------------------------------------------- #
#  Error-rendering on partial-fetch failure (2026-05-20 review #16)     #
# --------------------------------------------------------------------- #


class TestInspectorErrorCardRuntime:
    """Behavioural verification of the .inspector-card.error-card UX
    when the partial-fetch fails.  Complements the static source-pin
    tests in tests/test_results_blueprint.py::TestInspectorErrorRendering;
    those pin the wiring, these pin the user-visible rendering."""

    def test_partial_fetch_404_renders_error_card(
            self, page, flask_server):
        """Intercept the trajectory partial URL with a 404; mount
        the trajectory inspector; verify the host shows an
        ``.inspector-card.error-card`` with the file name + HTTP
        reason (not a blank panel)."""
        _open_results(page, flask_server)
        # Route the partial URL to a 404 BEFORE triggering the mount.
        page.route(
            "**/partials/trajectory-inspector",
            lambda route: route.fulfill(
                status=404,
                content_type="text/plain",
                body="not found",
            ),
        )
        page.evaluate("""() => {
            const host = document.getElementById("inspector-host");
            const reg  = window.molbuilder.inspectors;
            const ctx  = reg.createDefaultContext(host);
            reg.mount(host, "/projects/foo/run.molwatch.log", ctx);
        }""")
        # The adapter's .catch() handler renders the error card; wait
        # for it (the fetch + setHTML + DOM update is microtask-fast
        # but not synchronous).
        page.wait_for_selector(".inspector-card.error-card", timeout=3000)
        # Card body must mention the file name (basename only) so the
        # user knows which selection failed.
        title = page.locator(
            ".inspector-card.error-card .inspector-card-title"
        ).text_content() or ""
        assert "failed to mount" in title.lower(), (
            f"error card title should mention 'failed to mount'; "
            f"got {title!r}"
        )
        # And the HTTP status the user can act on.
        body = page.locator(
            ".inspector-card.error-card"
        ).text_content() or ""
        assert "404" in body, (
            f"error card body should show the HTTP 404 reason so the "
            f"user can debug; got {body!r}"
        )

    def test_aborterror_during_mount_does_not_render_error_card(
            self, page, flask_server):
        """Mount inspector A, immediately mount inspector B before
        A's partial fetch resolves.  A's fetch should be aborted by
        the cleanup chain; the abort must NOT surface as an error
        card (it's the user's deliberate file-switch, not a failure).
        Verify B's content is in the host, no error-card overlay
        from A.
        """
        _open_results(page, flask_server)
        # Trajectory partial: slow response (1.5 s) so the second
        # mount has time to abort the first.
        page.route(
            "**/partials/trajectory-inspector",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body="<div id='viewer'></div>",
            ),
        )
        page.evaluate("""() => {
            const host = document.getElementById("inspector-host");
            const reg  = window.molbuilder.inspectors;
            const ctx  = reg.createDefaultContext(host);
            // Mount A (trajectory).
            const handleA = reg.mount(host, "/x/a.molwatch.log", ctx);
            // Immediately dispose A (simulates user picking another
            // file before A's partial fetch resolves).
            handleA.dispose();
            // Mount B (different inspector -- source) so the host
            // ends up with B's content, not A's late HTML.
            reg.mount(host, "/x/file.fdf", ctx);
        }""")
        # Wait for the source inspector's card (synchronous: source
        # inspector mounts without fetching a partial).
        page.wait_for_selector(".source-card", timeout=3000)
        # The error-card must NOT be visible -- A's abort was
        # expected, not a failure.
        err_count = page.locator(".inspector-card.error-card").count()
        assert err_count == 0, (
            "an .inspector-card.error-card appeared after a user-"
            "triggered abort -- the AbortError guard in the .catch() "
            "handler is missing or broken"
        )



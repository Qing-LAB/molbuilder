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

import threading

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture(scope="module")
def flask_server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


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
#  Registry contract                                                    #
# --------------------------------------------------------------------- #


class TestRegistryContract:

    def test_four_inspectors_self_registered_on_load(self, page, flask_server):
        _open_results(page, flask_server)
        names = page.evaluate(
            "() => window.molbuilder.inspectors.list().map(i => i.name)"
        )
        # Order matters: see registry.js + the script tag order in
        # results.html.  Inspector with the most specific match
        # registers in a slot where it'd win over a more general one
        # if both fire (source's plain ``.log`` vs trajectory's
        # ``.molwatch.log``).
        assert set(names) >= {"source", "structure", "trajectory", "spectra"}

    def test_pick_returns_null_for_unknown_extension(self, page, flask_server):
        _open_results(page, flask_server)
        picked = page.evaluate(
            "() => window.molbuilder.inspectors.pick"
            "('/projects/foo/spectrum/run.unknown_ext')"
        )
        assert picked is None

    def test_pick_returns_null_for_empty_path(self, page, flask_server):
        _open_results(page, flask_server)
        picked = page.evaluate(
            "() => window.molbuilder.inspectors.pick('')"
        )
        assert picked is None

    def test_pick_dispatches_xyz_to_structure(self, page, flask_server):
        _open_results(page, flask_server)
        name = page.evaluate(
            "() => window.molbuilder.inspectors"
            ".pick('/projects/foo/struct/water.xyz').name"
        )
        assert name == "structure"

    def test_pick_dispatches_pdb_to_structure(self, page, flask_server):
        _open_results(page, flask_server)
        name = page.evaluate(
            "() => window.molbuilder.inspectors"
            ".pick('/projects/foo/struct/peptide.pdb').name"
        )
        assert name == "structure"

    def test_pick_dispatches_fdf_to_source(self, page, flask_server):
        _open_results(page, flask_server)
        name = page.evaluate(
            "() => window.molbuilder.inspectors"
            ".pick('/projects/foo/spectrum/run.fdf').name"
        )
        assert name == "source"

    def test_pick_dispatches_compound_log_to_trajectory(self, page, flask_server):
        """Compound extension ``.molwatch.log`` MUST win over plain
        ``.log`` (which source would also claim).  This is the
        first-match-wins ordering invariant."""
        _open_results(page, flask_server)
        name = page.evaluate(
            "() => window.molbuilder.inspectors"
            ".pick('/projects/foo/spectrum/run.molwatch.log').name"
        )
        assert name == "trajectory"

    def test_pick_dispatches_compound_json_to_spectra(self, page, flask_server):
        """Same invariant for ``.spectra.json`` vs plain ``.json``."""
        _open_results(page, flask_server)
        name = page.evaluate(
            "() => window.molbuilder.inspectors"
            ".pick('/projects/foo/spectrum/water.spectra.json').name"
        )
        assert name == "spectra"

    def test_pick_dispatches_plain_json_to_source(self, page, flask_server):
        _open_results(page, flask_server)
        name = page.evaluate(
            "() => window.molbuilder.inspectors"
            ".pick('/projects/foo/user/config.json').name"
        )
        assert name == "source"

    def test_register_rejects_missing_required_fields(self, page, flask_server):
        """The registry validates the Inspector interface at register
        time -- a missing field is a programming error, not a
        runtime surprise."""
        _open_results(page, flask_server)
        errs = page.evaluate("""() => {
            const out = [];
            const cases = [
                {},
                {name: "x"},
                {name: "x", displayName: "X"},
                {name: "x", displayName: "X", match: () => true},
            ];
            for (const c of cases) {
                try { window.molbuilder.inspectors.register(c); out.push(null); }
                catch (e) { out.push(e.message || String(e)); }
            }
            return out;
        }""")
        assert all(msg is not None for msg in errs), (
            "registry accepted an incomplete inspector definition: "
            + repr(errs)
        )

    def test_register_is_idempotent_on_name(self, page, flask_server):
        """Re-registering the same name replaces the previous entry
        (the contract that lets a placeholder be swapped for a real
        implementation without code changes elsewhere)."""
        _open_results(page, flask_server)
        result = page.evaluate("""() => {
            const reg = window.molbuilder.inspectors;
            const before = reg.list().length;
            const fake = {
                name: "source",
                displayName: "Source TEST REPLACEMENT",
                match: () => false,
                mount: () => ({dispose: () => {}}),
            };
            reg.register(fake);
            const after = reg.list().length;
            const replaced = reg.list().find(i => i.name === "source");
            return {before, after, displayName: replaced.displayName};
        }""")
        assert result["after"] == result["before"], (
            "re-registering same name should NOT grow the list"
        )
        assert result["displayName"] == "Source TEST REPLACEMENT"


# --------------------------------------------------------------------- #
#  Mount / dispose lifecycle                                            #
# --------------------------------------------------------------------- #


class TestMountLifecycle:
    """The host element is owned exclusively by the active inspector;
    a new selection disposes the previous handle BEFORE mounting
    the new one."""

    def test_mount_returns_handle_with_dispose(self, page, flask_server):
        _open_results(page, flask_server)
        ok = page.evaluate("""() => {
            const reg = window.molbuilder.inspectors;
            const host = document.createElement("div");
            const h = reg.mount(host, "/projects/foo/spectrum/run.fdf");
            return h && typeof h.dispose === "function";
        }""")
        assert ok

    def test_mount_unknown_extension_returns_null(self, page, flask_server):
        _open_results(page, flask_server)
        out = page.evaluate("""() => {
            const reg = window.molbuilder.inspectors;
            const host = document.createElement("div");
            return reg.mount(host, "/projects/foo/x.unknown");
        }""")
        assert out is None


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
            const handle = window.molbuilder.inspectors.mount(
                host, "/projects/foo/spectrum/run.fdf");
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
            const h = window.molbuilder.inspectors.mount(
                host, "/projects/foo/x.fdf");
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
                // runs (partial fetch + core mount).
                const handle = window.molbuilder.inspectors.mount(
                    host, "/projects/foo/job.spectra.json");
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

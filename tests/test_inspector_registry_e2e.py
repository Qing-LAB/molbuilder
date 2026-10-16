"""The inspector registry (`web/presenters.md`) on a live ``/results``
page, where alone it exists -- in JavaScript, with the presenters the page
registered, in the order it registered them:

  * which viewer opens a file, which lists it in the dropdown, and under
    which heading -- each row of ``tests/data/presenters.toml``;
  * a viewer mounted draws into its host, and its dispose empties it;
  * opening a viewer again and again leaves on the page what once left;
  * a viewer whose file cannot load says why, and lets the host go.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
PRESENTERS = tomllib.loads(
    (ROOT / "tests" / "data" / "presenters.toml").read_text())


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
#  Which viewer opens a file -- in the order the page registered them    #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def registry_answers(browser, flask_server):
    """Every row of `tests/data/presenters.toml`, asked of the registry on
    the live `/results` page in one visit, once every presenter the rows
    name has registered: ``{table: {row name: answer}}``."""
    page = browser.new_page()
    try:
        page.goto(f"{flask_server}/results")
        named = sorted({r[k] for tbl, k in (("pick", "opens"),
                                             ("result", "lists"))
                        for r in PRESENTERS[tbl] if k in r})
        page.wait_for_function(
            "(names) => { const reg = (window.molbuilder || {}).inspectors;"
            "  return !!reg && names.every(n =>"
            "    reg.list().some(i => i.name === n)); }",
            arg=named, timeout=10000)
        return page.evaluate("""(t) => {
            const reg = window.molbuilder.inspectors;
            const name = (p) => p ? p.name : null;
            const out = { pick: {}, result: {}, heading: {} };
            for (const r of t.pick) out.pick[r.name] = name(reg.pick(r.file));
            for (const r of t.result)
                out.result[r.name] = name(reg.pickResult(r.file));
            for (const r of t.heading) {
                const p = reg.pickResult(r.file, r.meta || null);
                out.heading[r.name] = p ? p.resultCategory(r.file, r.meta || null)
                                        : null;
            }
            return out; }""",
            {k: PRESENTERS[k] for k in ("pick", "result", "heading")})
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
    """`presenters.md` § 2: the heading names the engine when, and only
    when, the directory's answer does; the spelling is what the person
    reads."""
    assert registry_answers["heading"][row["name"]] == row.get("heading")


# --------------------------------------------------------------------- #
#  Mount lifecycle -- the real DOM mount, dispose and listener teardown  #
# --------------------------------------------------------------------- #


class TestMountLifecycle:
    """A viewer put away takes what it drew with it: the Results tab puts
    the old viewer away before it mounts the next (`presenters.md` § 3,
    step 2), and the next draws into an empty host.  (A file no presenter
    claims opens in nothing: the table above, `presenters.toml`'s last two
    `pick` rows.)"""

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
    """A viewer put away leaves nothing listening: a listener left on a
    document or a window after its viewer is gone fires against a dead
    panel, on every later mount (`presenters.md` § 3, step 2)."""

    @pytest.mark.parametrize("viewer,file", [
        ("structure", "water.xyz"),
        ("trajectory", "relax_optim.xyz"),
        ("spectra", "job.spectra.json"),
    ])
    def test_opening_a_viewer_again_and_again_leaves_what_once_left(
            self, page, flask_server, isolated_projects_root, viewer, file):
        """Opening and closing a viewer any number of times leaves on the
        page what opening and closing it once left: the listeners on the
        window, the document or an element still on the page, and the live
        GL contexts (`molview.md` § 8, § 9.9a; `presenters.md` § 3, step 2).

        THE SILENT FAILURES: a listener a mount adds and its dispose leaves,
        firing against a dead panel on every later mount (2026-08-23: the
        shared scope took the registrations while each core drained an old
        array of its own, 19 left a mount); a drawing window made for every
        file opened, each left alive with its listeners for the page's life
        (2026-10-15: twenty of twenty).

        Paired add to remove on ``EventTarget`` itself, so a listener added
        outside the viewers' shared scope (`lib/inspectors/lifecycle.js`)
        counts as surely as one inside it, on a settled page, each mount
        waited out by the viewer's own word that its load has ended
        (`EVENT_INSPECTOR_READY`).

        MUTATIONS THIS MUST FAIL AGAINST: a listener added outside the scope;
        a drawing window made for every mount."""
        folder = isolated_projects_root.resolve() / "P" / "s"
        folder.mkdir(parents=True)
        (folder / "water.xyz").write_text(
            "3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
        (folder / "relax_optim.xyz").write_text("".join(
            f"3\nstep {k}\nO 0 0 0\nH {0.957 + 0.01 * k:.3f} 0 0\n"
            "H -0.239 0.927 0\n" for k in range(3)))
        path = str(folder / file)
        page.goto(f"{flask_server}/results", wait_until="networkidle")
        page.wait_for_function(
            "(c) => { const r = (window.molbuilder || {}).inspectors;"
            "  return !!r && (r.pick(c.file) || {}).name === c.viewer; }",
            arg={"file": path, "viewer": viewer}, timeout=10000)
        page.wait_for_selector(".ps-entry, .ps-list-error, #ps-list.is-empty",
                               state="attached", timeout=10000)
        left = page.evaluate("""async (c) => {
            const reg = window.molbuilder.inspectors;
            const live = [];
            const contexts = [];
            const name = (t) => t === window ? "window"
                : t === document ? "document"
                : (t && t.nodeName) ? t.nodeName + (t.id ? "#" + t.id : "")
                : String(t);
            const add = EventTarget.prototype.addEventListener;
            const remove = EventTarget.prototype.removeEventListener;
            const getContext = HTMLCanvasElement.prototype.getContext;
            EventTarget.prototype.addEventListener = function (type, fn, o) {
                const by = String(new Error().stack || "").split("\\n")
                    .slice(2, 4).map((s) => s.trim()).join(" | ");
                live.push([this, type, fn, by]);
                return add.call(this, type, fn, o); };
            EventTarget.prototype.removeEventListener = function (type, fn, o) {
                const i = live.findIndex(([t, ty, f]) =>
                    t === this && ty === type && f === fn);
                if (i >= 0) live.splice(i, 1);
                return remove.call(this, type, fn, o); };
            HTMLCanvasElement.prototype.getContext = function (kind, ...a) {
                const g = getContext.call(this, kind, ...a);
                if (g && /webgl/.test(kind) && !contexts.includes(g))
                    contexts.push(g);
                return g; };
            const outlives = (t) => t === window || t === document
                || (t instanceof Node && t.isConnected);
            const state = () => ({
                listeners: live.filter(([t]) => outlives(t))
                    .map(([t, ty, , by]) => name(t) + ":" + ty + " <= " + by)
                    .sort(),
                contexts: contexts.filter((g) => !g.isContextLost()).length });
            const cycle = async () => {
                const host = document.createElement("div");
                host.style.width = "900px";
                document.body.appendChild(host);
                // The test's own wait, kept out of what it counts.
                const ended = new Promise((resolve) => add.call(document,
                    window.molbuilder.constants.EVENT_INSPECTOR_READY,
                    () => resolve(true), { once: true }));
                const handle = reg.mount(host, c.file,
                                         reg.createDefaultContext(host));
                const done = await Promise.race([ended,
                    new Promise((r) => setTimeout(() => r(false), 8000))]);
                handle.dispose();
                document.body.removeChild(host);
                return done;
            };
            try {
                if (!(await cycle())) return { error: "the first never ended" };
                const once = state();
                for (let i = 1; i < c.times; i++) {
                    if (!(await cycle())) return { error: "a mount never ended" };
                }
                return { once, again: state() };
            } finally {
                EventTarget.prototype.addEventListener = add;
                EventTarget.prototype.removeEventListener = remove;
                HTMLCanvasElement.prototype.getContext = getContext;
            }
        }""", {"file": path, "times": 5})
        assert "error" not in left, left
        grew = [x for x in left["again"]["listeners"]
                if left["again"]["listeners"].count(x)
                > left["once"]["listeners"].count(x)]
        assert not grew, (
            f"opening the {viewer} viewer five times left more on the page "
            f"than opening it once:\n  " + "\n  ".join(sorted(set(grew))))
        assert left["again"]["contexts"] == left["once"]["contexts"], (
            f"the {viewer} viewer left {left['once']['contexts']} live GL "
            f"context(s) opened once and {left['again']['contexts']} opened "
            f"five times")

    def test_a_viewer_whose_file_cannot_load_says_why_and_lets_go(
            self, page, flask_server, isolated_projects_root_module):
        """A load that fails has ENDED, and says so twice: the viewer's own
        status line carries the reason, and the ready signal goes -- so the
        tab's cover and the picker's "Parsing…" line lift at once instead
        of sitting over the error for their safety timers (`web/results.md`
        § 3; the Results-tab review, 2026-09-28).  The trajectory panel's
        line keeps its own layout class through the write.

        MUTATIONS THIS MUST FAIL AGAINST: spectra's refused load with no
        signal; the one status writer replacing the line's whole class list
        (it stripped `trajectory-status` on the first write).
        """
        gone = isolated_projects_root_module / "gone"
        gone.mkdir(parents=True, exist_ok=True)
        files = {"trajectory": str(gone / "vanished.molwatch.log"),
                 "spectra": str(gone / "vanished.spectra.json")}
        _open_results(page, flask_server)
        out = page.evaluate("""async (files) => {
            const C = window.molbuilder.constants;
            const got = [];
            const onReady = (e) => got.push(e.detail || {});
            document.addEventListener(C.EVENT_INSPECTOR_READY, onReady);
            const reg = window.molbuilder.inspectors;
            const out = {};
            for (const [name, file] of Object.entries(files)) {
                const host = document.createElement("div");
                document.body.appendChild(host);
                got.length = 0;
                const t0 = performance.now();
                const h = reg.mount(host, file, reg.createDefaultContext(host));
                for (let i = 0; i < 60; i += 1) {
                    if (got.some(d => d.inspector === name)) break;
                    await new Promise(r => setTimeout(r, 100));
                }
                const line = host.querySelector(name === "trajectory"
                    ? "#trajectory-status" : "#watch-status");
                out[name] = {
                    signal: got.find(d => d.inspector === name) || null,
                    ms: Math.round(performance.now() - t0),
                    text: line ? line.textContent : null,
                    cls: line ? line.className : null };
                h.dispose();
                host.remove();
            }
            document.removeEventListener(C.EVENT_INSPECTOR_READY, onReady);
            return out;
        }""", files)
        for name in ("trajectory", "spectra"):
            got = out[name]
            assert got["signal"] and got["signal"].get("error"), (name, got)
            assert got["ms"] < 6000, (name, got)
            assert got["text"], (name, got)
            assert "error" in got["cls"].split(), (name, got)
        assert "trajectory-status" in out["trajectory"]["cls"].split(), out


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

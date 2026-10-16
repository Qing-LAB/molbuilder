"""The inspector registry (`web/presenters.md`) on a live ``/results``
page, where alone it exists -- in JavaScript, with the presenters the page
registered, in the order it registered them:

  * which viewer opens a file, which lists it in the dropdown, and under
    which heading -- each row of ``tests/data/presenters.toml``;
  * opening a viewer again and again, or switching files in the dropdown,
    leaves on the page what once left;
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
    name has registered -- each with its row's ``meta``, as the dropdown
    passes the directory's answer: ``{table: {row name: answer}}``.

    A row's answer is one the server can give: its role one the catalogue
    declares, its parser one the registry holds."""
    from molbuilder.parse.registry import _FILE_PARSERS
    from molbuilder.runfiles import roles
    parsers = {k.name for k in _FILE_PARSERS}
    for tbl in ("pick", "result", "heading"):
        for r in PRESENTERS[tbl]:
            meta = r.get("meta", {})
            assert meta.get("role") in (None, *roles()), (r["name"], meta)
            assert meta.get("parser") in (None, *parsers), (r["name"], meta)
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
            for (const r of t.pick)
                out.pick[r.name] = name(reg.pick(r.file, r.meta || null));
            for (const r of t.result)
                out.result[r.name] = name(reg.pickResult(r.file, r.meta || null));
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
#  Listener-leak regression test (review P1 #1 from 2026-05-18)         #
# --------------------------------------------------------------------- #


class TestInspectorListenerTeardown:
    """A viewer put away leaves on the page no more than opening it once
    left: a listener left on a document or a window after its viewer is
    gone fires against a dead panel, on every later mount (`presenters.md`
    § 3, step 2)."""

    @pytest.mark.parametrize("viewer,file,loads", [
        ("structure", "water.xyz", True),
        ("spectra", "job.spectra.json", False),
    ])
    def test_opening_a_viewer_again_and_again_leaves_what_once_left(
            self, page, flask_server, isolated_projects_root, viewer, file,
            loads):
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

        Paired add to remove on ``EventTarget`` itself, as the browser pairs
        them (`support/leftovers.SPY`), so a listener added outside the
        viewers' shared scope (`lib/inspectors/lifecycle.js`) counts as
        surely as one inside it, on a settled page, each mount waited out by
        the viewer's own word that its load has ended
        (`EVENT_INSPECTOR_READY`) -- the structure loaded from its file, the
        word carrying no error.  The spectra row's file is not there: its
        viewer mounts, refuses the load, and is put away -- the mount's
        wiring and the refused road, no spectrum drawn.  The trajectory's
        row is on a run the engine made
        (`test_siesta_run_in_the_viewer_e2e.py`): what it opens is output.

        MUTATIONS THIS MUST FAIL AGAINST: a listener added outside the scope;
        a drawing window made for every mount; a capturing listener taken off
        without its capture flag."""
        folder = isolated_projects_root.resolve() / "P" / "s"
        folder.mkdir(parents=True)
        (folder / "water.xyz").write_text(
            "3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
        path = str(folder / file)
        page.goto(f"{flask_server}/results", wait_until="networkidle")
        page.wait_for_function(
            "(c) => { const r = (window.molbuilder || {}).inspectors;"
            "  return !!r && (r.pick(c.file) || {}).name === c.viewer; }",
            arg={"file": path, "viewer": viewer}, timeout=10000)
        page.wait_for_selector(".ps-entry, .ps-list-error, #ps-list.is-empty",
                               state="attached", timeout=10000)
        from support.leftovers import SPY, grew
        left = page.evaluate("""async (c) => {
            const reg = window.molbuilder.inspectors;
            const spy = """ + SPY + """;
            const cycle = async () => {
                const host = document.createElement("div");
                host.style.width = "900px";
                document.body.appendChild(host);
                // The test's own wait, kept out of what it counts.
                const ended = new Promise((resolve) => spy.quietly(document,
                    window.molbuilder.constants.EVENT_INSPECTOR_READY,
                    (e) => resolve((e && e.detail) || {}), { once: true }));
                const handle = reg.mount(host, c.file,
                                         reg.createDefaultContext(host));
                const said = await Promise.race([ended,
                    new Promise((r) => setTimeout(() => r(null), 8000))]);
                handle.dispose();
                document.body.removeChild(host);
                if (!said) return "a mount never ended";
                if (c.loads && said.error) return "it could not load: " + said.error;
                return null;
            };
            try {
                const first = await cycle();
                if (first) return { error: first };
                const once = spy.state();
                for (let i = 1; i < c.times; i++) {
                    const again = await cycle();
                    if (again) return { error: again };
                }
                return { once, again: spy.state() };
            } finally {
                spy.restore();
            }
        }""", {"file": path, "times": 5, "loads": loads})
        assert "error" not in left, left
        assert not grew(left), (
            f"opening the {viewer} viewer five times left more on the page "
            f"than opening it once:\n  " + "\n  ".join(grew(left)))
        assert left["again"]["contexts"] == left["once"]["contexts"], (
            f"the {viewer} viewer left {left['once']['contexts']} live GL "
            f"context(s) opened once and {left['again']['contexts']} opened "
            f"five times")

    def test_switching_files_in_the_dropdown_again_and_again_leaves_what_once_left(
            self, page, flask_server, isolated_projects_root):
        """`presenters.md` § 3, step 2: picking a file in the dropdown puts
        away the viewer showing the last one -- its timers, listeners and
        drawing -- BEFORE it mounts the next.

        THE SILENT FAILURE: every pick leaving the last viewer mounted
        behind the new one, its listeners firing against a panel that is
        gone, its drawing window held, until the page runs out of them.

        Two structures in one folder, the Results tab bound to it as a
        returning tab finds it, each picked in the dropdown in turn five
        times, each pick waited out by the viewer's word that its load has
        ended; what the page holds after five is what it held after one
        (`support/leftovers.SPY`).

        MUTATION THIS MUST FAIL AGAINST: the tab mounting the next viewer
        without putting the last one away."""
        import json as _json
        from support.leftovers import SPY, grew
        folder = isolated_projects_root.resolve() / "P" / "two"
        folder.mkdir(parents=True)
        files = [folder / "water.xyz", folder / "co2.xyz"]
        files[0].write_text(
            "3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
        files[1].write_text("3\nco2\nC 0 0 0\nO 1.16 0 0\nO -1.16 0 0\n")
        page.add_init_script(
            "try { sessionStorage.setItem('molbuilder.current_dir', "
            f"{_json.dumps(str(folder))}); }} catch (_) {{}}")
        page.goto(f"{flask_server}/results", wait_until="domcontentloaded")
        page.wait_for_function(
            "(want) => { const s = document.getElementById("
            "  'results-file-picker-select');"
            "  return !!s && want.every(w => [...s.options]"
            "    .some(o => o.value === w)); }",
            arg=[str(f) for f in files], timeout=15000)
        left = page.evaluate("""async (c) => {
            const spy = """ + SPY + """;
            const sel = document.getElementById("results-file-picker-select");
            const READY = window.molbuilder.constants.EVENT_INSPECTOR_READY;
            const pick = async (path) => {
                const ended = new Promise((resolve) => spy.quietly(document,
                    READY, (e) => resolve((e && e.detail) || {}),
                    { once: true }));
                sel.value = path;
                sel.dispatchEvent(new Event("change"));
                const said = await Promise.race([ended,
                    new Promise((r) => setTimeout(() => r(null), 8000))]);
                if (!said) return "a pick never ended";
                if (said.error) return "it could not load: " + said.error;
                return null;
            };
            const cycle = async () =>
                (await pick(c.files[1])) || (await pick(c.files[0]));
            try {
                const first = await cycle();
                if (first) return { error: first };
                const once = spy.state();
                for (let i = 1; i < c.times; i++) {
                    const again = await cycle();
                    if (again) return { error: again };
                }
                return { once, again: spy.state() };
            } finally {
                spy.restore();
            }
        }""", {"files": [str(f) for f in files], "times": 5})
        assert "error" not in left, left
        assert not grew(left), (
            "switching files five times left more on the page than switching "
            "once:\n  " + "\n  ".join(grew(left)))
        assert left["again"]["contexts"] == left["once"]["contexts"], (
            f"{left['once']['contexts']} live GL context(s) after one switch, "
            f"{left['again']['contexts']} after five")

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
    """What a viewer shows when the panel it fetches cannot be had: the
    ``.inspector-card.error-card``, saying which file and why."""

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

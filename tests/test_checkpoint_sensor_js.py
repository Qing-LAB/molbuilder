"""Sidebar sensor refresh-contract tests (docs/web/projects.md;
test design: docs/execution/checkpointing.md § 13).

This file pins the *refresh model* the checkpoint sensor depends on,
after the 2026-06-26 decision to replace background polling with
explicit refresh (docs/web/projects.md).  Four
guarantees, the last two about the panel's own source and behaviour:

  1. ``/api/checkpoint/state`` is CHEAP: it returns the documented
     fields and does NOT read a big file it can rule out by size and
     timestamp.  It reports no archive total **at all** -- this used to
     say "archive size stays at its ``0`` default", describing a field
     that has since been removed rather than defaulted (checkpointing.md
     § 12: the number was never true, since hard links counted in full,
     and it fed no decision because nothing prunes).  The two assertions
     below are therefore *the file is never opened* and *the field is
     absent*, which is what the section now claims.
  2. A git failure inside ``state()`` surfaces as a structured
     ``{ok:false, error}`` envelope (HTTP 500, web-api.md bucket D)
     that the JS sensor renders as a 🔴 error pill -- NOT an unhandled
     crash or empty body.
  3. ``checkpoint.js`` contains no ``setInterval`` -- there is no
     background poll loop to reintroduce.
  4. **The panel, RUN.**  The last section imports the module into
     Node against a stub DOM and a stub ``fetch``, drives it, and
     asserts on the requests it made and the DOM it wrote.

That fourth part replaces a claim this file used to make: that the
refresh behaviour "is browser-scope and belongs to the Playwright
E2E".  It does not, entirely -- which requests fire on
directory-enter, whether re-entering the same directory re-reads it,
and whether a refusal the user can answer actually gets a question,
are all decidable in Node in milliseconds.  Deferring them to an E2E
meant deferring them indefinitely, and § 13.3 is explicit: *run the
thing and look at what moved*, because a grep passes for a panel that
contains the right words in the wrong order or throws on first click.

What genuinely does need a browser -- layout, the lazy @gitgraph
render, focus and keyboard behaviour -- is still not covered here.

Real Flask test client + real filesystem + real git, mirroring the
no-mocks discipline of test_checkpoint_routes.py.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from molbuilder.web.app import create_app


def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"],
                       capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(
    not _have_git(),
    reason="git not on PATH; checkpoint sensor tests need git ≥ 2.20",
)


@pytest.fixture
def client():
    return create_app(config={}).test_client()


def _seed_with_archive(tmp_path: Path) -> Path:
    """A plain run dir: an .fdf and a 2048-byte .DM beside it.

    The name is historical and the old docstring claimed ``init()``
    archived the .DM -- neither is true of its one caller, which stubs
    ``Repo.status`` to raise and never initialises anything.  What the
    test needs is a directory that exists and is not a checkpoint
    folder; nothing here is saved, tracked or archived.
    """
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x00" * 2048)
    return tmp_path


# ----------------------------------------------------------------- #
#  1. /state is cheap -- no .binsnapshots/ walk on the refresh path #
# ----------------------------------------------------------------- #


def test_state_does_not_read_big_files_it_does_not_have_to(
        client, tmp_path, checkpoint_config):
    """The sidebar calls this on every directory-enter (docs/web/projects.md).

    A folder holding a 2 GB density matrix must not be read end-to-end to
    answer "is anything unsaved".  A file whose size matches its record and
    which has not been touched since the state was saved cannot have changed,
    so it is ruled out by two ``stat`` fields rather than by hashing.

    The gate is behavioural, not a timing assertion: make the file unreadable
    after saving it.  If the answer still comes back clean, nothing opened it.
    """
    import os
    from molbuilder.checkpoint import Repo

    checkpoint_config(size_limit_bytes=1024, engines={"generic": []})
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    big = tmp_path / "siesta-test.DM"
    big.write_bytes(b"\x00" * 4096)
    Repo(str(tmp_path)).init(note="set up")
    assert list((tmp_path / ".binsnapshots").rglob("siesta-test.DM")), (
        "precondition: the .DM is over the limit and must be archived")

    # Age it past the save.  A state's timestamp has one-second resolution, so
    # a file written in the same second is deliberately NOT ruled out -- this
    # makes the "untouched since the save" case the one under test rather than
    # a race.
    saved = os.stat(big).st_mtime - 60
    os.utime(big, (saved, saved))
    os.chmod(big, 0o000)
    try:
        r = client.get("/api/checkpoint/state",
                       query_string={"path": str(tmp_path)})
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] and body["clean"], (
            "the sensor read the file it did not need to read")
    finally:
        os.chmod(big, 0o644)


def test_state_returns_the_shape_the_sensor_reads(client, tmp_path):
    """§ 5's vocabulary, and nothing that would need an archive walk.

    No archive size: the number was never true (hard links counted in full)
    and fed no decision, since nothing prunes (checkpointing.md § 12).
    """
    from molbuilder.checkpoint import Repo
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    Repo(str(tmp_path)).init(note="set up")
    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path)})
    # Guard BEFORE the body checks.  ``archive_total_bytes not in body`` is a
    # negative assertion, and a 404 would satisfy it against Flask's error
    # page while pinning nothing at all.
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    for key in ("path", "initialized", "standing_at", "clean",
                "changed", "added", "deleted", "unsaved", "ignore_edited"):
        assert key in body, f"sensor field {key!r} missing from /state"
    assert "archive_total_bytes" not in body
    assert body["standing_at"]["note"] == "set up"


# ----------------------------------------------------------------- #
#  2. git failure -> structured envelope, not a crash               #
# ----------------------------------------------------------------- #


def test_state_git_failure_is_structured_error_not_crash(
        client, tmp_path, monkeypatch):
    """When state() raises (transient git failure), the route returns a
    structured {ok:false, error} envelope (HTTP 500, web-api.md bucket
    D) the sensor renders as a 🔴 error pill -- never a bare crash or
    empty body."""
    import molbuilder.web.blueprints.checkpoint as bp
    _seed_with_archive(tmp_path)

    def _boom(self, deep=False):
        raise bp.CheckpointError("git rev-parse exploded (simulated)")

    monkeypatch.setattr(bp.Repo, "status", _boom)
    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 500
    body = r.get_json()
    assert body is not None, "must return a JSON envelope, not empty body"
    assert body["ok"] is False
    assert isinstance(body.get("error"), str) and body["error"]
    assert "simulated" in body["error"]


# ----------------------------------------------------------------- #
#  3. no background poll loop in the JS                              #
# ----------------------------------------------------------------- #


def _js(name: str) -> Path:
    return (Path(__file__).resolve().parent.parent / "molbuilder" / "web"
            / "static" / "lib" / "projects" / name)


@pytest.mark.parametrize("name", ["checkpoint.js", "projects-sidebar.js"])
def test_the_panel_and_its_importer_actually_parse(name):
    """The cheapest test in the file, and it would have caught a dead sidebar.

    A stray brace made `checkpoint.js` a SyntaxError.  `projects-sidebar.js`
    imports it STATICALLY, so the failure took the importer down too -- and
    that module is loaded by five page templates, which means the whole
    projects sidebar was gone from every page.  Every other test here reads
    this file as *text* and greps it, and text greps cannot tell you a file
    does not parse.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node not on PATH; cannot parse-check ES modules")
    source = _js(name)
    # `--check` reads .js as CommonJS, where `export` is a syntax error in
    # itself; the .mjs copy is what makes it judge the module as a module.
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / (source.stem + ".mjs")
        probe.write_text(source.read_text(), encoding="utf-8")
        result = subprocess.run([node, "--check", str(probe)],
                                capture_output=True, text=True)
    assert result.returncode == 0, (
        f"{name} does not parse:\n{result.stderr}")


def test_the_panel_speaks_the_routes_vocabulary():
    """The panel's request bodies must be the ones the routes accept.

    `save` and `tag` sent `message` and `label` -- the pre-rework names -- so
    every save and every tag from the sidebar was refused with HTTP 400 while
    the module itself was correct.  Nothing else in the suite crosses that
    seam: the route tests post their own bodies, and the JS tests grep text.
    """
    js = _js("checkpoint.js").read_text()

    def body_of(route):
        """The object literal passed to the CALL, not the header comment.

        Splitting on the bare route path finds the module docstring's list of
        endpoints first, which contains no request body at all -- a grep that
        matches prose can only ever agree with prose.
        """
        anchor = f'_fetchJSON("POST", "/api/checkpoint/{route}"'
        assert anchor in js, f"no POST call to {route} in checkpoint.js"
        return js.split(anchor, 1)[1][:400]

    save = body_of("save")
    assert "note:" in save and "message:" not in save, (
        "save must send `note` -- the field L3 requires and the route reads")
    tag = body_of("tag")
    assert "name:" in tag and "note:" in tag, "tag sends `name` and `note`"
    assert "label:" not in tag


def test_the_panel_never_offers_to_generate_a_note():
    """L3: the note is required and never generated.

    The prompt offered "leave blank for ISO timestamp" -- the exact generated
    stand-in the contract rules out, on the one surface where the question is
    actually asked.
    """
    js = _js("checkpoint.js").read_text().lower()
    assert "leave blank" not in js
    assert "iso timestamp" not in js


def test_checkpoint_js_has_no_polling_timer():
    """The explicit-refresh model (docs/web/projects.md)
    forbids a background poll loop.  Guard against setInterval creeping
    back into the sensor.  We strip line comments first so the doc-
    comment mention of 'no setInterval' doesn't trip the check."""
    js = (Path(__file__).resolve().parent.parent
          / "molbuilder" / "web" / "static" / "lib"
          / "projects" / "checkpoint.js").read_text()
    code_lines = []
    for ln in js.splitlines():
        stripped = ln.strip()
        if stripped.startswith("*") or stripped.startswith("//") \
                or stripped.startswith("/*"):
            continue
        code_lines.append(ln)
    code = "\n".join(code_lines)
    assert "setInterval" not in code, (
        "checkpoint.js must not poll -- refresh is explicit "
        "(directory-enter + manual Refresh) per docs/web/projects.md "
        "docs/web/projects.md")


def test_the_panel_can_supply_a_calculation_name_when_the_folder_is_refused():
    """The panel's only way out of a refused save (L3), and it had none.

    A folder whose own name cannot be a calculation name is refused, and the
    only thing that resolves it is a name — which the panel could not send,
    because Set-up posted `{path}` and nothing else.  Every button gave the
    same refusal and none of them could answer it.

    The server marks that refusal `where: "calculation"` so a surface can tell
    it from the other advisory on the same route — a folder holding several
    calculations, which a name does **not** fix and where a prompt would be a
    lie.  Branching on `where` is therefore the assertion, not the presence of
    a prompt.
    """
    js = _js("checkpoint.js").read_text()
    init = js.split("async function _onInitClick", 1)[1].split("\nasync function", 1)[0]
    assert 'where === "calculation"' in init, (
        "the panel must branch on WHICH refusal it got; prompting for a name "
        "on the nested-calculations advisory would ask a question that cannot "
        "help")
    assert "calculation:" in init, (
        "and it must actually send the name it collected")
    assert "prompt(" in init


def test_the_sensor_pill_does_not_keep_the_previous_folders_tooltip():
    """Navigating to an un-set-up folder cleared the text and not the title.

    The pill then read "no states saved" while its tooltip still listed the
    unsaved files of the directory you just left -- naming files that are not
    in this folder at all.
    """
    js = _js("checkpoint.js").read_text()
    uninit = js.split("if (!repoState || !repoState.initialized) {", 1)[1] \
               .split("return;", 1)[0]
    assert "elSensor.title" in uninit, (
        "the branch that sets the pill must clear the title the other two set")


def test_the_panel_says_so_when_the_generated_ignore_block_was_edited():
    """A hand edit inside the markers is not supposed to happen, so say so.

    The block is rewritten from the classification on every save -- which is
    what makes the edit detectable, and also what makes it disappear.  The
    server sends `ignore_edited`; a panel that receives it and stays quiet
    leaves somebody editing a file that never survives.
    """
    js = _js("checkpoint.js").read_text()
    assert "ignore_edited" in js, (
        "the panel must read the flag the route sends")
    lowered = js.lower()
    assert "rewritten from the classification" in lowered, (
        "the note must say WHY the edit will not survive")
    # A phrase short enough not to straddle a line break in the source: this
    # file reads the JS as text, so an assertion spanning two concatenated
    # string literals fails for the formatting rather than for the meaning.
    assert "outside the molbuilder" in lowered, (
        "and where to put entries that should survive -- a note with no way "
        "to get what you wanted is just a scolding")


# ================================================================== #
#  The panel RUN, not grepped                                        #
#                                                                    #
#  § 13.3 rules out "asserting on emitted text where the end result  #
#  is what matters -- run the thing and look at what moved."  Every  #
#  test above this line reads checkpoint.js as a STRING: they would  #
#  pass for a panel that contains the right words in the wrong       #
#  order, or that throws on the first click.  These import the       #
#  module into Node against a stub DOM and a stub fetch, drive it,   #
#  and assert on the requests it made and the DOM it wrote.          #
# ================================================================== #

from _node_esm import run_node                              # noqa: E402

#: A DOM small enough to read and real enough to drive the panel.  Only the
#: handful of methods checkpoint.js actually touches; anything it reaches for
#: that is missing shows up as a Node error rather than a silent pass.
_DOM = r"""
const _els = {};
function _mk(id) {
  const el = {
    id, textContent: "", innerHTML: "", hidden: false, title: "", type: "",
    className: "", dataset: {}, children: [], _events: {},
    addEventListener(ev, fn) { (this._events[ev] ||= []).push(fn); },
    setAttribute(k, v) { this.dataset["attr_" + k] = String(v); },
    getAttribute(k) { return this.dataset["attr_" + k]; },
    appendChild(c) { this.children.push(c); return c; },
    classList: { toggle() {}, add() {}, remove() {} },
    closest() { return null; },
    click() { for (const fn of (this._events.click || [])) fn({}); },
  };
  return el;
}
globalThis.document = {
  getElementById: (id) => (_els[id] ||= _mk(id)),
  createElement: (t) => _mk("new:" + t),
  createElementNS: (ns, t) => _mk("ns:" + t),
  documentElement: _mk("html"),
  head: _mk("head"),
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
globalThis.sessionStorage = {
  _d: {}, getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; },
};
globalThis.__els = _els;

// Every request the panel makes, in order, with its body.
globalThis.__calls = [];
globalThis.__replies = [];      // [{match, http, body}] -- first match wins
globalThis.fetch = async (url, opts = {}) => {
  const body = opts.body ? JSON.parse(opts.body) : null;
  __calls.push({ url, method: opts.method || "GET", body });
  const hit = __replies.find(r => url.includes(r.match)) || {};
  return { status: hit.http ?? 200, json: async () => hit.body ?? { ok: true } };
};
globalThis.__prompts = [];      // answers handed to prompt(), in order
globalThis.__asked   = [];      // the questions it asked
globalThis.prompt = (q) => { __asked.push(q); return __prompts.shift() ?? null; };
globalThis.confirm = () => true;
globalThis.molbuilder = {
  projects: { getProjectsRoot: () => "/p", onChange() {}, getCurrentDir: () => null },
};
"""

_PANEL = (Path(__file__).resolve().parent.parent / "molbuilder" / "web"
          / "static" / "lib" / "projects" / "checkpoint.js")

_RUN_DIR = "/p/BDT-Au/optimization/relax"      # rel-depth 3: the gate opens


def test_entering_a_run_dir_reads_the_state_only_until_opened():
    """Directory-enter with the view closed: ONE request.

    The state read is what colors the toggle -- it must be honest without
    the view ever opening -- but the fifty list rows nobody is looking at
    are not fetched (docs/web/projects.md, 2026-08-19).  A second request
    here would be paying for a view that is not on screen.
    """
    out = run_node([_PANEL], r"""
      globalThis.__replies = [
        {match: "/state", body: {ok: true, initialized: true, clean: true,
                                 unsaved: [], changed: [], added: [], deleted: [],
                                 standing_at: {id: "abc1234", short: "abc1234",
                                               note: "set up", parent: null,
                                               tags: []}}},
        {match: "/list",  body: {ok: true, states: [
            {id: "abc1234", short: "abc1234", note: "set up",
             parent: null, tags: []}]}},
      ];
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      await new Promise(r => setTimeout(r, 30));
      console.log(JSON.stringify({
        calls:  __calls.map(c => c.method + " " + c.url.split("?")[0]),
        toggleHidden: __els["ps-checkpoint-toggle"].hidden,
        toggleState:  __els["ps-checkpoint-toggle"].dataset["attr_data-state"],
        panelHidden:  __els["ps-checkpoint"].hidden,
        filesHidden:  __els["ps-list"].hidden,
      }));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["calls"] == ["GET /api/checkpoint/state"], out["calls"]
    assert out["toggleHidden"] is False
    assert out["toggleState"] == "clean"          # the color IS the status
    assert out["panelHidden"] is True             # files hold the space
    assert out["filesHidden"] is False


def test_the_toggle_swaps_the_space_and_only_then_reads_the_list():
    """The swap, executed: open -> the checkpoint view takes the list's
    space (exclusively), the filter disables, and the list is fetched;
    close -> the files return and the filter re-enables."""
    out = run_node([_PANEL], r"""
      globalThis.__replies = [
        {match: "/state", body: {ok: true, initialized: true, clean: true,
                                 unsaved: [], changed: [], added: [], deleted: [],
                                 standing_at: {id: "abc1234", short: "abc1234",
                                               note: "set up", parent: null,
                                               tags: []}}},
        {match: "/list",  body: {ok: true, states: [
            {id: "abc1234", short: "abc1234", note: "set up",
             parent: null, tags: []}]}},
      ];
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      await new Promise(r => setTimeout(r, 10));
      __els["ps-checkpoint-toggle"].click();
      await new Promise(r => setTimeout(r, 30));
      const open = {
        calls: __calls.map(c => c.method + " " + c.url.split("?")[0]),
        panelHidden: __els["ps-checkpoint"].hidden,
        filesHidden: __els["ps-list"].hidden,
        filterDisabled: __els["ps-filter-input"].disabled === true,
        rows: __els["ps-checkpoint-list"].children.length,
        pref: sessionStorage.getItem("ps.checkpoint.open"),
      };
      __els["ps-checkpoint-toggle"].click();
      await new Promise(r => setTimeout(r, 10));
      const closed = {
        panelHidden: __els["ps-checkpoint"].hidden,
        filesHidden: __els["ps-list"].hidden,
        filterDisabled: __els["ps-filter-input"].disabled === true,
        pref: sessionStorage.getItem("ps.checkpoint.open"),
      };
      console.log(JSON.stringify({open, closed}));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    o = out["open"]
    assert o["calls"] == ["GET /api/checkpoint/state",
                          "GET /api/checkpoint/state",
                          "GET /api/checkpoint/list"], o["calls"]
    assert o["panelHidden"] is False and o["filesHidden"] is True
    assert o["filterDisabled"] is True            # it filters FILES
    assert o["rows"] == 1
    assert o["pref"] == "1"                       # the choice survives
    c = out["closed"]
    assert c["panelHidden"] is True and c["filesHidden"] is False
    assert c["filterDisabled"] is False
    assert c["pref"] is None


def test_a_stored_open_preference_greets_a_run_dir_with_the_view():
    """The sessionStorage pref is read at wiring, so a reload -- or stepping
    out of a run dir and back -- returns the view you had, list included."""
    out = run_node([_PANEL], r"""
      sessionStorage.setItem("ps.checkpoint.open", "1");
      globalThis.__replies = [
        {match: "/state", body: {ok: true, initialized: true, clean: true,
                                 unsaved: [], changed: [], added: [], deleted: [],
                                 standing_at: {id: "abc1234", short: "abc1234",
                                               note: "set up", parent: null,
                                               tags: []}}},
        {match: "/list",  body: {ok: true, states: [
            {id: "abc1234", short: "abc1234", note: "set up",
             parent: null, tags: []}]}},
      ];
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      await new Promise(r => setTimeout(r, 30));
      console.log(JSON.stringify({
        calls: __calls.map(c => c.method + " " + c.url.split("?")[0]),
        panelHidden: __els["ps-checkpoint"].hidden,
        rows: __els["ps-checkpoint-list"].children.length,
      }));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["calls"] == ["GET /api/checkpoint/state",
                            "GET /api/checkpoint/list"], out["calls"]
    assert out["panelHidden"] is False
    assert out["rows"] == 1


def test_leaving_the_run_dir_hides_the_toggle_and_restores_the_files():
    """A non-run dir has no history surface at all: the button goes, the
    files return, the filter re-enables -- but the OPEN PREFERENCE survives,
    so coming back keeps your view."""
    out = run_node([_PANEL], r"""
      sessionStorage.setItem("ps.checkpoint.open", "1");
      globalThis.__replies = [
        {match: "/state", body: {ok: true, initialized: true, clean: true,
                                 unsaved: [], changed: [], added: [], deleted: [],
                                 standing_at: null}},
        {match: "/list",  body: {ok: true, states: []}},
      ];
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      await new Promise(r => setTimeout(r, 20));
      await m.onDirectoryChange("/p/BDT-Au/optimization");
      console.log(JSON.stringify({
        toggleHidden: __els["ps-checkpoint-toggle"].hidden,
        panelHidden:  __els["ps-checkpoint"].hidden,
        filesHidden:  __els["ps-list"].hidden,
        filterDisabled: __els["ps-filter-input"].disabled === true,
        pref: sessionStorage.getItem("ps.checkpoint.open"),
      }));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["toggleHidden"] is True
    assert out["panelHidden"] is True
    assert out["filesHidden"] is False
    assert out["filterDisabled"] is False
    assert out["pref"] == "1"


def test_a_shallower_directory_never_reaches_the_network():
    """The activation gate, executed rather than described.

    A project or category directory is not a run dir, so the panel hides and
    asks nothing.  Asserting the REQUEST COUNT is what makes this real: the
    gate could be drawn correctly and still fetch.
    """
    out = run_node([_PANEL], r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      for (const d of ["/p", "/p/BDT-Au", "/p/BDT-Au/optimization", null]) {
        await m.onDirectoryChange(d);
      }
      await new Promise(r => setTimeout(r, 20));
      console.log(JSON.stringify({calls: __calls.length,
                                  hidden: __els["ps-checkpoint"].hidden}));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["calls"] == 0, "the panel fetched for a directory it does not serve"
    assert out["hidden"] is True


def test_re_entering_the_same_directory_does_not_re_fetch():
    """The no-op guard, executed.

    `projects.onChange` fires on every single-click FILE selection too -- the
    cursor moves inside the same directory -- so without this guard the panel
    re-reads the folder on every click in the file list.
    """
    out = run_node([_PANEL], r"""
      __replies = [{match: "/state", body: {ok: true, initialized: false}}];
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      for (let i = 0; i < 4; i++) {
        await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      }
      await new Promise(r => setTimeout(r, 20));
      console.log(JSON.stringify({calls: __calls.length}));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["calls"] == 1, (
        "entering the same directory four times cost four reads")


def test_setup_asks_for_a_name_only_when_a_name_is_what_is_missing():
    """The refusal→prompt→retry path, executed end to end.

    The grep above proves the branch is *written*.  This proves it *works*:
    the first POST carries no name, the server refuses with
    `where: "calculation"`, the panel asks, and the SECOND POST carries what
    the person typed.  Nothing above this line would notice if the retry sent
    the name under the wrong key, or never sent it at all.
    """
    out = run_node([_PANEL], r"""
      let seen = 0;
      globalThis.fetch = async (url, opts = {}) => {
        const body = opts.body ? JSON.parse(opts.body) : null;
        __calls.push({url, method: opts.method || "GET", body});
        if (url.includes("/init")) {
          seen++;
          if (seen === 1) return {status: 200, json: async () => ({
            ok: false, error: "bad name",
            errors_only: [{severity: "error", message: "bad name",
                           where: "calculation"}]})};
          return {status: 200, json: async () => ({ok: true, already: true,
                                                   calculation: "BDT_Au_relax"})};
        }
        return {status: 200, json: async () => ({ok: true, initialized: false})};
      };
      __prompts = ["BDT_Au_relax"];
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      await new Promise(r => setTimeout(r, 20));
      __els["ps-checkpoint-init"].click();
      await new Promise(r => setTimeout(r, 40));
      const inits = __calls.filter(c => c.url.includes("/init"));
      console.log(JSON.stringify({
        n: inits.length,
        first_had_name: inits[0] ? ("calculation" in inits[0].body) : null,
        second_name: inits[1] ? inits[1].body.calculation : null,
        asked: __asked.length,
      }));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["n"] == 2, "the panel did not retry after the refusal"
    assert out["first_had_name"] is False, "the ordinary attempt sends no name"
    assert out["second_name"] == "BDT_Au_relax", (
        "the retry did not carry the name the person typed")
    assert out["asked"] == 1, "exactly one question, and only because of `where`"


def test_setup_does_not_ask_for_a_name_when_a_name_would_not_help():
    """The other advisory on the same route, and the reason `where` exists.

    A folder holding several independent calculations is refused too — and a
    name does not fix that. Prompting for one would be asking a question whose
    answer cannot help, which is how a surface teaches people that its
    questions are noise.
    """
    out = run_node([_PANEL], r"""
      globalThis.fetch = async (url, opts = {}) => {
        const body = opts.body ? JSON.parse(opts.body) : null;
        __calls.push({url, method: opts.method || "GET", body});
        if (url.includes("/init")) return {status: 200, json: async () => ({
          ok: false, error: "several calculations",
          errors_only: [{severity: "error", message: "several calculations",
                         where: "path"}]})};
        return {status: 200, json: async () => ({ok: true, initialized: false})};
      };
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      await new Promise(r => setTimeout(r, 20));
      __els["ps-checkpoint-init"].click();
      await new Promise(r => setTimeout(r, 40));
      console.log(JSON.stringify({
        inits: __calls.filter(c => c.url.includes("/init")).length,
        asked: __asked.length,
        advisory: __els["ps-checkpoint-advisory"].textContent,
      }));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["asked"] == 0, "it asked for a name that cannot help"
    assert out["inits"] == 1, "and it must not retry blindly"
    assert "several calculations" in out["advisory"]


def test_wiring_the_panel_twice_does_not_double_every_click():
    """`initCheckpointPanel` says it is safe to call from several bootstrap
    paths; before the guard, a second call double-bound every listener and one
    click on Set-up posted twice.  One caller exists today — the docstring
    invites the second."""
    out = run_node([_PANEL], r"""
      __replies = [{match: "/", body: {ok: true, initialized: false}}];
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      m.initCheckpointPanel();
      await m.onDirectoryChange("/p/BDT-Au/optimization/relax");
      await new Promise(r => setTimeout(r, 20));
      __calls.length = 0;
      __els["ps-checkpoint-init"].click();
      await new Promise(r => setTimeout(r, 40));
      console.log(JSON.stringify({
        inits: __calls.filter(c => c.url.includes("/init")).length}));
    """, globals_js=_DOM + f'\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};')
    assert out["inits"] == 1, (
        "one click produced more than one request; the panel was wired twice")

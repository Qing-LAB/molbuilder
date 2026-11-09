"""The checkpoint panel's refresh model (docs/web/projects.md, the
run-history view; test design: docs/execution/checkpointing.md § 13).

The panel refreshes only when asked -- no polling: entering a folder reads
its state once, which decides whether the button appears and colors it, and
Refresh reads it again.  Two parts:

  1. ``/api/checkpoint/state`` is CHEAP: it returns the documented
     fields and does NOT read a big file it can rule out by size and
     timestamp.  It reports no archive total **at all** (checkpointing.md
     § 12: the number was never true, since hard links counted in full,
     and it fed no decision because nothing prunes).  The two assertions
     below are therefore *the file is never opened* and *the field is
     absent*.
  2. **The panel, RUN.**  The last section imports the module into
     Node against a stub DOM, its requests answered by the app's own
     routes (`_node_esm.served`), drives it, and asserts on the requests
     it made and the DOM it wrote.

Which requests fire on directory-enter, whether re-entering the same
directory re-reads it, and whether a refusal the user can answer
actually gets a question, are all decidable in Node in milliseconds,
and § 13.3 is explicit: *run the thing and look at what moved*, because a
grep passes for a panel that contains the right words in the wrong order or
throws on first click.

What genuinely does need a browser -- layout, the lazy @gitgraph
render, focus and keyboard behaviour -- is still not covered here.

The routes are asked through the Flask test client, on real folders and
real git, and the panel's requests go to the app's own routes; what stands
in is the page's DOM and the person's answers to the panel's dialogs.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


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


# ----------------------------------------------------------------- #
#  1. /state is cheap -- no .binsnapshots/ walk on the refresh path #
# ----------------------------------------------------------------- #


@pytest.fixture
def run_dir(isolated_projects_root):
    """A folder in this test's own projects tree, which the projects root
    names as a deployment names it (`isolated_projects_root`): since
    web-api.md § 2.1 every route fences a browser-supplied path to that
    tree before calling the module behind it."""
    folder = isolated_projects_root / "P" / "run"
    folder.mkdir(parents=True)
    return folder


def test_state_does_not_read_big_files_it_does_not_have_to(
        web_client, run_dir, checkpoint_config):
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
    (run_dir / "siesta-test.fdf").write_text("SystemLabel test\n")
    big = run_dir / "siesta-test.DM"
    big.write_bytes(b"\x00" * 4096)
    Repo(str(run_dir)).init(note="set up")
    assert list((run_dir / ".binsnapshots").rglob("siesta-test.DM")), (
        "precondition: the .DM is over the limit and must be archived")

    # Age it past the save.  A state's timestamp has one-second resolution, so
    # a file written in the same second is deliberately NOT ruled out -- this
    # makes the "untouched since the save" case the one under test rather than
    # a race.
    saved = os.stat(big).st_mtime - 60
    os.utime(big, (saved, saved))
    os.chmod(big, 0o000)
    try:
        r = web_client.get("/api/checkpoint/state",
                           query_string={"path": str(run_dir)})
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] and body["clean"], (
            "the sensor read the file it did not need to read")
    finally:
        os.chmod(big, 0o644)


def test_state_returns_the_shape_the_sensor_reads(web_client, run_dir):
    """§ 5's vocabulary, and nothing that would need an archive walk.

    No archive size: the number was never true (hard links counted in full)
    and fed no decision, since nothing prunes (checkpointing.md § 12).
    """
    from molbuilder.checkpoint import Repo
    (run_dir / "siesta-test.fdf").write_text("SystemLabel test\n")
    Repo(str(run_dir)).init(note="set up")
    r = web_client.get("/api/checkpoint/state",
                       query_string={"path": str(run_dir)})
    # Guard BEFORE the body checks.  ``archive_total_bytes not in body`` is a
    # negative assertion, and a 404 would satisfy it against Flask's error
    # page while pinning nothing at all.
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    for key in ("path", "is_calculation", "initialized", "standing_at",
                "clean", "changed", "added", "deleted", "unsaved",
                "ignore_edited"):
        assert key in body, f"sensor field {key!r} missing from /state"
    assert "archive_total_bytes" not in body
    assert body["standing_at"]["note"] == "set up"


# ================================================================== #
#  The panel RUN, not grepped                                        #
#                                                                    #
#  § 13.3 rules out "asserting on emitted text where the end result  #
#  is what matters -- run the thing and look at what moved."  These  #
#  import the module into Node against a stub DOM, its requests      #
#  answered by the app's own routes on a folder the road made        #
#  (`_node_esm.served`), drive it, and assert on the requests it     #
#  made and the DOM it wrote.                                        #
# ================================================================== #

import json                                                 # noqa: E402
from _node_esm import run_node, served                      # noqa: E402

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
    // ...AS A BROWSER'S: the nodes put in, and their text the element's.
    replaceChildren(...nodes) {
      const flat = nodes.flatMap((n) => (n && n.isFragment ? n.children : [n]));
      this.children = flat;
      this.textContent = flat.map((n) => n.textContent).join("");
    },
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
  createTextNode: (t) => ({ textContent: String(t) }),
  createDocumentFragment: () => ({
    isFragment: true, children: [],
    appendChild(c) { this.children.push(c); return c; },
  }),
  documentElement: _mk("html"),
  head: _mk("head"),
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
globalThis.sessionStorage = {
  _d: {}, getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; },
};
globalThis.__els = _els;
// A RELOAD: the page's DOM built afresh and the panel's module run again --
// what the browser keeps across one, its session storage, kept.
globalThis.__reload = async () => {
  for (const k of Object.keys(_els)) delete _els[k];
  return import(process.env.PANEL_URL + "?reloaded");
};
// THE APP'S OWN DIALOGS, answered: the panel's `../dialog.js` resolves to a
// module whose `askFields` records the question and hands back the next of
// `__answers` (null, a cancel, when none is left); the one confirm says yes.
globalThis.__answers = [];      // what each dialog is answered, in order
globalThis.__asked   = [];      // the questions it asked
{
  const { registerHooks } = await import("node:module");
  const stub = "data:text/javascript," + encodeURIComponent(
    "export async function askFields(o) { globalThis.__asked.push(o.title);"
    + " return globalThis.__answers.shift() ?? null; }\n");
  registerHooks({ resolve(spec, ctx, next) {
    return spec === "../dialog.js"
        && String(ctx.parentURL).includes("/lib/projects/checkpoint.js")
      ? { url: stub, shortCircuit: true } : next(spec, ctx);
  } });
}
globalThis.molbuilder = {
  warningModal: { confirm: async () => true },
  projects: { getProjectsRoot: () => null, onChange() {}, getCurrentDir: () => null },
  // The one tab strip, which every page's header loads (`lib/tab-strip.js`);
  // the List / Graph strip is not what these tests drive.
  tabStrip: { mount: () => ({ select() {} }) },
};
"""

_PANEL = (Path(__file__).resolve().parent.parent / "molbuilder" / "web"
          / "static" / "lib" / "projects" / "checkpoint.js")
#: THE SERVER'S WORDS, DRAWN -- the renderer every page's header loads
#: (`lib/dom.js`, `ui-contract.md` § 12.1).
_WORDS = _PANEL.parent.parent / "dom.js"


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the panel's requests (`support.
    live_server.serve`).  Each test's tree is its own -- the projects root
    and the allowed roots are read as each request arrives."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.fixture
def described(tmp_path, monkeypatch):
    """A calculation `molbuilder describe` made, in this test's tree."""
    from support.road import described, road_tree
    return described(road_tree(tmp_path, monkeypatch))


def _set_up(folder):
    """``folder`` made a checkpoint folder, its first state saved, as a
    person does at a terminal (`molbuilder checkpoint init`)."""
    from support.road import molbuilder
    r = molbuilder("checkpoint", "init", "--path", str(folder))
    assert r.exit_code == 0, r.output


def _drive(app_at, script):
    """``script`` run against the panel, its requests answered by the app."""
    return run_node([_WORDS, _PANEL], script, globals_js=(
        _DOM + served(app_at)
        + f"\nprocess.env.PANEL_URL = {_PANEL.resolve().as_uri()!r};"))


def _calls():
    """The requests the panel made, as the snippet reports them."""
    return '__calls.map(c => c.method + " " + c.url.split("?")[0])'


def test_entering_a_run_dir_reads_the_state_only_until_opened(app_at, described):
    """Directory-enter with the view closed: ONE request.

    The state read is what colors the toggle -- it must be honest without
    the view ever opening -- but the fifty list rows nobody is looking at
    are not fetched (docs/web/projects.md, 2026-08-19).  A second request
    here would be paying for a view that is not on screen.
    """
    _set_up(described)
    out = _drive(app_at, r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      console.log(JSON.stringify({
        calls:  %s,
        toggleHidden: __els["ps-checkpoint-toggle"].hidden,
        toggleState:  __els["ps-checkpoint-toggle"].dataset["attr_data-state"],
        panelHidden:  __els["ps-checkpoint"].hidden,
        filesHidden:  __els["ps-list"].hidden,
      }));
    """ % (json.dumps(str(described)), _calls()))
    assert out["calls"] == ["GET /api/checkpoint/state"], out["calls"]
    assert out["toggleHidden"] is False
    assert out["toggleState"] == "clean"          # the color IS the status
    assert out["panelHidden"] is True             # files hold the space
    assert out["filesHidden"] is False


def test_the_toggle_swaps_the_space_and_only_then_reads_the_list(
        app_at, described):
    """The swap, executed: open -> the checkpoint view takes the list's
    space (exclusively), the filter disables, and the list is fetched --
    one row, the state the set-up saved; close -> the files return and the
    filter re-enables."""
    _set_up(described)
    out = _drive(app_at, r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      __els["ps-checkpoint-toggle"].click();
      await __quiet();
      const open = {
        calls: %s,
        panelHidden: __els["ps-checkpoint"].hidden,
        filesHidden: __els["ps-list"].hidden,
        filterDisabled: __els["ps-filter-input"].disabled === true,
        rows: __els["ps-checkpoint-list"].children.length,
        pref: sessionStorage.getItem("ps.checkpoint.open"),
      };
      __els["ps-checkpoint-toggle"].click();
      await __quiet();
      const closed = {
        panelHidden: __els["ps-checkpoint"].hidden,
        filesHidden: __els["ps-list"].hidden,
        filterDisabled: __els["ps-filter-input"].disabled === true,
        pref: sessionStorage.getItem("ps.checkpoint.open"),
      };
      console.log(JSON.stringify({open, closed}));
    """ % (json.dumps(str(described)), _calls()))
    o = out["open"]
    assert o["calls"] == ["GET /api/checkpoint/state",
                          "GET /api/checkpoint/state",
                          "GET /api/checkpoint/list"], o["calls"]
    assert o["panelHidden"] is False and o["filesHidden"] is True
    assert o["filterDisabled"] is True            # it filters FILES
    assert o["rows"] == 1                         # the one state saved
    assert o["pref"] == "1"                       # the choice survives
    c = out["closed"]
    assert c["panelHidden"] is True and c["filesHidden"] is False
    assert c["filterDisabled"] is False
    assert c["pref"] is None


def test_the_open_view_greets_a_run_dir_again_after_a_reload(
        app_at, described):
    """The view opened by its toggle survives a reload: the page comes back
    to the run dir with the view you had, list included -- the choice the
    toggle stored is read when the panel is wired."""
    _set_up(described)
    out = _drive(app_at, r"""
      const first = await import(process.env.PANEL_URL);
      first.initCheckpointPanel();
      await first.onDirectoryChange(%s);
      await __quiet();
      __els["ps-checkpoint-toggle"].click();
      await __quiet();
      const m = await __reload();
      __calls.length = 0;
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      console.log(JSON.stringify({
        calls: %s,
        panelHidden: __els["ps-checkpoint"].hidden,
        rows: __els["ps-checkpoint-list"].children.length,
      }));
    """ % (json.dumps(str(described)), json.dumps(str(described)), _calls()))
    assert out["calls"] == ["GET /api/checkpoint/state",
                            "GET /api/checkpoint/list"], out["calls"]
    assert out["panelHidden"] is False
    assert out["rows"] == 1


def test_leaving_the_run_dir_hides_the_toggle_and_restores_the_files(
        app_at, described):
    """A folder that is not a calculation has no history surface at all:
    the button goes, the files return, the filter re-enables -- but the OPEN
    PREFERENCE survives, so coming back keeps your view."""
    _set_up(described)
    out = _drive(app_at, r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      __els["ps-checkpoint-toggle"].click();
      await __quiet();
      await m.onDirectoryChange(%s);
      await __quiet();
      console.log(JSON.stringify({
        toggleHidden: __els["ps-checkpoint-toggle"].hidden,
        panelHidden:  __els["ps-checkpoint"].hidden,
        filesHidden:  __els["ps-list"].hidden,
        filterDisabled: __els["ps-filter-input"].disabled === true,
        pref: sessionStorage.getItem("ps.checkpoint.open"),
      }));
    """ % (json.dumps(str(described)), json.dumps(str(described.parent))))
    assert out["toggleHidden"] is True
    assert out["panelHidden"] is True
    assert out["filesHidden"] is False
    assert out["filterDisabled"] is False
    assert out["pref"] == "1"


def test_a_folder_that_is_not_a_calculation_shows_no_panel(app_at, described):
    """The activation gate, executed rather than described — and the gate is
    the DOOR'S ANSWER, not a count of path segments.

    The projects root, a project and a kind's folder are not calculations,
    so the panel hides; nor is a folder of loose files beside the
    calculation, at its very depth.  It knows from `is_calculation`, which
    comes back from `/api/checkpoint/state` (`task.json` is in the
    calculation's own folder — `project-layout.md` invariant 2;
    `web/projects.md` § 4, the run-history view: a count of path segments
    put a calculation one folder deeper out of reach and offered loose
    files at the right depth a repository).

    ONE request per distinct directory is the cost, and it is the same
    request that fills the panel — measured at 1.3–1.7 ms against a
    non-repository folder, no subprocess on that path.  What must not
    happen is a storm, so the count is asserted.  `null` asks nothing:
    there is no directory to ask about.

    The panel is read after EACH folder's answer has landed, while that
    folder is still the current one: the panel hides on entering a folder,
    before any answer, so a read taken later, or after `null`, sees that
    hiding whatever the answer said.
    """
    loose = described.parent / "loose"
    loose.mkdir()
    (loose / "a.xyz").write_text("1\nx\nH 0 0 0\n")
    above = [described.parent.parent.parent, described.parent.parent,
             described.parent, loose]
    out = _drive(app_at, r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      const seen = [];
      for (const d of %s) {
        m.onDirectoryChange(d);
        await __quiet();
        const said = __answered[__answered.length - 1];
        seen.push({
          said: [said.is_calculation, said.initialized],
          hidden: __els["ps-checkpoint"].hidden,
          toggleHidden: __els["ps-checkpoint-toggle"].hidden,
          filesHidden: __els["ps-list"].hidden});
      }
      m.onDirectoryChange(null);
      await __quiet();
      console.log(JSON.stringify({calls: __calls.length, seen}));
    """ % json.dumps([str(d) for d in above]))
    assert out["calls"] == len(above), (
        "one ask per distinct directory and nothing for `null`; got "
        + str(out["calls"]))
    for folder, seen in zip(above, out["seen"]):
        assert seen["said"] == [False, False], (folder, seen)
        assert seen["hidden"] is True, (folder, seen)
        assert seen["toggleHidden"] is True, (
            f"the button appeared on {folder}, a folder the panel does not "
            f"serve")
        assert seen["filesHidden"] is False, (
            folder, "the file list keeps the space")


def test_a_described_folder_with_no_repository_shows_the_panel(
        app_at, described):
    """The case `initialized` alone cannot express, and the reason
    `is_calculation` is a separate field.

    A calculation that has never been checkpointed is exactly where the
    Set-up button belongs — `initialized: false` — so a panel gated on
    `initialized` would hide from the one folder it is there to help.  A
    description is the answer at any depth: the server says this folder is
    a calculation with no repository, and the button shows.
    """
    out = _drive(app_at, r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      const said = __answered[0];
      console.log(JSON.stringify({
        said: [said.is_calculation, said.initialized],
        toggleHidden: __els["ps-checkpoint-toggle"].hidden}));
    """ % json.dumps(str(described)))
    assert out["said"] == [True, False], (
        "a described folder with no repo is exactly the Set-up case")
    assert out["toggleHidden"] is False


def test_re_entering_the_same_directory_does_not_re_fetch(app_at, described):
    """The no-op guard, executed.

    `projects.onChange` fires on every single-click FILE selection too -- the
    cursor moves inside the same directory -- so without this guard the panel
    re-reads the folder on every click in the file list.
    """
    out = _drive(app_at, r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      for (let i = 0; i < 4; i++) {
        await m.onDirectoryChange(%s);
      }
      await __quiet();
      console.log(JSON.stringify({calls: __calls.length}));
    """ % json.dumps(str(described)))
    assert out["calls"] == 1, (
        "entering the same directory four times cost four reads")


def _set_up_by_the_panel(app_at, folder, answers):
    """The panel's Set-up clicked on ``folder``, each dialog it opens
    answered from ``answers`` in turn: the init requests it sent, the
    questions it asked, and the advisory it shows."""
    return _drive(app_at, r"""
      __answers = %s;
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      __els["ps-checkpoint-init"].click();
      await __quiet();
      console.log(JSON.stringify({
        inits: __calls.filter(c => c.url.includes("/init")).map(c => c.body),
        asked: __asked.length,
        advisory: __els["ps-checkpoint-advisory"].textContent,
      }));
    """ % (json.dumps(answers), json.dumps(str(folder))))


def test_setup_asks_for_a_name_only_when_a_name_is_what_is_missing(
        tmp_path, monkeypatch, app_at):
    """The refusal→question→retry path, executed end to end.

    A calculation whose folder's name cannot be a history's name -- `H2.v2`,
    the dot outside [A-Za-z0-9_-] (`checkpointing.md` L3): the first POST
    carries no name, the server refuses it with `where: "calculation"`, the
    panel asks, and the SECOND POST carries what the person typed -- under
    the right key -- and the folder is set up under it.
    """
    from molbuilder.checkpoint import Repo
    from support.road import described, road_tree
    folder = described(road_tree(tmp_path, monkeypatch),
                       "P/optimization/H2.v2")
    out = _set_up_by_the_panel(app_at, folder, [{"name": "H2_v2"}])
    inits = out["inits"]
    assert len(inits) == 2, "the panel did not retry after the refusal"
    assert "calculation" not in inits[0], "the ordinary attempt sends no name"
    assert inits[1]["calculation"] == "H2_v2", (
        "the retry did not carry the name the person typed")
    assert out["asked"] == 1, "exactly one question, and only because of `where`"
    assert Repo(str(folder)).calculation() == "H2_v2"


def test_setup_does_not_ask_for_a_name_when_a_name_would_not_help(
        app_at, described):
    """The other advisory on the same route, and the reason `where` exists.

    A person ran `git init` by hand in a folder of this calculation's,
    `notes/`, so that inner folder holds a history.  Set-up on the
    calculation is then refused -- a history inside a history -- and a name
    does not fix that.  Prompting for one would be asking a question whose
    answer cannot help, which is how a surface teaches people that its
    questions are noise.

    WHAT GOES UNSEEN: this advisory reached through molbuilder's own acts.
    Molbuilder begins no history inside a calculation's folder
    (`checkpointing.md` L1) and describes no calculation inside another's
    (`project-layout.md` § 7, invariant 2a), so git run by hand is the one
    road left to it.
    """
    notes = described / "notes"
    notes.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=notes, check=True)
    out = _set_up_by_the_panel(app_at, described, [{"name": "anything"}])
    assert out["asked"] == 0, "it asked for a name that cannot help"
    assert len(out["inits"]) == 1, "and it must not retry blindly"
    assert "already checkpoint folders" in out["advisory"], out["advisory"]


def test_wiring_the_panel_twice_does_not_double_every_click(app_at, described):
    """`initCheckpointPanel` says it is safe to call from several bootstrap
    paths; a second call that double-bound every listener would make one
    click on Set-up post twice.  One caller exists today — the docstring
    invites the second."""
    out = _drive(app_at, r"""
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      __calls.length = 0;
      __els["ps-checkpoint-init"].click();
      await __quiet();
      console.log(JSON.stringify({
        inits: __calls.filter(c => c.url.includes("/init")).length}));
    """ % json.dumps(str(described)))
    assert out["inits"] == 1, (
        "one click produced more than one request; the panel was wired twice")


# ================================================================== #
#  The deep read is covered; the ordinary one is not                 #
#                                                                    #
#  `Repo.status(deep=False)` -- every directory-enter -- runs four    #
#  git subprocesses and stats each file, reading no file's contents.  #
#  `deep=True`, which only the Refresh button asks for, hashes every  #
#  big file whose size still matches: `checkpoint.py` says a folder   #
#  holding a 2 GB density matrix must not be read end-to-end, which   #
#  is exactly what that path then does.  One is ui-contract § 10's    #
#  heavy user-triggered operation; the other is a click nobody waits  #
#  on, and covering it would flash the window for nothing.            #
# ================================================================== #


def _fence_walk(app_at, folder, body: str) -> dict:
    """Drive the real panel on ``folder`` and sample the fence from inside
    `fetch`.

    Sampled there rather than from a timer because a request leaves
    synchronously: anything scheduled from outside runs after the read is
    already under way, and would pass against a panel that claimed late.
    """
    return _drive(app_at, r"""
      const busy = (await import(new URL("../page-busy.js",
                                 process.env.PANEL_URL).href)).pageBusy;
      const seen = [];
      const inner = globalThis.fetch;
      globalThis.fetch = (url, opts = {}) => {
        seen.push({deep: url.includes("deep=1"), held: busy.isClaimed()});
        return inner(url, opts);
      };
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      %s
      console.log(JSON.stringify({seen, after: busy.isClaimed()}));
    """ % (json.dumps(str(folder)), body))


def test_entering_a_folder_does_not_cover_the_window(app_at, described):
    """A directory-enter reads no file's contents, so it takes nothing."""
    _set_up(described)
    out = _fence_walk(app_at, described, "")
    assert out["seen"], "the panel made no request at all"
    assert all(c["deep"] is False for c in out["seen"]), out["seen"]
    assert all(c["held"] is False for c in out["seen"]), (
        "entering a folder covered the window -- § 10 keeps the cover for "
        "work the person waits on")
    assert out["after"] is False


def test_the_refresh_button_covers_the_window_while_it_hashes(
        app_at, described):
    """Refresh asks for `deep=1`, which hashes every big file."""
    _set_up(described)
    out = _fence_walk(app_at, described, r"""
      __els["ps-checkpoint-refresh-btn"].click();
      await __quiet();
    """)
    deep = [c for c in out["seen"] if c["deep"]]
    assert deep, (
        f"Refresh did not ask for a deep read: {out['seen']}")
    assert all(c["held"] for c in deep), (
        "the deep read ran with the window uncovered -- it hashes "
        "gigabytes and nothing was blocking the click behind it")
    assert out["after"] is False, "the window stayed covered afterwards"


def test_cancelling_a_deep_read_says_nothing_rather_than_HTTP_200(
        app_at, described):
    """Cancel after the headers land but before the body is read.

    The `AbortError` a Cancel raises mid-body is a cancelled question
    (`lib/server.js`, `web/ui-contract.md` § 12), never a server that
    answered with nothing: read as one, pressing Cancel painted "HTTP 200"
    as an error.  Only the deep read carries a signal, so only it gets here.
    The server's answer is the route's; Cancel is pressed the moment the
    deep read's response reaches the panel, before the panel reads its
    body, and the read fails with the runtime's own AbortError
    (`_node_esm.served`).
    """
    _set_up(described)
    out = _drive(app_at, r"""
      const busy = (await import(new URL("../page-busy.js",
                                 process.env.PANEL_URL).href)).pageBusy;
      const inner = globalThis.fetch;
      globalThis.fetch = async (url, opts = {}) => {
        const r = await inner(url, opts);
        // THE HEADERS LANDED: Cancel, before the panel reads the body
        if (url.includes("deep=1")) queueMicrotask(() => busy._runCancelers());
        return r;
      };
      const m = await import(process.env.PANEL_URL);
      m.initCheckpointPanel();
      await m.onDirectoryChange(%s);
      await __quiet();
      const before = __els["ps-checkpoint-toggle"].dataset["attr_data-state"];
      __els["ps-checkpoint-refresh-btn"].click();
      await __quiet();
      console.log(JSON.stringify({
        before,
        pill: __els["ps-checkpoint-toggle"].dataset["attr_data-state"],
        held: busy.isClaimed(),
      }));
    """ % json.dumps(str(described)))
    assert out["before"] == "clean", out
    assert out["pill"] != "error", (
        f"Cancel painted a failure the user did not have: pill={out['pill']}")
    assert out["held"] is False, "the window stayed covered after Cancel"

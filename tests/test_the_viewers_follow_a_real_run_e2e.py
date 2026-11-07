"""THE VIEWERS FOLLOW A REAL RUN -- `web/results.md` § 4, § 4.1, § 4.2;
`web/trajectory.md` § 4, § 5; `web/spectra.md` § 3, § 7.

The trajectory viewer's and the spectra viewer's own functions -- lifted from
the shipped modules (`tests/_node_esm.lifted`) and run in node against fakes
at the edge: timers, abort controllers, the drawing calls -- each fed what the
viewer's routes answered of a real run made in this pass
(`support.real_runs`; `process/testing.md` § 6): ``/api/watch/load`` and
``/api/watch/data`` of an H2 relaxation, asked as the viewer asks them while
the run went and once it ended, and ``/api/spectra/load`` of a water
vibration likewise.  How the run is doing reaches a viewer only in those
answers (`runstatus.run_answer`), so a viewer that follows a run is asked
here of the answers a run really gives: live while it goes, finished, or
failed where SIESTA stopped it.

The runs, each launched in a copy of a shared calculation (`copy_of`), in the
background as a launch leaves a run, and followed while they go:

* :func:`followed` -- the shared H2's `coarse` launched again, cold;
* :func:`stopped` -- the shared stopped H2's `coarse` (``real_h2_stopped``,
  its SCF unable to converge in its cap) launched again, cold: SIESTA stops
  it, and the run is failed;
* :func:`vibration_followed` -- the shared straight water's `freq` launched
  again, cold;

and :func:`stage_shaped`, an H2 optimization on PySCF made by the shared
run's maker under the label `au_2_bdt` (`support.real_runs.pyscf`), whose
folder's listing the Results tab's picker is asked of.

Beside them, no engine: an H2 calculation prepared and never launched, whose
folder the viewer opens (prep writes the run's log) and the server answers
not launched.

Beside the viewers' follow: the trajectory viewer's rebuild into a read-only
MolView, its install answered by the app (`_node_esm.served`); the badge's
clocks on each answer; and the picker on real run folders' listings.

WHAT NO RUN HERE REACHES GOES UNSEEN (each retired from the node files whose
rows typed it, `process/testing.md` § 3b): a run ``queued``, which only a
scheduler holds (the field pass on a cluster); a point a launched run of
points never started, ``not run``, which no session run leaves (the
junction's own run has one, and its test asks the run door of it, not the
viewer); a launch record that does not read, ``unreadable``, which only a
record broken by hand makes; an uploaded output that ran out of
memory, or one whose format states no ending; an answer naming no file, which
no route sends; a poll whose write ended the run -- its new content and the
run over in one answer, which the server sends only when the last write lands
between its two reads of the file (`web/results.md` § 4.1), a moment no follow
here can be sure to meet; a parse stating one clock series without the other,
or an answer with no time at all; an ended run whose record states no wall; a
listing holding a broken link -- a file with no time, listed last and labelled
by its name alone -- two folders' files or two rungs' files, or a run whose
master was deleted; two files of a listing with one time, which the picker
orders by name -- the listing states each file's time as the filesystem keeps
it, to the fraction of a second, and no run here is sure to write two at once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from _road import conda_hook, env_available
from tests._node_esm import lifted, run_node

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")
             and env_available("molbuilder-pySCF")),
        reason="needs the molbuilder-siesta and molbuilder-pySCF envs + a "
               "detectable conda hook"),
]

_LIB = Path(__file__).resolve().parents[1] / "molbuilder/web/static/lib"
TRAJECTORY = _LIB / "trajectory" / "core.js"
SPECTRA = _LIB / "spectra" / "core.js"
LIFECYCLE = _LIB / "inspectors" / "lifecycle.js"

# --------------------------------------------------------------------- #
#  The runs, and what the viewers' routes answered of them               #
# --------------------------------------------------------------------- #

def _web():
    """The app's own routes, as a page reaches them (the rate limit off: a
    viewer following a run asks often)."""
    from molbuilder.web.app import create_app
    return create_app(config={"rate_limit": {"enabled": False}}).test_client()


def _launched(c, stage: str, *cold: str) -> Path:
    """``stage`` of ``c`` launched here, ``cold`` (``"--cold"``) or not,
    left in the background as a launch leaves a run (`job-system.md`
    § 6.0): the new run's folder, by the status door."""
    from support.real_runs import stage_status, taken
    taken(c, f"launch {stage}", "launch", "task", "--stage", stage,
          "--mode", "direct", "--yes", *cold)
    return c.attempt(stage, 0).parent / stage_status(c.bundle, stage).attempt


def _as_the_trajectory_viewer_follows(c, folder: Path) -> None:
    """THE RUN IN ``folder`` FOLLOWED AS THE TRAJECTORY VIEWER FOLLOWS IT
    (`web/results.md` § 4.1): its folder loaded once it opens, then polled
    with the mtime the last answer named, until an answer says the run is
    no longer live and one poll after.  Kept on ``c``: ``answers``, every
    answer in order, ``(what, body)``; ``growing``, the output's bytes and
    name as they stood once while the run was live and the file said it was
    running -- a run's file kept as it grew, replayed under its own name
    (`process/testing.md` § 6); ``final``, the folder loaded again once the
    run's process ended."""
    from support.real_runs import left_to_end
    web = _web()
    c.answers, c.growing = [], None
    mtime, over, end = None, False, time.monotonic() + 900
    while True:
        assert time.monotonic() < end, "the run did not end within 900 s"
        if mtime is None:
            r = web.post("/api/watch/load", json={"path": str(folder)})
            if r.status_code != 200:
                time.sleep(0.05)
                continue
            body, what = r.get_json(), "load"
        else:
            body = web.get("/api/watch/data",
                           query_string={"mtime": mtime}).get_json()
            what = "poll"
        assert body["ok"], body
        c.answers.append((what, body))
        mtime = body["mtime"]
        run = body.get("run")
        data = body.get("data")
        if (c.growing is None and run and run["live"] and data
                and data.get("run_state") == "running"):
            out = Path(body["path"])
            c.growing = (out.name, out.read_bytes())
        if over:
            break
        over = bool(run) and not run["live"]
        time.sleep(0.05)
    left_to_end(c)
    r = web.post("/api/watch/load", json={"path": str(folder)})
    assert r.status_code == 200, r.get_json()
    c.final = r.get_json()


def _caught(c, sights: dict) -> None:
    """THE FOLLOW SAW WHAT THE TESTS BELOW READ OF IT: each of ``sights`` --
    a state in words, and what of ``c``'s answers holds it -- or the fixture
    fails saying which the follow never saw.  A poll every 50 ms can miss a
    short state, and a test reaching for one gets a bare StopIteration."""
    missed = [words for words, held in sights.items() if not held]
    assert not missed, "the follow never saw " + "; nor ".join(missed)


def _uploaded(name: str, data: bytes) -> dict:
    """What ``/api/watch/load`` answers of ``data`` uploaded as ``name`` --
    the file picker's road for a file of no run."""
    import io
    r = _web().post("/api/watch/load", data={"file": (io.BytesIO(data), name)},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    return r.get_json()


@pytest.fixture(scope="module")
def followed(real_h2_layered, tmp_path_factory):
    """THE SHARED H2'S `coarse`, LAUNCHED AGAIN COLD IN A COPY of its
    calculation and followed while it went (:func:`_as_the_trajectory_viewer_
    follows`); then the copy's first run loaded, and polled, as a tab does
    that opens another file -- the poll's answer, ``other_poll`` -- and the
    output as it grew and as it ended, each uploaded -- ``uploads``."""
    from support.real_runs import copy_of, road_at

    def follow(c):
        with road_at(c.tree.parent, tmp_path_factory):
            began = time.monotonic()
            _as_the_trajectory_viewer_follows(c, _launched(c, "coarse", "--cold"))
            c.took = time.monotonic() - began
            print(f"\n[the followed run: {c.took:.1f} s]")
            web = _web()
            r = web.post("/api/watch/load",
                         json={"path": str(c.attempt("coarse", 0))})
            assert r.status_code == 200, r.get_json()
            c.other_poll = web.get("/api/watch/data").get_json()
        _caught(c, {
            "the run live with its output saying it was running": c.growing,
            "a live answer whose run was running": [
                b for b in _live(c.answers) if b["run"]["state"] == "running"],
            "a live answer carrying the output's parse": [
                b for b in _live(c.answers) if b.get("data")],
            "a poll bringing new content": _polls(c, changed=True,
                                                  run="absent"),
            "a quiet poll while the run was live": _polls(c, changed=False,
                                                          run="live"),
            "an answer holding frames and a later one holding more":
                _first_and_grown(c),
        })
        final_out = Path(c.final["path"])
        c.uploads = {"running": _uploaded(*c.growing),
                     "ended": _uploaded(final_out.name,
                                        final_out.read_bytes())}
    return copy_of(real_h2_layered, "followed", follow)


@pytest.fixture(scope="module")
def stopped(real_h2_stopped, tmp_path_factory):
    """THE SHARED STOPPED H2'S `coarse` (``real_h2_stopped``: its SCF unable
    to converge in its cap, set by the person in its template), LAUNCHED
    AGAIN COLD IN A COPY of its calculation and followed while it went until
    SIESTA stopped it (:func:`_as_the_trajectory_viewer_follows`); its
    output as it stopped, uploaded."""
    from support.real_runs import copy_of, road_at

    def follow(c):
        with road_at(c.tree.parent, tmp_path_factory):
            began = time.monotonic()
            _as_the_trajectory_viewer_follows(c, _launched(c, "coarse",
                                                           "--cold"))
            c.took = time.monotonic() - began
        print(f"\n[the stopped run, followed: {c.took:.1f} s]")
        assert c.final["run"]["state"] == "failed", c.final["run"]
        _caught(c, {"the run live": _live(c.answers),
                    "a quiet poll that found the run over":
                        _polls(c, changed=False, run="over")})
        out = Path(c.final["path"])
        c.uploads = {"stopped": _uploaded(out.name, out.read_bytes())}
    return copy_of(real_h2_stopped, "followed", follow)


@pytest.fixture(scope="module")
def vibration_followed(real_water_linear, tmp_path_factory):
    """THE SHARED STRAIGHT WATER'S `freq`, LAUNCHED AGAIN COLD IN A COPY and
    followed as the spectra viewer follows it (`web/spectra.md` § 7): its
    result asked of ``/api/spectra/load`` from its first write until an
    answer says the run is no longer live -- ``answers``, each that differs
    from the one before."""
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from molbuilder.task import of as task_of
    from support.real_runs import copy_of, left_to_end, road_at

    def follow(c):
        # THE ROAD OF THE RUN'S OWN ENGINE, as the calculation states it --
        # PySCF's, nothing put on PATH (`_road.live_pyscf`).
        with road_at(c.tree.parent, tmp_path_factory,
                     engine=str(task_of(c.bundle).engine)):
            began = time.monotonic()
            folder = _launched(c, "freq", "--cold")
            web, c.answers, end = _web(), [], time.monotonic() + 900
            while True:
                assert time.monotonic() < end, "not over within 900 s"
                result = on_label_at(folder, ROLE_SPECTRA) \
                    if folder.is_dir() else None
                if result is None or not result.is_file():
                    time.sleep(0.1)
                    continue
                r = web.post("/api/spectra/load", json={"path": str(result)})
                assert r.status_code == 200, r.get_json()
                body = r.get_json()
                if not c.answers or c.answers[-1] != body:
                    c.answers.append(body)
                if not body["run"]["live"]:
                    break
                time.sleep(0.1)
            left_to_end(c)
            c.took = time.monotonic() - began
            print(f"\n[the followed vibration: {c.took:.1f} s]")
            _caught(c, {"its result with the run live":
                        [a for a in c.answers if a["run"]["live"]]})
    return copy_of(real_water_linear, "followed", follow)


def _not_launched(prepared) -> dict:
    """What ``/api/watch/load`` answers of an H2 calculation prepared and
    never launched, asked of its run's folder as the viewer asks: prep opens
    the run and writes its log, so the folder opens, and the server says how
    the run is doing -- not launched.  No engine: prep alone."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.paths import attempt_dir
    from molbuilder.task import of as task_of
    from support.road import H2
    calculation, stage, _deck = prepared(H2, "siesta")
    folder = attempt_dir(
        stage_home(calculation, task_of(calculation), stage).dir, 0)
    r = _web().post("/api/watch/load", json={"path": str(folder)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _live(answers):
    """The answers that carried the run live."""
    return [b for _w, b in answers if b.get("run") and b["run"]["live"]]


# --------------------------------------------------------------------- #
#  The trajectory viewer settles by the run, or a file of no run by its  #
#  own ending (`web/results.md` § 4.1, `web/trajectory.md` § 4)          #
# --------------------------------------------------------------------- #

def _settled(answer) -> dict:
    """The REAL `_settlePostLoad` and `_badgeKind` on ``answer`` -- the run
    it carries and the file's own parse: the transition the settle made and
    the badge's kind."""
    run_state = lifted(TRAJECTORY, "const RUN_STATE = Object.freeze",
                       "});") + "});"
    settle = lifted(TRAJECTORY, "function _settlePostLoad",
                    "function plottableFrames")
    badge = lifted(TRAJECTORY, "function _badgeKind(run, data) {",
                   "\n    function ")
    return run_node([], f"""
        {run_state}
        const _seen = [];
        function transition(name) {{ _seen.push(name); state.machine = name; }}
        const answer = {json.dumps(answer)};
        const state = {{
            machine: "LOADING",
            fileState: {{ data: answer.data, run: answer.run }},
        }};
        {settle}
        {badge}
        _settlePostLoad();
        console.log(JSON.stringify({{
            seen: _seen, badge: _badgeKind(answer.run, answer.data) }}));
    """)


#: (case, how the answer is found -- of the followed run, the stopped one,
#: or a calculation prepared here -- the transition the settle makes, the
#: badge's kind).  The badge's words for its kinds, `web/trajectory.md` § 4:
#: `finished` Finished, `stopped` Stopped, `pending` Not launched, `running`
#: Running.
_SETTLES = [
    ("a run the server says is running is followed",
     lambda f, s, p: next(b for b in _live(f.answers)
                          if b["run"]["state"] == "running"),
     "WATCHING", "running"),
    ("a finished run is not followed",
     lambda f, s, p: f.final, "LOADED", "finished"),
    ("a run SIESTA stopped is failed, and stops",
     lambda f, s, p: s.final, "ERROR", "stopped"),
    ("a run never launched is not followed -- nothing arrives until it is",
     lambda f, s, p: _not_launched(p), "LOADED", "pending"),
    ("an upload is never followed, whatever its file says",
     lambda f, s, p: f.uploads["running"], "LOADED", "running"),
    ("an upload that ended is loaded",
     lambda f, s, p: f.uploads["ended"], "LOADED", "finished"),
    ("an upload that stopped settles on its own ending",
     lambda f, s, p: s.uploads["stopped"], "ERROR", "stopped"),
]


@pytest.mark.parametrize("case, answer, want, badge", _SETTLES,
                         ids=[c[0] for c in _SETTLES])
def test_the_trajectory_viewer_follows_the_run(case, answer, want, badge,
                                               followed, stopped, prepared):
    """After every answer ``_settlePostLoad`` decides whether the viewer
    keeps polling -- by the run the server sent, never by the file's own
    ending: an output that states its end belongs to a job that may still be
    deriving its result, and a run killed mid-step states nothing at all.  A
    file that belongs to no run -- an upload -- has nothing to follow, and
    its own ending says only whether it stopped.  The badge reads the run
    the file belongs to, or an upload's own ending.

    THE SILENT FAILURES: a run killed mid-step polled every 15 s until the
    page closes (plan W38 M2f); a live run left unwatched; a run never
    launched polled, or shown Running; an upload that ended shown Running
    for as long as anyone looks."""
    got = _settled(answer(followed, stopped, prepared))
    assert got == {"seen": [want], "badge": badge}, case


# --------------------------------------------------------------------- #
#  The poll loop takes how the run is doing from each answer             #
# --------------------------------------------------------------------- #

def _poll_once(answer, *, held, path):
    """Run the REAL `pollOnce` -- `/api/watch/data` answering ``answer`` --
    on a viewer WATCHING ``path`` and holding ``held``, the run's state the
    last answer that carried one said.  Faked at the edge: the transitions
    it asks for (recorded), the applier, the badge, the status line --
    except `_settlePostLoad`, lifted and run for real."""
    run_state_const = lifted(TRAJECTORY, "const RUN_STATE = Object.freeze",
                             "});") + "});"
    settle = lifted(TRAJECTORY, "    function _settlePostLoad",
                    "\n    function plottableFrames")
    poll = lifted(TRAJECTORY, "    async function pollOnce()",
                  "\n    function ")
    return run_node([], f"""
        {run_state_const}
        const _seen = [];
        function transition(name, payload) {{
            _seen.push(name);
            if (name === "APPLY") {{
                if (payload.run !== undefined) state.fileState.run = payload.run;
                return;
            }}
            state.machine = name;
        }}
        function applyNewData(r) {{
            state.fileState.data = r.data;
            if (r.run !== undefined) state.fileState.run = r.run;
        }}
        function _renderBadge() {{}}
        function setStatus() {{}}
        function _withRunNotes() {{}}
        function dispose() {{}}
        const state = {{
            machine: "WATCHING",
            mtime: 1,
            data: {{ frames: [] }},
            fileState: {{ path: {json.dumps(path)}, mtime: 1,
                         data: {{ run_state: "running", frames: [] }},
                         run: {json.dumps(held)} }},
            lifecycle: {{ pollInFlight: false, pollAbort: null }},
            derived: {{}},
        }};
        globalThis.fetch = async () => ({{
            ok: true,
            json: async () => ({json.dumps(answer)}),
        }});
        {settle}
        {poll}
        pollOnce().then(() => console.log(JSON.stringify({{
            transitions: _seen, machine: state.machine,
        }})));
    """)


def _polls(f, *, changed, run):
    """``f``'s poll answers: with new content or none, and carrying the run
    ``"live"``, ``"over"`` or ``"absent"``."""
    def carries(b):
        r = b.get("run", "absent")
        return {"absent": r == "absent",
                "live": isinstance(r, dict) and r["live"],
                "over": isinstance(r, dict) and not r["live"]}[run]
    return [b for w, b in f.answers
            if w == "poll" and b["changed"] is changed and carries(b)]


def test_a_quiet_poll_that_finds_the_run_over_settles_it(stopped):
    """THE CASE M2f EXISTS FOR (`results.md` § 4.1).  A run killed mid-step
    leaves an output that never changes again and states nothing; the quiet
    poll brings how the run is doing -- failed, as SIESTA stopped it -- and
    the poll loop must take it and settle, or the viewer polls for as long
    as the tab is open."""
    held = _live(stopped.answers)[0]["run"]
    quiet = _polls(stopped, changed=False, run="over")[-1]
    out = _poll_once(quiet, held=held, path=stopped.final["path"])
    assert out["machine"] == "ERROR", out["transitions"]


def test_a_quiet_poll_on_a_live_run_keeps_watching(followed):
    """A quiet poll while the run is live brings its state, live, and the
    viewer keeps following (`results.md` § 4.1: it polls while the run is
    live).  THE SILENT FAILURE: a run still going taken as over at its first
    quiet poll, its later frames never shown."""
    quiet = _polls(followed, changed=False, run="live")[0]
    out = _poll_once(quiet, held=quiet["run"], path=followed.final["path"])
    assert out["machine"] == "WATCHING", out["transitions"]


def test_a_poll_with_new_content_keeps_the_runs_state_it_held(followed):
    """New content leaves the run's state out -- the run was writing -- so
    the viewer keeps following on what it held."""
    held = _live(followed.answers)[0]["run"]
    grown = _polls(followed, changed=True, run="absent")[0]
    out = _poll_once(grown, held=held, path=grown["path"])
    assert out["machine"] == "WATCHING", out["transitions"]


# --------------------------------------------------------------------- #
#  An answer is written under its own file's name, or not at all         #
# --------------------------------------------------------------------- #

def _applied(answer, *, machine, path, held, movie=False):
    """Run the REAL `applyNewData(answer)` -- and through it the REAL
    `transition("APPLY")` -- on the REAL state, the viewer in ``machine``
    showing ``path`` with ``held`` parsed; report the file it then holds,
    its time, what reached the movie, what it drew, or what it threw.

    Faked at the edge: MolView -- with ``movie``, a stand-in holding as many
    frames as ``held`` that records what is appended to it, so a poll that
    provably extends the movie takes the append; without, none mounted --
    and the drawing calls, each recorded by name."""
    frames = len(held["frames"]) if held else 0
    harness = f"""
        const root = globalThis;
        {lifted(TRAJECTORY, "    const state = {",
                "\n    (function _wireBackcompatAliases")}
        {lifted(TRAJECTORY, "    (function _wireBackcompatAliases",
                "\n    // Transition orchestrator")}
        {lifted(TRAJECTORY, "    function transition(target, payload) {",
                "\n    function ")}
        {lifted(TRAJECTORY, "    function _latticeEqual(a, b) {",
                "\n    // Compare frame")}
        {lifted(TRAJECTORY, "    function _frameEqualAt(",
                "\n    function _scfFingerprint(")}
        {lifted(TRAJECTORY, "    function _scfFingerprint(data) {",
                "\n    function ")}
        {lifted(TRAJECTORY, "    function applyNewData(r) {",
                "\n    function ")}
        const _appended = [];
        let _count = {frames};
        const _movie = {{
            frameCount: () => _count,
            currentFrame: () => _count - 1,
            addFrames: (fs) => {{ _appended.push(...fs); _count += fs.length; }},
            setCurrentFrame: () => {{}},
        }};
        let _mv = {json.dumps(movie)} ? {{}} : null;
        const _mvdata = () => (_mv ? _movie : null);
        function startPolling() {{}}
        function stopPolling() {{}}
        const _drawn = [];
        function _renderRuntimeInfo() {{ _drawn.push("runtime info"); }}
        function _renderParseWarnings() {{ _drawn.push("parse warnings"); }}
        function makePlots() {{ _drawn.push("plots"); }}
        function _renderBadge() {{ _drawn.push("badge"); }}
        function setStatus() {{ _drawn.push("status"); }}
        function rebuildModel() {{ _drawn.push("movie"); }}
        function drawForces() {{ _drawn.push("forces"); }}
        function _announceReady() {{}}
        function _withRunNotes() {{}}
        function _cellCameFromTheRun() {{ return false; }}
        state.machine = {json.dumps(machine)};
        state.fileState.path = {json.dumps(path)};
        state.fileState.data = {json.dumps(held)};
        state.fileState.mtime = 1;
        let threw = null;
        try {{ applyNewData({json.dumps(answer)}); }}
        catch (e) {{ threw = String((e && e.message) || e); }}
        console.log(JSON.stringify({{
            path: state.fileState.path,
            data: state.fileState.data,
            mtime: state.fileState.mtime,
            appended: _appended.length,
            drawn: _drawn,
            threw,
        }}));
    """
    return run_node([LIFECYCLE], harness)


def _another_files(f):
    """A poll's answer for ANOTHER file than the followed run's: the copy's
    first run, loaded and polled as a tab does that opens it."""
    assert f.other_poll["changed"] and f.other_poll["path"] != f.final["path"]
    return f.other_poll


def _first_and_grown(f):
    """The followed run's first answer that holds frames, and the first
    later answer for the same file holding more -- what a viewer showing
    the run holds, and a poll that grew it; ``None`` where the follow saw
    no such pair."""
    with_frames = [b for _w, b in f.answers
                   if b.get("data") and b["data"]["frames"]]
    return next(((with_frames[0], b) for b in with_frames
                 if len(b["data"]["frames"])
                 > len(with_frames[0]["data"]["frames"])), None)


@pytest.mark.parametrize("machine", ["WATCHING", "LOADED", "ERROR"])
def test_another_files_answer_writes_nothing_and_draws_nothing(followed,
                                                               machine):
    """`results.md` § 4.2: *"A name and the data that belongs to it are
    written together, or neither is"* -- a poll's answer for a file not on
    screen is dropped and draws nothing, whatever the viewer was doing when
    the poll went out: a tab coming back into view polls a file that settled
    or stopped too.

    THE SILENT FAILURE.  `/api/watch/data` answers for whatever file the
    server is watching, so a poll's answer can belong to another file than
    the one on screen; written -- or drawn -- its frames appear under this
    file's name, or its run stops this file's follow, and nothing says so.
    Here the server watches the copy's first run, as it does once another
    tab opened it, and the viewer shows the followed run."""
    held = followed.final
    out = _applied(_another_files(followed), machine=machine,
                   path=held["path"], held=held["data"])
    assert out["threw"] is None, out
    assert out["path"] == held["path"] and out["data"] == held["data"], out
    assert out["mtime"] == 1 and out["drawn"] == [], out


def test_another_files_new_frames_never_reach_a_movie_that_could_take_them(
        followed):
    """The append arm guarded as the rebuild is: a movie holding the
    followed run's first frames is offered another file's answer, and
    nothing is appended."""
    first, _grown = _first_and_grown(followed)
    out = _applied(_another_files(followed), machine="WATCHING",
                   path=first["path"], held=first["data"], movie=True)
    assert out["threw"] is None, out
    assert out["appended"] == 0 and out["drawn"] == [], out
    assert out["data"] == first["data"], out


@pytest.mark.parametrize("movie", [True, False],
                         ids=["appended to its movie", "no movie mounted"])
def test_the_file_on_screens_new_frames_are_written(followed, movie):
    """The followed run's poll that grew its file: written under the file's
    name, with its time -- on a movie holding its first frames, the new
    frames appended, the same start making it a provable continuation."""
    first, grown = _first_and_grown(followed)
    out = _applied(grown, machine="WATCHING", path=first["path"],
                   held=first["data"], movie=movie)
    assert out["threw"] is None, out
    assert out["data"] == grown["data"] and out["mtime"] == grown["mtime"]
    assert out["appended"] == ((len(grown["data"]["frames"])
                                - len(first["data"]["frames"])) if movie
                               else 0), out


def test_an_answer_with_no_new_frame_for_the_file_on_screen_is_written(
        followed):
    """A poll whose file grew by SCF cycles and no frame -- the step in
    flight -- written under the file's name with its time: the arm that
    keeps the movie as it is still takes the answer."""
    answers = [b for _w, b in followed.answers
               if b.get("data") and b["data"]["frames"]]
    pair = next(((a, b) for a, b in zip(answers, answers[1:])
                 if len(a["data"]["frames"]) == len(b["data"]["frames"])
                 and a["data"] != b["data"]), None)
    assert pair, ("the followed run gave no two answers of one frame count "
                  "with different content")
    held, answer = pair
    out = _applied(answer, machine="WATCHING", path=held["path"],
                   held=held["data"])
    assert out["threw"] is None, out
    assert out["data"] == answer["data"], out
    assert out["mtime"] == answer["mtime"], out


def test_a_load_takes_the_name_of_the_file_the_server_read(followed):
    """The viewer asked for the run's folder; the load names the output the
    run door opened in it, and the viewer holds it under that name."""
    load = followed.final
    folder = str(Path(load["path"]).parent)
    out = _applied(load, machine="LOADING", path=folder, held=None)
    assert out["threw"] is None, out
    assert out["path"] == load["path"] and out["data"] == load["data"], out


def test_after_dispose_a_late_answer_writes_nothing(followed):
    """An answer that lands after the viewer was disposed -- IDLE, its file
    null -- writes nothing (`results.md` § 4.2: a fetch in flight when the
    viewer was disposed fails the name's check, the file on screen null by
    then).  THE SILENT FAILURE: a poll that left before the viewer closed
    writing the old run into the state it left."""
    _first, grown = _first_and_grown(followed)
    out = _applied(grown, machine="IDLE", path=None, held=None)
    assert out["path"] is None and out["data"] is None, out


# --------------------------------------------------------------------- #
#  The spectra viewer follows the run while it is live                   #
# --------------------------------------------------------------------- #

def _spectra_viewer(script, *, machine, path, results, run, replies=None,
                    polling=True):
    """The spectra viewer's state -- ``machine``, showing ``path`` with
    ``results`` and holding ``run``, its poll timer running when
    ``polling`` -- driven by ``script`` through the REAL `transition`,
    `_settlePostLoad` and `watchTick`, each request for the result answered
    by the next of ``replies``: what it then holds, what it drew, and how
    many times it asked.

    Faked at the edge: the timers and the abort controllers, the drawing
    calls, and how the request is carried (``fetchResults``: the answer, and
    the ask it went with)."""
    harness = f"""
        const _aborted = [];
        const _ac = (name) => ({{ abort: () => _aborted.push(name) }});
        let _polling = {json.dumps(polling)};
        globalThis.setInterval = () => {{ _polling = true; return 7; }};
        globalThis.clearInterval = () => {{ _polling = false; }};
        {lifted(SPECTRA, "const WATCH_INTERVAL_MS", ";")};
        {lifted(SPECTRA, "const WATCH_MAX_ERRORS", ";")};
        const _drawn = [];
        // THE ANSWERS, one a request in turn -- each as the route gave it --
        // then none: a request past them stays on the wire.  ``_onAsk``
        // runs as a request leaves, after what it asks is read.
        const _replies = {json.dumps(replies or [])};
        let _ask = {{}};
        let _asks = 0;
        let _onAsk = () => {{}};
        function _askOf() {{ return _ask; }}
        const fetchResults = async (p, signal) => {{
            const asked = JSON.stringify(_askOf());
            const n = _asks++;
            _onAsk(n, p);
            if (n >= _replies.length) return new Promise(() => {{}});
            return {{ path: p, body: _replies[n], asked }};
        }};
        function setStatus() {{ _drawn.push("status"); }}
        function stopWatch() {{ _drawn.push("stopped"); }}
        function renderResults() {{ _drawn.push("results"); }}
        function updatePhaseIndicator() {{ _drawn.push("phases"); }}
        function _showRunStatus() {{ _drawn.push("run status"); }}
        const _listeners = {{ disposeAll() {{}} }};
        const document = {{}};
        let chart = null, chartReady = null;
        const els = {{}};
        const root = globalThis;
        const state = {{
            machine: {json.dumps(machine)},
            fileState: {{ path: {json.dumps(path)},
                         results: {json.dumps(results)},
                         run: {json.dumps(run)} }},
            viewState: {{ selectedMode: 3 }},
            uiPrefs: {{}},
            lifecycle: {{ loadAbort: _ac("load"), watchAbort: _ac("watch"),
                         watchInFlight: false,
                         watchTimer: {json.dumps(7 if polling else None)},
                         watchErrors: 0 }},
            derived: {{}},
            vib: null,
        }};
        {lifted(SPECTRA, "    function transition(target, payload) {",
                "\n    function ")}
        {lifted(SPECTRA, "    function _abortViews() {", "\n    }")}
        }}
        {lifted(SPECTRA, "    function _settlePostLoad() {", "\n    }")}
        }}
        {lifted(SPECTRA, "    async function watchTick() {",
                "\n    function ")}
        {lifted(SPECTRA, "    function _askAgainIfChanged(asked) {", "\n    }")}
        }}
        (async () => {{
        {script}
        console.log(JSON.stringify({{
            machine: state.machine,
            run: state.fileState.run,
            results: state.fileState.results,
            polling: _polling,
            drawn: _drawn,
            asks: _asks,
        }}));
        }})();
    """
    return run_node([], harness)


def test_a_run_the_server_says_is_live_is_followed(vibration_followed):
    """`web/spectra.md` § 7: after a load, the viewer follows the run the
    server named while it is live -- here the water vibration's first
    answer, its run going."""
    load = vibration_followed.answers[0]
    assert load["run"]["live"], load["run"]
    out = _spectra_viewer("_settlePostLoad();", machine="LOADING",
                          path="the result", results=load["results"],
                          run=load["run"], polling=False)
    assert out["machine"] == "WATCHING" and out["polling"] is True, out


@pytest.mark.parametrize("which", ["finished", "of no run"])
def test_a_run_no_longer_live_is_not_followed(which, vibration_followed,
                                              real_water_linear, monkeypatch):
    """...and not once it is over -- the vibration's last answer, its run
    finished -- or for a file that belongs to no run: a version of the
    shared water's result kept as its run wrote it, in a tree of its own
    (`RealRun.versions`), which the route answers with no run."""
    from conftest import point_the_one_door_at
    from molbuilder.projects import find_projects_root
    if which == "finished":
        answer = vibration_followed.answers[-1]
        assert answer["run"]["state"] == "finished", answer["run"]
    else:
        kept = real_water_linear.versions[-1]
        point_the_one_door_at(find_projects_root(kept).parent,
                              monkeypatch.setenv)
        answer = _web().post("/api/spectra/load",
                             json={"path": str(kept)}).get_json()
        assert answer["ok"] and answer["run"] is None, answer
    out = _spectra_viewer("_settlePostLoad();", machine="LOADING",
                          path="the result", results=answer["results"],
                          run=answer["run"])
    assert out["machine"] == "LOADED" and out["polling"] is False, out


def test_a_poll_that_says_the_run_finished_is_drawn_and_ends_the_follow(
        vibration_followed):
    """A tick on a live run answered with the run finished: the answer is
    drawn, and the follow ends."""
    live = next(a for a in vibration_followed.answers if a["run"]["live"])
    last = vibration_followed.answers[-1]
    out = _spectra_viewer("await watchTick();", machine="WATCHING",
                          path="the result", results=live["results"],
                          run=live["run"], replies=[last])
    assert out["machine"] == "LOADED" and out["polling"] is False, out
    assert out["run"] == last["run"], out
    assert out["drawn"] == ["results", "phases", "run status"], out


def test_a_poll_on_a_live_run_is_drawn_and_keeps_following(vibration_followed):
    """A tick on a live run answered with the run still going: drawn, the
    follow kept, and -- the answer to the ask as it stands -- asked nothing
    more."""
    live = [a for a in vibration_followed.answers if a["run"]["live"]]
    out = _spectra_viewer("await watchTick();", machine="WATCHING",
                          path="the result", results=live[0]["results"],
                          run=live[0]["run"], replies=[live[-1]])
    assert out["machine"] == "WATCHING" and out["polling"] is True, out
    assert out["drawn"] == ["results", "phases", "run status"], out
    assert out["asks"] == 1, out


def test_a_poll_answering_an_older_ask_is_drawn_then_the_file_asked_again(
        vibration_followed):
    """The person applies a laser by Update while the poll is out: its
    answer is to the older ask -- drawn, and the file asked again at the
    ask as it stands."""
    live = [a for a in vibration_followed.answers if a["run"]["live"]]
    out = _spectra_viewer(
        '_onAsk = (n) => { if (n === 0) _ask = { raman: { laser_nm: 633 } }; };'
        ' await watchTick();',
        machine="WATCHING", path="the result", results=live[0]["results"],
        run=live[0]["run"], replies=[live[-1]])
    assert out["drawn"] == ["results", "phases", "run status"], out
    assert out["asks"] == 2, out


def test_a_poll_answered_after_the_same_file_was_opened_again_draws_nothing(
        vibration_followed):
    """`results.md` § 4.2: the same file opened again while a tick is out
    lets that tick go -- its answer, the run's own, draws nothing, and the
    reload's state stands."""
    live = [a for a in vibration_followed.answers if a["run"]["live"]]
    out = _spectra_viewer(
        '_onAsk = (n, p) => { if (n === 0) transition("LOADING", { path: p }); };'
        ' await watchTick();',
        machine="WATCHING", path="the result", results=live[0]["results"],
        run=live[0]["run"], replies=[live[-1]])
    assert out["machine"] == "LOADING" and out["results"] is None, out
    assert out["polling"] is False and out["drawn"] == [], out


# --------------------------------------------------------------------- #
#  The phase dots read the result's own record (`web/spectra.md` § 3)    #
# --------------------------------------------------------------------- #

_PHASES = ["relaxation", "frequencies", "raman", "ir", "es"]


def _dots(results) -> dict:
    """The REAL `updatePhaseIndicator(results)` -- with what it asks of the
    record, `_routeHas` and `_asked` -- on the partial's phase indicator: a
    dot per phase, each in its own `.spectra-phase` label, the indicator
    hidden as the partial ships it.  Whether it then shows, and each dot's
    class, title and whether its label shows."""
    def fn(name):
        return lifted(SPECTRA, f"    function {name}(", "\n    }") + "\n    }"
    return run_node([], f"""
        const dots = {json.dumps(_PHASES)}.map((phase) => {{
            const wrap = {{ hidden: false }};
            return {{ dataset: {{ phase }}, className: "spectra-phase-dot",
                     title: "", _wrap: wrap,
                     closest: (sel) => (sel === ".spectra-phase" ? wrap : null) }};
        }});
        const els = {{ phaseIndicator: {{
            hidden: true,
            querySelectorAll: (sel) => (sel === ".spectra-phase-dot" ? dots : []),
        }} }};
        {fn("_routeHas")}
        {fn("_asked")}
        {fn("updatePhaseIndicator")}
        updatePhaseIndicator({json.dumps(results)});
        console.log(JSON.stringify({{
            hidden: els.phaseIndicator.hidden,
            dots: Object.fromEntries(dots.map((d) => [d.dataset.phase,
                {{ className: d.className, title: d.title,
                   shown: !d._wrap.hidden }}])),
        }}));
    """)


def _every_state(runs):
    """Each distinct phase record the shared vibrations' results passed
    through, as the load door serves it: every version kept as each run
    wrote it, and each run's own file at its end."""
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    seen = {}
    for run in runs:
        for path in [*run.versions, on_label_at(run.attempt("freq"),
                                                ROLE_SPECTRA)]:
            d = json.loads(path.read_text(encoding="utf-8"))
            key = json.dumps({p: d.get(f"phase_{p}") for p in _PHASES},
                             sort_keys=True)
            seen.setdefault(key, path)
    return list(seen.values())


def test_each_dot_says_its_phase_as_the_result_records_it(
        real_water_vibration, real_water_held, real_water_probed,
        real_water_linear, monkeypatch):
    """`web/spectra.md` § 3, on every phase record a real run's result
    passed through: the indicator shows on the first answer; each dot's
    class is its phase's state as one token, `spectra-phase-<state>`, and
    its title `<phase>: <state>`; a dot shows for each phase the run asked
    for -- its record's `not requested` none -- and the relaxation and the
    frequencies always.

    THE SILENT FAILURES: the dots left hidden; a phase drawn in another's
    state; a dot for a phase the run never asked for, read from the
    switch's presence in the file's config instead of its record."""
    from conftest import point_the_one_door_at
    from molbuilder.projects import find_projects_root
    web = _web()
    for path in _every_state([real_water_vibration, real_water_held,
                              real_water_probed, real_water_linear]):
        point_the_one_door_at(find_projects_root(path).parent,
                              monkeypatch.setenv)
        results = web.post("/api/spectra/load",
                           json={"path": str(path)}).get_json()["results"]
        out = _dots(results)
        assert out["hidden"] is False, path
        for phase in _PHASES:
            said = results[f"phase_{phase}"]
            dot = out["dots"][phase]
            assert dot["className"] == ("spectra-phase-dot spectra-phase-"
                                        + said.replace(" ", "-")), (path, phase)
            assert dot["title"] == f"{phase}: {said}", (path, phase)
            always = phase in ("relaxation", "frequencies")
            assert dot["shown"] is (always or said != "not requested"), (
                path, phase, said)


# --------------------------------------------------------------------- #
#  The rebuild reaches the read-only viewer (`web/trajectory.md` § 5)   #
# --------------------------------------------------------------------- #

MOLVIEW_MODEL = _LIB / "molview" / "model.js"


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers MolView's own requests
    (`support.live_server.serve`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


def test_a_rebuild_replaces_the_movie_in_a_read_only_viewer(followed, app_at):
    """A live run's poll is applied as an append when it provably extends
    the movie, and as a REBUILD otherwise; the rebuild installs the whole
    series again into the viewer the tab mounted read-only, which takes a
    second install only when its host says it means it (`web/molview.md`
    § 9.4, ``enforce``).  Without it the install was refused in silence: the
    frame counter grew while the movie kept the old frames.

    The viewer's REAL `rebuildModel` against MolView's real model,
    read-only, its install answered by the app's own `/api/build/load`: the
    followed run's first answer holding frames, then the poll that grew it.
    The viewer holds every frame of each, where the run says it is."""
    from tests._node_esm import served
    first, grown = _first_and_grown(followed)
    # FRAME 0 AS THE LOAD ASSEMBLES IT from the run's first frame -- a load
    # made before the run wrote a frame carries none, so the folder loaded
    # again once it ended carries it.
    frame0 = followed.final["structure"]
    assert frame0, followed.final
    rebuild = lifted(TRAJECTORY, "    async function rebuildModel(seekIdx) {",
                     "    // (No local showFrame()")
    out = run_node([], f"""
        const {{ createModel }} = await import(
            {json.dumps(MOLVIEW_MODEL.resolve().as_uri())});
        const m = createModel({{ mode: "readonly" }});
        const _mv = {{}};
        const _mvReady = Promise.resolve();
        const _mvdata = () => m;
        const $ = () => null;
        const said = [];
        function setStatus(text) {{ said.push(text); }}
        function buildForcesPerFrame() {{ return null; }}
        const state = {{ structure: {json.dumps(frame0)},
                        path: {json.dumps(first["path"])},
                        data: {json.dumps(first["data"])} }};
        {rebuild}
        const held = [];
        await rebuildModel(state.data.frames.length - 1);
        let s = m.getStructure();
        held.push(s ? (s.frames || [s.positions]) : null);
        state.data = {json.dumps(grown["data"])};
        await rebuildModel(state.data.frames.length - 1);
        s = m.getStructure();
        held.push(s ? (s.frames || null) : null);
        console.log(JSON.stringify({{ held, said }}));
    """, globals_js=served(app_at))
    assert out["said"] == [], out["said"]
    assert [bool(frames) for frames in out["held"]] == [True, True], (
        "a rebuild left the viewer holding no frame set", out["held"])
    for answer, frames in zip((first, grown), out["held"]):
        want = [[atom[1:4] for atom in frame]
                for frame in answer["data"]["frames"]]
        assert len(frames) == len(want), (len(frames), len(want))
        for got_frame, want_frame in zip(frames, want):
            for got, w in zip(got_frame, want_frame):
                assert got == pytest.approx(w, abs=1e-9)


# --------------------------------------------------------------------- #
#  The badge reads the right clock (`model/parse.md` § 2a,             #
#  `web/trajectory.md` § 4)                                             #
# --------------------------------------------------------------------- #

def _clocks(answer) -> dict:
    """The REAL `badgeClocks` on the viewer holding ``answer``: its parse,
    its file's time and the run it carries."""
    fn = lifted(TRAJECTORY, "    function badgeClocks(state) {",
                "\n    /* Wall-clock formatter")
    return run_node([], f"""
        {fn}
        const a = {json.dumps(answer)};
        console.log(JSON.stringify(badgeClocks({{
            data: a.data, mtime: a.mtime,
            fileState: {{ run: a.run === undefined ? null : a.run }} }})));
    """)


def _last(series):
    return series[-1] if series else None


def _live_with_data(f):
    """``f``'s answers that carried the run live and the output's parse."""
    return [b for b in _live(f.answers) if b.get("data")]


def _a_pyscf_log_uploaded(shaped, monkeypatch) -> dict:
    """What ``/api/watch/load`` answers of ``shaped``'s progress log, the
    PySCF run's, uploaded: a file of no run."""
    from molbuilder.runfiles import ROLE_MOLWATCH
    listing = _listing(shaped.attempt("coarse"), monkeypatch)
    log = Path(next(f["path"] for f in listing["files"]
                    if f["role"] == ROLE_MOLWATCH))
    return _uploaded(log.name, log.read_bytes())


#: (case, how the answer is found -- of the followed run, the stopped one,
#: the PySCF run labelled `au_2_bdt`, with the test's monkeypatch -- and what
#: the case's answer states, so the rule asks something of it: a run that
#: ended, its engine's wall; a PySCF log uploaded, BOTH CLOCKS, each in its
#: own slot -- an epoch as the time, how far in as the duration -- where a
#: SIESTA `.out` states neither, its steps carrying no time of day, and the
#: file's own time stands in).
_CLOCKS = [
    ("running, before its first step",
     lambda f, s, p, mp: _live_with_data(f)[0], None),
    ("running, in its steps",
     lambda f, s, p, mp: _live_with_data(f)[-1], None),
    ("finished", lambda f, s, p, mp: f.final,
     lambda got, run: isinstance(run["engine_elapsed_s"], (int, float))),
    ("failed", lambda f, s, p, mp: s.final, None),
    ("an upload", lambda f, s, p, mp: f.uploads["ended"], None),
    ("a PySCF log uploaded", lambda f, s, p, mp: _a_pyscf_log_uploaded(p, mp),
     lambda got, run: (got["elapsed"] is not None and got["elapsed"] < 1e6
                       and got["lastResultEpoch"] > 1e9)),
]


@pytest.mark.parametrize("case, answer, states", _CLOCKS,
                         ids=[c[0] for c in _CLOCKS])
def test_the_badge_reads_each_clock_from_its_own_series(
        case, answer, states, followed, stopped, stage_shaped, monkeypatch):
    """Two per-frame time series, not interchangeable: ``wall_clock_s`` is an
    absolute epoch and may be shown as a date; ``elapsed_s`` counts from the
    run's start and may be shown as a duration.  Feeding one to the other's
    formatter made a six-minute SIESTA run show "last result Dec 31,
    5:06 PM".  And a run that has ended says its own length, the engine's
    wall its record states -- served with the run's state -- where a run
    still going, or a file of no run, says how far in its last step was; the
    file's own time stands in for an epoch the parse has none of.

    THE SILENT FAILURES: an epoch offered as the duration ('55 years'); a
    duration offered as the timestamp; an ended SIESTA run reading *total
    0s* from a timer line that timed one SCF iteration, beside a Run panel
    saying 24 s (2.56.1)."""
    answer = answer(followed, stopped, stage_shaped, monkeypatch)
    data, run = answer["data"], answer.get("run")
    got = _clocks(answer)
    assert states is None or states(got, run), (case, got, run)
    if run and not run["live"]:
        assert got["elapsed"] == run["engine_elapsed_s"], (got, run)
    else:
        assert got["elapsed"] == _last(data.get("elapsed_s") or []), got
    wall = _last(data.get("wall_clock_s") or [])
    assert got["lastResultEpoch"] == (wall if wall is not None
                                      else answer["mtime"]), got
    assert got["endedLocal"] == ((data.get("runtime_info") or {})
                                 .get("run_end_local") or None), got


# --------------------------------------------------------------------- #
#  The Results tab's picker on a real run folder's listing              #
#  (`web/results.md` § 0.3, § 2.3)                                      #
# --------------------------------------------------------------------- #

PICKER = _LIB / "results" / "file-picker.js"
TIME_AGO = _LIB / "time-ago.js"
INSPECTORS = _LIB / "inspectors"


@pytest.fixture(scope="module")
def stage_shaped(tmp_path_factory):
    """AN H2 OPTIMIZATION ON PYSCF LABELLED `au_2_bdt` -- a label holding a
    part shaped like a stage -- made by the shared PySCF run's maker under
    that label (`support.real_runs.pyscf`): `coarse` prepared, launched and
    waited on, the run a picker once listed as four entries (2026-09-19),
    its label cut at the leftmost `_<digits>_`."""
    from support.real_runs import pyscf
    began = time.monotonic()
    run = pyscf(tmp_path_factory, label="au_2_bdt")
    run.took = time.monotonic() - began
    print(f"\n[the au_2_bdt PySCF run: {run.took:.1f} s]")
    return run


def _listing(folder, monkeypatch) -> dict:
    """What `/api/results/dir` answers of ``folder``, asked inside the tree
    it lies in (the fence, `web-api.md` § 2.1)."""
    from conftest import point_the_one_door_at
    from molbuilder.projects import find_projects_root
    point_the_one_door_at(find_projects_root(folder).parent, monkeypatch.setenv)
    r = _web().get("/api/results/dir", query_string={"path": str(folder)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _picker(snippet: str):
    """``snippet`` against the picker as `results.html` loads it -- the one
    "ago" first -- with the trajectory presenter beside it."""
    return run_node(
        [TIME_AGO, PICKER, INSPECTORS / "_partial_inspector_factory.js",
         INSPECTORS / "trajectory.js"],
        "const traj = window.molbuilder.inspectors.trajectoryInspector;\n"
        "const P = window.molbuilder.resultsFilePicker;\n" + snippet,
        globals_js=(
            "globalThis.document = { createElement: () => ({ setAttribute() {},"
            " appendChild() {} }), getElementById: () => null,"
            " querySelector: () => null };\n"
            "window.molbuilder.inspectors = window.molbuilder.inspectors"
            " || {};"),
        timeout=30)


@pytest.mark.parametrize("which", ["SIESTA", "PySCF"])
def test_the_menu_lists_what_can_be_read_and_is_claimed_newest_first(
        which, followed, stage_shaped, monkeypatch):
    """`filterToResultFiles` on a real run folder's listing: a file the
    server's registry can read (`parser`) AND a presenter claims, each with
    the path the server sent, whole -- the page joins no folder to a name
    -- newest first.  THE SILENT FAILURES: a file nothing can open offered;
    a menu counting files nobody can pick ("7 of 7" under two options,
    measured 2026-09-19); a path the page composed."""
    folder = (Path(followed.final["path"]).parent if which == "SIESTA"
              else stage_shaped.attempt("coarse"))
    answer = _listing(folder, monkeypatch)
    out = _picker(f"""
        const a = {json.dumps(answer)};
        const pick = (p, m) => (traj.match(p, m) ? traj : null);
        const every = P.filterToResultFiles(a.files, a);
        const listed = P.filterToResultFiles(a.files, a, pick);
        console.log(JSON.stringify({{
            every, listed,
            claimed: a.files.map(f => !!f.parser && traj.match(f.path, f)) }}));
    """)
    files = answer["files"]
    assert {e["path"] for e in out["every"]} == {
        f["path"] for f in files if f["parser"]}
    assert [e["path"] for e in out["listed"]] == [
        e["path"] for e in out["every"]
        if out["claimed"][[f["path"] for f in files].index(e["path"])]]
    assert out["listed"], files
    stamps = [e["mtime"] for e in out["every"]]
    assert stamps == sorted(stamps, reverse=True)
    for e in out["every"]:
        f = next(f for f in files if f["path"] == e["path"])
        assert (e["name"], e["mtime"], e["role"], e["label"]) == (
            f["name"], f["mtime"], f["role"], f["label"])
        assert e["engine"] == answer.get("engine")


def test_a_run_is_one_entry_whatever_its_label_looks_like(stage_shaped,
                                                         monkeypatch):
    """`results.md` § 2.3: a PySCF relaxation writes its unified
    `.molwatch.log` beside the input it echoed back, the geometry it
    reached and geomeTRIC's stream; the master absorbs its own run's parts,
    read back with the run's label as the server reads it -- never cut out
    of the name, which a label shaped like `au_2_bdt` would defeat.  The
    REAL trajectory presenter's `absorbs`, on the real run's listing.

    THE SILENT FAILURE: one relaxation listed as four entries (measured
    2026-09-19)."""
    from molbuilder.runfiles import (PYSCF_OPTIMIZED, ROLE_GEOM_TRAJ,
                                     ROLE_INITIAL, ROLE_MOLWATCH)
    answer = _listing(stage_shaped.attempt("coarse"), monkeypatch)
    out = _picker(f"""
        const a = {json.dumps(answer)};
        const pick = (p, m) => (traj.match(p, m) ? traj
            : (p.endsWith(".xyz") ? {{ name: "structure",
                                       displayName: "Structure" }} : null));
        const listed = P.filterToResultFiles(a.files, a, pick);
        const kept = P.absorbSatellites(listed, pick);
        console.log(JSON.stringify({{
            listed: listed.map(e => e.name), kept: kept.map(e => e.name) }}));
    """)
    roles = {f["name"]: f["role"] for f in answer["files"]}
    masters = [n for n in out["listed"] if roles[n] == ROLE_MOLWATCH]
    parts = [n for n in out["listed"]
             if roles[n] in (ROLE_INITIAL, PYSCF_OPTIMIZED, ROLE_GEOM_TRAJ)]
    assert len(masters) == 1 and parts, out
    assert all(answer_label == "au_2_bdt" for answer_label in
               {f["label"] for f in answer["files"]
                if f["name"] in masters + parts}), answer["files"]
    assert out["kept"] == [n for n in out["listed"] if n not in parts], out


def test_the_heading_is_built_from_the_folders_answer(followed, monkeypatch):
    """The picker's heading for a run's files comes from what the folder's
    answer says of it -- the engine's words and the calculation's noun --
    through the presenter's own `resultCategory`.  THE SILENT FAILURE: the
    entry handed to the presenter and dropped before the heading, so a
    SIESTA directory's runs sat under the engine-less heading while the same
    response named the engine (measured 2026-09-19)."""
    answer = _listing(Path(followed.final["path"]).parent, monkeypatch)
    assert answer.get("engine_words") and answer.get("noun"), answer
    out = _picker(f"""
        const a = {json.dumps(answer)};
        const pick = (p, m) => (traj.match(p, m) ? traj : null);
        const listed = P.filterToResultFiles(a.files, a, pick);
        console.log(JSON.stringify(P.groupResultFiles(listed, pick)
                                   .map(g => g.label)));
    """)
    assert out == [answer["engine_words"] + " " + answer["noun"]]

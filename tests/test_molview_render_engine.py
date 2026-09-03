"""The renderEngine — every test derived from ``docs/web/molview.md``, never from
the source it checks (§ 13).

Step F of the rebuild (``docs/web/molview.md``). The rows of § 13.3
guarded here:

    § 10.5  the cost matches what changed
    § 10.4  playing does not re-process
    § 10.1  a drawing setting derives nothing
    § 10.6  shapes move with the frames
    § 10.7  a selection never restyles the model
    § 10.8  same atoms, every frame
    § 10.9  nothing is lost during a rebuild
    § 10.10 the offered frames are drawable
    § 10.10 only the master copy's count is offered
    § 9.7   the renderEngine answers nothing

Level 2 of § 13.2: boundary behaviour, with a stand-in that obeys *this
document* — here a recording stand-in for the sealed layer, which is the level
below. § 13.1's warning is taken literally: it never claims a movie is loaded
while reporting that none exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
ENGINE = MODULE_DIR / "render-engine.js"

# A stand-in for the sealed layer. It offers exactly the doors of § 9.8 and the
# two self-check questions of § 9.9, and NOTHING ELSE — so a renderEngine that
# tried to read coordinates or the shown frame back out would fail here rather
# than quietly work.
EMBED = """
globalThis.__embedCalls = [];
globalThis.makeEmbed = function () {
    let movieFrames = 0;
    // A stand-in may be told to come back SHORT of what it was given, which is
    // the only way to exercise § 10.10's heal.
    let shortBy = 0;
    const rec = (name, args) => globalThis.__embedCalls.push({ name, args });
    return {
        __setShortBy(n) { shortBy = n; },
        loadFrames(elements, frames) {
            rec("loadFrames", [elements.length, frames.length]);
            movieFrames = Math.max(0, frames.length - shortBy);
            return true;
        },
        appendFrames(frames) {
            rec("appendFrames", [frames.length]);
            if (!movieFrames) return false;
            movieFrames += frames.length;
            return true;
        },
        showFrame(i)      { rec("showFrame", [i]); },
        setStyle(v)       { rec("setStyle", [v]); },
        setProjection(p)  { rec("setProjection", [p]); },
        setCell(c)        { rec("setCell", [c]); },
        // The COUNT first, which is what most tests ask; the arrows themselves
        // beside it, for the one that has to compare their geometry between
        // frames (§ 10.3 — worked out once, not per frame).
        setArrows(a)      { rec("setArrows", [(a || []).length, a || []]); },
        setOverlays(o)    { rec("setOverlays", [o]); },
        setBusy(m)        { rec("setBusy", [m]); },
        beginBatch()      { rec("beginBatch", []); },
        endBatch()        { rec("endBatch", []); },
        fitCamera()       { rec("fitCamera", []); },
        capture()         { return Promise.resolve(null); },
        onPick()          {},
        // The only two questions (§ 9.9), and they are COHERENT with each other.
        hasMovie()        { return movieFrames > 0; },
        drawnFrameCount() { return movieFrames; },
        dispose()         { rec("dispose", []); movieFrames = 0; },
    };
};
"""

PRELUDE = f"""
const E = await import({json.dumps(ENGINE.resolve().as_uri())});

// The model's end of the data source: the engine is HANDED the master copy.
function makeSource(atoms, frameCount) {{
    const s = {{
        structure: {{
            elements: Array.from({{length: atoms}}, () => "C"),
            annotations: Array.from({{length: atoms}}, () => ({{ labels: [] }})),
            cell: null,
        }},
        frames: Array.from({{length: frameCount}},
            (_, f) => Array.from({{length: atoms}}, (_, a) => [a + f * 100, 0, 0])),
        forces: null,
        frame: 0,
        selection: [],
        switches: {{ isolate: false, showIndex: false, showForces: false,
                    showCell: false, showAxis: false, forceScale: 1 }},
    }};
    s.door = {{
        structure: () => s.structure,
        frames:    () => s.frames,
        forces:    () => s.forces,
        frame:     () => s.frame,
        selection: () => s.selection,
        switches:  () => s.switches,
    }};
    return s;
}}

function wired(atoms, frameCount) {{
    globalThis.__embedCalls = [];
    const embed = globalThis.makeEmbed();
    const src = makeSource(atoms, frameCount);
    const engine = E.createRenderEngine(embed);
    engine.setDataSource(src.door);
    return {{ engine, embed, src }};
}}

function calls(name) {{
    return globalThis.__embedCalls.filter(c => c.name === name);
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=EMBED)


# ---------------------------------------------------------------------------
# § 10.5 — the cost matches what changed
# ---------------------------------------------------------------------------

def test_each_change_costs_what_the_table_says():
    """§ 10.5's table is exhaustive of the changes that reach this pipeline. Each
    row is driven here and the cost asserted.

    The two worth reading twice, because both are easy to get wrong:
      - a CELL EDIT is an overlay refresh, not a rebuild — the atoms did not move
      - a streamed APPEND extends the movie; it is not a reload
    """
    out = _run(
        """
        const { engine, src } = wired(4, 3);
        await engine.dataChanged();                     // a new structure
        engine.__resetCosts();

        src.frame = 1;  engine.showFrame();             // scrubbing
        const swap = engine.__costs().slice(-1)[0];

        src.switches.showIndex = true; engine.switchesChanged();
        const overlaySwitch = engine.__costs().slice(-1)[0];

        src.structure.cell = { lattice: [[4,0,0],[0,4,0],[0,0,4]] };
        engine.cellChanged();                           // a CELL EDIT
        const cellEdit = engine.__costs().slice(-1)[0];

        src.forces = src.frames.map(() => [[1,0,0],[1,0,0],[1,0,0],[1,0,0]]);
        engine.forcesChanged();
        const forces = engine.__costs().slice(-1)[0];

        src.frames.push([[9,0,0],[10,0,0],[11,0,0],[12,0,0]]);
        engine.appendFrames(3);                          // a streamed APPEND
        const append = engine.__costs().slice(-1)[0];

        src.selection = [1, 2];
        src.switches.isolate = true; await engine.switchesChanged();
        const isolate = engine.__costs().slice(-1)[0];

        console.log(JSON.stringify({
            swap, overlaySwitch, cellEdit, forces, append, isolate,
        }));
        """
    )
    assert out["swap"] == "swap", "scrubbing must be a frame swap"
    assert out["overlaySwitch"] == "overlay", (
        "an overlay switch with the same atoms drawn is an overlay refresh"
    )
    assert out["cellEdit"] == "overlay", (
        "a cell edit is an overlay refresh, NOT a rebuild — the atoms did not move"
    )
    assert out["forces"] == "overlay"
    assert out["append"] == "append", (
        "a streamed append extends the movie; it is not a reload"
    )
    assert out["isolate"] == "rebuild", (
        "toggling isolate changes the set of drawn atoms, so it rebuilds"
    )


def test_the_axes_and_the_force_arrows_are_drawn_together():
    """§ 10.3 lists the overlays as independent things a switch adds, and § 10.11
    says the scene is "a fixed stack of independent layers … no switch can
    corrupt another".

    The axis triad rides the ordinary arrow door carrying its own colours, and
    the drawing has ONE such door — so the two must be composed into a single
    set, not written one after the other. Written in sequence, whichever went
    last erased the other: with both switches on the force arrows disappeared,
    and a frame swap (which re-places the overlays but not the scene) erased the
    axes instead.

    Counting is enough here and is the honest assertion: the question is whether
    both sets survived the same write, not what an arrow looks like.
    """
    out = _run(
        """
        const { engine, src } = wired(3, 2);
        src.forces = [
            [[1,0,0],[0,1,0],[0,0,1]],
            [[1,0,0],[0,1,0],[0,0,1]],
        ];
        await engine.dataChanged();

        // Forces alone: one arrow per atom.
        src.switches.showForces = true;   await engine.switchesChanged();
        const forcesOnly = calls("setArrows").pop().args[0];

        // Both: the three axes ride along with them.
        src.switches.showAxis = true;     await engine.switchesChanged();
        const both = calls("setArrows").pop().args[0];

        // A swap re-places the overlays — and must not drop the axes doing it.
        src.frame = 1;                    engine.showFrame();
        const afterSwap = calls("setArrows").pop().args[0];

        // Axes alone, with the forces switched back off.
        src.switches.showForces = false;  await engine.switchesChanged();
        const axesOnly = calls("setArrows").pop().args[0];

        console.log(JSON.stringify({ forcesOnly, both, afterSwap, axesOnly }));
        """
    )
    assert out["forcesOnly"] == 3, "one force arrow per atom"
    assert out["axesOnly"] == 3, "three axes"
    assert out["both"] == 6, (
        f"with both switches on the drawing got {out['both']} arrows, not both "
        f"sets — one write erased the other"
    )
    assert out["afterSwap"] == 6, (
        f"a frame swap dropped one of the two sets: {out['afterSwap']} arrows"
    )


def test_the_cost_never_consults_the_atom_count():
    """§ 10.5: "THE COST IS CHOSEN BY WHAT CHANGED, NEVER BY HOW BIG THE SYSTEM
    IS. There is no atom-count threshold and no magic number anywhere in this
    decision", and "a change that adds it has changed the design."
    """
    out = _run(
        """
        async function costsFor(atoms) {
            const { engine, src } = wired(atoms, 3);
            await engine.dataChanged();
            engine.__resetCosts();
            src.frame = 1;                    engine.showFrame();
            src.switches.showIndex = true;    engine.switchesChanged();
            src.frames.push(Array.from({length: atoms}, (_, a) => [a, 9, 0]));
            engine.appendFrames(3);
            src.selection = [0];
            src.switches.isolate = true;      await engine.switchesChanged();
            return engine.__costs();
        }
        console.log(JSON.stringify({
            tiny: await costsFor(2),
            huge: await costsFor(5000),
        }));
        """
    )
    assert out["tiny"] == out["huge"], (
        "the same sequence of changes cost differently at a different atom "
        f"count: {out['tiny']} vs {out['huge']}"
    )
    assert out["tiny"] == ["swap", "overlay", "append", "rebuild"]


def test_only_a_rebuild_raises_the_cover():
    """§ 10.5's last column: the cover is shown for a rebuild and for nothing
    else. § 10.9: "it is the only one that raises the cover."
    """
    out = _run(
        """
        const { engine, src } = wired(4, 3);
        await engine.dataChanged();
        const afterRebuild = calls("setBusy").map(c => c.args[0]);

        globalThis.__embedCalls = [];
        src.frame = 1;                     engine.showFrame();
        src.switches.showIndex = true;     engine.switchesChanged();
        engine.cellChanged();
        engine.forcesChanged();
        src.frames.push([[1,0,0],[2,0,0],[3,0,0],[4,0,0]]);
        engine.appendFrames(3);
        console.log(JSON.stringify({
            afterRebuild, cheapCovers: calls("setBusy").length,
        }));
        """
    )
    assert out["afterRebuild"] == ["Updating view…", False], (
        f"a rebuild must raise and lower the cover: {out['afterRebuild']}"
    )
    assert out["cheapCovers"] == 0, (
        "a swap, an overlay refresh or an append must not raise the cover"
    )


# ---------------------------------------------------------------------------
# § 10.4 / § 10.1 — what does NOT re-derive
# ---------------------------------------------------------------------------

def test_playing_reloads_nothing():
    """§ 13.3: "stepping or playing issues no per-frame derivation; the frames
    were finished at load."

    § 10.4: load once; playing is a frame swap, not a redraw.
    """
    out = _run(
        """
        const { engine, src } = wired(4, 20);
        await engine.dataChanged();
        globalThis.__embedCalls = [];
        for (let f = 0; f < 20; f++) { src.frame = f; engine.showFrame(); }
        console.log(JSON.stringify({
            loads: calls("loadFrames").length,
            appends: calls("appendFrames").length,
            swaps: calls("showFrame").length,
        }));
        """
    )
    assert out["loads"] == 0 and out["appends"] == 0, (
        "playing reloaded the movie — the frames were finished at load"
    )
    assert out["swaps"] == 20


def test_an_append_leaves_the_displayed_frame_where_it_was():
    """§ 10.8 rule 5: "A user watching frame 12 keeps watching frame 12 while the
    run grows past it."
    """
    out = _run(
        """
        const { engine, src } = wired(4, 20);
        await engine.dataChanged();
        src.frame = 12; engine.showFrame();
        globalThis.__embedCalls = [];
        for (let k = 0; k < 5; k++) {
            src.frames.push([[k,0,0],[k,1,0],[k,2,0],[k,3,0]]);
            engine.appendFrames(src.frames.length - 1);
        }
        console.log(JSON.stringify({
            moved: calls("showFrame").map(c => c.args[0]),
            frame: src.frame,
        }));
        """
    )
    assert out["moved"] == [], (
        f"an append moved the displayed frame: {out['moved']}"
    )
    assert out["frame"] == 12


# ---------------------------------------------------------------------------
# § 10.9 — nothing is lost during a rebuild
# ---------------------------------------------------------------------------

def test_frames_arriving_mid_rebuild_all_appear_afterwards():
    """§ 13.3: "frames that arrive mid-rebuild all appear afterwards."

    § 10.9: appended frames ACCUMULATE, "because each poll tick's frames are a
    distinct piece of the run, and losing one would leave a hole in the middle of
    it."
    """
    out = _run(
        """
        const { engine, src } = wired(4, 2);
        await engine.dataChanged();
        globalThis.__embedCalls = [];

        const rebuilding = engine.dataChanged();       // not awaited: the window
        for (let k = 0; k < 3; k++) {
            src.frames.push([[k,0,0],[k,1,0],[k,2,0],[k,3,0]]);
            engine.appendFrames(src.frames.length - 1);
        }
        await rebuilding;
        console.log(JSON.stringify({
            appends: calls("appendFrames").map(c => c.args[0]),
        }));
        """
    )
    assert len(out["appends"]) == 3, (
        f"poll ticks arriving during a rebuild were dropped, leaving a hole in "
        f"the run: {out['appends']}"
    )


def test_a_seek_and_new_forces_keep_only_the_last():
    """§ 13.3: "a seek and a force update keep only the last."

    § 10.9: "only the frame you end on matters", and only the last forces are the
    current answer.
    """
    out = _run(
        """
        const { engine, src } = wired(4, 10);
        await engine.dataChanged();
        globalThis.__embedCalls = [];

        const rebuilding = engine.dataChanged();
        for (const f of [2, 5, 9]) { src.frame = f; engine.showFrame(); }
        for (let k = 0; k < 3; k++) {
            src.forces = src.frames.map(() => [[k,0,0],[k,0,0],[k,0,0],[k,0,0]]);
            engine.forcesChanged();
        }
        await rebuilding;

        // The rebuild itself shows the current frame; the replayed seek is the
        // one that follows it.
        const shown = calls("showFrame").map(c => c.args[0]);
        console.log(JSON.stringify({ shown, last: shown[shown.length - 1] }));
        """
    )
    assert out["last"] == 9, (
        f"only the frame the user ended on should be replayed: {out['shown']}"
    )
    assert out["shown"].count(2) == 0 and out["shown"].count(5) == 0, (
        f"a superseded seek was replayed: {out['shown']}"
    )


def test_a_switch_change_during_a_rebuild_is_not_held():
    """§ 10.9: "nothing is held. The rebuild reads the switches WHEN IT RUNS, not
    when it was scheduled — the latest value is the one it should use, so there
    is nothing to replay."
    """
    out = _run(
        """
        const { engine, src } = wired(4, 3);
        await engine.dataChanged();
        globalThis.__embedCalls = [];

        const rebuilding = engine.dataChanged();
        src.switches.showIndex = true;
        engine.switchesChanged();               // must be dropped, not queued
        await rebuilding;

        const overlays = calls("setOverlays");
        console.log(JSON.stringify({
            overlayCalls: overlays.length,
            labelsDrawn: overlays.length ? overlays[0].args[0].labels.length : 0,
        }));
        """
    )
    assert out["overlayCalls"] == 1, (
        f"a switch change during a rebuild was replayed as extra work: "
        f"{out['overlayCalls']} overlay applications"
    )
    assert out["labelsDrawn"] == 4, (
        "the rebuild must read the switch value as it stood WHEN IT RAN, so the "
        "labels turned on mid-rebuild are drawn"
    )


def test_a_full_load_supersedes_a_rebuild_and_drops_what_was_held():
    """§ 13.3: "a full load cancels what was queued and supersedes the rebuild."

    § 10.9: anything held "refers to atoms or frames that no longer exist. A full
    load is never itself refused: it is the more authoritative statement about
    what the structure is."
    """
    out = _run(
        """
        const { engine, src } = wired(4, 2);
        await engine.dataChanged();
        globalThis.__embedCalls = [];

        const first = engine.dataChanged();
        src.frames.push([[1,0,0],[2,0,0],[3,0,0],[4,0,0]]);
        engine.appendFrames(2);                  // held...
        const second = engine.dataChanged();     // ...and dropped by this
        await Promise.all([first, second]);

        console.log(JSON.stringify({
            appends: calls("appendFrames").length,
            loads: calls("loadFrames").length,
        }));
        """
    )
    assert out["appends"] == 0, (
        "a held append was replayed after a full load replaced the atoms it "
        "referred to"
    )
    assert out["loads"] >= 1


# ---------------------------------------------------------------------------
# § 10.10 — keeping the offered frames drawable
# ---------------------------------------------------------------------------

def test_appending_with_no_movie_rebuilds_instead_of_extending_nothing():
    """§ 13.3: "appending to a structure with no movie rebuilds instead of
    extending nothing."

    § 10.10: a run caught at its very first geometry has no movie, and appending
    to one that does not exist quietly does nothing at all. This is the case the
    "is there a movie?" question exists to catch.
    """
    out = _run(
        """
        const { engine, src, embed } = wired(4, 1);
        // Nothing drawn yet at all — the run is at its first geometry.
        src.frames.push([[1,0,0],[2,0,0],[3,0,0],[4,0,0]]);
        engine.appendFrames(1);
        console.log(JSON.stringify({
            costs: engine.__costs(),
            hasMovie: embed.hasMovie(),
            drawn: embed.drawnFrameCount(),
        }));
        """
    )
    assert out["costs"] == ["rebuild"], (
        f"appending with no movie must become a rebuild: {out['costs']}"
    )
    assert out["hasMovie"] is True and out["drawn"] == 2, (
        "after the rebuild the drawing must hold every frame the structure has"
    )


def test_a_drawing_found_short_of_the_master_copy_is_rebuilt():
    """§ 10.10: "A drawing found short of the master copy is rebuilt from it."

    The check is only worth making against something that could disagree —
    asking the copy you just grew how big it is confirms nothing.
    """
    out = _run(
        """
        const { engine, src, embed } = wired(4, 5);
        embed.__setShortBy(2);            // the drawing silently comes back short
        await engine.dataChanged();
        const afterShort = { drawn: embed.drawnFrameCount(), costs: engine.__costs() };

        embed.__setShortBy(0);            // it heals on the retry
        engine.__resetCosts();
        await engine.dataChanged();
        console.log(JSON.stringify({
            afterShort,
            healed: embed.drawnFrameCount(),
            master: src.frames.length,
        }));
        """
    )
    assert out["afterShort"]["costs"].count("rebuild") > 1, (
        "a short drawing must trigger another rebuild rather than being left "
        f"unable to show frames the bar offers: {out['afterShort']}"
    )
    assert out["healed"] == out["master"] == 5


# ---------------------------------------------------------------------------
# § 9.7 — the renderEngine answers nothing
# ---------------------------------------------------------------------------

def test_the_world_triad_and_the_cell_triad_are_on_screen_together():
    """The two triads answer different questions and a user needs both at once:
    x/y/z is the frame every coordinate in the file is written in, a/b/c is the
    way the box repeats. On a skewed or rotated cell those are different
    directions, and the angle between them is the thing worth looking at.

    `sceneFor` used to return ONE of them — a/b/c if the structure had a cell,
    x/y/z if it did not — so the world frame vanished the moment a cell appeared
    and there was nothing to compare the cell against.

    Each rides its own switch, so each switch means one thing: "Show axes" is the
    world triad; the cell's own directions belong to the cell and come and go
    with the box they describe.
    """
    out = _run(
        """
        const { engine, src } = wired(4, 1);
        src.structure.periodicity = { cell: [[8,0,0],[0,8,0],[0,0,8]],
                                      cell_origin: [1,1,1] };
        await engine.dataChanged();

        // Whatever reaches the drawing, by label — the arrows door carries the
        // force arrows and both triads together (§ 10.6).
        const shown = () => {
            const c = calls("setArrows");
            const last = c.length ? c[c.length - 1].args[1] : [];
            return last.map(a => a.label).filter(Boolean);
        };
        const seen = {};

        globalThis.__embedCalls = [];
        engine.switchesChanged();                     // both switches off
        seen.neither = shown();

        globalThis.__embedCalls = [];
        src.switches.showAxis = true;
        engine.switchesChanged();
        seen.axesOnly = shown();

        globalThis.__embedCalls = [];
        src.switches.showCell = true;
        engine.switchesChanged();
        seen.both = shown();

        globalThis.__embedCalls = [];
        src.switches.showAxis = false;
        engine.switchesChanged();
        seen.cellOnly = shown();

        // And with no cell at all there is no cell triad to show, however the
        // cell switch is set — whether there IS one is the structure's business.
        globalThis.__embedCalls = [];
        src.structure.periodicity = null;
        engine.cellChanged();
        seen.noCell = shown();

        console.log(JSON.stringify(seen));
        """
    )
    assert out["neither"] == [], "a triad was drawn with both switches off"
    assert out["axesOnly"] == ["x", "y", "z"], (
        f"'Show axes' must draw the world triad: {out['axesOnly']}"
    )
    assert out["both"] == ["x", "y", "z", "a", "b", "c"], (
        "the two triads must be on screen TOGETHER — one did not replace the "
        f"other: {out['both']}"
    )
    assert out["cellOnly"] == ["a", "b", "c"], (
        f"the cell's triad rides the cell switch, on its own: {out['cellOnly']}"
    )
    assert out["noCell"] == [], (
        f"a cell triad was drawn for a structure with no cell: {out['noCell']}"
    )


def test_the_camera_is_fitted_on_load_and_on_reset_and_at_no_other_moment():
    """§ 9.6: "On load, and on Reset, the camera is fitted to the structure."
    Those two moments, and no other.

    Isolate is a REBUILD (§ 10.5), and fitting on every rebuild threw the user's
    angle away the moment they pressed the isolate switch. Nothing above the
    drawing keeps the camera (§ 9.6), so a fit that should not have happened
    cannot be undone afterwards — it has to be withheld.
    """
    out = _run(
        """
        const { engine, src } = wired(4, 3);

        const fits = () => calls("fitCamera").length;
        globalThis.__embedCalls = [];

        await engine.dataChanged();                 // a load
        const onLoad = fits();

        globalThis.__embedCalls = [];
        src.selection = [0, 1];
        src.switches.isolate = true;
        await engine.switchesChanged();             // isolate — a rebuild
        const onIsolate = fits();

        globalThis.__embedCalls = [];
        src.switches.isolate = false;
        await engine.switchesChanged();             // and back off — a rebuild
        const offIsolate = fits();

        globalThis.__embedCalls = [];
        src.switches.showIndex = true;
        engine.switchesChanged();                   // an overlay refresh
        engine.showFrame();                         // a frame swap
        const onOverlayAndSwap = fits();

        globalThis.__embedCalls = [];
        engine.resetView();                         // the other sanctioned moment
        const onReset = fits();

        console.log(JSON.stringify({
            onLoad, onIsolate, offIsolate, onOverlayAndSwap, onReset }));
        """
    )
    assert out["onLoad"] == 1, "a load must fit the camera on the structure"
    assert out["onReset"] == 1, "Reset must re-fit the camera"
    assert out["onIsolate"] == 0 and out["offIsolate"] == 0, (
        "isolating re-fitted the camera, so the angle the user set was thrown "
        "away by a switch that only changes which atoms are drawn"
    )
    assert out["onOverlayAndSwap"] == 0, (
        "an overlay refresh or a frame swap moved the camera"
    )


def test_the_scene_is_worked_out_once_and_follows_the_cell_when_it_changes():
    """§ 6.5 / § 10.3: the cell box and the axes "are the same for every frame
    unless the cell itself changes, so they are worked out once as scene-level
    data and are not recomputed per frame."

    **What this test can and cannot see.** Deriving the scene once versus four
    hundred times produces the *same* answer, so the saving is a cost and not an
    output — there is nothing at this boundary that tells them apart, and a test
    claiming otherwise would be asserting its own fixture. What IS observable, and
    what holding the scene between frames could break, is the other half of the
    same sentence: *unless the cell itself changes*. A scene held too long is a
    box drawn at the previous structure's corner, which is § 10.3's own failure
    written from the other side.

    So: it stays put across a played trajectory, and it moves for each of the two
    things allowed to move it — a cell edit, and a new structure.
    """
    out = _run(
        """
        const axesOf = (c) => JSON.stringify((c.args[1] || []).slice(-3));
        const lastAxes = () => {
            const c = calls("setArrows");
            return c.length ? axesOf(c[c.length - 1]) : null;
        };

        const { engine, src } = wired(4, 400);
        // The CELL's triad, which is the one that moves when the cell does; it
        // rides the cell switch beside the box it describes.
        src.switches.showCell = true;
        src.structure.periodicity = { cell: [[8,0,0],[0,8,0],[0,0,8]],
                                      cell_origin: [1,1,1] };
        await engine.dataChanged();

        // Play the whole trajectory: the axes ride every swap (they share the
        // one arrow door with the frame's force arrows, § 10.6) and none of the
        // four hundred may differ from the first.
        globalThis.__embedCalls = [];
        for (let f = 0; f < 400; f++) { src.frame = f; engine.showFrame(); }
        const perSwap = calls("setArrows").length;
        const distinctWhilePlaying = new Set(
            calls("setArrows").map(axesOf)).size;
        const whilePlaying = lastAxes();

        // A CELL EDIT must move them.
        globalThis.__embedCalls = [];
        src.structure.periodicity = { cell: [[20,0,0],[0,20,0],[0,0,20]],
                                      cell_origin: [5,5,5] };
        engine.cellChanged();
        const afterCellEdit = lastAxes();

        // A NEW STRUCTURE must move them too.
        globalThis.__embedCalls = [];
        src.structure = { elements: ["C","C","C","C"],
                          periodicity: { cell: [[3,0,0],[0,3,0],[0,0,3]],
                                         cell_origin: [0,0,0] } };
        await engine.dataChanged();
        const afterNewStructure = lastAxes();

        console.log(JSON.stringify({
            perSwap, distinctWhilePlaying,
            whilePlaying, afterCellEdit, afterNewStructure }));
        """
    )
    assert out["perSwap"] == 400, (
        "the axes must still be re-sent on each swap — they share the one arrow "
        "door with the frame's force arrows (§ 10.6)"
    )
    assert out["distinctWhilePlaying"] == 1, (
        "the axis geometry changed while only the frame moved"
    )
    assert out["afterCellEdit"] and out["afterCellEdit"] != out["whilePlaying"], (
        "a cell edit left the axes at the old cell — the scene is being held "
        "past the one thing allowed to change it (§ 10.3)"
    )
    assert out["afterNewStructure"] and out["afterNewStructure"] != out["afterCellEdit"], (
        "a new structure left the axes at the previous structure's cell"
    )


def test_the_engine_offers_no_read_of_the_data_or_the_frame():
    """§ 13.3: "it offers no read of the data and no read of the displayed
    frame."

    § 9.7: every entry is an instruction; none is a question, "because the
    renderEngine is told what to draw and is never consulted about what the data
    is."
    """
    out = _run(
        """
        const { engine, src } = wired(4, 6);
        await engine.dataChanged();
        src.frame = 3; engine.showFrame();

        // Call every door and look at what comes back. The coordinates carry a
        // value that appears nowhere else, so a leak is visible.
        src.frames[3] = [[7777,0,0],[1,0,0],[2,0,0],[3,0,0]];
        const answers = {};
        for (const name of Object.keys(engine)) {
            if (name === "dispose" || name === "setDataSource") continue;
            let r;
            try { r = engine[name](); } catch (e) { r = "threw"; }
            if (r && typeof r.then === "function") r = "promise";
            answers[name] = JSON.stringify(r === undefined ? null : r);
        }
        const blob = JSON.stringify(answers);
        console.log(JSON.stringify({
            answers,
            leaksCoordinate: blob.indexOf("7777") >= 0,
            leaksFrame: /"(currentFrame|frame|frameCount)"/.test(blob),
        }));
        """
    )
    assert out["leaksCoordinate"] is False, (
        f"the renderEngine answered a question about the data: {out['answers']}"
    )
    assert out["leaksFrame"] is False, (
        "the renderEngine offered a read of the displayed frame"
    )


def test_the_engine_runs_no_change_notification_of_its_own():
    """§ 7 level 5's third never: "run a change notification of its own."

    Notifying is the model's job (§ 6.4), and a second notifier is how two
    subscribers come to see different states.
    """
    code = ENGINE.read_text()
    for token in ("subscribe", "addEventListener", "notify", "listeners"):
        assert token not in code, (
            f"the renderEngine has a notification mechanism ({token}) — telling "
            "anyone is the model's job"
        )


def test_the_engine_never_names_the_drawing_library():
    """§ 5.3: everything above the sealed layer reads end to end without learning
    which library draws the molecule. The engine is handed an embed and calls its
    doors; it does not know what is behind them.
    """
    assert "3Dmol" not in ENGINE.read_text()


def test_one_menu_two_owners():
    """§ 8.5: the View menu writes to two different stores, and § 9.6's question
    is what sorts them — does working out what a frame contains require reading
    this?

    § 13.3: "turning on atom numbers re-derives frames; changing the style does
    not — both from the same menu."

    The menu is one piece of UI. That does not make its contents one kind of
    thing, and the cost is where the difference shows.
    """
    out = _run(
        """
        const { engine, src } = wired(4, 3);
        await engine.dataChanged();
        engine.__resetCosts();

        // A SWITCH: atom-number labels change what is IN a frame.
        src.switches.showIndex = true;
        engine.switchesChanged();
        const afterSwitch = engine.__costs().slice();

        // A DRAWING SETTING smuggled in beside them: style, background and
        // projection change how the same frame is PAINTED, so the pipeline must
        // not see them at all (§ 10.5's table has no row for one).
        Object.assign(src.switches, {
            style: "sphere", background: "black", orthographic: true, radius: 9,
        });
        engine.switchesChanged();
        const afterSetting = engine.__costs().slice();

        console.log(JSON.stringify({ afterSwitch, afterSetting }));
        """
    )
    assert out["afterSwitch"] == ["overlay"], (
        f"turning on atom numbers must re-derive the overlays: {out['afterSwitch']}"
    )
    assert out["afterSetting"] == ["overlay", "overlay"], (
        "a drawing setting must not change the COST of anything — the second "
        f"call cost the same as the first: {out['afterSetting']}"
    )


def test_the_cover_reaches_the_screen_before_the_work_starts():
    """§ 10.9: the cover is the lock, so it has to be VISIBLE while the lock is on.

    THE DEFECT THIS CLOSES, measured on a real page before the fix: the cover was
    raised and lowered 8.8 ms apart with NO task boundary between them. A browser
    only redraws BETWEEN pieces of work, so raising the cover, drawing the
    molecule and lowering it as one piece meant the first chance to redraw came
    after it was already hidden. The sign went up and came down while the door
    was shut.

    WHAT THIS HAD TO ASSERT, and did not on the first attempt. The old test
    checked `setBusy` was CALLED with the right arguments -- true throughout the
    bug, because a call is not a pixel. My replacement checked that a task
    boundary fell somewhere inside the cover's window, which the
    minimum-on-screen wait satisfies ON ITS OWN; it passed with the bug
    deliberately restored. The condition that actually separates them is WHERE
    the boundary falls: between raising the cover and STARTING THE WORK.
    `loadFrames` is the work.
    """
    out = _run(
        """
        const { engine } = wired(4, 3);
        // A macrotask can only run BETWEEN pieces of work. Scheduled BEFORE the
        // rebuild starts, so it takes the first boundary that opens up.
        let atBoundary = null;
        setTimeout(() => { atBoundary = {
            covers: calls("setBusy").length,
            workStarted: calls("loadFrames").length,
        }; }, 0);
        await engine.dataChanged();
        console.log(JSON.stringify({
            atBoundary,
            sequence: calls("setBusy").map(c => c.args[0]),
        }));
        """
    )
    assert out["sequence"] == ["Updating view…", False], (
        f"the cover must still be raised then lowered: {out['sequence']}")
    assert out["atBoundary"] is not None, (
        "no task boundary ran during the rebuild at all -- the whole thing is "
        "one piece of work and the browser never gets to draw the cover")
    assert out["atBoundary"]["covers"] == 1, (
        "the boundary fell before the cover was even raised")
    assert out["atBoundary"]["workStarted"] == 0, (
        "the browser's first chance to draw came only AFTER the rebuild work had "
        "started, so the cover cannot have been painted -- raise it, yield to a "
        "real task boundary, THEN work")


def test_a_model_emptied_to_NOTHING_still_clears_the_drawing():
    """**The reported bug** *(user, 2026-09-02: "when i click 'start empty'
    the atom list is clear, but 3dmol is not updated with an empty window.
    it stays with the old model displayed")*.

    `doRebuild` opened with `if (!s || !processed.length) return;` — it told
    the drawing NOTHING when there was nothing to draw, so a model emptied to
    null left the previous molecule on screen beside an empty atom list.

    Deleting every atom took a different route and so looked fixed: there a
    structure still EXISTS with zero elements, `loadFrames` is reached, and
    the embed clears itself (molview.md § 6.7a). This is the same mistake one
    layer up — *nothing to draw* read as *nothing to do* — and it is the layer
    `clear()` goes through."""
    out = _run("""
        const w = wired(3, 2);
        await w.engine.dataChanged();          // a real molecule is drawn
        const drew = calls("loadFrames").length;
        globalThis.__embedCalls = [];
        // what `model.clear()` does: no structure at all
        w.src.structure = null;
        w.src.frames = [];
        await w.engine.dataChanged();
        const after = calls("loadFrames");
        console.log(JSON.stringify({
            drew: drew,
            clearedWith: after.length ? after[after.length - 1].args : null,
        }));
    """)
    assert out["drew"] >= 1, "the fixture never drew anything to begin with"
    assert out["clearedWith"] is not None, (
        "emptying the model told the drawing nothing -- the previous molecule "
        "is still on screen")
    elements, frames = out["clearedWith"][0], out["clearedWith"][1]
    assert not elements and not frames, (
        f"the drawing was not asked to clear: {out['clearedWith']!r}")


def test_an_empty_rebuild_HANDS_DOWN_the_triad_rather_than_assuming_it_is_there():
    """§ 6.7a: *"the axis triad, turned on for you"* -- and the arrows have to
    ARRIVE for that to be true.

    The embed's empty path redraws the arrows it is HOLDING, which reads as
    correct when a structure was on screen a moment ago.  It is false on the
    path that matters: a page reopened onto an empty canvas has handed no
    arrows down at all, so there was nothing to redraw and the window came
    back a blank rectangle with *Show axes* reading ON -- exactly the
    "working viewer or broken one?" question the triad exists to answer
    (browser walk, 2026-09-02).

    Two arrivals, one rule, so both are checked here: an emptied model, and a
    FIRST rebuild that was empty from the start.
    """
    out = _run("""
        // (a) a molecule on screen, then emptied
        const w = wired(3, 2);
        w.src.switches.showAxis = true;
        await w.engine.dataChanged();
        globalThis.__embedCalls = [];
        w.src.structure = null;
        w.src.frames = [];
        await w.engine.dataChanged();
        const emptied = calls("setArrows").map((c) => c.args[0]);

        // (b) a viewer whose FIRST rebuild is the empty one -- a reopened page
        const fresh = wired(3, 2);
        fresh.src.switches.showAxis = true;
        fresh.src.structure = null;
        fresh.src.frames = [];
        globalThis.__embedCalls = [];
        await fresh.engine.dataChanged();
        const firstEver = calls("setArrows").map((c) => c.args[0]);

        // (c) and the switch is still obeyed: axes off means no arrows
        const off = wired(3, 2);
        off.src.switches.showAxis = false;
        off.src.structure = null;
        off.src.frames = [];
        globalThis.__embedCalls = [];
        await off.engine.dataChanged();
        const switchedOff = calls("setArrows").map((c) => c.args[0]);

        console.log(JSON.stringify({ emptied, firstEver, switchedOff }));
    """)
    assert out["emptied"] and max(out["emptied"]) == 3, (
        "emptying the model did not hand the three axis arrows down: "
        f"{out['emptied']}")
    assert out["firstEver"] and max(out["firstEver"]) == 3, (
        "a viewer whose first rebuild was empty drew no triad -- the drawing "
        "was left to redraw arrows it had never been given: "
        f"{out['firstEver']}")
    assert not out["switchedOff"] or max(out["switchedOff"]) == 0, (
        "the empty path drew the triad with `Show axes` switched off -- the "
        f"switch is still the switch: {out['switchedOff']}")

"""VibrationView's handle — derived from the contract, not from the source.

Every test names the rule in ``docs/web/vibrationview.md`` it guards.

THE STAND-IN IS THE DRAWING LIBRARY, not the sealed layer.  Substituting the seal
would mean the module offering a way to replace its own internals, and a seam a
test can reach is a seam anything can reach (§ 4).  Standing in one level lower —
at ``$3Dmol``, the boundary the module actually has — lets ``index.js`` and the
sealed layer both run for real, and makes the assertions about the calls that
actually leave the module.

This file replaces a predecessor of the same name whose stand-in was a fake
``molbuilder.viewer`` global that the test itself installed.  Those five tests
passed for months while the module could not mount on any page, because the thing
they supplied was the thing that was missing.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"

#: A recording stand-in for the drawing library.  It obeys the one rule its level
#: has: it draws what it is told and answers nothing about the structure.
HARNESS = r"""
    const calls = [];
    let atoms = [];
    const viewer = {
        removeAllModels() { calls.push(["removeAllModels"]); atoms = []; },
        addModel(text) {
            calls.push(["addModel", text.split("\n")[0]]);
            atoms = text.split("\n").slice(2).filter(Boolean).map((l) => {
                const p = l.trim().split(/\s+/);
                return { elem: p[0], x: +p[1], y: +p[2], z: +p[3] };
            });
        },
        getModel() { return { selectedAtoms: () => atoms }; },
        setStyle(sel, spec) { calls.push(["setStyle", sel.index || "all"]); },
        render()   { calls.push(["render"]); },
        zoomTo()   { calls.push(["zoomTo"]); },
        addLabel(text, o) { calls.push(["addLabel", text, o.fontSize]); return { text }; },
        removeLabel(l) { calls.push(["removeLabel", l && l.text]); },
        setBackgroundColor(c, a) { calls.push(["background", c, a]); },
        resize()   { calls.push(["resize"]); },
        pngURI()   { return "data:image/png;base64,AAAA"; },
        clear()    { calls.push(["clear"]); },
    };
    global.$3Dmol = {
        createViewer: () => viewer,
        elementColors: { Jmol: {} },
    };
    // A DOM stand-in, at the same level as the $3Dmol one above: the browser is
    // the boundary the module has, so that is where a test stands in.  The caption
    // is a real element (§ 12.3), so a test that could not see elements would be
    // testing something the module does not do.
    const classes = new Set();
    const children = [];
    function makeEl(tag) {
        const el = { tagName: tag, className: "", textContent: "", hidden: false,
                     children: [], style: {},
                     appendChild(c) { this.children.push(c); return c; },
                     remove() { const i = children.indexOf(el); if (i >= 0) children.splice(i, 1); } };
        return el;
    }
    global.document = {
        createElement: makeEl,
        getElementById: () => null,
        head: { appendChild: () => {} },
    };
    const host = {
        classList: { add: (...c) => c.forEach((x) => classes.add(x)),
                     remove: (...c) => c.forEach((x) => classes.delete(x)) },
        querySelector: () => null,
        appendChild: (c) => { children.push(c); return c; },
        innerHTML: "", textContent: "",
    };
    const caption = () => children.find((c) => c.className === "vibview-caption");
    // The clock is the handle's, so it is the one thing a test must drive.
    var rafQ = [];
    let clock = 0;
    global.requestAnimationFrame = (fn) => { rafQ.push(fn); return rafQ.length; };
    global.cancelAnimationFrame  = () => { rafQ = []; };
    function pump(n, msPerStep) {
        for (let i = 0; i < n && rafQ.length; i++) {
            clock += (msPerStep === undefined ? 1000 : msPerStep);
            const fn = rafQ.shift();
            fn(clock);
        }
    }
    const { mount } = await import("/static/lib/vibrationview/index.js");
    const WATER = { elements: ["O", "H", "H"],
                    positions: [[0,0,0], [1,0,0], [0,1,0]] };
"""


def _run(snippet: str) -> object:
    return run_node([], HARNESS + snippet, static_root=STATIC)


# ---------------------------------------------------------------------------
# § 4 — self-contained
# ---------------------------------------------------------------------------

def test_the_module_reads_no_global_and_writes_none():
    """§ 4: "It does not read `window.molbuilder` and it does not write to it."

    Both halves failed in the module this replaces: it looked its drawing surface
    up in a global that nothing published, and published itself into another.
    """
    out = _run("""
        const before = JSON.stringify(Object.keys(global.molbuilder || {}));
        const vib = await mount(host, {});
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        console.log(JSON.stringify({
            before, after: JSON.stringify(Object.keys(global.molbuilder || {})),
            worked: vib.ok,
        }));
    """)
    assert out["worked"] is True
    assert out["after"] == out["before"]     # nothing published, nothing consulted


# ---------------------------------------------------------------------------
# § 8 — making and tearing down
# ---------------------------------------------------------------------------

def test_a_mount_that_cannot_build_a_surface_still_resolves_with_a_working_dispose():
    """§ 8: "a mount that cannot build a surface still resolves with ok === false
    AND a working dispose; nothing rejects and nothing returns null" — teardown
    must never have to ask whether setup worked."""
    out = _run("""
        delete global.$3Dmol;                 // the library is missing
        let rejected = false, threwOnDispose = false;
        let handle = null;
        try { handle = await mount(host, {}); } catch (_) { rejected = true; }
        try { handle.dispose(); } catch (_) { threwOnDispose = true; }
        console.log(JSON.stringify({
            rejected, threwOnDispose,
            isNull: handle === null, ok: handle && handle.ok,
            hasError: !!(handle && handle.error),
        }));
    """)
    assert out["rejected"] is False
    assert out["isNull"] is False
    assert out["ok"] is False
    assert out["hasError"] is True
    assert out["threwOnDispose"] is False


def test_the_handle_is_live_on_the_first_call():
    """§ 8: "every door works on the first call after await, with no readiness
    wait and nothing deferred".

    The predecessor deferred a mode requested before an onReady callback fired,
    which is a state a caller can get wrong; there is nothing to get wrong here.
    """
    out = _run("""
        const vib = await mount(host, {});
        const installed = vib.setStructure(WATER);
        const shown = vib.showMode({ index: 4,
                                     displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        console.log(JSON.stringify({
            installed, shown,
            drewStructure: calls.some((c) => c[0] === "addModel"),
        }));
    """)
    assert out["installed"] is True
    assert out["shown"] is True
    assert out["drewStructure"] is True


# ---------------------------------------------------------------------------
# § 5.1 — the door you call says what it costs
# ---------------------------------------------------------------------------

def test_setStructure_redraws_and_refits_and_showMode_does_neither():
    """§ 5.1 / § 10: "showMode on an installed structure issues no redraw and no
    refit; setStructure issues both" — which is why browsing mode to mode of one
    result never disturbs the camera."""
    out = _run("""
        const vib = await mount(host, {});
        calls.length = 0;
        vib.setStructure(WATER);
        const onStructure = calls.map((c) => c[0]);
        calls.length = 0;
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        const onMode = calls.map((c) => c[0]);
        calls.length = 0;
        vib.showMode({ index: 2, displacements: [[0,1,0],[0,0,1],[1,0,0]] });
        const onSecondMode = calls.map((c) => c[0]);
        console.log(JSON.stringify({ onStructure, onMode, onSecondMode }));
    """)
    assert "addModel" in out["onStructure"]
    assert "zoomTo" in out["onStructure"]
    assert "addModel" not in out["onMode"]        # no reload
    assert "zoomTo" not in out["onMode"]          # and no camera move
    assert "addModel" not in out["onSecondMode"]
    assert "zoomTo" not in out["onSecondMode"]


def test_a_new_structure_ends_the_mode_that_was_running():
    """§ 5.1: "after setStructure the clock is stopped and nothing is animating,
    whatever was running before".

    A mode belongs to the structure it was computed against.  Animating structure
    B with structure A's eigenvector would look entirely plausible on screen,
    which is exactly what makes it worth forbidding.
    """
    out = _run("""
        const vib = await mount(host, {});
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.play();
        const playingBefore = vib.isPlaying();
        vib.setStructure({ elements: ["H","H"], positions: [[0,0,0],[1,0,0]] });
        const playingAfter = vib.isPlaying();
        calls.length = 0;
        pump(4);
        console.log(JSON.stringify({
            playingBefore, playingAfter,
            keptAnimating: calls.some((c) => c[0] === "render"),
        }));
    """)
    assert out["playingBefore"] is True
    assert out["playingAfter"] is False
    assert out["keptAnimating"] is False


# ---------------------------------------------------------------------------
# § 6.3 — a mode that does not fit is refused
# ---------------------------------------------------------------------------

def test_a_mode_that_does_not_fit_the_structure_draws_nothing():
    """§ 6.3: "turned away with nothing drawn — never padded with zeros into a
    partial animation"."""
    out = _run("""
        const vib = await mount(host, {});
        vib.setStructure(WATER);
        calls.length = 0;
        const accepted = vib.showMode({ index: 1,
                                        displacements: [[1,0,0]], basis: [9] });
        const drewAnything = calls.length > 0;
        // ...and the viewer is still usable afterwards
        const goodOne = vib.showMode({ index: 2,
                                       displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        console.log(JSON.stringify({ accepted, drewAnything, goodOne }));
    """)
    assert out["accepted"] is False
    assert out["drewAnything"] is False
    assert out["goodOne"] is True


def test_showMode_before_a_structure_is_a_refusal_not_a_queue():
    """§ 9.2: "showMode before a structure does nothing, and says so; it is not an
    error and not a queue"."""
    out = _run("""
        const vib = await mount(host, {});
        const shown = vib.showMode({ index: 1, displacements: [[1,0,0]] });
        calls.length = 0;
        vib.setStructure(WATER);            // must NOT flush a deferred mode
        const flushed = calls.some((c) => c[0] === "setStyle" && c[1] !== "all");
        console.log(JSON.stringify({ shown, flushed, playing: vib.isPlaying() }));
    """)
    assert out["shown"] is False
    assert out["flushed"] is False
    assert out["playing"] is False


# ---------------------------------------------------------------------------
# § 9.2 / § 10.1 — the clock and the live knobs
# ---------------------------------------------------------------------------

def test_the_knobs_are_live_and_a_drag_never_stops_the_animation():
    """§ 9.2: "amplitude, fps and cycle-length changes take effect on the next
    frame, issue no call to the drawing surface, and never stop a running
    animation".

    This is the regression home for the slider bug the predecessor's design was
    built to avoid: a knob that re-registered anything would stop the loop.
    """
    out = _run("""
        const vib = await mount(host, { fps: 30 });
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.play();
        calls.length = 0;
        vib.setAmplitude(0.5);
        vib.setFps(60);
        vib.setCycleSec(2.0);
        const duringEdits = calls.length;
        const stillPlaying = vib.isPlaying();
        pump(3, 1000);
        console.log(JSON.stringify({
            duringEdits, stillPlaying,
            keptDrawing: calls.some((c) => c[0] === "render"),
        }));
    """)
    assert out["duringEdits"] == 0        # pure variable writes while playing
    assert out["stillPlaying"] is True
    assert out["keptDrawing"] is True


def test_a_rate_change_does_not_jump_the_motion():
    """§ 9.2 / § 10.1: "a rate change mid-flight does not jump the motion".

    A frame number means nothing without the count it is measured against, so
    when the count changes the number is re-expressed against the new one.
    Without that, nudging a smoothness slider would visibly move the molecule.
    """
    out = _run("""
        // Read where the atoms ACTUALLY ARE, off the drawing library's own
        // model, before and after the change — which is what a user sees.
        const drawn = () => atoms.map((a) => [a.x, a.y, a.z]);
        const worst = (a, b) => Math.max(...a.flatMap(
            (row, i) => row.map((v, j) => Math.abs(v - b[i][j]))));

        const vib = await mount(host, { fps: 30, amplitude: 1.0 });
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.play();
        pump(9, 1000);                        // walk part-way round the cycle
        vib.pause();

        const before = drawn();
        vib.setFps(60);                       // 30 -> 60 frames: an exact remap
        const afterFiner = drawn();

        vib.setFps(24);                       // 60 -> 24: a coarser grid
        const afterCoarser = drawn();

        console.log(JSON.stringify({
            finerShift:   worst(before, afterFiner),
            coarserShift: worst(afterFiner, afterCoarser),
            framesFine:   60,
            // half a frame of phase at the COARSER rate, times the amplitude:
            // the largest the nearest-frame remap can be off by
            bound: 1.0 * Math.PI / 24,
        }));
    """)
    # a finer grid contains every phase of a coarser one: exact, no movement
    assert out["finerShift"] < 1e-12
    # a coarser grid does not, so the phase lands on the nearest frame it has --
    # bounded by half a frame-step and never more
    assert out["coarserShift"] <= out["bound"]


def test_the_loop_advances_at_the_chosen_rate_not_every_repaint():
    """§ 10.1: `fps` means frames per second, not "every repaint" — so a display
    painting faster than the chosen rate does not speed the vibration up."""
    out = _run("""
        const vib = await mount(host, { fps: 10 });   // one frame per 100 ms
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.play();
        calls.length = 0;
        pump(12, 16);                     // twelve repaints at 60 Hz = ~192 ms
        const drawn = calls.filter((c) => c[0] === "render").length;
        console.log(JSON.stringify({ drawn }));
    """)
    # ~192 ms at 10 fps is one or two frames, not twelve
    assert 1 <= out["drawn"] <= 3


def test_pause_keeps_the_place_and_play_resumes_from_it():
    """§ 9.2: "pause then play resumes on the frame it stopped on"."""
    out = _run("""
        const vib = await mount(host, { fps: 30, amplitude: 1.0 });
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.play();
        pump(5, 1000);
        vib.pause();
        const paused = vib.isPlaying();
        calls.length = 0;
        pump(5, 1000);                                  // a paused loop draws nothing
        const drewWhilePaused = calls.some((c) => c[0] === "render");
        vib.play();
        pump(1, 1000);
        console.log(JSON.stringify({
            paused, drewWhilePaused,
            drewAfterResume: calls.some((c) => c[0] === "render"),
        }));
    """)
    assert out["paused"] is False
    assert out["drewWhilePaused"] is False
    assert out["drewAfterResume"] is True


# ---------------------------------------------------------------------------
# § 12.3 — the caption
# ---------------------------------------------------------------------------

def test_the_caption_is_drawn_exactly_as_given():
    """§ 12.3: "a mode's label is drawn exactly as given: no rounding, no unit
    added, no sign reinterpreted".

    The imaginary-mode case is the one that matters: the backend reports a saddle
    point as a negative frequency, and a viewer that formatted numbers itself
    would sooner or later present it as an ordinary mode.
    """
    out = _run("""
        const vib = await mount(host, {});
        vib.setStructure(WATER);
        vib.showMode({ index: 3, displacements: [[1,0,0],[0,1,0],[0,0,1]],
                       label: "Mode 3 · −212.4 cm⁻¹ (imag)" });
        const el = caption();
        console.log(JSON.stringify({
            text: el && el.textContent, shown: el && !el.hidden,
            // an OVERLAY, not a mark in the scene: the drawing library is not
            // asked to draw text, because measured against a real browser it
            // draws none (see _seal.js, carried knowledge 3 of 3)
            askedTheLibrary: calls.some((c) => c[0] === "addLabel"),
        }));
    """)
    assert out["text"] == "Mode 3 · −212.4 cm⁻¹ (imag)"
    assert out["shown"] is True
    assert out["askedTheLibrary"] is False


def test_the_caption_switch_shows_and_hides_it():
    """§ 12.3: the switch is the tab's, the drawing is the module's — and off
    means off (§ 12: "hidden, it appears in none")."""
    out = _run("""
        const vib = await mount(host, { showLabel: true });
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]],
                       label: "Mode 1 · 1584.2 cm-1" });
        vib.setLabelVisible(false);
        const off = caption().hidden;
        vib.setLabelVisible(true);
        const on = caption().hidden;
        console.log(JSON.stringify({ off, on, text: caption().textContent }));
    """)
    assert out["off"] is True                        # hidden is the switch
    assert out["on"] is False
    assert out["text"] == "Mode 1 · 1584.2 cm-1"


def test_a_new_structure_takes_the_caption_with_it():
    """§ 5.1: a new structure ends the mode — and the caption NAMES a mode.

    Left up, it is a confident label on nothing: "Mode 7 · 1584.2 cm⁻¹" written
    across a molecule that has no mode at all, which is worse than no label.
    """
    out = _run("""
        const vib = await mount(host, {});
        vib.setStructure(WATER);
        vib.showMode({ index: 7, displacements: [[1,0,0],[0,1,0],[0,0,1]],
                       label: "Mode 7 · 1584.2 cm⁻¹" });
        const before = caption().textContent;
        vib.setStructure({ elements: ["H","H"], positions: [[0,0,0],[1,0,0]] });
        const el = caption();
        console.log(JSON.stringify({
            before, after: el.textContent, hidden: el.hidden }));
    """)
    assert out["before"] == "Mode 7 · 1584.2 cm⁻¹"
    assert out["after"] == ""
    assert out["hidden"] is True


def test_the_chosen_rate_is_honoured_on_a_display_that_repaints_faster():
    """§ 10.1: `fps` is frames per second.

    THE REGRESSION HOME for the commonest configuration there is.  A 60 Hz display
    repaints every 16.67 ms and the default 30 fps asks for 33.33 ms — exactly two
    repaints.  Comparing strictly makes that a coin flip decided by timestamp
    jitter: land a hair under and the loop waits a THIRD repaint, so the animation
    silently runs at 20 fps and alternates as the jitter moves.

    Twelve repaints of a 60 Hz display is 200 ms, which at 30 fps is six frames.
    """
    out = _run("""
        const vib = await mount(host, { fps: 30 });
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.play();
        calls.length = 0;
        pump(12, 16.67);                       // 60 Hz, twelve repaints
        const drawn = calls.filter((c) => c[0] === "render").length;
        vib.dispose();      // its loop shares this recorder; retire it first
        rafQ = [];
        // ...and asking for the display's own rate draws on every repaint
        const vib2 = await mount(host, { fps: 60 });
        vib2.setStructure(WATER);
        vib2.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib2.play();
        calls.length = 0;
        pump(12, 16.67);
        const drawn60 = calls.filter((c) => c[0] === "render").length;
        console.log(JSON.stringify({ drawn, drawn60 }));
    """)
    assert out["drawn"] == 6      # every second repaint, not every third
    assert out["drawn60"] == 12   # every repaint


def test_a_rate_the_door_cannot_honour_is_brought_into_range():
    """§ 10.1: "a frame rate below the floor or above the ceiling is brought into
    range rather than honoured or refused, and the animation keeps running across
    the change" — asserted AT THE DOOR, which is where a caller reaches.

    The clock divides by the rate to decide when a frame is due, so an unclamped
    one does not merely animate oddly: ``1000/0`` is Infinity, a frame is never
    due, and the animation stops for good while the handle goes on reporting
    itself as playing.  A negative rate makes every repaint due instead.

    The floor is 5 fps, so 12 repaints of a 60 Hz display (200 ms) is one frame.
    The ceiling is 120, which a 60 Hz display can only meet on every repaint.
    """
    out = _run("""
        async function drawsIn12Repaints(rate) {
            const v = await mount(host, {});
            v.setStructure(WATER);
            v.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
            v.setFps(rate);
            v.play();
            calls.length = 0;
            pump(12, 16.67);
            const n = calls.filter((c) => c[0] === "render").length;
            const stillPlaying = v.isPlaying();
            v.dispose(); rafQ = [];
            return { n, stillPlaying };
        }
        console.log(JSON.stringify({
            zero:     await drawsIn12Repaints(0),
            negative: await drawsIn12Repaints(-10),
            huge:     await drawsIn12Repaints(10000),
            sane:     await drawsIn12Repaints(30),
        }));
    """)
    # zero and negative are slider end-stops, not caller bugs: they land on the
    # floor and keep animating rather than freezing or running flat out
    assert out["zero"]["n"] >= 1,     "a zero rate froze the animation"
    assert out["zero"]["n"] <= 2,     "a zero rate ran faster than the floor"
    assert out["zero"]["stillPlaying"] is True
    assert out["negative"]["n"] >= 1, "a negative rate froze the animation"
    assert out["negative"]["n"] <= 2, "a negative rate ran flat out"
    # the ceiling is above what a 60 Hz display offers, so it draws every repaint
    assert out["huge"]["n"] == 12
    assert out["sane"]["n"] == 6      # unchanged: every second repaint


def test_nonsense_is_ignored_rather_than_reset_to_a_default():
    """A value out of range and a value of the wrong kind are different answers.

    A slider at its end stop has a clear intention this module cannot honour, so
    it is honoured as far as it goes.  ``setFps("fast")`` is a caller bug, and
    quietly substituting the default would hide it — the viewer keeps the rate it
    already had.
    """
    out = _run("""
        const vib = await mount(host, { fps: 60 });
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.setFps("fast");        // ignored -- 60 stands
        vib.setFps(NaN);           // ignored
        vib.play();
        calls.length = 0;
        pump(12, 16.67);
        console.log(JSON.stringify({
            drawn: calls.filter((c) => c[0] === "render").length }));
    """)
    assert out["drawn"] == 12       # still 60 fps, not reset to the 30 default


def test_a_viewer_mounted_with_the_caption_off_never_draws_one():
    """§ 8: the mount option is the default, and it is honoured from the first
    mode rather than after a first draw."""
    out = _run("""
        const vib = await mount(host, { showLabel: false });
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]],
                       label: "Mode 1" });
        const el = caption();
        console.log(JSON.stringify({ hidden: !el || el.hidden }));
    """)
    assert out["hidden"] is True


# ---------------------------------------------------------------------------
# § 6.4 — what a viewer does not hold
# ---------------------------------------------------------------------------

def test_the_handle_offers_no_read_of_what_it_holds():
    """§ 6.4: "the handle offers no read of the structure, the mode's vectors, the
    mode's index, or the camera".

    A read that hands back what the caller passed in one line earlier is a second
    place to believe something.  `isPlaying` is the deliberate exception (§ 9.2):
    a play/pause button has to draw itself from somewhere.
    """
    out = _run("""
        const vib = await mount(host, {});
        // Every key, unfiltered.  An earlier draft carried a `_capture` hatch and
        // this line filtered it out to stay green -- a check laundering the thing
        // it exists to catch.  If a hatch comes back, this fails.
        const doors = Object.keys(vib);
        console.log(JSON.stringify({ doors: doors.sort() }));
    """)
    assert out["doors"] == sorted([
        "ok", "setStructure", "showMode", "play", "pause", "isPlaying",
        "setAmplitude", "setFps", "setCycleSec", "setLabelVisible", "dispose",
    ])


def test_dispose_stops_the_clock_and_empties_the_host():
    """§ 8: "dispose stops the clock, releases the drawing surface, and leaves the
    host element empty.  Calling it twice is safe"."""
    out = _run("""
        const vib = await mount(host, {});
        vib.setStructure(WATER);
        vib.showMode({ index: 1, displacements: [[1,0,0],[0,1,0],[0,0,1]] });
        vib.play();
        vib.dispose();
        calls.length = 0;
        pump(4, 1000);
        let threwOnSecond = false;
        try { vib.dispose(); } catch (_) { threwOnSecond = true; }
        console.log(JSON.stringify({
            keptDrawing: calls.some((c) => c[0] === "render"),
            playing: vib.isPlaying(), threwOnSecond,
            classGone: !classes.has("vibview"),
        }));
    """)
    assert out["keptDrawing"] is False
    assert out["playing"] is False
    assert out["threwOnSecond"] is False
    assert out["classGone"] is True

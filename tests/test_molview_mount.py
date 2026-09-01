"""Mounting, and the handle — every test derived from ``docs/web/molview.md``,
never from the source it checks (§ 13).

Step G of the rebuild (``docs/web/molview.md``). The rows of § 13.3
guarded here:

    § 4    the module is self-contained
    § 8    mount always resolves
    § 8.2  the floor is the stacked minimum, not the row sum
    § 8.3  the arrow turns, the handle does not
    § 8.5  a control reads a fact from where it lives
    § 9.2  the handle refuses appearance
    § 5.6  a viewer is owned

Level 2 of § 13.2: boundary behaviour, with stand-ins that obey this document —
a DOM stand-in and the drawing-library stand-in from step B. What a viewer LOOKS
like is § 13.2's third level, in a real page, and nothing here pretends to check
it.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node
from tests._molview_sources import module_code, module_files

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
INDEX = MODULE_DIR / "index.js"
SUPPORT = Path(__file__).parent / "support"

# ORDER MATTERS: both stand-ins offer a host element, and the DOM one must win —
# mount builds real elements inside it, which the embed's minimal host cannot do.
GLOBALS = (SUPPORT / "molview_3dmol_standin.js").read_text() + """
// The DOM, loaded after so its richer host element is the one in play.
""" + (SUPPORT / "molview_dom_standin.js").read_text() + """
globalThis.__requests = [];
/* The atom identity the stand-in server hands back. A test that needs a
 * different one sets it BEFORE installing — because § 10.8 fixes the atoms at
 * load and every frame after that has to match, so a test cannot reach a
 * three-atom trajectory by loading two atoms and reloading three frames. */
globalThis.__nextAtoms = null;
globalThis.fetch = async function (route, init) {
    globalThis.__requests.push({ route, body: JSON.parse(init.body) });
    const atoms = globalThis.__nextAtoms || [
        { index: 0, element: "C", x: 0, y: 0, z: 0, regions: [] },
        { index: 1, element: "O", x: 1, y: 0, z: 0, regions: [] },
    ];
    const extra = globalThis.__nextNotices
        ? { notices: globalThis.__nextNotices } : {};
    return { ok: true, status: 200, json: async () => Object.assign({ atoms }, extra) };
};
globalThis.__nextNotices = null;
globalThis.setInterval = globalThis.setInterval;
"""

PRELUDE = f"""
const MV = await import({json.dumps(INDEX.resolve().as_uri())});

const workspace = {{ read: async () => null, write: async () => {{}} }};

/* The drawing lands a TURN after the model does.
 *
 * `installMolecule` resolves when the MODEL has the structure; the rebuild that
 * puts it on screen yields a turn first, deliberately -- that turn is what lets
 * the busy cover be painted and lets it catch the input a freeze queues up
 * (molview.md § 10.9). So a test that reads the drawing's call log straight
 * after the await reads it too early, and these two did.
 *
 * Waits for the CONDITION rather than a fixed delay: a sleep long enough to be
 * safe is long enough to be slow, and one that is neither is a flake. */
async function waitForDrawing(check, why) {{
    for (let i = 0; i < 200; i++) {{
        if (check()) return true;
        await new Promise((r) => setTimeout(r, 5));
    }}
    throw new Error("the drawing never received it: " + why);
}}

async function mounted(opts) {{
    const host = globalThis.__makeHost(opts && opts.width, opts && opts.minWidth);
    const viewer = await MV.mount(host, workspace,
        Object.assign({{ owner: "test" }}, (opts && opts.mount) || {{}}));
    return {{ host, viewer }};
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=GLOBALS)


# ---------------------------------------------------------------------------
# § 4 — the module is self-contained
# ---------------------------------------------------------------------------

def test_the_entry_point_offers_two_names_and_no_others():
    """§ 4, § 9.1: "`mount` and `formula`. Nothing else in the module is
    importable, and this is the only import a consumer ever writes."
    """
    out = _run(
        """
        console.log(JSON.stringify({
            exports: Object.keys(MV).sort(),
            formula: MV.formula(["C","H","H","H","H"]),
            empty: MV.formula([]),
        }));
        """
    )
    assert out["exports"] == ["formula", "mount"], (
        f"the entry point exports more than the contract allows: {out['exports']}"
    )
    assert out["formula"] == "CH4", "C and H first, then the rest"
    assert out["empty"] == "—"


def test_mounting_needs_only_a_host_and_a_workspace_door():
    """§ 13.3: "the module mounts given only a host element and something that
    satisfies the workspace door."

    § 8: "anything that can store and return bytes satisfies it. That is what
    lets a viewer mount in a test page with a stand-in, and it is the whole of
    § 4's 'nothing it needs comes from a global'."
    """
    out = _run(
        """
        const host = globalThis.__makeHost();
        // A door, not a module: two functions and nothing else.
        const door = { read: async () => null, write: async () => {} };
        const viewer = await MV.mount(host, door, { owner: "results-structure" });
        console.log(JSON.stringify({
            ok: viewer.ok,
            published: Object.keys(globalThis.molbuilder),
        }));
        """
    )
    assert out["ok"] is True, "a viewer must mount with a stand-in workspace"
    assert out["published"] == [], (
        f"mounting published into the app's global namespace: {out['published']}"
    )


# ---------------------------------------------------------------------------
# § 8 — mount always resolves
# ---------------------------------------------------------------------------

def test_mount_always_resolves_with_a_working_dispose():
    """§ 13.3: "a mount that cannot fit still returns `ok === false` AND a working
    `dispose`; nothing rejects, nothing returns nothing."

    § 8: so a caller's teardown path needs no special case for the viewer that
    did not build.
    """
    out = _run(
        """
        const results = [];
        for (const [name, run] of [
            ["no host",  () => MV.mount(null, workspace, { owner: "x" })],
            ["no owner", () => MV.mount(globalThis.__makeHost(), workspace, {})],
            ["too narrow", () => MV.mount(
                globalThis.__makeHost(200, 350), workspace, { owner: "x" })],
        ]) {
            let v, threw = false, disposeThrew = false;
            try { v = await run(); } catch (e) { threw = true; }
            if (v) { try { v.dispose(); } catch (e) { disposeThrew = true; } }
            results.push({
                name, threw, returned: v !== undefined && v !== null,
                ok: v && v.ok, hasError: !!(v && v.error), disposeThrew,
            });
        }
        console.log(JSON.stringify({ results }));
        """
    )
    for row in out["results"]:
        assert row["threw"] is False, f"{row['name']}: mount rejected"
        assert row["returned"] is True, f"{row['name']}: mount returned nothing"
        assert row["ok"] is False, f"{row['name']}: should not have mounted"
        assert row["hasError"] is True, f"{row['name']}: no error said why"
        assert row["disposeThrew"] is False, f"{row['name']}: dispose did not work"


def test_the_floor_is_the_stacked_minimum_not_the_row_sum():
    """§ 8.2: "below the sum the card does not break — it STACKS, window above
    panel, and both are still usable. So the floor a host has to respect is the
    stacked minimum."

    § 13.3: "a host narrower than the side-by-side sum still mounts, and stacks;
    only one narrower than the wider single piece gets the blank card and the
    error."

    Getting this backwards makes MolView refuse hosts where it would have worked
    perfectly well by stacking — which is the failure this pins.
    """
    out = _run(
        """
        // A card whose stacked floor is 350 and whose side-by-side sum is 692.
        const narrow = await MV.mount(globalThis.__makeHost(500, 350), workspace,
                                      { owner: "x" });
        const broken = await MV.mount(globalThis.__makeHost(200, 350), workspace,
                                      { owner: "x" });
        console.log(JSON.stringify({
            betweenSumAndFloor: narrow.ok,
            belowFloor: broken.ok,
            error: broken.error,
        }));
        """
    )
    assert out["betweenSumAndFloor"] is True, (
        "a host narrower than the side-by-side sum but wider than the stacked "
        "minimum must MOUNT and stack — refusing it is the mistake § 8.2 names"
    )
    assert out["belowFloor"] is False
    assert "350" in out["error"], (
        f"the error must say what MolView needed: {out['error']}"
    )


def test_a_viewer_that_cannot_fit_draws_a_blank_card_with_the_error_in_it():
    """§ 8: "renders a blank card with the error written in it, rather than a
    half-built viewer."
    """
    out = _run(
        """
        const host = globalThis.__makeHost(200, 350);
        const viewer = await MV.mount(host, workspace, { owner: "x" });
        const card = host.querySelector(".molviewer-card");
        console.log(JSON.stringify({
            ok: viewer.ok,
            hasCard: !!card,
            hasViewer: !!(card && card.querySelector(".molviewer-window")),
            hasPanel: !!(card && card.querySelector(".molviewer-panel")),
            errorText: card && card.querySelector(".molviewer-embed-error")
                ? card.querySelector(".molviewer-embed-error").textContent : null,
        }));
        """
    )
    assert out["hasCard"] is True, "there must still be a card"
    assert out["hasViewer"] is False and out["hasPanel"] is False, (
        "a half-built viewer was left on the page instead of a blank card"
    )
    assert out["errorText"] and "350" in out["errorText"]


# ---------------------------------------------------------------------------
# § 9.2 — the handle
# ---------------------------------------------------------------------------

def test_the_handle_refuses_appearance():
    """§ 13.3: "there is no way through the handle to push arrows, labels, a busy
    state or a toggle — arrows come from the forces in the data or are not drawn
    at all."

    § 9.2: arrows, labels and the highlight are WORKED OUT FROM THE DATA by the
    renderEngine, never given to it.
    """
    out = _run(
        """
        const { viewer } = await mounted();
        const names = Object.keys(viewer);
        const forbidden = names.filter(n =>
            /^set(Arrows|Labels|Busy|Highlight|Style|Toggle)/.test(n)
            || /^(addToggle|showBusy|setAppearance)$/.test(n));
        console.log(JSON.stringify({ names: names.sort(), forbidden }));
        """
    )
    assert out["forbidden"] == [], (
        f"the handle accepts a finished appearance: {out['forbidden']}"
    )


def test_the_handle_contains_the_model_and_does_not_mirror_it():
    """§ 9.2: "The handle CONTAINS the model; it does not mirror it … Adding a
    read to the handle that the model already answers is the specific move this
    rule forbids."

    A mirrored read is a second surface over the same fact, and one of the two is
    the one somebody forgets to update.
    """
    out = _run(
        """
        const { viewer } = await mounted();
        const handleNames = Object.keys(viewer);
        const modelNames = Object.keys(viewer.data);
        const mirrored = handleNames.filter(n => n !== "data" && modelNames.includes(n));
        console.log(JSON.stringify({ handleNames: handleNames.sort(), mirrored }));
        """
    )
    assert out["mirrored"] == [], (
        f"the handle mirrors reads the model already answers: {out['mirrored']}"
    )
    assert "data" in out["handleNames"], (
        "the handle must carry one route to the model"
    )


def test_playback_moves_the_frame_through_the_same_write_everyone_uses():
    """§ 9.2: the handle owns the timer, but "playback lives in the mount layer
    and moves the frame through the same write everyone else uses (§ 6.4)".

    § 6.4: the write "tells EVERY subscriber regardless of what did the moving",
    so a subscriber never has to know which.
    """
    out = _run(
        """
        const { viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.reloadFrames([[[0,0,0],[1,0,0]], [[2,0,0],[3,0,0]]]);

        const heard = [];
        viewer.data.onFrameChange((f) => heard.push(f));

        // Moved by the model directly, and by playback — indistinguishable
        // downstream, which is the point.
        viewer.data.setCurrentFrame(1);
        const byHand = heard.slice();
        viewer.data.setCurrentFrame(0);
        // Speed is SET and then played at — `play` takes no knobs (§ 9.2).
        viewer.setSpeed(20);
        viewer.play();
        await new Promise(r => setTimeout(r, 150));
        viewer.pause();

        console.log(JSON.stringify({
            byHand, heard, playing: viewer.isPlaying(),
        }));
        """
    )
    assert out["byHand"] == [1], "a direct write must reach the subscriber"
    assert len(out["heard"]) > 1, (
        "playback must move the frame through the same write, so the same "
        f"subscriber hears it: {out['heard']}"
    )
    assert out["playing"] is False, "pause must stop the timer"


# ---------------------------------------------------------------------------
# § 8.3 / § 8.5 — the card's own controls
# ---------------------------------------------------------------------------

def test_the_arrow_turns_and_the_handle_does_not():
    """§ 8.3: "Rotating the handle itself would swap its width and height, which
    in the stacked layout turns a small grip into a tall rail lying across the
    window. Only the glyph turns."

    So folding must change a class on the CARD (which the stylesheet keys both
    layouts off) and leave the button's own box alone.
    """
    out = _run(
        """
        const { host } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const fold = card.querySelector(".molviewer-panel-fold-btn");
        const chevron = card.querySelector(".molviewer-panel-fold-chevron");

        const before = { folded: card.classList.contains("molviewer-is-folded"),
                         expanded: fold.getAttribute("aria-expanded"),
                         btnStyle: JSON.stringify(fold.style) };
        fold.click();
        const after = { folded: card.classList.contains("molviewer-is-folded"),
                        expanded: fold.getAttribute("aria-expanded"),
                        btnStyle: JSON.stringify(fold.style) };
        console.log(JSON.stringify({ before, after, hasChevron: !!chevron }));
        """
    )
    assert out["hasChevron"] is True, (
        "the glyph must be its own element, so only it can be rotated"
    )
    assert out["before"]["folded"] is False and out["after"]["folded"] is True
    assert out["before"]["expanded"] == "true" and out["after"]["expanded"] == "false", (
        "folding must say so, for anyone not looking at the arrow"
    )
    assert out["before"]["btnStyle"] == out["after"]["btnStyle"], (
        "folding changed the handle's own box — the stylesheet decides both "
        "layouts, and rotating the box lays a rail across the window"
    )


def test_the_frame_bar_appears_only_once_there_is_more_than_one_frame():
    """§ 8: "the frame bar is the one piece that is not decided at mount: a viewer
    mounts before it has a structure, and the bar appears once a structure with
    more than one frame is loaded into it."
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        const bar = host.querySelector(".molviewer-frames-bar");
        const atMount = bar.hidden;

        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const oneFrame = bar.hidden;

        viewer.data.reloadFrames([[[0,0,0],[1,0,0]], [[2,0,0],[3,0,0]]]);
        const manyFrames = bar.hidden;

        viewer.data.setCurrentFrame(1);
        const counter = bar.querySelector(".molviewer-frames-counter").textContent;
        console.log(JSON.stringify({ atMount, oneFrame, manyFrames, counter }));
        """
    )
    assert out["atMount"] is True, "no structure, no bar"
    assert out["oneFrame"] is True, "one frame is not a trajectory"
    assert out["manyFrames"] is False, "the bar appears once there is a movie"
    assert out["counter"] == "2 / 2", (
        f"the counter reads 1-based, through the one translation: {out['counter']}"
    )


def test_a_switch_and_a_selection_reach_the_drawing():
    """§ 9.5: the switches live in the selection store, and the renderEngine
    reads them "when working out a processed frame" (§ 9.6's table). § 10.5: a
    switch flip costs an overlay refresh; a change to the set of drawn atoms
    costs a rebuild.

    This is the test the suite did not have, and its absence cost six features at
    once. The renderEngine was correct and was tested against a stand-in data
    source that offered `switches()` and `selection()`; the MODEL handed it four
    doors and neither of those, so every frame was worked out from no switches
    and an empty selection. Atom-number labels, force arrows, the cell, the axes,
    the highlight and isolate were dead together, and every layer passed its own
    tests.

    So this one asserts across the seam — a switch set through the model's own
    door, and what arrives at the drawing.
    """
    out = _run(
        """
        const { viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        // The load's own rebuild has to FINISH first. While it is in flight the
        // engine is in its rebuild window, where a switch is deliberately not
        // held and not drawn -- "the rebuild reads the switches when it runs"
        // (§ 10.9). Exercising switches before then tests the window, not the
        // wiring this test is about.
        await waitForDrawing(() => globalThis.__callNames().includes("addModelsAsFrames"),
                    "the load's rebuild to finish");

        globalThis.__resetCalls();
        viewer.data.selection.setSwitch("showIndex", true);
        const labelled = globalThis.__countCalls("addLabel");

        globalThis.__resetCalls();
        viewer.data.pickAtom(1);
        const highlighted = globalThis.__countCalls("addSphere");

        // Isolate changes WHICH atoms are drawn, so the movie is reloaded and
        // carries only the selected one (§ 10.5's rebuild).
        globalThis.__resetCalls();
        viewer.data.selection.setSwitch("isolate", true);
        await new Promise((r) => setTimeout(r, 0));
        const reloads = globalThis.__countCalls("addModelsAsFrames");
        const drawn = globalThis.__lastCall("setStyle") ? true : false;

        console.log(JSON.stringify({
            labelled, highlighted, reloads, drawn,
            frames: globalThis.__lastCall("addModelsAsFrames") ? 1 : 0,
        }));
        """
    )
    assert out["labelled"] > 0, (
        "turning on atom numbers drew no labels — the switch reached nothing"
    )
    assert out["highlighted"] > 0, (
        "selecting an atom drew no highlight — the selection reached nothing"
    )
    assert out["reloads"] == 1, (
        f"isolate did not rebuild the movie: {out['reloads']} reloads"
    )


def test_the_switches_are_a_rail_of_buttons_outside_the_window():
    """§ 1.1 / § 8.5: icon buttons down the left edge, "always outside the
    canvas, never on top of the molecule" — Reset view, axes, atom labels, force
    vectors, unit cell, show-selected-only, and (2026-08-30) **measure**, in
    that order.

    The seventh is the one that is not the selection's: § 8.5 now says each
    button reads its lit state "from the store that owns it", so this test also
    pins that a switch living in `measurement` lights from there.

    Two things are asserted and both are the design rather than decoration: that
    they are one-press controls a user can see without opening anything, and that
    they are OUTSIDE the window — the card's own arithmetic pays for the column
    (`--rail-w`) precisely so the controls never cover the molecule.

    Each reads its lit state from the store (§ 5.2), so a switch flipped
    somewhere else lights the right button with nothing kept in step.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const rail = card.querySelector(".molviewer-rail");
        const buttons = rail.children;

        const shape = {
            glyphs: buttons.map((b) => b.textContent),
            names:  buttons.map((b) => b.getAttribute("aria-label")),
            // Reset is an action, so it is the one with no lit state.
            pressed: buttons.map((b) => b.getAttribute("aria-pressed")),
            insideCanvas: !!card.querySelector(".molviewer-window-canvas")
                              .querySelector(".molviewer-rail"),
        };

        // A press writes the store...
        buttons[4].click();                       // the unit cell
        const afterPress = viewer.data.selection.getState().showCell;
        // ...and a switch set anywhere else lights the button.
        viewer.data.selection.setSwitch("showAxis", true);
        const litFromStore = buttons[1].getAttribute("aria-pressed");

        // The seventh button reads a DIFFERENT store (§ 11.6), so both
        // directions are checked against it too.
        buttons[6].click();
        const measuringAfterPress = viewer.data.measurement.getState().active;
        viewer.data.measurement.setActive(false);
        const measureLitFromStore = buttons[6].getAttribute("aria-pressed");
        // ...and pressing it must not have moved anything in the selection.
        const selectionUntouched = viewer.data.selection.get();

        globalThis.__resetCalls();
        buttons[0].click();                       // Reset view
        const refit = globalThis.__countCalls("zoomTo");

        console.log(JSON.stringify({ shape, afterPress, litFromStore, refit,
                                    measuringAfterPress, measureLitFromStore,
                                    selectionUntouched }));
        """
    )
    assert out["shape"]["glyphs"] == ["⟲", "✚", "#", "➤", "▦", "◉", "∡"], (
        f"the rail is not § 8.5's buttons in order: {out['shape']['glyphs']}"
    )
    assert out["shape"]["names"] == [
        "Reset view", "Show axes", "Show atom labels",
        "Show force vectors", "Show unit cell", "Show selected only",
        "Measure",
    ]
    assert out["shape"]["pressed"][0] is None, (
        "Reset view carries a pressed state; it is an action, not a switch"
    )
    assert all(p == "false" for p in out["shape"]["pressed"][1:]), (
        "every switch starts off (§ 9.5)"
    )
    assert out["shape"]["insideCanvas"] is False, (
        "the rail is inside the 3D window — § 1.1 puts it outside, so it can "
        "never cover the molecule"
    )
    assert out["afterPress"] is True, "pressing a rail button set no switch"
    assert out["litFromStore"] == "true", (
        "a switch set elsewhere did not light its button — the rail is "
        "remembering its own state instead of reading the store"
    )
    assert out["refit"] == 1, "Reset view did not re-fit the camera"
    assert out["measuringAfterPress"] is True, (
        "the measure button wrote nothing — a rail switch whose home is not "
        "`selection` still has to reach it (§ 8.5)"
    )
    assert out["measureLitFromStore"] == "false", (
        "turning measuring off elsewhere left its button lit — the rail is "
        "reading the wrong store, or remembering its own answer"
    )
    assert out["selectionUntouched"] == [], (
        "turning the ruler on moved the selection (§ 11.6's wall)"
    )


def test_an_open_menu_is_placed_against_its_own_trigger():
    """§ 8.5: MolView's menus are controls the module draws, and a control that
    opens onto nothing is not one.

    The popover is fixed to the viewport — it has to be, because it opens over
    the 3D window and the window clips its own contents, so an in-flow popover is
    cut off at the canvas edge. Fixed positioning has no anchor, so the module
    measures: the panel hangs under its trigger, and is pulled back inside the
    window rather than allowed off the edge.

    Until this was wired the menu opened, took its open state, and showed
    nothing: the stylesheet parks the panel off-screen until something places it,
    so it was on screen the whole time at -9999px.
    """
    out = _run(
        """
        const { host } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const menu = card.querySelector("DETAILS");
        const summary = menu.querySelector("SUMMARY");
        const body = menu.querySelector(".molviewer-menu-body");

        const parked = { top: body.style.top || null, left: body.style.left || null };

        // A trigger with room below it.
        summary._rect = { top: 100, left: 40, width: 60, height: 20 };
        body._rect = { width: 200, height: 150 };
        menu.open = true;
        const under = { top: body.style.top, left: body.style.left };

        // The same menu, its trigger against the right edge (window: 1200).
        summary._rect = { top: 100, left: 1150, width: 60, height: 20 };
        menu.open = false;
        menu.open = true;
        const clamped = body.style.left;

        // And with no room below it (window: 800 tall).
        summary._rect = { top: 700, left: 40, width: 60, height: 20 };
        menu.open = false;
        menu.open = true;
        const flipped = body.style.top;

        console.log(JSON.stringify({ parked, under, clamped, flipped }));
        """
    )
    assert out["parked"] == {"top": None, "left": None}, (
        "the module placed the panel before it was opened — the stylesheet owns "
        "where it rests"
    )
    # 100 + 20 + the 4px it hangs by; the trigger's own left edge.
    assert out["under"] == {"top": "124px", "left": "40px"}, (
        f"an open menu was not placed under its trigger: {out['under']}"
    )
    # 1200 - 8 of margin - 200 of panel: pulled back, not left hanging off.
    assert out["clamped"] == "992px", (
        f"a menu near the edge was left off the screen: {out['clamped']}"
    )
    # 700 - 4 - 150: it opens upwards rather than off the bottom.
    assert out["flipped"] == "546px", (
        f"a menu with no room below it did not open upwards: {out['flipped']}"
    )


def test_a_menu_closes_on_a_click_elsewhere_and_leaves_nothing_behind():
    """§ 8.5 again, and § 8's teardown: "dispose still works".

    An open popover is fixed to the VIEWPORT, so the module listens for the
    things that move its trigger out from under it. Those listeners are on the
    window and the document — outside the card — so nothing removes them when the
    card goes. Disposing has to.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const menus = card.querySelectorAll("DETAILS");
        menus[0].open = true;

        // Opening the second closes the first: one open at a time.
        menus[1].open = true;
        const exclusive = { first: menus[0].open, second: menus[1].open };

        // A click anywhere else closes it.
        document.dispatch("click", { target: document.body });
        const afterOutsideClick = menus[1].open;

        const listening = {
            scroll: globalThis.listenerCount("scroll"),
            resize: globalThis.listenerCount("resize"),
            click:  document.listenerCount("click"),
        };
        viewer.dispose();
        const afterDispose = {
            scroll: globalThis.listenerCount("scroll"),
            resize: globalThis.listenerCount("resize"),
            click:  document.listenerCount("click"),
        };
        console.log(JSON.stringify({
            exclusive, afterOutsideClick, listening, afterDispose,
        }));
        """
    )
    assert out["exclusive"] == {"first": False, "second": True}, (
        "two menus were open at once"
    )
    assert out["afterOutsideClick"] is False, (
        "a click outside left the menu open — the trigger is not the only way out"
    )
    assert out["listening"] == {"scroll": 1, "resize": 1, "click": 1}, (
        f"the open menu is not being followed: {out['listening']}"
    )
    assert out["afterDispose"] == {"scroll": 0, "resize": 0, "click": 0}, (
        f"disposing left listeners on the window and the document: "
        f"{out['afterDispose']}"
    )


def test_each_write_costs_what_the_cost_table_says_it_costs():
    """§ 10.5: the pipeline "does the least work that still produces the correct
    result", and the cost is chosen by WHAT CHANGED — a frame swap, an overlay
    refresh, an append, or a rebuild.

    That decision is the engine's, but it can only make it if the model says
    which change this was. Saying "the data changed" for every write, which is
    what the model did, collapses the four into one: a streamed append reloaded
    the whole movie, a cell edit reloaded it, and tagging an atom — which moves
    nothing and draws nothing — reloaded it too. The table was correct and
    unreachable, and no test noticed because every test that knew about costs
    drove the engine directly.

    So this drives the MODEL's own doors and counts what the drawing was asked
    to do. `addModelsAsFrames` is a movie reload; nothing else is.
    """
    out = _run(
        """
        const { viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.reloadFrames([[[0,0,0],[1,0,0]], [[0,1,0],[1,1,0]]]);
        await new Promise((r) => setTimeout(r, 0));

        const reloads = () => globalThis.__countCalls("addModelsAsFrames");
        const counted = {};

        globalThis.__resetCalls();
        viewer.data.addFrames([[[0,2,0],[1,2,0]]]);
        await new Promise((r) => setTimeout(r, 0));
        counted.append = reloads();

        globalThis.__resetCalls();
        viewer.data.pickAtom(0);
        viewer.data.selection.writeLabel("L-electrode");
        await new Promise((r) => setTimeout(r, 0));
        counted.tag = reloads();

        globalThis.__resetCalls();
        viewer.data.setForces([[[0,0,1],[0,0,1]], [[0,0,1],[0,0,1]],
                               [[0,0,1],[0,0,1]]]);
        await new Promise((r) => setTimeout(r, 0));
        counted.forces = reloads();

        globalThis.__resetCalls();
        viewer.data.setCurrentFrame(1);
        await new Promise((r) => setTimeout(r, 0));
        counted.seek = reloads();

        // And the one that IS a rebuild, so the counter is not simply dead.
        globalThis.__resetCalls();
        viewer.data.reloadFrames([[[9,0,0],[9,1,0]]]);
        await new Promise((r) => setTimeout(r, 0));
        counted.load = reloads();

        console.log(JSON.stringify(counted));
        """
    )
    assert out["append"] == 0, "a streamed append reloaded the movie"
    assert out["tag"] == 0, (
        "tagging an atom reloaded the movie — a label moves nothing and draws "
        "nothing (§ 6.6)"
    )
    assert out["forces"] == 0, "new forces reloaded the movie"
    assert out["seek"] == 0, "moving the displayed frame reloaded the movie"
    assert out["load"] == 1, (
        "a full load must rebuild — if this is 0 the test is counting nothing"
    )


def test_the_readout_measures_from_the_truth_in_pick_order():
    """§ 11.6: the readout's atoms come from the MEASUREMENT TRACK in pick order
    — "which is why the vertex of a three-atom angle is the atom picked second,
    not the middle one by number" — and its coordinates from the master copy at
    the current frame, never from the drawing.

    Its input changed on 2026-08-30: it used to read `selection`, which is what
    an edit acts on.  The geometric-vertex fallback went with that change and is
    asserted GONE here — a selection can arrive with no pick order (All, Invert,
    a filter), a track cannot, so the guess has no case left to cover.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        // THREE atoms, fixed at load (§ 10.8) — the frames below must match the
        // identity the structure was opened with.
        globalThis.__nextAtoms = [
            { index: 0, element: "C", x: -1, y: 0, z: 0, regions: [] },
            { index: 1, element: "C", x:  0, y: 0, z: 0, regions: [] },
            { index: 2, element: "C", x:  0, y: 1, z: 0, regions: [] },
        ];
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        // A bent three-atom frame: 1 is the geometric middle, 0 is not.
        viewer.data.reloadFrames([
            [[-1, 0, 0], [0, 0, 0], [0, 1, 0]],
            [[-2, 0, 0], [0, 0, 0], [0, 2, 0]],
        ]);
        const card = host.querySelector(".molviewer-card");
        const readout = card.querySelector(".molviewer-overlay--measure");
        const lines = () => Array.from(
            readout.querySelectorAll(".molviewer-measure-line"))
            .map((n) => n.textContent);
        const result = () => Array.from(
            readout.querySelectorAll(".molviewer-measure-result"))
            .map((n) => n.textContent);

        // Off, it says nothing at all — even with atoms selected.
        viewer.data.selection.add([0, 1, 2]);
        const whileOff = readout.hidden;

        viewer.data.measurement.setActive(true);
        const emptyTrack = readout.hidden;

        // Picks go in through `pickAtom`, the one router -- writing them
        // straight into the store would skip the rule that decides whether a
        // click means measuring or selecting, which is the thing under test.
        // (`toggle` is not on the handed-out surface at all any more.)

        // One atom: where it is.
        viewer.data.pickAtom(1);
        const one = lines();

        // Two: both positions, the distance, and the signed delta.
        viewer.data.pickAtom(2);
        const two = lines();

        // Three, picked 2nd-1st-3rd, so the VERTEX IS ATOM 0 — the one picked
        // second — even though atom 1 is the middle by number and by geometry.
        viewer.data.measurement.clear();
        viewer.data.pickAtom(1);
        viewer.data.pickAtom(0);
        viewer.data.pickAtom(2);
        const picked = result();

        // A BULK SELECTION CANNOT REACH THE TRACK, so the no-trail case the
        // geometric guess existed for cannot arise.
        viewer.data.selection.all(3);
        const afterBulk = result();
        const trackAfterBulk = viewer.data.measurement.getState().picks;

        // It follows the frame, because it re-reads the master copy.
        viewer.data.setCurrentFrame(1);
        const laterFrame = result();
        const laterLines = lines();

        // And it is not the drawing's numbering: under isolate the drawn set is
        // cut down and renumbered, and the answer must not move.
        viewer.data.selection.setSwitch("isolate", true);
        const isolated = result();

        // Clear is on the chip, and it clears the RULER, not the selection.
        readout.querySelector(".molviewer-measure-clear").click();
        console.log(JSON.stringify({
            whileOff, emptyTrack, one, two, picked, afterBulk, trackAfterBulk,
            laterFrame, laterLines, isolated,
            afterClear: readout.hidden,
            selectionAfterClear: viewer.data.selection.get(),
        }));
        """
    )
    assert out["whileOff"] is True, (
        "the chip showed with the ruler off — it was reading the selection"
    )
    assert out["emptyTrack"] is True, "nothing picked is nothing to say"

    # § 1.1's formats, WORD FOR WORD. These assertions used to be
    # `startswith("#2")` — copied from what the code printed, in a file whose
    # docstring says every test comes from the document. So the readout had
    # never carried the element at all, and this test was what kept it that way.
    assert out["one"] == ["C #2 — (0.000, 0.000, 0.000) Å"], out["one"]
    # EVERY picked atom's coordinates, then the derived answers: the position is
    # what a reader checks the distance against (§ 11.6).
    assert out["two"] == [
        "C #2 — (0.000, 0.000, 0.000) Å",
        "C #3 — (0.000, 1.000, 0.000) Å",
        "|C #2 – C #3| = 1.000 Å",
        "Δ = (0.000, 1.000, 0.000) Å",
    ], out["two"]

    # Picked 1→0→2, so the vertex is atom 0 — off to the side, giving 45°. By
    # number or by geometry the vertex would be atom 1, giving 90°: the two
    # answers differ, which is what makes this test able to tell them apart.
    assert out["picked"] == ["∠C #2 – C #1 – C #3 = 45.0°"], out["picked"]
    assert out["afterBulk"] == out["picked"], (
        "a bulk SELECTION changed the measurement — the two tracks are joined"
    )
    assert out["trackAfterBulk"] == [1, 0, 2]

    assert out["laterFrame"] == out["picked"], (
        "the angle is 45° in both frames; only the coordinates should move"
    )
    assert out["laterLines"][0] == "C #2 — (0.000, 0.000, 0.000) Å"
    assert out["laterLines"][1] == "C #1 — (-2.000, 0.000, 0.000) Å", (
        "the readout did not re-read the master copy at the new frame: "
        f"{out['laterLines']}"
    )
    assert out["isolated"] == out["laterFrame"], (
        "isolate changed the measurement — it is reading the drawing's numbering "
        "instead of the master copy (§ 6.5)"
    )
    assert out["afterClear"] is True, "the chip's Clear did not empty the track"
    assert out["selectionAfterClear"] == [0, 1, 2], (
        "the chip's Clear emptied the SELECTION — the two Clears in this card "
        "are exactly the confusion § 11.6 separates them to avoid"
    )


def test_two_viewers_on_one_page_share_no_name():
    """§ 5.6 and § 12.6: two mounts are two viewers that share nothing — and a
    NAME is something they can share by accident, because ids and radio-group
    names are global to the document, not to the card.

    Both halves are guarded here because only one of them was fixed the first
    time: the radio groups were made owner-specific while the panel's structural
    hooks stayed ids, so a second viewer still duplicated `panel-page-cell` and
    the two sections beneath it.
    """
    out = _run(
        """
        const first = globalThis.__makeHost();
        const second = globalThis.__makeHost();
        await MV.mount(first, workspace, { owner: "left" });
        await MV.mount(second, workspace, { owner: "right" });

        const ids = [];
        (function walk(node) {
            const id = node.getAttribute && node.getAttribute("id");
            if (id) ids.push(id);
            for (const child of node.children || []) walk(child);
        })(globalThis.document.body);

        const groups = [first, second].map((host) =>
            host.querySelector(".molviewer-panel-tab-switch").children[0].children[0].name);

        console.log(JSON.stringify({
            ids, duplicates: ids.filter((id, at) => ids.indexOf(id) !== at),
            groups, groupsDiffer: groups[0] !== groups[1],
        }));
        """
    )
    assert out["duplicates"] == [], (
        f"two viewers wrote the same id twice: {out['duplicates']}"
    )
    assert out["ids"] == [], (
        f"MolView wrote ids at all — an id is a document-global name, and a "
        f"module that mounts twice may not claim one: {out['ids']}"
    )
    assert out["groupsDiffer"] is True, (
        "both viewers' page tabs are in one radio group, so choosing a page in "
        "one un-chooses it in the other"
    )


# ---------------------------------------------------------------------------
# § 6.7 — no hand-rolled file handling
# ---------------------------------------------------------------------------

def test_the_module_writes_no_file_handling_of_its_own():
    """§ 6.7: "MolView does not build a download link, does not make an object
    URL, does not touch a filesystem API, and does not call a file endpoint."

    The rule is worth checking flatly because the alternative looks so reasonable
    in the moment: offering a download is four lines of DOM, and writing them is
    quicker than threading a door through. It went that way once, and the result
    was a viewer that knew how to put a file on a user's disk — a second place
    that knowledge lived, outside every rule that applies to the real one.
    """
    # THE MODULE IS WHAT THE ENTRY POINT REACHES, not what the directory holds.
    # The demo page is a CONSUMER — it supplies the door rather than living
    # behind it — so it is not a layer and this rule is not its. That is
    # structural, not a name on a list (§ 13.1).
    offenders = {}
    for name, code in module_code().items():
        hits = [t for t in ("createObjectURL", ".download =", "showSaveFilePicker",
                            "createWritable", "FileSystemHandle", "/api/files/")
                if t in code]
        if hits:
            offenders[name] = hits
    assert offenders == {}, (
        f"the module handles files itself instead of through the door: {offenders}"
    )


def test_the_demo_imports_nothing_but_the_entry_point():
    """§ 13.4 / § 4: the demo is worth having precisely because it is held to the
    single import like every other consumer. A demo with a private door proves
    nothing.

    The page it runs on loads the drawing library and this file and nothing else
    — so if the module ever grows a hidden dependency on something the app
    happens to have loaded, the demo is where it stops working.
    """
    import re

    code = (MODULE_DIR / "demo.js").read_text()
    # AN IMPORT, not the word "from". This matched `from\s+"..."` anywhere in the
    # file, so the ordinary sentence `say("nothing to load from " + name)` read
    # as a second import and failed the test — a scanner loose enough to be
    # tripped by prose is one that will also be quietly satisfied by the wrong
    # thing. Both spellings are checked: the static form and the dynamic one,
    # which the old pattern could not see at all.
    imports = re.findall(r'\bimport\s[^;]*?\bfrom\s+"([^"]+)"', code)
    imports += re.findall(r'\bimport\s*\(\s*"([^"]+)"', code)
    imports += re.findall(r'\bimport\s+"([^"]+)"', code)
    assert imports == ["/static/lib/molview/index.js"], (
        f"the demo reaches past the entry point: {imports}"
    )


def test_the_structure_leaves_through_the_door_and_its_facts_go_with_it():
    """§ 11.7: what leaves is "what the viewer holds — the atoms, their positions,
    and the facts about them". NOT bytes: a coordinate document is a format the
    server owns, and a second writer in the browser is a second answer to what a
    saved structure looks like on disk. Both halves had already drifted — the
    document in its decimals, the sidecar in the version key that makes one
    loadable at all.

    One call with a destination argument, not two paths to keep in step: a
    separate download path is how the sidecar came to be dropped from one of
    them. Whether a sidecar is written at all is the codec's rule now, applied
    server-side on the one generator both destinations go through.

    Since 2026-08-19 the menu is § 11.3's two sections; the DATA rows are the
    truth's, the payload is `{structure, frames}`, and a one-frame structure
    asks no range (the dialog exists for a quantity there is only one of).
    """
    out = _run(
        """
        const saved = [];
        const host = globalThis.__makeHost();
        const viewer = await MV.mount(host, workspace, {
            owner: "x",
            files: {
                save: async (destination, stem, payload) => {
                    saved.push({ destination, stem, payload });
                    return { ok: true };
                },
                saveBinary: async () => ({ ok: true }),
            },
        });
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });

        // A label, so there is something for the export to lose.
        viewer.data.pickAtom(0);
        viewer.data.selection.writeLabel("L-electrode");

        const card = host.querySelector(".molviewer-card");
        const sections = card.querySelectorAll(".molviewer-export-section");
        const dataButtons = sections[0].querySelectorAll(".molviewer-export-btn");
        for (const item of dataButtons) item.click();
        await new Promise((r) => setTimeout(r, 20));   // the sends are async

        console.log(JSON.stringify({
            sections: sections.length,
            destinations: saved.map(s => s.destination),
            stems:        saved.map(s => s.stem),
            keys:         Object.keys(saved[0].payload.structure).sort(),
            regions:      saved[0].payload.structure.metadata.regions,
            framesSent:   saved.map(s => s.payload.frames !== undefined),
            dialogOpen:   !!card.querySelector(".molviewer-export-dialog"),
            identical:    JSON.stringify(saved[0].payload.structure)
                          === JSON.stringify(saved[1].payload.structure),
        }));
        """
    )
    assert out["sections"] == 2, (
        f"§ 11.3's menu is Data and Image — {out['sections']} section(s) found"
    )
    assert out["destinations"] == ["project", "download"], (
        "both destinations must go through the same door, differing only in the "
        f"destination they name: {out['destinations']}"
    )
    assert out["stems"] == ["x", "x"], (
        f"the stem is the name the structure came in under (§ 11.4): {out['stems']}"
    )
    assert out["dialogOpen"] is False, (
        "a one-frame structure never asks (§ 11.3) — a range dialog opened"
    )
    assert out["framesSent"] == [False, False], (
        "a one-frame export carries no frames array"
    )
    assert out["identical"] is True, (
        "the two destinations handed over different structures"
    )

    # WHAT LEAVES, not merely that something did. Asserting only that bytes went
    # out is what let this ship a server-request payload into a `.molstruct.json`
    # — a good .xyz paired with a .json the codec cannot read, losing every label
    # at the next open.
    assert out["keys"] == ["elements", "metadata", "positions"], (
        f"what leaves is not the structure the envelope defines: {out['keys']}"
    )
    assert out["regions"] == {"L-electrode": [0]}, (
        f"the labels did not leave with the atoms: {out['regions']}"
    )


def test_a_picture_is_the_view_and_leaves_through_saveBinary():
    """§ 11.3's Image, single frame: the dialog asks a RESOLUTION (the user's
    2026-08-19 addition; no range — one frame never asks one), the picture is
    asked of the drawing (§ 9.7's bounded asking), and the bytes leave
    through the door's binary half named `<stem>.png`."""
    out = _run(
        """
        const saved = [];
        const host = globalThis.__makeHost();
        const viewer = await MV.mount(host, workspace, {
            owner: "x",
            files: {
                save: async () => ({ ok: true }),
                saveBinary: async (destination, filename, blob) => {
                    saved.push({ destination, filename,
                                 isBlob: typeof blob === "object" && !!blob });
                    return { ok: true };
                },
            },
        });
        await viewer.data.installMolecule({ text: "x", filename: "wire.xyz" });

        const card = host.querySelector(".molviewer-card");
        const imageRow = card.querySelectorAll(".molviewer-export-section")[1];
        imageRow.querySelectorAll(".molviewer-export-btn")[1].click(); // Download
        await new Promise((r) => setTimeout(r, 10));
        const dialog = card.querySelector(".molviewer-export-dialog");
        const rangeInputs = dialog ? dialog.querySelectorAll("input").length : -1;
        const actions = dialog.querySelectorAll(".molviewer-export-btn");
        actions[actions.length - 1].click();          // Export at defaults
        await new Promise((r) => setTimeout(r, 30));

        console.log(JSON.stringify({
            rangeInputs,
            saved,
        }));
        """
    )
    assert out["rangeInputs"] == 0, (
        "a one-frame Image export asked a frame range (§ 11.3: it never asks)"
    )
    assert out["saved"] == [{"destination": "download",
                             "filename": "wire.png", "isBlob": True}], (
        f"the picture did not leave as <stem>.png through saveBinary: "
        f"{out['saved']}"
    )


def test_a_picture_is_the_view_and_leaves_through_saveBinary():
    """§ 11.3's Image, single frame: the dialog asks a RESOLUTION (the user's
    2026-08-19 addition; no range — one frame never asks one), the picture is
    asked of the drawing (§ 9.7's bounded asking), and the bytes leave
    through the door's binary half named `<stem>.png`."""
    out = _run(
        """
        const saved = [];
        const host = globalThis.__makeHost();
        const viewer = await MV.mount(host, workspace, {
            owner: "x",
            files: {
                save: async () => ({ ok: true }),
                saveBinary: async (destination, filename, blob) => {
                    saved.push({ destination, filename,
                                 isBlob: typeof blob === "object" && !!blob });
                    return { ok: true };
                },
            },
        });
        await viewer.data.installMolecule({ text: "x", filename: "wire.xyz" });

        const card = host.querySelector(".molviewer-card");
        const imageRow = card.querySelectorAll(".molviewer-export-section")[1];
        imageRow.querySelectorAll(".molviewer-export-btn")[1].click(); // Download
        await new Promise((r) => setTimeout(r, 10));
        const dialog = card.querySelector(".molviewer-export-dialog");
        const rangeInputs = dialog ? dialog.querySelectorAll("input").length : -1;
        const actions = dialog.querySelectorAll(".molviewer-export-btn");
        actions[actions.length - 1].click();          // Export at defaults
        await new Promise((r) => setTimeout(r, 30));

        console.log(JSON.stringify({ rangeInputs, saved }));
        """
    )
    assert out["rangeInputs"] == 0, (
        "a one-frame Image export asked a frame range (§ 11.3: it never asks)"
    )
    assert out["saved"] == [{"destination": "download",
                             "filename": "wire.png", "isBlob": True}], (
        f"the picture did not leave as <stem>.png through saveBinary: "
        f"{out['saved']}"
    )


def test_a_trajectorys_data_export_asks_the_range_and_sends_it():
    """§ 11.3: the dialog opens ON THE DISPLAYED FRAME (accepting it unchanged
    is the common case), widening it is what the dialog is for, and the range
    reaches `exportFile` — frames ride the payload, and the stem names both
    ends (§ 11.4's `_frameA-B`, which existed only in prose until 2026-08-19).
    """
    out = _run(
        """
        const saved = [];
        const host = globalThis.__makeHost();
        const viewer = await MV.mount(host, workspace, {
            owner: "x",
            files: {
                save: async (destination, stem, payload) => {
                    saved.push({ destination, stem, payload });
                    return { ok: true };
                },
                saveBinary: async () => ({ ok: true }),
            },
        });
        await viewer.data.installMolecule({ text: "x", filename: "wire.xyz" });
        viewer.data.addFrames([
            [[0,0,1],[1,0,1]], [[0,0,2],[1,0,2]], [[0,0,3],[1,0,3]],
        ]);
        viewer.data.setCurrentFrame(1);

        const card = host.querySelector(".molviewer-card");
        const dataRow = card.querySelectorAll(".molviewer-export-section")[0];
        dataRow.querySelectorAll(".molviewer-export-btn")[1].click();  // Download
        await new Promise((r) => setTimeout(r, 10));

        const dialog = card.querySelector(".molviewer-export-dialog");
        const opened = {
            exists: !!dialog,
            defaults: dialog
                ? Array.from(dialog.querySelectorAll("input"))
                      .map((i) => i.value)
                : null,
        };
        // Widen: frames 2..4 on screen = 1..3 in code.
        const inputs = dialog.querySelectorAll("input");
        inputs[0].value = "2";
        inputs[1].value = "4";
        const actions = dialog.querySelectorAll(".molviewer-export-btn");
        actions[actions.length - 1].click();          // Export
        await new Promise((r) => setTimeout(r, 20));

        console.log(JSON.stringify({
            opened,
            stem: saved[0] && saved[0].stem,
            frameCount: saved[0] && saved[0].payload.frames
                ? saved[0].payload.frames.length : null,
            dialogGone: !card.querySelector(".molviewer-export-dialog"),
        }));
        """
    )
    assert out["opened"]["exists"] is True
    assert out["opened"]["defaults"] == ["2", "2"], (
        f"the dialog must open on the DISPLAYED frame: {out['opened']['defaults']}"
    )
    assert out["stem"] == "wire_frame2-4", (
        f"the stem names both ends (§ 11.4): {out['stem']!r}"
    )
    assert out["frameCount"] == 3, (
        f"the chosen range did not reach exportFile: {out['frameCount']}"
    )
    assert out["dialogGone"] is True


def test_the_browser_writes_no_coordinate_document():
    """§ 11.7's rule, checked at the source rather than at one call site: the
    module contains no coordinate writer at all. One exception survives — a
    trajectory frame the server has never seen — and it is named there; a writer
    that reappears anywhere else has broken the rule quietly.

    Checked against the CODE, comments stripped: naming a format in prose is how
    the rule gets explained, and using it is how the rule gets broken.
    """
    import re as _re

    def code_of(text):
        text = _re.sub(r"/\*[\s\S]*?\*/", "", text)     # block comments
        return _re.sub(r"(?m)^\s*//.*$", "", text)      # line comments

    for path in sorted(MODULE_DIR.glob("*.js")):
        # `demo.js` is a HOST, not the module -- it reaches MolView only through
        # the entry point (pinned above), and turning a structure into bytes is
        # exactly a host's job. It asks the server for them.
        if path.name == "demo.js":
            continue
        body = code_of(path.read_text(encoding="utf-8"))
        assert ".molstruct.json" not in body, (
            f"{path.name} writes the sidecar filename -- the server owns that format"
        )
        assert ".xyz" not in body, (
            f"{path.name} writes a coordinate filename"
        )
        assert not _re.search(r"lines\.push\(.*elements\[", body), (
            f"{path.name} assembles a coordinate document"
        )


# ---------------------------------------------------------------------------
# § 8.4 / § 9.5 — the panel
# ---------------------------------------------------------------------------

def test_switching_editors_redraws_the_panel_and_moves_no_selection():
    """§ 9.5: "the selection is the truth; click and filter are two EDITORS of
    it. Switching between them does not touch what is selected."

    § 13.3: "moving between click and filter mode leaves the selection exactly as
    it was."
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.selection.add([0, 1]);

        const card = host.querySelector(".molviewer-card");
        const clickBody = card.querySelector(".molviewer-selection-click-section");
        const filterBody = card.querySelector(".molviewer-filter-section");

        const inClick = { click: clickBody.hidden, filter: filterBody.hidden,
                          selection: viewer.data.selection.get() };
        viewer.data.selection.setEditor("filter");
        const inFilter = { click: clickBody.hidden, filter: filterBody.hidden,
                           selection: viewer.data.selection.get() };
        console.log(JSON.stringify({ inClick, inFilter }));
        """
    )
    assert out["inClick"] == {"click": False, "filter": True, "selection": [0, 1]}
    assert out["inFilter"] == {"click": True, "filter": False, "selection": [0, 1]}, (
        f"switching editors moved the selection: {out['inFilter']}"
    )


def test_the_page_tabs_are_the_switch_the_stylesheet_draws():
    """§ 8.1: the panel has two pages and a tab bar that switches them — and the
    carried stylesheet is what draws that bar, so the markup is as much its
    contract as the class name is.

    It draws the chosen tab from a checked input inside the option and its type
    from a `span` inside it. Written as a plain button carrying its own text,
    NEITHER rule can match: the switch renders as two words in the browser's
    default font with nothing saying which page you are looking at. That is what
    shipped, and it is invisible to any test that only asks whether the click
    worked.

    Two viewers on a page get two groups, because a radio group is named
    document-wide (§ 5.6: a viewer owns everything in it).
    """
    out = _run(
        """
        const { host } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const options = card.querySelector(".molviewer-panel-tab-switch").children;
        const pages = card.querySelectorAll(".molviewer-panel-tab");

        const shape = options.map((o) => ({
            tag:    o.tagName,
            inside: o.children.map((c) => c.tagName),
            typed:  o.children[0].type,
            text:   o.textContent,
        }));
        const atMount = {
            checked: options.map((o) => !!o.children[0].checked),
            shown:   pages.map((p) => !p.hidden),
        };

        options[1].click();                            // choose Cell, as a user does
        const afterClick = {
            checked: options.map((o) => !!o.children[0].checked),
            shown:   pages.map((p) => !p.hidden),
        };

        // A second viewer on the same page: its tabs are its own.
        const otherHost = globalThis.__makeHost();
        await MV.mount(otherHost, workspace, { owner: "second-viewer" });
        const otherOptions = otherHost.querySelector(".molviewer-panel-tab-switch").children;

        console.log(JSON.stringify({
            shape, atMount, afterClick,
            groups: [options[0].children[0].name, otherOptions[0].children[0].name],
        }));
        """
    )
    assert [o["tag"] for o in out["shape"]] == ["LABEL", "LABEL",
                                                "LABEL"], (
        f"a tab is not the option the stylesheet draws: {out['shape']}"
    )
    for option in out["shape"]:
        assert option["inside"] == ["INPUT", "SPAN"], (
            f"the stylesheet draws the tab's state from the input and its type "
            f"from the span; this option has {option['inside']}"
        )
        assert option["typed"] == "radio", "two tabs, one choice: a radio group"
    assert [o["text"] for o in out["shape"]] == ["Selection", "Cell",
                                                 "Metadata"]
    assert out["atMount"] == {"checked": [True, False, False],
                              "shown": [True, False, False]}, (
        f"nothing said which page the panel opened on: {out['atMount']}"
    )
    assert out["afterClick"] == {"checked": [False, True, False],
                                 "shown": [False, True, False]}, (
        f"choosing the other tab did not move both the switch and the page: "
        f"{out['afterClick']}"
    )
    assert out["groups"][0] == "molview-page-test", (
        f"the radio group is not named after the viewer that owns it: "
        f"{out['groups'][0]}"
    )
    assert out["groups"][0] != out["groups"][1], (
        "two viewers on one page share a radio group, so choosing a page in one "
        "un-chooses it in the other"
    )


def test_the_atom_list_reads_one_based_and_a_click_goes_through_the_store():
    """§ 11.5: the first atom reads as #1 everywhere even though the code sees 0.

    And the panel holds nothing: a click asks the STORE to change, and the list
    redraws from the snapshot that comes back (§ 8.4).
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const card = host.querySelector(".molviewer-card");
        const rows = card.querySelectorAll(".molviewer-atoms-table")[0].children;

        const numbers = Array.from(rows).map(
            r => r.querySelectorAll(".molviewer-atoms-column-idx")[0].textContent);
        rows[1].click();                       // the SECOND atom, which reads #2
        console.log(JSON.stringify({
            numbers,
            selection: viewer.data.selection.get(),
            count: card.querySelector(".molviewer-selection-count").textContent,
        }));
        """
    )
    assert out["numbers"] == ["1", "2"], (
        f"the atom list must read 1-based: {out['numbers']}"
    )
    assert out["selection"] == [1], (
        "clicking the row that reads #2 must select the atom the code calls 1"
    )
    assert "1 of 2 selected" in out["count"]


def test_an_atom_row_ticks_shows_its_labels_and_lets_one_be_taken_off():
    """The atom list is a list of things you TICK, and each row shows the labels
    that atom carries with a × to take one off.

    § 6.2 fixes the columns: an atom's facts are its element, the labels it
    carries and its residue — so those are what a row shows, beside the number
    (§ 11.5, 1-based) and the tick.

    The × is a change to the STRUCTURE (§ 9.5), so it goes through the same
    label door with the same gate — it just names the atom instead of letting the
    door default to the selection. Two things follow, and both are asserted here:
    it works without the atom being selected first, and it leaves the selection
    exactly as it was.

    A reserved label is an ordinary one (§ 6.6): it arrives in the same list,
    wears the same chip, and comes off the same way. There is no second kind of
    tag and no case for it here.
    """
    out = _run(
        """
        globalThis.__nextAtoms = [
            { index: 0, element: "C", x: 0, y: 0, z: 0,
              regions: ["bridge", "frozen_atoms"], residue_name: "ALA" },
            { index: 1, element: "O", x: 1, y: 0, z: 0,
              regions: [], residue_name: "ALA" },
        ];
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const card = host.querySelector(".molviewer-card");
        const rows = () => card.querySelectorAll(".molviewer-atoms-table")[0].children;

        const first = rows()[0];
        const columns = {
            index:   first.querySelectorAll(".molviewer-atoms-column-idx")[0].textContent,
            element: first.querySelectorAll(".molviewer-atoms-column-el")[0].textContent,
            residue: first.querySelectorAll(".molviewer-atoms-column-res")[0].textContent,
            labels:  first.querySelectorAll(".molviewer-atoms-column-labels .molviewer-selection-tag")
                          .map(t => t.children[0].textContent),
        };

        // THE TICK. Driven as a change on the box itself, not a row click.
        const box = first.querySelectorAll(".molviewer-atoms-column-check input")[0];
        const tickedBefore = box.checked;
        box.dispatch("change", { target: box });
        const afterTick = viewer.data.selection.get();

        // Set from elsewhere: the box follows the store (§ 8.4).
        viewer.data.selection.clear();
        const afterClear = rows()[0].querySelectorAll(".molviewer-atoms-column-check input")[0].checked;

        // THE ×, on an atom nothing has selected — and on the RESERVED label,
        // because § 6.6 says it comes off exactly like any other.
        const selectionBefore = viewer.data.selection.get();
        rows()[0].querySelectorAll(".molviewer-atoms-column-labels .molviewer-selection-tag-remove")[1]
                 .dispatch("click", {});
        const left = rows()[0].querySelectorAll(".molviewer-atoms-column-labels .molviewer-selection-tag")
                              .map(t => t.children[0].textContent);

        console.log(JSON.stringify({
            columns, tickedBefore, afterTick, afterClear,
            selectionBefore, left,
            selectionAfter: viewer.data.selection.get(),
            otherAtomUntouched: viewer.data.getAtoms()[1].labels,
        }));
        """
    )
    assert out["columns"]["index"] == "1", "the row must read 1-based (§ 11.5)"
    assert out["columns"]["element"] == "C"
    assert out["columns"]["residue"] == "ALA", (
        "§ 6.2 makes the residue one of the three facts an atom carries"
    )
    assert out["columns"]["labels"] == ["bridge", "frozen_atoms"], (
        "every label the atom carries must show, the reserved one among them "
        f"and wearing the same chip (§ 6.6): {out['columns']['labels']}"
    )
    assert out["tickedBefore"] is False
    assert out["afterTick"] == [0], (
        f"ticking the box did not reach the selection store: {out['afterTick']}"
    )
    assert out["afterClear"] is False, (
        "the box kept its own answer instead of following the store (§ 8.4)"
    )
    assert out["selectionBefore"] == [], "the × is being tested on an unselected atom"
    assert out["left"] == ["bridge"], (
        f"the × did not take that one label off that one atom: {out['left']}"
    )
    assert out["selectionAfter"] == [], (
        "taking a label off disturbed the selection — the door defaulted to the "
        "selection instead of using the atom it was given"
    )
    assert out["otherAtomUntouched"] == [], (
        "removing a label from one atom reached another"
    )


def test_a_warning_from_a_load_is_put_in_front_of_the_user():
    """A LOAD REPORTS, IT DOES NOT REFUSE — so the report has to be visible.

    A structure whose box is unusable still opens: the user has to be able to
    see it to fix it, and the doors that GENERATE a calculation are the ones
    that refuse (nothing worth running comes out of an impossible box). That
    trade only works if the sentence the server sent actually reaches the
    screen — a warning nobody sees is the same as no check at all.

    MolView shows it, worded exactly as the server wrote it: the numbers in
    these messages — clearances, determinants, axes — were computed there, and
    rewording would put a second author on a sentence only one of them can
    write.
    """
    out = _run(
        """
        globalThis.__nextNotices = [
            { level: "warn",
              message: "cell must be right-handed (det > 0); got det = -1." },
        ];
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const card = host.querySelector(".molviewer-card");
        const box = card.querySelector(".molviewer-notices");
        const lines = card.querySelectorAll(".molviewer-notice")
            .map(n => ({ text: n.textContent,
                         classes: Array.from(n._classes).join(" ") }));
        console.log(JSON.stringify({ hidden: !!(box && box.hidden), lines }));
        """
    )
    assert out["lines"], (
        "the server said the box was unusable and the viewer showed nothing. "
        "A load reports rather than refuses, so this IS the whole of the check "
        "reaching the user"
    )
    assert not out["hidden"], "the message was drawn into a hidden box"
    assert out["lines"][0]["text"] == (
        "cell must be right-handed (det > 0); got det = -1."
    ), f"the sentence was reworded on the way to the screen: {out['lines'][0]}"
    assert "warn" in out["lines"][0]["classes"], (
        f"a warning was drawn as ordinary information: {out['lines'][0]}"
    )


def test_a_notice_is_drawn_where_its_subject_is_and_the_tab_says_so():
    """A message goes where it can be ACTED ON, not where it came from.

    A warning about the box names an axis and a clearance — those numbers are
    the Cell page's four rows, so that is where it belongs and where a user
    would go to change them. It used to be routed by ORIGIN instead: the same
    warning went above the atom list when it arrived with a file, and under the
    Cell rows when it arrived from a cell edit. Same sentence, same subject, two
    places.

    AND A PAGE YOU ARE NOT LOOKING AT STILL HAS TO REACH YOU. Loading happens
    from the Selection page, so putting the words on the Cell page alone would
    warn into a page nobody is on. The TAB carries a mark — visible from either
    page — and the words stay where they are useful. Opening the page clears it,
    because by then it has been seen.
    """
    out = _run(
        """
        globalThis.__nextNotices = [
            { level: "warn", about: "cell",
              message: "the box does NOT contain the structure along z." },
            { level: "info",
              message: "something about the structure as a whole." },
        ];
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const card = host.querySelector(".molviewer-card");

        // The stand-in understands tags, classes and descendant chains -- no
        // attribute selectors -- so the two pages are told apart by the
        // attribute the stylesheet keys on, read off the elements themselves.
        let cellPage = null;
        for (const pg of card.querySelectorAll(".molviewer-panel-tab")) {
            if (pg.getAttribute("data-page") === "cell") cellPage = pg;
        }
        const notesIn = (box) => box.querySelectorAll(".molviewer-notice")
            .map(n => n.textContent);
        const boxes = card.querySelectorAll(".molviewer-notices");
        const onCell = [], aboveTabs = [];
        for (const box of boxes) {
            (cellPage.contains(box) ? onCell : aboveTabs).push(...notesIn(box));
        }
        const tab = () => {
            for (const el of card.querySelectorAll(".molviewer-panel-tab-option")) {
                if (el.textContent.indexOf("Cell") >= 0) return el;
            }
            return null;
        };
        const marked = () => !!(tab() && tab().getAttribute("data-has-notices"));

        const before = { onCell, aboveTabs, tabMarked: marked() };
        // Open the Cell page: the mark has done its job.
        card.querySelectorAll(".molviewer-panel-tab-option input")[1]
            .dispatch("change", {});
        console.log(JSON.stringify({ before, markedAfterOpening: marked() }));
        """
    )
    assert out["before"]["onCell"] == [
        "the box does NOT contain the structure along z."
    ], (
        f"the box warning is not beside the box: {out['before']}. Routed by "
        f"where it came from, it lands above the atom list — nowhere near the "
        f"only control that can fix it"
    )
    assert out["before"]["aboveTabs"] == [
        "something about the structure as a whole."
    ], (
        f"a notice about no particular part was hidden on the Cell page, or the "
        f"cell warning was drawn twice: {out['before']}"
    )
    assert out["before"]["tabMarked"] is True, (
        "a warning was put on a page the user is not looking at with nothing to "
        "say it was there"
    )
    assert out["markedAfterOpening"] is False, (
        "the mark stayed after the page was opened, so it says nothing"
    )


def test_the_names_molview_offers_each_read_differently_and_frozen_stands_out():
    """MolView offers four names before anyone has used them — `L-electrode`,
    `R-electrode`, `bridge`, `interface` — because nearly every device structure
    needs them and typing one by hand is where `L-Electrode` comes from. They
    are SPELLINGS: MolView reads no meaning into any of them.

    Each wears its own colour for one reason: so you can tell them apart on a
    crowded atom list.

    `frozen_atoms` IS DIFFERENT, and looks it. The calculation acts on it — the
    atoms wearing it are held still — so it is the one a user is owed a warning
    about before tagging atoms with it by accident. "This changes your run" and
    "this is a name I picked" must not look the same.

    THE LIST IS MOLVIEW'S OWN (user decision, 2026-08-03). It was handed in at
    mount, so five pages would each have had to repeat the same four names —
    and only the module's demo page ever did, which is why every chip on every
    real page came out the same colour.
    """
    out = _run(
        """
        globalThis.__nextAtoms = [
            { index: 0, element: "C", x: 0, y: 0, z: 0,
              regions: ["frozen_atoms", "L-electrode", "bridge", "my-notes"] },
            { index: 1, element: "O", x: 1, y: 0, z: 0, regions: [] },
        ];
        const m = await mounted();
        await m.viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const chips = m.host.querySelector(".molviewer-card")
            .querySelectorAll(".molviewer-atoms-column-labels .molviewer-selection-tag")
            .map(t => ({ name: t.children[0].textContent,
                         classes: Array.from(t._classes).sort().join(" "),
                         title: t.title || null }));
        console.log(JSON.stringify({ chips }));
        """
    )
    chip = {c["name"]: c for c in out["chips"]}
    assert set(chip) == {"frozen_atoms", "L-electrode", "bridge", "my-notes"}, (
        "every label the atom carries must show, offered by MolView or not"
    )
    assert "molviewer-label-region" in chip["my-notes"]["classes"], (
        "a name the user invented must read as an ordinary label"
    )
    for name in ("L-electrode", "bridge", "frozen_atoms"):
        assert "molviewer-label-region" not in chip[name]["classes"], (
            f"{name} is one of the names MolView offers and reads as an "
            f"ordinary label — which is what it looked like when nobody handed "
            f"the list in: every chip the same colour"
        )
    assert chip["L-electrode"]["classes"] != chip["bridge"]["classes"], (
        "two of the offered names look identical, so the colour says only that "
        "they came from the list — the point is telling them apart"
    )
    assert "molviewer-label-frozen" in chip["frozen_atoms"]["classes"], (
        "frozen_atoms is the one the calculation acts on and must not look like "
        f"the conveniences beside it: {chip['frozen_atoms']['classes']}"
    )
    assert "held still" in (chip["frozen_atoms"]["title"] or ""), (
        "the chip must say what the name does, so a user can find out without "
        "leaving the panel"
    )


def test_the_labels_in_play_are_offered_so_none_has_to_be_retyped():
    """A label is offered, never retyped — a typo while retyping makes a SECOND
    label that looks like the first, which is a whole extra region as far as
    anything downstream is concerned.

    TWO SOURCES, ONE LIST. The five predefined names are offered before anyone
    has used them, because nearly every device structure needs them and typing
    `L-electrode` by hand is where `L-Electrode` comes from. Everything the
    structure itself carries is offered beside them, read from `getRegions` and
    never kept (§ 5.2), so the list follows the structure with nothing to keep in
    step. A predefined name is a SPELLING, not a meaning: MolView assigns none of
    them (§ 6.6).
    """
    out = _run(
        """
        globalThis.__nextAtoms = [
            { index: 0, element: "C", x: 0, y: 0, z: 0,
              regions: ["bridge", "frozen_atoms"] },
            { index: 1, element: "O", x: 1, y: 0, z: 0, regions: ["bridge"] },
        ];
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const card = host.querySelector(".molviewer-card");
        const chooser = card.querySelector(".molviewer-selection-assign-select");
        const box = card.querySelector(".molviewer-selection-new-label");
        const options = () => chooser.children.map(o => o.value);

        const offered = options();
        // Choosing an existing one hides the typing box; it is not needed.
        chooser.value = "bridge";
        chooser.dispatch("change", { target: chooser });
        const boxHiddenForExisting = box.hidden;

        // Applying uses the CHOSEN name, with nothing typed anywhere.
        viewer.data.selection.adopt([1]);
        card.querySelectorAll(".molviewer-selection-assign-btn")[0].click();
        const afterAssign = viewer.data.getRegions();

        // A brand-new name still needs the box, so choosing "+ new" shows it.
        chooser.value = chooser.children[chooser.children.length - 1].value;
        chooser.dispatch("change", { target: chooser });
        const boxShownForNew = !box.hidden;
        box.value = "interface";
        viewer.data.selection.adopt([0]);
        card.querySelectorAll(".molviewer-selection-assign-btn")[0].click();

        console.log(JSON.stringify({
            offered, boxHiddenForExisting, boxShownForNew,
            afterAssign,
            nowOffered: options(),
            regions: viewer.data.getRegions(),
        }));
        """
    )
    assert out["offered"][:6] == ["L-electrode", "R-electrode", "bridge",
                                  "interface", "buffer",
                                  "frozen_atoms"], (
        "the six predefined names come first, in the order the device reads "
        "(buffer joined with the transport composite -- TS.Atoms.Buffer is "
        f"a real region the emitter consumes): {out['offered']}"
    )
    assert out["offered"][-1] == "", (
        f"the last entry is the way to name a new label: {out['offered']}"
    )
    # `bridge` and `frozen_atoms` are BOTH carried here and predefined -- offered
    # once, not twice, or the same name would appear as two choices.
    assert out["offered"].count("bridge") == 1, out["offered"]
    assert out["offered"].count("frozen_atoms") == 1, (
        "the reserved label is offered like any other, and only once — it is an "
        f"ordinary label (§ 6.6): {out['offered']}"
    )
    assert len(out["offered"]) == 7, (
        f"six predefined + nothing else carried + a way to name a new one: {out['offered']}"
    )
    assert out["boxHiddenForExisting"] is True, (
        "the typing box is in the way once an existing label is chosen"
    )
    assert out["afterAssign"]["bridge"] == [1], (
        "applying used the chosen label with nothing typed — and replace means "
        f"the label's set BECOMES the selection (§ 9.5): {out['afterAssign']}"
    )
    assert out["boxShownForNew"] is True, "a new name still needs somewhere to type"
    assert out["regions"]["interface"] == [0], (
        f"the typed name was not applied: {out['regions']}"
    )
    assert "interface" in out["nowOffered"], (
        "a label just created is not offered next time, so it would have to be "
        f"retyped after all: {out['nowOffered']}"
    )


def test_typing_in_a_filter_row_does_not_replace_the_control_being_typed_in():
    """§ 8.4: the filter "is edited a row at a time... A surface that only
    accepted the whole set of rows at once would make the panel rebuild and
    re-send state it was in the middle of editing."

    The store took one row at a time exactly as that says — and the panel
    redrew the whole set on every snapshot anyway, so each keystroke emptied the
    container and built fresh rows. The input the user was typing in was
    destroyed and replaced between characters, taking the caret with it: typing
    a two-letter element meant clicking back into the box in between.

    ELEMENT IDENTITY is the assertion. Focus lives in the browser, but "is the
    thing I was typing in still the thing that is there" is the same question and
    it is answerable here. What must survive is the node; what must change is
    only the store.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.selection.setEditor("filter");
        viewer.data.selection.addFilter();
        const card = host.querySelector(".molviewer-card");

        const typedInto = card.querySelector(".molviewer-filter-text");
        const kindSelect = card.querySelector(".molviewer-filter-kind");

        // Type, one character at a time, the way a user does.
        const survived = [];
        for (const text of ["A", "Au", "Au,"]) {
            typedInto.value = text;
            typedInto.dispatch("input", { target: typedInto });
            survived.push(card.querySelector(".molviewer-filter-text") === typedInto);
        }

        // Changing the kind is also a change to a row, not to the set.
        kindSelect.value = "by_element";
        kindSelect.dispatch("change", { target: kindSelect });
        const survivedKind =
            card.querySelector(".molviewer-filter-text") === typedInto;

        // Selecting an atom redraws the panel from the same snapshot — and must
        // not take the row out from under the typing either.
        viewer.data.pickAtom(0);
        const survivedSelection =
            card.querySelector(".molviewer-filter-text") === typedInto;

        // But ADDING a row is a change to the SET, so the rows are rebuilt.
        viewer.data.selection.addFilter();
        const rowsNow = card.querySelectorAll(".molviewer-filter-row").length;

        console.log(JSON.stringify({
            survived, survivedKind, survivedSelection, rowsNow,
            stored: viewer.data.selection.getState().filters[0],
        }));
        """
    )
    assert out["survived"] == [True, True, True], (
        "the input was replaced between keystrokes, so the caret and the focus "
        f"went with it: {out['survived']}"
    )
    assert out["survivedKind"] is True, (
        "changing a row's kind rebuilt the row being typed in"
    )
    assert out["survivedSelection"] is True, (
        "clicking an atom rebuilt the filter row being typed in — the panel is "
        "redrawing the whole set because something unrelated changed"
    )
    assert out["stored"] == {"kind": "by_element", "value": "Au,"}, (
        f"what was typed did not reach the store: {out['stored']}"
    )
    assert out["rowsNow"] == 2, (
        "adding a row IS a change to the set and must rebuild them"
    )


def test_by_label_offers_the_defined_names_and_nothing_else():
    """§ 9.5: what is worth offering is read from the structure — and for a label
    rule, what is worth offering is the whole of what can match.

    A label that is not defined matches no atom, so a free-text box on this rule
    can only ever produce an empty selection and a user wondering why. It is also
    the second place a name gets retyped into a near-duplicate, after the Assign
    chooser. So `by label` CHOOSES; the other three rules still type, because an
    element symbol, an index range and a residue name are typed, not chosen.
    """
    out = _run(
        """
        globalThis.__nextAtoms = [
            { index: 0, element: "C", x: 0, y: 0, z: 0, regions: ["custom-tag"] },
            { index: 1, element: "O", x: 1, y: 0, z: 0, regions: [] },
        ];
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.selection.setEditor("filter");
        const card = host.querySelector(".molviewer-card");
        card.querySelector(".molviewer-selection-add-filter-row").click();

        const kindOf = () => card.querySelector(".molviewer-filter-kind");
        const valueOf = () => card.querySelector(".molviewer-filter-text");

        // A fresh row is `by_element`: typed, not chosen.
        const asElement = valueOf().tagName;

        // Type an atom range FIRST, then re-kind: the value must not follow.
        const typed = valueOf();
        typed.value = "3-7";
        typed.dispatch("input", { target: typed });

        kindOf().value = "by_label";
        kindOf().dispatch("change", { target: kindOf() });
        const asLabel = valueOf().tagName;
        const offered = valueOf().children.map(o => o.value);
        const shownAfterReKind = valueOf().value;
        const storedAfterReKind = viewer.data.selection.getState().filters[0].value;

        // Choosing one reaches the store like any other row edit.
        const box = valueOf();
        box.value = "custom-tag";
        box.dispatch("change", { target: box });
        const stored = viewer.data.selection.getState().filters[0];

        // Back to a typed rule: the control swaps back.
        kindOf().value = "by_residue";
        kindOf().dispatch("change", { target: kindOf() });
        const backToTyped = valueOf().tagName;

        console.log(JSON.stringify({
            asElement, asLabel, backToTyped, offered, stored,
            shownAfterReKind, storedAfterReKind,
        }));
        """
    )
    assert out["asElement"] == "INPUT", (
        f"`by element` is typed, not chosen: {out['asElement']}"
    )
    assert out["asLabel"] == "SELECT", (
        "`by label` must offer the defined names rather than a free-text box: "
        f"{out['asLabel']}"
    )
    assert out["backToTyped"] == "INPUT", (
        "re-kinding back to a typed rule must swap the control back — a chooser "
        f"left behind is a rule the user cannot express: {out['backToTyped']}"
    )
    assert out["storedAfterReKind"] == "", (
        f"the atom range survived the change of rule: "
        f"{out['storedAfterReKind']!r}. A value belongs to its kind — \"3-7\" "
        f"is not a label, and carrying it across left the row saying something "
        f"its new rule cannot mean"
    )
    assert out["shownAfterReKind"] == "", (
        f"the chooser opened on something instead of nothing: "
        f"{out['shownAfterReKind']!r}"
    )
    assert out["offered"][0] == "", (
        f"a row nobody has filled in must say so, not pick the first label on "
        f"the user's behalf: {out['offered']}"
    )
    assert "3-7" not in out["offered"], (
        f"the leftover atom range was offered as though somebody had defined a "
        f"label called it: {out['offered']}"
    )
    assert out["offered"][1:7] == ["L-electrode", "R-electrode", "bridge",
                                   "interface", "buffer",
                                   "frozen_atoms"], (
        f"the six names MolView offers come first: {out['offered']}"
    )
    assert "custom-tag" in out["offered"], (
        "a label the structure carries must be offered beside the predefined "
        f"ones: {out['offered']}"
    )
    assert out["stored"]["value"] == "custom-tag", (
        f"choosing a name must reach the store like any other row edit: {out['stored']}"
    )


def test_a_filter_row_is_added_typed_and_removed_from_the_panel():
    """§ 8.4: "a user adds a row, types in it, changes its kind, removes it" —
    each its own change, because that is what the controls are.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.selection.setEditor("filter");
        const card = host.querySelector(".molviewer-card");

        const empty = card.querySelector(".molviewer-filter-empty") !== null;
        card.querySelector(".molviewer-selection-add-filter-row").click();
        const afterAdd = viewer.data.selection.getState().filters;

        const text = card.querySelector(".molviewer-filter-text");
        text.value = "Au";
        text.dispatch("input", { target: text });
        const afterType = viewer.data.selection.getState().filters;

        card.querySelector(".molviewer-filter-remove").click();
        console.log(JSON.stringify({
            empty, afterAdd, afterType,
            afterRemove: viewer.data.selection.getState().filters,
        }));
        """
    )
    assert out["empty"] is True, "no rows yet means the panel says so"
    assert out["afterAdd"] == [{"kind": "by_element", "value": ""}]
    assert out["afterType"] == [{"kind": "by_element", "value": "Au"}], (
        f"typing must reach the store a row at a time: {out['afterType']}"
    )
    assert out["afterRemove"] == []


def test_the_view_menu_holds_all_four_drawing_settings():
    """§ 1.1: "The View menu holds style (stick, ball & stick, sphere, line), a
    radius slider from 0.2 to 2.5 that scales stick thickness / sphere size /
    line width, and a background colour with preset swatches plus a picker. One
    preset is transparent — choose it before exporting a picture to drop onto a
    slide." Plus Perspective / Orthographic.

    All four are § 9.6's `view` — settings that change how the same frame is
    PAINTED — so each control writes there and reads its state back from there
    (§ 8.5), never from what it last did.

    The radius and the background lived in the store with no control that could
    write them: two of the four settings the menu promises were unreachable, and
    the stylesheet had been carrying their design the whole time.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const view = viewer.data.view;

        // § 1.1's range, read off the control itself.
        const slider = card.querySelector(".molviewer-menu-radius-row input");
        const spec = { min: slider.min, max: slider.max, step: slider.step };
        const startedAt = slider.value;

        // Drive it the way a user does.
        slider.value = "2.5";
        slider.dispatch("input", { target: slider });
        const afterDrag = view.get().radius;
        const shownAfterDrag = card.querySelector(".molviewer-menu-radius-row output")
                                   .textContent;

        // Set it from ELSEWHERE: the control must follow the store, not itself.
        view.set("radius", 0.5);
        const followed = { value: slider.value,
                           shown: card.querySelector(
                               ".molviewer-menu-radius-row output").textContent };

        // The background presets, and the one that is transparent.
        const swatches = card.querySelectorAll(".molviewer-menu-background-swatch");
        const transparent = card.querySelectorAll(
            ".molviewer-menu-background-swatch.molviewer-is-transparent");
        const picker = card.querySelectorAll(".molviewer-menu-background-custom input");

        // Nothing is lit until the user chooses: `background: null` is "the
        // window's own ground", not a colour (§ 9.6).
        const litAtStart = card.querySelectorAll(
            ".molviewer-menu-background-swatch.molviewer-is-active").length;

        transparent[0].click();
        const afterSwatch = view.get().background;
        const litAfter = card.querySelectorAll(
            ".molviewer-menu-background-swatch.molviewer-is-active").length;

        console.log(JSON.stringify({
            spec, startedAt, afterDrag, shownAfterDrag, followed,
            swatches: swatches.length, transparent: transparent.length,
            picker: picker.length, litAtStart, afterSwatch, litAfter,
        }));
        """
    )
    assert out["spec"] == {"min": "0.2", "max": "2.5", "step": "0.05"}, (
        f"§ 1.1 fixes the radius range at 0.2 to 2.5: {out['spec']}"
    )
    assert out["startedAt"] == "1", "the slider must open on the store's default"
    assert out["afterDrag"] == 2.5, (
        f"dragging the slider did not reach `view`: {out['afterDrag']}"
    )
    assert out["shownAfterDrag"] == "2.50"
    assert out["followed"] == {"value": "0.5", "shown": "0.50"}, (
        "the slider did not follow a radius set from elsewhere — it is holding "
        f"its own answer rather than reading the store (§ 8.5): {out['followed']}"
    )
    assert out["swatches"] == 3 and out["transparent"] == 1, (
        f"§ 1.1 asks for preset swatches, one of them transparent: {out}"
    )
    assert out["picker"] == 1, "§ 1.1 asks for a picker beside the presets"
    assert out["litAtStart"] == 0, (
        "a swatch was lit before the user chose one — `background: null` means "
        "the window's own ground, which is not one of the presets (§ 9.6)"
    )
    assert out["afterSwatch"] == "transparent", (
        f"clicking a swatch did not reach `view`: {out['afterSwatch']}"
    )
    assert out["litAfter"] == 1, "the chosen swatch must light, and only it"


def test_the_view_menu_opens_showing_what_the_store_already_says():
    """§ 8.5: "None of them holds a fact of its own."

    `view.subscribe` hands nothing over when you subscribe — unlike
    `selection.subscribe`, which fires immediately — so a view subscriber has to
    take its own first pass. The menu used to take it for the RADIUS ALONE, and
    style, background and projection opened showing hand-written initial markup
    instead of the store.

    That was visible, not theoretical: the store says `style: "stick"` from the
    first moment, and NO style button was lit until the user changed something.
    The menu opened claiming nothing was selected while the drawing was already
    drawing sticks.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const lit = card.querySelectorAll(".molviewer-menu-style-btn.molviewer-is-active");

        console.log(JSON.stringify({
            style:       viewer.data.view.get().style,
            litCount:    lit.length,
            litLabel:    lit.length === 1 ? lit[0].textContent : null,
            radiusShown: card.querySelector(".molviewer-menu-radius-row input").value,
            projection:  card.querySelector(".molviewer-menu-projection")
                             .getAttribute("aria-pressed"),
        }));
        """
    )
    assert out["litCount"] == 1, (
        "the View menu opened with "
        f"{out['litCount']} style buttons lit. The store already says "
        f"style={out['style']!r} — a control that shows nothing until the user "
        "touches it is holding its own answer instead of reading `view` (§ 8.5)"
    )
    assert out["litLabel"] == "Sticks", (
        f"the lit style must be the store's, not the first in the list: {out}"
    )
    assert out["radiusShown"] == "1", "the slider must open on the store's radius"
    assert out["projection"] == "false", (
        "the projection toggle must open showing the store's `orthographic`, "
        f"not a hand-written attribute: {out['projection']}"
    )


def test_the_projection_toggle_reads_the_store_not_its_own_attribute():
    """§ 8.5: every control reads its state back from `view`, "never from what
    it last did".

    The toggle used to compute its next value from its own `aria-pressed`, and
    that is worse than untidy because `view.set` DEDUPES — `if (settings[name]
    === value) return`. Let the attribute drift from the store once and every
    click computes the same stale value, sets what the store already holds,
    fires nothing and repaints nothing. The control is dead for good, and
    silently: no error, no visible change, just a button that stops working.

    The drift is forced here rather than waited for. Writing the attribute IS
    the test — if the DOM were where the setting lived, that write would matter,
    and it must not.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const view = viewer.data.view;
        const toggle = card.querySelector(".molviewer-menu-projection");
        const readBack = () => toggle.getAttribute("aria-pressed");

        toggle.click();
        const afterOne = { store: view.get().orthographic, shown: readBack() };

        // Set from ELSEWHERE: the control must follow the store.
        view.set("orthographic", false);
        const followed = readBack();

        // Now make the ATTRIBUTE lie, and click. The store says false; the
        // attribute claims true. A control that reads itself would set false,
        // hit the dedupe, fire nothing, and stick here forever.
        view.set("orthographic", true);
        toggle.setAttribute("aria-pressed", "false");
        toggle.click();
        const afterLie = { store: view.get().orthographic, shown: readBack() };

        console.log(JSON.stringify({ afterOne, followed, afterLie }));
        """
    )
    assert out["afterOne"] == {"store": True, "shown": "true"}, (
        f"one click must set `orthographic` and light the button: {out['afterOne']}"
    )
    assert out["followed"] == "false", (
        "the toggle did not follow a projection set from elsewhere — it is "
        f"holding its own answer rather than reading `view` (§ 8.5): {out['followed']}"
    )
    assert out["afterLie"] == {"store": False, "shown": "false"}, (
        "with the store at `true` and the attribute lying `false`, the click "
        "must still flip the STORE to false. It did not, which means the next "
        "value was computed from the DOM: `view.set` then deduped the write, "
        f"nothing fired, and the toggle is now stuck: {out['afterLie']}"
    )


def test_the_speed_box_sets_playback_and_shows_what_playback_took():
    """§ 8.5: "None of them holds a fact of its own." § 9.2: the handle exposes
    what a viewer DOES, not the knobs it does it with.

    Speed used to live in the `<input>` and nowhere else, with `mount.js`
    keeping a private `fps` it would accept through `play({fps})`. Two partial
    homes, neither authoritative — and they disagreed: the box displayed
    § 1.1's 150 ms while the timer started at `DEFAULT_FPS = 12`, 83 ms.

    Now the handle owns it in the module's own unit, and the box is a control
    over it like any other: it sets, then shows what was taken.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        const card = host.querySelector(".molviewer-card");
        const box = card.querySelector(".molviewer-frames-speed-input");
        const type = (v) => { box.value = String(v); box.dispatch("change", { target: box }); };

        const opened = { shown: box.value, held: viewer.getSpeed(),
                         min: box.min, max: box.max };

        type(500);
        const afterTyping = { shown: box.value, held: viewer.getSpeed() };

        // The clamp is the HANDLE's, so it holds for every caller — and the box
        // settles to what playback actually took, not to what was typed.
        type(99999);
        const clampedHigh = { shown: box.value, held: viewer.getSpeed() };
        type(1);
        const clampedLow = { shown: box.value, held: viewer.getSpeed() };

        // Nonsense leaves the speed alone rather than stopping the timer dead.
        viewer.setSpeed("not a number");
        const afterNonsense = viewer.getSpeed();

        // The knob is gone: play() takes no arguments.
        const playArity = viewer.play.length;

        console.log(JSON.stringify({
            opened, afterTyping, clampedHigh, clampedLow, afterNonsense, playArity,
        }));
        """
    )
    assert out["opened"] == {"shown": "150", "held": 150, "min": "20", "max": "3000"}, (
        "the box must open on what playback is set to, within § 1.1's range — "
        f"not on a number written into the control: {out['opened']}"
    )
    assert out["afterTyping"] == {"shown": "500", "held": 500}, (
        f"typing a speed must reach the handle: {out['afterTyping']}"
    )
    assert out["clampedHigh"] == {"shown": "3000", "held": 3000}, (
        "§ 1.1 caps the speed at 3000 ms and the HANDLE must enforce it, then "
        f"the box must show what was actually taken: {out['clampedHigh']}"
    )
    assert out["clampedLow"] == {"shown": "20", "held": 20}, (
        f"§ 1.1 floors the speed at 20 ms: {out['clampedLow']}"
    )
    assert out["afterNonsense"] == 20, (
        f"nonsense must leave the speed alone, not zero the timer: {out['afterNonsense']}"
    )
    assert out["playArity"] == 0, (
        "`play` must take no arguments — `play({fps})` published the timer's own "
        f"parameter and let a caller run at a speed the bar never showed: {out['playArity']}"
    )


def test_the_slowest_setting_really_is_the_slowest_setting():
    """§ 1.1 offers 20–3000 ms per frame. The top of that range did not work.

    Speed was carried as frames per second, and the guard against a zero rate
    — `Math.max(1, fps)` — also capped the SLOW end at 1 fps. 3000 ms is
    0.33 fps, so the contract's slowest setting played at 1000 ms: three times
    too fast, with nothing on screen to say so. Carrying milliseconds instead
    removes the division and the floor together.

    1500 ms over a 1200 ms window separates the two cleanly and without a
    stopwatch's flakiness: correctly, the interval has not elapsed even once,
    so the frame CANNOT have moved. Under the old cap it fires at 1000 ms and
    moves once. `setInterval` never fires early, so a loaded machine can only
    make the count lower — and it is already zero.
    """
    out = _run(
        """
        const { viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.reloadFrames([[[0,0,0],[1,0,0]], [[2,0,0],[3,0,0]]]);

        let moved = 0;
        viewer.data.onFrameChange(() => { moved += 1; });
        viewer.setSpeed(1500);
        viewer.play();
        await new Promise(r => setTimeout(r, 1200));
        viewer.pause();

        console.log(JSON.stringify({ moved, held: viewer.getSpeed() }));
        """
    )
    assert out["held"] == 1500, f"1500 ms is inside § 1.1's range: {out['held']}"
    assert out["moved"] == 0, (
        f"the frame moved {out['moved']} time(s) in 1200 ms at a setting of "
        "1500 ms per frame — it cannot have, unless playback is running faster "
        "than it was told. That is the fps floor capping the slow end at 1000 ms"
    )


def test_a_read_only_viewer_hides_the_control_the_gate_would_swallow():
    """§ 9.4: "A control that would do nothing should not be offered … MolView
    hides the ones it draws — the label box, the edit operations."

    "The gate is the contract; the hiding is courtesy, and it may never be the
    only thing standing between a read-only viewer and a changed structure" — so
    this test checks the hiding, and the model tests check the gate.
    """
    out = _run(
        """
        async function panelFor(mode) {
            const host = globalThis.__makeHost();
            const viewer = await MV.mount(host, workspace, { owner: "x", mode });
            await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
            const card = host.querySelector(".molviewer-card");
            return {
                assignHidden: card.querySelector(".molviewer-selection-assign").hidden,
                isolateOffered: card.querySelector(".molviewer-selection-mode-option") !== null,
            };
        }
        console.log(JSON.stringify({
            editable: await panelFor(undefined),
            readonly: await panelFor("readonly"),
        }));
        """
    )
    assert out["editable"]["assignHidden"] is False
    assert out["readonly"]["assignHidden"] is True, (
        "a read-only viewer must not offer the label box — the gate would "
        "swallow it, and a button that silently does nothing is a bad answer"
    )
    assert out["readonly"]["isolateOffered"] is True, (
        "isolate is NOT an edit (§ 9.4) — a read-only viewer isolates freely"
    )


def test_loading_a_structure_reaches_the_drawing():
    """The first attempt's worst bug, guarded.

    A rename left `mount` asking for a factory the module no longer published,
    and the guard around it turned a missing factory into a no-op — so EVERY
    VIEWER MOUNTED AND THEN NEVER DREW. It was invisible to every node test in
    the suite, because they all stubbed the engine.

    This is the cheapest check that the wiring is continuous: load a structure
    through the public door and assert the frames arrived at the bottom. It does
    not replace the browser (§ 13.2's third level) — nothing here proves anything
    LOOKS right — but it does prove the chain is joined.
    """
    out = _run(
        """
        globalThis.__resetCalls();
        const { viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        await waitForDrawing(() => globalThis.__callNames().includes("addModelsAsFrames"),
                    "addModelsAsFrames after a load");
        const afterLoad = globalThis.__callNames().slice();

        viewer.data.reloadFrames([[[0,0,0],[1,0,0]], [[5,0,0],[6,0,0]]]);
        viewer.data.setCurrentFrame(1);

        console.log(JSON.stringify({
            drewOnLoad: afterLoad.includes("addModelsAsFrames"),
            swapped: globalThis.__countCalls("setFrame") > 0,
            painted: globalThis.__countCalls("render") > 0,
            styled: globalThis.__countCalls("setStyle") > 0,
            fitted: globalThis.__countCalls("zoomTo") > 0,
        }));
        """
    )
    assert out["drewOnLoad"] is True, (
        "a structure was loaded and nothing reached the drawing — this is the "
        "mount-but-never-draw failure, and it is what every stubbed test misses"
    )
    assert out["styled"] is True, "the structure was drawn with no style applied"
    assert out["fitted"] is True, (
        "the camera must be fitted to the structure on load (§ 9.6)"
    )
    assert out["swapped"] is True, "scrubbing did not reach the drawing"
    assert out["painted"] is True, "nothing was ever painted"


def test_every_layer_is_reachable_from_the_entry_point():
    """The guard on every other source scan in the suite.

    Those scans ask a question of "the module", and the module is computed as
    what `index.js` reaches (``tests/_molview_sources.py``) rather than what the
    directory holds. That definition is right — § 4 says the entry point is the
    module — but it has one failure mode: if the walk resolved too little, every
    scan would pass by finding nothing, and the suite would go quiet exactly when
    it should shout.

    So this pins the two ends. Every file the plan's tree calls a layer must be
    reached, and the demo must NOT be — it is a consumer, and holding it to the
    module's internal rules would be holding the wrong side of the boundary.
    """
    layers = set(module_files())
    expected = {
        "index.js", "_atom.js", "3dmol-embed.js", "render-engine.js",
        "model.js", "model-jobs.js", "history.js", "stores.js",
        "mount.js", "ui.js",
    }
    missing = expected - layers
    assert missing == set(), (
        f"these layers are not reachable from the entry point, so every source "
        f"scan in the suite silently skips them: {sorted(missing)}"
    )
    assert "demo.js" not in layers, (
        "the demo is a consumer — it imports the entry point rather than being "
        "imported by it — so it is not a layer"
    )


# ---------------------------------------------------------------------------
# § 8.1–8.3 — the stylesheet is the design system, not a suggestion
# ---------------------------------------------------------------------------

def test_every_class_the_module_writes_is_one_the_stylesheet_defines():
    """The stylesheet IS MolView's design system (§ 8.1–8.3), so a class name the
    module invents is a control with no design — it falls through to the
    browser's defaults and looks like nothing else on the card.

    That is not a cosmetic problem. It happened because writing `class="…"` is
    the same four keystrokes whether the rule exists or not, so nothing pushes
    back at the moment of writing. This is what pushes back: the vocabulary is
    the stylesheet's, and the module may only use words it already has.

    It also catches the reverse of a real bug — a control styled by a rule that
    was renamed out from under it.
    """
    import re

    css = (MODULE_DIR / "molview.css").read_text()
    defined = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))

    invented = {}
    for name, path in module_files().items():
        code = path.read_text()
        used = set()
        for value in re.findall(r'el\((?:"[^"]*"|\w+),\s*"([^"]+)"\)', code):
            used.update(value.split())
        for value in re.findall(r'className\s*=\s*"([^"]+)"', code):
            used.update(value.split())
        for value in re.findall(r'classList\.(?:add|toggle)\("([^"]+)"', code):
            used.update(value.split())
        missing = sorted(c for c in used if c not in defined)
        if missing:
            invented[name] = missing
    assert invented == {}, (
        f"these controls are drawn with classes the stylesheet never defines, so "
        f"they have no design at all: {invented}"
    )


# ---------------------------------------------------------------------------
# § 9.2 — what leaves through the handle is narrower than the module
# ---------------------------------------------------------------------------

def test_the_handle_hands_out_surfaces_not_the_stores():
    """User, 2026-08-31: *"why would you expose all internal when there is no
    need for outside user to have them?"*

    `mount` returns `data`, and a tab is not the module.  The one door left out
    of each store is the one with a ROUTER: a pick is written by `pickAtom`,
    which decides measuring-vs-selecting, so `toggle` must not be reachable
    beside it.  (`measurement` never had a public `toggle`; `selection` did.)
    """
    out = _run("""
        const { viewer } = await mounted();
        console.log(JSON.stringify({
            selToggle:  typeof viewer.data.selection.toggle,
            measToggle: typeof viewer.data.measurement.toggle,
            measAdopt:  typeof viewer.data.measurement.adopt,
            measure:    Object.keys(viewer.data.measurement).sort(),
            // the doors a consumer was found to need still work
            hasGet:     typeof viewer.data.selection.get,
            hasSwitch:  typeof viewer.data.selection.setSwitch,
            hasViewSet: typeof viewer.data.view.set,
            // and everything else on the model still passes through
            hasApplyOp: typeof viewer.data.applyOp,
            hasInstall: typeof viewer.data.installMolecule,
            hasPick:    typeof viewer.data.pickAtom,
        }));
    """)
    assert out["selToggle"] == "undefined", (
        "`selection.toggle` is reachable from a tab — a caller can write a pick "
        "without `pickAtom`, so the measuring-vs-selecting rule never runs"
    )
    assert out["measToggle"] == "undefined"
    assert out["measAdopt"] == "undefined"
    assert out["measure"] == ["clear", "getState", "positions",
                              "requestPicking", "setActive", "subscribe"]
    for door in ("hasGet", "hasSwitch", "hasViewSet", "hasApplyOp",
                 "hasInstall", "hasPick"):
        assert out[door] == "function", f"{door} stopped passing through"


# ---------------------------------------------------------------------------
# § 11.6 — the ruler's marks carry the ORDER
# ---------------------------------------------------------------------------

def test_a_pick_reaches_the_drawing_as_a_mark_then_an_arrow():
    """The whole chain, end to end: a click goes through `pickAtom`, the engine
    derives the marks, and the sealed layer draws them — one pick a mark, two
    an arrow running first→second (§ 11.6).

    Two atoms, because that is what this harness's server stand-in returns
    whatever is installed; the three-pick chain is covered where the drawing
    layer is driven directly (`test_molview_3dmol_embed.py`).
    """
    out = _run("""
        const { viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        await waitForDrawing(() => globalThis.__lastCall("addModelsAsFrames"),
                             "the structure never reached the window");
        const frame = viewer.data.getFrameAllAtoms(viewer.data.currentFrame());
        viewer.data.measurement.setActive(true);

        globalThis.__resetCalls();
        viewer.data.pickAtom(1);
        const one = { arrows: globalThis.__countCalls("addArrow"),
                      marks:  globalThis.__countCalls("addSphere") };

        globalThis.__resetCalls();
        viewer.data.pickAtom(0);
        const arrow = globalThis.__lastCall("addArrow");
        console.log(JSON.stringify({
            one,
            twoArrows: globalThis.__countCalls("addArrow"),
            start: arrow ? arrow.args[0].start.x : null,
            end:   arrow ? arrow.args[0].end.x   : null,
            atom1x: frame[1][0], atom0x: frame[0][0],
        }));
    """)
    assert out["one"]["arrows"] == 0, "one pick has no direction to draw"
    assert out["one"]["marks"] >= 1, "...but it is marked, or the pick is invisible"
    assert out["twoArrows"] == 1, "the second pick makes it an arrow"
    assert (out["start"], out["end"]) == (out["atom1x"], out["atom0x"]), (
        f"the arrow runs {out['start']} -> {out['end']}, but atom 1 was picked "
        f"FIRST (x={out['atom1x']}) and atom 0 second (x={out['atom0x']}) — the "
        f"arrow must follow the click order, which is what it exists to show"
    )

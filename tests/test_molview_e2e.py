"""MolView in a real page — § 13.2's THIRD level, which this module has
never had.

    | Level | Runs | Derived from | Shows |
    | End to end | a real page | § 1.1 | what a user does: select, isolate,
      measure, scrub, play, export |

The first two levels run in node against stand-ins, and both were green while
six features were dead: the renderEngine was proven against a data source the
model does not resemble, and nothing anywhere put a browser in front of the
result. A level that cannot see a blank window cannot defend § 1.1.

So every assertion here is about what a USER can see: a control that is on
screen, a press that changes the picture, a number that reads correctly. Where
"the picture changed" is the claim, it is checked against the CANVAS PIXELS —
the only end-to-end evidence that a switch reached WebGL rather than merely
reaching a store.

The page under test is the in-repo demo (§ 13.4), which is MolView mounted
through its one public import and nothing else.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  A real server, on its own port, with no login                        #
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def molview_server():
    """The app on a free port in a daemon thread.

    ``config={}`` is the no-auth, no-TLS build — the same one the other e2e
    files use, so a developer's ``molbuilder.json`` (which may enable auth)
    cannot change what these tests are looking at. Port 0 lets the OS pick,
    so this never collides with a `molbuilder serve` already running.
    """
    from werkzeug.serving import make_server

    from molbuilder.web.app import create_app

    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def demo(page, molview_server):
    """The demo page with a structure loaded and the drawing settled."""
    page.goto(f"{molview_server}/molview-demo")
    page.wait_for_selector(".molview-card .molviewer-rail button")
    page.click("#demo-benzene")
    page.wait_for_function(
        "() => document.querySelectorAll('.molviewer-atoms-table tr').length === 12"
    )
    _settle(page)
    return page


def _settle(page):
    """Let the drawing library finish a frame before the pixels are read."""
    page.wait_for_timeout(250)


def _canvas_pixels(page):
    """What is actually painted, as bytes.

    The sealed layer offers no way to ask what it drew (§ 9.9) and that is
    correct — so the honest end-to-end question is the one a user asks by
    looking: did the picture change?
    """
    return page.evaluate(
        """() => {
            const canvas = document.querySelector(".molviewer-window-canvas canvas");
            return canvas ? canvas.toDataURL().length : 0;
        }"""
    )


# --------------------------------------------------------------------- #
#  § 1.1 — the toolbar switches                                         #
# --------------------------------------------------------------------- #

def test_the_rail_is_six_buttons_beside_the_window_not_over_it(demo):
    """§ 1.1: "Six icon buttons sit down the left edge, always outside the
    canvas, never on top of the molecule."

    Both halves are the design: one press each, visible without opening
    anything, and never covering what the user is looking at.
    """
    buttons = demo.locator(".molviewer-rail button")
    assert buttons.count() == 6

    glyphs = [buttons.nth(i).inner_text() for i in range(6)]
    assert glyphs == ["⟲", "✚", "#", "➤", "▦", "◉"], glyphs

    rail = demo.locator(".molviewer-rail").bounding_box()
    canvas = demo.locator(".molviewer-window-canvas").bounding_box()
    assert rail["x"] + rail["width"] <= canvas["x"] + 1, (
        f"the rail overlaps the 3D window: rail ends at "
        f"{rail['x'] + rail['width']}, canvas starts at {canvas['x']}"
    )
    for i in range(6):
        assert buttons.nth(i).is_visible()


@pytest.mark.parametrize(
    "index,name",
    [(1, "Show axes"), (2, "Show atom labels")],
)
def test_a_switch_changes_the_picture(demo, index, name):
    """§ 1.1 through § 10.5: pressing a switch draws something.

    This is the test the suite could not have: at the two node levels the
    switches wrote a store correctly, the store fired correctly, and the
    drawing never heard about any of it. The only place that is visible is a
    page — so the assertion is the canvas itself, before and after.
    """
    before = _canvas_pixels(demo)
    assert before > 0, "nothing is painted at all"

    button = demo.locator(".molviewer-rail button").nth(index)
    assert button.get_attribute("aria-pressed") == "false"
    button.click()
    _settle(demo)

    assert button.get_attribute("aria-pressed") == "true", f"{name} did not light"
    assert _canvas_pixels(demo) != before, (
        f"{name} lit its button and changed nothing on screen"
    )


def test_showing_the_unit_cell_draws_the_box_the_structure_uses(demo):
    """§ 1.1's `▦` switch, in a real page: pressing it draws the cell.

    § 10.3: "The visibility switch carries ONLY a boolean: show it or don't. It
    never carries geometry" — the geometry travels on its own, unconditionally,
    so it is already there when the switch goes on. And § 9.3: the cell a reader
    is given is the one "as it will actually be used... so it always has an
    answer", which means EVERY structure has a box to show, not only one that was
    handed an explicit lattice.

    THIS TEST USED TO ASSERT THE OPPOSITE — that the canvas does not change — and
    it passed, because the box was derived from the raw `cell` field, which is
    null for every structure nobody gave an explicit lattice to. It then reasoned
    from its own green run that the demo has no periodic sample. The drawing was
    wrong and the test agreed with it, so "Show unit cell" did nothing at all and
    nothing said so.
    """
    before = _canvas_pixels(demo)
    button = demo.locator(".molviewer-rail button").nth(4)
    button.click()
    _settle(demo)

    assert button.get_attribute("aria-pressed") == "true"
    assert _canvas_pixels(demo) != before, (
        "pressing 'Show unit cell' changed nothing on screen — the box the "
        "structure actually uses was not drawn"
    )
    assert demo.locator(".molview-card").is_visible(), "the switch broke the card"

    # And off again: a boolean both ways, with the geometry untouched underneath.
    button.click()
    _settle(demo)
    assert button.get_attribute("aria-pressed") == "false"
    assert _canvas_pixels(demo) == before, (
        "turning the cell off left something of the box behind"
    )


def test_selecting_an_atom_draws_a_highlight(demo):
    """§ 1.1: "Click an atom and a soft amber sphere appears over it."

    Selected from the PANEL, because the panel and the window select into one
    store (§ 9.5) — and because a click in the window would be testing 3Dmol's
    hit detection rather than MolView's.
    """
    before = _canvas_pixels(demo)
    demo.locator(".molviewer-atoms-table tr").first.click()
    _settle(demo)

    assert demo.locator(".molviewer-selection-count").inner_text().startswith("1 of 12")
    assert _canvas_pixels(demo) != before, (
        "an atom was selected and the drawing did not change — the highlight "
        "never reached the window"
    )


def test_isolate_hides_the_rest_and_gives_them_back(demo):
    """§ 6.3: under isolate the drawing holds only the selection, and "the whole
    structure comes back the moment isolate is turned off" — because the master
    copy was never cut down.

    § 1.1: isolate turns itself off when the selection empties.
    """
    demo.locator(".molviewer-atoms-table tr").first.click()
    _settle(demo)
    whole = _canvas_pixels(demo)

    isolate = demo.locator(".molviewer-rail button").nth(5)
    isolate.click()
    _settle(demo)
    isolated = _canvas_pixels(demo)
    assert isolated != whole, "isolate drew the same picture"

    isolate.click()
    _settle(demo)
    assert _canvas_pixels(demo) != isolated, "the structure did not come back"

    # Clearing the selection with isolate on turns it off rather than leaving a
    # viewer that hides everything it has.
    isolate.click()
    demo.locator(".molviewer-selection-clear-btn").click()
    _settle(demo)
    assert isolate.get_attribute("aria-pressed") == "false", (
        "isolate stayed on with nothing selected"
    )


# --------------------------------------------------------------------- #
#  § 1.1 — measuring                                                    #
# --------------------------------------------------------------------- #

def test_measuring_reads_one_two_and_three_atoms(demo):
    """§ 1.1: one atom gives its coordinates, two give a distance, three give an
    angle — and § 11.6 puts the vertex at the atom picked SECOND.

    Benzene's ring carbons are 1.396 Å from the centre and 120° apart, so the
    numbers are checkable rather than merely present.
    """
    rows = demo.locator(".molviewer-atoms-table tr")
    readout = demo.locator(".molview-overlay--info")

    rows.nth(0).click()
    _settle(demo)
    assert readout.is_visible()
    assert readout.inner_text().startswith("#1"), readout.inner_text()

    rows.nth(1).click()
    _settle(demo)
    assert readout.inner_text().endswith("Å"), readout.inner_text()

    rows.nth(2).click()
    _settle(demo)
    text = readout.inner_text()
    assert text.endswith("°"), text
    # Picked 1 → 2 → 3, so atom 2 is the vertex: the interior angle of the ring.
    assert text.startswith("120."), f"the vertex is not the atom picked second: {text}"


# --------------------------------------------------------------------- #
#  § 1.1 — the menus, the panel, playing a trajectory                   #
# --------------------------------------------------------------------- #

def test_the_view_menu_opens_onto_something(demo):
    """§ 8.5: the menus are controls MolView draws, and a control that opens onto
    nothing is not one.

    The popover is fixed to the viewport and placed by script; when that
    placement was missing the menu opened, took its open state, and sat at
    -9999px. Nothing in node could see it, and this is what does.
    """
    body = demo.locator(".molviewer-menu-body").first
    assert not body.is_visible()

    demo.locator(".molviewer-menu summary").first.click()
    _settle(demo)

    assert body.is_visible()
    box = body.bounding_box()
    viewport = demo.viewport_size
    assert box["x"] >= 0 and box["y"] >= 0, f"the menu opened off-screen: {box}"
    assert box["x"] + box["width"] <= viewport["width"], f"clipped right: {box}"
    assert box["height"] > 0 and box["width"] > 0

    trigger = demo.locator(".molviewer-menu summary").first.bounding_box()
    assert box["y"] >= trigger["y"], "the menu is not below its own trigger"

    # A click elsewhere closes it.
    demo.locator(".molviewer-selection-count").click()
    _settle(demo)
    assert not body.is_visible()


def test_the_panel_switches_pages_without_resizing_the_card(demo):
    """§ 8.1: "Switching pages never resizes the card." The panel is given the
    window's extent, so its height cannot depend on what is inside it (§ 8.2).
    """
    before = demo.locator(".molview-card").bounding_box()
    pages = demo.locator(".molviewer-panel-tab")

    demo.locator(".molviewer-panel-tab-option").nth(1).click()      # Cell
    _settle(demo)
    assert pages.nth(1).is_visible() and not pages.nth(0).is_visible()

    after = demo.locator(".molview-card").bounding_box()
    assert (before["width"], before["height"]) == (after["width"], after["height"]), (
        f"switching pages resized the card: {before} -> {after}"
    )

    demo.locator(".molviewer-panel-tab-option").nth(0).click()      # Selection
    _settle(demo)
    assert pages.nth(0).is_visible()


def test_the_panel_bottom_aligns_with_the_window(demo):
    """§ 8.2: the panel is not measured and is not told a height by script — it
    is given the SAME extent the square is, "so the two bottom-align at every
    width with no JavaScript and no fixed number anywhere".

    A layout that computed one from the other would be a second place the same
    fact lives, and it would be wrong for one frame after every resize.
    """
    window = demo.locator(".molview-viewer").bounding_box()
    panel = demo.locator(".molview-panel").bounding_box()
    assert abs((window["y"] + window["height"]) - (panel["y"] + panel["height"])) <= 1, (
        f"window ends at {window['y'] + window['height']}, "
        f"panel at {panel['y'] + panel['height']}"
    )
    assert abs(window["width"] - window["height"]) <= 1, (
        f"the 3D window is not a square: {window}"
    )


def test_a_trajectory_gets_a_frame_bar_that_scrubs(demo):
    """§ 1.1: "When a structure has more than one frame, a playback bar appears
    under the viewer… a slider with an `i / N` counter. One frame shows no bar."

    § 6.4: every control reads the frame from the model, so the counter follows
    whatever moved it.
    """
    bar = demo.locator(".molview-frame-controls")
    assert not bar.is_visible(), "a single structure showed a playback bar"

    demo.click("#demo-trajectory")
    demo.wait_for_function(
        "() => document.querySelector('.molviewer-frames-counter')"
        " && document.querySelector('.molviewer-frames-counter').textContent.includes('/')"
    )
    _settle(demo)

    assert bar.is_visible()
    assert demo.locator(".molviewer-frames-counter").inner_text() == "1 / 6"

    painted = _canvas_pixels(demo)
    demo.locator(".molviewer-frames-step").nth(1).click()               # ›
    _settle(demo)
    assert demo.locator(".molviewer-frames-counter").inner_text() == "2 / 6"
    assert _canvas_pixels(demo) != painted, "the frame moved and the picture did not"


def test_playing_runs_the_movie_and_pauses(demo):
    """§ 1.1's play-pause, and § 10.4: playing is a frame swap at the library's
    own speed, not a re-render.
    """
    demo.click("#demo-trajectory")
    demo.wait_for_function(
        "() => document.querySelector('.molviewer-frames-counter')"
        " && document.querySelector('.molviewer-frames-counter').textContent.includes('/')"
    )
    play = demo.locator(".molviewer-frames-play")
    play.click()
    demo.wait_for_function(
        "() => document.querySelector('.molviewer-frames-counter').textContent !== '1 / 6'"
    )
    play.click()                                            # pause
    _settle(demo)
    stopped = demo.locator(".molviewer-frames-counter").inner_text()
    demo.wait_for_timeout(400)
    assert demo.locator(".molviewer-frames-counter").inner_text() == stopped, (
        "pausing did not stop the movie"
    )


# --------------------------------------------------------------------- #
#  § 8.1 — the card is dark, and it is the module's own                 #
# --------------------------------------------------------------------- #

def test_the_card_brings_its_own_surfaces(demo):
    """§ 5.4: a host hands over an empty element and gets a working viewer — so
    the card's surfaces are the module's, not a class the host happens to have.

    Three grounds, and which is which is the whole of the visual organisation
    (§ 8.1): the module's box is the ground, the window and the panel are the
    cards raised on it, and the drawing has a darker ground of its own.
    """
    colour = lambda sel: demo.eval_on_selector(
        sel, "el => getComputedStyle(el).backgroundColor"
    )
    ground = colour(".molview-card")
    window_card = colour(".molview-card .molviewer-window-frame")
    panel_card = colour(".molview-card .molviewer-selection-card")
    scene = colour(".molviewer-window-canvas")

    assert window_card == panel_card, "the two cards are not one surface"
    assert ground != window_card, (
        "the ground and the cards are the same colour, so nothing separates them"
    )
    assert scene != window_card, "the 3D window has no ground of its own"
    for name, value in [("ground", ground), ("card", window_card), ("scene", scene)]:
        assert "255, 255, 255" not in value, f"the {name} is white: {value}"


def test_the_cell_door_speaks_the_route_it_posts_to(demo):
    """§ 6.2: the cell is "one fact that travels together, which is why there is
    one door to change it" — and a door the server refuses is not one.

    This runs against the REAL route, deliberately. Both halves of this had been
    wrong for as long as they existed and no stand-in could have caught either,
    because a stand-in written beside the code agrees with the code:

      - the door posted `{op, params, structure}`; the route reads
        `{data: {xyz, sidecar}, op, payload}` and answers 400 without it;
      - the block that comes back is `{cell, cell_origin, …}` and the module read
        `lattice` and `origin`, so the cell was null however the round trip went.

    § 13.1's rule about stand-ins is exactly this failure: one that "copies how
    the code happens to work" confirms behaviour that cannot happen.
    """
    got = demo.evaluate(
        """async () => {
            const { mount } = await import("/static/lib/molview/index.js");
            const host = document.createElement("div");
            host.style.width = "900px";
            document.body.appendChild(host);
            const ws = { read: async () => null, write: async () => {} };
            const v = await mount(host, ws, { owner: "cell-door" });

            await v.data.installMolecule({
                text: "2\\n\\nAu 0 0 0\\nAu 2 2 0\\n", filename: "au.xyz" });

            // Give it a cell through the one door, then read it back through
            // the one read (§ 9.3's main way in).
            // The payload IS the value for that op — the route's four ops are
            // vacuum / axis_kind / cell / cell_origin, each taking its own field
            // directly. MolView carries it through and interprets none of it
            // (§ 6.2).
            const answer = await v.data.commitPeriodicityOp(
                "cell", [[8,0,0],[0,8,0],[0,0,8]]);
            const info = v.data.getUnitCellInfo();   // the block's own names
            // What LEAVES the viewer is the structure, not a file (§ 11.7):
            // `exportFile()` "returns the structure as data and stops", and the
            // cell rides in its metadata under the names it arrived with (§ 6.2).
            const leaving = v.data.exportFile().structure;
            return {
                answered: answer !== null,
                cell:     info.cell,
                exported: leaving.metadata.cell,
            };
        }"""
    )
    assert got["answered"] is True, (
        "the cell door was refused by the route it posts to"
    )
    assert got["cell"] == [[8, 0, 0], [0, 8, 0], [0, 0, 8]], (
        f"the cell did not come back through the module's own read: {got}"
    )
    assert got["exported"] == got["cell"], (
        "the cell reached the viewer but not the sidecar, so a saved structure "
        f"would lose it: {got}"
    )


def test_two_viewers_on_one_page_collide_over_nothing(demo, molview_server):
    """§ 5.6 and § 12.6: two mounts are two viewers that share nothing — and the
    things they can collide over in a real document are NAMES: element ids, and
    radio-group names.

    Mounted through the module's one public import, exactly as a second tab
    would (§ 4).
    """
    demo.evaluate(
        """async () => {
            const host = document.createElement("div");
            host.style.width = "900px";
            document.body.appendChild(host);
            const { mount } = await import("/static/lib/molview/index.js");
            const ws = { read: async () => null, write: async () => {} };
            window.__second = await mount(host, ws, { owner: "second-viewer" });
        }"""
    )
    demo.wait_for_function("() => document.querySelectorAll('.molview-card').length === 2")

    report = demo.evaluate(
        """() => {
            // MolView's OWN markup. The drawing library builds its canvas
            // inside the window and names it as it likes — that DOM is its
            // business, and § 5.3 keeps it invisible from up here.
            const ids = [...document.querySelectorAll(".molview-card [id]")]
                .filter(el => el.tagName !== "CANVAS")
                .map(el => el.tagName + "." + String(el.className) + "#" + el.id);
            const groups = [...document.querySelectorAll(".molviewer-panel-tab-switch")]
                .map(sw => sw.querySelector("input").name);
            return { ids, groups, unique: new Set(groups).size };
        }"""
    )
    assert report["ids"] == [], (
        f"MolView wrote element ids, which are document-global: {report['ids']}"
    )
    assert report["unique"] == 2, (
        f"both viewers' page tabs share one radio group: {report['groups']}"
    )

    # And they are genuinely separate: choosing in one leaves the other alone.
    demo.locator(".molview-card").nth(1).locator(".molviewer-panel-tab-option").nth(1).click()
    _settle(demo)
    first_pages = demo.locator(".molview-card").nth(0).locator(".molviewer-panel-tab")
    assert first_pages.nth(0).is_visible(), (
        "choosing a page in the second viewer moved the first viewer's"
    )

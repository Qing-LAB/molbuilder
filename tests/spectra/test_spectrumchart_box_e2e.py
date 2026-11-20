"""The chart's box, and the words above its plot, measured in a real browser.

`spectrumchart.md` § 11: *"The frame states its size; the surface fills it,
carries no padding, and never spills past it."*  All three from real breakage
on one day: a height that came from the plot collapsed to a strip; a padded
surface drew wider than its frame, because the plotting library asks the
surface how wide it is and the browser's answer counts padding in; and the
drawing then overflowed across the panel beside it.

WHY THIS IS NOT A REGEX ON `_style.css`.  A stylesheet is not a file, it is a
cascade: a page sheet, a reset, a `.spectrumchart *` shorthand or a later
rule in this very file can put the padding back, and the regex goes on
finding the declaration it was told to look for.  Computed style is the
answer the browser actually gives the library.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def test_the_frame_states_its_size_and_the_surface_carries_no_padding(
        page, flask_server):
    """Mount a real chart on the real server; read what the browser computed.

    The real app is used rather than a synthetic page because the module
    fetches two things through absolute URLs -- its own stylesheet at
    `/static/lib/spectrumchart/_style.css` and the plotting library at
    `/vendor/plotly.min.js`, which the app serves out of the installed
    `plotly` package -- and a hand-rolled fixture would have to reproduce
    both routes to prove anything about either.
    """
    page.goto(f"{flask_server}/results")
    got = page.evaluate("""async () => {
        const m = await import("/static/lib/spectrumchart/index.js");
        const pad = (el) => {
            const s = getComputedStyle(el);
            return [s.paddingTop, s.paddingRight, s.paddingBottom, s.paddingLeft];
        };
        const host = document.createElement("div");
        host.style.width = "640px";
        host.style.height = "320px";
        document.body.appendChild(host);
        const handle = await m.mount(host);
        const frame = host.querySelector(".spectrumchart");
        const surf  = host.querySelector(".spectrumchart-surface");
        if (!frame || !surf) {
            return {error: "the module mounted no frame/surface pair; "
                           + "host held: " + host.innerHTML.slice(0, 200)};
        }
        const cs = getComputedStyle(frame);
        const out = {surface: pad(surf),
                     overflow: cs.overflow,
                     // WHERE THE NUMBER CAME FROM, which is the whole claim.
                     // The token is `clamp(252px, 34vh, 460px)`, so its raw
                     // text can never equal a resolved `cs.height`: a PROBE
                     // is given that height and the browser resolves it, and
                     // the two resolved numbers are what get compared.
                     //
                     // It has to be this and not a two-mounts-two-hosts
                     // comparison, measured 2026-09-07: under `height: auto`
                     // the frame is the SAME height in a 320px host and a
                     // 700px one, because the plotting library sets an
                     // explicit height on the div it owns.  That version of
                     // this assertion passed the mutation it existed for.
                     stated: (() => {
                         const probe = document.createElement("div");
                         probe.style.height = "var(--spectrumchart-height)";
                         frame.appendChild(probe);
                         const h = getComputedStyle(probe).height;
                         probe.remove();
                         return h;
                     })(),
                     painted: cs.height,
                     surfaceInner: surf.clientWidth,
                     frameInner: frame.clientWidth};
        if (handle && handle.dispose) handle.dispose();
        host.remove();
        return out;
    }""")

    assert "error" not in got, got.get("error")
    assert got["surface"] == ["0px", "0px", "0px", "0px"], (
        f"the surface computes padding {got['surface']} -- the library asks "
        f"this element how wide it is and the browser counts padding in, so "
        f"the drawing comes out wider than the frame and is clipped")
    assert 0 < got["surfaceInner"] <= got["frameInner"], (
        f"the surface reports {got['surfaceInner']}px of drawable width "
        f"inside a {got['frameInner']}px frame")

    assert got["stated"], (
        "the frame declares no --spectrumchart-height token, so there is no "
        "stated height for the painted one to have come from")
    assert got["painted"] == got["stated"], (
        f"the frame paints {got['painted']} while its token says "
        f"{got['stated']} -- the height is coming from the content, and a "
        f"frame that takes its height from a child sized in PERCENT is a "
        f"definition in a circle the browser resolves to a strip")
    assert got["overflow"] == "hidden", (
        f"the frame computes overflow:{got['overflow']} -- a drawing that "
        f"exceeds it spills across whatever sits beside it")


#: (what, the window's height, the page's font size, the chart's width,
#: whether a mode waits on its strength): 1400 and 600 px windows put the
#: frame at its height's ceiling and floor (§ 11); a large page font is what
#: the toolbar must not take its height from; a mode waiting draws the note's
#: row; and a narrow chart puts the note and the readout side by side.
STRIPS = [
    ("a tall window", 1400, None, 900, False),
    ("a short window", 600, None, 900, False),
    ("a large page font", 1400, "24px", 900, False),
    ("a mode waiting on its strength", 600, None, 900, True),
    ("a narrow chart, a mode waiting", 900, None, 400, True),
]


@pytest.mark.parametrize("what, height, page_font, width, waits", STRIPS,
                         ids=[s[0] for s in STRIPS])
def test_the_words_above_the_plot_keep_clear_of_the_toolbar(
        page, flask_server, what, height, page_font, width, waits):
    """§ 6.3.1: the line naming the mode nearest the pointer is drawn on the
    chart's surface, "in the strip above the plot area, below the library's
    toolbar, which shows in the same corner on the same hover (§ 5.3) and
    never covers it; the note of § 6.2 ... has a row of its own above it".

    THE SILENT FAILURE: on a tall window the readout climbed under the
    toolbar and its words were covered as the pointer moved; beside the note
    on a narrow chart the two lines wrote over each other."""
    page.set_viewport_size({"width": 1200, "height": height})
    page.goto(f"{flask_server}/results")
    page.evaluate("""async ([font, width, waits]) => {
        if (font) {
            document.documentElement.style.fontSize = font;
            document.body.style.fontSize = font;
        }
        const m = await import("/static/lib/spectrumchart/index.js");
        const host = document.createElement("div");
        host.id = "readout-probe";
        host.style.width = width + "px";
        document.body.prepend(host);
        const chart = await m.mount(host);
        chart.setModes([
            { index: 1, freq:  323.5, values: { activity:  2.9 } },
            { index: 2, freq:  826.6, values: { activity:  5.4 } },
            { index: 3, freq: 1131.8, values: { activity: 12.9 } },
        ].concat(waits ? [{ index: 4, freq: 1490.0 }] : []));
        chart.setBroadening(20);
    }""", [page_font, width, waits])
    area = page.locator("#readout-probe .nsewdrag").bounding_box()
    page.mouse.move(area["x"] + area["width"] * 0.6, area["y"] + area["height"] / 2)
    page.wait_for_function("""() => [...document.querySelectorAll(
        '#readout-probe .annotation-text')].some((t) => /mode \\d/.test(t.textContent))""")
    got = page.evaluate("""() => {
        const box = (el) => { const r = el.getBoundingClientRect();
                              return { top: r.top, bottom: r.bottom,
                                       left: r.left, right: r.right }; };
        const said = (re) => [...document.querySelectorAll(
            '#readout-probe .annotation-text')].find((t) => re.test(t.textContent));
        const bar = document.querySelector('#readout-probe .modebar');
        const note = said(/not computed/);
        return { readout: box(said(/mode \\d/)), note: note && box(note),
                 toolbar: bar && box(bar),
                 surface: box(document.querySelector('#readout-probe .spectrumchart-surface')),
                 plot: box(document.querySelector('#readout-probe .nsewdrag')) };
    }""")
    readout, note, toolbar = got["readout"], got["note"], got["toolbar"]
    surface, plot = got["surface"], got["plot"]
    apart = lambda a, b: (a["bottom"] <= b["top"] or b["bottom"] <= a["top"]
                          or a["right"] <= b["left"] or b["right"] <= a["left"])
    assert toolbar, "no toolbar drawn to keep clear of"
    assert apart(readout, toolbar), (
        f"on {what} the toolbar {toolbar} covers the readout {readout}")
    assert bool(note) == waits, f"on {what} the note is {'missing' if waits else 'drawn'}"
    if waits:
        assert apart(note, toolbar), f"on {what} the toolbar {toolbar} covers the note {note}"
        assert apart(note, readout), (
            f"on {what} the note {note} and the readout {readout} overlap")
    for name, line in (("readout", readout), ("note", note)):
        if not line:
            continue
        assert surface["top"] <= line["top"], (
            f"on {what} the {name} {line} is cut off above the surface {surface}")
        assert line["bottom"] <= plot["top"], (
            f"on {what} the {name} {line} is inside the plot {plot}")

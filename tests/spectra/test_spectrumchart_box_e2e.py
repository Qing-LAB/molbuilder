"""The chart's box traps, measured after the cascade rather than read off it.

`spectrumchart.md` § 11, all three traps, all three from real breakage:\nthe frame states a height, the frame clips, and **the frame pads while\nthe surface does not.**  The
plotting library asks the surface how wide it is, and the browser's answer
counts padding in -- so a padded surface draws wider than the frame meant to
contain it, and the overflow is clipped rather than reported.

WHY THIS IS NOT A REGEX ON `_style.css`, which is what stood here.
`tests/spectra/test_spectrumchart_seal.py` asserted

    re.search(r"\\.spectrumchart-surface\\s*\\{[^}]*padding:\\s*0", css, re.S)

-- one declaration, in one file.  A stylesheet is not a file, it is a
cascade: a page sheet, a reset, a `.spectrumchart *` shorthand or a later
rule in this very file can put the padding back, and the regex goes on
finding the declaration it was told to look for.  Computed style is the
answer the browser actually gives the library.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def test_the_frame_pads_and_the_surface_does_not(page, flask_server):
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
        const out = {surface: pad(surf), frame: pad(frame),
                     overflow: cs.overflow,
                     // WHERE THE NUMBER CAME FROM, which is the whole claim.
                     // The token is `clamp(220px, 34vh, 460px)`, so its raw
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
    assert any(p != "0px" for p in got["frame"]), (
        f"the FRAME computes no padding either ({got['frame']}); the rule is "
        f"that the frame pads and the surface does not, and half of it "
        f"holding by accident is not the rule holding")
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

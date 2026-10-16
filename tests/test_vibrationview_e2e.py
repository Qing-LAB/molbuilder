"""VibrationView in a real browser, through ``mount`` as a page uses it
(`web/vibrationview.md` § 14, *end to end*): what a viewer leaves on the page
when it is closed, and what a viewer on a kept window draws (§ 8, § 9.3a).

The page is ``/results``, which loads the drawing library; the module is
imported by name, as a page's module script imports it.
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


#: A structure and one of its modes, as a tab hands them over (§ 6.1).
WATER = {"elements": ["O", "H", "H"],
         "positions": [[0, 0, 0], [0.957, 0, 0], [-0.239, 0.927, 0]]}
MODE = {"index": 7,
        "displacements": [[0, 0.07, 0], [0.4, -0.56, 0], [-0.4, -0.56, 0]]}


def _open(page, base):
    """The page settled: the drawing library loaded and the page's own
    sidebar listed, so what it adds while it starts is not counted."""
    page.goto(f"{base}/results", wait_until="networkidle")
    page.wait_for_function("() => !!window.$3Dmol", timeout=15000)
    page.wait_for_selector(".ps-entry, .ps-list-error, #ps-list.is-empty",
                           state="attached", timeout=10000)


def test_opening_and_closing_a_viewer_again_and_again_leaves_what_once_left(
        page, flask_server, isolated_projects_root):
    """§ 8, § 9.3a: a viewer opened, given a structure and a mode, played and
    closed, five times, leaves on the page what doing it once left -- the
    listeners on the window, the document or an element still on the page,
    and the live graphics contexts.

    THE SILENT FAILURE: every viewer opened leaving its window alive for the
    page's life, listening to every click and resize, until the graphics
    contexts run out and the browser takes one from a viewer on screen
    (measured 2026-10-15: twenty of twenty, sixteen contexts).

    Paired add to remove on ``EventTarget`` itself, as the browser pairs
    them, each listener named with what added it (`support/leftovers.SPY`).

    MUTATION THIS MUST FAIL AGAINST: a window made for every mount."""
    _open(page, flask_server)
    from support.leftovers import SPY, grew
    left = page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
        const spy = """ + SPY + """;
        const cycle = async () => {
            const host = document.createElement("div");
            host.style.width = "400px"; host.style.height = "300px";
            document.body.appendChild(host);
            const vib = await mount(host, {});
            if (!vib.ok) throw new Error(vib.error);
            vib.setStructure(c.water);
            vib.showMode(c.mode);
            vib.play();
            await new Promise((r) => setTimeout(r, 150));
            vib.dispose();
            host.remove();
        };
        try {
            await cycle();
            const once = spy.state();
            for (let i = 1; i < c.times; i++) await cycle();
            return { once, again: spy.state() };
        } finally {
            spy.restore();
        }
    }""", {"water": WATER, "mode": MODE, "times": 5})
    assert not grew(left), (
        "opening a viewer five times left more on the page than opening it "
        "once:\n  " + "\n  ".join(grew(left)))
    assert left["again"]["contexts"] == left["once"]["contexts"], (
        f"{left['once']['contexts']} live graphics context(s) opened once, "
        f"{left['again']['contexts']} opened five times")


def test_a_viewer_on_a_kept_window_draws_as_a_new_one(
        page, flask_server, isolated_projects_root):
    """§ 9.3a: a window taken is a new one.  Viewer A, on a white ground, is
    given a structure and a mode, turned with the mouse, and closed in the
    middle of an export at 1600 x 1200 on a transparent ground.  Viewer B,
    opened next in a host of another size, takes A's window; viewer C,
    beside it, a new one.  B and C draw the same picture -- empty, and with
    the same structure and mode.

    THE SILENT FAILURE: the next spectrum's molecule opening on the last
    one's mouse turn, on the last page's ground, or at the size and ground of
    an export cut short -- or the last molecule still in the window.

    MUTATIONS THIS MUST FAIL AGAINST: the camera kept; the window not
    cleared; the ground not the new host's."""
    _open(page, flask_server)
    box = page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
        const host = document.createElement("div");
        host.style.cssText = "width: 400px; height: 300px;";
        host.style.setProperty("--vibview-ground", "#ffffff");
        document.body.prepend(host);
        const a = await mount(host, {});
        a.setStructure(c.water);
        a.showMode(c.mode);
        window.__a = { vib: a, host: host };
        const r = host.getBoundingClientRect();
        return { x: r.x, y: r.y, w: r.width, h: r.height };
    }""", {"water": WATER, "mode": MODE})
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + 120, cy + 40, steps=8)
    page.mouse.up()
    out = page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
        const settle = () => new Promise((r) => requestAnimationFrame(
            () => requestAnimationFrame(() => setTimeout(r, 300))));
        const a = window.__a;
        let closed = false;
        await a.vib.exportAnimation({
            format: "png-zip", width: 1600, height: 1200,
            background: "transparent",
            onProgress: () => {
                if (!closed) { closed = true; a.vib.dispose(); }
            },
        }).catch(() => null);
        a.host.remove();
        const host = () => {
            const h = document.createElement("div");
            h.style.cssText = "width: 500px; height: 360px;";
            document.body.prepend(h);
            return h;
        };
        const picture = (h) => {
            const cv = h.querySelector("canvas");
            return cv ? cv.toDataURL() : "";
        };
        const hb = host(), hc = host();
        const b = await mount(hb, {});
        const cc = await mount(hc, {});
        await settle();
        const empty = [picture(hb), picture(hc)];
        for (const v of [b, cc]) { v.setStructure(c.water); v.showMode(c.mode); }
        await settle();
        const drawn = [picture(hb), picture(hc)];
        b.dispose(); cc.dispose(); hb.remove(); hc.remove();
        return { closed, empty, drawn };
    }""", {"water": WATER, "mode": MODE})
    assert out["closed"], "the export never reached its first frame"
    assert out["empty"][0] and out["drawn"][0], "a window drew nothing at all"
    assert out["empty"][0] == out["empty"][1], (
        "the viewer on a kept window opened differently from one on a new "
        "window -- the last viewer's molecule, ground or export size")
    assert out["drawn"][0] == out["drawn"][1], (
        "the viewer on a kept window drew its molecule differently from one "
        "on a new window -- the last viewer's camera or ground")

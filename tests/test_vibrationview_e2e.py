"""VibrationView in a real browser, through ``mount`` as a page uses it
(`web/vibrationview.md` § 14, *end to end*): what a viewer leaves on the page
when it is closed, and what a viewer on a kept window draws (§ 8, § 9.3a).

The page is ``/results``, which loads the drawing library; the module is
imported by name, as a page's module script imports it.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
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
        page, flask_server):
    """§ 8, § 9.3a: a viewer opened, given a structure and a mode, played and
    closed, five times, leaves on the page what doing it once left -- the
    listeners on the window, the document or an element still on the page,
    and the live graphics contexts.

    THE SILENT FAILURE: every viewer opened leaving its window alive for the
    page's life, listening to every click and resize, until the graphics
    contexts run out and the browser takes one from a viewer on screen
    (measured 2026-10-15: twenty of twenty, sixteen contexts).

    Paired add to remove on ``EventTarget`` itself, as the browser pairs
    them, each listener named with what added it
    (`support.leftovers.what_grew`).

    MUTATION THIS MUST FAIL AGAINST: a window made for every mount."""
    from support.leftovers import what_grew
    _open(page, flask_server)
    grew = what_grew(page, lambda: page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
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
    }""", {"water": WATER, "mode": MODE}))
    assert not grew, (
        "opening a viewer five times left more on the page than opening it "
        "once:\n  " + "\n  ".join(grew))


def test_a_viewer_on_a_kept_window_draws_as_a_new_one(page, flask_server):
    """§ 9.3a: a window taken is a new one.  Viewer A, on a white ground, is
    given a structure and a mode, turned with the mouse -- its picture
    asserted turned -- and closed in the
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
        await new Promise((r) => requestAnimationFrame(
            () => requestAnimationFrame(r)));
        const r = host.getBoundingClientRect();
        return { x: r.x, y: r.y, w: r.width, h: r.height,
                 picture: host.querySelector("canvas").toDataURL() };
    }""", {"water": WATER, "mode": MODE})
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + 120, cy + 40, steps=8)
    page.mouse.up()
    turned = page.evaluate("""() => new Promise((r) => requestAnimationFrame(
        () => requestAnimationFrame(() => r(window.__a.host
            .querySelector("canvas").toDataURL()))))""")
    assert turned != box["picture"], "the drag did not turn viewer A"
    out = page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
        const settle = () => new Promise((r) => requestAnimationFrame(
            () => requestAnimationFrame(r)));
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
    assert all(d != e for d, e in zip(out["drawn"], out["empty"])), (
        "a viewer given a structure and a mode drew nothing of them")
    assert out["empty"][0] == out["empty"][1], (
        "the viewer on a kept window opened differently from one on a new "
        "window -- the last viewer's molecule, ground or export size")
    assert out["drawn"][0] == out["drawn"][1], (
        "the viewer on a kept window drew its molecule differently from one "
        "on a new window -- the last viewer's camera or ground")


#: An export's size, a row each: the window's own, twice it, and a shape
#: the window is not -- each asked of a viewer in a 400 x 300 window.
_EXPORT_SIZES = ((400, 300), (800, 600), (640, 640))


@pytest.mark.parametrize("width,height", _EXPORT_SIZES,
                         ids=[f"{w}x{h}" for w, h in _EXPORT_SIZES])
def test_an_export_is_the_size_asked(page, flask_server, width, height):
    """§ 12 (`exportAnimation`): `width` and `height` are the picture's,
    "independent of the on-screen box" -- every frame of a PNG export is the
    size asked, the manifest says so, and the window is put back.

    THE SILENT FAILURE (F10's reading, 2026-11-16): the capture set the
    canvas to the size asked and then asked the library to resize, which
    reads the on-screen box again -- every export came out the window's
    size, whatever was asked."""
    _open(page, flask_server)
    out = page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
        const host = document.createElement("div");
        host.style.cssText = "width: 400px; height: 300px;";
        document.body.prepend(host);
        const vib = await mount(host, {});
        vib.setStructure(c.water);
        vib.showMode(c.mode);
        const before = host.querySelector("canvas").getBoundingClientRect();
        const got = await vib.exportAnimation({
            format: "png-zip", width: c.w, height: c.h, cycles: 1 });
        const text = new TextDecoder("latin1").decode(
            new Uint8Array(await got.blob.arrayBuffer()));
        const at = text.indexOf('"width"');
        const manifest = JSON.parse(text.slice(text.lastIndexOf("{", at),
                                               text.indexOf("}", at) + 1));
        // Every frame's own header: a PNG states its size in its IHDR.
        const bytes = new Uint8Array(await got.blob.arrayBuffer());
        const sizes = [];
        for (let i = 0; i + 24 < bytes.length; i++) {
            if (bytes[i] === 0x89 && bytes[i + 1] === 0x50 && bytes[i + 2] === 0x4E
                    && bytes[i + 3] === 0x47 && bytes[i + 12] === 0x49
                    && bytes[i + 13] === 0x48) {
                const dv = new DataView(bytes.buffer, i + 16, 8);
                sizes.push([dv.getUint32(0), dv.getUint32(4)]);
            }
        }
        const after = host.querySelector("canvas").getBoundingClientRect();
        vib.dispose(); host.remove();
        return { manifest: [manifest.width, manifest.height], sizes,
                 meta: [got.meta.width, got.meta.height],
                 window: [Math.round(before.width), Math.round(before.height),
                          Math.round(after.width), Math.round(after.height)] };
    }""", {"water": WATER, "mode": MODE, "w": width, "h": height})
    assert out["manifest"] == [width, height], out
    assert out["meta"] == [width, height], (
        f"the export's description does not say its size: {out['meta']}")
    assert out["sizes"] and all(s == [width, height] for s in out["sizes"]), (
        out["sizes"][:3])
    assert out["window"][:2] == out["window"][2:], (
        f"the window was not put back after the export: {out['window']}")


def test_a_gif_on_a_transparent_ground_is_transparent(page, flask_server):
    """§ 12.1: a GIF asked for on a transparent ground has one -- 1-bit, a
    pixel opaque or gone (its edges fringe, as the chapter says), never an
    opaque ground.  The file decoded by the browser itself: the corner, where
    no atom is, shows through.

    THE SILENT FAILURE (F10's reading, 2026-11-16): the encoder was never
    told which colour is transparent and painted each frame over black, so a
    "transparent" GIF came out on an opaque black ground."""
    _open(page, flask_server)
    out = page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
        const host = document.createElement("div");
        host.style.cssText = "width: 400px; height: 300px;";
        document.body.prepend(host);
        const vib = await mount(host, {});
        vib.setStructure(c.water);
        vib.showMode(c.mode);
        const got = await vib.exportAnimation({
            format: "gif", width: 200, height: 150, cycles: 1,
            background: "transparent" });
        vib.dispose(); host.remove();
        // The browser's own decoder, the file's first frame.
        const img = await createImageBitmap(got.blob);
        const cv = document.createElement("canvas");
        cv.width = img.width; cv.height = img.height;
        const g = cv.getContext("2d");
        g.drawImage(img, 0, 0);
        const corner = Array.from(g.getImageData(0, 0, 1, 1).data);
        return { size: [img.width, img.height], corner };
    }""", {"water": WATER, "mode": MODE})
    assert out["size"] == [200, 150], out
    assert out["corner"][3] == 0, (
        f"the GIF's ground is opaque, not transparent: corner {out['corner']}")


def test_a_gifs_encoding_reports_progress_and_can_be_cancelled(page,
                                                               flask_server):
    """§ 12.2: a GIF's progress covers its encoding -- the fraction reaches 1
    only when the last frame is encoded -- and a cancel while it encodes
    rejects, handing back no file and leaving the viewer free for the next
    export.

    THE SILENT FAILURE (F10's reading, 2026-11-16): the fraction reached 1
    when the last frame was COPIED, the encoding after it reported nothing,
    and a cancel during it was ignored -- the file resolved anyway."""
    _open(page, flask_server)
    out = page.evaluate("""async (c) => {
        const { mount } = await import("/static/lib/vibrationview/index.js");
        const host = document.createElement("div");
        host.style.cssText = "width: 400px; height: 300px;";
        document.body.prepend(host);
        const vib = await mount(host, {});
        vib.setStructure(c.water);
        vib.showMode(c.mode);
        const seen = [];
        await vib.exportAnimation({ format: "gif", width: 120, height: 90,
                                    onProgress: (f, label) => seen.push([f, label]) });
        const ac = new AbortController();
        let cancelled = "resolved";
        try {
            await vib.exportAnimation({
                format: "gif", width: 120, height: 90, signal: ac.signal,
                onProgress: (f, label) => { if (label === "encoding") ac.abort(); } });
        } catch (e) { cancelled = e.message; }
        let next = "refused";
        try {
            await vib.exportAnimation({ format: "gif", width: 120, height: 90 });
            next = "ran";
        } catch (e) { next = e.message; }
        vib.dispose(); host.remove();
        return { seen, cancelled, next };
    }""", {"water": WATER, "mode": MODE})
    fractions = [f for f, _ in out["seen"]]
    encoding = [f for f, label in out["seen"] if label == "encoding"]
    assert encoding, "the encoding reported no progress"
    assert fractions[-1] == 1.0 and fractions.count(1.0) == 1, fractions[-5:]
    assert all(b >= a for a, b in zip(fractions, fractions[1:])), fractions
    assert out["cancelled"] == "the export was cancelled", out["cancelled"]
    assert out["next"] == "ran", out["next"]


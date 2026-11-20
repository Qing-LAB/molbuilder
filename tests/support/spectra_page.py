"""THE SPECTRA PAGE, DRIVEN AS A PERSON DRIVES IT (`web/spectra.md`).

The doors a browser test opens the Results tab's spectra viewer through, and
reads its mode box by: a run's spectrum picked from the file menu
(:func:`spectrum_opened`), the box's molecule paused and turned by a drag
(:func:`mode_box_turned`), and its picture (:func:`mode_box_picture`),
compared within a rounding (:func:`pictures_alike`).
"""
from __future__ import annotations

from contextlib import contextmanager

from support.sidebar import WAIT_MS, result_picked, results_on


@contextmanager
def spectrum_opened(page, run, setenv):
    """``run``'s spectrum (its `freq` run's) picked on the Results tab of the
    app's own server, the one door pointed at its tree with ``setenv``, and
    the mode box's host shown: the server's address."""
    from conftest import point_the_one_door_at
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from support.live_server import serve
    folder = run.attempt("freq")
    point_the_one_door_at(run.tree.parent, setenv)
    with serve() as base:
        results_on(page, base, folder)
        result_picked(page, on_label_at(folder, ROLE_SPECTRA))
        page.locator("#mode-viewer").wait_for(state="visible", timeout=WAIT_MS)
        yield base


def mode_box_picture(page) -> bytes:
    """What the spectra page's mode box draws: its canvas, as a screenshot --
    the molecule and its caption, not the box's frame around them."""
    canvas = page.locator("#mode-viewer canvas").first
    canvas.scroll_into_view_if_needed()
    return canvas.screenshot()


def mode_box_turned(page) -> bytes:
    """The spectra page's mode box, once it shows a mode, paused and its
    molecule turned by a drag, as a person turns it: the box's picture after
    the turn.  It differs from the picture before, or the drag turned nothing
    and a test comparing pictures after it would compare the fitted camera
    with itself."""
    from playwright.sync_api import expect
    # The line under the box is written once the box shows its mode: a
    # Pause pressed before that finds no box and is lost.
    expect(page.locator("#viewer-status")).to_contain_text(
        "further than", timeout=WAIT_MS)
    page.locator("#anim-toggle").click()
    expect(page.locator("#anim-toggle")).to_have_text("Play")
    fitted = mode_box_picture(page)
    box = page.locator("#mode-viewer canvas").first.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.6,
                    steps=8)
    page.mouse.up()
    turned = mode_box_picture(page)
    assert not pictures_alike(turned, fitted), "the drag turned nothing"
    return turned


def pictures_alike(a: bytes, b: bytes) -> bool:
    """Two screenshots of one drawing alike: of one size, and no channel of
    any pixel off by more than 8 of 255.  Measured 2026-11-19: the mode box
    laid out a fraction of a pixel aside re-antialiased its rounded corner,
    9 pixels off by 1; a molecule turned by a drag moves hundreds of pixels
    by tens to hundreds.  Read with the standard library alone."""
    one, two = _pixels(a), _pixels(b)
    if one[:3] != two[:3]:
        return False
    return max(abs(x - y) for x, y in zip(one[3], two[3])) <= 8


def _pixels(png: bytes):
    """A screenshot's pixels -- (width, height, bytes a pixel, the rows) --
    from the PNG a browser writes: 8 bits a channel, RGB or RGBA, not
    interlaced, each row's filter undone (the PNG specification, § 9)."""
    import struct
    import zlib
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    at, packed, head = 8, b"", None
    while at < len(png):
        size, kind = struct.unpack(">I4s", png[at:at + 8])
        data = png[at + 8:at + 8 + size]
        if kind == b"IHDR":
            head = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            packed += data
        at += 12 + size
    width, height, depth, colour, _method, _filter, interlace = head
    assert depth == 8 and colour in (2, 6) and interlace == 0, head
    step = 3 if colour == 2 else 4
    raw, stride = zlib.decompress(packed), width * step
    rows, above = [], bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for i in range(stride):
            left = line[i - step] if i >= step else 0
            up = above[i]
            corner = above[i - step] if i >= step else 0
            if kind == 1:
                line[i] = (line[i] + left) & 255
            elif kind == 2:
                line[i] = (line[i] + up) & 255
            elif kind == 3:
                line[i] = (line[i] + (left + up) // 2) & 255
            elif kind == 4:
                guess = left + up - corner
                near = min((abs(guess - left), 0, left), (abs(guess - up), 1, up),
                           (abs(guess - corner), 2, corner))
                line[i] = (line[i] + near[2]) & 255
        rows.append(bytes(line))
        above = line
    return width, height, step, b"".join(rows)

"""SpectrumChart `_seal.js` — the obligations § 4, § 8.4 and § 11 lay on it.

Two kinds of check, and the split is deliberate:

  * what can be read off the files — the library is named in one place, the
    stylesheet keeps to its own names and holds no loose values (§ 4, § 11).
    These are static facts, so they are checked statically rather than mimed;
  * what the seal DOES with a drawing library — the five doors, the palette, the
    click that comes up as a position (§ 8.4). Run in node against a stand-in
    that takes Plotly's place, per § 12's middle level: the stand-in replaces the
    LIBRARY, never a file of this module.

What is left over is the loading path itself — a real `<link>` and a real
`<script>` reaching a real server — which belongs to § 12's third level, a real
page, and is not mimed here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests._node_esm import run_node

MODULE = (
    Path(__file__).resolve().parents[2]
    / "molbuilder" / "web" / "static" / "lib" / "spectrumchart"
)
SEAL = MODULE / "_seal.js"
STYLE = MODULE / "_style.css"

# The browser stand-in already used by the MolView JS tests: it stands in for the
# BROWSER, not for any module, so it serves here unchanged rather than being
# copied. What it does not carry -- a head to append to, a link/script that
# announces itself loaded -- is added below, because those are this module's
# needs and not everyone's.
DOM = (Path(__file__).resolve().parents[1] / "support" / "molview_dom_standin.js").read_text()

BROWSER = DOM + """
/* A browser loads what you append to the head and says so. The stand-in has to
   do that too, or a module that waits for its stylesheet waits forever -- which
   is a real thing this module does (contract § 11). */
const head = document.createElement("head");
const appendToHead = head.appendChild.bind(head);
head.appendChild = (node) => {
    appendToHead(node);
    queueMicrotask(() => { if (node.onload) node.onload(); });
    return node;
};
document.head = head;
document.querySelector = (sel) => null;

const PLOTLY_CALLS = [];
globalThis.__calls = PLOTLY_CALLS;
globalThis.Plotly = {
    react(el, traces, layout, config) {
        PLOTLY_CALLS.push({ call: "react", traces, layout, config });
        el.on = (type, fn) => el.addEventListener(type, fn);
        /* What Plotly leaves on the graph element once it has drawn: the drag
           layer covering exactly the plot area, and the range it drew against.
           The seal reads both to turn a pointer position into a frequency. */
        const area = document.createElement("div");
        area.className = "nsewdrag";
        area.getBoundingClientRect = () => ({ left: 100, width: 400, top: 0, height: 200 });
        el.appendChild(area);
        el.querySelector = (sel) => (sel === ".nsewdrag" ? area : null);
        el._fullLayout = { xaxis: { range: [0, 4000] } };
    },
    restyle(el, update, indices) { PLOTLY_CALLS.push({ call: "restyle", update, indices }); },
    purge(el) { PLOTLY_CALLS.push({ call: "purge" }); },
    Plots: { resize(el) { PLOTLY_CALLS.push({ call: "resize" }); } },
};

globalThis.__host = function (values) {
    const host = document.createElement("div");
    host.clientWidth = 900;
    // What _style.css declares on the frame; the stand-in serves custom
    // properties from the nearest ancestor that has one.
    host._customProps = values || {
        "--spectrumchart-bg": "#111111",
        "--spectrumchart-ink": "#eeeeee",
        "--spectrumchart-grid": "#222222",
        "--spectrumchart-stick": "#3333ff",
        "--spectrumchart-chosen": "#00ff00",
        "--spectrumchart-curve": "#999999",
        "--spectrumchart-pending": "#666666",
        "--spectrumchart-imaginary": "#ff0000",
    };
    document.body.appendChild(host);
    return host;
};
"""

PICTURE = """
const picture = {
    sticks: { x: [100, 200, 300], y: [1, 2, 3], width: [2, 2, 2],
              state: ["plain", "chosen", "imaginary"] },
    curve: { x: [90, 100, 110], y: [0.1, 1.0, 0.1] },
    xTitle: "frequency (cm-1)", yTitle: "strength",
};
"""


def seal(snippet: str):
    """Open a surface on a stand-in host and evaluate `snippet` against it."""
    program = (
        f"const S = await import({json.dumps(SEAL.resolve().as_uri())});\n"
        "const host = globalThis.__host();\n"
        "const surface = await S.openSurface(host);\n"
        f"{PICTURE}\n{snippet}"
    )
    return run_node([], program, globals_js=BROWSER)


# --- § 4  the library is named once, and nothing above the seal names it -----

def test_plotly_is_named_in_exactly_one_file_of_the_module():
    """§ 4 — the library name appears once."""
    named = [
        p.name for p in MODULE.glob("*.js")
        if re.search(r"\bPlotly\b", p.read_text())
    ]
    assert named == ["_seal.js"], f"the library is named in {named}"


def test_nothing_in_the_module_imports_from_outside_it():
    """§ 4 — self-contained: it reaches nothing else in the app by name."""
    for path in MODULE.glob("*.js"):
        source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
        specs = re.findall(r"""(?m)^\s*import\s[^;]*?from\s*["']([^"']+)["']""", source)
        specs += re.findall(r"""import\(\s*["']([^"']+)["']""", source)
        for spec in specs:
            assert spec.startswith("."), f"{path.name} imports {spec}"


# --- § 11  the stylesheet keeps to itself -----------------------------------

def test_every_selector_is_the_modules_own_name():
    """§ 11 — the sheet stays this module's: no bare names, no element selectors."""
    body = re.sub(r"/\*.*?\*/", "", STYLE.read_text(), flags=re.S)
    for selector in re.findall(r"^\s*([^@{}]+?)\s*\{", body, flags=re.M):
        for part in selector.split(","):
            part = part.strip()
            assert part.startswith(".spectrumchart"), f"selector escapes the module: {part!r}"


def test_no_rule_is_keyed_to_the_window_size():
    """§ 11 — the chart responds to its own box, never to the window."""
    assert "@media" not in STYLE.read_text()


def test_values_are_declared_once_and_read_by_name():
    """§ 11 — its values sit in one block; the rules below read them."""
    body = re.sub(r"/\*.*?\*/", "", STYLE.read_text(), flags=re.S)
    blocks = re.findall(r"\{(.*?)\}", body, flags=re.S)
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("--"):
                continue  # the declaration block itself
            assert not re.search(r"#[0-9a-fA-F]{3,8}\b", line), f"loose colour: {line}"
            if re.search(r"\b\d+px\b", line) and "var(" not in line:
                pytest.fail(f"loose length: {line}")


def test_the_frame_states_a_height_and_the_surface_carries_no_padding():
    """§ 11 — the box traps, all three from real breakage."""
    css = STYLE.read_text()
    frame = css.split(".spectrumchart-surface")[0]
    assert re.search(r"height:\s*var\(--spectrumchart-height\)", frame)
    assert not re.search(r"height:\s*auto", css)
    assert re.search(r"\.spectrumchart-surface\s*\{[^}]*padding:\s*0", css, flags=re.S)
    assert re.search(r"overflow:\s*hidden", frame)


# --- § 8.4  commands down, one number up ------------------------------------

class TestTheDoors:

    def test_draw_puts_the_picture_on_the_surface(self):
        """§ 8.4 — draw(picture): the curve, the sticks, and the bands."""
        calls = seal("surface.draw(picture);\nconsole.log(JSON.stringify(__calls));")
        assert [c["call"] for c in calls] == ["react"]
        traces = calls[0]["traces"]
        assert [t["type"] for t in traces] == ["scatter", "bar"]
        assert traces[1]["x"] == [100, 200, 300]

    def test_a_click_anywhere_on_the_plot_comes_up_as_a_frequency(self):
        """§ 8.4 — the seal reports WHERE the pointer was, in the units of the
        picture. Nothing it can ask the library reports a click over empty space,
        so the position is read off the surface and converted against the axis.

        The stand-in draws over 400px starting at x=100, across a 0–4000 range:
        a click a quarter of the way in is 1000 cm-1.
        """
        got = seal(
            "const seen = [];\n"
            "surface.onClick((x) => seen.push(x));\n"
            "surface.draw(picture);\n"
            "const el = host.children[0].children[0];\n"
            "el.dispatch('click', { clientX: 200 });\n"     # 100px into 400
            "console.log(JSON.stringify(seen));"
        )
        assert got == [1000]

    def test_a_click_in_empty_space_reports_just_as_well_as_one_on_a_peak(self):
        """§ 6.3 — the space beside a peak is the whole purpose of a band, and it
        holds no mark for the library to report."""
        got = seal(
            "const seen = [];\n"
            "surface.onClick((x) => seen.push(x));\n"
            "surface.draw(picture);\n"
            "const el = host.children[0].children[0];\n"
            "el.dispatch('click', { clientX: 337.5 });\n"   # nothing drawn here
            "console.log(JSON.stringify(seen));"
        )
        assert got == [2375]

    def test_a_click_outside_the_plot_area_is_not_a_position(self):
        """§ 8.4 — the margins are not the picture; a click there names nothing."""
        got = seal(
            "const seen = [];\n"
            "surface.onClick((x) => seen.push(x));\n"
            "surface.draw(picture);\n"
            "const el = host.children[0].children[0];\n"
            "el.dispatch('click', { clientX: 40 });\n"
            "console.log(JSON.stringify(seen));"
        )
        assert got == []

    def test_a_state_becomes_a_colour_here_and_only_here(self):
        """§ 8.4 — the layer above says which mark is chosen, never what colour."""
        calls = seal("surface.draw(picture);\nconsole.log(JSON.stringify(__calls));")
        colours = calls[0]["traces"][1]["marker"]["color"]
        assert colours == ["#3333ff", "#00ff00", "#ff0000"]

    def test_every_colour_comes_from_a_value_on_the_frame(self):
        """§ 11 — the values are declared, not sampled: change them, the chart changes."""
        program = (
            "const host = globalThis.__host({"
            '"--spectrumchart-bg": "#abcabc", "--spectrumchart-ink": "#defdef",'
            '"--spectrumchart-stick": "#010101", "--spectrumchart-chosen": "#020202",'
            '"--spectrumchart-imaginary": "#030303", "--spectrumchart-curve": "#040404",'
            '"--spectrumchart-grid": "#050505", "--spectrumchart-pending": "#060606"'
            "});\n"
            f"const S = await import({json.dumps(SEAL.resolve().as_uri())});\n"
            "const surface = await S.openSurface(host);\n"
            f"{PICTURE}\n"
            "surface.draw(picture);\n"
            "console.log(JSON.stringify(__calls));"
        )
        calls = run_node([], program, globals_js=BROWSER)
        layout = calls[0]["layout"]
        assert layout["paper_bgcolor"] == "#abcabc"
        assert layout["font"]["color"] == "#defdef"
        assert calls[0]["traces"][1]["marker"]["color"] == ["#010101", "#020202", "#030303"]

    def test_recolour_changes_colours_without_a_rebuild(self):
        """§ 5.1 — the cheap door: no react, no layout, no axis."""
        calls = seal(
            "surface.draw(picture);\n"
            "__calls.length = 0;\n"
            'surface.recolour(["chosen", "plain", "plain"]);\n'
            "console.log(JSON.stringify(__calls));"
        )
        assert [c["call"] for c in calls] == ["restyle"]
        assert calls[0]["update"]["marker.color"] == [["#00ff00", "#3333ff", "#3333ff"]]

    def test_resize_fills_the_box_without_redrawing(self):
        """§ 8.4 — resize(): the box changed; fill it."""
        calls = seal(
            "surface.draw(picture);\n__calls.length = 0;\n"
            "surface.resize();\nconsole.log(JSON.stringify(__calls));"
        )
        assert [c["call"] for c in calls] == ["resize"]

    def test_purge_releases_the_surface_and_empties_the_host(self):
        """§ 7 — dispose leaves the host element empty, and is safe twice."""
        got = seal(
            "surface.draw(picture);\n__calls.length = 0;\n"
            "surface.purge();\nsurface.purge();\n"
            "console.log(JSON.stringify({ calls: __calls.map(c => c.call),"
            " left: host.children.length }));"
        )
        assert got == {"calls": ["purge"], "left": 0}

    def test_the_seal_answers_no_question_about_what_is_drawn(self):
        """§ 8.4 — the seal faces downward: five doors, and nothing to read back."""
        got = seal("console.log(JSON.stringify(Object.keys(surface).sort()));")
        assert got == ["draw", "onClick", "purge", "recolour", "resize"]

    def test_the_frame_and_the_surface_are_the_only_markup(self):
        """§ 11 — the sheet styles the frame and the surface and nothing else."""
        got = seal(
            "console.log(JSON.stringify({"
            " frame: host.children.map(c => c.className),"
            " inside: host.children[0].children.map(c => c.className) }));"
        )
        assert got == {"frame": ["spectrumchart"], "inside": ["spectrumchart-surface"]}

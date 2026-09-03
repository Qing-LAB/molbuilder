"""SpectrumChart `index.js` — the handle, driven exactly as a page drives it.

`web/spectrumchart.md` § 12's middle level: node, with a stand-in that takes
the drawing LIBRARY's
place. Nothing here reaches past `mount`; every fact goes in through a door and
the only thing that comes back is a click.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

MODULE = (
    Path(__file__).resolve().parents[2]
    / "molbuilder" / "web" / "static" / "lib" / "spectrumchart"
)
ENTRY = MODULE / "index.js"

DOM = (Path(__file__).resolve().parents[1] / "support" / "molview_dom_standin.js").read_text()

BROWSER = DOM + """
/* Browser behaviours the shared stand-in does not carry, added here rather than
   in it: a head that loads what you append, and replaceChildren. */
const head = document.createElement("head");
const appendToHead = head.appendChild.bind(head);
head.appendChild = (node) => {
    appendToHead(node);
    queueMicrotask(() => { if (node.onload) node.onload(); });
    return node;
};
document.head = head;
document.querySelector = () => null;

const PLOTLY_CALLS = [];
globalThis.__calls = PLOTLY_CALLS;
globalThis.Plotly = {
    react(el, traces, layout, config) {
        PLOTLY_CALLS.push({ call: "react", traces, layout });
        const area = document.createElement("div");
        area.className = "nsewdrag";
        area.getBoundingClientRect = () => ({ left: 0, width: 1000, top: 0, height: 200 });
        el.appendChild(area);
        el.querySelector = (sel) => (sel === ".nsewdrag" ? area : null);
        el._fullLayout = { xaxis: { range: [0, 1000] } };   // 1 px == 1 cm-1
    },
    restyle(el, update, indices) { PLOTLY_CALLS.push({ call: "restyle", update, indices }); },
    relayout(el, patch) { PLOTLY_CALLS.push({ call: "relayout", patch }); },
    purge() { PLOTLY_CALLS.push({ call: "purge" }); },
    Plots: { resize() { PLOTLY_CALLS.push({ call: "resize" }); } },
};

globalThis.__observed = [];
globalThis.ResizeObserver = class {
    constructor(fn) { this.fn = fn; }
    observe(el) { globalThis.__observed.push(el); el.__resize = this.fn; }
    disconnect() { globalThis.__observed = globalThis.__observed.filter(e => e.__resize !== this.fn); }
};

globalThis.__host = function () {
    const host = document.createElement("div");
    host.clientWidth = 900;
    host.replaceChildren = () => { host.children.length = 0; };
    host._customProps = {
        "--spectrumchart-bg": "#111111", "--spectrumchart-ink": "#eeeeee",
        "--spectrumchart-grid": "#222222", "--spectrumchart-stick": "#3333ff",
        "--spectrumchart-chosen": "#00ff00", "--spectrumchart-curve": "#999999",
        "--spectrumchart-pending": "#666666", "--spectrumchart-imaginary": "#ff0000",
        "--spectrumchart-hovered": "#cccc00",
    };
    document.body.appendChild(host);
    return host;
};

globalThis.__move = function (host, atCm1) {
    const el = host.children[0].children[0];
    el.dispatch("mousemove", { clientX: atCm1, clientY: 100, target: el });
};
globalThis.__leave = function (host) {
    host.children[0].children[0].dispatch("mouseleave", {});
};
globalThis.__click = function (host, atCm1) {
    const el = host.children[0].children[0];
    el.dispatch("click", { clientX: atCm1, clientY: 100, target: el });
};
"""

MODES = [
    {"index": 1, "freq": 100.0, "activity": 4.0, "imaginary": False},
    {"index": 2, "freq": 500.0, "activity": 9.0, "imaginary": False},
    {"index": 3, "freq": 900.0, "activity": 2.0, "imaginary": False},
]


def drive(body: str, *, modes=MODES):
    program = (
        f"const {{ mount }} = await import({json.dumps(ENTRY.resolve().as_uri())});\n"
        f"const MODES = {json.dumps(modes)};\n"
        "const host = globalThis.__host();\n"
        "const picked = [];\n"
        "const chart = await mount(host, { onSelect: (i) => picked.push(i) });\n"
        f"{body}"
    )
    return run_node([], program, globals_js=BROWSER)


def last_react(calls):
    return [c for c in calls if c["call"] == "react"][-1]


# --- § 7  mount and dispose --------------------------------------------------

def test_a_host_that_is_not_an_element_is_the_one_refusal():
    """§ 8.2 — it resolves with ok false and a message; it does not throw."""
    program = (
        f"const {{ mount }} = await import({json.dumps(ENTRY.resolve().as_uri())});\n"
        "const bad = await mount(null);\n"
        "const alsoBad = await mount('#chart');\n"
        "bad.dispose(); alsoBad.dispose();\n"
        "console.log(JSON.stringify({ keys: Object.keys(bad).sort(),"
        " ok: bad.ok, hasMessage: typeof bad.error === 'string' && bad.error.length > 0,"
        " second: alsoBad.ok }));"
    )
    got = run_node([], program, globals_js=BROWSER)
    assert got == {"keys": ["dispose", "error", "ok"], "ok": False,
                   "hasMessage": True, "second": False}


def test_the_module_owns_the_inside_of_its_host():
    """§ 7 — mount empties it, dispose empties it again, twice is safe."""
    got = drive(
        "console.log(JSON.stringify({"
        " afterMount: host.children.map(c => c.className),"
        " disposedTwice: (chart.dispose(), chart.dispose(), host.children.length) }));"
    )
    assert got == {"afterMount": ["spectrumchart"], "disposedTwice": 0}


def test_dispose_stops_the_box_watcher():
    """§ 7 — a resize after dispose draws nothing."""
    got = drive(
        "host.__resize();\n"
        "chart.dispose();\n"
        "__calls.length = 0;\n"
        "if (host.__resize) host.__resize();\n"
        "console.log(JSON.stringify(__calls.map(c => c.call)));"
    )
    assert got == []


def test_a_box_that_changes_size_redraws_while_the_window_sits_still():
    """§ 5.4 — the module watches its own box."""
    got = drive(
        "__calls.length = 0;\n"
        "host.__resize();\n"
        "console.log(JSON.stringify(__calls.map(c => c.call)));"
    )
    assert got == ["resize"]


# --- § 8.3  the doors --------------------------------------------------------

def test_the_handle_is_the_doors_and_nothing_else():
    """§ 4 — no hidden way in; § 8.3 — nothing is read back."""
    got = drive("console.log(JSON.stringify(Object.keys(chart).sort()));")
    assert got == ["dispose", "ok", "refit", "setBroadening", "setModes", "setSelected"]


def test_selecting_recolours_and_does_not_redraw():
    """§ 5.1 — the door says the cost: no curve, no axis, one restyle."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n__calls.length = 0;\n"
        "chart.setSelected(2);\n"
        "console.log(JSON.stringify(__calls.map(c => c.call)));"
    )
    assert got == ["restyle"]


def test_the_chosen_mode_is_the_one_the_caller_named():
    calls = drive(
        "chart.setModes(MODES);\nchart.setSelected(2);\n"
        "console.log(JSON.stringify(__calls));"
    )
    assert calls[-1]["update"]["marker.color"] == [["#3333ff", "#00ff00", "#3333ff"]]


def test_the_selection_is_recorded_before_any_modes_exist():
    """§ 8.3 — order does not matter, and a mirror that refused would drift."""
    calls = drive(
        "chart.setSelected(3);\nchart.setModes(MODES);\n"
        "console.log(JSON.stringify(__calls));"
    )
    colours = last_react(calls)["traces"][-1]["marker"]["color"]
    assert colours == ["#3333ff", "#3333ff", "#00ff00"]


def test_an_index_no_list_holds_highlights_nothing_and_raises_nothing():
    """§ 8.3 — and it costs nothing either: no colour changes, so no call is
    made. What is asserted is the state on screen, not the traffic."""
    calls = drive(
        "chart.setModes(MODES);\n__calls.length = 0;\nchart.setSelected(99);\n"
        "console.log(JSON.stringify(__calls));"
    )
    assert calls == []


def test_a_bad_width_leaves_the_one_already_set_standing():
    """§ 8.3 — substituting a default would hide the caller's bug."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n"
        "const before = __calls.length;\n"
        "chart.setBroadening(-5);\nchart.setBroadening('wide');\nchart.setBroadening(NaN);\n"
        "console.log(JSON.stringify(__calls.length - before));"
    )
    assert got == 0


def test_an_empty_list_is_a_state_not_a_failure():
    got = drive(
        "chart.setModes([]);\n"
        "console.log(JSON.stringify(last(__calls).traces[0].x));\n"
        "function last(a) { return a.filter(c => c.call === 'react').pop(); }"
    )
    assert got == []


# --- § 6.1  what a list must be ----------------------------------------------

def test_one_malformed_record_empties_the_chart():
    """§ 6.1 — the refusal takes the whole call, and the last spectrum does not
    stay on screen looking as though it worked."""
    got = drive(
        "chart.setModes(MODES);\n"
        "chart.setModes([{ index: 1, freq: 100 }, { index: 2 }]);\n"
        "console.log(JSON.stringify(__calls.filter(c => c.call === 'react').pop().traces.pop().x));"
    )
    assert got == []


def test_the_same_index_twice_is_refused():
    got = drive(
        "chart.setModes([{ index: 7, freq: 100, activity: 1 },"
        " { index: 7, freq: 200, activity: 1 }]);\n"
        "console.log(JSON.stringify(__calls.filter(c => c.call === 'react').pop().traces.pop().x));"
    )
    assert got == []


def test_a_list_that_is_not_a_list_is_refused():
    got = drive(
        "chart.setModes('modes');\n"
        "console.log(JSON.stringify(__calls.filter(c => c.call === 'react').pop().traces.pop().x));"
    )
    assert got == []


# --- § 6.2  the two pictures -------------------------------------------------

def test_with_no_strengths_anywhere_every_stick_is_the_same_height():
    calls = drive(
        "chart.setModes(MODES.map(m => ({ index: m.index, freq: m.freq })));\n"
        "console.log(JSON.stringify(__calls));"
    )
    react = last_react(calls)
    bars = [t for t in react["traces"] if t["type"] == "bar"][0]
    assert bars["y"] == [1, 1, 1]
    assert "strengths not computed" in react["layout"]["annotations"][0]["text"]


def test_a_mode_with_no_strength_is_marked_not_drawn_at_zero():
    """§ 6.2 — not computed can never be read as a strength of zero."""
    calls = drive(
        "chart.setModes([{ index: 1, freq: 100, activity: 4 }, { index: 2, freq: 500 }]);\n"
        "console.log(JSON.stringify(__calls));"
    )
    react = last_react(calls)
    bars = [t for t in react["traces"] if t["type"] == "bar"][0]
    marks = [t for t in react["traces"] if t.get("mode") == "markers"]
    assert bars["x"] == [100]                    # the one with a strength
    assert marks and marks[0]["x"] == [500]      # the other is marked, not zero
    assert marks[0]["marker"]["symbol"] == "x"


# --- § 6.4  imaginary modes --------------------------------------------------

def test_an_imaginary_mode_is_drawn_marked_and_clickable():
    calls = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n__click(host, 100);\n"
        "console.log(JSON.stringify({ calls: __calls, picked }));",
        modes=[
            {"index": 1, "freq": 100.0, "activity": 4.0, "imaginary": True},
            {"index": 2, "freq": 500.0, "activity": 9.0, "imaginary": False},
        ],
    )
    react = last_react(calls["calls"])
    bars = [t for t in react["traces"] if t["type"] == "bar"][0]
    assert bars["marker"]["color"][0] == "#ff0000"   # marked apart
    assert calls["picked"] == [1]                     # and clickable like any other


# --- § 6.3 / § 8.4  a click becomes a mode -----------------------------------

def test_a_click_beside_a_peak_selects_that_mode():
    """§ 6.3 — the band is why picking a mode is not a test of aim. The stand-in
    maps one pixel to one cm-1, so a click 15 cm-1 off a mode is inside its
    20 cm-1 band."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n"
        "__click(host, 515);\n"
        "console.log(JSON.stringify(picked));"
    )
    assert got == [2]


def test_a_click_in_a_gap_reports_nothing_at_all():
    """§ 8.3 — not `null`: the event means a user picked a mode."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n"
        "__click(host, 300);\n"
        "console.log(JSON.stringify(picked));"
    )
    assert got == []


def test_the_index_reported_is_the_one_that_was_handed_in():
    """§ 6.1 — the caller's numbering is carried, not interpreted."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n__click(host, 900);\n"
        "console.log(JSON.stringify(picked));",
        modes=[
            {"index": 41, "freq": 900.0, "activity": 1.0, "imaginary": False},
            {"index": 7, "freq": 100.0, "activity": 1.0, "imaginary": False},
        ],
    )
    assert got == [41]


def test_setting_the_selection_never_reports_a_click():
    """§ 8.3 — the event means a user clicked, not the selection changed."""
    got = drive(
        "chart.setModes(MODES);\nchart.setSelected(2);\nchart.setSelected(null);\n"
        "console.log(JSON.stringify(picked));"
    )
    assert got == []


def test_a_chart_with_no_onSelect_still_draws():
    """§ 5.2 — a figure nobody is meant to click is a legitimate thing to be."""
    program = (
        f"const {{ mount }} = await import({json.dumps(ENTRY.resolve().as_uri())});\n"
        "const host = globalThis.__host();\n"
        "const chart = await mount(host);\n"
        f"chart.setModes({json.dumps(MODES)});\n"
        "globalThis.__click(host, 100);\n"
        "console.log(JSON.stringify(__calls.filter(c => c.call === 'react').length > 0));"
    )
    assert run_node([], program, globals_js=BROWSER) is True


# --- § 8.2  the mount options ------------------------------------------------

def test_a_mount_option_is_the_first_write_through_the_same_door():
    """§ 8.2 — `mount({broadening: 20})` and `setBroadening(20)` reach the same
    one place, so the chart ends in the same state either way.

    What is compared is the state, not the calls: through the options the
    selection is known before the last draw, through the doors it arrives after
    as a recolour. Same chart, different route — which is the point.
    """
    onscreen = """
    function onscreen() {
        const react = __calls.filter(c => c.call === "react").pop();
        const restyle = __calls.filter(c => c.call === "restyle").pop();
        const bar = react.traces.filter(t => t.type === "bar")[0];
        return {
            x: bar.x, y: bar.y,
            colours: restyle ? restyle.update["marker.color"][0] : bar.marker.color,
            curve: react.traces.filter(t => t.mode === "lines").map(t => t.y.length),
            note: (react.layout.annotations[0] || {}).text || "",
        };
    }
    """
    program = (
        f"const {{ mount }} = await import({json.dumps(ENTRY.resolve().as_uri())});\n"
        f"const MODES = {json.dumps(MODES)};\n"
        + onscreen +
        "const a = await mount(globalThis.__host(), { modes: MODES, broadening: 20, selected: 2 });\n"
        "const viaOptions = onscreen();\n"
        "__calls.length = 0;\n"
        "const b = await mount(globalThis.__host());\n"
        "b.setModes(MODES); b.setBroadening(20); b.setSelected(2);\n"
        "const viaDoors = onscreen();\n"
        "console.log(JSON.stringify({ viaOptions, viaDoors }));"
    )
    got = run_node([], program, globals_js=BROWSER)
    assert got["viaOptions"] == got["viaDoors"]
    assert got["viaOptions"]["colours"] == ["#3333ff", "#00ff00", "#3333ff"]


def test_the_host_is_never_resized_by_the_module():
    """§ 8.2 — the host sizes the chart, not the other way round."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\nchart.refit();\n"
        "console.log(JSON.stringify({ w: host.style.width || null, h: host.style.height || null }));"
    )
    assert got == {"w": None, "h": None}


def test_the_entry_resolves_by_the_path_the_page_uses():
    """§ 4 — one entry point, reached the way a browser reaches it.

    The unit tests above import by file path; the Spectrum tab imports
    "/static/lib/spectrumchart/index.js". This walks the real module graph
    through that specifier, so a wrong relative import inside the package
    fails here rather than in a browser.
    """
    static = Path(__file__).resolve().parents[2] / "molbuilder" / "web" / "static"
    got = run_node(
        [],
        'const m = await import("/static/lib/spectrumchart/index.js");\n'
        "console.log(JSON.stringify(Object.keys(m).sort()));",
        globals_js=BROWSER,
        static_root=static,
    )
    assert got == ["mount"]


# --- the hover indicator: the band, made visible -----------------------------

def test_the_mode_a_click_would_pick_lights_up_under_the_pointer():
    """A band is invisible, so which mode a click would take is a guess until the
    chart says so."""
    calls = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n"
        "__calls.length = 0;\n__move(host, 505);\n"
        "console.log(JSON.stringify(__calls));"
    )
    assert [c["call"] for c in calls] == ["restyle", "relayout"]
    assert calls[0]["update"]["marker.color"] == [["#3333ff", "#cccc00", "#3333ff"]]
    assert "mode 2" in list(calls[1]["patch"].values())[0]


def test_sliding_along_inside_one_band_costs_nothing():
    """§ 5.1 — a pointer fires hundreds of times a second; the chart redraws only
    when the ANSWER changes."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n"
        "__move(host, 495);\n__calls.length = 0;\n"
        "for (const x of [496, 498, 500, 502, 505]) __move(host, x);\n"
        "console.log(JSON.stringify(__calls.length));"
    )
    assert got == 0


def test_leaving_the_plot_puts_the_indicator_out():
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n__move(host, 500);\n"
        "__calls.length = 0;\n__leave(host);\n"
        "console.log(JSON.stringify(__calls));"
    )
    colours = [c["update"]["marker.color"][0] for c in got if c["call"] == "restyle"]
    assert colours == [["#3333ff", "#3333ff", "#3333ff"]]
    assert readouts(got) == [""]


def test_what_you_picked_does_not_flicker_away_under_the_pointer():
    """Chosen outranks hovered — so pointing at the mode you already picked
    changes the words and not one colour, and no colour is repainted to say so."""
    got = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\nchart.setSelected(2);\n"
        "__calls.length = 0;\n__move(host, 500);\n"
        "console.log(JSON.stringify(__calls));"
    )
    assert [c["call"] for c in got] == ["relayout"]
    assert "mode 2" in list(got[0]["patch"].values())[0]


# --- the readout: what am I near, at every moment -----------------------------

def readouts(calls):
    """Every line of text the chart put on screen, in order."""
    out = []
    for c in calls:
        if c["call"] == "react":
            notes = c["layout"]["annotations"]
            out.append(notes[-1]["text"] if notes else "")
        elif c["call"] == "relayout":
            out.extend(v for k, v in c["patch"].items() if "text" in k)
    return out


def test_the_mode_nearest_the_pointer_is_named_at_every_position():
    """Naming what you are near answers a question that has an answer everywhere
    on the plot, unlike "what would a click take"."""
    got = readouts(drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n__calls.length = 0;\n"
        "__move(host, 100);\n__move(host, 300);\n__move(host, 880);\n"
        "console.log(JSON.stringify(__calls));"
    ))
    # Three moves, two lines: 300 is a gap where mode 1 is still the nearest, so
    # the words do not change and nothing is redrawn to repeat them.
    assert got == [
        "mode 1  ·  100.0 cm⁻¹  ·  4.00 Å⁴/amu",
        "mode 3  ·  900.0 cm⁻¹  ·  2.00 Å⁴/amu",
    ]


def test_a_gap_names_the_nearest_without_lighting_anything_up():
    """The two questions have different answers there, and the chart shows both:
    the words say what is near, the colours say what a click would take."""
    calls = drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n__calls.length = 0;\n"
        "__move(host, 300);\n"
        "console.log(JSON.stringify(__calls));"
    )
    # Nothing lights up — a click at 300 would take nothing — so no colours are
    # repainted at all, and the only thing that happens is the words.
    assert [c["call"] for c in calls] == ["relayout"]
    assert "mode 1" in readouts(calls)[0]


def test_leaving_the_plot_takes_the_words_with_it():
    got = readouts(drive(
        "chart.setModes(MODES);\nchart.setBroadening(20);\n__move(host, 100);\n"
        "__calls.length = 0;\n__leave(host);\n"
        "console.log(JSON.stringify(__calls));"
    ))
    assert got == [""]


def test_the_readout_is_drawn_in_the_plot_not_beside_it():
    """§ 6.2 — the module's markup is a frame and a surface; every word the user
    reads is drawn on the surface."""
    got = drive(
        "chart.setModes(MODES);\n__move(host, 100);\n"
        "console.log(JSON.stringify({ frame: host.children.map(c => c.className),"
        " inside: host.children[0].children.map(c => c.className) }));"
    )
    assert got == {"frame": ["spectrumchart"], "inside": ["spectrumchart-surface"]}

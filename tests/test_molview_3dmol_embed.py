"""The sealed layer — every test derived from ``docs/web/molview.md``, never from
the source it checks (§ 13).

Step B of the rebuild (``docs/web/molview-rework-plan.md``). The rows of § 13.3
guarded here:

    § 9.9  the sealed layer faces downward only
    § 9.8  the drawing commands answer nothing upward
    § 10.10 the offered frames are drawable / only the master copy's count is offered
    § 6.1  one frame is not a special case
    § 6.5  the highlight is content, not styling
    § 10.6 shapes move with the frames
    § 10.7 a selection never restyles the model
    § 10.1 one render place
    § 6.7  no file route
    § 4    the module is self-contained

The library beneath is a stand-in that obeys the library's own behaviour
(``tests/support/molview_3dmol_standin.js``), per § 13.1.

Nothing here pins a list of method names. § 13.1: "A pinned list of method names
is a transcription, not a contract." Where a test needs the whole surface it
enumerates it at run time and asserts the RULE over whatever it finds.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
EMBED = MODULE_DIR / "3dmol-embed.js"
STANDIN = Path(__file__).parent / "support" / "molview_3dmol_standin.js"

# Distinctive coordinates: if one of these ever comes back out of a door, the
# layer is answering a question about the truth (§ 9.9).
FRAME_0 = [[0.0, 0.0, 0.0], [1.11, 0.0, 0.0]]
FRAME_1 = [[5.55, 0.0, 0.0], [6.66, 0.0, 0.0]]

PRELUDE = f"""
const MOD = await import({json.dumps(EMBED.resolve().as_uri())});
const ELEMENTS = ["C", "O"];
const F0 = {json.dumps(FRAME_0)};
const F1 = {json.dumps(FRAME_1)};

function fresh(frames) {{
    globalThis.__resetCalls();
    const e = MOD.create(globalThis.__makeHost(), {{}});
    if (frames !== null) e.loadFrames(ELEMENTS, frames || [F0, F1]);
    return e;
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=STANDIN.read_text())


# ---------------------------------------------------------------------------
# § 9.9 — the sealed layer faces downward only
# ---------------------------------------------------------------------------

def test_nothing_that_comes_back_out_depends_on_which_frame_is_shown():
    """§ 9.9: the shown frame cannot be read out of it.

    Every door is called twice over identical data — once with frame 0 drawn and
    once with frame 1 — and the two sets of answers must be the same. If any
    door leaked the playhead, the frame-1 sweep would differ.

    Each door gets a fresh viewer so that one whose job is to tear the viewer
    down cannot affect the rest, and so the sweep needs no list of names.
    """
    out = _run(
        """
        async function sweep(atFrame) {
            const names = Object.keys(fresh());
            const answers = {};
            for (const k of names) {
                const e = fresh();
                if (atFrame) e.showFrame(atFrame);
                let r;
                try { r = e[k](); }
                catch (err) { r = "threw"; }
                if (r && typeof r.then === "function") {
                    r = await r.then(v => "resolved", () => "rejected");
                }
                answers[k] = JSON.stringify(r === undefined ? null : r);
            }
            return answers;
        }
        const atZero = await sweep(0);
        const atOne  = await sweep(1);
        const differ = Object.keys(atZero).filter(k => atZero[k] !== atOne[k]);
        console.log(JSON.stringify({
            doors: Object.keys(atZero).length,
            differ: differ,
            answers: atZero,
        }));
        """
    )
    assert out["doors"] > 0, "no surface to check"
    assert out["differ"] == [], (
        "these doors answer differently depending on which frame is drawn, "
        f"so the shown frame can be read out of the sealed layer: {out['differ']}"
    )


def test_no_door_hands_back_a_coordinate():
    """§ 9.9: "There is no way to read coordinates out."

    The atoms are loaded at coordinates that appear nowhere else, then every
    door's answer is searched for them.
    """
    out = _run(
        """
        async function answers() {
            const names = Object.keys(fresh());
            const seen = [];
            for (const k of names) {
                const e = fresh();
                let r;
                try { r = e[k](); }
                catch (err) { r = "threw"; }
                if (r && typeof r.then === "function") {
                    r = await r.then(v => v, () => "rejected");
                }
                seen.push(k + "=" + JSON.stringify(r === undefined ? null : r));
            }
            return seen.join(" | ");
        }
        const blob = await answers();
        console.log(JSON.stringify({
            blob: blob,
            leaks: ["1.11", "5.55", "6.66"].filter(c => blob.indexOf(c) >= 0),
        }));
        """
    )
    assert out["leaks"] == [], (
        f"coordinates came back out of the sealed layer: {out['leaks']}\n{out['blob']}"
    )


def test_the_camera_cannot_be_read_back():
    """§ 9.6 / § 9.9: nothing reports where the camera is pointing.

    Moving the camera must change no answer the layer gives.
    """
    out = _run(
        """
        async function sweep(moveCamera) {
            const names = Object.keys(fresh());
            const answers = {};
            for (const k of names) {
                const e = fresh();
                if (moveCamera) e.fitCamera();
                let r;
                try { r = e[k](); } catch (err) { r = "threw"; }
                if (r && typeof r.then === "function") {
                    r = await r.then(() => "resolved", () => "rejected");
                }
                answers[k] = JSON.stringify(r === undefined ? null : r);
            }
            return answers;
        }
        const before = await sweep(false);
        const after  = await sweep(true);
        console.log(JSON.stringify({
            differ: Object.keys(before).filter(k => before[k] !== after[k]),
        }));
        """
    )
    assert out["differ"] == [], (
        f"the camera can be read back through: {out['differ']}"
    )


def test_the_two_self_checks_agree_with_what_is_drawn():
    """§ 10.10 + § 13.1: the layer must never report frames while claiming no
    movie, nor claim a movie with no frames — the incoherence § 13.1 names.

    These two answers exist so the layer above can check its own last
    instruction landed, so they have to track the drawing exactly.
    """
    out = _run(
        """
        const e = fresh(null);                       // nothing loaded yet
        const empty = { movie: e.hasMovie(), frames: e.drawnFrameCount() };

        e.loadFrames(ELEMENTS, [F0, F1]);
        const loaded = { movie: e.hasMovie(), frames: e.drawnFrameCount() };

        e.appendFrames([F0, F1, F0]);
        const grown = { movie: e.hasMovie(), frames: e.drawnFrameCount() };

        console.log(JSON.stringify({ empty, loaded, grown }));
        """
    )
    assert out["empty"] == {"movie": False, "frames": 0}, (
        "with nothing loaded the layer must report no movie and no frames"
    )
    assert out["loaded"] == {"movie": True, "frames": 2}
    assert out["grown"] == {"movie": True, "frames": 5}, (
        "appended frames must show up in the drawing's own count"
    )


def test_appending_with_no_movie_reports_that_it_did_not_land():
    """§ 10.10: "appending to a structure with no movie rebuilds instead of
    extending nothing."

    The rebuild decision is the layer above's; what this layer owes it is an
    honest answer that the append had nothing to extend.
    """
    out = _run(
        """
        const e = fresh(null);
        const landed = e.appendFrames([F0]);
        console.log(JSON.stringify({ landed, movie: e.hasMovie(), frames: e.drawnFrameCount() }));
        """
    )
    assert out["landed"] is False
    assert out["movie"] is False and out["frames"] == 0


# ---------------------------------------------------------------------------
# § 6.1 — one frame is not a special case
# ---------------------------------------------------------------------------

def test_one_frame_takes_the_same_path_as_many():
    """§ 6.1: no path treats a single frame differently from four hundred."""
    out = _run(
        """
        function trace(frames) {
            const e = fresh(frames);
            return {
                calls: globalThis.__callNames(),
                movie: e.hasMovie(),
                frames: e.drawnFrameCount(),
            };
        }
        const one  = trace([F0]);
        const many = trace([F0, F1, F0, F1]);
        console.log(JSON.stringify({ one, many }));
        """
    )
    assert out["one"]["movie"] is True and out["one"]["frames"] == 1
    assert out["many"]["frames"] == 4
    assert out["one"]["calls"] == out["many"]["calls"], (
        "a one-frame structure took a different path through the drawing than a "
        f"four-frame one:\n one:  {out['one']['calls']}\n many: {out['many']['calls']}"
    )


# ---------------------------------------------------------------------------
# § 10.7 — a selection never restyles the model
# ---------------------------------------------------------------------------

def test_a_selection_adds_shapes_and_never_restyles():
    """§ 10.7: "a click adds or removes shapes and issues no model restyle, and
    its cost does not grow with atom count."

    Restyling is what rebuilds the whole model's geometry, so a selection that
    restyled would cost the structure's size on every click.
    """
    out = _run(
        """
        function pickCost(atomCount) {
            const els = Array.from({length: atomCount}, () => "C");
            const frame = Array.from({length: atomCount}, (_, i) => [i, 0, 0]);
            globalThis.__resetCalls();
            const e = MOD.create(globalThis.__makeHost(), {});
            e.loadFrames(els, [frame]);
            globalThis.__resetCalls();                 // measure only the pick
            e.setOverlays({ highlight: [0, 1] });
            return {
                setStyle: globalThis.__countCalls("setStyle"),
                spheres:  globalThis.__countCalls("addSphere"),
                total:    globalThis.__calls.length,
            };
        }
        console.log(JSON.stringify({ small: pickCost(4), large: pickCost(400) }));
        """
    )
    assert out["small"]["setStyle"] == 0, "a selection restyled the model"
    assert out["large"]["setStyle"] == 0, "a selection restyled the model"
    assert out["small"]["spheres"] == 2, "the highlight is drawn as shapes"
    assert out["small"]["total"] == out["large"]["total"], (
        "the cost of picking two atoms grew with the structure: "
        f"{out['small']['total']} vs {out['large']['total']}"
    )


# ---------------------------------------------------------------------------
# § 10.6 — shapes move with the frames
# ---------------------------------------------------------------------------

def test_overlays_sit_on_the_new_positions_after_a_swap():
    """§ 10.6: "after a swap, labels and the highlight sit on the atoms' new
    positions, not where frame 0 left them."

    Nothing above re-sends the overlays for this to be true.
    """
    out = _run(
        """
        const e = fresh();
        e.setOverlays({ highlight: [0], labels: [{ atom: 0, text: "42" }] });
        const at0 = {
            halo:  globalThis.__lastCall("addSphere").args[0].center,
            label: globalThis.__lastCall("addLabel").args[1].position,
            text:  globalThis.__lastCall("addLabel").args[0],
        };
        e.showFrame(1);
        const at1 = {
            halo:  globalThis.__lastCall("addSphere").args[0].center,
            label: globalThis.__lastCall("addLabel").args[1].position,
            text:  globalThis.__lastCall("addLabel").args[0],
        };
        console.log(JSON.stringify({ at0, at1 }));
        """
    )
    assert out["at0"]["halo"]["x"] == 0.0
    assert out["at1"]["halo"]["x"] == 5.55, (
        "the highlight stayed where frame 0 left it after a frame swap"
    )
    assert out["at1"]["label"]["x"] == 5.55, (
        "the label stayed where frame 0 left it after a frame swap"
    )
    assert out["at0"]["text"] == "42" and out["at1"]["text"] == "42", (
        "the label's text is the caller's, and a frame swap must not rewrite it "
        "— under isolate it carries the ORIGINAL atom number (§ 6.5)"
    )


# ---------------------------------------------------------------------------
# § 6.5 — appearance is decided here, not carried on the data
# ---------------------------------------------------------------------------

def test_arrows_arrive_as_geometry_and_are_given_their_appearance_here():
    """§ 6.5: per-frame data says which arrows exist and where they point, and
    nothing else. What they look like is this layer's decision, taken from the
    set as a whole — so the largest force reads differently from the rest.
    """
    out = _run(
        """
        const e = fresh();
        e.setArrows([
            { start: [0,0,0], end: [1,0,0] },      // small
            { start: [0,0,0], end: [4,0,0] },      // the largest of the set
        ]);
        const drawn = globalThis.__calls
            .filter(c => c.name === "addArrow" || c.name === "addArrow:batched")
            .map(c => c.args[0]);
        console.log(JSON.stringify({
            drawn: drawn.length,
            colors:  drawn.map(a => a.color),
            radii:   drawn.map(a => a.radius),
            distinct: new Set(drawn.map(a => a.color)).size,
        }));
        """
    )
    assert out["drawn"] == 2
    assert all(c for c in out["colors"]), (
        "an arrow reached the drawing with no colour — appearance must be "
        "decided in this layer, since the data carries none"
    )
    assert all(isinstance(r, (int, float)) for r in out["radii"]), (
        "an arrow reached the drawing with no radius"
    )
    assert out["distinct"] == 2, (
        "the largest arrow of the set must read differently from the rest"
    )


def test_an_arrow_that_brings_its_own_colour_keeps_it():
    """§ 6.5 again, from the other side: the axis triad hands its own per-axis
    colours through this same door, and those are its to choose.
    """
    out = _run(
        """
        const e = fresh();
        e.setArrows([
            { start: [0,0,0], end: [1,0,0], color: "#ff0000" },
            { start: [0,0,0], end: [9,0,0] },
        ]);
        const drawn = globalThis.__calls
            .filter(c => c.name === "addArrow" || c.name === "addArrow:batched")
            .map(c => c.args[0]);
        console.log(JSON.stringify({ colors: drawn.map(a => a.color) }));
        """
    )
    assert out["colors"][0] == "#ff0000", (
        "a caller's own arrow colour was overwritten by this layer"
    )


# ---------------------------------------------------------------------------
# § 10.1 — one render place
# ---------------------------------------------------------------------------

def test_a_batch_paints_once_however_many_doors_it_touches():
    """§ 10.1: every interaction is a write followed by ONE render.

    Without this, a change that touches four doors paints four times and the
    user sees the screen assemble itself.
    """
    out = _run(
        """
        const e = fresh();
        globalThis.__resetCalls();
        e.beginBatch();
        e.setOverlays({ highlight: [0] });
        e.setArrows([{ start: [0,0,0], end: [1,0,0] }]);
        e.setCell({ lattice: [[4,0,0],[0,4,0],[0,0,4]], origin: [0,0,0] });
        e.showFrame(1);
        const during = globalThis.__countCalls("render");
        e.endBatch();
        const after = globalThis.__countCalls("render");

        // Nested opens must still paint once, at the outermost close.
        globalThis.__resetCalls();
        e.beginBatch();
        e.beginBatch();
        e.setArrows([]);
        e.endBatch();
        const nestedInner = globalThis.__countCalls("render");
        e.endBatch();
        const nestedOuter = globalThis.__countCalls("render");

        console.log(JSON.stringify({ during, after, nestedInner, nestedOuter }));
        """
    )
    assert out["during"] == 0, "the screen was painted while a batch was open"
    assert out["after"] == 1, (
        f"a batch touching four doors painted {out['after']} times, not once"
    )
    assert out["nestedInner"] == 0, "a nested batch painted at the inner close"
    assert out["nestedOuter"] == 1


# ---------------------------------------------------------------------------
# § 4, § 5.3, § 6.7 — what the module as a whole must not contain
# ---------------------------------------------------------------------------

def _module_sources():
    return sorted(p for p in MODULE_DIR.iterdir() if p.suffix in {".js", ".css"})


def test_the_graphics_library_is_named_in_exactly_one_file():
    """§ 4: "within this module the name 3Dmol occurs in exactly one file."
    § 5.3: everything above it could be read end to end without learning which
    library draws the molecule.
    """
    named = [p.name for p in _module_sources() if "3Dmol" in p.read_text()]
    assert named == ["3dmol-embed.js"], (
        f"the graphics library is named outside the sealed layer: {named}"
    )


def _module_code():
    """Every module source with its comments stripped — comments explain what was
    deleted and why, and a rule about what the code DOES must not fire on them."""
    out = {}
    for path in _module_sources():
        out[path.name] = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.lstrip().startswith(("*", "//", "/*"))
        )
    return out


def test_the_module_owns_no_file_route():
    """§ 6.7: "MolView owns no file route" — files on disk are the projects
    module's. `exportFile()` returns bytes and stops there (§ 11.3: "neither has
    MolView writing a file").

    Note what this does NOT ban: the network. § 11.1 says MolView calls three
    routes, so a blanket ban on `fetch` would contradict the contract — see
    ``test_the_module_calls_only_the_routes_named_in_the_contract``.
    """
    offenders = {}
    for name, code in _module_code().items():
        # NOT banned: the anchor `download` attribute. § 11.3 names it
        # explicitly — "save-to-project and download differ only in destination
        # … and NEITHER has MolView writing a file". Handing the browser bytes
        # the user asked for is the Export menu's whole job; what § 6.7 forbids
        # is a file ROUTE and a filesystem handle.
        hits = [t for t in ("/api/files/", "showSaveFilePicker", "createWritable",
                            "FileSystemHandle", "require(\"fs\")") if t in code]
        if hits:
            offenders[name] = hits
    assert offenders == {}, f"the module reaches a file route: {offenders}"


def test_the_module_calls_only_the_routes_the_contract_describes():
    """Every route the module reaches must be one the contract describes, so that
    a fact leaving by an undescribed path is visible — which is how the file
    route got in last time.

    § 11.1's sentence says THREE: load a structure, perform one geometry edit,
    resolve a cell. § 9.5 describes a fourth in prose — "filtering is a question
    asked of the server, not a scan done here" — which that sentence omits. The
    plan records the wording as an open item; this test counts what the module
    actually reaches, so the gap is stated rather than papered over.

    If a fifth appears, either the contract gained a route or the module grew
    one it should not have. Both are worth stopping for.
    """
    import re

    found = set()
    for code in _module_code().values():
        for hit in re.findall(r'"(/api/[^"]*)"', code):
            found.add(hit.split("/api/")[1].split('" +')[0])
    # `/api/modify/` is completed with the operation name, which IS the route
    # segment (§ 11.1), so it appears as a prefix.
    assert found == {
        "build/load",                # § 11.1 — load a structure
        "modify/",                   # § 11.1 — one geometry edit
        "structure/periodicity",     # § 11.1 — resolve a cell
        "selection/eval",            # § 9.5  — the filter, omitted by § 11.1's count
    }, f"the module reaches a route the contract does not describe: {sorted(found)}"


def test_the_module_neither_publishes_nor_reads_a_global():
    """§ 4: "it is imported by name, it reaches nothing else by name, and nothing
    in the app can reach inside it … Nothing it needs comes from a global."

    The harness creates an empty ``window.molbuilder`` before the import, so a
    module that publishes into it leaves a key behind.
    """
    out = _run(
        """
        const e = fresh();
        e.setOverlays({ highlight: [0] });
        console.log(JSON.stringify({
            published: Object.keys(globalThis.molbuilder),
        }));
        """
    )
    assert out["published"] == [], (
        f"the module published into the app's global namespace: {out['published']}"
    )


def test_the_app_namespace_is_never_typed_in_the_source():
    """§ 4 in both directions. Publishing and reading are the same mechanism, so
    a rebuild that removes one and leaves the other has sealed nothing.
    """
    offenders = {}
    for path in _module_sources():
        text = path.read_text()
        code = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith(("*", "//", "/*"))
        )
        if "molbuilder." in code or "window.molbuilder" in code:
            offenders[path.name] = [
                line.strip() for line in code.splitlines() if "molbuilder." in line
            ]
    assert offenders == {}, f"the app's global namespace is named in: {offenders}"

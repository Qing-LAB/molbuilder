"""Mounting, and the handle — every test derived from ``docs/web/molview.md``,
never from the source it checks (§ 13).

Step G of the rebuild (``docs/web/molview-rework-plan.md``). The rows of § 13.3
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
globalThis.fetch = async function (route, init) {
    globalThis.__requests.push({ route, body: JSON.parse(init.body) });
    return { ok: true, status: 200, json: async () => ({ atoms: [
        { index: 0, element: "C", x: 0, y: 0, z: 0, regions: [], is_frozen: false },
        { index: 1, element: "O", x: 1, y: 0, z: 0, regions: [], is_frozen: false },
    ] }) };
};
globalThis.setInterval = globalThis.setInterval;
"""

PRELUDE = f"""
const MV = await import({json.dumps(INDEX.resolve().as_uri())});

const workspace = {{ read: async () => null, write: async () => {{}} }};

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
        const card = host.querySelector(".molview-card");
        console.log(JSON.stringify({
            ok: viewer.ok,
            hasCard: !!card,
            hasViewer: !!(card && card.querySelector(".molview-viewer")),
            hasPanel: !!(card && card.querySelector(".molview-panel")),
            errorText: card && card.querySelector(".molview-mount-error")
                ? card.querySelector(".molview-mount-error").textContent : null,
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
        viewer.play({ fps: 1000 });
        await new Promise(r => setTimeout(r, 20));
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
        const card = host.querySelector(".molview-card");
        const fold = card.querySelector(".molview-fold-btn");
        const chevron = card.querySelector(".molview-fold-chevron");

        const before = { folded: card.classList.contains("is-folded"),
                         expanded: fold.getAttribute("aria-expanded"),
                         btnStyle: JSON.stringify(fold.style) };
        fold.click();
        const after = { folded: card.classList.contains("is-folded"),
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
        const bar = host.querySelector(".molview-frame-controls");
        const atMount = bar.hidden;

        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const oneFrame = bar.hidden;

        viewer.data.reloadFrames([[[0,0,0],[1,0,0]], [[2,0,0],[3,0,0]]]);
        const manyFrames = bar.hidden;

        viewer.data.setCurrentFrame(1);
        const counter = bar.querySelector(".molview-frame-counter").textContent;
        console.log(JSON.stringify({ atMount, oneFrame, manyFrames, counter }));
        """
    )
    assert out["atMount"] is True, "no structure, no bar"
    assert out["oneFrame"] is True, "one frame is not a trajectory"
    assert out["manyFrames"] is False, "the bar appears once there is a movie"
    assert out["counter"] == "2 / 2", (
        f"the counter reads 1-based, through the one translation: {out['counter']}"
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
    offenders = {}
    for path in sorted(MODULE_DIR.glob("*.js")):
        code = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.lstrip().startswith(("*", "//", "/*"))
        )
        hits = [t for t in ("createObjectURL", ".download =", "showSaveFilePicker",
                            "createWritable", "FileSystemHandle", "/api/files/")
                if t in code]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}, (
        f"the module handles files itself instead of through the door: {offenders}"
    )


def test_bytes_leave_through_the_door_and_the_sidecar_goes_with_them():
    """§ 11.3: "the `.json` goes with the `.xyz`, so labels and frozen atoms
    survive into whatever is generated from it", and "save-to-project and
    download differ ONLY in destination — both produce identical bytes."

    One call with a destination argument, not two paths to keep in step: a
    separate download path is how the sidecar came to be dropped from one of them.
    """
    out = _run(
        """
        const saved = [];
        const host = globalThis.__makeHost();
        const viewer = await MV.mount(host, workspace, {
            owner: "x",
            files: { save: (destination, filename, contents) =>
                        saved.push({ destination, filename, length: contents.length }) },
        });
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });

        const card = host.querySelector(".molview-card");
        const items = card.querySelectorAll(".mol-viewer-menu-item");
        for (const item of items) item.click();

        console.log(JSON.stringify({
            saved,
            destinations: saved.map(s => s.destination),
            names: saved.map(s => s.filename),
        }));
        """
    )
    assert out["destinations"] == ["project", "project", "download", "download"], (
        "both destinations must go through the same door, differing only in the "
        f"destination they name: {out['destinations']}"
    )
    assert out["names"] == ["structure.xyz", "structure.molstruct.json",
                            "structure.xyz", "structure.molstruct.json"], (
        f"the sidecar must go with the geometry, both times: {out['names']}"
    )
    assert all(s["length"] > 0 for s in out["saved"])


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

        const card = host.querySelector(".molview-card");
        const clickBody = card.querySelector(".selection-click-section");
        const filterBody = card.querySelector(".selection-filter-section");

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


def test_the_atom_list_reads_one_based_and_a_click_goes_through_the_store():
    """§ 11.5: the first atom reads as #1 everywhere even though the code sees 0.

    And the panel holds nothing: a click asks the STORE to change, and the list
    redraws from the snapshot that comes back (§ 8.4).
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        const card = host.querySelector(".molview-card");
        const rows = card.querySelectorAll(".selection-atom-table")[0].children;

        const numbers = Array.from(rows).map(r => r.children[0].textContent);
        rows[1].click();                       // the SECOND atom, which reads #2
        console.log(JSON.stringify({
            numbers,
            selection: viewer.data.selection.get(),
            count: card.querySelector(".selection-count").textContent,
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


def test_a_filter_row_is_added_typed_and_removed_from_the_panel():
    """§ 8.4: "a user adds a row, types in it, changes its kind, removes it" —
    each its own change, because that is what the controls are.
    """
    out = _run(
        """
        const { host, viewer } = await mounted();
        await viewer.data.installMolecule({ text: "x", filename: "x.xyz" });
        viewer.data.selection.setEditor("filter");
        const card = host.querySelector(".molview-card");

        const empty = card.querySelector(".selection-filter-empty") !== null;
        card.querySelector(".selection-add-filter-row").click();
        const afterAdd = viewer.data.selection.getState().filters;

        const text = card.querySelector(".selection-filter-text");
        text.value = "Au";
        text.dispatch("input", { target: text });
        const afterType = viewer.data.selection.getState().filters;

        card.querySelector(".selection-filter-remove").click();
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
            const card = host.querySelector(".molview-card");
            return {
                assignHidden: card.querySelector(".selection-assign").hidden,
                isolateOffered: card.querySelector(".selection-mode-option") !== null,
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

"""Task Setup tab — the page renders, and it renders the design's shape.

`docs/web/task-setup.md` is the contract. This pins the parts a reader of that
document would expect to find on the page, plus the two properties that are
easy to lose in a later edit:

  * the tab **writes nothing today** — Save is disabled and says why. A future
    change that wires saving has to change this test deliberately, which is the
    point: enabling a write path by accident is the failure worth catching.
  * the tab sheet is **layer 5** of `docs/web/ui-contract.md` § 1 — composition
    only, every value a token. A raw hex colour or a raw px/rem spacing value in
    `task-setup/style.css` is the drift that made the older tab sheets
    unmaintainable, and it is checkable by reading the file.

The CSS check is an L2 source-text invariant (`docs/process/testing.md`): no
browser, no JS runtime, just a read of the stylesheet the page loads.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT   = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"
SHEET  = STATIC / "task-setup/style.css"
VIEWER = STATIC / "task-setup/viewer.js"


# --------------------------------------------------------------------- #
#  The page                                                             #
# --------------------------------------------------------------------- #

def test_the_tab_is_in_the_roster_and_routes():
    """The nav order is one place (`web/tabs.py`); the route matches its path."""
    from molbuilder.web.tabs import TABS
    entry = next((t for t in TABS if t["key"] == "task-setup"), None)
    assert entry is not None, "task-setup missing from TABS"
    assert entry["path"] == "/task-setup"
    assert entry["label"] == "Task setup"


def test_the_page_renders_the_designs_parts(web_client):
    body = web_client.get("/task-setup").data.decode()
    for needle in (
        'id="ts-path"',            # § 2  where it saves
        'id="ts-state"',           # § 2  the three folder states
        'id="ts-stage-table"',     # § 5  stages
        'id="ts-machine-rows"',    # § 6  the machine half
        'id="ts-file-list"',       # what will be written
        'id="ts-editor"',          # the CodeMirror mount
        "task-setup/style.css",
        "task-setup/viewer.js",
    ):
        assert needle in body, f"missing {needle!r} on /task-setup"


def test_save_is_disabled_until_it_could_succeed(web_client):
    """Save now exists (T2).  It renders DISABLED and the page says why —
    the button is enabled by `refreshSave()` only once a folder is open, the
    folder has something to save, and a hand-over has been given its shape.

    This replaced `test_the_tab_writes_nothing_yet` when the write path landed,
    which is what that test was for: enabling a write path had to be a
    deliberate edit, not a side effect.
    """
    body = web_client.get("/task-setup").data.decode()
    m = re.search(r"<button[^>]*id=\"ts-save\"[^>]*>", body)
    assert m, "the Save button is gone"
    assert "disabled" in m.group(0), (
        "Save renders enabled — nothing is open yet, so it cannot succeed")
    src = VIEWER.read_text()
    assert "refreshSave" in src, "nothing decides when Save is usable"
    assert "/api/task-setup/save" in src, "Save is not wired to the endpoint"


def test_the_page_loads_the_shared_stylesheet_layers(web_client):
    """Layers 1-4 before layer 5, so tokens land first (`ui-contract.md` § 1)."""
    body = web_client.get("/task-setup").data.decode()
    order = [body.index(x) for x in (
        "lib/tokens.css", "lib/page-shell.css",
        "lib/form-components.css", "task-setup/style.css")]
    assert order == sorted(order), "stylesheet layers load out of order"


# --------------------------------------------------------------------- #
#  The stylesheet is composition only                                   #
# --------------------------------------------------------------------- #

def _declarations(css: str) -> list[tuple[str, str]]:
    """(property, value) pairs, comments stripped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(m.group(1).strip(), m.group(2).strip())
            for m in re.finditer(r"([a-z-]+)\s*:\s*([^;{}]+)[;}]", css)]


def test_the_sheet_writes_no_raw_palette_colour():
    """`ui-contract.md` § 2: components never write a raw palette colour."""
    offenders = [f"{p}: {v}" for p, v in _declarations(SHEET.read_text())
                 if re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(", v)]
    assert not offenders, (
        "raw colours in task-setup/style.css — use a var(--token) from "
        f"lib/tokens.css: {offenders}")


def test_the_sheet_takes_spacing_and_type_from_the_scales():
    """No magic numbers for spacing, radius or type size.

    The `--space-*` / `--text-*` / `--radius*` scales exist so the rhythm is
    uniform and retunable in one file.  Exempt: `0`, and the handful of
    properties whose value is genuinely not a scale step (border widths,
    percentages, `1px` hairlines, and the media-query breakpoints, which are
    not declarations at all).
    """
    scale_props = (
        "padding", "padding-top", "padding-right", "padding-bottom",
        "padding-left", "margin", "margin-top", "margin-right",
        "margin-bottom", "margin-left", "gap", "row-gap", "column-gap",
        "font-size", "border-radius", "top",
    )
    offenders = []
    for prop, val in _declarations(SHEET.read_text()):
        if prop not in scale_props:
            continue
        for tok in val.split():
            if re.fullmatch(r"-?\d*\.?\d+(px|rem|em)", tok) and not tok.startswith("0"):
                offenders.append(f"{prop}: {val}")
                break
    assert not offenders, (
        "magic numbers in task-setup/style.css — use --space-*, --text-* or "
        f"--radius*: {offenders}")


def test_every_jp_token_the_sheet_uses_is_defined():
    """A `var(--ts-x)` with no definition renders as nothing at all."""
    tokens = (STATIC / "lib/tokens.css").read_text()
    defined = set(re.findall(r"(--ts-[a-z0-9-]+)\s*:", tokens))
    used    = set(re.findall(r"var\((--ts-[a-z0-9-]+)", SHEET.read_text()))
    assert used <= defined, f"undefined --ts-* tokens: {sorted(used - defined)}"
    assert defined, "no --ts-* tokens declared in lib/tokens.css"


def test_the_jp_tokens_live_in_the_one_palette_file():
    """`ui-contract.md` § 2: module-private tokens live in lib/tokens.css,
    promoted out of per-file :root blocks."""
    assert ":root" not in SHEET.read_text(), (
        "task-setup/style.css declares its own :root block — module tokens "
        "belong in lib/tokens.css")


# --------------------------------------------------------------------- #
#  The controller                                                       #
# --------------------------------------------------------------------- #

def test_the_editor_uses_the_shared_codemirror_loader():
    """One loader, not a second copy (`lib/codemirror-load.js`)."""
    src = VIEWER.read_text()
    assert "codemirror-load.js" in src
    assert "loadCodeMirror" in src
    for copied in ("_injectScript", "injectScript(", "codemirror.min.js"):
        assert copied not in src, (
            f"{copied!r} in task-setup/viewer.js — the bundle's asset list has "
            "ONE home, and the vendor-integrity test reads it there")


def test_the_editor_picks_its_mode_from_the_suffix():
    """Highlighting is chosen by suffix through the shared loader, not by a
    mode string typed here.

    Until 2026-08-16 the editor passed `mode: null` because the bundle carried
    no json mode.  It carries eight now, so `task.json` gets the JSON dialect —
    and the tab asks for it by PATH, so nothing here has to know which mode
    that is.
    """
    src = VIEWER.read_text()
    assert "modeFor" in src, "the editor no longer resolves its mode by path"
    assert re.search(r"modeFor\(TASK_JSON\)", src), (
        "the mode should be resolved from the file's own name")
    # Narrowly: the MODE option, not the file — `application/json` is also a
    # legitimate Content-Type header on the save fetch.
    assert not re.search(r"mode:\s*[\"']", src), (
        "a mode string is hard-coded in the tab — the suffix map in "
        "lib/codemirror-load.js is the one place that decides")


def test_the_optional_read_uses_the_camelCase_option():
    """`lib/projects/api.js` takes `missingOk` and maps it to the wire's
    `missing_ok`.  Passing the wire spelling is silently ignored — the read
    then takes the 404 path and logs a failed-resource console error for the
    perfectly normal "this folder has no description yet" case.

    Caught by reading api.js after the tab was already written and green.
    """
    src = VIEWER.read_text()
    assert "missingOk" in src, "the optional read does not pass missingOk"
    assert not re.search(r"missing_ok\s*:", src), (
        "task-setup/viewer.js passes the WIRE spelling to the projects API; "
        "api.js expects `missingOk` and silently drops the other")


def test_the_page_reads_the_current_dir_through_the_public_accessor():
    """`projects.getCurrentDir()` exists; reaching into sessionStorage for the
    sidebar's own key would put that key name in a second place."""
    src = VIEWER.read_text()
    assert "getCurrentDir" in src
    assert "molbuilder.current_dir" not in src, (
        "the tab duplicates the sidebar's sessionStorage key — call "
        "projects.getCurrentDir() instead")


def test_the_machine_answered_settings_are_named():
    """mpi_np / omp_threads / max_memory_mb may never carry a value in a
    description (`engines/template.md` § 6.4), so the page must not show them
    as a choice."""
    src = VIEWER.read_text()
    for name in ("mpi_np", "omp_threads", "max_memory_mb"):
        assert name in src, f"{name} not distinguished in the controller"
    assert "MACHINE_ANSWERED" in src


@pytest.mark.parametrize("endpoint", ["/api/files/read", "/api/files/list"])
def test_the_tab_reuses_the_shipped_files_api(endpoint, web_client):
    """No new endpoint was added for this tab; both already ship."""
    r = web_client.get(endpoint, query_string={"path": ""})
    assert r.status_code != 404, f"{endpoint} is missing"


# --------------------------------------------------------------------- #
#  The hand-over (`stages.md` § 6.5a)                                    #
# --------------------------------------------------------------------- #

import json as _json
import shutil as _shutil


def _fresh_calc_dir():
    """A directory inside the configured root — the picker refuses anything
    outside it, which is the guard working, not a test problem."""
    d = ROOT / "projects/_t_handover/optimization/probe_calc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _envelope():
    return {"structure": {"elements": ["H", "H"],
                          "positions": [[0, 0, 0], [0, 0, 0.74]],
                          "metadata": {}}}


def test_handover_renders_and_writes_nothing(web_client):
    """The endpoint returns TEXTS; the browser writes them.

    `web/projects.md` § 1 puts raw bytes in the content-blind file layer that
    "every tab can use" — a tab that writes files itself bypasses the roots
    guard, the lock, the uniform envelope and the sidebar re-list.  What is
    genuinely server-side is the RENDER: only Python can turn a config into
    `<label>.template.toml`.
    """
    d = _fresh_calc_dir()
    try:
        r = web_client.post("/api/task-setup/handover", json=dict(
            _envelope(), engine="siesta", name="probe calc",
            params={"system_label": "probe"}))
        assert r.status_code == 200, r.get_json()
        out = r.get_json()
        assert out["ok"] is True
        assert out["template_text"].strip(), "no template rendered"
        assert out["handover_name"] == "task.1st.json"
        assert list(d.iterdir()) == [], (
            "the render endpoint wrote into the folder; the browser writes, "
            "through projects.safeSave")

        h = _json.loads(out["handover_text"])
        assert h["schema"] == "molbuilder/task-handover@1"
        assert "shape" not in h and "stages" not in h
        assert h["awaiting"] == ["shape", "stages"]
        assert h["_what"], "the file does not say what it is"
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_the_tab_moves_bytes_through_the_file_layer():
    """No hand-rolled writes or deletes on either surface."""
    opt = (STATIC / "structure-optimization/viewer.js").read_text()
    ts  = VIEWER.read_text()
    assert "projects.safeSave(text, name," in opt, (
        "the hand-over does not write through safeSave(TEXT, FILENAME, opts)")
    assert "projects.deleteEntry(" in ts, (
        "the hand-over is not removed through the file layer")
    for surface, src in (("optimization", opt), ("task-setup", ts)):
        assert "/api/files/write" not in src, (
            f"{surface} calls the write route directly instead of the door")


def test_save_refuses_outside_the_roots(web_client):
    """The save door owns `task.json` because it owns that schema — but it is
    still inside the picker's roots guard like every other write."""
    r = web_client.post("/api/task-setup/save",
                        json={"dest": "/tmp", "text": "{}"})
    assert r.status_code >= 400
    assert "root" in str(r.get_json().get("error", "")).lower()


def test_the_tab_reads_the_handover_only_when_there_is_no_description():
    """A folder holding both is a save that did not finish; the description
    wins, because it is the one that passed the preflight."""
    src = VIEWER.read_text()
    assert "task.1st.json" in src, "the tab does not know the hand-over"
    assert re.search(r"taskText\s*\n?\s*\?\s*null", src) or "taskText ?" in src, (
        "the hand-over should be read only when task.json is absent")


def test_the_handover_button_calls_only_helpers_that_exist():
    """The button lives in `structure-optimization/viewer.js` because the two
    things it needs are private to that file.

    Written first as its own module against `structurePage.structureEnvelope()`
    and `formSchema.lastSchema()` — **neither of which exists**.  It would have
    parsed, loaded, and silently done nothing useful.  Same failure class as
    passing `missing_ok` where the API takes `missingOk`: a name that is wrong
    rather than code that is wrong.
    """
    src = (STATIC / "structure-optimization/viewer.js").read_text()
    assert "send-to-task-setup" in src, "the hand-over button is not wired"
    assert "/api/task-setup/handover" in src

    for helper in ("_structureForRequest", "collectFdfParams",
                   "collectPyscfParams", "_activeEngine"):
        assert f"function {helper}" in src, (
            f"{helper} is called by the hand-over but not defined here")

    for invented in ("structureEnvelope", "lastSchema"):
        assert invented not in src, (
            f"{invented} does not exist anywhere in the tree")

    assert not (STATIC / "structure-optimization/handover.js").exists(), (
        "the standalone hand-over module is back; it cannot reach the "
        "private helpers and would have to duplicate them")


# --------------------------------------------------------------------- #
#  T1 shape · T2 save                                                    #
# --------------------------------------------------------------------- #

def test_the_shape_is_asked_and_never_defaulted(web_client):
    """`stages.md § 6.7`: required with no default, because inferring it
    "would hand somebody a directory tree they never asked for"."""
    body = web_client.get("/task-setup").data.decode()
    assert 'id="ts-shape-card"' in body
    for shape in ("flat", "hierarchical"):
        assert f'data-shape="{shape}"' in body, f"no {shape} option"
    assert body.count('aria-pressed="false"') >= 2, (
        "a shape renders pre-selected — the page must ask")
    src = VIEWER.read_text()
    assert 'let _shape      = "";' in src, "shape is initialised to a value"


def test_the_editor_shows_what_will_be_written_not_the_handover(web_client):
    """A person checking a description before a week of compute should be
    reading the thing that lands, not its input."""
    src = VIEWER.read_text()
    assert "proposedFromHandover" in src
    assert '"molbuilder/task@1"' in src, (
        "the proposed description does not carry the real task schema")
    assert re.search(r'name:\s*"coarse"', src), (
        "the proposal should start with one stage named coarse (§ 6.5)")


def test_save_writes_the_description_and_reports_the_handover(web_client):
    """The save door owns `task.json` — the same reason `/api/structure/save`
    owns the sidecar: a browser-authored, schema-stamped file that the loader
    would reject is the save-then-reload trap `projects.md` § 3 describes.

    It does NOT delete the hand-over; it reports that one is there, and the
    browser removes it through `projects.deleteEntry`.
    """
    d = _fresh_calc_dir()
    try:
        rendered = web_client.post("/api/task-setup/handover", json=dict(
            _envelope(), engine="siesta", name="probe",
            params={"system_label": "probe"})).get_json()
        # the browser's half, through the file layer
        (d / rendered["handover_name"]).write_text(rendered["handover_text"])
        over = _json.loads(rendered["handover_text"])

        proposed = {"schema": "molbuilder/task@1", "engine": over["engine"],
                    "shape": "flat", "run": over["run"],
                    "structure": over["structure"], "varies": [],
                    "stages": [{"name": "coarse", "enabled": True,
                                "overrides": {}}]}
        r = web_client.post("/api/task-setup/save", json={
            "dest": str(d), "text": _json.dumps(proposed)})
        assert r.status_code == 200, r.get_json()
        out = r.get_json()
        assert (d / "task.json").is_file()
        assert out["stages"] == ["coarse"]
        assert out["handover_here"] is True, (
            "the save door should report the hand-over for the browser to remove")
        assert (d / "task.1st.json").is_file(), (
            "the save door deleted it; moving bytes is the file layer's job")
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_save_refuses_rather_than_repairs(web_client):
    """The text goes through `task.read_task` — the same door `prep` uses — so
    a browser cannot become a second, drifting writer of descriptions."""
    d = _fresh_calc_dir()
    try:
        r = web_client.post("/api/task-setup/save",
                            json={"dest": str(d), "text": "{ not json"})
        assert r.status_code == 400
        assert not (d / "task.json").exists(), "a bad description was written"

        # complete but for `stages`, so the STAGES refusal is what fires
        from molbuilder.identity import run_id
        no_stages = {"schema": "molbuilder/task@1", "engine": {"name": "siesta"},
                     "shape": "flat",
                     "run": {"name": "x", "id": run_id("x", "H2"),
                             "created": "2026-08-16T00:00:00-07:00"},
                     "structure": {"source": "s.xyz", "formula": "H2",
                                   "atoms": 2},
                     "varies": []}
        r2 = web_client.post("/api/task-setup/save", json={
            "dest": str(d), "text": _json.dumps(no_stages)})
        assert r2.status_code == 400
        assert "stage" in r2.get_json()["error"].lower(), (
            "the refusal should be the reader's own words")
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


# --------------------------------------------------------------------- #
#  T3 — the stage table edits                                            #
# --------------------------------------------------------------------- #

def test_the_table_edits_the_description_in_one_direction():
    """The table is a VIEW of the buffer, not a second source.

    Two-way binding between a table and a text buffer is how you get an edit
    loop, so a table edit mutates the model → re-serialises into the editor →
    repaints; a hand edit re-parses the other way, debounced and silent while
    the text is mid-typing.  The BUFFER stays what `save` sends.
    """
    src = VIEWER.read_text()
    assert "function syncFromModel" in src
    assert "JSON.stringify(_task" in src, "table edits do not reach the buffer"
    assert "_reparse" in src, "a hand edit never reaches the table"


def test_removing_the_last_stage_is_refused():
    """`stages.md` § 6.5 — a job always has at least one stage, so there is no
    stage-less shape to fall back to."""
    src = VIEWER.read_text()
    assert "function removeStage" in src
    assert re.search(r"stages\.length\s*<=\s*1", src), (
        "nothing stops the last stage being removed")


def test_a_new_stage_copies_the_previous_ones_values():
    """`task-setup.md` § 9 — a refinement starts from what came before; a
    stage that inherits nothing is a different calculation, not a next step."""
    src = VIEWER.read_text()
    assert "function addStage" in src
    assert re.search(r"Object\.assign\(\{\},\s*\(prev && prev\.overrides\)", src), (
        "a new stage does not copy the previous overrides")


def test_an_empty_cell_deletes_the_override_rather_than_storing_blank():
    """Absent means "this stage uses the template's value" — a real state
    (`stages.md` § 6.2), expressed by the key being gone."""
    src = VIEWER.read_text()
    assert "function setCell" in src
    assert re.search(r'if \(text === ""\)[\s\S]{0,200}delete ov\[col\]', src), (
        "an emptied cell stores a blank instead of removing the override")


def test_a_stage_name_is_checked_against_the_descriptions_rule():
    """The name keys filenames, so the rule is `stages.md` § 2's — letters,
    digits, underscore; no hyphen, which means 'a counter follows'."""
    src = VIEWER.read_text()
    assert "/^[A-Za-z0-9_]+$/" in src, "stage names are not validated"


def test_disabling_a_stage_keeps_its_values():
    """It changes what `prep` builds; it does not delete the row's values.

    The first version of this test matched the word "delete" in the function's
    own COMMENT — asserting on prose rather than on code.  It now strips
    comments and reads the statements.
    """
    src = VIEWER.read_text()
    assert "function toggleStage" in src
    body = src.split("function toggleStage", 1)[1].split("\nfunction ", 1)[0]
    body = re.sub(r"//.*", "", body)          # the code, not the commentary
    assert "delete" not in body, "disabling a stage discards its values"
    assert ".enabled" in body, "toggle does not touch `enabled`"


# --------------------------------------------------------------------- #
#  T4 machine rows · T5 what has run                                     #
# --------------------------------------------------------------------- #

def test_one_point_is_a_choice_and_several_a_measurement():
    """`generator.md § 2` — a run is a sweep of length one, so both states are
    the same structure at different lengths.  Verified legal by the reader:
    `bench` takes a non-empty list, and a one-element list parses."""
    src = VIEWER.read_text()
    assert re.search(r'pts\.length === 1 \? "chosen" : "measured"', src), (
        "the row's length no longer decides what it is")


def test_a_machine_answered_setting_is_never_a_choice():
    """mpi_np / omp_threads / max_memory_mb may never carry a value in a
    description (`template.md § 6.4`), so even one point is a point to TRY."""
    src = VIEWER.read_text()
    assert re.search(r'MACHINE_ANSWERED\.has\(name\)\s*\n?\s*\?\s*"machine"', src), (
        "a machine-answered setting can render as `chosen`")


def test_measuring_never_discards_the_chosen_value():
    """`task-setup.md § 9` — adding a point keeps the value as the first one."""
    src = VIEWER.read_text()
    assert "function addPoint" in src
    body = src.split("function addPoint", 1)[1].split("\nfunction ", 1)[0]
    assert "pts.push(v)" in body, "a new point replaces rather than appends"
    assert "splice" not in body, "adding a point removes an existing one"


def test_a_setting_with_no_points_is_removed_not_left_empty():
    """`bench` takes a NON-EMPTY list — the reader refuses an empty one, so a
    setting with no points is a setting that is not being measured."""
    src = VIEWER.read_text()
    body = src.split("function removePoint", 1)[1].split("\nfunction ", 1)[0]
    assert "delete b[name]" in body, "an emptied setting stays as an empty list"


def test_what_has_run_is_counted_from_the_directory_and_not_judged():
    """No target machine needed, which is why it belongs here.  It counts
    attempts; whether one CONVERGED is in its output and belongs to Results."""
    src = VIEWER.read_text()
    assert "function runsForStages" in src
    assert "run-" in src, "attempts are not counted"
    body = src.split("function runsForStages", 1)[1].split("\n/* ", 1)[0]
    for verdict in ("converged", "failed", "success"):
        assert f'"{verdict}"' not in body, (
            f"the page judges a run as {verdict!r} from a listing")


# --------------------------------------------------------------------- #
#  The checkpoint API (F3) and the two guards (F1, F2)                   #
# --------------------------------------------------------------------- #

def test_checkpoint_is_a_public_api_not_a_private_click_handler():
    """`projects.md` § 5 — a sub-namespace on the one door, like
    `projects.parser`.  The panel's save WAS a private click handler, so a tab
    needing it had to POST the route itself or reach into the panel's DOM."""
    ck = (STATIC / "lib/projects/checkpoint.js").read_text()
    for fn in ("export async function status",
               "export async function init",
               "export async function saveState"):
        assert fn in ck, f"missing from the API: {fn}"
    for absent in ("export async function restore", "export async function tag"):
        assert absent not in ck, (
            "restore/tag are decisions taken at the panel, not another tab's "
            "side effect (`checkpointing.md` L4)")
    door = (STATIC / "lib/projects/projects-sidebar.js").read_text()
    assert "projects.checkpoint" in door, "the API is not on the one door"


def test_the_panel_calls_the_api_rather_than_a_second_implementation():
    ck = (STATIC / "lib/projects/checkpoint.js").read_text()
    body = ck.split("async function _onCommitClick", 1)[1].split("\nasync function", 1)[0]
    assert "saveState(" in body, "the panel does not go through the API"
    assert "/api/checkpoint/save" not in body, (
        "the panel still POSTs the route itself — two implementations")


def test_a_state_needs_a_note():
    """`checkpointing.md` L4 retired automatic messages, so nothing writes one
    on your behalf."""
    ck = (STATIC / "lib/projects/checkpoint.js").read_text()
    body = ck.split("export async function saveState", 1)[1]
    assert "A state needs a note" in body, "saveState invents a note"


def test_send_refuses_onto_a_described_calculation():
    """F1 — this guard was a 409 in the endpoint and was LOST when it became
    render-only.  Restored on the side that chooses where to write."""
    src = (STATIC / "structure-optimization/viewer.js").read_text()
    assert "one job per folder" in src, "Send can overwrite another calculation"
    assert re.search(r'readFile\(dest \+ "/task\.json",\s*\n?\s*\{ missingOk: true \}',
                     src), "the check does not go through the file layer"


def test_save_refuses_a_different_calculation(web_client):
    """F2 — the ids say these are different calculations, and overwriting one
    with the other orphans every warm file keyed to it."""
    d = _fresh_calc_dir()
    try:
        from molbuilder.identity import run_id
        base = {"schema": "molbuilder/task@1", "engine": {"name": "siesta"},
                "shape": "flat", "structure": {"source": "a.xyz",
                                               "formula": "Au2", "atoms": 2},
                "varies": [], "stages": [{"name": "coarse", "enabled": True,
                                          "overrides": {}}]}
        theirs = dict(base, run={"name": "theirs", "id": run_id("theirs", "Au2"),
                                 "created": "2026-08-01T00:00:00-07:00"})
        (d / "task.json").write_text(_json.dumps(theirs))

        mine = dict(base, run={"name": "mine", "id": run_id("mine", "Au2"),
                               "created": "2026-08-16T00:00:00-07:00"})
        r = web_client.post("/api/task-setup/save",
                            json={"dest": str(d), "text": _json.dumps(mine)})
        assert r.status_code == 409, r.status_code
        assert "one job per folder" in r.get_json()["error"].lower()
        back = _json.loads((d / "task.json").read_text())
        assert back["run"]["id"] == theirs["run"]["id"], "it was overwritten"

        # re-saving the SAME calculation stays free
        again = web_client.post("/api/task-setup/save",
                                json={"dest": str(d), "text": _json.dumps(theirs)})
        assert again.status_code == 200, again.get_json()
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_saving_is_two_steps_and_the_first_is_the_checkpoint(web_client):
    """`task-setup.md` § 8 — the state is saved before the description, so
    whatever you are about to change can be brought back."""
    body = web_client.get("/task-setup").data.decode()
    assert 'id="ts-ckpt"' in body, "no checkpoint step on the save card"
    assert 'id="ts-ckpt-note"' in body, "no note field (`checkpointing.md` L4)"

    src = VIEWER.read_text()
    assert "projects0.checkpoint.saveState" in src, (
        "save does not take a state")
    assert "/api/checkpoint/save" not in src, (
        "the tab POSTs the checkpoint route itself instead of the API")
    # the offer is a tick the user can clear — never taken silently
    assert 'checked' in body and "ts-ckpt" in body
    assert "wantCkpt" in src, "the checkpoint is unconditional"


def test_a_failed_checkpoint_stops_the_save():
    """The step exists so what you change can be brought back; writing anyway
    would silently spend the safety net you asked for."""
    src = VIEWER.read_text()
    body = src.split("async function save()", 1)[1].split("\n/* ", 1)[0]
    i_fail = body.index("No state was saved, so nothing was written")
    i_post = body.index("/api/task-setup/save")
    assert i_fail < i_post, "the description is written before the state"
    assert body.count("return;", 0, i_post) >= 2, (
        "a failed checkpoint does not stop the save")


def test_a_new_column_seeds_every_stage_with_the_template_value():
    """`task-setup.md` § 9 — adding a column "changes nothing on screen": it is
    a statement about structure, never about values.

    The tab has no template, but it does not need one: an ABSENT override
    already means "this stage uses the template's value" (`stages.md` § 6.2),
    so adding the column and touching no cell IS seeding them all.
    """
    src = VIEWER.read_text()
    assert "function addColumn" in src
    body = src.split("function addColumn", 1)[1].split("\n/* ", 1)[0]
    assert "overrides" not in body, (
        "adding a column writes cell values; absence is the seed")
    assert "v.push(name)" in body


def test_removing_a_column_says_what_is_lost_before_it_goes():
    """§ 9 — "the page says which value it kept, and says it BEFORE the click"."""
    src = VIEWER.read_text()
    assert "_pendingDrop" in src, "removing a column is a single click"
    body = src.split("function removeColumn", 1)[1].split("\n/* ", 1)[0]
    assert "last enabled" in body.lower() or "enabled" in body, (
        "the surviving value is not the last enabled stage's")
    assert "Click × again" in body, "there is no second, confirming click"


def test_removing_a_column_does_not_pretend_the_value_survives():
    """This page edits `task.json`, not the template — so it must not imply
    the kept value lands anywhere."""
    src = VIEWER.read_text()
    body = src.split("function removeColumn", 1)[1].split("\n/* ", 1)[0]
    assert "not the template" in body, (
        "the message implies the value is preserved somewhere it is not")


def test_the_identity_facts_are_shown_and_read_only(web_client):
    """`task-setup.md` § 3 — shown because you are about to commit a week of
    compute against them, not so they can be changed."""
    body = web_client.get("/task-setup").data.decode()
    assert 'id="ts-came-card"' in body and 'id="ts-facts"' in body
    src = VIEWER.read_text()
    assert "function renderCameOver" in src
    fn = src.split("function renderCameOver", 1)[1].split("\n/* ", 1)[0]
    assert '"Run id"' in fn, "the id is not shown — nothing says which calculation"
    assert "<input" not in fn and 'el("input"' not in fn, (
        "the identity facts are editable here")


def test_the_tab_waits_for_the_sidebar_instead_of_reading_it():
    """`projects.md` § 1: *"A tab waits for it with
    `runtime.whenReady("projects")` instead of polling."*

    The sidebar is a `type=module` script, so its deferred initialisation has
    NOT run at DOMContentLoaded.  Reading `window.molbuilder.projects` there
    finds `undefined`, and the page reported "the projects sidebar did not
    load" on every single load — a user-visible bug from ignoring a documented
    facility, which is why this is pinned rather than just fixed.
    """
    src = VIEWER.read_text()
    assert 'whenReady("projects")' in src, (
        "the tab does not wait for the sidebar through the runtime registry")
    boot = src.split("function boot()", 1)[1]
    assert "window.molbuilder.projects" not in boot, (
        "boot still reads the namespace directly instead of awaiting it")


def test_the_runtime_loads_before_every_other_script(web_client):
    """The registry can only hand out what registered with it, so it has to be
    parsed first — `molbuilder-runtime.js` before the sidebar and the tab."""
    body = web_client.get("/task-setup").data.decode()
    i_rt   = body.index("lib/molbuilder-runtime.js")
    i_tab  = body.index("task-setup/viewer.js")
    assert i_rt < i_tab, "the runtime loads after the tab's own script"


def test_the_tab_hands_you_the_next_command(web_client):
    """`task-setup.md` § 1 — this page "turns that into a description on disk
    and **hands you the command to run it somewhere else**".  Half the tab's
    purpose, and it was missing entirely until 2026-08-16."""
    body = web_client.get("/task-setup").data.decode()
    assert 'id="ts-next-card"' in body and 'id="ts-next"' in body
    src = VIEWER.read_text()
    assert "function renderNext" in src
    assert "jobset prep run" in src and "jobset submit run" in src


def test_the_command_names_its_stage_and_what_it_continues_from():
    """Exact, not generic: every verb is given the stage's name
    (`stages.md` § 6.5), and a continuing stage names the attempt because
    `prep` is told, never left to guess (`project-layout.md` § 1.6)."""
    src = VIEWER.read_text()
    body = src.split("function renderNext", 1)[1].split("\n/* ", 1)[0]
    assert '"molbuilder jobset prep run " + name' in body, (
        "the command does not name its stage")
    assert "--from" in body, "a continuing stage does not name its source"
    assert 'restart' in body and 'continue' in body, (
        "`--from` is emitted regardless of whether the stage continues")
    assert "padStart(2" in body, "the --from token drops its ordinal"


def test_a_disabled_stage_gets_no_command():
    """It changes what `prep` will build, so offering a command for it would
    be offering to run something the description says to skip."""
    src = VIEWER.read_text()
    body = src.split("function renderNext", 1)[1].split("\n/* ", 1)[0]
    assert "enabled !== false" in body, "disabled stages still get commands"

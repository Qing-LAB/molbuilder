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


def test_the_machine_answered_set_is_derived_and_not_listed():
    """An item whose resolver is an allocation resolver may never carry a
    value in a description (`engines/template.md` § 6.4), so the page must not
    show it as a choice.

    **The page must DERIVE which those are, not list them.**  It listed them
    -- ``new Set(["mpi_np", "omp_threads", "max_memory_mb"])`` -- until
    2026-08-17, which was a third answer to a question the page already had
    two answers to: ``/api/task-setup/sweepable`` ships ``machine_answers``
    per item, computed from ``template.ALLOCATION_RESOLVERS``, and the page
    already read that field in two other places.  A fourth allocation-backed
    item would have rendered as the user's own choice, silently.

    This asserts the direction that drifts: the derivation is present and the
    hard-coded list is not.
    """
    src = VIEWER.read_text()
    assert "machineAnswers" in src, "the page no longer derives the set"
    assert '"/api/task-setup/sweepable' in src, (
        "the derivation's source endpoint is not called")
    assert "MACHINE_ANSWERED" not in src, (
        "the hard-coded set is back -- derive it from `machine_answers`")
    # And no fresh literal list of the three, in any order.
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    for name in ("mpi_np", "omp_threads", "max_memory_mb"):
        assert f'"{name}"' not in code or "ROW_NOTE" in code, (
            f"{name} is spelled in executable code again")


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
    """No hand-rolled writes or deletes on any surface.  The send moved to
    lib/task-handover.js at P2 (spectra-migration-plan.md) -- ONE door two
    tabs share -- so the write lives there now."""
    opt = (STATIC / "structure-optimization/viewer.js").read_text()
    lib = (STATIC / "lib/task-handover.js").read_text()
    ts  = VIEWER.read_text()
    assert "projects.safeSave(text, name," in lib, (
        "the hand-over does not write through safeSave(TEXT, FILENAME, opts)")
    assert "projects.deleteEntry(" in ts, (
        "the hand-over is not removed through the file layer")
    for surface, src in (("optimization", opt), ("task-handover", lib),
                         ("task-setup", ts)):
        assert "/api/files/write" not in src, (
            f"{surface} calls the write route directly instead of the door")


def test_the_handover_refuses_the_cited_junction_as_destination():
    """The calculation never lives in its citation (archive/2026-09-01-transport-design.md
    § 4.1b): the sidebar selection lingers on the attempt the person
    just browsed to cite, and describing there would drop task.json --
    and every prep attempt -- inside the finished relaxation.  The
    success message names the folder it DID write, so the person sees
    where the description went."""
    lib = (STATIC / "lib/task-handover.js").read_text()
    assert "never lives inside its citation" in lib, (
        "the citation-destination guard is gone -- a transport describe "
        "into the cited attempt would pollute the relaxation it cites")
    assert 'relNorm === citeNorm' in lib, (
        "the guard must compare the DESTINATION against the citation")
    assert '+ " into " + (rel || "the selected folder")' in lib, (
        "the transport success message no longer names its destination")


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

    # `collectFdfParams` / `collectPyscfParams` collapsed into ONE
    # `collectParams(engine)` on 2026-08-17: the SIESTA one had become a
    # pure pass-through and the two differed by a container id.
    for helper in ("_structureForRequest", "collectParams",
                   "_activeEngine"):
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

def _row_badge(name, points, machine):
    """What the machine card says a row IS — the real source, driven.

    A regex over `viewer.js` stood here until 2026-09-01 and broke on a
    refactor that kept the rule exactly ("the row's length no longer decides
    what it is" — it still did). A source grep pins a spelling; this pins the
    behaviour, which is the thing the contract states.
    """
    import json as _json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = VIEWER.read_text(encoding="utf-8")
    i = src.index("        const chosen = pts.length === 1 && !machineAnswers(name);")
    j = src.index("`;", i) + 2
    prog = (f"const pts = {_json.dumps(points)};\n"
            f"const name = {_json.dumps(name)};\n"
            f"const machineAnswers = () => {_json.dumps(bool(machine))};\n"
            + src[i:j].replace("const ", "var ") + "\n"
            "console.log(JSON.stringify({kind: kind, verdict: verdict}));")
    out = subprocess.run([node, "--input-type=commonjs", "-e", prog],
                         capture_output=True, text=True, timeout=20)
    if out.returncode != 0:
        pytest.fail(out.stderr)
    return _json.loads(out.stdout.strip().splitlines()[-1])


def _continue_tail(shape, restart, prev_runs):
    """What `renderNext` teaches as the continue tail -- the real source,
    driven under node rather than grepped.

    The block is lifted whole and given its three inputs, so a refactor that
    keeps the rule keeps this passing, and one that breaks it fails here."""
    import json as _json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = VIEWER.read_text(encoding="utf-8")
    i = src.index("        let from = \"\";")
    j = src.index("        const runs = _runs[name];", i)
    prog = ("const i = 1;\n"
            f"const _shape = {_json.dumps(shape)};\n"
            f"const _runs = {_json.dumps(prev_runs)};\n"
            f"const ov = {{ restart: {_json.dumps(restart)} }};\n"
            "const enabled = [{ full: 0, st: { name: 'coarse' } },\n"
            "                 { full: 1, st: { name: 'tight'  } }];\n"
            + src[i:j].replace("const ", "var ").replace("let ", "var ")
            + "\nconsole.log(JSON.stringify({from: from}));")
    out = subprocess.run([node, "--input-type=commonjs", "-e", prog],
                         capture_output=True, text=True, timeout=20)
    if out.returncode != 0:
        pytest.fail(out.stderr)
    return _json.loads(out.stdout.strip().splitlines()[-1])["from"]


def test_a_FLAT_bundle_is_taught_no_from_at_all():
    """**Flat keeps no attempt directories**, so it has no `--from`: the
    shared warm set lies in the bundle root and continuing is free
    (`project-layout.md` § 1).  `prepare_attempt` refuses an explicit
    `--from` there by name.

    The page taught `--from 01_coarse/run-0` on every continuing rung
    regardless of shape until 2026-09-02 -- a command that cannot run,
    naming a directory a flat bundle does not have.  Two wrongs at once,
    and the second hid the first."""
    assert _continue_tail("flat", "continue", {"coarse": 2}) == ""


def test_a_HIERARCHICAL_continue_names_the_LAST_attempt_not_the_first():
    """A stage with three attempts was taught to continue from `run-0`.

    The page knows the count -- `_runs`, filled by `runsForStages` -- and a
    hard-coded index is the kind of guess that reads as a fact: `run-0`
    exists, so the command succeeds and continues from the wrong place."""
    assert _continue_tail("hierarchical", "continue",
                          {"coarse": 3}) == " --from 01_coarse/run-2"
    # one attempt, and none at all, both name run-0
    assert _continue_tail("hierarchical", "continue",
                          {"coarse": 1}) == " --from 01_coarse/run-0"
    assert _continue_tail("hierarchical", "continue",
                          {}) == " --from 01_coarse/run-0"


def test_a_rung_that_does_not_continue_is_taught_no_tail():
    """The other half, so the fix is not "always emit a tail"."""
    assert _continue_tail("hierarchical", "", {"coarse": 3}) == ""


def test_the_run_preview_shows_the_EMITTED_launch(web_client):
    """`architecture.md` A13: the run's surface shows the value the `.sbatch`
    will carry and where it came from — resolved by the emitter, and never
    left blank when blank resolves to a number.

    **The failure this pins**: the preview called `header_ntasks` WITHOUT
    `auto=`, so an unstated rank count rendered the no-record refusal while
    the header carried the domain's own width. The card was wrong about the
    one number it exists to surface (caught by review, 2026-09-02).
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/build.py").read_text(encoding="utf-8")
    door = src[src.index("def api_task_setup_prep("):
               src.index("def api_task_setup_save(")]
    assert "_emitted_launch" in door, "the preview reports no emitted launch"
    call = door[door.index("ntasks, why = header_ntasks("):]
    call = call[:call.index(")\n")]
    assert "auto=" in call, (
        "the card asks the emitter a DIFFERENT question than the emitter "
        "asks itself -- `auto=` is what turns 'cannot size' into the "
        "domain's width")
    assert "auto_ranks(" in call, (
        "the default must come from the same producer the header uses")


def test_the_emitted_rows_cover_every_launch_parameter(web_client):
    """A13 again, on the shape: `-n`, `-c`, `--gres`, `--mem`, `-t`, `-p`.
    A parameter missing from the block is a parameter that can surprise."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/build.py").read_text(encoding="utf-8")
    body = src[src.index("def _emitted_launch("):]
    body = body[:body.index("\n    # ---- the PLAN")]
    for flag in ("-n", "-c", "--gres", "--mem", "-t", "-p"):
        assert f'"{flag}"' in body, f"{flag} is not reported to the card"
    assert '"source"' in body, "a value without its source is half the rule"


def test_one_point_is_a_choice_and_several_a_measurement():
    """`generator.md` § 2 — a run is a sweep of length one, so both states are
    the same structure at different lengths.  Verified legal by the reader:
    `bench` takes a non-empty list, and a one-element list parses."""
    assert _row_badge("diag_algorithm", ["ELPA-1STAGE"], False)["verdict"] \
        == "chosen \u00b7 1 point"
    assert _row_badge("diag_algorithm", ["A", "B"], False)["verdict"] \
        == "measured \u00b7 2 points"


def test_a_ONE_POINT_MACHINE_row_is_a_trial_not_a_decision():
    """**Replaced the test that asserted the opposite** (2026-09-02).

    It read `test_length_decides_for_a_MACHINE_row_too` and pinned the
    design that lasted one day: *one point is a decision on every axis*.
    Narrowing a `bench` row to say what the RUN uses destroyed the plan to
    measure, so the two were separated -- `mpi_np: [8]` is *measure eight*,
    one trial, and `prep run` never reads it (`stages.md` § 6.8).  What the
    run uses is `execution`, asked in the rung's own tab.

    The measure card said "chosen · 1 point" beside such a row, which told
    a person their run was decided by a row the run does not consult.

    The `machine` KIND survives, and that is deliberate: it tints the row,
    because which kind of setting this is stays worth seeing."""
    one = _row_badge("mpi_np", [8], True)
    assert one["verdict"] == "measured \u00b7 1 point", (
        "a one-point machine row still claims to be the run's decision")
    assert one["kind"] == "machine", "the row lost its tint"
    many = _row_badge("mpi_np", [4, 8, 16], True)
    assert many["verdict"] == "measured \u00b7 3 points"
    assert many["kind"] == "machine"


def test_a_one_point_row_that_is_NOT_machine_answered_is_still_a_choice():
    """The other half, so the fix is not "call everything measured": a deck
    knob with one point is the value that stage runs at (`generator.md`
    § 4.3a).  Only the machine-answered rows changed."""
    one = _row_badge("diag_algorithm", ["ELPA-1STAGE"], False)
    assert one["verdict"] == "chosen \u00b7 1 point"
    assert one["kind"] == "chosen"


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
    src = (STATIC / "lib/task-handover.js").read_text()
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
    assert "jobset prep run" in src and "jobset launch run" in src


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


def test_parameters_are_PICKED_from_the_catalogue_not_typed(web_client):
    """A free-text box is not a list.  The catalogue knows every parameter, so
    the picker reads it rather than asking the user to spell a name.

    **From the columns endpoint, not the parameter form's schema**
    (2026-08-18).  This asserted `/api/build/schema/`, which is the FORM's
    schema and filters the whole `staging` group out on purpose — a form does
    not ask how many ranks the scheduler granted.  Borrowing that answer cost
    the table `restart`, the field that decides whether a ladder is a ladder,
    so a ladder built here ran every stage clean.  `stages.md` § 6.2 gives the
    right question: anything the description may HOLD may be a column."""
    body = web_client.get("/task-setup").data.decode()
    for sel in ('id="ts-add-col"', 'id="ts-add-setting"'):
        assert f"<select {sel[:0]}" or sel in body
    assert body.count("<select") >= 2, "the add controls are still text inputs"
    src = VIEWER.read_text()
    assert "/api/task-setup/columns" in src, \
        "columns are not drawn from the catalogue"
    assert "/api/build/schema/" not in src, \
        "the column picker is reading the parameter FORM's schema again, " \
        "which filters out the staging group -- restart among it"
    assert "/api/task-setup/sweepable" in src, \
        "bench settings are not drawn from it"


def test_the_column_picker_offers_restart(web_client):
    """The regression that motivated the endpoint, pinned end to end.

    `restart` sits in the `staging` group because it is not a physics
    parameter, and it is per-stage because it decides whether each rung starts
    from what the one before it produced.  One `group` cannot say both, so the
    table's columns are drawn from what the description may HOLD instead."""
    j = web_client.get("/api/task-setup/columns?engine=siesta").get_json()
    names = [i["name"] for i in j["items"]]
    assert "restart" in names
    # ...and the machine's settings are still not the description's to hold
    assert not ({"mpi_np", "omp_threads", "max_memory_mb", "gpu_count"}
                & set(names))


def test_only_execution_category_parameters_may_be_swept(web_client):
    """`stages.md § 6.8` — sweeping anything else means each point silently
    measures a DIFFERENT calculation."""
    j = web_client.get("/api/task-setup/sweepable?engine=siesta").get_json()
    assert j["ok"] and j["items"]
    from molbuilder.template import load_catalogue, read_template, one
    t = read_template(load_catalogue())
    for item in j["items"]:
        it = one(t, item["name"])
        assert "execution" in it.category, (
            f"{item['name']} is offered for sweeping but is not `execution`")


def test_the_sweepable_list_says_which_the_machine_answers(web_client):
    """An allocation resolver means a description may never carry a value
    (`template.md § 6.4`), so those can only ever be measured."""
    j = web_client.get("/api/task-setup/sweepable?engine=siesta").get_json()
    by = {i["name"]: i["machine_answers"] for i in j["items"]}
    for machine in ("mpi_np", "omp_threads", "max_memory_mb", "gpu_count"):
        assert by.get(machine) is True, f"{machine} not flagged machine-answered"
    assert by.get("use_gpu") is False, (
        "the GPU is a user decision, not something the machine answers "
        "(engines/overview.md § 3a: the user decides the GPU)")


def test_a_picker_offers_only_what_is_not_already_used():
    src = VIEWER.read_text()
    assert "function fillPicker" in src
    body = src.split("function fillPicker", 1)[1].split("\nasync function", 1)[0]
    assert "taken.indexOf(i.name) === -1" in body, (
        "the picker offers parameters that are already columns/settings")


def test_the_chosen_shape_carries_a_tick():
    """A border tint reads as "hovered" as easily as "chosen", and the shape is
    a decision the page refuses to guess — so which one you picked must be
    unmistakable.  The gutter is reserved on both, so ticking shifts nothing."""
    css = (STATIC / "task-setup/style.css").read_text()
    assert '.ts-choice .opt b::before' in css, "no tick on the shape options"
    assert 'visibility: hidden' in css and 'visibility: visible' in css, (
        "the tick is added/removed rather than shown/hidden, so the label "
        "shifts when you choose")
    assert '.ts-choice .opt[aria-pressed="true"] b::before' in css


def test_a_handover_opens_with_a_starting_matrix_not_an_empty_table():
    """`stages.md § 1.3` — *"`varies` defaults to the engine's `stage` group,
    and the user adds to or removes from it"*.

    An empty `varies` is not a neutral start: the table opens with no columns
    and nothing to edit, which is a dead end.  The group is a DEFAULT, never a
    restriction — any parameter can be added and any of these removed (§ 1.2).
    """
    src = VIEWER.read_text()
    assert 'c.group === "stage"' in src, (
        "the proposal does not seed varies from the stage group")


def test_the_bench_opens_with_a_starting_sweep():
    """The machine-answered settings can ONLY be measured, so an empty bench
    leaves the user typing point lists from scratch.  Safe to propose because
    `bench` records points to TRY and never an answer (`stages.md § 6.8`)."""
    src = VIEWER.read_text()
    assert "BENCH_START" in src
    assert "it.machine_answers" in src, (
        "the starting sweep is not restricted to what the machine answers")


def test_the_starting_sweep_only_covers_settings_the_engine_has():
    """Proposed rows are intersected with the sweepable set, so a PySCF
    description never opens with a SIESTA-only knob in its grid."""
    src = VIEWER.read_text()
    body = src.split("Promise.all([loadColumnChoices", 1)[1].split("}).catch", 1)[0]
    assert "for (const it of sweep)" in body, (
        "the grid is not intersected with what this engine can sweep")


def test_the_presets_come_from_the_shipped_table(web_client):
    """The same table `default_siesta_stages` builds the ladder from, so a
    stage filled here and stage N of that ladder cannot drift.  `tuning.md § 4`
    is the authority for the numbers; this serves them, never restates them."""
    j = web_client.get("/api/task-setup/presets?engine=siesta").get_json()
    assert j["ok"] and len(j["presets"]) == 3
    from molbuilder.config.siesta import SIESTA_STAGE_NAMES, SIESTA_STAGE_PRESETS
    for ps in j["presets"]:
        assert ps["name"] == SIESTA_STAGE_NAMES[ps["tier"]]
        assert ps["values"] == SIESTA_STAGE_PRESETS[ps["tier"]], (
            "the endpoint restates the tier values instead of serving them")


def test_a_preset_adds_its_missing_columns_first():
    """`task-setup.md § 9` — "a preset knows several fields.  If some are not
    columns yet it ADDS THEM FIRST — a preset that half-applied would be worse
    than one that refused"."""
    src = VIEWER.read_text()
    assert "function applyPreset" in src
    body = src.split("function applyPreset", 1)[1].split("\n/* ", 1)[0]
    i_add = body.index("v.push(key)")
    i_set = body.index("Object.assign(ov, values)")
    assert i_add < i_set, "values are written before the columns exist"
    assert "added.join" in body, "the page does not say which columns it added"


def test_an_empty_cell_shows_what_the_stage_will_actually_use():
    """An empty cell is not blank — it says the stage uses the template's
    value.  Showing the NUMBER rather than the word "template" is what makes
    "adding a column changes nothing on screen" (§ 9) visible, not merely true.
    """
    src = VIEWER.read_text()
    assert "function defaultText" in src
    assert "placeholder: fallback ||" in src, (
        "an unset cell shows no recommended value")


def test_every_parameter_carries_its_note_on_hover():
    """The catalogue already holds `help`, `unit` and `default`; a second copy
    would be the drift the one-source rule exists to prevent, so the tab looks
    them up from what the schema returned."""
    src = VIEWER.read_text()
    assert "function helpText" in src
    for surface in ('title: helpText(col)',        # a cell
                    'title: helpText(col) }, col', # the column header
                    'title: helpText(name)'):      # a machine row
        assert surface in src, f"no hover note: {surface}"
    assert "title: helpText(i.name)" in src, "the picker options carry no note"


def test_the_sweepable_notes_reach_the_lookup():
    """`staging` items are filtered out of the form schema, so the sweepable
    endpoint is the only place their help arrives."""
    src = VIEWER.read_text()
    # The fill moved into `_fillSweepMeta` on 2026-08-24 so BOTH of the
    # loader's paths -- the cached one and the fetching one -- publish it;
    # the cached path returned early without it, which is how every enum
    # became a text box.  What this pins is unchanged: the sweepable items
    # reach `_meta`, whichever path ran.
    body = src.split("async function loadSweepChoices", 1)[1].split("\n/**", 1)[0]
    assert "_fillSweepMeta(_sweep)" in body, (
        "machine settings would hover with no note at all")
    filler = src.split("function _fillSweepMeta(", 1)[1].split("\n}", 1)[0]
    assert "_meta[i.name]" in filler, (
        "the fill no longer writes into the lookup the hovers read")


def test_the_folder_template_is_what_an_empty_cell_names(web_client):
    """`stages.md` § 6.2: a stage that sets nothing uses THE TEMPLATE'S value.

    So the whole point of the hand-over — the k-grid a person chose in the
    parameter tab — has to survive into what Task setup shows.  Before this
    endpoint the tab read the catalogue's default and named a number the job
    would not run whenever the sender had changed that parameter.
    """
    d = _fresh_calc_dir()
    try:
        rendered = web_client.post("/api/task-setup/handover", json=dict(
            _envelope(), engine="siesta", name="probe",
            params={"system_label": "probe", "kgrid": [4, 4, 1],
                    "mesh_cutoff": 450.0})).get_json()
        (d / rendered["template_name"]).write_text(rendered["template_text"])

        j = web_client.get("/api/task-setup/template-values?dir="
                           + str(d)).get_json()
        assert j["ok"], j
        assert j["name"] == rendered["template_name"]
        assert j["values"]["mesh_cutoff"] == 450.0, (
            "the value the parameter tab collected did not reach Task setup; "
            "the catalogue default (300.0) would be shown instead")
        assert j["values"]["kgrid"] == [4, 4, 1], j["values"].get("kgrid")
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_a_folder_with_no_template_is_not_an_error(web_client):
    """An empty folder is an ordinary state, not a failure — the cells fall
    back to the catalogue, which is exactly right when nothing was sent."""
    d = _fresh_calc_dir()
    try:
        j = web_client.get("/api/task-setup/template-values?dir="
                           + str(d)).get_json()
        assert j["ok"] and j["name"] is None and j["values"] == {}
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_the_template_values_outrank_the_catalogue_default():
    """The order matters and is easy to write backwards."""
    src = VIEWER.read_text()
    body = src.split("function defaultText", 1)[1].split("\n/**", 1)[0]
    assert "_tmpl.values" in body, (
        "defaultText ignores the folder's template — it would show the "
        "catalogue recommendation where the contract says the template's value")
    assert body.index("_tmpl.values") < body.index("m.default"), (
        "the catalogue default is consulted FIRST, so a template that answers "
        "the parameter is overridden by the recommendation")
    assert "await loadTemplateValues(dir)" in src, (
        "nothing loads the template when the folder changes")


def test_the_handover_carries_the_STRUCTURE_not_a_summary_of_it(web_client):
    """The hole this closes.  The hand-over used to record a formula, an atom
    count, and `structure_path` -- the projects sidebar's selected file, which
    in a real folder pointed at the calculation's own `.template.toml`.  The
    geometry, the cell and the region labels all crossed the wire and were
    thrown away at the last step.

    `molview.md` § 11.7: the SERVER writes every file, from the one generator,
    because a browser-authored pair drifts -- it shipped once with no
    `schema_version` and every label in it was dropped on the next open.  So
    the pair comes from `StructureCodec`, exactly as `/api/structure/export`
    builds it.

    This is the check `handover-procedure.md` § 7 should have had from the
    start: OPEN what `source` names and see whether the structure survived.
    """
    import json as _json
    d = _fresh_calc_dir()
    try:
        env = _envelope()
        env["structure"]["metadata"] = {
            "regions": {"frozen_atoms": [0], "L-electrode": [1]},
            "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 12.0]],
            "cell_origin": None, "axis_kind": None, "vacuum": None,
        }
        r = web_client.post("/api/task-setup/handover", json=dict(
            env, engine="siesta", name="probe",
            params={"system_label": "probe"}))
        assert r.status_code == 200, r.get_json()
        out = r.get_json()

        files = out["structure_files"]
        assert files, "the structure was not written at all"
        names = [f["name"] for f in files]
        assert out["label"] + ".source.xyz" in names, (
            "the hand-over's structure pair must carry the .source "
            "reservation (job-contracts.md 6.3): " + repr(names))
        assert any(n.endswith(".molstruct.json") for n in names), (
            "the cell and the region labels have nowhere to live")

        # the browser's half, through the file layer
        for f in files:
            (d / f["name"]).write_text(f["text"])
        (d / out["handover_name"]).write_text(out["handover_text"])

        over = _json.loads(out["handover_text"])
        src = over["structure"]["source"]
        assert src and not src.endswith(".json"), src
        assert "/" not in src, "the reference must be folder-relative"
        assert (d / src).is_file(), f"{src} is named but not there"

        # READ IT BACK through the one authority, pairing and all.
        from molbuilder.workingcopy_structure import StructureCodec
        back = StructureCodec().read(d / src)
        assert len(back.elements) == over["structure"]["atoms"] == 2
        assert back.cell is not None, "the CELL did not survive the hand-over"
        assert [list(v) for v in back.cell][2][2] == 12.0, back.cell
        assert back.frozen_atoms, "the frozen-atom tags did not survive"
        assert "L-electrode" in (back.regions or {}), (
            f"the region labels did not survive: {back.regions}")

        side = _json.loads((d / (out["label"] + ".source.molstruct.json")).read_text())
        assert side.get("schema_version"), (
            "the sidecar has no schema version — the load door refuses the "
            "pair on the next open and every label is dropped silently")
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_the_sidebar_cursor_is_not_in_the_payload():
    """`molview.md` § 9.3a: the facts that leave together were read together.
    `getCurrentFile()` is a second fact sampled at a second moment, and it is
    what made a calculation claim to be OF its own parameter file."""
    src = (ROOT / "molbuilder/web/static/lib/task-handover.js").read_text()
    # Anchored on the CALL, not on any mention of the route: the file
    # header names its endpoints (fixed 2026-08-17), so splitting on the
    # bare path started the slice in the header and swept in an
    # unrelated `structure_path` from a different fetch.
    body = src.split("const r = await fetch(url", 1)[1]\
              .split("out = await r.json()", 1)[0]
    # Comments out first — this file's own note explains what was removed, and
    # a test that matches its own explanation proves nothing.
    body = re.sub(r"//.*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "structure_path" not in body, "the sidebar's cursor is still sent"
    assert "getCurrentFile" not in body


def test_a_folder_with_no_history_gets_one_before_the_first_save():
    """A folder that has no history yet must get `init` before the first
    save.  Gating `init` on `st.ok` skipped it for exactly the folders that
    needed it, and the save then failed on `saveState`'s "not a checkpoint
    folder; run init first": a refusal naming a step the page had chosen to
    skip.  The question is `initialized` — the server's own field
    (2026-08-19: `status()` now really asks `/api/checkpoint/state`, whose
    `ok` means "the query worked")."""
    src = VIEWER.read_text()
    body = src.split("checkpoint.status(_dir)", 1)[1].split("saveState", 1)[0]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "st.ok &&" not in body, (
        "init is still gated on `ok`, which is false for every folder that "
        "has never been checkpointed — the ones that need init")
    assert "!st.initialized" in body, "nothing asks whether a history exists"


def test_save_is_blocked_while_the_checkpoint_has_no_note():
    """`saveState` requires a note (`checkpointing.md` L4 — nothing writes a
    message on your behalf), and the save aborts rather than writing without
    the state it was told to keep.  With the box ticked by default and the note
    empty, the button was live and the only way to learn that was to press it
    and read a refusal.  The condition is knowable before the click."""
    src = VIEWER.read_text()
    body = src.split("function refreshSave", 1)[1].split("\nasync function save", 1)[0]
    assert "wantsCheckpoint()" in body and "checkpointNote()" in body, (
        "the button does not consider the checkpoint it is about to run")
    assert "blocked" in body.split("checkpointNote()", 1)[1][:200], (
        "an empty note does not block the button")
    watch = src.split("function watchCheckpointControls", 1)[1].split("\n}", 1)[0]
    assert "ts-ckpt-note" in watch and "ts-ckpt" in watch, (
        "nothing re-decides the button as the note is typed")


def test_the_structure_pair_is_not_reported_as_engine_state():
    """`warm_files_present` answers *has anything run here* by SUBTRACTION —
    anything named after the label that is not on `OUR_FILE_PATTERNS` is the
    engine's.  The hand-over started writing `<label>.xyz` +
    `<label>.molstruct.json` into the bundle, so `prep` announced a brand-new
    calculation as "already under way here: warm files at the root" and offered
    a person their own input back as engine state."""
    from molbuilder.validation.identity import warm_files_present
    import tempfile, pathlib as _pl
    with tempfile.TemporaryDirectory() as d:
        base = _pl.Path(d)
        for n in ("slab.source.xyz", "slab.source.molstruct.json",
                  "slab.template.toml"):
            (base / n).write_text("x")
        assert warm_files_present(base, "slab", "siesta") == [], (
            "the hand-over's own files are reported as engine warm files")
        (base / "slab.XV").write_text("x")
        assert warm_files_present(base, "slab", "siesta") == ["slab.XV"], (
            "a real warm file stopped being detected")
        # Post-reservation the bare `<label>.xyz` is the ENGINE's name --
        # WriteCoorXmol writes it at the root of a flat run, which is the
        # file the 2026-08-19 clobber overwrote the input with.  The
        # subtraction must report it as run state now, not claim it.
        (base / "slab.xyz").write_text("x")
        assert "slab.xyz" in warm_files_present(base, "slab", "siesta"), (
            "a bare <label>.xyz is WriteCoorXmol's output and is engine "
            "state under the .source reservation")


from conftest import write_pseudos as _pseudos_for



def _child_env_with_a_config(tmp_path):
    """Environment for a spawned `molbuilder`, with a machine config of its own.

    These two tests run the real CLI in a SUBPROCESS from the repo root, and
    it used to pick up a ``./molbuilder.json`` sitting there -- the developer's
    own file, which no test had put under control.  That is proving something
    with found state, and it ended the moment the machine config stopped being
    looked for in the working directory (`configuration.md` § 2.1a).

    So the child is given a root of its own, holding the one thing the render
    requires: ``script_generation.activation``.  ``monkeypatch`` cannot reach
    across a process boundary, which is why the environment is built here
    rather than set on the parent.
    """
    import json
    import os
    root = tmp_path / "child-config-root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preamble": "module load mamba",
                              "activation": "source activate"},
    }))
    env = dict(os.environ)
    env["MOLBUILDER_CONFIG_DIR"] = str(root)
    return env


def test_the_whole_chain_from_structure_to_rendered_deck(web_client, tmp_path):
    """§ 7's bar, automated: structure -> hand-over -> description -> deck.

    **Every link can hold while the thing being carried is lost between them.**
    That is what the four shape-checks could not see, and why this compares the
    DECK's own lattice against the structure that started the chain rather than
    checking that each step returned `ok`.

    Written 2026-08-17 because § 7 had just been rewritten to claim the chain
    was verified — and it was, by hand, in a browser. A claim in a contract
    that rests on somebody having driven it once is the same false assurance
    the section above it retracts.
    """
    import json as _json, subprocess, sys
    d = _fresh_calc_dir()
    try:
        # a periodic slab: a cell, a region label, a frozen atom
        cell = [[5.77, 0.0, 0.0], [2.885, 4.997, 0.0], [0.0, 0.0, 20.0]]
        env = _envelope()
        env["structure"]["elements"] = ["Au", "Au", "S"]
        env["structure"]["positions"] = [[0, 0, 0], [2.885, 0, 0], [0, 0, 4.755]]
        env["structure"]["metadata"] = {
            "regions": {"frozen_atoms": [0], "slab": [0, 1]},
            "cell": cell, "cell_origin": None, "axis_kind": None, "vacuum": None,
        }
        r = web_client.post("/api/task-setup/handover", json=dict(
            env, engine="siesta", name="chain",
            params={"system_label": "chain", "kgrid": [8, 8, 1],
                    "kgrid_displacement": [0.5, 0.5, 0.0], "mesh_cutoff": 350.0}))
        assert r.status_code == 200, r.get_json()
        out = r.get_json()

        for f in out["structure_files"]:
            (d / f["name"]).write_text(f["text"])
        (d / out["template_name"]).write_text(out["template_text"])
        over = _json.loads(out["handover_text"])

        described = {"schema": "molbuilder/task@1", "engine": over["engine"],
                     "shape": "flat", "run": over["run"],
                     "structure": over["structure"], "varies": [],
                     "stages": [{"name": "coarse", "enabled": True,
                                 "overrides": {}}]}
        s = web_client.post("/api/task-setup/save",
                            json={"dest": str(d), "text": _json.dumps(described)})
        assert s.status_code == 200, s.get_json()

        # the data files the engine will open -- prep refuses without them
        _pseudos_for(d, ["Au", "S"])

        # …and now the part no shape-check reaches: RENDER IT.
        p = subprocess.run(
            [sys.executable, "-m", "molbuilder.cli", "jobset", "prep", "run",
             "coarse", "--bundle", str(d)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=300,
            env=_child_env_with_a_config(tmp_path))
        assert p.returncode == 0, p.stdout + p.stderr
        assert "already under way" not in p.stdout, (
            "the hand-over's own files are being reported as engine leftovers:\n"
            + p.stdout)

        decks = sorted(d.glob("*_01_coarse.fdf"))
        assert decks, sorted(x.name for x in d.iterdir())
        deck = decks[0].read_text()

        # THE LATTICE the k-grid indexes — the thing the hand-over used to drop
        lat = re.search(r"%block LatticeVectors(.*?)%endblock", deck, re.S)
        assert lat, "the deck has no cell — a periodic run became a molecule"
        rows = [[float(v) for v in ln.split()]
                for ln in lat.group(1).strip().splitlines()]
        assert rows == cell, f"deck lattice {rows} != source {cell}"

        kg = re.search(r"%block kgrid_Monkhorst_Pack(.*?)%endblock", deck, re.S)
        assert kg and "8 0 0 0.5" in kg.group(1), kg and kg.group(1)
        assert "MeshCutoff 350.0 Ry" in deck

        # the ATOMS, row by row, against the .xyz the hand-over wrote
        blk = re.search(r"%block AtomicCoordinatesAndAtomicSpecies(.*?)%endblock",
                        deck, re.S)
        fdf = [ln.split() for ln in blk.group(1).strip().splitlines()]
        xyz = [ln.split() for ln
               in (d / over["structure"]["source"]).read_text().splitlines()[2:]
               if ln.strip()]
        assert len(fdf) == len(xyz) == 3
        for a, b in zip(fdf, xyz):
            assert max(abs(float(x) - float(y))
                       for x, y in zip(a[:3], b[1:4])) < 1e-4, (a, b)
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_a_cpu_description_gets_a_cpu_benchmark(web_client, tmp_path):
    """The machine half of § 7's bar, which the chain test above does not reach.

    **Where the grid comes from is settled and it is not the description.**
    `generator.md` § 4.3 — *a sweep and an allocation are both inputs to `prep`,
    never fields of the description* — and § 10's class 3 puts `mpi_np`,
    `omp_threads` and `max_memory_mb` at prep, *"never floor 2"*.  So this test
    does NOT write points into `task.json`; it writes a description and asks
    `prep bench` to enumerate.

    **What the description DOES answer is the GPU**, and that is equally
    settled: `web/task-setup.md` § 6.2, *"use GPU or not is set up only at the
    Job Prep UI"*.  `use_gpu` is a `staging` item carrying a real value, and
    it rides the template like any other.

    Until 2026-08-17 `_bench_inputs` pinned `use_gpu=True` and
    `diag_algorithm='ELPA-1STAGE'` flat, so **every trial measured a GPU
    whatever was asked for** — and on a machine whose probe finds no GPU the
    verb refused outright, which made a CPU benchmark impossible to run at all.
    That is the case here: an ordinary CPU description, benchmarked.

    Three claims:

    * it RUNS, and produces more than one trial;
    * no trial carries the GPU keyword the description declined;
    * every trial is separately labelled (`project-layout.md` § 7 invariant 5)
      — two trials sharing a SystemLabel warm-start off each other's `.DM` and
      the timings stop being comparable, which is the one thing a benchmark
      exists to produce.
    """
    import json as _json, subprocess, sys
    d = _fresh_calc_dir()
    try:
        env = _envelope()
        r = web_client.post("/api/task-setup/handover", json=dict(
            env, engine="siesta", name="grid",
            params={"system_label": "grid", "mesh_cutoff": 200.0,
                    "use_gpu": False}))
        assert r.status_code == 200, r.get_json()
        out = r.get_json()
        for f in out["structure_files"]:
            (d / f["name"]).write_text(f["text"])
        (d / out["template_name"]).write_text(out["template_text"])
        over = _json.loads(out["handover_text"])

        described = {"schema": "molbuilder/task@1", "engine": over["engine"],
                     "shape": "hierarchical", "run": over["run"],
                     "structure": over["structure"], "varies": [],
                     "stages": [{"name": "coarse", "enabled": True,
                                 "overrides": {}}]}
        s = web_client.post("/api/task-setup/save",
                            json={"dest": str(d), "text": _json.dumps(described)})
        assert s.status_code == 200, s.get_json()
        _pseudos_for(d, ["H"])

        p = subprocess.run(
            [sys.executable, "-m", "molbuilder.cli", "jobset", "prep", "bench",
             "coarse", "--bundle", str(d)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=300,
            env=_child_env_with_a_config(tmp_path))
        assert p.returncode == 0, (
            "a CPU description cannot be benchmarked:\n" + p.stdout + p.stderr)

        # `prep` LINKS each deck into its attempt directory, so a bare rglob
        # counts every trial twice.  The rendered deck is the real file.
        decks = sorted(p for p in d.rglob("*.fdf") if not p.is_symlink())
        assert len(decks) > 1, (
            "a benchmark is a set of points; got "
            + repr([str(x.relative_to(d)) for x in decks]))

        labels = []
        for deck in decks:
            text = deck.read_text()
            m = re.search(r"^SystemLabel\s+(\S+)", text, re.M)
            assert m, f"{deck.name} has no SystemLabel"
            labels.append(m.group(1))
            # The description said CPU.  A trial that turns the GPU on is
            # measuring a calculation nobody asked to run.
            g = re.search(r"^Diag\.ELPA\.GPU\s+(\S+)", text, re.M)
            assert not (g and g.group(1).lower().strip(".") == "true"), (
                f"{deck.name} enables the GPU against the description's "
                f"use_gpu = false -- the Job Prep UI's answer was "
                f"overridden by a pin (web/task-setup.md § 6.2)")
        assert len(set(labels)) == len(labels), (
            f"trials share a SystemLabel {labels} -- they will warm-start off "
            f"each other's .DM and the timings stop being comparable")

        js = sorted(d.rglob("job-set.json"))
        assert js, sorted(str(x.relative_to(d)) for x in d.rglob("*"))
        plan = _json.loads(js[0].read_text())
        assert plan["kind"] == "sweep", plan["kind"]
        assert len(plan["jobs"]) == len(decks), (
            [j.get("script") for j in plan["jobs"]])
        # A CPU sweep asks for no GPU.  `gres` set here would queue every
        # trial behind a GPU node it never uses.
        for j in plan["jobs"]:
            res = j.get("resources") or {}
            assert not (j.get("gres") or res.get("gres")), (
                f"a CPU trial asks for {j.get('gres') or res.get('gres')!r}")
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_the_cell_gate_s_notices_reach_the_person():
    """Every structure door runs the same cell gate, and it answers with
    NOTICES as well as refusals: a refusal is the door's 400, but a notice is a
    box the gate accepted and wants read.  The endpoint returned them on every
    send and the browser dropped them, so somebody whose cell was questioned
    found out from the run instead of from the page.

    A notice does not hold the WRITE back — the files are the person's own
    parameters.  It holds back the NAVIGATION, because a page that jumps to the
    next tab is a page whose warning was never read."""
    src = (ROOT / "molbuilder/web/static/lib/task-handover.js").read_text()
    # anchored after the write loop; `toWrite` is the null-filtered list
    # (a transport hand-over is ONE file, P7b)
    body = src.split("const written = toWrite.map", 1)[1]
    assert "out.notices" in body, "the gate's notices are still dropped"
    i_notice = body.index("out.notices")
    i_nav = body.index('window.location.href = "/task-setup"')
    assert i_notice < i_nav, "the page navigates before the notices are shown"
    guard = body[i_notice:i_nav]
    assert "return;" in guard, (
        "notices are shown and then navigated past — nobody reads them")
    # The gate's four-key contract spells the badness key `level`
    # (periodicity_gate.py, since 2026-08-03) -- this pin read `n.severity`
    # until 2026-08-21, asserting the very bug (E-B10) that kept the error
    # arm from ever firing.
    assert "n.level" in guard, (
        "the notice badness key is `level` (the gate's four-key "
        "contract), not `severity`")


def test_a_refused_cell_is_the_door_s_400_not_a_500(web_client):
    """`checked_periodicity` RAISES on a box it will not accept, and the app
    turns that into a 400 carrying the gate's own sentence.  The hand-over runs
    the same gate, so a refusal has to leave as the same answer — a 500 would
    show a stack trace where the reason belongs."""
    env = _envelope()
    env["structure"]["metadata"] = {
        "regions": {}, "cell": [[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]],
        "cell_origin": None, "axis_kind": None, "vacuum": None,
    }
    r = web_client.post("/api/task-setup/handover", json=dict(
        env, engine="siesta", name="bad", params={"system_label": "bad"}))
    assert r.status_code in (200, 400), r.status_code
    if r.status_code == 400:
        assert (r.get_json() or {}).get("error"), "a refusal with no reason"


def test_the_file_list_shows_the_structure_the_calculation_is_of():
    """The list said "Two files" while the folder holds four — so the page
    whose job is showing you the folder omitted the geometry, on the screen
    where a person checks a description before a week of compute.

    It is looked up **by the name the description gives it**, never by
    globbing for a `.xyz`: globbing answers "is there a geometry here", and the
    case worth showing is a folder holding somebody else's structure and not
    its own.  `prep` refuses on that, late; this says it where it can be fixed.
    """
    html = (ROOT / "molbuilder/web/templates/task_setup.html").read_text()
    for probe in ("ts-f-struct", "ts-f-side", "ts-f-struct-name", "ts-f-side-name"):
        assert probe in html, f"the file list has no {probe} row"
    assert "Two files, into the folder above." not in html, (
        "the hint still claims two files")

    src = VIEWER.read_text()
    body = src.split('markFile("ts-f-tmpl"', 1)[1].split("A hand-over:", 1)[0]
    assert "ref.source" in body, "the structure row is not driven by the description"
    assert ".endsWith(\".xyz\")" not in body, (
        "the structure is being found by globbing rather than by name")
    assert 'markFile("ts-f-struct"' in body and 'markFile("ts-f-side"' in body


def test_a_bench_edit_repaints_the_row_it_changed():
    """`syncFromModel` is where every editing verb ends, and it re-rendered
    everything except the machine card.  So `addPoint` put the point in the
    model and in the JSON on screen while the row went on showing the old
    chips — the bench panel looked inert while the file underneath it moved."""
    src = VIEWER.read_text()
    body = src.split("async function syncFromModel", 1)[1].split("\n}", 1)[0]
    for verb in ("renderStages", "renderMachine", "renderNext", "refreshPickers"):
        assert verb + "(" in body, f"syncFromModel does not repaint via {verb}"


def test_the_editor_is_mounted_once_even_under_concurrent_callers():
    """`ensureEditor` is async and its guard was on the RESULT, so two callers
    arriving before the first finished both saw `_cm === null` and both
    constructed an editor into `#ts-editor`.  Three were stacked in the live
    page: every edit went to the newest and every reading came from the
    oldest, which is what made the whole panel look dead.

    The guard has to be on the PROMISE for it to hold across the await."""
    src = VIEWER.read_text()
    body = src.split("function ensureEditor", 1)[1].split("\n}", 1)[0]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "async" not in src.split("function ensureEditor", 1)[0][-20:], (
        "ensureEditor is async again — its guard cannot span the await")
    assert "_cmBooting" in body, "nothing caches the in-flight mount"


def test_changing_the_shape_does_not_discard_the_table():
    """Picking a shape turns a hand-over into a proposal — stages, varies and a
    seeded bench.  That happens ONCE.  Re-running it on the second click meant
    building a two-stage table with its overrides and a bench grid, changing
    your mind about the shape, and losing all of it without a word.

    Seen end to end: two stages varying mesh_cutoff 200/500 and a three-setting
    bench became one bare `coarse` stage and the seed."""
    src = VIEWER.read_text()
    body = src.split("function setShape", 1)[1].split("\n}", 1)[0]
    i_guard = body.find('_task.schema === "molbuilder/task@1"')
    i_build = body.find("proposedFromHandover")
    assert i_guard != -1, "nothing stops the proposal being rebuilt"
    assert i_guard < i_build, (
        "the rebuild runs before the guard, so the table is discarded first")
    guard = body[i_guard:i_build]
    assert "_task.shape = shape" in guard, "the shape is not simply edited"


# --------------------------------------------------------------------- #
#  The value SHAPE reaches the tab (user, 2026-08-20)                    #
# --------------------------------------------------------------------- #

def test_both_pickers_payloads_carry_the_value_shape(web_client):
    """A bool or enum parameter edits through a dropdown of its legal
    values, and the tab can only build one by asking the catalogue -- so
    BOTH payloads carry `type`/`choices` (+ the sweepable's `default`,
    which births a row at its value in force).  Until 2026-08-20 the
    sweepable payload had no type at all, and every added setting was born
    as the number 1 -- `use_gpu` included."""
    sw = web_client.get("/api/task-setup/sweepable?engine=siesta").get_json()
    items = {i["name"]: i for i in sw["items"]}
    assert items["use_gpu"]["type"] == "bool"
    assert items["diag_algorithm"]["type"] == "enum"
    assert items["diag_algorithm"]["choices"] == [
        "ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE"]
    assert items["diag_algorithm"]["default"] == "ScaLAPACK"

    cols = web_client.get("/api/task-setup/columns?engine=siesta").get_json()
    citems = {i["name"]: i for i in cols["items"]}
    assert citems["diag_algorithm"]["type"] == "enum"
    assert citems["diag_algorithm"]["choices"] == [
        "ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE"]


def test_the_viewer_dispatches_widgets_on_the_shape_not_the_look():
    """Source-text pins (this page has no node harness -- the live browser
    walk covers behavior; these keep the structure from regressing):

      * the ONE widget rule exists (`legalValues`) and BOTH surfaces ask
        it -- the machine card's adder and the stage table's cell;
      * a new row is born at its value in force, never the literal "1";
      * a cell's VALUE is read by the declared type too, not by the look of
        what was typed."""
    src = (STATIC / "task-setup" / "viewer.js").read_text()
    assert "function legalValues(" in src
    assert src.count("legalValues(") >= 3, (
        "both surfaces must ask the one widget rule")
    assert 'addPoint(sel.value, "1")' not in src, (
        "a row born as the literal 1 is the bug this closed")
    assert "String(valueInForce(sel.value))" in src
    # This asserted the ONE case the writer handled -- `ov[col] = text ===
    # "true"` -- and passed for seven months over a `setCell` that guessed
    # every other type from `Number(text)`.  The rule is the lookup, not the
    # bool branch (2026-08-25).
    body = src.split("function setCell", 1)[1].split("\n}", 1)[0]
    assert "CELL_READERS[(_meta[col] || {}).type]" in body, (
        "the cell reader is not chosen by the parameter's declared type")
    assert "Number.isFinite(Number(text))" not in body, (
        "a value's look picks its type again -- that is the `use_gpu` bug "
        "and the `kgrid` one")


# `test_every_declared_type_has_a_cell_reader` moved to
# `tests/test_task_setup_cell_readers_js.py` on 2026-08-25, with the table
# it pins: the readers left `viewer.js` for `task-setup/cell-readers.js` so
# a test could run them instead of grepping for their names.  Key-existence
# was all this file could ever check -- `int3: (t) => t` would have passed.

def test_another_kinds_items_stay_out_of_the_optimization_surfaces(
        web_client):
    """`template.md` § 6.3's sibling rule at the two web doors
    (spectra-migration P0, 2026-08-20): the vibration items leaked into
    the Build form and the column picker the day their catalogue rows
    landed -- an item another calculation kind owns stays out by its own
    `calculations` declaration until the vibration surface threads the
    real kind."""
    import json as _json
    schema = _json.dumps(
        web_client.get("/api/build/schema/pyscf").get_json())
    cols = _json.dumps(
        web_client.get("/api/task-setup/columns?engine=pyscf").get_json())
    for name in ("already_relaxed", "compute_raman", "es_mode_selection",
                 "displacement_amplitude_ang"):
        assert f'"{name}"' not in schema, f"{name} leaked into the form"
        assert f'"{name}"' not in cols, f"{name} leaked into the columns"
    assert '"basis"' in schema, "shared items must stay"


def test_the_hand_over_carries_the_vibration_kind_end_to_end(web_client):
    """handover-procedure § 6, landed (spectra-migration P2, 2026-08-20):
    the hand-over is a Send button on the SAME endpoint.  With
    calculation=vibration the response's task.1st.json carries the kind,
    the template text carries the twelve vibration items, and the
    schema/columns doors serve the kind's form -- while an optimization
    hand-over stays byte-for-byte what it was (absent-is-a-state)."""
    import json as _json
    body = {
        "engine": "pyscf", "calculation": "vibration", "name": "V",
        "structure": {"elements": ["O", "H", "H"],
                      "positions": [[0, 0, 0.119], [0, 0.757, -0.477],
                                    [0, -0.757, -0.477]]},
        "params": {},
    }
    r = web_client.post("/api/task-setup/handover", json=body).get_json()
    assert r["ok"], r
    over = _json.loads(r["handover_text"])
    assert over["calculation"] == "vibration"
    assert "[item.already_relaxed]" in r["template_text"]
    assert "[item.compute_ir]" in r["template_text"]

    body2 = dict(body); body2.pop("calculation")
    r2 = web_client.post("/api/task-setup/handover", json=body2).get_json()
    over2 = _json.loads(r2["handover_text"])
    assert "calculation" not in over2, "absent IS the optimization state"

    sch = _json.dumps(web_client.get(
        "/api/build/schema/pyscf?calculation=vibration").get_json())
    assert '"already_relaxed"' in sch and '"es_mode_selection"' in sch
    cols = _json.dumps(web_client.get(
        "/api/task-setup/columns?engine=pyscf&calculation=vibration"
    ).get_json())
    assert '"compute_ir"' in cols

    src = (STATIC / "task-setup" / "viewer.js").read_text()
    assert '"freq"' in src and 'kind === "vibration"' in src, (
        "the receiver must propose the kind's own one-stage ladder")


def test_the_spectra_tab_sends_through_the_shared_door():
    """One door, two tabs (handover-procedure.md § 6; delivered by the
    archived spectra migration's P2): the spectra tab
    renders the CATALOGUE's vibration form and hands over through the
    same lib/task-handover.js door as /structure-optimization -- one
    spelling of the guards and the write order, two tabs."""
    core = (STATIC / "lib" / "spectra" / "core.js").read_text()
    assert '"pyscf", { calculation: "vibration" }' in core, (
        "the tab no longer fetches the catalogue's vibration schema")
    send_body = core.split("async function sendToTaskSetup", 1)[1]\
                    .split("// ----- Render button", 1)[0]
    assert 'calculation: "vibration"' in send_body, (
        "the send does not carry the kind (matching the string anywhere "
        "in the file is vacuous -- the schema fetch spells it too)")
    assert "taskHandover.send" in send_body, (
        "the tab bypasses the shared door")
    assert "/api/spectra/render" not in send_body, (
        "the send path still touches the retiring render route")

    lib = (STATIC / "lib" / "task-handover.js").read_text()
    assert 'o.calculation !== "optimization"' in lib, (
        "the lib must ride the kind only when it is not the default "
        "(absent IS the optimization state, handover-procedure § 6)")

    for page in ("spectra.html", "index.html"):
        html = (ROOT / "molbuilder/web/templates" / page).read_text()
        assert "lib/task-handover.js" in html, (
            f"{page} does not load the shared send door")
    spectra_html = (ROOT / "molbuilder/web/templates/spectra.html").read_text()
    assert "generate-btn" not in spectra_html, (
        "the retiring Generate flow is still offered next to Send")


def test_the_browser_hand_over_writes_the_cli_s_files(web_client, tmp_path):
    """The one-writer bar (handover-procedure.md § 4; the archived
    spectra migration's P2 acceptance): the files the browser's
    Send writes are the CLI's own.  The same water + defaults through
    (a) /api/task-setup/handover and (b) describe.build_description /
    write_description must agree BYTE-FOR-BYTE on the template and the
    structure pair, and on every identity field of the description head
    (run.created is the send's timestamp and is excluded by name)."""
    import json as _json
    import numpy as _np
    from molbuilder import describe as _D
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.pyscf.stages import vibration_stages
    from molbuilder.structure import Structure
    from molbuilder.identity import normalise_id

    elements = ["O", "H", "H"]
    positions = [[0.0, 0.0, 0.119], [0.0, 0.757, -0.477],
                 [0.0, -0.757, -0.477]]

    # (a) the browser door
    r = web_client.post("/api/task-setup/handover", json={
        "engine": "pyscf", "calculation": "vibration", "name": "vib",
        "structure": {"elements": elements, "positions": positions},
        "params": {},
    }).get_json()
    assert r["ok"], r
    browser_files = {f["name"]: f["text"] for f in r["structure_files"]}
    browser_files[r["template_name"]] = r["template_text"]
    over = _json.loads(r["handover_text"])

    # (b) the CLI door, for the identical inputs
    struct = Structure(elements=elements,
                       positions=_np.array(positions, dtype=float))
    src = tmp_path / "vib.xyz"
    src.write_text(struct.to_xyz())
    # The CLI arm folds the calculation's name into the engine's own
    # identity field BEFORE describing (jobset/_cli.py: "the template's
    # SystemLabel and the description's id cannot disagree") -- the
    # hand-over route holds the same rule, so the comparison must too.
    stages = vibration_stages()
    label = normalise_id("vib", what="name",
                         stage_names=tuple(s.name for s in stages))
    desc = _D.build_description(
        struct, PySCFConfig(job_name=label), stages,
        engine="pyscf", shape="flat", name="vib", source=str(src),
        calculation="vibration")
    dest = tmp_path / "calc"
    _D.write_description(desc, dest, struct=struct)

    cli_task = _json.loads((dest / "task.json").read_text())
    cli_files = {p.name: p.read_text() for p in dest.iterdir()
                 if p.name != "task.json"}

    for name, text in browser_files.items():
        assert name in cli_files, (
            f"the browser writes {name!r}; the CLI writes "
            f"{sorted(cli_files)}")
        assert text == cli_files[name], (
            f"{name} differs between the browser's send and the CLI")

    # The description head: same identity, same kind, same structure.
    assert over["engine"] == cli_task["engine"]
    assert over["calculation"] == cli_task["calculation"] == "vibration"
    assert over["run"]["id"] == cli_task["run"]["id"]
    assert over["structure"]["formula"] == cli_task["structure"]["formula"]
    assert over["structure"]["atoms"] == cli_task["structure"]["atoms"]


def test_the_next_steps_teach_the_bench_lane_and_true_ordinals():
    """U1(b)+(d): the tab's notes teach the whole bench flow through
    the launcher, and the --from ordinal comes from the FULL ladder
    (a disabled stage still occupies its number)."""
    src = VIEWER.read_text()
    body = src.split("function renderNext", 1)[1].split(
        "what has already run", 1)[0]
    # `run-config.toml` was an eighth needle until 2026-09-02.  It named a
    # file `prep run` folded in; the file is a report now and the taught
    # sequence does not mention it (`architecture.md` § 5.2).
    for needle in ("molbuilder jobset prep bench",
                   "molbuilder jobset launch bench",
                   "one job per resource shelf",
                   "summarize bench",
                   "prev.full + 1", "task.bench"):
        assert needle in body, f"next-steps lost {needle!r}"
    assert 'String(i).padStart' not in body, (
        "the enabled-filtered ordinal is back (E-T4)")
    # The declaration-time note for multi-point VALUE axes: the 2β rule
    # is BUILT (generator.md § 4.3a, 2026-08-21), so the U1 refusal
    # warning became teaching -- the sweep multiplies, and submission
    # groups by exact resource ask (one job per shelf, 2026-08-21).
    assert "sweep as a value axis" in src, \
        "the value-axis note left the bench table"
    assert "cpu-vs-gpu axis" in src, \
        "the use_gpu family note left the bench table"
    assert "exact resource ask" in src, \
        "the per-shelf teaching left the bench table"


def test_save_runs_gate_three_and_refuses_a_failing_preflight(web_client):
    """Gate ③ fires at save (G-1b, 2026-08-21): `workflow.md` § 9 names this
    door beside `describe` and dispatch, and until then only the codec ran
    here -- a stage override outside its field's bounds saved cleanly and
    failed at prep, on the cluster, hours later.  The refusal is the SAME
    function the CLI runs (`validation.task.preflight`), so the two
    surfaces answer alike."""
    d = _fresh_calc_dir()
    try:
        from molbuilder.identity import run_id
        bad_bounds = {
            "schema": "molbuilder/task@1", "engine": {"name": "siesta"},
            "shape": "hierarchical",
            "run": {"name": "x", "id": run_id("x", "H2"),
                    "created": "2026-08-16T00:00:00-07:00"},
            "structure": {"source": "s.xyz", "formula": "H2", "atoms": 2},
            "varies": ["mesh_cutoff"],
            # mesh_cutoff's schema range floors at 50 Ry; the codec has no
            # opinion about values, so only the preflight can catch this.
            "stages": [{"name": "coarse", "enabled": True,
                        "overrides": {"mesh_cutoff": 1.0}}]}
        r = web_client.post("/api/task-setup/save", json={
            "dest": str(d), "text": _json.dumps(bad_bounds)})
        assert r.status_code == 400, r.get_json()
        body = r.get_json()
        assert "preflight" in body["error"], body
        assert "mesh_cutoff" in body["error"], (
            "the refusal must name the failing field")
        assert not (d / "task.json").exists(), (
            "a description that fails its own preflight was written anyway")
        # The findings ride as data too, for the tab to render.
        assert any("mesh_cutoff" in f.get("message", "")
                   for f in body.get("findings", []))
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_the_two_engine_caches_are_keyed_by_engine():
    """R2-1: `loadSweepChoices` and `loadPresets` memoised with a bare
    `if (_x) return _x` -- the first folder's ENGINE was served to every
    folder opened after it, so a PySCF description showed SIESTA's
    machine rows, and applying a tier preset wrote SIESTA values into a
    PySCF ladder.  Both caches now follow `_cols`' settled pattern: the
    key is checked before the cache is served."""
    src = VIEWER.read_text()
    for fn, key in (("loadSweepChoices", "_sweepKey"),
                    ("loadPresets", "_presetsKey")):
        body = src.split(f"async function {fn}(", 1)[1].split(
            "\nasync function", 1)[0]
        assert f"{key} === key" in body, (
            f"{fn} caches without comparing the engine key")


def test_load_folder_resets_the_previous_folders_state_first():
    """U6 close: the hand-over and empty branches never wrote _task /
    _shape, so a description opened FIRST leaked into the next folder
    -- setShape(_shape) re-fired on the stale _task, syncFromModel()
    overwrote the editor with the previous folder's task.json, and
    Save was enabled over the wrong calculation.  The reset must come
    before any branch."""
    src = VIEWER.read_text()
    body = src.split("async function loadFolder(", 1)[1]
    head = body.split("const taskText", 1)[0]
    for needle in ("_task = null;", '_shape = "";', "_handover = null;"):
        assert needle in head, (
            f"loadFolder does not reset {needle!r} before branching")


def test_the_saves_preflight_findings_reach_the_person():
    """Gate ③'s warnings ride the OK save response ("the reader deserves
    what the CLI would have echoed"), and the tab must SHOW them --
    until the U6 close they went on the floor while loadFolder
    repainted the box to "loaded".  The refusal path's multi-line list
    needs the state body to keep its newlines."""
    src = VIEWER.read_text()
    tail = src.split("async function save(", 1)[1]
    assert "body.findings" in tail, "save() never reads the findings"
    assert tail.index("loadFolder") < tail.index("body.findings"), (
        "the findings must be painted AFTER loadFolder or it wipes them")
    css = (STATIC / "task-setup" / "style.css").read_text()
    body_rule = css.split(".ts-state-body", 1)[1].split("}", 1)[0]
    assert "pre-line" in body_rule, (
        "gate 3's multi-line refusal renders as one run-on line")


# --------------------------------------------------------------------- #
#  Which machine this is prepared FOR                                    #
# --------------------------------------------------------------------- #


class TestTheMachineChoiceIsAskedNotGuessed:
    """`preparing-for-another-machine.md` § 4: the choice is the user's
    whenever more than one machine could be meant.

    The tab does not carry its own rule for this -- the CLI refuses without
    a choice, and the tab asks the same question from the same records, so
    the two cannot come to different conclusions about what is ambiguous.
    """

    def test_the_card_reuses_the_shape_choosers_component(self, web_client):
        """No new panel and no new CSS: `.ts-choice`/`.opt` already exists
        for exactly this shape of question -- a choice with no default --
        and the stylesheet already carries its pressed/hover states.
        Inventing a second chooser would be two components for one idea."""
        body = web_client.get("/task-setup").data.decode()
        # `ts-target-*`, not `ts-machine-*`: the bench card below already
        # owns that name for the settings measured ON a machine, and when
        # this card briefly shared it (2026-08-22) getElementById handed
        # the bench card's renderer THIS card and the bench panel vanished.
        # Counting, not `in`, is what tells one card from two -- the
        # substring form of this assertion passed throughout the outage.
        assert body.count('id="ts-target-card"') == 1
        assert body.count('id="ts-target-choice"') == 1
        # the component is the shared one
        m = re.search(r'id="ts-target-choice"[^>]*class="[^"]*"'
                      r'|class="ts-choice" id="ts-target-choice"', body)
        assert m, "the machine chooser is not the shared .ts-choice component"
        # and it says the choice is required, in the card the design uses
        assert 'id="ts-target-needs"' in body

    def test_no_bespoke_css_was_added_for_it(self):
        """Every class the card uses already existed."""
        css = (ROOT / "molbuilder/web/static/task-setup/style.css").read_text()
        body = web_client_css = None  # noqa: F841 - readability
        for cls in ("ts-choice", "ts-needs", "ts-state", "ts-state-title",
                    "ts-state-body"):
            assert f".{cls}" in css, (
                f".{cls} is used by the machine card but is not in the "
                f"module's stylesheet -- it must not be invented inline")
        assert ".ts-target-" not in css, (
            "a bespoke class was added for the machine card; the shared "
            "components already cover it")
        # The one stylesheet change this card required is a CORRECTNESS fix
        # to a shared component, not a new one: `.ts-state` sets
        # `display: flex`, which ties with the browser's `[hidden]` rule and
        # wins on source order, so anything hidden by JS stayed on screen.
        # The folder state is never hidden, so the gap had never shown.
        assert ".ts-state[hidden]" in css, (
            "`.ts-state` can be hidden by JS now, so it needs the "
            "[hidden]-precedence guard the other shared components carry")

    def test_the_route_lists_the_records_and_says_when_a_choice_is_required(
            self, web_client, tmp_path, monkeypatch):
        """`choice_required` mirrors the CLI's C1 rule: any NAMED record
        makes 'this machine' ambiguous."""
        import json as _json
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        cfg = tmp_path / "home" / ".config" / "molbuilder"
        (cfg / "environments").mkdir(parents=True)
        rec = {"schema": "molbuilder/environment@2", "scheduler": "slurm",
               "domains": [], "topology": {}, "site": {}, "source": {}}
        (cfg / "environment.json").write_text(
            _json.dumps({**rec, "scheduler": "workstation"}))

        r = web_client.get("/api/task-setup/machines").get_json()
        assert r["ok"] and r["choice_required"] is False, r
        assert [m["name"] for m in r["machines"]] == ["(this machine)"]

        (cfg / "environments" / "sol.json").write_text(_json.dumps(rec))
        r = web_client.get("/api/task-setup/machines").get_json()
        assert r["choice_required"] is True, r
        assert {m["name"] for m in r["machines"]} == {"sol", "(this machine)"}
        assert all(m["readable"] for m in r["machines"])

    def test_an_unreadable_record_is_listed_and_marked(
            self, web_client, tmp_path, monkeypatch):
        """Shown rather than hidden: a record the user wrote and molbuilder
        cannot read is a thing to fix, and hiding it leaves them waiting."""
        import json as _json
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        cfg = tmp_path / "home" / ".config" / "molbuilder"
        (cfg / "environments").mkdir(parents=True)
        (cfg / "environments" / "sol.json").write_text('{"schema": "nope@9"}')
        r = web_client.get("/api/task-setup/machines").get_json()
        sol = next(m for m in r["machines"] if m["name"] == "sol")
        assert sol["readable"] is False
        assert "probe --write --name sol" in sol["summary"]

    def test_the_taught_command_carries_the_chosen_target(self):
        """A remote choice reaches the command the user copies -- and
        `launch` does not take it, because launching happens ON the
        machine."""
        src = VIEWER.read_text()
        assert "_targetArg()" in src
        # bench is per-STAGE now (§ 11), so both commands read `name`.
        assert 'prep bench " + name + _bundleArg() + _targetArg()' in src
        assert 'prep run " + name + from + _bundleArg() + _targetArg()' in src
        assert "_targetArg()" not in src.split("jobset launch")[1][:80], (
            "launch must not carry --target -- launching happens ON the "
            "machine, so there is nothing to target")


class TestEveryStageOffersBothThingsYouCanDoWithIt:
    """`task-setup.md` § 11: a stage is either something to MEASURE or
    something to RUN, and the page hands over the command for each rather
    than choosing between them."""

    def test_bench_is_offered_per_stage_not_only_for_the_first(self):
        """The card offered `prep bench` for `enabled[0]` alone -- a guess
        dressed as an answer.  The bench axes are declared once for the
        calculation, so every enabled stage can be measured; which one is
        worth measuring is a judgement the page states as a hint."""
        src = VIEWER.read_text()
        assert "enabled[0].st.name" not in src, (
            "the bench command is still hardwired to the first stage")
        # both commands are built inside the per-stage loop, from `name`
        assert 'prep bench " + name' in src
        assert 'prep run " + name' in src

    def test_the_order_that_matters_is_shown(self):
        """`summarize` writes run-config.toml and `prep run` APPLIES it, so
        skipping the middle step does not fail -- it quietly prepares a run
        with no measured verdict behind it.  The three commands appear in
        order, in one block."""
        src = VIEWER.read_text()
        i_prep = src.index("prep bench \" + name")
        i_launch = src.index("launch bench \" + name")
        i_summ = src.index("summarize bench \" + name")
        assert i_prep < i_launch < i_summ, "the bench order is not shown in order"

    def test_help_uses_the_pages_own_hint_component(self):
        """Not a new affordance and not another module's stylesheet:
        `form-schema.css` (which carries `.schema-help`) is not loaded on
        this page, and importing a parameter-form stylesheet to get a
        details widget would be the wrong wheel.  `.hint` is what this
        page already explains things with."""
        src = VIEWER.read_text()
        assert src.count('el("p", { class: "hint" }') >= 2, (
            "the per-stage blocks do not explain themselves with the "
            "page's own hint component")
        assert "Measure it" in src and "Run it" in src
        head = (ROOT / "molbuilder/web/templates/task_setup.html").read_text()
        assert "form-schema.css" not in head, (
            "the parameter form's stylesheet was pulled in for a hint")

    def test_the_hint_says_a_flag_still_overrides_the_card(self):
        """A person who filled the card and then wants something else for one
        prep needs to know a flag wins -- otherwise the only visible path is
        editing `task.json` again.

        *(It also asserted `run-config.toml` appeared, until 2026-09-02.
        That file is a report now and the UI does not name it: a benchmark
        reports, and what the run uses is this card.)*"""
        src = VIEWER.read_text()
        assert "--np / --omp / --time" in src


class TestTheTabShowsWhatAPrepWouldResolve:
    """`preparing-for-another-machine.md` § 5: the tab shows what `prep`
    resolved using the provenance `prep` already computes -- not a
    hand-written notice, which would be a second account of the same facts,
    free to drift from the one the terminal prints."""

    def _folder(self, tmp_path, monkeypatch, bundle_cfg=None):
        import json as _json
        from molbuilder.projects import PROJECTS_ROOT_ENV
        tree = tmp_path / "projects"
        b = tree / "P" / "optimization" / "w"
        b.mkdir(parents=True)
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
        monkeypatch.chdir(tmp_path)
        # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
        # working-directory step, which is gone (configuration.md § 2.1a) --
        # without naming the directory the write lands in a file nothing
        # opens, and the test passes having configured nothing.
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
        (tmp_path / "molbuilder.json").write_text(_json.dumps(
            {"script_generation": {"preamble": "source /home/local/conda.sh",
                                   "activation": "conda activate"}}))
        if bundle_cfg:
            (b / ".molbuilder.json").write_text(_json.dumps(bundle_cfg))
        return b

    def test_it_serves_the_same_facts_prep_prints(
            self, web_client, tmp_path, monkeypatch):
        b = self._folder(tmp_path, monkeypatch)
        r = web_client.get("/api/task-setup/resolved",
                           query_string={"dest": str(b)}).get_json()
        assert r["ok"], r
        # the shape config_provenance produces, not a re-description of it
        assert {s["scope"] for s in r["sources"]} >= {"machine", "project"}
        assert "script_generation.preamble" in r["effective"]
        assert "from" in r["effective"]["script_generation.preamble"]

    # `test_a_remote_target_is_warned_and_a_local_one_is_not` and
    # `test_the_warning_is_the_same_rule_the_cli_uses` were RETIRED
    # 2026-08-25 with the warning they pinned.  Both asserted § 3's rule as
    # it read before 2026-08-24 -- *"a preamble is a preference, so it stays
    # local"* -- which that section retracted in a boxed note the same day:
    # the bootstrap is a fact about the machine and rides its probed record,
    # and `runwrap` reads the record and nothing else.  The warning fired on
    # every named-target prep regardless, because it asked the local config
    # cascade a question the target's record had already answered.  A test
    # that keeps a retracted rule alive is worse than no test: it makes
    # deleting the dead code look like a regression.

    def test_the_provenance_list_uses_the_pages_facts_component(self):
        css = (ROOT / "molbuilder/web/static/task-setup/style.css").read_text()
        head = (ROOT / "molbuilder/web/templates/task_setup.html").read_text()
        assert 'class="ts-facts" id="ts-resolved"' in head
        # hidden by JS, so it needs the [hidden]-precedence guard
        assert ".ts-facts[hidden]" in css, (
            "`.ts-facts` sets display:grid and is now hidden by JS -- "
            "without the guard it stays in the layout")


def test_the_sheet_names_no_token_that_does_not_exist():
    """**A gap the raw-colour test leaves open, found 2026-08-27.**

    `test_the_sheet_writes_no_raw_palette_colour` forbids a hex literal —
    so `var(--danger, #e06c6c)` fails it and `var(--danger)` passes. But
    `--danger` is defined nowhere: the palette calls it `--error`. A
    `var()` naming a token that does not exist resolves to *nothing*, so
    the colour is simply unset and the rule silently does not apply.

    Passing the first test made the second failure invisible, which is the
    shape worth guarding: the rule that catches the loud mistake let the
    quiet one through.
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "molbuilder/web/static"
    defined = set()
    for f in root.rglob("*.css"):
        defined |= set(re.findall(r"^\s*(--[a-z0-9-]+)\s*:", f.read_text(),
                                  re.M))
    sheet = (root / "task-setup/style.css").read_text()
    # strip comments first -- prose about `var(--token)` is not a usage
    sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.S)
    used = set(re.findall(r"var\((--[a-z0-9-]+)", sheet))
    missing = sorted(used - defined)
    assert not missing, (
        f"task-setup/style.css names token(s) nothing defines: {missing}. "
        f"A var() with no definition resolves to nothing, so the property "
        f"silently does not apply.")


def test_every_page_class_in_the_markup_is_styled_somewhere():
    """**A class that matches no rule fails exactly like a token that names
    nothing: silently.** Found in the browser 2026-08-27 — renaming a card
    fixed its ids but left three classes as `ts-dest-*` while the sheet had
    moved to `.ts-reports-*`, so the layout rules simply did not apply and
    the inputs rendered at their default width.

    Only this page's own prefixes are checked: `card`, `hint`, `btn` and
    friends are the app's and live elsewhere.
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "molbuilder/web"
    html = (root / "templates/task_setup.html").read_text()
    used = set()
    for attr in re.findall(r'class="([^"{}]+)"', html):
        used |= {c for c in attr.split() if c.startswith(("ts-", "ps-"))}
    styled = set()
    for f in (root / "static").rglob("*.css"):
        styled |= set(re.findall(r"\.((?:ts|ps)-[a-z0-9-]+)", f.read_text()))
    # classes the JS toggles rather than the sheet naming them directly
    from_js = set()
    for f in (root / "static").rglob("*.js"):
        from_js |= set(re.findall(r"[\"'`]((?:ts|ps)-[a-z0-9-]+)", f.read_text()))
    orphans = sorted(used - styled - from_js)
    assert not orphans, (
        f"class(es) in task_setup.html that no stylesheet and no script "
        f"ever names: {orphans}. A class matching nothing applies nothing, "
        f"and says so nowhere.")


# --------------------------------------------------------------------- #
#  The TRANSPORT hand-over (archive/2026-09-01-transport-design.md § 4.1, P7b) +           #
#  the slot picker's describe seam                                      #
# --------------------------------------------------------------------- #

_T_CITE = "_t_transport/optimization/Relax/01_only/run-0"


def _cited_junction(*, concluded=True):
    """A minimal citable directory (4.1b form A: one .fdf + one .XV
    together).  ``concluded=False`` writes a run RECORD that does not
    conclude (mid-run / force-stopped); the classification itself
    never needs molbuilder's layout."""
    calc = ROOT / "projects/_t_transport/optimization/Relax"
    attempt = calc / "01_only" / "run-0"
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "Relax_01_only.fdf").write_text(
        "SystemLabel Relax\nMeshCutoff 250.0 Ry\nPAO.BasisSize SZ\n"
        "XC.functional GGA\nXC.authors PBE\n")
    bohr = 1.0 / 0.529177210903
    (attempt / "Relax.XV").write_text(
        f"  {10*bohr:.8f} 0.0 0.0  0.0 0.0 0.0\n"
        f"  0.0 {10*bohr:.8f} 0.0  0.0 0.0 0.0\n"
        f"  0.0 0.0 {10*bohr:.8f}  0.0 0.0 0.0\n"
        "  2\n"
        f"  1  6  0.0 0.0 0.0  0.0 0.0 0.0\n"
        f"  1  6  0.0 0.0 {1.3*bohr:.8f}  0.0 0.0 0.0\n")
    if concluded:
        (attempt / "Relax_01_only-run0.concluded").write_text("rc=0\n")
    else:
        (attempt / "run.json").write_text("{}")
    return calc


def test_transport_describe_answers_the_finished_description(web_client):
    """No hand-over for the composite (user ruling 2026-08-29): nothing
    is awaiting, so /api/transport/describe answers with the COMPLETE
    task.json -- one file, readable by the shipped reader, knobs on the
    device stage's overrides promoted through varies."""
    import pathlib as _pl
    import tempfile as _tf
    _cited_junction()
    try:
        r = web_client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T",
            junction=_T_CITE, bias=[0.0, 0.2],
            overrides={"transmission_n_points": 101}))
        assert r.status_code == 200, r.get_json()
        out = r.get_json()
        assert out["ok"] is True
        files = out["files"]
        assert [f["name"] for f in files] == ["task.json"], (
            "floor 2 is task.json ALONE")
        # the text is a real description: the shipped reader accepts it
        from molbuilder.task import read_task
        with _tf.TemporaryDirectory() as td:
            probe = _pl.Path(td) / "task.json"
            probe.write_text(files[0]["text"])
            task = read_task(probe)
        assert task.calculation == "transport"
        assert task.slots == {"junction": _T_CITE}
        assert task.bias == (0.0, 0.2)
        assert task.shape == "hierarchical"
        assert [s.name for s in task.stages] == [
            "seed", "electrode_L", "electrode_R", "device",
            "transmission"]
        dev = next(s for s in task.stages if s.name == "device")
        assert dev.overrides == {"transmission_n_points": 101}
        assert task.varies == ("transmission_n_points",)
    finally:
        _shutil.rmtree(ROOT / "projects/_t_transport",
                       ignore_errors=True)


def test_the_handover_door_refuses_transport_by_name(web_client):
    r = web_client.post("/api/task-setup/handover", json=dict(
        engine="siesta", calculation="transport", name="T",
        junction=_T_CITE, bias=[0.0]))
    assert r.status_code == 400
    assert "/api/transport/describe" in r.get_json()["error"]


def test_transport_describe_refuses_a_citation_the_tree_lacks(web_client):
    r = web_client.post("/api/transport/describe", json=dict(
        engine="siesta", name="T",
        junction="_t_nope/optimization/Gone@run-0", bias=[0.0]))
    assert r.status_code == 400
    assert "not a directory" in r.get_json()["error"]


def test_transport_describe_refuses_a_bias_off_equilibrium(web_client):
    _cited_junction()
    try:
        r = web_client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T",
            junction=_T_CITE, bias=[0.2, 0.4]))
        assert r.status_code == 400
        assert "0.0" in r.get_json()["error"]
    finally:
        _shutil.rmtree(ROOT / "projects/_t_transport",
                       ignore_errors=True)


def test_describe_attempt_reads_the_attempts_own_deck(web_client):
    """The describe seam: fdf-is-truth, plus the honest concluded
    line."""
    _cited_junction()
    try:
        r = web_client.get("/api/transport/describe_attempt?path="
                           "_t_transport/optimization/Relax/01_only/run-0")
        assert r.status_code == 200, r.get_json()
        out = r.get_json()
        assert out["ok"] and out["concluded"] is True
        assert "SZ" in out["summary"] and "250" in out["summary"]
        assert "CONCLUDED (rc=0)" in out["summary"]
    finally:
        _shutil.rmtree(ROOT / "projects/_t_transport",
                       ignore_errors=True)


def test_describe_attempt_names_both_unconcluded_states(web_client):
    _cited_junction(concluded=False)
    try:
        r = web_client.get("/api/transport/describe_attempt?path="
                           "_t_transport/optimization/Relax/01_only/run-0")
        out = r.get_json()
        assert out["concluded"] is False
        assert "still running" in out["summary"]
        assert "force-stopped" in out["summary"]
    finally:
        _shutil.rmtree(ROOT / "projects/_t_transport",
                       ignore_errors=True)


def test_describe_attempt_stays_inside_the_tree(web_client):
    r = web_client.get("/api/transport/describe_attempt?path=../../etc")
    assert r.status_code == 400


import re as _re


def test_the_viewer_carries_no_transport_handover_arm():
    """No hand-over for the composite (user ruling 2026-08-29): the
    Transport tab writes the finished task.json itself, so a transport
    task.1st.json never exists and Task setup's PROPOSAL machinery must
    carry no transport branch -- dead arms grow stale rulings.
    (Description mode may still know the kind: the file-list panel
    tells the transport truth about a SAVED description.)"""
    src = (ROOT / "molbuilder/web/static/task-setup/viewer.js").read_text()
    proposal = src.split("function proposedFromHandover(", 1)[1].split(
        "\nfunction ", 1)[0]
    assert '"transport"' not in proposal, (
        "a transport branch is back in the hand-over proposal -- the "
        "tab describes directly (POST /api/transport/describe)")
    shape = src.split("function setShape(", 1)[1].split(
        "\nfunction ", 1)[0]
    assert '"transport"' not in shape


def test_the_transport_page_loads_the_handover_door():
    """Found live 2026-08-29: the composite card's Send said
    'lib/task-handover.js is not loaded' because the template never
    included the door the other two sender tabs load."""
    tpl = (ROOT / "molbuilder/web/templates/transport_calculation.html"
           ).read_text()
    assert "lib/task-handover.js" in tpl


def test_transport_describe_refuses_a_sealed_override_by_name(web_client):
    """The electronic contract is the citation's to say (ruling Q5) --
    refused at the DESCRIBE door, from the same constant prep refuses
    from, while changing it is still free."""
    _cited_junction()
    try:
        r = web_client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T",
            junction=_T_CITE, bias=[0.0],
            overrides={"basis_size": "DZP"}))
        assert r.status_code == 400
        assert "citation's to say" in r.get_json()["error"]
    finally:
        _shutil.rmtree(ROOT / "projects/_t_transport",
                       ignore_errors=True)


def test_describe_attempt_names_a_recorded_contract(web_client):
    """4.1b's third shade at the seam: a pair whose sidecar carries
    info.calculation answers contract="cited" and says RECORDED."""
    import numpy as _np
    from molbuilder.structure import Structure as _S
    from molbuilder.workingcopy_structure import StructureCodec as _C
    d = ROOT / "projects/_t_transport/exported"
    d.mkdir(parents=True, exist_ok=True)
    s2 = _S(elements=["Au"] * 12 + ["S", "C", "C", "S"] + ["Au"] * 12,
            positions=_np.array([[1.0, 1.0, float(i)] for i in range(28)]),
            regions={"L-electrode": list(range(12)),
                     "R-electrode": list(range(16, 28))},
            cell=_np.diag([8.0, 8.0, 40.0]))
    s2.info = {"calculation": {"engine": "siesta",
                               "contract": {"basis_size": "DZP"},
                               "source": "Relax.fdf",
                               "source_sha256": "c" * 64}}
    _C().write(s2, d / "junction.xyz")
    try:
        r = web_client.get("/api/transport/describe_attempt"
                           "?path=_t_transport/exported")
        out = r.get_json()
        assert out["ok"] and out["form"] == "structure"
        assert out["contract"] == "cited"
        assert "RECORDED" in out["summary"]
    finally:
        import shutil as _shutil
        _shutil.rmtree(ROOT / "projects/_t_transport",
                       ignore_errors=True)


def test_every_command_the_page_teaches_is_a_REAL_cli_verb():
    """The page composes `molbuilder jobset …` command lines for the user
    to paste on the cluster.  That is the CLI's grammar living in a second
    place, and the only thing that keeps the two together is this check.

    The pin beside it asserts the STRINGS are still present -- which proves
    the page still says them, not that they still work.  Verbs in this
    project do get renamed (`molbuilder bench` folded into `jobset`
    2026-08-17; the bare `serve` form retired 2026-08-28), and under a
    presence-only pin the page would go on teaching a dead command with
    every test green.

    So this resolves each one against the live click tree.
    """
    import re

    import click

    from molbuilder.jobset._cli import jobset_group

    src = VIEWER.read_text()
    taught = set(re.findall(r'"molbuilder jobset (\w+) (\w+)', src))
    assert taught, "no jobset commands found in the page -- re-anchor this"

    for verb, kind in sorted(taught):
        cmd = jobset_group.commands.get(verb)
        assert cmd is not None, (
            f"the page teaches `molbuilder jobset {verb} …`, and `jobset` "
            f"has no such command.  It offers: "
            f"{', '.join(sorted(jobset_group.commands))}")
        # …and the kind it passes must be one the verb accepts.
        choices = next(
            (p.type.choices for p in cmd.params
             if isinstance(p, click.Argument)
             and getattr(p.type, "choices", None)), None)
        if choices is not None:
            assert kind in choices, (
                f"the page teaches `jobset {verb} {kind}`, but {verb} takes "
                f"{choices}")

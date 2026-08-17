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


def test_the_tab_writes_nothing_yet(web_client):
    """Save is present but disabled, and the page says why.

    A page that looks like it can save and cannot is worse than one that says
    so.  When saving lands, this test changes on purpose.
    """
    body = web_client.get("/task-setup").data.decode()
    m = re.search(r"<button[^>]*id=\"ts-save\"[^>]*>", body)
    assert m, "the Save button is gone"
    assert "disabled" in m.group(0), "Save is enabled but nothing writes yet"
    assert "jobset describe" in body, "the page does not say what writes today"


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
    for hardcoded in ("application/json", '"javascript"', "'javascript'"):
        assert hardcoded not in src, (
            f"{hardcoded!r} is hard-coded in the tab — the suffix map in "
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


def test_handover_writes_a_partial_that_declares_itself(web_client):
    """It writes the template + `task.1st.json`, and that file says what it is.

    The schema is deliberately NOT `molbuilder/task@1`: without `shape` it
    would fail that reader, and `check_schema` refuses a wrong artifact by
    name.  `_what` and `awaiting` exist because a person reads this file in an
    editor and should not need a document open beside it.
    """
    d = _fresh_calc_dir()
    try:
        body = dict(_envelope(), engine="siesta", dest=str(d),
                    name="probe calc", params={"system_label": "probe"})
        r = web_client.post("/api/task-setup/handover", json=body)
        assert r.status_code == 200, r.get_json()
        out = r.get_json()
        assert out["ok"] is True
        assert "task.1st.json" in out["files"]

        h = _json.loads((d / "task.1st.json").read_text())
        assert h["schema"] == "molbuilder/task-handover@1"
        assert "shape" not in h and "stages" not in h, (
            "the hand-over must not assert a shape — it is required with no "
            "default because inferring it hands you a tree you did not ask for")
        assert h["awaiting"] == ["shape", "stages"]
        assert h["_what"], "the file does not say what it is"
        assert list(d.glob("*.template.toml")), "no template was written"
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_handover_refuses_onto_a_described_calculation(web_client):
    """One job per folder (`job-contracts.md § 2.1` Rule 1).  Overwriting a
    description with a form's contents is the worst thing this could do."""
    d = _fresh_calc_dir()
    try:
        (d / "task.json").write_text("{}")
        body = dict(_envelope(), engine="siesta", dest=str(d),
                    name="probe calc", params={"system_label": "probe"})
        r = web_client.post("/api/task-setup/handover", json=body)
        assert r.status_code == 409, r.status_code
        assert "one job per folder" in r.get_json()["error"]
    finally:
        _shutil.rmtree(ROOT / "projects/_t_handover", ignore_errors=True)


def test_handover_refuses_outside_the_roots(web_client):
    """The picker's roots guard covers this door like every other."""
    body = dict(_envelope(), engine="siesta", dest="/tmp",
                name="x", params={"system_label": "probe"})
    r = web_client.post("/api/task-setup/handover", json=body)
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

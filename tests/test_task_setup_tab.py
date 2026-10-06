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
VIEWER = STATIC / "task-setup/viewer.js"
SHEET  = STATIC / "task-setup/style.css"


# --------------------------------------------------------------------- #
#  The page                                                             #
# --------------------------------------------------------------------- #

def _declarations(css: str) -> list[tuple[str, str]]:
    """(property, value) pairs, comments stripped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(m.group(1).strip(), m.group(2).strip())
            for m in re.finditer(r"([a-z-]+)\s*:\s*([^;{}]+)[;}]", css)]


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


# --------------------------------------------------------------------- #
#  The controller                                                       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("endpoint", ["/api/files/read", "/api/files/list"])
def test_the_tab_reuses_the_shipped_files_api(endpoint, web_client):
    """No new endpoint was added for this tab; both already ship."""
    r = web_client.get(endpoint, query_string={"path": ""})
    assert r.status_code != 404, f"{endpoint} is missing"


# --------------------------------------------------------------------- #
#  The hand-over (`stages.md` § 6.5a)                                    #
# --------------------------------------------------------------------- #

import json as _json


def _folder(client, d):
    """The folder answer -- the one door Task setup reads a folder through
    (`web/web-api.md`, `/api/task-setup/folder`)."""
    return client.get("/api/task-setup/folder?dir=" + str(d)).get_json()


def _fresh_calc_dir(root):
    """A directory inside the configured root — the picker refuses anything
    outside it, which is the guard working, not a test problem.

    Takes the root, so the tree is wherever the caller's
    `isolated_projects_root` put it.  It was
    `ROOT / "projects/_t_handover/..."` until 2026-09-06 -- inside the
    developer's own data, which a crashed run left behind.
    """
    d = root / "handover/optimization/probe_calc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _envelope():
    """Through the ONE builder (`tests/support/envelope.py`), so a field the
    envelope grows lands here rather than being forgotten in a hand-listed
    copy."""
    from support.envelope import envelope
    return {"structure": envelope(["H", "H"],
                                  [[0, 0, 0], [0, 0, 0.74]])}


def test_handover_renders_and_writes_nothing(web_client, isolated_projects_root):
    """The endpoint returns TEXTS; the browser writes them.

    `web/projects.md` § 1 puts raw bytes in the content-blind file layer that
    "every tab can use" — a tab that writes files itself bypasses the roots
    guard, the lock, the uniform envelope and the sidebar re-list.  What is
    genuinely server-side is the RENDER: only Python can turn a config into
    `<label>.template.toml`.
    """
    d = _fresh_calc_dir(isolated_projects_root)
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
        pass    # tmp_path removes the tree


def test_save_refuses_outside_the_roots(web_client):
    """The save door owns `task.json` because it owns that schema — but it is
    still inside the picker's roots guard like every other write."""
    r = web_client.post("/api/task-setup/save",
                        json={"dest": "/tmp", "text": "{}"})
    assert r.status_code >= 400
    assert "root" in str(r.get_json().get("error", "")).lower()


# --------------------------------------------------------------------- #
#  T1 shape · T2 save                                                    #
# --------------------------------------------------------------------- #


def test_save_writes_the_description_and_reports_the_handover(web_client, isolated_projects_root):
    """The save door owns `task.json` — the same reason `/api/structure/save`
    owns the sidecar: a browser-authored, schema-stamped file that the loader
    would reject is the save-then-reload trap `projects.md` § 3 describes.

    It does NOT delete the hand-over; it reports that one is there, and the
    browser removes it through `projects.deleteEntry`.
    """
    d = _fresh_calc_dir(isolated_projects_root)
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
        pass    # tmp_path removes the tree


def test_save_refuses_rather_than_repairs(web_client, isolated_projects_root):
    """The text goes through `task.read_task` — the same door `prep` uses — so
    a browser cannot become a second, drifting writer of descriptions."""
    d = _fresh_calc_dir(isolated_projects_root)
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
        pass    # tmp_path removes the tree


# --------------------------------------------------------------------- #
#  T3 — the stage table edits                                            #
# --------------------------------------------------------------------- #


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


def test_one_point_is_the_trials_value_and_several_a_measurement():
    """A one-point non-machine row is what EVERY TRIAL runs with -- never the
    run's value, which is its run card's (`stages.md` § 6.8, § 6.8d; user,
    2026-09-30: *"trials only"*).  The card called it "chosen", which told a
    person their run was decided by a row `prep run` no longer reads.
    Several points are a measurement."""
    one = _row_badge("diag_algorithm", ["ELPA-1STAGE"], False)
    assert one["verdict"] == "every trial \u00b7 1 point"
    assert one["kind"] == "chosen", "the row lost its tint"
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


# --------------------------------------------------------------------- #
#  The checkpoint API (F3) and the two guards (F1, F2)                   #
# --------------------------------------------------------------------- #


def test_the_runtime_loads_before_every_other_script(web_client):
    """The registry can only hand out what registered with it, so it has to be
    parsed first — `molbuilder-runtime.js` before the sidebar and the tab."""
    body = web_client.get("/task-setup").data.decode()
    i_rt   = body.index("lib/molbuilder-runtime.js")
    i_tab  = body.index("task-setup/viewer.js")
    assert i_rt < i_tab, "the runtime loads after the tab's own script"


def test_the_column_picker_offers_no_run_setting(web_client):
    """A run setting is the rung's run card's, never a column (`stages.md`
    § 6.2, § 6.8d; plan § 5w K5): the machine's answers, and a person's --
    `restart`, `use_gpu`, the solver.  `restart` and `use_gpu` were columns
    until 2026-09-30, and a rung's `use_gpu` set as one reached its deck and
    not the scheduler's device ask (SO-C1)."""
    from molbuilder.template import run_settings
    # The set is the catalogue's own answer; that it holds the settings this
    # rule is about is asserted, so an empty answer cannot pass vacuously.
    assert {"restart", "use_gpu", "diag_algorithm", "mpi_np"} <= run_settings(
        "siesta")
    j = web_client.get("/api/task-setup/columns?engine=siesta").get_json()
    names = [i["name"] for i in j["items"]]
    assert names, "the picker offered nothing at all"
    assert not (run_settings("siesta") & set(names)), sorted(
        run_settings("siesta") & set(names))
    # ...and a value that binds EVERY rung of the folder's kind is not a
    # column either (template.md § 6.4 `shared`: "no stage overrides it").
    # Measured 2026-09-24: a transport folder's picker offered thirteen of
    # them, `mesh_cutoff` and `basis_size` among them -- each a per-rung
    # override `prep` refuses by name.  A rung-owned item stays a column,
    # and the payload names its owners, so the table can disable the
    # other rungs' cells.
    from molbuilder.template import catalogue, select
    cat = catalogue()
    shared = {it.name for it in select(cat, engine="siesta", shared=True)
              if "transport" in it.shared}
    assert shared, "the fixture catalogue marks nothing shared"
    t = web_client.get("/api/task-setup/columns?engine=siesta"
                       "&calculation=transport").get_json()
    by = {i["name"]: i for i in t["items"]}
    assert not (set(by) & shared), sorted(set(by) & shared)
    assert by["transmission_n_points"]["stages"] == ["transmission"]
    # ...while an optimization's picker, whose kind shares nothing, keeps
    # the same rows as controls: `mesh_cutoff` is a legitimate per-stage
    # column of a relaxation ladder.
    assert "mesh_cutoff" in names


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
    # OFFERED PER KIND, by the columns' own membership rule (`task-setup.md`
    # § 9): a tier is offered only where every field it carries may be a
    # column.  A transport rung owns none of the relaxation fields, so its
    # rows get no menu; a vibration ladder's relax rung owns them all.
    # Until 2026-09-24 every transport rung offered coarse/medium/tight
    # (plan W31, archived 2026-09-29), a menu whose every entry `prep` would refuse.
    t = web_client.get("/api/task-setup/presets?engine=siesta"
                       "&calculation=transport").get_json()
    assert t["ok"] and t["presets"] == [], t
    v = web_client.get("/api/task-setup/presets?engine=siesta"
                       "&calculation=vibration").get_json()
    assert v["ok"] and len(v["presets"]) == 3


def test_the_folder_template_is_what_an_empty_cell_names(web_client, isolated_projects_root):
    """`stages.md` § 6.2: a stage that sets nothing uses THE TEMPLATE'S value.

    So the whole point of the hand-over — the k-grid a person chose in the
    parameter tab — has to survive into what Task setup shows.  Before this
    endpoint the tab read the catalogue's default and named a number the job
    would not run whenever the sender had changed that parameter.
    """
    d = _fresh_calc_dir(isolated_projects_root)
    try:
        rendered = web_client.post("/api/task-setup/handover", json=dict(
            _envelope(), engine="siesta", name="probe",
            params={"system_label": "probe", "kgrid": [4, 4, 1],
                    "mesh_cutoff": 450.0})).get_json()
        # WHAT THE BROWSER WRITES: the hand-over's files, as the route
        # returned them -- the template beside the hand-over that names it.
        (d / rendered["template_name"]).write_text(rendered["template_text"])
        (d / rendered["handover_name"]).write_text(rendered["handover_text"])

        j = _folder(web_client, d)["template"]
        assert j["ok"], j
        assert j["name"] == rendered["template_name"]
        assert j["values"]["mesh_cutoff"] == 450.0, (
            "the value the parameter tab collected did not reach Task setup; "
            "the catalogue default (300.0) would be shown instead")
        assert j["values"]["kgrid"] == [4, 4, 1], j["values"].get("kgrid")
    finally:
        pass    # tmp_path removes the tree


def test_each_template_value_says_whose_it_is(web_client, isolated_projects_root):
    """`engines/template.md` § 6.6 obligation 2 (plan § 5w K7): the template
    a form hands over records whose each value is, and the folder door says
    it in the template's own words (`template.SOURCE_WORDS`).

    The form sends what it holds, a field nobody chose blank
    (`web/form-schema.md` § 1.1), and the one door reads a blank as not
    chosen: the field is written at THIS KIND's recommendation and recorded
    as nobody's.  Until K7 the form drew every default as a value and sent
    it, so no template could tell the person's 450 Ry from nobody's 300."""
    from molbuilder.template import one, read_template
    d = _fresh_calc_dir(isolated_projects_root)
    rendered = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", name="probe", calculation="vibration",
        params={"system_label": "probe", "mesh_cutoff": 450.0,
                "relax_force_tol": None, "basis_size": None})).get_json()
    assert rendered["ok"], rendered
    tmpl = read_template(rendered["template_text"])
    assert one(tmpl, "mesh_cutoff").source == "person"
    assert one(tmpl, "system_label").source == "person"
    # Not chosen: the vibration's own recommendation, held tighter than an
    # optimization's 0.02 (`template.md` § 6.3a), and nobody's.
    rft = one(tmpl, "relax_force_tol")
    assert (rft.value, rft.source) == (0.01, "default"), rft
    assert one(tmpl, "basis_size").source == "default"

    (d / rendered["template_name"]).write_text(rendered["template_text"])
    (d / rendered["handover_name"]).write_text(rendered["handover_text"])
    said = _folder(web_client, d)["template"]["said"]
    assert said["mesh_cutoff"] == "you set this", said["mesh_cutoff"]
    assert said["relax_force_tol"] == "not chosen", said["relax_force_tol"]


def test_a_folder_with_no_template_is_not_an_error(web_client, isolated_projects_root):
    """An empty folder is an ordinary state, not a failure — the cells fall
    back to the catalogue, which is exactly right when nothing was sent."""
    d = _fresh_calc_dir(isolated_projects_root)
    try:
        j = _folder(web_client, d)["template"]
        assert j["ok"] and j["name"] is None and j["values"] == {}
    finally:
        pass    # tmp_path removes the tree


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
        # Its two engine-file halves -- a `slab.XV` and a bare `slab.xyz`
        # written by hand to stand for a run -- retired 2026-10-04 (user:
        # "any fucking faking tests should be retired"; `process/testing.md`
        # § 6).


from conftest import write_pseudos as _pseudos_for


def _child_env_with_a_config(tmp_path):
    """Environment for a spawned `molbuilder`, with a machine config of its own.

    These two tests run the real CLI in a SUBPROCESS from the repo root, and
    it used to pick up a ``./molbuilder.json`` sitting there -- the developer's
    own file, which no test had put under control.  That is proving something
    with found state, and it ended the moment the machine config stopped being
    looked for in the working directory (`configuration.md` § 2.1a).

    So the child is given a root of its own, holding the one thing the render
    requires: a probed record, carrying the activation the generator reads
    (`configuration.md` § 4).  ``monkeypatch`` cannot reach across a
    process boundary, which is why the environment is built here rather than
    set on the parent.
    """
    import os
    root = tmp_path / "child-config-root"
    root.mkdir(parents=True, exist_ok=True)
    # THE CHILD'S MACHINE IS PROBED.  The child resolves its scope from the
    # env var below, not from this process's -- and prep refuses without a
    # record there (`running-a-job.md` § 3.1).
    from conftest import write_machine_record
    write_machine_record(at=root)
    env = dict(os.environ)
    env["MOLBUILDER_CONFIG_DIR"] = str(root)
    return env


def test_the_whole_chain_from_structure_to_rendered_deck(web_client, tmp_path, isolated_projects_root):
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
    d = _fresh_calc_dir(isolated_projects_root)
    try:
        # a periodic slab: a cell, a region label, a frozen atom
        cell = [[5.77, 0.0, 0.0], [2.885, 4.997, 0.0], [0.0, 0.0, 20.0]]
        # BUILT, not a two-atom envelope overwritten field by field: the
        # canonical dict carries the per-atom columns too, and replacing
        # `elements` alone left `atom_names` describing the old atoms.
        from support.envelope import envelope
        env = {"structure": envelope(
            ["Au", "Au", "S"],
            [[0, 0, 0], [2.885, 0, 0], [0, 0, 4.755]],
            regions={"frozen_atoms": [0], "slab": [0, 1]},
            cell=cell)}
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

        # The run card states its launch shape, as a run's must
        # (`architecture.md` § 5.2).
        described = {"schema": "molbuilder/task@1", "engine": over["engine"],
                     "shape": "flat", "run": over["run"],
                     "structure": over["structure"], "varies": [],
                     "execution": {"mpi_np": 1, "omp_threads": 1},
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
        assert kg and "8 0 0 0.5" in " ".join(kg.group(1).split()), (
            kg and kg.group(1))
        assert "MeshCutoff 350.0 Ry" in deck

        # the ATOMS, row by row, against the pair the hand-over wrote: the same
        # atoms in the same order, placed by the one rule
        # (`model/structure-periodicity.md` § 6.0) -- the design coordinates
        # plus the engine offset -- and the deck's own record states that
        # offset.  Until 2026-09-25 this compared them EQUAL, because a slab
        # authored at the origin was written untranslated; now "nothing lost"
        # means one translation for every atom, and the one the deck says.
        import numpy as _np
        from molbuilder import cell as _cell
        from molbuilder.deck_record import extract_engine_offset
        from molbuilder.workingcopy_structure import StructureCodec
        blk = re.search(r"%block AtomicCoordinatesAndAtomicSpecies(.*?)%endblock",
                        deck, re.S)
        fdf = _np.array([[float(v) for v in ln.split()[:3]]
                         for ln in blk.group(1).strip().splitlines()])
        design = StructureCodec().load(d / over["structure"]["source"])
        assert fdf.shape == (3, 3)
        # Nobody assigned an origin, so the pair the hand-over wrote leaves the
        # placement to the rule (Automatic) rather than freezing a corner.
        assert design.engine_offset is None, design.engine_offset
        offset = _cell.engine_offset(design)
        assert _np.allclose(fdf, design.positions + offset, atol=1e-6), (
            fdf, design.positions, offset)
        record = extract_engine_offset(deck)
        assert record is not None, "the deck does not say where it put the atoms"
        assert _np.allclose(record["applied_offset"], offset, atol=1e-7), record
        assert record["stated"] is False, record
    finally:
        pass    # tmp_path removes the tree


def test_a_dispersion_turned_off_on_the_form_is_off_in_the_deck(
        web_client, tmp_path, isolated_projects_root):
    """``dispersion = "none"`` from a form reaches the deck as no correction.

    "none" is the item's VALUE for no correction (`config/pyscf.py`'s note on
    the field).  Until 2026-09-28 the form's server side turned it into None,
    None is what an UNSET item reads as, the template wrote the item valueless
    and `prep` filled the class default: a person who chose "none" ran D3BJ.
    Every link returned ok -- so this follows the value through all of them,
    hand-over to template to `prep` to the deck `prep` renders.
    """
    import json as _json, subprocess, sys
    d = _fresh_calc_dir(isolated_projects_root)
    from support.envelope import envelope
    env = {"structure": envelope(
        ["O", "H", "H"],
        [[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]])}
    r = web_client.post("/api/task-setup/handover", json=dict(
        env, engine="pyscf", name="nodisp",
        params={"method": "DFT", "dispersion": "none"}))
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    for f in out["structure_files"]:
        (d / f["name"]).write_text(f["text"])
    (d / out["template_name"]).write_text(out["template_text"])
    blk = out["template_text"].split("[item.dispersion]", 1)[1].split("help", 1)[0]
    assert 'value = "none"' in blk, blk

    over = _json.loads(out["handover_text"])
    described = {"schema": "molbuilder/task@1", "engine": over["engine"],
                 "shape": "flat", "run": over["run"],
                 "structure": over["structure"], "varies": [],
                 "execution": {"threads": 1},
                 "stages": [{"name": "coarse", "enabled": True,
                             "overrides": {}}]}
    s = web_client.post("/api/task-setup/save",
                        json={"dest": str(d), "text": _json.dumps(described)})
    assert s.status_code == 200, s.get_json()

    p = subprocess.run(
        [sys.executable, "-m", "molbuilder.cli", "jobset", "prep", "run",
         "coarse", "--bundle", str(d)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
        env=_child_env_with_a_config(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    decks = sorted(d.glob("*_01_coarse.py"))
    assert decks, sorted(x.name for x in d.iterdir())
    deck = decks[0].read_text()
    assert 'mf.xc = "' in deck, "not a DFT deck -- the check below proves nothing"
    assert "mf.disp = " not in deck, [
        ln for ln in deck.splitlines() if "mf.disp" in ln]


def test_a_cpu_description_gets_a_cpu_benchmark(web_client, tmp_path, isolated_projects_root):
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
    d = _fresh_calc_dir(isolated_projects_root)
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
        pass    # tmp_path removes the tree


def test_a_refused_cell_is_the_door_s_400_not_a_500(web_client):
    """`checked_periodicity` RAISES on a box it will not accept, and the app
    turns that into a 400 carrying the gate's own sentence.  The hand-over runs
    the same gate, so a refusal has to leave as the same answer — a 500 would
    show a stack trace where the reason belongs."""
    env = _envelope()
    env["structure"]["metadata"] = {
        "regions": {}, "cell": [[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]],
        "engine_offset": None, "axis_kind": None, "vacuum": None,
    }
    r = web_client.post("/api/task-setup/handover", json=dict(
        env, engine="siesta", name="bad", params={"system_label": "bad"}))
    # Two equal rows: a flat box, an ERROR finding (`cell.no_volume`), so the
    # gate refuses and the door answers 400 with that sentence.
    assert r.status_code == 400, (r.status_code, r.get_json())
    assert "flat" in ((r.get_json() or {}).get("error") or ""), (
        "the 400 does not carry the gate's reason", r.get_json())


# --------------------------------------------------------------------- #
#  The value SHAPE reaches the tab (user, 2026-08-20)                    #
# --------------------------------------------------------------------- #

def test_the_run_card_offers_only_what_the_kind_carries(web_client):
    """A kind's run card offers the run settings that kind carries
    (`template.md` § 6.3's sibling rule, as the columns are narrowed):
    `restart` is an optimization's, and a vibration's card offered it -- a
    value its deck would ignore (the K5 review's C2, 2026-09-30)."""
    def names(kind):
        j = web_client.get("/api/task-setup/sweepable?engine=siesta"
                           f"&calculation={kind}").get_json()
        return {i["name"] for i in j["items"]}
    assert "restart" in names("optimization")
    assert "restart" not in names("vibration")
    assert "use_gpu" in names("vibration")


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

    from molbuilder.template import catalogue, one
    cols = web_client.get("/api/task-setup/columns?engine=siesta").get_json()
    citems = {i["name"]: i for i in cols["items"]}
    assert citems["relax_type"]["type"] == "enum"
    assert citems["relax_type"]["choices"] == list(
        one(catalogue(), "relax_type", engine="siesta").choices)

    # THE KIND'S OWN DEFAULT, never the general one -- what the hover calls
    # *Recommended* (the M11 review's PS-C12: a vibration folder was told
    # an optimization's 0.02).
    vib = web_client.get("/api/task-setup/columns?engine=siesta"
                         "&calculation=vibration").get_json()
    vitems = {i["name"]: i for i in vib["items"]}
    assert vitems["relax_force_tol"]["default"] == 0.01
    assert citems["relax_force_tol"]["default"] == 0.02


# RETIRED 2026-09-03 — test_the_viewer_dispatches_widgets_on_the_shape_not
# _the_look.  It opened "Source-text pins (this page has no node harness --
# the live browser walk covers behavior)", which is the admission that decides
# it (`process/testing.md` § 3a.1).  Its three claims, and where each went:
#
#   * the widget rule (`legalValues`) -- it counted call sites,
#     `src.count("legalValues(") >= 3`.  A count cannot tell a call that runs
#     from one moved into a branch nothing reaches.  Now driven:
#     test_task_setup_cell_types_e2e.py::test_a_bool_column_is_a_chooser_not
#     _a_box, mutation-verified against legalValues().
#   * the cell READER chosen by declared type -- already driven under node in
#     test_task_setup_cell_readers_js.py::test_a_cell_reads_as_its_declared
#     _type, so the pin was a duplicate.
#   * "a new row is born at its value in force, never the literal 1" -- an
#     absence assertion (`'addPoint(sel.value, "1")' not in src`), which
#     passes on any spelling of the same bug.  Not replaced: it states no
#     rule any document carries, and a test may not invent one.
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


def test_save_runs_gate_three_and_refuses_a_failing_preflight(web_client, isolated_projects_root):
    """Gate ③ fires at save (G-1b, 2026-08-21): `workflow.md` § 9 names this
    door beside `describe` and dispatch, and until then only the codec ran
    here -- a stage override outside its field's bounds saved cleanly and
    failed at prep, on the cluster, hours later.  The refusal is the SAME
    function the CLI runs (`validation.task.preflight`), so the two
    surfaces answer alike."""
    d = _fresh_calc_dir(isolated_projects_root)
    try:
        from molbuilder.identity import run_id
        bad_bounds = {
            "schema": "molbuilder/task@1", "engine": {"name": "siesta"},
            "shape": "hierarchical",
            "run": {"name": "x", "id": run_id("x", "H2"),
                    "created": "2026-08-16T00:00:00-07:00"},
            "structure": {"source": "s.xyz", "formula": "H2", "atoms": 2},
            "varies": ["kgrid"],
            # A k-point count of 0 is past kgrid's hard limit
            # (`engines/template.md` § 5.3); the codec has no opinion about
            # values, so only the preflight can catch this.  (A value merely
            # outside a recommended range is warned, and saves.)
            "stages": [{"name": "coarse", "enabled": True,
                        "overrides": {"kgrid": [0, 4, 4]}}]}
        r = web_client.post("/api/task-setup/save", json={
            "dest": str(d), "text": _json.dumps(bad_bounds)})
        assert r.status_code == 400, r.get_json()
        body = r.get_json()
        assert "preflight" in body["error"], body
        assert "kgrid" in body["error"], (
            "the refusal must name the failing field")
        assert not (d / "task.json").exists(), (
            "a description that fails its own preflight was written anyway")
        # The findings ride as data too, for the tab to render.
        assert any("kgrid" in f.get("message", "")
                   for f in body.get("findings", []))
    finally:
        pass    # tmp_path removes the tree


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

        # The `[hidden]`-precedence guard this used to pin by name was
        # covered by `test_css_hidden_attribute_audit.py`, which DERIVED the
        # ids JS toggles (58 of them, `#ts-target-state` among them) and
        # required a guard for each -- until that audit was retired on
        # 2026-09-10 (`082ba979`).  So the guard is UNTESTED now, and held by
        # review (process/code-audit.md § 3.1).

    def test_the_route_lists_the_records_and_says_when_a_choice_is_required(
            self, web_client, tmp_path, monkeypatch):
        """`choice_required` mirrors the CLI's C1 rule: any NAMED record
        makes 'this machine' ambiguous."""
        import json as _json
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        # ...and the box is probed: this moves the machine scope, and
        # prep refuses without a record (`running-a-job.md` § 3.1).
        from conftest import write_machine_record
        write_machine_record()
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
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        # ...and the box is probed: this moves the machine scope, and
        # prep refuses without a record (`running-a-job.md` § 3.1).
        from conftest import write_machine_record
        write_machine_record()
        cfg = tmp_path / "home" / ".config" / "molbuilder"
        (cfg / "environments").mkdir(parents=True)
        (cfg / "environments" / "sol.json").write_text('{"schema": "nope@9"}')
        r = web_client.get("/api/task-setup/machines").get_json()
        sol = next(m for m in r["machines"] if m["name"] == "sol")
        assert sol["readable"] is False
        assert "probe --write --name sol" in sol["summary"]

    # `test_the_taught_command_carries_the_chosen_target` stood here and read
    # `viewer.js` as text: `"_targetArg()" in src`, and absent from the 80
    # characters after "jobset launch".  Both were true of the source while
    # the page was broken -- choosing a machine never re-rendered the block,
    # so the line a person copied carried no `--target` at all.  Driven now,
    # against a real named record, by test_task_setup_prep_e2e.py::
    # test_choosing_a_machine_puts_it_in_the_command_you_copy.


class TestEveryStageOffersBothThingsYouCanDoWithIt:
    """`task-setup.md` § 11: a stage is either something to MEASURE or
    something to RUN, and the page hands over the command for each rather
    than choosing between them."""


class TestTheTabShowsWhatAPrepWouldResolve:
    """`preparing-for-another-machine.md` § 5: the tab shows what `prep`
    resolved using the provenance `prep` already computes -- not a
    hand-written notice, which would be a second account of the same facts,
    free to drift from the one the terminal prints."""

    def _folder(self, tmp_path, monkeypatch):
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
        # ...and the box is probed: this moves the machine scope, and
        # prep refuses without a record (`running-a-job.md` § 3.1).
        from conftest import write_machine_record
        write_machine_record()
        # ...and a NAMED record beside it, so which machine a fresh folder is
        # for is not decided yet: the card shows no queues, and is not hidden
        # (W54 C10 -- it answered `ok: false` here).
        from molbuilder.scheduler import (Environment,
                                          named_environment_path,
                                          write_environment)
        named = named_environment_path("sol")
        named.parent.mkdir(parents=True, exist_ok=True)
        write_environment(Environment(scheduler="workstation"), named)
        (tmp_path / "molbuilder.json").write_text(_json.dumps(
            {"launch": {"mode": "direct"}}))
        return b

    def test_it_serves_the_same_facts_prep_prints(
            self, web_client, tmp_path, monkeypatch):
        b = self._folder(tmp_path, monkeypatch)
        r = _folder(web_client, b)["provenance"]
        assert r["ok"], r
        # the shape config_provenance produces, not a re-description of it --
        # this machine's molbuilder.json.  WHICH MACHINE RECORD answers is the
        # prep answer's, a preview's too: it depends on the machine a prep
        # names (`test_prep_from_the_browser`).
        assert {s["scope"] for s in r["sources"]} == {"machine"}
        assert "launch.mode" in r["effective"]
        assert "from" in r["effective"]["launch.mode"]

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

    # `test_the_provenance_list_uses_the_pages_facts_component` stood here
    # and asserted the string `'class="ts-facts" id="ts-resolved"'` appeared
    # in the template -- that someone typed two attributes in one order.  It
    # said nothing about whether `loadResolved()` runs, reaches the route, or
    # writes a row.  The other half is now driven:
    # test_task_setup_prep_e2e.py::
    # test_choosing_a_machine_shows_what_a_prep_would_resolve reads the
    # rendered block and checks the preamble's value AND the file it came
    # from.  The `[hidden]` guard belonged to
    # test_css_hidden_attribute_audit.py, which derived `#ts-resolved` with
    # the other 57 -- retired on 2026-09-10 (`082ba979`), so the guard is
    # untested now and held by review (process/code-audit.md § 3.1).


# --------------------------------------------------------------------- #
#  The TRANSPORT hand-over (archive/2026-09-01-transport-design.md § 4.1, P7b) +           #
#  the slot picker's describe seam                                      #
# --------------------------------------------------------------------- #

_T_CITE = "transport/optimization/Relax/01_only/run-0"


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


def test_describe_attempt_stays_inside_the_tree(web_client):
    r = web_client.get("/api/transport/describe_attempt?path=../../etc")
    assert r.status_code == 400




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


# `test_every_command_the_page_teaches_is_a_REAL_cli_verb` retired
# 2026-10-03 with the page's own command lines (W55 B4): the page composes
# none, and the lines it shows -- the terminal's composer's,
# `/api/task-setup/commands` -- are typed back down the road
# (`test_printed_commands_run.py`).


# --------------------------------------------------------------------- #
#  The folder door -- Task setup's one per-directory answer              #
# --------------------------------------------------------------------- #


class TestTheFolderDoor:
    """`web/task-setup.md` § 2.1: *"the page holds no state of its own…
    the folder is the only link."*

    The page did not keep that.  It assembled from twelve endpoints and
    cleared the per-folder ones with `_resetPerFolderState()`, a
    hand-written list of eight clears in a file with twenty-five module
    variables -- so a six-trial bench plan rode into a calculation that
    declared none (2026-09-19).  This door answers the folder once.
    """

    @pytest.fixture
    def described(self, tmp_path, monkeypatch):
        """A described SIESTA calculation, through the product's own doors."""
        from conftest import write_pseudos
        from molbuilder import describe as D
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.siesta.stages import default_siesta_stages
        from molbuilder.structure import Structure
        import numpy as np

        root = tmp_path / "projects"
        root.mkdir()
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        st = Structure(elements=["H", "H"],
                       positions=np.array([[0, 0, 0], [0, 0, 0.74]], float))
        src = root / "h2.xyz"
        src.write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
        dest = root / "proj" / "optimization" / "run"
        D.write_description(D.build_description(
            st, SiestaConfig(system_label="JOB", mesh_cutoff=250.0),
            default_siesta_stages("publishable"),
            engine="siesta", shape="hierarchical", name="JOB",
            source=str(src)), dest)
        write_pseudos(dest, ["H"])
        from molbuilder.web.app import create_app
        return dest, create_app(config={}).test_client()

    # `test_the_door_answers_what_the_four_calls_answer` retired 2026-10-03
    # with the three routes it compared the folder answer to
    # (`/template-values`, `/resolved`, `/attempts`): no page called them,
    # and the folder answer is the one door left.

    def test_the_answer_names_the_folder_it_is_about(self, described):
        """The half a reset list cannot cover.

        Two things make a page show the wrong folder: state that LINGERS,
        and an answer that LANDS LATE.  Clearing fixes only the first, and
        Task setup has no `AbortController` across any of its twelve calls
        -- so a response for the folder you just left can still arrive.
        `dir` is what lets a consumer that has moved on discard it, which
        is `calcdir.json`'s rule (§ 1.4a) applied to the wire.
        """
        dest, client = described
        one = client.get("/api/task-setup/folder?dir=" + str(dest)).get_json()
        assert one["dir"] == str(dest.resolve()), (
            "the answer must name its own subject; got " + repr(one["dir"]))

    def test_it_reports_the_mode_rather_than_the_page_keeping_one(
            self, described):
        """A save writes `task.json` and deletes the hand-over, so which is
        on disk IS the mode -- the page need not remember one."""
        dest, client = described
        from molbuilder.task import FILENAME as TASK_FILENAME
        assert client.get("/api/task-setup/folder?dir=" + str(dest)
                          ).get_json()["mode"] == "description"

        (dest / TASK_FILENAME).unlink()
        (dest / "task.1st.json").write_text('{"schema": "x", "engine": {}}')
        body = client.get(
            "/api/task-setup/folder?dir=" + str(dest)).get_json()
        assert body["mode"] == "handover"
        assert body["description"] is None and body["handover"] is not None

        (dest / "task.1st.json").unlink()
        assert client.get("/api/task-setup/folder?dir=" + str(dest)
                          ).get_json()["mode"] == "empty"

    def test_one_bad_part_does_not_cost_the_others(self, described):
        """A malformed template must not take the description with it."""
        dest, client = described
        tmpl = next(dest.glob("*.template.toml"))
        tmpl.write_text("this is not toml = = =\n")
        body = client.get(
            "/api/task-setup/folder?dir=" + str(dest)).get_json()
        assert body["ok"] is True, "the answer survives one bad part"
        assert body["template"]["ok"] is False, "and says which part failed"
        assert body["description"] is not None, (
            "the description is still there to read")


def _labelled_au_lead_junction(root, n_layers):
    """A form-B citable pair (§ 4.1b) whose leads are real fcc(111) Au.

    The seam note needs a junction that actually has leads.
    """
    import numpy as np
    from ase.build import fcc111
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    slab = fcc111("Au", size=(1, 1, n_layers), a=4.158,
                  orthogonal=False, vacuum=0.0)
    lead = np.asarray(slab.positions, dtype=float)
    lead[:, 2] -= lead[:, 2].min()
    span = lead[:, 2].max()
    right = lead.copy()
    right[:, 2] += span + 8.0
    bridge = np.array([[lead[0, 0], lead[0, 1], span + 4.0]])

    pos = np.vstack([lead, bridge, right])
    n = len(lead)
    s = Structure(
        elements=["Au"] * n + ["S"] + ["Au"] * n,
        positions=pos,
        regions={"L-electrode": list(range(n)),
                 "bridge": [n],
                 "R-electrode": list(range(n + 1, 2 * n + 1))})
    s.frozen_atoms = list(range(n)) + list(range(n + 1, 2 * n + 1))
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, pos[:, 2].max() + 10.0]
    s.cell = cell

    d = root / "seamjunction"
    d.mkdir(parents=True, exist_ok=True)
    StructureCodec().write(s, d / "junction.xyz")
    return "seamjunction"


@pytest.mark.parametrize("n_layers,verdict", [(4, "ECLIPSED"), (6, "CONTINUES")])
def test_the_leads_own_measurements_reach_the_card(
        web_client, isolated_projects_root, n_layers, verdict):
    """WIRED, NOT MERELY COMPUTED.

    `extract_electrode_model` measures the periodic seam and the
    principal-layer condition into `ElectrodeModel.notes`; a note no
    reader reaches is not a limit stated.

    Reported, never enforced -- the rule the electrode orientation is
    drawn under -- so the 4-layer case comes back as a description, not
    a refusal.
    """
    cite = _labelled_au_lead_junction(isolated_projects_root, n_layers)
    r = web_client.get(f"/api/transport/describe_attempt?path={cite}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["form"] == "structure", body
    summary = body["summary"]
    assert verdict in summary, f"the seam verdict must reach the card: {summary}"
    # BOTH leads emit the same seam sentence, so without the prefix a
    # person reads the verdict twice with nothing saying which end it is
    # about.  Pin the prefix, not merely the label -- the principal-layer
    # note already contains "L-electrode", so a looser assertion passes
    # with the prefix deleted.
    for lead in ("L-electrode", "R-electrode"):
        assert f"{lead}: the periodic seam" in summary, (
            f"each lead's seam note must be attributed to it: {summary}")

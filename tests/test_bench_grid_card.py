"""The bench grid, live in the card that sets the axes.

User, 2026-08-30: *"can't this list be just updated in the same card where
the parameters are set? this update is trivial when target and parameter
list is known and can be updated in real time. this does not need to be a
message with a window."*

So `/api/task-setup/bench-grid` serves the report `_bench_inputs` already
computes — **the one enumerator**, handed the axes as they are being
edited.  The browser paints it; it never enumerates a grid of its own,
because a second enumerator is exactly the drifting decider that let a
cell look fine in one place and be refused in another.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.scheduler import Domain, Environment, Topology
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure

REPO = Path(__file__).resolve().parents[1]
VIEWER = REPO / "molbuilder" / "web" / "static" / "task-setup" / "viewer.js"

#: A small Sol-SHAPED menu, written by hand so the fits list is a fact about
#: this fixture rather than about whatever the developer's cluster last
#: probed.  `public` deliberately stocks a100 and NOT a100.40gb -- the very
#: asymmetry that made a real submission unrunnable.
_DOMAINS = [
    {"name": "short", "partition": "short", "qos": "public",
     "max_time": "04:00:00", "max_cores": 128,
     "gpu": {"a100": 4, "a100.40gb": 4},
     "node_types": [{"cores": 128, "nodes": 20},
                    {"cores": 48, "nodes": 10, "gpu": {"a100": 4}},
                    {"cores": 64, "nodes": 4, "gpu": {"a100.40gb": 4}}]},
    {"name": "public", "partition": "public", "qos": "public",
     "max_time": "7-00:00:00", "max_cores": 128,
     "gpu": {"a100": 4},
     "node_types": [{"cores": 128, "nodes": 100},
                    {"cores": 48, "nodes": 50, "gpu": {"a100": 4}}]},
]


@pytest.fixture()
def bundle(tmp_path):
    """A described SIESTA calculation on a machine WITH QUEUES.

    Built here, never copied from the developer's `projects/` tree.  It was
    copied, until 2026-08-30, and a browser walk that typed a point into the
    real folder then saved it made this file fail -- a test proving its claim
    against found state, which is the one thing a fixture must never do.
    """
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "bundle"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB", use_gpu=True,
                                         diag_algorithm="ELPA-1STAGE"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    env = Environment(scheduler="slurm",
                      topology=Topology(sockets=2, cores_per_socket=32,
                                        gpus_per_node=4,
                                        gpu_type="a100.40gb"),
                      domains=[Domain.from_row(r) for r in _DOMAINS])
    (dest / "environment.json").write_text(env.to_json() + "\n")
    return dest


@pytest.fixture(autouse=True)
def _picker_root(tmp_path, monkeypatch):
    """The doors read only inside a picker root."""
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


def _post(client, dest, bench, target="(this machine)"):
    return client.post("/api/task-setup/bench-grid",
                       json={"dest": str(dest), "target": target,
                             "bench": bench}).get_json()


class TestTheDoorServesTheOneEnumerator:

    def test_it_answers_the_cells_and_where_each_would_go(self, client, bundle):
        # A CPU-family grid states no device counts: declaring `gpu_count`
        # beside `use_gpu: [false]` is refused by name upstream, because
        # the counts would be silently ignored.
        d = _post(client, bundle, {"mpi_np": [48], "omp_threads": [1],
                                   "use_gpu": [False]})
        assert d["ok"] is True, d
        assert d["cells"], "a resolvable grid must answer its cells"
        one = d["cells"][0]
        assert set(one) >= {"label", "shape", "family", "ranks",
                            "cores_each", "gpus", "gpu_type", "fits", "why"}
        assert set(one["fits"]) == {"short", "public"}, (
            f"both queues hold a 48-rank CPU cell; got {one['fits']}")

    def test_the_axes_SENT_win_over_the_axes_on_disk(self, client, bundle):
        """The card's edits live in the browser's model until the person
        saves, so a list read from `task.json` would describe the previous
        state.  The point of the door is that it does not.

        Both halves are asserted: what is SAVED must not appear, and what is
        SENT must.  Checking only the second would pass on a door that
        merged the two."""
        saved = {"mpi_np": [16], "omp_threads": [1], "use_gpu": [False]}
        tj = bundle / "task.json"
        doc = json.loads(tj.read_text())
        doc["bench"] = saved
        tj.write_text(json.dumps(doc, indent=2))

        d = _post(client, bundle, {"mpi_np": [64], "omp_threads": [1],
                                   "use_gpu": [False]})
        labels = {c["label"] for c in d["cells"]}
        assert labels == {"K64C1"}, (
            f"the in-flight 64 must be the whole grid, and the saved 16 must "
            f"not appear; got {sorted(labels)}")

    def test_a_cell_no_queue_takes_is_returned_struck_not_dropped(
            self, client, bundle):
        """R4 -- the struck row carries the numbers to change.  Dropping it
        would leave the person guessing why their point vanished."""
        d = _post(client, bundle, {"mpi_np": [48, 128], "omp_threads": [1],
                                   "use_gpu": [True], "gpu_count": [4]})
        kept = {c["label"] for c in d["cells"] if not c["why"]}
        struck = [c for c in d["cells"] if c["why"]]
        assert "G4K12C1" in kept, f"48 ranks x 4 a100.40gb fits `short`: {d}"
        assert struck, "a 128-rank a100.40gb cell fits no queue here"
        # R4 -- the struck row names the number to change, and the card it
        # could not get: only `short` stocks a100.40gb, on 64-core nodes.
        assert "64" in struck[0]["why"][0], struck[0]["why"]
        assert "a100.40gb" in struck[0]["why"][0], struck[0]["why"]

    def test_nothing_surviving_is_a_result_not_an_error(self, client, bundle):
        """*Nothing here fits* is the answer, and the crossed-out rows are
        how the person sees why.  A 400 would throw them away."""
        d = _post(client, bundle, {"mpi_np": [999999], "omp_threads": [1],
                                   "use_gpu": [False]})
        assert d["ok"] is True and d["kept"] == 0
        assert d["cells"] and all(c["why"] for c in d["cells"])

    def test_a_bench_that_is_not_an_object_is_refused(self, client, bundle):
        r = client.post("/api/task-setup/bench-grid",
                        json={"dest": str(bundle), "bench": [1, 2]})
        assert r.status_code == 400
        assert "bench" in r.get_json()["error"]

    def test_the_folder_must_be_inside_a_picker_root(self, client):
        r = client.post("/api/task-setup/bench-grid",
                        json={"dest": "/etc", "bench": {}})
        assert r.status_code >= 400


    def test_an_unresolvable_declaration_answers_the_readers_own_words(
            self, client, bundle):
        """A grid that cannot be resolved at all is a 400 carrying the
        refusal the terminal gives -- not a browser-side paraphrase, which
        would be a second account of the same rule, free to drift."""
        d = _post(client, bundle, {"mpi_np": [48], "omp_threads": [1],
                                   "use_gpu": [False], "gpu_count": [2]})
        assert d["ok"] is False
        assert "gpu_count" in d["error"] and "use_gpu" in d["error"]


class TestTheBrowserDoesNotEnumerate:
    """Source pins.  The browser must ASK for the grid, never derive it."""

    def _src(self) -> str:
        src = VIEWER.read_text()
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        return re.sub(r"^\s*//.*$", "", src, flags=re.M)

    def test_the_card_asks_the_one_door(self):
        assert "/api/task-setup/bench-grid" in self._src()

    def test_it_sends_the_axes_THE_ROWS_WERE_PAINTED_FROM(self):
        """One source for both, or they disagree.

        `renderMachine` takes the task as an ARGUMENT and is called with
        the handover object in handover mode; reading the module's `_task`
        instead described a different object than the rows above it.  And
        the axes must be the in-memory model's, not the file's, so the
        list tracks typing rather than the last save.
        """
        src = self._src()
        rm = src.split("function renderMachine", 1)[1]
        assert "scheduleFitRefresh(bench)" in rm, (
            "the list must be refreshed from the same `bench` the rows "
            "were painted from, not from a module global")
        body = src.split("async function refreshFit", 1)[1]
        assert "_fitBench" in body

    def test_it_stays_quiet_where_there_is_no_description(self):
        """Handover and empty modes have no `task.json`, and the door
        reads one -- so a request there could only 400."""
        src = self._src()
        body = src.split("async function refreshFit", 1)[1].split(
            "function paintFit", 1)[0]
        assert '_mode !== "description"' in body

    def test_a_stale_answer_is_dropped(self):
        """Typing outruns the network; an earlier reply landing after a
        later one would paint a grid for axes that no longer exist."""
        src = self._src()
        assert "_fitSeq" in src and "seq !== _fitSeq" in src

    def test_the_request_is_debounced(self):
        """Every keystroke repaints the rows; the server answer is worth
        one request per PAUSE.

        Asserting the name ``_fitTimer`` appears was the first version of
        this pin, and it survived deleting the `setTimeout` -- the name
        still occurs in its own declaration.  A presence check cannot tell
        a working debounce from a dead one, which is the same fault that
        let `payload.info` stay green over a bug.  So this reads the
        scheduler's BODY.
        """
        src = self._src()
        # To the NEXT function, not to the first `}` -- the body holds a
        # `{}` literal, and splitting on a brace truncated before the line
        # under test, so the pin reported on its own arithmetic.
        body = src.split("function scheduleFitRefresh", 1)[1].split(
            "async function refreshFit", 1)[0]
        assert "setTimeout(refreshFit" in body, (
            "scheduleFitRefresh must DEFER the call, not make it")
        assert "clearTimeout" in body, (
            "and cancel the pending one, or every keystroke still fires")

    def test_a_failed_fetch_hides_the_list_and_keeps_the_card(self):
        """`loadSweepChoices`' recorded lesson: a card that cannot get a
        nicety shows what it has.  Only a card that cannot get its
        SUBSTANCE may refuse -- and this list is not the substance."""
        src = self._src()
        tail = src.split("async function refreshFit", 1)[1].split(
            "function paintFit", 1)[0]
        assert tail.count("host.hidden = true") >= 2, (
            "a failed or unparseable answer must hide the list, never "
            "throw out of renderMachine and strand the card")


# --------------------------------------------------------------------- #
#  The RUN's own numbers — the same door, a grid of one                  #
# --------------------------------------------------------------------- #

def test_the_browser_assembles_nothing_of_its_own():
    """**The UI is not a second framework** *(user, 2026-09-02: "if you
    handcraft two branches to collect the parameter and generate the thing,
    then you have to maintain two branches of the logic… The user can do the
    same thing using CLI").*

    So the prep door calls `prep_run_inputs` and composes nothing itself.
    It DID compose its own for an hour on 2026-09-02, and every divergence
    was a different run: no `run-config.toml` verdict, no bench pins, and at
    first no condition pins -- a solver chosen on the run card reached the
    sbatch's neighbour and not the deck.

    Source-level because that is where the rule lives: a behavioural test
    would need a benchmark, a verdict and a condition all at once, and would
    still pass the day someone re-derived ONE of the three here.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/build.py").read_text(encoding="utf-8")
    door = src[src.index("def api_task_setup_prep("):
               src.index("def api_task_setup_save(")]
    assert "prep_run_inputs(" in door, (
        "the browser's prep door no longer calls the one assembly")
    # CALLS, not mentions, and on a WORD boundary: the door's own comments
    # name these to say why it does not reach for them, and `run_inputs` is
    # a substring of the very function it is supposed to call.
    #
    # `run_condition` is allowed: a pure read of the posted document for
    # DISPLAY (the preview's fallback when the machine cannot hold the
    # condition), not a step in assembling what prep receives.
    import re as _re
    # `_apply_run_config` was the fourth name here until 2026-09-02.  It is
    # deleted -- a benchmark no longer reaches a run at all -- and a name in
    # this list that nothing defines is a check that cannot fail.
    for reassembled in ("_declared_execution_pins",
                        "declared_run_shape", "run_inputs"):
        assert not _re.search(rf"(?<![\w.]){reassembled}\(", door), (
            f"the door calls {reassembled} itself -- that is the second "
            f"branch of logic the one assembly exists to prevent")


def test_there_is_exactly_ONE_admission_door():
    """`generator.md` § 2: a run is a sweep of length one.

    So there is one door and not two that agree: a `/run-fit` endpoint stood
    beside `bench-grid` on 2026-09-01, building a one-point axis from each
    value and calling the same `_bench_inputs` -- its own contract row said
    so. Deleted the same day. This fails if a second one comes back, because
    two doors is how one comes to say a run fits where `launch` refuses it.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/build.py").read_text(encoding="utf-8")
    calls = [ln for ln in src.splitlines()
             if "_bench_inputs(" in ln and not ln.lstrip().startswith("#")]
    assert len(calls) == 2, (
        "the browser asks the enumerator from exactly two places -- the grid "
        "card's door and the prep door -- and neither enumerates its own: "
        + "\n".join(calls))
    assert "run_fit" not in src and "run-fit" not in src
    door = src[src.index("def api_task_setup_bench_grid"):
               src.index("def api_task_setup_prep_plan")]
    for invented in ("max_cores", "admit(", "def _fits"):
        assert invented not in door, f"the grid door computes {invented} itself"


# --------------------------------------------------------------------- #
#  What a prep will write, per stage — task-setup.md § 7.1               #
# --------------------------------------------------------------------- #

_PLAN_TASK = {
    "schema": "molbuilder/task@1", "engine": {"name": "siesta"},
    "shape": "hierarchical", "run": {"name": "r", "id": "r_H2"},
    "structure": {"source": "a.xyz", "formula": "H2", "atoms": 2},
    "varies": [],
    "stages": [{"name": "coarse", "enabled": True, "overrides": {}},
               {"name": "tight", "enabled": True, "overrides": {}}],
    "allocation": {"domain": "htc", "time": "1-00:00:00"},
    # THREE points on one axis and ONE on the other: an axis to measure
    # beside a decision already made (`generator.md` § 4.3a).
    "bench": {"mpi_np": [4, 8, 16], "omp_threads": [4]},
}


def _plan(client, task):
    return client.post("/api/task-setup/prep-plan",
                       json={"task": task}).get_json()


class TestTheBenchPreviewSaysNothingAboutTheRun:
    """A13 is the RUN's rule, and it is told in the run's card.

    The plan door returned `emitted` and `chosen` for `kind == "bench"` as
    well, and the page renders them with no kind check -- so a **bench**
    preview carried the heading "What this run will actually be launched
    with" over the run condition's numbers, which no trial uses.  A surprise
    of exactly the kind A13 exists to prevent, told in the wrong card."""

    def test_a_bench_preview_carries_no_emitted_launch(self, client, bundle):
        r = client.post("/api/task-setup/prep", json={
            "dest": str(bundle), "kind": "bench", "stage": "coarse",
            "plan": True})
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("emitted") == [], (
            "the bench preview named the run's launch: "
            + repr(body.get("emitted")))
        assert not body.get("chosen"), (
            "the bench preview named the run's condition: "
            + repr(body.get("chosen")))

    def test_a_run_preview_still_carries_one(self, client, bundle):
        """The other half -- the gate must not have emptied both cards."""
        r = client.post("/api/task-setup/prep", json={
            "dest": str(bundle), "kind": "run", "stage": "coarse",
            "plan": True})
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json().get("emitted"), "the run lost its A13 block"


class TestThePlanComesFromTheProducer:
    """§ 7.1: a confirmation, not a second answer.  Flat and hierarchical
    name directories differently, and a list the page composed would be free
    to disagree with the thing it describes."""

    def test_each_stage_names_its_directory_from_the_one_namer(self, client):
        d = _plan(client, _PLAN_TASK)
        assert d["ok"] is True, d
        assert [r["dir"] for r in d["stages"]] == ["01_coarse", "02_tight"]

    def test_flat_puts_every_stage_in_the_bundle_root(self, client):
        """The layout question, and the reason the page may not answer it:
        `Shape.stage_dir` gives flat a real path (`.`) so no caller needs an
        `if` for "no directory"."""
        t = dict(_PLAN_TASK, shape="flat",
                 stages=[_PLAN_TASK["stages"][0]])
        d = _plan(client, t)
        assert [r["dir"] for r in d["stages"]] == ["."]

    def test_every_stage_shows_the_ONE_allocation(self, client):
        """§ 6.8a: the calculation asks the scheduler for one queue, one
        wall, one memory.  A per-rung block stood here on 2026-09-01 and was
        deleted with the key -- `prep run <stage>` is already per stage, so
        a rung that wants a different wall says so on its own command."""
        d = _plan(client, _PLAN_TASK)
        for row in d["stages"]:
            assert row["allocation"] == {"domain": "htc",
                                         "time": "1-00:00:00", "mem": ""}

    def test_the_chosen_shape_is_EXECUTION_never_the_bench(self, client):
        """`stages.md` § 6.8d: the run's condition is `execution`, its own
        block.  A `bench` row is a thing to measure at any length -- including
        length one, which is one trial.

        This asserted the opposite for one day (2026-09-01), when a one-point
        bench row WAS the run's shape.  The two blocks are independent now,
        and this is the test that says so from the surface's side."""
        d = _plan(client, _PLAN_TASK)
        for row in d["stages"]:
            assert row["chosen"] == {}, (
                "a bench row reached the run's shape: " + repr(row["chosen"]))
        withcond = dict(_PLAN_TASK, execution={"mpi_np": 8})
        for row in _plan(client, withcond)["stages"]:
            assert row["chosen"] == {"mpi_np": 8}

    def test_the_bench_row_carries_every_axis(self, client):
        d = _plan(client, _PLAN_TASK)
        assert d["bench"]["axes"] == {"mpi_np": [4, 8, 16], "omp_threads": [4]}
        assert d["bench"]["allocation"]["domain"] == "htc"

    def test_a_disabled_rung_is_not_listed(self, client):
        t = dict(_PLAN_TASK,
                 stages=[dict(_PLAN_TASK["stages"][0], enabled=False),
                         _PLAN_TASK["stages"][1]])
        assert [r["stage"] for r in _plan(client, t)["stages"]] == ["tight"]

    def test_a_description_mid_edit_is_refused_in_its_own_words(self, client):
        """Unreadable is ordinary while someone types, and the card hides the
        list rather than showing a plan for a document that no longer says
        what it did."""
        d = _plan(client, {"schema": "molbuilder/task@1"})
        assert d["ok"] is False and d["error"]


def test_the_plan_door_composes_no_name_of_its_own():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/build.py").read_text(encoding="utf-8")
    door = src[src.index("def api_task_setup_prep_plan"):
               src.index("def api_task_setup_machines")]
    assert "token_for" in door and "stage_dir" in door
    for invented in ('f"{i:02d}_', "zfill", "01_", "bench-"):
        if invented == "bench-":
            continue          # the LITERAL label of the bench row, not a name
        assert invented not in door, f"the door builds {invented} itself"

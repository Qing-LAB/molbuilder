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

def _fit(client, dest, values, target="(this machine)"):
    return client.post("/api/task-setup/run-fit",
                       json={"dest": str(dest), "target": target,
                             "values": values}).get_json()


class TestTheRunFitIsTheSameEnumerator:
    """`task-setup.md` § 6.2b: the run states one value per parameter, which
    is a sweep of length one — so the check is the grid door with a grid of
    one, and there is no second admission path to keep in step."""

    def test_a_run_that_fits_says_which_queues_would_take_it(self, client,
                                                             bundle):
        d = _fit(client, bundle, {"mpi_np": 48, "omp_threads": 1,
                                  "use_gpu": False})
        assert d["ok"] is True, d
        assert d["stated"] is True and d["fits"] is True
        assert set(d["cell"]["fits"]) == {"short", "public"}, d["cell"]
        assert d["cell"]["ranks"] == 48

    def test_a_run_no_queue_can_hold_is_a_RESULT_not_a_failure(self, client,
                                                               bundle):
        """A struck answer beside the field that caused it is the point.  A
        400 would leave the card with nothing to show and send the person to
        the CLI to find out why — the failure this whole lane was rebuilt to
        remove."""
        d = _fit(client, bundle, {"mpi_np": 100000, "omp_threads": 1,
                                  "use_gpu": False})
        assert d["ok"] is True, d
        assert d.get("fits") is not True
        assert d["cell"] is None or d["cell"]["why"], d

    def test_stating_nothing_is_ordinary_and_checks_nothing(self, client,
                                                            bundle):
        """A description that leaves the run to `run-config.toml` and the
        wrapper's policy is the state every description was in before
        2026-09-01."""
        d = _fit(client, bundle, {})
        assert d["ok"] is True and d["stated"] is False and d["cell"] is None

    def test_it_reads_the_values_SENT_not_the_ones_on_disk(self, client,
                                                           bundle):
        """Same reason the grid door does: the card's edits live in the
        browser's model until the person saves."""
        a = _fit(client, bundle, {"mpi_np": 48, "omp_threads": 1,
                                  "use_gpu": False})
        b = _fit(client, bundle, {"mpi_np": 24, "omp_threads": 2,
                                  "use_gpu": False})
        assert a["cell"]["ranks"] == 48 and b["cell"]["ranks"] == 24


def test_there_is_exactly_one_admission_path():
    """Both doors call `_bench_inputs` and neither computes a verdict of its
    own.  A second one could say a run is fine where `launch` refuses it."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/build.py").read_text(encoding="utf-8")
    door = src[src.index("def api_task_setup_run_fit"):
               src.index("def api_task_setup_machines")]
    assert "_bench_inputs" in door
    for invented in ("max_cores", "admit(", "def _fits"):
        assert invented not in door, f"the run door computes {invented} itself"


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
    "allocation": {"domain": "htc", "time": "1-00:00:00", "mpi_np": 8},
    "stage_allocation": {"tight": {"time": "2-00:00:00", "mpi_np": 16}},
    "bench": {"mpi_np": [4, 8, 16]},
    "bench_allocation": {"time": "0-00:30:00"},
}


def _plan(client, task):
    return client.post("/api/task-setup/prep-plan",
                       json={"task": task}).get_json()


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

    def test_a_rung_shows_what_IT_will_ask_for(self, client):
        """§ 6.8b at the surface: `tight` overrides the wall and the ranks
        and inherits the queue nobody restated."""
        d = _plan(client, _PLAN_TASK)
        coarse, tight = d["stages"]
        assert coarse["allocation"]["time"] == "1-00:00:00"
        assert coarse["allocation"]["values"] == {"mpi_np": 8}
        assert tight["allocation"]["time"] == "2-00:00:00"
        assert tight["allocation"]["values"] == {"mpi_np": 16}
        assert tight["allocation"]["domain"] == "htc", "the queue is inherited"

    def test_the_bench_row_shows_its_OWN_wall(self, client):
        """§ 6.8c: measuring is short and running is not.  The row exists so
        that is visible beside the runs rather than inferred."""
        d = _plan(client, _PLAN_TASK)
        assert d["bench"]["allocation"]["time"] == "0-00:30:00"
        assert d["bench"]["allocation"]["domain"] == "htc"
        assert d["bench"]["axes"] == {"mpi_np": [4, 8, 16]}

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

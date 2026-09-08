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
from pathlib import Path

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.scheduler import Domain, Environment, Topology
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure

REPO = Path(__file__).resolve().parents[1]

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




# --------------------------------------------------------------------- #
#  The RUN's own numbers — the same door, a grid of one                  #
# --------------------------------------------------------------------- #





class TestPickingThisMachineIsAnAnswer:
    """`"(this machine)"` and *nobody said* are two states, and the door
    collapsed them into one `None`.

    The `None` is right and load-bearing: it lets `record_scopes` prefer the
    bundle's own `environment.json`, the snapshot a described calculation
    carries.  Forcing this door to `LOCAL_TARGET` on 2026-09-02 threw that
    away and a GPU grid stopped resolving.  But `machine_for` raises
    `AmbiguousTarget` for `None` when named records exist and the folder has
    no snapshot -- so on a not-yet-prepped calculation the card showed an
    ambiguity refusal naming a `--target` flag nobody can type in a browser,
    beside a Prep button that worked, because prep had been told the answer
    and this door had discarded it.
    """

    @pytest.fixture()
    def machine_with_named_records(self, tmp_path, monkeypatch):
        """A workstation that also holds a cluster's record — the setup that
        makes the question real (`choice_required`: any named record does)."""
        cfg = tmp_path / "cfg"
        (cfg / "environments").mkdir(parents=True)
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cfg))
        # A DIFFERENT GPU FROM THE BUNDLE'S, deliberately.  The bundle
        # snapshot says `a100.40gb`; these say plain `a100`.  Writing the
        # same topology in both places makes every "which record answered?"
        # assertion pass whichever one did -- which is how a first draft of
        # these tests let the 2026-09-02 regression through untouched.
        env = Environment(scheduler="slurm",
                          topology=Topology(sockets=2, cores_per_socket=32,
                                            gpus_per_node=4,
                                            gpu_type="a100"),
                          domains=[Domain.from_row(r) for r in _DOMAINS])
        (cfg / "environment.json").write_text(env.to_json() + "\n")
        (cfg / "environments" / "sol.json").write_text(env.to_json() + "\n")
        return cfg

    @staticmethod
    def _unprepped(bundle):
        """A described calculation that has not been prepped: no snapshot."""
        (bundle / "environment.json").unlink()
        return bundle

    def test_the_grid_resolves_when_this_machine_was_picked(
            self, client, bundle, machine_with_named_records):
        d = _post(client, self._unprepped(bundle),
                  {"mpi_np": [48], "omp_threads": [1], "use_gpu": [False]},
                  target="(this machine)")
        assert d["ok"] is True, d
        assert d["cells"], "a picked machine must yield cells"

    def test_naming_nothing_is_still_refused(
            self, client, bundle, machine_with_named_records):
        """The other half.  The refusal is right when nobody answered -- what
        was wrong was applying it to somebody who did."""
        d = _post(client, self._unprepped(bundle),
                  {"mpi_np": [48], "omp_threads": [1], "use_gpu": [False]},
                  target=None)
        assert d["ok"] is False and d["error"]

    def test_a_prepped_folders_own_snapshot_still_wins(
            self, client, bundle, machine_with_named_records):
        """What the `None` is FOR, and the reason this is not fixed by
        mapping the label to `LOCAL_TARGET`: the bundle here carries a
        record with a100.40gb, and that is the machine the grid is measured
        against even though a named record and a local one both exist."""
        d = _post(client, bundle,
                  {"mpi_np": [4], "omp_threads": [1], "use_gpu": [True],
                   "gpu_count": [1]},
                  target="(this machine)")
        assert d["ok"] is True, d
        assert d["cells"][0]["gpu_type"] == "a100.40gb", d["cells"][0]


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


def test_the_prep_door_reads_the_FILE_and_not_a_posted_document(client, bundle):
    """`build.py` does `task = read_task(desc)`: the prep door takes a FOLDER.

    One assembly serves the CLI and the browser (A12), and the CLI has only
    the file -- so the document is not a channel here, and a browser holding
    unsaved edits is holding something prep cannot see.  That asymmetry is
    real and load-bearing; what it must never do is go unsaid, which is why
    the page now refuses to prep while its buffer differs from disk
    (2026-09-02: the run card's values reached the fit line, which IS posted
    them, and not the A13 block, which is not).

    Asserted by CONTRADICTION: post a document that disagrees with the file
    and check the answer follows the file."""
    import json
    d = json.loads((bundle / "task.json").read_text())
    d["stages"][0]["execution"] = {"mpi_np": 7}
    (bundle / "task.json").write_text(json.dumps(d, indent=2))

    r = client.post("/api/task-setup/prep", json={
        "dest": str(bundle), "kind": "run", "stage": "coarse", "plan": True,
        # a document saying something ELSE -- it must be ignored
        "task": {"stages": [{"name": "coarse", "execution": {"mpi_np": 999}}]},
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    chosen = r.get_json().get("chosen") or {}
    assert chosen.get("mpi_np") == 7, (
        "the prep door honoured a POSTED document -- it must read the file, "
        f"or the CLI and the browser are two assemblies: {chosen}")


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

    def test_the_bench_rungs_name_the_container_the_sweep_LANDS_in(self, client):
        """The card showed `bench-<token>/`, composed in the browser.

        It named nothing: the container is `<NN>_<stage>/bench` in the
        hierarchy, and the dash form it showed is a TRIAL's name
        (`bench-<point>`), which lives INSIDE one.  ONE ENTRY PER RUNG,
        because `prep bench` takes a stage and the container lives inside
        the stage it measures.
        """
        d = _plan(client, _PLAN_TASK)
        assert d["bench"]["rungs"] == [
            {"stage": "coarse", "dir": "01_coarse/bench"},
            {"stage": "tight", "dir": "02_tight/bench"}]

    def test_flat_qualifies_the_container_instead_of_nesting_it(self, client):
        """The other layout, and the reason the browser may not guess: flat
        has no stage directory to sit inside, so the token qualifies the
        container's own name (`bench_<NN>_<stage>`).  Two flat stages sharing
        one root `bench/` is the bug that rule was written for."""
        t = dict(_PLAN_TASK, shape="flat")
        assert [r["dir"] for r in _plan(client, t)["bench"]["rungs"]] == [
            "bench_01_coarse", "bench_02_tight"]

    def test_a_disabled_rung_takes_its_container_with_it(self, client):
        t = dict(_PLAN_TASK,
                 stages=[dict(_PLAN_TASK["stages"][0], enabled=False),
                         _PLAN_TASK["stages"][1]])
        assert _plan(client, t)["bench"]["rungs"] == [
            {"stage": "tight", "dir": "02_tight/bench"}]

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



"""The bench grid, live in the card that sets the axes.

User, 2026-08-30: *"can't this list be just updated in the same card where
the parameters are set? this update is trivial when target and parameter
list is known and can be updated in real time. this does not need to be a
message with a window."*

So `/api/task-setup/bench-grid` serves the report `bench_inputs` already
computes — **the one enumerator**, handed the axes as they are being
edited.  The browser paints it; it never enumerates a grid of its own,
because a second enumerator is exactly the drifting decider that let a
cell look fine in one place and be refused in another.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.scheduler import Domain, Environment, Topology
from molbuilder.jobset.ladder import engine_ladder
from molbuilder.structure import Structure
from support.machines import SLURM
from support.road import read_request, write_request


#: A small Sol-SHAPED menu, written by hand so each fit is a fact about this
#: fixture rather than about whatever the developer's cluster last probed.
#: `short`'s GPU nodes are 48 and 64 cores wide and `public`'s 48, so the
#: widest a GPU job can be anywhere here is 64 cores; `cpu` lists no GPUs.
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
    {"name": "cpu", "partition": "cpu", "qos": "public",
     "max_time": "7-00:00:00", "max_cores": 128,
     "node_types": [{"cores": 128, "nodes": 40}]},
]


def _queues(cell):
    """The queues a cell was fit on, and whether a node of each holds it --
    read off the answer's fields, never its sentences."""
    return {f["queue"]: f["holds"] for f in cell["fit"]}


def _with_queue(bundle, name):
    """The job request names the benchmark's one queue
    (`allocation.domain`, `generator.md` § 4.3a)."""
    req = read_request(bundle)
    req["allocation"] = {**req.get("allocation", {}), "domain": name}
    write_request(bundle, req)


@pytest.fixture()
def bundle(tmp_path):
    """A described SIESTA calculation on a machine WITH QUEUES.

    Built here, never copied from the developer's `projects/` tree: a test
    proving its claim against found state is the one thing a fixture must
    never do.
    """
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "bundle"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB"),
                            engine_ladder("siesta", "publishable"),
                            engine="siesta",
                            calculation="optimization",
                            shape="hierarchical", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    env = Environment(machine="this", scheduler=SLURM,
                      topology=Topology(sockets=2, cores_per_socket=32,
                                        gpus_per_node=4),
                      domains=[Domain.from_row(r) for r in _DOMAINS],
                      # how a shell enters an environment there -- the
                      # record's fact (`configuration.md` § 5 M-1)
                      env_init={"activation": "conda activate",
                                         "preamble": "true"})
    (dest / "environment.json").write_text(env.to_json() + "\n")
    # THE RUN ON THE GPU, and how many (`execution/gpu.md` G5), on the
    # calculation's run card; the bench takes its GPU from it, never its
    # count.
    req = read_request(dest)
    req["execution"] = {"use_gpu": True, "diag_algorithm": "ELPA-1STAGE",
                        "gpu_count": 1}
    write_request(dest, req)
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


def _post(client, dest, bench, target="(this machine)", domain=None):
    return client.post("/api/task-setup/bench-grid",
                       json={"dest": str(dest), "target": target,
                             "bench": bench,
                             **({"domain": domain} if domain is not None
                                else {})}
                       ).get_json()


class TestTheDoorServesTheOneEnumerator:

    def test_a_cell_is_fit_on_every_queue_while_none_is_named(self, client,
                                                              bundle):
        """`task-setup.md` § 6.2a: before the description names the
        benchmark's queue, the card fits each cell on every queue of the
        target's record, so the person sees each one's while choosing."""
        # A CPU-family grid states no device counts: declaring `gpu_count`
        # beside `use_gpu: [false]` is refused by name upstream, because
        # the counts would be silently ignored.
        d = _post(client, bundle, {"mpi_np": [48], "omp_threads": [1],
                                   "use_gpu": [False]})
        assert d["ok"] is True, d
        assert d["cells"], "a resolvable grid must answer its cells"
        one = d["cells"][0]
        assert set(one) >= {"label", "shape", "family", "ranks",
                            "cores_each", "gpus", "fit", "why"}
        assert _queues(one) == {"short": True, "public": True, "cpu": True}, (
            f"every queue holds a 48-rank CPU cell; got {one['fit']}")

    def test_the_axes_SENT_win_over_the_axes_on_disk(self, client, bundle):
        """The card's edits live in the browser's model until the person
        saves, so a list read from `job-request.json` would describe the previous
        state.  The point of the door is that it does not.

        Both halves are asserted: what is SAVED must not appear, and what is
        SENT must.  Checking only the second would pass on a door that
        merged the two."""
        saved = {"mpi_np": [16], "omp_threads": [1], "use_gpu": [False]}
        req = read_request(bundle)
        req["bench"] = saved
        write_request(bundle, req)

        d = _post(client, bundle, {"mpi_np": [64], "omp_threads": [1],
                                   "use_gpu": [False]})
        labels = {c["label"] for c in d["cells"]}
        assert labels == {"G0N64C1"}, (
            f"the in-flight 64 must be the whole grid, and the saved 16 must "
            f"not appear; got {sorted(labels)}")

    def test_a_declared_cell_no_node_holds_is_kept_with_its_fit(
            self, client, bundle):
        """`scheduler.md` R6, `generator.md` § 4.3a: a declared cell is
        kept whatever the record says, and its fit names what each kind of
        machine lacks (R4, R10) -- the scheduler decides."""
        _with_queue(bundle, "short")
        d = _post(client, bundle, {"mpi_np": [48, 128], "omp_threads": [1],
                                   "use_gpu": [True], "gpu_count": [4]})
        cells = {c["label"]: c for c in d["cells"]}
        assert _queues(cells["G4N48C1"]) == {"short": True}, cells
        wide = cells["G4N128C1"]
        assert not wide["why"], f"a declared cell was left out: {wide}"
        assert _queues(wide) == {"short": False}, wide
        assert "the 128-core machines have no GPU" in wide["fit"][0]["line"]

    def test_the_card_fits_the_queue_the_description_names(self, client,
                                                           bundle):
        """A benchmark measures on one queue, the one its description names
        (`allocation.domain`, `generator.md` § 4.3a), and the card fits each
        cell on that queue alone, by the one fold prep asks
        (`prep_inputs.fold_allocation`) -- so a cell the card keeps is one
        prep keeps.  `public`'s GPU nodes are 48 cores wide: a GPU cell the
        machine proposes wider than that is left out by `public`'s own
        numbers, and the narrower ones are kept."""
        _with_queue(bundle, "public")
        d = _post(client, bundle, {"use_gpu": [True], "gpu_count": [1]})
        assert d["ok"] is True, d
        assert all(_queues(c) == {"public": not c["why"]}
                   for c in d["cells"]), d["cells"]
        left = [c for c in d["cells"] if c["why"]]
        assert left and all(c["ranks"] * c["cores_each"] > 48
                            for c in left), left
        assert any(not c["why"] for c in d["cells"]), d["cells"]

    def test_the_card_fits_the_queue_picked_on_the_page(self, client,
                                                        bundle):
        """`task-setup.md` § 6.2a: the cells are fit on the queue picked on
        the page -- sent with the axes, before it is saved -- as a prep's
        `--domain` would place them (`prep_inputs.fold_allocation`), over
        the queue the saved description names."""
        _with_queue(bundle, "public")
        d = _post(client, bundle, {"mpi_np": [48], "omp_threads": [1],
                                   "use_gpu": [False]}, domain="short")
        assert d["ok"] is True, d
        assert all(_queues(c) == {"short": True} for c in d["cells"]), d

    def test_with_none_picked_on_the_page_every_queue_is_asked(
            self, client, bundle):
        """`task-setup.md` § 6.2a: the page shows no queue picked -- its
        model's allocation names none -- so each cell is fit on every queue
        of the record, whatever the saved description still names."""
        _with_queue(bundle, "public")
        d = _post(client, bundle, {"mpi_np": [48], "omp_threads": [1],
                                   "use_gpu": [False]}, domain="")
        assert d["ok"] is True, d
        assert all(set(_queues(c)) == {"short", "public", "cpu"}
                   for c in d["cells"]), d

    def test_nothing_left_is_a_result_not_an_error(self, client, bundle):
        """*Nothing here fits* is the answer, and the rows left out are how
        the person sees why.  A 400 would throw them away.  The machine's
        GPU proposal on `cpu`, whose record lists no GPUs, holds nothing."""
        _with_queue(bundle, "cpu")
        d = _post(client, bundle, {"use_gpu": [True]})
        assert d["ok"] is True and d["kept"] == 0, d
        assert d["cells"] and all(c["why"] for c in d["cells"])
        assert all(_queues(c) == {"cpu": False} for c in d["cells"])

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
    no snapshot -- so on a not-yet-prepared calculation the card showed an
    ambiguity refusal naming a `--target` flag nobody can type in a browser,
    beside a Prep button that worked, because prep had been told the answer
    and this door had discarded it.
    """

    @pytest.fixture()
    def machine_with_named_records(self, tmp_path, monkeypatch):
        """A workstation that also holds a cluster's record — the setup that
        makes the question real (`choice_required`: any named record does)."""
        from molbuilder.config_dir import config_dir, ensure_private_dir
        from molbuilder.scheduler import (environments_dir, machine_scope_path,
                                          named_environment_path)
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
        ensure_private_dir(environments_dir())
        # QUEUES NAMED APART FROM THE BUNDLE'S, deliberately.  The bundle
        # snapshot's are `short` and `public`; these are `sol-short` and
        # `sol-public`.  Writing the same menu in both places makes every
        # "which record answered?" assertion pass whichever one did.
        env = Environment(scheduler=SLURM,
                          topology=Topology(sockets=2, cores_per_socket=32,
                                            gpus_per_node=4),
                          domains=[Domain.from_row(
                              dict(r, name="sol-" + r["name"]))
                              for r in _DOMAINS])
        machine_scope_path().write_text(env.to_json() + "\n")
        named_environment_path("sol").write_text(env.to_json() + "\n")
        return config_dir()

    @staticmethod
    def _unprepped(bundle):
        """A described calculation that has not been prepared: no snapshot."""
        (bundle / "environment.json").unlink()
        return bundle

    def test_the_grid_resolves_when_this_machine_was_picked(
            self, client, bundle, machine_with_named_records):
        """...and the card, a preview, writes nothing: it reads the record a
        prep would snapshot (`configuration.md` M-3; W52 -- every edit pause
        snapshotted the picked machine into the calculation, so a later Prep
        for another was refused as a contradiction of a prep never made)."""
        bundle = self._unprepped(bundle)
        for target in ("(this machine)", "sol"):
            d = _post(client, bundle,
                      {"mpi_np": [48], "omp_threads": [1],
                       "use_gpu": [False]}, target=target)
            assert d["ok"] is True, (target, d)
            assert d["cells"], "a picked machine must yield cells"
            # ...fit on the PICKED machine's queues: with several machines
            # on file and nothing snapshotted, the folder names no menu, and
            # every cell was fit on none (the W52 fix-6 review: the card's
            # fits came back empty)
            assert set(_queues(d["cells"][0])) == {
                "sol-short", "sol-public", "sol-cpu"}, (target, d["cells"][0])
            assert not (bundle / "environment.json").exists(), (
                "the card snapshotted the machine into the calculation")

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
        record whose queues are `short` and `public`, and those are the
        queues the grid is measured against even though a named record and
        a local one both exist."""
        d = _post(client, bundle,
                  {"mpi_np": [4], "omp_threads": [1], "use_gpu": [True],
                   "gpu_count": [1]},
                  target="(this machine)")
        assert d["ok"] is True, d
        assert set(_queues(d["cells"][0])) == {"short", "public", "cpu"}, (
            d["cells"][0])


# --------------------------------------------------------------------- #
#  What a prep will write, per stage — task-setup.md § 7.1               #
# --------------------------------------------------------------------- #

_PLAN_TASK = {
    "schema": "molbuilder/task@2", "engine": {"name": "siesta"},
                                              "calculation": "optimization",
    "shape": "hierarchical", "run": {"name": "r", "id": "r_H2"},
    "structure": {"source": "a.xyz", "formula": "H2", "atoms": 2},
    "varies": [],
    "stages": [{"name": "coarse", "overrides": {}},
               {"name": "tight", "overrides": {}}],
}
_PLAN_REQUEST = {
    "schema": "molbuilder/job-request@1",
    "stages": {"coarse": {"group": 1}, "tight": {"group": 2}},
    "allocation": {"domain": "htc", "time": "1-00:00:00"},
    # THREE points on one axis and ONE on the other: an axis to measure
    # beside a decision already made (`generator.md` § 4.3a).
    "bench": {"mpi_np": [4, 8, 16], "omp_threads": [4]},
}


def _plan(client, task, request=None):
    return client.post("/api/task-setup/prep-plan",
                       json={"task": task,
                             "request": request or _PLAN_REQUEST}).get_json()


def _runnable(bundle):
    """The launch values a run states, or prep refuses it (`architecture.md`
    § 5.2): its ranks and cores per rank, its GPU count -- the run card
    runs it on the GPU -- and, on this machine with a scheduler, its queue,
    wall and memory."""
    req = read_request(bundle)
    req["execution"] = {**req.get("execution", {}), "mpi_np": 4,
                        "omp_threads": 1, "gpu_count": 1}
    req["allocation"] = {"domain": "short", "time": "01:00:00", "mem": "8G"}
    write_request(bundle, req)


def test_the_prep_door_reads_the_FILE_and_not_a_posted_document(client, bundle):
    """The prep door takes a FOLDER: the one entry reads its `task.json`
    and its `job-request.json`.

    One assembly serves the CLI and the browser (A12), and the CLI has only
    the file -- so the document is not a channel here, and a browser holding
    unsaved edits is holding something prep cannot see.  That asymmetry is
    real and load-bearing; what it must never do is go unsaid, which is why
    the page refuses to prep while its buffer differs from disk.

    Asserted by CONTRADICTION: post a document that disagrees with the file
    and check the answer follows the file."""
    _runnable(bundle)
    req = read_request(bundle)
    req["stages"]["coarse"]["execution"] = {"mpi_np": 7}
    write_request(bundle, req)

    r = client.post("/api/task-setup/prep", json={
        "dest": str(bundle), "kind": "task", "groups": [1], "plan": True,
        # a document saying something ELSE -- it must be ignored
        "request": {"stages": {"coarse": {"group": 1,
                                          "execution": {"mpi_np": 999}}}},
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    header = (r.get_json()["answers"][0].get("launch") or {}).get("header") or []
    assert "#SBATCH -n 7" in header, (
        "the prep door honoured a POSTED document -- it must read the file, "
        f"or the CLI and the browser are two assemblies: {header}")


class TestTheBenchPreviewSaysNothingAboutTheRun:
    """A13 is the RUN's rule, and it is told in the run's card.

    The plan door returned the run's launch and condition for a **bench**
    too, and the page renders them with no kind check -- so a bench preview
    carried the heading "What this run will actually be launched with" over
    the run condition's numbers, which no trial uses.  The preview is the
    entry's now, and a bench's answer carries neither."""

    def test_a_bench_preview_carries_no_launch(self, client, bundle):
        _runnable(bundle)
        r = client.post("/api/task-setup/prep", json={
            "dest": str(bundle), "kind": "bench", "stages": ["coarse"],
            "plan": True})
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()["answers"][0]
        assert body.get("launch") is None, (
            "the bench preview named the run's launch: "
            + repr(body.get("launch")))

    def test_a_run_preview_carries_its_header_line_for_line(self, client,
                                                             bundle):
        """The other half -- and A13's end point: the header the plan holds,
        as it will be written: every line the preview shows is the line the
        Prep of that plan writes, and no other."""
        _runnable(bundle)
        r = client.post("/api/task-setup/prep", json={
            "dest": str(bundle), "kind": "task", "groups": [1],
            "plan": True})
        assert r.status_code == 200, r.get_data(as_text=True)
        pv = r.get_json()
        header = (pv["answers"][0].get("launch") or {}).get("header") or []
        assert "#SBATCH -n 4" in header and "#SBATCH -p short" in header, (
            "the run lost its A13 block: " + repr(header))
        r = client.post("/api/task-setup/prep", json={
            "dest": str(bundle), "kind": "task", "groups": [1],
            "plan_id": pv["plan_id"]})
        assert r.status_code == 200, r.get_data(as_text=True)
        written = [ln.strip() for sb in bundle.rglob("01_coarse/*.sbatch")
                   for ln in sb.read_text().splitlines()
                   if ln.startswith("#SBATCH")]
        assert header == written, (header, written)


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
        d = _plan(client, t, dict(_PLAN_REQUEST,
                                  stages={"coarse": {"group": 1}}))
        assert [r["dir"] for r in d["stages"]] == ["."]

    def test_every_stage_shows_the_ONE_allocation(self, client):
        """§ 6.8a: the calculation asks the scheduler for one queue, one
        wall, one memory.  `prep task <stage>` is already per stage, so a
        rung that wants a different wall says so on its own command."""
        d = _plan(client, _PLAN_TASK)
        # THE EFFECTIVE ASK per stage (`ask`: the run card's over the
        # calculation's, the fold prep uses) -- here every stage's is the
        # calculation's one allocation.
        for row in d["stages"]:
            assert row["ask"] == {"domain": "htc",
                                  "time": "1-00:00:00", "mem": None}

    def test_the_chosen_shape_is_EXECUTION_never_the_bench(self, client):
        """`stages.md` § 6.8d: the run's condition is `execution`, its own
        block.  A `bench` row is a thing to measure at any length -- including
        length one, which is one trial.  The two blocks are independent, and
        this is the test that says so from the surface's side."""
        d = _plan(client, _PLAN_TASK)
        for row in d["stages"]:
            assert row["chosen"] == {}, (
                "a bench row reached the run's shape: " + repr(row["chosen"]))
        withcond = dict(_PLAN_REQUEST, execution={"mpi_np": 8})
        for row in _plan(client, _PLAN_TASK, withcond)["stages"]:
            assert row["chosen"] == {"mpi_np": 8}

    def test_the_bench_row_carries_every_axis(self, client):
        d = _plan(client, _PLAN_TASK)
        assert d["bench"]["axes"] == {"mpi_np": [4, 8, 16], "omp_threads": [4]}
        assert d["bench"]["allocation"]["domain"] == "htc"

    def test_the_bench_rungs_name_the_container_the_sweep_LANDS_in(self, client):
        """The container is `<NN>_<stage>/bench` in the hierarchy; the dash
        form (`bench-<point>`) is a TRIAL's name, which lives INSIDE one.  ONE ENTRY PER RUNG,
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

    def test_a_description_mid_edit_is_refused_in_its_own_words(self, client):
        """Unreadable is ordinary while someone types, and the card hides the
        list rather than showing a plan for a document that no longer says
        what it did."""
        d = _plan(client, {"schema": "molbuilder/task@2"})
        assert d["ok"] is False and d["error"]


# --------------------------------------------------------------------- #
#  How many attempts each stage has — asked, not worked out in the page  #
# --------------------------------------------------------------------- #

def _attempts(client, dest):
    """The folder answer's attempts -- the one door (`/api/task-setup/folder`)."""
    return client.get("/api/task-setup/folder?dir=" + str(dest)).get_json()[
        "attempts"]


def test_the_attempts_door_answers_from_the_DECLARED_shape(client, bundle):
    """Worked out in the browser this needs four of our rules — the stage
    token, the `run-<n>` pattern, the flat `_<token>-run` form, and a branch
    between the last two chosen **by looking at what was on disk**.

    `project-layout.md` § 4.5 forbids that by name: the shape is DECLARED.
    A hierarchical calculation whose stage directory does not exist yet is
    still hierarchical.
    """
    body = _attempts(client, bundle)
    assert body["ok"] is True, body
    assert body["shape"] == "hierarchical"
    first = json.loads((bundle / "task.json").read_text())["stages"][0]["name"]
    row = body["stages"][first]
    assert row["token"] == "01_" + first
    assert row["dir"] == "01_" + first
    assert row["attempts"] == 0, "nothing is prepared in this fixture"


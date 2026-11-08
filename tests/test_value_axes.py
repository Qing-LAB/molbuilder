"""2β value axes, end to end — `generator.md` § 4.3a, built 2026-08-21.

The acceptance case is the USER's live matrix (their
``projects/Au-BDT-Au/optimization/Relax/task.json``, declared 2026-08-21):
``mpi_np × use_gpu × diag_algorithm × block_size``, prepared from a
Sol-shaped record — a login node with NO local GPU, a menu whose
gpu-capable row carries the sinfo inventory and the probed 48-core cap
(the GPU node group's own core count; hand-editable).
The family split, each declared cell kept with its fit said, the per-trial
coordinates in the decks and in ``job-set.json``, and the per-side grouped
submission, every shelf to the benchmark's one queue, are pinned here; what a
declared `gpu_count` means, and what a bench's points may be, are rows
(`tests/data/gpu_contract.toml`, `tests/data/prep_protocol.toml`).
"""
from __future__ import annotations

import re

import pytest

from molbuilder.config_dir import ensure_private_dir
from molbuilder.jobset.materialize import latest_attempt
from molbuilder.jobset.model import JobSet
from molbuilder.scheduler import (Domain, Environment, Topology,
                                  environments_dir, named_environment_path,
                                  write_environment)
from support.machines import SLURM
from support.road import (declare_bench, described, jobset, road_tree,
                          shelf_headers)


#: The user's declared matrix, with the catalogue's canonical enum casing.
USERS_MATRIX = {
    "mpi_np": [32, 64, 128],
    "omp_threads": [1],
    "use_gpu": [True, False],
    "gpu_count": [1, 2, 4],
    "diag_algorithm": ["ELPA-1STAGE", "ELPA-2STAGE"],
    "block_size": [64, 128, 256],
}


@pytest.fixture
def sol_calc(tmp_path, monkeypatch):
    """A described calculation, and Sol's record where a record copied here
    lies (`preparing-for-another-machine.md` § 1): a 128-core login node with
    no local GPU, and a probed menu -- ``htc`` cpu-only; ``general`` and
    ``public`` carrying the a100 inventory and the probed ``max_cores`` cap."""
    ensure_private_dir(environments_dir())
    write_environment(
        Environment(
            scheduler=SLURM,
            topology=Topology(sockets=2, cores_per_socket=64,
                              gpus_per_node=None, gpu_type=None),
            domains=[Domain(name="htc", partition="htc", qos="public",
                            max_time="0-04:00:00"),
                     Domain(name="general", partition="general", qos="public",
                            max_time="7-00:00:00", max_cores=48,
                            gpu={"a100": 4}),
                     Domain(name="public", partition="public", qos="public",
                            max_time="7-00:00:00", max_cores=48,
                            gpu={"a100": 4})],
            # how a shell enters an environment there -- the record's fact
            env_init={"activation": "conda activate", "preamble": "true"}),
        named_environment_path("sol"))
    # The queue, wall and memory every job sent to this scheduler states --
    # the trials share the calculation's (`architecture.md` § 5.2); a
    # benchmark measures on the one queue it names (`generator.md` § 4.3a).
    # ...and the solver its run card states, on the CPU.
    return described(road_tree(tmp_path, monkeypatch), name="JOB",
                     allocation={"domain": "general", "time": "0-01:00:00",
                                 "mem": "8G"},
                     execution={"use_gpu": False,
                                "diag_algorithm": "ELPA-1STAGE"})


def _prep_on(calc, queue="general"):
    """`prep bench coarse --domain QUEUE --target sol`, as typed: the
    benchmark's one queue, its headers written (`job-system.md` § 6.0).
    What it said."""
    r = jobset("prep", "bench", "coarse", "--bundle", calc, "--target", "sol",
               "--domain", queue)
    assert r.exit_code == 0, r.output
    return r.output


def _bench(calc):
    return calc / "01_coarse" / "bench"


def _trials(calc) -> list:
    """The sweep's plan, each trial's point and resources, through its door
    (`JobSet.of`)."""
    return JobSet.of(_bench(calc)).to_dict()["jobs"]


def _deck_of(calc, trial) -> str:
    """The deck a trial's run holds (`project-layout.md` § 1.5a)."""
    run = latest_attempt(_bench(calc) / f"bench-{trial}")
    return next(run.glob("*.fdf")).read_text()


# --------------------------------------------------------------------- #
#  enumeration: families, inventory fallback                             #
# --------------------------------------------------------------------- #

def test_the_users_matrix_enumerates_both_families(sol_calc):
    """The acceptance matrix on `general`: 18 CPU trials and 54 GPU trials,
    every stated cell kept whatever the record says, its fit said under it
    (`generator.md` § 4.3a): np64 and np128 are over the GPU nodes' 48 cores,
    and the scheduler decides (`scheduler.md` R4, R6).  `gpu_count` shapes
    only the GPU family, and `use_gpu`'s two points beside it are no
    cpu-only refusal.  The record's TOPOLOGY has no GPU (a login node), and a
    stated count does not ask it: a trial asks what it states, a count naming
    no card (`scheduler.md` R2a), and a CPU trial asks no device at all."""
    declare_bench(sol_calc, USERS_MATRIX)
    out = _prep_on(sol_calc)
    jobs = _trials(sol_calc)
    points = [j["point"] for j in jobs]

    cpu = [p for p in points if not p["G"]]
    gpu = [p for p in points if p["G"]]
    assert len(cpu) == 18 and len(gpu) == 54
    assert sorted({p["N"] for p in cpu}) == [32, 64, 128]
    assert sorted({p["N"] for p in gpu}) == [32, 64, 128]
    assert {p["G"] for p in gpu} == {1, 2, 4}
    assert all(p["use_gpu"] is False for p in cpu)
    assert all(p["use_gpu"] is True for p in gpu)
    # every point carries the full value coordinate
    assert all({"diag_algorithm", "block_size"} <= set(p) for p in points)
    for job in jobs:
        res, pt = job["resources"], job["point"]
        assert res["mpi_np"] == pt["N"]
        assert res.get("gres") == (f"gpu:{pt['G']}" if pt["G"] else None), job
    assert "left out" not in out, out
    # EVERY np64 AND np128 GPU CELL IS OVER THE GPU NODES' 48 CORES, whatever
    # its G -- a trial's total cores are what a node must hold -- kept as
    # declared, each named with its fit under it.
    lines = [ln.strip() for ln in out.splitlines()]
    for cell, cores in (("G1N64C1", 64), ("G2N64C1", 64), ("G4N64C1", 64),
                        ("G1N128C1", 128)):
        at = next((i for i, ln in enumerate(lines)
                   if ln.startswith(f"{cell} ")), None)
        assert at is not None, f"the kept cell {cell} must be named"
        assert (f"{cores} cores is over general's largest machine (48 cores)"
                in lines[at + 1]), lines[at + 1]
    # an axis is never a pin; the measurement pins still ride
    deck = _deck_of(sol_calc, jobs[0]["name"])
    assert re.search(rf"^BlockSize\s+{points[0]['block_size']}\s*$", deck, re.M)
    assert re.search(r"^MaxSCFIterations\s+3\s*$", deck, re.M)


# --------------------------------------------------------------------- #
#  the coordinates reach the decks and the record                        #
# --------------------------------------------------------------------- #

def _small_matrix():
    return {"mpi_np": [4], "omp_threads": [1],
            "use_gpu": [True, False], "gpu_count": [1, 2, 4],
            "diag_algorithm": ["ELPA-1STAGE"],
            "block_size": [64, 128]}


def test_per_trial_coordinates_reach_the_decks(sol_calc):
    """Render-probed, not assumed: two trials differing only in
    ``block_size`` render different ``BlockSize`` lines; the CPU-family
    deck says ``Diag.ELPA.GPU .false.`` while its GPU sibling says
    ``.true.`` — and `job-set.json` records every trial's point as data."""
    declare_bench(sol_calc, _small_matrix())
    _prep_on(sol_calc)
    assert len(_trials(sol_calc)) == 2 + 3 * 2   # cpu: 1 cell x2 blocks; gpu: G in 1,2,4

    # ``diag_algorithm`` has ONE declared point, so it is a PIN, not an
    # axis (§ 4.3a: one point = the value in force) -- it reaches every
    # deck but never a name.  Only the multi-point axes coordinate.
    b64 = _deck_of(sol_calc, "G0N4C1block_size64")
    b128 = _deck_of(sol_calc, "G0N4C1block_size128")
    def kw(text, key):
        # the KEYWORD line, not the item's comment banner above it
        return next(l for l in text.splitlines()
                    if l.split()[:1] == [key])

    assert b64 != b128
    assert kw(b64, "BlockSize").split()[1] == "64"
    assert kw(b128, "BlockSize").split()[1] == "128"
    assert ".false." in kw(b64, "Diag.ELPA.GPU")
    assert "ELPA-1STAGE" in b64, "the one-point pin reaches the deck"
    gpu = _deck_of(sol_calc, "G1N4C1block_size64")
    assert ".true." in kw(gpu, "Diag.ELPA.GPU")

    for job in _trials(sol_calc):
        assert job["point"]["block_size"] in (64, 128)
        assert isinstance(job["point"]["use_gpu"], bool)


# --------------------------------------------------------------------- #
#  split submission: one group per side and shelf, on the one queue     #
# --------------------------------------------------------------------- #

def _submit_dry(calc, *extra):
    """A scripted submit, as a person types it -- `--yes` because there is
    no terminal to ask; its output."""
    r = jobset("launch", "bench", "coarse", "--bundle", calc,
               "--mode", "submit", "--dry-run", "--yes", *extra)
    assert r.exit_code == 0, r.output
    return r.output


def test_split_submission_one_group_per_side_and_shelf(sol_calc):
    """One grouped job per side AND resource shelf (user rulings,
    2026-08-21): the CPU side is one shelf here (one exact-fit group,
    named in full, its side and its cell -- `generator.md` § 4.3a); the
    GPU side spans three device counts, so it submits three exact-fit
    groups -- a G1 trial never idles inside a gres:4 envelope.  Every shelf
    goes to the benchmark's one queue, and only GPU shelves ask for
    devices."""
    declare_bench(sol_calc, _small_matrix())
    _prep_on(sol_calc, "general")
    shelves = shelf_headers(sol_calc, "coarse")
    assert len(shelves) == 4
    cpu = shelves["bench-group-cpu-G0N4C1"]
    assert "#SBATCH -p general" in cpu
    assert not any(ln.startswith("#SBATCH --gres") for ln in cpu), cpu
    # The shelf token is the SAME spelling its trials carry:
    # `G<gpus>N<ranks>C<cores-per-rank>`, 4 ranks on each g.
    for g in (1, 2, 4):
        header = shelves[f"bench-group-gpu-G{g}N4C1"]
        assert "#SBATCH -p general" in header, header
        assert f"#SBATCH --gres=gpu:{g}" in header, header
    # widest first ACROSS shelves: the G4 group precedes G2 precedes G1
    assert [n for n in shelves if "-gpu-" in n] == [
        "bench-group-gpu-G4N4C1", "bench-group-gpu-G2N4C1",
        "bench-group-gpu-G1N4C1"]
    assert _submit_dry(sol_calc).count("rides the group") == 8


def test_only_sends_one_side_to_the_queue_its_prep_named(sol_calc):
    """``--only gpu`` carries just that side, to the queue the benchmark's
    prep named -- a second queue with GPUs here, not the first on the
    menu."""
    declare_bench(sol_calc, _small_matrix())
    _prep_on(sol_calc, "public")
    shelves = shelf_headers(sol_calc, "coarse", side="gpu")
    assert len(shelves) == 3
    assert all("-gpu-" in name for name in shelves), list(shelves)
    assert all("#SBATCH -p public" in header
               for header in shelves.values()), shelves


def test_the_shelves_submit_widest_first(sol_calc):
    """User rules, 2026-08-21: two declared widths are two exact-fit
    SHELVES (nothing idles inside a group), submitted widest first so the
    expensive measurements land first and an early stop still summarizes
    to a verdict.  Declared ASCENDING here, so the order must flip it."""
    declare_bench(sol_calc, {"mpi_np": [2, 4], "omp_threads": [1],
                        "use_gpu": [False]})
    _prep_on(sol_calc, "htc")
    shelves = shelf_headers(sol_calc, "coarse")
    # The shelf token is the SAME spelling its trials carry
    # (`G<gpus>N<ranks>C<cores-per-rank>`), named in full, side and cell
    # (`generator.md` § 4.3a).
    assert list(shelves) == ["bench-group-cpu-G0N4C1",
                             "bench-group-cpu-G0N2C1"]
    assert "#SBATCH -n 4" in shelves["bench-group-cpu-G0N4C1"]
    assert "#SBATCH -n 2" in shelves["bench-group-cpu-G0N2C1"]
    out = _submit_dry(sol_calc)
    riders = [l.split()[1] for l in out.splitlines() if "rides the group" in l]
    assert riders == ["G0N4C1", "G0N2C1"], riders


def test_submission_gates_the_cold_start_against_the_deck(sol_calc):
    """User-settled 2026-08-21: prep bakes the intent (the measurement
    pin), but the SUBMISSION determines the run's actual starting state --
    so the door verifies the artifact.  A trial deck edited to warm-start
    is refused by name; one with its restart group stripped cannot be
    vouched for and refuses too."""
    declare_bench(sol_calc, {"mpi_np": [4], "omp_threads": [1],
                        "use_gpu": [False], "block_size": [64, 128]})
    _prep_on(sol_calc, "htc")
    # The trial's ARTIFACT directory, not its container: the deck lives in
    # the trial's run (`project-layout.md` § 1.5a).
    deck = next(latest_attempt(_bench(sol_calc) / "bench-G0N4C1block_size64"
                               ).glob("*.fdf"))
    text = deck.read_text()
    assert "DM.UseSaveDM" in text and ".false." in text

    warm_text = re.sub(r"(DM\.UseSaveDM\s+)\.false\.", r"\1.true.", text)
    assert warm_text != text, "the flip must actually land"
    deck.write_text(warm_text)
    r = jobset("launch", "bench", "coarse", "--bundle", sol_calc,
               "--mode", "submit", "--dry-run", "--yes")
    assert r.exit_code != 0
    assert "WARM-start" in r.output and "G0N4C1block_size64" in r.output

    deck.write_text(re.sub(r"^[ \t]*[A-Za-z.]*UseSave.*$", "",
                           text, flags=re.M))
    r = jobset("launch", "bench", "coarse", "--bundle", sol_calc,
               "--mode", "submit", "--dry-run", "--yes")
    assert r.exit_code != 0
    assert "no restart group" in r.output

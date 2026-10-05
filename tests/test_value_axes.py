"""2β value axes, end to end — `generator.md` § 4.3a, built 2026-08-21.

The acceptance case is the USER's live matrix (their
``projects/Au-BDT-Au/optimization/Relax/task.json``, declared 2026-08-21):
``mpi_np × use_gpu × diag_algorithm × block_size``, prepped from a
Sol-shaped record — a login node with NO local GPU, a menu whose
gpu-capable row carries the sinfo inventory and the probed 48-core cap
(the GPU node group's own core count; hand-editable).
Every rule the section states is pinned here: the family split, the
inventory fallback, the cap drop BY NAME, the per-trial coordinates in the
decks and in ``job-set.json``, the per-side grouped submission with the
routing preference, and the winner's coordinates riding ``run-config.toml``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.scheduler import Domain, Environment, Topology
from molbuilder.jobset.prep_inputs import bench_inputs
from molbuilder.jobset.materialize import latest_attempt
from molbuilder.jobset.errors import PrepError
from molbuilder.jobset.prep import prep_calculation
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure


@pytest.fixture(autouse=True)
def _tmp_is_the_projects_tree(tmp_path, monkeypatch):
    """These tests build a calculation under ``tmp_path`` and hand its path
    to a verb.  ``--bundle`` is fenced to the projects tree
    (`job-contracts.md` § 2.5b), so the test declares where its tree IS --
    exactly what a user does when calculations live on scratch.
    """
    from molbuilder.projects import PROJECTS_ROOT_ENV
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))

#: The user's declared matrix, with the catalogue's canonical enum casing
#: (their saved file spelled "ELPA-1Stage" — the preflight test below pins
#: that THIS is now caught at describe time, not at prep on Sol).
USERS_MATRIX = {
    "mpi_np": [32, 64, 128],
    "use_gpu": [True, False],
    "diag_algorithm": ["ELPA-1STAGE", "ELPA-2STAGE"],
    "block_size": [64, 128, 256],
}


@pytest.fixture
def sol_calc(tmp_path):
    """A described calculation on a Sol-shaped machine: 128-core login
    node, no local GPU, and a probed menu — ``htc`` cpu-only, ``general``
    carrying the a100 inventory and the probed ``max_cores`` cap."""
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB",
                                         use_gpu=False,
                                         diag_algorithm="ELPA-1STAGE"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, ["H"])
    # The wall and memory every job sent to this scheduler states -- the
    # trials share the calculation's (`architecture.md` § 5.2).
    tj = dest / "task.json"
    d = json.loads(tj.read_text())
    d["allocation"] = {"time": "0-01:00:00", "mem": "8G"}
    tj.write_text(json.dumps(d, indent=2))
    env = Environment(
        scheduler="slurm",
        topology=Topology(sockets=2, cores_per_socket=64,
                          gpus_per_node=None, gpu_type=None),
        domains=[Domain(name="htc", partition="htc", qos="public",
                        max_time="0-04:00:00"),
                 Domain(name="general", partition="general", qos="public",
                        max_time="7-00:00:00", max_cores=48,
                        gpu={"a100": 4})],
        # how a shell enters an environment there -- the record's fact
        env_init={"activation": "conda activate",
                           "preamble": "true"})
    (dest / "environment.json").write_text(env.to_json() + "\n")
    return dest


def _declare(calc, axes):
    p = calc / "task.json"
    obj = json.loads(p.read_text())
    obj["bench"] = axes
    p.write_text(json.dumps(obj, indent=2) + "\n")


# --------------------------------------------------------------------- #
#  enumeration: families, inventory fallback, cap drop                   #
# --------------------------------------------------------------------- #

def test_the_users_matrix_enumerates_both_families(sol_calc):
    """The acceptance matrix: 18 CPU trials (every declared rank count)
    and 18 GPU trials (only np32 survives the 48-core cap; its G ranges
    over the divisors 1/2/4) — and the dropped cells are said BY NAME."""
    _declare(sol_calc, USERS_MATRIX)
    notes: list = []
    points, pins, tr = bench_inputs(sol_calc, None, notes=notes)

    cpu = [p for p in points if not p["G"]]
    gpu = [p for p in points if p["G"]]
    assert len(cpu) == 18 and len(gpu) == 18
    assert sorted({p["K"] for p in cpu}) == [32, 64, 128]
    assert all(p["G"] * p["K"] == 32 for p in gpu), \
        "np64/np128 exceed the gpu domain's 48 cores/node"
    assert all(p["use_gpu"] is False for p in cpu)
    assert all(p["use_gpu"] is True for p in gpu)
    # every point carries the full value coordinate
    assert all({"diag_algorithm", "block_size"} <= set(p) for p in points)
    # The struck cells are named, with the queue and the number that
    # struck them.  The wording changed on 2026-08-30 -- the machine used
    # to print "dropped from the GPU family (domain 'general' allows 48
    # cores/node)" and now prints the whole grid, kept and crossed out,
    # each struck row carrying its own reason.  The CLAIM is unchanged:
    # nothing is dropped silently -- the notes say it, which both prep
    # doors show (`prep.prep_stage`).
    out = "\n".join(notes)
    assert "crossed out" in out
    assert "general" in out and "48" in out
    assert "G1K64C1" in out and "G1K128C1" in out
    # an axis is never a pin; the measurement pins still ride
    assert "block_size" not in pins and pins["max_scf_iter"] == 3


def test_the_gpu_family_answers_from_the_domain_inventory(sol_calc):
    """The fixture's TOPOLOGY has no GPU (a login node) — the family's
    device count comes from the menu's recorded inventory, which is what
    lets a login node enumerate for the cluster behind it; the ask is a
    count, and names no card (`scheduler.md` R2a)."""
    _declare(sol_calc, USERS_MATRIX)
    points, _pins, tr = bench_inputs(sol_calc, None)
    g2 = next(p for p in points if p["G"] == 2)
    res = tr.to_resources(g2, None)
    assert res["gres"] == "gpu:2"
    assert res["mpi_np"] == 2 * g2["K"]
    # and a CPU point of the SAME translation asks for no device at all
    c = next(p for p in points if not p["G"])
    res_c = tr.to_resources(c, None)
    assert "gres" not in res_c and res_c["mpi_np"] == c["K"]


def test_a_queue_listing_several_cards_answers_the_most_gpus_a_node_holds(
        sol_calc):
    """A queue whose record lists two cards is no question to refuse: the
    GPU count a node can hold is the most any of its nodes holds -- here
    the 16 of the second -- and no card is chosen (`scheduler.md` R2a).
    It refused, asking the person to curate the record, until 2026-10-01."""
    env = json.loads((sol_calc / "environment.json").read_text())
    env["domains"][1]["gpu"] = {"a100": 4, "a100.20gb": 16}
    (sol_calc / "environment.json").write_text(json.dumps(env))
    _declare(sol_calc, {"mpi_np": [32], "omp_threads": [1],
                        "use_gpu": [True], "gpu_count": [8]})
    points, _pins, tr = bench_inputs(sol_calc, None)
    assert [p["G"] for p in points] == [8], points
    assert tr.to_resources(points[0], None)["gres"] == "gpu:8"


# `test_a_hand_declared_device_row_enumerates_like_a_probed_one` retired
# 2026-10-02 with the hand-declared device spelling it read (W54 R22,
# `execution/scheduler.md` § 4: the probe's map is the one spelling).


def test_no_gpu_anywhere_refuses_with_both_remedies(sol_calc):
    """No local GPU and no recorded inventory: the family cannot be
    enumerated, and the refusal names the probe AND the menu -- and the
    two steps a re-probed record takes to reach a calculation
    (`configuration.md` M-3: it is set to the record of its first prep, so
    a re-probe alone re-reads the same copy; W52)."""
    env = json.loads((sol_calc / "environment.json").read_text())
    del env["domains"][1]["gpu"]
    (sol_calc / "environment.json").write_text(json.dumps(env))
    _declare(sol_calc, USERS_MATRIX)
    with pytest.raises(PrepError) as e:
        bench_inputs(sol_calc, None)
    said = str(e.value)
    assert "no queue with recorded GPUs" in said, said
    # the record that answered is the calculation's own snapshot, named as
    # such whatever the flag said (the fix-6 review), with its two steps
    assert "this calculation's environment.json" in said, said
    assert "`molbuilder jobset probe --write` here" in said, said
    assert "then go back to the state saved before its first prep" \
        in said, said


# --------------------------------------------------------------------- #
#  the coordinates reach the decks and the record                        #
# --------------------------------------------------------------------- #

def _small_matrix():
    return {"mpi_np": [4], "omp_threads": [1],
            "use_gpu": [True, False],
            "diag_algorithm": ["ELPA-1STAGE"],
            "block_size": [64, 128]}


def _prep(calc):
    from molbuilder.jobset.model import Resources
    points, pins, tr = bench_inputs(calc, None)
    dirs = prep_calculation(calc, "coarse", allocation=Resources(),
                            emit_sbatch=False, sweep=points, pins=pins,
                            translation=tr)
    return points, dirs


def test_per_trial_coordinates_reach_the_decks(sol_calc):
    """Render-probed, not assumed: two trials differing only in
    ``block_size`` render different ``BlockSize`` lines; the CPU-family
    deck says ``Diag.ELPA.GPU .false.`` while its GPU sibling says
    ``.true.`` — and `job-set.json` records every trial's point as data."""
    _declare(sol_calc, _small_matrix())
    points, dirs = _prep(sol_calc)
    assert len(dirs) == 2 + 3 * 2       # cpu: 1 cell x2 blocks; gpu: G in 1,2,4

    def deck(sub):
        # MATCHED ON THE PATH, not the leaf.  A trial keeps attempts since
        # 2026-08-27 (`project-layout.md` § 1.5a), so the leaf is `run-0`
        # and the coordinate names the directory above it.  The property
        # under test -- that each trial's own coordinate reaches its deck
        # -- is unchanged; only where the deck sits moved.
        d = next(p for p in dirs if sub in str(p))
        return next(d.glob("*.fdf")).read_text()

    # ``diag_algorithm`` has ONE declared point, so it is a PIN, not an
    # axis (§ 4.3a: one point = the value in force) -- it reaches every
    # deck but never a name.  Only the multi-point axes coordinate.
    b64 = deck("G0K4C1block_size64")
    b128 = deck("G0K4C1block_size128")
    def kw(text, key):
        # the KEYWORD line, not the item's comment banner above it
        return next(l for l in text.splitlines()
                    if l.split()[:1] == [key])

    assert b64 != b128
    assert kw(b64, "BlockSize").split()[1] == "64"
    assert kw(b128, "BlockSize").split()[1] == "128"
    assert ".false." in kw(b64, "Diag.ELPA.GPU")
    assert "ELPA-1STAGE" in b64, "the one-point pin reaches the deck"
    gpu = deck("G1K4C1block_size64")
    assert ".true." in kw(gpu, "Diag.ELPA.GPU")

    js = json.loads(
        (sol_calc / "01_coarse" / "bench" / "job-set.json").read_text())
    for job in js["jobs"]:
        assert job["point"]["block_size"] in (64, 128)
        assert isinstance(job["point"]["use_gpu"], bool)


# --------------------------------------------------------------------- #
#  split submission: one group per side and shelf; domains; override    #
# --------------------------------------------------------------------- #

def _submit_dry(calc, *extra):
    """A scripted submit names its queues, exactly as a person now does.

    `htc` for the CPU side and `general` for the GPU side: a cpu-only
    partition cannot take a GPU group, so one queue cannot answer for both
    when a sweep splits.  `--yes` because there is no terminal to ask.
    """
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(
        jobset_group, ["launch", "bench", "coarse", "--bundle", str(calc),
                       "--mode", "submit", "--dry-run", "--yes",
                       "--domain", "htc", "--gpu-domain", "general", *extra])
    assert r.exit_code == 0, r.output
    return r.output


def test_split_submission_one_group_per_side_and_shelf(sol_calc):
    """One grouped job per side AND resource shelf (user rulings,
    2026-08-21): the CPU side is one shelf here (one exact-fit group,
    named by side alone); the GPU side spans three device counts, so it
    submits three exact-fit groups -- a G1 trial never idles inside a
    gres:4 envelope.  The CPU group prefers the cpu-only domain, every
    GPU group the gpu-capable one, and only GPU groups ask for devices."""
    _declare(sol_calc, _small_matrix())
    _prep(sol_calc)
    out = _submit_dry(sol_calc)
    plans = [l for l in out.splitlines() if "WOULD run" in l]
    assert len(plans) == 4
    cpu = next(l for l in plans if "bench-group-cpu " in l)
    assert "-p htc" in cpu and "--gres" not in cpu
    # The shelf token is the SAME spelling its trials carry (2026-08-24):
    # `G<gpus>K<ranks-per-gpu>C<cores>`, where 4 ranks over g devices is
    # K = 4/g.  It was `g<gpus>n<TOTAL-ranks>c<cores>` -- the same three
    # facts in a second vocabulary, beside directories using the first.
    for g in (1, 2, 4):
        tok = f"G{g}K{4 // g}C1"
        line = next(l for l in plans if f"bench-group-gpu-{tok}" in l)
        assert "-p general" in line and f"--gres=gpu:{g}" in line
    # widest first ACROSS shelves: the G4 group precedes G2 precedes G1
    order = [l for l in plans if "bench-group-gpu-" in l]
    assert ["G4" in order[0], "G2" in order[1], "G1" in order[2]] == \
        [True, True, True]
    assert out.count("rides the group") == 8


def test_only_submits_one_side_and_the_named_queue_wins(sol_calc):
    """``--only gpu`` carries just that side, and the queue named for that
    side is where it goes — nothing infers a preference over a stated answer.

    *Migrated 2026-08-23.*  This passed `--domain htc` and expected it to
    place the GPU shelves.  With a queue named per side, forcing the GPU side
    means naming the GPU side's queue: a cpu-only partition cannot take a GPU
    group, so one flag answering for both was exactly the conflation that made
    `--gpu-domain` necessary.  The claim is unchanged — a stated queue wins —
    only the spelling is per-side now.
    """
    # A SECOND QUEUE WITH GPUS, named for the GPU side: the stated queue
    # must win over the one placement would pick (`general`, first on the
    # menu).  This named the cpu-only `htc` until 2026-10-01 -- the gap
    # that let a GPU group onto a queue with no GPUs (`scheduler.md` R2a).
    env = json.loads((sol_calc / "environment.json").read_text())
    env["domains"].append({"name": "public", "partition": "public",
                           "qos": "public", "max_time": "7-00:00:00",
                           "max_cores": 48, "gpu": {"a100": 4}})
    (sol_calc / "environment.json").write_text(json.dumps(env, indent=2))
    _declare(sol_calc, _small_matrix())
    _prep(sol_calc)
    out = _submit_dry(sol_calc, "--only", "gpu", "--gpu-domain", "public")
    plans = [l for l in out.splitlines() if "WOULD run" in l]
    assert len(plans) == 3
    assert all("bench-group-gpu-" in l for l in plans)
    assert all("-p public" in l for l in plans), \
        "the named queue wins over the preference for every shelf"
    assert "bench-group-cpu" not in out


def test_the_shelves_submit_widest_first(sol_calc):
    """User rules, 2026-08-21: two declared widths are two exact-fit
    SHELVES (nothing idles inside a group), submitted widest first so the
    expensive measurements land first and an early stop still summarizes
    to a verdict.  Declared ASCENDING here, so the order must flip it."""
    _declare(sol_calc, {"mpi_np": [2, 4], "omp_threads": [1],
                        "use_gpu": [False]})
    _prep(sol_calc)
    out = _submit_dry(sol_calc)
    plans = [l for l in out.splitlines() if "WOULD run" in l]
    assert len(plans) == 2
    # The shelf token is the SAME spelling its trials carry
    # (`G<gpus>K<ranks-per-gpu>C<cores>`, 2026-08-24) -- it was
    # `g<gpus>n<TOTAL-ranks>c<cores>` while the directories that
    # same job launches were named the other way.
    assert "bench-group-G0K4C1" in plans[0] and " -n 4 " in plans[0]
    assert "bench-group-G0K2C1" in plans[1] and " -n 2 " in plans[1]
    riders = [l.split()[1] for l in out.splitlines() if "rides the group" in l]
    assert riders == ["K4C1", "K2C1"], riders


def test_submission_gates_the_cold_start_against_the_deck(sol_calc):
    """User-settled 2026-08-21: prep bakes the intent (the measurement
    pin), but the SUBMISSION determines the run's actual starting state --
    so the door verifies the artifact.  A trial deck edited to warm-start
    is refused by name; one with its restart group stripped cannot be
    vouched for and refuses too."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    _declare(sol_calc, {"mpi_np": [4], "omp_threads": [1],
                        "use_gpu": [False], "block_size": [64, 128]})
    _prep(sol_calc)
    # The trial's ARTIFACT directory, not a hardcoded container: since
    # 2026-08-27 the deck lives in the attempt (`project-layout.md`
    # § 1.5a).  The gate this exercises -- submission verifies the deck it
    # is about to launch starts cold -- is unchanged.
    _c = sol_calc / "01_coarse" / "bench" / "bench-K4C1block_size64"
    deck = next((latest_attempt(_c) or _c).glob("*.fdf"))
    text = deck.read_text()
    assert "DM.UseSaveDM" in text and ".false." in text

    import re
    warm_text = re.sub(r"(DM\.UseSaveDM\s+)\.false\.", r"\1.true.", text)
    assert warm_text != text, "the flip must actually land"
    deck.write_text(warm_text)
    r = CliRunner().invoke(
        jobset_group, ["launch", "bench", "coarse", "--bundle",
                       str(sol_calc), "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert r.exit_code != 0
    assert "WARM-start" in r.output and "K4C1block_size64" in r.output

    deck.write_text(re.sub(r"^[ \t]*[A-Za-z.]*UseSave.*$", "",
                           text, flags=re.M))
    r = CliRunner().invoke(
        jobset_group, ["launch", "bench", "coarse", "--bundle",
                       str(sol_calc), "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert r.exit_code != 0
    assert "no restart group" in r.output


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# `test_the_winners_value_coordinates_reach_the_report`:
# it wrote a SIESTA output and an SCF-timing log by hand into benchmark
# trial folders that never ran, so the summary had a winner to pick: an
# output invented as text (`process/testing.md` § 6).


# --------------------------------------------------------------------- #
#  gpu_count: the device count is DECLARED, not derived                  #
# --------------------------------------------------------------------- #

def test_declared_gpu_count_is_exact(sol_calc):
    """User ruling, 2026-08-21 ("explicit is what we need"): declared
    device counts are measured exactly -- no divisor invention -- and an
    uneven (mpi_np, G) pair is dropped BY NAME (ELPA's equal-share rule),
    never rounded."""
    _declare(sol_calc, {"mpi_np": [32], "omp_threads": [1],
                        "use_gpu": [True], "gpu_count": [1, 2, 3]})
    points, _pins, _tr = bench_inputs(sol_calc, None)
    assert sorted(p["G"] for p in points) == [1, 2],         "exactly the declared counts that divide -- G4 must NOT appear"
    assert all(p["G"] * p["K"] == 32 for p in points)


def test_uneven_split_is_dropped_by_name(sol_calc):
    _declare(sol_calc, {"mpi_np": [32], "omp_threads": [1],
                        "use_gpu": [True], "gpu_count": [2, 3]})
    notes: list = []
    bench_inputs(sol_calc, None, notes=notes)
    out = "\n".join(notes)
    assert "split EVENLY" in out and "mpi_np=32 x gpu_count=3" in out


def test_gpu_count_beyond_the_record_is_refused(sol_calc):
    _declare(sol_calc, {"mpi_np": [32], "use_gpu": [True],
                        "gpu_count": [8]})
    with pytest.raises(PrepError) as e:
        bench_inputs(sol_calc, None)
    assert "gpu_count = [8]" in str(e.value)
    assert "4 device(s)" in str(e.value)


def test_gpu_count_on_a_cpu_bench_is_refused_not_ignored(sol_calc):
    _declare(sol_calc, {"mpi_np": [32], "use_gpu": [False],
                        "gpu_count": [2]})
    with pytest.raises(PrepError) as e:
        bench_inputs(sol_calc, None)
    assert "silently ignored" in str(e.value)


def test_every_cell_uneven_refuses_a_gpu_only_bench(sol_calc):
    _declare(sol_calc, {"mpi_np": [32], "use_gpu": [True],
                        "gpu_count": [3]})
    with pytest.raises(PrepError) as e:
        bench_inputs(sol_calc, None)
    assert "no GPU cell survived" in str(e.value)
    assert "crossed-out list" in str(e.value), (
        "the refusal must point at the list that names each struck cell "
        "and why -- not merely say that everything went")


def test_the_worked_example_matrix_enumerates_as_the_doc_states(sol_calc):
    """tuning.md § 2.12's own worked declaration, end to end: gpu_count
    filters ONLY the GPU family (the CPU family keeps every declared rank
    count); use_gpu as a two-point axis beside gpu_count does NOT
    trigger the cpu-only refusal; and the probed 48-core cap drops a
    DECLARED count by name (np64 × G1 = 64 cores) — the cap lane, distinct
    from the even-split lane."""
    _declare(sol_calc, {"mpi_np": [32, 64], "omp_threads": [1],
                        "use_gpu": [True, False],
                        "gpu_count": [1, 2, 4]})
    notes: list = []
    points, _pins, _tr = bench_inputs(sol_calc, None, notes=notes)
    cpu = [p for p in points if not p["G"]]
    gpu = [p for p in points if p["G"]]
    assert sorted({p["K"] for p in cpu}) == [32, 64], \
        "gpu_count must not touch the CPU family's rank counts"
    # the cap bounds a trial's TOTAL cores (ranks × cores-per-rank ≤ the
    # GPU nodes' 48 — § 4.3a's "ranks × cores"), so EVERY np64 GPU cell
    # drops whatever its G (a 64-rank single-node trial cannot fit a
    # 48-core node), each named in the echo; np32 keeps all three counts
    assert sorted((p["G"], p["K"]) for p in gpu) == \
        [(1, 32), (2, 16), (4, 8)]
    out = "\n".join(notes)
    assert "crossed out" in out
    for cell in ("G1K64C1", "G2K32C1", "G4K16C1"):
        assert cell in out, f"the struck cell {cell} must be named"


def test_gpu_count_alone_filters_the_proposed_grid(sol_calc):
    """gpu_count without mpi_np does NOT invent a rank grid: the K x C
    half stays the machine's proposal, filtered to the declared device
    counts."""
    _declare(sol_calc, {"use_gpu": [True], "gpu_count": [2]})
    points, _pins, _tr = bench_inputs(sol_calc, None)
    assert points, "the probed ladder must survive the filter"
    assert {p["G"] for p in points} == {2}
    assert len({p["K"] for p in points}) > 1,         "K must still range over the machine's proposal"


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# `test_summarize_mid_flight_lists_unfinished_and_refreshes`:
# it wrote a SIESTA output and an SCF-timing log by hand into benchmark
# trial folders that never ran, so the summary had a winner to pick: an
# output invented as text (`process/testing.md` § 6).


# --------------------------------------------------------------------- #
#  the describe-time shape checks (the casing that hit Sol)              #
# --------------------------------------------------------------------- #

def test_preflight_flags_a_typod_choice_per_point(sol_calc):
    """The live 2026-08-21 case: a matrix saved as 'ELPA-1Stage' (pre-U1
    UI) failed only at `prep bench` on Sol.  The preflight now says it at
    describe time, per point, with the choices — and a repeated point is
    refused too."""
    from molbuilder.task import read_task
    from molbuilder.validation.task import preflight
    _declare(sol_calc, {"diag_algorithm": ["ELPA-1Stage", "ELPA-2STAGE"],
                        "use_gpu": [True, True]})
    issues = preflight(read_task(sol_calc / "task.json"))
    msgs = [i.message for i in issues]
    assert any("'ELPA-1Stage'" in m and "ELPA-1STAGE" in m for m in msgs), \
        "the typo'd point must be named WITH the choices"
    assert any("repeated point" in m for m in msgs)


def test_a_duplicated_allocation_point_refuses_at_prep(sol_calc):
    """R2-5's named divergence, closed: the shape rules live ONCE
    (`validation/task.py::_bench_points_fit_their_items`) and prep's
    reader calls that same checker.  Before the fold, prep's local copy
    classified allocation entries before checking them, so
    ``mpi_np: [8, 8, 16]`` was refused at describe and ACCEPTED at prep
    -- two identical grid cells measuring one configuration twice."""
    _declare(sol_calc, {"mpi_np": [8, 8, 16]})
    with pytest.raises(PrepError) as e:
        bench_inputs(sol_calc, None)
    assert "repeated point" in str(e.value)
    assert "mpi_np" in str(e.value)


@pytest.fixture
def flat_sol_calc(tmp_path):
    """The same Sol-shaped machine, FLAT calculation shape — the container
    form the hierarchical fixtures never exercise (R2-8 gap): in flat
    there is no stage directory, so the bench container is the
    root-level ``bench_<NN>_<stage>``."""
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "flatcalc"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB",
                                         use_gpu=False,
                                         diag_algorithm="ELPA-1STAGE"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="flat",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, ["H"])
    # The wall and memory every job sent to this scheduler states -- the
    # trials share the calculation's (`architecture.md` § 5.2).
    tj = dest / "task.json"
    d = json.loads(tj.read_text())
    d["allocation"] = {"time": "0-01:00:00", "mem": "8G"}
    tj.write_text(json.dumps(d, indent=2))
    env = Environment(
        scheduler="slurm",
        topology=Topology(sockets=2, cores_per_socket=64,
                          gpus_per_node=None, gpu_type=None),
        domains=[Domain(name="htc", partition="htc", qos="public",
                        max_time="0-04:00:00"),
                 Domain(name="general", partition="general", qos="public",
                        max_time="7-00:00:00", max_cores=48,
                        gpu={"a100": 4})],
        # how a shell enters an environment there -- the record's fact
        env_init={"activation": "conda activate",
                           "preamble": "true"})
    (dest / "environment.json").write_text(env.to_json() + "\n")
    return dest


def test_a_grouped_bench_on_a_flat_calculation(flat_sol_calc):
    """R2-8's first gap: every grouped-bench test rode the hierarchical
    shape.  In FLAT the trials and the record live in the root-level
    ``bench_<NN>_<stage>`` container (job-contracts.md § 6.3's flat
    arm), and the split submission works the same from there."""
    from molbuilder.jobset._cli import _stage_bench_dir
    _declare(flat_sol_calc, _small_matrix())
    _prep(flat_sol_calc)
    container, token = _stage_bench_dir(flat_sol_calc, "coarse")
    assert container.name.startswith("bench_"), (
        f"flat container is {container.name!r}, not the root-level "
        f"bench_<NN>_<stage> form")
    assert container.is_dir(), "prep did not lay the flat container down"
    assert (container / "job-set.json").is_file(), (
        "the sweep's record is not in the flat container")
    trials = [d.name for d in container.iterdir() if d.is_dir()]
    assert trials and all(t.startswith("bench-") for t in trials), (
        f"trials misplaced: {trials}")
    out = _submit_dry(flat_sol_calc)
    plans = [l for l in out.splitlines() if "WOULD run" in l]
    assert len(plans) == 4, out    # one CPU shelf + three GPU shelves
    assert any("bench-group-cpu " in l for l in plans)
    assert sum("bench-group-gpu-" in l for l in plans) == 3


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# `test_a_gpu_winner_rides_run_config_on_a_mixed_sweep`:
# it wrote a SIESTA output and an SCF-timing log by hand into benchmark
# trial folders that never ran, so the summary had a winner to pick: an
# output invented as text (`process/testing.md` § 6).

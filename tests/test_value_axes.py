"""2β value axes, end to end — `generator.md` § 4.3a, built 2026-08-21.

The acceptance case is the USER's live matrix (their
``projects/Au-BDT-Au/optimization/Relax/task.json``, declared 2026-08-21):
``mpi_np × enable_gpu × diag_algorithm × block_size``, prepped from a
Sol-shaped record — a login node with NO local GPU, a menu whose
gpu-capable row carries the sinfo inventory and the CURATED 48-core cap.
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
from molbuilder.environment import Domain, Environment, Topology
from molbuilder.jobset._cli import _bench_inputs
from molbuilder.jobset.prep import prep_calculation
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure

#: The user's declared matrix, with the catalogue's canonical enum casing
#: (their saved file spelled "ELPA-1Stage" — the preflight test below pins
#: that THIS is now caught at describe time, not at prep on Sol).
USERS_MATRIX = {
    "mpi_np": [32, 64, 128],
    "enable_gpu": [True, False],
    "diag_algorithm": ["ELPA-1STAGE", "ELPA-2STAGE"],
    "block_size": [64, 128, 256],
}


@pytest.fixture
def sol_calc(tmp_path):
    """A described calculation on a Sol-shaped machine: 128-core login
    node, no local GPU, and a probed menu — ``htc`` cpu-only, ``general``
    carrying the a100 inventory and the curated ``max_cores`` cap."""
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB",
                                         enable_gpu=False,
                                         diag_algorithm="ELPA-1STAGE"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, ["H"])
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    env = Environment(
        scheduler="slurm",
        topology=Topology(sockets=2, cores_per_socket=64,
                          gpus_per_node=None, gpu_type=None),
        domains=[Domain(name="htc", partition="htc", qos="public",
                        max_time="0-04:00:00"),
                 Domain(name="general", partition="general", qos="public",
                        max_time="7-00:00:00", max_cores=48,
                        gpu={"a100": 4})])
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

def test_the_users_matrix_enumerates_both_families(sol_calc, capsys):
    """The acceptance matrix: 18 CPU trials (every declared rank count)
    and 18 GPU trials (only np32 survives the 48-core cap; its G ranges
    over the divisors 1/2/4) — and the dropped cells are said BY NAME."""
    _declare(sol_calc, USERS_MATRIX)
    points, pins, tr = _bench_inputs(sol_calc)

    cpu = [p for p in points if not p["G"]]
    gpu = [p for p in points if p["G"]]
    assert len(cpu) == 18 and len(gpu) == 18
    assert sorted({p["K"] for p in cpu}) == [32, 64, 128]
    assert all(p["G"] * p["K"] == 32 for p in gpu), \
        "np64/np128 exceed the gpu domain's 48 cores/node"
    assert all(p["enable_gpu"] is False for p in cpu)
    assert all(p["enable_gpu"] is True for p in gpu)
    # every point carries the full value coordinate
    assert all({"diag_algorithm", "block_size"} <= set(p) for p in points)
    # the drop is loud and names its cells and the cap's source
    out = capsys.readouterr().out
    assert "dropped from the GPU family" in out
    assert "'general'" in out and "48 cores/node" in out
    assert "G1K64C1" in out and "G1K128C1" in out
    # an axis is never a pin; the measurement pins still ride
    assert "block_size" not in pins and pins["max_scf_iter"] == 3


def test_the_gpu_family_answers_from_the_domain_inventory(sol_calc):
    """The fixture's TOPOLOGY has no GPU (a login node) — the family's
    device count and gres type come from the menu's recorded inventory,
    which is what lets a login node enumerate for the cluster behind it."""
    _declare(sol_calc, USERS_MATRIX)
    points, _pins, tr = _bench_inputs(sol_calc)
    g2 = next(p for p in points if p["G"] == 2)
    res = tr.to_resources(g2, None)
    assert res["gres"] == "gpu:a100:2"
    assert res["mpi_np"] == 2 * g2["K"]
    # and a CPU point of the SAME translation asks for no device at all
    c = next(p for p in points if not p["G"])
    res_c = tr.to_resources(c, None)
    assert "gres" not in res_c and res_c["mpi_np"] == c["K"]


def test_two_inventory_types_refuse_with_the_curation_remedy(sol_calc):
    """Choosing between recorded GPU types would be a ranking — the probe
    buried ``best_gpu_type`` for exactly that.  Refused, with the row
    named."""
    import click
    env = json.loads((sol_calc / "environment.json").read_text())
    env["domains"][1]["gpu"] = {"a100": 4, "a100.20gb": 16}
    (sol_calc / "environment.json").write_text(json.dumps(env))
    _declare(sol_calc, USERS_MATRIX)
    with pytest.raises(click.ClickException) as e:
        _bench_inputs(sol_calc)
    assert "several GPU types" in str(e.value)
    assert "a100.20gb" in str(e.value)


def test_no_gpu_anywhere_refuses_with_both_remedies(sol_calc):
    """No local GPU and no recorded inventory: the family cannot be
    enumerated, and the refusal names the probe AND the menu."""
    import click
    env = json.loads((sol_calc / "environment.json").read_text())
    del env["domains"][1]["gpu"]
    (sol_calc / "environment.json").write_text(json.dumps(env))
    _declare(sol_calc, USERS_MATRIX)
    with pytest.raises(click.ClickException) as e:
        _bench_inputs(sol_calc)
    assert "no domain row with a recorded GPU inventory" in str(e.value)


# --------------------------------------------------------------------- #
#  the coordinates reach the decks and the record                        #
# --------------------------------------------------------------------- #

def _small_matrix():
    return {"mpi_np": [4], "omp_threads": [1],
            "enable_gpu": [True, False],
            "diag_algorithm": ["ELPA-1STAGE"],
            "block_size": [64, 128]}


def _prep(calc):
    from molbuilder.jobset.model import Resources
    points, pins, tr = _bench_inputs(calc)
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
        d = next(p for p in dirs if sub in p.name)
        return next(d.glob("*.fdf")).read_text()

    # ``diag_algorithm`` has ONE declared point, so it is a PIN, not an
    # axis (§ 4.3a: one point = the value in force) -- it reaches every
    # deck but never a name.  Only the multi-point axes coordinate.
    b64 = deck("G0K4C1enable_gpuFalseblock_size64")
    b128 = deck("G0K4C1enable_gpuFalseblock_size128")
    def kw(text, key):
        # the KEYWORD line, not the item's comment banner above it
        return next(l for l in text.splitlines()
                    if l.split()[:1] == [key])

    assert b64 != b128
    assert kw(b64, "BlockSize").split()[1] == "64"
    assert kw(b128, "BlockSize").split()[1] == "128"
    assert ".false." in kw(b64, "Diag.ELPA.GPU")
    assert "ELPA-1STAGE" in b64, "the one-point pin reaches the deck"
    gpu = deck("G1K4C1enable_gpuTrueblock_size64")
    assert ".true." in kw(gpu, "Diag.ELPA.GPU")

    js = json.loads(
        (sol_calc / "01_coarse" / "bench" / "job-set.json").read_text())
    for job in js["jobs"]:
        assert job["point"]["block_size"] in (64, 128)
        assert isinstance(job["point"]["enable_gpu"], bool)


# --------------------------------------------------------------------- #
#  split submission: one group per side, preferred domains, override     #
# --------------------------------------------------------------------- #

def _submit_dry(calc, *extra):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(
        jobset_group, ["submit", "bench", "coarse", "--bundle", str(calc),
                       "--mode", "submit", "--dry-run", *extra])
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
    for g in (1, 2, 4):
        line = next(l for l in plans if f"bench-group-gpu-g{g}n4c1" in l)
        assert "-p general" in line and f"--gres=gpu:a100:{g}" in line
    # widest first ACROSS shelves: the g4 group precedes g2 precedes g1
    order = [l for l in plans if "bench-group-gpu-" in l]
    assert ["g4" in order[0], "g2" in order[1], "g1" in order[2]] ==         [True, True, True]
    assert out.count("rides the group") == 8


def test_only_submits_one_side_and_domain_overrides(sol_calc):
    """``--only gpu`` carries just that side; ``--domain`` overrides the
    per-side preference for whatever it carries — together they place each
    side wherever the user says."""
    _declare(sol_calc, _small_matrix())
    _prep(sol_calc)
    out = _submit_dry(sol_calc, "--only", "gpu", "--domain", "htc")
    plans = [l for l in out.splitlines() if "WOULD run" in l]
    assert len(plans) == 3
    assert all("bench-group-gpu-" in l for l in plans)
    assert all("-p htc" in l for l in plans), \
        "--domain overrides the preference for every shelf"
    assert "bench-group-cpu" not in out


def test_the_shelves_submit_widest_first(sol_calc):
    """User rules, 2026-08-21: two declared widths are two exact-fit
    SHELVES (nothing idles inside a group), submitted widest first so the
    expensive measurements land first and an early stop still summarizes
    to a verdict.  Declared ASCENDING here, so the order must flip it."""
    _declare(sol_calc, {"mpi_np": [2, 4], "omp_threads": [1],
                        "enable_gpu": [False]})
    _prep(sol_calc)
    out = _submit_dry(sol_calc)
    plans = [l for l in out.splitlines() if "WOULD run" in l]
    assert len(plans) == 2
    assert "bench-group-n4c1" in plans[0] and " -n 4 " in plans[0]
    assert "bench-group-n2c1" in plans[1] and " -n 2 " in plans[1]
    riders = [l.split()[1] for l in out.splitlines() if "rides the group" in l]
    assert riders == ["K4C1", "K2C1"], riders


# --------------------------------------------------------------------- #
#  summarize: the coordinate rides the record and the proposal           #
# --------------------------------------------------------------------- #

def test_the_winners_coordinates_ride_run_config(sol_calc):
    """The winner's VALUE coordinates land in the proposal's ``[pins]``
    (typed: int stays bare TOML), and `prep run`'s reader accepts its own
    writer's output — the hand vocabulary that refused ``block_size`` is
    gone."""
    from molbuilder.jobset._cli import _load_bench_set
    from molbuilder.jobset.materialize import job_dir_names, shape_of
    from molbuilder.jobset.summarize import (read_run_config,
                                             run_summarize_jobset)
    _declare(sol_calc, _small_matrix())
    _prep(sol_calc)
    js, base = _load_bench_set(str(sol_calc), "coarse")
    dirs = job_dir_names(js, shape_of(js, sol_calc))
    winner = "G0K4C1enable_gpuFalseblock_size128"
    d = sol_calc / dirs[winner]
    stem = next(d.glob("*.fdf")).name[:-4]
    (d / f"{stem}-run0.out").write_text(
        "* Running on 4 nodes in parallel\nx\n"
        "siesta: Final energy (eV):\nJob completed\n")
    t0 = 1000000.0
    (d / f"{stem}-run0.scf-timing.log").write_text(
        "\n".join(f"{t0 + i * 2.0} iter" for i in range(6)) + "\n")

    res, _out, (cfg_path, status) = run_summarize_jobset(
        js, sol_calc, now_iso="2026-08-21T00:00:00Z", stage="coarse")
    assert status == "written"
    assert res.choice["label"] == winner
    assert res.choice["point"]["block_size"] == 128
    by_label = {p.label: p for p in res.points}
    assert by_label[winner].point["block_size"] == 128
    assert by_label[winner].point["enable_gpu"] is False

    text = cfg_path.read_text()
    assert "block_size = 128" in text
    assert 'diag_algorithm = "ELPA-1STAGE"' in text
    cfg = read_run_config(cfg_path, engine="siesta")
    assert cfg["pins"]["block_size"] == 128
    assert cfg["pins"]["enable_gpu"] is False


def test_summarize_mid_flight_lists_unfinished_and_refreshes(sol_calc):
    """User rule, 2026-08-21: summarize works WHILE the bench runs -- the
    finished trials are summarized consistently, the unfinished ones are
    listed as unfinished (never a failure of the set), the coverage clause
    says how partial the verdict is, and a later summarize refreshes the
    record over the fuller evidence."""
    from molbuilder.jobset._cli import _load_bench_set
    from molbuilder.jobset.materialize import job_dir_names, shape_of
    from molbuilder.jobset.summarize import run_summarize_jobset, summary_text
    _declare(sol_calc, _small_matrix())
    _prep(sol_calc)
    js, base = _load_bench_set(str(sol_calc), "coarse")
    dirs = job_dir_names(js, shape_of(js, sol_calc))

    def finish(name, spi):
        d = sol_calc / dirs[name]
        stem = next(d.glob("*.fdf")).name[:-4]
        (d / f"{stem}-run0.out").write_text(
            "* Running on 4 nodes in parallel\nx\n"
            "siesta: Final energy (eV):\nJob completed\n")
        t0 = 1000000.0
        (d / f"{stem}-run0.scf-timing.log").write_text(
            "\n".join(f"{t0 + i * spi} iter" for i in range(4)) + "\n")

    finish("G0K4C1enable_gpuFalseblock_size64", 5.0)
    res, out_path, rc = run_summarize_jobset(
        js, sol_calc, now_iso="2026-08-21T00:00:00Z", stage="coarse")
    text = summary_text(res, out_path, run_config=rc, stage="coarse")
    states = {p.label: p.state for p in res.points}
    assert states["G0K4C1enable_gpuFalseblock_size64"] == "completed"
    assert sum(1 for s in states.values() if s != "completed") == 7
    assert "coverage: 1 of 8 prepped points measured" in text
    assert "the verdict ranks what ran" in text

    # more evidence lands; the RECORD refreshes on the next summarize
    finish("G0K4C1enable_gpuFalseblock_size128", 3.0)
    res2, _o, rc2 = run_summarize_jobset(
        js, sol_calc, now_iso="2026-08-21T01:00:00Z", stage="coarse")
    assert res2.choice["label"] == "G0K4C1enable_gpuFalseblock_size128"
    # the PROPOSAL file is the user's after first write: kept, with the
    # refresh taught in the summary text
    assert rc2[1] == "kept"
    assert "delete it and summarize again" in summary_text(
        res2, _o, run_config=rc2, stage="coarse")


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
                        "enable_gpu": [True, True]})
    issues = preflight(read_task(sol_calc / "task.json"))
    msgs = [i.message for i in issues]
    assert any("'ELPA-1Stage'" in m and "ELPA-1STAGE" in m for m in msgs), \
        "the typo'd point must be named WITH the choices"
    assert any("repeated point" in m for m in msgs)

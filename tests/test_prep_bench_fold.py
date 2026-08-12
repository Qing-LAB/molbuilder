"""`jobset prep bench` — step 6 u2: the grid through the ONE builder.

Contract: `project-layout.md` § 2.3.1a (benchmarking is `prep` whose
parameters are a set) · § 2.3.2 (trials relabelled + forced cold; submission
one trial per invocation) · `generator.md` §§ 2, 5 · `template.md` § 8.1
(rebuild and render, never splice — the pins land as schema values).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.environment import Environment, Topology
from molbuilder.jobset._cli import _bench_inputs
from molbuilder.jobset.model import Resources
from molbuilder.jobset.prep import prep_calculation
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure


@pytest.fixture
def calc(tmp_path):
    """A described calculation on a machine whose probe found one a100."""
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        dest)
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    # The probe's answer, pre-seeded: resolve_target early-returns on an
    # existing environment.json, so the grid enumerates THIS topology.
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100")).to_json() + "\n")
    return dest


def _prep_bench(calc):
    sweep, pins, translation = _bench_inputs(calc)
    prep_calculation(calc, "coarse",
                     allocation=Resources(mpi_np=8, cpus_per_task=8),
                     sweep=sweep, pins=pins, translation=translation,
                     emit_sbatch=False)
    return json.loads((calc / "job-set.json").read_text())


def test_the_grid_becomes_a_sweep_jobset_of_relabelled_trials(calc):
    js = _prep_bench(calc)
    assert js["kind"] == "sweep"
    names = [j["name"] for j in js["jobs"]]
    assert len(names) == len(set(names)) and len(names) >= 2
    assert all(n.startswith("G1K") and "C" in n for n in names)
    # one deck per trial, named by the TRIAL's label + the stage token
    import re
    for n in names:
        deck = calc / f"JOB-{n}_01_coarse.fdf"
        assert deck.is_file()
        assert re.search(rf"^SystemLabel\s+JOB-{n}\s*$",
                         deck.read_text(), re.M)


def test_the_pins_land_as_rendered_schema_values_not_splices(calc):
    """What `transform_fdf` spliced is now resolved and rendered: the capped
    SCF, the single point, the eigensolver — readable in the deck."""
    import re
    js = _prep_bench(calc)
    deck = (calc / f"JOB-{js['jobs'][0]['name']}_01_coarse.fdf").read_text()
    assert re.search(r"^MaxSCFIterations\s+5\s*$", deck, re.M)
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE\s*$", deck, re.M)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    assert re.search(r"^MD\.NumCGsteps\s+0\s*$", deck, re.M)


def test_each_trials_resources_carry_its_own_coordinate(calc):
    """G·K ranks, C cores, the machine's GPU type in gres — translated once,
    per element, never re-derived downstream (generator.md § 5)."""
    js = _prep_bench(calc)
    for job in js["jobs"]:
        g = int(job["name"][1])
        k = int(job["name"][job["name"].index("K") + 1:job["name"].index("C")])
        c = int(job["name"][job["name"].index("C") + 1:])
        r = job["resources"]
        assert r["mpi_np"] == g * k
        assert r["cpus_per_task"] == c
        assert r["gres"] == f"gpu:a100:{g}"


def test_a_machine_without_gpus_is_refused_by_name(calc):
    import click
    (calc / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1,
                                      cores_per_socket=4)).to_json() + "\n")
    with pytest.raises(click.ClickException, match=r"no GPU topology"):
        _bench_inputs(calc)


def test_cli_prep_bench_requires_a_stage(calc):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group,
                           ["prep", "bench", "--bundle", str(calc)])
    assert r.exit_code != 0
    assert "name it" in r.output


def test_cli_prep_bench_end_to_end_lists_trials_not_attempts(calc):
    """The whole verb through the CLI: trials prepped and listed, and NO
    attempt machinery -- a sweep's jobs are named by coordinate, so the
    run-kind's stage/attempt tail must not run."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group,
                           ["prep", "bench", "coarse", "--bundle", str(calc),
                            "--np", "8", "--cpus-per-task", "8",
                            "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "trial dir(s) for stage 'coarse'" in r.output
    assert "one trial per invocation" in r.output
    assert "config:" in r.output          # provenance rides every prep


def test_cli_submit_bench_is_one_trial_per_invocation(calc):
    """Bare `submit bench` refuses and names the trials; naming one plans
    exactly one launch (§ 2.3.2, decided 2026-08-12)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    trial = js["jobs"][0]["name"]
    runner = CliRunner()
    # SUBMIT mode: the whole set is refused -- a scheduler takes one job per
    # invocation.  (DIRECT is not submission: it runs points sequentially
    # in-shell and is exempt, the same rule as everywhere else.)
    r = runner.invoke(jobset_group, ["submit", "bench", "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run"])
    assert r.exit_code != 0 and "one at a time" in r.output
    r = runner.invoke(jobset_group, ["submit", "bench", trial,
                                     "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert r.output.count("WOULD run") == 1

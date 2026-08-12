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


def test_trials_nest_inside_the_stage_they_measure(calc):
    """u3, `generator.md` § 5: ``<NN>_<stage>/bench-<point>/`` — the trial
    directory is § 6.3's authority name, INSIDE the stage, and its links
    reach the bundle root through a COMPUTED depth, not a hardcoded hop."""
    js = _prep_bench(calc)
    name = js["jobs"][0]["name"]
    d = calc / "01_coarse" / f"bench-{name}"
    assert d.is_dir()
    script = d / f"JOB-{name}_01_coarse.fdf"
    assert script.is_symlink() and script.resolve().is_file()
    wrapper = d / f"JOB-{name}_01_coarse.run.sh"
    assert wrapper.is_symlink() and wrapper.resolve().is_file()
    # nothing landed at the old flat location
    assert not (calc / f"bench-{name}").exists()
    assert not (calc / f"point-{name}").exists()


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


def test_cli_summarize_bench_reads_trials_by_data(calc):
    """u4: discovery keyed by job-set.json's data, results through the same
    artifacts as any run, ASYNC — a trial with no output yet reports
    ``incomplete`` rather than failing the set (user, 2026-08-12)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    name = js["jobs"][0]["name"]
    d = calc / "01_coarse" / f"bench-{name}"
    (d / f"JOB-{name}_01_coarse-run0.out").write_text(
        "banner\nsiesta: Final energy (eV) = -1.0\n")
    r = CliRunner().invoke(jobset_group, ["summarize", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    assert (calc / "bench-result.json").is_file()
    assert name in r.output
    assert "completed" in r.output       # the trial with the finished .out
    assert "unknown" in r.output         # siblings with no output yet
    # A run's outputs are the calculation's, not a benchmark's to rank.
    r = CliRunner().invoke(jobset_group,
                           ["summarize", "run", "--bundle", str(calc)])
    assert r.exit_code != 0


def _finished_trial_and_verdict(calc):
    """prep bench + one completed trial + summarize -> bench-result.json."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    name = js["jobs"][0]["name"]
    d = calc / "01_coarse" / f"bench-{name}"
    (d / f"JOB-{name}_01_coarse-run0.out").write_text(
        "x\n>> End of run:\n")
    # epoch-per-line format: consecutive deltas are the per-iter durations
    (d / f"JOB-{name}_01_coarse-run0.scf-timing.log").write_text(
        "100.0 scf 1\n104.0 scf 2\n108.0 scf 3\n112.0 scf 4\n")
    r = CliRunner().invoke(jobset_group, ["summarize", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    return name


def test_prep_run_offers_the_verdict_and_silence_is_no(calc):
    """§ 2.3.2: it asks; it does not just take it — and a non-interactive
    shell's silence is No, so nothing is ever applied by default."""
    import re
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    _finished_trial_and_verdict(calc)
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "coarse", "--bundle", str(calc),
                            "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "a benchmark result exists" in r.output
    assert "use it?" in r.output
    assert "not applied" in r.output
    deck = (calc / "JOB_01_coarse.fdf").read_text()
    assert not re.search(r"^Diag\.ELPA\.GPU\s+\.true\.", deck, re.M)


def test_prep_run_applies_an_accepted_verdict_but_flags_win(calc):
    """On yes, the measured machine half fills only what the user did NOT
    state, and the winner's eigensolver arrives as pins."""
    import re
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.parse.scripts.bench_marks import _extract_bench_marks_dict
    name = _finished_trial_and_verdict(calc)
    g = int(name[1])
    k = int(name[name.index("K") + 1:name.index("C")])
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "coarse", "--bundle", str(calc),
                            "--cpus-per-task", "3", "--no-sbatch"],
                           input="y\n")
    assert r.exit_code == 0, r.output
    assert "applied:" in r.output
    deck = (calc / "JOB_01_coarse.fdf").read_text()
    marks = _extract_bench_marks_dict(deck)
    assert marks.get("mpi_np") == g * k          # measured, user said nothing
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE\s*$", deck, re.M)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    # the flag the user DID state beat the verdict
    js = json.loads((calc / "job-set.json").read_text())
    assert js["jobs"][0]["resources"]["cpus_per_task"] == 3


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

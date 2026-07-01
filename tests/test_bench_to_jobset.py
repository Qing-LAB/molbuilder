"""The benchmark sweep -> JobSet producer (molbuilder/bench/to_jobset.py) --
D4: bench as a second jobset producer, sharing the grid + point-dir
convention with the legacy bash sweep and with summarize."""
from __future__ import annotations

import re

from molbuilder.bench import adapters, prep
from molbuilder.bench.environment import Environment, Topology
from molbuilder.bench.to_jobset import sweep_to_jobset
from molbuilder.jobset.materialize import materialize
from molbuilder.jobset.model import JobSet


def _env(cores=24, gpn=2, gtype="a100"):
    return Environment(scheduler="slurm",
                       topology=Topology(cores_per_socket=cores,
                                         gpus_per_node=gpn, gpu_type=gtype))


def test_sweep_to_jobset_points_and_resources():
    js = sweep_to_jobset(adapters.SlurmAdapter(), _env(cores=24, gpn=2))
    assert js.kind == "sweep" and js.engine == "siesta"
    names = {j.name for j in js.jobs}
    assert "G1K8C3" in names and "G2K8C3" in names       # K=8 -> c=24//8=3
    j = next(x for x in js.jobs if x.name == "G1K8C3")
    assert j.script == "job-gpu.fdf"
    assert j.resources.mpi_np == 8 and j.resources.cpus_per_task == 3
    assert j.resources.gres == "gpu:a100:1"
    assert js.validate() == []                            # framework-valid


def test_sweep_to_jobset_grid_matches_bash_sweep():
    """The JobSet's points are EXACTLY the bash sweep's point dirs -- both
    iterate the shared adapters.sweep_grid, so they can't diverge."""
    env = _env(cores=24, gpn=2)
    adapter = adapters.SlurmAdapter()
    js = sweep_to_jobset(adapter, env)
    bash = adapter.format_bench(env)["job-gpu-sweep.sh"]
    bash_points = set(re.findall(r"_mb_point (point-G\d+K\d+C\d+)", bash))
    jobset_points = {f"point-{j.name}" for j in js.jobs}
    assert jobset_points == bash_points and jobset_points


def test_sweep_jobset_materializes_to_summarize_dirs(tmp_path):
    """Materializing the JobSet yields point-G<g>K<k>C<c>/ dirs -- exactly the
    convention summarize._POINT_RE globs."""
    js = sweep_to_jobset(adapters.SlurmAdapter(), _env(cores=24, gpn=1))
    (tmp_path / "job-gpu.fdf").write_text("x")
    dirs = materialize(js, tmp_path)
    R = re.compile(r"^point-G\d+K\d+C\d+$")               # == summarize._POINT_RE
    assert dirs and all(R.match(d.name) for d in dirs)


def test_ks_cs_overrides():
    js = sweep_to_jobset(adapters.SlurmAdapter(), _env(cores=24, gpn=1),
                         ks=[4], cs=[6])
    assert [j.name for j in js.jobs] == ["G1K4C6"]
    assert js.jobs[0].resources.mpi_np == 4
    assert js.jobs[0].resources.cpus_per_task == 6


def test_prep_writes_loadable_job_set_json(tmp_path):
    """prep-bench now also emits job-set.json (bench = jobset producer)."""
    prep.run_prep_bench(
        tmp_path,
        overrides={"cores_per_socket": 24, "gpus_per_node": 1,
                   "gpu_type": "a100"},
        scheduler_override="slurm", mode="submit")
    jf = tmp_path / "job-set.json"
    assert jf.is_file()
    js = JobSet.load(jf)
    assert js.kind == "sweep" and js.jobs and js.validate() == []

"""A generated run wrapper never changes directory (`job-contracts.md` § 2.1)."""
from __future__ import annotations

import re

import pytest

from molbuilder import runwrap
from molbuilder.jobset.model import Resources
from molbuilder.runfiles import RunNames


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """The renders read the machine's record (B-9, 2026-08-13: unsandboxed,
    every wrapper here folded in the developer's own configuration, so the
    banner/mover surfaces under test varied by machine).  Sandboxed, with
    the activation the writer requires DECLARED by the test, in the record
    the generator reads it from (`configuration.md` § 4)."""
    cwd = tmp_path_factory.mktemp("cwd")
    monkeypatch.chdir(cwd)
    # THE SANDBOX IS THE CONFIG ROOT: without naming the directory the
    # write lands in a file nothing opens, and the test passes having
    # configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cwd))
    # A PROBED MACHINE.  A rank count is read from a record
    # and nowhere else -- no probe of the running box, no fallback
    # (`running-a-job.md` § 3.1).  A wrapper cannot be rendered on an
    # unprobed machine, so a fixture that renders one probes first,
    # exactly as a person does:  molbuilder jobset probe --write
    from molbuilder.scheduler import Environment as _Env, Topology as _Topo
    (cwd / "environment.json").write_text(
        _Env(scheduler="slurm",
             topology=_Topo(sockets=2, cores_per_socket=32),
             env_init={"activation": "conda activate",
                                "preamble": "true"}).to_json()
        + "\n")


ENGINES = (
    pytest.param("siesta", ".fdf", "SystemLabel job\n", id="siesta"),
    pytest.param("pyscf", ".py", 'JOB = "job"\nimport pyscf\n', id="pyscf"),
)


def _wrapper(tmp_path, ext, body, warm=None, **kw):
    """The wrapper of a run that STATES its shape -- one written for an
    unstated one is refused (`architecture.md` § 5.2) -- handed ``warm``,
    the restart files in effect, as prep hands it."""
    names = RunNames.of("job", "01_coarse", "hierarchical")
    p = tmp_path / names.name(ext)
    p.write_text(body)
    return runwrap.render_run_wrapper(
        p, names=names, env="molbuilder-siesta", warm=warm,
        resources=Resources(**{"mpi_np": 2, "cpus_per_task": 1, **kw}))


#: The generated comments that open the two blocks, in the order the wrapper
#: emits them: the ``--cold`` mover first, the startup banner after it.
_MOVER_HEADING = "--- Cold restart: SAY WHAT WOULD BE LOST, THEN STOP"
_BANNER_HEADING = "--- Runtime status banner"


# PySCF's run script: no basic-tier run starts it, so this lint is what
# holds it in the folder it was launched in.
@pytest.mark.parametrize("engine,ext,body",
                         [p for p in ENGINES if p.id == "pyscf"])
def test_no_generated_wrapper_changes_directory(tmp_path, engine, ext, body):
    """`job-contracts.md § 2.1`: **the caller's working directory is the
    contract** — both launchers establish it, and neither the wrapper nor the
    engine ever navigates.
    """
    text = _wrapper(tmp_path, ext, body)
    cds = re.findall(r"(?m)^\s*cd\s+\S+", text)
    assert cds == [], f"{engine} wrapper navigates: {cds}"

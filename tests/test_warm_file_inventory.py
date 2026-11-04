"""A generated run wrapper never changes directory (`job-contracts.md` § 2.1)."""
from __future__ import annotations

from molbuilder.scheduler.record import write_environment
import re

import pytest

from molbuilder import runwrap
from molbuilder.jobset.model import Resources
from molbuilder.runfiles import RunNames
from support.machines import SLURM


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
    write_environment(
        _Env(machine="this", scheduler=SLURM,
             topology=_Topo(sockets=2, cores_per_socket=32),
             env_init={"activation": "conda activate",
                                "preamble": "true"}),
        (cwd / "environment.json"))


ENGINES = (
    pytest.param("siesta", ".fdf", "SystemLabel job\n", id="siesta"),
    pytest.param("pyscf", ".py", 'JOB = "job"\nimport pyscf\n', id="pyscf"),
)


def _wrapper(tmp_path, engine, ext, body, warm=None, *,
             names=RunNames("job", "01_coarse"), calculation="optimization",
             program=None, **kw):
    """The wrapper of a run that STATES its shape -- one written for an
    unstated one is refused (`architecture.md` § 5.2) -- and carries the run
    settings every job carries, its wall's slack
    (`engines/template.md` § 5.4), handed ``warm``, the restart files in
    effect, and ``program``, the one its deck states, as prep hands them."""
    p = tmp_path / names.name(ext)
    p.write_text(body)
    # THE RECORD PREP READ AND HANDS ON -- the probed machine's
    # (`scheduler.machine_for`).
    from molbuilder.scheduler import machine_for
    return runwrap.render_run_wrapper(
        p, names=names, engine=engine, calculation=calculation,
        program=program, env="molbuilder-siesta",
        machine_record=machine_for(),
        warm=warm,
        resumes=True, relaunch_continues=True, trial=False,
        resources=Resources(**{"mpi_np": 2, "cpus_per_task": 1,
                               "wall_slack": 600, **kw}))


# PySCF's run script: no basic-tier run starts it, so this lint is what
# holds it in the folder it was launched in.
@pytest.mark.parametrize("engine,ext,body",
                         [p for p in ENGINES if p.id == "pyscf"])
def test_no_generated_wrapper_changes_directory(tmp_path, engine, ext, body):
    """`job-contracts.md § 2.1`: **the caller's working directory is the
    contract** — both launchers establish it, and neither the wrapper nor the
    engine ever navigates.
    """
    text = _wrapper(tmp_path, engine, ext, body)
    cds = re.findall(r"(?m)^\s*cd\s+\S+", text)
    assert cds == [], f"{engine} wrapper navigates: {cds}"


def test_a_tbtrans_step_s_run_script_starts_tbtrans_its_banner_lined_up(
        tmp_path):
    """A transport calculation's transmission step: its deck states TBtrans
    as the program that reads it (`transport.deck.transport_spec`), and the
    run script starts TBtrans and names it in its banner -- whose labels
    line up on the longest one it prints, ``TBtrans version``
    (`running-a-job.md` § 1).  API-level: a transmission step is prepared
    only on a finished relaxation, which the road makes only with a real
    engine (F9).

    Until 2.64.0 prep picked TBtrans by the step's name, and the banner's
    ``TBtrans version`` stood one column past its other labels."""
    text = _wrapper(tmp_path, "siesta", ".fdf", "SystemLabel T\n",
                    names=RunNames("T", "05_transmission"),
                    calculation="transport", program="tbtrans")
    assert '_program_bin="$(command -v tbtrans ' in text, \
        "the run script does not start tbtrans"
    assert "command -v siesta " not in text
    head = text[text.index("===== molbuilder run script"):]
    head = head[:head.index('echo "========')]
    rows = re.findall(r'echo "  (\S[^"]*?) : ', head)
    assert "TBtrans version" in [r.rstrip() for r in rows], rows
    colons = {len(r) for r in rows}
    assert len(colons) == 1, f"the banner's labels do not line up: {rows}"


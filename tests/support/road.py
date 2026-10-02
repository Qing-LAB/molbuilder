"""The road a test drives -- `jobset init`, `prep`, `launch` -- and the two
things a test may put beside it: a machine whose record names queues, with a
scheduler that queues nothing, and a run that has already happened.

WHY THIS FILE EXISTS.  A test drives the designed workflow (user, 2026-09-23:
*"tests should be using our established jobset workflow, unless you have a
strong reason to focus on api"*).  Several files grew their own copy of the
same steps -- an `init` of H2, a stub `sbatch` -- and a copy is free to drift
from the road it imitates.  The steps live here once.

* :func:`jobset` -- the verbs, as a person types them;
* :func:`each_is_taken` -- every command an output prints, typed back as
  printed (`job-system.md` § 5.3: what molbuilder prints, you can type);
* :func:`describe_h2` -- `jobset init` of a held H2 in a box, SIESTA, the
  shipped `publishable` ladder (coarse, medium; tight disabled) -- or
  PySCF's own;
* :func:`a_queue_that_answers` -- a machine record naming queues, and an
  `sbatch` on PATH that queues nothing: it writes down every call -- where
  it was made and what it said -- answers ``--test-only`` the way Sol's
  did, and otherwise gives a job id;
* :func:`sbatch_line` -- the `sbatch` line a launch showed, as its words;
* :func:`gpus_given` -- the GPUs a machine hands a job, with an
  `nvidia-smi` that knows them;
* :func:`strip_preamble_activation` -- a generated run script, runnable in
  a bare shell: its preamble and environment activation cut out;
* :func:`a_finished_run` -- the measured H2 relaxation
  (``tests/fixtures/siesta_relax``, its README says what it pins), put where
  a run of the stage would have left it: the one thing the road cannot make
  without an engine.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path

import numpy as np

#: The measured relaxation a finished run stands on.
RELAX = (Path(__file__).resolve().parent.parent / "fixtures" / "siesta_relax"
         / "01_relax" / "run-0")

#: Sol's own answer to `sbatch --test-only`, verbatim (2026-08-27).
SOL_PREDICTION = ("sbatch: Job 62266174 to start at 2026-08-27T11:22:03 a "
                  "using 4 processors on nodes sc078 in partition htc")


def jobset(*args):
    """`molbuilder jobset <args>`, as typed."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def printed_commands(output: str):
    """Every ``molbuilder jobset`` command line an output prints, as a shell
    splits it -- a ``#`` comment cut off, as bash does -- without the
    program's two words."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("molbuilder jobset "):
            yield shlex.split(line, comments=True)[2:]


def each_is_taken(output: str) -> int:
    """Type back every command ``output`` prints, from where the test
    stands, and assert each is taken -- a launch only planned
    (``--dry-run``), so nothing is sent.  Returns how many there were."""
    printed = list(printed_commands(output))
    for words in printed:
        got = jobset(*words, *(["--dry-run"] if words[0] == "launch" else []))
        assert got.exit_code == 0, (words, got.output)
    return len(printed)


def describe_h2(tmp_path, monkeypatch, *, shape: str = "hierarchical",
                name: str = "H2", calculation: str = "optimization",
                engine: str = "siesta") -> Path:
    """`jobset init` on a held H2 in a box -- the bundle, at
    ``<projects>/P/<calculation>/<name>``: a SIESTA optimization's shipped
    `publishable` ladder, a vibration's own (`relax`, `freq`), or PySCF's
    own ladder."""
    from conftest import write_pseudos
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    write_pseudos(tree / "pseudopotential", ["H"])
    StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]]),
                  regions={"frozen_atoms": [0]},
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3),
        tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    siesta = engine == "siesta"
    r = jobset("init", "--structure", "P/structure/h2.xyz",
               "--bundle", f"P/{calculation}/{name}", "--engine", engine,
               "--shape", shape, "--name", name,
               "--calculation", calculation,
               *(("--stage-strategy", "publishable")
                 if calculation == "optimization" and siesta else ()),
               *(("--psml-lib", "pseudopotential") if siesta else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / calculation / name
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    return bundle


def a_queue_that_answers(tmp_path, monkeypatch, domains, **record) -> Path:
    """This machine's record names ``domains`` (`scheduler.Domain` rows) --
    and ``record``'s other fields, when given -- and the `sbatch` first on
    PATH queues nothing.  Returns the file each call is written to, one line
    each: ``<where it was run> | <its arguments>``."""
    from conftest import write_machine_record
    write_machine_record(scheduler="slurm", domains=list(domains), **record)
    bin_dir = tmp_path / "scheduler-bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch-calls.log"
    f = bin_dir / "sbatch"
    f.write_text(
        "#!/bin/sh\n"
        f'echo "$(pwd) | $*" >> "{calls}"\n'
        'case " $* " in\n'
        f'  *" --test-only "*) echo "{SOL_PREDICTION}" >&2; exit 0 ;;\n'
        "esac\n"
        'echo "Submitted batch job 4242"\n')
    f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return calls


def gpus_given(tmp_path, monkeypatch, visible: str) -> None:
    """The GPUs a machine hands a job: ``CUDA_VISIBLE_DEVICES`` as a
    scheduler sets it, and an `nvidia-smi` first on PATH that knows each of
    them -- listed by ``-L``, and a PCI address for ``--id`` that no real
    device has, so the NUMA lookup reads *unknown* on any box.  The run
    script then asks about its GPUs as it would on the node, wherever the
    test runs."""
    n = len([g for g in visible.split(",") if g])
    bin_dir = tmp_path / "gpu-bin"
    bin_dir.mkdir(exist_ok=True)
    f = bin_dir / "nvidia-smi"
    listing = "".join(f"GPU {i}: Stand-in GPU (UUID: GPU-stand-in-{i})\\n"
                      for i in range(n))
    f.write_text(
        "#!/bin/sh\n"
        'case " $* " in\n'
        f'  *" -L "*) printf "{listing}" ;;\n'
        '  *) echo "00000000:FE:1F.7" ;;\n'
        "esac\n")
    f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)


def strip_preamble_activation(text: str) -> str:
    """Remove the baked preamble + conda-activation block (script-
    execution blocks 3-4) from a rendered wrapper so the behaviour
    tests can EXECUTE it in a bare CI shell.  ``module load mamba`` /
    ``source activate`` exit 127 without an HPC module system or conda;
    under ``set -e`` that aborts the wrapper before the cold block ever
    runs.  ``_log`` is defined earlier (block 2) so the cold block's
    logging survives the strip -- the wrapper is RUN, here, as the
    person's machine would run it, short of entering an environment."""
    pre = text.find("# --- Baked preamble")
    assert pre >= 0, "baked-preamble marker not found in wrapper"
    # Since U10 the bootstrap AND the post-activation state dump each sit
    # inside a help guard (if [ "$_mb_help" = "0" ]); the cut must span
    # from the FIRST guard's opener through the SECOND guard's close, or
    # the truncated wrapper keeps an unopened fi.
    start = text.rfind('if [ "$_mb_help" = "0" ]; then', 0, pre)
    assert start >= 0, "help-guard opener not found before the preamble"
    em = text.find("which python:", pre)
    assert em >= 0, "activation conda-dump end marker not found"
    close = text.find("\nfi\n", em)
    assert close >= 0, "post-activation guard close not found"
    # ``set -u`` is restored explicitly: the real wrapper disables
    # nounset around the activation (NVCC_PREPEND_FLAGS) and re-enables
    # it INSIDE the region cut here, so without this line the stripped
    # harness runs everything after the preamble with nounset off --
    # which is how the unbraced-$_warm_label death (redo NEW-1) stayed
    # invisible to every executed test in this file.
    return (
        text[:start]
        + "# preamble + activation stripped for CI (no conda here).\n"
        + "set -u\n"
        + text[close + 4:]
    )


def sbatch_line(output: str):
    """The `sbatch` line a launch showed -- in its question, or as a dry
    run's ``WOULD run`` -- as its arguments."""
    for ln in output.splitlines():
        words = ln.split()
        if "sbatch" in words:
            words = words[words.index("sbatch"):]
            return [w for w in words if not w.startswith("[")]
    raise AssertionError(f"no sbatch line shown:\n{output}")


def calls_made(calls: Path):
    """The `sbatch` calls written down so far, as ``(where, argv)``."""
    if not calls.is_file():
        return []
    out = []
    for line in calls.read_text().splitlines():
        where, _sep, argv = line.partition(" | ")
        out.append((Path(where), argv.split()))
    return out


def a_finished_run(where: Path, *, stem: str = "H2_01_coarse",
                   rc: int = 0, tolerance: str = "0.0100",
                   concluded: bool = True) -> None:
    """A run of the stage ``stem`` names, ended, in ``where``: the measured
    relaxation's output and geometry -- and, when it ``concluded``, its
    conclusion marker.  A failed one (``rc`` nonzero) died partway, so its
    output stops before the engine's end -- the output's own ending is the
    strongest evidence of how a run ended (`running-a-job.md` § 4.2)."""
    text = (RELAX / "H2_01_relax-run0.out").read_text().replace(
        "Force tolerance                             =     0.0100 eV/Ang",
        f"Force tolerance                             =     {tolerance} "
        f"eV/Ang")
    (where / f"{stem}-run0.out").write_text(
        text if rc == 0 else text[: len(text) // 3])
    shutil.copy2(RELAX / "H2.XV", where / "H2.XV")
    if concluded:
        (where / f"{stem}-run0.concluded").write_text(
            f"rc={rc} at Thu Sep 24 02:38:51 PM MST 2026\n")

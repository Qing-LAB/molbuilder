"""Regression test for the 2026-06-14 ``--cold`` / ``--from-scratch``
flag on the SIESTA + PySCF run wrappers.

User-visible contract (job-contracts.md § 4.1 -- a NAME SWEEP since
U17, 2026-08-12; the suffix list the sentence below used to carry was a
snapshot of one build, and a file nobody listed was a file --cold
walked past):

  * ``bash <name>.run.sh --cold`` NAMES everything matching the run's
    id -- minus what molbuilder itself wrote (deck, template, .psml,
    wrappers, molbuilder's logs) -- and **refuses**, changing nothing;
    ``--force`` then proceeds and the run overwrites them as it goes.
    SIESTA's ``DM.UseSaveDM`` / ``MD.UseSaveCG`` / ``MD.UseSaveXV``
    find nothing surviving, so the calc starts strictly from the .fdf
    coords + conditions.
  * **Nothing is moved or copied** *(user, 2026-08-18)*.  It moved the
    files into ``<basename>-restart-aside-<UTC>/`` until then, which
    left two mechanisms for preserving a state; keeping one is
    ``molbuilder checkpoint save`` and it is never automatic
    (`checkpointing.md` § 2).
  * Distinct from ``--force``: ``--force`` only resets the
    run-index sequence; the warm-start files stay on disk and the
    engine still loads them.  ``--cold`` is about the engine state.
  * Combinable with ``--force`` (cold + restart run-index) and
    ``--continue`` (cold = no-op when there is nothing to name,
    which is the typical case mid-run).
  * Idempotent: re-running with ``--cold`` on a directory that is
    already clean says so and proceeds.

Motivation (2026-06-14 BDT incident): stage 2 ran without the
frozen-atom constraints the user intended (separate UI bug).  The
resulting .DM/.XV/.CG were physically inconsistent with what the
user wanted; any subsequent run that warm-started from them would
inherit the contamination.  ``--cold`` lets the user re-run from a
known clean state without having to manually ``rm`` the files.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.runwrap import write_run_wrapper
from molbuilder.jobset.model import Resources


@pytest.fixture(autouse=True)
def _autosetup_minimal_config(tmp_path, monkeypatch):
    """Every wrapper render needs a ``script_generation.activation`` or
    the v2 wrapper-independence contract's refuse-to-emit guard
    (docs/execution/running-a-job.md § 5, landed 2026-06-25) rejects it.  Mirror
    test_runwrap.py's fixture: a cwd molbuilder.json with the canonical
    Sol defaults so write_run_wrapper can emit.  Without this the whole
    file fails with RuntimeConfigError ("activation is not set")."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / "molbuilder.json").write_text(json.dumps({
        "script_generation": {
            "preamble":   "module load mamba",
            "activation": "source activate",
        }
    }))
    # A PROBED MACHINE.  Since 2026-09-02 a rank count is read from a record
    # and nowhere else -- no probe of the box that happens to be running, no
    # fallback (`running-a-job.md` § 3.1, user: "so we are not guess at
    # all").  A wrapper cannot be rendered on an unprobed machine, so a
    # fixture that renders one must probe first, exactly as a person does:
    #     molbuilder jobset probe --write
    from molbuilder.scheduler import Environment as _Env, Topology as _Topo
    (tmp_path / "environment.json").write_text(
        _Env(scheduler="slurm",
             topology=_Topo(sockets=2, cores_per_socket=32)).to_json() + "\n")
    yield tmp_path


def _bind():
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary="/usr/bin/conda",
        conda_envs=frozenset(["molbuilder-siesta", "molbuilder-pySCF"]),
    ))


# The engine/conda stubs this suite needs are `conftest.py`'s
# `product_toolchain_is_the_suites_own` -- ONE home, because the hostile
# sweep that found this hole here found it in three more suites
# (2026-08-25).  It lived in this file for about an hour.

# --------------------------------------------------------------------- #
#  Cold-restart bash block: text shape                                   #
# --------------------------------------------------------------------- #


class TestColdFlagText:
    """Source-text guards: the cold flag must appear in the help, in the
    arg parser, and in a sweep that VISITS every warm-start file.

    *Visits*, not moves: since 2026-08-18 the sweep names what it found and
    refuses.  Which files it selects is unchanged and is what these check.
    """

    def _siesta_wrapper(self, tmp_path: Path) -> str:
        _bind()
        script = tmp_path / "myjob.fdf"
        script.write_text(
            "SystemLabel  myjob\nNumberOfAtoms 1\n%block AtomicCoordinatesAndAtomicSpecies\n"
            "0 0 0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n"
        )
        return write_run_wrapper(script, resources=Resources()).read_text()

    def _pyscf_wrapper(self, tmp_path: Path) -> str:
        _bind()
        script = tmp_path / "myjob.py"
        script.write_text("# fake\n")
        return write_run_wrapper(script, resources=Resources()).read_text()

    def test_siesta_cold_in_help(self, tmp_path):
        text = self._siesta_wrapper(tmp_path)
        assert "--cold" in text
        assert "--from-scratch" in text
        # Each flag listed in the usage line.
        assert "[--cold]" in text

    def test_pyscf_cold_in_help(self, tmp_path):
        text = self._pyscf_wrapper(tmp_path)
        assert "--cold" in text
        assert "--from-scratch" in text

    def test_siesta_cold_arg_parsed(self, tmp_path):
        text = self._siesta_wrapper(tmp_path)
        # The case-line in the shared parser AND the engine arg loop
        # both need to know the flag (the shared parser consumes it
        # before the engine loop sees ``$@``).
        assert "--cold|--from-scratch)" in text
        assert "_cold=1" in text

    def test_pyscf_cold_arg_parsed(self, tmp_path):
        text = self._pyscf_wrapper(tmp_path)
        assert "--cold|--from-scratch)" in text
        assert "_cold=1" in text

    def test_siesta_sweep_visits_all_warmstart_exts(self, tmp_path):
        text = self._siesta_wrapper(tmp_path)
        # Each of SIESTA's warm-start extensions must fall inside the
        # sweep's globs.  Missing one would leave a file the run then
        # overwrites without ever having named it.
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            assert f"myjob.{ext}" in text, (
                f"cold block missing myjob.{ext} glob"
            )

    def test_pyscf_sweep_visits_chk(self, tmp_path):
        text = self._pyscf_wrapper(tmp_path)
        assert "myjob.chk" in text


# --------------------------------------------------------------------- #
#  Cold-restart end-to-end: actually run the bash + check behaviour      #
# --------------------------------------------------------------------- #


def _strip_preamble_activation(text: str) -> str:
    """Remove the baked preamble + conda-activation block (script-
    execution blocks 3-4) from a rendered wrapper so the behaviour
    tests can EXECUTE it in a bare CI shell.  ``module load mamba`` /
    ``source activate`` exit 127 without an HPC module system or conda;
    under ``set -e`` that aborts the wrapper before the cold block ever
    runs.  ``_log`` is defined earlier (block 2) so the cold block's
    logging survives the strip.  Mirrors test_runwrap.py's executable-
    test stripping; the difference here is we KEEP ``_log``."""
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


def _truncated_siesta(tmp_path: Path, basename: str = "myjob") -> Path:
    """Build a SIESTA wrapper but truncate the bash BEFORE the
    actual ``mpirun siesta`` invocation so the script exits cleanly
    after the run-index + cold-restart logic.  Lets us exercise the
    cold block in CI without needing a real SIESTA install."""
    _bind()
    script = tmp_path / f"{basename}.fdf"
    script.write_text(
        "SystemLabel myjob\nNumberOfAtoms 1\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    wrapper = write_run_wrapper(script, resources=Resources())
    text = _strip_preamble_activation(wrapper.read_text())
    # Truncate at the first ``mpirun`` so the cold block has executed
    # but the SIESTA launch is skipped.  Append explicit exit 0 so
    # the test doesn't depend on what the wrapper would emit after.
    cut = text.find("mpirun")
    if cut < 0:
        cut = text.find("exec ")
    assert cut > 0, "no mpirun/exec in wrapper to truncate at"
    wrapper.write_text(text[:cut] + "\nexit 0\n")
    return wrapper


def _truncated_pyscf(tmp_path: Path, basename: str = "myjob") -> Path:
    """Build a PySCF wrapper truncated before the ``exec`` engine launch,
    same trick as :func:`_truncated_siesta`: the run-index, cold-restart
    and warm-start-detection logic all execute, the engine does not."""
    _bind()
    script = tmp_path / f"{basename}.py"
    script.write_text("# fake\n")
    wrapper = write_run_wrapper(script, resources=Resources())
    text = _strip_preamble_activation(wrapper.read_text())
    # "\nexec python", NOT "\nexec ": the log-redirect line
    # (``exec > >(tee ...)``) matches the bare form FIRST, and cutting
    # there skips the warm-start detection this harness exists to reach
    # (that vacuous cut let the first version of the fresh-dir pin pass
    # against the broken render).
    cut = text.find("\nexec python")
    assert cut > 0, "no exec-python launch line in the PySCF wrapper"
    wrapper.write_text(text[:cut] + "\nexit 0\n")
    return wrapper


def _cold(wrapper, tmp_path, *args):
    """Run ``--cold`` and return ``(proc, the set of files it NAMED)``.

    **``--cold`` reports what it would overwrite and refuses; ``--force``
    proceeds** *(user, 2026-08-18)*.  It moved those files into a timestamped
    aside directory until then, and that is what the tests below used to read.

    What they were protecting is unchanged and is why they are repointed
    rather than retired: **which files the name sweep selects.**  The sweep
    picks the same files; they are now named in a refusal instead of moved,
    so the report is where the selection is visible.
    """
    proc = subprocess.run(
        ["bash", str(wrapper), "--cold", *args], cwd=tmp_path,
        capture_output=True, text=True, timeout=20,
        # U10's launch-door gate: these tests exercise the SWEEP; the claim
        # is the deliberate manual door.
        env={**os.environ, "MB_LAUNCHED_BY": "manual"})
    named, collecting = set(), False
    for ln in proc.stderr.splitlines():
        if "would OVERWRITE prior state:" in ln:
            collecting = True
            continue
        if collecting:
            m = re.match(r"^\[molbuilder\]     (\S.*)$", ln)
            if m:
                named.add(m.group(1))
            else:
                collecting = False
    return proc, named


def _no_aside(tmp_path) -> bool:
    """Nothing was moved anywhere.  The launcher keeps no copies -- keeping a
    state is ``molbuilder checkpoint save`` and it is never automatic."""
    return not list(tmp_path.glob("*-restart-aside-*"))


def _has_bash() -> bool:
    return shutil.which("bash") is not None


@pytest.mark.skipif(not _has_bash(), reason="bash not available")
class TestPyscfFreshDirectorySurvives:
    """Redo NEW-1 (2026-08-12, introduced 8981376a): the warm-start test
    emitted ``[ -e "$_warm_label_optimized.xyz" ]`` -- four of PySCF's
    five warm suffixes start with ``_``, so the shell parsed the whole
    thing as ONE variable name, unbound under ``set -u``, and EVERY
    fresh-directory run died before launch.  A ``.chk`` on disk
    short-circuits the ``||`` chain and hides it, which is why only the
    fresh directory -- the most common state there is -- was the death
    scenario, and why this pin plants NOTHING."""

    def test_fresh_directory_reaches_the_launch_line(self, tmp_path):
        wrapper = _truncated_pyscf(tmp_path)
        proc = subprocess.run(
            ["bash", str(wrapper)],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
            env={**os.environ, "MB_LAUNCHED_BY": "manual"},
        )
        assert "unbound variable" not in proc.stderr, proc.stderr
        assert proc.returncode == 0, (
            f"wrapper exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )

    def test_no_unbraced_warm_label_concatenation_renders(self, tmp_path):
        """The render-side half: no ``$_warm_label`` immediately followed
        by a name character may appear anywhere in either engine's
        wrapper -- braces or a ``.`` must terminate the expansion."""
        for make in ("myjob.py", "myjob.fdf"):
            d = tmp_path / make.replace(".", "_")
            d.mkdir()
            script = d / make
            script.write_text("# fake\n" if make.endswith(".py") else
                              "SystemLabel myjob\n")
            _bind()
            text = write_run_wrapper(script, resources=Resources()).read_text()
            assert not re.search(r"\$_warm_label[A-Za-z0-9_]", text), (
                f"{make}: unbraced $_warm_label concatenation renders"
            )


def _bind_gpu():
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary="/usr/bin/conda",
        conda_envs=frozenset(["molbuilder-siesta", "molbuilder-siesta-gpu"]),
    ))


def _gpu_wrapper(tmp_path: Path, fdf_text: str) -> Path:
    """A GPU-mode wrapper, stripped for bare-shell execution."""
    _bind_gpu()
    fdf = tmp_path / "myjob.fdf"
    fdf.write_text(fdf_text)
    wrapper = write_run_wrapper(fdf, resources=Resources())
    wrapper.write_text(_strip_preamble_activation(wrapper.read_text()))
    return wrapper


def _dry(wrapper: Path, tmp_path: Path, *args: str):
    """--dry-run the wrapper with every rank/OMP env override scrubbed,
    so the resolution under test is the FLAG chain, not this shell's."""
    env = {**os.environ, "MB_LAUNCHED_BY": "manual"}
    for k in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "MB_NP",
              "SLURM_NTASKS", "PBS_NP", "MOLBUILDER_MPI_NP",
              "MOLBUILDER_OMP_NUM_THREADS"):
        env.pop(k, None)
    return subprocess.run(["bash", str(wrapper), "--dry-run", *args],
                          cwd=tmp_path, capture_output=True, text=True,
                          timeout=30, env=env)


_GPU_FDF = "SystemLabel myjob\nNumberOfAtoms 444\nDiag.ELPA.GPU .true.\n"


@pytest.mark.skipif(not _has_bash(), reason="bash not available")
class TestTrialLabelledCold:
    """G2 I-list (2026-08-13): a TRIAL's deck carries the coordinate-
    qualified label (``JOB-G1K4C6``), and its warm files are keyed on it
    (project-layout § 2.3.2 -- the relabelling exists so trial warm state
    never collides with the run's).  The name sweep must move THOSE files
    on --cold; every prior cold test used a plain label."""

    def test_cold_moves_trial_labelled_warm_files(self, tmp_path):
        _bind()
        script = tmp_path / "JOB-G1K4C6.fdf"
        script.write_text(
            "SystemLabel JOB-G1K4C6\nNumberOfAtoms 1\n"
            "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
            "%endblock AtomicCoordinatesAndAtomicSpecies\n")
        wrapper = write_run_wrapper(script, resources=Resources())
        text = _strip_preamble_activation(wrapper.read_text())
        cut = text.find("mpirun")
        if cut < 0:
            cut = text.find("\nexec ")
        assert cut > 0
        wrapper.write_text(text[:cut] + "\nexit 0\n")
        for ext in ("DM", "XV", "CG"):
            (tmp_path / f"JOB-G1K4C6.{ext}").write_text(f"trial {ext}")
        proc, named = _cold(wrapper, tmp_path)
        assert proc.returncode == 1, proc.stderr
        for ext in ("DM", "XV", "CG"):
            assert f"JOB-G1K4C6.{ext}" in named, (
                f"trial-labelled JOB-G1K4C6.{ext} was not named by --cold")
        assert _no_aside(tmp_path)


@pytest.mark.skipif(not _has_bash(), reason="bash not available")
class TestGpuFlagPrecedence:
    """Redo F6 (2026-08-12, runtime-proven): ``-np 9 --no-mps`` ran 2
    ranks -- the MPS arm's re-resolve chain read MB_NP/SLURM (unset) and
    fell through to the regime policy default, clobbering the flag the
    comment claimed still won.  The fix: explicit-flag markers guard the
    re-resolve, and the auto-OMP width derives from the EFFECTIVE rank
    count in a post-parse epilogue (inside the loop it depended on flag
    order)."""

    def test_np_flag_survives_no_mps_in_both_orders(self, tmp_path):
        wrapper = _gpu_wrapper(tmp_path, _GPU_FDF)
        for order in (("-np", "9", "--no-mps"), ("--no-mps", "-np", "9")):
            proc = _dry(wrapper, tmp_path, *order)
            out = proc.stdout + proc.stderr
            assert proc.returncode == 0, out[-800:]
            assert re.search(r"mpirun -np 9\b", out), (
                f"{order}: flag-set rank count lost:\n{out[-800:]}"
            )

    def test_bare_no_mps_takes_the_no_mps_policy_count(self, tmp_path):
        wrapper = _gpu_wrapper(tmp_path, _GPU_FDF)
        proc = _dry(wrapper, tmp_path, "--no-mps")
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out[-800:]
        m = re.search(r"mpirun -np (\d+)\b", out)
        assert m, out[-800:]
        # 2 on dual-socket / >=16-core-socket boxes, 1 on small ones --
        # never the 4-rank MPS-regime default the original R9/F6 bug
        # kept after the regime flipped.
        assert int(m.group(1)) in (1, 2), out[-800:]

    def test_auto_omp_width_divides_by_the_effective_count(self, tmp_path):
        """9 ranks x the 2-rank width oversubscribed the box; the
        epilogue's invariant is ranks x width <= physical cores."""
        wrapper = _gpu_wrapper(tmp_path, _GPU_FDF)
        proc = _dry(wrapper, tmp_path, "--no-mps", "-np", "9")
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out[-800:]
        cores = re.search(r"detected phys_cores=(\d+)", out)
        width = re.search(r"package:PE=(\d+)", out)
        assert cores and width, out[-800:]
        assert 9 * int(width.group(1)) <= int(cores.group(1)), (
            f"9 ranks x PE={width.group(1)} oversubscribes "
            f"{cores.group(1)} cores:\n{out[-800:]}"
        )

    def test_dry_run_names_sources_and_flags_a_stale_header(self, tmp_path):
        """User design 2026-08-13: dry-run is the pre-submission
        inspection.  It must say WHERE each number came from, and -- run
        locally with a sibling .sbatch -- read the header back and WARN
        when the header's -n would override the resolved count once
        SLURM_NTASKS exists (the header always wins inside a job)."""
        _bind_gpu()
        (tmp_path / "molbuilder.json").write_text(json.dumps({
            "script_generation": {"activation": "source activate"},
            "scheduler": {"kind": "slurm",
                          "directives": {"partition": "general",
                                         "qos": "public"},
                          "gpu": {"default_type": "a100"}},
        }))
        fdf = tmp_path / "myjob.fdf"
        fdf.write_text(_GPU_FDF)
        wrapper = write_run_wrapper(fdf, resources=Resources())   # emits myjob.sbatch too (-n 1)
        assert "#SBATCH -n 1" in (tmp_path / "myjob.sbatch").read_text()
        wrapper.write_text(_strip_preamble_activation(wrapper.read_text()))
        proc = _dry(wrapper, tmp_path, "-np", "3")
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out[-800:]
        assert "(source: -np flag)" in out, out[-800:]
        assert "sbatch header:" in out and "-n 1" in out, out[-800:]
        assert "WARNING" in out and "OVERRIDE" in out, out[-800:]
        assert "sbatch -n 3 myjob.sbatch" in out, out[-800:]

    def test_mps_gate_is_any_shared_gpu(self, tmp_path):
        """User decision 2026-08-13: MPS starts whenever ranks exceed
        GPUs -- the floor-division gate missed the uneven split (3 ranks
        over 2 GPUs shared GPU0 by time-slicing, without the funnel)."""
        wrapper = _gpu_wrapper(tmp_path, _GPU_FDF)
        text = wrapper.read_text()
        assert '[ "$_mpi_np" -gt "${_ngpu:-0}" ]' in text
        assert '[ "$_ranks_per_gpu" -ge 2 ]' not in text

    def test_gpu_fdf_without_numberofatoms_still_launches(self, tmp_path):
        """The rank-policy function must return 0: without the n_atoms
        clamp (NumberOfAtoms is OPTIONAL in SIESTA) its body ended on a
        failed ``[ ... ] && ...`` guard, and under ``set -e`` the first
        bare call killed every such wrapper pre-launch (found by the F6
        probe, 2026-08-12)."""
        wrapper = _gpu_wrapper(
            tmp_path, "SystemLabel myjob\nDiag.ELPA.GPU .true.\n")
        proc = _dry(wrapper, tmp_path)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out[-800:]
        assert "resolved launch" in out, out[-800:]


@pytest.mark.skipif(not _has_bash(), reason="bash not available")
class TestColdBehaviour:
    """Exec the wrapper with --cold and verify warm-start files
    actually move."""

    def test_cold_moves_dm_xv_cg_aside(self, tmp_path):
        wrapper = _truncated_siesta(tmp_path)
        # Plant warm-start files the cold block should move.
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            (tmp_path / f"myjob.{ext}").write_text(f"fake {ext}")
        proc, named = _cold(wrapper, tmp_path)
        assert proc.returncode == 1, (
            f"--cold with prior state must refuse\nstderr:\n{proc.stderr}")
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            assert f"myjob.{ext}" in named, (
                f"myjob.{ext} would be overwritten and was not named")
            assert (tmp_path / f"myjob.{ext}").exists(), (
                f"myjob.{ext} was touched -- a refusal changes nothing")
        assert _no_aside(tmp_path)

    def test_cold_with_no_warmstart_files_is_noop(self, tmp_path):
        """Idempotent: ``--cold`` on a clean directory must not
        fail and must NOT create an empty aside dir."""
        wrapper = _truncated_siesta(tmp_path)
        proc = subprocess.run(
            ["bash", str(wrapper), "--cold"],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
            # U10's launch-door gate: these tests exercise the SWEEP;
            # the claim is the deliberate manual door
            env={**os.environ, "MB_LAUNCHED_BY": "manual"},
        )
        assert proc.returncode == 0, (
            f"wrapper exited {proc.returncode}\nstderr:\n{proc.stderr}"
        )
        # No aside dir should exist (nothing to move).
        asides = list(tmp_path.glob("myjob-restart-aside-*"))
        assert not asides, (
            f"empty aside dir should NOT have been created; got {asides}"
        )
        assert "already a clean start" in proc.stderr or \
               "already a clean start" in proc.stdout

    def test_no_cold_leaves_warmstart_in_place(self, tmp_path):
        """Default behaviour (no --cold) MUST NOT move the files.
        Pins that the cold logic is gated by the flag and doesn't
        run unconditionally."""
        wrapper = _truncated_siesta(tmp_path)
        for ext in ("DM", "CG", "XV"):
            (tmp_path / f"myjob.{ext}").write_text(f"fake {ext}")
        proc = subprocess.run(
            ["bash", str(wrapper)],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
            # U10's launch-door gate: these tests exercise the SWEEP;
            # the claim is the deliberate manual door
            env={**os.environ, "MB_LAUNCHED_BY": "manual"},
        )
        assert proc.returncode == 0
        # Originals remain.
        for ext in ("DM", "CG", "XV"):
            assert (tmp_path / f"myjob.{ext}").exists(), (
                f"myjob.{ext} should have stayed put without --cold"
            )
        # No aside dir.
        asides = list(tmp_path.glob("myjob-restart-aside-*"))
        assert not asides


@pytest.mark.skipif(not _has_bash(), reason="bash not available")
class TestColdBehaviourSystemLabelMismatch:
    """Pins the 2026-06-14 BDT incident's actual root cause: the
    SIESTA SystemLabel inside the .fdf is OFTEN different from the
    .fdf's filename basename.  An .fdf named ``foo-stage2.fdf``
    whose ``SystemLabel`` line says ``foo`` writes ``foo.DM`` /
    ``foo.XV`` / ``foo.CG`` -- NOT ``foo-stage2.DM`` etc.

    The first ``--cold`` ship globbed only against the wrapper
    basename and missed every staged-relaxation project (basename
    ``foo-stage2`` vs SystemLabel ``foo``); the BDT-stage-2
    contamination went uncleaned and re-contaminated stage 3.

    The fix reads ``SystemLabel`` from the .fdf at runtime via awk
    and globs against both the SystemLabel-keyed AND wrapper-
    basename-keyed patterns.  These tests reproduce the actual BDT
    file layout (different label vs filename) and verify the move.
    """

    def _truncated_siesta_with_label(
            self,
            tmp_path: Path,
            *,
            fdf_basename: str,
            system_label: str,
    ) -> Path:
        """Build a wrapper for ``<fdf_basename>.fdf`` whose
        SystemLabel line points at ``system_label`` (a DIFFERENT
        string from the basename when they differ).  Truncates the
        bash at the first ``mpirun`` so the cold block fires but
        SIESTA never launches."""
        _bind()
        script = tmp_path / f"{fdf_basename}.fdf"
        script.write_text(
            f"SystemLabel {system_label}\n"
            f"NumberOfAtoms 1\n"
            f"%block AtomicCoordinatesAndAtomicSpecies\n"
            f"0 0 0 1\n"
            f"%endblock AtomicCoordinatesAndAtomicSpecies\n"
        )
        wrapper = write_run_wrapper(script, resources=Resources())
        text = _strip_preamble_activation(wrapper.read_text())
        # Truncate AFTER the closing banner separator (which prints
        # the Mode + Constraints lines we want to observe).  The
        # naive ``find("mpirun")`` matches the FIRST occurrence which
        # is usually a comment ("Default to mpirun (safe...)") --
        # the banner sits between that comment and the actual
        # launch line, so cutting at the comment kills the banner.
        end = text.find('echo "================================')
        if end < 0:
            # Fallback for runs without the closing-banner echo
            # (PySCF wrapper uses a slightly shorter separator).
            end = text.find('mpirun')
        end = text.find("\n", end) + 1
        wrapper.write_text(text[:end] + "\nexit 0\n")
        return wrapper

    def test_systemlabel_keyed_warmstart_files_move(self, tmp_path):
        """The BDT-stage-2 case: .fdf is ``siesta-foo-stage2.fdf``
        with ``SystemLabel siesta-foo``.  SIESTA wrote
        ``siesta-foo.DM`` / ``.XV`` / ``.CG``.  --cold MUST move
        them aside even though the basename doesn't match."""
        wrapper = self._truncated_siesta_with_label(
            tmp_path,
            fdf_basename="siesta-foo-stage2",
            system_label="siesta-foo",
        )
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            (tmp_path / f"siesta-foo.{ext}").write_text(f"fake {ext}")

        proc, named = _cold(wrapper, tmp_path)
        assert proc.returncode == 1, proc.stderr
        # SystemLabel-keyed names must be NAMED -- pre-fix the glob only
        # matched siesta-foo-stage2.{ext} (basename-keyed) and silently
        # missed these, which under the old behaviour meant they were not
        # moved and under this one means they are overwritten unannounced.
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            assert f"siesta-foo.{ext}" in named, (
                f"siesta-foo.{ext} (SystemLabel-keyed) was not named")
        assert _no_aside(tmp_path)

    def test_both_systemlabel_and_basename_files_move(self, tmp_path):
        """Defensive: if BOTH naming patterns exist (a project
        that's been through a SystemLabel rename), --cold should
        move ALL of them.  The glob covers both patterns."""
        wrapper = self._truncated_siesta_with_label(
            tmp_path,
            fdf_basename="job-stage3",
            system_label="job",
        )
        # SystemLabel-keyed files (the "new" naming):
        (tmp_path / "job.DM").write_text("fake")
        (tmp_path / "job.XV").write_text("fake")
        # Basename-keyed files (legacy / external):
        (tmp_path / "job-stage3.DM").write_text("fake")
        (tmp_path / "job-stage3.XV").write_text("fake")

        proc, named = _cold(wrapper, tmp_path)
        assert proc.returncode == 1, proc.stderr
        # All four -- both keyings -- must be named.
        for name in ("job.DM", "job.XV", "job-stage3.DM", "job-stage3.XV"):
            assert name in named, f"{name} was not named by --cold"
        assert _no_aside(tmp_path)

    def test_quoted_systemlabel_stripped_in_glob(self, tmp_path):
        """SIESTA accepts ``SystemLabel "my job"`` (quoted, with
        embedded space).  The wrapper's awk must strip the surrounding
        quotes BEFORE using the label as a glob anchor -- otherwise
        ``"my`` becomes the prefix and the warm-start files are
        missed.  Embedded spaces are a separate concern (the wrapper's
        SAFE_WRAPPER_NAME_RE rejects them at emission time) so we
        only test the quote-strip here."""
        wrapper = self._truncated_siesta_with_label(
            tmp_path,
            fdf_basename="job-stage1",
            system_label='"foo"',  # quoted label
        )
        # Plant warm-start files keyed on the UNQUOTED label.
        (tmp_path / "foo.DM").write_text("fake")
        (tmp_path / "foo.XV").write_text("fake")

        proc, named = _cold(wrapper, tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "foo.DM" in named, (
            "quoted SystemLabel must be quote-stripped before globbing "
            "-- pre-fix the glob looked for ``\"foo\".DM`` and missed "
            "the unquoted filename SIESTA actually wrote, so the file "
            "would be overwritten with no warning."
        )
        assert "foo.XV" in named

    def test_lowercase_systemlabel_keyword_still_matched(
            self, tmp_path):
        """Replaces gawk's ``IGNORECASE`` with ``tolower($1) ==
        "systemlabel"`` for awk portability.  Pin that a .fdf
        whose ``systemlabel`` keyword is lowercase (or any other
        case) still drives the glob.  Pre-fix mawk / BSD awk
        silently ignored IGNORECASE and the glob fell back to
        wrapper basename -- the exact bug the SystemLabel
        extraction was added to fix."""
        # Author the .fdf manually so we can pick the keyword case.
        _bind()
        script = tmp_path / "job-stage1.fdf"
        script.write_text(
            "systemlabel foo\n"   # lowercase keyword
            "NumberOfAtoms 1\n"
            "%block AtomicCoordinatesAndAtomicSpecies\n"
            "0 0 0 1\n"
            "%endblock AtomicCoordinatesAndAtomicSpecies\n"
        )
        wrapper = write_run_wrapper(script, resources=Resources())
        text = _strip_preamble_activation(wrapper.read_text())
        end = text.find('echo "================================')
        if end < 0:
            end = text.find("mpirun")
        end = text.find("\n", end) + 1
        wrapper.write_text(text[:end] + "\nexit 0\n")
        (tmp_path / "foo.DM").write_text("fake")

        proc, named = _cold(wrapper, tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "foo.DM" in named, (
            "lowercase ``systemlabel`` keyword must still be matched "
            "by the awk (uses tolower($1) for portability across "
            "gawk / mawk / BSD awk)"
        )

    def test_status_banner_detects_systemlabel_keyed_files(self, tmp_path):
        """When the user runs WITHOUT --cold but
        ``$SystemLabel.{DM,XV,CG}`` files are on disk, the status
        banner must report ``WARM-RESTART (silent; ...)`` so the
        user can see they need --cold.  Same SystemLabel-keyed
        detection logic; if the status block only checked the
        basename, this would silently report ``initial-run``."""
        wrapper = self._truncated_siesta_with_label(
            tmp_path,
            fdf_basename="bdt-stage2",
            system_label="bdt",
        )
        # Truncate AFTER the banner so we capture its output.
        text = wrapper.read_text()
        # Wrapper was already truncated at mpirun; banner is in
        # the env_prefix which runs before mpirun, so banner is
        # still in.
        (tmp_path / "bdt.DM").write_text("fake")

        proc = subprocess.run(
            ["bash", str(wrapper)],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
            # U10's launch-door gate: these tests exercise the SWEEP;
            # the claim is the deliberate manual door
            env={**os.environ, "MB_LAUNCHED_BY": "manual"},
        )
        assert proc.returncode == 0
        combined = proc.stdout + proc.stderr
        assert "WARM-RESTART" in combined, (
            "status banner must detect SystemLabel-keyed warm-start "
            "files when no --cold flag is passed.  Pre-fix the "
            "banner read basename-keyed only and reported "
            "``initial-run`` even with bdt.DM on disk.\n\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


class TestNameSweep:
    """U17: the sweep is BY NAME (job-contracts § 4.1), so completeness
    is by construction -- the cases a suffix list could never pass."""

    def test_a_file_no_list_ever_named_is_swept(self, tmp_path):
        """The defect the name sweep ends: an engine build writing a
        state file nobody enumerated.  Under the list this survived
        --cold and silently warmed the 'clean' run."""
        wrapper = _truncated_siesta(tmp_path)
        (tmp_path / "myjob.DM").write_text("x")
        (tmp_path / "myjob.NEWFANGLED_STATE").write_text("x")
        (tmp_path / "myjob.orbdata.v99").write_text("x")
        proc, named = _cold(wrapper, tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert {"myjob.DM", "myjob.NEWFANGLED_STATE",
                "myjob.orbdata.v99"} <= named
        assert _no_aside(tmp_path)

    def test_what_molbuilder_wrote_survives_the_sweep(self, tmp_path):
        """§ 4.1's exception: everything molbuilder wrote stays put, and
        since E-1 (2026-08-13) the bash exception list is DERIVED from
        ``identity.OUR_FILE_PATTERNS`` -- so the run-indexed HISTORY
        survives in every shape.  The hand list this replaces lacked
        ``*.out`` and the monitor/util/scf-timing logs, and its comment
        claimed prior outputs "survive by construction (hyphen-joined)"
        -- false for a FLAT STAGED calculation, whose
        ``myjob_01_coarse-run0.out`` matches the ``myjob_*`` glob:
        ``--cold`` on stage 2 counted stage 1's stdout and timing history
        as engine state."""
        wrapper = _truncated_siesta(tmp_path)
        (tmp_path / "myjob.template.toml").write_text("x")
        (tmp_path / "myjob.molwatch.log").write_text("x")
        (tmp_path / "myjob-run0.out").write_text("results")
        (tmp_path / "myjob.out").write_text("results")
        (tmp_path / "myjob_01_coarse-run0.out").write_text("stage 1 stdout")
        (tmp_path / "myjob-run0.scf-timing.log").write_text("timing")
        (tmp_path / "myjob.monitor.log").write_text("status")
        (tmp_path / "myjob.util.csv").write_text("samples")
        (tmp_path / "myjob.runwrap-20260813-000000.log").write_text("session")
        (tmp_path / "myjob.DM").write_text("state")
        proc, named = _cold(wrapper, tmp_path)
        # The engine state IS named -- that is the whole point of the run.
        assert proc.returncode == 1, proc.stderr
        assert "myjob.DM" in named
        for kept in ("myjob.fdf", "myjob.run.sh", "myjob.template.toml",
                     "myjob.molwatch.log", "myjob-run0.out", "myjob.out",
                     "myjob_01_coarse-run0.out",
                     "myjob-run0.scf-timing.log", "myjob.monitor.log",
                     "myjob.util.csv", "myjob.runwrap-20260813-000000.log"):
            assert kept not in named, (
                f"{kept} was named as prior engine state -- molbuilder's own "
                f"history is the § 4.1 exception and is not what --cold is "
                f"warning about")
            assert (tmp_path / kept).is_file(), f"{kept} was touched"
        # And the engine state is still on disk: a refusal changes nothing.
        assert (tmp_path / "myjob.DM").is_file()


def test_the_exception_is_anchored_on_the_id_not_widened_to_a_star():
    """§ 4.1's exception must name OUR files, not every file of that shape.

    ``--cold``'s "except what molbuilder wrote" list is derived from the one
    enumeration, ``identity.OUR_FILE_PATTERNS`` (E-1, 2026-08-13).  How it is
    READ is the thing this pins: each pattern's ``{label}`` becomes the run's
    actual id, never ``*``.

    **The widening was defended as harmless and was not.**  It read
    ``{label}`` -> ``*`` until 2026-08-17, on the argument that the sweep's own
    globs already anchor on the id — which says the widening cannot make the
    sweep visit MORE files, and says nothing about the exception matching more
    of them.  It held only while every pattern ended in a suffix nobody but
    molbuilder writes.  ``{label}.xyz`` joined the list on 2026-08-16 (so
    ``prep`` would stop calling a hand-over's input structure an engine
    leftover) and widened to ``*.xyz``, which claimed PySCF's
    ``<JOB>_optimized.xyz`` — warm state, and the whole reason ``--cold``
    exists.

    So this guards the CLASS rather than that one file: the next shared suffix
    added to ``OUR_FILE_PATTERNS`` for the other reader's sake must not quietly
    re-open it.  Two globs are exempt by design and named here — ``*.psml`` is
    element-named, not run-named, and the aside directories are what the sweep
    must never recurse into.
    """
    from molbuilder.runwrap import _cold_restart_block

    block = _cold_restart_block("myjob", engine="pyscf")
    line = [l for l in block.splitlines()
            if l.strip().endswith(") continue ;;")]
    assert len(line) == 1, "the exception case arm moved or multiplied"
    pats = line[0].strip()[:-len(") continue ;;")].split("|")

    bare = sorted(p for p in pats if p.startswith("*"))
    assert bare == ["*-restart-aside-*", "*.psml"], (
        f"an exception is anchored on nothing but a suffix: {bare}.\n"
        f"A pattern that starts with `*` protects every file of that shape "
        f"from --cold, including the engine output the sweep exists to move. "
        f"Anchor it on the run's id -- `\"$_warm_label\"` and the basename.")

    # ...and both spellings are present, because the sweep visits both.
    assert any("_warm_label" in p for p in pats)
    assert any(p.startswith("myjob") for p in pats)


# --------------------------------------------------------------------- #
#  The --help text is part of the contract, and it drifted              #
# --------------------------------------------------------------------- #

def _usage(engine: str) -> str:
    """The USAGE heredoc a generated wrapper prints for ``-h``.

    Rendered, not read out of the generator's source: the defect this guards
    was in the *emitted* text, and a test that reads the f-strings would have
    passed just as happily.
    """
    from molbuilder.runwrap import render_run_wrapper
    from molbuilder.jobset.model import Resources

    deck = "deck.fdf" if engine == "siesta" else "deck.py"
    text = render_run_wrapper(deck, resources=Resources(mpi_np=1), env="e")
    start = text.index("cat <<USAGE")
    return text[start:text.index("\nUSAGE\n", start)]


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_the_help_does_not_promise_a_backup_the_launcher_never_makes(engine):
    """**The one way this text can be wrong that costs a user their data.**

    ``--cold`` moved the prior state into a timestamped aside directory until
    2026-08-18, when it became a refusal (`job-contracts.md` § 4.1: *"the
    safety net for the other direction is a REFUSAL, not a copy"*).  SIESTA's
    help was not swept and went on promising the backup for a day -- while
    citing § 4.1, the section stating its opposite.  A reader who believed it
    would pass ``--cold --force`` expecting a copy and get an overwrite.

    Nothing read the generated help, which is why only one of the two engines
    was corrected.  This reads it.
    """
    text = _usage(engine)
    for promise in ("backup dir", "aside", "restart-aside",
                    "move EVERYTHING", "moves EVERYTHING"):
        assert promise not in text, (
            f"{engine}: --help offers `{promise}`; the launcher names the "
            f"files and refuses, and --force overwrites them "
            f"(job-contracts.md § 4.1)")


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_the_help_says_what_cold_actually_does(engine):
    """The other half: absence of the wrong claim is not presence of the
    right one.  A reader must be able to learn from ``-h`` that ``--cold``
    alone changes nothing, and that keeping a state is a separate verb."""
    text = _usage(engine)
    assert "REFUSES" in text and "--force then" in text, (
        f"{engine}: --help does not say that --cold refuses and --force "
        f"proceeds")
    assert "molbuilder\n                   checkpoint save" in text, (
        f"{engine}: --help does not point at the tool that keeps a state; "
        f"`checkpointing.md` § 2 -- it is never automatic, so the one place "
        f"a user is told about discarding state must name it")


def test_both_engines_get_that_entry_from_one_writer():
    """**The structural half, and the reason the drift was possible.**

    The sweep is engine-independent *by construction* -- it reads no list of
    extensions -- so the entry describing it is one fact.  Written out per
    engine, the two copies disagreed for a day.  Here the shared body is
    identical and only the EXAMPLE of what a run leaves behind differs.
    """
    import re as _re
    si, py = _usage("siesta"), _usage("pyscf")

    def entry(text):
        start = text.index("  --cold,")
        rest = text[start:]
        # the entry ends where the next flag begins
        m = _re.search(r"\n  -(?!-cold|-from-scratch)\S", rest)
        return rest[:m.start()] if m else rest

    a, b = entry(si), entry(py)
    assert ".DM/.CG/.XV among them" in a
    assert ".chk and _optimized.xyz among them" in b
    # everything except the example line is character-for-character shared
    strip = lambda t: [l for l in t.splitlines() if "among them" not in l]
    assert strip(a) == strip(b), (
        "the two engines' --cold entries have diverged again; the rule has "
        "one writer, `runwrap._cold_usage_entry`")

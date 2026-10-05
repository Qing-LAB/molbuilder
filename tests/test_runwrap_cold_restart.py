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
    """Every wrapper render reads the activation off the machine's record
    (`configuration.md` § 5 M-1) or refuses to emit.  Mirror
    test_runwrap.py's fixture: the config root holds this machine's
    record, with the canonical Sol activation, so write_run_wrapper can
    emit."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    # A PROBED MACHINE.  Since 2026-09-02 a rank count is read from a record
    # and nowhere else -- no probe of the box that happens to be running, no
    # fallback (`running-a-job.md` § 3.1, user: "so we are not guess at
    # all").  A wrapper cannot be rendered on an unprobed machine, so a
    # fixture that renders one must probe first, exactly as a person does:
    #     molbuilder jobset probe --write
    from molbuilder.scheduler import Environment as _Env, Topology as _Topo
    (tmp_path / "environment.json").write_text(
        _Env(scheduler="workstation",
             topology=_Topo(sockets=2, cores_per_socket=32),
             env_init={"preamble": "module load mamba",
                                "activation": "source activate"}).to_json()
        + "\n")
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
        return write_run_wrapper(script, resources=Resources(mpi_np=4, cpus_per_task=1)).read_text()

    def _pyscf_wrapper(self, tmp_path: Path) -> str:
        _bind()
        script = tmp_path / "myjob.py"
        script.write_text("# fake\n")
        return write_run_wrapper(script, resources=Resources(cpus_per_task=1)).read_text()

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


# The stripping lives in `tests/support/road.py`, beside the road the
# GPU contract's run-script cases drive.
from support.road import strip_preamble_activation as _strip_preamble_activation


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
    wrapper = write_run_wrapper(script, resources=Resources(mpi_np=4, cpus_per_task=1))
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
    """Build a PySCF wrapper truncated before the engine launch, same trick
    as :func:`_truncated_siesta`: the run-index, cold-restart and
    warm-start-detection logic all execute, the engine does not."""
    _bind()
    script = tmp_path / f"{basename}.py"
    script.write_text("# fake\n")
    wrapper = write_run_wrapper(script, resources=Resources(cpus_per_task=1))
    text = _strip_preamble_activation(wrapper.read_text())
    # The launch line, anchored on the `set +e` that immediately precedes
    # it.  It was `exec python` until 2026-09-08, when the PySCF branch
    # stopped `exec`-ing so that the shell could outlive the engine and
    # write the conclusion marker.  The anchor must stay PRECISE: a bare
    # "\nexec " matched the log-redirect line (``exec > >(tee ...)``) first
    # and cut before the warm-start detection this harness exists to reach,
    # which let an earlier version of the fresh-dir pin pass against a
    # broken render.  A bare "\npython " would be the same mistake again.
    cut = text.find("\nset +e\npython ")
    assert cut > 0, "no python launch line in the PySCF wrapper"
    wrapper.write_text(text[:cut] + "\nexit 0\n")
    return wrapper


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
            text = write_run_wrapper(
                script, resources=(Resources(cpus_per_task=1) if make.endswith(".py")
                                   else Resources(mpi_np=4, cpus_per_task=1))).read_text()
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
    # a GPU job as `resolve` writes one: told, with its count (`gpu.md` G5)
    wrapper = write_run_wrapper(fdf, resources=Resources(
        mpi_np=4, cpus_per_task=1, use_gpu=True, gres="gpu:1"))
    wrapper.write_text(_strip_preamble_activation(wrapper.read_text()))
    return wrapper


def _dry(wrapper: Path, tmp_path: Path, *args: str):
    """--dry-run the wrapper with every rank/OMP env override scrubbed,
    so the resolution under test is the FLAG chain, not this shell's."""
    env = {**os.environ, "MB_LAUNCHED_BY": "manual"}
    for k in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "MB_NP",
              "SLURM_NTASKS", "PBS_NP"):
        env.pop(k, None)
    return subprocess.run(["bash", str(wrapper), "--dry-run", *args],
                          cwd=tmp_path, capture_output=True, text=True,
                          timeout=30, env=env)


_GPU_FDF = "SystemLabel myjob\nNumberOfAtoms 444\nDiag.ELPA.GPU .true.\n"


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 9 tests here planted restart files or run records by hand for the cold
# start to move -- `TestTrialLabelledCold` and `TestNameSweep` whole, and
# some of the classes below (`process/testing.md` § 6).


@pytest.mark.skipif(not _has_bash(), reason="bash not available")
class TestGpuFlagPrecedence:
    """Redo F6 (2026-08-12, runtime-proven): ``-np 9 --no-mps`` ran 2
    ranks -- the MPS arm's re-resolve chain read MB_NP/SLURM (unset) and
    fell through to the regime policy default, clobbering the flag the
    comment claimed still won.  Since 2026-10-02 the MPS flags switch the
    daemon and nothing else -- there is no policy to fall through to -- and
    the flag-set count is still what runs, in either order."""

    def test_np_flag_survives_no_mps_in_both_orders(self, tmp_path):
        wrapper = _gpu_wrapper(tmp_path, _GPU_FDF)
        for order in (("-np", "9", "--no-mps"), ("--no-mps", "-np", "9")):
            proc = _dry(wrapper, tmp_path, *order)
            out = proc.stdout + proc.stderr
            assert proc.returncode == 0, out[-800:]
            assert re.search(r"mpirun -np 9\b", out), (
                f"{order}: flag-set rank count lost:\n{out[-800:]}"
            )

    # `test_auto_omp_width_divides_by_the_effective_count` stood here.  It
    # asked for 9 MPI ranks and asserted `9 * PE <= phys_cores`, so it could
    # only pass on a box with at least 9 cores.  On a 4-core machine it fails
    # for the machine, not for the code -- and it had been failing here for
    # exactly that reason, in every lane, as the one permanent red.
    #
    # Retired 2026-09-10.  A test that encodes the developer's hardware is not
    # a test of molbuilder.  The arithmetic it guarded (ranks x width never
    # exceeding what the box has) is the wrapper's own, asserted on the
    # RESOLVED numbers by the sizing tests that do not hardcode a rank count.

    def test_dry_run_names_sources_and_flags_a_stale_header(self, tmp_path):
        """User design 2026-08-13: dry-run is the pre-submission
        inspection.  It must say WHERE each number came from, and -- run
        locally with a sibling .sbatch -- read the header back and WARN
        when the header's -n would override the resolved count once
        SLURM_NTASKS exists (the header always wins inside a job)."""
        _bind_gpu()
        # A MACHINE WITH A QUEUE is a record listing one; the job names it
        # and states every value its header carries (`architecture.md`
        # § 5.2).
        from molbuilder.scheduler import Domain, Environment, Topology
        (tmp_path / "environment.json").write_text(Environment(
            scheduler="slurm", topology=Topology(sockets=2,
                                                 cores_per_socket=32),
            domains=[Domain(name="general", partition="general",
                            qos="public", max_time="1-00:00:00")],
            env_init={"activation": "source activate"}).to_json())
        fdf = tmp_path / "myjob.fdf"
        fdf.write_text(_GPU_FDF)
        # one GPU and one rank, stated -- a GPU job states its count
        # (`gpu.md` G5) and its ranks; -n 1
        wrapper = write_run_wrapper(fdf, resources=Resources(
            use_gpu=True, gres="gpu:1", mpi_np=1, cpus_per_task=1,
            domain="general",
            time="0-01:00:00", mem="8G"))
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
        """A GPU deck with no NumberOfAtoms (it is OPTIONAL in SIESTA) runs
        its dry run to the launch line.  Under ``set -e`` a guard that
        fails on such a deck kills the wrapper pre-launch -- the GPU rank
        policy's did, until the F6 probe found it (2026-08-12)."""
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


    def test_an_unusable_label_falls_back_AND_SAYS_SO(self):
        """A label that cannot be a filename falls back to the basename,
        **and the wrapper tells the person it did.**

        The check itself is old -- it was a `case "$_warm_label" in
        *[!A-Za-z0-9._-]*)` in the emitted bash, guarding a value the awk had
        just read out of the deck, and it printed a warning before falling
        back.  The read moved to prep on 2026-09-17 (`gpu.md` G7) and the
        guard came with it; **the warning did not**, and this test exists
        because that silence is the dangerous half.

        Silently sweeping under the wrong name means `--cold` finds nothing,
        reports nothing to clean, and the engine then warm-starts off files
        the person believed were gone.  A wrong answer with no signal.

        Mutation-tested: with the charset guard disabled nothing else in the
        suite fails, which is why this is here and not assumed covered.
        """
        from molbuilder.runwrap import _cold_restart_block
        block = _cold_restart_block("job_01_coarse", engine="siesta",
                                    label="my job")   # a space: unusable
        assert '_warm_label="job_01_coarse"' in block, (
            "an unusable label must fall back to the basename")
        assert "NOTE" in block and "--cold" in block, (
            "the fallback must be announced -- a silent one lets --cold "
            "report a clean directory that is not clean")

    def test_a_usable_label_says_nothing(self):
        """THE DISCRIMINATING HALF.  An ordinary label emits no notice, so
        the test above cannot pass on a wrapper that warns unconditionally."""
        from molbuilder.runwrap import _cold_restart_block
        block = _cold_restart_block("job_01_coarse", engine="siesta",
                                    label="bdt")
        assert '_warm_label="bdt"' in block
        assert "NOTE" not in block, (
            "a nameable label is not worth a line of output")

    # `test_quoted_systemlabel_stripped_in_glob` and
    # `test_lowercase_systemlabel_keyword_still_matched` stood here until
    # 2026-09-17.  Both pinned an AWK that read `SystemLabel` out of the deck
    # at LAUNCH -- one that a quoted value was unquoted before globbing, the
    # other that `tolower($1) == "systemlabel"` matched a lowercase keyword
    # where mawk/BSD awk ignore gawk's IGNORECASE.
    #
    # **The awk is deleted and the wrapper is TOLD its label** (`gpu.md` G7:
    # the value travels, the deck is not re-read for it).  `task.label` is a
    # validated basename, so it can be neither quoted nor oddly-cased, and no
    # keyword is matched here at all any more.  These tested a mechanism, and
    # the mechanism is gone.
    #
    # **The behaviours are not gone, and neither is their coverage.**  Both
    # still matter to `parse/fdf.system_label`, which `web/blueprints/watch.py`
    # runs over a directory a person points at -- no description to ask, the
    # deck is all there is.  They moved to `tests/parse/test_fdf.py`, which is
    # also where the reader finally got tests of its own: it was added on
    # 2026-09-17 with none.


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
    re-open it.  One glob is exempt by design and named here — ``*.psml`` is
    element-named, not run-named.  *(The aside directories were a second until
    2026-10-04: nothing has made one since 2026-08-18, plan D27.)*
    """
    from molbuilder.runwrap import _cold_restart_block

    block = _cold_restart_block("myjob", engine="pyscf", label="myjob")
    line = [l for l in block.splitlines()
            if l.strip().endswith(") continue ;;")]
    assert len(line) == 1, "the exception case arm moved or multiplied"
    pats = line[0].strip()[:-len(") continue ;;")].split("|")

    bare = sorted(p for p in pats if p.startswith("*"))
    assert bare == ["*.psml"], (
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
    text = render_run_wrapper(deck, resources=Resources(mpi_np=1,
                                                        cpus_per_task=1),
                              env="e")
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


# ---- warm state means CONTENT, not mere existence --------------------- #

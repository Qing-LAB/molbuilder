"""Regression test for the 2026-06-14 ``--cold`` / ``--from-scratch``
flag on the SIESTA + PySCF run wrappers.

User-visible contract:

  * ``bash <name>.run.sh --cold`` moves SIESTA's warm-start files
    (.DM, .CG, .XV, .LWF, .ZM, .Bonds, .PARTIAL, .EIG) -- and
    PySCF's .chk -- into ``<basename>-restart-aside-<UTC>/``
    BEFORE the engine launches.  SIESTA's ``DM.UseSaveDM`` /
    ``MD.UseSaveCG`` / ``MD.UseSaveXV`` then find nothing, so the
    calc starts strictly from the .fdf coords + conditions.
  * Distinct from ``--force``: ``--force`` only resets the
    run-index sequence; the warm-start files stay on disk and the
    engine still loads them.  ``--cold`` resets the engine state.
  * Combinable with ``--force`` (cold + restart run-index) and
    ``--continue`` (cold = no-op when there's nothing to move,
    which is the typical case mid-run).
  * Idempotent: re-running with ``--cold`` on a directory that's
    already clean emits a "no warm-start files to move" line and
    proceeds.

Motivation (2026-06-14 BDT incident): stage 2 ran without the
frozen-atom constraints the user intended (separate UI bug).  The
resulting .DM/.XV/.CG were physically inconsistent with what the
user wanted; any subsequent run that warm-started from them would
inherit the contamination.  ``--cold`` lets the user re-run from a
known clean state without having to manually ``rm`` the files.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.runwrap import write_run_wrapper


def _bind():
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary="/usr/bin/conda",
        conda_envs=frozenset(["molbuilder-siesta", "molbuilder-pySCF"]),
    ))


# --------------------------------------------------------------------- #
#  Cold-restart bash block: text shape                                   #
# --------------------------------------------------------------------- #


class TestColdFlagText:
    """Source-text guards: the cold flag must appear in the help,
    in the arg parser, and as a working aside-move block."""

    def _siesta_wrapper(self, tmp_path: Path) -> str:
        _bind()
        script = tmp_path / "myjob.fdf"
        script.write_text(
            "SystemLabel  myjob\nNumberOfAtoms 1\n%block AtomicCoordinatesAndAtomicSpecies\n"
            "0 0 0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n"
        )
        return write_run_wrapper(script).read_text()

    def _pyscf_wrapper(self, tmp_path: Path) -> str:
        _bind()
        script = tmp_path / "myjob.py"
        script.write_text("# fake\n")
        return write_run_wrapper(script).read_text()

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

    def test_siesta_aside_block_moves_all_warmstart_exts(self, tmp_path):
        text = self._siesta_wrapper(tmp_path)
        # Each of SIESTA's warm-start extensions must appear in the
        # move loop.  Missing one would leave a partial warm-start
        # that triggers an inconsistent restart.
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            assert f"myjob.{ext}" in text, (
                f"cold block missing myjob.{ext} glob"
            )

    def test_pyscf_aside_block_moves_chk(self, tmp_path):
        text = self._pyscf_wrapper(tmp_path)
        assert "myjob.chk" in text


# --------------------------------------------------------------------- #
#  Cold-restart end-to-end: actually run the bash + check behaviour      #
# --------------------------------------------------------------------- #


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
    wrapper = write_run_wrapper(script)
    text = wrapper.read_text()
    # Truncate at the first ``mpirun`` so the cold block has executed
    # but the SIESTA launch is skipped.  Append explicit exit 0 so
    # the test doesn't depend on what the wrapper would emit after.
    cut = text.find("mpirun")
    if cut < 0:
        cut = text.find("exec ")
    assert cut > 0, "no mpirun/exec in wrapper to truncate at"
    wrapper.write_text(text[:cut] + "\nexit 0\n")
    return wrapper


def _has_bash() -> bool:
    return shutil.which("bash") is not None


@pytest.mark.skipif(not _has_bash(), reason="bash not available")
class TestColdBehaviour:
    """Exec the wrapper with --cold and verify warm-start files
    actually move."""

    def test_cold_moves_dm_xv_cg_aside(self, tmp_path):
        wrapper = _truncated_siesta(tmp_path)
        # Plant warm-start files the cold block should move.
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            (tmp_path / f"myjob.{ext}").write_text(f"fake {ext}")
        proc = subprocess.run(
            ["bash", str(wrapper), "--cold"],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, (
            f"wrapper exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
        # Originals are gone.
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            assert not (tmp_path / f"myjob.{ext}").exists(), (
                f"myjob.{ext} should have been moved aside"
            )
        # An aside dir was created and holds the originals.
        asides = list(tmp_path.glob("myjob-restart-aside-*"))
        assert len(asides) == 1, f"expected 1 aside dir, got {asides}"
        aside = asides[0]
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            assert (aside / f"myjob.{ext}").exists(), (
                f"aside dir missing myjob.{ext}"
            )

    def test_cold_with_no_warmstart_files_is_noop(self, tmp_path):
        """Idempotent: ``--cold`` on a clean directory must not
        fail and must NOT create an empty aside dir."""
        wrapper = _truncated_siesta(tmp_path)
        proc = subprocess.run(
            ["bash", str(wrapper), "--cold"],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
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
        wrapper = write_run_wrapper(script)
        text = wrapper.read_text()
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

        proc = subprocess.run(
            ["bash", str(wrapper), "--cold"],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, (
            f"wrapper exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
        # Originals -- SystemLabel-keyed names -- must be gone.
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            f = tmp_path / f"siesta-foo.{ext}"
            assert not f.exists(), (
                f"siesta-foo.{ext} (SystemLabel-keyed) should "
                f"have been moved aside by --cold.  Pre-fix the "
                f"glob only matched siesta-foo-stage2.{ext} "
                f"(basename-keyed) and silently missed this file."
            )
        # Aside dir contains them.
        asides = list(tmp_path.glob("siesta-foo-stage2-restart-aside-*"))
        assert len(asides) == 1
        aside = asides[0]
        for ext in ("DM", "CG", "XV", "LWF", "ZM"):
            assert (aside / f"siesta-foo.{ext}").exists(), (
                f"aside dir missing siesta-foo.{ext}"
            )

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

        proc = subprocess.run(
            ["bash", str(wrapper), "--cold"],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0
        # All four should be gone.
        for name in ("job.DM", "job.XV", "job-stage3.DM", "job-stage3.XV"):
            assert not (tmp_path / name).exists(), (
                f"{name} should have been moved aside under --cold"
            )
        # And present in the aside.
        asides = list(tmp_path.glob("job-stage3-restart-aside-*"))
        assert len(asides) == 1
        aside = asides[0]
        for name in ("job.DM", "job.XV", "job-stage3.DM", "job-stage3.XV"):
            assert (aside / name).exists()

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

        proc = subprocess.run(
            ["bash", str(wrapper), "--cold"],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, (
            f"wrapper exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
        assert not (tmp_path / "foo.DM").exists(), (
            "quoted SystemLabel must be quote-stripped before globbing "
            "-- pre-fix the glob looked for ``\"foo\".DM`` and missed "
            "the unquoted filename SIESTA actually wrote."
        )
        assert not (tmp_path / "foo.XV").exists()

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
        wrapper = write_run_wrapper(script)
        text = wrapper.read_text()
        end = text.find('echo "================================')
        if end < 0:
            end = text.find("mpirun")
        end = text.find("\n", end) + 1
        wrapper.write_text(text[:end] + "\nexit 0\n")
        (tmp_path / "foo.DM").write_text("fake")

        proc = subprocess.run(
            ["bash", str(wrapper), "--cold"],
            cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0
        assert not (tmp_path / "foo.DM").exists(), (
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

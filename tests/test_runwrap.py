"""Shell-wrapper emission (``molbuilder.runwrap``).

Each test binds a synthetic Capabilities via ``set_capabilities``;
the autouse fixture in ``tests/conftest.py`` resets it afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.diagnostics import (Capabilities, EXTENSION_TO_CATEGORY,
                                      set_capabilities)
from molbuilder.runwrap import render_run_wrapper
from molbuilder.jobset.model import Resources
from molbuilder.runfiles import RunNames


from molbuilder.scheduler import Environment as _Env, Topology as _Topo

#: THIS MACHINE'S RECORD, as `jobset probe --write` writes it, carrying the
#: copy of this machine's `env_init`: every wrapper reads how a shell enters
#: an environment off the record (`configuration.md` § 4).  A render of a deck in a folder that does not exist names it, as
#: prep names the target's.
_MACHINE = _Env(scheduler="workstation",
                topology=_Topo(sockets=2, cores_per_socket=32),
                env_init={"preamble": "module load mamba",
                                   "activation": "source activate"})


def _bind(envs_overrides=None):
    """Bind a synthetic snapshot for the current test."""
    cfg = {"envs": envs_overrides} if envs_overrides else {}
    set_capabilities(Capabilities(
        runtime_config = cfg,
        conda_binary   = "/usr/bin/conda",
    ))


# --------------------------------------------------------------------- #
#  Extension routing                                                    #
# --------------------------------------------------------------------- #


def test_extension_table_covers_two_engines():
    assert EXTENSION_TO_CATEGORY[".fdf"] == "siesta"
    assert EXTENSION_TO_CATEGORY[".py"]  == "pyscf"


# --------------------------------------------------------------------- #
#  SIESTA (.fdf) wrapper text                                           #
# --------------------------------------------------------------------- #


def test_render_siesta_emits_propor_diagnostic():
    """On SIESTA exit-non-zero with propor: ERROR, the wrapper prints
    a MULTI-CAUSE diagnostic (2026-06-26): pseudopotential FIRST (a
    null KB projector / mismatched pseudo), then -np as a legitimate
    tunable.  Pin the key text so a regression doesn't silently drop the
    diagnostic or revert to the old 'it's-never-a-config-bug,
    just-lower-np' framing that masks a defective pseudo.

    Spin is not a cause: `propor` is called only from `matel_table.F90`,
    and IMAX = 0 is an all-zero radial table."""
    _bind()
    names = RunNames.of("hemeC", "01_coarse", "hierarchical")
    text = render_run_wrapper(Path("/x") / names.name(".fdf"),
                              machine_record=_MACHINE, names=names,
                              resources=Resources(mpi_np=15, cpus_per_task=1))
    # Captured run, not exec.  2026-06-26: the launch is piped through
    # the _mb_scf_tee timing filter (§ 11.0b), so the exit code is read
    # from ${PIPESTATUS[0]} (awk must not mask SIESTA's exit), not $?.
    assert "set +e" in text
    assert "_siesta_exit=${PIPESTATUS[0]}" in text
    # The hint is gated by the framework's reading of the cause -- the
    # table's own marker, asked of the door beside the job
    # (`test_the_wrapper_asks_how_the_run_ended_over_both_channels`).
    from molbuilder.parse.engines.siesta_grammar import PROPOR_MARKER
    assert f'_mb_ending stopped-by "{PROPOR_MARKER}"' in text
    # THE ORDER IS THE CLAIM: `-np` first would make a defective
    # pseudopotential read as "you asked for too many ranks".  `.index`
    # still raises on an absent cause, so presence is checked by the same
    # line that checks sequence.
    causes = ["Kleinman-Bylander",              # 1: the pseudopotential
              "np IS a legitimate tunable"]     # 2: ranks, as a tunable
    at = [text.index(c) for c in causes]
    assert at == sorted(at), (
        "the propor diagnostic must name the pseudopotential FIRST; got "
        + " -> ".join(c for _, c in sorted(zip(at, causes))))
    # The rest of each cause's text, which ordering does not cover.
    assert "ekb=0" in text
    assert "molbuilder pseudo check" in text
    assert "Spin.Total" not in text, "the retracted spin cause is back"
    # Re-exit with SIESTA's code.
    assert 'exit "$_siesta_exit"' in text


def test_the_notice_that_replaced_it_is_about_ORBITALS_and_only_advises():
    """**The objective number, and it never refuses** *(user ruling)*.

    SIESTA distributes ORBITALS across ranks, so ``n_orbitals / mpi_np`` is
    the occupancy and it wants to be greater than one.  At or below it the
    user is told *"your CPUs are not going to be fully used"* -- and the run
    proceeds.

    The orbital count is the ``10 x n_atoms`` DZP estimate the BlockSize
    bound and the deck's BENCH-MARKS block already use, so the deck and the
    notice cannot disagree.
    """
    import subprocess
    from molbuilder.runwrap import _orbitals_per_rank_notice

    notice = _orbitals_per_rank_notice(10)          # ~100 orbitals
    assert "_norb_est=100" in notice

    def _say(ranks):
        r = subprocess.run(["bash", "-c", f"_mpi_np={ranks}\n" + notice],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return (r.stdout + r.stderr).strip()

    assert _say(20) == "", "20 ranks over ~100 orbitals is 5 each -- nothing to say"
    for ranks in (100, 200):
        out = _say(ranks)
        assert "not going to be fully used" in out, out
        assert str(ranks) in out and "100" in out, (
            "the notice must show both numbers so the claim is checkable")
        assert "not a limit" in out

    assert _orbitals_per_rank_notice(None) == "", (
        "a deck with no NumberOfAtoms gets no notice -- inventing the number "
        "is what the ruling removed")


@pytest.fixture(autouse=True)
def _autosetup_minimal_config(tmp_path, monkeypatch):
    """Every render in this file reads the activation off the machine's
    record (`configuration.md` § 5 M-1), so each test's config root holds
    that record -- `_MACHINE`, the canonical Sol activation.  A test that
    wants another rewrites `environment.json`."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT -- and so the machine scope, where the
    # record a deck under it is rendered with lives.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    (tmp_path / "environment.json").write_text(_MACHINE.to_json() + "\n")
    yield tmp_path


def test_pyscf_cold_block_sweeps_by_name_not_by_inventory():
    """U17 (job-contracts § 4.1): the sweep is by NAME (id-keyed globs,
    molbuilder's own writes excepted), not a per-suffix list -- a file
    nobody listed is a file --cold walks past.  This pin: the PySCF block
    carries the id-keyed glob forms (braced for underscore suffixes) and no
    suffix enumeration."""
    from molbuilder.runwrap import _cold_restart_block
    block = _cold_restart_block("myjob", engine="pyscf", label="myjob")
    assert '"$_warm_label".*' in block
    assert '"${_warm_label}"_*' in block
    assert "myjob.*" in block and "myjob_*" in block
    for retired in ("_optimized.xyz", "_geom_optim", ".chk\""):
        assert retired not in block, (
            f"a suffix enumeration crept back in: {retired}")

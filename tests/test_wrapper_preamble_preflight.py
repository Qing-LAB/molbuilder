"""A baked preamble must not fail as a bare bash error on the target.

**The failure this closes (Sol, 2026-08-24).** The preamble (`env_init`'s
since 2026-10-02) is baked VERBATIM into the `.run.sh` from the machine that ran `prep`.  The
workstation's config says

    source /home/u/miniconda3/etc/profile.d/conda.sh

so prepping from the browser — where the server runs on the workstation —
put that line into every trial's wrapper.  The bundle then travelled to
Sol, which has no `/home/u/miniconda3` and activates with
`module load mamba` instead, and every job died with

    siesta-...-run.sh: line 196: /home/u/.../conda.sh: No such file or directory

naming neither the config key that put the path there, nor the machine it
came from, nor what to do about it.

**Why the check has to be in the script.** Nothing at prep time can know:
on the prepping machine the file is right there.  The only molbuilder code
that runs on the target is the wrapper, so the wrapper checks its own
preconditions before relying on them.

**Why the existing generate-time warning could never catch it.**
`runwrap.py` warns when the preamble does NOT name a conda hook or a
module — the opposite condition.  This preamble names one.
"""
from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import pytest

from molbuilder.jobset.model import Resources
from molbuilder.runwrap import _preamble_source_targets, render_run_wrapper


def _render(tmp_path: Path, preamble: str, monkeypatch=None) -> Path:
    """Render a wrapper whose ONLY preamble is the one under test -- the
    record's, the preamble's one home (`configuration.md` § 5 M-1), handed
    over as prep hands the target's."""
    from molbuilder.scheduler import Environment, Topology
    if monkeypatch is not None:
        monkeypatch.chdir(tmp_path)
    (tmp_path / "JOB.fdf").write_text(
        "SystemName t\nSystemLabel t\nNumberOfAtoms 1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = render_run_wrapper(
            tmp_path / "JOB.fdf",
            resources=Resources(mpi_np=1, cpus_per_task=1),
            env="some-env", project_dir=tmp_path,
            machine_record=Environment(
                scheduler="workstation", topology=Topology(),
                env_init={"preamble": preamble,
                                   "activation": "conda activate"}))
    sh = tmp_path / "JOB.run.sh"
    sh.write_text(text)
    os.chmod(sh, 0o755)
    return sh


class TestWhichPathsAreChecked:
    """Only ABSOLUTE `source`/`.` targets — the ones that can silently
    refer to a machine that is not this one.  A wrong guard is worse than
    none: it would refuse a run that would have worked."""

    @pytest.mark.parametrize("line,expected", [
        ("source /home/u/miniconda3/etc/profile.d/conda.sh",
         ["/home/u/miniconda3/etc/profile.d/conda.sh"]),
        ('source "/opt/conda/etc/profile.d/conda.sh"',
         ["/opt/conda/etc/profile.d/conda.sh"]),
        (". /opt/x/conda.sh", ["/opt/x/conda.sh"]),
        ("source /a/b.sh   # the hook", ["/a/b.sh"]),
        ("module load mamba", []),          # no path to check
        ("source ./local.sh", []),          # relative: author's business
        ("source $HOME/x.sh", []),          # built from a variable
        ("", []),
    ])
    def test_extractor(self, line, expected):
        assert _preamble_source_targets([("target", line)]) == expected

class TestTheGeneratedScriptRefusesActionably:

    def test_a_missing_baked_path_fails_with_a_message_not_a_bash_error(
            self, tmp_path, monkeypatch):
        """THE REGRESSION.  Runs the generated artifact, because the text
        looking right is exactly what shipped the bug: the first version of
        this guard put `prep` in backticks inside a double-quoted bash
        string, so the message printed with two holes in it and only
        RUNNING it showed that."""
        sh = _render(tmp_path, "source /opt/definitely-not-here/conda.sh",
                     monkeypatch)
        cp = subprocess.run(["bash", str(sh)], capture_output=True, text=True,
                            cwd=str(tmp_path),
                            env={**os.environ, "MB_LAUNCHED_BY": "manual"},
                            timeout=60)
        out = cp.stdout + cp.stderr
        assert cp.returncode == 78, out          # EX_CONFIG, not a bash 1/127
        assert "/opt/definitely-not-here/conda.sh" in out
        assert "does not exist on this machine" in out
        # the message must be COMPLETE -- no empty command substitutions --
        # and name the fix: the env_init.preamble of the record this
        # calculation was prepped with -- its own copy, beside task.json
        assert "the record prep read" in out
        assert "module load mamba" in out
        assert "env_init.preamble" in out
        assert "environment.json beside task.json" in out

    def test_a_preamble_with_no_absolute_source_gets_no_guard(
            self, tmp_path, monkeypatch):
        """`module load mamba` has nothing to check, so nothing is emitted
        -- the guard must not appear where it has no work to do."""
        text = _render(tmp_path, "module load mamba",
                       monkeypatch).read_text()
        assert "Preamble preflight" not in text

    def test_a_path_that_exists_is_not_blocked(self, tmp_path, monkeypatch):
        """The guard refuses only what is genuinely absent."""
        real = tmp_path / "hook.sh"
        real.write_text("true\n")
        sh = _render(tmp_path, f"source {real}", monkeypatch)
        cp = subprocess.run(["bash", str(sh)], capture_output=True, text=True,
                            cwd=str(tmp_path),
                            env={**os.environ, "MB_LAUNCHED_BY": "manual"},
                            timeout=60)
        out = cp.stdout + cp.stderr
        assert "does not exist on this machine" not in out
        assert cp.returncode != 78, out


# `TestActivationComesFromTheMachineRecord` retired 2026-10-02 (W54 T2, T26):
# its leak test could not fail -- no local config existed to leak -- and is
# a `launch_values.toml` row now, down the road; the record's activation
# baked is `test_runwrap_v2.py`'s; and its "probe" test ran no probe.


class TestTheEnvGateAsksTheTargetMachine:
    """*Which* env you want is a preference; whether it EXISTS there is a
    fact about that machine (`configuration.md` § 5 M-1).  So the gate asks
    the target's record, not the box the generator happens to run on.

    Asking here is the same mistake as baking this machine's activation:
    `molbuilder-siesta-gpu` installed on a workstation says nothing about
    ASU Sol, and the answer otherwise arrives as a `conda activate` failure
    on a compute node after a queue wait.

    The apparent circularity -- *"probing needs an env"* -- is only about
    the probe's OWN env: `conda env list --json` enumerates every env from
    inside any one of them, never entering the ones a generated script
    will use.
    """

    @staticmethod
    def _rec(envs):
        from molbuilder.scheduler import Environment, Topology
        return Environment(
            scheduler="slurm", topology=Topology(),
            env_init={"preamble": "module load mamba",
                               "activation": "source activate"},
            conda_envs=envs)

    @staticmethod
    def _gpu_deck(tmp_path):
        (tmp_path / "JOB.fdf").write_text(
            "SystemName t\nSystemLabel t\nNumberOfAtoms 1\n"
            "Diag.ELPA.GPU .true.\n")
        return tmp_path / "JOB.fdf"

    def _render(self, tmp_path, record):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return render_run_wrapper(
                self._gpu_deck(tmp_path),
                resources=Resources(mpi_np=4, cpus_per_task=1),
                project_dir=tmp_path, machine_record=record)

    def test_the_target_having_it_is_what_permits_generation(self, tmp_path):
        text = self._render(tmp_path, self._rec(["molbuilder-siesta-gpu"]))
        assert "molbuilder-siesta-gpu" in text

    def test_the_target_lacking_it_is_refused_at_prep(self, tmp_path):
        from molbuilder.runwrap import WrapperError
        with pytest.raises(WrapperError) as e:
            self._render(tmp_path, self._rec(["molbuilder-siesta"]))
        # and the message says WHICH machine was asked
        assert "prepared FOR" in str(e.value)
        assert "re-probe" in str(e.value)

    def test_a_record_that_cannot_answer_refuses_nothing(self, tmp_path):
        """Empty is UNKNOWN, not "none": a record written before the field,
        or a machine with no conda on PATH.  A gate cannot refuse on it."""
        assert self._render(tmp_path, self._rec([]))

    def test_a_record_without_an_inventory_does_not_fall_back_to_HERE(
            self, tmp_path, monkeypatch):
        """The subtle half.  Falling back to this box's inventory for a
        record that carries none re-asks the wrong machine by another
        route: a workstation without the GPU env would refuse a bundle for
        a cluster that has it.  So the fallback is used only when there is
        no record at all."""
        import molbuilder.runwrap as rw
        class _Caps:
            conda_envs = frozenset()          # nothing installed HERE
            def env_for_category(self, c): return "molbuilder-siesta-gpu"
            def env_available(self, n): return False
        monkeypatch.setattr(rw, "get_capabilities", lambda: _Caps())
        # a record that cannot answer must NOT inherit this machine's "no"
        assert self._render(tmp_path, self._rec([]))


# `TestTheHeaderNamesTheQueueTheAllocationAsKED` retired 2026-10-02 (W54
# T26): the queue a job names decides `-p` and `-q` in the rows of
# `tests/data/launch_values.toml` (same partition, another QoS) and
# `gpu_contract.toml` (each queue's own partition), down the road.


class TestOneReaderOfSlurmsGresSpelling:
    """There were three `_parse_gres`, and one of them was wrong about the
    hardware ASU Sol actually has.

    `record.py`'s matched the type against a hard-coded list of GPU names,
    so it reported `gh200` as `h200` (substring), `a100.40gb` as `a100` (a
    MIG slice as the whole card), and `hl225` as nothing.  `--gpus`' own
    help says the MIG slices "are separate askable types, not a smaller ask
    of the same one" -- and that reader conflated exactly those, into the
    machine record a bundle is prepped against.

    They also returned the same pair in OPPOSITE ORDER under one name:
    `(count, type)` in `record`, `(type, count)` in `runwrap`.
    """

    @staticmethod
    def _q():
        from molbuilder.scheduler.quantities import parse_gres
        return parse_gres

    def test_the_type_is_read_from_the_token_not_guessed(self):
        q = self._q()
        assert q("gpu:gh200:1") == {"gh200": 1}          # not h200
        assert q("gpu:a100.40gb:4") == {"a100.40gb": 4}  # not a100
        assert q("gpu:h200.35gb:4") == {"h200.35gb": 4}
        assert q("gpu:hl225:8") == {"hl225": 8}          # Habana, not None

    def test_the_slurm_shapes_it_must_survive(self):
        q = self._q()
        assert q("gpu:a100:4(S:0-1)") == {"a100": 4}   # affinity tail
        assert q("gpu:a100:4,mps:400") == {"a100": 4}  # mps is not a GPU count
        assert q("gpu:4") == {"gpu": 4}                # untyped
        assert q("(null)") == {} and q("none") == {} and q("") == {}

    def test_a_partition_merged_across_node_groups_keeps_the_larger(self):
        assert self._q()("gpu:a100:2,gpu:a100:8") == {"a100": 8}

    def test_the_record_narrows_the_same_reading(self):
        """`Topology` states ONE device kind, so it narrows -- it does not
        re-read.  Untyped stays None there: the field means *which device*,
        and "gpu" answers nothing."""
        from molbuilder.scheduler.record import _parse_gres
        assert _parse_gres("gpu:gh200:1") == (1, "gh200")
        assert _parse_gres("gpu:a100.40gb:4") == (4, "a100.40gb")
        assert _parse_gres("gpu:4") == (4, None)
        assert _parse_gres("(null)") == (None, None)


"""A baked preamble must not fail as a bare bash error on the target.

**The failure this closes (Sol, 2026-08-24).** `script_generation.preamble`
is baked VERBATIM into the `.run.sh` from the machine that ran `prep`.  The
workstation's config says

    source /home/qqing/miniconda3/etc/profile.d/conda.sh

so prepping from the browser — where the server runs on the workstation —
put that line into every trial's wrapper.  The bundle then travelled to
Sol, which has no `/home/qqing/miniconda3` and activates with
`module load mamba` instead, and every job died with

    siesta-...-run.sh: line 196: /home/qqing/.../conda.sh: No such file or directory

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

import json
import os
import subprocess
import warnings
from pathlib import Path

import pytest

from molbuilder.jobset.model import Resources
from molbuilder.runwrap import _preamble_source_targets, render_run_wrapper


def _render(tmp_path: Path, preamble: str, monkeypatch=None) -> Path:
    """Render a wrapper whose ONLY preamble is the one under test.

    The server scope is read from the cwd's `molbuilder.json`, and this
    repo's own root has one carrying `source
    /home/qqing/miniconda3/etc/profile.d/conda.sh` -- the very line that
    caused the Sol failure.  Without chdir'ing away, every case here would
    silently inherit it and the "no absolute path" case could never be
    expressed.
    """
    if monkeypatch is not None:
        monkeypatch.chdir(tmp_path)
    (tmp_path / ".molbuilder.json").write_text(json.dumps({
        "script_generation": {"preamble": preamble,
                              "activation": "conda activate"},
        "execution": {"mode": "direct"},
    }))
    (tmp_path / "JOB.fdf").write_text(
        "SystemName t\nSystemLabel t\nNumberOfAtoms 1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = render_run_wrapper(tmp_path / "JOB.fdf",
                                  resources=Resources(mpi_np=1),
                                  env="some-env", project_dir=tmp_path)
    sh = tmp_path / "JOB.run.sh"
    sh.write_text(text)
    os.chmod(sh, 0o755)
    return sh


class TestWhichPathsAreChecked:
    """Only ABSOLUTE `source`/`.` targets — the ones that can silently
    refer to a machine that is not this one.  A wrong guard is worse than
    none: it would refuse a run that would have worked."""

    @pytest.mark.parametrize("line,expected", [
        ("source /home/qqing/miniconda3/etc/profile.d/conda.sh",
         ["/home/qqing/miniconda3/etc/profile.d/conda.sh"]),
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
        assert _preamble_source_targets([("server", line)]) == expected

    def test_both_scopes_in_order(self):
        assert _preamble_source_targets(
            [("server", "source /a/b.sh"), ("project", "source /c/d.sh")]
        ) == ["/a/b.sh", "/c/d.sh"]


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
        # the message must be COMPLETE -- no empty command substitutions
        assert "the machine that ran prep" in out
        assert "module load mamba" in out
        assert "script_generation.preamble" in out

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


class TestActivationComesFromTheMachineRecord:
    """THE FIX for the Sol failure, and the rule behind it.

    A wrapper is generated on one machine and executed on another.  How a
    shell enters its environment differs between them -- `module load
    mamba` + `source activate` on ASU Sol, a `conda.sh` hook on the
    workstation -- so that fact travels on the TARGET'S RECORD, which is
    the thing that crosses (user, 2026-08-24: *"the jobset probe should do
    its job whether it's running on the local machine or a remote HPC
    environment.  Either way, it should provide the only set of data the
    script generator would need"*).

    Before this, `prep --target sol` took Sol's queues and topology from
    its record and the WORKSTATION's preamble from `molbuilder.json`, so
    every job died on `source /home/qqing/miniconda3/.../conda.sh`.
    """

    @staticmethod
    def _sol():
        from molbuilder.scheduler import Environment, Topology
        return Environment(
            scheduler="slurm", topology=Topology(),
            script_generation={"preamble": "module load mamba",
                               "activation": "source activate"})

    def test_the_targets_activation_is_what_gets_baked(self, tmp_path):
        (tmp_path / "JOB.fdf").write_text(
            "SystemName t\nSystemLabel t\nNumberOfAtoms 1\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = render_run_wrapper(
                tmp_path / "JOB.fdf", resources=Resources(mpi_np=48),
                env="molbuilder-siesta", project_dir=tmp_path,
                machine_record=self._sol())
        assert "module load mamba" in text
        assert "source activate molbuilder-siesta" in text
        assert "TARGET PREAMBLE" in text

    def test_this_machines_activation_does_not_leak_into_a_remote_wrapper(
            self, tmp_path, monkeypatch):
        """The regression, stated as the thing that must NOT appear.

        This repo's own root `molbuilder.json` carries
        `source /home/qqing/miniconda3/etc/profile.d/conda.sh`, and it is
        what got baked.  With a target record present it must not be
        consulted at all -- so this does NOT chdir away: the local config
        is deliberately in scope and must still be ignored.
        """
        (tmp_path / "JOB.fdf").write_text(
            "SystemName t\nSystemLabel t\nNumberOfAtoms 1\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = render_run_wrapper(
                tmp_path / "JOB.fdf", resources=Resources(mpi_np=4),
                env="e", project_dir=tmp_path, machine_record=self._sol())
        assert "conda.sh" not in text
        assert "miniconda3" not in text

    def test_a_record_that_states_nothing_falls_back_to_THIS_machine(
            self, tmp_path):
        """The only legitimate substitute for a silent record is the config
        of the machine that record describes -- reachable only when it is
        this one.  `prep` refuses before reaching here when the record
        names somewhere else."""
        from molbuilder.scheduler import Environment, Topology
        (tmp_path / "JOB.fdf").write_text(
            "SystemName t\nSystemLabel t\nNumberOfAtoms 1\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = render_run_wrapper(
                tmp_path / "JOB.fdf", resources=Resources(mpi_np=4),
                env="e", project_dir=tmp_path,
                machine_record=Environment(scheduler="workstation",
                                           topology=Topology()))
        # this repo root's molbuilder.json is the server scope here
        assert "conda.sh" in text

    def test_the_probe_records_this_machines_activation(self):
        """`probe` writes it wherever it runs -- which is what makes
        copying a record here sufficient to generate a script that runs
        there."""
        from molbuilder.scheduler import Environment
        rec = Environment.from_dict({
            "schema": "molbuilder/environment@2", "scheduler": "slurm",
            "script_generation": {"preamble": "module load mamba",
                                  "activation": "source activate"}})
        assert rec.script_generation["activation"] == "source activate"
        # and a record written before the field still loads
        old = Environment.from_dict({"schema": "molbuilder/environment@2",
                                     "scheduler": "slurm"})
        assert old.script_generation == {}


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
            script_generation={"preamble": "module load mamba",
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
                self._gpu_deck(tmp_path), resources=Resources(mpi_np=4),
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

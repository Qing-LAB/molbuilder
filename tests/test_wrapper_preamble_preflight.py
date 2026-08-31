"""A baked preamble must not fail as a bare bash error on the target.

**The failure this closes (Sol, 2026-08-24).** `script_generation.preamble`
is baked VERBATIM into the `.run.sh` from the machine that ran `prep`.  The
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
    /home/u/miniconda3/etc/profile.d/conda.sh` -- the very line that
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
    every job died on `source /home/u/miniconda3/.../conda.sh`.
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
        `source /home/u/miniconda3/etc/profile.d/conda.sh`, and it is
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
            self, tmp_path, monkeypatch):
        """The only legitimate substitute for a silent record is the config
        of the machine that record describes -- reachable only when it is
        this one.  `prep` refuses before reaching here when the record
        names somewhere else.

        THIS MACHINE'S CONFIG IS SUPPLIED, not found.  It read whatever
        `./molbuilder.json` sat in the repo root -- the developer's own, whose
        preamble happens to source a `conda.sh` -- so the assertion below was
        really about the checkout rather than about the fallback.  It passed
        or failed on a file no test controlled, and stopped meaning anything
        the moment the machine scope left the working directory.
        """
        from molbuilder.scheduler import Environment, Topology
        import json as _json
        root = tmp_path / "machine-config"
        root.mkdir()
        (root / "molbuilder.json").write_text(_json.dumps({
            "script_generation": {
                "preamble": "source /opt/conda/etc/profile.d/conda.sh",
                "activation": "conda activate"}}))
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(root))
        (tmp_path / "JOB.fdf").write_text(
            "SystemName t\nSystemLabel t\nNumberOfAtoms 1\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = render_run_wrapper(
                tmp_path / "JOB.fdf", resources=Resources(mpi_np=4),
                env="e", project_dir=tmp_path,
                machine_record=Environment(scheduler="workstation",
                                           topology=Topology()))
        # the machine config supplied above is the server scope here
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


class TestTheHeaderNamesTheQueueTheAllocationAsKED:
    """R1 (`execution/scheduler.md`): the `#SBATCH` header and the `sbatch`
    command line are two RENDERINGS of one placement, never two decisions.

    The command line honoured it -- `submit` builds
    `Directives.of(placement, r)` from a placement resolved against the
    TARGET's record.  The header did not: it read `partition`/`qos` straight
    out of the local `molbuilder.json`'s `scheduler.directives`, falling back
    to the record only when no such block existed.  So a bundle prepped on a
    workstation FOR Sol carried `-p public -q public` -- the workstation's
    default -- while its allocation asked for `htc` (`-p htc -q public`).

    It fails SILENTLY, which is why it is worth a test: `public` IS a real
    Sol domain, so `sbatch` accepts the file and the job runs on hardware
    nobody chose.  `jobset launch` masks it because flags beat the header --
    but the header's own comment tells you to `sbatch` the file directly.
    """

    @staticmethod
    def _sol_with_menu(tmp_path):
        from molbuilder.scheduler import (Domain, Environment, Topology,
                                          write_environment, FILENAME)
        env = Environment(
            scheduler="slurm", topology=Topology(),
            script_generation={"preamble": "module load mamba",
                               "activation": "source activate"},
            domains=[Domain(name="debug", partition="htc", qos="debug",
                            max_time="00:15:00"),
                     Domain(name="htc", partition="htc", qos="public",
                            max_time="4:00:00"),
                     Domain(name="general", partition="general", qos="public",
                            max_time="14-00:00:00")])
        write_environment(env, tmp_path / FILENAME)
        return env

    def _header(self, tmp_path, monkeypatch, domain):
        # a LOCAL config whose directives name a different queue entirely --
        # the situation that produced the failure
        import json
        monkeypatch.chdir(tmp_path)
        # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
        # working-directory step, which is gone (configuration.md § 2.1a) --
        # without naming the directory the write lands in a file nothing
        # opens, and the test passes having configured nothing.
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
        (tmp_path / "molbuilder.json").write_text(json.dumps({
            "scheduler": {"kind": "slurm",
                          "directives": {"partition": "public",
                                         "qos": "public"}}}))
        rec = self._sol_with_menu(tmp_path)
        (tmp_path / "JOB.fdf").write_text(
            "SystemName t\nSystemLabel t\nNumberOfAtoms 2\n")
        from molbuilder.runwrap import render_wrappers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = render_wrappers(
                tmp_path / "JOB.fdf",
                resources=Resources(mpi_np=4, cpus_per_task=1, domain=domain,
                                    time="0-04:00:00", mem="8G"),
                project_dir=tmp_path, machine_record=rec, emit_sbatch=True)
        sb = [x for n, x in out.files if n.endswith(".sbatch")]
        assert sb, "no .sbatch was emitted"
        return [l.replace("#SBATCH ", "") for l in sb[0].splitlines()
                if l.startswith("#SBATCH -p") or l.startswith("#SBATCH -q")]

    def test_the_named_domain_decides_the_pair(self, tmp_path, monkeypatch):
        assert self._header(tmp_path, monkeypatch, "htc") == [
            "-p htc", "-q public"]

    def test_a_different_domain_gives_a_different_partition(
            self, tmp_path, monkeypatch):
        assert self._header(tmp_path, monkeypatch, "general") == [
            "-p general", "-q public"]

    def test_same_partition_different_qos_is_honoured(
            self, tmp_path, monkeypatch):
        """`debug` and `htc` are the SAME partition; only the QoS differs,
        and it is what drops the wall from 4 h to 15 min.  A resolution that
        carried only the partition would lose the whole distinction."""
        assert self._header(tmp_path, monkeypatch, "debug") == [
            "-p htc", "-q debug"]

    def test_the_local_configs_queue_never_leaks_in(
            self, tmp_path, monkeypatch):
        """The regression, stated as what must NOT appear: the local block
        says `public/public` in every case above and must never win."""
        for dom in ("htc", "general", "debug"):
            got = self._header(tmp_path, monkeypatch, dom)
            assert got != ["-p public", "-q public"], (dom, got)


def test_a_gpu_job_goes_to_the_domains_gpu_partition(tmp_path, monkeypatch):
    """`Placement`, not a `(partition, qos)` tuple.

    A domain may declare `gpu_partition` -- where GPU work goes when that
    differs from the same domain's ordinary partition -- and
    `scheduler.place._bind` is what reads it: ``(gpu_partition or
    partition) if prefer_gpu else partition``.

    The first version of this resolution was a hand-written loop over the
    routing rows returning ``(row.partition, row.qos)``, which is `place`'s
    named branch reimplemented WITHOUT that line: both decks would have gone
    to the ordinary partition, so the GPU job would run on the wrong queue
    -- and only for the jobs that care about the distinction.
    """
    import os
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      write_environment, FILENAME)
    from molbuilder.runwrap import render_wrappers
    monkeypatch.chdir(tmp_path)
    env = Environment(
        scheduler="slurm", topology=Topology(),
        script_generation={"preamble": "module load mamba",
                           "activation": "source activate"},
        domains=[Domain(name="mix", partition="cpu-part", qos="public",
                        gpu_partition="gpu-part", max_time="4:00:00",
                        gpu={"type": "a100", "per_node": 4})])
    write_environment(env, tmp_path / FILENAME)

    def header(deck_text, res):
        (tmp_path / "D.fdf").write_text(deck_text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = render_wrappers(tmp_path / "D.fdf", resources=res,
                                  project_dir=tmp_path, machine_record=env,
                                  emit_sbatch=True)
        sb = [x for n, x in out.files if n.endswith(".sbatch")]
        assert sb, "no .sbatch emitted"
        return [l.replace("#SBATCH ", "") for l in sb[0].splitlines()
                if l.startswith("#SBATCH -p")]

    gpu = header("SystemName t\nSystemLabel t\nNumberOfAtoms 2\n"
                 "Diag.ELPA.GPU .true.\n",
                 Resources(mpi_np=4, cpus_per_task=1, domain="mix",
                           time="0-04:00:00", mem="8G", gres="gpu:a100:1"))
    cpu = header("SystemName t\nSystemLabel t\nNumberOfAtoms 2\n",
                 Resources(mpi_np=4, cpus_per_task=1, domain="mix",
                           time="0-04:00:00", mem="8G"))
    assert gpu == ["-p gpu-part"], gpu
    assert cpu == ["-p cpu-part"], cpu


def test_an_unstated_wall_takes_the_NAMED_queues_ceiling(tmp_path, monkeypatch):
    """With no `--time`, the header states the ceiling of the queue it
    names -- the only value that queue can never reject as too long.

    It found that row by matching `(partition, qos)` back against the whole
    menu, which is a second lookup for something `_placement_for` had just
    returned, and it cannot tell two domains apart that share a pair.
    `Placement.domain` IS the row.

    `debug` is the case that proves it: same PARTITION as `htc` on ASU Sol,
    and only its QoS carries the 15-minute wall.  A partition-only match
    would hand a debug job four hours.
    """
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      write_environment, FILENAME)
    from molbuilder.runwrap import render_wrappers
    monkeypatch.chdir(tmp_path)
    env = Environment(
        scheduler="slurm", topology=Topology(),
        script_generation={"preamble": "module load mamba",
                           "activation": "source activate"},
        domains=[Domain(name="debug", partition="htc", qos="debug",
                        max_time="00:15:00"),
                 Domain(name="htc", partition="htc", qos="public",
                        max_time="4:00:00")])
    write_environment(env, tmp_path / FILENAME)
    (tmp_path / "D.fdf").write_text(
        "SystemName t\nSystemLabel t\nNumberOfAtoms 2\n")

    def wall(domain):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = render_wrappers(
                tmp_path / "D.fdf",
                resources=Resources(mpi_np=4, cpus_per_task=1,
                                    domain=domain, mem="8G"),   # NO time
                project_dir=tmp_path, machine_record=env, emit_sbatch=True)
        sb = [x for n, x in out.files if n.endswith(".sbatch")][0]
        return [l.replace("#SBATCH ", "") for l in sb.splitlines()
                if l.startswith("#SBATCH -t")]

    assert wall("htc") == ["-t 0-04:00:00"]
    assert wall("debug") == ["-t 0-00:15:00"], (
        "same partition as htc -- only the QoS carries the shorter wall")


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

    def test_only_one_module_reads_slurms_gres_spelling(self):
        """The guard.  `runwrap._parse_gres_flag` is excluded by NAME as
        well as by module: it reads what a PERSON typed and raises on a
        typo, which is a different dialect and must stay separate."""
        import re
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[1] / "molbuilder"
        hits = [f.relative_to(root).as_posix()
                for f in sorted(root.rglob("*.py"))
                if "static" not in f.parts
                and re.search(r"^def parse_gres\(|^def _parse_gres\(",
                              f.read_text(encoding="utf-8"), re.M)]
        assert hits == ["scheduler/quantities.py", "scheduler/record.py"], hits


def test_both_spellings_carry_gres_flags(tmp_path):
    """R1: the header and the command line are two renderings of one
    placement.  `--gres-flags=enforce-binding` was appended by the header
    alone, from `runwrap`, on the reasoning that "the command line never
    states it".  That is backwards -- the command line not stating it is
    the disagreement R1 forbids.  It rides with the gres now, because it
    is meaningless without one."""
    from molbuilder.scheduler.emit import Directives
    d = Directives(partition="general", qos="public", gres="gpu:a100:1",
                   ntasks=4, walltime="0-04:00:00")
    hdr = " ".join(d.header_lines())
    cli = " ".join(d.sbatch_flags())
    assert "--gres-flags=enforce-binding" in hdr
    assert "--gres-flags=enforce-binding" in cli
    # and no gres at all -> no binding flag in either
    bare = Directives(partition="p", qos="q", ntasks=4)
    assert "gres" not in " ".join(bare.header_lines())
    assert "gres" not in " ".join(bare.sbatch_flags())

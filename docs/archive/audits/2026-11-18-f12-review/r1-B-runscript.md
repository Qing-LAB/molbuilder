# F12 review, round 1 — slice B: the run script, its execution and the scheduler

Checkout `the checkout` at f578482db (2.77.2), read only.

**Read end to end:** `docs/execution/` job-contracts, script-preparation,
running-a-job, scheduler, submission, gpu, asu-sol,
preparing-for-another-machine, architecture.
**Read whole:** `runwrap.py`, `script_emit.py`, `scheduler/` (`__init__`,
`emit`, `fit`, `kinds` + `kinds.toml`, `place`, `probe`, `quantities`, `queue`,
`record`), `runtime_info.py`, `runtime_config.py`, `wrapper_log.py`,
`diagnostics.py`, `identity.py`, `placement.py`. Traced outside the slice as
far as a claim needed: `jobset/placement.py` (the scheduler placement the
chapters mean by `placement.placed` / `queue_table` / `launch_refusal`),
`jobset/model.py` (`Resources`, `gpu_request`, `Job`), `jobset/prep.py`
(`plan_stage`), `jobset/submit.py` (`_sbatch_request`, `_counts_of`,
`_scheduler_job_name`, `ask_launch`), the catalogue's `wall_slack` /
`max_memory_mb` items, `engine_table.toml`, `kind_table.toml`,
`siesta/warm-files.toml`, `pyscf/input.py` (deck tail).

**Checked by running read-only Python** (`python -B`, nothing written in the
checkout; scratch under `/tmp/f12rev_b_scratch/`): three run scripts rendered
through `runwrap.render_run_wrapper` (SIESTA CPU, SIESTA GPU, PySCF), each
passing `bash -n`; every `module.name` the nine chapters cite that falls in
this slice resolves to a real name; the bundles' import closure.

Note on the slice: `molbuilder/placement.py` is the configured-files table
(`configuration.md` § 3.1), which none of these nine chapters governs — read,
no finding. The scheduler placement the chapters mean is
`molbuilder/jobset/placement.py`, traced above.

Overall: the run script's blocks, their order, the counts chains, the exit
trap and signals, the launch-door gate, the one-start guard, the wall, the
ending questions, MPS, the header and send line, the kinds file, the fit's
sentences (R4/R10 examples reproduce exactly), placement and the machine
record's scopes (C1–C4) all match their chapters. What follows is what does
not.

---

## Defects — 4

### D1. Off the identity GPU list, the dry run's rank → GPU map, the banner's "GPU0 NUMA" and the whole-job NUMA wrap describe GPUs the run does not use

- **Where:** `molbuilder/runwrap.py:1090` (dry run), `:951` (rank helper),
  `:1251–1252` and `:1264–1270` (GPU 0's NUMA node, the whole-job wrap),
  `:986` (the wrap cleared only for two or more GPUs).
- **Rule:** running-a-job § 3.4: "`--dry-run` | … where ranks are placed on
  GPUs — the rank → GPU / NUMA map"; § 3.3: "A generated helper gives each
  rank a GPU (`CUDA_VISIBLE_DEVICES`), block by block (`rank × GPUs /
  ranks`)"; the dry-run block's own docstring: "the block-distribution the
  per-rank launcher will apply".
- **What the code does:** the rank helper takes the device from the list
  itself — `_gpu=${_gpus[$_idx]}` where `_gpus` is `CUDA_VISIBLE_DEVICES`
  split on commas — and asks `nvidia-smi --id=$_gpu`. The dry run instead
  prints an index: `_dr_g=$(( _dr_r * _dr_ng / _mpi_np ))`, then
  `nvidia-smi --id=$_dr_g` and `echo "rank $_dr_r -> GPU $_dr_g
  (numa=$_dr_numa)"`. The placement block asks `nvidia-smi --id=0`
  (`_bash_numa_from_gpu("0", "_gpu_numa_probed")`) for "GPU0 NUMA" and builds
  `_numa_wrap_gpu="numactl --cpunodebind=$_gpu_numa --membind=$_gpu_numa"`
  from it; `nvidia-smi` numbers devices physically and ignores
  `CUDA_VISIBLE_DEVICES`.
- **Failure:** a two-socket workstation with GPUs 0 and 1 on different
  sockets. A person starts a one-GPU SIESTA run with
  `CUDA_VISIBLE_DEVICES=1` (or a scheduler without device isolation hands the
  job GPU 1, and the person sends the header with `--dry-run`):
  - the dry run prints `rank 0 -> GPU 0 (numa=0)`, and the real run puts
    rank 0 on GPU 1 (numa 1);
  - the banner says `GPU0 NUMA=0`;
  - off a scheduler, with one GPU in play, the wrap is kept (it is cleared
    only for `_ngpu >= 2` or under `MB_JOB_ID`), so every rank's memory is
    `--membind`-ed to socket 0 while its GPU sits on socket 1.
  When `CUDA_VISIBLE_DEVICES` is unset, or reads `0..n-1`, the three agree
  with the run.
- **Confidence:** read and traced in the rendered script. Reachable only when
  `CUDA_VISIBLE_DEVICES` names devices other than `0..n-1`.

### D2. The run script's USER-CUSTOM zone runs after the conclusion marker, so it cannot affect the run, and a failing line there changes the exit after `rc=0` is written

- **Where:** `molbuilder/runwrap.py:2901–2903` and `:3034`. The launch block
  ends with `echo "{words} completed: …"`, the point-ended block and
  `conclude_line(0)`; then the zone is appended: `f"\n{_user_custom}\n"`.
- **Rule:** running-a-job § 1 step 21: "writes `<stem>.concluded`,
  `rc=<code> at <date>`, its last act on the main line, and exits with the
  engine's code". job-contracts § 3.1 gives `.run.sh` a USER-CUSTOM zone
  (✅), and § 3.5 says: "anything between the markers is yours … the engine
  judges it" and "Your own additions go here".
- **What the code does:** the zone is the script's tail, under `set -euo
  pipefail`, reached only after `printf "rc=%s at %s\n" 0 "$(date)" >
  "$_concluded"`. On a failed engine the script has already done `exit
  "$_engine_exit"`, so the zone never runs.
- **Failure:**
  - A person puts `ulimit -s unlimited`, an `export`, or a `module load` in
    the run script's zone, which the next prep keeps. None of it reaches the
    engine: it runs after the engine, and only when the engine succeeded.
  - A line there that fails exits the script non-zero after the marker reads
    `rc=0`. The exit trap does not overwrite an existing marker, so the run
    reads `finished` while the scheduler records the job as failed.
- **Confidence:** read and traced in the rendered script (lines 585–590 of
  `/tmp/f12rev_b_scratch/siesta_cpu.sh`). The design choice of where the
  zone sits is the user's; the chapter sentence and the behaviour disagree.

### D3. When the probe falls back to the four-column QoS query, `environment.json` records `max_submit_jobs: null` ("asked, none stated") for a field it never asked

- **Where:**
  - `molbuilder/scheduler/record.py:885–890`: the fallback query
    `format=Name,MaxWall,MaxTRES,Flags`, used when the five-column query is
    refused;
  - `molbuilder/scheduler/probe.py:277–278`: `submit = None` when there is
    no fifth column;
  - `probe.py:458`: `row.max_submit_jobs = qos[q].max_submit_jobs_pu`;
  - `record.py:190–191, 219`: `max_submit_jobs` is in `_NULLABLE`, so
    `None` is written as `null`.
- **Rule:** scheduler.md § 2: "`null` means *asked, and the scheduler states
  none*; the key absent means *never asked*. R3 forbids reading … a
  never-asked one as permission"; R13: "A field the probe did not request is
  not an absence the record may report as silence." `parse_qos`'s own
  docstring says a four-column row "reads as *this probe never asked* rather
  than as *no limit*".
- **What the code does:** the fallback rows carry no fifth cell. `parse_qos`
  then stores `None`, the same value it stores for a blank (uncapped) cell,
  and `derive_domains` copies it onto every domain whose QoS the table lists.
  `to_row` writes `"max_submit_jobs": null`, where it should have left the
  key `UNSET` and off the disk.
- **Failure:** `jobset probe --write` on a cluster whose `sacctmgr` rejects
  `MaxSubmitJobsPerUser` writes a record that claims each QoS was asked and
  states no submit cap. The probe's note says otherwise, but only once, on
  the terminal. Today's one reader (`prep_inputs`, R14) treats `null` and
  absent alike, so nothing is mis-decided yet; the file is wrong.
- **Confidence:** read and traced. Whether a given cluster takes the
  fallback depends on its Slurm version.

### D4. The command a run shows is not the command it runs: the wall arguments are missing from the banner, the log, the dry run and the "completed" line

- **Where:** `molbuilder/runwrap.py:2383`:
  `launch_line = f"{launch_shown} {script_name}{to_output}"`, with
  `launch_shown = "$_launch_cmd"`. It is printed at `:2627` (banner
  `Command`), `:1012` (`resolved launch :`), `:1068` (dry run `Resolved
  cmd`) and `:2901` (`… completed:`). What runs is `launch_words`
  (`:2377`): `$_launch_cmd ${_mb_wall_args[@]+…} <deck> | _mb_scf_tee …`.
- **Rule:** job-contracts § 2.6: "Record resolved launch command + placement
  | the command about to run, logged". The block's own docstring: "the log
  must carry the exact command that ran".
- **What the code does:** every printed form leaves out `-fdf
  MaxWalltime=<left>:s -fdf MaxWalltime.Slack=<slack>:s`, which the engine
  is started with whenever a wall was told. The printed forms also show
  `> $_out_file` where the run pipes through the SCF tee.
- **Failure:** a SIESTA run sent with a 4 h wall shows `Command : mpirun -np
  4 siesta x.fdf > x.out`. A person who re-runs that line by hand to
  reproduce the run runs it with no `MaxWalltime`. Someone reading the log
  for the exact command does not see the arguments SIESTA got; they are on a
  separate `time limit:` line.
- **Confidence:** read and traced; low severity.

---

## Untrue sentences — 3

### U1. "Each absolute path the preamble sources is checked": only a line that is nothing but `source /abs/path` is checked

- **Where:** `molbuilder/runwrap.py:1517–1518`:
  `pat = _re.compile(r"^\s*(?:source|\.)\s+(?P<q>[\"']?)(?P<path>/[^\s\"';|&]+)(?P=q)\s*(?:#.*)?$")`.
  The function's docstring says "Deliberately not a shell parser. It reads
  the one form that caused a real failure".
- **Chapter:**
  - running-a-job § 2.0a: "checks each absolute path the preamble sources
    before it runs; a missing one stops the script (exit 78)";
  - § 5.2: "each absolute path it sources is checked on the node first";
  - job-contracts § 2.6: "each such file checked readable on the node before
    the preamble runs".
- **Scenario:** a record whose preamble is `source
  /home/me/miniconda3/etc/profile.d/conda.sh && conda activate base`, or
  `source /x/conda.sh hook`, or `[ -f /x ] && . /x`. None of these lines is
  checked. On a node without the file, bash dies at that line with its own
  message — no exit 78, no record named — which is exactly the failure the
  check was written for.
- **Confidence:** read and traced. The code is deliberate; the chapters say
  more than it does.

### U2. "A dry run … without writing anything": a GPU-rank dry run writes the rank helper into its folder

- **Where:** `molbuilder/runwrap.py:927–928`. The rank-helper block runs
  inside the program-launcher block, before the dry-run exit, and does `cat >
  "$_rank_helper"` (`.mb-rank-launch-$$.sh` in the cwd). The exit trap
  removes it (`:2728`).
- **Chapter:**
  - running-a-job § 1: "A dry run passes steps 3 to 5 without writing
    anything and stops at step 12";
  - job-contracts § 2.6 *Arguments*: "none of them writes a file".
- **Scenario:** `bash h2_01_coarse.run.sh --dry-run` for a SIESTA job on the
  GPU creates and then deletes `.mb-rank-launch-<pid>.sh` in the run folder.
  The program probe also writes, then removes, a `mktemp` file. Transient,
  but the sentence is not true; a dry run killed with SIGKILL leaves the
  helper behind.
- **Confidence:** read and traced; low.

### U3. "Nothing is refused and nothing is dropped" (scheduler.md R2): a domain row with no `qos` key is dropped silently

- **Where:** `molbuilder/scheduler/record.py:204–205`: `if not {"name",
  "partition", "qos"} <= set(d): return None`. `Environment.from_dict` then
  filters the `None` out without a word (`:490–492`).
- **Chapter:** scheduler.md R2: "**`extra` is the one exception, and what it
  holds is said** … Nothing is refused and nothing is dropped; a typo stops
  being invisible."
- **Scenario:** a hand-edited record that spells one queue's key `"QoS"`, or
  leaves `qos` out instead of writing `null`. That queue disappears from
  `jobset machines` and from the record's queue list with no "?? not
  understood" mark. A job naming it is refused with "the record does not
  list it", and nothing says why the row vanished.
- **Confidence:** read and traced on the reader. *Plausible* as a road: it
  needs a record not written by `jobset probe`, which always writes `qos`.

---

## Plausible, untraced

- **P1. Two hand-listed bundles leave out a module their members say they
  need beside a job.**
  - `runwrap.VIBRATION_COMPANIONS` ships `runrecord.py`, whose
    `TRAVELS_WITH` names `molbuilder.scheduler.quantities`; no
    `quantities.py` is in `mb_vibration.pyz`.
  - `runwrap.PYSCF_COMPANIONS` ships `runfiles.py`, whose `TRAVELS_WITH`
    names `molbuilder.runpoints`; no `runpoints.py` is in `mb_pyscf.pyz`.
  - The import that would fail is in `runrecord.sent_wall_s` (`from
    quantities import parse_walltime`) and in `runfiles.reader` (`from
    runpoints import part`).
  - No caller of either on a path a vibration job or a PySCF script runs was
    found, so this agrees with architecture § 3.2's measurement ("none lies
    on a path a job runs"). It is latent: a future caller would fail with
    `ImportError` beside the job.
  - What is missing to call it a defect is a traced caller.

---

## Final message (for the caller)

Slice B found 4 defects and 3 untrue sentences, plus 1 plausible latent gap.
The most severe five:

1. **D2:** the run script's USER-CUSTOM zone runs after `rc=0` is written, so
   nothing there can affect the run, and a failing line there leaves a
   `finished` run behind a non-zero job exit (`runwrap.py:2901–2903, 3034`).
2. **D1:** with `CUDA_VISIBLE_DEVICES` not `0..n-1`, the dry run's rank →
   GPU map, the banner's "GPU0 NUMA" and the whole-job `numactl --membind`
   name GPU 0 / index n, not the GPUs the ranks use (`runwrap.py:951, 1090,
   1251, 986`).
3. **D3:** when `sacctmgr` refuses the five-column query, the probe writes
   `max_submit_jobs: null` ("asked, none") for a field it never asked
   (`probe.py:277, 458`; `record.py:885–890`), against R13's tri-state.
4. **D4:** the "Command" / `resolved launch` / dry-run / `completed` lines
   leave out the `-fdf MaxWalltime…` arguments SIESTA is actually started
   with (`runwrap.py:2383`).
5. **U1:** the preamble preflight checks only lines that are nothing but
   `source /abs/path`; the three chapters say "each absolute path the
   preamble sources" (`runwrap.py:1517`).

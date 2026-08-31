# Audit 2026-08-28 — the whole road, walked with a real molecule

**Role:** audit report — **evidence, not a plan.** Open items go to
[`roadmap.md`](?doc=roadmap.md) (rule R3); what stays here is what was
measured, how each finding was proven, and what was fixed on sight with the
guard that keeps it fixed.
**Domain:** the full workflow minus transport (its redesign is scheduled
last, user ruling 2026-08-28): CLI verbs end-to-end, the monitor, the
scheduler layer, docs-vs-code in full text, UI/JS/CSS via the guard suites —
and, after the server restart, the **browser integration lane in full**
(§ 6).
**Method:** contracts read in full text (`workflow.md`, `task-setup.md`,
plus the § 2 cross-checks), then the road walked for real: a fresh
`review-bdt/optimization/BDTRelax` (14-atom BDT, SIESTA, this workstation)
taken through init → prep bench → launch bench → summarize → prep run →
launch run, every refusal and artifact compared against the sentence that
promised it. *No diff-and-grep verification: each claim was checked by
reading the named code or by driving it.*

---

## 1. What the walk proved WORKS, against the contract that claims it

| claim | where stated | observed |
|---|---|---|
| init writes exactly the four portable files, names no machine | `workflow.md` § 3, § 6 | `BDTRelax.template.toml`, `task.json`, structure pair; `task@1` shape exact, id `BDTRelax_C6H6S2` |
| the pseudo refusal names the elements, the field, the convention | `project-layout.md` § 2.6 | *"needs H, C, S … set `psml_lib` … the bare name `pseudopotential` means the projects tree"* — followed it, worked |
| gate ④ fires at prep, inside the conductor | `workflow.md` § 9 | thin-vacuum + capped-SCF warnings at `prep bench`, per trial |
| a bench trial is its own attempt | `project-layout.md` § 1.5a | `bench-K2C1/run-0/` on a fresh calculation |
| the grouped launch runs trials in sequence, records per trial | `job-system.md` § 7 | two trials, one invocation, per-trial `run.json`, rc=0 |
| the `[MACHINE]` line, first, model-not-count | `scheduler.md` R12 | `node=devbox cores=40 mem_gb=251.8 gpu=NVIDIA GeForce RTX 3060 Ti` — written by production |
| one machine kind → column shown, census silent | `generator.md` § 4.4b, R11 | machine column present, no *"kinds of node"* line — T1 holding in the wild |
| summarize writes a recommendation, never a decision | `project-layout.md` § 2.3.2 | `run-config.toml`: *"yours to edit … delete a line … delete the file"* |
| prep run applies the file to unstated fields, says so | `task-setup.md` § 11 | *"rendered for mpi_np 4 — agrees with this launch"* |
| status mid-run reads the attempt | § 1.5 | *coarse · run-0 · running*, warm files listed |
| `--mode ask` on a scheduler-less box says so | `submission.md` S5 | *"nothing to wait for — a job here starts immediately"* |

## 2. Findings — fixed on sight, each with its mutation-tested guard

Same-day-regression rule applied: each is a violation of a stated contract,
fixed with the guard named, committed `fa0bf437` / `1222a8a9`.

**F1 — every production run's monitor was dead on arrival** *(severe)*.
`config_dir.py` travels beside `mb_monitor.py` since notify (2026-08-26);
the wrapper writer staged both, `materialize._bring`'s extras named only the
monitor. Bench trials render *into* their attempt → got both; run attempts
*link* → got one. The shipped monitor died at import with stderr on
`/dev/null`: **no `[STATUS]`, no util.csv, no `[UTIL-SUMMARY]`, no
`[MACHINE]`, no reports, on every run since — silently.** Found because the
review's relax had a bench with all four beside a run with none. Fixed
twice over: `runwrap.MONITOR_COMPANIONS` is the ONE list (writer and stager
read it, the staging test asserts through it), and the monitor now survives
the missing companion — *where reports go* being unanswerable means
**reports off, said in the log** (`run-reports.md`: absent is off; a monitor
that cannot report must still monitor).

**F2 — `--mode ask` wrote to the tree** *(severe)*. The launched-stage
auto-continue gated on `dry_run` alone, so asking *when would this start*
opened `run-<n+1>` and copied warm files — **from an attempt that was still
running** (a torn `.DM`/`.XV` copy) — and the fresh empty attempt then hid
the running one from `status`, which reports the latest. One question, and
a live relax vanished from the status table. Ask now takes the non-creating
arm and prints the honest *"WOULD continue … into run-2 (warm), then launch
it"*. Guard counts attempts across an ask.

**F3 — `prep bench` leaked `AmbiguousTarget` as a traceback** where
`prep run` speaks (`workflow.md` § 9: *a gate refuses with the reason,
never a stack trace*). The bench arm reached the environment before the run
arm's catch. First command of the whole walk. Guard drives the real CLI and
forbids `Traceback` in the output.

**F4 — every full-suite run dropped five zero-byte install logs into the
real `~/.molbuilder/logs`.** The bootstrap tests stub `run_install`, but the
log tee is the CLI's, opened around the stub — and `_LOG_ROOT` was
`expanduser`'d at import, freezing the developer's real home before any test
could isolate `HOME`. The root is a function now (*a path that depends on
the environment is a question, asked when asked*), the env tests isolate
`HOME` autouse, and a guard pins the log to the HOME of the moment.

## 3. Findings — open, needing a decision (not fixed, by the align rule)

**O1 — CLOSED 2026-08-28, by the user's conclusion-marker design**
(`project-layout.md` § 1.6, *the other file*): the wrapper's last act on
its main path writes `<basename>-run<N>.concluded` (an error is a
conclusion; a kill never reaches the write), and the launch lane treats a
launched-but-unconcluded attempt as a QUESTION — still running and
force-stopped look identical on disk, so the refusal names both and
`--yes` records the user's judgement. Guards: the real wrapper run
concludes with rc, a SIGTERM'd one leaves no marker, the gate asks, the
judgement is honoured — each mutation-tested.

**O2 — CLOSED 2026-08-28** with the `serve` verbs
(`deployment.md` § 1.0c): both supervisors now respawn a child that died
by signal — killing a hung child brings a fresh one, flap-guarded — and
`serve restart` recycles from outside when the Reload route cannot
answer.

**O3 — CLOSED 2026-08-28**: the server registers a stack-dump hook —
`kill -USR1 <child pid>` appends every thread's stack to
`<state dir>/logs/serve-<port>.stacks.log` (`configuration.md` § 2.1d;
it was `~/.molbuilder/logs/` when this audit was written). The next wedge is read,
not theorized.

**O4 — in-request heavy work can freeze every user.** *Narrowed
2026-08-28:* the GPU half is closed — the frozen child held
`/dev/nvidia*` open, the page widget's per-request `pynvml` reads were
the only in-process driver path, and every one is now an
nvidia-smi-subprocess with a hard timeout behind a cache (`bb946172`;
user rule: *a temporary failure of the inquiry, never a failure of the
system*). What stays open is the general case — RDKit embeds, giant
parses — still run in request threads; worth a contract sentence about
what may run in-request.

**O5 — three wording papercuts**, recorded for their next visit: the
prepped-trials list prints `run-0, run-0` without naming whose attempt each
is; the science-gate warning block repeats once per trial (noise at 16); a
scheduler-less `ask` prints *"nothing to wait for"* and then an
`sbatch --test-only` preview in the next breath.

## 4. The wedged dev server — state and the way back

At review start the 8888 child (pid 2455057) **accepted TCP and answered
nothing**: 41 threads — 39 in `futex_wait`, 11 s total CPU after 7 h, 8
connections in CLOSE-WAIT. Both package locks (`watch._lock`,
`rate_limit`'s) are tight `with` blocks with parses outside — a pure-Python
deadlock is unlikely; the signature fits one thread blocked in a C call
holding the GIL (O4), unprovable without O3.

**The way back is yours** (the supervisor quits on any non-sentinel exit,
and starting servers is not the assistant's call): `Ctrl-C` in the terminal
holding `python -m molbuilder serve --port 8888`, then start it again. The
browser half of this review — Build → form → handover → Task setup →
Results live-watch, gates ① ② ⑤ under the eye — resumes at that moment;
`review-bdt`'s relax keeps a live monitored run to watch.

## 5. What was NOT covered, honestly

- **Transport** — excluded by ruling; last to migrate.
- **Sol** — the machine-identity plan's two remaining facts (a real
  `[MACHINE]` line from a Sol job; the `lightwork` policy cap) need the
  user's Sol session; F1's fix is what makes tomorrow's run monitored.
- **PySCF/spectra decks** were exercised only through the existing guard
  suites this round, not a fresh live walk.

## 6. The browser lane — walked after the restart, all five gates seen

A second BDT, born and driven **entirely in the browser**
(`review-bdt-ui/optimization/BDTUIRelax`), every artifact then verified on
disk:

| step | observed |
|---|---|
| SMILES → 3-D | *"Generated 14 atoms from SC1=CC=C(S)C=C1 · RDKit ETKDGv3"*; cell box updated live when vacuum 3→8 Å |
| save the pair | the dialog says it plainly — *"Saving writes TWO files… coordinates and a sidecar"*; on disk: `bdt-ui.xyz` (SMILES provenance in the comment) + sidecar `vacuum:[8,8,8]`, `cell:null` (derived at generation, by design) |
| load + gate ① | the sidecar found by the pairing rule (box drawn, never named); the amber preflight banner appeared on the first form edit |
| handover, gate ② | four files written; *"Handed over — not a description yet … Still needed: shape and stages"* |
| Task setup, gate ③ | shape/machine asked never guessed; the queue card speaks the workstation truth (*"no queues — runs directly … memory still has a real ceiling"*); save gated on a checkpoint note; `task.json` written and `task.1st.json` **removed** |
| prep from the browser, gate ④ | the two-click confirm states the whole plan (*NO QUEUE STATED · NO MEMORY STATED*…), and then **`generator.md` § 4.1 refused in words with the numbers**: *"declared bench point mpi_np=16, omp_threads=2 needs 32 cores and this machine's probe found 20 … Trim the declaration, or benchmark on the machine it is meant to measure"* — trimmed in the editor (*"edited — not saved"* held until saved), re-prepped clean |
| launch + summarize | CLI (the page teaches the command and deliberately has no launch button); the browser-born trial carries the `[MACHINE]` line first **and** `.concluded` `rc=0` — the whole day's stack in one directory |
| bench summary, B1–B5 | one-kind census riding the header (*40c 250G GeForce RTX 3060 Ti*), per-card *"on … (devbox)"*, gold winner outline, single-iteration caution, both clocks |
| trajectory, gate ⑤ | the finished 140-step relax: phase chip (*Finished · SIESTA 5.4.2 · MPI*), convergence-targets card, 142-frame player with force overlay, four charts with their target lines |
| the fence | a programmatic path outside the tree was refused naming the allowed roots |
| SERVER LOAD | the fenced GPU sampler live through the widget (RTX named, no error) |
| Spectrum tab | full PySCF form renders; a picked `.out` refused politely (*"not a structure file — .xyz/.pdb only"*) |
| console | zero errors across the whole session |

**One suggestion, not a defect:** the bench card's default proposal
(`4/8/16 × 1/2`) exceeded the very machine chosen two cards above, and only
`prep` said so. The refusal is correct and teaches well — but the tab holds
the machine's probe on the same page, so a quiet early note there would save
the round-trip. Left as a suggestion because the current behaviour follows
the contract exactly (the declaration is portable; prep is where it meets a
machine).

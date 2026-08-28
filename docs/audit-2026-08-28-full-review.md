# Audit 2026-08-28 — the whole road, walked with a real molecule

**Role:** audit report — **evidence, not a plan.** Open items go to
[`roadmap.md`](?doc=roadmap.md) (rule R3); what stays here is what was
measured, how each finding was proven, and what was fixed on sight with the
guard that keeps it fixed.
**Domain:** the full workflow minus transport (its redesign is scheduled
last, user ruling 2026-08-28): CLI verbs end-to-end, the monitor, the
scheduler layer, docs-vs-code in full text, UI/JS/CSS via the guard suites —
the **browser integration half is blocked** on a wedged dev server (§ 4) and
is recorded as the one unfinished lane.
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
| the `[MACHINE]` line, first, model-not-count | `scheduler.md` R12 | `node=qlabsrv cores=40 mem_gb=251.8 gpu=NVIDIA GeForce RTX 3060 Ti` — written by production |
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

**O1 — re-submitting over a still-RUNNING attempt.** The ladder
auto-continue reads *launched* (`run.json` present) and continues the
latest attempt — but *launched* is not *finished*: a `launch run` while the
attempt is mid-flight would copy torn warm state and set two engines on one
directory. `prep` has `_ask_if_underway`; the launch lane has no liveness
look. Needs a rule: refuse (or ask) when the latest attempt shows no ending
and its monitor/pid still moves.

**O2 — the supervisor cannot recover a HUNG child** (§ 4). It respawns
only on the sentinel exit code; a wedge is not an exit, and killing the
child by signal makes the supervisor quit — so the designed recovery
(`Reload server`) is unreachable exactly when it is needed. Proposal:
respawn on failed `/api/health` after a grace period, or at minimum treat
signal deaths as respawnable.

**O3 — no way to see a hung server's stacks.** No `faulthandler` hook;
`ptrace` is blocked on this box. Proposal: register
`faulthandler.register(SIGUSR1, file=<log>)` at serve startup — one line,
and the next wedge is diagnosable.

**O4 — in-request heavy work can freeze every user.** The dev server is
one threaded process; a request that enters a long C call holding the GIL
(RDKit embed of a pathological SMILES, a giant parse) stalls all users.
The 2026-08-28 wedge (39 threads on one futex, 11 s CPU after 7 h) fits
this shape; without stacks (O3) the trigger is unproven. Worth a contract
sentence about what may run in-request.

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

- **The browser integration half** (§ 4) — the one open lane of this audit.
- **Transport** — excluded by ruling; last to migrate.
- **Sol** — the machine-identity plan's two remaining facts (a real
  `[MACHINE]` line from a Sol job; the `lightwork` policy cap) need the
  user's Sol session; F1's fix is what makes tomorrow's run monitored.
- **PySCF/spectra decks** were exercised only through the existing guard
  suites this round, not a fresh live walk.

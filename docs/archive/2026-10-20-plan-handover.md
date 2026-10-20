# The plan's handover as it stood on 2026-10-18 (evening) — archived 2026-10-20

**Role:** archive — the handover `plans/plan.md` carried from 2026-10-18 until the 2026-10-20 consolidation, verbatim. The work it lists as done is in the changelog; what it lists as open was carried into the plan's new handover and § 0a.

---

## ▶ HANDOVER — where we are, and where to start *(2026-10-18, evening)*

**Where we are.** THE ORDER NOW *(the user, 2026-10-18: "focus on getting the
transport calculation done correctly. Make sure you get the multi-frame support
correctly. Make sure you get the result parser to be correctly working, and
that is your job")* is done on this machine: the transport calculation, the
multi-frame support and the result parser each read whole, contract against
code, every finding fixed (2.13.1–2.13.2; what each read found is in the
changelog's rows); the five items of API unification (2.13.3); V1.39 whole —
the reader serving one shape, the consumers, both writers, the NetCDF data
file (2.13.3–2.15.0); `netcdf4` listed where a python of ours imports it
(2.15.1–2.15.2); then, on the user's *"continue to fix other issues"*: V1.33
(2.15.3), W48 (2.15.4), V1.37 (closed by 2.15.0) and W44 (closed with no code:
the gather at a rung's first launch is the check, the contracts said
otherwise and are rewritten). **The plan was consolidated the same evening**
*(the user: "consolidate all review results, validate all findings, archive
properly and clean up plan")*: three read-only readers re-derived every
finding in this file against the code at `5d0a33de`, each verdict re-read at
its cited line before anything moved; 34 rows left § 2 for
[`archive/2026-10-18-plan-rows.md`](?doc=archive/2026-10-18-plan-rows.md), each
verbatim under its verdict; the found lists of §§ 5s–5w, 7 and 8 carry what
is fixed struck and what stands at its line of today; the push history this
handover carried from 2.0.0 to 2.13.2 is
[`archive/2026-10-18-plan-consolidation.md`](?doc=archive/2026-10-18-plan-consolidation.md).

**Reviewed whole, the same evening** *(the user: "use agents to do thorough
document and code full reviews and validate all those findings … detect
handcrafted code … fix the mistakes and inconsistencies")*: seven fresh
readers, one lane each — transport, multi-frame, optimization, spectrum, Task
setup and the hand-over, execution and launch, the result parser — each read
its contracts and code whole; their 90 findings were re-read at the cited
lines, and every measured error was fixed in 2.16.3 (the changelog's row says
what; [`archive/2026-10-18-review-findings.md`](?doc=archive/2026-10-18-review-findings.md)
holds every finding with its verdict). Two fresh reviewers then read the
whole diff, and their findings were fixed before the push. **The twelve
design calls and the two open rows were built the next day as 2.17.0** (the
user, 2026-10-19: "go ahead and validate every finding, then fix them"): the
GPU build's absence said at prep, never a veto (E13); the thermal spread
served per mode (S3); one domain per allowed partition+QoS pair, no QoS
preferred (E3); a lead's files on its own stem (T13); the held set and the
SystemLabel read by the one pass (P4); and M9, M10, E2, P2, K14, K19, O6, K5,
K22 — the changelog's 2.17.0 row says what each is; the archive's record
carries the rows as they stood. **The same day, from the Sol runs, 2.18.0**:
the wall reaches every engine through its own channel and a SIESTA wall stop
reads as what it is (`OUT_OF_TIME`); a warm file written as its job died is
named at the next launch; a queued job's standing on the ladder, through
the scheduler kind; the listener's reports on the This machine tab; the
monitor stdlib-only on any python, its vocabulary split from the log's
writer. **Then 2.19.0, the user's yes on the 49 GB `.TSGFL` of every
device point**: the leads' surface Green's functions are the run's, once
per voltage, named in the device deck's `TS.Elec` blocks the point's own
way up and reused by every frame's point at that voltage
(`transport.md` § 2a.11, *What a run's points share*). **Then 2.20.0,
from Sol's device job that died with nothing written**: a sent run the
scheduler no longer lists is written down once by the first reader on that
machine and reads `failed` everywhere; a run past its wall is over anywhere
(`running-a-job.md` § 4.2). **Then 2.21.0**: a running run's row and
the monitor's line say where it is — the SCF row's phase and iteration,
or a device's set-up until its first SCF row (`model/parse.md` § 2b,
`doing`). **Then 2.22.0, contract first**: the leads' surface Green's
functions in a device run's own `tsgf/`, one pair at a time, the walk
round by round; what the files cost and the four levers; the size said on
the device's row before the launch (`transport.md` § 2a.11, *What a run's
points share*). **Then 2.23.0, contract first**: the chat card's layout as
data per receiver — one layout file, read beside the job and by the server,
a copy in the config dir replacing it whole (`run-reports.md` § 4.1c).
**Then 2.23.1, from Sol's `jobset migrate` and `launch`**: a calculation's
records through their owner's door, the calculation passed whole to every
function that asks about it (rule A16, `architecture.md` § 3.2), one writer
staging every temp beside its target, and two decks compared as SIESTA reads
them — every SIESTA keyword's default stated (K9's declaration), the
engine's scratch, the bias-only contours, the contour it builds itself,
every difference said (`transport.md` § 6.1). **Open, awaiting the user's
yes**: whether keywords that steer only how the engine runs —
`Diag.ParallelOverK`, `BlockSize`, `Diag.Algorithm`, the catalogue's
`execution` category — count in the same-input comparison; an older device
run here differs from today's render in `Diag.ParallelOverK` alone.
**Open, the user's yes given (2026-10-19)**: a full-text, full-code review
of `transport.md` and the frame-set contract against the code, two fresh
rounds — the first two reviewers were cut off by the session limit.

**What decides next is the user's, on Sol**: the junction (E13), the
multi-frame run, a vibration on 2.15.x — the PySCF environment there needs
`molbuilder envs doctor molbuilder-pySCF`, then `envs repair`, before a
vibration deck runs (it stops at its first write otherwise) — and the AuBDTAu
workflow. What those runs show is the work; nothing is started here on their
account.

**Off the plan** (the user, 2026-10-18): every feature the three roads do not
need; the 2026-10-18 scan's findings off those roads (migrate, bench-result);
any check, warning or format the contracts do not state.

**Then, in § 0a's order** — the table there is current as of this evening.
Step 1 has nothing left but V1.9's ruling, the user's. Step 2 is the
half-built framework pieces (W35 P2's remainder, W34 P5, § 5w's K13, K14,
K16, K18, K9, Q11, Q8, W33 P4, W45, N10, X4 ② ③, W41, W57-M/K/O, A1.14,
V1.11) — each row re-derived against the code when it is taken, its line
numbers of 2026-10-18 in its state cell. Step 3 the tidy-up (EXCL, Q19's later
phases, § 5r's remainder, § 8 rows 10 and 11, F9, A1.5's and A1.13's
remainders, V1.14's vacuum advice, TST-R, the tests — PYF-T, L2, T-ENV, R11,
W57-T, DOC-TOC, Q2f). Steps 4–6 as the table says.

**The AuBDTAu workflow is the user's, on Sol**, once the framework is
established *(the user, 2026-10-10: "no, you should not run it on this
workstation"; "we focus on workflow and validation of design of framework,
not yet on science. i'll test real system on sol")*. Found for it on
2026-10-10: `AuBDTAu/spectrum/siesta-vib`'s own result says its reference
geometry is not a stationary point (0.37 eV/Å on the free atoms against its
0.01 eV/Å tolerance; MeshCutoff 300 Ry against the relaxation's 350 Ry); its
`job-set.json` and `optimization/min_force_relax`'s predate the 2026-10-06
job keys (`jobset migrate`); and the walk's AuBDTAu relaxation
(`claude-transport-walk/optimization/au-dta-relax2/01_coarse/run-1`) ran with
one k-point on a slab periodic in x and y — check a relaxation's mesh before
citing it.

**Decisions that wait on the user** (§ 0a lists them: GKN, RL15, V1.9's
remainder, F11's go; SCHED-R's advance signal). **Only Sol can show**, when
the user is there: the junction on Sol (E13), the Sol memory measurement, a
group's header under a real scheduler, the AuBDTAu workflow, and why no
instrument recorded SIGTERM on Sol job 65999920 (SCHED-R).


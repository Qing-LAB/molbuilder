# The plan — one file

**Role:** plan — **the only one.** Every open item from the nine plan
documents that preceded it lives here; those nine are archived under
`docs/archive/2026-09-01-*.md` as records of what was decided and built.
**Domain:** all
**Started:** 2026-09-01, by consolidation

> *(user, 2026-09-01: "We don't need ten plan files scattered. We want one
> plan folder or one plan file and stay with that file.")*

**Status, 2026-10-09 — consolidated.** *(user, 2026-10-09: "review your
overall design, make sure all are integrated in plan with context and details,
consolidate the plan, archive obsolete or done items, consolidate design
document, archive and make sure current document contracts are updated
correctly based on validated facts")*. Ten read-only validators read every
section against the code at `61db1660`, each claim re-read at its cited place
before anything moved. What was done, superseded or measured untrue is in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md),
verbatim, each section under its verdict. What stands here is open, with what
is left, where it is in the code and which contract owns it. The duplicate
lists that had grown beside § 2 — *Unscheduled*, the leftovers of § 0b and
§ 0c, W56's and W57's open lines, the items "carried" by archived sections —
are folded into § 2, so R3's one table is one table again.

**Cleaned 2026-10-13** *(user: "clean up plan/todo")*: the rows done since
moved to [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md)
(Q17, Q18, eleven of § 2's, Q19's phase-1 record); § 5aa, the groups, built
from 2.7.0, moved whole to
[`archive/2026-10-13-5aa-groups.md`](?doc=archive/2026-10-13-5aa-groups.md),
nothing of it open (two small code items in Q19's Python list); Q13 says what
is done since; the handover is current. **Consolidated 2026-10-14** *(user:
"consolidate the plan, review the plan and document with agent holistically
and check where missing gap, error, inconsistency, design problem")*: the rows
done in 2.9.5–2.9.6 moved to
[`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md); V1.9
kept for what is left of it; the day's decisions written as rows (V1.39,
V1.40, T-CO2) and step 1's list brought to them. **Ordered 2026-10-13** *(user:
"consolidate the plan for a logical priority for the remaining work")*: the
queue and the milestone table, drifted apart and behind the work, replaced by
one list in priority order (§ 0a, *THE ORDER*). There is no other to-do list:
this file is it.
**Consolidated 2026-10-20** *(user: "conslidate the plan and give me an
update")*: the handover rewritten for where things stand (the 2026-10-18 one
is [`archive/2026-10-20-plan-handover.md`](?doc=archive/2026-10-20-plan-handover.md),
verbatim); one queue, *NOW*, at the head of § 0a, each item placed with its
reason; § 1u, the memory a run needs, added; §§ 1s and 1t point to the queue
instead of carrying their own order.
**Consolidated 2026-10-27** *(user: "consolidate the plan and give me an update")*: every section and row done or settled since moved
verbatim, each under its verdict, to
[`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md)
— §§ 1s, 1v, 1x, 1y, 1ab, 1z, 1aj whole, §§ 1t, 1u, 1w, 1aa and § 1ad.1
as they stood (each kept here as what is left), the 2026-10-20 handover
and its queue, six rows of § 2; four reviewers read every section
against the code and the changelog first. The handover and the queue
rewritten for 2.46.2.

---

## ▶ HANDOVER — where we are, and where to start *(2026-10-27)*

**Where we are: 2.46.3.** Since the last handover (2026-10-20, kept
verbatim in [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md)
§ 3), each a changelog row: the folder download (2.25.0); several groups as
one job (2.26.0); the layers (2.26.2–2.26.4); two records, `task.json` and
`job-request.json` (2.27.0); one door per setting (2.28.0); the memory a
stage needs (2.29.0); a transport calculation that reuses a finished one
(2.30.0); TBtrans's files read by the parser (2.31.0); the transport export
and the atom index map (2.32.0); extending a finished calculation, release 1
(2.33.0); the device's charges under bias and one data-file axis table
(2.34.0); a vibration mode as a frame set (2.35.0); SIESTA's infrared and the
reference displacement (2.35.2–2.37.0); frame sets in reuse (2.38.0–2.40.0);
molecular dynamics out of the optimization kind (2.40.2); what each stage was
prepared with, and the fifteen decisions (2.41.0); SIESTA's vibration as one
run in phases and its Raman at a laser (2.42.0–2.44.1); the vibration form by
result (2.45.0); a frame set saying what its numbers mean, and the charge and
spin in the Spectrum tab's chemistry card (2.46.0–2.46.2); the review's shipped
errors (2.46.3).

**Last checked**: the fast batch on 2.46.3 (6,877 passed); the real-junction e2e on 2.46.3 (189 passed); the stopped-run e2e on 2.46.3 (8 passed); the SIESTA CO₂ vibration e2e on
2.45.0 (8 passed); every road of 2.45.0–2.46.0 through the pages and the
printed commands (vibration → frame set → transport on the H chain).

**What runs next** is § 0a's queue, *NOW* (2026-10-27) — proposed, the
user's to change.

**Decisions that wait on the user** (§ 0a): SEQ-ALLOC, WAE, GKN, RL42,
GATHER-WARM, V1.9's remainder, F11's go, SCHED-R's advance signal; K22's
design and § 9.2's three choices when step 6 reaches them; the one comment
in `envs/recipes.py` § 1ae's handover (k) names, the user's file.

**Only Sol can show**, when the user is there: the junction on Sol (E13 —
the changelog records AuBDTAu transport jobs and their currents on Sol
since 2.20.0, so it may be the user's to close), a group's header under a
real scheduler (Q13), a benchmark's shelf (BENCH-SHELF), the SIGTERM no
instrument recorded (SCHED-R); before a vibration runs there, Sol's PySCF
environment repaired for `netcdf4` (`envs doctor`, then `envs repair`).

**Off the plan** (the user, 2026-10-18): every feature the three roads do not
need; any check, warning or format the contracts do not state. The AuBDTAu
workflow is the user's, on Sol *(the user, 2026-10-10: "no, you should not
run it on this workstation")*.

---

## 1t. The layers — built 2.26.2–2.26.4; one wording left *(the section as it stood: [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md) § 2)*

**Left**: the note the Results tab gives a calculation root whose
description does not read still says *not marked as part of a calculation*
(`web/static/results/viewer.js` `PLACE_NOTE_UNKNOWN`) — its words the user's
(`web/results.md`). The import scan's *no upward import* is to be re-run
when the step that touches the layers next is taken.

## 1u. The memory a run needs — (a) built 2.29.0; (b) and (c) left *(N6; the section as it stood: [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md) § 2)*

**Left** (`execution/job-request.md` § 3.1a): **(b)** the engine's own
analysis for a stage's first run — TBtrans's `TBT.Analyze`, TranSIESTA's
`TS.Analyze` — read and shown beside the request, nothing in the code yet;
**(c)** a formula from the system, only where (a) and (b) leave a gap.

## 1w. One door per setting — built 2.28.0–2.28.1; four parked items left *(the section as it stood: [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md) § 2)*

**Left**, each still true in the code:
- the parameter forms' own field reader, `lib/form-schema.js` `collectField`
  (`web/form-schema.md`);
- the *Recommended:* hover word on Task setup, `task-setup/viewer.js`
  (`web/task-setup.md`);
- a GPU run with a solver that is not ELPA refused only when the deck is
  written (`siesta/input.py`), never at Save (`engines/siesta.md` § 7);
- the `run_card` marker in the catalogue (`template.py`), kept while
  `jobset migrate` reads 2.27.x templates that hold it.

Whether the four documents § 1w found contradicting each other on the
gather and versions were reconciled is to be re-read when one is taken.

## 0. Rule R3, restated

`docs/README.md` used to say **R3: `roadmap.md` is THE one plan.** That is now
this file *(user, 2026-09-01: "the old roadmap and audit should be archived")*,
and R3 reads:

> **`plans/plan.md` is the one plan. Every open item lives there, and nowhere
> else. A document that finds work records the evidence and sends the item
> here.**

That roadmap and the two audit reports are archived under
`archive/2026-09-01-*`. They were 2044 lines between them.

**R3a — a plan section is kept current while its work runs** *(user,
2026-09-16)*:

> **A programme with an implementation section updates that section when each
> step lands, and records a problem found mid-step before working around it.**

Not afterwards, and not in a commit message alone. A step that changed shape, a
step that turned out to depend on something unlisted, a defect found while doing
it: each goes in, with what was measured. **A plan written once describes an
intention; a plan kept current describes the work.** The first section under
this rule is § 5p (transport).

**And R3 now means what it says: there is ONE table** *(2026-09-10)*. Until
that day the open items sat in six tables across §§ 2, 3, 4, 4a, 5, 5b, 5f, 5l
and 5m, which is how three different rows came to be numbered `P3`, how `W8`
could cite line numbers for three endpoints that are not registered, and how
`W9` could carry one sentence naming no behaviour for weeks. **§ 2 is the
list.** A design section keeps the *why* and sends its rows there; a row that
is done, withdrawn or measured untrue leaves for
[`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md)
the same day, with what it turned out to be.

---

## 1ad. The six-dimension review of 2.40 — what it found, and what waits on the user *(2026-10-25; molecular dynamics removed 2.40.2; the fifteen decisions built 2.41.0; the verified findings of § 1ad.2 are last-number moves after it)*

Six fresh reviewers (science, back end, workflow, front end, documents, test
design) read 2.40.1 whole against the contract, each claim reproduced on a
scratch copy or re-derived; every finding below was cross-validated by a
second reader before it was kept. **Done in 2.40.2:** SIESTA's molecular
dynamics left the optimization kind *(the user, 2026-10-25: "yes, remove MD
from the optimization kind, and put a note on MD.FinalTimeStep in generated
script")* — `Verlet` and `Nose` and their three settings, retired; the
relaxation section's note says why `MD.FinalTimeStep` is not written — and
the 2.40.1 tests' own findings TS1–TS4, TS6, TS7.

### 1ad.1 The decisions — built 2.41.0 *(as they stood: [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md) § 2)*

### 1ad.2 Verified findings — fixed after the decisions, last-number moves

**Fixed 2.46.3** (the changelog row says each): SCI4, SCI7, SCI9–SCI14,
BE2, BE3, BE5, BE7, BE10, BE11, WF3 (a voltage added after launch), WF5, WF6,
WF9, WF12, UI1, UI4–UI6, UI11, UI12, DOC1–DOC4, DOC6, DOC8–DOC12, DOC15,
STAGE-PLAN written twice, the two finished-calculation openers; WF8 was
already fixed. **Earlier**: SCI1 (2.41.0), SCI3, SCI6, SCI8 (2.44.0), WF1
(2.41.0), DOC7 (2.43.0), the current integrated two ways and `gf_files`'
second fdf reader (2.43.2). **Dropped as polish**: BE9, WF11, DOC16's asides.
**Left** — re-derived at their lines when taken: BE8, BE12, WF10 (not
located), UI7–UI10, UI13–UI16, DOC5, DOC13, DOC14; and **one design call for
the user**, WF3's other half — a voltage *removed* from the description after
its rung was launched is still read the description's way.

**Science.** SCI1 — the device's Fermi level is moved to the seed's frame
with the last NEGF step's `ts-Vha`; the periodic step's is the right one
(on the e2e chain: 1 µeV apart, the record says 31 meV); the 0.01 eV *equal*
cut is a tolerance of ours, removed (`transport/record.py` `fermi_frames`,
`_science`). SCI3 — the bias sign's text is reversed (`transport.md` ~5814).
SCI4 — IETS described as elastic near zero bias; it is inelastic at eV = ħω.
SCI6 — *Fermi tails beyond 5 kT add nothing*: state the bound. SCI7 — the
frame step is the person's maximum displacement, not the thermal amplitude
(`normal-modes.md`, `vibrational-averaging.md`). SCI8 — Raman without
(Bohr/Å)⁶ is 45.5× too large, not 50× too small. SCI9 — a wire's rigid-motion
count with held atoms is 1 when a free atom is off their line (the doc; the
code is right). SCI10 — the charged-molecule infrared caveat holds only with
held atoms; PySCF's dipole origin is (0,0,0). SCI11 — the electronic
temperature help's *raise to 1000–2000 K* has no source, and above ~1847 K
the 10 eV pole default gives TranSIESTA fewer than 20 poles. SCI12 — *the
step Gaussian and ORCA use* unsupported. SCI13 — seven references cited and
missing from `references.bib`. SCI14 — minor wording.

**Back end.** BE2 — a point's Run panel lacks the launch details
(`runs.record_of` never passes `launched_in`). BE3 — a warm launch again of a
transport rung names the leads' gathered files as continued from its own
run; the ledger's `gathers` is written only without a continuation. BE5 —
`about()` calls the leads' prep logs not ours. BE7 — a frame row in a wrong
unit loses the whole record; the contract keeps the page with the refusal
where the frames would be. BE8 — the read-only `status` prints the deck
check's `info [lead.seam]`. BE9 — an edited cited pair that no longer
classifies is used as *moved*. BE10 — prep decides the reuse before
`checked()`. BE11 — the gather takes a run of points with no launch record.
BE12 — the ledger records a kept relaxation as *named*. Smells, for the next
touch: STAGE-PLAN.md written twice, two doors opening the finished
calculation, the header check written five times, the current integrated two
ways, a second fdf reader in `gf_files`.

**Workflow.** WF1 — the way back after changing a launched stage prints
no command. WF3 — a voltage added after launch: status calls the finished
run failed, a dry-run launch plans a point never prepared. WF5 — `summarize`
writes its own next step. WF6 — a bare `prep task` on an unchanged launched
calculation offers every group. WF8 — `checkpoint` takes `-p`, the docs say
`--bundle`. WF9 — the copy-in refusal prints only the `--cold` line. WF10,
WF11 — one state in different words across verbs; internal paths and old
clauses printed. WF12 — `worked-example.md` stale.

**Front end.** UI1 — Task setup's stage table on a transport folder is
2,400–2,800 px wide in a 708 px box (`task-setup/style.css` ~208: `nowrap` on
every cell). UI4 — the restored-values line prints `null`. UI5 — the fit
refusal's sentence (`window.py` ~63). UI6 — Results' file dropdown after a
ladder pick. UI7–UI9 — doc drift. UI10 — a stale comment in `core.js`.
UI11 — a copied-in group's tab names the newest run only. UI12 —
`#transport-extend-out` shows while `[hidden]`. UI13–UI16 — polish.

**Documents.** DOC1 — *a run still going is never copied from* against the
take-over of finished points. DOC2 — *every run copied under its own name*,
stale since 2.40.0's re-laid copies. DOC3 — `web-api.md`'s Transport rows.
DOC4 — a lead continues from its own density (it starts over). DOC5 — a first
launch's check of *the same calculation*. DOC6 — *nothing is ever saved
without you saying so* against the automatic saves. DOC7 — one Python per job
not stated everywhere. DOC8 — the vibration result's schema number. DOC9 —
the mode frame set described as future work. DOC10–DOC12 — stale names and
references in `transport.md`, `tabs.md`, `workflow.md`, `task-setup.md`,
`parse.md`. DOC13 — this plan's HANDOVER and done rows. DOC14, DOC15 — stale
lines and broken references. DOC16 — correction stories and unexplained codes
the writing rules forbid.

**Tests** (older files, as each is next touched — TEST-AUDIT (5)–(9)): the
walk script called with invented arguments; hand-built `task.json` and
`run.json`; private-step tests; reads by name in the road, flat and launch
e2e files; support helpers duplicated.


## 1ae. SIESTA's vibration as one run in phases — Raman its optical phase: the build *(2026-10-25; the design [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11, the user's B; R-a, R-c, R1 decided; the SIESTA side verified by the proof of concept, § 5.11 § 12)*

*Checked 2026-10-27*: **M1–M4 built** (2.42.0–2.44.1). **Left — M5, the
contract closed**: `engines/vibration.md` § 5.11 folded into the chapter
(it still stands as its own section and says *not yet built* of what is),
R1 measured (`science/vibrational-optical-response.md`; it gates D-e's named
set's share and D-h's six-run table, § 1ac), its two review rounds; and the
handover items still open — (a) and (g) the words § 5.11 keeps, (c)
`continuation.py`, (f) `jobset/ready.py` asking `calculation == "transport"`,
(k) the comment in `envs/recipes.py`, the user's file to approve. After it:
the named set's share (D-e), the `electronic` kind, the Stokes factor.

*(The user, 2026-10-25: "i don't see why pyscf can do all this in one run,
while you struggle so much. there is no intrinsic difference"; "i choose B";
"consolidate how these steps are joint, and how output record/log are
correctly unified to save in one place for parser"; "it should share the same
structure as pySCF results"; "make sure the monitor probe etc should be
integrated correctly"; "consolidate the plan and design document to include
all these aspects, review and if no missing gaps, continue to implement".)*

**What it is**, in one line each (the rules are § 5.11's, not restated here):
one stage `freq` on both engines; SIESTA's phases `relax` → `fc` →
`optical`, each SIESTA call a point of the run in transport's folder shape;
the walk a loop over phases — `plan` (Python) → each call through prep's run
script for its phase (bash) → `record` (Python) → `finish`; every deck by
the one deck writer, its coordinates from a file beside it; the records: each
call's own, the index, the result pair in PySCF's structure, `optical.nc`
through the data-file framework; one monitor per call, the run reporting as
one run; warm relaunch taking over calls done; Raman presented at the laser
and width asked. The `relax` stage retires.

**The milestones** — each closed as § 0a closes one: a full code-text
review of what it touched, its tests green, then the next.

| # | milestone | pieces | done when |
|---|---|---|---|
| M1 | **the framework** | a rung of points declared — each `RUNGS` row states where its points come from, what makes a point done, its start and its finish; every reader asking `calculation == "transport"` today asks the row (prep, the walk, `status`, `continuation`, the gather, Task setup, the record); `siesta/vibration_phases.py` — `PHASES`, the index's writer and reader, its `runfiles.WRITTEN` row and registry parser, the state rule over `points_state`; `parse/fdf.py` reads `%block Label < Filename`; `selection`, `calcdirs`, `paths` join the vibration bundle; `materialize.bring_files`' copy core split into a floor-1 function both call; `siesta/deck_lines.coordinates_rows` (one for `spec_for`, TranSIESTA, the plan); the displacement rule one function (`mode_frames` and the plan) | **built 2.42.0** — the fast batch and the real-junction e2e pass unchanged — nothing a transport calculation does differs |
| M2 | **prep and launch** | `spec_for(part=)` for the relax, force-constant and optical decks; the stage's phase folders with their run scripts; `relax`'s coordinates file; the count beside the wall; `vibration_stages` one stage on both engines and the `relax` stage's rules retired (`ladder`, `ready`'s *waits before it can be prepared*, `_vibration_stage_geometry`, `first_launch_takes`' branch, § 5.2a); the phased walk at launch; the calls a warm launch takes over through `materialize.HandOver`; `warm-files.toml` per phase; the monitor's `--member` flag and `notify` verb, the run script's monitor block rendered for a member, the N-hour clock from the run's start | **built 2.42.0** — the scratch H₂ (held and free) through `describe → prep → launch --mode direct`: `01_freq/{relax,fc}/` each with its deck, `coordinates.fdf` and run script; `run-0/relax`, `run-0/fc` seeded from its `.DM`; the index with both calls, its phases and the finish; `notified.json` start + finish; the result pair at the run's root; the CO₂ e2e (`tests/test_siesta_vibration_e2e.py`) through `describe → prep → launch`: the `relax` and `fc` calls, one mode's three optical calls, the index, the result, one `start` and one `finish`, the sweep — and the relaxation's, the monitor's and the inspector's e2e on the same run |
| M3 | **the job's Python** | `plan`, `record`, `finish` in `spectra/siesta_vibration.py`, the optical calls' record composed there; `parse/engines/siesta_epsimg_reader.py` and its registered parser `siesta_epsimg.py`; `spectra/optical_data.py` (its `Var` table, `photon_energy_ev` in `ncdata.AXES`); the result's `phase_*` on SIESTA, `raman_route` `optical`, `raman_fd_step_ang`, `optical` (schema 13; the `.nc` @7); prep's count and sizes (§ 5.11 § 9) | **built 2.43.0** — the CO₂ e2e's index, result and data file agree, each frame's geometry the displacement rule's and its spectrum its call's `.EPSIMG` through the registry; the result's phases and route right; the data file reads through its reader and `xarray.load_dataset`; the warm-relaunch e2e (a copy stopped once an optical call started) takes `relax` and `fc` over and runs only the optical calls — it caught the optical record reading a result only a run computing its own force constants writes |
| M4 | **the view, the export and the readers** *(designed 2026-10-26, § 5.11 § 10; M4-a–e decided)* | `spectra/raman.py` — `raman_view` (KK, the width composed, the laser), its ask's reader, the `raman.nc` variable table; `phase_ir` and `ir_route` `charges` written by the finish; the viewer's gates by route, `_ramanOf`, the *Laser wavelength* and *Electronic broadening* boxes, the Raman column and lane named by their quantity, the Export buttons; the load door's `raman` ask; `spectra/export.py`, `GET /api/spectra/export`, `jobset export` for a vibration (`--laser-nm`, `--broaden-ev`), the export's shape shared with transport's; `postprocess.raman_view`, `read_optical_data`, `read_raman_data`; `methods`' sentence; the charge switch as items (L4); Task setup's card per phase | **built 2.44.0** — the KK and composition API tests; the CO₂ e2e's symmetry check (633 nm: 45.1 against 0.23 Å⁴/amu) and its export read back through the readers and `xarray`; the page's own e2e asking the laser and exporting in a browser; the viewer on the user's running server (Claude in Chrome) still to walk |
| M5 | **the contract, closed** | § 5.11 becomes the rules of § 2.2, § 5.2a–§ 5.5, § 5.8, § 5.9 as built; `job-system.md` § 5.4's vibration rows; `project-layout.md` § 5 (the manifest regenerated); `job-contracts.md`'s rows; `web/results.md`; `web/spectra.md` § 3.2; `web-api.md`; the method doc's § 5 and § 7; the changelog; two review rounds | R1 measured on the toy junction through the built run (two `optical_mesh` values), its numbers in the method doc's § 3.1 |

**Handover, M2 built (2026-10-26)** — what M3–M5 pick up: (a) a sweep's
later stage taking the first stage's `relax` call over (§ 5.11 § 1, § 6) is
not built — each stage relaxes in its own run today, and § 5.9's summary
refuses two geometries; (c) `status` reads a run whose calls are all done
while its job still goes as *running (after its last point)* — the record
step or the finish at work; (d) § 5.11 is still one
section beside the rules it changes (M5 folds it into § 2.2, § 5.2a–§ 5.5,
§ 5.8, § 5.9); (f) `ready.first_launch_takes` and `at_launch` still
ask `calculation == "transport"` — the next framework pass moves the
question to the rung-of-points declaration (`jobset.points`); (g) a
transport sweep's points still send their own `start` and `finish` — the
run's one pair needs a step its walk can run (R-e's transport half,
`engines/vibration.md` § 5.11 § 7a); (h) a call whose bundle does not load
on the job's python (no numpy) records nothing and sends no event — its
run's state still reads failed, by the run door; (i) a warm launch with
every call done and the finish not done runs the last call again so its
record step finishes the run (§ 5.11 § 6) — no e2e reaches it: the real
road is a finish that failed, and no job environment of ours lacks what the
finish imports; a finish entry removed by hand is a planted state. (b) and
(e) closed with M3 (2.43.0).

**Handover, M3 built (2026-10-26)** — what the review left for later: (j)
a run stopped inside its optical phase keeps the result the force
constants' record step wrote — `phase_raman` *running*, its citation the
data file's sha256 at that write — since the result is rewritten at a
phase's end (§ 5.11 § 8); (k) `envs/recipes.py`'s comments say the SIESTA
env's python is the monitor's and ASE a monitor reader (~l.1466, 1477) —
the person's file to approve; (m) a
spin-polarized optical frame's Fermi level reads none (SIESTA prints
`ef_up`/`ef_dn` with a fixed spin); (n) § 5.5 still describes the finish
as a step after the force-constant run alone — M5 folds § 5.11 in.

**Open after it**: the named set's share (D-e), designed after R1; the
`electronic` kind (§ 5.10 ④) for the response along a mode, unchanged;
**the measured intensity as an option** (the user, 2026-10-26: "add the
Stokes shift factor as an option later") — the factors of the method's
§ 3.2a, contract first (`engines/vibration.md` § 5.11 § 13, M4-d).

## 1af. What a run reports, by the kind of calculation *(2026-10-26; the design agreed with the user — the contract to write in `execution/run-reports.md` § 2, then the build)*

*(The user, 2026-10-26: "the report should report the progress and/or the
periodic update, as the monitor is designed for. i am asking you to do a
logical design rather than aligning with what happens to be the case"; on
transport: "one voltage across every frame would be a transport milestone.
these calculation is heavy and it deserves a report".)*

**The rule an optimization's reports follow, carried to every kind**: a unit
of work is reported when its outcome is something the person can judge or
act on — a geometry move changes the largest force against its tolerance;
otherwise only its count and pace are said, in the periodic report.

| report | when | what it says |
|---|---|---|
| start | once per run | the run started, where |
| step | each geometry move, where the person ticks it | the energy, the largest force against its tolerance |
| milestone | the end of a phase whose outcome the person can act on | that phase's outcome |
| part failed | at once | which call or point, and why |
| periodic | every N hours, the person's setting | the phase, units done of all, the time per unit, the time left at that pace, the running SCF's state |
| wall warning | once per run | the step underway may not finish; or, for a run of points, the units left need more than the wall leaves at the measured pace |
| finish | once per run | how it ended, and what it produced |

**Vibration (SIESTA's run in phases)**: the relaxation's moves are steps;
no message per displacement or per optical call — their count and pace in
the periodic report; milestones at the relaxation's end (converged, the
largest force), the force constants' end (the modes, how many imaginary, the
stationarity verdict — the checkpoint at which the person may stop a Raman
phase that would be wasted) and the optical calls' end. **Transport**: the
seed and the leads are single points, start and finish their whole report;
a sweep's point that fails is reported at once; **one voltage done across
every frame is a milestone** (the user's); the periodic report carries the
points done and failed, the time left at the measured pace, and the running
point's NEGF iteration and residuals; the SCF trouble signs, once built
(`model/parse.md` § 5d.6, P4), once each per point.

**What changes**: `scf_converged` fires on geometry moves only, never on a
force-constant displacement; milestones and part failures are new events;
the wall warning gains the run-level projection; a transport run of points
reports as one run — which needs its run to keep the index a vibration's
keeps, so the monitor can tell its first and last point (R-e's transport
half, `engines/vibration.md` § 5.11 § 7a, handover (g) above).

**With it, the environment the monitor runs in** *(the user, 2026-10-26: one
consistent theme; scanned the same day, the code's claims checked)*. **The
governing rule** is `execution/running-a-job.md` § 2.2a (since 2.35.1): every
Python a job runs is its activated engine environment's own,
`$_mb_py = $CONDA_PREFIX/bin/python3`, set right after the activation
(`runwrap.py`, the activation block) and checked once the run index is known
— the env's name and its python — else `no env python`, nothing started.
The monitor, the ending reader, the retry's record, the vibration bundle's
`loads`/`plan`/`record` steps and the `notify` verb they run, and a PySCF
deck all run on it. **One exception, the launcher layer's**: a sequence of
groups' walk runs the host molbuilder on the node
(`sys.executable -m molbuilder jobset launch`, `submit.py`, the sequence
walk; `job-system.md` § 6.0) — every other walk is bash, the sbatch header
runs no Python. **Aligned in 2.43.0** (docs, comments, and the run script's
unreachable checks for a missing job python dropped; no change of design):
`run-reports.md` § 2.3 and its layout file, `running-a-job.md` § 2.2a
(the launcher layer's one exception; three bundles, the monitor's alone
stdlib-only) and § 3.5, `job-contracts.md` § 2.6 (the PySCF deck on
`$_mb_py`; the job-python and plan-step rows; no *no python* case),
`configuration.md` § 2.3 (`persist` travels; the finish's and the
PySCF bundle's imports), `vibration.md`'s finish command and the
environments' self-tests, `installation.md` and `pyscf.md`'s package
lists, `runwrap.py`'s and `monitor.py`'s comments. **Still open**:
`job-contracts.md` § 2.6's block table has no row for the preamble
preflight. **Why the monitor stays stdlib-only**, restated: it is the
instrument that reports a failure, so it must load when the env's own
packages do not — a broken numpy is reported by it, never hidden by it.
**A job sending several groups** starts each next group by running
molbuilder on the node, in molbuilder's own environment entered by the
machine record's preamble and activation, as a run script enters its
engine's (the user, 2026-10-26: "make sure the correct activation is done …
this should be a framework level. installation of env is assumed";
`job-system.md` § 6.0, *The walk*). The SIESTA envs' self-test imports
numpy and netCDF4 before `siesta --version` (2.43.1; the user, 2026-10-26:
"siesta can check for numpy and netcdf4"). numpy is declared in the PySCF
env too, its self-test importing it (2.44.1; the user, 2026-10-27: "pyscf
include numpy. confirmed") — an existing env gains it with `molbuilder envs
repair molbuilder-pySCF`.


## 1ag. Corrections to a junction's electronic structure — a framework before any one correction *(2026-10-27; planned, not designed)*

*(The user, 2026-10-27: "let's put this deltaH, IP/EA design into a plan
instead of directly build it now because we need a better framework to
integrate such consideration rather than one arbitrary parameter".)*

**The question it answers.** PBE places a molecule's frontier levels too
close to the electrodes' Fermi level, so a junction's conductance comes out
high: AuBDTAu (thiolate on flat Au(111), three-fold sites) reads
T(E_F) = 0.225 G₀, its HOMO-derived resonance at −0.68 eV, against Kaneko
2016's 0.02 G₀ (bridge-tip state; JACS 138, 1294). The corrections in the
literature all act on a converged device Hamiltonian, and TBtrans takes any
of them as a correction to it — `TBT.dH`, a NetCDF file sisl writes
(`deltancSileTBtrans`, eV in, Ry stored; SIESTA 5.4.2
`Util/TS/TBtrans/m_tbt_delta.F90`, manual `tbtrans.tex` § δH) — so only the
transmission re-runs, never the device SCF.

**Why a framework, not a parameter.** Several corrections share one road and
differ in what they move and why:

| correction | moves | its inputs |
|---|---|---|
| DFT+Σ (scissor; Neaton 2006, Quek 2011, Mowbray 2008) | occupied down, empty up — the gap opens — less the image charge | IP, EA and the same functional's gas-phase HOMO/LUMO of a reference molecule; the image plane z₀ |
| a rigid shift (a non-self-consistent gate scan) | every molecular level by one Δ | Δ, or a range of them |
| an optimally tuned hybrid's levels (Egger 2015) | each level by its own gas-phase difference plus the image term | a gas-phase calculation of another functional |

A framework names, for every correction: the subspace it acts on (a label
the person names), the projection (δH_MM = Σₙ Δₙ S_MM ψₙψₙ† S_MM with
ψ†S_MMψ = 1, exact in a non-orthogonal basis), the levels it moves and by
what, each input's unit and source, the terms molbuilder derives from the
geometry (the image term, P = 3.600 eV·Å / (z − z_surface − z₀) per
electrode), and where the record keeps it — the deck, the data file's
settings, the report's curve label, the Methods sentence.

**What to design, in order** (contract first, `engines/transport.md`):
1. The correction as a record: its kind, inputs with units and sources,
   derived terms, and the δH's sha256 — one writer in the job's python
   (sisl, in the SIESTA env), one reader.
2. Where it sits: the transmission rung, and an extension of a finished
   calculation changing only that tab (§ 2a.16 road, built 2.33.0).
3. Inputs from another calculation — a gas-phase reference's IP/EA and
   levels — cited as a calculation of molbuilder's or typed with a source.
4. The comparison: corrected and uncorrected T(E) side by side.
5. The help each correction carries: when it applies (weakly coupled, levels
   away from E_F) and when the papers call it unjustified — a thiolate on
   gold (Mowbray 2008, Liu 2017, Souza 2014) — stated, never refused.

**Facts already verified** (2026-10-27, the agent's report, DOIs
Crossref-checked): z₀ = 1.0 Å (Quek 2011; Souza 2014: 0.79–1.13 Å), 0.9 Å for
Au(111) (Egger 2015); no measured IP for 1,4-benzenedithiol (computed: PBE
ΔSCF 7.5 eV, EOM-CCSD 7.8–7.9 eV; PBE HOMO −5.1 eV — Strange 2011, Bâldea
2014); PBE ΔSCF IP error 0.16–0.24 eV (Ernzerhof & Scuseria 1999, Rostgaard
2010), EAs unreliable with semi-local functionals; Kaneko 2019 shifted BDT's
HOMO by 1.08–1.17 eV and gets 0.029 G₀ (bridge) against 0.024 measured, its
split into the two terms unreported. On AuBDTAu the image term is 1.4–1.8 eV
at the ring centre for z₀ = 0–1 Å, and its 1/d diverges at the sulfurs —
the classical picture's limit for a bonded atom. The step with the best
support for the 0.02-vs-0.22 question is the structure: the thiol hydrogens
kept, or Au adatoms (Strange 2011: GW 1.0×10⁻² G₀ with H kept, 0.83 for the
thiolate; Inkpen 2019: H retained under common conditions) — ordinary
transport calculations molbuilder runs today.

## 1ah. Which modes get Raman — after the run, and by symmetry *(2026-10-27; planned, not designed)*

*(The user, 2026-10-27, on *Raman for*: "because this is done before any
calculation how could there be anything to select from?"; "is there a way
to exclude obvious ones that is insensitive? or there is no general
simplification?")*

Before a run there are no modes, so 2.45.0's *Raman for* offers what can be
decided then — all modes, or a frequency range a chemist knows
(`engines/vibration.md` § 3.1). Two ways to spend less, each its own design:

1. **Pick modes from a finished spectrum.** A vibration whose force
   constants are done, extended with the optical calls of the modes a person
   ticks on the Spectra page — only those run, on the run's own modes; the
   extension road of a finished calculation (`engines/transport.md`
   § 2a.16) for the vibration kind, the run's index already planning optical
   calls after `fc` (§ 5.11 § 3).
2. **Skip the modes symmetry forbids.** A mode's Raman is exactly zero when
   its symmetry forbids it — CO₂'s antisymmetric stretch and two bends, 3 of
   its 4 modes (our run: 0.23 against 45.1 Å⁴/amu, zero to numerical noise),
   9 optical calls cut to 3. To design: the point group found, each mode's
   symmetry species with a tolerance for numerical noise, a forbidden mode
   recorded *not computed: forbidden by symmetry* — never a silent zero —
   and which library in the SIESTA environment does it. It saves nothing on
   a junction, which has little or no symmetry; the frequency range stays
   the general cost control there (above about 300 cm⁻¹ leaves out the slow
   gold modes).

## 1ai. The Structure-optimization tab's two locks into the catalogue *(2026-10-27; planned)*

2.45.0's `active_when` greys a field out from one catalogue declaration,
and the settings gate says the same (`web/form-schema.md` § 1.3). The
Structure-optimization tab still keeps two locks of its own in its page
script (`structure-optimization/viewer.js`): a relaxation's step settings
under *Relaxation type* `none`, and the solvent model with no solvent —
which the gate never mentions. To design: `active_when` naming a value that is
*any one given* (the solvent's blank is its gas phase, not a member to list).
A vibration's relaxation settings stay live under *Structure is already
relaxed* — the person may relax again with other settings; the choice is
theirs, never greyed *(the user, 2026-10-27: "structure that is already
relaxed should not stop user from relax with a different set of parameter,
the user's choice for this, no greying")*. The Spectrum tab's charge and
spin fields sit in step 2, in the chemistry card above the answer they are
worked into — the user's choice, 2026-10-27, built 2.46.0 (`web/spectra.md`
§ 5, `web/form-schema.md` § 1.3: the common part's electronic state served
apart, `&surface=state`). Its card keeps the title *Run profile*: a shared
part's titles name no engine, one engine's own carry its name — the user's
rule, 2026-10-27 (`web/form-schema.md` § 1.3).

## 1ac. A SIESTA vibration's optical activity, and the displacement *(2026-10-24; the charge switch built 2.35.2; the infrared and the displacement reference built 2.36.0; the review and the user's choices 2.37.0; SIESTA's vibration redesigned 2026-10-25 as one run in phases, Raman its optical phase (the user's B); R-a, R-c and R1 decided as recommended, 2026-10-25 — to build)*

*Checked 2026-10-27*: rows 1, 2, 4, 5 and A built (2.35.2–2.44.0), the
review's choices built (2.37.0–2.45.0). **Left**: 3a — the water proof on
both engines (CO₂ done); 3b — the Raman proof, unblocked now piece 5 is
built; B — energies along a mode, waiting on the `electronic` kind; D-e's
named set's share and D-h's six-run table, after R1 (§ 1ae M5); R-e's
transport half.

*(The user, 2026-10-24: "(1) optical activity — is the vibration mode
calculated have polarization or dipole strength changes that it will be IR or
Raman active, and how strong (relatively) are they … what's the correct
scientific process to get these from the siesta calculation? (2) evaluate the
'reasonable' max displacement for suggestion — maybe this can be provided as
deltaE vs displacement amplitude?"; its scope: "it is simple: siesta calculate
correctly optical activity and evalaute the displacement"; "all i asked is a
vibration calculation, assumed an EM in the vacuum")*

### 1ac.1 The method *([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md))*

SIESTA alone, light in vacuum — a unit field along a direction the person
chooses; no field applied, no other calculation:

- **Infrared, every mode, no extra calculation**: the dipole change and the
  charge each mode moves, from the partial charges SIESTA prints at every
  force-constant step (§ 2 of the method).
- **The charge partition — decided** *(user, 2026-10-24: "both on"; "default
  to hirshfeld")*: both computed, every number stated with its partition,
  Hirshfeld shown unless the person picks Voronoi (the method's § 1a;
  `web/spectra.md` § 3.2; `engines/vibration.md` § 5.10 ②).
- **Raman, chosen modes**: SIESTA's optical calculation, one single point
  per frame — the equilibrium and ±δQ along each chosen mode — the optical
  phase of one vibration run whose phases are the relaxation, the force
  constants and Raman, PySCF's shape ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11; the user,
  2026-10-25: "i choose B"); both
  numbers reported, the whole cell's change and the named set's share
  (D-e below; the method's § 3.4, its cost § 5.1); the polarizability at
  the laser's frequency the person enters after the run, the static one
  beside it (D-f). The measurement before it is built is R1 below.
- **The displacement**: § 1ac.2.

### 1ac.2 The maximum displacement — the reference at a temperature, and a driven mode *(the science: [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 2a, § 5)*

- **The input stays the person's maximum displacement** `d_max` — the move,
  at the outermost frames, of the atom that moves most in the mode (the
  largest `|L_A|`; often a light atom, not always).
- **The reference beside it**: `d_T = √2 x_out σ_T max_A |L_A|` (`√3 σ_T
  max_A |L_A|` for three frames), the `d_max` whose frames sample the
  thermal distribution at the experiment's temperature T — from the
  vibration result alone, no calculation. It is the floor.
- **A mode driven by light spreads further** (§ 2a): pumped at random times
  (spontaneous Raman, a current), its spread is an effective temperature,
  measured from the anti-Stokes/Stokes ratio; driven by an infrared laser at
  its frequency, `σ² = σ_T² + Q_0²/2`, `Q_0` from the optical coupling, the
  local field and the linewidth. To second order the transport average needs
  the spread alone, so the set at `σ_T` gives the curvature and a driven
  spread rescales it (§ 2a.3).
- **The harmonic energy** `½ω²q²` is SIESTA's force-constant curvature
  extended to `q` — an estimate, not a SIESTA energy there; the energies
  measured at the outer frames (item B below) check the harmonic picture.

**Built 2.36.0**: the *Write frame set…* dialog shows `d_T` at the page's
temperature, beside `d_max`, and `molbuilder.postprocess.reference_max_displacement`
gives it to a script; `molbuilder frameset from-mode` does not print it. The
value stays the person's.

**Measured — the harmonic range checked.** Single-point SIESTA energies at
the frame set's frames — the junction, the vibration's settings — against
the harmonic `(ħω/4)(d/d_zp)²` say where the outer frames leave the harmonic
range (§ 7's assumption 1 of `science/vibrational-averaging.md`). Cost: about
one force-constant step per frame (AuBDTAu: 13.6 h for 121 steps on 16 ranks
here, ~7 min a frame). It needs a calculation that runs one SCF per frame of
a frame set — the kind `engines/vibration.md` § 5.10 ④ decided (D4,
`electronic`, not built), whose first output this would be.

### 1ac.3 Order

| # | piece | size | depends on |
|---|---|---|---|
| 1 | **the charge switch** in the force-constant input: `Charge.Hirshfeld geometry` and `Charge.Voronoi geometry` (D3, `engines/vibration.md` § 5.3, § 5.10 ②) — **built 2.35.2** | small | — |
| 2 | **the charges measured** at every force-constant step on the toy junction — **done 2026-10-24** (the method's § 7) | small | 1 |
| 3a | **the infrared proof** — **on CO₂ through the SIESTA e2e, 2.36.0**: what the molecule's symmetry demands, under both partitions, on the relaxed and the unrelaxed molecule; the forbidden stretch's 0.5 % residue at `MeshCutoff` 150 Ry falls to 6.5·10⁻⁸ at 500 Ry (the method's § 7). **Still owed**: water in a box on both engines, SIESTA's infrared against PySCF's, mode by mode and in rank | small: two vibration runs | 2 |
| 3b | **the Raman proof**: the same water, SIESTA's optical calculation at each mode's ±δQ against PySCF's Raman — it checks the chain to the Raman strength on SIESTA's molecular formulation, not the junction's (the method's § 7) | medium | the Raman runs ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11) |
| 4 | **infrared, built 2.36.0**: the finish stores every step's charges (schema 11, `partial_charges`); one function derives the infrared of a set of atoms along a direction by a partition (`spectra.infrared`), served by the load door and `molbuilder.postprocess`; the viewer's atoms, direction and partition controls, its columns and chart (`web/spectra.md` § 3.2) | medium | 3a |
| 5 | **Raman, built**: SIESTA's vibration as one run in phases (`relax` stage retired) — the call function and the phase loop in the run script, the phases table, the index, the optical phase and its data file, `raman_view` at the laser and the broadening asked, the viewer ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11) | large | 3b |

**Separate items:**

| # | piece | size | depends on |
|---|---|---|---|
| A | **the reference displacement `d_T`** beside the maximum displacement — **built 2.36.0**: the load door serves it per mode at the page's temperature, the *Write frame set…* dialog shows it, `molbuilder.postprocess.reference_max_displacement` for a script (§ 1ac.2) | small: no calculation | — |
| B | **energies along a mode**, measured (§ 1ac.2): one SCF per frame | medium | the `electronic` kind (§ 5.10 ④) |

**The 2026-10-24 review** (two rounds, six agents, the code and documents in
full, each finding checked against the code — and SIESTA's source — before it
was acted on; 2.37.0 fixed what broke a written rule and built the user's
choices below — `process/changelog.md`).

**Decided by the user, 2026-10-24, built 2.37.0:**

| # | what | the choice |
|---|---|---|
| D-a | the band floor | kept for the infrared view and every chart lane: a lane whose strongest real value is at or below its channel's floor draws no band, and no mode of such a view is ranked *("threshold should exist")* |
| D-b | the dipole's centre | the set's centre of mass at equilibrium *("center should be mass center")* |
| D-c | the light's direction | any vector on the viewer, z the default *("allow setting polarization direction of light with a vector, default to z direction")*; the infrared needs no extra calculation for it |
| D-d | held or free | a result holding atoms: the strength along the light's direction, the molecule in one orientation; holding none: the conventional orientation average, no direction *("if no fix atoms are listed, that means gas phase … when we have fixed atoms that means we have rigidity and surface effects")* |
| D-e | Raman, what is reported | both numbers — the whole cell's change, which keeps the held gold's response at the contact, and the named set's share beside it; the method's § 3.4, its cost measured in § 5.1 |
| D-f | Raman's polarizability | at the laser's frequency, the wavelength the person enters **after the run** — one integral over the absorption spectrum each frame's optical calculation already wrote, so no SIESTA run is added and the laser can be changed or compared on the same runs; the static value beside it as the textbook comparison where it is stable (a free molecule, or where the toy junction's measurement shows it settling) *(the user, 2026-10-24: "yes", to that proposal; the method's § 3.1)* — designed, not built |

**Decided by the user, 2026-10-24, on the recommendations** *("go ahead with
your recommendation"; for PySCF with held atoms, "i will accept your
recommendation, but would need document to be updated explaining how pyscf
calculates, and how saved values are used, justified by scientific
arguments")*:

| # | what | the choice | state |
|---|---|---|---|
| D-g (R2a) | Raman's step | taken from the vibration result — the force-constant run's own nudge, the largest atom moved by it; nothing typed; stated beside the result | designed (the method's § 3.2) |
| D-h (R2b) | Raman's light direction | z by default, another direction chosen before the run; the whole polarizability table (six optical runs a geometry, any direction after) decided once R1 measures the optical calculation's cost | designed (the method's § 3.2) |
| D-i (R2c) | Raman's frame 0 | once for every mode: one Raman calculation takes the modes chosen and shares the undisplaced geometry, $1 + 2N$ runs | designed (the method's § 3.2, § 5.1) |
| D-j (R3) | the reference displacement's temperature | its own box in the *Write frame set…* dialog, starting at the run's thermochemistry temperature | 2.37.0 |
| D-k (R4) | the infrared's sets | a built-in *the free atoms* — everything the vibration moved — beside the structure's labels | 2.37.0 |
| D-l (R5) | a mode's frame set's settings | the record of the relaxation its frame 0 came from: the vibration's own `relax` stage's (its own deck's, as before), or, for a structure stated already relaxed, the record that structure states; one stating none gives a set with none, said | 2.37.0 |
| D-m (R7) | PySCF with held atoms | each mode's dipole change stored whole; a result holding atoms gives the strength along the light's direction, as SIESTA's does; one holding none, PySCF's orientation average, unchanged; `engines/vibration.md` § 4.6 explains how PySCF computes it and how the stored values are used | 2.37.0 |
| L1, L5, L7 | the frame-set dialog's frame numbers; the relaxation tolerance's provenance; the polish | fixed | 2.37.0 |
| L4 | the charge switch shown in Task setup and the deck's provenance | with the Raman inputs, when they are declared | with Raman |
| L2, L3, L6, L8 | — | closed, as recommended | — |
| D-o (R8) | a frame set on the Transport tab | the extension panel names the junction it describes with and offers *Cite a frame set of this junction…*, the picker asking the one rule; skipping it describes the finished junction as it ran *(the user, 2026-10-25: "Build it"; "the user can skip citing frameset")* | 2.39.0 |
| D-p | a narrower question; the extension's layout; the live plan | a stage whose every asked result finished in the finished calculation's newest run is copied in at prep — fewer voltages, or one geometry from a frame set (its first frame) — re-laid copies keeping what each was given; the extension's own layout, five parts, and its plan before Describe *(the user, 2026-10-25: "build the layout, live plan before describe, when citing a multi frame calculation, first frame result is used"; "One rule: reuse completes")* | 2.40.0 |
| D-n (R6) | a frame set whose frame 0 is a finished transport calculation's junction | a calculation that reuses the finished one may cite the set in its place: checked at describe — the same atoms in the user order at the same places, the same cell and axes, the same transport partition; the held set and other labels may differ — and reused deck by deck, as every reuse is, so the seed, the leads and frame 0's points are taken — a finished run launched warm read through what it carries (`runrecord.carried_gather`); on the command line, `describe --reuses <finished> --slot junction=<set>` (`engines/transport.md` § 2a.16, *A frame set of the same junction*) *(the user, 2026-10-24: "yes to R6", "go ahead, contract first")* | 2.38.0 |

**Open, each the user's:**

| # | decision | the choice |
|---|---|---|
| R1 | Raman's numbers, measured | through the built runs, on the toy junction: whether the whole cell's change, at a laser frequency and static, is stable against the optical calculation's k-point mesh and broadening; and what the optical calculation costs per geometry (the method's § 3.1, § 5.1) |
| R-a | how SIESTA's form asks for Raman — which modes is PySCF's selector, reused (`es_mode_selection`, its window and list) | **decided 2026-10-25, revised at 2.45.0**: *Compute Raman* and *Raman for* on both engines, PySCF's selector the orbital probe's alone ([`engines/vibration.md`](?doc=engines/vibration.md) § 3.1, R-a) |
| R-e | a run of points' monitors report as one run — one `start`, one `finish`, the walk's; transport's points lose their per-point pair ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 7a) | taken with the build unless the user objects |
| R-c | the run's resolution, `Optical.Broaden` — the presentation's width is the viewer's, applied after the run: Gaussians compose, **verified exact on H₂** ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 12) | 0.002 Ry (recommended: 2.7 MB of `.EPSIMG` text a frame, 0.4 MB in the data file), revisited with R1 ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 13) |

**Found, not fixed — each with why** (to fix, or to close):

| # | finding | where | why not fixed |
|---|---|---|---|
| L1 | the *Write frame set…* dialog's answer numbers frames from 0 ("frame 0: q/σ …"), the frame-set card from 1 | `core.js` (the dialog's status text), `lib/frameset/card.js` | the dialog predates the review; a wording change of the frame generator's answer |
| L2 | the chart's infrared lane is titled *IR absorption*, never naming the atoms, direction or partition | `core.js` `CHART_CHANNELS`, `spectrumchart` | a lane's title is fixed when the chart is mounted; retitling needs a new chart door; the column beside it names the view |
| L3 | the view's `relative` and the chart's own lane scaling compute the same ranking twice | `spectra/infrared.py`, `spectrumchart/index.js` | `relative` serves a script; the chart scales every channel, PySCF's too |
| L4 | the charge switch (`Charge.Hirshfeld` / `Charge.Voronoi geometry`) is written by the deck builder, not declared as a catalogue item: no PROVENANCE row in the deck, no locked line on Task setup | `siesta/vibration_deck.py` `fc_block` | the force-constant block's other fixed lines (`MD.TypeOfRun FC`, `FC.First`) are written the same way |
| L5 | a vibration stated already relaxed judges its reference forces by `relax_force_tol`, an item declared for the relaxation rung; the deck's provenance does not trace it (the pipeline log does) | `jobset/prep.py` | older than this work |
| L6 | `temperature_K` (template, deck block, thermo) beside `temperature_k` (load door, `postprocess`, `mode_frames`, the data file's axis) | several | naming only |
| L7 | polish: "--" where the page writes "—" in the dialog's reference sentence; the infrared status line has no live-region announcement; *Frames (3 or 5)* is a number box where the page uses selects; the explanation's link has no § 1a anchor | `core.js`, `_spectra_inspector.html` | polish |
| L8 | a vibration run at cheaper settings than its relaxation (AuBDTAu: 1×1×1 against 5×5×1, 300 against 400 Ry) — prep warned, the run went ahead — so its geometry is not a stationary point there | `engines/vibration.md` § 2.2 (warn only) | the person's choice, warned; tied to R5 |

**From the multi-frame check on AuBDTAu** (2026-10-24, the synced Sol copy, a
scratch prep, nothing launched): the citation door accepts the mode-56 set —
312 atoms, one cell, no electrode atom moved, the 5-point Gauss–Hermite nodes
and weights; frame 0 is atom for atom the junction `AuBDTAu_singlebias_equilibrium`
cited; prep writes the seed and leads once and the device and transmission
as five points each, one walk a rung. Said to the person, not ours to change:
each walk's wall is its points' summed — 5 × the 2-day card = 10 days, over
`public`'s 7 — and at the measured 45 h a transmission point the walk needs
about 9.4 days; the set's σ is 2.1 times mode 56's thermal spread at 300 K
(the reference for five frames at 300 K is 0.047 Å, the set used 0.1 Å).
R5 and R6 above are the two decisions it raised, both built (D-l, D-n).

**Verified 2026-10-25, a proof of concept on H₂ outside jobset** ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 12): SIESTA reads a deck's coordinates from a file beside it; the phases chain by the density (the optical frame 0 converges in one SCF iteration from the force-constant run's `.DM`); the broadening composes exactly after the run.

Then: R-a and R-c, then Raman built (piece 5, [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11), R1
measured through it, and the water comparison of 3a and 3b.

**References**: the method's own list
([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)),
each checked against Crossref 2026-10-24, in `references.bib`.

## 1aa. Extending a finished transport calculation — release 1 built 2.33.0; release 2 left *(N11; the section as it stood: [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md) § 2)*

**Left**: **release 2, the page split** — the Transport tab the physics
editor, Task setup the running: `web/task-setup.md` still has Task setup
editing the physics after Describe, so the contract changes first; and a
door for each control's run time (none reads a run's wall time yet).

---

## 0a. THE WORK ORDER — milestones, each closed by a full code-text review *(2026-09-27)*

> *(user: "you need a consolidated persistent plan that get's updated at every
> milestone and reviewed with agent full code text review to validate
> results"; the order itself: "yes, that order works. go ahead")*

### NOW — the queue *(2026-10-27; proposed, the user's to change)*

*(the user, 2026-10-20: "prioritize among all remaining issues, don't switch
if not justified")* — the order: errors on the roads first, then what the
user asked for and is half built, then the agreed designs, then § 0a's
steps 2–6. The queue it replaces (2026-10-20, every row shipped but four)
is [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md) § 4.

| # | what | where | why here |
|---|---|---|---|
| **1** | **§ 1ad.2's findings left** — BE8, BE12, WF10, UI7–UI10, UI13–UI16, DOC5, DOC13, DOC14, each re-derived at its line when taken; WF3's removed voltage, the user's call | § 1ad.2 | the rest of the review's errors; 2.46.3 fixed the others |
| **2** | **The data files describe themselves** (#231) — the definitions in `model/datafiles.md` first, then the writers; § 1aj (2.46.0) was its first slice | `model/datafiles.md`, the writers | the user's ask; half built |
| **3** | **§ 1ae M5 — the vibration contract closed**, with § 1ac's proofs: R1 measured, 3a water on both engines, 3b the Raman proof | § 1ae, § 1ac | what was built in 2.42–2.44 is checked against the science before more is built on it |
| **4** | **§ 1ai — the Structure-optimization tab's two locks into the catalogue** (`active_when` with *any value given*) | § 1ai | the last page keeping a lock of its own after 2.45.0 |
| **5** | **N6 (b) — the engine's own memory analysis** for a stage's first run | § 1u | the user's feature, (a) built |
| **6** | **N11 release 2 — the page split**, contract first | § 1aa | the user's four answers, release 1 built |
| **7** | **N12 group 4** — messages in the wrong words, unit spellings, a node's two core counts, dead whole-node code (`runwrap.py`, `jobset/model.py`), the sweep presenter's columns, `read_data_file`'s atom map, a copied device's provenance, PySCF's two convergence doors, `--one-job`'s nothing-to-write — brought for approval | this table | the six-agent review's last group |
| **8** | **§ 1af — what a run reports, by kind**: the contract in `execution/run-reports.md` § 2, then the build | § 1af | its design agreed with the user |
| **9** | **§ 1ag, § 1ah** — corrections to a junction's electronic structure; which modes get Raman | §§ 1ag, 1ah | planned, each designed with the user first |
| **10** | **THE ORDER's steps 2–6** | below | as the table says |

### THE ORDER — the remaining work, by priority *(2026-10-13)*

> *(user, 2026-10-13: "consolidate the plan for a logical priority for the
> remaining work")* — proposed by Claude from a read of every open row and
> section; **the user's to change**. It replaces the queue (the Q rows) and
> the milestone table (the M rows), which had drifted apart and behind the
> work — transport's surface (Q5, M5) was finished on 2026-10-11 and still
> listed, M2j was closed and still listed, M8's CSS was the same work as
> Q19's. Both, as they stood, are in
> [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md).

**Why this order.** The user's goal, 2026-10-10: *"we focus on workflow and
validation of design of framework, not yet on science"*. So: first what can
give a wrong answer without saying so; then the framework pieces left half
built; then the tidy-up, while nothing new is being built on the code; then
the whole road proved in the browser on the result; then the documents,
written from what is then true; new science last, each designed with the user
first. Inside a step the listed order holds; every row is re-derived against
the code when it is taken (§ 2's rule), and a step closes as a milestone does
(below).

| # | step | what it is, plainly | its items — the detail is in § 2 or the section named |
|---|---|---|---|
| **1** | **Wrong or lost results** | places where molbuilder can show, read or keep the wrong data and not say so | *Done 2.9.5–2.9.6, archived 2026-10-14:* T1, F-MDNC, F19b, W40, the pin of V1.9 (M2m), A1.11, A1.4, the `.pdb` half of A1.5 / A1.13, and the held-set rule of `engines/vibration.md` § 2.3a. *Done 2.10.0–2.15.0, archived 2026-10-18:* V1.10 and A1.1 (§ 5r steps 3–4), X3, PDB-T, F14, K17 (§ 5w.1), PL-R (no defect: a trial runs once), V1.40 (retired: a hand edit between stages), V1.43, RMP (2.10.0), V1.39 (2.13.3–2.15.0). **Open:** V1.9's remainder — the ruling, the user's |
| **2** | **Finish what the framework half-built** | pieces whose design is agreed and partly built, so two ways of doing one thing live side by side | M3 the run record (W35, § 5t.3: P2's remainder, P4 the contour, divergence and the monitor's warnings); M6 P5 (W34, § 5s.3: the charge and spin a run actually used, read back and set against what was asked); § 5w's remainder (M11: K13 one atomic writer, K14 each engine's parallel model as data, K16 the wrapper fence, K18 a PySCF run's level of theory (with K17's two PySCF items: the ECP read back, a `-V` functional beside D3), K9 each engine default declared once with its source); Q11 (the named-queue check in one door); Q8 (one `proxy.trust`; a benchmark's grid stated by its description); M4 (W33 P4, § 5q.6: the design-frame exports, the transport validation subject); ~~W44~~ *(closed 2026-10-18: the gather is the check)*; W45 (checkpointing's owed half); ~~W48~~ *(done 2.15.4)*; N10's two calculation-root checks; X4 ② ③; W41; W57-M, W57-K, W57-O; A1.14; V1.9's remainder (the broad hash, the join check, derived structures — its ruling the user's); V1.11 |
| **3** | **Tidy the code and the pages** | no change to what molbuilder does; the same thing done one way, in one place | EXCL (`exclusive`, a request field nothing sets); Q19's later phases — small UI P3–P4; the CSS phases, M8's W1–W6, W13 and W15 (sealing MolView) folded in as one CSS programme; Python phases 2–5 with S1 (§ 0a's Q19 row lists them); § 5r's remainder (A1.6, A1.8–A1.10, A1.15, A1.17–A1.20), then A1.21 (the audit of the rest of the tree); § 8 rows 10 and 11; F9 (a group prep's notes once, not per member); A1.5's remainder (one home for the structure-path rule, renaming a structure vs a file) and A1.13's; V1.14's vacuum advice; TST-R (what TST's replacements recorded and did not do); the tests — PYF-T, L2, T-ENV, R11, W57-T, DOC-TOC (retire); Q2f (absolute imports — agreed 2026-10-03, done as its own focused session with agent reviews, as the user asked) |
| **4** | **Prove the whole road** | each kind of calculation driven start to finish through the browser and the printed commands, on the code as it then is | M3 P6 = W50 step 3 (an optimization, a vibration and a transport calculation: the parameter tab, the hand-over, Task setup, `prep` / `launch`, the Results tab — every step through the designed doors); M2n's leftovers (the sweep for tests that read text; W57-G, rules held by an assertion on a real run; W56-H, a PySCF relaxation with a held atom; K11 judged); F11's walk (the user's go); the full batch after it |
| **5** | **The documents** | the contracts brought to what is true, and made findable | § 7's four indexes (steps 2–4); § 8 row 13; M4's document sweep (§ 5q.5's list); § 5t P5 (the Makov-Payne script); § 5w.2's text sweep (after K9); E14 (two references checked into `references.bib`); TPL, SMALL and CARRY, each re-derived when taken |
| **6** | **New science** | new things a person can do — each brought to the user as a design, with its science (`scientific-review`: the literature, the numbers, the cost), before any code | M10, from a vibration mode to the current (W42; V1.24, V1.26 — its decision owed, V1.27, V1.34) with V1.25, the mode-displaced frame set (its amplitude the user's); the vibration's other open rows (V1.12, V1.15, V1.17, V1.38; ~~V1.33~~, ~~V1.37~~ done 2.15.3 and 2.15.0); § 5e, a person's own engine text (W47); S13, transport's convergence sweep; K22, per-atom starting spins (the user's design word); § 9.2, ATOM as an optional pseudopotential check |

**Decisions that wait on the user** *(2026-10-14; 2026-10-17; 2026-10-20; 2026-10-21)*: **SEQ-ALLOC** — a sequence of groups' one request (`group.sequence_request`): the largest rank count and the largest cores per rank taken apart, as 2.26.0 builds it, or the request of the group asking the most cores, whole; **WAE** — the *written as the job ended* check compares a file with its run's closing record, to the whole second: on the real-junction e2e's own runs (30 run folders, 2026-10-21) a normally ended run wrote the files a continuation takes (`.DM`, `.XV`, `.CG`) 1.1 s or more before it, so none was named, while one device point's `.TSHS` (gathered, not compared) fell in that second; read from the code, not observed, a write a kill cut in the last fifth of a second before a new second begins falls before the record's second and is not named — closing that needs the monitor to record the signal's time to the fraction of a second, a change brought only on the user's word; **GKN** — an
uneven ranks-over-GPUs cell in a benchmark (32 ranks on 3 GPUs): kept, by
naming every trial by its total ranks (`G3N32C1`), or dropped as now
(RL15, launch's one check, was settled by D15 — § 1ad.1, built 2.41.0); V1.9's
remainder — whether a broad-hash difference blocks (two rulings disagree);
F11 — the walk's go (~45 min of this machine). The questions this list
carried before were answered on 2026-10-13 from the contracts and the user's
own rulings (W40, F-MDNC, PL-R in § 2; EL-W and E7 in
[`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md)).
Step 6's designs — V1.25's amplitude, V1.26, K22 — are brought with their
science when that step starts. **Only Sol can show** (the user's runs, when
the user is there; none on this workstation, *the user, 2026-10-10*): E13 (the
junction on Sol), the Sol memory measurement (M2n), a group's header under a
real scheduler (Q13), the AuBDTAu workflow, the SIGTERM no instrument recorded
(SCHED-R); and, before a vibration runs there on 2.15.x, Sol's PySCF
environment repaired for `netcdf4` (`envs doctor`, then `envs repair`) — the
user's. This machine's was repaired on 2026-10-18.

**How a step closes** (each is a milestone):
1. its items are built — the contract first wherever an item changes a rule,
   then the code, then a test that drives the road (`jobset
   init/prep/launch/summarize`, or the page on a live server) and is broken on
   purpose once to watch it fail;
2. the targeted tests pass (the unit's own and what it touched), and the
   package is `pyflakes`-clean (`conventions.md` § 1);
3. **an agent reads the FULL code text** of the milestone's diff and of the
   contract sections it touches — not a grep — and reports each finding with
   its evidence;
4. every finding is re-read against the code before it is acted on, then fixed
   or answered;
5. this section records the commits, the review and its outcome — and only
   then does the next step start.

A full test batch (`tools/testrun.py`) runs at the end of each step, with nothing changing under it; the last non-e2e batch ran 2026-10-27 on 2.46.1 (6,876 passed, none failed); the real-junction e2e ran 2026-10-27 on 2.46.0 (186 passed); the SIESTA CO₂ vibration e2e ran 2026-10-27 on 2.45.0 (8 passed); the e2e batch last ran whole 2026-10-16 (396, all passing).

**The Q19 row** — the three passes, its phase 1 done and its later phases
listed — stays here as step 3's detail:

| # | item | state | next |
|---|---|---|---|
| **Q19** | **Three passes — CSS, small UI, Python** *(proposed 2026-10-13 from three read-only surveys; the user: "go with your recommendations … think at framework level or larger scope where the snippet sits, need a systematic point of view for fixing"; for Python: "make sure final outcome to be checked against design contract, and reviewed in their full code/text and validated correctly")* | **phase 1 done 2026-10-13**, each part reviewed by fresh agents on the full code and contract text: 2.9.2 (small UI P1–P2 — one page fence, the shape shown fixed, the chip's advice gone, the five generators on one chain; CSS P1 — one dialog component, no browser dialog left; Python P1 — the package pyflakes-clean), 2.9.3 (the server's one numerical thread), 2.9.4 (Task setup as a checklist and the save bar, the user's request). Its record: [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md) | **open — step 3, in this order** (the user approved the passes' recommendations, 2026-10-13) — **small UI**: P3 prep's answer served as lines (the wording to keep is the user's call), P4 one element builder, one path helper, one download helper, and keyboard access (the sidebar rows, glyph buttons named); Structure optimization's *discard unsaved changes?* on Load (a design call: the viewer is read-only, `runtime.md` puts the question on a dirty canvas). **CSS**, decisions 1–8 as recommended: one button vocabulary, a flat `.card` variant for Results' restatements, `.num-table` reused, tokens for durations, z-index layers, weights and line heights, the sealed modules' palettes (`vibrationview`, `spectrumchart`, `docs-render`), the stylesheet order, `.lock-reason`'s raw values, the junction card's control under its prose; MolView's export dialog, a third modal outside `lib/dialog.js`; three animations that ignore reduced motion; the layout test skipping Transport, This machine and JupyterNB. **Python** phases 2–5: the files a calculation holds read once (`JobSet.of`, the template text, `Shape.named(task.shape)`), `done` returning its `RunStatus`, one instruments reader, the long functions (`render_run_wrapper`, the engines' `spec_for`), `RunRef` for `written_at` and the continuation's fields, S1's wrapper hook per engine, prep's planning — one path for a single run and a swept stage (named when the groups closed; measured before acting), and three small dead guards with the cold sweep's docstring (`next_step`'s `s is None`, the `bench=` that is always false, `_with_notify`'s `None` tests — round B of 2026-10-12 did not re-read them); phase 5 the layer index's missing rows and four upward imports |

---

## 2. OPEN — the one list

**Every open item, in one table** *(consolidated 2026-09-10 at the user's instruction: "consolidate plan and to do, archive finished things and untrue things, clean it up so we have one list"; re-consolidated 2026-10-09)*. A design section (§ 5e–§ 9) keeps the *why* and the order of its own work and sends its rows here; the order (§ 0a) says when.

**A row is evidence of when it was written** — re-derive it against the code before acting (the rule, `archive/2026-10-08-plan-consolidation.md` § 5a). Each row below was re-read at its cited place on 2026-10-09; *Re-checked* says what was found. The long histories of the rows that had grown to tens of lines (W36, W38, W41, W30, W32, X4, W52, W54) are in [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md), verbatim; the row here is what is left.

**Rows that were done, withdrawn or measured untrue are not here** — [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md), [`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md) [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md) [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md), [`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md) and [`archive/2026-10-18-plan-rows.md`](?doc=archive/2026-10-18-plan-rows.md), each with what it turned out to be. Every row left here was re-derived against the code on 2026-10-18 by three read-only readers, each verdict re-read at its cited line; *2026-10-18:* in a state cell is what was found. Archived ids still cited elsewhere resolve there: A1.1, F14, GPU-PLACE, GPU-V, JN, JOBS, PDB-T, PL-R, RMP, SCHED-END, SD, SK, SRV, STEPS, T-CO2, TBK, TST, UPG, V1.10, V1.33, V1.37, V1.40, V1.41, V1.42, V1.43, VVW, W4, W5, W36, W38, W44, W48, W56-2, X3 (2026-10-18); A1.12, A1.16, A1.2, A1.3, A1.7, B2, B3, B4, D2, D4, DOC-TEX, E1, E10, E11, N5, N9, R5, S-T2, S18, S3, TD10 text, TR1–TR3, TR4–TR6, TR7–TR8, TR9–TR10, TD6, TD12, R2-15, SCF-RUNG, CHIP, V1.31, W10, W20, W27, W37, W39, W46, W51, W52, W53, W54, X1.

| # | area | item | from | state |
|---|---|---|---|---|
| **TEST-AUDIT** | tests / calculations | **The calculation e2e tests, audited against the contract** (the user, 2026-10-25: "check all your tests on calculation, e2e particularly, against contract/design document to see if they are doing meaningful and correct job"; five agents, every test of 14 e2e files read beside the section it cites, the high-impact claims re-read at their lines). 2.40.0's own reuse tests were fixed in the release (one table for a question the finished results answer, the plan's groups read by their own lines, the record through its parser, a stopped result's reuse, a relaunch's engine run dropped for the record it reads). **Open, by weight:** **(1) a road no test reaches** — a run here told its wall from its card (`test_wall_reaches_the_engine_e2e` and the PySCF wall tests set `MB_WALL_S` in the test's own environment); a warm launch again of a sweep with points not done (`submit._points_walk`'s take-over branch); `freq`'s first launch recording its `relax` hand-over; a frame set's average waiting for points not done; a selection's PDOS refused before the digest is kept; a failed run here printing FAILED; § 4.8's listed mode that names no mode. **(2) checks that cannot fail** — the transmission taking the device's newest run (`is_dir` of a folder prep made); a refused warm launch (no `run-2`, where it would reuse `run-1`); a citation's evidence compared with the value the code copied; the atom-index test falling back to identity when its record is missing; `_mb_warm_retry not in …` (never printed); the infrared route accepting the fallback it names; Task setup's typed value equal to the saved one; refusals checked by one letter. **(3) a regression a fixture hides** — the vibration runs at the relaxation's settings, so a frame set stamped with the vibration's (2026-10-24) passes. **(4) state no road makes** — Task setup's fixtures writing a SLURM `environment.json` into the calculation (`_a_described_probe`, `filled_dir`, `binding_dir`), two browser tests answering the page with made-up server replies, `mkdir`'d run folders, a hand-built pre-v14 sidecar (`an_older_junction…`, retire), a compose record no describe made (`record_answers_for_the_citation…`, retire), a hand-built second vibration stage, `_engine_only(…, set())`, a raw edit of `task.json`'s shape. **(5) reads around the doors** — the transport record (`_record`, fixed 2.40.0), the spectrum and sweep JSON, `task.json`, the ledger (seven `json.loads`; the product has no ledger reader), `.out` and deck text split by hand, `_optimized.xyz` by lines, files by glob or composed name. **(6) expectations from the code** — `pct_words`, `scf_words`, `Point.rel`, `atom_index_map`, `collect_record`, the export's own composition, `toPrecision(4)`, whole-dict equality on undocumented keys. **(7) category and cost** — prep-only checks in engine files never run in the fast batch (PySCF prep tests, the listed-modes refusal, two ticked refusals duplicating `hand_overs.toml`, four reuse API tests); engine runs no test needs (a SIESTA force-constant run for words `--dry-run` prints, two PySCF runs that merge, a third SIESTA group, ten SIESTA starts probing SIESTA's own unit reading, which no doc states). **(8) copies → table rows** — the seven record-gate tests, the dry-run triplet, the card-names and history tests across two files, describe refusals, the two ×8 Results-tab tests, the PDOS-style rows elsewhere. **(9) stale words** — "design §15", "§7.5", "the sweep runs H2", a per-trial bound "5 s" that is 5 minutes. | the test audit, 2026-10-25 | **(1)–(4) done 2.40.1** (the user's yes, 2026-10-25: "yes, go with that order after 2.40.0 ships"); the review's findings on the 2.40.1 tests themselves (§ 1ad: the queue test reaching the unlisted queue, the walk stopped judged in its test, reads through the doors) **done 2.40.2**; **open**: (5)–(9) as each file is next touched; Task setup's late-answer test still waits a bounded 300 ms after the stale reply is delivered (no page signal says it was dropped); the machine-record copy naming no machine is § 1ad's D14 |
| **SCR-UI** | web / transport | **Two pages do not say the device's scratch as the contract does** (found by the 2.24.0 documents' verifiers, 2026-10-20): Task setup's stage row appends `r.tsgf` (`task-setup/viewer.js` ~2703) but the folder answer never sends it (`build._folder_ladder` sends stage, group, state, detail, why), so the `tsgf/` line `transport.md` § 2a.11 puts on the Task setup row never shows; the Results sidebar's place note has no `scratch` entry (`results/viewer.js` `PLACE_NOTE`), so *the engine's scratch of the run above* is not said | 2.24.0 review | open — the UI's, when the user takes it |
| **A4-SR** | jobset | **`StageRef` is built at three sites beside its resolver** -- `jobset/prep.py`, `jobset/_cli.py`, `jobset/runstatus.py` construct it directly, where A4 (`execution/architecture.md` § 3, § 7) names `stage_refs` and the resolver its builders | the 2.26.2 review (older than it) | open |
| **SE-F** | layers | **`script_emit` is on no floor**: it writes decks (floor 3's work), and the deck-block readers in it (`read_provenance_block`, `same_calculation`, `atom_metadata_has_something`, `apply_atom_metadata`, and the parameters fence `read_parameters_fence`, `deck_values`) are asked by floors 1 and 2 (`parse.contract`, `parse.dirs.record`, `parse.dirs.setup`, `runs`) -- the readers belong with the blocks' grammar (`deck_record`, L1), and `script_emit` then on floor 3 | the 2.26.2 review | open |
| **BENCH-2** | engines / bench | **BENCH-MARKS' field list is a second, hand-kept source**: `script_emit.SIESTA_BENCH_FIELDS` (e.g. `BlockSize` `pow2` there, `int` in the catalogue); only the type is checked against the catalogue (`tests/test_template_declarations.py`) | 2.24.0 docs sweep | open |
| **BENCH-SHELF** | engines / bench | **Two benchmark findings that only a real shelf on a cluster can confirm** (the six-agent review, 2026-10-24; both read off the code, neither seen on a run): **(I)** a shelf walks its trials inside one job (`submit.py`), and each trial's monitor reads the job's cgroup memory counter (`monitor.py`, `memory.peak` / `memory.max_usage_in_bytes`), which `utilisation()` prefers — so every trial after a shelf's first may report the shelf's peak as its own, labelled *kernel counter* ([`bench-summary.md`](?doc=web/bench-summary.md) § 3 says the peak is the trial's own); **(J)** the measurement pins (`template.py`, `prep_inputs.py`) leave the run's 600 s `wall_slack` in force, so on a 15-minute queue (`htc`/`debug`) each trial has 300 s of compute and a trial starting after ~5 min may stop `OUT_OF_TIME` at its first check, read failed and never ranked ([`running-a-job.md`](?doc=execution/running-a-job.md) § 3.6). | six-agent review, benchmarks reader | **unconfirmed** — to confirm, one 2-trial shelf on Sol's `htc`/`debug`: the second trial's `UTIL-BASIS` peak against its own `util.csv` (I), and whether the second trial stops at its first check (J); a fix only after that |
| **DOC-SMALL** | various | **Small text errors in code found by the docs sweep**: `molbuilder jupyter --help` says a restart "does not destroy notebook state" while `serve_daemon` stops it; `web/app.py` cites "deployment.md § 1.3" (no such section); `ui-contract` § 1's stylesheet order broken in the index, spectra, modify and results pages; the ruler notice written twice (`modify/viewer.js` ~669, `periodicity.js` ~268); `checkpoint.js` ~982 cites L4 where the rule is L3; the zip export carrying an engine's scratch is § 1v's; dangling codes (D3–D5, `M5:`, the unit-11 review) in launch's comments | 2.24.0 docs sweep and review | open — step 3 (tidy) |
| **RL42** | execution / launch | **R-L42 (a run script missing) is checked by two builders only** (`_plan_stage` run here, `_bench_trials`): a group's walk, a run of points' walk and a queue send meet it at run time — make it one check where every submission is built, or drop it as a launch-time re-check (the user's rule: prep decides, launch runs) | 2.24.0 launch review | open — the user's word |
| **GATHER-WARM** | transport | **A warm `--gather` relaunch of a run of points takes its done points over with their old inputs** (N3 round 1, U1): the run's points then rest on two upstream runs, and the record reads every point's device facts from the run the first point names (`record.gathered_device_run`); the deck-agreement gate holds, so the difference is at SCF tolerance. Whether a warm `--gather` may take points over at all is the user's | N3 | open -- the user's word |
| **E13** | engine / science | **The first real junction walk — BDT–Au on Sol. DROPPED BY THE CONSOLIDATION.** Named in the roadmap and again in `transport-design` § 7's *"order of proof"* as the run that follows P6. plan.md has the machine-blocked infrastructure (E5 — archived 2026-09-10 — and E7) and the browser and deck walks (W12 — archived 2026-09-10 — and E11) but no row for the transport composite's first real science run | roadmap · `transport-design` § 7 | **the Sol half only** — after § 5u step 4; the workstation half ran 2026-09-29 (the Au–BDT–Au ladder to T(E) on `claude-au-bdt-au`, TD8's acceptance ladder) and is archived *(2026-10-08)* · *Re-checked 2026-10-09:* open — the Sol half only; the workstation half closed by TD8 and Q15's run E. |
| **E14** | science / references | **Reed 2006 and Stokbro 2003 into `science/references.bib`.** `engines/transport.md` § 8 cites both (and Solomon 2008 beside them); none is in the `.bib`, and the rule is that an entry is added when a passage cites it. The 2026-10-08 archive reason — *no live document cites them* — was untrue | roadmap § 7.3 | **reopened 2026-10-09** — each entry checked against the paper before it goes in (§ 7.3) |
| **V1** | engine / science / web | **The vibration calculation — one path on two engines.** The contract is [`engines/vibration.md`](?doc=engines/vibration.md) (§ 10 says what stands); the science is [`science/normal-modes.md`](?doc=science/normal-modes.md); the design and the first audit are archived (`archive/2026-09-24-*`). Built 2026-09-21 → 24: one harmonic path with the rank rule and its gate against PySCF, one mass convention, stationarity on the free atoms, the runs write the pair, the Hessian over the free atoms with the two measured corrections, the SIESTA arm (deck on a sorted copy, `atom-permutation.json` with its key, the `.FC` reader, `summarize run`, the warm-file section, the kind's start state), the equilibrium block optional, the wrapper's banner. **Open, the sub-rows:** | `engines/vibration.md` § 10 | open · *Re-checked 2026-10-09:* a container: V1.11, V1.15 and V1.38 narrowed, V1.25 unblocked, V1.31 archived. |
| ↳ | **V1.9** | **The structure identity rule — what is left** (the rule settled in discussion 2026-09-22; its first half, the pin, built 2026-10-14 — 2.9.6, M2m: the geometry hash over the atom lines as written and the cell and axis kinds, never the comment line, in every pair's sidecar, compared by the one read door, prep refusing a changed structure and the browser's load door saying so; the row as it stood is in [`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md)). **Open:** a *broad hash* over the labels, regions and identity columns, for provenance; the check when two runs are joined (the geometry hash only); and **derived structures** — a reordered copy minting its own identity and recording its parent's with the permutation (`atom-permutation.json` records the permutation and no parent today). Two rulings disagree and are the user's to settle when this is taken: V1.9 says a broad-hash difference is reported, never blocking; W39's customized parameters, as agreed 2026-09-27, enter the hash and prep refuses when it moves ([`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md), M2m, W39). `spectra.json`'s own `structure_hash` is V1.39's (decided 2026-10-14: retired, the result states its structure) | `model/structure-molstruct.md` § 3 · `engines/vibration.md` § 6.10 | open — the pin done 2.9.6 · *Re-checked 2026-10-18:* `atom-permutation.json` is retired (RMP R5); the order record is `Structure.atom_order` (`structure.py:736-743`), key and both directions, no parent. What is left is the ruling (broad hash, join check, a derived structure's parent) |
| ↳ | **V1.11** | **Untested and open**: the reduced Hessian with a GPU mean field; a composed permutation for a structure sorted for two reasons (ruled one record, no caller yet); `transport/compose.py` writing its record through `write_permutation` and stamping its key (U10; reading through `read_permutation` done 2026-09-28) | `vibration.md` § 4.4, § 5.2 | → **§ 5u step 3** *(2026-09-29: compose writes the permutation by hand and calls the sort directly, so no key is stamped; `sort_by` / `write_permutation` exist for it)* — **the compose half built 2026-10-02 (M5 step 3)**: it sorts with `sort_by(…, "transport")` and writes through `write_permutation`, the key stamped (`test_transport_prep.py` reads it back) · *Re-checked 2026-10-09:* the compose half done (`transport/compose.py:757` `sort_by`, `:831` `write_permutation`); the test the row names (`test_transport_prep.py`) went in `93977f5d`, so no test reads the transport permutation key back. Untested: the reduced Hessian with a GPU mean field; one record for a structure sorted for two reasons. · *Re-checked 2026-10-18:* the compose half's words are stale — `write_permutation` is gone with RMP R5; compose calls `sort_by(relaxed, deck_sort_key("siesta", "transport"))` (`transport/compose.py:746`) and the order rides `atom_order`. Left: the reduced Hessian with a GPU mean field, and one record for a structure sorted for two reasons — both need a run or a design |
| ↳ | **V1.12** | **Not done**: a mode-by-mode intensity cross-check against an external code (Gaussian / ORCA / Turbomole); absolute intensities carry the caveat until then | `vibration.md` § 9 | open · *Re-checked 2026-10-09:* open (`vibration.md` § 10). |
| ↳ | **V1.14** | **From the UI walk of 2026-09-23, not this kind's**: the Molbuilder tab's save prompt doubles a typed suffix (`x.xyz.xyz`); the `#`-labelled provenance regions (`O#`) warned as unconsumed on every SMILES-built molecule — needs a ruling; the vacuum notice advising 8 Å on a gas-phase PySCF run | unification audit § 1.18 U3, U4, U7 | **the suffix done 2026-10-13** — the door adds the extension to a name typed without it and takes one typed with it, any case (`molview-doors._askProjectPath`, both the structure and the picture; `projects.md` § 3; `test_molview_files_door_js.py`, mutation red); *open:* the vacuum advice · *Re-checked 2026-10-09:* the doubled suffix stood — `lib/projects/molview-doors.js:114-117` appends `.xyz` to a typed `x.xyz`, the save route writes the path as given; the 8 Å vacuum advice is engine-blind (`cell.py:398-405`); the `#`-label warning is settled by *labels are the user's* — nothing to build. · *2026-10-18:* the vacuum advice `cell.py:397-408`, engine-blind still |
| ↳ | **V1.17** | **Four held systems through the whole road**, today rank rows only: acetylene with both carbons held (the collinear trap end to end), NH₃ with its three hydrogens held (nothing removed), an empty held list reproducing the free path *exactly* (what makes "the free case is the held case with nothing held" a fact), the water dimer with one molecule held (the over-removal guard) | `vibration.md` § 9 | open · *Re-checked 2026-10-09:* open (`vibration.md` § 10). |
| ↳ | **V1.24** | **Mode matching across runs** (needs a design): the overlap of eigenvectors in the shared free subspace, mass-weighted — the active-region convergence test (Models A/B/C) and the PySCF-against-SIESTA comparison of one molecule are the same calculation | `science/normal-modes.md` § 4b.6 F, § 4b.3 | open · *Re-checked 2026-10-09:* open — M10. |
| ↳ | **V1.26** | **`Δρ_ν(r)` maps** (decided with M10's design, step 6): SIESTA's density grid at `±Q_ν`, differenced — the discussion's intermediate quantity before any oscillator strength on a metal | `science/normal-modes.md` § 4b.7 | open — M10; its decision is owed · *Re-checked 2026-10-09:* open — M10; its decision owed. |
| ↳ | **V1.27** | **The PySCF probe** (decided — W42's D8): five points per mode instead of two; the coupling per zero-point amplitude `g_ν = ∂ε/∂Q · √(ħ/2ω)` in meV written beside `ΔE/(2A)`; a molecule-projected frontier quantity for a cluster whose HOMO and LUMO are metal states. *Decided with W42 (D8, 2026-09-28): the probe runs on the thermal nodes* | `vibration.md` § 4.8; `science/normal-modes.md` § 4b.5 F–G | open — M10 · *Re-checked 2026-10-09:* decided (D8), not built — M10. |
| ↳ | **V1.34** | **The response screen — which mode to take to transport** (decided with W42, 2026-09-28): the vibration kind's half of W42 — each mode's character at load, the charge each mode moves from the force-constant run, the frame set at thermal nodes (V1.25), the PySCF probe on them (V1.27) | `vibration.md` § 5.10 · `science/normal-modes.md` § 4c | open — M10 (W42 decided) · *Re-checked 2026-10-09:* open — M10. |
| ↳ | **V1.38** | **A δ-convergence comparison that also varies the mesh.** `engines/vibration.md` § 9 still names it as owed; V1.23 closed as built with the δ axis only (the displacement sweep), so the mesh half — the one that separates the grid's error from δ's (§ 9's H₂ measurement) — has no row. Found by the 2026-09-29 coverage check | `vibration.md` § 9 · `science/normal-modes.md` § 4b.6 C | open · *Re-checked 2026-10-09:* narrower: the sweep takes a mesh stage already (`spectra/displacement_sweep.py:8,17,152`; `vibration.md` § 5.9); what is left is the measured δ+mesh comparison on H₂. |
| ↳ | **LEAD1** | **One lead calculation for both electrodes, when they are the same lead cell** *(decided with the user 2026-10-14: "this can be done with a note to the user when the calculation was cited: the L- and R- electrode can be compared to confirm if such arrangement is scientifically correct"; then: "You, with a note", "Prep refuses, naming it"; and, asked to investigate: "mirror or translational? their contact to the bridge are on different sides")*. Replaces `engines/transport.md`'s decision *Always two lead stages*. **The condition — investigated 2026-10-14, read in SIESTA 5.4.2's own source (`Src/ts_electrode.F90`, `m_ts_electrode.F90`, `m_ts_options.F90`), its manual and the TranSIESTA/sisl tutorial**: **translation, never mirror**. One bulk lead run serves both when the right block is the left block shifted by one vector — the same atoms in the same order, the same species atom for atom, the same in-plane cell, no rotation, no reflection, no atom re-wrapped — within `TS.Elecs.Coord.Eps`, 0.001 Å (`m_ts_options.F90:380`; the manual's electrode chapter still says 1e-4 Bohr, an older value). **Why the contacts facing opposite sides do not matter**: the bulk run is periodic along the wire and holds the coupling to both neighbours; the side is only `semi-inf-direction`'s sign (`ts_electrode.F90:1892-1900` takes the coupling at +1 for −a3, at −1 for +a3); TranSIESTA builds each side's surface Green's function itself from the same input (`m_ts_electrode.F90`, "handles both the left and right one"), and each side's bias through its own chemical potential; the contact's own surface lives in the device region. **Why mirror is not**: TranSIESTA compares the device's electrode atoms with the lead's atom by atom, each relative to its first (`ts_electrode.F90:1396-1416`), no reordering, no reflection — fcc(111)'s mirror is its twin (CBA), another crystal; fcc(100)/(110) mirrors pass only once rewritten as a shift. **What molbuilder builds**: one run serves both for fcc(111) with a layer count a multiple of 3 and the two stacking starts matched so the crystal continues; fcc(100)/(110) with an even layer count; the hydrogen chain of the tests — not for other (111) starts (off 2.50–2.88 Å) nor a mirror-symmetric (111) junction. The developers do it: the manual (*identical structures … "the same files can and should be used"*, `TranSIESTA_Description.tex:46-48`), the tutorial (TS_01 `RUN.fdf`, both `HS ELEC.TSHS`), SIESTA's test `16.TranSiesta/ts_chain.fdf`. References (Crossref-checked, in `science/references.bib`): Papior et al., CPC 212, 8 (2017), 10.1016/j.cpc.2016.09.022; Brandbyge et al., PRB 65, 165401 (2002), 10.1103/PhysRevB.65.165401; Sancho et al., J. Phys. F 15, 851 (1985). **The comparison** (at citation, the two `extract_electrode_model`s in the deck's order): the atom counts; the species atom for atom — molbuilder's alone, since TranSIESTA does not compare species (`ts_electrode.F90:1118`), so a mismatch would pass it in silence; the positions by TranSIESTA's own rule, the worst atom within 0.001 Å; the initial spin, when the structure states one — antiparallel magnetic leads (Ni) need two runs. A mismatch names the atom, in your numbering, and by how much. **The design**: the note at citation, *the same lead cell: one lead run can serve both* or *they differ: <the first atom, by how much>*; a transport template choice, off by default, the person's, making one lead rung serve both (both `TS.Elec` blocks naming its `.TSHS`); asked of electrodes that differ, prep refuses, naming the first difference. **Also to check**: `molbuilder modify --help`'s example (`registry=B:+z` / `registry=A:-z` on Au(111) 3×3×2) is not a shifted pair, and may not continue the crystal across the cell's boundary — to be confirmed with the seam check (`classify_seam`). Contract text first (`engines/transport.md` § 3.1 and the decision table), then code | `engines/transport.md` § 3.1 | open — new functionality, after RMP |
| **W33** | structure / engines / web | **THE ENGINE OFFSET — one placement rule for every engine** *(user, 2026-09-25: "always adjust it before sending to siesta or other engines that the coordinates of all atoms are centered inside the cell"; "we don't have to have special logic to treat isolated, periodic, transport axis_info differently")*. Every engine receives the design coordinates + `engine_offset` with the cell at the origin; the offset is computed from the cell and every atom (the fractional span centred, never re-wrapped), recorded in every deck, and stated 0 for an engine's own output, so the Results tab draws what the engine had. Retires `cell_origin` (an origin the person assigns is kept, stored as the offset), the derive-on-null corner and its per-axis rules, calibrate, and the two hand translations. Found by the fake-junction ladder: a TranSIESTA device refused on a flush corner, a Results box the engine never had, the molwatch step-0 jump (the 2026-09-25 audit's finding X4 — not § 2's row X4) | `model/structure-periodicity.md` § 6.0 · § 5q | **P0–P3 done** (P3's T3 on 2026-09-27, `tests/test_results_export_e2e.py`; its dev-server check on 2026-09-25)**; P4 open** — P5 closed by TD8, the Au–BDT–Au ladder run to T(E) on 2026-09-29 (§ 5q.6 is the per-phase record). **D1–D16 settled** (§ 5q.8) · *Re-checked 2026-10-09:* P4's three items (§ 5q.6) — M4. · *2026-10-18:* `structure.py:2341-2360` `to_ase` still emits design coordinates with `resolve_cell()`; `validation/__init__.py:135,156-158`; `wizard.py:394-398` |
| **W34** | science / engines / web | **THE ELECTRONIC STATE — charge and spin as one answer per calculation** *(user, 2026-09-25: "put spin and charge setup in to a unified framework so that these information can be produced consistently and systematically for different engines"; "investigate holistically ... from template to validation ... a design gap/framework level investigation rather than a patch"; "documentation should have an explicit discussion on species with these properties")*. Four engine-neutral template items — `net_charge`, `spin_treatment`, `unpaired_electrons`, `method` — resolved once with what the structure adds (the charge's source, the electron count, finite or repeating) and read by every deck writer, check, hand-over, form and read-back; restricted-open explicit; a species-by-species chapter. Found by the transport ladder's rung reports (a gold lead and a formate ion both told "switch to open-shell") | `science/chemistry-correctness.md` §§ 2a–2b · § 5s | **P1–P4 built** (M6, 2026-09-28/29; decisions 1–9, § 5s.2); **P5 open** · *Re-checked 2026-10-09:* P5 open — the TBtrans two-channel half built (`transport/record.py:157-165`, through `tbtrans.transmission_files`); no reader of SIESTA's charge and moment, none of PySCF's ⟨S²⟩. · *2026-10-18:* still no reader of SIESTA's charge and moment or PySCF's ⟨S²⟩ (`parse/`, `pyscf/`, `spectra/`) |
| **W35** | parse / execution / web | **THE RUN RECORD — what ran, with what, and how it went, for every run; one SIESTA-family reader; the transport report** *(user, 2026-09-26: "include in the report of transport results ... all indicators that can provide scientific information, progress tracking and symptom of non-convergence"; "parameters used for calculating transport should be similarly reported ... check what we did for the other tasks, and find the best way to honestly and fully record what is the computation setup and scientific setup, and how result evolves in the record, and output report"; "fdf parser need unified upgrade"; "get the framework and code unified and finalized")*. One record per attempt, composed on read — computation, setup (default · asked · engine used), the deck, every iteration of every phase from one grammar, the verdict with its symptoms — read by a Run panel for every kind and by a transport report built from its rungs; the NEGF contour stated rather than left to TranSIESTA's fallback. Found by a device that diverged for eleven hours with every reader saying otherwise | `model/parse.md` § 5d · `engines/transport.md` § 2a.12 · § 5t | **P0–P1 done (`a4c1d28e`); decision 10 done — the monitor, the wrapper's endings and the report fields through the framework; P2: the record and the Run panel built (2026-09-27), the rest below** · *Re-checked 2026-10-09:* M3 — P2's remainder, P4, P5, P6 (§ 5t.3). |
| **W41** | engines / results | **A SIESTA vibration's finish, the remainder** *(user, 2026-09-28: "siesta only gave some optimization ...")* — the finish built (M2b′, M2b″: the job derives its spectrum, `mb_vibration.pyz`); the measurement and the rulings are in the 2026-10-09 archive. **Left** (each re-read 2026-10-18): ~~the force-constant `.out` titled *"SIESTA optimization"* — the title guessed from the suffix~~ — the heading takes the engine from the folder answer now (`lib/inspectors/trajectory.js:83-97`, `ae945ccf`), but it is still *\<ENGINE\> optimization* for every file the inspector claims, no calculation kind read; Makov–Payne run by hand (`jobset/engines.py:100-115` emits the script, nothing runs it); the finish parsing the whole output where FC step 0 suffices (`spectra/siesta_vibration.py:262-265`); PySCF's `ir_fd_step_ang` — the reader re-emits it `null` (`spectra/results.py:1046`) but the deck's own write omits it off the finite-difference route (`pyscf/vibration_emitters.py:808-814`): the writer states it, under the explicitness rule; I7's sorted positions and the ⚠ beside an in-tolerance stationarity sentence — located nowhere on 2026-10-18 (no front-end file mentions stationarity but `fc-sweep.js:90-98`, a yes/no column) — dropped unless seen again. Re-running a failed finish is settled by rule (no recovery machinery; launch again) | `engines/vibration.md` § 5.5, `web/results.md` § 0.3 | open |
| **W42** | engines / science | **FROM A MODE TO THE CURRENT — which vibration to take to transport** *(user, 2026-09-28: "think about this and how the spectrum calculation can be enriched by the additional electronic state/PDOS calculation, and propose an improvement in the vibration calculation contract/document/design and see how we can move forward ... this is after all the other issues are done, so this is just a draft for next step, do not implement")*. The science, `science/normal-modes.md` § 4c (proposed): at a metal the molecule's orbitals are resonances and the junction has a Fermi level, so the current sees `T(E)` near `E_F`, not the HOMO–LUMO gap; a mode acts on a level `ε_r` or on the coupling `Γ` (the contact modes, whose exponential coupling gives a large positive curvature); a DC measurement, thermal or light-driven, sees the second derivative, `⟨ΔG⟩_ν/G = ½ T″σ²/T₀`, and the two scores — the level shift at `σ`, the conductance change — are kept side by side; three frames at the Gauss–Hermite nodes of the thermal distribution give the curvature and the second-order average at once; one transmission at the base screens every mode by the rigid-shift estimate `T″ ≈ g_ε² ∂²T₀/∂E²`; the free atoms' projected DOS and their own states (the extended molecule's MPSH) stand in for HOMO and LUMO. **The design, proposed:** `engines/vibration.md` § 5.10 — ① each mode's character (element shares, the free part's whole-body share, the bond stretches at `σ`), derived at load, no calculation and no new label; ② the charge each mode moves between the held anchor and the free part, from the force-constant run's own displaced SCFs; ③ V1.25's frame set for many modes at the Gauss–Hermite nodes — *made by V1.25's separate procedure, each frame's parameters in the structure's `customized` section (TD10, 2026-09-29)*; ④ an `electronic` kind — one SCF per frame, each job's finish writing `E_F`, the charges, the projected DOS and, with sisl, the levels; ⑤ `summarize` over a frame set, electronic or transport, writing `<label>.mode-response.json` — slope, curvature and thermal average per mode, levels followed by overlap, the rigid-shift estimate beside the explicit value; ⑥ the ranking table in that record's presenter; the PySCF probe on the same nodes (V1.27). Folds in V1.24–V1.27. **Decided with the draft** *(user, 2026-09-28: "the anchor is the fixed atoms. we don't specify the anchor, the atoms is labeled already with fixing or unfixed")*: D1 — the anchor is the held atoms and every projection is over the free atoms, per element, with no new label; D2 — then not needed, the result already carries the free and held atoms. **Decided 2026-09-28, the user's answers:** D3 — yes, the charge each mode moves, *"and result presentation should include the correct data and UI design to present it"*; D4 — yes, an `electronic` kind, *"and make it clear the difference between pySCF and siesta, from setup and calculation and presentation"*; D5 — sisl in every SIESTA env, *"so it is always available when we need them ... fix the install-env and other env related setup to get this consistently done - do not reinvent wheel as we have a full system of install implemented"* (built with the M2b‴ milestone); D7 — the thermal nodes, as recommended (the displaced structures the electronic and transport runs compute on — not the animation, which the user asked about); D8 — the PySCF probe on the same nodes, as recommended. **Taken as recommended unless the user objects:** D6, one displaced-structure set for many modes. **Registered with it:** E2 — a vibrational-only result records no pressure (built with M2b‴); E3 — a command that re-runs a failed finish on its attempt, the user's yes owed. **Order, once agreed:** P0 the decisions; P1 ① (no engine); P2 ② — measured first, whether SIESTA prints the populations at every FC step; P3 ③ with its displacement statement, checked by the frame set's citation promises (W32 ②); P4 ④ and ⑤ for the `electronic` kind — the frame axis for one kind first; P5 transport on the frame set (W32 ③–⑤, behind W32's gate) and ⑤'s transport half; P6 ⑥. Validation H₂, then the carbon-chain junction, then Au–BDT–Au. **Found while drafting, fixed with M2b‴:** the SIESTA deck's commented population hint spelled the retired `WriteMullikenPop` (`siesta/input.py`); it names 5.4.2's `Charge.Mulliken` and `Charge.Mulliken.Format` | `science/normal-modes.md` § 4c · `engines/vibration.md` § 5.10 · `engines/transport.md` § 2a.9 | **draft 2026-09-28 — decided D1–D5, D7, D8; D6 as recommended; after the current work order** · *Re-checked 2026-10-09:* decided, M10; V1.26's decision owed; the frame set it uses is V1.25's — **D6 (*one set for many modes*) reversed 2026-10-09: one mode a set, each its own transport calculation**. |
| **W45** | execution / checkpoint | **Checkpointing's owed half.** The save is always (prep and Task setup's Save call `checkpoint.save_before`, `prep.py:1890`, `:2851-2856`; `checkpointing.md` § 12 A3 ✅). **Left:** a `checkpoint verify` verb (the CLI has init, save, list, tag, restore, config — `cli.py:2885-3132`; `verify_archive` is reached only from restore, `checkpoint.py:839,1695`); invariants S2, S3, S4, S6 and L8 without a test (`checkpointing.md` § 13.4 points here), A3's test and S5's run-layer half with no row before this one; `project-layout.md`'s invariant 16 and § 6.1 say the archive does not reach a bench trial's `.DM`, against S1 (*every regular file goes to git or the archive*), which the code follows — the layout's sentence is corrected to S1 when this is built; § 12's S2/S4/S6/L8 still read *"needs the layout / the description"*, which exist | `execution/checkpointing.md` § 12, § 13.4 | open · *2026-10-18:* no `checkpoint verify` verb (`cli.py:3017-3265`); `verify_archive` (`checkpoint.py:921`) called from restore alone |
| **W47** | engines / execution | **A person's own deck text: § 5e's engine additions and the USER-CUSTOM zone's carrier.** No `user_custom` item exists, so in the staged path the zone is emitted empty and a person's text does not survive (`template.md`, `job-contracts.md`, `worked-example.md`'s gap 12); § 5e (*a person's own engine text, as an input*) is a design with no row and says the zone's defects are *"listed where they belong"* — they were listed nowhere | § 5e · `template.md` § 12.1 row 1 | open — § 5e not started · *Re-checked 2026-10-09:* open, § 5e not started. `engines/template.md` § 9.2 disagrees with § 5e on four points — the carrier (a catalogue item `user_custom` against a separate input), the zone's role, stage overrides against a per-addition switch, the placement; § 5e is the user's ruling of 2026-09-05 and owns it, its open question 1 leaves the zone's fate to its design. `template.md` § 12.1 row 1. · *2026-10-18:* no `user_custom` item in the catalogue; the zone emitted empty by `script_emit.emit_user_custom_placeholder` (`:601`) |
| **W50** | front end / engines / execution | **EVERY PARAMETER OF THE STRUCTURE OPTIMIZATION, SPECTRUM AND TRANSPORT WORKFLOWS, CONFIRMED END TO END** *(user, 2026-09-29: the quote in M11; "this is not a poking test, a static review followed with e2e validation")*. Found the day it was asked: the PySCF vibration refused a periodic structure where the engine's own rule notes it (fixed, `762ca298`), and gold on def2-SVP received no core potential while the hint stayed silent on a false belief about PySCF (fixed, `5a8133cc`) — each a parameter whose chain nobody had read end to end since the Spectrum tab gained its second engine (2026-09-24). **The method**, as the structure optimization's full-text review (`archive/2026-08-14-template-execution-review.md`): (1) a static review per track — Structure optimization on SIESTA and on PySCF, Spectrum on PySCF and on SIESTA, Transport's five rungs — every file of the chain read in full (the tab's HTML and JS, the routes, the catalogue rows and `template.py`, config / resolve / prep / validation, the deck writer) and each parameter traced from its control to its line in the generated script and to where the engine's own source reads it, nothing changed and nothing run; (2) every finding re-read against the code and fixed at its owner, the contract first; (3) the road in the browser per track — the tab, the hand-over, Task setup, the printed `prep` / `launch`, a small run, the Results tab | `web/spectra.md` · `engines/vibration.md` · `engines/pyscf.md` · `engines/siesta.md` · `engines/transport.md` · `web/task-setup.md` | **started 2026-09-29** — step (1) done the same day: five reports, every defect verified; step (2) is § 5w — the findings grouped by the mechanism that produced them, each class one declaration in the template and one door (K1–K17), approved 2026-09-29 — K6 first · *Re-checked 2026-10-09:* a pointer — K6 done; the remainder is Q9, § 5w. |
| ↳ | **V1.15** | **Recorded, not in scope**: the transport connection (displace along a mode, then transport; the electron–vibration coupling from `FC.Save.dHS`) and Born-charge infrared on SIESTA — each a feature to design as one | `vibration.md` § 5.6 | not started · *Re-checked 2026-10-09:* narrower: displace-then-transport is V1.25 + Q17; left: the electron–vibration coupling from `FC.Save.dHS`, and Born-charge IR on SIESTA. |
| **W1** | front end | **The document tier (step C).** `html, body`, `header`, `button`, `footer`, `textarea` genuinely differ per page; the `*` reset is already deleted. Blocked on a browser pass over all pages | `css-system` § 4C | partly · *Re-checked 2026-10-09:* open — M8; re-derive at its start. |
| **W2** | front end | **One home per component (step D).** `.card`, `.status`, `header .tagline`. **One value to settle first:** `.card`'s padding is `var(--space-md) 18px 18px` and 18 is off the 4px grid the contract declares — moving it shifts every page by 2px | `css-system` § 4D | not started · *Re-checked 2026-10-09:* `.card` has one home now (`lib/page-shell.css:284-288`, tokens since `a09a8d60`); left: `.status` in two sheets (`spectra/style.css`, `page-shell.css`), `header .tagline` in four. · *2026-10-18:* `.status {` in four sheets (`spectra/style.css:48`, `modify/style.css:74`, `structure-optimization/style.css:57`, `lib/page-shell.css:512`); `.app-header .tagline` in two (`page-shell.css:86`, `spectra/style.css:34`) |
| **W3** | front end | **Per-page token/namespace passes (step E)**, one page per commit: `spectra`, `structure-optimization`, `transport`, `results`, `documents` | `css-system` § 4E | partly · *Re-checked 2026-10-09:* open — M8; re-derive at its start. |
| **W6** | front end | **The editor module.** The loader half is confirmed and accurate: `lib/codemirror-load.js` is the one loader, two of three surfaces import it, and `lib/inspectors/markdown.js` still hand-rolls its own pair — *definitions* at `markdown.js:31` and `:38` (this row cited only the call site; 10 CodeMirror references in the file on 2026-10-08). **The sheet number was wrong twice over, re-derived 2026-09-07: 21 rule blocks / 60 declarations, not 30 and not 40.** The original 25+4+1 was never reproducible as a block count either — `projects-sidebar.css` has held 16 CodeMirror blocks at every commit back to 2026-08-28. The caps (1500-line selection, 1 MB view-only) are on `preview.js` alone, confirmed | `editor-module` | partly · *Re-checked 2026-10-09:* `markdown.js:45-50` loads CodeMirror itself. · *2026-10-18:* still `markdown.js:45-50` |
| **W13** | front end | **Raw px/rem literals — re-derived a THIRD time, 2026-09-07, and the definition finally holds still.** 160 / 740 reproduce exactly, but only because the regex reads raw file text *including comments*. Counting literals **in declarations**: **133** across the eight page sheets, **650** in `lib/`. Two things the row hides: `lib/tokens.css`'s 44 literals ARE the scale definitions — the token layer, not violations — and `lib/molview/molview.css` alone is **252**, 39% of the whole `lib/` figure. So "lib/ carries 740" is really "MolView carries 252, and the rest of lib carries ~400". 777 → 384 → 160/740 → 133/650 are four scopes, not four measurements | roadmap § 7.4c | partly · *Re-checked 2026-10-09:* open — M8; re-derive at its start. |
| **W15** | front end | **Sealing the MolView module's internals and finishing the ES-module conversion** — both **browser-verified** before they count. ~~Plus routing the CLI through the shared codec~~ — **DONE 2026-09-22**: `xv2xyz` (`cli.py:968`, a pair on 2026-10-08) was the last CLI converter writing a lone geometry, and the writers no longer take a path at all, so the class is closed rather than swept (`model/structure.md` §§ 2.3, 2.4). Still open here: exercising the last annotation-channel kind. Re-measured 2026-10-08 (first 2026-09-07): `lib/molview/` has **no `_seal.js`** where `spectrumchart/` and `vibrationview/` both do, and `results.html` loads 5 scripts as classic `<script defer>` against 10 on `type="module"` (seven against two in September) | `web/overview.md` § 4, § 6 | partly — the ES-module half's remaining list is `web/overview.md` § 6 (the `spectra` engine on it: `lib/spectra/core.js` loads as a classic script, `results.html:250`); the sealing half is re-measured at M8 — the shared 3Dmol embed it named (`lib/viewer/`, task #104) is gone *(2026-09-29)* · *Re-checked 2026-10-09:* no `lib/molview/_seal.js`; `spectra/core.js` loads as a classic script at `results.html:241`; 14 more plain classic `<script>` tags are uncounted. Its *task #102* and *#103* (the classic registry and `lib/inspectors` → `presenters`, the results module and the shared primitives as modules) are cited at `web/overview.md:183-189`, `spectrumchart.md:1163-1164`, `spectra.md:652` — none done. · *2026-10-18:* `lib/molview/_seal.js` absent (spectrumchart and vibrationview have theirs); `results.html:247` loads `lib/spectra/core.js` classic; 17 classic `<script src>` and 5 `defer` against 10 modules |
| **X4** | transport / metadata | **The transport structure's metadata seam** — ①, ④ and ⑥ built (`transiesta.py:360-386` `species_order`, now I15; the catalogue's `shared`; `sidecar.py:13-16`); the measurement is in the 2026-10-09 archive. **② open**: `wizard.as_structure` builds the lead's `Structure` field by field and drops `annotations`, `info`, the identity columns and now `customized` (`transport/wizard.py:106-146`) — the lead is carried from the device's through its doors (`frame_at`, `replace`, the region subset), never rebuilt. **③ open**: `transiesta._compute_cell_from_extents` is live (`transiesta.py:171,333`; `wizard.py:392`; `test_transport_cell.py:158-173`) against `transport.md` § 2a.9, *a transport calculation derives no cell* — read which road reaches it; if none, it is deleted with its test, if one does, that road is refused by I14. **⑤** rides with V1.9 (M2m) | `engines/transport.md` § 2a.9, § 6.2; `model/structure.md` § 2.2a | ② ③ open; ⑤ M2m · *2026-10-18:* ② `transport/wizard.py:170-176` builds the structure from five fields; ③ `_compute_cell_from_extents` live at `transiesta.py:176`, called `:338` and `wizard.py:391` |
| **A1** | structure / validation / tests | **The structure-API audit's open findings — moved here 2026-09-25, when the audit was archived.** The audit (seven reviews of the structure API, 2026-09-22/23) held its findings and its cleanup order outside this list; they are the rows below and § 5r, and the audit itself is the evidence record, measurement by measurement: [`archive/2026-09-22-unification-audit.md`](?doc=archive/2026-09-22-unification-audit.md). **Every row is as measured on 2026-09-23 — re-derive before acting (§ 5a).** The origin-rule sites it found are W33's, not these | the unification audit, archived | open · *Re-checked 2026-10-09:* open — § 5r keeps the order; A1.2, A1.3, A1.7 and A1.16 archived. |
| ↳ | **A1.5** | **The pair's doors re-derive one rule.** Eleven spellings of *"is this a structure path"* (`workingcopy_structure.py`, `files.py:257`, `selection.py:99` — dead, `siesta/input.py:1872,1902`, `build.py:240,722,813`, `cli.py:93,773`, `structure.py:94`) and a fourth in the browser (`task-setup/viewer.js:1144` derives `<stem>.molstruct.json`). Rename strands labels: `water.xyz → notes.txt` leaves `notes.molstruct.json`, and the reverse adopts a foreign sidecar unchecked. ~~A `.pdb` source travels under three names — describe records `c.source.pdb`, the codec writes `c.source.pdb.xyz` (`files()` never passes `fmt`), prep looks for `c.source.xyz` → *"the structure this calculation describes is not here"*~~ — *done 2026-10-14 (2.9.6)*: every source is `<label>.source.xyz` (`source_files`), and `files()` takes the format its target names, as `write` does (`StructureCodec.format_named`, the one rule; the export door states its XYZ) | audit §§ 1.8, 1.8a, 1.18 | open (the rest) · *Re-checked 2026-10-09:* the browser derives the sidecar name itself (`task-setup/viewer.js:1053`); a `.pdb` target is not replaceable (`workingcopy_structure.py:117,330-333`); `files('mol.pdb')` writes `mol.pdb.xyz` (A1.13). · *2026-10-18:* the browser derives nothing now — `task-setup/viewer.js:1314` is a placeholder label; a `.pdb` is read and written through the codec (`workingcopy_structure.py:199-220, 371-400`). Left: the Python spellings, re-counted when taken |
| ↳ | **A1.6** | **`Structure.replace()` still hand-enumerates.** Nine fields in a literal, five more from `_carry_nonatom()`. Completeness is pinned by a test iterating `dataclasses.fields()` — but its fixture has `annotations={}`, so a `replace()` that drops annotations passes it. Deriving the carried set from `dataclasses.fields()` makes it complete by construction; `frozen_atoms` stays excluded (a property over `regions`, no storage). Read `replace()`, `_carry_nonatom()` and `__post_init__` end to end first — it is the most load-bearing method in the model (`structure.py:1091`, `:1778`; 2026-10-08) | audit § 1.16e; the 2026-09-22 handover | open · *Re-checked 2026-10-09:* `replace()` lists the fields by hand (`structure.py:1272-1336`). · *2026-10-18:* `structure.py:1511` |
| ↳ | **A1.8** | **Rules re-derived at the call site.** `estimate_partial_charges` (`chemistry.py:1392`) is label-blind (water `O1,H2,H3` → **0.0 D**; `_DEFAULT_EN = 2.20` is hydrogen's value); **twelve dead `axis_kind` fallbacks** (13 on 2026-10-08) (11 × `("isolated",)*3`, 1 × `()`, all unreachable, plus `validation/siesta.py:734–737` resolving the opposite way — and `transiesta.py`'s was `("periodic",)*3`) — pure deletion, first; the k-sampling hint measures the gap in two frames (hexagonal cell: the hint says ~5.5 Å, the perpendicular gap is 4.16, `_min_image_distance` 6.5) | audit § 1.12a–c | open · *Re-checked 2026-10-09:* twelve dead `axis_kind` fallbacks (e.g. `validation/geometry.py:116`, `validation/__init__.py:115`); `chemistry.py:1392`. · *2026-10-18:* 14 `("isolated",) * 3` fallbacks package-wide; `chemistry.py:1409` |
| ↳ | **A1.10** | **Placeholders stored as facts, and two builder defects.** The backbone check keys on `rid − 1`, so a 5P duplex is refused (*"residue 4 O3' → residue 5 P 16.74 Å"*) with the blame on `$X3DNA`; rdkit-added hydrogens carry `(1, MOL, A)` and are persisted as real identity (7 of 12 atoms); `smiles.py:155,187` and `_common.py:35,40` spell index names `C1, O3, H4…` as stated identity, so every SMILES-built molecule ships a sidecar; `_amber.py:80` warns *"requested B-form … not enforced"* on every build. The placeholder must be carried apart from the data — a shape decision before a fix | audit §§ 1.15, 1.18 | open · *Re-checked 2026-10-09:* open. · *2026-10-18:* `peptide.py:115`, `builders/backends/_common.py:36,41`, `smiles.py:155,187`, `_amber.py:80` |
| ↳ | **A1.13** | **One rule, several enumerations — and two strictnesses.** ~~Three `_CONTAIN_EPS`/`_EPS` (one unread); `cell._contains` re-implements `Structure.cell_contains_atoms`~~ — gone with W33 (`6c705058`): containment is one distance tolerance, the hand-off's; `affine`/`concat` hand-list the columns; `vacuum=-5` is accepted by the model and refused by the gate; `annotations` accepts a `str` index `regions` refuses; `set_channel` installs then validates, so a refused channel stays and breaks `copy()`; ~~`files('mol.pdb')` writes an XYZ inside~~ (done 2.9.6, A1.5); `apply_to_structure` on a partial payload resets the cell. Fix each at its owner, never at the instance | audit §§ 3, 4 | open · *Re-checked 2026-10-09:* open — ~~`files('mol.pdb')` writes `mol.pdb.xyz`~~ (done 2.9.6). · *2026-10-18:* `structure.py:1508-1509` assigns then validates; `apply_metadata_dict`'s absent-key reset `structure.py:1121-1122` |
| ↳ | **A1.14** | **The same shape, latent** (unreachable or harmless today). A second PDB reader (`builders/backends/_common.py:57–93`: `Mg → M`, `Cl → C` on a blank element column); a second deserialiser (`selection.py:131–173`; `_shared.py:124–165, 1349–1406`); eight metadata-dropping rebuilds (`add_hydrogens`, `protonate_phosphate_oxygens`, `_drop_overlapping_hydrogens`, `relieve_clashes`, `_strip_5prime_phosphate`, `select_chain`, `_patch_residue`, `_fix_methylene_hydrogens`); `describe.write_description(struct=None)`; `_reset_to_derived`'s own 1e-6 threshold; `_validate_transport_kind` reading the raw cell; `transiesta.py:601,705` bare `+ 1` into the deck; `pyscf/input.py:1575` serialising with `json.dump`; `chemistry.py`'s `_adjacency` on raw `"H"` and a second periodic table; `emit_atom_metadata` dropping a kind-less channel silently; `build.py:410/1296`; the backend set spelled in seven places | audit § 1.18 | open — its transport half is the electrode block's hand-written `+1` (`transiesta.emit_electrode_declarations`, `:432`); § 5r's order *(2026-09-29: not done by § 5u step 1, which had claimed it)* · *Re-checked 2026-10-09:* the transport `+1` sits at `transiesta.py:500,580`. · *2026-10-18:* the bare `+ 1` at `transiesta.py:371, 519, 608-609`; `pyscf/input.py:911` still `json.dumps` |
| ↳ | **A1.15** | **Rules nobody wrote down** — the stale comments are fixed (`structure.py:5-16` and `workingcopy_structure.py:26-32` read right on 2026-10-08; `structure.py:892` and `validation/sidecar.py:3–7` not re-read). To state: what the CLI's single stdout stream carries; `cell: null` in a load body; a partial `apply_to_structure` payload; a `.XV`'s companion sidecar; whether *Delete file* pairs the sidecar; `Frame.lattice` against `Structure.cell` (which frame a run artifact is in is W33's) | audit § 1.18 | open · *Re-checked 2026-10-09:* the single-stdout-stream item is answered by the 2026-10-09 ruling (a lone file is read as atoms and says so, `cli.py:421-424`); `Frame` still exists (`frame.py:64`). |
| ↳ | **A1.17** | **The tests: the count must come down, and three rules are unpinned.** Unpinned: `write(struct, "x.pdb")` producing a readable pair (restoring the pre-2026-09-07 bug leaves 460 passed); geometry-before-sidecar, both-or-neither (reversed → 441 passed); the explicit-cell centring branch (deleted → 441 passed — W33 makes it the rule and pins it). Blind or shape-asserting: ~~`test_periodicity_gate.py:1126` (inverted — fails on cosmetics, passes on deletion), `test_cell.py:202` (`"a " in message`)~~, `test_cell.py:588` (a signature), ~~`TestDocMatchesTheDoor` (blind to a shrinking `OPS`)~~ — the struck three fixed or retired with W33 (`6c705058`), which also pins the explicit-cell centring branch as the rule (T5, T1). ~~~24 duplicates in twelve clusters — 16–17 tests carry *"the default isolated vacuum gap is 3 Å"*~~ (gone, 2026-10-08). ~~`structure_hash` is still hand-built in test fixtures (18 test files name it on 2026-10-08; 15 fixtures in September)~~ — converted 2026-10-14 with V1.9: every pair a test writes goes through the codec, and the files still naming it write an older version on purpose or are the vibration result's own hash (V1.39), three matching a different error than they name | audit § 5a | open · *Re-checked 2026-10-09:* 19 test files name `structure_hash`; the front-end half of the atom-number invariant (`model/overview.md`; `structure-annotations.md` § 6 now names the real homes, `_atom.js` `toDisplay`) is this row. · *2026-10-18:* 8 test files name `structure_hash` (the spectra ones read the v4–v8 key the reader still takes) |
| ↳ | **A1.18** | **The contracts' line numbers are 15 % right.** 6 of 39 `file.py:NNN` references resolve (re-derived exhaustively, 2026-09-22); seven symbol names have never existed; five retired concepts are written as current. `model/parse.md:355` already states the rule — a line number is a pin; the fix is one mechanical pass, then the behavioural list | audit § 2 | open · *Re-checked 2026-10-09:* open. |
| ↳ | **A1.19** | **The test harness can report a green suite that is not green.** In `tools/`, open and not held: ~~`run lf` with nothing to rerun shouts NOT GREEN (exit 5 on a green last run — `exit 4 \| 0/0 ran` appears 302 times in history, so the guard gets trained away)~~ — **closed 2026-09-30** with the spread runs' review: `run lf` runs only when pytest's last-failed list names a test file here, and otherwise says *nothing to rerun* and exits 0 (`testing.md` § 6.1a); ~~`testrun.py failed` emits node-ids with a `[teardown]` suffix pytest refuses; the head line stops summing (`2/2 ran \| pass 1 FAIL 3`)~~ — fixed (`testrun.py:319`, `:234`; 2026-10-08). Same class, open: `progress_plugin.py:68`'s `except OSError` silently disables the writer and `cmd_status` returns 0 for no-data; the env canary's *DISARMED* goes through `warnings.warn` with no hook, so a run whose canary proved nothing reads like one that proved everything | audit § 0c | open · *Re-checked 2026-10-09:* open — `tools/progress_plugin.py`'s `except OSError: pass`; `conftest.py:401`. · *2026-10-18:* `tools/progress_plugin.py:68-69`; `tests/conftest.py:411-416` |
| ↳ | **A1.20** | **Residue, with the step-0 read done** (`process/code-audit.md` § 1d): ~~five `_enumerate_files` buckets with no reader, built on the Watch polling path with four directory scans~~ (gone, 2026-10-08); three legacy shim classes and the five test-only names that depend on them (one decision); ~~`sha256_of_file` (`sidecars/molstruct.py:220`), whose docstring calls it the `structure_hash` pin (X4 ⑤)~~ (gone 2026-10-14 with V1.9's pin); `selection_rules` (`script_emit.py:391-448`), a format field with no producer and no consumer; `sidecars.molstruct.load_text` (×2 on 2026-10-08 — re-read before acting); ~~the `*-electrode` convention two modules advertised and `sort.PARTITION_LABELS` could not compose~~ (gone 2026-10-02, § 0b item 3). Last in § 5r's order | audit § 5 | open · *Re-checked 2026-10-09:* open — `sidecars/molstruct.py:230,706`. · *2026-10-18:* `selection_rules` lives in `sidecars/molstruct.py:168,407` alone, out of `script_emit.py`; `load_text` twice (`parse/sidecars/molstruct.py:135`, `sidecars/molstruct.py:826-827`) |
| ↳ | **A1.21** | **Audit #2 — the rest of the tree, planned and not started.** ~70,000 of ~110,000 lines were outside the audit: `web/static/lib/` (33k — a language boundary), `jobset/` (12k — a process boundary), the rest of `web/blueprints/` (~9k), `runwrap.py` (5k), `runtime_config`/`template`/`monitor`/`checkpoint`/`task` (~9k), `config/` + `validation/` (~8k). Kept separate because the failure shape differs (one fact per side of a boundary, not one door per operation) and so does the evidence (a JS finding needs no Python fixture; a wrapper one needs a submitted job) | audit § 7c | planned · *Re-checked 2026-10-09:* open, with A14: the 44 hand-built run-file names counted 2026-09-07 predate `runfiles.RunNames` — re-measure by review. |
| **S1** | architecture seams | **`runwrap` reaches into the engines.** The wrapper writer branches on which engine it is writing for — what a cold restart clears, how the label is read back out of a deck, how the launch line is formed. Until it moves, *adding an engine edits `runwrap.py`*, which is exactly what `generator.md` § 7's *"adding an engine adds files and edits none"* exists to catch | `backend-architecture.md` § 5 (its **W1**) | **measured open** — three engine branches in `runwrap.py` and 22 lines carrying an engine-name literal (2026-10-08; four branches and 128 literals when last counted, 2026-09-06) · *Re-checked 2026-10-13:* conditions on an engine at `runwrap.py:592`, `:772`/`:816`, `:1914`, `:1962`, `:2116`, `:3171`, `:3183`, `:3432`, `:3920`; 22 lines carry an engine-name literal. Closed by the Python pass's phase 4 as proposed (Q19: one wrapper hook per engine on `jobset/engines.engine_seam`), when the user takes it. · *2026-10-18:* seven engine branches (`runwrap.py:839, 883, 2019, 2173, 3256, 3268, 4033`), 24 lines with an engine literal |
| **S13** | architecture seams | **Transport convergence sweep** — auto-vary transverse-k / `MeshCutoff` / electrode thickness and report where `T(E_F)` stops moving. `transport.md` § 2 already tells a reader not to trust a single point blindly, so the document promises what the code does not offer | `engines/transport.md` § 8 | **measured: not built, re-verified 2026-09-20.** Nothing in the tree names it at all now — even the `transport/wizard.py` comment that used to is gone, so the only record that it is owed is `engines/transport.md` § 8 and this row · **→ after § 5u step 4, unscheduled** (§ 5u.5: a sweep needs a ladder that has produced a curve) · *Re-checked 2026-10-09:* open; `transport.md` § 8 now cites this row. |
| **N10** | parse / front end | **A CALCULATION ROOT IS NOT A RUN DIRECTORY, and the Results tab has only one notion.** Measured on the real transport ladder: `jobset_status` answers 5 stages all `pending, prepped, not launched`; `run_status` -- which `/api/results/dir` calls unconditionally -- answers `running, no result file yet`, so the tab tells a person a calculation nobody launched is running. The tab offers that root its own INPUT structure as the result, and says nothing about the five stages. `transport.md` § 2a.12 has required the ladder's state, the curve with its treatment named, and the provenance chain since before the surface was built. **The predicate already exists** -- `checkpoint._is_bundle_root` -- and a second copy in `parse/dirs` would be instance 14 of § 8 | § 5c.3, `transport.md` § 2a.12 | **RE-MEASURED 2026-09-23 — the headline defect is CLOSED; (e) folded into § 5u step 9 (2026-10-08); the two-predicate question open.** § 5c.3's two premises are both stale: `/api/results/dir` no longer calls `run_status` unconditionally (`results.py:185-229` asks `calcdirs` first — `record` / `read` / `root_of` — and builds the ladder there; a container gets `status: None`), and the public owner that landed is a NEW module, `calcdirs`, **not** the `checkpoint._is_bundle_root` step (a) proposed — which is still private at `checkpoint.py:379`. So (a) is superseded, (b) is done in substance but via `place == CONTAINER` rather than a `jobset_status` ladder (`jobset_status` has zero hits in `results.py`), (f) is DONE (`inspectors/transport.js:180-205` renders the provenance chain), and **(c) the payload and (d) the ladder view are DONE 2026-09-24** (`ladder` on `/api/results/dir`, the empty-state card's table, `results.md` § 2.4); **(e) the inspector's parser → § 5u step 9** — `transport.js` no longer calls `JSON.parse` (2026-10-08); what remains of the reader is step 9's. **AND A NEW, SMALLER QUESTION:** there are now TWO predicates for *is this a calculation root* on DIFFERENT evidence — `_is_bundle_root` tests for `task.json`/`job-set.json` existing, `container_or_run` reads `task.json`'s `shape`. The § 8 duplication § 5c.3 warned about is real in a milder form, found by a browser walk that 2,800 passing tests missed · *Re-checked 2026-10-09:* `container_or_run` is gone (`runs.place_of`, `runs.py:56`); two root checks remain on different evidence — `checkpoint._is_bundle_root` (`:379-382`, task.json or job-set.json) and `calcdirs.root_of` (`:109-130`, task.json). No failure on molbuilder's road (job-set.json always sits beside task.json): the unification only. · *2026-10-18:* `checkpoint.py:372-378` (`task.json`, `job-set.json`) and `calcdirs.py:109-115` (`task.json`) |
| **F9** | execution | **A group prep prints its pseudopotential copy lines and the `geometry.h_ratio` note (a warning until 2026-10-11) once per rung** (three times each) — found on the road walk 2026-10-08, not reached by B8 | § 5x.7 F9 | open · *Re-checked 2026-10-09:* `prep_group` (`prep.py:2970`) does not de-duplicate the members' notes; its h_ratio half is A1.12. · *2026-10-18:* `jobset/_cli.py:1431-1434` prints each answer's notes; `prep.py:2745` de-duplicates nothing |
| **F11** | execution | **The wrapper's kill line is unseen on a killed point**, and two more mechanisms have not been seen on the road: a **cold sweep** (`launch task --stage device --cold` on a finished sweep) and the **take-over hop** (a point taken over from a run that took it over). One walk closes all three — ~45 minutes of this machine (a device point ≈ 15 min), launched through Task setup and left to run; **waits for the user's go** | § 5x.7 | open — the user's · *Re-checked 2026-10-09:* the cold sweep is seen on Q15's run E (`tests/test_transport_on_a_real_junction_e2e.py:238-240, 396-409`); the kill line and the take-over hop are not. |
| **T-DOCERR** | tests | **Five or six setup errors in the documents' test set, twice, cause unknown** — the set `tests/test_doc_claims.py` plus the `doc`/`toc`/`link` files, run right after a document edit on 2026-10-14 (6 errors, then 5); three runs since passed, a new archive file added on purpose among them (so not `toc.json`'s write-back, which is atomic). The error text was lost both times: the harness records setup errors with their reason in the run's own file (`tools/progress_plugin.py`), but `status --fails` was read filtered for `ERROR` — it prints `FAIL` — and a rerun of the same batch overwrote it. **Next time: read `testrun.py status --fails` whole before any rerun.** The set's browser fixtures (`test_docs_tab.py`, `test_structure_envelope_protocol.py`) are the only ones in it that can fail at setup; the machine was at load 18 — a hypothesis, unproven | `process/testing.md` § 6.1a | open — step 3 |
| **T-ENV** | tests | **Installer tests answer the environment manager in-process**: `tests/test_envs_one_answer_about_an_env.py` (H6, `dispatch_into_env` replaced to return a typed PySCF verify answer) and the stubs `tests/test_envs_install.py` writes -- a command's answer swapped in-process, which tier 1 forbids (`testing.md` § 0); judged under § 5y's rule in their own pass | § 5y | open · *Re-checked 2026-10-09:* open (`tests/test_envs_one_answer_about_an_env.py:182`). · *2026-10-18:* `:182-184` |
| **PYF-T** | code hygiene | **The tests, `tools/` and `scripts/` are not pyflakes-clean**: 56 findings — 52 in tests, 3 in `tools/`, 1 in `scripts/` (47 unused imports, 7 unused locals, 2 f-strings without a placeholder), measured 2026-10-13 with `python -m pyflakes $(git ls-files '*.py' ':!:molbuilder/*' ':!:tests/data/*')` — the package is clean (`conventions.md` § 1) | Q19's Python P1 | open — each read before it is removed: an unused import in a test can be a fixture pulled in by name |
| **L2** | tests | **`tests/field/test_ask_the_target.py` fails instead of skipping when invoked outside its batch** (`. does not read as a machine record`: the record path is unset) — a field test gates itself on its backend (`testing.md` § 3) | B8 round 2 | open · *Re-checked 2026-10-09:* open — `tests/field/test_ask_the_target.py:49-51` asserts instead of skipping. · *2026-10-18:* `:44-45` |
| **R11** | execution / tests | **Launch's test-only renderer fallback.** Three test files call the wrapper renderer; two rely on its fallback, and both hand-write `environment.json` — a hand-built record `testing.md:54-55` retires: `tests/test_launch_door_gate.py:37-46`, `tests/test_warm_file_inventory.py:30-55`. Then `machine_record` becomes required in `render_run_wrapper` and the fallback goes. *Re-derived 2026-10-17:* the header's own fallback is gone — the header is rendered at prep only, for the target's record it is handed | § 0b (W54) | open — judge the two tests first · *2026-10-18:* `tests/test_launch_door_gate.py:39`, `tests/test_warm_file_inventory.py:33,54`; `runwrap.py:1853-1859` |
| **W57-G** | execution / tests | **Rules no test holds since the W57 retirements**: a killed run leaves no conclusion marker (an engine-tier kill); a stage's `overrides` (no road key for it); the PySCF effective-parameters fence (an engine-tier assertion on the H₂ log); `set -e` across the preamble; a warm retry's own launch record (`retry_of`, 0 test hits); MPS and ranks-per-GPU (only the `run_sh_lacks` side, `gpu_contract.toml:188`) | W57 (2026-10-07) | open — each an assertion on a real run of M2n's batch, never a fake |
| **W57-M** | execution / PySCF | **A stated PySCF memory cap reaching `max_memory`** — only the unset rows exist (`launch_values.toml:206-216`); none for a stated cap reaching `max_memory = N` (`pyscf/input.py:494`) | W57 | open · *2026-10-18:* `tests/data/launch_values.toml:224-228` covers `ulimit -v` alone; `pyscf/input.py:502` emits the cap |
| **W57-T** | tests | **Tests built on hand-made inputs, for retirement**: `tests/parse/test_tbtrans_out.py` writes empty files named by TBtrans's rule (`:13-22`); `tests/test_jobset.py` builds job sets by hand (`:56-59`, `:107-115`) — `testing.md:50-55` retires both, never sweeps them; and the ten test files that render through `spec_for` + `render_deck` directly, swept to prep's road | W57 | open · *2026-10-18:* `tests/parse/test_tbtrans_out.py:16-18`; `tests/test_jobset.py:54-56,113,130`; `render_deck` in ten test files |
| **W57-O** | execution / records | **Old-data tolerances**, each an explicit error under *build molbuilder, don't clean up handcrafted input*: `record.py:1323-1324` skips the conflict check when a copy names no machine; `script_generation` refused by name in three doors (`runtime_config.py:705-715,958`; `jobset/machine.py:137`; `envs/initconfig.py:253-259`) — one door; `Environment.from_dict` drops unknown keys (`record.py:363-398`) | W57 | open · *2026-10-18:* `scheduler/record.py:1551-1557`; `runtime_config.py:714`, `jobset/machine.py:145`, `envs/initconfig.py:257`; `record.py:467` |
| **W57-K** | execution / library | **A kind left to a default.** Fifteen functions default their kind to `"optimization"` — `template.py:1003`, `siesta/input.py:555`, `script_emit.py:1043`, `pyscf/stages.py:33`, `validation/siesta.py:648`, `validation/__init__.py:122`, `validation/stages.py:49,117`, `pyscf/input.py:171`, `_shared.py:540`, `siesta/stages.py:91`, `kmesh.py:130`, `validation/siesta.py:197`, `validation/pyscf.py:166`, `_shared.py:711` — every caller on the road states it; ~150 calls, most in tests, lean on the default. And `template.MIGRATE_HINT` (`template.py:2290-2293`) names `--bundle` without a complete command | W57 (explicitness) | open · *2026-10-18:* 5 defaults now; `template.py:2465-2468` `MIGRATE_HINT` |
| **W56-H** | tests / PySCF | **A PySCF relaxation with a held atom, end to end** — its progress log's `# frozen_atoms` line and a trajectory read with the atom held; only the vibration e2e holds one today (`test_vibration_e2e.py:501-537`) | W56 | open — M2n's batch |
| **TST-R** | tests / front end | **Recorded by TST's replacements, not done** (each re-read 2026-10-18): (4) two of the sidebar's three file-slot readers redo what `openDir` → `setShared` → `renderSidebar` did (`lib/projects/list.js:636` `_renderList`, `:759` `openDir`, `:827-837` `restoreSelection`) — one reader wanted; (6) every SIESTA relaxation deck carries the same troubleshooting advice whatever it states (`siesta/input.py:1524-1531`: *shrink MD.MaxDispl to 0.02*, *loosen MD.MaxForceTol to 0.04*, literal, reading nothing of the deck's own values); (14) the spectra panel's hand-picked `REQUIRED_IDS` (`tests/test_results_blueprint.py:236`) want (14)'s mount test on a spectra run; (15) `results.html:246` loads the findings renderer's global and the only reader on the page is the Spectrum form's preflight (`lib/spectra/core.js:958, 976`) — nothing on Results draws a finding; the transport describe door's findings carry no `where` (`web/blueprints/transport.py:220, 230, 235`, against the description route's `:575-579`); no browser test opens a real spectrum and animates a mode through the Spectrum code (`test_siesta_vibration_e2e.py:363-512` clicks the ES and thermo tabs only; the `anim-*` ids are presence rows); no test reads a PySCF vibration's result file DURING its relaxation (an in-progress read — the equilibrium block absent whole, 2.16.3; the water e2e reads the file at its end) | TST (archived 2026-10-18), VVW | open — step 3 |
| **SCHED-R** | scheduler / monitor | **SCHED-END's remainder** (built 2.12.0, archived 2026-10-18): why neither the wrapper's EXIT trap (`runwrap.py:3154`, `_mb_stop_monitor TERM`) nor the monitor (`monitor.py:2249-2256`, `_STOPPED_BY = "SIGTERM"`) recorded SIGTERM on Sol job 65999920 — the job's files, the user's; and the kind's advance signal (`sbatch --signal`), named only in a comment (`monitor.py:2248`), `scheduler/kinds.toml:46` carries none | SCHED-END | open — the first needs the user's files; the second a decision |
| **PAGE-PARITY** | Task setup | Three lines the page and the terminal say differently, found by the 2.11.0 review: a benchmark's "takes nothing from another run" (the page only), the resources line's memory cap (the terminal only), a benchmark continuation's label (*written at* vs *every trial's geometry*); and the card's pairs dropped by the even-split rule, shown once GKN is decided | the 2.11.0 review | open · *2026-10-18:* *takes nothing from another run* is on both sides now (`jobset/ready.py:85`, `_cli.py:1409`); the memory-cap line is the terminal's alone (`_cli.py:1444`); the benchmark label line not located |
| **EXCL** | execution / model | **`exclusive` — a request field nothing sets.** No flag, `allocation` key, run-card item or page writes it, so the header's whole-node branch (`--exclusive`, `--mem=0`) never renders; every `job-set.json` carries `"exclusive": null`, so removing the field needs the retired-key path (`jobset migrate`) | round 2 of the launch review, 2026-10-17 | open — step 3 · *2026-10-18:* `jobset/model.py:94`; readers `shelves.py:123`, `runwrap.py:4299,4429`, `scheduler/emit.py:103`; no writer |
| **GKN** | bench / naming | **An uneven ranks-over-GPUs cell is dropped** (32 ranks on 3 GPUs) because a trial is named and sized by its ranks per GPU (`G3K?C1`, `mpi_np = G·K`) — not for the hardware, and no ELPA source says it cannot run (a run with the same numbers is prepared, its split said). Kept only by naming every trial by its total ranks (`G0N32C1`, `G3N32C1`) | the user, 2026-10-17 | **decision owed** — § 0a |
| **Q11** | execution / scheduler | **The named-queue check in two doors** — `placement.launch_refusal` and `scheduler/place.place`. *Re-derived 2026-10-17:* launch asks neither; at prep `launch_refusal` answers first, so `place`'s two membership branches are not reached | the queue of 2026-10-02 | open — step 2: one door · *2026-10-18:* `jobset/placement.py:33`, `scheduler/place.py:48` |
| **Q8** | configuration | **Ruled 2026-10-02, not built**: one `proxy.trust` setting replacing `auth.trust_proxy` and `rate_limit.trust_proxy` (`configuration.md` § 8); a benchmark's grid stated by the description, refused when it is not — no grid the machine proposes | the queue of 2026-10-02 | ruled, not built — step 2 · *2026-10-18:* `runtime_config.py:584,605-610` still `auth.trust_proxy` |
| **Q2f** | code hygiene | **Absolute imports, package-wide** *(user, 2026-10-03: "i agree with a, but like a focused session for that with agents and reviews. let's finish this main goal of transport and framework related tasks")* — every `from ..x import y` inside `molbuilder/` becomes `from molbuilder.x import y`, the rule written into the conventions; **and the shipped files' imports in the same session** *(ruled 2026-10-06, W57 decision 1: "B, with Q2f as one session")*: each zip (`mb_monitor.pyz`, `mb_vibration.pyz`, `mb_pyscf.pyz`) keeps molbuilder's folder layout so the package's absolute import resolves inside it (user, 2026-10-06: "for the first form of import why can't we use absolute path?"); the try/except ImportError pairs leave the 22 shipped files. Until then a NEW module is written absolute | the queue of 2026-10-03 | agreed (option a) — its own focused session in step 3, as the user asked: a script with an agent review, then the full batch · *2026-10-18:* 1,899 relative imports in the package |
| **CARRY** | various | **Smaller items carried by archived sections**, each to become its own row when taken: TS9's undocumented-test count to re-measure (§ 5m); `generator_mismatch`'s severity (`validation/siesta.py:149`), `detect()` unable to say why it refused (`parse/registry.py:46-89`), Task setup acting on `onChange` (`task-setup/viewer.js:3811-3839`) (§ 10); P6 — an A/B swap passes `tests/test_backends.py:306`; P7 — the `ByElement(("Au",))` mutant lives in `tests/test_atom_selection.py:342-355`; P12 — `lib/projects/checkpoint.js:33`'s no-poll property is prose; § 11.5a's error surface, unwritten; `LAYER_TOL_ANG` untested (§ 11); the route-catalogue sweep (`web-api.md`: 11 documented routes that did not exist, 9 live ones undocumented, 2026-10-08 — re-measure after 2026-10-09's corrections); the noted items in M2a (an unreadable zip or an exception escaping `main` reports in Python's words; a failing `ending` exits 1, which the wrapper treats as 2), M2b (the Build route's peptide branch collects no warnings, so the C-terminus warning reaches no web user) and M2c (`create_app(config={})` reads the disk config through the snapshot); N5d (the label handed to the wrapper through a `Resources` field — waits for a yes); `BlockSize` as a benchmark axis — built as a value axis (`prep_inputs.py:94-106`), re-derive what is left | the archived sections | open |
| **TPL** | engines / template | **`engines/template.md` § 12.1's open rows without one of their own**: row 3 — G3's `kind`, which no layer dispatches on; row 4 — `read_by` declared and unconsumed (no caller asks `select(read_by=…)`); row 5 — `kind = "monitor"` with no items; row 7 — BENCH-MARKS SIESTA-only. (Row 1 is W47, row 8 W44, row 6 closed) | `engines/template.md` § 12.1 | open · *2026-10-18:* `read_by` declared on one item (`catalogue:1003`) and read by nothing outside `template.py`; no `kind = "monitor"` in the catalogue — § 12.1 row 5 re-read when taken |
| **SMALL** | various | **Smaller open items found in the contracts** (2026-09-29, each searched in the plan and not found): the Raman/IR finite-difference steps untested for convergence, ΔF ≫ σ_F not estimated, two textbook references owed (`normal-modes.md`); V1.22's dropped warning half; `dm0` seeding (`pyscf.md`); a SIESTA road run whose held-first sort reorders atoms, `info.calculation` on the PySCF road (K18), the v1 reader's residue (`vibration.md`); the G-5a `gpu_used` read-back (`engines/overview.md`); § 6b's two questions (`stages.md`); multi-node MPI and `config init --site`; hand-typed wrapper comment claims (`script-preparation.md`); a test `gpu.md` lists that does not exist; a PySCF residual shown with no tolerance (`run-reports.md`); the per-trial SCF plot (`bench-summary.md`); five known gaps in `projects.md`; `ttl` / `detail` / a `success` level for notifications; whether the markdown editor returns (`presenters.md`); legacy response keys (`web-api.md:76,246-251`); an unexercised APPLY arm and the getter/setter bridge (`results.md`); the calculation-wide block editable only by hand (`task-setup.md`); a per-layer overlay refresh and `.pdb` on the download row (`molview.md`); `title` carried by hand at ten sites (`structure.md`); `MOLBUILDER_LOG` (A1.15); the SLURM-shaped `Resources` (`backend-architecture.md` § 7); the gcc 15 migration against the GPU SIESTA pin (`installation.md`); `references.bib`'s unverified entries | the contracts, 2026-09-29 | open — re-derive each when taken |


---

## 5e. Engine additions — a person's own engine text, as an INPUT

*(Your ruling, 2026-09-05: option 2 — a distinct input to the writer, not a
catalogue extension; **engine-specific**; and **the person is told the
consequence**. Contract-first: this section is the design. **Not started.**
No code has moved.)*

*Re-checked 2026-10-09:* the measurements hold in shape, at new lines —
`render_deck`'s concatenation is `script_emit.py:1382`. Two of the known
defects named below have moved: a stray marker no longer drops text silently —
`check_deck` refuses a deck holding more than one pair of USER-CUSTOM markers
(`script_emit.py:1522-1543`); and transport has the zone — its decks are
rendered by `render_deck` like every other. § 3.5's *byte-for-byte* now reads
*line for line, unchanged* (`job-contracts.md`). `engines/template.md` § 9.2
still describes a `user_custom` catalogue item — the carrier this ruling
rejected; it is corrected when this is built (§ 2, W47).

### The need

A person wants to run a SPECIFIC task on a structure that is already
optimised, using engine content molbuilder does not model: a Lua setup driving
SIESTA, a `%block` for a feature with no form field, a PySCF call.

Today the only place for that is the USER-CUSTOM zone of a relaxation deck,
which is the wrong shape twice over — it is the wrong kind of task, and (below)
the zone cannot carry the content anyway.

### Why the zone cannot carry it — measured, not argued

`render_deck` assembles the deck like this (`script_emit.py:1319`):

```python
text = (science + "\n\n" + emit_user_custom_placeholder()
        + "\n\n" + machine_record_banner()
        + "\n\n" + "\n\n".join(record) + "\n")
```

**The zone is not in the layout.** `spec.layout` never sees it; the framework
concatenates a placeholder after the walk. Everything else in a deck arrives
through the model — `Section.items` are CATALOGUE NAMES, turned into a
`Parameter` and handed to the engine's `line`, one line each, recorded in
`emitted` so `check_deck` can close the loop against the written file.

Three consequences follow from that one fact, and they are not three bugs:

1. **A user's line can duplicate a declared item's**, because nothing compares
   them — the writer never saw it.
2. **Position is fixed below the science.** Measured on a real deck: engine
   body at lines 12 / 520 / 980, zone at 1222. libfdf takes the FIRST
   occurrence, so *anything the deck already writes cannot be overridden from
   the zone*.
3. **The engine's own rules refuse it.** SIESTA's `check_rules`
   (`siesta/layout.py:354`) splits the whole file and knows nothing of the
   fence, so a duplicate raises **error** severity and `prep` refuses.

The Lua case is decided by (2) and (3) together: SIESTA engages Lua with
`MD.TypeOfRun Lua`, which the catalogue already declares. Measured — that deck
is refused, and the refusal is *correct*, because libfdf would have ignored the
line anyway. **The zone can never carry the feature it is documented for.**

### The shape

An **engine addition** is a person-supplied contribution to one engine's deck.
It is an INPUT, alongside the structure and the config — never text recovered
from a previous output.

| | |
|---|---|
| **engine-specific** | an addition is SIESTA text or PySCF text; there is no engine-neutral addition, because the content is engine syntax. It is declared for one engine and ignored by the other, the way `[item.*]`'s `engines` key already works |
| **the writer places it** | whatever writes that task's script takes additions as an input and places them, the way `render_deck` places a `Section`'s parameters today — by the engine's authority, in a position the engine chooses, never concatenated after the walk |
| **emitted once** | if an addition writes a keyword a declared item also writes, ONE line is written, not two |
| **recorded** | its lines join `emitted`, so `check_deck` closes the same loop over them as over every other line |

### The consequence a person is told

Silent resolution is the thing to avoid. `Parameter.writes` already answers
*"which engine keywords does this item put in the deck"* (from `expands`, else
`anchor`), so a collision is **detectable, not guessable**:

- an addition writing a keyword **no** declared item writes — accepted, placed,
  no notice;
- an addition writing a keyword a declared item **also** writes — the person is
  told, at the point of entry, what is about to happen: *your value replaces
  what `MD.TypeOfRun` would have written (`CG`)*. Their value wins, because
  they said it last and more specifically — but never without being told;
- an addition molbuilder cannot attribute to any keyword (a `%block`, free
  prose) — accepted verbatim, and the engine judges it, which is the honest
  half of today's § 3.5.

**This is the rule the current design cannot state**: today a person is either
refused (duplicate) or silently ignored (first-wins), and which one depends on
whether SIESTA's rule happens to notice.

### What this does NOT disturb — which is the point

**This is ADDITIVE. It removes nothing that ships.** The USER-CUSTOM zone, the
read-back merge, `write_script`'s round trip and `check_deck`'s reason to read
the written file all stay exactly as they are, serving the relaxation decks
they serve today. A task kind that does not exist yet cannot be a reason to
disturb one that does.

*(An earlier draft of this section claimed the design "deletes the read-back
merge" and closes the transport gap. **Withdrawn.** That followed from the
withdrawn assumption that additions would flow through the relaxation deck
path. They do not, so those mechanisms are untouched and their known defects —
the stray marker that silently drops text above it, transport having no zone at
all, § 3.5's inaccurate "byte-for-byte" — remain open on their own terms,
listed where they belong rather than as credit claimed here.)*

**What it buys instead** is that the new need lands in its own layer:

- nothing in the relaxation path changes to accommodate it;
- the mechanism is defined by what it IS (an input, engine-placed, switchable,
  unvalidated) rather than by which existing function it borrows;
- when a task kind does need it, that task brings its own writer and this
  mechanism plugs into it — no structural change to make room.

### THIS IS NOT A STAGED RUN, and must not be fitted into one

*(Your correction, 2026-09-05, replacing what this section said first.)*

**Stages exist for one reason: a calculation that needs several steps to fit
the computational resources and constraints** — coarse before tight, a ladder
that accommodates a machine. That is a different problem from this one.

A customised block is for **a specific task, on a structure that is already
optimised**. It is the mechanism molbuilder EXPOSES for a future kind of task
that needs it — not an extension of the relaxation path.

So the following, which this section asserted in its first draft, is **wrong
and withdrawn**:

> ~~"An addition needs no new home: it follows the path a parameter already
> takes — collected in Structure optimization, overridden per stage in Task
> setup."~~

That reasoned from the tabs that exist to the need, which is backwards: it took
a mechanism for a *future task kind* and forced it into the ladder built for
multi-step resource accommodation. Per-stage override is a stage concept, and
this has no stages to override across.

**What survives that correction, because it does not depend on staging:**

- an addition is an **INPUT** to whatever writes the task's script, never text
  spliced into a written file (the whole of § 5e above);
- it is **engine-specific**, and where it goes is engine knowledge;
- it carries an **include switch**, which is what makes the responsibility
  workable;
- **molbuilder does not validate it.**

**What is deliberately left open**: which task kind first needs this, and what
its own description looks like. That question belongs to that task, not to this
mechanism — and answering it early is how this would get forced into stages
again.

### Who is responsible
molbuilder does not understand the content; it places it and records it. For
"your responsibility" to be a fair deal rather than a disclaimer, three things
have to be true, and only the first is about the text:

1. **A stated format** — a clear start and end, so the addition is a bounded
   thing rather than loose text. (Not the current marker fence, which is
   file-level and is what a stray paste can break; the bound belongs to the
   addition as data.)
2. **It is separable at generation time.** `prep` can write the deck WITHOUT
   the additions and WITH them, because they are an input rather than text
   fused into the file. That gives a person the bisection directly: run it
   clean, run it with, and the difference is theirs.
3. **The consequence is stated before it is saved**, per `Parameter.writes` —
   *your value replaces what `MD.TypeOfRun` would have written (`CG`)*.

(2) is the one that turns responsibility into something a person can act on,
and it is a capability the input model gives for free. Under a zone it is
possible only by hand-stripping a section from a written file, which changes
the deck in more ways than the one being tested.

### The toggle is part of the block, and it is the whole mechanism

*(Your ruling, 2026-09-05.)* An addition carries an **include** switch in the
UI: *do you want this customised block in the final task?* That is not a
convenience and not a `prep` flag — it is the instrument that makes the
responsibility workable.

- **Off** — the task is prepared and run exactly as molbuilder would have
  written it. This is the reference.
- **On** — the same task with the addition placed.

A person compares the two and decides for themselves whether a failure belongs
to their block. They can do it through whatever they are already doing —
a benchmark trial, a debug run — because the two differ in one input and
nothing else. **That is only true because the addition is an input**; stripping
a zone out of a written deck changes more than the thing under test.

The switch belongs to each addition, not to the calculation, so several can be
carried and enabled one at a time.

### What an addition IS, per engine

Open-ended by design — a script, a variable, a setting molbuilder has not
exposed. What it means is the engine's business, and the two engines differ in
a way that matters for placement:

| engine | an addition is | why placement differs |
|---|---|---|
| **PySCF** | Python that RUNS — a call, a hook, a few statements | the deck is a program, so an addition must land where the objects it uses already exist |
| **SIESTA** | fdf settings molbuilder has not exposed, or a Lua setup (`MD.TypeOfRun Lua` + a script path) | the deck is a settings file, so what matters is libfdf's first-wins and the block structure |

So **where an addition goes is engine knowledge**, which is already where the
framework puts layout: the engine owns its `Section`/`Block` layout, and an
addition is placed by the same authority rather than by a framework rule that
would have to be right for both.

### Whose responsibility, stated plainly

**molbuilder does not validate the content and does not claim to understand
it.** It places it, records it, states the consequence when it collides with a
declared item, and gives the person the on/off pair to test with. Making the
addition correct — that it parses, that the engine accepts it, that the run
completes — is the person's.

That is a fair deal only because of the switch. Without it, "not our
responsibility" would leave someone with a failing run and no way to tell which
half caused it.

### Open, and deliberately not decided here

1. **Does the zone survive at all** for genuinely free-form text (a comment
   with no variable in it), or does that become an addition with no attributed
   keyword?
2. **What the card shows** when an addition collides — refuse, warn-and-accept,
   or show the resolved line before saving. (The consequence must be stated;
   whether it can be overridden is separate.)
3. **Ordering among additions**, when two of them write to the same section.
4. ~~Whether the include switch is per-stage~~ — withdrawn. There are no
   stages here; see the correction above.

### Before any code

This section is the contract. The measurements it rests on
(`script_emit.py:1319`'s concatenation, the 12/520/980-vs-1222 positions, the
reproduced duplicate-keyword refusal, `Parameter.writes`) are re-checkable, and
should be re-checked rather than trusted if this is picked up later.


## 5q. The engine offset — scope and order of work *(W33, 2026-09-25)*

*The rule, the name, the operations and the checks are the contract's:
[`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 6.0.
This section is the scope and the order of work, and restates no rule; its
row is **W33** in § 2. Trimmed 2026-10-08 after the validation walk: P0–P3
are built and P5 is closed by TD8; what stays is P4's remainder (§ 5q.6).
The measured facts, the data-structure and file-access tables, the risks and
the decisions D1–D16 are archived with their verdicts.*

### 5q.3 Protocol agreement — the wire

* **Built** (`6c705058`): one periodicity block from `Structure.to_wire`
  (`structure.py:896–950`); MolView draws the box at `box_corner`, verbatim
  (`render-engine.js:313–317`, `model.js:879–884`); the Cell page's origin
  group sends `box_corner` (`modify/periodicity.js:508–621`).
* The load/save payload's door is `validate_periodicity`
  (`web/blueprints/_shared.py:36–37, 354–385`).
* The Results door reads the run deck's ENGINE-OFFSET record for the axis
  kinds (D5) and states 0. The `.source` pair fallback for a run made before
  the record was retired by B12 4b (§ 0c unit 4): a run reads its own deck.
* ~~The transport deck viewer has no home since step 9 closed~~ — built 2026-10-11 on the Results ladder, for every calculation (§ 2, W30 ④).

### 5q.4 Validation

* **The checks** (contract § 6.0): *the atoms fit* (fractional span `< 1`) at
  the edit; along the transport axis the check is TD3's
  `cell.transport_vacuum` (`validation/__init__.py:459–500`) — a gap at the
  face beyond one layer spacing of the lead is **refused**, measured from the
  lead. T4's *d/2 warning* and R1 are superseded by it.
* **Where the tests live**: T1 `tests/test_engine_offset_reaches_every_deck.py`;
  T2 `tests/test_siesta_flat_run_e2e.py:168–191` (the Results load of the
  measured flat H2: the box at 0, each stage reading its own deck);
  T3 `tests/test_results_export_e2e.py`; T5 `tests/test_cell.py:543–576`.
* **"Nothing translates by hand"** is checked by a code-text review of
  § 5q.5's rows at the end of P4, not by a lint test.

### 5q.5 The inventory — what P4 still reaches

| site | role today | phase |
|---|---|---|
| `structure.py` `to_ase` (`:2034`), `cli.py --pyscf-atom-block` | design coordinates beside a lattice that implies the box at the origin — `to_ase` alone now: `to_extxyz` is gone (`61db1660`) and `to_pyscf` writes no lattice | P4 |
| `transport/deck.py:657`, `validation/__init__.py` | no validation subject: the transport validators judge design coordinates in `resolve_cell()`, which for a lead that states no cell is not the deck's box | P4 |
| `transport/wizard.py:396-400` | the lead's z hand-shifted by its lowest layer — harmless (re-centred), and a translation outside the one door | P4 review |
| multi-frame pairs | **built 2026-10-09**: `Structure.frame_at(i)` states the set's one offset — frame 0's — so every frame reaches `cell.to_engine` with it (`structure-periodicity.md` § 6.0); each frame's deck built 2026-10-10 (Q17-d, `prep._at_frame`) | — |

**The documents to sweep** in P4 — the owner first (the line numbers are 2026-09-25's; re-derive them, and § 6.0's own list was brought to the code on 2026-10-09): `structure-periodicity.md`'s
superseded clauses are still in the text — § 6 (`:606`), § 6.1 clause 4
(`:697`), § 6.1a's corner column (`:781`), § 6.2 (`:961`) — deleted, as § 7
and clause 5 were on 2026-09-25. Then the restatements, counted at planning:
`web/molview.md` (11), `model/structure.md` (11), `web/web-api.md` (6),
`model/structure-molstruct.md` (5), `engines/transport.md` (3),
`engines/vibration.md` (2), `architecture.md` (2), `science/normal-modes.md`,
`README.md`, `model/overview.md`, `backend-architecture.md` (1 each);
`model/parse.md` (the premise *"no deck shifts a structure that states one"*),
`model/structure.md` § 2.2 (the removal rule D2 relied on),
`execution/job-contracts.md` § 3.1 (the block grammar lacks ENGINE-OFFSET),
`engines/vibration.md` § 5.2a, `engines/siesta.md` (centring described for
the derived box only). Dated plans and handovers are history and are not
rewritten; `plans/plan.md`'s live rows are.

### 5q.6 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0–P3** | the name and the contract; data structure + file access; every emitter through `to_engine`, each deck carrying its record; the readers, the wire, MolView and the Cell page | **Done** 2026-09-25/27 (`a2901f4d`, `dcfb371b`, `1df5cc24`, `4adc6832`, `6c705058`, `4a3172e1`, `dba66a9a`); the detail is archived. T2's pin is `tests/test_siesta_flat_run_e2e.py:168–191` (§ 5q.4) |
| **P4** | the checks, the test retirement, the document sweep | the test retirements and the contract's own tables **done 2026-09-25**; the clearance check is TD3's (§ 5q.4). **Open — three items**: ① the document sweep (§ 5q.5's list, the owner's four clauses first); ② the design-frame exports (§ 5q.5 row 1); ③ the transport validation subject (§ 5q.5 row 2) — `wizard.py:395` read in the same review. Done when the review of § 5q.5's rows finds no hand translation and no design-frame export, and the four clauses are gone |
| **P5** | acceptance | **Closed by TD8** (§ 5u.2): the acceptance ladder is `claude-au-bdt-au`, run to its record on 2026-09-29 (§ 5u step 4). The one piece P5 left — the Results tab's view of each rung's engine frame — is § 5x R9 (the 3D viewer's default view, B6) |

---


## 5r. The structure-API cleanup — its order, and what not to fix *(A1; from the unification audit, 2026-09-25)*

*The findings are § 2's **A1** rows. The evidence, measurement by measurement, is
the audit itself, archived as the record:
[`archive/2026-09-22-unification-audit.md`](?doc=archive/2026-09-22-unification-audit.md).
This section keeps the two things the rows cannot: the order, and the
non-findings.*

### 5r.1 The order, and why

1. **The map first** — the stale comments and misleading docstrings (A1.15),
   in one pass, and **D5 before X1 ②–⑤**: a bad map is what made X1 ① read
   as a sanctioned live path. Nothing else is safe to touch until the
   documents describe the code.
2. ~~**The four live defects that need no ruling** — A1.11 (`X`), A1.12 (the
   H-ratio on raw labels), A1.5's `.pdb` travelling name (fix `files()`'s
   `fmt`, and A1.13's root falls with it), A1.4's silent `JSONDecodeError`.~~
   *Done: A1.12 2026-10-11; the other three 2026-10-14 (2.9.6).*
3. ~~**V1.10 — the generator renders both halves.**~~ *Done 2026-10-17.* The only structural change
   in the set; it also closes the half-written pair (a NaN in `info` raising
   at the sidecar write after the `.xyz` is on disk). Do A1.3's doc fix in
   the same commit — both are in `structure.md` § 2.4's four-clause block.
4. ~~**A1.1**~~ (done 2026-10-17), ~~A1.4~~ (done 2.9.6) — the rest of the data-loss and uncaught-exception set;
   independent, small, user-visible.
5. **A1.5** — one home for the structure-path rule, then rename-structure vs
   rename-file, the no-delete rule, and the hex check (X4 ⑤). One missing
   door and four things that grew where it should be.
6. **A1.7** — the registry seam; beside A1.1, all three are *a check that
   does not happen, and the absence is invisible*.
7. **A1.8** — the twelve `axis_kind` fallback deletions first (pure removal,
   and it makes the rest safe to read), then the partial-charge and k-hint
   doors. A1.14's instances join here. The transport no-cell arms are ruled
   and are X4 ③'s.
8. **A1.9, A1.10** — the second and third conditions: `schema_version` is two
   one-liners; the builder defects need the placeholder carried apart from
   the data, a shape decision before a fix.
9. **The origin-rule sites** — containment without the origin, and
   `wrap_into_cell` (retired 2026-09-25). **These are W33's** (§ 5q) and go with the engine
   offset, not here.
10. ~~**A1.2**~~ — superseded 2026-10-09 by the comment-line ruling (`61db1660`): a lone stream is atoms and coordinates, and the read says so.
11. **A1.18** — the line-number pass, mechanical; then its behavioural list.
12. **A1.13** — each at its owner, never at the instance, starting with the
    `replace()` guard (A1.6), which is what makes the rest safe to touch.
13. **A1.17** — the three unpinned rules first (the `.pdb` one before all: a
    regression with its numbers already written down), then the six blind
    tests (worse than absent — they read as coverage), then the duplicates,
    re-running each cluster's mutant to confirm the keeper still goes red.
14. **A1.20** — residue last, and only what has a clean step-0 verdict.

**A1.21** (audit #2) comes after steps 1–4. **Standing on its own:** the 15
hand-built `structure_hash` fixtures (A1.17) ~~agree with the writer only
because the gate is loose — convert each as its file is touched for another
reason~~ — converted 2026-10-14 with V1.9: every pair a test writes goes
through the codec, and a file of an older version is written as that version
wrote it.

### 5r.2 Do NOT "fix" these — confirmed non-findings

Recorded so nobody fixes them by analogy. Re-read against the code 2026-10-08.

- **`frozen_atoms`'s shape is benign.** A data descriptor sends every read and
  write through `regions[FROZEN_LABEL]`, so the two cannot disagree in any
  order; `replace()` handles it by not re-passing it. **Not the defect `pbc`
  was** — the init-field spelling exists so `Structure(…, frozen_atoms=[…])`
  reaches the one place that spells the reserved label.
- ~~**`resolve_cell` really is the one resolver.**~~ **Untrue (2026-10-09):**
  `transiesta._compute_cell_from_extents` computes one for a transport lead
  (`transiesta.py:171,333`; `wizard.py:392`) — X4 ③.
- **The companion-lookup merge stays withdrawn.** What forbids it is a test
  pinning a deliberately more permissive guard, not `parse.md` § 5.3; a shared
  helper with the guard as a parameter would keep both.
- **The second multi-frame XYZ reader is forced**, not duplicated: it
  tolerates a torn final frame (`parse/engines/pyscf.py:405–439`), which every
  live geomeTRIC run has and ase refuses. Unifying it would break live-run
  viewing.
- **Registry overlap: none** — 14 parsers against a 29-file synthetic run
  directory, no file claimed twice.
- **Bare atom-index arithmetic in `parse/`: none that carries data.** But
  `transport/transiesta.py` has real ones (A1.14, still open:
  `:500`, `:580` on 2026-10-09), in the block whose own docstring says an
  off-by-one *"computes transmission through a region that is not the
  molecule, and converges while doing it."*
- **The CLI's seam verdict** was raised and dropped on the user's ruling
  (2026-09-22: *"people working with CLI would know what they're doing… leave
  that out"*). X3 is the browser half only.

---


## 5ab. The remapped state — ARCHIVED 2026-10-18

RMP built 2026-10-14 (R0–R6, R4 dropped), reviewed twice 2026-10-15, pushed as 2.10.0; nothing of it open. The section whole: [`archive/2026-10-18-5ab-remapped-state.md`](?doc=archive/2026-10-18-5ab-remapped-state.md). The contract is `model/overview.md` § 2.2a.

---

## 5s. The electronic state — charge and spin as one answer *(W34, 2026-09-25)*

*The contract is [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md)
§§ 2a–2b; this section is the order of work. The row is **W34** in § 2; the
milestone is **M6**. Amended 2026-09-28 (decisions 8–9): spin is decided by the
same class as charge, and Auto-detect's fill is retired. Trimmed 2026-10-08
after the validation walk: P0–P3 are built, P4 all but one site; what stays is
P5 (ES10), decision 4's refusal site, and § 5s.4's open items. The measured
causes, decisions 1–3 and 6–9, the P0–P3 rows and the review record are
archived with their verdicts.*

### 5s.1 The design

The contract, § 2a: four items, one resolver, ten rules (ES1–ES10), a capability
table and the engines' semantics, each fact read from the engine's source. § 2b:
the species.

### 5s.2 Decisions *(user, 2026-09-25: "go with your recommendations on all seven")*

*Decisions 1–3 and 6–9 are built and archived. Two are partly verified:*

4. **Transport's spin defaults from the cited run** (`transport/citation_defaults.py:56–72,
   126–129, 161–194`), and a cited run carrying a net charge is refused.
   **Located 2026-10-09**: `citation_defaults.py:196-207` refuses a charged
   citation by name (ES7) — built.
5. **A vibration built from a relaxed structure inherits its state**
   (`electronic_state.py:253–283`, `_recorded`), and a change is warned (the
   record check, `validation/spectra.py:638–640`) — PARTLY on 2026-10-08.

### 5s.3 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0–P3** | the contract and its pointers; the items and the class; the checks; the forms | **Done** 2026-09-29 (`chemistry-correctness.md:271–720`; `electronic_state.py:160–252`; `tests/test_electronic_state.py`, `tests/test_chemistry_card_e2e.py`; `lib/detection-chip.js`; `web/blueprints/build.py:122–151`); the rows are archived |
| **P4** | the hand-over: `parse/fdf.py` reads `NetCharge`/`Spin`/`Spin.Fix`/`Spin.Total`; transport defaults its spin from the citation and refuses a charged one; the relaxation record carries the state, the vibration defaults from it, and the record check compares it | **Built** 2026-09-29 — the fdf reader `parse/fdf.py:158–175`; the citation's spin `citation_defaults.py:56–72, 126–129, 161–194`; the inherited state `electronic_state.py:253–283`; the comparison `validation/spectra.py:638–640`. the charged citation's refusal `citation_defaults.py:196-207` (decision 4) |
| **P5** | the read-back: SIESTA's `.out` (net charge, fixed or converged moment) into the run record; PySCF's class, ⟨S²⟩ and stability recorded; the transport record reads both TBtrans channels; the Results tab shows asked against used, and a difference is a finding | **Open** (ES10): no reader of SIESTA's moment or charge, none of PySCF's ⟨S²⟩. The TBtrans half is § 5x B6 / R1 — `transport/record.py:453`'s own glob replaced by `tbtrans.transmission_files`, which knows the channels. Done when a spin-polarized run's moment and a UKS run's ⟨S²⟩ appear on the Results tab, and a two-channel transmission is read |

### 5s.4 Found on the way — named here, fixed only where a phase says so

* ~~A fresh PySCF bundle's first `prep` says *"this calculation is already under way
  here: warm files at the root: <name>.source.xyz"* — the structure copy `init`
  writes is read as a warm file.~~ **Gone, read 2026-10-18**: no such sentence
  exists; prep's only warm-file refusals go through `warmfiles.warm_list`
  (`jobset/prep.py:186-189, 263-267`) and the pair is written by the hand-over's
  own call (`workingcopy_structure.py:470-477`).
* ~~The wrapper's failure hint (`runwrap.py`) still says *"SpinPolarized with
  Spin.Total unset or 0 on a d/f-shell metal also triggers IMAX=0"*~~ — **fixed
  2026-09-29** (P2): the retracted cause is gone (`runwrap.py:3289–3305`).
* ~~The transport tab's shared panel is not persisted~~ — **fixed 2026-10-11**
  (step 8): the shared panel and the rung tabs keep what they hold in one
  workspace note, `transport:form`.
* ~~The PySCF vibration record keeps the raw `net_charge`~~ — **fixed**: its
  `config` carries the resolved `electronic_state` block.
* **Left by the review (2026-09-29):**
  * **The state is resolved several times per prep** — 11 call sites of
    `electronic_state()` (2026-10-09; still 11 on 2026-10-18 — `siesta/input.py:649`, `validation/siesta.py:207, :573`, `validation/spectra.py:628`, `validation/chemistry.py:279`, `pyscf/input.py:236`, `pyscf/vibration_deck.py:111`, `jobset/transport_rung.py:185`, `jobset/engines.py:111`, `transport/deck.py:653`, `web/blueprints/build.py:211`): the family, the
    SIESTA checks, the vibration record check, the deck writer and prep's
    sibling files. Each gets the same answer from the same function; resolving
    once in `validate()` and handing it down is a tidy-up, not a defect.
  * **A re-prepped stage folder keeps its `makov_payne_correction.py`**
    (`runfiles.py:838`) from an earlier charged prep after the charge goes
    to 0. Prep writes siblings; it does not remove the ones a deck no longer
    promises. Not reproduced on 2026-10-08; by reading on 2026-10-18 it stands:
    `jobset/engines.py:96-116` writes the script only when the deck is charged
    and finite, and no prep step removes a sibling (prep removes a stale
    `.sbatch` alone, `prep.py:204-209`).
  * **`restart = continue` keeps the run id across a template edit.** So a spin
    edited between continuations starts from the previous state's `.DM` — the
    hazard ES1 closes for stages, one level up
    (`execution/run-identity.md` § 2.1). **Check against § 5x's ruling**
    before keeping: M2h (the stage hand-over) is done 2026-10-05, and § 5x.3
    rules *every launch a new run; warm or cold; every run kept* — **closed
    2026-10-09** under the user's rule of 2026-10-07: *"run continue warm or
    cold is user's decision, and error or not, that's user's responsibility"*.


## 5t. The run record — scope and order of work *(W35, 2026-09-26)*

*The contract is [`model/parse.md`](?doc=model/parse.md) § 5d and
[`engines/transport.md`](?doc=engines/transport.md) § 2a.12; this is the order
of work. The row is **W35** in § 2 (M3 carries P2's remainder and P3–P6).
Re-read against the code 2026-10-08: what was built or overtaken is in the
archive; what stands here is open.*

### 5t.1 The design

`model/parse.md` § 5d: one record per attempt, four parts (computation,
setup, deck, verdict), composed on read;
three columns for every parameter; one grammar per engine family; the verdict
and its symptoms. `engines/transport.md` § 2a.12: the transport report from
the rung records, and the stated contour.

### 5t.2 Decisions *(user, 2026-09-26: "get the framework and code unified and finalized")*

Taken as a yes to the three recommendations made the same day, with the
user's own additions — what is still open of them:

* **2.** **The contour is stated**: a circle and a tail whose lower bound sits
  below the seed's lowest eigenvalue — the manual's rule — rather than a
  pole count guessed from one measurement. Open — M3 P4.
* **3.** **On divergence the monitor warns, and the wrapper does not warm-retry**;
  stopping stays the person's call. Open — M3 P4: the wrapper's warm retries
  (`runwrap.py:3343-3349`, any `scf_not_conv` stop) ask no divergence
  verdict, and none exists; `running-a-job.md` marks the rule not built
  (2026-10-09), `model/parse.md:1706-1708` to re-check.
* **4.** **The parameters are reported as optimization's are — and better**: with
  what the engine used, and with the deck as it ran (user). Built
  (`parse/dirs/setup.py:151-205`, `siesta_fdflog.py:126-135`,
  `parse/dirs/record.py:467`) but for the setup rows' `echo` column —
  `parse.md:1538` lists it, `setup.py` writes none. P2.

And the same day, after P1 and its review:

* **5.** **P2 is built with all eight of the review's corrections** (§ 5t.5; user:
  *"go with all eight"*) — 4 (the echo column) and 6 (P4's symptoms) remain.
* **7.** **The transport result is shown, not listed** (user: *"plots that show
  the convergence of the calculation and … the DOS … in a more graphical
  way"*): each rung's convergence, and T(E), the DOS and the eigenchannels
  as plots. TBtrans's outputs are on by default
  (`data/catalogue.template.toml:2301-2347`); T(E), the DOS and the I–V have
  their card (`lib/inspectors/transport.js:142`); the eigenchannels and each
  rung's convergence plot are M5 step 9 / § 5x B6.
* **9.** **Then the whole road in the browser** (user: *"an end-to-end test
  through the browser … all based on the web UI design and the correct
  contract … and inspect the graphical presentation of the results and the
  graphical placement of the elements in the settings of the jobs … using
  the correct CSS framework"*): P6 — W50 step 3 (§ 5w.4).

### 5t.3 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P2** | **the record** — built: the Run panel (`web/results.md:646`, § 3a), the SCF plots by phase (`scfplot.js:89-108`), the parameters fence narrowed to the calculation's items (`parse/dirs/setup.py:151-153` → `script_emit.declarations:796-802`). **Owed**: the setup rows' `echo` column (§ 5d.3; `parse.md:1538` lists it, `setup.py:151-205` writes none); the launch rows asked against those run — `compare_asked_to_ran` (`bench/result.py:191`) is called by the bench alone (`jobset/summarize.py:178`) | on the dev server, an optimization, a vibration and each transport rung show their Run panel; the SCF plots draw each phase against its own criterion |
| **P3** | **the transport report** — **done** (§ 5x B6, M5 step 9; the record reads both spin channels and both `.TBT.nc` files, and the panels draw each spin and their total — K21, 2.4.0). What § 5t.0 measured and still holds goes with it: `energies_relative_to_ef` hard-coded (`transport/record.py:537`); the spin channels' files missed — `.TBT.AVTRANS_*` globbed alone (`:453`) | B6's and step 9's own |
| **P4** | **the contour, the divergence, and the monitor's warnings**: the device deck states its contour from the seed's eigenvalues; the settings gate refuses one that cannot cover the spectrum; no warm retry of a diverged run; the symptoms, which the monitor then warns on. *(One monitor for every engine, reading through the framework's shipped readers and reporting the run's state, is decision 10 — built; the archive.)* **Why** (measured 2026-09-26): no `contour.eq`, so TranSIESTA's 42-pole continued fraction — dQ −29 at step 1 and −584 at the cap; 123 poles on the same deck conserved the charge to 0.024 by step 4; the wrapper then warm-retried the diverged device from its own density for three more hours. **Doc drift**: `execution/running-a-job.md:481-484` and `model/parse.md:1706-1708` state the no-retry rule as built; `runwrap.py:3335,3367` retries with no divergence verdict to ask | a device deck carries its contour; a diverging output yields the symptom verdict, and the monitor warns on it |
| **P5** | **the document sweep** — left: the Makov-Payne script (§ 5t.4) and § 2's document rows | the review finds none |
| **P6** | **the whole road in the browser** (decision 9; W50 step 3, § 5w.4): an optimization, a vibration and a transport calculation, each from the UI through the hand-over, Task setup, the printed `prep` / `launch` verbs and the Results tab — every step through the designed doors, no script; the settings' layout and the results' plots inspected against `web/ui-contract.md` and the CSS framework | each road finishes in the browser, and the inspection's findings are fixed at their owners |

### 5t.4 Found on the way — named here, fixed where a phase says so

* ~~**A TranSIESTA device's `max_force` is meaningless** — TranSIESTA prints
  *"TranSiesta will NOT update forces … ALL FORCES AFTER TRANSIESTA HAS RUN
  ARE WRONG"* — and `report_fields.py:58-61` offers it to every run, its
  comment saying a transport rung's single point states one.~~ **Fixed
  2026-10-17 (F14)**: the SIESTA reading pass catches the line
  (`TS_FORCES_WRONG`, `siesta_reader.py:523-524`) and answers `max_force`, the
  forces and the stress as not computed (`:320-330`); the monitor says so.
* ~~A PySCF run's manifest promises the monitor, `util.csv` and SCF-timing
  files, which the wrapper never writes for PySCF (`runfiles.py`).~~ The
  monitor's two files (`.monitor.log`, `.util.csv`) are every engine's since
  decision 10 — the wrapper starts the monitor from its shared part; the
  SCF-timing tee alone is SIESTA's, and its row names the engine
  (`runfiles.py:1107-1118`). This row had said all three rows were SIESTA's.
* The SIESTA vibration record's `runtime_info` is empty, so its Host/CPU/GPU
  rows read *—* (`spectra/siesta_vibration.py:272` takes only `siesta_build`
  from the reading; the page draws the blank, `lib/spectra/core.js:1623`) —
  read again 2026-10-18, still so. P2.
* **The Makov-Payne post-process script** (`siesta/makov_payne.py`, written
  beside a charged deck) keeps its own `.out` regexes for `E_KS` and the
  cell (`:279`, `:306-307`, `:346`) — a second reader of the SIESTA family;
  it should read the run the way the monitor does. P5. ~~It defaulted to
  `<label>.out`, a name no wrapper writes, then globbed `*-run*.out` for the
  newest~~ — since 2.12.x it reads its own stage's newest run by run number
  through the catalogue's template (`:371-395`, `:491`).
* **Found reading the Results tab's code (2026-09-27)**:
  ~~loading one run parses its output three times~~ — the directory's
  metadata was composed twice per load, each composing the relaxation
  record by a full parse, and a stopped run's poll re-read the file for
  its stop reason; now the load's own parse serves all three (2026-09-27) —
  verified 2026-10-18: one keyed cache per file version
  (`parse/engines/_run_ending.py:330-349`).
* The runaway symptom (§ 5d.6) cannot be "|ts-Vha| exceeds 1 eV": the
  converging device swung −1.39 → +1.95 eV in its first two NEGF iterations.
  P4 sets the rule on the two measured runs. M3 P4.

### 5t.5 P2's design, corrected by the 2026-09-26 review — **decided: all eight** *(user, 2026-09-26)*

The review read the P2 drafts against every door the record would use and
the files real attempts leave. Five of its corrections are built and one
overtaken (2: the record is read through `runs.folder_answer`,
`runs.py:429`) — the archive; what remains:

* **4.** **The columns from the run itself.** *Default* and *asked* as the run
  recorded them — PySCF's fence; SIESTA's wrapper fence extended to list
  every item's default — not today's catalogue. *Used* never picks one of
  several readings, and shows the engine's own echo where one exists
  (`Number of poles = 42`). Rows by engine, calculation and stage; items
  sharing a block as one row. Owed: the echo column (`parse.md:1535, 1620`
  promise it; `parse/dirs/setup.py:148-215` builds default / asked / used /
  differs and no echo — read 2026-10-18).
* **6.** **Symptoms are P4's**; P2 carries asked ≠ used only. Open with P4.


## 5u. Transport — the one list (M5, W43) *(consolidated 2026-09-29)* — DONE 2026-10-11 but for S13 (the convergence sweep, step 6) and E13 (the junction on Sol, the user's)

*(user, 2026-09-29: "after all work on spectrum is done and review is green,
continue to work on transport as the next item on plan remember to correctly
update the plan and consolidate")*

**What this section is now** *(trimmed 2026-10-08; re-checked 2026-10-09)*.
§ 5x — archived 2026-10-09 — took steps 1–6 and 9; what stays here: step 5's
last line, steps 7, 8 and 10 (the queue's Q5), the TD1–TD14 decision record;
step 11 is Q17 (§ 5z). The paragraph as it stood: **§ 5x holds the order for transport since 2026-10-08** — the sweep,
the record and the Results tab (B5, B6), the small composition rules (B7), the
milestone review (B8) and the document sweep (B9). What stays here: the steps
§ 5x did not take over — 5's remainder, 7, 8, 10, 11 — the two owed contract
texts, the TD1–TD14 decision record (cited across the plan), and the stale
text § 5x B9 takes as its transport input (§ 5u.3, § 5u.4). Steps 1–4 are
done and archived (`4c612262`, `b63f219d`, `325cb1d4`; step 4's ladder
`claude-au-bdt-au` is no longer on disk — § 5x.0's road junction is the
acceptance now); step 6 is superseded by § 5x B1. The rows this section
absorbed point here (§ 5u.5).

### 5u.0 What is true today — ARCHIVED 2026-10-08

*Superseded by § 5x: the road is the ordinary one and the three cases ran on the road junction (§ 5x.7's record, archived).*

### 5u.1 The order — each step closes as a milestone does (§ 0a)

| # | step | absorbs | done when |
|---|---|---|---|
| **5** | **Submission groups — one queue wait per rung, not per point** — TD2's ruling; its order against M2 is TD9. **The group built 2026-10-07** *(user: "group at prep")*: `prep task --stage seed --stage electrode_L --stage electrode_R` names a group — each member prepped as alone, one save, none building on another (refused by name), one shared allocation; `Job.group`; the group's header at prep; `launch task` by the same names sends one job walking them (`project-layout.md` § 1.6.6; `jobset/group.py`). Walked on the road junction (§ 5x.7: seed + leads, one group job). The bias-point walk is § 5x B4's one walker (`submit._walk_script`). **Left**: the group's header checked on a machine with a scheduler (Q13). The frame walk built 2026-10-10 (Q17-d): frames are points of one run (D4), walked as voltages are, a chain per frame (`transport.md` § 2a.11) | TR9; § 2a.7's *default grouping*; W32 ③'s *the gather runs per frame* | the frame walk on the group: the device and the transmission ONE submission per scan walking frames, each from the shared seed, continuing past a failed frame; members stay prepared runs with their own records; the allocation sized from the members' own requests, a member that cannot share it refused by name; the order and the stop-or-continue rule read from the ladder's own `stage_inputs` and the axis's declaration. Written into the contract first — `project-layout.md` § 1.6.6 and `task-setup.md` § 1 carry the group (2026-10-07); the frame walk is written in `transport.md` § 2a.11 and `job-system.md`. The group's header seen on a scheduler |
| **7** | **The rest of the TranSIESTA / TBtrans surface**, each through step 1's sections and checked in the engine's source. **Done 2026-10-11, at 2.1.0** (`6c518057`): the four new rows (`TS.SolutionMethod`, `TBT.SolutionMethod`, `TS.Hartree.Fix`, `TS.Contours.nEq.Fermi.Cutoff` with `at_least`), blank ≠ 0 on the two device broadenings with `written_by` and `jobset migrate` for the old zeros (user, 2026-10-11), TD11, A1.12's counting half, the `bloch`, reservoir-temperature and tbtrans-log notes (`transport.md` § 2a.13, `template.md` §§ 3, 5, 5.3, `changelog.md` 2.1.0). Two review rounds, findings fixed (a rung's old 0 in `task.json` refused and migrated too). **Open**: the spin-polarised device run (`TBT.Spin`, K21's two channels). **Tried 2026-10-11** on the user's word, the task made through the browser (Transport tab → Task setup → the printed prep/launch, `projects/claude-transport-walk/transport/h-spin`: the relaxed H-chain pair, unrestricted, the moment free, 0 V): the seed and both leads converged fully polarised (moment 22, and 9 per lead, from their first step); the device's NEGF cycle held at dQ = 2.3 and a moment of 19 through 1000 iterations and its warm retry — `SCF_NOT_CONV`. **Finished procedurally** on the user's word (*"relax the convergence parameter, and let it finish just procedurally"*): the device's charge check is a setting now (`ts_scf_dq_converge` / `ts_scf_dq_tolerance`, 2.3.0 — SIESTA writes `.TS.HSX` only on convergence), set off with the cycle cap at 100 on the device's row in Task setup; the transmission ran and `summarize task` read **both channels' transmission** (`channels: up, down`, the current their sum). ~~**Found**: the DOS, PDOS and eigenchannels are not read for a polarised run — tbtrans writes `TBT_UP.nc` / `TBT_DN.nc` and the record looks for `TBT.nc` alone (`dos_why`)~~ — **fixed 2.4.0** (K21: `tbtnc.tbt_files`, `spin_dos`; `transport/record.py:831-850` reads both channels, `dos_why` only for a missing file; re-read 2026-10-18). The Fermi levels read `nan`: SIESTA printed an overflow for this state (`Fermi = 2178 eV`, `**********`), which the reader passes on. TD14 retired (below). **A known limit**: the template's `written_by` dates the whole calculation, so a calculation described before 2.1.0 whose template never needed migrating (both broadenings set to numbers) reads a 0 typed on a rung today as an old one — refused at prep, and `migrate` writes it blank and says so. **Re-checked 2026-10-09**: `TS.Elecs.Eta` has its row; ~~no `TS.Contours.nEq.Fermi.Cutoff` or `TS.Hartree.Fix` row~~ — *2026-10-18: every row of 2.1.0 is in the catalogue (`TS.SolutionMethod` `:2648`, `TBT.SolutionMethod` `:2672`, `TS.Hartree.Fix` `:2693`, `TS.Contours.nEq.Fermi.Cutoff` `:2714`, `TS.Elecs.Eta` `:2633`)*; `electrodes_bulk` declares `shared` and no `stages` (`[item.electrodes_bulk]` — TD11); the transmission's SCF settled 2026-10-09 (§ 2, TD12); the transverse check measures per lead, not per layer (`validation/__init__.py:516-543` — TD14). **Re-checked against SIESTA 5.4.2's source 2026-10-11**: `TS.SolutionMethod` and `TBT.SolutionMethod` default `BTD` (`m_ts_options.F90:180`, `m_tbt_options.F90:490`); `TS.Hartree.Fix` defaults to the transport axis's letter, `C`, and any other spelling — `-C` included — prints *"Requesting a different hartree plane than the default! … YOUR responsibility"* (`m_ts_hartree.F90:50-72`); `TS.Contours.nEq.Fermi.Cutoff` is in kT, default 5, and below 2.5 the run dies (`m_ts_contour_neq.f90:141-144`). **Taken off 2026-10-11** (said to the user): `TBT.T.Out` — tbtrans's default is `N_Elec == 1` (`m_tbt_options.F90:634`) and a transport run is two-terminal (`transport.md` § 4), so there is nothing to state; a per-lead temperature (`TBT.ChemPot.<>.ElectronicTemperature`) — both leads take the one shared electronic temperature, and a per-lead one is a thermoelectric feature nobody designed; the chemical-potential block says so in a comment. TD13 retired (§ 5u.2) | W25; § 5o.6's open rows; TD11, TD12, TD14 | `TS.Elecs.Eta` has a row (today TranSIESTA's 1 meV applies unsaid); `TS.Contours.nEq.Fermi.Cutoff` has a row, 5 kT, refused below 2.5; `TS.Hartree.Fix C` written, fixed by the rung, never typed; the transmission header saying which program reads what and pointing to `tbtrans`'s own fdf log (TD12 — nothing is cut); `electrodes_bulk` carried by the two rungs that read it — `shared` with `stages` meaning one value for the named rungs (TD11); ~~the transverse check measuring each atomic layer of the lead (TD14)~~ — retired 2026-10-11; a spin-polarised device run end to end (`TBT.Spin`; its record reads both channels' `.TBT.nc` — K21 — and the run is e2e, on the user's word); the chemical-potential block saying both leads take the shared temperature; A1.12's counting half (the element, not the raw label — the warning itself stays as written, F9 saying it once per group); both solvers (`TS.SolutionMethod`, `TBT.SolutionMethod`, `BTD`, today unsaid) and `bloch 1 1 1` with its note stated in the deck — TD1's *every parameter explicit*, after step 10 so each new line carries its source |
| **8** | **One panel per engine** (step 3, which it waited on, done 2026-10-02). **Done 2026-10-11**: card 4's engine strip — TranSIESTA live, PySCF-NEGF disabled with its reason — from `transport.stages.engine_tabs`; the schema route and the describe door take the engine and refuse one that runs no transport (`transport.md` § 3.8.8). The shared panel kept across a reload too (the user, 2026-10-11: *"you should fix #2 … a framework, like using the correct prefix/tag when calling persistence module"*): what the shared panel and each rung's tab hold is one workspace note under `transport:form`, through the workspace's one door (`web/tabs.md`, `workspace.md` § 4) | W24 = W30 ⑤ = `engines/transport.md` § 3.8.8; § 5s.4's *the transport shared panel is not persisted* | the tab's engine strip with the disabled PySCF-NEGF tab, its fields from the catalogue (§ 3.8.8's *"`TransportConfig` keeps its name"* superseded by TD4); an engine dimension on the schema route (a PySCF-NEGF backend itself — its `spec_for` arm, catalogue rows and config — is `transport.md` § 8's follow-up, not this step) |
| **9** | **The Results surface, finished** — **→ § 5x B6.** Designed 2026-10-07, `web/results.md` § 2.5 (:440-463), approved (user: "yes, go ahead with the build"); ① record fields (DOS, PDOS by label, eigenchannels, per-rung SCF, NEGF figures, chain, caveat) + the shared SCF-plot module and ② the report's cards, tabs, two selections, PDOS of a selection by atoms and orbital type — **built**; ③ the frame axis waits on step 11; the rest is § 5x B6 (R1–R4, R6, R7, R9, the low-bias I(V)) | W10; § 5c.3 (e) and (f)'s rung half — (f)'s citation half is built (`transport.js` draws `provenance.slot`); TR10's chain; W30 ④ (the deck viewer) | § 5x B6's: `summarize task` and the Results tab on the road junction, the three cases. Named here from the first done-when, for B6 to carry: presenters read through a door rather than `JSON.parse` (a framework door, not transport's alone); the decks viewable per rung, run and bias point beside their `.validation.txt` (W30 ④) |
| **10** | **Every value says where it came from** — TD5's ruling, framework-wide. **Done 2026-10-11** (`bac8ccbe`, `2c951f48`): each deck's PROVENANCE block says who set each value it is rendered from (`parameters:`, `ResolvedConfig.deck_sources`, both render roads; `job-contracts.md` § 3.2); a setting left to the engine says so where its line would be, in the item's own `not_set` — the catalogue key the user named (*"unset may suggest removed or cleared by user"*), required on every optional engine/deck item (`template.md` § 5, § 6.6); Task setup's *What came over* lists every value shared by every stage with whose it is (`task-setup.md` § 3). Reviewed in two rounds by fresh agents, every finding re-read against the code and fixed (`048824a7`, and the commit after it): the table names the step that set a value after `resolve` and who answers a blank; both PySCF decks say their unset solvent and fitting basis; the diagonaliser and output items declare the SCF rungs, so the transmission's table and tab carry none of them | W30 ③'s remainder; `engines/template.md` § 6.6 obligations 2–4; the read-only echo of shared values | a template item carries `source` (`cited` · `person` · `default` — `record` retired, `template.py:1168-1172`; built, K7), written at `init` and by the describe door, read by every surface; each deck's provenance block carries the per-parameter table; an `optional` item left unset writes a line naming the engine default that applies |
| **11** | **The frame axis → Q17 (§ 5z)** — on step 5's group (its *Left*). **Done 2026-10-10** (Q17, archived) | W32 ②, ③, ⑤ (④ left transport by TD10's ruling: transport is handed a multi-frame structure and reads each frame's details from its `customized` section; the generator is V1.25's own procedure) | as W32 states them, the frame walk on step 5's groups; step 9 ③ with it |
| **12** | **A transport description's life** *(user, 2026-10-11, on the AuBDTAu walk; design B of the framework analysis — who writes which part, when)*: ① Describe is one act and the server writes it (`/api/transport/describe` writes the template and `task.json` into a folder holding none, through `describe.write_description` with `jobset init`'s two refusals) — `engines/transport.md` § 3.8.2a, `web/handover-procedure.md` § 4, § 6; ② once written, Task setup is its editor as every kind's: the Stages card's rows are the ladder's rungs, fixed and numbered — only the seed removed or added back (ruling Q4) — the picker naming the rungs that read each item, the shape shown fixed — what a kind fixes is `task.fixed_by_kind`, which the reader enforces (`web/task-setup.md` § 4, § 5.3); ③ the launch rows stand on every rung's run card, blank — the catalogue's `run_card` (§ 6.2b); ④ ~~the junction card's centring offer~~ — built at 2.5.0, retired at 2.6.0: the junction enters the calculation centred (row 13 ①); ⑤ the swap choice reaches the description (the sender never sent it) | — | built and reviewed twice; `AuBDTAu/transport/transport_lowbias`'s junction composes centred |
| **13** | **Found on the 2026-10-11 browser walk — the user's word first** (the road ran: Describe wrote the description through the server, Task setup echoed it, Preview refused until the launch rows were stated, then composed the AuBDTAu junction centred): ① ~~D7~~ — **ruled 2026-10-11, built at 2.6.0**: a structure entering a new calculation is centred unless its origin is a person's; a vibration's input stated relaxed keeps the engine's frame (`structure-periodicity.md` § 6.0, *Entering a new calculation*; the sidecar records whose the origin is, v12); D7, lost in `d7f37054` with no decision, is subsumed and 2.5.0's `centre_junction` retired; ② **closed 2026-10-13 (2.9.1): `stages = "scf"` on both, `template.fixed_answers` for the one-answer rule** — the transmission tab echoed `WriteForces` / `WriteCoorStep` as fixed on, though tbtrans writes neither — § 6.1b keeps their one answer for every rung on purpose; limit them to the SIESTA rungs (`stages`), or keep; ③ **closed 2026-10-13 (2.9.1): the locked box shows the list's points (`locked.per_point`, `formSchema.showPoints`)** — the device's and transmission's *Bias voltage* echo was an empty box — show the bias list's points; ④ **closed 2026-10-13 (2.9.1): counted against the heavy non-metals, none on a gold lead** — the H/heavy reference note counted gold as heavy — 0/75 on every lead; ⑤ **closed 2026-10-13 (2.9.1): it names Task setup's card, then the key and the flag** — the run card's refusal named `task.json` and flags, not the card on the page; ⑥ ~~at ~1200 px the Transport tab's main column is narrow: the rung strip overflows and the run card's number boxes show a dash~~ — **fixed 2026-10-11** (second walk): the `.tabs` strip wraps (tabs 4–5 were clipped out of reach), and the run card's boxes are typed text inputs, so the theme styles them. **The second walk, end to end on `claude-transport-walk/transport/e2e-walk{,2}`** (Describe → refused again → Task setup: stages edited, seed removed and added back, Save, run cards, Preview, Prep → Results), fixed: ⑦ the machine card and `STAGE-PLAN.md` listed `molbuilder.json`'s `launch._comment` as a value it supplied (`config_provenance`); ⑧ a Prep's answer stopped before the terminal's *next* line, and every rung's tab named its stage alone, so a group's one launch was shown nowhere — the answer carries `next`, and the tabs ask the status door's rule (`runstatus.GroupStep.next`, `runstatus.py:164-179` — the row had named a `launch_words` door that does not exist, corrected 2026-10-18; D4); ⑨ a waiting rung read *"…from seed, whose newest run, 01_seed/run-0, which has not been launched"*; ⑩ the Transport tab drew all five rung forms three times on load — restoring the shared panel fired its change listener; ⑪ the describe refusals named the folder by its absolute path. **Open, the user's word first**: ⑫ **closed 2026-10-13 (2.8.0): a blank transmission mesh is the rule's, and its hint reads *not chosen · (3 x the shared k-point mesh)*** — the transmission grid's one-line hint read *"not chosen · 15, 15, 1 you set this"*: the value is the 3× rule's (`kmesh.transmission_start`) on the shared mesh the person set, and the line says neither the rule nor its purpose, which only ⓘ help gives (§ 0.3b) — proposed: *"3 × the shared k-point mesh (5, 5): T(E) needs a finer in-plane mesh than the density; converge until T(E_F) moves < 1 %"*. Traced 2026-10-11: the cited AuBDTAu run is 3 × 3 × 1, offset 0; the user's `transport_lowbias` set 5 × 5 × 1 (→ 15 × 15 × 1); the 4 × 4 seen on the walk was the walk's own test edit, since undone in the user's note. Parity and Γ, from SIESTA 5.4.2's source: the block form reaches `find_kgrid` with `firm_displ = .true.` (`kpoint_t.F90` `setup_mp`; tbtrans's `m_tbt_kpoint.F90` likewise), so the 0.0 offset is kept, and the index runs −(n−1)/2 … n/2 — Γ is a sample on an odd or an even count; an even count adds the zone edge. The 3× rule keeps parity (5 → 15, 4 → 12). A parity rule, if wanted, is chosen with the literature first; ⑬ ~~GPU binding on a machine with no queue~~ — **withdrawn**: the box is the calculation's own switch (`execution/gpu.md` G9, the person's column of § 1), and the queue card writes `allocation` whole (`web/task-setup.md` § 6); a machine fact decides nothing about it; ⑭ **moot 2026-10-13: the seed is always kept** — without the seed the stage folders renumbered (`01_electrode_L/`) while the rows read *2 · electrode_L*; ⑮ **closed 2026-10-13 (2.9.1): MolView's hidden radios positioned in their labels — they had stretched the document** — a focus jump (find-in-page, `scrollIntoView`) scrolled the window itself, leaving the header off-screen and a blank band below — `body`'s `overflow: hidden` reaches the viewport, which can still be scrolled by script | — | each ruled; the ruled ones built |
| **14** | **Groups are stages' numbers; prep writes every group; launch gathers** *(user, 2026-10-11, after the browser walk found the page treating a group's one job as three stages: "the prep UI clearly still treats each stage separately ... steps that are beyond logic"; "the checkbox for grouping is awkward: it cannot assign different groups"; "prep should be able to prepare all groups at the same time ... that's the launch status, not the prep")*. **Ruled 2026-10-11, the four questions as put**: ① each stage states a **group number** in `task.json` (`stages[i].group`), edited on Task setup's Stages card: the first stage is group 1, each later one repeats the number above it or adds one; a stage may not repeat the number of a group holding a stage it builds on; the defaults repeat where allowed (transport 1 1 1 2 3); ② one tab per group on Task setup, holding the group's one run card — what the job shares (ranks, cores per rank, GPU use and count, queue) written to every member's `execution`; the wall per stage, the job asking their sum; ③ **one prep prepares every group** — each group one job — and **the gather moves to launch**: `launch task` sends the next group, first checking each stage's upstream newest run finished and ran the deck described now, copying the inputs with their `.gathered-from`, refusing before anything is sent when not; ④ a stage **prepared and never launched may be prepared again** (no run's record to falsify). **The design, traced and reviewed: § 5aa** — V settled (the force-constant stage keeps its geometry in its deck), R answered by a save before a group's first launch (put to the user with the contract). Contract first: `job-system.md` (*The task*, the states, § 5.0, § 5.4, § 6.0), `project-layout.md` § 1.6.6, `stages.md` (the field), `transport.md` § 1.1, `task-setup.md` (§ 5.3, § 11, the run card), `web-api.md`; then code; then the user's `lowbias` setup proved in the browser (this machine; a Sol preview) | the walk's grouped-prep refusal on `AuBDTAu/transport/lowbias` (2026-10-11: the job's resources stated on the seed only) | proved through the browser on the user's setup, two review rounds — **done 2026-10-12** (2.7.0; § 5aa, archived) |

**TD12 — settled 2026-10-09** (§ 2, TD12): `engines/transport.md` § 6.1b states what tbtrans reads and the deck's header says so; the four SCF rungs carry their own SCF settings and the transmission none, its tab saying why (`RUNG_NOTES`), the user's word on B8 U1's cut.

**Owed by TD10's ruling** — **written** (`36a485e6`; 2026-10-09; `archive/2026-10-10-q17.md`, § 5z.2). The paragraph as it stood *(2026-09-29; open 2026-10-08 — `transport.md` names no `customized`)*: the contract text — `engines/transport.md` § 2a.9 (the frame axis: handed a multi-frame structure, each frame's details in `customized`), `model/structure-molstruct.md` § 6.1, `engines/vibration.md` § 5.6 and § 5.10 ③, `science/normal-modes.md` § 4b.6 G (the generator a separate procedure; its record in `customized`, not `info`) — written once the M11 reviews, which are reading those documents, report.

**Beside every step: the documents.** Each step fixes the false claims and
stale citations in the documents it touches; the sweep at the end is **§ 5x
B9**, whose transport input is § 5u.3's residue and § 5u.4.

### 5u.2 Rulings *(user, 2026-09-29, on the questions as they were put)* — the decision record; every one ruled (TD6 on 2026-10-09, `engines/transport.md` § 2a.7)

*(Numbered TD — transport decisions — because D1–D7 already name § 2 rows. The
questions as put, and the user's words, are in the archive.)*

* **TD1** — the device's equilibrium contour (TranSIESTA's fallback 42 points lost the charge on `claude-w33`'s device; `TS.Contours.Eq.Pole 10 eV` = 123 points held it): **10 eV, always written, the count stated beside it** (an ENERGY — TranSIESTA counts its points by how far up the imaginary axis they reach, `N = int(E / (π·k_B·T))`); interim until M3 P4's stated `contour.eq` (§ 5x's parked P3). Step 2 (done); the rule is `transport.md` § 6.1c.
* **TD2** — grouping the ladder's submissions (TR9): *"not a hack but a sequence-aware design framework"* — **build it, on the chain submission**. Step 5.
* **TD3** — vacuum in a transport structure: refused along transport beyond one layer spacing of the lead (measured from the lead) and on a transverse axis declared periodic; **allowed** on an isolated transverse axis, from the structure's stated vacuum, and so is the box built from it; the k-grid along transport 1 on the device and the seed, dense on the electrode rungs. Step 2 (done); § 6.1c.
* **TD4** — `TransportConfig`: **retired** — the Results tab reads the record and the ladder's status, never the class. Step 3 (done).
* **TD5** — how a value records where it came from (`engines/template.md` § 6.6's three proposals): **all three, as proposed**. Step 10.
* **TD6** — per-rung convergence defaults (§ 2a.7): **decided after the first ladder**; every rung's convergence settings stay on its own tab, the person's to change. After step 4 — **due**: the ladders have run (§ 2, TD6).
* **TD7** — the bias treatment as a named choice: **build it** — superseded in form by § 5x decision 1 (one explicit switch, `low_bias_approximation`; B1, `7351a3ff`).
* **TD8** — one acceptance ladder *(user: "TD8 yes")*: `claude-au-bdt-au` closed step 4, E13's workstation half, W33 P5 and W35 P3's done-when; `claude-w33` and `claude-vib-ui`'s fake junction retired as acceptance targets. **2026-10-08: `projects/claude-au-bdt-au` is no longer on disk; § 5x.0's road junction (`claude-transport-walk/transport/au-dta-t`) is the acceptance now** — flagged to the user.
* **TD9** — step 5 against M2 (which attempt *concluded* — M2f; the bias chain's restart list — M2g; a seed skipped by removal — M2i; one placement record — M2k): **decided 2026-10-02** *(user: "#9 ok")* **and done** (Q4: M2f, M2g, M2i done; M2k's placement done, its GPU half M2k's own; step 5 built `0e11a018`). Recommended then: steps 1–4, then M2f, M2g, M2i, M2k, then step 5 onward. On 2026-10-08: M2f, M2g, M2i done; M2k partly; step 5's group built. Step 5.
* **TD10** — the displacement generator's one home: **transport builds no displacement.** It is handed a multi-frame structure (one pair, many frames); the person is responsible for the frames being one structure with only the relevant atoms moved; the structure's `customized` section (W39) carries each frame's parameter set — its displacement, the mode's frequency, whatever else tells that frame's details. The generator is a separate backend procedure (V1.25): the equilibrium relaxed structure and a normal mode in, the multi-frame structure out. W32 loses ④; W42 ③ uses the procedure. Step 11 · V1.25 · W39; the contract text owed above.
* **TD11** — `electrodes_bulk`'s scope *(user: "TD11 a")*: `shared` and `stages` declared together mean **one value, carried by the named rungs only**; `electrodes_bulk` declares both, and the shared panel says which rungs read it. Step 7.
* **TD12** — what the transmission deck carries *(user: "a comment or log showing what are the parameter taken by tbtrans would be enough … no point adding complexity between steps")*: **nothing is cut.** The header says which program reads what — `tbtrans` reads the `TBT.*` settings, the `TS.*` declarations and, by fallback, `ElectronicTemperature`; the SCF and output settings are the ladder's shared set it does not read — and points to `tbtrans`'s own fdf log in the run (`fdf.<time>.log`), the engine's word rather than ours. Step 7 (the header only). *B8 U1 then cut the SCF items from the transmission; the user settled it 2026-10-09 — the four SCF rungs carry their own, the transmission none, with its tab saying why (§ 2, TD12).*
* **TD13** — correcting a junction's transverse kinds without a new relaxation — **retired 2026-10-11** *(the user: "Retire TD13"; ruled "TD13 a" on 2026-09-29, a transport-side declaration of the transverse kinds)*: the code builds no recovery machinery for a wrong state (2026-10-07), so the kinds are what was cited — the relaxation's deck, or a pair's own. A wrong declaration is refused at the first prep, and the way out is the road that exists: relax again after declaring on the Cell page, or save the run as a pair, set the cell on the Cell page and cite the pair — the edit marks the pair's record outdated and keeps it, and the citation reads its settings (`transport.md` § 6.1c).
* ~~**TD14** — the transverse check measuring within each atomic layer of the lead~~ — **retired 2026-10-11** *(user: "i would leave scientific check to the user. our check should really be the most obvious, and consistency check. we are creating tools, not to invent or limit scientific exploration.")*: whether a lead's cell tiles is the person's to judge. Measured before building, the per-layer rule would have refused correct leads — a graphene sheet with transport along zigzag 2.0, a bcc(111) lead 1.63 against the 1.5 limit. The whole-lead check stays as it is: vacuum on an axis declared periodic, the structure against its own declaration (`transport.md` § 6.1c).

### 5u.3 / 5u.4 / 5u.5 — ARCHIVED 2026-10-08 (B9 done)

*The residue list, the stale citations and the absorption map are in the archive; B9 swept them (the commit after `93977f5d`). What B9 did not reach is a § 2 row.*

---


## 5aa. Groups, every group prepared at once, a run's inputs taken at launch — ARCHIVED 2026-10-13

Built from 2.7.0: a stage's group number in `task.json`, prep preparing every
group picked in one plan, a group's inputs gathered at its first launch, one
hand-over, and review rounds A and B. The design, its trace, its reviews and
its status are verbatim in
[`archive/2026-10-13-5aa-groups.md`](?doc=archive/2026-10-13-5aa-groups.md);
the rules live in `execution/job-system.md`, *The task*, and
`web/task-setup.md` §§ 5, 11. Nothing of it is open: the engine e2e batch
passed on the grouped code (302 tests, 2026-10-13), and two small code items
it named are in Q19's Python list, with its design calls for the UI, the CSS
and the Python.

## 5z. Q17 — transport over a set of frames — ARCHIVED 2026-10-10

*Done 2026-10-10 (Q17-c … Q17-f, each with its real-junction e2e — 41 passed
at the close; the milestone review's two rounds; an H2 frame set walked
through the road in Chrome). Its rulings, its build history and its design
(A–E: the parameters, the cycle, the folders, the launch) are in
[`archive/2026-10-10-q17.md`](?doc=archive/2026-10-10-q17.md). The rules it
wrote live in the contracts: `engines/transport.md` § 2a.9 (the input — a run
or a pair, one frame or a frame set), § 2a.11 (a point is a frame and a
voltage), § 2a.12 (the record, the data file and its notebook recipe), § 3.1
(what you can cite, *a cited pair is left alone*, *a calculation's folder is
its own*), § 3.8.1 (the frame-set card); `model/structure.md` § 2.2d–2.2f
(frames, `customized`, a mode's frame set's definition); `model/overview.md`
§ 2 (the index convention); `web/results.md` § 2.4–2.5 (the frame family, one
Plotly box); `web/task-setup.md` § 3. Open from it: the frame generator,
V1.25 (§ 2), its amplitude waiting on the user; the AuBDTAu workflow, the
user's, on Sol (the handover).*

---

## 5w. The M11 review — what is left of its mechanism classes *(W50, 2026-09-29; rewritten 2026-10-09)*

*The five static reviews of 2026-09-29 (Structure optimization on SIESTA /
PySCF, Spectrum on SIESTA / PySCF, Transport), each defect re-read in the code
and the engine's source, are
[`archive/2026-09-29-m11-static-review.md`](?doc=archive/2026-09-29-m11-static-review.md);
the classes K1–K22 and their rulings as they stood, with the 2026-10-09
verdicts, are in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
§ 5w. **The fix is the framework's, never the call site's** *(user,
2026-09-29: "the solution to the issues identified through the review should
be a systematic holistic and framework level with good api unification")*:
each class closes with one declaration in the catalogue and one door every
surface asks (`template.md` § 6.5, § 6.6).*

**Done:** K1–K8, K10–K12, K19, K20 (archive 2026-10-08); **K15** — the deck's
values written to two significant figures by one formatter, `as_written`
(`pyscf/layout.py:132`; the user's ruling, `pyscf.md:701-704`, `164ecbfe`);
**K16's record** — `script_emit.declarations` (`:863`), read narrowed by
`parse/dirs/setup.py:172`; **K14's name map** — `AS_RESOURCE`
(`jobset/model.py:58-63`), read by `resolve.py:160-169`, `prep_inputs.py:259, 300`,
`placement.py:69`; its blank budget superseded — an unstated value is refused
(`jobset/model.py:67-71`, `runwrap.py:3513`, the 2026-10-02 rule; lines re-read 2026-10-18).

### 5w.1 Open, in this order (step 2 of § 0a's order, but K17's three named items in step 1 and K22 in step 6)

| | what is left | where it is | done when |
|---|---|---|---|
| **K13** | **one atomic writer** — private writers remain at `checkpoint.py:660` (`_atomic_write_bytes`), `sidecars/molstruct.py:648-680`, `workingcopy_structure.py:565-572` (~~`sidecars/spectra.py`~~ — through `persist.write_bytes` since 2.15.0, V1.37); 19 raw `write_text` calls in 12 files on 2026-10-18 (`envs/builds.py` ×5, `script_emit.py` ×2, `projects.py` ×2, `jupyter.py` ×2, `web/blueprints/build.py`, `trajectory_log/format.py`, `serve_daemon.py`, `monitor.py`, `envs/initconfig.py`, `envs/abi.py`, `describe.py`, `_threedna.py`; `compose.py`, `jobset/summarize.py` and `jobset/_cli.py` have none now) | `persist.py` | every file molbuilder writes goes through `persist`'s atomic writer |
| **K14** | **the engine's parallel model as data** — today a name branch: `runwrap.py:4291` asks `placement.deck_runs_one_process` (`:4154-4158`), which is `str(engine) == "pyscf"` (`placement.py:172-176`), and `:3510`, `:3564-3572` still branch on the suffix (re-read 2026-10-18) | `job-contracts.md` § 6.2 | the model (MPI or OpenMP) declared per engine and read by the scheduler header |
| **K16** | **the wrapper fence narrowed** — the SIESTA fence lists every SIESTA item (`runwrap.py:1774`), PySCF narrows by calculation only (`pyscf/layout.py:305-320`); the record side narrows by stage already (`parse/dirs/setup.py:172`) — against `parse.md` § 5d.3a's *engine, calculation and stage* (marked not built) | `model/parse.md` § 5d.3a | the fence writes the rung's own items |
| **K18** | **a PySCF run's level of theory** — `parse/contract.py:45-48` reads SIESTA decks only (its docstring `:14-17` says so) | `parse.contract.contract_of` | a PySCF deck's basis, method, functional, charge and spin read by the one reader; the Results export carries `info.calculation` for a PySCF run |
| **K17** | **physics needing a build or a refusal** — **done 2026-10-17**, each re-derived against the code first: **fixed** — the k-point split, off with ELPA (SIESTA's default solver here, which runs over orbitals only and turned it off itself on every multi-k run while the deck said `.true.`), warned when set on with ELPA, the manual's words, and written as set (`siesta.input`, `config.siesta._validate_parallel_over_k`); a pseudopotential copied by the exact name every check asks (`pseudos.psml_sources`; `find_psml`'s case and prefix fallback deleted); geomeTRIC's prefix derived from the trajectory it writes, the `_geom.log` it never writes (PySCF hands it a screen-only log setup) gone from the run-file list, the deck and the documents; a vibration's structure compared with its relaxation's k-point grid, as the run's deck will write it (`kmesh.mesh_for` through `parse.contract.contract_fields_of`). **Already handled** — mode numbers out of range (dropped by the run, which says so); a transport ladder's rungs (fixed in Task setup); the grid counted before time reversal (only a hand-set half shift on a two-point grid). **Moved** — the ECP read back and a `-V` functional with D3 to K18; PS-C2's and PS-C3's builds, refused today, are new science if wanted | as listed | done |
| **K22** | **per-atom starting spins** — **designed in step 6, brought with its science**: where the starting moments live (the structure's per-atom data, or a template item) and how SIESTA is told them (`DM.InitSpin`). Measured 2026-10-08: an unrestricted hydrogen junction cannot run — every rung's moment floats, the aligned start leaves no minority density, the empty channel's GGA potential puts its states near 3000 eV; the suite's transport run is restricted until this is built (`tests/test_transport_on_a_real_junction_e2e.py:26-32`) | no `DM.InitSpin` writer | designed, ruled, built, and a polarized junction run end to end |
| **K21** | ~~**a polarized junction's panels**~~ — **done 2026-10-11**: the polarized run `claude-transport-walk/transport/h-spin` made through the browser and run on the road; both `.nc` files read (`tbtnc.tbt_files`, `spin_dos`), the report drawing ↑, ↓ and their total by colour and legend in T(E), DOS, eigenchannels and the I–V *(user: "use color/legend to correctly present up and down spin and total current")* — `web/results.md` § 2.5 | `transport/tbtnc.py` · `transport/record.py` · `lib/inspectors/transport.js` | ✅ |
| **K9** | **engine facts restated in prose** — the declaration is done (2.23.1): every SIESTA keyword item states `engine_default`, the engine's value with its source line or what it follows, the catalogue refusing one that states neither, each checked against SIESTA 5.4.2's source and, where a run used it, its fdf log. Open: the help's deviation sentence, the template comment and the deck's *not set* line written from it, and the e2e step that checks the table against a minimal run's fdf log | `engines/template.md` § 5, 6.5–6.6 | the e2e check, then the prose written from the declaration (ruled 2026-09-29) |
| **K11** | its road test — the leads and the seed concluded, `mesh_cutoff` changed, the device's prep refused — ~~is not in `tests/`~~ — **judged 2026-10-18: kept**, it is an assertion on the real junction's run (`tests/test_transport_on_a_real_junction_e2e.py:309-319`, `_ANOTHER_INPUT`: `mesh_cutoff` and the k-grid displacement changed, the launch asked dry refused naming the keyword) | — | ✅ |
| **K3** | ~~parked~~ — **done 2.16.3**: `transport/sort.py` and `transport/compose.py` import `kmesh.TRANSPORT_AXIS` | — | — |
| **K10** | ~~parked~~ — **superseded 2.42.0**: the relax rung it spoke of is retired, SIESTA's vibration one run in phases (`siesta/input.py`) | — | — |

### 5w.2 Then the text

The doc-drift and nit findings are one sweep, **after** K9 — several of those
texts become data there. The list (the archive's § 5w.2) is partly stale —
T-F21 is done (`validation/__init__.py:393-395`), PS-C5 has no *auto-load*
text left — and is re-derived before the sweep. Then W50's step 3: the road in
the browser per track, with the reviewers' probes among its runs.

---

## 5v. Documents that lag the code — ARCHIVED 2026-10-09

*In [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
— its entries corrected on 2026-10-09 (the transport invariant table's
`species_order` row, I15; I14's owed clause; § 2a.13's `TS.HS.Save`;
`template.md` § 6.6; `junction-cell.md` against X3; `normal-modes.md`'s
comparison tool; `vibration.md` § 10; `stages.md`'s conversion;
`execution/overview.md`'s fork; `parse.md`'s *one test*; the archived plans
named as *the plan*); what is left is § 2 (W15's #102 / #103, TPL,
`checkpointing.md` § 13.4 → W45).*

---

## 5x. Q14 — Transport, made whole — ARCHIVED 2026-10-09

*Done 2026-10-08 (B0–B9, the review's two rounds clean); in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
with its decisions (§ 5x.5) and its road walks (§ 5x.7) — the contracts it
wrote are `engines/transport.md` § 2a.10–2a.12, § 3.1 and § 3.8. Its open
remainder is § 2's F9, F11 and F14 (F19b done 2.9.6; R2-15 closed).*

---

## 5y. Q15 — No fake engine — ARCHIVED 2026-10-09

*Done 2026-10-09 (`22ac10d9`, `4963406b`); in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md).
The rule lives in `process/testing.md` — tier 1 (`:28-40`: no engine runs
in the basic suite, fake or real) and its patterns (`:757-779`): an API test
checks what molbuilder prepares, a test that reads an engine's output is
end-to-end on the real engine, and one end-to-end pass makes each run once.*

---

## 7. The document survey and the four indexes — OPEN; step 1 done (§ 5v, archived), steps 2–4 not started

*(User, 2026-09-07: "the documents are very fragmented and a lot of key
information cannot be easily found. we also should have an index that
summarizes key design aspects, such as api, module, data structure etc. we
also should have a reference list that gives the foundation of all the
constants, scientific validation foundation and paper citations summarized in
one place.")*

### 7.1 What is actually wrong — measured 2026-09-07, re-measured 2026-10-08 and 2026-10-09

**Coverage too, since 2026-10-08.** The 2026-09-07 measurement said every
`/api/*` route was named in `web-api.md` and planned no sweep. The
route-catalogue sweep (§ 0a *Unscheduled*) overturned it: **11** documented
routes do not exist and **9** live ones are undocumented, against **86** unique `/api` routes live on 2026-10-09 (still 86 on 2026-10-18 —
74 `@bp.route`, 10 `@bp.get` / `@bp.post`, 6 on the app itself, `app.py:658-718`, `rate_limit.py:565-571`; a naive path diff of `web-api.md` against the live routes on 2026-10-18 gives 19 documented-not-live and 26 live-not-documented, the checkpoint, files, modify and workspace-storage families among the latter — the exact counts depend on how the document names a family, re-derived when step 2 is taken)
(`app.py:641-701`, `rate_limit.py:565-571`); the 71 counted only the first kind. And coverage was never truthfulness:
the 2026-09-29 check read every live document against the code and found
statements the code contradicts — § 5v holds that list, and the survey starts
from it rather than repeating it.

**Findability.** **79** live documents, **73,853 lines** (2026-10-18; 78 and 69,899 on 2026-10-09), nine directories
(`docs/` and eight below it; `archive/` and `plans/` excluded). The
information is there and a person cannot reach it.

**The existing index has the problem it is meant to solve.** `README.md`
§ Index (`docs/README.md:211`) is one row per DOCUMENT, and the longest row
runs 823 words (`:251`, 2026-10-18). To find out which document owns a fact you read
essays about documents. It is a good map of the tree and a poor answer to
"where is X".

**Two documents contradicted themselves in one day** (2026-09-07), both found
by accident while doing something else — `science/pseudopotentials.md` § 3
listed a check as not-done that its own § 2a.1 documented as done (since
corrected; § 3's note at `:468-479` records the removal), and
`process/testing.md` § 6 quoted a number after telling the reader never to
quote one (corrected since: `testing.md:702-704`). Neither is exotic.
Nothing systematic would have found them, which is the survey's real
justification.

### 7.2 The four indexes — organized by THING, not by document

Each is a lookup table, each row pointing at the document that owns the
detail. **None of them may become a second home**: a row says where a fact
lives, never what the fact is. (`README.md` R-W1, and the reason `docs/`
survived its 2026-06-02 over-compression.)

| index | one row per | the row answers | generated or written |
|---|---|---|---|
| **API** | route | what it answers · who calls it · which doc owns it | **generated** — the route list is derivable from the blueprints, and a generated index cannot drift. Two findings are the argument: W8 (2026-09-07: `web-api.md` claimed the Documents tab read `/api/docs/list`, which it has not since the commit after the one that added it) and the 2026-10-08 route sweep (11 documented routes that do not exist, 9 live ones undocumented) |
| **Module** | module | its role · its layer (L1/L2/L3) · its doc | **generated** from `architecture.md` § 3's index, the one list of each module's layer (the scan that classified every module, `test_layering.py`, was retired 2026-09-27, `a291fbbf`) |
| **Data structure** | persisted file / schema | its shape · its version · its one reader and one writer | **written** — the doors are a design fact, not derivable. The substance for a calculation's files exists: `execution/project-layout.md` § 5, every file, who writes it, the one door that reads it — **generated** from `runfiles.WRITTEN` by `tools/manifest.py`, so the index points at it |
| **Reference** | constant · citation | its value · its source · **why this value** | **written** — see 7.3 |

### 7.3 The reference index, which is the one with real content in it

Constants have one home already (`molbuilder/constants.py`), and the dialects
a quantity is written in have theirs (`molbuilder/units.py`); the rule for both
is `architecture.md` § 3.  There is no lint -- `082ba979` retired it in favour
of that written rule. What has no home is the **reasoning**, and it is
genuinely scientific. The worked example: `constants.py:101-106` defines
`HARTREE_BOHR_EV_ANGSTROM_ASE = 51.42208619`, the ASE/NIST value, and says why
the name carries the convention — `HARTREE_EV / BOHR_ANGSTROM` gives
51.422067476, 0.36 ppm apart, and a force and the threshold line drawn over it
must use the SAME one. The emitter imports it (`trajectory_log/emitter.py:40,
43`) so emitted forces line up with what a person reads in ASE, VASP and QE
logs. That is a real convention choice with a real justification, and it lives
in a code comment nobody will find.

Citations are split the same way: `science/references.bib` holds **34**
entries, **8** live documents reference it (`README.md`, `engines/pyscf.md`,
`engines/template.md`, `engines/tuning.md`, `engines/vibration.md`,
`science/normal-modes.md`, `science/vibrational-averaging.md`,
`science/refs/README.md`), and the rest cite
literature in prose, if at all. Two keys the roadmap called for — Reed 2006 and
Stokbro 2003 — are still in neither, though `engines/transport.md` § 8 cites both (row **E14**, reopened 2026-10-09).

So the reference index carries three things per row: the value, where it is
defined, and the sentence saying why that value and not the neighbouring one.

### 7.4 Order, and the one risk

1. **Survey first, and record findings without fixing them** — **done as
   § 5v** (2026-09-29): every live doc read against the code, findings in one
   list, an entry leaving it when its document is corrected. The route sweep
   (§ 0a *Unscheduled*) is the same kind of finding for `web-api.md`.
2. **The two generated indexes**, which are cheap and cannot go stale. Inputs:
   the 86 live routes; `architecture.md` § 3's module list.
3. **The two written indexes**, from the survey's findings. Input for the
   data-structure index: `execution/project-layout.md` § 5 is
   already the manifest (generated) of a calculation's files — the index points at it, it
   does not restate it.
4. **Only then** decide what merges. Fragmentation may be the symptom; the
   fix might be four indexes and no merges at all.

**THE RISK IS THIS PLAN'S OWN HISTORY.** The 2026-09-01 consolidation merged
nine documents and dropped § 5f entirely; a re-check on 2026-09-07 found
**seven more** items lost (E12–E14, N5, W16–W18), from source documents
credited with "nothing open" that were never checked row by row. A survey
that reorganizes without a per-item carry-over check will do it again. The
rule for step 4: **nothing moves until the thing it says is written where it
is going**, which is the archive's substance-first rule, and it is the reason
§ 5i could not be archived until `process/testing.md` § 2a existed.

---


---

## 8. One nature, fourteen instances — the reading doors have no guard — OPEN: rows 10, 11, 13

*(Consolidated 2026-09-07, after the selection/MolView review; re-read
2026-10-08 and 2026-10-09. Rows 1–9, 12 and 14 are fixed; the guard of § 8.3
is ruled out — a door's rival is found by reading the code, which is how every
row was found (`code-audit.md` § 1c (e)). The closed rows, § 8.2's account of
the four remaining guard tests and § 8.3–8.5 are in the archives of 2026-10-08
and 2026-10-09.)*

**The pattern.** A door exists. Someone needs it to behave slightly
differently for a local reason. They write a second implementation instead of
widening the first. The two then drift, and the copy is the one that is wrong.

| # | the door | the second implementation | how it drifted | state |
|---|---|---|---|---|
| 10 | the `forceScale` switch | the trajectory template's slider | **confirmed by reading 2026-10-09**: `trajectory/core.js:918-922` sets the switch from `#force-scale` before the load; `molview/ui-context.js:145-152` then runs `restore()` after the first structure and re-sets every saved switch (`:107-112`) — the saved value wins, and the slider (`#force-scale` / `#force-scale-val`, written only in its own input handler, `core.js:2438-2446`) shows another number (lines re-read 2026-10-18) | open — one owner, the switch; the slider reads it |
| 11 | the filter row kinds | two literal lists: the menu (`molview/ui.js:2737`) and the translator `rowToRule` (`stores.js:353-382`), with the default at `stores.js:224` (re-read 2026-10-18; `structure-annotations.md:359-360` still marks the menu's read from `channelKinds` as owed) — `_atom.js`'s `KIND` is the atom-channel taxonomy, not these (the 2026-10-08 premise was untrue) | a kind added to one list and not the other is a menu entry that builds no rule; `structure-annotations.md` § 6 marks the kind menu read from `channelKinds` and the `value` kind's range predicate not built | open — one table, read by both |
| 13 | `molview.md` § 1.1 vs §§ 6.6 / 9.5 / 11.6 | the same facts stated twice | five stale claims on 2026-09-07; unverified since | open — § 7's survey |

*Row 14 closed 2026-10-09: `model/structure-annotations.md` § 6 names the real
homes (`_atom.js`'s `toDisplay`, `atomChannels`, `channelKinds`; `stores.js`'s
`buildRule`, `rowToRule`; `ui.js`; `/api/selection/eval`).*

---

## 9. ATOM as a pseudopotential validator — § 9.1 DONE, § 9.2 PENDING, not started

*(User, 2026-09-19.)* **§ 9.1 — done, archived 2026-10-08**: the user's two
rules — the pseudopotentials are explicitly provided, and the file for element
`E` is named `E.psml` — are refusals when the calculation is configured
(`validation/siesta.py:19`, `:59-100`; `pseudos.py:636`; contract
`science/pseudopotentials.md` § C1a, `:139`).

### 9.2 ATOM as an optional extra layer — PLANNED, not started

*(User, 2026-09-19: treat it the way 3DNA is treated — ask whether they have
it, look for it under the molbuilder root, and enable the extra validation
only when it is there.)*

ATOM 4.2.7, the SIESTA project's own pseudopotential program (Froyen /
Troullier / Martins, maintained by Alberto García). Not in the repo; the copy
read on 2026-09-19 is gone, and the facts below are from that reading.

**It is the 3DNA case exactly, including the licence.** `COPYRIGHT` says
**"REDISTRIBUTION OF THIS CODE IS PROHIBITED"**, so molbuilder must not
bundle, mirror or fetch it — the same standing we already give 3DNA
(`builders/backends/_threedna.py`, "Licensing"). That makes the discovery
design a copy rather than an invention:

| 3DNA today | ATOM |
|---|---|
| in-tree `<repo_root>/x3dna*/`, version-agnostic glob | `<repo_root>/atom*/` |
| completeness filter: `bin/fiber` executable **and** `config/` present | `atm` built, **and** `Tutorial/Utils/` present (the `pt.sh` driver lives there, not beside the binary) |
| `$X3DNA`, then `fiber` on PATH | `$ATOM_PROGRAM` — which `pt.sh` already honours — then `atm` on PATH |
| `x3dna*/` in `.gitignore` | `atom*/` likewise, added (`.gitignore:85-88`): the ignore has to exist *before* the folder does, or a redistribution-prohibited package lands in `git status` |
| `BackendUnavailable` names where to download | same, pointing at the SIESTA pseudopotential page |

**How it gets there is the user's to do, once** (user, 2026-09-19), and it
is the 3DNA sequence unchanged: go to the site, accept the licence, download
the package, unpack it under the molbuilder root. molbuilder does no part of
that. From then on it is present, detection finds it, and the extra layer is
simply available — there is no enable switch to forget, and no state to keep
beyond the folder being there.

**What the extra layer measures.** `Tutorial/Utils/` holds the drivers:
`ae.sh` (all-electron), `pg.sh` (generation), `pt.sh` (**the pseudopotential
test**). The workflow the manual prescribes is `ae` over a series of atomic
configurations, then `pt` over the same series with the pseudopotential, and
compare — eigenvalues, and the inter-configuration energy changes the tutorial
greps as `&d`. That is **transferability**, measured rather than trusted. The
AE-vs-PS **logarithmic derivatives** (`logder.f`, plotted per channel) are the
standard **ghost-state** diagnostic. Those are precisely the two items
[`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) § 3 declares
out of scope today (`:446-470`, which points back here), and the ATOM
tutorial's own warning is the argument for them: *"You should thoroughly test
a pseudopotential before using it."*

**It also broadens which formats molbuilder can accept** (user). ATOM's native
pseudopotential formats are `.vps` / `.psf` — `pt.sh` takes
`<ptname.inp> <psname.vps>` — and `Util/` converts among VPS spellings
(`cdf2vps`, `vps2cdf`, `vpsa2bin`, `vpsb2asc`). So a library need not be PSML
to be usable or checkable.

> **BROADER INPUT, SAME STRICTNESS** — stated here because it is the thing a
> second format would quietly erode. § 9.1's two rules are about the FILE, not
> about PSML: whichever formats are accepted, the source is still stated
> explicitly, and the file for element `E` is still named `E.<ext>`. Accepting
> `.vps` therefore adds one question and it must be ANSWERED, not guessed: if
> a folder holds both `Au.psml` and `Au.vps`, that is a refusal naming both,
> never a preference order. Nothing about a wider door makes it right to pick
> for the user.

**The open question that keeps this from being a task.** ATOM has
`write_psml.f90` and **no PSML reader**, and `pt.sh` reads `.vps`. So it
cannot be pointed at a downloaded PseudoDojo PSML as-is. Either a PSML→VPS
path is needed, or the layer is scoped to the formats ATOM already reads —
which is also the format-broadening above, and may be the same piece of work.
That choice has not been made, and it decides whether this validates the
library we have or the ones we could start accepting.

**Also open:** whether `atm` is built as part of an env
(`feedback_no_env_deployment_changes` — nothing is installed without asking),
and whether the check runs per prep or on demand like `molbuilder pseudo
check`.


---

## Archived sections — pointers

- [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md)
  — §§ 1s, 1v, 1x, 1y, 1ab, 1z and 1aj whole, done; §§ 1t, 1u, 1w, 1aa and
  § 1ad.1 as they stood, partly done; the handover of 2026-10-20 and § 0a's
  NOW queue as they stood; § 2's TPL-L1, DOC-TOC, LEAD-TOL, RL15, V1.25 and
  A1.9 rows — each under its verdict, read against the code on 2026-10-27.
- [`archive/2026-10-21-1s-job-request.md`](?doc=archive/2026-10-21-1s-job-request.md)
  — § 1s's job-request design record.
- [`archive/2026-10-20-plan-handover.md`](?doc=archive/2026-10-20-plan-handover.md)
  — the handover of 2026-10-18, verbatim.
- [`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md)
  — the rows done in 2.9.5–2.9.6.
- [`archive/2026-10-14-rmp-design-rounds.md`](?doc=archive/2026-10-14-rmp-design-rounds.md)
  — the remapped state's design rounds.
- [`archive/2026-10-18-review-findings.md`](?doc=archive/2026-10-18-review-findings.md)
  — the seven-lane review of 2026-10-18: every finding with its verdict
  (fixed, rewritten, a decision, judged).
- [`archive/2026-10-18-5ab-remapped-state.md`](?doc=archive/2026-10-18-5ab-remapped-state.md)
  — § 5ab (the remapped state, RMP) whole, as it stood when the plan was
  cleaned on 2026-10-18.
- [`archive/2026-10-18-plan-rows.md`](?doc=archive/2026-10-18-plan-rows.md)
  — the rows of § 2 done, built, closed, retired or answered between
  2026-10-14 and 2026-10-18 (2.10.0–2.15.4), each verbatim under its verdict.
- [`archive/2026-10-18-plan-consolidation.md`](?doc=archive/2026-10-18-plan-consolidation.md)
  — the handover's push-history paragraphs, 2.0.0 to 2.13.2, as they stood
  when the plan was cleaned on 2026-10-18.
- [`archive/2026-10-13-5aa-groups.md`](?doc=archive/2026-10-13-5aa-groups.md)
  — § 5aa (the groups) whole, as it stood when the plan was cleaned on
  2026-10-13.
- [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md)
  — the rows of § 2 and the queue done, settled or closed by 2026-10-13
  (after 2.9.3), each verbatim under its verdict; and Q19's row as it stood
  after its phase 1.
- [`archive/2026-10-10-q17.md`](?doc=archive/2026-10-10-q17.md)
  — § 5z (Q17) as it stood at its close, its queue row, and the handover
  of 2026-10-09/10.
- [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
  — the queue and milestone tables as they stood; § 0b; § 0c (W55, W56, W57,
  unit 12); *Unscheduled*; the whole of § 2 as it stood, with a verdict per
  row; the pointer sections § 5a–5c, § 5f–5p, § 10 and § 11 with what they
  carried; § 5v; § 5w as it stood; § 5x; § 5y; § 5z as it stood (Q16, F.1–F.7);
  § 8.2–8.5.
- [`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md)
  — §§ 1, 3, 4, 4a, 5, 5d/5i/5j, 5k, 5l and 6 (pointer sections since
  2026-09), and §§ 5a, 5b, 5c, 5f, 5h, 5m, 5n, 5o, 5p, 10, 11 in full.
- [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md)
  — the rows closed when § 2 became the one table.

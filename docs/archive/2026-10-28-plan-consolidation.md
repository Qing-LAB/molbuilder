# The plan's consolidation of 2026-10-28

**Role:** archive
**Domain:** all

*(The user, 2026-10-28: "consolidate plan and correctly update items in the to
do, archive as needed".)* Three fresh readers checked every section and row of
[`plans/plan.md`](?doc=plans/plan.md) against the code, the changelog and the
tree at 2.47.12 — each claim at its cited place, read-only — and every verdict
below was read again before the text moved. What was done or settled is here
verbatim, each under its verdict; what is open stays in the plan, rewritten
from the same findings.

## 1. Done or settled — moved whole

### § 1ad — the six-dimension review of 2.40

*Verdict, 2026-10-28:* done — its decisions built 2.41.0, its verified findings fixed 2.46.3–2.46.5 (spot-checked: `runs.concluded(launched_in=)`; SCI11's and SCI12's text gone from the catalogue and the documents); WF10 never located, UI13–UI16, BE9 and WF11 dropped as polish; TEST-AUDIT (5)–(9) carried to § 1ak S4.

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
**Fixed 2.46.5**: UI7–UI10, DOC5, DOC14 (the dead code names and the
label checklist that contradicted *labels are the user's*), BE12's stale
`web-api.md` clauses — the error itself went with 2.42.0's `relax` stage —
and DOC13 by the plan's consolidation (2026-10-27). **Not reproduced on
2.46.4**: BE8 (`status`, `status <stage>`, on described, prepared and
finished junctions). **Left**: WF10, never located; UI13–UI16 are polish,
dropped.

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

### § 5u — transport, the one list

*Verdict, 2026-10-28:* done 2026-10-11 — steps 1–14 built or ruled (step 6 the bias treatment, TD7, built by § 5x B1, `7351a3ff`; the spin-polarised device finished procedurally, both channels read, K21); TD1–TD14 ruled; TD10's owed text written (`36a485e6`). Its header's *but for S13 (the convergence sweep, step 6)* was stale: S13 is a § 2 row, unbuilt (`transport.md:7039`). Left as § 2 rows: S13, E13, SCHED-HDR (step 5's *the group's header on a scheduler*, Q13's leftover). The contracts that cited it point here now.

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

### § 2's rows done or settled

- **RL42** — done 2.46.5 (`0cc35f95`): `_plan_stage` (`submit.py:3430`) and `_bench_trials` (`:2441`) check no run script.
- **GATHER-WARM** — closed 2026-10-27 (no road a person takes; the contract states which run the device facts come from, `record.py:649`).
- **E14** — done: Stokbro 2003 is in `references.bib` (2.46.3, Crossref-checked 2026-10-27); no live document cites Reed 2006 or Solomon 2008 now.
- **V1.9** — settled 2026-10-27 (no broad hash; an edit between stages is not designed for); the order record is `Structure.atom_order`. Three document sentences still call it open — `model/structure-molstruct.md:422-426`, `model/structure.md:726`, `engines/vibration.md` § 10 — carried to DOC-SMALL.
- **V1.14** — done: the suffix fixed 2026-10-13; the vacuum advice filtered by engine since K8 (`cell.box_findings_for`, `cell.py:705-717`); the `#` label settled.
- **GKN** — done 2.46.6 (`33929678`): the cell kept, its trials named by their total ranks.

| # | area | item | from | state |
|---|---|---|---|---|
| **RL42** | execution / launch | **R-L42 (a run script missing) is checked by two builders only** (`_plan_stage` run here, `_bench_trials`): a group's walk, a run of points' walk and a queue send meet it at run time — make it one check where every submission is built, or drop it as a launch-time re-check (the user's rule: prep decides, launch runs) | 2.24.0 launch review | done 2.46.5 — the launch-time check dropped, by the user's rule (prep decides, launch runs) |
| **GATHER-WARM** | transport | **A warm `--gather` relaunch of a run of points takes its done points over with their old inputs** (N3 round 1, U1): the run's points then rest on two upstream runs, and the record reads every point's device facts from the run the first point names (`record.gathered_device_run`); the deck-agreement gate holds, so the difference is at SCF tolerance. Re-checked 2026-10-27: reached only by relaunching a finished device cold, no road a person takes; the contract states the run (§ 2a.12) | N3 | closed — no change |
| **E14** | science / references | **Reed 2006 and Stokbro 2003 into `science/references.bib`.** `engines/transport.md` § 8 cites both (and Solomon 2008 beside them); none is in the `.bib`, and the rule is that an entry is added when a passage cites it. The 2026-10-08 archive reason — *no live document cites them* — was untrue | roadmap § 7.3 | **reopened 2026-10-09** — each entry checked against the paper before it goes in (§ 7.3) |
| ↳ | **V1.9** | **The structure identity rule — what is left** (the rule settled in discussion 2026-09-22; its first half, the pin, built 2026-10-14 — 2.9.6, M2m: the geometry hash over the atom lines as written and the cell and axis kinds, never the comment line, in every pair's sidecar, compared by the one read door, prep refusing a changed structure and the browser's load door saying so; the row as it stood is in [`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md)). **Open:** a *broad hash* over the labels, regions and identity columns, for provenance; the check when two runs are joined (the geometry hash only); and **derived structures** — a reordered copy minting its own identity and recording its parent's with the permutation (`atom-permutation.json` records the permutation and no parent today). Two rulings disagree and are the user's to settle when this is taken: V1.9 says a broad-hash difference is reported, never blocking; W39's customized parameters, as agreed 2026-09-27, enter the hash and prep refuses when it moves ([`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md), M2m, W39). `spectra.json`'s own `structure_hash` is V1.39's (decided 2026-10-14: retired, the result states its structure) | `model/structure-molstruct.md` § 3 · `engines/vibration.md` § 6.10 | open — the pin done 2.9.6 · *Re-checked 2026-10-18:* `atom-permutation.json` is retired (RMP R5); the order record is `Structure.atom_order` (`structure.py:736-743`), key and both directions, no parent. Settled 2026-10-27 from the user's rules: no broad hash and no join check — a cited pair edited afterwards is refused, and an edit between stages is not designed for; a derived structure's order record is `Structure.atom_order` |
| ↳ | **V1.14** | **From the UI walk of 2026-09-23, not this kind's**: the Molbuilder tab's save prompt doubles a typed suffix (`x.xyz.xyz`); the `#`-labelled provenance regions (`O#`) warned as unconsumed on every SMILES-built molecule — needs a ruling; the vacuum notice advising 8 Å on a gas-phase PySCF run | unification audit § 1.18 U3, U4, U7 | **the suffix done 2026-10-13** — the door adds the extension to a name typed without it and takes one typed with it, any case (`molview-doors._askProjectPath`, both the structure and the picture; `projects.md` § 3; `test_molview_files_door_js.py`, mutation red); *open:* the vacuum advice · *Re-checked 2026-10-09:* the doubled suffix stood — `lib/projects/molview-doors.js:114-117` appends `.xyz` to a typed `x.xyz`, the save route writes the path as given; the 8 Å vacuum advice is engine-blind (`cell.py:398-405`); the `#`-label warning is settled by *labels are the user's* — nothing to build. · *2026-10-18:* the vacuum advice `cell.py:397-408`, engine-blind still |
| **GKN** | bench / naming | **An uneven ranks-over-GPUs cell is dropped** (32 ranks on 3 GPUs) because a trial is named and sized by its ranks per GPU (`G3K?C1`, `mpi_np = G·K`) — not for the hardware, and no ELPA source says it cannot run (a run with the same numbers is prepared, its split said). Kept only by naming every trial by its total ranks (`G0N32C1`, `G3N32C1`) | the user, 2026-10-17 | done 2.46.6 — kept, every trial named by its ranks in all (`G3N32C1`), its split said (the user's rule: a request is prepared, never vetoed) |


## 2. Trimmed or rewritten — as they stood

### The status paragraphs and the handover of 2026-10-27

*Verdict, 2026-10-28:* history — replaced by one line per consolidation and a handover for 2.47.12 (its *Where we are* line had said 2.47.11 and its queue *waiting on the user's yes* for § 1ak, which ran S0–S2.3).

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

**Where we are: 2.47.12.** Since the last handover (2026-10-20, kept
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
errors (2.46.3); Save refusing to change a stage that has run (2.46.4); launch re-checking
nothing, and stale words put right (2.46.5); a benchmark keeping every
declared cell, its trials named by their ranks in all (2.46.6); every data
file saying what it is, in the file — its title, schema, molbuilder version
and source, every group's title, every number a variable in one spelling of
its unit — and a SIESTA vibration result citing the deck it was computed
from (2.47.0); the data files' fresh-eyes review — two errors, every value's
meaning said right, the deck cited whole with its pseudopotentials (2.47.1);
a SIESTA vibration's contract written as rules, every page telling the one
stage (2.47.2), and read against its code by two rounds of fresh reviewers
(2.47.3, 2.47.4); its shape fixed where every road reads it, and the ladder
naming each call of its run (2.47.5); the hand-over's shape card (2.47.6);
the optical calculation's cutoff at 3 Ry, measured (2.47.7); what differs
between kinds and between engines declared in two tables, and why a choice is
not offered beside it (2.47.8); what a stage builds on and what a description
carries read from the kind table, the table made smaller (2.47.9); a kind's
fixed charge, benchmark and hand-over, an engine's method and benchmark lane
(2.47.10); a structure lying along z opened side on, and a project's topics
the kinds' (2.47.11); a lead, a summary and a retired stage read from the
kind table, a vibration writing no `.HSX`, and the monitor's line from the
run's structure decided (2.47.12).

**Last checked**: the fast batch on 2.47.12 (6,921 passed), the files its last revision touched after it (151 passed), its chapters rendered; the fast batch on 2.47.11 (6,919 passed), the MolView, project, sidebar and kind-table files (385 passed), the new camera and topic rows broken one by one and seen to fail, its chapters rendered; the fast batch on 2.47.10 (6,917 passed), its moved rules broken and seen to fail, its chapters rendered; the fast batch on 2.47.9 (6,917 passed), its moved rules broken one by one, its chapters rendered; the fast batch on 2.47.8's code before its second review's revision (6,917 passed), the 16 files that revision touched after it (310 passed), the render of every chapter it changed; the fast batch on 2.47.6's code before its review's revision (6,893 passed), the Task setup and describe tests after it (62), the Task setup browser files (26); the fast batch on 2.47.5 (6,891 passed); the ladder's e2e
on the real CO₂ run on 2.47.5; the run-script, launch, warm-file and deck
tests on 2.47.3 (101 passed; comments changed); the catalogue, bundle and launch tests on
2.47.2 (922 and 114 passed); the
fast batch on 2.47.1 (6,889 passed); the SIESTA CO₂
vibration e2e and the real-junction e2e on 2.47.1 (196 passed); the PySCF
vibration e2e on 2.47.0; the real-runs road e2e on 2.46.6 (19 passed); the stopped-run e2e on 2.46.3 (8 passed); every road of 2.45.0–2.46.0 through the pages and the
printed commands (vibration → frame set → transport on the H chain).

**What runs next** is § 0a's queue, *NOW* (2026-10-27) — proposed, the
user's to change.

**Decisions that wait on the user**: none now. The list § 0a carried was
settled on 2026-10-27 from the contracts and the user's own rules (§ 0a);
K22's design and § 9.2's three choices come with their science when step 6
reaches them.

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

### § 1ae — SIESTA's vibration as one run in phases

*Verdict, 2026-10-28:* M1–M5, p1 and p2 built (2.42.0–2.44.1, 2.47.2–2.47.5); handover items (c), (d), (f), (n) done — (f): `ready.py` asks the kind table since 2.47.9; (b), (e) with M3; (h), (i), (j) design notes, closed. Left in the plan: p3, p4 (§ 1ak S3), (a) → SWEEP-RELAX, (g) → § 1af, (k) the user's, (m) → SPIN-EF, M4's viewer walk → S5.

## 1ae. SIESTA's vibration as one run in phases — Raman its optical phase: the build *(2026-10-25; the design [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11, the user's B; R-a, R-c, R1 decided; the SIESTA side verified by the proof of concept, § 5.11 § 12)*

*Checked 2026-10-27*: **M1–M4 built** (2.42.0–2.44.1). **M5, the contract
closed**: `engines/vibration.md` § 5.11 written as rules (2.47.2) — it stays
the run's own section, § 2.2, § 5.2a–§ 5.5, § 5.8 and § 5.9 pointing into it;
nothing built is in the design's tense, and the sweep's take-over and a
transport sweep's one start and finish are said not built where they are
described; every page tells the one stage; two review rounds — the chapter
against the code and every other page against the chapter (2.47.3), then the
records, result and data files against the code and the pages against the
page code (2.47.4) — each finding read again in the code and put right. **M5
closed 2026-10-28.** (p1) the shape a SIESTA vibration fixes and (p2) the
ladder naming each call — built 2.47.5. **Left — two page errors round 2
measured**: (p3) the Run panel stays hidden on a run in phases' `run-<n>/`, which is served
its state and no record, so no page names the call that is running (§ 7a);
(p4) two doors find a stage's result file — the picker's `runs.openable`
(`runfiles.find` and a parser) and `runs.on_label_at` (the export, the frame
set). And § 5.10's banner still holds ② and ③, built. R1 withdrawn 2026-10-27 — the toy chain's run and
the published work in the method's § 3.1, § 7, the science checks the
user's, on Sol (the user, 2026-10-27: *"you can focus on workflow, let me submit scientific result and you can help me detect issues later"*); and the
handover items still open — (a) and (g) the words § 5.11 keeps, (c)
`continuation.py`, (f) `jobset/ready.py` asking `calculation == "transport"`,
(k) the comment in `envs/recipes.py` (reworded 2.46.4, the user's word). After it:
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
| M5 | **the contract, closed** | § 5.11 becomes the rules of § 2.2, § 5.2a–§ 5.5, § 5.8, § 5.9 as built; `job-system.md` § 5.4's vibration rows; `project-layout.md` § 5 (the manifest regenerated); `job-contracts.md`'s rows; `web/results.md`; `web/spectra.md` § 3.2; `web-api.md`; the method doc's § 5 and § 7; the changelog; two review rounds | the chapter states the run in phases as rules, nothing *designed* or *not yet built* of what is built; the method doc's § 3.1 and § 7 carry the published work and the toy chain's run (2026-10-27) |

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

**Open after it**: the named set's share (D-e), designed from the method's
§ 3.4 (the published counterpart: the polarizability projected on the
molecule); the
`electronic` kind (§ 5.10 ④) for the response along a mode, unchanged;
**the measured intensity as an option** (the user, 2026-10-26: "add the
Stokes shift factor as an option later") — the factors of the method's
§ 3.2a, contract first (`engines/vibration.md` § 5.11 § 13, M4-d).

### § 1af's environment paragraph

*Verdict, 2026-10-28:* aligned 2.43.0–2.44.1; the one item left — `job-contracts.md` § 2.6's row for the preamble preflight — stays in § 1af.

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

### § 1ai's last sentences

*Verdict, 2026-10-28:* done 2.46.0 — the Spectrum tab's charge and spin in step 2's chemistry card, its card titled *Run profile*.

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

### § 1ak — one system for a calculation's files

*Verdict, 2026-10-28:* S0 built 2.47.6, S1 written and revised 2026-10-28, S2.1–S2.3 built 2.47.8–2.47.12, the S2 walk done. Of the 67-row inventory: rows 5–10, 21, 26–28 done 2.47.8–2.47.9; 1, 22, 23, 29, 51, 52, 55, 61 done 2.47.8–2.47.10; 32, 42, 58 done 2.47.8; 12–14, 19, 35, 56 (the label tail), 57, 59 done 2.47.12; 11, 15, 44–47, 50, 53 and 67's `spectra/` own modules, staying; 48, 49, 54 — their KIND verdict stale: `validation/siesta.py` and `validation/pyscf.py` are engines' own modules, OWN; 20 and 66 superseded (migrate's rewrites stay code); 25 a docstring (S2.4); 16–18, 30, 31, 33, 34, 36–41, 43, 62–65 and 67's route open in S2.4, with 24 and 60's half the first list missed; 2–4 open in S3. *What S2 moves into the records* done: every named table gone or derived.

## 1ak. One system for a calculation's files — the code, the page and the folders as one design, driven by declared data *(2026-10-28; the direction agreed — the user: "unified and well layered/structured api and framework is the right way to go"; S0 built 2.47.6; S1's contract written, reviewed and revised — its three choices agreed by the user)*

*(The user, 2026-10-28: "did you make sure your work did not [bypass] existing
parser framework and make sure you are not handcrafting code to work with dir
and files? this needs to be systematically and holistically evaluated"; "the
goal is to have the code, ui and file structure all part of a logical, well
organized modularized system design"; "different situations should be mainly
driven by config file and meta data as appropriate, minimize handcrafting code
or design when possible".)*

**What the review of 2026-10-28 measured** (two fresh agents — the directory
door end to end, and the description reader with the tests — each finding
read again in the code):

| group | what | whose |
|---|---|---|
| errors on the roads | **E1** a SIESTA vibration's hand-over: the shape card offers *flat* and Save refuses it — `shape_fixed` is asked only of a calculation that opens, and the page's hand-over branch never reads it (2.47.5's p1 reached a described folder only); **E2** `/api/results/dir` re-lists the door's fields by hand and drops `at_work`; **E3** what a run of points' folder opens answered by `folder_answer` and not by `openable`; **E4** that fallback loses a transmission point and the page calls the file *not in this folder* | mine (E1 shipped; E2–E4 p3, not committed) |
| structure | the run door (`runs.py`, floor 2) calls up into `calculation.open` (floor 3) and `continuation` (floor 4) for a run of points' root — since 2.42, extended by p3's `at_work`; a run's points walked three times per answer (`point_rows`, `every_run`, `at_work`) beside the one per-point reader; the calculation opened twice per answer; a vibration's phases read from two doors | 2.42 on, and mine |
| files by hand | about a dozen in the door, status and export: `export.py` composes the `.spectra.nc` beside the result; `transmission_point` rebuilds the run folder and calls transport's points directly; `_openable` and `on_label_at` answer "the run's result" by two filters; *is this point opened* by `is_dir()` in four places; `continuation.done` composes `shared/.copied-from`; `_warm_present`, `_held` compose names and test `is_file`; the ladder never lists a run in phases' own result | older |
| the page | the browser composes paths (`rootDir + "/" + r.opens`, the deck's **View**, p3's `record_dir`), decides from a path's shape that a row opens the report, and picks a run by a file's basename | older, and mine |
| tests | support helpers and e2e build `01_coarse/run-0`, `04_device/run-0/v0.2`, `task.json`, `job-request.json`, `CO2.fc-sweep.json` by hand; read the bench plan as raw JSON; parse the ledger by hand twice; hand-build the hand-over → description dict four times (which hid E1) | older, and mine |
| kinds by code | **54 branches** found first — about 40 more by S1's review (below) — on a kind's or engine's name (`calculation == "transport"`, `engine == "siesta"`) in 25 modules, beside the kind facts that ARE data, scattered over seven tables — `paths.KIND_ROLES`, `paths.ROLE_SETS`, `jobset.points.DECLARED`, `prep.RUNGS`, `template.RESULTS`, `electronic_state.KINDS`, `task.fixed_by_kind` (itself two branches, mine) | older, and mine |

**The design is the contract now** (S1): the four declarations — the kind
table and the engine table (`kind_table.toml`, `engine_table.toml`, one
reader each on floor 1), the file catalogue, the folder record — in
`execution/architecture.md` § 3.3 with rule A18; one job per floor in § 2.1
and § 3.2 — the run door one folder, a run of points' points read once per
answer (`point_rows`) and the point a folder is (`point_of`) on floor 4, the
views (`views_of`: the ladder, the run of points' view, the point) and
`run_answer` on floor 6, a route serving whole on floor 7, the page composing
no path; what each folder is asked and opens, `model/parse.md` § 5.2.

**What each finding becomes**: E1 — the card's shape is the kind table's,
asked of a hand-over as of a description; E2 — the route serves the answers
whole; E3, E4 — one *what it opens* at floor 6, the page reading why; the
floor breach and the three walks — floors 2, 4 and 6 as above; files by hand
— each to its floor-1 or floor-2 door; the page — floor 7 sends paths, words
and why; the tests — every artifact through these doors, the support helpers
first (one hand-over → description helper, taking the shape from the folder
answer); every branch on a kind's or an engine's name — the kind table and
the engine table. p3 is rebuilt on floors 4 and 6; p4 falls out of floor 2's
one *what it opens*.

**The order** — each stage: contract first, code bottom-up, and **a
fresh-agent review at its end** — its findings verified in the code, triaged
and revised before the stage ships; the two automated rounds over the whole at
S5 *(the user, 2026-10-28: "make sure review is conducted at milestone and/or
stages")*:

| # | milestone | done when |
|---|---|---|
| S0 | **E1** (an error on the road, a last-number move): the hand-over's shape card and Save from `fixed_by_kind`; a road-table row through the hand-over | **built 2.47.6** — a SIESTA vibration's hand-over starts at *hierarchical*, fixed, with why; `describe` without `--shape` names the kind's shape (the CLI stays explicit, the user's decision of 2026-10-28). Its review: card, Save, reader and CLI agree on every road; the hand-over's own words and two contract rows fixed with it; to S2 — `fixed_by_kind`'s branches and `points.DECLARED` keyed differently (transport on any engine, SIESTA only), `fixed_shape` a second door; to S4 — the road helper's record paths by name, a protocol row running a prep it does not need |
| S1 | **the contract**: `architecture.md` § 2.1 and § 3.2 (the kind table as a door; the run door's limits; the run of points' view on floor 6), `model/parse.md` § 5 (the folder kinds and what each opens, as data), `project-layout.md` § 1.4a, `web/results.md` and `web-api.md` (the page shows; the wire carries paths, words and why) | **written and revised 2026-10-28** — `execution/architecture.md` § 0 (six axes: the calculation kind beside the job kind), § 2.1 (floors: the two tables on 1, the run door one folder, a run of points and `point_of` on 4, `views_of`, the views and `run_answer` on 6, a route serving whole and a path whole on 7; a kind's own modules by what they hold), § 3, § 3.2 (the kind door; what a folder opens; a run of points' points; a run of points as a reader sees it; the point a folder is; the views a folder is read with), **§ 3.3 the declarations** (four; a fact with a home keeps it, the two tables the lists the others are checked against; the kind's record, its part per engine, the engine's record; a part named `module:NAME` living on the floor that resolves it; what stays code by package), A18; `model/parse.md` § 5 (one folder; § 5.2's five questions and its table of what each folder is asked and opens; the record's deck path whole); `project-layout.md` § 1.4a; `web/results.md` § 0.1–§ 0.3, § 2, § 2.3, § 2.4, § 3a; `web-api.md` `/api/results/dir`; the rows naming the old doors in `architecture.md`, `job-contracts.md`, `running-a-job.md`, `engines/vibration.md`. **Its review** (two fresh agents, the tables and the floors, every finding read again in the code): the kind/engine split corrected (roles, results, the benchmark, the sum and the retired stages are the pair's where the code says so); facts with a home kept there; the tables renamed apart from the scheduler kinds; a part's floor; `point_of` and `views_of` added (a transmission that runs once, `shared/`, the route deciding no view); a flat root's ladder; what a run of points' root opens; a whole path defined; the page knowing where a file lies by its answer. The chapters that name today's homes are swept in S2, as the records move |
| S2 | **the kind table and the engine table** — first the inventory: every branch on a kind's or engine's name in `molbuilder/`, read in its function and judged (the table below, with the review's locators); then the two files and their readers, the seven tables and their kin into them (below), each branch into its field; the parts named and moved to their floors — a run of points' declaration out of `siesta/vibration_phases` and `transport/stages` into floor-4 modules, the rung builders named by their pairs (the conductor resolving them), the config class one part (`validation.task._shipped` and `config_class_for` go); the other declarations' names checked against the tables at load; the chapters naming the old homes swept (`engines/template.md` § 6.2a, § 6.4, `web/task-setup.md`, `engines/stages.md`, `engines/vibration.md`, `engines/transport.md`, `science/chemistry-correctness.md`, `science/validation.md`, `web/web-api.md`, `execution/job-contracts.md`) | **S2.1 and S2.2 a built 2.47.8.** S2.1: `molbuilder/data/kind_table.toml` + `kind_table.py`, `data/engine_table.toml` + `engine_table.py`; the seven tables and `STAGE_FACT`, `ENGINE_WORDS` (both), `cell.MOLECULAR`, `placement.one_process`, the seam's `if` into them; the runs of points' declarations in `jobset/vibration_points.py` and `jobset/transport_points.py`, the rung builders `prep.siesta_vibration_rung_of` and `transport_rung_of` named by their pairs; the catalogue's engines and item kinds and the shipped warm-file sections (a kind's, or a role's of one) checked against the tables at load; the chapters swept. Its review (a fresh agent, every finding read again in the code): one error — the reader stopped refusing a transport description on PySCF, now refused there by the kind's part — and stale names, a name clash, a claimed check not built (now built), the table reader's own refusals, two routes' 500s now the tables' 400s. `validation.task._shipped` asks the engine table; `config_class_for` stays the preflight's *none for an engine there is not*. S2.2 a: `not_offered` rows beside `offered` (`engines/template.md` § 6.3a), `template.why_not_offered` their lookup — every one of the 36 reasons the same as before. **S2.2 b built 2.47.9**: what a stage builds on, one door — `jobset/ladder` (`independent`, `inputs_of`, `upstream_of`, `stage_before`): a kind's ladder read off its roles (`job-system.md` § 5.4), and on a linked kind's pair the `inputs` its rungs take and their `gather`; `group.linked`, `group.upstream_of` and `ladder.source_stage` gone, `continuation`, `ready`, `submit` and prep's linked line read it; a kind's `description` — `cites` (the slot it cites instead of a structure; its identity) and `carries` (the fields only it states; `bias` stated whenever carried) — read by the description's reader, its migrate hint and prep's cited pair. **Its design review** (a fresh agent asked whether the design is reasonable or over-built, and whether the declarations overlap — the user's question): sound, with small local over-building, removed — a `ladder` field restating the roles, a `"fermi"` value nothing told from `"scf"` (and the `ROLE_SETS` it fed), a part's `stages = "rungs"` restating `stage_roles = "own"`, `PointsOf.kind` read nowhere, one pair-to-part lookup written four times (now `kind_table.part_of`), load-time `module:NAME` format checks that resolving the part already makes; checked and kept — the `module:NAME` parts (a floor-1 reader cannot import the floors that hold them), `inputs` and `gather` as two parts on two floors, `description` at its size, the catalogue's per-item kind lists (which items do what per kind, not what a kind is). **For S2.2 c** (the review's caution): about fifteen of the inventory's verdicts are flags only one kind would set (`checks.*`, `opens.*`, `handover.from`, `frame_when_stated_relaxed`, `describer`) — each new field must delete code, not rename a branch; where the code under a flag is the kind's own logic, it moves into the kind's module or behind an existing part, and a table field is added only where kinds or engines differ by a value. **S2.2 c built 2.47.10**: a kind's `charge` {value, why} (a transport junction's 0 — the electronic state, the chemistry check's refusal, the peptide advisory skipped where the charge is fixed), `no_benchmark` (why), `handover` (the tab that hands it over, `web/tabs.py`'s key; none — it describes on its own tab); an engine's `method` (SIESTA's DFT, where the catalogue offers no choice) and `benchmark` lane. Deleted without a field: `cell.keeps_engine_frame` with its kind test (`already_relaxed` is the vibration's catalogue item), the OMM check's SIESTA guard. Kept as code and listed in § 3.3's *What stays code*: the electronic state's two-setting rules by engine (`count_must_float`, the odd count under restricted) — **a call for the user**: whether they move into SIESTA's own module (`config/siesta.py`, reached through the config's class) instead. Its review: the contract's general exception against A18 made one listed entry, the benchmark's stale third home removed, a hand-over file naming a pair the table lacks answered without a 500, two prose fields guarding only hand-made requests removed. **Left**: S2.3 with **a lead role's facts as one design** — its files' own stem (`identity._LEAD_REGION_OF`, in a module that travels beside a job, which cannot read the table), the region labels (`transport/sort.py`'s), `kmesh`'s lead rungs and fixed axis — three homes of one fact; S2.4 — the run script's engine facts (`runwrap.py` 771, 951, 995, 2273, 2325, 2484, 3466), the surfaces' engine and kind tests (`web/blueprints/build.py` 209, 922–939, 1082, 1119, 1153, 1871, 2811 `_folder_frame_set`, 3054–3062; `web/blueprints/transport.py` 176; `web/blueprints/spectra.py` 270), `cli.py` 392, `parse/dirs/setup.py` 156–181, `runfiles.py` 1040, `validation/sidecar.py` 254 and 441, `validation/spectra.py` 707, and the describe command line (`jobset/_cli.py` 806, 845, 902, 935–951, 1858–1864) — line numbers as of 2.47.10, locators only. Done when: no branch on a kind's or engine's name in shared code (A18); the review's two rounds clean; **every stage's review walks the pages it touched on the running server through Claude in Chrome** — the Structure-optimization and Spectrum forms (`/api/build/schema`), the Transport tab's engine strip, Task setup's hand-over, shape card, stage table cells (`/columns`, `/value`) and folder answer (bench, hand-over), the chemistry card — the console read, the network answers read beside the page *(the user, 2026-10-28: "integrate this in your design/test/review so we can validate all UI components, workflow and backend together")* |
| S2 walk | **the first browser walk** *(2026-10-28, Claude in Chrome, the user's server on 2.47.10)* | **done** — the Transport tab's engine strip (TranSIESTA live, PySCF disabled with the table's reason; the rung tabs the kind's roles), its chemistry card's charge from the kind's `charge`; the Spectrum tab's forms by result in the kind's order, each engine's part named by the engine table; the Structure-optimization forms; and one workflow carried by clicks into `projects/claude-walk-2-47-10/spectrum/walkh2` — an H₂ SIESTA vibration handed over (*from the Spectrum calculation tab*; only `stages` awaited), its shape card fixed with the table's sentence, saved, its run card stated through `/api/task-setup/value`, previewed (*relax · fc 13 SCFs (1 + 6 × 2 free atoms)*) and prepared (`relax/`, `fc/`; *builds on nothing*, *takes nothing from another run*), never launched — every API answer 200, no error of the app's in the console. Its two findings — the viewer's first camera looking along z, and the project scaffold's topics restating the kinds — fixed in 2.47.11, and walked again: New project's six topic folders, a topic folder's ⋯ offering only *Delete directory…* and an older project's `frequency/` the ordinary menu; H₂ lying along z opened side on, an atom added kept the turned view, Retract and Reset fitted it again |
| S3 | **the floors**: floor 2 limited (`points_root_state`, `transmission_point` and the `transport` branches leave the run door), floor 4's rows read once per answer and `point_of`, floor 6's `views_of` — the ladder view with each stage's runs in it, at a flat root too, a run of points listed as the run it is; the run of points' view; `run_answer` — floor 7 serving whole, every path the page opens whole, the record's deck path whole; the page knowing where a file lies by its answer — p3 rebuilt, p4 with it | the review's structure, files and page rows closed |
| S4 | **the tests through the doors** — the support helpers first | no test composes a run's path or a file's name |
| S5 | two review rounds over S0–S4, and a browser walk of every page S2–S4 touched | — |

**Decided 2026-10-28**: (a) S2's reach — the kind table, every branch on
a kind's name judged into it; the engine's branches into the engine table,
the seam a lookup of it; an engine's own modules choosing by kind stay, listed
(`architecture.md` § 3.3, *What stays code*); (b) `molbuilder describe` stays
explicit — `--shape` asked, its refusal naming the shape a kind fixes and why
(*"cli shall be more explicit because on web user typically has visual
guidance and know the final outcome"*; built 2.47.6); (c) p3's uncommitted
code set aside (the job's scratch, `p3-setaside/`) and rebuilt on S3.

**Agreed with S1** *(the user, 2026-10-28: "go with your recommendation on all
three")*: (d) **the kind table is a data
file** — `molbuilder/kind_table.toml`, one reader `kind_table.py`, and the
engine table beside it, `engine_table.toml` and `engine_table.py` (named apart
from `scheduler.kinds`), as the scheduler
kinds are (`scheduler/kinds.toml`, `scheduler.md` § 6a) — the user's *"driven
by config file and meta data"*; a part that is code is named in it as
`module:NAME` and resolved by the floor it lives on (the rule
`jobset.points.DECLARED` keeps today, moved into the record); (e) **one record
per kind, with a part per engine it runs on** — refining *one record per
(engine, kind)*: a kind's roles, results' order and description fields are the
kind's whichever engine runs it, and the shape, fixed stages, how stages play
the roles, the run of points, the rung builder, the benchmark, the sum and the
retired stages are the pair's, so nothing is written twice; the engines a
kind runs on are its parts (no second list — `electronic_state._RUNS`,
`pyscf.stages.VIBRATION_ENGINES` go); (f) **`run_answer` moves to floor 6**
— a file at a run of points' root answers its run's state, which only the
points can say.

**What S2 moves into the records** — the seven tables, each a field:
`paths.KIND_ROLES` and `ROLE_SETS` (a rung's role and whether it runs an
SCF), `jobset.points.DECLARED` (the pair's run of points), `prep.RUNGS` (the
pair's rung builder), `template.RESULTS` (the kind's results),
`electronic_state.KINDS` and `_RUNS` (the kinds, and which engines run them —
the records' keys), `task.fixed_by_kind` (the pair's shape and stages);
and their kin the review found — `transport/stages.STAGE_FACT` (what a rung
computes, beside `ROLE_SETS`), `kmesh._FIXED_ON_TRANSPORT` and `LEAD_RUNGS`,
`identity._LEAD_REGION_OF`; `template.ENGINE_WORDS`,
`transport/stages.ENGINE_WORDS`, `cell.MOLECULAR`, `placement.one_process`,
`validation.task._shipped` and `engines.engine_seam`'s branches into the engine
table — then every branch, each judged again against the record (the first
verdict, 2026-10-28: KIND — a field of the kind's record; PAIR — of its part
on an engine; ENGINE — of the engine's; CATALOGUE — a fact the parameter
catalogue already declares; OWN — an engine's or a kind's own module, which
stays; line numbers as of 2.47.5, a locator only):

| # | where | what it decides | verdict | becomes |
|---|---|---|---|---|
| 1 | cell.py:628 | a vibration stated relaxed keeps its engine frame | KIND | `frame_when_stated_relaxed: kept` |
| 2 | runs.py:1017 | a transmission point's folder opens the report at its point | KIND | `opens.point: report_at_point` (rung) |
| 3 | runs.py:1087 | a transport root opens its description | KIND | `opens.root: description` |
| 4 | task_reader.py:62 | the description read as a report only for transport | KIND | the same `opens.root` field |
| 5 | task.py:121 | transport fixes shape + five stages | KIND | `shape`, `stages` |
| 6 | task.py:129 | SIESTA vibration fixes shape | KIND (engine-qualified) | `shape` on the (siesta, vibration) record |
| 7 | task.py:325 | a bias treatment exists for transport | KIND | `description.bias` |
| 8 | task.py:360 | transport carries a junction slot, no structure | KIND | `description.slots`, `description.structure` |
| 9 | task.py:413 | transport states its voltages | KIND | `description.bias: required` |
| 10 | task.py:549 | the id's identity: junction citation vs formula | KIND | `identity_from` |
| 11 | transport/record.py:881 | transport's own doors refuse another kind | OWN (the kind's module) | stays |
| 12–13 | summarize.py:752/754 | what `summarize` does per kind | PAIR | the pair's sum (a part) |
| 14 | validation/task.py:93 | the retired `relax` stage, naming migrate | PAIR | the pair's retired stages |
| 15 | transport/reuse.py:51 | reuse refuses another kind | OWN | stays |
| 16 | _cli.py:806 | `--reuses` only for transport | KIND | `description.reuses` |
| 17 | _cli.py:844 | describe's road per kind | KIND | `describer` |
| 18 | _cli.py:901 | the vibration ladder | KIND | `fresh_ladder` (with ladder.py:86) |
| 19 | _cli.py:1859 | `--tolerance-cm1` only for a vibration | PAIR | the pair's sum and its options |
| 20 | migrate.py:1412 | the Raman selector retired on SIESTA | CATALOGUE | `template.RETIRED_ITEMS` (`Retired.engines`) |
| 21 | prep.py:405 | the junction citation at prep | KIND | `description.slots` |
| 22 | prep_inputs.py:398/408 | no bench for transport; none off SIESTA | PAIR, ENGINE | the pair's benchmark; the engine's benchmark lane |
| 23 | build.py:1038 | no hand-over for transport | KIND | `handover` |
| 24 | ladder.py:86 | the fresh ladder per kind | KIND | `fresh_ladder` |
| 25 | points.py:9 | docstring only | — | — |
| 26–27 | ready.py:521/570 | a rung's inputs gathered (transport) | KIND | `inputs` |
| 28 | submit.py:1067 | `--gather` only for transport | KIND | `inputs` |
| 29 | electronic_state.py:308 | SIESTA's method is DFT | CATALOGUE | the `method` item's `engines` and `offered` |
| 30 | runwrap.py:951 | the run script's held-atom preview | ENGINE | engine record |
| 31 | runwrap.py:2325 | SIESTA's GPU env category | ENGINE | engine record `gpu_environment` |
| 32 | template.py:2404–2445 | SIESTA has no ROKS; every `why_not_offered` reason | CATALOGUE | the item's `offered` and its reason, beside it |
| 33–34 | parse/dirs/setup.py:163/181 | the deck read; keyword-less items | ENGINE | engine record |
| 35 | chemistry.py:426 | SIESTA's restricted half-filled hint | ENGINE | engine record |
| 36–38 | build.py:208/917/1057 | config from params per engine | ENGINE | the seam's config builder |
| 39 | build.py:1094 | pseudopotential findings for SIESTA | ENGINE | the seam's data files |
| 40 | build.py:1128 | the restart identity group | ENGINE | engine record |
| 41 | build.py:3011 | stage presets per engine | ENGINE | engine record |
| 42 | engines.py:280 | the seam picked by `if` | ENGINE | the seam table's lookup |
| 43 | cli.py:392 | `validate`'s config class | ENGINE | the seam's config class |
| 44 | pyscf/input.py:213 | PySCF's deck by kind | OWN (engine module) | stays |
| 45–46 | siesta/input.py:593/636 | SIESTA's deck by kind | OWN | stays |
| 47 | validation/siesta.py:565 | SIESTA validator's vibration path | OWN | stays |
| 48 | validation/siesta.py:583 | junction boundary: refused on transport, warned elsewhere — **written twice** (pyscf.py:264) | KIND | `checks.junction_boundary` |
| 49 | validation/siesta.py:639 | frozen atoms consumed, not on vibration/transport | KIND | `checks.frozen_consumed` |
| 50 | validation/siesta.py:684 | the device's bias window | OWN | stays |
| 51 | chemistry.py:276 | transport's charge rule | KIND | `checks.charge` |
| 52 | chemistry.py:287 | peptide protonation not on transport | KIND | `checks.protonation` |
| 53 | validation/pyscf.py:220 | PySCF validator's vibration path | OWN | stays |
| 54 | validation/pyscf.py:264 | junction boundary (twice, see 48) | KIND | `checks.junction_boundary` |
| 55 | build.py, the hand-over door's `_what` | which tab sent the hand-over, named by `if calculation == 'vibration'` | KIND | `handover.from` (the sending tab) |
| 56 | task.py:503, :700, :983 | a lead's label tail; the structure block (#8's second site); the migrate hint | to judge — KIND | — |
| 57 | electronic_state.py:97, :103, :289 | `count_must_float` on SIESTA / transport; the transport charge rule (#51's second home) | to judge — KIND, ENGINE | — |
| 58 | template.py:2092–2095, :2298 | `stage_role_rule` (SIESTA all roles, PySCF `force_constants`); `a_kind`'s noun | PAIR (how stages play the roles); KIND (its name) | — |
| 59 | kmesh.py:195, :224; validation/chemistry.py:298 | the transport k-mesh's fixed axis and lead rungs; a chemistry check by kind | to judge — KIND (a role's fact) | — |
| 60 | prep.py:1269, :2621; group.py:86, :100; ladder.py:32; continuation.py:408 | whether a deck relaxes; whether a ladder's stages are linked — two answers today (`group.linked` transport only, `ladder.independent` by `KIND_ROLES`) | KIND — one field, linked | — |
| 61 | prep_inputs.py:408; placement.py:176; cell.py:691/701 | the bench lane off SIESTA; one process (PySCF); free space (PySCF) | ENGINE | — |
| 62 | runwrap.py:771, :995, :2273, :2484, :3466 (beside #30, #31) | the run script's launch line, threads, finish, GPU environment | ENGINE (its run script) | — |
| 63 | runfiles.py:1040 | PySCF's code bundle | to judge — ENGINE | — |
| 64 | build.py:900, :1036, :1844, :1941; parse/dirs/setup.py:156 | lists of known engines | the engine table's list | — |
| 65 | build.py:2783; _cli.py:899, :936, :950, :952, :1865 | describe's per-engine ladder dict, config class and identity field; a surface's kind test | ENGINE (identity field, presets, config class); PAIR | — |
| 66 | migrate.py:141, :704, :767 | old shapes a rewrite names | to judge — PAIR (retired stages) or the rewrite's own | — |
| 67 | spectra/displacement_sweep.py:122; spectra/infrared.py:163, :214; spectra/vibrational_analysis.py:298; web/blueprints/spectra.py:270 | the vibration's own modules by engine; a route's | OWN (the kind's `spectra/`), the route to judge | — |

### § 1ac — a SIESTA vibration's optical activity, and the displacement

*Verdict, 2026-10-28:* built 2.35.2–2.45.0 but the proofs and the science named in the plan: 1ac.1–1ac.3's rows 1, 2, 4, 5 and A built; D-a–D-d and D-j–D-p built 2.37.0–2.40.0; D-f, D-g and D-i, marked *designed*, built (D-f 2.44.0, `spectra/raman.py`; D-g and D-i 2.42.0–2.43.0, `raman_fd_step_ang`, frame 0 shared); D-h's direction built (`Optical.Vector`); R-a 2.45.0, R-c 2.42.0, R1 withdrawn; R-e's vibration half built; *Found, not fixed* L1–L8 fixed or closed (its own D-table says so; L4 built 2.44.0: `compute_ir` writes `Charge.Hirshfeld` / `Voronoi`). Left in the plan: 3a, 3b, B, D-e, D-h's table, the measured intensity's factors.

## 1ac. A SIESTA vibration's optical activity, and the displacement *(2026-10-24; the charge switch built 2.35.2; the infrared and the displacement reference built 2.36.0; the review and the user's choices 2.37.0; SIESTA's vibration redesigned 2026-10-25 as one run in phases, Raman its optical phase (the user's B); R-a, R-c and R1 decided as recommended, 2026-10-25 — to build)*

*Checked 2026-10-27*: rows 1, 2, 4, 5 and A built (2.35.2–2.44.0), the
review's choices built (2.37.0–2.45.0). **Left**: 3a — the water proof on
both engines (CO₂ done); 3b — the Raman proof, unblocked now piece 5 is
built; B — energies along a mode, waiting on the `electronic` kind; D-e's
named set's share (the method's § 3.4) and D-h's six-run table, once an
optical run's cost on a junction frame is measured (the user's, on Sol); R-e's
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
  beside it (D-f); beside a metal only the laser's is defined (the
  method's § 3.1, 2026-10-27).
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
| D-h (R2b) | Raman's light direction | z by default, another direction chosen before the run; the whole polarizability table (six optical runs a geometry, any direction after) decided once an optical run's cost on a junction frame is measured — the user's, on Sol | designed (the method's § 3.2) |
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
| R1 | Raman's numbers, measured | **withdrawn 2026-10-27**: the toy chain's free atoms are links of its metal, so it measures the route's limit, not a junction's number (run and recorded, the method's § 7); the published work settles the static value beside a metal (§ 3.1); a junction's Raman and its optical cost are the user's to run on Sol |
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

Then: R-a and R-c, then Raman built (piece 5, [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11) —
all built — and the water comparison of 3a and 3b, the user's to run on Sol.

**References**: the method's own list
([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)),
each checked against Crossref 2026-10-24, in `references.bib`.

### § 0a's NOW queue and its settled decisions

*Verdict, 2026-10-28:* row 1 done 2.47.0; row 2's § 1ak under way (S0–S2.3 built), its proofs the user's on Sol; the rest carried. The decisions paragraph's settled list kept here; THE ORDER's stale words (step 1's V1.9, the struck W44, W48, V1.33, V1.37, V1.14's advice, E14, V1.25's amplitude, F11 *the user's go*) corrected in place.

### NOW — the queue *(2026-10-27; proposed, the user's to change)*

*(the user, 2026-10-20: "prioritize among all remaining issues, don't switch
if not justified")* — the order: errors on the roads first, then what the
user asked for and is half built, then the agreed designs, then § 0a's
steps 2–6. The queue it replaces (2026-10-20, every row shipped but four)
is [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md) § 4.

| # | what | where | why here |
|---|---|---|---|
| **1** | **The data files describe themselves** (#231) — the definitions in `model/datafiles.md` first, then the writers; § 1aj (2.46.0) was its first slice | `model/datafiles.md`, the writers | **done 2.47.0** |
| **2** | **§ 1ak — one system for a calculation's files, driven by declared data** (proposed 2026-10-28, waiting on the user's yes: S0 the hand-over's shape card, then the contract, the kind table, the floors, the tests; § 1ae's p3 and p4 rebuilt within it; M5 closed 2026-10-28, 2.47.2–2.47.4); § 1ac's proofs — 3a water on both engines, 3b the Raman proof — the user's to run on Sol, molbuilder checking the results with them (the user, 2026-10-27: *"you can focus on workflow, let me submit scientific result and you can help me detect issues later"*) | § 1ae, § 1ac | what was built in 2.42–2.44 is written as rules before more is built on it |
| **3** | **§ 1ai — the Structure-optimization tab's two locks into the catalogue** (`active_when` with *any value given*) | § 1ai | the last page keeping a lock of its own after 2.45.0 |
| **4** | **N6 (b) — the engine's own memory analysis** for a stage's first run | § 1u | the user's feature, (a) built |
| **5** | **N11 release 2 — the page split**, contract first | § 1aa | the user's four answers, release 1 built |
| **6** | **N12 group 4** — messages in the wrong words, unit spellings, a node's two core counts, dead whole-node code (`runwrap.py`, `jobset/model.py`), the sweep presenter's columns, `read_data_file`'s atom map, a copied device's provenance, PySCF's two convergence doors, `--one-job`'s nothing-to-write — brought for approval | this table | the six-agent review's last group |
| **7** | **§ 1af — what a run reports, by kind**: the contract in `execution/run-reports.md` § 2, then the build | § 1af | its design agreed with the user |
| **8** | **§ 1ag, § 1ah** — corrections to a junction's electronic structure; which modes get Raman | §§ 1ag, 1ah | planned, each designed with the user first |
| **9** | **THE ORDER's steps 2–6** | below | as the table says |

**Decisions that wait on the user**: none. *Settled 2026-10-27 from the
contracts and the user's own rules, each re-derived when its row is taken:*
**SEQ-ALLOC** — one job running several groups in turn asks the largest rank
count and the largest cores per rank, so every group fits (as 2.26.0
builds it); **WAE** — the *written as the job ended* check's sub-second edge,
never observed: dropped; **RL42** — launch re-checks nothing (prep decides,
launch runs): the launch-time check of a missing run script is dropped (2.46.5);
**V1.9's remainder** — no broad hash: a cited pair edited afterwards is
refused already, and an edit between stages is not designed for;
**F11** — a walk to see a killed point's kill line: our own test gap, not a
decision; **SCHED-R's advance signal** — beyond what the person requests:
not added; **GATHER-WARM** — a warm `--gather` relaunch keeping done points
from the old upstream run — re-checked 2026-10-27: reached only by
relaunching a finished device cold and then gathering a transmission that
had points done, no road a person takes (a change goes back to before the
first launch, 2.46.4); the contract states which run the device facts come
from (`engines/transport.md` § 2a.12), no change; **GKN** — a benchmark cell of uneven ranks per GPU
(32 on 3) is dropped against the rule that a request is prepared, never
vetoed: kept, its trials named by their total ranks (2.46.6).
Step 6's designs — V1.25's amplitude, V1.26, K22 — are brought with their
science when that step starts. **Only Sol can show** (the user's runs, when
the user is there; none on this workstation, *the user, 2026-10-10*): E13 (the
junction on Sol), the Sol memory measurement (M2n), a group's header under a
real scheduler (Q13), the AuBDTAu workflow, the SIGTERM no instrument recorded
(SCHED-R); and, before a vibration runs there on 2.15.x, Sol's PySCF
environment repaired for `netcdf4` (`envs doctor`, then `envs repair`) — the
user's. This machine's was repaired on 2026-10-18.

### § 5q — the engine offset

*Verdict, 2026-10-28:* P0–P3 done, P5 closed by TD8, 5q.3's wire and 5q.4's checks and tests done (their references drifted: `to_wire` `structure.py:1351`, `validate_periodicity` now `periodicity_gate.py:158`, `cell.transport_vacuum` in `validation/sidecar.py:165,179` called from `validation/__init__.py:510-511`); the multi-frame row done. Left in the plan: P4's three items, references re-read.

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

### § 5r — the structure-API cleanup's order

*Verdict, 2026-10-28:* steps 2–4 done; step 1's *D5 before X1 ②–⑤* past (X1 archived 2026-10-09); step 6's A1.7 obsolete (2026-10-08); step 8's A1.9 done (2.32.0); step 9 is § 5q's; step 10's A1.2 superseded; 5r.2's transport `+1` bullet past (2.32.0, `engine_atom_index.siesta_numbering`).

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

### § 5s — the electronic state

*Verdict, 2026-10-28:* P0–P4 built; decision 4 built (`citation_defaults.py:184-195`); 5s.4's struck items and `restart = continue` closed. P5's TBtrans half built (K21) and its `record.py:453` glob text stale; the call-site count 12 now.

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

### § 5t — the run record

*Verdict, 2026-10-28:* P3 done (its `:453` spin-channel text stale; `energies_relative_to_ef` at `transport/record.py:1136`, whether a defect unverified); P5 done 2.43.2 (`09ed49ae`); decision 7 built (`inspectors/transport.js:231, 629-717, 1041-1044`); 5t.4's TranSIESTA forces (F14), PySCF manifest and three parses done; P4's *doc drift* note stale — the documents say not built; the pointer *§ 5w.4* named a section that does not exist.

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

### § 5w — the M11 review's mechanism classes

*Verdict, 2026-10-28:* K14 done 2.47.8 (`f82c1bdf`); K13's three private writers through `persist` now; K17, K21, K11, K3 done; K10 superseded 2.42.0. Left in the plan: K13's raw `write_text` calls, K16, K18, K22, K9, K14's residue, § 5w.2.

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

# The seven-lane review of 2026-10-18 — every finding, with its verdict

**Role:** archive — the record of the review the user asked for on 2026-10-18 (*"use agents to do thorough document and code full reviews and validate all those findings … detect handcrafted code … fix the mistakes and inconsistencies"*): seven fresh readers, one lane each (transport, multi-frame, optimization, spectrum, Task setup and the hand-over, execution and launch, the result parser), each reading its contracts and code whole at `6c3ea317`; every finding re-read at its cited line before anything moved. FIX = a measured error against a written rule, fixed in 2.16.3; TEXT = stale or self-contradicting contract text, rewritten; DESIGN = a decision brought to the user (plan § 2, REV-D); DROP = not verified. The changelog's 2.16.3 row says what was built; this file is why.


## Multi-frame lane (M)
- M1 FIX — the frame-set card drops the folder answer's `why` (build.py:2401-2407 returns it; viewer.js:1687 nulls the answer unless n_frames; card.js:57-59 hides unless n>1). Fix: viewer passes the answer whole; card draws `why` as one line when present.
- M2 TEXT — transport.md:3343 and task-setup.md:182 name `compose.frame_set_answer`, which does not exist; the door is `frame_average.answer`.
- M3 TEXT — job-contracts.md:2213 owner cell names `transport/average.py` (gone in 2.13.0) → `frame_average.py`.
- M4 TEXT — spectra/derived.py:14-17 header says `frameset` uses it; frameset.py imports only `frame_words`. Delete the line.
- M5 FIX — card.js:68 spells `mode_index_1based` (the server's `_states_a_mode`, frame_average.py:120). Fix: the answer carries `mode: bool`; the card reads it.
- M6 TEXT/DESIGN — vibration.md:1862-1870 names generator rows (`max_displacement_ang`, NN-distance) that § 2.2f's vocabulary lacks. Fix: § 5.10 ③ points at § 2.2f and names no row (verify the lines first).
- M7 TEXT — vibration.md:1925-1940 ⑤ bullets still say "on transport" slope/curvature, struck by § 2a.12. Strike (verify first).
- M8 TEXT — vibrational-averaging.md:28-30 ("from the mode the customized rows announce"), structure.md:654 ("waits on M2m" — built), structure-molstruct.md:377 ("W39's" → V1.9's).
- M9 DESIGN — structure.frame_at (structure.py:1684-1695) derives frame 0's engine offset for a lone frame f>0 and stores it with `offset_by` None: a fourth state. Why it is there: every frame of a set must sit in frame 0's box (a transport point per frame). Contract decision: state the rule (a frame taken from a set keeps the set's placement) and whose the offset is in the sidecar.
- M10 DESIGN (low) — frame_average._said (333-343) answers the vocabulary's unit for a known name, ignoring a written `unit`; the data file keeps the written one. Decide: the name's unit is the name's (say so in § 2.2f), or a differing written unit is said beside the row.
- M11 FIX — runstatus.points_summary:444 re-derives n_frames from the rows; take it from the one door that made `pts`.

## Spectrum lane (S)
- S1 FIX — PySCF's first write states equilibrium placeholders (scf 0.0, mo [], homo 0; vibration_emitters.py:549-553, deck :215) and the reader refuses homo 0 against 0 MOs (results.py:792-800): every in-progress file before the SCF is refused (viewer ERROR / "stopped following" during the relaxation). Fix: the equilibrium block absent whole before the SCF (None ×3, the shape § 6.2 allows); delete the placeholders and the stale comment (:572-575); an e2e assertion that an in-progress file reads.
- S2 FIX — three hand-written walks of the structure dict beside the one door: results.py:1092-1095 (frozen from metadata.regions), datafile.py:335-344 (elements/positions/cell/axis_kind), blueprints/spectra.py:142 (elements). Fix: Structure.from_dict once; datafile and the blueprint read the instance.
- S3 FIX/DESIGN — the thermal-spread rule lives in JS (core.js:2542-2560) and an unused Python twin (derived.py:68-80, USED-BY false). Decide the home: serve the amplitude per mode at T from the load door (Python), delete the JS arithmetic; contracts § 4.1 / § 12.2 say which. (Recommend: Python, the page derives nothing.)
- S4 FIX — core.js:794-798 the engine note says PySCF "refuses" a periodic structure; the contract (§ 3.3, 2026-09-29) says it computes it as a cluster and notes it. Reword.
- S5 TEXT — vibration.md § 2.1 (l.191 "schema 8"), § 4.1 (l.547 schema 7), § 6.7 (l.2170 "{4,5,6,7}") → 9 / {4..9}.
- S6 TEXT — job-contracts.md:2210 vibration result row says schema 8 → 9 (the structure whole).
- S7 TEXT — vibrationview.md § 12.2 l.905-906, 937 "largest per-atom vector = 1" → § 6.3's wording (largest Cartesian component).
- S8 TEXT — web/spectra.md § 2 l.65-71 "Nothing is built yet" banner; spectrumchart is built. Delete.
- S9 FIX — siesta_vibration.py:260 reads the cell from LatticeVectors (fdf), :313 from the ENGINE-OFFSET block (box_of_run): one fact, two statements. Fix: one read (the box the structure states feeds rigid_motions); § 5.5/§ 6.10.1 say it once.
- S10 FIX — results.py:801-828: § 6.10.2 promises free_atom_idxs checked ascending and hessian_scope vs held; the check is set-based. Add the order check and the scope↔held check (cheap), or trim the contract. (Add.)
- S11 FIX — vibration_deck.py:279 `else float(_ga.max())` judges all atoms on an unexpected shape; drop the arm (None).
- S12 FIX — methods.py:430-433 and :406-411 mock arms (duck-typed struct) inside the one derivation's caller; delete.
- S13 TEXT — vibration_deck.py:31-33 "an anchored molecule does not rotate" (retracted 2026-09-21). Delete clause.
- S14 TEXT — core.js:2289-2293 comment misstates why g_ν is not shown; cite V1.27.
- S15 TEXT — vibration.md § 5.3 block listing and deck_record.py:233-235 omit title/identity; siesta_vibration.py:352 message says "format v3" where § 5.5 says v2.
- S16 TEXT — vibration.md § 5.5 l.1430 "imports the finish with numpy" → numpy and netCDF4.
- S17 TEXT — normal-modes.md § 3.1 l.324-327 "centre of mass, as PySCF does" vs code's geometric centre (origin-invariant). Say geometric centre.

## Execution lane (E)
- E1 FIX — runwrap.py:966-1032 probes GPU 0's NUMA node at PREP (nvidia-smi on the prepping machine) and bakes it into the run script (`_gpu_numa="${MOLBUILDER_GPU_NUMA:-N}"`); not masked by `_STAMPS` (deck_record.py:83-86). (a) prep --target sol bakes the workstation's node; (b) a first launch where nvidia-smi answers differently is refused by R-L15 for a line that is not the description's. Fix: drop the prep-time probe; the single-GPU wrap resolves the node at run time with the same sysfs lookup the per-rank helper uses (`_bash_numa_from_gpu`). running-a-job.md § 3.3 l.440-442 and job-contracts.md § 2.6 said "probed at generation time" — rewrite.
- E2 FIX — prep_inputs.py:596-600, 664-668: a bench proposes G from `topology.gpus_per_node` (one sampled node) first, the queue's node types only when that is 0; gpu.md § 1.3 says the sample is never the cluster's. Fix: propose from the named queue's node_types; the topology only for a machine with no queue.
- E3 DESIGN — probe.py:304-316, 344-350, 419-424: `_pick_qos` prefers `public`, assumes `public` when sacctmgr answers nothing; one domain per partition. configuration.md M-1: a probe never writes a preference. Fix shape: one domain per allowed (partition, qos) pair; no QoS recorded when unreadable, said in the notes. Changes what `jobset machines` lists — the user's call.
- E4 FIX — probe.py:28-31, 428-435 records `30-00:00:00` for an infinite partition (an invented limit; R3 "an unstated limit is not counted"). Fix: `max_time` absent (`domain_ceiling_s` answers None already); keep the note.
- E5 TEXT — job-contracts.md § 4.3 l.2122-2131 says the wrapper awks the SystemLabel; runwrap.py:645-664 bakes it.
- E6 TEXT — job-contracts.md § 4.4 table l.2141-2147 lists five Mode lines; the code writes eight (runwrap.py:945-959, 1618, 1647).
- E7 TEXT — job-contracts.md § 3.2 l.1209-1212, 1272-1281: resolved-defaults keys and the "auto-policy" sentence; the code writes target_env, omp_threads, max_memory_mb, mpi_np (runwrap.py:3511-3523).
- E8 TEXT — job-system.md § 3 class diagram l.965-999 and the job-set.json example l.1235-1255 lack JOB_KEYS (placement, members, written_at, asked, Resources fields); from_dict refuses the example.
- E9 FIX — 26 literals "sweep"/"ladder" across prep.py, submit.py, _cli.py, materialize.py, runstatus.py, plan.py where model.KIND_SWEEP/KIND_LADDER are the one home. Fix: compare the constants (`JobSet.is_sweep`).
- E10 FIX — submit.py:346, 1810 spell "jobset-launch"; runwrap.py:3644-3680 spells the gate's name and "manual". One constant pair beside the gate.
- E11 FIX — submit.py:2155-2160 re-checks a trial's ranks/cores at launch (prep states them). Delete.
- E12 TEXT — runrecord.py:480-500 parses D-HH:MM:SS itself (ships beside the job; quantities does not): state the shipping reason in the docstring.
- E13 DESIGN — runwrap.py:2060-2096 refuses at prep a SIESTA GPU run whose target record lists no molbuilder-siesta-gpu env (gpu.md G6, contract-stated) vs "the record is shown, not a veto". The user's call: keep, or print the fit's sentence and let the run script refuse at run time.

## Parser lane (P)
- P1 FIX — runs.py:881 (folder_answer) and runstatus.py:331 (every_run) ask `run_state_of(run.folder, names, run)` without `launched_in`, while the stage row (runstatus.py:275) and transport/record.py pass it; a point's folder is run-<n>/<pt.rel>, its launch record lies in the run folder, so a point's launch record reads None: a queued point reads "pending — not launched", a wall-cancelled point reads "running" for ever (`_scheduler_ended` never reached). Fix: `Run` carries the folder its launch record lies in; the callers pass `launched_in=`.
- P2 DESIGN — runs.py:146-150 `speaking` answers only with ONE deck; a flat root with two prepared, unlaunched stages answers status null, no ladder; results.md § 2.4 says both "no ladder for a flat root" (:423) and "lists its stages' runs the same way" (:479). Settle one sentence; recommend: a flat root with no run yet answers pending for its first stage, and its runs are listed by number.
- P3 TEXT — parse.md:385 § 2b row "ended → finished" vs running-a-job.md § 4.2 (ended → running until it concludes); parse.md:371 run_status signature stale.
- P4 FIX — parse/engines/siesta.py:93-104, 107-187 own regexes and a second walk of the .out for the held set; siesta_mdnc.py:364-379 a third walk for the SystemLabel. § 4a: one reader per line. Fix: grammar rows for `siesta: Constraint (N): pos` and `reinit: System Label:` read by the pass; the parser and sibling_md_nc take both from finish().
- P5 FIX — suffix literals outside the catalogue: siesta.py:237 (.molwatch.log), :243 (.py); pyscf.py:125 (_geom_optim.xyz), :488, :519 (_optimized.xyz); sidecars/spectra.py:207 (.spectra.json); parse/dirs/record.py:210 (glob fdf.*.log vs siesta_fdflog._NAME). R-RO1. Fix: `role_of`/the catalogue's constant at each site.
- P6 FIX — trajectory.js:69-72 keeps TRAJECTORY_ROLES / PART_ROLES (role suffixes) and matches meta.role; R-RO1 says a page takes its roles from the Python site. Fix: the folder answer names each file's parser and its part-of relation; the page keeps no list (as the other inspectors since 2.13.3).
- P7 TEXT — run-reports.md:494 "the monitor converts nothing, no tolerance beside a PySCF residual" vs parse.md § 4a :639 and monitor.py:219-221 (prints the tolerance in eV).
- P8 TEXT — results.md:42 lists `-runN.pyscf.log` as what opens; parse.md § 5.5 :1112 says no parser claims it.
- P9 FIX — _run_ending.py:199-201, 243, 258: the PySCF/molwatch ending carries neither `relaxed` nor `scf_converged`; results.md § 2.4 promises "geometry yes/NO" for a relaxation. A finished PySCF relaxation reads "—" for ever. Fix: the molwatch ending reader states `relaxed` from the log's footer/final block.
- P10 FIX — summarize.py:150-151, 513, 733-734 and bench/result.py:345: a trial's `state` = "completed" when the output ENDED (not the door's finished); the terminal column is headed `state` while the wire calls it `artifacts`; choose_winner ranks on it, so a trial that exited rc≠0 after ">> End of run" can be crowned. Fix: head the column by what it prints; rank on the door's finished.
- P11 TEXT — parse.md:956-960 claims cross-file checks (atom count, handedness) no code makes.
- P12 TEXT — presenters.md:43 `*_optim.xyz`, :229-230 "because the name ends in .molwatch.log" — the role is `_geom_optim.xyz`; a name alone matches nothing.
- P13 FIX — pyscf.py:402 `run_state = mw_meta.get("run_state", "unknown")` where the same log read directly answers running (scan_conclusion). Take scan_conclusion's word.

## Task setup lane (K)
- K1 FIX — viewer.js:4230-4241 seeds `varies` from every stage-group column (~28) where `jobset init` writes `varies: []` (task.py:322-327, _cli.py:685-689): two writers, two files for one hand-over; the page's catch path writes [] — one click yields either. Fix: `task.fresh_varies` beside `fresh_stages`, carried on the hand-over as `proposed_varies`; the page copies.
- K2 FIX — viewer.js:650-659 LANE_ASKS (time, domain: label/help/placeholder) is a page-held copy of task.LANE_ASKS. Serve from the folder answer / sweepable door; reword § 6.2b's "by name rather than from the catalogue".
- K3 FIX — viewer.js:353-400 (groupCell) and :1630-1637 decide "same as above" from builds_on/launched alone and compose their own sentence; group.may_join also requires equal shared card values: the page offers a join the envelope refuses at prep. Fix: the folder answer's groups.stages[i] carries may_repeat + why; addStage asks the same.
- K4 FIX — viewer.js:3356-3364 coercePoint types a value by its look (true/on/yes, Number) on setRunValue/addPoint; § 5.2: never by look; CELL_READERS exist. Fix: read through CELL_READERS[_meta[name].type]; lane asks as text.
- K5 FIX — viewer.js:1936-1941 valueInForce falls back to 1: + Add setting on a valueless item births a one-point row [1] nobody typed. Fix: born empty.
- K6 FIX — viewer.js:3426-3427 `engine: (over && over.engine) || {name: "siesta"}` and `schema: "molbuilder/task@1"` spelled on the page; build.py:1119-1120 `or "calculation"`. Refuse by name; the folder answer serves the schema string.
- K7 FIX — task_setup.html:376,382 `<label>.xyz` (a file nothing writes); viewer.js:1320-1323 spells the source pair; :4482 `wrote = ["task.json"]` where the door answers `wrote`. Take the names from the answers.
- K8 FIX — viewer.js:3540-3636 a JS codec for time/mem mirroring scheduler.quantities (kept equal by a parity test), marking aria-invalid on its own verdict (§ 6.1: never marking what is typed invalid). Fix: write the typed text; the save door canonicalises; the note shows the queue's range only; retire the parity test.
- K9 TEXT — task-setup.md:1067-1070 (§ 9 "only the seed is removed") vs :412-416 and task.py:536-543 (every rung, in order, refused otherwise).
- K10 TEXT — handover-procedure.md:139-140 "three" structure keys; task.py:161 has four (fingerprint).
- K11 TEXT — stages.md § 6.8a cites `jobset/ask.py`'s canonical_time; it is scheduler/quantities.
- K12 TEXT — build.py:1205-1209 the hand-over's own note names `structure.files`, a key it has not.
- K13 TEXT — task_setup.html:213 "task-setup.md 6.2c" → 6.2b.
- K14 DESIGN — build.py:2358-2366 answers groups.shared (SHARED rows) that no reader uses; web-api.md:602 and task-setup.md:681-683 describe two card behaviours. Recommend: the card writes every row to every reader (the built, approved behaviour); delete `shared` and the web-api sentence.
- K15 FIX — build.py:1211-1212 spells `run_id(typed, formula)` inline where derive_run is the codec.
- K16 TEXT — task-setup.md:394-396 "coarse/medium/tight ladder shipped" vs one stage (task.py:322-327).
- K17 TEXT — viewer.js:1060-1062 "submit bench" → launch bench.
- K18 FIX — viewer.js:206, 2190 stage-name patterns copied from task.py:97; let the save door refuse (one sentence).
- K19 DESIGN — task.py:1228-1235 rewrites a pre-rule `enabled` silently on Save while centre_junction and a missing group refuse naming migrate; stages.md § 2 sanctions the silent read. Ruling: refuse naming migrate (one policy); amend § 2.
- K20 FIX — build.py:1628 `list(it.stages) or None` vs :1745 `list(it.stages)`: one shape.
- K21 FIX — build.py:1963-1990 sends per stage the calculation's allocation AND the rung's chosen (time/domain included); viewer.js:2402-2417 prints both — two walls on one line; web-api.md:605 names a producer (`prep._allocation_for`) that does not exist. Fix: the door answers the effective ask per stage (the fold prep uses); the page prints one line.
- K22 FIX — viewer.js:997-1006 decides a bench row's class (machine/chosen/measured) in JS; the grid door says it.
- K23 TEXT — web-api.md:597 the hand-over door returns four things, not two.

## Optimization lane (O)
- O1 FIX — pyscf/layout.py:235-241 writes level_shift/diis_space/damp only off their defaults (template.md § 6.6 ob. 4: every item written, a default marked). Write unconditionally.
- O2 FIX — the trajectory-log leaf keys spelled in four places: emitter.py:52-60 (_LEAF_KEYS), pyscf/input.py:1272-1278, prep.py:1667-1668, siesta_grammar.py:884-897, parse/contract.py:132-138, trajectory/core.js:1098-1125. One Python home (the emitter's tuple); the PySCF script's dict and prep's seed rendered from it; the viewer reads keys from the data.
- O3 FIX — prep.py:1664-1670 seeds FLAT targets; pyscf/input.py:1270-1280 writes NESTED {stage: {...}}; readers each carry a flat-or-nested branch (core.js:997-1024, contract.py:130-139). One shape (nested); pyscf.md § 4's "unstaged run" sentence goes (O17).
- O4 TEXT — siesta.md:772-777 "# stage:" header the emitter never writes.
- O5 FIX — identity.py:106-113 `getattr(cfg, "restart", "clean")`: a second default spelled in code where the catalogue's is `continue` (run-identity.md § 4 rule 3). Read the catalogue's default or make the missing attribute an error.
- O6 DESIGN — vibration.md:1349-1356: a vibration's relax rung continues by `restart`'s default, but the catalogue row is optimization-only, so the value comes from the config class default, stated on no card and in no provenance block. Offer `restart` to the vibration kind, or record the kind-fixed value in the deck's provenance as the kind's rule.
- O7 FIX — identity.py:131-136 hand-spells the longest extension (.STRUCT_NEXT_ITER, 17 B); the catalogue has `_optimized.molstruct.json` (25 B): MAX_LABEL_BYTES admits a label whose final structure write exceeds 255 B. Derive from runfiles.WRITTEN.
- O8 FIX — parse/engines/pyscf.py:125-155, 519 hand-split stems and spell roles (with P5/P6).
- O9 FIX — parse/contract.py:58-80 three hand-listed tuples mirroring the catalogue's `citation` tag. Build from `template.select(citation=...)`.
- O10 FIX — trajectory/core.js:1779-1780 captions "CPU (GPU requested, fell back)" — a state PySCF refuses (pyscf.md § 2a). Delete the branch.
- O11 TEXT — template.md § 10a.5 (2605-2606), § 11.3 (2775), § 12.1 row 4 (2882) say the wrapper greps the deck; § 1.1a says gone.
- O12 TEXT — template.md § 4.0a (428), § 5.1 (801), § 6.0 (1053) and tests/test_catalogue_agreement.py:1-30,111 + web/blueprints/_shared.py:54-56 docstrings describe the two-home state paid in 2.15.4.
- O13 TEXT — template.md § 10b (2708-2712) describes a fixed bug as current.
- O14 TEXT — template.md § 9.1 row + ⚠ on `frozen_indices` vs § 12.1 row 2 (closed).
- O15 TEXT — task.py:47-48 P2 names a refusal that does not exist.
- O16 TEXT — "enabled" wording: siesta/stages.py:33-37, pyscf/stages.py:63-66, resolve.py:743, continuation.py:385,395 (user-facing), trajectory/core.js:997-1024 comment.
- O17 TEXT — pyscf.md § 4 (with O3).
- O18 FIX — pyscf/stages.py:81,99 imports SIESTA_STAGE_NAMES for PySCF's ladder; an engine-neutral home.
- O19 FIX — trajectory/core.js:1341,1438 x-axis "CG step" for every engine → "geometry step".
- O20 FIX — pyscf/input.py:1135-1150: the resume (`keep`/`resumable`) gated on `save_optimized_xyz` (a produce item) not on `restart`; `_GEOM_CONVERGED` assigned, never read. Gate on continues(cfg); delete the unread name.

## Transport lane (T)
- T1 FIX — parse/engines/tbtrans.py:122 globs `{label}{tag}AVTRANS_*` by hand (and `[0]` by alphabet); job-contracts § 2.2a. One door on the label, the lead pair from LEAD_NAMES.
- T2 FIX — role literals at call sites: stages.py:268-275 (.TSHS, .DM), :282 and deck.py:329 (.TS.HSX), :307-311 and tbtnc.py:55,68 (tag + "nc"), transiesta.py:507. One vocabulary (warm-files.toml [transport] / siesta_grammar), the DAG and the deck line reading it.
- T3 FIX — tbtnc.py:64-72 copies Run.on_label's body (runs.py:196-203). Through run_of(where).on_label.
- T4 — K3 parked (plan § 5w.1); three copies of TRANSPORT_AXIS. Fix with C-group (one constant).
- T5 FIX — wizard.py:332,402-414 `z_period` parameter reachable only from a test (the flag went 2026-09-17). Delete, with its test row.
- T6 FIX — wizard.py:72-74, 422-431: a ~12 Å thickness note the contract retired (I11); shown as info beside "UNVERIFIED" for the same lead. Delete.
- T7 TEXT — transport.md § 3.6a (3246) says validate_subject unanswered; deck.py:681-683 answers it.
- T8 TEXT — transport.md § 2a.12 (2171) "E_F reference checked" vs § 1.1 (932) "Not yet"; record.py:909 asserts. One statement: TBtrans's convention, asserted.
- T9 TEXT — job-contracts.md § 4.2 (1826-1830) says .TS.HSX is in the transport warm-file section; it is not. Add the row (T2 reads it) or correct.
- T10 FIX — bias: `jobset init` writes `voltages_v: []` (_cli.py:407), the tab `[0.0]` (blueprints/transport.py:421, core.js:1072); readers patch `or (0.0,)` at record.py:1048, datafile.py:353,373, stages.py:184. One spelling written by both roads (`[0.0]`, § 2a.10 stated), the patches gone.
- T11 FIX — record.py:317 keys a voltage note by `%g`; transport.js:338 by String(Number(v)). Key by the row.
- T12 FIX — deck.py:310 types `DM.UseSaveDM true` as a literal outside SIESTA_RESTART_GROUP. Through the declaration.
- T13 DESIGN — the leads' decks carry SystemLabel `<label>_L-electrode` (transiesta.py:85; transport_rung.py:150-171) while named on the description's label (:364-367): the run door (on_label) cannot see a lead run's own .TSHS/.DM/.XV; § 2.1's one-basename rule does not hold for two rungs. Decide: the lead's files on its own stem, or a run door that knows a rung's engine label.
- T14 TEXT — job-contracts.md § 6.1 (2213) `transport/average.py` (= M3).
- T15 FIX — blueprints/transport.py:555-558 types shape="hierarchical"; take fixed_by_kind("transport")["shape"].
- T16 FIX — blueprints/transport.py:755-769 filters the rung form a second time by resolvable_override_names; the schema builder's surface="rung" drops run settings by marker.
- T17 FIX — citation_defaults.py:60-73 drops a malformed cited kgrid silently (shown "not chosen"). Refuse by name.
- T18 FIX — compose.py:1127-1132 a vanished citation is told as "no readable .ion"; name the cause.

## Decisions for the user (design, not touched)
M9 (a lone frame's derived box corner: whose), M10 (a known row name's unit vs a written unit), S3 (the thermal amplitude's home — recommend Python, the page derives nothing), E3 (one domain per allowed partition+QoS pair; no assumed `public`), E13 (prep's refusal of a SIESTA GPU run with no GPU env in the record — keep or show as the fit), P2 (a flat root's state before its first run), K14 (`shared` rows vs every row — recommend delete `shared`), K19 (a pre-rule `enabled` on Save — refuse naming migrate), O6 (`restart` for a vibration's relax rung), T13 (the leads' SystemLabel vs the one-basename rule).


## Judged, not changed (2026-10-18)
- M11 — the point rows ARE the one door's answer; counting their distinct axes is not a second rule.
- T12 — the device deck's one `DM.UseSaveDM true` is the contract's own statement (§ 2a.15); the three-member group is the optimization's.
- E2 — the GPU count a benchmark proposes from: the probed node first (gpu_contract.toml's rows, `generator.md` § 4.3a) or the queue's node types (`gpu.md` § 1.3) — two contracts disagree; brought to the user (REV-D), the code left as the rows state.
- K11 — kept: an assertion on the real junction's run.
- K22 — open (plan § 2): the bench row's class decided on the page.
- P4 — not built (plan § 2, its own row): the two second readers of SIESTA's output stay until the grammar rows are written.
- K5 — the empty-row birth was tried and reverted the same night: the reader takes no empty row; brought to the user (REV-D).
- S3 — brought to the user (REV-D): the thermal-spread rule's home.


## Built 2026-10-19 (2.17.0) — the design calls and the open rows

The user, 2026-10-19: "go ahead and validate every finding, then fix them".
Each row below is the plan's, verbatim as it stood, and what was built:

| **REV-D** | design | **The seven-lane review's decisions for the user** (2026-10-18; each verified at its line, none changed — `archive/2026-10-18-review-findings.md`): **M9** a frame taken from a set (`Structure.frame_at`, f > 0) carries frame 0's engine offset with `offset_by` unset — a fourth state; state the rule (a frame keeps its set's placement) and whose the offset is. **M10** a known row name's unit is the vocabulary's on the card while a differing written `unit` goes to the data file; decide: the name's unit is the name's (said in § 2.2f), or a differing written unit is said beside the row. **S3** the thermal-spread rule lives in the page (`spectra/core.js` `_physicalAmplitude`) and an unused Python twin (`derived.thermal_spread_amu12_ang`); `spectra.md` § 4.1 and `vibrationview.md` § 12.2 assign it to the tab — recommend: served per mode at the request's temperature by the load door, the JS arithmetic deleted. **E2** a benchmark proposes its GPU count from the probed node first (`gpu_contract.toml`, `generator.md` § 4.3a) though `gpu.md` § 1.3 says the sample is never the cluster's — a 4-GPU sample proposes no 8-GPU cell on an 8-GPU queue; recommend: the named queue's node types first, the sample only with no queue. **E3** the probe records one domain per partition with a QoS it prefers (`public`, else the first allowed), and assumes `public` when `sacctmgr` answers nothing — a preference written by a probe (M-1); recommend: one domain per allowed (partition, QoS) pair, no QoS recorded when unreadable. **E13** prep refuses a SIESTA GPU run whose target record lists no `molbuilder-siesta-gpu` env (`gpu.md` G6) against *the record is shown, not a veto* — keep, or show it as the fit and let the run script refuse. **P2** a flat root with two prepared, unlaunched stages answers no state and no ladder (`runs.speaking` answers with one deck only); `results.md` § 2.4 says both *no ladder* and *lists its stages' runs*; recommend: pending for the first stage, its runs by number. **K14** the folder answer's `groups.shared` names rows no reader uses; `web-api.md` and `task-setup.md` § 6.2b describe two card behaviours; recommend: the card writes every row to every reader (built), `shared` deleted. **K19** a pre-rule `enabled` is rewritten silently on Save while two sibling cases refuse naming `migrate`; recommend: refuse naming `migrate`, `stages.md` § 2 amended. **O6** a vibration's relax rung continues by `restart`'s default, but the item is optimization-only, so the value is on no card and in no provenance block; offer `restart` to the vibration kind, or record the kind-fixed value in the deck's provenance. **K5** a bench row added for a setting with no value in force (no template value, no catalogue default) is born at `1`, an invented number (`task-setup/viewer.js` `valueInForce`); the reader takes no empty row (`task.py`: a non-empty list), so the row cannot be born empty — decide what a valueless setting's first point is (typed before the row exists; the item's `null_label`; refused). **T13** the leads' decks carry `SystemLabel <label>_L-electrode` while named on the description's label, so the run door cannot see a lead run's own `.TSHS`/`.DM`/`.XV` and § 2.1's one-basename rule does not hold for two rungs; decide: the lead's files on its own stem, or a run door that knows a rung's engine label | the review, 2026-10-18 | **decisions owed** — each a design change; none built unasked |

- **M9** — `Structure.frame_at(f)` states `offset_by = "engine"` for a frame
  derived from a set's engine offset (`structure-periodicity.md`).
- **M10** — a vocabulary row name written with a differing unit is refused by
  name (`frame_average._said`; `structure.md` § 2.2f).
- **S3** — the load door serves each mode's `thermal_spread_amu12_ang` at the
  request's `temperature_k` (`spectra.derived.thermal_spread_of`); the page's
  arithmetic and its served constant deleted; the tab asks again when the
  number changes (`web/spectra.md` § 4.1, `vibrationview.md` § 12.2).
- **E2** — a benchmark's GPU count from the named queue's node types first,
  the probed node only with no queue (`prep_inputs`; `generator.md` § 4.3a).
- **E3** — one domain per allowed (partition, QoS) pair from `sacctmgr show
  assoc`'s partition and QoS columns, named `<partition>` for `public` or the
  partition's own name and `<partition>-<qos>` else; no QoS recorded when the
  list is unreadable (`qos: null`, no QoS line in the header); the `debug`
  domain on the cheapest partition and the assumed `public` gone
  (`probe.derive_domains`; `scheduler.md` § 4).
- **E13** — prep prepares the job and says in its notes that the target's
  record lists no `molbuilder-siesta-gpu`, that the run script enters it at
  start and stops without it, and the install command (`RenderedWrapper.notes`,
  carried to `PrepAnswer.notes`); `gpu.md` G6/G8; the protocol row that used
  the refusal as its last-step example retired with the road's `reprobed` hook.
- **P2** — a flat root with no run yet speaks for its first stage by ladder
  order (`runs.speaking`; `results.md` § 2.4).
- **K14** — `groups.shared` deleted; the card writes every row to every reader.
- **K19** — a pre-rule `enabled` is refused naming `jobset migrate`, which
  strips it (`stages.md` § 2).
- **O6** — the catalogue's `restart` item is offered to the vibration kind
  (`[item.restart] calculations`), so a relax rung's value is on its card and
  in its provenance.
- **K5** — a setting with no value in force is born from a typed first point
  (the add-setting box; the reader still takes no empty row).
- **T13** — a lead's files are on its own stem, `<label>_L-electrode`, the one
  door `transport.stages.rung_label` asked by the element's label, the job's
  run names and the run door's position reader; the label cap budgets the
  tail (`identity.normalise_id(label_tail=)`); rule 2 holds for every rung.

| **P4** | parse | **Two second readers of SIESTA's output**: `parse/engines/siesta.py:93-104, 107-187` keeps its own regexes and a second walk of the `.out` for the held set (`siesta: Constraint (N): pos`), and `siesta_mdnc.py:364-379` a third walk for the SystemLabel (`reinit: System Label:`), against `parse.md` § 4a (one reader per line: the reading pass, through the grammar). No surface wrong today (the pass carries neither fact, so nothing asks it). Fix shape: two grammar rows read by the pass; the parser and `sibling_md_nc` take both from `finish()` | the review, 2026-10-18 | open — step 3 |

- **P4** — grammar rows `CONSTRAINT_LINE`/`CONSTRAINT_RANGES` and the
  `system_label` rule in the one pass; `finish()` carries `held` (1-based) and
  `runtime_info["system_label"]`; the parser translates the held set through
  the engine index API and hands the label to `sibling_md_nc`; the two second
  walks of the output deleted, with the test that read one.

| **K22** | Task setup | **A bench row's class is decided on the page** (`task-setup/viewer.js` `renderBench`: `chosen` = one point and not machine-answered; `machine`; `measured`) — the one-point rule of `generator.md` § 4.3a, copied; `bench_inputs` reports cells, not the row's class. Fix shape: the bench-grid door (called on every edit with the axes) says each row's class | the review, 2026-10-18 | open — step 3 |

- **K22** — the bench-grid door says each row's class (`rows`), the page
  paints it (`_declared_execution_pins`, `_rowKinds`).

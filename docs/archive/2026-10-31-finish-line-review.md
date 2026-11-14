# The finish-line review of 2026-10-30, verified

**Role:** archive
**Domain:** all

*(The user, 2026-10-30: "shall we review again to include all document, UI and backend, and consolidate with previous review results to unify a single plan to finish"; "go ahead, science after the finish line"; 2026-10-31: "make sure after these checked validated results come, your fixing should be framework level, holistically and unify api and data structure design, such that you are not patching and hacking to generate handcrafted redundant/duplicated codes".)*

Fourteen readers covered the code at 2.56.1 (`e60b9827`): two settled every open item the plan and the 2.54.1 and 2.55.1 reviews listed (445 items); six read the 85 contract chapters against the code; two drove every page headless on the user's real runs; four read the backend by door — § 1am steps 2–7 — re-measuring the 2026-10-29 review, and the tests. Eleven more then checked every finding against the code and the documents, both sides: **785 confirmed, 83 partly true (the part said), 10 rejected, 6 unsure**; 28 inventory items given their right status; 84 open inventory items not re-checked one by one (a random quarter of them was: 1 in 29 wrong).

Each item below is a **symptom**. The plan ([`plans/plan.md`](?doc=plans/plan.md), *The one plan*) closes them by release, each release one redesign of the door or the data structure they point to — never one patch per item. A finding's *door* column is the reviewer's pointer to the existing door, checked by the verifier; the release decides the design.

Ids: **I1/I2** the inventory of open items (plan lines 1–639 and the 2.54.1/2.55.1 leftovers; plan lines 640–1567) · **B1** the run script and the engine/kind records · **B2** names, records, formats · **B3** vocabulary, floors, objects, the results views · **B4** the tests · **D1–D6** the documents · **U1/U2** the pages.

## F0 — Wrong answers now — each through its owning door (29)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B2-1 | wrong-result | structure.py:2298 `to_xyz` / :2066 `from_xyz` — a structure holding a labelled species (`Au1`) is written into its pair and its own reader refuses it | a species is a label and an element: one shape through the structure codec (writer and reader) — **closed 2.57.0** | confirmed |
| B2-2 | road | runfiles.py:159-162 `FIELDS["element"]` shape `[A-Z][a-z]?` — the pseudopotential's name family refuses the species label SIESTA opens | a species is a label and an element: the pseudopotential row's field is the label — **closed 2.57.0** | confirmed |
| B2-36 | door | jobset/migrate.py:1155 — a point's `.continued-from` written by hand beside its one writer | `.continued-from` bytes from `runrecord` alone (2.56.0 left) — **closed 2.56.2** | confirmed |
| B3-28 | door | two module groups import each other, and one function reads an unbound name | the diagnostics door — **the unbound name closed 2.56.2**; the import cycle **left 2.65.15**: function imports only, every module loads alone in any order, and moving the measurement would separate it from the door it is measured beside. The module says so | partial: PARTIAL \| diagnostics.py:532-552 `try: caps = get_capabilities() … except Exception as exc:  # pragma: no cover … envs_here, why_none = [], f"listing them failed ({exc})"` then :552 `if caps.conda_b… |
| B3-30 | words | NEW (2.56.0 left it) — a comment that says the opposite of the code: a point never reached is *not run*, the docstring says *failed* | *not run* everywhere the walk's never-run is said (2.56.0 left) — **closed 2.56.2** | confirmed |
| B3-31 | door | one rule written twice, and two lists of engines | status's next step over the run door: a group the sequence sent is launched — **the rule closed 2.56.2**; the two lists of engines F6's (F0a-R2e-3) | confirmed |
| D1-JSa-1 | wrong-result | jobset/runstatus.py:_steps — a launched group one of whose members' run folders is not on disk is said **finished** in the next step | status's next step over the run door: a missing member is said — **closed 2.56.2** | confirmed |
| D1-RR-1 | docs-untrue | § 2.8 — a member monitor's line cannot say *not run* | *not run* only where an ended walk is read (2.56.0 left) — **closed 2.56.2** | confirmed |
| D2-RAJ-A | docs-stale | running-a-job.md:868-872 (§ 4.2 *The order of evidence*) — a scheduler-stopped sweep said to read *failed* at points the walk never reached | *not run* in running-a-job (2.56.0 left) — **closed 2.56.2** | confirmed |
| D2-RAJ-B | docs-stale | running-a-job.md:998 (§ 4.2 *The words on each surface*) and the state diagram :980-989 — `not run` missing | *not run* in running-a-job (2.56.0 left) — **closed 2.56.2** | confirmed |
| D2-CKP-1 | docs-stale | checkpointing.md:1252-1258 (L8) — "a run that ended never differs afterwards, but for its `run.json`'s `ended`, written once" — `jobset migrate` writes into ended runs | migrate's writes into finished runs stated as the rule's exception (2.56.0 left) — **closed 2.56.2** | confirmed |
| D2-PAM-1 | docs-untrue | preparing-for-another-machine.md:15-23 (§ 1's diagram) — on the target the run is started as `./…run.sh` | a document that makes a person act wrongly — **closed 2.56.2** | confirmed |
| D4-18 | docs-untrue | tuning.md § 2.12 — "one thread per rank is the shipped default" and "the wrapper widens each rank's threads"; a launch value is stated or refused | a document that makes a person act wrongly — **closed 2.56.2** | confirmed |
| D6-P6 | docs-untrue | § 1 :34-36 "the package is pyflakes-clean … answers nothing" — runwrap.py:3622 "f-string is missing placeholders". § 1am step 8, S, no. | the package pyflakes-clean (conventions § 1) — **closed 2.56.2** | confirmed |
| D6-O1 | docs-untrue | access-control § 2 row 4 (:88), :91-93, § 2.1 diagram (:165), deployment § 4 (:495-498) — a non-admin "gets 404 … absent, not refused" from the restart route; app.py:662-672 → admin.py:78-81 answer 403, tests/test_admin_reload.py:118-124 asserts 403, access-c… | a document that makes a person act wrongly — **closed 2.56.2** | confirmed |
| D6-O2 | docs-untrue | access-control § 3.2 (:206-211), deployment § 3.4 (:390-392) — the public-endpoint list omits the run-report listener `notify.api_notify` (auth.py:64-88, :85), which answers without a session (its own HMAC gate, run-reports.md § 4.1); § 2's gate table has no … | a document that makes a person act wrongly — **closed 2.56.2** | confirmed |
| D6-E6 | docs-untrue | § 4 :480-483 top-level commands `run`, `bench`, `transport`, `fdf`, `pyscf` — each "No such command"; describe, jobs, checkpoint, frameset, … unnamed. S, no. | a document that makes a person act wrongly — **closed 2.56.2** | confirmed |
| D6-E7 | docs-untrue | § 5 :501-503 "A SIESTA job you launch from the UI now runs via `conda run -n molbuilder-siesta`" — the run script activates through the target record's `env_init` (runwrap.py:2031, :2084-2085), as installation.md:15 and env-framework.md § 5.6 say; launching i… | a document that makes a person act wrongly — **closed 2.56.2** | confirmed |
| U1-2 | road | Transport › extending a finished calculation whose recorded junction states no atom order — the plan and Describe both answer 500 | the composed junction's atom order through `Structure.replace` honouring the view field — **closed 2.57.0** | confirmed |
| U2-2 | wrong-result | Results › spectra viewer › Modes table, a PySCF result written before `phase_ir` (PDT/spectrum/pySCF_PDT run-0) — the IR intensities the run computed are not in the table (nor its CSV), while the chart draws their lane and the summary says computed | one *IR asked* reading of a result, older files included — **closed 2.57.0** | confirmed |
| U2-3 | wrong-result | Results › trajectory viewer › run badge, PDT/optimization/PDT_FIX_moleculeonly/03_final/run-0 — "Finished \| ended Sep 27 9:10:01 PM · total 0s" for a run whose engine wall time is 24 s (the Run panel beside it) | the run's clock read once (its engine wall), the badge asking it — **closed 2.56.2** | confirmed |
| U2-4 | words | Results › spectra viewer, PDT/spectrum/siesta_PDT run-1 — the sentence over the mode lines says intensities "were not requested in this run" while the run summary on the same page says the infrared is missing because the result states no charges | one *IR asked* reading of a result, older files included — **closed 2.57.0** | confirmed |
| I1-023 | inventory | a displacement sweep's later stage taking the first stage's `relax` call over | **closed 2.58.0** | confirmed |
| I1-100 | inventory | `status` names no next step for a group member that is missing | status's next step over the run door: a missing member is said — **closed 2.56.2** | confirmed |
| I2-133 | inventory | a failing `ending` reader exits 1, read by the wrapper as "no" | the run-ending reader's answer: a crash is an answer, never *no* — **closed 2.56.2** | confirmed |
| I2-134 | inventory | the Build route's peptide branch collects no warnings, so the C-terminus warning never reaches a web user | the build route returns the builder's own findings — **closed 2.57.0** (decided 11 (b): one build door and one check after it) | confirmed |
| I2-168 | inventory | a SIESTA displacement sweep's later stage does not take the first stage's `relax` call over, so a sweep that relaxes is refused at its summary | **closed 2.58.0** | confirmed |
| I2-169 | inventory | a spin-polarized optical frame's Fermi level is lost | **closed 2.58.0** (a fixed spin's: SIESTA prints one `Ef` otherwise) | confirmed |
| I2-199 | inventory | a run prepared again after the charge goes to 0 keeps its old `makov_payne_correction.py` | prep's writes of a stage: what it no longer writes is not left behind — **closed 2.56.2** | confirmed |

## F1 — The results views (§ 1ak S3): one point view every reader asks (34)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B1-47 | door | runs.py:1027-1069, :1238-1266 — the run door imports transport's points and words transport's folders in its own sentences | the words on the role's and the folder record's own rows; `point_of` (floor 4) for the point — **closed 2.59.0** | confirmed |
| B2-17 | door | atom_index.py — a point's `shared/` and a lead's file told by their names | `calcdirs.where(folder).role == SHARED`; `runfiles.parse(p.name, rung_label(...))`; the registry (`detect`) or each record's own reader — **closed 2.59.0** | confirmed |
| B3-1 | door | runs.py:933 `points_root_state` — floor 2 opens the calculation and asks floor 4 for a run of points' state | the run of points' state at its root through floor 6's view (`points_view`, `views_of`), the run door answering only its folder — **closed 2.59.0** | confirmed |
| B3-2 | door | runs.py:1027 `transmission_point` — the run door walks a transport run's points through the kind's module | floor 4's `point_of` over the kind's declaration (`jobset.points.points_of`), served by `views_of` beside the folder answer — **closed 2.59.0** | confirmed |
| B3-3 | door | runs.py:894 `_as_its_point` — NEW in 2.55.0/2.56.0: floor 2 imports floor 4 and reads a whole run's points to answer one point's state | a point's state as its run's rows read it is floor 4's (`continuation.point_rows`) asked by floor 6's view; the run door keeps its own folder's answer — **closed 2.59.0** | confirmed |
| B3-4 | door | runs.py:200 `Run.prepared` — NEW in 2.55.0: the run door walks the kind's declaration to find a point's prepared folder, and for a SIESTA vibration that walk reads the plan | the folder rule the contract states, through `calcdirs.RunPlace` (`prepared`), with no declaration walked — **closed 2.59.0** | confirmed |
| B3-5 | door | runs.py:868 `run_answer` — on floor 2; the contract puts it on floor 6 | move it to `jobset/runstatus.py`, asking the run door for a run's file and `points_view` for a root's — **closed 2.59.0** | confirmed |
| B3-6 | door | web/blueprints/results.py:194–243 — `/api/results/dir` decides the views, builds the ladder and re-lists the door's fields | `views_of(folder)` on floor 6 (the ladder, `points_view`, `point_of`), the route returning `{**folder_answer(d), **views_of(d)}` — **closed 2.59.0** | confirmed |
| B3-7 | door | runs.py:1149 `_file_row` — a file's row carries its `name`, not its whole `path` | `_file_row` states `path` (the absolute path) beside `name`; the page reads it — **closed 2.59.0** | confirmed |
| B3-8 | door | runs.py:1246–1266 — a `shared/` folder's walk up to its point is the folder answer's, not `openable`'s | `point_of` answers a `shared/` folder's point (floor 4) and `openable` its report, so every reader of the folder gets one answer — **closed 2.59.0** | confirmed |
| B3-10 | door | runs.py:348, :493 — the transport branches are still in the run door: the pair's sort key and an electrode's own structure asked of `transport.sort`, an engine's deck read by its suffix | the pair's sort key and own-structure fact as the kind table's part (`kind_table.kind(...).on(engine)`), the atom count through the engine's deck reader (`keyword_deck`) — **moved to F3** — the kind's sort key and own structure are the kind/pair record's (F3's row) — **closed 2.65.0** — the pair's `atom_order` and `own_structure`; the atom count's fdf fallback deleted (B1-29) | confirmed |
| B3-11 | door | the upward imports A7 forbids — 15 edges (A 12 + B 2 + C 1) | each fact to the floor below its readers — **closed 2.65.15**: measured 11 upward imports and 2 from the core layer, not 15, none at a module's top; five close by the sentence that put `transport/stages` on the wrong floor, the rest moved or deleted. Four stand on purpose, named in A7: `parse/dirs/setup` asking `script_emit`'s read API, `submit._go` asking `runstatus.job_members` at a stop, `parse/coords`' two readers through the structure codec, `transport/reuse` reading a finished point's energies through `energies.held`; `runfiles.reader` asking `runpoints.part` is the contract's own door | partial: PARTIAL \| each of the 15 lines read now (table below) \| 10 stand against § 2.1's floors table / A7 as written (runs ×7, commands:429, mode_frames:94, and selection:50 against docs/architecture.md §… |
| B3-33 | door | runstatus.py:309 `every_run` — a run of points' points walked again for every point, and its paths sent root-relative | the ladder's run rows composed from `point_rows` once — each row's point, folder, prepared folder and what it opens from the declaration's walk already made — paths whole, `opens_report` from the role's `opens` — **closed 2.59.0** | confirmed |
| B3-34 | door | a transmission sweep's run folder opens nothing — `openable`'s container branch never asks the role's `opens` | `points_view`'s `opens` (the point at work's: the report at that point), served by `views_of` — **closed 2.59.0** | partial: PARTIAL \| runs.py:1086-1090 `openable`: `transmission_point(d) if place.ours and place.role == calcdirs.RUN else None`; :1109-1110 `if place.role == calcdirs.CONTAINER: return _container_product(…)`… |
| B3-36 | docs-untrue | architecture § 2.1, § 3.2 and web-api `/api/results/dir` state the views as built; only results.md § 0.4 says they are not | whichever the user picks; until S3 is built, web-api.md and architecture § 3.2 say what `/api/results/dir` serves today — **closed 2.59.0** | confirmed |
| D2-ARC-1 | docs-untrue | architecture.md § 2.1 / § 3.2 — three doors the contract names that no code holds, and one on the wrong floor | S3 builds them (floor 6 `views_of`/`points_view`, floor 4 `point_of`), or the rows state the live road (`runs.folder_answer` + `runstatus.every_run`) — **closed 2.59.0** | confirmed |
| D3-T13 | docs-untrue | transport.md § 2a.9 / § 5 I14 / § 6.2 hop 3 — the citation door's "states no cell" refusal is presented as a live gate; the plan measured that no road reaches it | whichever the user picks, through compose's existing refusal — **kept 2.59.0** — a road reaches the refusal (the section below); its words now name the way out, the Cell page | unsure: UNSURE — the docs side is true; the plan's "no road" premise looks false by code (see the D3-T13 section below) \| transport.md:1914-1916 "a structure that states none is refused at the citation door… |
| D4-1 | docs-untrue | vibration.md § 5.11 § 7 and § 11 — the run root's view `runstatus.points_view` does not exist | whichever the user picks; if "state the live road", name `runs.points_root_state` / `run_answer` here. — **closed 2.59.0** | confirmed |
| D5-5 | docs-untrue | parse.md §§ 5.0, 5.2 — `views_of` and `point_of` named as the doors beside the folder answer; no code holds them | add parse.md §§ 5.0/5.2 to the § 1ak S3 sweep list; until S3 is built, name the live road (`folder_answer`'s `point`, the route's ladder). — **closed 2.59.0** | confirmed |
| U2-1 | wrong-result | Results › transport report › DOS tab, a transmission run once (lowbias_local, lowbias_local_extended) — the page says no data file exists and offers no way to read the DOS per orbital, while the record lists the finished point with its 1.5 GiB `.TBT.nc` and i… | `point_readings` takes a finished point from the run door's answer for a run-once rung (the record's own `state`, `record.result_folders` + `run_state_of`) as it does for a run of points' rows — one door for "this point… — **closed 2.59.0** | confirmed |
| U2-21 | wrong-result | Transport report › I–V points table — a point *not run* (2.56.0's record list `not_run`) gets no row: the head counts it, the table lists only `pending` and `failed` | `pending.concat(notRun, failed)` at `transport.js:606`, its chip the warn tone; results.md § 0.1 "each point not ended, not run or failed". — **closed 2.59.0** | confirmed |
| I1-014 | inventory | the import scan's *no upward import* re-run | **closed 2.59.0** | confirmed |
| I1-021 | inventory | the Run panel on a run of points' `run-<n>/` is served its state and no record | **closed 2.59.0** | confirmed |
| I1-022 | inventory | two doors find a stage's result (`runs.openable`, `runs.on_label_at`) | **closed 2.59.0** | confirmed |
| I1-066 | inventory | `transmission_point` asks `transport.stages.points_in` directly (not `point_of`) | **closed 2.59.0** | confirmed |
| I1-088 | inventory | the `shared/` walk-up lives in `folder_answer`, not `openable` | **closed 2.59.0** | confirmed |
| I1-089 | inventory | the contract names `points_view`, `point_of`, `views_of`, which no code holds | **closed 2.59.0** | confirmed |
| I1-090 | inventory | `compose`'s "states no cell" refusal is reached by no road — say so or delete it | **kept 2.59.0** — a road reaches it (D3-T13); its words name the Cell page | confirmed |
| I1-103 | inventory | `/api/results/dir` re-lists the door's fields by hand and builds the ladder in the route | **closed 2.59.0** | confirmed |
| I1-105 | inventory | the run door (floor 2) calls up into floors 3 and 4 | **closed 2.59.0** | confirmed |
| I1-107 | inventory | paths composed in the browser | **in part 2.59.0** — the Results page composes no path; the transport report's junction join and Task setup's three → F8 (F1-R1-12) | confirmed |
| I1-109 | inventory | `run_answer` on floor 6 | **closed 2.59.0** | confirmed |
| I1-113 | inventory | the floors (whole milestone) | **closed 2.65.15** — the run door 2.59.0; floor 1 loads nothing above it (the two package markers empty, measured); the rest by B3-11…B3-16 | confirmed |
| I2-087 | inventory | a point taken over from a run that itself took it over | **moved to F9** — a real run's chain, the e2e's | unchecked |

## F2 — § 1am step 2 — the run script names no engine (34)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B1-1 | wrong-result | runwrap.py:3289 — a PySCF vibration's banner says the engine will load a `.chk` its deck never reads | pass the job's `resumes` to the PySCF arm as to the SIESTA arm (one banner call once step 2 merges the arms) — **closed 2.63.0** — one banner for every engine, from the `resumes` and `warm` the job hands in | confirmed |
| B1-2 | road | runwrap.py:771-776, :791 — `--cold` names, and with `--force` removes, a device run's gathered leads | the sweep's exceptions from the gather's own record (`.gathered-from`, `runrecord.read_gathered_from`) — **closed 2.60.0** by removal: the sweep removed with the run script's `--cold`/`--force` (a run is one start) | confirmed |
| B1-3 | road | runwrap.py:791 — `rm -f` on geomeTRIC's scratch folder stops `--cold --force` under `set -e` | the sweep removes what it named, folders included, through the one sweep it already is — **closed 2.60.0** by removal: the sweep removed | confirmed |
| B1-4 | residue | runwrap.py:3508, :3523 — the retry count and the try number are two counters for one fact | derive the retry from `--run N` (the try's number the launch gives as 0) — **closed 2.60.0** by removal: both counters removed | confirmed |
| B1-5 | road | runwrap.py:156, :3864, :3874 — two starts in one second share one session log | the catalogue's `stamp` field with the `pid` field `_BASH_FILLS` already fills — **closed 2.60.0** by removal: one session log per run, opened by its one start | confirmed |
| B1-6 | door | runwrap.py, render_run_wrapper — the run script is two engine arms chosen by name, not one skeleton reading the engine's record | step 2 as designed — one skeleton over the engine table's record (program, one_process, keyword_deck) and the parse layer asked by meaning (s24/design_2533.md) — **closed 2.63.0** — one run script, each block decided by the engine record or the job, or asked of the parse layer by the output's role | confirmed |
| B1-7 | door | runwrap.py:3741, :3762, :2412, :2414, :2425, :3795-3803 — the deck's suffix stands for its engine | `category` (the catalogue's `engine_of_role`) and `engine(category).keyword_deck` — **closed 2.63.0** — prep hands in the engine (`engine=jobset.engine`); the deck is checked as that engine's (`runfiles.deck_role`) | confirmed |
| B1-8 | door | runwrap.py:1785-1801 — whether a PySCF deck reads prior state is sniffed from its text | `resumes` and `warm` the job hands in (already parameters of `render_run_wrapper`) — **closed 2.63.0** — `resumes` and `warm` handed in; a script deck's text is not read | confirmed |
| B1-9 | door | runwrap.py:2326-2337, :2378; diagnostics.py:63-73 — the env a run enters is the engine's NAME read as an env category, a GPU run's by `"siesta-gpu"` spelled here | the engine record's environment and GPU environment, each a configuration.md § 4 category — **closed 2.63.0** — the engine record's `environment` and `gpu_environment`, each a registered category | partial: PARTIAL \| runwrap.py:2326-2328 `env_lookup_category = category` / `if category == "siesta" and env is None and gpus.uses: env_lookup_category = "siesta-gpu"`; :2331 `caps.env_for_category(env_lookup… |
| B1-10 | door | runwrap.py:2913, :3114-3117 — the retry budget is dropped by the program's name, not by what the rung computes | `stages = "scf"` on `continue_retries` — `resolve` already carries a rung-scoped item onto the job as it does `use_gpu` — **closed 2.60.0** by removal: the retry budget removed (`continue_retries` retired) | confirmed |
| B1-11 | door | wrapper_log.py:97, :100; runwrap.py:2908 — the program's label is spelled by the writer and matched by the reader | one labels table in `wrapper_log` (the session log's own module), the writer and the reader asking it — **closed 2.63.0** — `wrapper_log.BINARY_LABEL`, the program named by its engine record's `programs` words | confirmed |
| B1-12 | door | runwrap.py:1863-1866 — the run script inlines `%block … < file` with its own awk | the monitor bundle's deck reader (`parse.fdf.read_deck`), asked as the design names it (`mb_monitor.pyz deck`) — **closed 2.63.0** — `mb_monitor.pyz deck` (`parse.fdf.read_deck`) | confirmed |
| B1-13 | door | runwrap.py:808-845 — the orbital estimate `10 * n_atoms` restated beside the deck's BENCH-MARKS `n_orbitals_est` | the deck's `n_orbitals_est` through the bench-marks block's reader — **closed 2.63.0** — `runwrap._n_orbitals_est` through `script_emit.read_bench_marks_block` | confirmed |
| B1-14 | door | runwrap.py:1951-1964 — `env_entry` refuses a missing activation in its own words beside prep's door | `jobset.machine.require_activation`, or the run script trusting the prep that asked it — **closed 2.63.0** — `jobset.machine.require_activation` | confirmed |
| B1-15 | door | runwrap.py:4637-4646, :4674 — `Resources` taken apart to call `render_sbatch` | `render_sbatch(names, resources, …)` whole — **closed 2.63.0** — `render_sbatch(names, scheduler, partition, qos, resources, one_process, where)` | confirmed |
| B1-16 | door | runwrap.py:742-747 — the cold sweep's `*.psml` glob spelled by hand | the glob of the `PSEUDO_FILE` row (`runfiles.patterns`) — **closed 2.60.0** by removal: the sweep removed | confirmed |
| B1-53 | docs-untrue | docs/execution/architecture.md:671-674 — the engine record is said to hold a `run_script` part; no engine table holds one, and plan § 1am step 2 describes another shape | § 3.3's engine paragraph as the set-aside patch words it, in step 2's release — **closed 2.63.0** — architecture § 3.3 lists the record's `environment`, `gpu_environment` and `programs` | confirmed |
| B1-54 | docs-untrue | docs/execution/job-contracts.md:925-958 — the wrapper's routing said to be by extension and by the deck's `Diag.ELPA.GPU` | "the job's GPU request (`gpu_request`) routes a SIESTA run to its GPU environment" — **closed 2.63.0** — job-contracts § 2.6: the GPU request routes, never the deck's text | confirmed |
| B2-29 | door | a block of our decks written beside its reader's module | one block writer and one span reader in `deck_record`/`script_emit`; the transport decks asking `siesta.deck_lines` for the SIESTA lines — **moved to F5** — one block writer and span reader is F5's codec work | confirmed |
| D2-ARC-2 | docs-untrue | architecture.md § 3.3 :670-673 — the engine record's `run_script` part | step 2 builds it; until then the bullet says the run script is runwrap's — **closed 2.63.0** — with B1-53 | partial: PARTIAL \| architecture.md:671-674 "what its run script runs and how — … (`run_script`, a part: the engine's block of the run script, in its own package …)"; engine_table.toml [siesta]/[pyscf] hold w… |
| U2-15 | road | Results › a real benchmark (AuBDTAu/optimization/min_force_relax/01_fine/bench) — listed under "Benchmark sweeps", then refused when picked; the file card calls molbuilder's own `job-set.json` "not written by molbuilder" | `jobset migrate` records the bench container's and each trial's place through `calcdirs.record` (as prep does now) where a described calculation's `<NN>_<stage>/bench/` has none — so `migrate` stops calling such a calcu… — **moved to F4** — the bench container's place is the record doors' work | partial: **PARTIAL** \| Facts true: `projects/AuBDTAu/optimization/min_force_relax/01_fine/bench/` and both `bench-G*/` hold no `calcdir.json` (only `01_fine/calcdir.json`, `01_fine/run-0/calcdir.json`; trial… |
| I1-041 | inventory | the run script names no engine | **closed 2.63.0** | confirmed |
| I1-083 | inventory | a device run's `--cold` names, and with `--force` removes, its gathered leads | **closed 2.60.0** (removed: a run is one start) | confirmed |
| I1-084 | inventory | geomeTRIC's scratch folder stops `--cold --force` (`rm -f` on a folder under `set -e`) | **closed 2.60.0** (removed: a run is one start) | confirmed |
| I1-085 | inventory | a PySCF vibration's banner promises a checkpoint its deck never reads | **closed 2.63.0** (B1-1) | confirmed |
| I1-086 | inventory | `MB_RETRY_N` and the try number carry one value | **closed 2.60.0** (removed: a run is one start) | confirmed |
| I1-087 | inventory | two starts in one second share a session log | **closed 2.60.0** — one session log per run, opened by its one start (B1-5) | confirmed |
| I1-127 | inventory | engine/program-name branches in `runwrap`, and `wrapper_log`'s `(?:SIESTA\|TBtrans)` | **closed 2.63.0** — no engine or program name branches in `runwrap`; `wrapper_log`'s program line matches any program's words | confirmed |
| I1-157 | inventory | an f-string with no placeholders in the run script's generator | **closed 2.63.0** — pyflakes clean over the run script's generator | confirmed |
| I2-096 | inventory | `set -e` across the run script's preamble | **decided 2.63.0** — a preamble line that fails stops the start with bash's own error (`running-a-job.md` § 5.2); `set +u` only around the preamble and activation | unchecked |
| I2-152 | inventory | hand-typed wrapper comment claims | **closed 2.63.0** — the run script's comments read against the code in two review rounds | unchecked |
| I2-204 | inventory | the wrapper warm-retries any `SCF_NOT_CONV`, diverging or not | **closed 2.60.0** (removed: a run is one start) | confirmed |
| I2-208 | inventory | the wrapper fence lists every SIESTA item, not the rung's | **moved to F3** — the rung's own items in the parameters record are the kind/pair record's — **closed 2.64.0 in part** — the run script's record of what the engine will read lists the items its step reads (`script_emit.declarations`, by `template.reads`), which its reader asks too; a SIESTA vibration's phases list their role's since 2.65.0 (F3-R-6) | unchecked |
| I2-279 | inventory | `render_run_wrapper`, the engines' `spec_for` | **open — longer**: `spec_for` stays the engines' own (F3) | unchecked |


### Found by F2's review rounds (2026-11-02)

| id | release | what | the door |
|---|---|---|---|
| F2-R-1 | **closed 2.64.0** — declared once, on the file catalogue's row (`cut_from`), asked through `runfiles.cut_from_output` | which output the run script stamps into `.scf-timing.log` is declared twice — the file catalogue's row (`engine = "siesta"`) and `scf_timing_rows.STAMPED_BY_RUN_SCRIPT` (role `.out`); they agree today | one declaration |
| F2-R-2 | **closed 2.64.0** — the engine record's `one_process` alone; `placement.one_process` deleted | "runs one process" is answered by two doors, the engine record's `one_process` and `placement.one_process` | one door |
| F2-R-3 | F9 | what only a real engine shows: a TBtrans run on GPUs, the advice printed for a real `propor` stop, `out-of-time` for a real wall stop, a PySCF GPU run's script on a GPU node | the e2e after F9 |
| F2-R-4 | F9 — the run script prints its thread line only after it finds the job's python, which only the real-engine road holds | two `restart_files.toml` rows pin a bash helper's spelling rather than its outcome | outcome rows |
| F2-R-5 | **closed 2.64.0** — the banner's labels padded to the longest it prints | the TBtrans banner's program label is one column wider than the others | — |

### Found by 2.64.0's review rounds (2026-11-02)

| id | release | what | the door |
|---|---|---|---|
| F3-R-1 | **closed 2.65.5** | the readers of an output's role — the monitor's live reading pass and loop words (`monitor.LIVE_READERS`, `LOOP_WORDS`), the used log (`record.USED_LOGS`), the contract reader (`contract.CONTRACT_READERS`) — are tables in shared modules and the bundle's own list; a new engine's output is a row in each | the file catalogue's row naming its reader parts, as the run-of-points words travel by their record — **closed 2.65.5**: the row's `reads` column (`runfiles.reader`); the monitor's, the record's and the contract reader's tables gone |
| F3-R-2 | **closed 2.65.0** | shared code still defaulted the kind to an optimization: `template.template_with_values`, `kmesh.mesh_for`, `DeckSpec.calculation` (PySCF's spec states none) | every caller states the kind |
| F3-R-3 | F5 | `BenchPoint.engine` holds the lane (`cpu`/`gpu`), not an engine — a written record's field | renamed with the benchmark record's schema |
| F3-R-4 | F9 | a TBtrans step's run script built through prep: a transport calculation cites a finished relaxation, which the road makes only with a real engine (T3); its API row is in 2.64.0 | the real-junction e2e |
| F3-R-5 | closed 2.64.0 | review rounds 1 and 2: `molbuilder validate` with a kind its engine does not run ending in a traceback; a description's step rule judging nothing when its ladder failed to resolve (a broad `except` hid the release's own signature change, in three places); the limit lookup's silent default; a bundle member's name clash; the step's items asked two ways; the timing file chosen four ways, and from a SIESTA run's seeded progress log; the engine worked out again in the metadata and transport checks; the seam's and the preflight's second door to the config class; a pair's pseudopotentials and a gathered deck read past their doors; the ending reader's special case; catches that took a defect of ours for an unreadable file; dead code and stale text | each fixed where it is decided |
| F3-R-6 | **closed 2.65.0** | a SIESTA vibration's phases each listed every role's settings under *what the engine will read*: one stage plays every role, and the run script is told its stage, not its phase's role | the phase's role handed to the run script with its point, the kind's roles being the kind side's |
| F3-R-7 | **closed 2.65.14** — a run whose ending cannot be read is *unreadable*, with why, in the closing line and message, never *running*; a notifier that raises is said in the monitor log, where its stdout is /dev/null | the monitor's closing state (`monitor.conclude`) and other monitor reads catch every exception, so a defect of ours there goes unsaid — deliberate for a process that must never stop the job, but unlogged | the catch logs what it caught |

### Found by 2.65.0's review rounds (2026-11-02)

*(The user, 2026-11-02: both rounds hunt duplicated, redundant and handcrafted code over whole touched
modules, and judge every refusal the touched code makes against the user's rules.)* Round 1: five whole-module
readers (the kind side, the conductor and the builders, the catalogue, the pages, the tests). Round 2: four fresh
ones (the kind side and catalogue, the conductor and builders, the pages and their answers, the tests and this
release's own documents), each told round 1's triage; it found five of round 1's fixes not made, now made. Each
finding checked in the code before it was acted on.

| id | release | what | the door |
|---|---|---|---|
| F3-L | **closed 2.65.2** — the user, 2026-11-02: *"Build it"*, on the design that named no new part | launch (`jobset/submit`) calls transport's reuse and gather by name: the gather of a rung's inputs, the take-over of a finished point, the shared copies (run on every kind's first launch, finding nothing), a point keyed by transport's axes (`point_key`), `GATHER_AT_FIRST`'s words; and the runs of points' declarations sit in `jobset/` | launch asks the description's `reuse` part as prep does, its doors listed once (`kind_table.REUSE_DOORS`, checked as it resolves); the gather handed over by `materialize.gather_into`; transport's modules in `transport/`, the vibration's in `spectra/` |
| F3-D | **closed 2.65.3** — the user, 2026-11-02: *"name ok, finished engine, option B yes, include the three"*; its trigger, *with a second citing kind*, was my own condition, never the user's | describe from a citation, and `describe --reuses`, were transport's code called by the command line (`_cli._init_transport`, `_init_reusing`, `_junction_slot`), and composed a second time by the Transport tab's describe door (the citation checked twice, the task built twice, an extension's engine written two ways, the window fit, the plan and the manifest each twice) | the kind's one describe, `transport/describe.py`, its record's `describe` part (`kind_table.DESCRIBE_DOORS`), asked by `_cli._described_by_its_kind`, taken by the tab; the inputs it does not take, its `not_taken`; the slot read by the general `_slot_of`; an extension's engine the finished calculation's |
| F3-H | **closed 2.65.11, in part** — `active_when = { method = ["DFT"] }` on both items *(the user, 2026-11-04: "Grey them out")*: the form greys them, the settings check says it, the deck's readback asks the one rule. The defaults read off the config class: 2.65.13, with B3-20 | PySCF's functional and grid level on Hartree–Fock are said by name (`validation/pyscf.py`, `pyscf/input.py`), a rule the catalogue could state (`active_when = { method = ["DFT"] }`, which would also grey the fields); defaults read off the config class (`validation/pyscf.py`, `validation/spectra.py`) where `default_of` answers | the catalogue's rule; `default_of` |
| F3-T | **closed 2.65.11** — a stage's own line in the deck and the targets the progress log hands the Watch tab ask `template.inactive_settings` | a setting the gate calls inactive is still stated as a target: the relaxation targets prep writes for the Watch tab with `relax_type = none`, a single point's deck comment | asked with `inactive_why` |
| F3-S | **closed 2.65.11** — both sentences deleted: `solvent_method`'s choices are all a value can be, and a value outside them never reached them; why neither is a choice is said once, on the catalogue row. Its `refs` the papers of the models offered | ddCOSMO and SMD are refused twice — the choices' check and each engine's own sentence (`validation/spectra.py`, `validation/pyscf.py`) | a `not_offered` row each |
| F3-E | **closed 2.65.1** — the user, 2026-11-02: *"you can have those checks give hint/info to the user but it is not your business to refuse"* — each said as information, the deck written as asked, the engine's own words kept when it stops | refusals that repeat an engine's own stop: TranSIESTA's temperature floor and pole count (`validation/transport.py`, ruled 2026-09-29), a spin state an engine cannot run and a PySCF parity (`validation/chemistry.py`) — against *don't do the engine's job* (2026-10-20) | the user's ruling between the two |
| F3-P | F9 | page readers no fast test runs: Task setup's rows by the kind's entry, the transport tab's Describe body and its pick's engine — each read headless in this release | F9's browser tests |
| F3-M | **closed 2.65.13** — `validation/stages` and `validation/siesta` ask `inactive_settings`. `siesta/input`'s own left: an engine's module, which raises on an unknown type | which relaxer moves atoms is stated three times: the catalogue's `active_when` on the step settings, the SIESTA deck writer's `relax_type != "none"`, and the stepless-relaxation check's `moves in (None, "none")` (`validation/stages.py`) | the catalogue's rule, asked (`template.inactive_settings`) |
| F3-N | **closed 2.65.7** | Task setup's extension entry composes outside a projects tree with `find_projects_root(...)` `None` (`build._folder_extension`; before this release) | the citation's tree, refused by name as the frame set's is |
| F3-K | kept | two wordings of one engine refusal from one record (`CalculationKind.on`, `why_not_on`); `identity.stage_name` reading a token-shaped name as a token (the doors handed either answer for a lead, whose name is its role's); a per-point value's field type unchecked at load (the kind table states none — the shipped one names the bias list); a sixth label's tone; a vibration's frame set asking transport's leads check (`compose.frames_keep_the_leads`); the sort serving both kinds from `transport/`; one `why_shared` sentence on eight rows (per row, as approved); prep's three paths saying their own end of one refusal | — |
| F3-R-9 | **closed 2.65.0** | round 2: a refused value's dependents said *no effect here*, and range advice for an override that enters nothing on its rung; *what enters nothing* composed four times (`template.inactive_settings`); the refused set three times; a kind that cites naming no `cited_frames` (Task setup's folder answer a `TypeError`); a per-point item naming no rungs; a part reference checked by its text alone; the composer and its record loader spelling TranSIESTA (the calculation's engine now, and the junction's pick names it as its sibling doors do, one refusal for the three); the cited slot read by name in transport's own modules and the blueprint; two `transport_spec` functions; a private import across packages; the preview refusal three times in prep; summarize's words built on every call; the window rung's catalogue defect answered as the person's error, and the window rule naming its rung by hand; the filter's `by label` chooser stale after a structure's arrival; `frozen_atoms` losing its tone to a served entry; MolView's own sentences naming transport's labels; the page's low-z lead from the table's order; the fit row's calculation from two sources; the hand-over library's door without its road; the setup reader's narrowing changing nothing (removed, the contract says what it reads); docs (the conductor's sentence, `reads`'s role, the folder answers' `kind` and heading words, the pick's engine, MolView's offered-labels read and order, the hand-over's needs) | each fixed where it is decided |
| F3-R-8 | **closed 2.65.0** | round 1: the structure answer's second serializer dropped `labels`; every chip copied the whole structure to find its label; the chooser's default stuck on `interface`; a refused value also said *no effect here*; a nameless transport describe defaulted its name; a citing kind's prep outside a projects tree a `TypeError`; a vibration call's record listing every role's settings; the cited slot read three ways; the stage name from a token read three ways; the transport deck's spec composed twice, one of them past the engine's seam; the window rule written twice; a pair's parts' form unchecked at load; dead code (a sweep check no sweep reaches, a `KindTableError` branch, `bias_points`); stale names in the docs and comments; tests reading a script's text, tautological expectations, vacuous rows | each fixed where it is decided |

### Found by 2.65.2's review rounds (2026-11-02)

Round 1: two whole-module readers (launch, the reuse module, the kind table
and materialize; the documents and the tests). Round 2: one fresh reader of
the whole release, told round 1's triage; it found every round-1 fix made.
Each finding checked in the code before it was acted on.

| id | release | what | the door |
|---|---|---|---|
| F3-L-R1 | **closed 2.65.2** | round 1: a launch again's carried gather composing the hand-over `materialize.gather_into` makes (`submit._carry_the_gather`); the walk composing the reuse's *nothing to run* words; `_shares` asking the record for the reuse part on every walked point; the finished calculation's name composed in three records; a whole-module part branched on its name, at load and as it resolves; `finished_of` and `fixed_rungs` exported with no caller outside; a doubled import; *a transport rung* said of a rung's gather in shared code; a door row testing a module with no door, which an all-or-nothing check would pass | `materialize.gather_into`; `Takes.refusal`; the part handed to `_shares`; one `_finished_name`; `kind_table.WHOLE_MODULE_PARTS`; private names; the real reuse module with one door taken away |
| F3-L-R2 | **closed 2.65.2** | round 2: the reuse module's door paragraph saying it writes only into a plan, its describe half writing the new folder; a finished run's shown name composed ten times; who asks each description part said twice in architecture § 3.3; what launch reads of the take-over unsaid; a door row's expectation copying the door list; the load check's `module:NAME` arm without a row; `covers` rebuilding its set per point; the changelog's seen and untested lists short; the moved modules' headers and imports unwrapped | `_shown`; one home, *A part that is code is named*; `Takes`'s fields said; the row's refusal alone; a row per arm; the set once |
| F3-L-X | **closed 2.65.7** | a kind's `export` is a module named whole resolved by its own door (`kind_table.export_of`), its doors (`export`, `name_of`, `ASKS`) listed in a comment and never checked, where a description's `reuse` is resolved and checked through `WHOLE_MODULE_PARTS` | one whole-module resolver taking the doors, for `export` as for `reuse` |
| F3-L-K | kept | a launch's own gather matched to a point by the point itself, where a point is matched across two calculations by `point_key` — two questions, each its rule (`reuse.takes_over`); `origin` and `frame_set_refusal` exported with no code caller outside, each a door the contract names (`runs-of-points.md` § 2.3, `engines/transport.md` § 2a.16) | — |

### Found by 2.65.3's review rounds (2026-11-02)

Round 1: two whole-module readers (the describe door and both roads; the
documents and the tests). Round 2: one fresh reader of the whole release,
told round 1's triage; it found every round-1 fix made. Each finding checked
in the code before it was acted on.

| id | release | what | the door |
|---|---|---|---|
| F3-D-R1 | **closed 2.65.3** | round 1: an extension's rung the tab sends empty refilled from the finished calculation, and an unknown rung dropped; shared values' names no longer checked, their blanks out of an extension's limits; an empty voltage list keeping the old voltages; the rename sent on an extension ignored (since before this release); the plan's refusals as 500; the finished calculation read again for a fit, the citation read twice by the schema route; the stage names in a refusal as `()`; `~` in `--slot` not expanded; the describe writer's warnings dropped; a hand-over stating no kind ending the folder answer in a 500; `finished_window` listed as a reuse door no shared caller asks; stale docstrings naming two describe doors; the docs naming a template door that does not exist and the pseudopotential rule for every kind | each fixed where it is decided: `describe.shared_of`, the bags by the road, the bias stated, the rename from the road, `Refused` for the plan, `_fitted`, the answers read once, `Description.notes` |
| F3-D-R2 | **closed 2.65.3** | round 2: a manifest whose folder does not open escaping as a traceback; an extension's rename left out of the tab's request read as *no rename*, refused; a new calculation's single bias defaulted by the command line alone; an extension's meshes judged on a frame set's axis kinds, not the junction it ran; a window guard no road reaches; the describe's refusal named `Refused` beside the extension's `Refused`; `shared_of` unexported; a hand-over guard its reader already holds; docstrings naming the describe doors and a writer's importer that is none; the changelog's visible changes short | the folder's refusal caught as the manifest's note; `None` for the rename; the single bias the describe's; the extension's junction for its meshes; `NotDescribed` |
| F3-D-K | kept | the pseudopotential check asked by two surfaces, the hand-over and the command line — one door, the engine seam's `describe_findings`, each surface saying its answer its own way; the command line's describe passing transport-shaped inputs (`junction`, `window`) — transport the one citing kind; the SIESTA validator's `cfg.psml_lib` words at the command line (the validator's, said on every road); the seam's `relabel` beside `engine_table.relabelled` — the contract's identity step (`script-preparation.md` § 3.4) delegating to the one implementation; the command line's generic one-calculation-per-folder refusal before a kind's own describe asks it again, unreachable there | — |

### Found by the e2e review and 2.65.4's review rounds (2026-11-02/03)

*(The user, 2026-11-02: "all e2e tests should be re-evaluated and reviewed in their full code since many details in
contract and design has been refined/revised. e2e tests may no longer pin the correct details. they need more work.
so make sure this happens before you run e2e tests to validate 2.65.4".)* Every e2e file read whole against the
contract as it reads now (nine readers), revised by eight fixers, the revisions read again by two rounds of readers
— each finding checked in the code before it was acted on. The reports:
[`audits/2026-11-03-e2e-review`](?doc=archive/audits/2026-11-03-e2e-review/README.md).

| id | release | what | the door |
|---|---|---|---|
| E2E-R | **closed 2.65.4** | e2e tests pinning details the contract no longer has — 5 heading rows on the pre-2.65.0 meta, the banner's fixed padding, the held-atoms line 2.63.0 replaced, a run-card step for a row that now stands, a GPU sweep that was a CPU sweep, a connection off this machine the sandbox forbids, negative checks on text gone from the code, a third load error 2.63.0 adds — revised to the contract; the text pane's ten tests (it mounts nowhere), duplicates and tests on faked failures retired; stale citations and goals; the checks the releases 2.57.0–2.65.3 named untested, added on runs the pass already makes | the contract's words, each test's docstring |
| E2E-P | **closed 2.65.4** | product and contract found by the review: the GPU notice telling the person to restart a server that asks the driver every few seconds; the chemistry card saying a method nobody chose was *stated*; molview § 11.2 contradicting § 1.1 (every edit lays a point); a catalogue row refusing a stated count on a transport rung, against ES6 and the 2.65.1 ruling; a heading branch no page reaches; contract text for four behaviours tests pinned and no chapter stated (a save's recovery, the docs pane's dark start, the tab-return ask, an out-of-set cell) | where each is decided: the notice, `electronic_state._method` + `stated_params`, molview.md, the catalogue's `offered`, `trajectory.js` |
| E2E-B | **closed 2.65.4** | option B's own review rounds: an extension's reuse decision rendering its decks before any pseudopotential was in its folder (every extension's first prep refused by the settings gate, its describe with no manifest), and the decision counting the new folder's frames, which hold none yet — the decision now asked under the folder its first prep leaves (`pseudos.PLANNED`, `reuse.carry_pseudos`), its frames the junction's (`ReadOnce.frames`, `rung.frames_had`); round 2: the describe's after-line, the ES5 advice's words, `stated_params` exported, a sequence test stopping a group before it started, docstrings claiming checks their asserts do not make | `rung.junction_record`, `reuse.decide` |
| E2E-F9 | **closed 2.67.9** for the page and browser tests; the rest, K2679-9 | handcrafted inputs in tests that are otherwise right — planted `sessionStorage` / `setShared` instead of sidebar clicks, the `Capabilities.file_picker_roots` patch (two files never restore `diagnostics._snapshot`), `page.route` fakes, synthetic DOM events, the typed GPU sampler, the forged `.XV` mtime, a typed `task.json`, the hand-built cell-type calculations, the notify file written by hand, the older-release deck cut by hand, the digests moved away; NOT-E2E checks to move to tier 1; table-row merges; copy pinned by text — each listed in the fixers' reports | through the road |
| E2E-OWED | F9 | rules only a new engine run or input shows, owed a test: an engine's own start stop prepared as asked (SIESTA, TranSIESTA on a stated count); the warm relaunch of a halted PySCF optimization; `launch --gather`; a `Spin.Fix` optical run's Fermi levels; the summary's two-geometry refusal; a labelled species through a run; the Spectrum tab's Send; the picker's own rules (one scan per visit, Reload, the folder it owns); the ladder pick and run heading on a vibration; a `mount` that throws drawing the error card and the AbortError guard (their faked tests retired); a monitor that dies loading and the ending the bundle cannot read (their hand-edited-bundle tests retired) | the real pass |
| E2E-Q | **the dead-code release** (with E2E-OWED's monitor that dies loading)  | `monitor.load_channels`' *reports off* branch for a bundle without `config_dir.py`, which `configuration.md` states and no road can make — recovery machinery for an unreachable state, now pinned by no test (its hand-edited-bundle test retired) | deleted with its row in `configuration.md`: a state against the design is an error, never recovered |
| E2E-K | kept | the *SIESTA trajectory* heading's removal (no page could show it); the junction written into prep's plan once per rung, as a citation's first prep already writes it (bytes identical); `finished_junction` reading the finished calculation's record directly, an unprepared extension never a finished one | — |

## F3 — § 1am step 3 — the engine record and the kind/pair record whole (47)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B1-17 | door | runfiles.py:190-193 — the file grammar's `engine` field restates the engine list | the shape built from `engine_table.engines()` (the catalogue already feeds that table's check) — **closed 2.64.0** — the grammar's `engine` field built from the file catalogue's own engines (`runfiles._ENGINES`), which the engine table checks itself against | confirmed |
| B1-18 | door | validation/__init__.py:52-53, :66, :301-304, :361-367 — the engines' settings checks keyed by config class, the engines listed by hand | an engine record part `validator`, resolved by the settings gate as the kind's is (`kind_table.part`) — **closed 2.64.0** — the engine record's `validator` part, resolved by the settings gate | confirmed |
| B1-19 | door | calculation.py:104-109 — each engine's reading of *the same calculation* registered by name | an engine record part for its reading, registered by walking `engine_table.engines()` — **closed 2.64.0** — the engine record's `same_calculation` part, resolved by `deck_record.same_calculation` | confirmed |
| B1-20 | door | resolve.py:967-989 — SIESTA's label bound in shared resolve | the engine record's label bound, asked by `resolve` with `task.engine` — **closed 2.64.0** — the engine record's `longest_label` (SIESTA's 48), asked by `resolve` with the task's engine | confirmed |
| B1-21 | door | template.py:1304-1313 — a second resolver of a config's engine, by its class name | the engine table read the other way (the engine whose `config` resolves to the class), or `task.engine` where the caller holds it — **closed 2.64.0** — `engine_table.engine_of_config` (the engine whose record names the class); `template.engine_name` deleted | confirmed |
| B1-22 | door | validation/siesta.py:591, :644; validation/pyscf.py:251, :302, :314, :328 — the engine's name as a person reads it, typed | `engine_table.engine(name).words` — **closed 2.64.0** — the engine record's `words` | confirmed |
| B1-23 | door | validation/chemistry.py:126-188, :192-, :293-316; chemistry.py:623 — SIESTA's and PySCF's own checks in the shared chemistry family | move each into its engine's validator (the S1 part above) — **closed 2.64.0** — the ECP hint and the metal-basis check in PySCF's validator; SIESTA's OMM rule a rule of its config (`spin_refusal`) | partial: PARTIAL \| validation/chemistry.py:126 `_check_ecp_declared_for_the_atoms_that_usually_want_one` (docstring: "PySCF's def2 files … `gto.M` applies one only when `ecp` is given"), :197 `_check_metal_b… |
| B1-24 | door | validation/siesta.py:572-586; validation/pyscf.py:255-265 — the junction checks written twice, one per engine, disagreeing on a vibration | once, in `validation.validate`, by the kind's record — **closed 2.64.0** — once, in the gate, where the kind's deck relaxes (`relaxes`) | confirmed |
| B1-25 | door | jobset/agreement.py:74-77 — a benchmark trial's cold start judged by SIESTA's restart keywords, by name | `deck_values("restart", job_set.engine, text)` (the `JobSet` carries its engine) — **closed 2.64.0** — one reader of a deck's restart group (`script_emit.restart_group`), the engine handed in | confirmed |
| B1-26 | door | parse/contract.py:44-46, :353, :507-516 — the relaxation record is SIESTA's by name, and an engine is inferred from a deck's suffix | the fdf extractor into `parse/engines/` by the deck's role; the engine from the deck role's catalogue row (`engine_of_role`) — **closed 2.64.0** — `CONTRACT_READERS` by the deck's role (`parse.engines.siesta_contract`); the `.fdf` vote removed | confirmed |
| B1-27 | door | parse/dirs/job.py:351-366 — SIESTA's stderr-apart rule read off the role `.out` | a field on the `_run_ending` reader's row: which stdout keeps stderr apart — **closed 2.63.0** — `_run_ending.STDERR_APART` / `stderr_apart(role)`, asked by `run_status` and the run script | confirmed |
| B1-28 | door | parse/dirs/record.py:71-72, :364; monitor.py:201, :366, :612 — engine- and program-named slots in shared readers | one stdout slot via `stdout_roles(engine)`; the reading pass's loop as a neutral slot — **closed 2.64.0** — one `stdout` slot by the engine's output role; the loop a neutral `loop`, its words TBtrans's own reader's | partial: PARTIAL \| parse/dirs/record.py:71-77 `out: Optional[Path] = None  # SIESTA family's stdout`, `pyscf_log … # PySCF's stdout`, `engine_log … # PySCF's own logger's file`, `fdf_log … # SIESTA's`, `timi… |
| B1-29 | door | runs.py:506-511 — the run door reads a deck's atom count as fdf when its suffix is `.fdf` | `engine(task.engine).keyword_deck` and the engine's own keyword reader — **closed 2.65.0** — deleted, the refusal naming such a deck kept (design § 4: build on what is live) | confirmed |
| B1-30 | door | web/blueprints/watch.py:155-186, :362-377 — the Results load asks which engine ran by inference, and words every engine's stop with SIESTA's table | `run_of(path).place.task.engine` first; the stop words from the ending the reader returns (`_run_ending`) — **closed 2.64.0** — the run door's engine first; a stop's words from its output's own table (`_run_ending.cause_words`) | partial: PARTIAL \| web/blueprints/watch.py:155-186 `_engine_of`: docstring "`running-a-job.md` § 4.2 owns the resolution order; `parse.contract.engine_of` is its one implementation" … `declared = engine_of(s… |
| B1-31 | door | web/blueprints/build.py:824 — each engine's form id prefix typed | a prefix composed from the engine table's names (none exists — the archive's NONE-EXISTS) — **→ F9** — the ids are pinned by the browser e2e's selectors, which run with F9 | confirmed — **closed 2.67.11, by judgement** (the user, 2026-11-07: "continue" on the scope): the prefix is stated by the server and the page, and the browser tests that pin the ids catch a drift |
| B1-32 | door | web/blueprints/build.py:1141-1163 — the hand-over relabels the config by hand beside the seam's relabel | `engine_seam(engine).relabel(cfg, label)` — **closed 2.65.0** — `engine_seam(engine).relabel` | partial: PARTIAL \| web/blueprints/build.py:1141 `_field = _engine(engine).identity_field` … :1163 `cfg = _dc.replace(cfg, **{_field: label})`; jobset/engines.py:308-315 `_relabel(engine)`: `field = _engine(e… |
| B1-34 | door | script_emit.py:224 — SIESTA's benchmark fields in the shared emitter | move it to `siesta/input.py` — **closed 2.64.0** — `siesta.input.BENCH_FIELDS` | confirmed |
| B1-35 | door | jobset/prep.py:302-348, :571-683, :687-760; data/kind_table.toml:123, :192 — both rung builders live in the conductor, each deciding the kind's and engine's facts by name | each builder into its kind's own module, the record's `rung` naming it there; the program a rung runs and its scratch from the role, as `SHAPE_OF_RUNG` (transport/deck.py:71) already tables the rung's deck — **closed 2.65.0** — `transport/rung.py` and `spectra/siesta_rung.py`, the record's `rung` naming each; `jobset/rung.py` shares the `Rung` and the structure step | confirmed |
| B1-36 | door | jobset/transport_rung.py:97-113, :193-201, :315-323, :341 — the transport rung's floor-3 module names its engine and its kind | `engine_seam(task.engine)`, `config_class(task.engine)`, `warm_list(task.engine, task.calculation, base)` — **closed 2.65.0** — the engine's seam (`spec_for`, `relabel`), the task's kind, the slot by `citation_of`; `compose_junction` sorts in the pair's order on the calculation's engine | confirmed |
| B1-37 | door | jobset/reuse.py:454; jobset/_cli.py:613-623; web/blueprints/build.py:2754-2761; web/blueprints/transport.py:378, :434 — the rung whose points share energies, a stage name in code; NEW: laid onto the extension's wire form in two doors | the window items' catalogue `stages` (the archive's verdict), or a role field in the kind table; `Extension.to_wire()` carrying it once — **closed 2.65.0** — `transport.window.window_rung()` from the window items' `stages`; `Extension.to_wire()` carries `window` | confirmed |
| B1-38 | door | jobset/submit.py:1110-1111, :1209, :2954, :2993-2994 — the shared walk and gather ledger name a transport point's `frame`/`volts` | `reuse.point_key(pt)` / `Point.to_wire()` — **closed 2.65.0** — the walk's rows carry the point; the ledger writes `point.to_wire()` | confirmed |
| B1-39 | door | jobset/runstatus.py:100-108, :539, :575-597 — the status door imports transport's lead-file line directly | a pair part for its scratch line, resolved by `kind_table.part_of` — **closed 2.65.0** — the pair's `scratch_status` part | confirmed |
| B1-40 | door | jobset/_cli.py:390-470, :812 — `describe` for a citing kind is transport's routine, its kind and rungs spelled | the kind from the record that cites; `kind(...).on(engine)`'s refusal; the rungs from `kind(...).role_names` — **closed 2.65.0** — the slot from `description.cites`, the stages from the pair's ladder (`fresh_stages`), the engine refused in the kind's one sentence (`why_not_on`) | confirmed |
| B1-41 | door | jobset/migrate.py:1045 — a deck found by its `.fdf` suffix | `deck_role(task.engine)` — **closed** — `jobset migrate` reads no deck by its suffix (checked at 2.65.0) | confirmed |
| B1-42 | door | validation/chemistry.py:318-336 — a kind's fixed charge refused with transport's ruling sentence beside the record's `why` | the sentence into the kind table's `charge.why` — **closed 2.65.0** — the kind table's `charge.why` | confirmed |
| B1-43 | door | validation/pyscf.py:177-178, :264; siesta/input.py:619-622; pyscf/input.py:207-210 — engines' own modules restate which kinds they run | `kind(c).on(engine)` (its refusal names the engines); delete the dead branch — **closed 2.64.0** — `kind(c).on(engine)` | confirmed |
| B1-44 | door | validation/siesta.py:208, :603; validation/spectra.py:613; validation/pyscf.py:176-179; validation/chemistry.py:273-282 — the electronic state's guarded read composed five times | read once in `validation.validate` and handed to each validator in `engine_kw` — **closed 2.64.0** — read once in the gate and handed to each check | confirmed |
| B1-45 | door | validation/task.py:418-446, :178 — the transmission window's description check in the shared preflight, by stage name | a description-level part of the kind (none exists — archive NONE-EXISTS), or the role marked in the record — **closed 2.65.0** — the kind's `description_checks` part | confirmed |
| B1-46 | door | transport/sort.py:317-348; runs.py:351-355, :500, :518; workingcopy_structure.py:216-227; jobset/prep.py:591-600, :1898 — a pair's atom-sort key held in a code dict, asked by shared readers | a field of the pair's record (`[<kind>.on.<engine>]`), read by `kind_table` — **closed 2.65.0** — the pair's `atom_order`, `kind_table.atom_order` | confirmed |
| B1-48 | door | mode_frames.py:217, :260 — the frame set generator falls back to SIESTA's force-constant call | into `spectra/` (the archive's verdict), or the run of points' declaration naming the call that measured — **closed 2.65.0** — `spectra/mode_frames.py` | confirmed |
| B1-49 | door | template.py:2093-2111, :2479-2483 — refusal reasons keyed by item name, a SIESTA and a transport sentence in the shared catalogue door | a `why` on the catalogue row (none exists — archive NONE-EXISTS) — **closed 2.65.0** — the catalogue's `why_fixed`, `why_shared`, `per_point_from` | confirmed |
| B1-50 | door | atom_index.py:318 — a cited deck's kind decided by its file name | the cited kind from the citing kind's description (`description.cites`) — **closed 2.65.0** — the citing kind's `cited_kind` | confirmed |
| B1-51 | door | web/blueprints/build.py:1652-1665, :2746-2763, :2840-2870 — Task setup's shared routes call transport's reuse and citation modules | kind parts for an extension and a citation's frames (none exists — archive NONE-EXISTS) — **closed 2.65.0** — the `extension` and `cited_frames` parts | confirmed |
| B1-52 | door | report_fields.py:83-133 — the run-of-points report fields gated by engine and kind names | gate by the pair's `run_of_points` record — **closed 2.65.0** — a field's `runs_points`, gated by `kind_table.run_of_points` | partial: PARTIAL \| report_fields.py:81-84 `ReportField("call_elapsed_s", …, engines=("siesta",), calculations=("vibration", "transport"))`, likewise `units_left` :93-95, `left_at_pace_s` :99-102, `done_pct`/… |
| B2-9 | door | a deck found by spelling `.fdf` in shared and kind code | `runfiles.deck_role(task.engine)` / `deck_roles()` | confirmed |
| D1-RP-1 | residue | runs-of-points.md § 2 — "nothing in the shared code names a kind, a phase or a file of one": the walk names a transport point's fields | the match asks the point type (`point_key`, already `reuse`'s) and the ledger row carries `pt.to_wire()` — or the sentence names the exception until step 3 | confirmed |
| I1-018 | inventory | a GPU run with a non-ELPA solver refused only when the deck is written, never at Save | **closed 2.64.0** — an error of SIESTA's settings check, heard by the page's | confirmed |
| I1-026 | inventory | a spin-polarized optical call's Fermi level reads none | **closed 2.58.0** | confirmed |
| I1-042 | inventory | every engine and kind fact asked of its record | OPEN | confirmed |
| I1-060 | inventory | which rung runs TBtrans decided by stage name in prep, beside `SHAPE_OF_RUNG` | **closed 2.64.0** — the program its deck states (`DeckSpec.program`, from the deck writer's `SHAPE_RUN`) | confirmed |
| I1-062 | inventory | the fit's rung laid onto `Extension.to_wire()` by hand in two doors | OPEN | confirmed |
| I1-072 | inventory | the monitor bundle's companion list names the kinds' modules | **closed 2.64.0 in part** — the kinds' parts and the engines' keyword readers travel by their records (`runwrap.monitor_companions`); the output readers by role → F4 (F3-R-1); **closed 2.65.5** — the output readers travel by their rows (`reads`) | confirmed |
| I1-073 | inventory | the run-of-points report fields gated by pair | OPEN | confirmed |
| I1-077 | inventory | `_across` and the gather's ledger rows name a transport point's `frame`/`volts` | OPEN | confirmed |
| I2-039 | inventory | the trajectory inspector titles every engine file "<ENGINE> optimization" | OPEN | confirmed |
| I2-109 | inventory | every SIESTA relaxation deck carries literal advice, reading nothing of its own values | OPEN | unchecked |
| I2-141 | inventory | BENCH-MARKS SIESTA-only | **closed 2.64.0** — the benchmark summary reads a trial's deck by setting (`summarize.deck_item`) | unchecked |

## F4 — § 1am step 4 — the file catalogue and the record doors (29)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B2-3 | door | across 41 modules — catalogue roles spelled as string literals at call sites; `runfiles` names 13 roles as `ROLE_*` constants (plus `TBT_DIGEST`, `TBT_DIGEST_CLAIM`) of its 40 role rows (`ON_THE_LABEL`), and `compose` refuses no role the catalogue lacks | a `ROLE_*` per catalogue row in `runfiles`, imported at every site; `compose` asks `_row` (an engine's own file on the label — `.XV`, `.FC`, `.ORB_INDX` — a row or `_ENGINE_ROLES`) — **closed 2.65.5** — a `ROLE_*` per row, imported at every site; one check (`runfiles.declared`) in every door that takes a role | confirmed |
| B2-4 | door | runfiles / template / script_emit — two roles spelled twice, each by a second module beside its catalogue row | the constants in `runfiles` (it is floor 1, so `template` importing it is downward), `template_file` composing through `runfiles.compose` — **closed 2.65.5** — `template.SUFFIX` gone, `template_file` composing `ROLE_TEMPLATE`; the validation report named by `runfiles.named_beside` | confirmed |
| B2-5 | door | the vibration's force-constant file spelled beside its phase's declared product | `phase(FC).products[0]`, as line 1051 asks for the optical product — **closed 2.65.5** — `phase(FC).products[0]` | confirmed |
| B2-6 | door | result and record names composed with a literal role in the kind modules | the `ROLE_*` constants of B2-3, a generated or stated name through `runfiles.compose` / `tail` — **closed 2.65.5** — the constants, through `compose` / `tail` | confirmed |
| B2-7 | door | the pseudopotential family globbed as `*.psml` beside `runfiles.find_fixed(dir, PSEUDO_FILE)` | `runfiles.find_fixed(dir, PSEUDO_FILE)` (after B2-2's field), `runfiles.patterns`/`fixed` for the run script's glob — **closed 2.65.5** — inside a calculation, the family (`find_fixed(dir, PSEUDO_FILE)`); a library folder scanned by format | confirmed |
| B2-8 | door | the TranSIESTA surface Green's function name composed in one module and split by hand in another | a catalogue row with fields (`{label}.{volts}.TSGF{lead}`, `FIELDS`) composed by `fixed_name`/`compose` and read back by `find_fixed` — **closed 2.65.5** — `ENGINE_FILES`' `SIESTA_TSGF` with its fields, composed by `transiesta.lead_gf_file`, read back by `parse` and `fields_of` | confirmed |
| B2-10 | door | a launch group's file path composed at every site — no door gives `<container>/launch/<group><tail>` | `GroupNames.path(container, tail)` beside `GroupNames.name`, asked by every site — **closed 2.65.7** — `GroupNames.path(container, row=None)` and `walked(row=None)`, every launch-group join | confirmed |
| B2-11 | door | the machine record's path probed by callers instead of asking its door | `calculation_machine(base)` / `calculation_environment(base)` asked, and the `UnknownTarget` it raises said as it is (`str(exc)`) — **closed 2.65.7** — `calculation_machine` for the launch lines' flags, `Environment.scope` set by `machine_for` for the refusal's words, the door's `UnknownTarget` said as it is | confirmed |
| B2-12 | door | the description's and the job request's files read by path beside `task.of` / `job_request.of` | `job_request.of(base) is None` in `refusal`; Save compares with the `Calculation` it opened (`calculation.open`); `_read_as` through `task.path_of` / `job_request.path_of` (or a door answering the bytes' hash); `_record… — **closed 2.65.7** — `job_request.of` in `refusal`; Save compares with the opened calculation's records; `_read_as` through `task.path_of`/`job_request.path_of`; `describe` through `write_task`/`job_request.write`; the data file from the opened calculation | confirmed |
| B2-13 | door | runs.py:1141 — a benchmark's plan path formed by the run door | `JobSet.path_of(folder)` (T6 wanted a floor-1 path door — the run door is floor 2, `jobset.model` floor 3; that is the open question of the earlier finding) — **closed 2.65.7** — `calcdirs.plan_path(folder)` on floor 1, `JobSet.path_of` handing off to it | confirmed |
| B2-14 | door | checkpoint.py:389-392 — a calculation's root told by probing two record names joined to the folder | `calcdirs.described(path)` (a bench container's plan: `JobSet.path_of`) — **closed 2.65.7** — `calcdirs.described` alone — a plan alone marks no root | confirmed |
| B2-15 | door | jobset/reuse.py — a run's `.copied-from` path formed outside `runrecord` | `runrecord.copied_from_path(folder)` beside `continued_from_marker`, asked by the reuse — **closed 2.65.7** — `runrecord.copied_from_path`, `gathered_from_path`, and `RUN_RECORDS` — what a copy does with each record | confirmed |
| B2-16 | door | the stage a folder is in, read off a typed path's first part | `calcdirs.where(base / from_attempt).stage` — **closed 2.65.7** — `calcdirs.where(...).stage` | confirmed |
| B2-18 | door | three climbs off a file's resolver or `__file__` to a root another door owns | `config_dir()` / `environments_dir()` (scheduler/record.py:1014 names it), `molbuilder.repo_root()`, the data folder by the package's own door — **closed 2.65.7** — `molbuilder.data_dir()`, `repo_root()`, the probe record's folder from its owners; `$MOLBUILDER_DATA_DIR` removed (the user, 2026-11-03) | partial: PARTIAL \| jobset/_cli.py:3138-3140 `record = named_environment_path(name)` / `machine_scope_path()`; `target = record.parent`, :3214-3215 `ensure_private_dir(target)`, `write_environment(env, target… |
| B2-19 | door | a runtime_info record named by `with_suffix` + a literal | `runfiles.compose(label, ROLE_RUNTIME_INFO, stage, run)` from the file's own `runfiles.parse` — **closed 2.65.5** — `runfiles.named_beside(input, ROLE_RUNTIME_INFO)` -- one per run; a name no role reads back refused, naming `--out` | confirmed |
| B2-20 | door | the notification reports folder and file named outside `config_dir` | `config_dir.reports_file(user)` (W53's missing door) and the parser asking `reports_dir()` — **closed 2.65.7** — `notify.reports_path(user)`, the one join; the reader telling the file by its folder's name → F7 (F4-267-f) | confirmed |
| B2-21 | words | record names re-spelled in what a person reads and in generated text | each message and generated line formats the catalogue's constant (`runfiles.TASK_FILE`, `JOB_REQUEST_FILE`, `MAKOV_PAYNE_SCRIPT`, the template's role) — **closed 2.65.5** — messages, help options and generated lines format the constants (command docstrings kept, F4-R-K) | confirmed |
| B2-22 | door | spectra/export.py:31 — a vibration export's name from `calc.task.label` beside the result it is of | the stem from the found result's name (`runfiles.stem_before(result, ROLE_SPECTRA)`) — **closed 2.65.5** — `stem_before(result, ROLE_SPECTRA)` | partial: PARTIAL \| spectra/export.py:31 `def _stem(calc): return f"{calc.task.label}.vibration"`, used by `name_of` (:39) and `exported(…, stem=_stem(calc))` (:149) \| The `_stem` site is as stated, but it n… |
| D6-11 | words | `MONITOR_BUNDLE` named under two homes | the documents name `runfiles.MONITOR_BUNDLE` — **closed 2.65.5** — the bundles named and imported from `runfiles` in code, tests and documents | partial: **PARTIAL** \| runfiles.py:1022 `MONITOR_BUNDLE = "mb_monitor.pyz"`; runwrap.py:4024 `from .runfiles import MONITOR_BUNDLE  # noqa: E402 -- the catalogue's name`; documents say `runwrap.MONITOR_BUNDL… |
| I1-043 | inventory | every name and record through its owner | **closed 2.65.7** — every record of a calculation and of a run through its owner (the record doors J–R) | confirmed |
| I1-056 | inventory | the vibration's `points_in` reads the plan before launch wrote an index | **closed 2.65.7** — the declaration's `points_in(run, calc, stage)` reads the plan the caller holds | confirmed |
| I1-076 | inventory | the walk's loop skips a planned point by its folder's last name | **closed 2.65.7** — a walk's skips as `RunPlace` names them | confirmed |
| I1-099 | inventory | `gf_files._lead_orbitals` reads a lead's latest run | **closed, not a defect** (measured 2.65.7): a device's gather takes each lead's newest run and refuses it unfinished (D5, `transport.gather.transport_inputs`), the run status reads | confirmed |
| I1-101 | inventory | the label cap leaves no room for the try counter | **closed 2.60.0** (removed: a run is one start) | confirmed |
| I2-010 | inventory | `StageRef` lists built by hand beside the door `naming.described_refs` | **closed 2.65.7** — `naming.described_refs` at the three sites | confirmed |
| I2-085 | inventory | the wrapper's kill line on a killed point never seen on the road | → F9 (E2E-OWED): only a real engine run shows it | unchecked |
| I2-093 | inventory | a killed run leaving no conclusion marker — no test | **closed 2.65.5** — pinned: a stopped run writes no marker (`test_launch_door_gate.py`, `test_a_sent_run_the_scheduler_forgot.py`) | unchecked |
| I2-107 | inventory | a PySCF relaxation with a held atom, end to end | → F9 (E2E-OWED): only a real engine run shows it | confirmed |
| I2-276 | inventory | the files a calculation holds read once (`JobSet.of`, the template text, `Shape.named`) | **closed 2.65.7** — the vibration's second read gone (`points_in` takes the calculation); the other `JobSet.of` calls are the root plan's door, prep's overlay and a benchmark's set (`Shape.named` gone 2.54.0) | unchecked |

### Found by 2.65.5's review rounds (2026-11-03)

*(The user, 2026-11-03: "One table in runfiles", "One column on the catalogue row", "Yes, move to F6", "Two
releases, as proposed".)* Three fixers changed the call sites under one brief; four readers then read the whole
touched modules against the contract (round 1), and three read them again after the fixes (round 2) — each
finding checked at its line before it was acted on.

| id | release | what | the door |
|---|---|---|---|
| F4-B | **closed 2.65.5** | the fast batch: a test pinning `job.transport.parse.log`, a parse-log name the catalogue dropped on 2026-10-04 as having no writer (D22) | retired into one table of the layout's two names and a name no row reads back |
| F4-R1-D | **closed 2.65.5** | round 1's defects: `jobset migrate` stopping on a traceback at an older run's file that names no role (the run left as it is again, said); a TBtrans data file whose deck names its leads otherwise refused whole; `RunNames.name` refusing an engine's file, so a PySCF deck named its restart files from the restart list; a shipped scratch suffix taking in other files' names accepted (`*.nc`); `stem_before` answering a templated role with nothing; § 5.4 rendering TBtrans's data files as the engine's scratch; `log_of_run` with no stage answering any stage's log; the readers' load check weaker than 2.65.4's (an `ending` on no run output, readers on a row with no role); the reader doors passing a misspelled role; a record reader that does not resolve swallowed with its fields; the record's files and the rows' `record` readers unchecked | migrate's own check first; the `lead` / `pair` shapes TBtrans's, `values` what our decks name; `_staged` of either table; `rows_ending_with`; the template's pattern; `_carried` by each name a row makes; the stage kept; `_check_reads`; `declared` in the reader doors; `_a_row_per_record_reader` |
| F4-R1-W | **closed 2.65.5** | round 1's one homes and words: the name of a role's file beside another's built four times (runtime-info, the parse log, the validation report, the deck hint); the TBtrans channel tags in three spellings; the leads' names twice, unchecked; the checkpoint's engine families typed with their stars, three of them no row's; catalogue templates filled with `str.format` for words; the GF-file's lead split by hand; a run script's and a header's name spelled two ways; `scf_rows` said asked beside a job; the `.sbatch` row spelling the walk's header; comments naming the bundles' old home, `runwrap`; parse.md's parser recipe spelling a suffix; docs and help naming what the code no longer does | `named_beside`; `TBT_CHANNEL_TAGS`; the lead field checked as `transiesta` loads; `family`, rows `TBTRANS_CC`, `TBTRANS_DOS`, `PYSCF_CUBE`, constants `SIESTA_CG`, `SIESTA_HSX`; `shown`; `fields_of`; `.run_script` / `.header`; the words |
| F4-R-K | kept | TranSIESTA's own name for a lead's Green's functions, `<label>.TSGF<lead>`, which a deck naming none gets: its lead is not read (our decks name every lead's file, `<label>.<bias>.TSGF<lead>`); the run record's `out` and `log` rows keyed by their files' roles (the fields each file answers, checked against the rows as the record loads); `<label>_optimized.xyz` an engine file (restart state is never also a row of ours, § 4.2, as at 2.65.4); a person's deck named `INPUT.FDF` gets no hint (a deck of ours is lower case); no parse log for a name no row reads back (handcrafted names are not ours); command docstrings naming records in prose (a docstring is no f-string); the job set's file named through its owner's re-export beside `runfiles`' constants (both the catalogue's); a call's output in words, `<label>_<stage>.out`; the Results file card showing a templated role for an engine file, as for ours | — |
| F4-R2-D | **closed 2.65.5** | round 2's defects: `molbuilder runtime-info` refusing geomeTRIC's trajectory and SIESTA's `.MD.nc`, both road files (round 1's door read a name's role as our dotted roles alone); the monitor stopping on a stray file beside a run (an editor's `.out~`), and a stopped output named outside the catalogue failing to load, both asking a refused role of the reader doors; a PySCF deck's `<job>_initial.xyz` claimed by two parsers, so it could not be opened (since 2026-10-19); the writer's temporary-file row taking in geomeTRIC's scratch folder, and the sidecar's lock named with a random part it has not; `named_beside` keeping a stage on a row named on the label alone; migrate deciding its fifth reason two ways | a name's own role the longest of either table whose stem a label begins; a file of a run named by the run, decided before the parse; the monitor and `_run_ending` asking only the rows that name a reader (`read_roles`); the trajectory reader leaving both pairs to the geometry reader; `{target}.{random}.tmp`, `{target}.lock`; `_staged`; one check |
| F4-R2-W | **closed 2.65.5** | round 2's one homes and words: the `tbt` field's shape typed beside its tags; our leads in three spellings (transiesta's blocks, the field's values, the pair's example); geomeTRIC's prefix spelled twice; `find_by_role` and the reader doors outside the one check, the readers looking a templated name up by its exact spelling; the PySCF header naming its restart files by hand; a session log's role typed in the catalogue itself; the PySCF log's hint a fifth copy of the name-beside question; the export's stem typing part of a role back; `fixed()` writing `shown` out; the contract's door list, `parse`'s `roles`, architecture's link and its *counter*, a mis-pointed section, the GF file's lead words, parse.md's load checks, module headers | the shape from `TBT_CHANNEL_TAGS`; `OUR_LEADS`; `GEOMETRIC_PREFIX`, `GEOMETRIC_APPENDS`; `declared` and `canonical_role` there; the rows' `staged`, through `names.name`; `ROLE_SESSION_LOG`; `named_beside`; the extension; `shown`; the words |
| F4-R2-N | **closed 2.65.6** | `role_of` and `find_by_role` read the fixed name `coordinates.fdf` as a SIESTA deck; SIESTA's fdf log named by its reader's own pattern beside the catalogue's family | *the user, 2026-11-03*: `coordinates.fdf` left as it is read (no harm measured; `junction.cited.fdf`, also named here, is a real deck — the cited relaxation's own — and was named in error); the fdf log's reader through the catalogue → 2.65.7, SIESTA 4's unstamped `fdf.log` no longer read |
| F4-R2-F9 | F9 | the file picker's page test building the server's rows by hand, its parser named by a hand map, not `detect` — it passed with the `_initial.xyz` collision present | through the road |
| F4-F7 | **closed 2.65.6** | `_run_ending` holds the SIESTA family's, PySCF's and the progress log's `ending` readers, so the monitor's own imports carried `siesta_reader` | not a defect: no rule asks the ending readers into the engines' own modules — the sentence that seemed to (parse.md § 5.5, *each reader in the engine's own module*) was written in 2.65.5 and never the user's; the parking was mine. The bundle by its members' word (2.65.6) carries `siesta_reader` by `_run_ending`'s `TRAVELS_WITH` |

### Found by the discipline review of 2.65.5 (2026-11-03)

*(The user, 2026-11-03: "did you do a discipline review? you have rackless coding and handcraft rogue code before".)*
Four readers read 2.65.5's whole diff against the user's working rules, not the contract — each new function, rule,
check and visible change traced to the user's words or not; each test's input to the road or not; each claim to a
measurement or not (the reports: `d-catalogue`, `d-readers`, `d-callsites`, `d-tests`). The user answered every
choice; 2.65.6 carries the answers, and two review rounds read it, one of them on the rules. What it changed in
how a release is made: before the ledger and the release, one pass over the whole diff on the rules, its findings to
the user first.

| id | release | what | the door / the answer |
|---|---|---|---|
| D-FOREIGN | **closed 2.65.6** | support for decks molbuilder did not write: the `lead`/`pair` fields taking any word, two test rows on invented lead names, three changelog lines calling fixed regressions changes a person sees | the fields closed to `runfiles.OUR_LEADS`, a foreign deck's data file refused in a sentence; rows retired; 2.65.5's row corrected |
| D-TESTS | **closed 2.65.6** | about 50 test cases on inputs no road makes — a typo role through every door, hand-built rows fed to load checks that run on the real tables, a hand-edited shipped list, invented names, a monitor test whose one assertion held for a monitor that read nothing — while the release unified an API and grew the suite by ~100 | retired; three a first pass retired unasked restored (travels beside a deck, a caller's own vocabulary, the PySCF geometry pair's parser) |
| D-UNASKED | **closed 2.65.6** | additions no approval named | the user's answers: the scratch-suffix rule and `rows_ending_with` gone; a skipped parse log said on stderr; `roles=` gone where no caller passes it. Kept: the 20 engine files explained by name, the wider checkpoint families, the export names' kind from the calculation, the narrow SIESTA hint |
| D-DOORS | **closed 2.65.6** | door I half done (`siesta_reader` listed by hand in the bundle), the run record's second table keyed by role | the user's yes: every bundle member's `TRAVELS_WITH`, the base list the monitor alone; the record's file rows from the catalogue rows' `record` readers, each stating its `fields`, the run's files handed by role (`RunFiles.files`) |
| D-WORDS | **closed 2.65.6** | contract sentences the code did not bear out (`reads_of` as every reader's door; *each reader in the engine's own module*; `cut_from` on the output's row; § 5.4's scope worded three ways), comments where removed tables stood, the manifest tool's copied door check and hand map | the words; one door check, the engine record's words, `compose` |
| D-R1 | **closed 2.65.6** | the first build of the user's (c) — a fixed name of ours read as no role, keyed on `counted_by` — made SIESTA's parser claim the charge-correction script prep writes beside a charged deck; it rested on a description of mine the user was given wrongly (`junction.cited.fdf` is a deck) | removed; the user then left `coordinates.fdf` as it is read; the correction script a row of the test that nothing shipped beside a job is taken for its output |
| D-REVIEW | review | a `TRAVELS_WITH` entry for a module imported only inside a function: loading the bundle proves nothing of it, and a test reading the source is not one the rules allow | review's (residue), and a real run's (F9) |

### Found by 2.65.7's review rounds (2026-11-03)

The record doors J–R (`design.md` § 4) and SIESTA's fdf log through the catalogue, built by four fixers on a brief that
forbade new doors and rules, then read in two rounds, each a launch reader, a records reader and one on the user's
rules (the reports `r267-1-*`, `r267-2-*`). The user answered every choice (2026-11-03): door Q "The four only"; the
vibration's plan read "Inside 2.65.7"; `$MOLBUILDER_DATA_DIR` removed, "agree with your recommendation"; the clean-ups
found "Plan rows for F6/F7", the last of them "F7 row, as filed"; the two choices handed up "Keep both".

| id | release | what | the door / the answer |
|---|---|---|---|
| R267-1 | **closed 2.65.7** (before it shipped) | migrate's *prepare each group again* named no group: a rename in its one long records step left a later use reading the job request's body | `_prepare_again` takes the rewritten `JobSet` whole (A8); the records step split → F4-267-a |
| R267-2 | **closed 2.65.7** (before it shipped) | the machine record's new `scope` printed in every prep's pipeline log, which dumps the record's fields | `Environment.scope` no field (a `ClassVar` set on the record `machine_for` hands out); the log's own dump → F4-267-b |
| R267-3 | **closed 2.65.7** (before it shipped) | a calculation a 2.41 prep wrote and an earlier migrate rewrote could not be migrated again: the removed keys' rule collided with the kept file's name | the removal written before 2.65.7, the release whose reader refuses the keys (`migrate._READ_PAST_UNTIL`) |
| R267-4 | **closed 2.65.7** (before it shipped) | `JobSet.of` worded a plan that is no JSON as its reader's refusal, so the sweep reader's own words for a killed write could not be said | the door words the reader's refusals alone |
| R267-5 | **closed 2.65.7** (before it shipped) | the viewer's answer at a run of points' root `null` for a calculation whose records do not read | `unreadable`, in the reader's words, as `views_of` says it (`web/results.md` § 4.1) |
| S267 | **closed 2.65.7** | the This machine tab's reports for `?user=../x` read a `.jsonl` file outside `reports/` (before this release) | the route asks `monitor.is_notify_user`, the rule's one home; a name no user id can be answers *no reports* |
| D267 | **closed 2.65.7** | from the rules reading: a second writer of the machine copy; the template's path formed by hand in `describe`; a test planting a value no molbuilder wrote (`asked` `{}` on a ladder job); tests on states no road makes (an export module missing a door, a root test checking a file it wrote, a producer called twice to prove two writers); comments and documents the change made untrue; the data README still inviting edits of the shared table | each fixed where it lies; the tests retired |
| C267 | **closed, not a defect** | I1-099: a lead's orbital count from the run the device takes | the device's gather takes each lead's newest run (D5) — the run the status reads |
| K267 | kept | the readers that read past a key beside door Q's four — the folder record, the machine record, `run.json`, a sequence's record, a job's address, a run of points' index | the user: "The four only" — two tolerant by contract, four with no key list |
| K267-2 | kept | a kind's own page importing its kind's modules | not a choice: the pages import their kind's package throughout; door R is the export's resolver |
| K267-3 | kept | the junction's sidecar named by the catalogue constant, not the codec's pairing rule; migrate writing the template's text itself | the user: "Keep both" — one home for the name (A11); migrate the one raw writer A16 names |
| K267-4 | kept | the fixed-name doors accept every engine file named whole, not the fdf log alone; the launch folder asked of a group's names (`GroupNames.path(container)`); the data file stating the records in their writers' text | the door extended, never one row special-cased; the launch folder is the container's, the names its door; the bytes are the writers' (`persist.json_text`) |
| F4-267-a…g | **closed 2.65.16: a (`_migrate_description`, `_migrate_request`, `_migrate_job_sets`), e (`Kind.said_of_job`), g (the calculation whole).** b left by the user's answer, c closed 2.65.13, d and f left: no caller joins a name, no road reaches the parser | migrate's records step one function per record (a); the pipeline log's machine record through its owner (b); one check of a record's keys (c, F6); a sequence's record one writer, its reader the folder (d); the scheduler's error file located once (e); the reports file's name where its reader can ask it (f); `point_done`, `must_hold`, `also_needs` taking the calculation whole (g) | `plans/plan.md` F6, F7 |

### Found by 2.65.10's review rounds (2026-11-04)

F5's third release, doors N–P of its design, built by hand, then read in two rounds, on the user's rules and on real runs:
a hydrogen-chain transport across 2.65.9 and this release, PySCF on the GPU, 183 real settings logs, 18 real builds through
the tools. Every finding judged by the user's test, 2026-11-04: "what actually goes wrong, and does it matter?".

| id | release | what | the door / the answer |
|---|---|---|---|
| R2610-1 | **closed 2.65.10** | a duplex asked with its 5' phosphates (`5P`, `PP`) refused, *backbone is broken*, sending the person after a configuration fault: the build's check took chain A's last residue for chain B's first's neighbour, the strands numbering their residues on | `verify_backbone_connectivity` per chain; a broken strand still refused, by name |
| R2610-2 | **closed 2.65.10** | a PDB with its element column blank read a four-letter hydrogen as a metal (`HG21` mercury, `HE21` helium, `HO5'` holmium) and `1HG2` as the element "1": measured by blanking the column of a real tleap peptide, 14 of 71 atoms. tleap's own file fills the column | `Structure.from_pdb`'s fallback |
| R2610-3 | **closed 2.65.10** (before it shipped) | two saved SIESTA logs added to the basic suite as fixtures, against `process/testing.md` *(the user: "when something reads the output of siesta output that's an end to end by definition")* | retired; rows in the two real-run tests |
| D2610 | **closed 2.65.10** | from the rules readings: the mercury fix's reason worded as if tleap's own file were misread; a name no contract held (`WINDOW_BLOCK`), and the window's reader imported at load beside a job that never asks it; by-name format checks left in three places; total memory read twice, the reason given for leaving it untrue | the words; the import inside the function; `format_named`; `runtime_info.mem_total_gb` |
| K2610 | kept | read a second way, and nothing goes wrong: the energy blocks' swap (`transport.reuse`: rewrites a deck's text, and the fdf reader writes none); the finished deck's check (`siesta.layout.check_rules`: must see a setting written twice, which the reader drops); the two conda readings (one package, 5 ms, against all of them, up to 0.5 s; they agree on all 11 environments here); the machine's GPU names, CPU layout and GPU use (each another question of its tool); `atom_index` picking a lone file's parser; the build page's load route falling back to the text on an unknown name | the user's test, 2026-11-04. A departure from door N, told |
| K2610-2 | kept, told | the Run panel's row `TS.Elec.L.UsedAtoms`, `-1`, *not in the deck*, beside a block whose `used-atoms 9` replaces it | true to the log: SIESTA looks the key up, then the block replaces it. Here both mean all the lead's atoms |
| F2610-a | **closed 2.67.2**, the Run panel's word; `cores` in the monitor's file is that file's key, and `48c` on a benchmark card its short spelling | the Run panel's and the monitor's `cores` is the node's logical count (40 here), beside the machine record's physical 20: nothing is computed from it | the word |

### Found by 2.65.9's review rounds (2026-11-04)

F5's second release, doors I–M of its design, built by hand, then read in two rounds, each a correctness reader (real
runs on both engines against 2.65.8) and one on the user's rules. The user's answers: "Rewrite the script", told the cost;
and the test every finding here was judged by: "what actually goes wrong, and does it matter?".

| id | release | what | the door / the answer |
|---|---|---|---|
| R269-1 | **closed 2.65.9** | every version move refused a launched PySCF vibration when prepared again: its script's line naming the release counted as a difference. Measured on 2.65.7 → 2.65.8, before this release's rewrite | the line is a stamp (`deck_record.BUILD_NAME`, masked by `without_stamps`); a real change is still refused (measured) |
| R269-2 | **closed 2.65.9** (before it shipped) | door I half built: the monitor still typed `[UTIL-SUMMARY]` and `[UTIL-BASIS]` where its reader asked `monitor_log` | the writer asks it too |
| R269-3 | **closed 2.65.9** (before it shipped) | a rename (`record._read`) missed a lazy import in `parse/dirs/setup.py`: six tests failed, the Results record would have | the import; every removed name swept, lazy ones included |
| R269-4 | **closed 2.65.9** (before it shipped) | the PySCF script still typed its equilibrium block, the probe's record and the full thermochemistry, where the user had been told it would build its records through the shared code | `equilibrium_block`, `ModeElectronicStructure`, `normal_modes.thermo_block`; every key of `relaxation` from the first write |
| D269 | **closed 2.65.9** | from the rules readings: a new test file whose rows were an invented race and a planted defect, retired for a writer-to-reader row in `tests/parse/test_instruments.py`; machinery the failure did not ask for (`_spaced`'s tolerance no writer uses, `TRAVELS_WITH = ()`, `say`'s aliases); the csv header moved away from its row, moved back; the contract's sentences this release made false, swept | the rows and sentences named |
| K269 | kept | written twice, or read a second way, and nothing goes wrong: the targets rule (`MolwatchReader.targets` knows the stage, `keys.targets_of` does not); PySCF's header read and `atom_index._of_log` (contract § 4a: *the pass or the table*; they differ only on a line no writer writes — both read by the one pass since **2.69.30**, X-PARSE); the two NetCDF checks (they accept different files on purpose); the species index (already through `read_fdf`); a SIESTA relaxation's frames (no gap on 13 real outputs); `.scf-timing.log`'s line layout; three helpers building a pattern from a template | the user's test, 2026-11-04. Told |
| K269-2 | kept | the PySCF script's `relaxation` block stays typed | the script fills it step by step, and no shared record holds it: `SpectraResults.relaxation` is a plain dict |
| K269-3 | kept, told | a PySCF vibration launched before 2.65.9 is refused when prepared again, and stays so until the way back is taken; launched again without preparing, it runs its old script | the user, 2026-11-04: "Rewrite the script" |
| F269-a | → F9 | typed copies of the monitor's words in tests (`test_util_csv_is_time_weighted.py`, `test_monitor_allocation_basis.py`) | F9's inputs through the road |

### Found by 2.65.8's review rounds (2026-11-04)

F5's first release, doors A–H of its design, built by three fixers on a brief that forbade new doors and rules, then read
in two rounds, each with a reader on the user's rules (the reports `r268-1-*`, `r268-2-*`). The user's answers: "Three
releases"; "retire selection_rules"; "Compare by meaning"; "i am ok with data for ncdata"; and the JSON attributes' list:
"if that key is so critical then the developer should keep that on the record of the list. otherwise it's a
complemntary and use should know that's not part of contract and it's a customized text".

| id | release | what | the door / the answer |
|---|---|---|---|
| R268-1 | **closed 2.65.8** (before it shipped) | a byte-order mark refused in records and read past in the person's files: two rules for one thing, and against the job request's chapter | one rule: `persist.read_json` reads past it |
| R268-2 | **closed 2.65.8** (before it shipped) | `selection_rules` retired with no version move, against the sidecar's own rule; the reason given was never measured | `SCHEMA_VERSION` 17; a PySCF deck's structure line compared by what it holds, the version and the retired keys set aside (`pyscf/script_input._as_read`), so a deck written before this release is the same calculation |
| R268-3 | **closed 2.65.8** (before it shipped) | the read invented a `created_at` for a file that stated none | the writer stamps (`molstruct.stamp`, passed by `StructureCodec.pair`); `to_dict` records what it is given |
| R268-4 | **closed 2.65.8** (before it shipped) | the door's first pass: the run script's two writes, `describe`'s staging, the result tags typed at their callers, spectra's re-exports | door C finished; `build_sidecar_result(kind=)`; the re-exports gone |
| D268 | **closed 2.65.8** | from the rules readings: a sentence inviting hand edits of every JSON file, against the user's rule and with two files' reason; a `task.json` or `job-request.json` a person broke refused as a raw decode error; a `.molstruct.json` not in UTF-8 refused as a raw decode error; a sidecar sample with a value no molbuilder wrote; a § 2.3 row's reason untrue | job-contracts § 6.1, the reason said for the files it covers; `task.of`, `job_request.of`, `molstruct.load` name the file; the sample `{}`; the row's reason |
| D268-2 | **closed 2.65.8** | from the correctness reading: the sidecar's chapter still said v16 in its example and its § 2, with the reader's set without 17; a `task.json` or `job-request.json` an editor saved in its own code page was called *not JSON*, with no way out; `prepared_with.read` read past the one reader | the chapter; both doors say *not UTF-8 text … save it as UTF-8*; `_stages_in(read_json, path)` |
| K268 | kept | `JSON_ATTRS` beside the writer's `json_attr` calls | not a defect: **the list is the contract** (the user, 2026-11-04; `model/datafiles.md` § 2). A name on it is a record the file promises; any other attribute is text and promises nothing. My *"spelled twice"* finding named no failure, and the deferral to 2.65.9 is dropped |
| K268-2 | kept | a pair saved by 2.65.8 is refused by a 2.65.7 install | every version move does this; the sidecar's rule asks for the move |
| K268-3 | kept | the sidecar's `created_at` and the run record's stamp, two clocks | nothing compares the sidecar's stamp with anything: nothing goes wrong |
| F268-a | → F7 | names no road uses: `selection.to_json`, `parse_spectra_json_dict`, `molstruct.save` and `with_lock` (tests only), `FirstN` sent by no page, whose two tests stay until then — **closed 2.65.17**: `parse_spectra_json_dict`, `molstruct.save`, `with_lock` and its `.lock` row deleted, a second reader and a second writer; `selection.to_json` and `FirstN` kept for a person's scripts (the user, 2026-11-05) | F7's dead-code pass, each read first |
| F268-b | → F7 | two `ParseWarning` classes (`frame.py`, `parse/types.py`) — **left 2.65.17**: they differ on purpose, and `parse.types` imports `frame`, so `frame` cannot take the typed one from below | F7, read first |
| F268-c | **left 2.65.16** — each writer its record's owner, the bytes one; the channels file read once, said to two audiences | `runpoints.write_index` is a sixth writer choosing a plan or the disk, and three reads choose the same way (`JobSet.of`, `prepared_with.record`, `compose`'s provenance read); the channels file judged twice | F7's objects. Nothing goes wrong on a road: those reads differ from `read_json` only on a byte-order mark in a record molbuilder wrote, and no road edits one by hand |
| F268-d | **closed 2.65.11** — `persist.unread_words`, at every file a person edits: description, job request, machine config, channels, report cards; `check_schema` in plain words. Left: the sidecar readers, `sweep_reader`, `scheduler/record.why_unread`, which read what molbuilder itself writes and say so in their own sentence | the refusal words of `persist.check_schema` and the report-cards copy's *minor versions tolerated*; a file not UTF-8 or not JSON said in six wordings across the readers | F6's words |

## F5 — § 1am step 5 — one codec per format (25)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B2-23 | door | our JSON records written in a byte shape of their own beside `persist.json_text` | `persist.json_text` / `write_json` at every site, `json_text` taking the sidecar's reason (molstruct.py:654-659: `ensure_ascii=False`, a non-ASCII label kept literal) for every record; the describe docstring corrected w… — **closed 2.65.8** — `persist.json_text`, the characters as themselves, a non-finite number `null` (or refused, `refuse_non_finite=`, for the structure sidecar and the spectra record); every record through it (`execution/job-contracts.md` § 6.1) | confirmed |
| B2-24 | door | our JSON records read raw beside `persist.read_json` | `persist.read_json` (+ `check_schema`) — R3's two loaders one function — **closed 2.65.8** — `persist.read_json` at every site; `parse.sidecars._helpers.read_record` the transport and sweep records' one loader; `toc.json` read once per request | partial: PARTIAL \| parse/sidecars/transport.py:30 and fc_sweep.py:33 `payload = json.loads(Path(path).read_text(encoding="utf-8"))` — the same `_load` twice; notify_setup.py:96; docs.py:140, :391 (toc.json t… |
| B2-25 | door | writer and reader of one of our formats each spell its vocabulary | one constant set each — `trajectory_log/keys.py` for the header lines, `MONITOR_ENDED` imported by the instrument parser, the `util.csv` columns and the peak's source words beside their writers, imported by the readers — **closed 2.65.9** — the progress log's words in `trajectory_log.keys`, the writer rendering them and `molwatch_grammar`'s patterns built on them; the SIESTA deck's `# runtime.` lines through them; the monitor's two files in `monitor_log`, the memory peak's words with them | confirmed |
| B2-26 | door | one rule or constant of a reading kept twice | each through its one home — `record._HEAD`, `persist.sha256_of`, `keys.targets_of`, `MolwatchReader`, one magic in `ncdata`, `types.SCHEMA_VERSION` — **closed 2.65.9, in part, the rest kept** — one read window (`record.SETUP_HEAD`, `read_part`), a deck's digest by its bytes and one digest of a file (`persist.sha256_of`, chunked; checkpoint's own gone), `types.SCHEMA_VERSION`. Kept (the user's test, 2026-11-04: nothing goes wrong, K269): the targets rule, PySCF's header read and `atom_index._of_log` (both on the one pass since **2.69.30**), the two NetCDF checks | confirmed |
| B2-27 | door | parse result schema tags typed by hand, two of them stale | the tag from the payload's `schema` (`persist.schema_name` / `schema_major`) — **closed 2.65.8** — `build_sidecar_result(kind=)` composes the tag from the payload's own version (`model/parse.md` § 6); `RESULT_SCHEMA` gone from the two readers | partial: PARTIAL \| parse/sidecars/transport.py:76 `schema="transport/v1"`, fc_sweep.py:77 `"fc-sweep/v1"`, run_points.py:57 `"run-points/v1"`, notified.py:56 `"run-notified/v1"`, jobset/sweep_reader.py:37 `R… |
| B2-28 | door | the SIESTA vibration's own steps read SIESTA's files a second way | `read_output(path).finish()`; `deck_atoms` (with the species' index it does not return yet); one `wrapper_log.say` — **closed 2.65.9, in part, the rest kept** — `read_output(path).finish()` at every read, one `wrapper_log.say`. Kept (K269): the species index, already through `read_fdf`; the relaxation's frames | confirmed |
| B2-30 | door | the structure-pair sidecar's envelope built on both sides, and renewed through a private name | one envelope builder in `sidecars.molstruct`, its renewal a public door — **closed 2.65.8** — `sidecars.molstruct.to_dict` the one builder, the reader asking it with the version the file states; `renewed` and `stamp` public | confirmed |
| B2-31 | door | which timing source a run's figures are read from, decided three times | one `timing_source(files)` beside `scf_timing_rows.timing_of` — **closed 2.64.0** — `scf_timing_rows.timing_file` | confirmed |
| B2-32 | door | a NetCDF data file's JSON attribute has one writer and no reader | `ncdata.unjson_attr` beside `json_attr`, asked by every reader — **closed 2.65.8** — `ncdata.unjson_attr`; the spectra, optical and Raman decodes ask it; `transport/datafile.read_data_file` hands over the list's attributes as data (`model/datafiles.md` § 2: the list is the contract) | confirmed |
| B2-33 | door | jobset/submit.py:2819 — the run's index written beside its owner's writer | `runpoints.write_index(path, index, plan=)`, as `JobSet.write(folder, plan=)` does — **closed 2.65.8** — `runpoints.write_index(path, index, *, plan=None)`, launch asks it | confirmed |
| B2-34 | door | the pidfile written twice, its name composed three times | `serve_daemon.write_pidfile` through `persist.write_bytes`; one composer in config_dir — **closed 2.65.8** — `serve_daemon.write_pidfile` through `persist.write_bytes`; `config_dir.pidfile_pattern`, `_pidfile`, `ports_with_pidfile(family)` the name's one home | partial: PARTIAL \| serve_daemon.py:393 `serve_pidfile(port).write_text(f"{os.getpid()}\n")`; jupyter.py:820 `jupyter_pidfile(serve_port).write_text(f"{os.getpid()}\n")` — two writers, neither atomic; one rea… |
| B2-35 | door | an engine's or a tool's file read outside `parse/` | each through its reader in `parse/` (the ORB_INDX and contour readers, `parse.fdf`, `Structure.from_pdb`); the conda and machine facts one reader each (`envs` and the travelling monitor's — X76's split is the stdlib-onl… — **closed 2.65.10, in part, the rest kept** — the device run's `.ORB_INDX` by `tbtrans_nc.read_orbitals`, its catalogue row naming it; a transmission deck's window through `parse.fdf`; the tools' PDB by `Structure.from_pdb`. the node's total memory one read (`runtime_info.mem_total_gb`). A departure from door N, told: kept (the user's test, 2026-11-04: nothing goes wrong, K2610) the energy blocks' text swap, the finished-deck check, the two conda readings, the machine's other facts | confirmed |
| U1-5 | wrong-result | Chemistry card › "Method DFT — stated" for a method nobody stated | `Resolved.said` for a template-sourced item takes the template row's `source` through `template.SOURCE_WORDS` — "not chosen" for the blank method, "from the junction you cited" for a cited spin — the analyze body alread… — **moved to F6** — words a page says, no file format — **closed 2.65.11** — gone already: a blank form says *default: not chosen*, a stated one *stated* (measured) | partial: PARTIAL \| Method part TRUE: blueprints/_shared.py:886-917 `config_from_params` lays a blank over `apply_recommended` (the kind's DFT); electronic_state.py:283-292 `_method` → `Resolved(cfg.method, "… |
| U1-18 | words | Spectrum › The calculation › "Structure is already relaxed" › engine names as tokens and an unrounded tolerance | sidecar.py says `engine_table.engine(x).words`; the tolerance through the one number formatter (3 significant figures) — **moved to F6** — words a page says, no file format — **closed 2.65.11** — the record's sentences name the engine by its words, and the tolerance through the one number formatter | confirmed |
| I1-016 | **closed 2.67.3, left**: it reads the page's own controls, and the server reads the values again and is the one that refuses; stricter at most (a count with a fraction refused first), never another value | the parameter forms' own field reader `collectField` | **moved to F8** — page code; two contract sentences disagree, asked then | confirmed |
| I1-044 | inventory | one reader and one writer per format | **closed 2.65.10** — every format of ours one writer and one reader, or left and said: each "no" of its list closed by B2-25, B2-26, B2-28, B2-30, B2-32, B2-33, I1-059; `.scf-timing.log`'s layout kept (K269); its example I1-016 moved to F8 | confirmed |
| I1-059 | inventory | the PySCF deck writes its mode records and removed-motions block as text beside the shared builder | **closed 2.65.9** — the PySCF script builds its result through the shared record: `SpectraResults`, `ModeData`, `ModeElectronicStructure`, `equilibrium_block`, `removed_motions_of`, `thermo_block` (the user, 2026-11-04: "Rewrite the script"); the relaxation's block stays the script's, filled step by step, no shared record holding it | confirmed |
| I1-151 | inventory | the fdf-log reader keeps a lookup's `# above item originally` line as a row of `%block LatticeVectors` | **closed 2.65.10** — a block's rows are the lines libfdf writes with one leading space; a line at column 0 inside a block is a lookup, and reaches `params` (`model/parse.md` § 5d.3) | confirmed |
| I2-041 | inventory | PySCF's `ir_fd_step_ang` written only on the finite-difference route | **closed 2.65.9** — by I1-059: every key from the first write (`engines/vibration.md` § 6.5) | confirmed |
| I2-072 | inventory | a second PDB reader (`Mg → M` on a blank element column) | **closed 2.65.10** — one PDB reader, `Structure.from_pdb`; a text with no name by `StructureCodec.format_of_text`, a name by `format_named` | unchecked |
| I2-079 | inventory | residue: `selection_rules`, `load_text` twice, shim classes | **closed 2.65.8** — `selection_rules` retired (the user, 2026-11-03: "retire selection_rules"), the `load_text` re-export gone; the "three legacy shim classes" the row carried from 2026-09-22 were never named, and a sweep finds none (the one candidate, two `ParseWarning` classes: F268-b above) | unchecked |
| I2-105 | inventory | `Environment.from_dict` drops unknown keys | **closed** — the user's answer at door Q, "The four only" (K267): this reader stays as it reads | unchecked |
| I2-159 | **closed 2.67.3**: `xyz`, `elements`, `n_residues` gone, no page read them; the four per-atom columns stay, the viewer folds them in | legacy response keys | **moved to F8** — page code | unchecked |
| I2-207 | inventory | one atomic writer for every file molbuilder writes | **closed 2.65.8** — every whole file through `persist.write_bytes` or a `Plan`; what stays outside is a row of `configuration.md` § 2.3 | unchecked |
| I2-278 | inventory | one instruments reader | not re-checked — **closed 2.65.9** — `parse.instruments.metrics_of`, the record and the bench summary asking it; the record's catch-all gone | unsure: UNSURE \| `parse/instruments/` holds the readers (`monitor.py`, `reports.py`, `scf_timing.py`, `util_csv.py`, `utilisation.py`), registered at import (`__init__.py:13-16`) \| what would tell: the 202… |

## F6 — § 1am step 6 — one home for each vocabulary, default and rule (45)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B3-9 | door | web/blueprints/results.py:128–136 — `/api/results/contract` composes a run's view by hand beside `runs.view_of` | `runs.view_of(run, output=output).info` — **closed 2.65.13** — the route asks `runs.view_of` | confirmed |
| B3-17 | residue | the run states — one home with no names, so every reader spells them; NEW sets written in 2.54.0–2.56.0 | a name per state beside `RUN_STATES` (as `NOT_RUN`), and the sets `LIVE_STATES` / a "not ended" set read from there; `OPTICAL_STATES` from them — **closed 2.65.13** — the sets by name, `job.LIVE_STATES` and `NOT_ENDED_STATES`, at every site that spelled them. A name per state left: about 70 sites, no failure, and `RunStatus` refuses a wrong one on the writer's side | partial: PARTIAL (substance CONFIRMED; two NEW datings wrong) \| home parse/dirs/job.py:381-393 `RUN_STATES … ("pending", "queued", "running", "finished", "failed", _re.NOT_RUN, "unreadable")`, `LIVE_STATES =… |
| B3-18 | residue | the job kinds `task` / `bench` / `sweep` spelled beside their names | `identity.TASK` / `BENCH`, `model.KIND_SWEEP` — **left 2.65.13** — the words of a verb's grammar, agreeing, nothing goes wrong | confirmed |
| B3-19 | residue | a job's request words and its GPU line spelled in each door | one field-words table beside `Resources` (P16's NONE-EXISTS), `canonical_gres` for the line — **left 2.65.13** — a request's words and its GPU line differ by what each door is saying; nothing goes wrong | confirmed |
| B3-20 | residue | defaults restated where their home states them | no kind default — the caller's `task.calculation`; the catalogue's `default_of` for the temperature, the solvent method and the pressure; the k-grid words from `TRANSMISSION_PER_SCF` — **closed 2.65.13** — pressure and the identity's placeholder through `default_of`. The 300 K of a deck that states none and the *three times the shared grid* sentences left: the first is the engine's default, not ours; the second one source for the number | confirmed |
| B3-21 | residue | a kind's or an engine's vocabulary restated outside its home | each through its home: `kind_table.COMPUTES` (beside the job, as `runpoints.part` resolves words), `SPIN_CHANNELS`, one axis-kind tuple, `PARTITIONS` with its words, the catalogue's choices and `active_when`, `ase.data`… — **closed 2.65.13** — a recorded treatment and count asked of the catalogue's `choices`; geomeTRIC's keywords its `anchor`; `SPIN_CHANNELS`, `PRODUCT`, `SOURCE_DIPOLE`, `ORIENTATION_FREE`, `FROZEN_LABEL`, `LISTED_*`, `AXIS_KINDS`. Left: partitions (facts keyed by the word), the periodic table (a fixed list), the DFT-only lines of PySCF's own modules | confirmed |
| B3-22 | words | one sentence composed in two or more places | `identity.migrate_as_typed`; a `commands` composer for the stop line; `config_dir.is_channel_name`'s sentence; `LOCAL_TARGET`'s label; `Readiness.detail` asked by status — **closed 2.65.11, in part** — a channel name's sentence, `config_dir.CHANNEL_NAME_RULE`, with its limit, at all five doors; the rest 2.65.13 — **closed 2.65.13** — with 2.65.11's channel name: `LOCAL_LABEL` and one translator, `ready_words`; the migrate hint and the `jobs stop` line were one already | confirmed |
| B3-23 | residue | the machine's and the environments' vocabulary spelled per site | each through its owner's constant (`_LAUNCH_MODES` made public, `JOB_VARIABLES`, `_OAUTH_KINDS`, one recipe version, `OUTPUT_LIMIT`, `config_dir.runtime_dir`) — **closed 2.65.13** — `OAUTH_KINDS` at its three sites, `DEFAULT_ENV_NAMES`, `OUTPUT_LIMIT`. The rest of the row left: launch modes (one travels beside the job), `MB_*` (kept to a list by review), a compiler's facts | partial: PARTIAL \| Launch modes: runtime_config.py:653-654 `#: The one key `launch` holds, and the two values it takes.` `_LAUNCH_MODES = ("direct", "submit")`; spelled at parse/dirs/job.py:191, :276, :296 `… |
| B3-24 | residue | smaller vocabularies with two homes | through each home (`_TOKEN_RE`, `projects`' topic, `SUPPORTED_FCC_PLANES`, a public channel-kinds name) — **left 2.65.13** — every row agrees; `pseudopotential` as a topic and as the library's name could be one name, and nothing goes wrong until a folder is renamed that the contract fixes | partial: PARTIAL (FCC planes not a restatement as stated) \| identity.py:258 `m = re.fullmatch(r"\d{2,}_([A-Za-z0-9_]+)", str(token))` beside :408 `_TOKEN_RE = re.compile(r"(\d{2,})_([A-Za-z0-9_]+)")` and :41… |
| B3-35 | residue | NEW (2.55.0, 2.56.0) — the count's tallies (done, failed, not run, not done) written in both kinds' count parts | one tally over rows in `runpoints` (beside `run_state`), each kind's `count` part wording it — **closed 2.65.13** — `runpoints.tally`, each kind wording it | confirmed |
| B3-37 | door | the web routes compose what their owners already compose — 3 NEW since 2.54.0 | the transport description built in `transport/` and asked by both doors (W2's NONE-EXISTS); `window` carried by `Extension.to_wire`; `runfiles.manifest`; each route asking its owner — **left 2.65.13** — the four that mattered gone already; the file row beside `manifest` and the state's tag row agree, a new key would be seen on the page | confirmed |
| B3-38 | residue | web/blueprints/files.py — one rule written in each route of the file browser | one helper per rule inside `files.py` (name, pair move, free name), `projects.create_project_skeleton`, `write_script`, `sidecars.molstruct` for the pair; `missing_ok` deleted — **closed 2.65.13** — `_replace_pair` for rename and move, `_next_free_name`, `_is_basename`, the depth through `_rel_parts_inside_root`. Left: copy, whose failure removes a half-copy where move's puts it back; rename taking a space where upload and mkdir refuse one, a real difference the contract does not rule on | confirmed |
| B3-39 | residue | the science and structure rules written twice (re-measured; none changed since 2026-10-29) | as the 2026-10-29 verifier named each door (`cell.check`/`ResolvedCell`, a `siesta/input` orbital estimate, `_stage_science`, `chemistry._adjacency`, `Structure` subset door, one lead collector in `transport/sort`, `war… — **left 2.65.13** — none gives a wrong answer: held-index wordings, a mode by number, stationarity, atom-order rows differ on purpose; the low ones (rigid-motions sentence, warm gate, `10 * n_atoms`, lead atoms, adjacency cut-offs) agree | confirmed |
| B3-40 | door | two readers of *does the scheduler still list this job* answer differently | `_slurm_standing` asks the one *listed* rule; the line composed once in `queue` — **closed 2.65.11** — `_slurm_listed` and `_slurm_standing` ask one `_slurm_any_standing`; a state with its reason composed once, `queue.state_said`, `runs.standing_of` asking it | confirmed |
| B3-41 | residue | the command line, the server daemon and the environments' tools each write their rules again (none of these files changed since 2026-10-29) | as the verifier named each (`require_activation`, one probe assembly, `_SECTIONS`, `child_exit_action`, `format_named`, `runtime_dir`, `relative_home`, `BuildPaths`, …) — **closed 2.65.13, in part** — `serve foreground` asks `child_exit_action`; `envs` one unknown-recipe refusal. The seeded record's probe left: two callers of the same two doors, in the same order, with nothing between them to merge. The rest of the row left as C-rules lists | confirmed |
| B3-42 | residue | the transport kind's rules written twice — one NEW copy in the 2.56.0 walk | as the verifier named each (`energies.same_grid`, `electrode_orientation`, `_restart_group_lines`, `Task.treatment`, `frame_token`, `not_transport`, `energies.spacing`, `fixed_rungs`, `rung_points_of`); the walk's start… — **closed 2.65.13** — `energies.same_grid` in `record` and `tbtnc`, `submit._walked_start`. `run_words`' exact comparison left: one run's printed columns, at 10⁻⁵ eV, where the tolerance and exact are the same test | partial: PARTIAL (10 of 12 sites as stated; the energy-grid site's "answer otherwise" and the `DM.UseSaveDM` door do not hold) \| **(L4-new, NEW)** `_points_walk` read whole (jobset/submit.py:2515-2846): :266… |
| B3-43 | residue | prep, the groups, the description's readers and status write their rules twice | as the verifier named each (one note composer in `group`, one envelope, `stage_config`, a reader-refusal helper in `persist`, the constructors, a run-records set in `runrecord`, `empty_run`, `verdict_words`, `warm_decla… — **closed 2.65.13** — `commands.copied_in_note`/`names_copy_of`, `gone` through `empty_run`. Left: *could not be saved* (four sentences, each what was not done), the builds-on refusal (two moments), three `MachineTranslation` | confirmed |
| D6-6 | words | "bundle" is retired for the calculation folder in the documents, and is the option every verb takes | one of the two, by the user's word — **closed 2.65.11** — *(the user, 2026-11-04: "Keep --bundle")*: the option every verb takes stays; the three stray sentences say *calculation* | confirmed |
| U2-7 | words | Run panel › Setup, a SIESTA vibration run (siesta-vib run-0) — the `already_relaxed` row names PySCF's relaxer and a Python function, the `temperature_K` row a Python call | the two `expands` say what the person reads ("the relaxation before the force constants", "the thermochemistry's temperature"), engine-neutral. — **closed 2.65.11** — the two `expands` say *the relaxation before the force constants*, *the thermochemistry's temperature* | confirmed |
| U2-18 | words | Task setup › a PySCF calculation's Preview refusal names a row the run card does not show by that name | the refusal names the row by the catalogue item's own label ("CPU threads" / "MPI ranks (np)"), read from the item, beside the key. — **closed 2.65.11** — the refusal names the run card's row by the item's label, beside the key | confirmed |
| I1-019 | inventory | the `run_card` marker in the catalogue, kept while `jobset migrate` reads 2.27.x templates | **closed 2.65.13** — the marker gone: a run setting is `template.is_run_setting`, and a new one draws its row | confirmed |
| I1-038 | inventory | the Structure-optimization tab's two locks in its page script (step settings under *Relaxation type* none; solvent model with no solvent) | **closed 2.65.11** — gone already: no page lock left, the catalogue's `active_when` since 2.65.0 | confirmed |
| I1-045 | inventory | one home for each vocabulary, default and rule | **closed 2.65.13** — this row's whole: F6's three releases, each word, default and rule one home where a copy could go wrong silently, the rest left and said | confirmed |
| I1-061 | inventory | the relaunch rule's words composed twice | **closed 2.65.11** — one sentence, `commands.relaunch_words`, asked by prep and `status`; prep now says *takes over every point* for a run of points | confirmed |
| I1-074 | inventory | `kind_table.FINISHED` spells `runpoints.DONE`/`RAN` again | **left 2.65.13** — a kind's table beside a travelling module, agreeing | confirmed |
| I1-075 | inventory | a point's state read twice by `status` | **closed 2.65.11** — gone already: `continuation.point_rows` reads `concluded` once and hands it over | confirmed |
| I1-128 | inventory | prep's *nothing to write* split plans the rest anew: the machine record checked again, every member re-planned, each deck's findings said twice | **left 2.65.11** — the terminal prints each finding once; the second call of the plan changes nothing a person sees | confirmed |
| I1-149 | inventory | two renderers of server words with backticks shown as code | **closed 2.65.11, in part** — `dom.codeWords`, one renderer: Task setup, the file card, the Results ladder's steps and details. A finding's message keeps its backticks as typed, a departure from the design told to the user: a stray backtick, nothing misread, and every surface's findings harness would change for it | confirmed |
| I2-059 | inventory | one home for "is this a structure path"; rename strands labels | **closed 2.65.13** — `molstruct.is_geometry`, whatever the suffix's case; a rename that strands a label left: the person's own act, undone by renaming back | confirmed |
| I2-061 | inventory | dead `axis_kind or ("isolated",)*3` fallbacks | **closed 2.65.11** — every `axis_kind or ...`, and the checks of it for `None`, gone: `Structure.__post_init__` makes `None` impossible. `kmesh.mesh_for`'s stays, whose argument is not a structure's | unchecked |
| I2-062 | inventory | the partial-charge estimate's unknown-element fallback is hydrogen's value | **left 2.65.13** — a hydrogen value for an unknown element in a charge estimate: the species check refuses such a label first | confirmed |
| I2-063 | inventory | the k-point vacuum hint measures the gap along a lattice vector, not perpendicular | **closed 2.65.11** — the atoms' span along each lattice vector, as a share of it; a skewed filled slab draws no hint. Measured along the vector, as the row's verifier prescribed, not perpendicular | confirmed |
| I2-069 | inventory | a negative vacuum accepted by the model, refused by the gate | **left 2.65.13** — the gate refuses it before anything runs | confirmed |
| I2-083 | inventory | two "is this a calculation root" checks on different evidence | **closed 2.65.11** — gone already: `checkpoint` asks `calcdirs.described` | confirmed |
| I2-104 | inventory | the retired key refused in three doors | **closed 2.65.11** — gone already: one refusal is left, the `_SECTIONS` row in `runtime_config` | unchecked |
| I2-106 | inventory | functions defaulting their kind to `"optimization"` | **in part 2.64.0** — 13 found; the MIGRATE_HINT half done earlier, and the settings check now takes the kind from its caller, never `"optimization"`; the rest F3-R-2 — **closed 2.65.11** — gone already, with B3-20's first row | unchecked |
| I2-119 | inventory | `exclusive`, a request field nothing sets | **left 2.65.13** — a request field nothing sets, harmless | unchecked |
| I2-120 | inventory | the named-queue check in two doors | **left 2.65.13** — two doors, two questions: a queue named at a header's write and at its placing | unchecked |
| I2-121 | inventory | one `proxy.trust` setting, ruled 2026-10-02, not built | **moved to AFTER-FEATURE** *(the user, 2026-11-04: "Leave it for now")*: nothing goes wrong behind the one proxy we run | confirmed |
| I2-122 | inventory | a benchmark's grid the machine proposes, against the 2026-10-02 ruling | **closed 2.65.12** — *(the user, 2026-11-04: "Refuse, you state it")*: a benchmark states its ranks and cores per rank, on the GPU its device counts too, or is refused naming each and where; `bench/grid.py` gone; every cell kept with its fit, only one past this prep's own ask left out | unchecked |
| I2-139 | inventory | `read_by` declared and read by nothing | **left 2.65.11** — `template.md` § 6.1 keeps the declaration on purpose, and says why | unchecked |
| I2-140 | inventory | `kind = "monitor"` has no items | **closed 2.65.11** — the member retired, `template.md` with it | confirmed |
| I2-198 | inventory | the electronic state resolved at many call sites per prep | **left 2.65.13** — a cost, not a wrong answer; one prep's calls agree | unchecked |
| I2-211 | inventory | engine facts restated in prose; no e2e comparing the table with a run's fdf log | **left 2.65.13** — engine facts in prose: words for a person | unchecked |
| I2-277 | inventory | `done` returning its `RunStatus` | **left 2.65.13** — nothing goes wrong | unchecked |

## F7 — § 1am step 7 — the floors and the objects (26)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B3-13 | door | modules holding two floors' parts — a deck or schema fact kept beside a higher floor's composition, so lower floors import the higher module | **closed 2.65.15** — each fact under its reader: two schema names at their readers, `spin_of` in `conductance`, the window rule in `energies`, `SIDECAR_NAME` in `script_input`, `unit_word` in `engine_input`, `BLOCK_SHARED` from `deck_record`. `gf_files` and `energies`' own composition left: each module's top loads nothing to speak of | confirmed |
| B3-14 | door | `script_emit` — its doors are listed on floor 3, and floors 1 and 2 import it | **closed 2.65.15** — the provenance and parameters readers in `deck_record`, the predicate in `structure` (a bundle holding `deck_record` has no `structure`), the one-line wrapper deleted; `declarations`, `parameter`, `deck_values` stay: some 200 lines with 30 callers on floor 3, for one reader that never runs beside a job | partial: PARTIAL \| parse/contract.py:432 `from molbuilder.script_emit import read_provenance_block`; parse/dirs/setup.py:151 `from ... import script_emit as _sc`; runs.py:380 `atom_metadata_has_something`, :… |
| B3-25 | door | an object taken apart, or a record its caller holds read again | **closed 2.65.16** — `_resolve_stage` takes the description read once, `cannot_be_named` the calculation; `render_sbatch` and `points_in` already took theirs whole | confirmed |
| B3-26 | door | a private name reached across packages — 2 NEW in 2.54.1, 7 earlier still open | **left 2.65.16** (decided at F7's design: nothing goes wrong, and a function-level import fails the same way when public): each owner's private name stays, the plan's *made doors or left alone* | confirmed |
| B3-27 | residue | a function that only renames another, or code nothing runs — **closed 2.65.17, after the audit** ([`archive/audits/2026-11-05-2.65.17-removals.md`](?doc=archive/audits/2026-11-05-2.65.17-removals.md)): the renames and what no road reached deleted, each by its role and history; what was a guard, an extension or a person's script door kept or brought back | callers call the door; the dead names deleted (their tests with them) | confirmed |
| B3-32 | door | floor 1 reaches the run door through the structure codec; two engine-owned readers sit at shared roots | **left 2.65.15, named in A7** — the coords readers go through the one door of a structure pair on purpose; registering them from the surfaces buys nothing and any caller below a surface would stop finding them. Nothing goes wrong. The PDB reader is the separate later plan; *R37* has no text anywhere to judge | partial: PARTIAL \| parse/coords/pdb.py:94 and parse/coords/pyscf_geom.py:33 (in functions) `from molbuilder.workingcopy_structure import StructureCodec`, then `.load(p)`; `load` reaches the run door: working… |
| D2-SCH-4 | residue | scheduler.md § 6a (:747, "the whole node `--exclusive` `--mem=0`") — a header directive no road can ask for | delete `Resources.exclusive`, the `whole_node` words and their rendering, or name the road that states it — **left 2.65.14** — *(the user, 2026-11-04: "Leave it")*: nothing goes wrong, and removing it would refuse every prepared folder once | confirmed |
| D5-10 | residue | parse.md §§ 4, 5c — `instruments/reports.py`, a registered parser no chapter section states, and filed against § 5c's own rule — **closed 2.65.17**: the rows in § 4 and § 5c, no move | state it in § 4/§ 5c, or move it under sidecars/ as the chapter's rule says; either way one row naming its reader (`GET /api/notify/reports`). | partial: PARTIAL \| parse/instruments/__init__.py:15 `register(ReportsFileParser)`; reports.py:1-9 reads the listener's `<state dir>/reports/<user>.jsonl` "as `web/blueprints/notify.py` writes it"; absent fro… |
| D5-11 | residue | parse.md § 1 — `FileParser.keeps_atom_index_map` and `footgun_hint_for` not stated in the ABC section — **closed 2.65.17**: a sentence each in § 1 | one sentence each in § 1. | partial: PARTIAL \| parse/base.py:30-39 `keeps_atom_index_map: bool = False`, :41 `footgun_hint_for`; neither name occurs in parse.md; § 1 (:64-66) lists name/label/output/hint \| NOT residue: both serve cont… |
| D6-7 | door | architecture.md § 3 — an L1 module imports an L2 package at its top | **closed 2.65.15**, the opposite way to the row: the grammar moved into `selection`, and `spectra/selection` keeps `select_modes` alone, because it travels and imports nothing | confirmed |
| D6-C6 | residue | § 2.3 "that list is the rule. A write that is not on it is the finding" — 23 raw whole-file writes outside persist.py on no list (script_emit.py:1505, :1956, jupyter.py:699, :820, envs/builds.py ×5, …). § 1am step 5, M, no. known: K13. — **closed 2.65.17**: `configuration.md` § 2.3's list says them |  | partial: PARTIAL \| docs/configuration.md:440-446 (§ 2.3) "`persist.write_bytes(target, data, mode=…)` replaces a whole file this package means to keep … A handful of shapes legitimately sit outside it — **ea… |
| I2-011 | inventory | deck-block readers live in `script_emit` (a writer, no floor) and are asked by floors 1–2 | **closed 2.65.15** with B3-14 | confirmed |
| I2-056 | inventory | a lead's `Structure` rebuilt from five fields, dropping annotations, info, identity, customized — **left 2.65.17**: a lead's deck and gate read none of them, measured; the docstring corrected | OPEN | confirmed |
| I2-060 | inventory | `Structure.replace()` lists its fields by hand | **closed 2.65.14** — `_nonatom` derived from the fields: a new field travels unless it says otherwise | unchecked |
| I2-064 | inventory | the backbone check keys on `rid − 1` | OPEN | unchecked |
| I2-065 | inventory | rdkit/SMILES atoms stored with placeholder identity `(1, MOL, A)` and index names | OPEN | unchecked |
| I2-066 | inventory | AmberTools warns "B-form not enforced" on every build | OPEN | unchecked |
| I2-067 | inventory | `set_channel` installs, then validates | **closed 2.65.14** — validated before it is set; a refused channel leaves the one it would have replaced | unchecked |
| I2-068 | inventory | `concat` hand-lists the identity columns | OPEN | unchecked |
| I2-071 | inventory | `annotations` accepts a `str` index `regions` refuses — **closed 2.65.17**: one rule for labels and channels, `structure._atom_index` | OPEN | restatused: V9 § 7: **OPEN** (located) |
| I2-073 | inventory | metadata-dropping rebuilds — **closed 2.65.17**: the rest through `Structure.without` and `replace` | **closed 2.65.14, in part** — `relieve_clashes` moves the atoms and nothing else, on the real road where it renamed 178 of 262 atoms and renumbered a strand. The other rebuilds are handed nothing to lose on any road (measured): 2.65.17's, or left | unchecked |
| I2-074 | inventory | `write_description(struct=None)`, the second periodic table, `emit_atom_metadata`, the backend set in seven places, … — **closed 2.65.17**: `SYMBOL_TO_Z` deleted, the dead guards gone; the rest gone before it or left, nothing going wrong | OPEN (sample) | unchecked |
| I2-123 | inventory | absolute imports, package-wide | OPEN — 2,681 relative-import lines (2,282 on 2026-10-28); new modules still written relative | unchecked |
| I2-163 | inventory | `title` carried by hand — **left 2.65.17**: every road keeps its title, measured; the false count removed from `structure.md` | OPEN — 8 sites now | confirmed |
| I2-178 | inventory | design-frame exports: `to_ase`, `--pyscf-atom-block` — `--pyscf-atom-block` **left 2.65.17** | **closed 2.65.14, in part** — `to_ase` places the atoms in its box. `--pyscf-atom-block` left: it states no box | unchecked |
| I2-179 | inventory | the transport validation subject is the arriving structure, not the placed box; the lead's hand shift — **left 2.65.17**: the gate's box equals the deck's, measured; the false *still open* removed from `transport.md` | OPEN | unchecked |

## F8 — The pages — words and residue (103)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| D1-JSa-13 | → F10, a document's text | job-system.md:465 and :801 — two broken machine tokens in the text | `runstatus.jobset_status` on one line; R-L45's cell `submit._plan_sequence`, `submit._planned` (as :802 has it) — once each | confirmed |
| D1-JSb-12 | → F10, a document's text | four sentences in the range a reader cannot follow | "`scratch_dir`: the folder inside each run where the engine keeps its scratch (a device run's `tsgf/`)". `finish`: "the bundle whose steps run around the engine at each point, and finish the run (a SIESTA vibration's `m… | confirmed |
| D1-JSc-6 | **gone**, checked at 2.67.0: the words no longer stand | § 7 — "Detect the machine" listed as a step of the benchmark workflow, which reads the record and never probes | "**The machine's record → `environment.json`** — what `jobset probe` measured on it (… `scontrol show node` …), read by prep, never probed by it" | partial: PARTIAL \| doc :3089-3094 "**Detect the machine → `environment.json`** … resolved to the compute node's real core and GPU counts (read from the scheduler via `scontrol show node` …)"; flowchart :3082… |
| D1-PLb-9 | **gone**, checked at 2.67.0: the words no longer stand | § 5.3 (the catalogue's row) — the run's index shown as `<base>-run<N>.points.json`, though it is always `-run0` | the row's `what` adds "named for the run's first try, `-run0`, whichever try a point is on" — **closed 2.60.0**: the index is `<stem>.points.json`, no try in its name | confirmed |
| D2-SUB-2 | → F10, a document's text | submission.md — the sections run § 1, § 3, § 4: there is no § 2 | renumber (or leave numbers and fix bench-summary.md:49's citation to the place that holds the "170 minutes" story, if any) | confirmed |
| D2-WEX-1 | **gone**, checked at 2.67.0: the words no longer stand | worked-example.md:167-169 vs :168, :187 — the benchmark's answer names a trial the sweep never ran | "G=1 N=4 C=5" | partial: PARTIAL \| worked-example.md:167 "bench-G1N2C5/ · bench-G1N4C5/ · bench-G2N8C5/ · …"; :168 "*choice*: elpa, G=1 N=4 C=6 — portable"; :187 "bench-G1N4C5/ ← a trial" \| the answer agrees with none of t… |
| D2-ARC-7 | **gone**, checked at 2.67.0: the words no longer stand | architecture.md:37 and § 4 :738 — the route that `jobset launch` walks is called *submit* | name the route *launch* | confirmed |
| D2-ARC-8 | **gone**, checked at 2.67.0: the words no longer stand | architecture.md — § 5.2 (:886) stands before § 5.1 (:1123) | swap the two numbers or the two sections | confirmed |
| D2-RID-3 | **gone**, checked at 2.67.0: the words no longer stand | run-identity.md:299, :609-611 — *produce* and *generate* for the verbs that are now `describe` and `prep` | "assigned once by the prep that first opens it"; "§ 6. Describing or preparing again in the same calculation folder" | confirmed |
| D2-PAM-2 | **gone**, checked at 2.67.0: the words no longer stand | preparing-for-another-machine.md:186-202 (§ 4) — the table's "today" column shows two refusals that hold as failures | one column, what prep does now, four refusals | confirmed |
| D3-O4 | → F10, a document's text | overview.md § 3a G-1 — the page that asks the GPU is called "the Job Prep page"; no page has that name | "Task setup's run card" | confirmed |
| D3-O5 | **gone**, checked at 2.67.0: the words no longer stand | catalogue `use_gpu` help says the wrapper refuses a missing GPU env; overview § 3a (owner) and the code say prep says so and prepares the job | the help's sentence says what prep says (the env, that the run stops at its start without it, the install command) | confirmed |
| D3-B3 | **gone**, checked at 2.67.0: the words no longer stand | `molbuilder dna --help` says ssDNA only; the command builds duplexes | "build DNA — a strand, `ds,<seq>` or two strands — from sequence" | confirmed |
| D3-PP1 | → F10, a document's text | transport-postprocessing.md has no role/domain header | add **Role:** guide · **Domain:** engines · Companions: transport.md § 2a.12, model/datafiles.md, web/jupyter.md | partial: PARTIAL \| transport-postprocessing.md:1-8 opens on "*(The user, 2026-10-23: "the best way to do this is to give an example code ...")*", no `**Role:**`/`**Domain:**`; every other engines/*.md has `*… |
| D3-P8 | → F10, a document's text | template.md — sentences that contradict their neighbours | one statement each, the code's | partial: PARTIAL \| (a) template.md:2573-2574 "`effective_config`, called six times" vs :2592 "**One operator, five callers, one rule.**" — true contradiction (code: resolve.py has 7 call sites, 5 in the main… |
| D3-P9 | → F10, a document's text | template.md § 3 :364-412 — the retired-items rule is one 49-line paragraph of release history | the rule in three sentences; the list is `template.RETIRED_ITEMS` | partial: PARTIAL \| template.md:364-412, one paragraph; :389-391 "**when its value runs as before without it** (each item's `runs`: any value of `copy_psml` and of the three dynamics settings, `optimizer = "g… |
| D3-T3 | → F10, a document's text | transport.md § 1.2 and § 2a.8 disagree on whether changing `electrode_kz` re-runs the seed | the § 1.2 cell says "runs again with its group (its result would stand)" | confirmed |
| D3-T4 | → F10, a document's text | transport.md § 0.5 — the bridge label is written `BRIDGE`; molbuilder's label is `bridge` | `bridge` | confirmed |
| D3-T12 | → F10, a document's text | transport.md § 2a.13 — `tbt_spin` is listed under Class C (the transmission owns) and Class D (role-fixed) | drop the Class C row | confirmed |
| D4-7 | → F10, a document's text | vibration.md — internal milestone and review tokens a reader cannot follow | drop the milestone tokens (the plan holds them); split the state cell into one row per state. | partial: PARTIAL \| tokens true: vibration.md:1233 "(2026-09-28, the review of M2b‴)", :1495 "the K6 review", :3897 "plan RMP, § 5ab (R4 …", :3923/:3929-3930 "the M2b⁗ review", :2270 "(L8's mismatch on AuBDTA… |
| D4-13 | → F10, a document's text | stages.md — for a reader, no scenario before the rules | open with § 7.1's tree and the task.json example of § 6 as the scenario; the rest follows once D4-12's narrative is gone. | partial: PARTIAL \| the layout tree is § 7.1 (:1448-1465) of 1802 lines — true. But § 1 (:36-46) opens in plain words ("No engine has a concept of a stage. SIESTA reads a `.fdf`; PySCF runs a `.py` …") with a… |
| D4-15 | → F10, a document's text | pyscf.md § 4 — the convergence-header grammar sentence does not parse | one sentence: the header is `convergence.<token>.<leaf>`, the token the stage's (`01_coarse`); the reader also takes the flat form. | confirmed |
| D4-25 | → F10, a document's text | vibrational-averaging.md and vibrational-optical-response.md — the header's Role is outside the closed vocabulary | the Role the user picks; the release list leaves the Role line. | confirmed |
| D5-4 | → F10, a document's text | overview.md § 2.2a I8, I9 and § 2.2b F4 — rules a person cannot follow | split each into the rule (one sentence), then a short list of the doors; old files as a separate "files written before" paragraph. | partial: PARTIAL \| overview.md:240 I9 is 389 words, :326 F4 421 words, :239 I8 157 words (not "the same shape") \| layout preference; rules about files written before are rules (refused by name), not history… |
| D5-15 | → F10, a document's text | parse.md — section order and the reader's way in | renumber 5a/5b in order; split the `finish()` cell into a key list. | partial: PARTIAL \| headings: :1223 "## 5b", :1264 "### 5b.1", :1320 "## 5a" — true. The `finish()` row (:684) is 133 words, not "25-line" \| ordering true (findability); the cell split is preference \| |
| D5-17 | → F10, a document's text | datafiles.md — sections out of order (2, 2b, 2c, 2a, 3) | put § 2a after § 2 (no renumbering needed). | confirmed |
| D5-40 | → F10, a document's text | junction-cell.md § 3.2 — the section asserts a rule and then measures it false in a box beneath it | state the rule once: the registry at the seam is set by `start_registry` (cli `--electrode …:registry=`), `sequence` only orders the walk; drop the measured-false box and "today". | confirmed |
| D5-53 | → F10, a document's text | science/overview.md §§ 3, 4 — labels a reader cannot resolve | say what the note is; drop "(E4)"; drop the history note (next). | confirmed |
| D6-4 | → F10, a document's text | README R2's role vocabulary is not what the headers use, and one chapter has no header | each header given a Role from R2's list (or R2's list states the roles in use) | confirmed |
| D6-20 | → F10, a document's text | workflow.md R-W3 (:53-63) — "a contract document leads with 'The short version'" … "Every contract page opens with what the design IS and HOW it is used" — few do under that name | the § 7 survey records, per contract, whether it leads with its rules (the heading's name aside) | confirmed |
| D6-C12 | → F10, a document's text | :529-531 "nothing of the vibration bundle's reaches the monitor" — 20 modules ride both (runwrap MONITOR_COMPANIONS ∩ VIBRATION_COMPANIONS); what holds: nothing needing numpy/netCDF4. S, no. |  | confirmed |
| D6-C14 | **in part 2.67.2**, the seeded comment; the documents' half → F10 | initconfig.py:174-175, the comment seeded into a person's molbuilder.json, "this file carries their PATHS" (secrets) — the contract: no secret path but the TLS cert files; runtime_config.py:17-18 calls config reading "L1", architecture.md:216 and configuratio… |  | confirmed |
| D6-P25 | → F10, a document's text | conventions/code-audit/package-layout/testing are Role `reference` (README R2: an external snapshot) though they state binding rules. S, no. (with D6-4) |  | confirmed |
| D6-E3 | → F10, a document's text | § 2 :80-91 `launch.mode` "the one key you are likely to change" — it is seeded empty (initconfig.py:186) and, unset, every launch refuses (runtime_config.py:1496-1502); § 7's road launches without `--mode`. S, no. |  | confirmed |
| D6-E4 | → F10, a document's text | § 3, § 4 "every prep without --target refuses" — only a calculation's first (record.py:1424-1429). S, no. |  | confirmed |
| D6-E19 | → F10, a document's text | header :7-8 credits "the runner's verify-before-signal rules" to deployment § 1.0a–1.0c; the chapter states no such rule. S, no. |  | confirmed |
| U1-1 | **closed 2.67.1** | Transport › 1. The junction › a citation the describe door itself marks refused is held, Describe writes it, and prep then refuses it — with a way out the written folder blocks | the describe door (`describe_attempt`, and `/api/transport/describe` for the same citation) answers a `ComposeError` from `compose_junction` as a refusal (no `citation`), so core.js keeps the held citation and Describe … | partial: PARTIAL \| Seen as stated (U1/tr_road-1440.log: `[jfind] … 17 more outside its cell along z … Move that origin … and cite it again`, `send disabled: False`, `describe status: Described — wrote au_tr.… |
| U1-3 | **closed 2.67.1** | Transport › Describe's warnings land on a hidden tab, on the wrong field, while the status says "read them" | core.js `_fieldIds()` adds the shared panel's schema (`_sharedSchema`) and the findings' `formScope` takes `#shared-card` beside `#parameters-card`, so a stage-less shared item lands beside its shared field; a row place… | confirmed |
| U1-4 | **closed 2.67.1** | Molbuilder › Modify › Atom › Delete — one atom deleted, the page says "Deleted: 18 atoms." | modify/viewer.js postOp: say what the op did against what it held before (`n_before - n_after` deleted, `+k` added), or "now N atoms" — the count it already reads | confirmed |
| U1-6 | **closed 2.67.1** | Spectrum › "Structure is already relaxed" › says the structure was "relaxed … to 0.005 eV/Å" from a record that says it did not converge | sidecar.py: "relaxed on SIESTA for 1 geometry step against a 0.005 eV/Å target — not converged (the run had not ended); the largest force left … 0.0087 eV/Å", reading `converged` and `run_state` the record already carri… | confirmed |
| U1-7 | **closed 2.67.2** | PySCF › Unpaired electrons › the parity error: "needs ann even count", code names, and a way out that is the value already stated | chemistry.py check_spin_charge_parity: "needs an even/odd count"; "the electron count (73)"; "the unpaired count 0"; offer only the counts of the right parity that differ from the stated one | confirmed |
| U1-8 | **closed 2.67.1** | Both parameter tabs › after Send, and their static hints — "it asks for the stages, which this hand-over leaves out on purpose", while the hand-over proposes them | one sentence in task-handover.js and the two hints: "Next: open Task setup on this folder — it shows the stages proposed and writes task.json when you save" (known in part: R3-B U20 is Task setup's banner) | partial: PARTIAL \| task-handover.js:333-337 the sentence; index.html:265-266 / spectra.html:193-195 "which is where the stages are asked for"; projects/claude-s28-U1/optimization/pdt_siesta/task.1st.json car… |
| U1-9 | **closed 2.67.1** | Structure optimization › SIESTA › "Run directory & restart" box describes a field the form does not draw, gives it the wrong default, and names a field by the wrong label | delete the static fieldset (prose that is the catalogue's help, out of date); if a line stays, it names "System label (output prefix)" and says the restart choice is made on Task setup's stage, not here | confirmed |
| U1-10 | **closed 2.67.2** | Every findings panel › each finding row prints its machine id under the sentence | `_row` carries `where` as `data-where` (the tests and placement keep it), not as visible text | partial: PARTIAL \| validation-findings.js:111-118 appends `issue.where` "shown so a user can quote it"; page-shell.css:828 "The stable id (R1), shown so a user can quote it"; BUT form-components.css:194-197 … |
| U1-11 | **closed 2.67.2** | Structure optimization › SIESTA › Send with no pseudopotential folder — one fact said twice in two severities, ~90 words each, with code names and a document number | the pseudopotential finding (validation's psml check) in one short sentence naming the elements this structure has (C, H, S) and the folder that covers them; the send's error replaces the live warn for the same `where` … | confirmed |
| U1-12 | **closed 2.67.2** | Transport › several voltages, no treatment chosen › the refusal beside the question is the door's code sentence | the task codec's refusal: "Several voltages: choose yes or no under “the low-bias approximation?”" (the description field's name kept in `where`) | confirmed |
| U1-13 | **closed 2.67.2** | Transport › every rung's locked field › the reason ends in a document reference | `why_role` drops the "(engines/template.md § 6.4)" tail; "a choice with one correct answer" → "this stage needs this value" | partial: PARTIAL \| template.py:2472-2485 `why_role`: "… (engines/template.md § 6.4)" twice; "it is what makes the rung that rung -- a choice with one correct answer". \| True as seen. No contract or user rul… |
| U1-14 | **closed 2.67.2** | Server sentences a person reads end in document section numbers and lead with codes | drop the trailing "(<doc> <§>)" from page-bound sentences in one sweep of the composers named (compose.py, kmesh.check, cell.check, the task codec, template.why_role, the hand-over door) | partial: PARTIAL \| kmesh.py:350-358 "{name} = 1 along {axis} … (engines/transport.md 0.3b)" with `name` the key (`kgrid[0]`); cell.py:1148-1151 "the boundary leaves {z_room:.2f} Å of room -- the box is not p… |
| U1-15 | **closed 2.67.2** | PySCF forms › four field labels are code: "scf.init_guess", "scf.conv_tol (Hartree)", "scf.conv_tol_grad", "scf.max_cycle"; and "If max_steps runs out" | catalogue labels: "Initial guess", "SCF energy tolerance", "SCF gradient tolerance", "Max SCF cycles", "If the step cap runs out" | confirmed |
| U1-16 | **closed 2.67.2** | Every page › the app header says the app "emit[s] SIESTA / PySCF input"; the Structure optimization tab is titled "build a molecule from a sequence" | tagline "Build and edit structures; describe optimization, vibration and transport calculations; read their results."; each `<title>` the tab's label | partial: PARTIAL \| _app_header.html:31 tagline; index.html:5 `<title>molbuilder &mdash; build a molecule from a sequence</title>`; spectra.html:5 "spectra (Raman / IR)"; index.html:104-110 tabs "SIESTA <code… |
| U1-17 | **closed 2.67.2** | Every page that loads a pair saved before 2026-10-14 › the 66-word atom-order notice, with a date and "user index" | "This file does not say which order its atoms are in; read in the order listed. Saving it again on the Molbuilder tab records that order." — one sentence, no date | partial: PARTIAL \| workingcopy_structure.py:133-146 `order_not_stated_said`: "… states no atom order (saved before 2026-10-14): read as listed by user index. A file saved then from a view of a sorted run may… |
| U1-19 | **closed 2.67.1** | Results › trajectory inspector › Convergence targets › "energy step 2.7211386245987997e-05 eV" | `trajectory_log.keys` formats the energy step like the SCF dE row (3 significant figures) | confirmed |
| U1-20 | **closed 2.67.2** | form help › "Write the molwatch trajectory log" › "the Watch tab reads" — there is no Watch tab | "…the Results tab reads" | confirmed |
| U1-21 | **closed 2.67.2** | form-schema › an optional yes/no reads "auto / true / false" | makeTriSelect labels the options "(default)" / "yes" / "no" (values unchanged) | confirmed |
| U1-22 | **closed 2.67.1** | Spectrum at 390 px › the content pane scrolls sideways — the MolView host never opted into the host rule | spectra.html: `<div id="spectra-molview-host" class="molviewer-host">` | confirmed |
| U1-23 | **closed 2.67.2** | Transport › 1. The junction › 476 words in seven paragraphs before the one button | two sentences above the button ("Cite a finished relaxation run's folder or a structure pair saved from one — or a transport calculation's folder to reuse it. The line under the button says what you picked."), the rest … | confirmed |
| U1-25 | **closed 2.67.2** | Structure optimization › "— residues" that nothing fills | delete `#info-residues` and its separator from index.html and the line in viewer.js | confirmed |
| U2-24 | **left** (the user, 2026-11-05: "Leave it off") | Results › calculation root of a SIESTA vibration migrated today (PDT/spectrum/siesta_PDT) — the folded `relax` stage's finished run is on disk but on no ladder row | if listed: `runstatus.every_run` also walks the folders of the description's `retired` stages, read-only rows, naming the stage they folded into. | partial: **PARTIAL** (observation true; not a defect by the contracts as written — a genuine user question, as the reviewer framed it) \| `projects/PDT/spectrum/siesta_PDT/`: `01_relax/` (calcdir, `run-0`) an… |
| U2-6 | **gone**, checked at 2.67.0: the words no longer stand | Results › Result file menu headings — every engine output is filed under "<ENGINE> optimization": a vibration's run (siesta-vib, siesta_PDT, pySCF_PDT), a TranSIESTA device and a TBtrans transmission run, and each calculation root's own source structure | the heading from the folder answer's own words for what the run is (its `place`/kind, e.g. "SIESTA run" / "the run's output"), never "optimization"; presenters.md row follows. | confirmed |
| U2-11 | **closed 2.67.1** | Transport report › Device card of a described, not-prepared calculation (AuBDTAu/transport/lowbias) — a raw load error with an absolute path, and a 404 in the console | the report asks the junction only once prep has composed it (the record's own fact); before that one line, "the junction is composed when the calculation is prepared". | confirmed |
| U2-10 | **closed 2.67.2** | Transport report › I–V card, a calculation that reuses a finished one (lowbias_local_extended) — the note explains a record key by its name | "Shares energies with a finished calculation's point: its current is summed over the joined grid — …" (the key name dropped). | confirmed |
| U2-9 | **closed 2.67.2** | Transport report › head — the treatment still said twice, once as a machine token (R3-A U17 applied as written; still visible) | drop `"treatment: " + rec.treatment` (`transport.js:162`); `low_bias_said` already says it. | confirmed |
| U2-8 | **closed 2.67.2** | Results › ladder detail and transport report › Rungs card — server sentences still carry raw backticks (still true from R3-B U17, 2.55.1 rendered them only on Task setup) | the ladder detail and the report's rung detail drawn through the same code-span helper Task setup uses. | confirmed |
| U2-25 | **closed 2.67.2** | Raw backticks on two more pages (adds to U2-8) |  | confirmed |
| U2-16 | **closed 2.67.1** | Task setup › a run of points' run card (my SIESTA vibration `claude-s28-U2/spectrum/h2_raman`) — the wall rule ("one wall for the whole run") is said on the card only after the first Prep, though the wall is typed before it | `_runs_points` asks the kind's declaration whether the stage runs points (`jobset.points.points_of(task)` / the kind table's record), not whether points are prepared yet. | confirmed |
| U2-17 | **closed 2.67.3** | Task setup › Preview / Prep answer and the Transport tab's findings — machine tokens and one long note said five times | the Prep answer renders a finding's sentence without its `where` (the field is for the parameter tabs, where it points at a field); a note the same for every stage is said once above the stages; "resources" in the run c… | confirmed |
| U2-19 | **closed 2.67.2** | Task setup › *Tell me how it is going* — each report field shows its wire key under its words | the key moves to the row's `title`, as the run card's rows do (R3-B U16's fix). | confirmed |
| U2-20 | **closed 2.67.2** | Task setup › *What to measure* on a PySCF calculation — 60 words and two document paths to say the bench lane speaks SIESTA only | "Benchmarks measure SIESTA runs only; a PySCF run is sized from its own guidance (engines/tuning.md)." | confirmed |
| U2-22 | **closed 2.67.1** | This machine at 390×844 — the page scrolls sideways: each channel row's **Remove** button and the listener's route line run past the card | the channel row wraps its actions under the name (`flex-wrap`), the route `code` breaks (`overflow-wrap: anywhere`). | confirmed |
| U2-12 | **closed 2.67.1** | Results › spectra summary, siesta_PDT run-1 — "Relaxation \| complete — 1 steps" | singular for one step. | confirmed |
| U2-23 | **closed 2.67.2** | Sign-in page — names tabs that no longer exist | "Sign in to use molbuilder." (no list), or the list from `TABS`. | confirmed |
| I1-017 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): a hover word, misleading nobody | the *Recommended:* hover word on Task setup | OPEN | confirmed |
| I1-147 | **closed 2.67.1** | a vibration not yet prepared still lists `<stem>.sbatch` on the files card, which a run of points never writes | OPEN | confirmed |
| I1-148 | **closed 2.67.2** | the transport digest `.<tbt>.digest.nc` listed under *once* "in each run's folder"; it lands only in transmission points | OPEN | confirmed |
| I1-150 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): the card lists each setting by the key its file holds it under | the resolved-settings card shows the key `launch.mode`, not words | OPEN | confirmed |
| I1-154 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): an old index says so on its row, and `jobset migrate` reads it again | a calculation still holding a 2.54 index: the ladder's stage row lists no runs ("its runs could not be listed") | OPEN | partial: PARTIAL \| code true: `runstatus.py:288-306` `run_folders` → `declared.points_in` → `index_of` raises `OlderIndexError` (a `LaunchRecordError`, a `ValueError`: `runrecord.py:46`, `runpoints.py:81`), … |
| I2-008 | **closed 2.67.3** | Task setup's stage row does not show the device's `tsgf/` line | OPEN | unchecked |
| I2-009 | **closed 2.67.3** | the Results place note says nothing for a scratch folder | OPEN | unchecked |
| I2-017 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): the order of four sheets: nothing on those pages draws wrong | four pages load sheets out of `ui-contract.md` § 1's order | OPEN | unchecked |
| I2-018 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): the notice is the same words twice in code, once on a page | the "Measuring is on" notice written twice | OPEN | unchecked |
| I2-024 | → F10, a document's text | `vibration.md` § 10 lists V1.14's UI-walk items as owed, done | OPEN | unchecked |
| I2-050 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): tidy: per-page declarations draw right | the document tier: `html, body`, `footer` and friends still declared per page | OPEN | confirmed |
| I2-051 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): tidy: `.status` and `.tagline` draw right | one home per component: `.status`, `.tagline`; `.card`'s 18px padding | OPEN (`.status`, `.tagline`); the `.card` padding part DONE | confirmed |
| I2-052 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): tidy: per-page token passes draw right | per-page token passes | OPEN | unchecked |
| I2-053 | **closed 2.67.1** | the markdown inspector loads CodeMirror itself | OPEN | confirmed |
| I2-054 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): tidy: literals draw right | raw px/rem literals | OPEN | unchecked |
| I2-055 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): how the scripts load: nothing goes wrong a person sees | MolView unsealed; scripts still classic | OPEN | confirmed |
| I2-084 | **closed 2.67.3** | a group prep prints every member's notes (pseudopotential copies, the h_ratio note) once per rung | OPEN | unchecked |
| I2-108 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): the sidebar reads right | the sidebar has more than one file-slot reader | OPEN | unchecked |
| I2-111 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): one more script loaded, no wrong drawing | Results loads the findings renderer though nothing on it draws a finding | OPEN | restatused: V9 § 7: **OPEN** (premise holds) |
| I2-112 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): the findings show in the panel, none lost | the transport describe door's findings carry no `where` | OPEN (part) | unchecked |
| I2-116 | **closed 2.67.3** | the resources line's memory cap is the terminal's only | OPEN | unchecked |
| I2-160 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): a bridge that works | an unexercised APPLY arm; the getter/setter bridge | OPEN | restatused: V9 § 7: **APPLY: DONE; bridge: OPEN** |
| I2-220 | **closed 2.67.1** | the force-scale slider and the `forceScale` switch can disagree after a restore | OPEN | confirmed |
| I2-221 | **closed 2.67.3, left**: the menu's four kinds with their words and the translation to a rule say different things, and a new kind needs both; `channelKinds` stays with no caller, the menu reading it recorded as not built (`structure-annotations.md`); `molview.md`'s sentence corrected | the filter row kinds in two literal lists | OPEN | confirmed |
| I2-226 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): its locks work | the Structure-optimization tab's two locks of its own | OPEN | unchecked |
| I2-267 | **closed 2.67.3** | prep's answer served as lines | OPEN; the wording the user's call | confirmed |
| I2-268 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): each builder draws right | one element builder | OPEN | unchecked |
| I2-269 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): each download works | one download helper | OPEN | unchecked |
| I2-270 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): a path helper and keyboard access to rows: nothing asked for | one path helper; keyboard access to sidebar rows and glyph buttons | OPEN | restatused: V9 § 7: path helper **OPEN**; keyboard UNSURE |
| I2-272 | **closed 2.67.1, left** (the user, 2026-11-05: "Close them with reasons"): decisions with nothing wrong on a page | the CSS decisions | OPEN (sampled) | unchecked |
| I2-273 | **closed 2.67.1** | MolView's export dialog, a third modal outside `lib/dialog.js` | OPEN | unchecked |
| I2-274 | **closed 2.67.1** | three animations ignore reduced motion | OPEN — one of three left | unchecked |

## F9 — Tests through the road (§ 1ak S4) (62)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B4-1 | **closed 2.67.8**: test | tests/support/road.py `_road_launched` — six case-table rows stand on a launch record written by hand, not by a launch | make the launched state by a launch — the six rows to tier 3 on `support.real_runs.layered`/`copy_of` (its `coarse` launched `--mode direct --foreground`), or the rows dropped where `tests/test_the_road_on_real_runs_e2e… | confirmed |
| B4-2 | test | the sent-run tests — a launch record and a job's address written by hand for a run never sent | the field tier sends one real held-H2 job (`launch task --mode submit`) and reads `concluded` on it; the basic pair retired or kept on your word; `launch_record_path(run)` for the composed path | confirmed |
| B4-3 | **closed 2.67.8**: test | tier-1 spectrum tests — the vibration result a run derives is made by hand and read by the route and the reader | what the route serves of a result is asked of the real runs that already load it through `/api/spectra/load` (test_siesta_vibration_e2e.py, test_vibration_e2e.py on `real_co2_vibration` and the PySCF run); the codec's o… | partial: PARTIAL \| spectra/_helpers.py:65 `_make_results` ("toy numerics"), test_blueprint.py:52 `_make_minimal_results`; counts re-derived: test_blueprint 16 sites/18 tests, test_parsers_json 49/34, test_ty… |
| B4-4 | **closed 2.67.8**: test | older releases' files built by hand — sidecars, deck metadata blocks and spectra as an older molbuilder wrote them | retire the older-file cases; the committed pair replaced by the pair the Structure API writes today (`StructureCodec().write`, as support/junction.py builds its junction) | confirmed |
| B4-5 | **closed 2.67.6**: test | the ledger read by hand — eighteen parsers of one format in ten files, and the product has no reader | one reader beside `ledger.record` in `jobset/ledger.py`, the support helpers and the ten files asking it | confirmed |
| B4-6 | **closed 2.67.7, by judgement**: literal paths in tests state the layout contract and stay; a path derived from the door under test could not fail; the planted file trees are the next release's: test | § 1ak S4's done-test still fails — tests compose run paths and file names by hand | `RealRun.attempt`/`call` (support/real_runs.py), `runs.run_of`, `runs.on_label_at`, `runfiles.compose` through `stage_home(...).names`, `runrecord.launch_record_path` — the doors the support helpers already use | confirmed |
| B4-7 | **closed 2.67.4**: test | watch/test_pattern_matcher.py + watch/test_section_rule.py — 54 tests restate a private matcher's shape on SIESTA lines typed by hand | retire; what the SIESTA parser reads is asked of the real runs (test_siesta_optimization_run_e2e.py, which test_pattern_matcher's own header names) | confirmed |
| B4-8 | **closed 2.67.8**: test | the `.spectra.json` codec — refusals of hand-edited payloads, exception-class shape, and one round trip tested in two files | one table of results the writer writes (a row each: complete, in progress, null intensities, imaginary mode, geometry) through `dump_spectra_json` → `parse_spectra_json`; the hand-edited refusals and the class-shape tes… | confirmed |
| B4-9 | **closed 2.67.7**: test | test_vibration_render_gate.py — decks rendered around prep, and a local name's spelling asserted | the rows of tests/data/the_deck.toml (`engine = "pyscf"`, `calculation = "vibration"`) through `support.road.run_road_case`, asserting what the deck sets (`deck_says`), not a local's name | confirmed |
| B4-10 | **closed 2.67.8**: test | test_warmfiles.py — the shipped restart-file list pinned to a literal copy of itself | retire the two inventory pins; the carry rows the contract states (`job-contracts.md` § 2.3.4's table, the next test in the file) stay | confirmed |
| B4-11 | **closed 2.67.7**: test | five files plant the calculation's own machine record by hand, and edit it as JSON | the machine's record written where a probe writes it (`conftest.write_machine_record(...)`, or a named target by `write_environment(..., named_environment_path("sol"))` as support/road.py `_road_target` does), then `pre… | confirmed |
| B4-12 | **closed 2.67.7**: test | the road's first step copied in thirteen files — a calculation built from describe's internals, with its pseudopotentials, run card and machine record laid by hand | `support.road.describe_calculation` (or `molbuilder describe` itself) and `save_both` for a change; the run card through `one_rank` | confirmed |
| B4-13 | **closed 2.67.6**: test | the plan read around its door — fourteen `json.loads` of `job-set.json` | `JobSet.of(folder)` (and `bench_container(...)` for the benchmark's own plan, as `shelf_headers` finds it) | confirmed |
| B4-14 | test | the run script's `--cold` sweep and the walk script tested on files planted by hand, through private renderers | the cold sweep read where the road leaves it — `support.real_runs.layered` already launches `coarse` "again cold" (test_the_road_on_real_runs_e2e.py:222); the walk's emptying and its not-run chain read on the real-junct… — **the cold-sweep half closed 2.60.0** (the sweep removed); the walk script's half stays F9's | confirmed |
| B4-15 | test | § 1am's three "tests owed, through the road" are still owed; and the 2.56.0 stopped-walk case still read by no test | the wall road's fixtures — `walled` (test_wall_reaches_the_engine_e2e.py:42, card wall `time="00:00:20"`) with `continue_retries=0` for (2); `failed_parts` (test_siesta_vibration_e2e.py, `copy_of(stopped, "walled", …)`)… | confirmed |
| B4-16 | test | stale words in the e2e tests — a § that does not exist, and expectations composed by the code under test | cite the live chapter (`engines/vibration.md`'s thermochemistry §) and spell the expected words as `run-reports.md` § 2.2 / `model/parse.md` § 5d.5 state them | confirmed |
| B4-17 | **closed 2.67.6**: test | tests/support/junction.py — a support helper importing a test module's privates, a header naming users and a sidecar it does not have, and the chain junction built twice | the chain built once in support (as a road dict for `describe_calculation`, `road_structure` making the `Structure`), its constants there; the header rewritten to its users | partial: PARTIAL \| support/junction.py:42 `from test_transport_compose import _BRIDGE, _LAYERS_L, _LAYERS_R`; header :20-25 "Used by `tests/test_script_emit.py`, `test_transport_compose.py` … and its spectru… |
| B4-18 | **closed 2.67.6**: test | two hand-rolled template editors beside the product's own | `template.with_values_set(..., source="person")` for both | partial: PARTIAL \| tests/_road.py:59-69 `set_template_value` (rewrites `value = ` only); test_transport_on_a_real_junction_e2e.py:301-319 `_set_by_the_person` (sets `value` and `source = "person"`); 19 uses … |
| B4-19 | **closed 2.67.8**: test | test_checkpoint_states.py `_staged` — a calculation's run records typed in a shape no writer writes | the fixture's run records dropped (the folder tool needs bytes and folders, which stay), or the stage folders made by `describe_calculation` + `prep` | confirmed |
| B4-20 | **closed 2.67.4**: test | test_envs_siesta_gpu_recipe.py — the recipe's declaration restated, "pinned" to documents by a copy | keep the measured pins (gcc minor, python, cmake floor, ELPA/CUDA flags tied to a failure); retire the restatements; the doc agreement is review's (`code-audit.md` § 1c) | confirmed |
| B4-21 | **closed 2.67.6**: test | support/live_server.py sets the projects-tree variable by hand | a `MonkeyPatch()` inside `serve()` (as conftest's `isolated_projects_root_module` does), undone at exit | partial: PARTIAL \| tests/support/live_server.py:46-50 `if not os.environ.get(PROJECTS_ROOT_ENV): … os.environ[PROJECTS_ROOT_ENV] = tree`; :61-63 `if own is not None: os.environ.pop(PROJECTS_ROOT_ENV, None)` … |
| B4-22 | **closed 2.67.6**: test | the run's monitor-log lines read by regex in the support helpers — a second reader of our own log | the monitor-log reader (`parse/instruments/monitor.py`) answers the status and notify lines, the two helpers ask it | confirmed |
| B4-23 | **closed 2.67.9**: test | a package JavaScript file's text read to syntax-check it | retire; a module that does not parse fails every page that loads the sidebar in the browser tests | partial: PARTIAL \| test_checkpoint_sensor_js.py:247-268 `probe.write_text(source.read_text())` of checkpoint.js / projects-sidebar.js, then `node --check` \| The § 3b list B4 cites names a read "to check a s… |
| B4-24 | **closed 2.67.7**: test | private renderers and helpers tested directly — a measured sample | the run script's blocks asked of the `.run.sh` prep writes (`support.road.stage_file(bundle, stage, ".run.sh")`, the `run_sh` rows of tests/data/*.toml); the rest judged file by file when each is next touched | confirmed |
| B4-25 | **closed 2.67.9**, by judgement: both tests were deleted in 2.65.4 and nothing on the road returns that 404, so what they guarded goes unseen (K2679-4): test | two browser tests answer the page with replies made up in the test, on files that do not exist | the failure shown on a real run's file (a run folder from `real_runs`, the partial asked for a role it does not serve), the mount by clicking the file in the Results tab; or retire if nothing on the road returns that 404 | confirmed |
| B4-26 | **closed 2.67.7**: test | test_describe.py — four refusals one function each, and one checking no words | one parametrized table `(input, the refusal's words)`; the name row states the words the refusal says | confirmed |
| D2-SCH-1 | test | scheduler.md:200-201 (R2) — the test named as R2's pin does not exist | name the test that holds it, or say the rule is review's | confirmed |
| I1-070 | inventory | a point a walk the scheduler stopped never began reads *not run* | OPEN | confirmed |
| I1-091 | inventory | launch records planted in `tests/support/road.py` | OPEN | confirmed |
| I1-092 | inventory | the sent-run tests plant a launch record | OPEN | confirmed |
| I1-093 | inventory | the cold sweep's table of planted files | **closed 2.60.0** (removed: a run is one start) | confirmed |
| I1-094 | inventory | a stage-folder test's hand-built plan | OPEN | unsure: UNSURE \| the plan names no test; `tests/test_jobset.py:428-435` (`test_each_stage_lays_out_in_the_folder_its_deck_token_names`) over `_token_ladder`'s hand-built `JobSet(...)` (:101-111) fits; no re… |
| I1-096 | inventory | a relaxation stopped at its card wall, launched warm twice — `status freq` names the run that computed it | OPEN | confirmed |
| I1-097 | inventory | a SIESTA stage with a retry budget of 0 stopped at its wall — the session log's wall line | OPEN | confirmed |
| I1-098 | inventory | a vibration with `scf_must_converge` off and an fc call ending SCF NO — status and the finish card say *fc failed* | OPEN | confirmed |
| I1-114 | inventory | tests through the doors, TEST-AUDIT (5)–(9), the 300 ms late-answer wait | OPEN | confirmed |
| I1-155 | inventory | JS tests feed `"job_completed"` as a run's detail — an input no road makes now | **closed 2.67.9**: the viewers fed what the server said of real runs as they ran, finished and failed | confirmed |
| I1-156 | inventory | a test helper's docstring still says `<file> <- <run>` | OPEN | confirmed |
| I2-006 | inventory | Task setup's late-answer test waits a fixed 300 ms | OPEN | unchecked |
| I2-027 | inventory | the reduced Hessian with a GPU mean field is untested | OPEN (a test owed) | unchecked |
| I2-044 | inventory | checkpoint rules A3, S2, S3, S4, L8 untested; S6 waits on a door; S5's run-layer half | OPEN | unchecked |
| I2-076 | inventory | three unpinned rules; blind tests; the count | UNSURE | restatused: V9 § 7: count CONFIRMED; the unpinned rules UNSURE |
| I2-078 | **closed 2.67.6** | the test harness can report green when not | OPEN | confirmed |
| I2-088 | inventory | five or six setup errors in the documents' test set, cause unknown | OPEN (unobserved since; cannot be re-checked without a run) | unchecked |
| I2-089 | inventory | installer tests replace the env manager in-process | OPEN | unchecked |
| I2-090 | **closed 2.67.6** | tests, `tools/`, `scripts/` not pyflakes-clean | OPEN — 32 now (tests 28, tools 3, scripts 1); and the package has a second finding | confirmed |
| I2-091 | inventory | a field test fails instead of skipping outside its batch | OPEN | confirmed |
| I2-092 | **closed 2.67.7** | two tests hand-write `environment.json`; the renderer keeps a `machine_record=None` fallback for them | OPEN | confirmed |
| I2-095 | inventory | the PySCF effective-parameters fence asserted on a real run | OPEN | unchecked |
| I2-098 | inventory | MPS start and ranks-per-GPU checked only by absence | OPEN | unchecked |
| I2-099 | inventory | a stated PySCF memory cap reaching `max_memory` — no row | OPEN | unchecked |
| I2-100 | inventory | a parser test writes empty files named by TBtrans's rule | OPEN | unchecked |
| I2-101 | **closed 2.67.7** | `test_jobset.py` builds job sets by hand | OPEN | unchecked |
| I2-102 | **closed 2.67.7** | test files render decks through `spec_for` + `render_deck`, not prep's road | OPEN | confirmed |
| I2-110 | inventory | the spectra panel's hand-picked `REQUIRED_IDS` | **closed 2.67.9**: every id the page's code asks for is in the served partial or page | unchecked |
| I2-114 | inventory | no test reads a PySCF vibration's result during its relaxation | OPEN | unchecked |
| I2-124 | inventory | TS9's undocumented-test count | OPEN | restatused: V9 § 7: **OPEN** (re-measured) |
| I2-128 | inventory | an A/B swap passes `test_backends.py`; the `ByElement(("Au",))` mutant lives | OPEN | restatused: V9 § 7: **OPEN by reading** (mutants not re-run) |
| I2-129 | inventory | `checkpoint.js`'s no-poll property is prose only | OPEN | restatused: V9 § 7: **OPEN** (located) |
| I2-131 | inventory | `LAYER_TOL_ANG` untested | OPEN | partial: PARTIAL \| `cell.py:731` `LAYER_TOL_ANG = 0.5`, the default of `detect_layers` (`:734`); no test names it — but `tests/test_cell.py:364-366` calls `detect_layers(z)` with that default and asserts thr… |
| I2-259 | inventory | the sweep for tests that read text | OPEN | restatused: V9 § 7: **OPEN** (one found) |
| I2-275 | inventory | the layout test skips Transport, This machine and JupyterNB | **closed 2.67.9**: the pages come from the tab list, the three laid out and measured | confirmed |

## F10 — The documents true (305)

### Inventory rows (the plan's own document items) (23)

| id | kind | what | checked |
|---|---|---|---|
| I1-020 | inventory | the four documents found contradicting each other (the gather and versions; the leads and the seed's density; the run-setting ladder; `web-api.md`'s launched prep — archive 2026-10-27 l.1064-1066) — reconciled? | unsure |
| I1-032 | inventory | `job-contracts.md` § 2.6's block table has no row for the preamble preflight | confirmed |
| I1-160 | inventory | placeholders the Documents page draws as nothing | confirmed |
| I2-016 | inventory | `web/app.py` cites a section that does not exist | unchecked |
| I2-019 | inventory | cites rule L4 for the note rule, which is L3 | unchecked |
| I2-021 | inventory | launch comments carrying review codes no reader can resolve | unchecked |
| I2-023 | inventory | sentences still calling V1.9's remainder open, though settled 2026-10-27 (no broad hash) | confirmed |
| I2-075 | inventory | rules nobody wrote down — **closed 2.65.17**: the delete pair and a frame's `lattice` written down; the untrue § 8.2 step removed | unchecked |
| I2-077 | inventory | contracts' line-number pins | confirmed |
| I2-143 | inventory | two textbook references owed in `normal-modes.md` | restatused |
| I2-164 | inventory | a log variable named in a comment, not wired | unchecked |
| I2-167 | inventory | `references.bib`'s unverified entries | restatused |
| I2-176 | inventory | `engines/template.md` still describes `user_custom` as a catalogue item, the carrier § 5e's ruling rejected | unchecked |
| I2-177 | inventory | the document sweep: four superseded clauses and the `cell_origin`/`calibrat` restatements | unchecked |
| I2-213 | inventory | the doc-drift and nit text sweep, after K9 | unchecked |
| I2-214 | inventory | a generated route index | unchecked |
| I2-215 | inventory | a generated module index from `architecture.md` § 3 | confirmed |
| I2-216 | inventory | a written data-structure index pointing at `project-layout.md` § 5 | confirmed |
| I2-217 | inventory | constants and citations with their "why this value" | unchecked |
| I2-218 | inventory | decide what merges | confirmed |
| I2-219 | inventory | the README index is one row per document, rows of essays | unchecked |
| I2-222 | inventory | `molview.md` § 1.1 restating §§ 6.6 / 9.5 / 11.6 | unsure |
| I2-227 | inventory | N11 release 2 — the page split (Transport tab the physics, Task setup the running), contract first; a door for a control's run time | confirmed |

### Other documents (named in a finding) (2)

| id | kind | what | checked |
|---|---|---|---|
| B3-15 | docs-untrue | `jobset/prepared_with.py` has no floor — A7 says every jobset module has one — **gone by 2.65.15's measure**: § 2.1 lists it on floor 4, and every `jobset` module has a floor | confirmed |
| U1-31 | docs-stale | The web chapters carry their own history — dated "until …", "used to", "This said …", retraction boxes — inside contracts | confirmed |

### README.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D6-3 | docs-link | README R7 (:123-126) "No `#` heading-anchor links" — eight anchor links remain in live documents | confirmed |
| D6-5 | docs-stale | README's own rules carry their history | partial: \| D6-5 [docs-stale] README's rules carry their history \| **PARTIAL** \| True for the stories: :13-18 the "Migration complete (2026-07-28) … Waves 0–9" banner… |
| D6-P22 | docs-untrue | :10-12 "README embeds only hero-molbuilder.png" vs :88-89 "exactly these 9 files, each referenced by the README" (only README.md:8 embeds one). S, no. | confirmed |

### architecture.md (4)

| id | kind | what | checked |
|---|---|---|---|
| D6-8 | docs-stale | architecture.md:61, :213 — the `jobset` verb list omits `export` | confirmed |
| D6-9 | docs-untrue | architecture.md:250-251 — an export that does not exist, a package that is a module | confirmed |
| D6-10 | docs-stale | architecture.md § 4 (:275) — ".molstruct.json, its envelope and schema versions v3–v6" | confirmed |
| D6-10b | docs-stale | architecture.md § 4 (:273-274) "New persisted artifacts MUST use `persist.check_schema` and be registered there" (job-contracts.md § 6.1) — seven current schemas are not in that registry | confirmed |

### backend-architecture.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D6-12 | docs-untrue | backend-architecture.md § 1 (:54-55), § 5 (:221) — its layer table contradicts architecture.md § 3, "the one list" | confirmed |
| D6-13 | docs-untrue | backend-architecture.md names code that is gone or never was | confirmed |
| D6-14 | docs-stale | backend-architecture.md is mostly history | confirmed |

### configuration.md (9)

| id | kind | what | checked |
|---|---|---|---|
| D6-C2 | docs-untrue | configuration.md:62, :634 — `task.1st.json` "written by the Task-setup tab"; a parameter tab's Send writes it (`/api/task-setup/handover` renders, the browser writes, build.py:1006-1037), Task setup removes it; job-contracts.md § 6.1 is right; README.md:208 repeats the error. Rule: configuration.md… | confirmed |
| D6-C3 | docs-untrue | configuration.md:60, :72-74 — "Nothing in the program may overwrite [molbuilder.json]… the one verb that ever wrote into a person's config was the scheduler prober"; `auth-setup` (cli.py:1765 `write_config_scope`) and `envs init-config` (initconfig.py:485, :495) write it, as the page's own § 2.1c a… | confirmed |
| D6-C4 | docs-untrue | configuration.md:61, :1069-1071 — `environment.json` written only by "a probe" (M-2: the manager run "the one case" besides); prep writes the calculation's copy (jobset/machine.py:70-87), migrate rewrites it (migrate.py:964-997). docs § 7, S, no. | confirmed |
| D6-C5 | docs-untrue | § 1, § 2.1, § 3 "every configuration file … every lookup" — missing `report-cards.json` (report_cards.py:134-148), `secrets/notify` (notify_setup.py:84-85), `$MOLBUILDER_DATA_DIR/fcc_lattice.json` (modify.py:541-553). docs § 7, S, no. | partial: \| D6-C5 \| PARTIAL \| docs/configuration.md:89 (§ 2.1) "**Every lookup is stated here**, and every one of them is *first-found-wins*" — its table (:92-98) has… |
| D6-C7 | docs-stale | § 8 row 4 "which section holds it is the user's call" — plan Q8 records the ruling (one `proxy.trust`, not built). docs § 7, S, no. known: Q8. | confirmed |
| D6-C10 | docs-untrue | § 3.1 "one row per entry … the table is the authority" — tree and `placement.places()` (placement.py:68-205) differ: report-cards.json and four jupyter-lab entries unpoliced; two modes undrawn. docs § 7, S, no. | confirmed |
| D6-C11 | docs-link | :692-694 "plan.md § 5n.8" (archived); :12-13, :1080 "project-layout.md § 2.3.1 (…M6)" — the content is § 2.3.1b. docs § 7, S, no. | confirmed |
| D6-C13 | docs-stale | about 45 passages of dated history in the contract ("until …", "no longer", "used to", a struck-through § 8 row, "my misreading") — each line listed in the full block. docs § 7, L, no. | confirmed |
| D6-C1 | docs-stale | execution/running-a-job.md:1100-1103 says `probe --write` asks about each `env_init` difference; configuration.md § 4 / M-6 (the user, 2026-10-24) and jobset/_cli.py:2627-2634 copy it and ask nothing. docs § 7, S, no. | confirmed |

### design.md (4)

| id | kind | what | checked |
|---|---|---|---|
| D6-15 | docs-untrue | design.md teaches the retired metadata design — the dataclass as the home of a field's facts, and no TOML for engine parameters | confirmed |
| D6-16 | docs-untrue | design.md:82-95, :116-119 — a second rule for which layer a module is, which the L1 index contradicts | confirmed |
| D6-17 | docs-stale | design.md:166-168 — "one admin list is currently read by two subsystems that need opposite readings of its empty default" | confirmed |
| D6-18 | docs-stale | design.md:79-80, :335 — tombstone and guards that are gone | partial: \| D6-18 [docs-stale] design.md tombstone and guards \| **PARTIAL** \| :79-80 "— until then `tests/test_layering.py` scanned every module's imports" (file gone… |

### engines/builders.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D3-B1 | docs-untrue | builders.md § 3 — the duplex gate is said to raise ValueError; it raises BackendUnavailable — **closed 2.57.0** | confirmed |
| D3-B2 | docs-stale | builders.md — file:line pointers have drifted — **closed 2.57.0** (by name) | confirmed |
| D3-B4 | docs-stale | builders.md:117-119 tombstone "(Before 2026-07-27 the DNA path was lenient ...)" — **closed 2.57.0** | confirmed |

### engines/overview.md (6)

| id | kind | what | checked |
|---|---|---|---|
| D3-O1 | docs-untrue | overview.md § 3a G-1 — the GPU item is described as one `kind = "engine"` item per engine, while the catalogue holds one merged `kind = "deck"` item | confirmed |
| D3-O2 | docs-stale | overview.md § 2 — the script-wrapper table claims "the wrapper itself is defined here" but lists an old block set and the wrong emitter | confirmed |
| D3-O3 | docs-stale | overview.md § 1 — the transport row names a deleted preflight and an old invariant count | confirmed |
| D3-O6 | docs-link | overview.md § 3a — "(§ 7)" and "(§ 7.1)" resolve to nothing in this chapter | confirmed |
| D3-O7 | docs-stale | overview.md § 3 Stage 1 — "(schema 7)" for the sidecar | partial: \| D3-O7 \| PARTIAL \| overview.md:117-118 "`frozen_atoms` is one of them, a *reserved* label rather than a key of its own (schema 7)"; sidecars/molstruct.py:9… |
| D3-O8 | docs-stale | overview.md — release narrative and tombstones throughout a contract (role: overview) | confirmed |

### engines/pyscf.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D4-14 | docs-untrue | pyscf.md § 8 — a versioning rule that contradicts the project's one versioning rule | confirmed |
| D4-16 | docs-untrue | pyscf.md § 7.4 — the verdict lines are said to be "quoted verbatim from the emitter" and are not | confirmed |
| D4-17 | docs-stale | pyscf.md throughout — tombstones in a contract | confirmed |

### engines/siesta.md (8)

| id | kind | what | checked |
|---|---|---|---|
| D3-S1 | docs-untrue | siesta.md § 3 row 3 — the species table is said to be "ordered by Z" | confirmed |
| D3-S2 | docs-untrue | siesta.md § 6.1 — "k = 1 on a periodic axis ... is checked not at all"; on a transport grid it is warned, and the findings table has no such row | confirmed |
| D3-S3 | docs-stale | siesta.md § 6.1 table — the Γ-device row says tbtrans always stops; the code's warning (worded after the user's 2026-10-20 ruling) says it stops only with ≥ 3 in-plane counts, else runs on with an invalid T(E) | confirmed |
| D3-S4 | docs-stale | siesta.md § 3 — "the optional `atom-metadata`" | confirmed |
| D3-S5 | docs-stale | siesta.md § 2 public API — `spec_for`'s signature lacks three arguments | confirmed |
| D3-S6 | docs-untrue | siesta.md § 4 — the Makov-Payne script is said to be written beside "each point of a transport rung"; a transport rung's charge is 0 by rule (same chapter), so none ever is | confirmed |
| D3-S7 | docs-link | siesta.md — three § references resolve to the wrong section of this chapter | confirmed |
| D3-S8 | docs-stale | siesta.md — tombstones and release narrative in a contract | confirmed |

### engines/stages.md (5)

| id | kind | what | checked |
|---|---|---|---|
| D4-8 | docs-stale | stages.md § 1.1a, § 1.4, § 7 — code names and callers that are gone | confirmed |
| D4-9 | docs-stale | stages.md § 1.1a, § 1.3 — the group counts no longer match the catalogue | confirmed |
| D4-10 | docs-untrue | stages.md § 6 — the `task.json` shape the chapter owns names transport's `bias` as a list and omits keys the reader takes | confirmed |
| D4-11 | docs-stale | stages.md § 6b — open questions kept in a contract | confirmed |
| D4-12 | docs-stale | stages.md throughout — the chapter is mostly its own correction history | confirmed |

### engines/template.md (8)

| id | kind | what | checked |
|---|---|---|---|
| D3-P1 | docs-stale | template.md § 1.1a, § 1.2 G3, § 8, § 12.1 row 3 — "no caller filters `select` by `kind`"; the wrapper writer does | confirmed |
| D3-P2 | docs-untrue | template.md § 9.2 and § 12.1 row 1 contradict each other and job-contracts.md § 3.5 on whether `jobset prep` keeps a person's USER-CUSTOM text | confirmed |
| D3-P3 | docs-link | template.md § 9.2 — "(§ 4.2 shows it)" for a `user_custom` item; § 4.2 says it does not exist | confirmed |
| D3-P4 | docs-stale | template.md § 9.1 — the sidecar's `frozen_atoms` "**seeds** the form" | confirmed |
| D3-P5 | docs-stale | template.md § 2.2 — `template.py`'s imports are misstated | confirmed |
| D3-P6 | docs-stale | template.md § 5, § 5.3, § 6.4 — catalogue facts and counts the chapter states have moved | confirmed |
| D3-P7 | docs-stale | template.md § 12.1 row 9 — struck "CLOSED" but its text still argues the open state and names a helper that is gone | confirmed |
| D3-P10 | docs-stale | template.md — release narrative and tombstones throughout a contract | confirmed |

### engines/transport-postprocessing.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D3-PP2 | docs-stale | transport-postprocessing.md:84 "*(built 2.52.0)*" — release narrative | partial: \| D3-PP2 \| PARTIAL \| transport-postprocessing.md:84 "**Averaged over another motion** *(built 2.52.0)*." — the only release tag in the file \| present as qu… |

### engines/transport.md (11)

| id | kind | what | checked |
|---|---|---|---|
| D3-T1 | docs-link | transport.md:23 — the in-page link `(#2a14-what-landed)` resolves to no heading | confirmed |
| D3-T2 | docs-untrue | transport.md § 0.4 — the "Corrected 2026-09-15" note says row 2 is not a guard on the road; row 2 above it now says it is held on every door | confirmed |
| D3-T5 | docs-untrue | transport.md § 2a.11 — a status line quoted "as the door prints it" is not what it prints | confirmed |
| D3-T6 | docs-untrue | transport.md § 3.2 — a section that tells the reader transport sits outside the framework, in the present tense, about code that no longer exists | confirmed |
| D3-T7 | docs-stale | transport.md — status ledgers and plans kept inside the contract | confirmed |
| D3-T8 | docs-untrue | transport.md § 3.8.0 and § 3.6 item 7 — two statements that contradict § 3.8.5 and the code about the form | confirmed |
| D3-T9 | docs-stale | transport.md § 3.3 / § 3.4 — counts and quotes measured once, now wrong | confirmed |
| D3-T10 | docs-untrue | transport.md § 2a.13 *Deferred* — the door said to refuse a charge on transport is not the one that does | confirmed |
| D3-T11 | docs-stale | transport.md § 2a.13 Class C — the SCF rows' `stages` list is spelled out and misses a role | confirmed |
| D3-T14 | docs-stale | transport.md § 8 and § 7 — two counts of the code that moved | confirmed |
| D3-T15 | docs-stale | transport.md — tombstones throughout the contract (beside the ledgers of D3-T7) | confirmed |

### engines/tuning.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D4-19 | docs-stale | tuning.md § 1, § 4 — "the `tight` rung is disabled by default", against stages.md § 2's "there is no `enabled`" | confirmed |
| D4-20 | docs-stale | tuning.md § 4 — the global defaults say `Diag.Algorithm` ScaLAPACK; molbuilder's default is ELPA-1STAGE | confirmed |
| D4-21 | docs-stale | tuning.md throughout — tombstones, and one aside that is now false | confirmed |

### engines/vibration.md (5)

| id | kind | what | checked |
|---|---|---|---|
| D4-2 | docs-stale | vibration.md § 7 I24, § 11 — code names that moved: `_phased_walk`, `runstatus.stage_run_state` | confirmed |
| D4-3 | docs-stale | vibration.md § 5.11 § 8 — the `.EPSIMG` size is the 10 Ry figure; the chapter's own R-c and prep's size line say 0.8 MB | confirmed |
| D4-4 | docs-stale | vibration.md § 6.9 — "What each file of a vibration holds" describes the pre-2.42 one-attempt layout | confirmed |
| D4-5 | docs-stale | vibration.md § 10 — a dated build log inside the contract, its last row now untrue | confirmed |
| D4-6 | docs-stale | vibration.md throughout — tombstones and release narrative in a contract | confirmed |

### execution/architecture.md (5)

| id | kind | what | checked |
|---|---|---|---|
| B3-12 | docs-untrue | execution/architecture.md § 2 note vs A7 — the contract both allows and forbids the gather's import of prep's assembly — **the document half gone; the import moved 2.65.15**: the gather asks `resolve.card_settings`, and `prep_inputs`' note names § 2.2 | confirmed |
| D2-ARC-3 | docs-stale | architecture.md:678-679 — "its stage presets — the table its ladder's tiers are read from, by the one ladder rule (`presets`, `config.stages.ladder_of`)" | confirmed |
| D2-ARC-4 | docs-stale | architecture.md:213-215 + § 2.1 — "**Every `jobset` module has a floor**, so an import into any of them can be judged; the table names each" — `jobset/prepared_with.py` has none | confirmed |
| D2-ARC-5 | docs-untrue | architecture.md:1651 (§ 10.2) — "\| **label → V7** \| naming any file \| every route \| **`identity` and nothing else** (A1) \|" | confirmed |
| D2-ARC-6 | docs-untrue | architecture.md:861 (§ 5, row 11) — the run-time precedence leaves out the person's own overrides | confirmed |

### execution/asu-sol.md (4)

| id | kind | what | checked |
|---|---|---|---|
| D2-SOL-1 | docs-stale | asu-sol.md:75 and :171-195 — `lightwork`'s "max 8 cores per node" left open, though the probe answered it | confirmed |
| D2-SOL-2 | docs-stale | asu-sol.md:201-216 (§ 5.3) — "Decision 38's shape, now concrete" names a domain the probe cannot write | confirmed |
| D2-SOL-3 | docs-link | asu-sol.md:193 — "(`machine-identity-plan.md` P7)" | confirmed |
| D2-SOL-4 | docs-stale | asu-sol.md — the page keeps its own corrections | partial: \| D2-SOL-4 \| PARTIAL \| asu-sol.md:134-135 "*Rewritten 2026-08-27 … What stood here is kept below, because how it was wrong is the useful part.*"; :156 "> **… |

### execution/checkpointing.md (4)

| id | kind | what | checked |
|---|---|---|---|
| D2-CKP-2 | docs-stale | checkpointing.md:1887 and :1906-1907 — `checkpoint diff` and archive pruning called *unbuilt*, a word for what is still coming | confirmed |
| D2-CKP-3 | docs-untrue | overview.md:47 — "the 31 rules behind `molbuilder checkpoint`" | confirmed |
| D2-CKP-4 | docs-stale | checkpointing.md § 12 vs § 13.4 — A3 marked held and owed at once | confirmed |
| D2-CKP-5 | docs-stale | checkpointing.md — a contract carrying its own revision history | confirmed |

### execution/generator.md (6)

| id | kind | what | checked |
|---|---|---|---|
| D2-GEN-1 | docs-stale | generator.md § 7.2 table (:919) and § 10.1 (:1019) — the catalogue's counts | confirmed |
| D2-GEN-2 | docs-stale | generator.md § 9 (:958-964) — "Open, and recorded rather than guessed" holds no open item, and its G3 row states a grammar and a command that are gone | confirmed |
| D2-GEN-3 | docs-link | generator.md:759 — "the field that structurally ends § 5h's finding B" | confirmed |
| D2-GEN-4 | docs-untrue | generator.md:986 (§ 10, class 3) — the launch values "decided … by the resolver, from Environment + config + what you asked for" | confirmed |
| D2-GEN-5 | docs-stale | generator.md — revision history inside the contract | confirmed |
| D6-P3 | docs-untrue | :262 "`jobset prep <run\|bench> [STAGE]`" — `{task\|bench} [WORDS]` (prep/launch/summarize --help); generator.md:963 repeats it. S, no. | confirmed |

### execution/job-contracts.md (16)

| id | kind | what | checked |
|---|---|---|---|
| D1-JCa-1 | docs-untrue | § 2.5b — "shared by all seven": eight verbs take `--bundle`, and `jobset export` is missing from the table | confirmed |
| D1-JCa-2 | docs-untrue | § 2.2a — `latest_run`'s signature as written is not the code's | confirmed |
| D1-JCa-3 | docs-untrue | the short version — "Reserved script blocks (provenance, bench-marks, resume)" | confirmed |
| D1-JCb-2 | docs-untrue | § 4.2a — the worked example runs `prep … --from`, which `prep` does not take | confirmed |
| D1-JCb-3 | docs-untrue | § 6.1 — the run-status row says `run_status` returns four keys | partial: \| D1-JCb-3 \| PARTIAL \| doc :2343 "\\| Run status \\| *(served, not written to disk)* \\| … \\| `parse/dirs/job.py` \\| `state`, `detail`, `last_change_at`, … |
| D1-JCa-5 | docs-link | § references inside the chapter that point at the wrong section | confirmed |
| D1-JCa-9 | docs-link | § 2.1 — "§ 8 of `running-a-job.md`" does not exist | confirmed |
| D1-JCa-4 | docs-stale | header — the chapter still says it pins "the object that carries a finished run forward" | confirmed |
| D1-JCa-6 | docs-stale | § 2.5 — a line-number citation that no longer holds, and a status line | confirmed |
| D1-JCa-7 | docs-stale | § 2.6 — a run of points' walk header named "a transport bias SCAN's" only | confirmed |
| D1-JCa-8 | docs-stale | header–§ 2 — tombstones and release history inside the contract: 29 entries (counted from the list) | confirmed |
| D1-JCb-1 | docs-stale | § 6.1 — the artifact registry has no row for three files every run of points or folder carries | confirmed |
| D1-JCb-4 | docs-stale | § 4.2a — the rules file is built, and the section still reads as its proposal and build plan | confirmed |
| D1-JCb-5 | docs-stale | § 6.2 — the PySCF non-convergence row explains the policy by the retired in-script ladder | confirmed |
| D1-JCb-6 | docs-stale | § 6.3 — the bias-point and launch-group rows still describe a run of points as a bias sweep (see D1-JCa-7, PLa-1) | confirmed |
| D1-JCb-7 | docs-stale | § 3–§ 7 — tombstones, retired sections and release history inside the contract: 62 entries (counted from the list) | confirmed |

### execution/job-request.md (2)

| id | kind | what | checked |
|---|---|---|---|
| D2-JRQ-1 | docs-stale | job-request.md:358-361 (§ 3.1a) — "A calculation's first run of a stage has no figure until the engine's own analysis answers — plan § 1u (b), not built." | confirmed |
| D2-JRQ-2 | docs-stale | job-request.md — status and dates in the headings and body of a finished contract | partial: \| D2-JRQ-2 \| PARTIAL \| job-request.md:266 "### 3.1a … *(…; built 2.29.0 — plan § 1u)*" (plan status in a contract — real); :855-857 "the agreement a launch … |

### execution/job-system.md (29)

| id | kind | what | checked |
|---|---|---|---|
| D1-JSa-2 | docs-untrue | job-system.md:474-481, the next-step table — its order and its line for a changed job differ from the code, and the line it gives is one R-P6 refuses | confirmed |
| D1-JSa-5 | docs-untrue | job-system.md:3 header — **Role: guide**, while the chapter declares itself the contract and holds the refusal ids the code cites | confirmed |
| D1-JSa-6 | docs-untrue | job-system.md:852-854 — "A stage launched before 2.7.0 ... goes back to the state saved before its prep; every way back a verb prints says so" | confirmed |
| D1-JSa-8 | docs-untrue | job-system.md:317-324 — the three printed lines: the prep line's `--bundle` is shown as a full path | confirmed |
| D1-JSa-9 | docs-untrue | job-system.md:880-885 — R-L18 cited as "a run that left none of the files the stage takes" | confirmed |
| D1-JSa-10 | docs-untrue | job-system.md:487 — the next step's word *stopped* is said by no reader | confirmed |
| D1-JSb-1 | docs-untrue | job-system.md § 3.1, the "real job-set.json" example — five parts of it are not what prep writes, though the note under it says "the key order and the value forms are the real ones" | confirmed |
| D1-JSb-7 | docs-untrue | § 5.5 — `molbuilder watch` does not take a run's directory | confirmed |
| D1-JSb-9 | docs-untrue | § 4.1 / § 4.2 — "N decks, N jobs" contradicts the group rule | partial: \| D1-JSb-9 \| PARTIAL \| doc :1496-1498 "Both engines run N decks as N jobs (`stages.md § 1.1a`)"; :1564 "N decks, N jobs, a person looks between the rungs"; … |
| D1-JSb-10 | docs-untrue | § 5.3 grammar box — `--cold` described as a first-launch flag only | confirmed |
| D1-JSc-1 | docs-untrue | § 7 — "a single-family CPU sweep drops `G` from its coordinates entirely (the points are `(K, c)`, the folders `bench-K<k>C<c>/`)" | confirmed |
| D1-JSa-11 | docs-link | job-system.md:1062 — "there is no web plan view (W14, § 8)" cites a row killed and archived | confirmed |
| D1-JSc-7 | docs-link | § 8 — "the cluster half among it (E7)" names a row the live plan does not hold | confirmed |
| D1-JSa-3 | docs-stale | job-system.md:1039-1047, § 1 "A one-paragraph scenario" — describe is said to produce a bundle holding `job-set.json` that is `scp`'d to the cluster | confirmed |
| D1-JSa-4 | docs-stale | job-system.md:57, the short version's key-rule table — "a bench sweep is ONE submission" | confirmed |
| D1-JSa-7 | docs-stale | job-system.md:215 and :433-439 — a run of points' *running* omits the state after its last point | confirmed |
| D1-JSa-12 | docs-stale | history inside the contract — release numbers and narrative in rules | confirmed |
| D1-JSb-2 | docs-stale | § 3 class diagram — `Job` lists a field `asked` that no longer exists | confirmed |
| D1-JSb-3 | docs-stale | § 3 `JobSet.shared` — the rule text still says the geometry is shared | confirmed |
| D1-JSb-4 | docs-stale | § 5.3, "the authority for what you type" — the verb `jobset export` is missing from it | confirmed |
| D1-JSb-5 | docs-stale | § 2, decisions 1 and 4 — they still describe the "producers" that the prep fold deleted | confirmed |
| D1-JSb-6 | docs-stale | § 5.2 "What `prep` lays out on disk" — a pointer to a tree that is no longer there, and a run folder described without its own files | confirmed |
| D1-JSb-8 | docs-stale | § 5.0 checkpoint 2 — names a `--from` that `prep` does not have | confirmed |
| D1-JSb-11 | docs-stale | history inside the contract (tombstones) | partial: \| D1-JSb-11 \| PARTIAL (2 entries: 2 carry history; 1 is a reason to keep) \| :1733-1735 "(`--engine`, `--calculation` — W57, 2026-10-06): a calculation descr… |
| D1-JSc-2 | docs-stale | § 7 — "`BlockSize` belongs on this grid too *(decided 2026-08-11, user; not yet built)* ... reported in `choice` ... so `prep` consumes it the same way" | confirmed |
| D1-JSc-3 | docs-stale | § 9.2 — `siesta/stages.py` said to hold "the shipped ladder" | confirmed |
| D1-JSc-4 | docs-stale | § 8 — the shipped verbs omit `jobset export` | confirmed |
| D1-JSc-5 | docs-stale | § 6.1 — a run of points' walk called "a bias scan" (a SIESTA vibration's calls are walked too) | confirmed |
| D1-JSc-8 | docs-stale | § 6.0a, § 7, § 8 — release history and status inside the chapter | partial: \| D1-JSc-8 \| PARTIAL (3 entries: 2 real history, 1 not a breach under the chapter's present role) \| :2879 "one written before 2.43.2 says `+00:00`, read ali… |

### execution/overview.md (5)

| id | kind | what | checked |
|---|---|---|---|
| D2-OVR-1 | docs-stale | overview.md § 1 map — six execution contracts not routed although the header says the table names them all | confirmed |
| D2-OVR-2 | docs-stale | overview.md:160-163 — "the cluster proof remains (`plans/plan.md` **E7**)" | confirmed |
| D2-OVR-3 | docs-stale | overview.md:174-177 — the diagram's "Still to build (roadmap workstream 1) — Web UI plan view + per-stage status roll-up" contradicts the same page | confirmed |
| D2-OVR-4 | docs-stale | overview.md § 1–§ 4 — release history and tombstones on the start-here page | confirmed |
| D2-OVR-5 | docs-link | overview.md:192 — "Launch a stage … `job-system.md § 3`" | confirmed |

### execution/preparing-for-another-machine.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D2-PAM-3 | docs-stale | preparing-for-another-machine.md — header out of the R-rules, and the history kept in the contract | confirmed |

### execution/project-layout.md (14)

| id | kind | what | checked |
|---|---|---|---|
| D1-PLa-3 | docs-untrue | § 1.4a *Where am I* — the table of what a folder is lacks `shared/` | confirmed |
| D1-PLa-5 | docs-untrue | § 1.0 — says `job-system.md` draws a `bdt_au-bundle/` root; it does not | partial: \| D1-PLa-5 \| PARTIAL \| project-layout.md:151-153 "[`job-system.md`] draws a `bdt_au-bundle/` root holding `job-set.json`, the decks…" — grep of docs/ finds … |
| D1-PLb-1 | docs-untrue | § 2.3.2 — the note under the bench table contradicts the paragraph above it and the code | confirmed |
| D1-PLb-2 | docs-untrue | § 2.3.2 — the printed report sample is not what `recommendation_text` prints for its own dataset | confirmed |
| D1-PLb-3 | docs-untrue | § 7 invariant 6 — "prepare writes only one attempt at ⑤" | confirmed |
| D1-PLb-4 | docs-untrue | § 7 invariant 9 — "the only sources at ③" omits the template | confirmed |
| D1-PLb-5 | docs-untrue | § 6.1 — "the calculation's archive does not reach" a trial's `.DM` | confirmed |
| D1-PLb-8 | docs-untrue | § 5.3 (the catalogue's row) — `notified.json` is written by every monitor, not only a member | confirmed |
| D1-PLa-2 | docs-link | § 1.1 — "§ 6.3's .source reservation" names this chapter's § 6.3, which is about checkpoints | confirmed |
| D1-PLb-7 | docs-link | plan and chapter § references that resolve to nothing | confirmed |
| D1-PLa-1 | docs-stale | § 1.5, § 5.5 — a run of points said of "a rung that sweeps the bias" only | confirmed |
| D1-PLa-4 | docs-stale | § 0–§ 1 — tombstones and release history inside the contract (twelve places, re-counted) | confirmed |
| D1-PLb-6 | docs-stale | § 4.5 — the Makov-Payne script's name-spelling, said "not closed", is closed | confirmed |
| D1-PLb-10 | docs-stale | § 2–§ 8 — tombstones and release history inside the contract: 52 entries below (counted from the list; each re-read) | confirmed |

### execution/run-identity.md (2)

| id | kind | what | checked |
|---|---|---|---|
| D2-RID-1 | docs-stale | run-identity.md:667 (§ 7) — "The warm-file inventory, the `--cold` move-aside glob, the restart banner wording …" — **closed 2.60.0**: run-identity § 7 names no move-aside glob | confirmed |
| D2-RID-2 | docs-stale | run-identity.md — the contract's own edit log, in its body | confirmed |

### execution/run-reports.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D1-RR-2 | docs-stale | § 2.6 — a release narrative inside the contract | confirmed |
| D1-RR-3 | docs-stale | § 3.0 — history of the `channels` key inside the rule | confirmed |
| D1-RR-4 | docs-stale | § 2.8, § 2.9 — release tags and a bug story in the headings and openings | confirmed |

### execution/running-a-job.md (5)

| id | kind | what | checked |
|---|---|---|---|
| D2-RAJ-C | docs-untrue | running-a-job.md:20-23 — "A legacy web install-wrapper endpoint survives as the low-level side door … `job-contracts.md § 2.6`'s note" | confirmed |
| D2-RAJ-D | docs-untrue | running-a-job.md:96-108 (§ 2.0a) — what the log says of MB_NP / OMP_NUM_THREADS, said two contrary ways, neither the code's | confirmed |
| D2-RAJ-E | docs-link | running-a-job.md:572-573 — "built by plan § 5t.3's P4" | confirmed |
| D2-RAJ-F | docs-stale | running-a-job.md — release history inside the guide's rules | confirmed |
| D2-RAJ-G | docs-untrue | running-a-job.md:1101-1104 (§ 5.2) — "an edit to `molbuilder.json` reaches a record at its next `probe --write`, which asks about each difference (`--yes` takes them all; silence keeps the record's)" | confirmed |

### execution/runs-of-points.md (2)

| id | kind | what | checked |
|---|---|---|---|
| B3-16 | docs-untrue | execution/runs-of-points.md § 6 puts `jobset.submit` on floor 4; architecture § 2.1 puts it on 5 — **closed 2.65.15**: `jobset.submit` on floor 5 | confirmed |
| D1-RP-2 | docs-stale | code docstrings that contradict § 1.2/§ 1.3 after 2.56.0 (`.continued-from` for a planned point; *not run*, never *failed*) | confirmed |

### execution/scheduler.md (2)

| id | kind | what | checked |
|---|---|---|---|
| D2-SCH-2 | docs-link | scheduler.md:518 (R11) — "(§ 2.12 — a run reports how well it used *what it was given*)" | confirmed |
| D2-SCH-3 | docs-stale | scheduler.md — change log inside the rules and the vocabulary | confirmed |

### execution/script-preparation.md (8)

| id | kind | what | checked |
|---|---|---|---|
| B1-55 | docs-untrue | docs/execution/script-preparation.md:30 vs :336 — the seam is "fifteen questions" in § 1 and "sixteen" in § 4 | confirmed |
| B1-56 | docs-stale | docs/execution/script-preparation.md:993-996 — "the monitor a SIESTA job carries" | confirmed |
| D2-SCP-1 | docs-stale | script-preparation.md:30 and generator.md:891 — "fifteen questions" against the chapter's own sixteen | confirmed |
| D2-SCP-2 | docs-stale | architecture.md:790-791 — "the eleven sub-steps inside step 3" against script-preparation.md's thirteen | confirmed |
| D2-SCP-3 | docs-stale | script-preparation.md:647-660 (§ 4.3) — "it has twelve slots", and the table lists eleven | confirmed |
| D2-SCP-4 | docs-stale | script-preparation.md:822-823 (§ 4.4) — the writers' catalogue counts disagree with generator.md and the catalogue | confirmed |
| D2-SCP-5 | docs-untrue | script-preparation.md:1111 (§ 6) — "02_tight/ … ← tight's own prep, once coarse has run" | confirmed |
| D2-SCP-6 | docs-stale | script-preparation.md — a dated revision log in the contract | confirmed |

### execution/submission.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D2-SUB-1 | docs-stale | submission.md § 4 — a correction log and a dated incident inside the contract | confirmed |

### model/chemistry.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D5-35 | docs-stale | chemistry.md §§ 1, 3, 3a, 5 — history and a stale process word in the contract | confirmed |

### model/datafiles.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D5-16 | docs-link | datafiles.md § 1 — the `.MD.nc` reading cited at the wrong section | confirmed |

### model/overview.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D5-1 | docs-stale | overview.md § 1 table, the parse.md row — says "three ABCs" | confirmed |
| D5-2 | docs-stale | overview.md §§ 2.2, 2.2a, 2.2b — release history inside the contract | confirmed |
| D5-3 | docs-link | overview.md §§ 2.2a, 2.2b — plan sections cited that the plan no longer holds | partial: \| D5-3 plan § 5ab / § 1ab cited \| PARTIAL \| plan.md:1139 "## 5ab. The remapped state — ARCHIVED 2026-10-18" with the archive link (:1141) — the § 5ab refere… |

### model/parse.md (7)

| id | kind | what | checked |
|---|---|---|---|
| D5-6 | docs-untrue | parse.md § 3 worked example — `run_status` called with one argument | confirmed |
| D5-7 | docs-untrue | parse.md §§ 5d.1a, 5d.3 — the setup rows' `echo` column and the asked-vs-ran launch rows stated as built | confirmed |
| D5-8 | docs-stale | parse.md § 2 — the discriminator list names `"rundir"`, which § 1 says went | confirmed |
| D5-9 | docs-stale | parse.md §§ 2, 3, 4 — the result hierarchy miscounted and partly not re-exported | confirmed |
| D5-12 | docs-stale | parse.md — line-number pins (the chapter's own L494-499 says "A line number is a pin … it rots") | confirmed |
| D5-13 | docs-link | parse.md — § references that do not resolve | confirmed |
| D5-14 | docs-stale | parse.md — history inside the contract, throughout, and a § 8 "History" | confirmed |

### model/structure-annotations.md (4)

| id | kind | what | checked |
|---|---|---|---|
| D5-31 | docs-stale | structure-annotations.md § 7 — names the sidecar schema as v11; the same chapter forbids re-spelling it | confirmed |
| D5-32 | docs-stale | structure-annotations.md §§ 2.1, 3, 4a — line pins, all wrong now | confirmed |
| D5-33 | docs-link | structure-annotations.md §§ 6, 7 — plan rows that no longer say what is cited | partial: \| D5-33 plan § 8 rows 11 and 14; W15 \| PARTIAL \| :360-361 "plan § 8, rows 11 and 14" — plan.md:1426 "*Row 14 closed 2026-10-09*" (true: row 14 is stale). Bu… |
| D5-34 | docs-stale | structure-annotations.md — history notes | confirmed |

### model/structure-molstruct.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D5-23 | docs-untrue | structure-molstruct.md § 7 — the "sidecar consumers" list names readers that do not read a sidecar and a route that is gone | confirmed |
| D5-24 | docs-stale | structure-molstruct.md § 2 — "What refused means" still keyed to v7 | confirmed |
| D5-25 | docs-stale | structure-molstruct.md §§ 1-3, 6 — line pins and history | partial: \| D5-25 molstruct pins + history \| PARTIAL \| pins: :483 "`_paired_sidecar_path` `:246` / `_existing_paired_sidecar` `:258`" — files.py:257/:269 (true); hist… |

### model/structure-periodicity.md (5)

| id | kind | what | checked |
|---|---|---|---|
| D5-26 | docs-stale | structure-periodicity.md §§ 6, 6.1, 6.1a, 6.2 — whole sections the chapter itself calls superseded, kept in the contract | confirmed |
| D5-27 | docs-untrue | structure-periodicity.md §§ 6.1, 6.2, 8.1 — where the gate runs | confirmed |
| D5-28 | docs-stale | structure-periodicity.md § 6.0 "The record" — says the ATOM-METADATA block is written only when there are labels | confirmed |
| D5-29 | docs-link | structure-periodicity.md — a malformed link, line pins, and plan rows that are gone | confirmed |
| D5-30 | docs-stale | structure-periodicity.md — history outside the superseded sections | confirmed |

### model/structure.md (5)

| id | kind | what | checked |
|---|---|---|---|
| D5-18 | docs-stale | structure.md §§ 2.1-2.4, 3.1, 4 — about 30 line-number pins, every one checked now wrong | confirmed |
| D5-19 | docs-stale | structure.md § 2.2c — a defect report and impact list kept as the contract for `title` | confirmed |
| D5-20 | docs-link | structure.md §§ 2.2a, 2.2c — § references with no target in the chapter | confirmed |
| D5-21 | docs-stale | structure.md §§ 2.2b, 2.3 — names that are not in the code | confirmed |
| D5-22 | docs-stale | structure.md §§ 2.2, 2.2e, 2.3, 2.4, 2.5, 3, 6 — history throughout, and a § 6 "Status" release log | confirmed |

### ops/access-control.md (9)

| id | kind | what | checked |
|---|---|---|---|
| D6-O5 | docs-untrue | counts that disagree: "four gates" (:18, deployment:407, design.md:151-158, README:291) vs "Five questions, five gates" (§ 2); "Four honest sharp edges" over five; "(11 classes)" — test_rate_limit.py has 12. S, no. | confirmed |
| D6-O6 | docs-untrue | access-control § 4.2 :276-277 "a loopback request really is local" — false behind the recommended same-host proxy with no sign-in (rate_limit.py:209-223, :252); `auth.trust_proxy` cannot be set without providers (runtime_config.py:584-590). S, no. known: Q8. | confirmed |
| D6-O7 | docs-untrue | § 3.4 "one file per provider" — one per provider KIND (config_dir.py:197-205); deployment § 5.1 is right. S, no. | partial: \| D6-O7 \| PARTIAL \| access-control.md:226-228 "Client secrets follow the same rule — one file per provider. See [`deployment.md`] § 5.1."; deployment.md:627… |
| D6-O11 | docs-stale | access-control (a contract) carries dated history in ~7 places (:95-105 "added 2026-09-15 … This document listed four gates while a fifth had shipped", :341-349, :370-375, :404-415, :503-504). S, no. | confirmed |
| D6-O12 | docs-untrue | deployment § 3.1/§ 3.3 "`serve start` prints the exact URI" — a best guess from the bind address (cli.py:1385-1426), and under `start` it goes to the server log (cli.py:2683-2686). S, no. | confirmed |
| D6-O13 | docs-untrue | deployment § 5 :561-562 "no server environment-variable knobs" — the server reads MOLBUILDER_SERVE_PORT (app.py:180-217, the guide itself tells you to set it), MOLBUILDER_PROJECTS (projects.py:173-178, ahead of paths.projects), MOLBUILDER_CONFIG_DIR/XDG_* (config_dir.py:72-104). S, no. | confirmed |
| D6-O14 | docs-untrue | CSRF defence: deployment § 4 (:423-427, SameSite + form-action) vs web-api.md:446-449 and app.py:229-230 (`connect-src 'self'` — which only limits molbuilder's own pages). Two homes disagree; deployment's reading is right. S, no. | partial: \| D6-O14 \| PARTIAL \| deployment.md:423-427 "**no CSRF token layer**: cross-site request forgery is defended by **`SameSite=Lax` cookies** … plus `form-actio… |
| D6-O15 | docs-untrue | deployment § 1.0a :49 stacks dumped "before any forced child kill" — no code does (only faulthandler on SIGUSR1, cli.py:2470-2490). S, no. | confirmed |
| D6-O16 | docs-untrue | deployment § 5 "the server's sections" omits `admin` and `paths`, lists `envs` (runtime_config.py:947-978 `_SECTIONS`). S, no. | confirmed |

### ops/env-framework.md (10)

| id | kind | what | checked |
|---|---|---|---|
| D6-E1 | docs-stale | § 6 :201-204 "the Task-setup card can write the [channels] file … hands you the exact command" — the This machine tab writes it and emits no command (this-machine.md § 1, § 3; task-setup.md:1318; notify_setup.py:36-39). S, no. | confirmed |
| D6-E2 | docs-untrue | § 0 :38-39 "Everything else … is made by the workflow" — the `.psml` files a SIESTA prep needs are the person's to put in `<projects>/pseudopotential/` (project-layout.md § 2.6; pseudos.py:118); prep stops without them. S, no. | confirmed |
| D6-E5 | docs-stale | :122-123 "This section said … until 2026-09-12". S, no. | confirmed |
| D6-E8 | docs-untrue | § 6.3 :955-964 "Both gaps are tracked" — no plan row (doctor skips `<` specs, envs/doctor.py:439; the libgfortran5 ceiling on an existing env). Rule: README R3. S, user: should the two gaps become plan rows? | confirmed |
| D6-E9 | docs-stale | M4's quoted note no longer the printed one (envs/builds.py:779-788). S, no. | confirmed |
| D6-E10 | docs-untrue | M1–M5 (how a step enters an env) live in the guide while env-framework.md (the contract) defers to it (:696-699) and code cites the guide (diagnostics.py:285). docs § 7, M, no. | confirmed |
| D6-E15 | docs-link | installation.md:351, :1234, env-framework.md:206 cite "plan W36 ⑤/⑥" — not in plan.md. S, no. | confirmed |
| D6-E16 | docs-stale | ~16 dated corrections/history passages (:43-48, :131-150, :577-579, :1015-1041, :1221-1225 …). M, no. | confirmed |
| D6-E17 | docs-stale | the contract carries its history in ~20 places ("used to", "until 2026-…", "Found and fixed"), and :790-798 repeats one sentence twice. M, no. | confirmed |
| D6-E18 | docs-untrue | :747-749 "there are a dozen envs" — six (envs/recipes.py:2682-2683). S, no. | confirmed |

### plans/plan.md (4)

| id | kind | what | checked |
|---|---|---|---|
| B1-57 | docs-stale | docs/plans/plan.md:458 — step 3 still lists "a kind's export" as to be made true | confirmed |
| D6-P17 | docs-stale | § 3b :464-477 an open list (two code defects, four tests owed a redesign, 18 verdicts) that plan.md does not carry. Rule: README R3 (every open item in plan.md). docs § 7, S, no. | confirmed |
| D6-P18 | docs-link | "B3", "B4", "§ 5k", "§ 5l" of plans/plan.md — archived ids/sections. S, no. | confirmed |
| D6-P23 | docs-stale | open items (the five-tab captures; the orphan results-bundle-card.png) not in plan.md. S, no. | confirmed |

### process/ (conventions, testing, code-audit, package-layout, screenshots) (10)

| id | kind | what | checked |
|---|---|---|---|
| D6-P1 | docs-untrue | § 3 command catalogue "19 top-level commands" (:157), "6 sub-groups" (:205), jobset "prep / launch / status / summarize / migrate" — the CLI has 23 (14 commands + 9 groups; `atom-index`, `frameset`, `jobs` in no row), `jobset` also `export`, `machines`, `probe`, `scratch`. docs § 7 (a generated ind… | confirmed |
| D6-P2 | docs-untrue | § 3 :283-288 "engine flags are derived from `SiestaConfig`/`PySCFConfig` field metadata" — no option is built from a config field (describe's are hand-declared, jobset/_cli.py:689-747); the test-map's "dataclass→flag bridge" test does not exist. Same retired design as D6-15. S, no. | confirmed |
| D6-P4 | docs-untrue | :271-273 a missing pseudopotential at prep "exits 2" — it exits 1 (engines.py:261-268 PrepError → ClickException). S, no. | confirmed |
| D6-P5 | docs-untrue | § 2 :82-84 "Each release is a git tag, `v<version>`" — tags end at v2.6.0; 2.7.0–2.56.1 untagged. S, user: tag every release again, or does the sentence go? | confirmed |
| D6-P10 | docs-untrue | § 1d step 3 :420-424 "Re-run the membership asserts … fails a named test that quotes the document" — no test reads conventions § 3 or presenters § 1. S, no. | confirmed |
| D6-P11 | docs-untrue | header :5-6, :528 "graduated into enforced artifact lints (§ 5 rule 5)" — § 5 has no rule 5; the § 3 traps are checked by hand (:490-494, :693-695 "its test was retired"). S, no. | confirmed |
| D6-P14 | docs-stale | § 2's tree is not the package: `parse/scripts/` (gone), `parse/instruments/` and `scheduler/` unnamed, four blueprints missing, `projects.py` placed under web/. S, no. | confirmed |
| D6-P15 | docs-untrue | § 3 "~275 test files … two conftest.py" — 394 `test_*.py`, one conftest.py, subdirs field/, support/, fixtures/ unnamed. S, no. | confirmed |
| D6-P16 | docs-stale | § 4's corrections and § 5 (struck-through packaging items) are history. S, no. | partial: \| D6-P16 \| PARTIAL \| package-layout.md:45 "*(`bundle_writer.py` retired 2026-08-29 …)*"; :76 "`tests/test_layering.py` scanned it until 2026-09-27"; :81-82 … |
| D6-P21 | docs-stale | the manifest's `projects/BDT/` and its file paths do not exist; a vibration's result is `<NN>_freq/run-<n>/<label>.spectra.json`. S, no. | confirmed |

### science/chemistry-correctness.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D5-37 | docs-untrue | chemistry-correctness.md § 3.5 — a test named that does not exist | confirmed |
| D5-38 | docs-link | chemistry-correctness.md header, §§ 1, 4 — documents that do not exist, and "not linked yet" | confirmed |
| D5-39 | docs-stale | chemistry-correctness.md — line pins and history | confirmed |

### science/junction-cell.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D5-41 | docs-stale | junction-cell.md — history across §§ 0, 2.1, 2.2, 3.2, 5, 6 | confirmed |

### science/normal-modes.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D4-22 | docs-untrue | normal-modes.md § 2, § 5 — "molbuilder's all-atoms-free path calls PySCF's `harmonic_analysis`"; no path does | confirmed |
| D4-23 | docs-stale | normal-modes.md § 4c, § 4c.7 — "proposed, not agreed" and "to be measured", against vibration.md's decided and built | confirmed |
| D4-24 | docs-stale | normal-modes.md — a contract carrying its build log and history | confirmed |

### science/overview.md (4)

| id | kind | what | checked |
|---|---|---|---|
| D5-50 | docs-stale | science/overview.md § 4 "Config field ranges" — says the range is dataclass field metadata; it is the catalogue's | confirmed |
| D5-51 | docs-link | science/overview.md § 4, transport table — § references that read as this chapter's | confirmed |
| D5-52 | docs-stale | science/overview.md §§ 3, 4, 7 — line pins and a test count | confirmed |
| D5-54 | docs-stale | science/overview.md — history in the catalogue and the glossary | confirmed |

### science/pseudopotentials.md (3)

| id | kind | what | checked |
|---|---|---|---|
| D5-47 | docs-untrue | pseudopotentials.md §§ 1, 2a, 4, 5 — the status set and the ERROR set are not the code's | confirmed |
| D5-48 | docs-untrue | pseudopotentials.md — the S.psml incident is told as two different defects | confirmed |
| D5-49 | docs-stale | pseudopotentials.md — line pins, the header's "not linked yet", plan refs, history | confirmed |

### science/validation.md (5)

| id | kind | what | checked |
|---|---|---|---|
| D5-42 | docs-untrue | validation.md § 7 — the package tree lists a module that does not exist | confirmed |
| D5-43 | docs-untrue | validation.md § 6 step 2 — "the four state items from `config/state.py`" | confirmed |
| D5-44 | docs-untrue | validation.md § 4.1 R5 — enforcement by a test that does not exist | confirmed |
| D5-45 | docs-link | validation.md §§ 4, 4.1, 7 — references that do not resolve here | confirmed |
| D5-46 | docs-stale | validation.md — history, and a § 10 "Design history" | confirmed |

### science/vibrational-averaging.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D4-26 | docs-stale | vibrational-averaging.md — dated narrative in the chapter | partial: \| D4-26 vibrational-averaging.md dated narrative \| PARTIAL \| :11-12 and :126-132 are the user's request quoted (provenance of a decision — durable, R3); :67… |

### science/vibrational-optical-response.md (2)

| id | kind | what | checked |
|---|---|---|---|
| D4-27 | docs-untrue | vibrational-optical-response.md § 5 — the energy cutoff's keyword is named `Optical.EnergyMaximum`; the deck writes `Optical.Energy.Maximum` | confirmed |
| D4-28 | docs-stale | vibrational-optical-response.md — release narrative and dated corrections in a method chapter | confirmed |

### web/molview.md (2)

| id | kind | what | checked |
|---|---|---|---|
| U1-26 | docs-untrue | molview.md and workspace.md › whether an edit lays down a point on the sequence — the documents say both, the code does one | confirmed |
| U1-27 | docs-stale | molview.md › claims about the code that are no longer true | confirmed |

### web/overview.md (1)

| id | kind | what | checked |
|---|---|---|---|
| U1-29 | docs-untrue | overview.md › § 1 says MolView is mounted editable on the Transport tab | confirmed |

### web/results.md (5)

| id | kind | what | checked |
|---|---|---|---|
| U2-28 | docs-untrue | runtime.md § 3 and § 4 (and overview.md l.110, l.170, l.208) — name two shared modules that no longer exist | confirmed |
| U2-13 | docs-stale | results.md — version history and tombstones inside the contract | partial: \| U2-13 results.md tombstones; § 0.4 designed \| **PARTIAL** — tombstones CONFIRMED; the § 0.4 part is self-declared and planned \| Re-read: l.178 "*(Until 20… |
| U2-14 | docs-stale | task-setup.md — the same, in the Task setup contract | confirmed |
| U2-26 | docs-stale | jupyter.md, this-machine.md, vibrationview.md, web-api.md — version history inside contracts (as U2-13) | confirmed |
| U2-27 | docs-link | web-api.md — four `?doc=` references written as code, so the Documents page shows them as text, not links | confirmed |

### web/tabs.md (1)

| id | kind | what | checked |
|---|---|---|---|
| U1-28 | docs-untrue | tabs.md › § 3 and form-schema.md › § 5/§ 6 describe a Structure-optimization page that is not the page | confirmed |

### web/workspace.md (1)

| id | kind | what | checked |
|---|---|---|---|
| U1-30 | docs-untrue | workspace.md › § 6 says the workspace keeps a browser copy and that `persist` reports it — § 1, § 2 and the code say there is none | confirmed |

### workflow.md (1)

| id | kind | what | checked |
|---|---|---|---|
| D6-19 | docs-link | workflow.md:129 — "the .source segment is reserved, § 6.3" | confirmed |

## F11 — The documents findable (§ 7) (2)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| D6-1 | docs-untrue | README.md:3-5, R1 (:71-75) — "Every document under `docs/` is listed here … If a doc is not listed here it does not exist", yet nine live documents have no row | a row each in § Index (step 2–3 of § 7.4 then point at them) | confirmed |
| D6-2 | docs-stale | README.md § Index rows and the folder table say things their chapters no longer say | rows cut to where a fact lives (the § 7.2 form), which removes most of these at once | partial: **CONFIRMED** (two sub-claims need a second edit in the chapter) \| README.md:220 "schema versioning (v7–v13 read)" vs model/structure-molstruct.md:183 "Current schema: v15. The reader accepts {7, … … |

## F12 — Proven — the three roads in the browser; two review rounds (8)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| I1-027 | inventory | M4's Raman viewer walked in a browser | OPEN | confirmed |
| I1-047 | inventory | two review rounds and a browser walk of every page steps 1–7 touched | OPEN | confirmed |
| I1-112 | inventory | the browser walk of the pages S2.3/S2.4 touched (Transport tab's k-mesh and leads, Spectrum form's *Write H+S* off, the odd-count warning, Task setup) | OPEN | confirmed |
| I1-115 | inventory | two review rounds over S0–S4 and a browser walk | OPEN | confirmed |
| I2-113 | inventory | no browser test animates a mode through the Spectrum code | OPEN | unchecked |
| I2-206 | inventory | the whole road in the browser | OPEN = THE ORDER step 4 (finish-line item 2) | unchecked |
| I2-258 | inventory | each road — optimization, vibration (SIESTA, PySCF), transport — start to finish in the browser on real runs | OPEN (finish-line item 2) | unchecked |
| I2-262 | inventory | the full batch at the step's end | OPEN (at the step's close) | unchecked |

## AFTER-SCIENCE — After the finish line — new science (28)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| I1-012 | inventory | D-h, the polarizability table (six optical runs a geometry) — its cost is now measured (about 2 h a call on AuBDTAu) | SCIENCE | confirmed |
| I1-028 | inventory | the `electronic` kind (§ 5.10 ④): energies along a mode | SCIENCE | confirmed |
| I1-029 | inventory | M4-d, the measured intensity's factors (Stokes shift) as an option | SCIENCE | confirmed |
| I1-033 | inventory | a transport point's SCF trouble signs, once each per point, "once built (`parse.md` § 5d.6, P4)" | SCIENCE | partial: PARTIAL \| `model/parse.md` § 5d.6 (l.1931-1946) states the symptoms with their thresholds and "*The symptoms and the divergence verdict are plan § 5t.3's P4, not yet built*"; plan.md:1179 P4 also ho… |
| I1-034 | inventory | a framework for corrections to a junction's electronic structure (δH: DFT+Σ, rigid shift, tuned hybrid) | SCIENCE | confirmed |
| I1-035 | inventory | the structure question (thiol H kept, Au adatoms) — ordinary transport runs | SCIENCE | confirmed |
| I1-036 | inventory | pick Raman modes from a finished spectrum (extend a vibration with ticked optical calls) | SCIENCE | confirmed |
| I1-037 | inventory | skip the modes symmetry forbids | SCIENCE | confirmed |
| I2-026 | inventory | the vibration calculation's open sub-rows | SCIENCE (container; the sub-rows below) | confirmed |
| I2-028 | inventory | mode-by-mode intensity cross-check against an external code | SCIENCE | confirmed |
| I2-029 | inventory | four held systems through the whole road | SCIENCE (THE ORDER step 6) | confirmed |
| I2-030 | inventory | mode matching across different free sets and across engines | SCIENCE | confirmed |
| I2-031 | inventory | Δρ maps along a mode | SCIENCE (its decision owed) | confirmed |
| I2-032 | inventory | the PySCF probe at five points per mode | SCIENCE (decided D8, not built) | confirmed |
| I2-033 | inventory | the response screen: ① character, ④ `electronic` kind, ⑤, ⑥ | SCIENCE | confirmed |
| I2-034 | inventory | the measured δ + mesh comparison on H₂ | SCIENCE | confirmed |
| I2-035 | inventory | one lead run serving both electrodes when they are one shifted cell | SCIENCE (new functionality, decided 2026-10-14, contract first) | partial: PARTIAL \| `engines/transport.md:1790` "**Always two lead stages** \\| Even when the leads are provably identical"; no code \| LEAD1 is decided NEW FUNCTIONALITY (the user, 2026-10-14), not new scien… |
| I2-042 | inventory | from a mode to the current (M10) | SCIENCE | confirmed |
| I2-049 | inventory | the e–ph coupling from `FC.Save.dHS`; Born-charge IR on SIESTA | SCIENCE | confirmed |
| I2-082 | inventory | transport's convergence sweep | SCIENCE | confirmed |
| I2-142 | inventory | Raman/IR finite-difference steps untested for convergence; ΔF ≫ σ_F not estimated | SCIENCE | confirmed |
| I2-145 | inventory | displaced points seeded from the equilibrium density | SCIENCE (a recorded future improvement) | confirmed |
| I2-171 | inventory | a person's engine text as an input, engine-placed, switchable, recorded | SCIENCE (THE ORDER step 6; not started) | confirmed |
| I2-175 | inventory | the measurements it rests on | still hold | confirmed |
| I2-210 | inventory | per-atom starting spins | SCIENCE (THE ORDER step 6) | confirmed |
| I2-223 | inventory | ATOM as an optional extra check, found like 3DNA | SCIENCE (not started) | confirmed |
| I2-237 | inventory | § 1ag (corrections to a junction's electronic structure), § 1ah (which modes get Raman) | SCIENCE | confirmed |
| I2-266 | inventory | new science | SCIENCE — DUP of I2-028–035, I2-042, I2-049, I2-171, I2-082, I2-210, I2-223 | confirmed |

## AFTER-SOL — After the finish line — only the user's Sol runs can show (13)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| I1-005 | inventory | E13 — the BDT–Au junction on Sol | SOL | confirmed |
| I1-006 | inventory | SCHED-HDR — a group's header under a real scheduler | SOL | confirmed |
| I1-007 | inventory | BENCH-SHELF (I) — every trial after a shelf's first may report the shelf's cgroup memory peak as its own | SOL | confirmed |
| I1-008 | inventory | BENCH-SHELF (J) — the 600 s `wall_slack` left in force on a 15-min queue stops a later trial at its first check | SOL | confirmed |
| I1-009 | inventory | SCHED-R — why SIGTERM was not recorded on Sol job 65999920 | SOL | confirmed |
| I1-010 | inventory | proof 3a: water on both engines, SIESTA's infrared against PySCF's mode by mode | SOL | confirmed |
| I1-011 | inventory | proof 3b: the same water's Raman, SIESTA's optical calls at ±δQ against PySCF's | SOL | confirmed |
| I1-013 | inventory | Sol's PySCF environment repaired for `netcdf4` before a vibration runs there | SOL | confirmed |
| I2-013 | inventory | a shelf's later trials may report the job's cgroup peak as their own | SOL | confirmed |
| I2-014 | inventory | a trial on a 15-minute queue keeps the 600 s wall slack | SOL | confirmed |
| I2-025 | inventory | the BDT–Au junction's first real run on Sol | SOL | confirmed |
| I2-166 | inventory | the gcc 15 migration against the GPU SIESTA pin | SOL / USER (environment work, the user's approval) | confirmed |
| I2-170 | inventory | a group's header under a real scheduler | SOL | confirmed |

## AFTER-FEATURE — After the finish line — features the three roads do not need (12)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| U-PDB | decided | PDB files, and maybe later `.cif`: what more molbuilder reads of a structure file | a plan of its own, not urgent *(the user, 2026-11-04: "pdb (and maybe later .cif) file support should be a separate plan - these are not urgent for now")*; no more of it in the finish-line releases | the user's |
| U-NAMES | decided | a structure read back from a finished SIESTA run keeps its labels, held atoms and cell, and loses atom names, residue names and numbers, chains and title | its own item after the finish line; every deck would record them *(the user, 2026-11-04: "Later")* |
| I2-043 | inventory | no `checkpoint verify` verb | OPEN (a verb the contract describes as missing) | unchecked |
| I2-150 | inventory | `stages.md` § 6b's two questions | USER | partial: PARTIAL \| `engines/stages.md:1356-1368` "## 6b. Open questions about the description … 1. **Is `task.json` the right name?** … 2. **Is a description editable by hand?**" \| The text stands, but neit… |
| I2-151 | inventory | multi-node MPI; `config init --site` | USER (new features) | partial: PARTIAL \| `execution/job-system.md:3288-3289` "Out of scope: **multi-node MPI** (v1 fixes one node), and a `molbuilder config init --site` command" \| The contract already rules both out of scope — … |
| I2-155 | inventory | the per-trial SCF plot | USER (a feature the doc marks not built) | confirmed |
| I2-157 | inventory | `ttl`, `detail`, a `success` level | USER (a feature) | confirmed |
| I2-158 | inventory | whether the markdown editor returns | USER | confirmed |
| I2-162 | inventory | a per-layer overlay refresh; `.pdb` on the download row | USER (features) | partial: PARTIAL \| the cited lines are wrong: `molview.md:2945`/`:2993` are § 10.5's cost table; the items are at `:3257-3259` "The rest is planned, not built ([`plans/plan.md`] **W15**)" and `:4153-4155` "`… |
| I2-165 | inventory | `Resources` spelled as SLURM reads it | USER (a design question for a second scheduler) | confirmed |
| I2-172 | inventory | does the zone survive for free-form text? | USER | confirmed |
| I2-173 | inventory | what the card shows on a collision | USER | confirmed |
| I2-174 | inventory | ordering among additions | USER | confirmed |

## USER — Waits on the user (10)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| I1-001 | inventory | a result written before 2.49.0 keeps its stored Methods sentence (*infrared of a chosen set of atoms*) beside the free-atoms sentence the load adds | USER | confirmed |
| I1-002 | inventory | the note for a root whose description does not read still says *not marked as part of a calculation* | USER | confirmed |
| I1-003 | inventory | `envs/recipes.py` comment calls ASE a monitor reader | USER | confirmed |
| I1-004 | inventory | the page's label palette keeps its own copy of the lead names | USER | confirmed |
| I1-121 | inventory | release 2, the page split (Transport tab the physics editor, Task setup the running) — the contract first | USER | confirmed |
| I1-122 | inventory | a door for each control's run time (nothing reads a run's wall time yet) | USER | confirmed |
| I1-142 | inventory | server sentences name a frame 0-based (*frame index 0 (of 3)*) beside the page's 1-based rows (*Frame 1 of 3*) on one card | USER | confirmed |
| I2-126 | inventory | `detect()` cannot say why each parser refused a file | USER (a nicety; no contract rule found) | partial: PARTIAL \| `parse/registry.py:46-89` `detect` → `_detect_one`: no per-parser reason; `UnknownFormatError` lists every parser \| Evidence right; but the item is a nicety/new feature with no contract r… |
| I2-225 | inventory | the page's label palette keeps its own copy of the lead names | USER (a page change, the user's call) | confirmed |
| I2-271 | inventory | Structure optimization's "discard unsaved changes?" on Load | USER (a design call) | confirmed |

## USER-5t — Waits on the user — half-built run-record framework (§ 5t, § 5s): inside the finish line or after? (7)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| I2-196 | inventory | a PySCF vibration compared with its relaxation by engine alone — no recorded contract for a PySCF deck | DUP of § 5w K18 (I2-209) | confirmed |
| I2-197 | inventory | a reader of a SIESTA run's net charge and spin moment; PySCF's SCF class, ⟨S²⟩, stability recorded | OPEN | confirmed |
| I2-200 | inventory | the setup rows' engine-echo column | OPEN | unchecked |
| I2-201 | inventory | the launch rows asked against those run, beyond the bench | OPEN | confirmed |
| I2-202 | inventory | a SIESTA vibration record's Host/CPU/GPU rows read "—" | OPEN (by reading) | confirmed |
| I2-203 | inventory | the device deck states its contour from the seed's eigenvalues | OPEN (the 10 eV pole energy is the interim) | unchecked |
| I2-205 | inventory | the divergence symptoms, the monitor's warning, the runaway rule | OPEN | confirmed |

## CLOSED — Closed — done, obsolete, not a finding, or rejected (178)

| id | kind | what | the door / status | checked |
|---|---|---|---|---|
| B1-33 | door | siesta/engine_input.py:241-242, :258-259 — the catalogue filtered by engine by hand | `select(catalogue(), engine="siesta")` | rejected: REJECTED \| siesta/engine_input.py:241-242 (`run_setting_keywords`) and :258-259 (`engine_defaults`) `if … (it.engines and "siesta" not in it.engines): continue` \| Two reasons. (1) `siesta/engine_in… |
| B3-29 | door | NEW (2.55.0) — status judges a run of points by the kind table's `finished` rule today, the job by the copy launch wrote into the index | `points_verdict` hands `run_state` the run's own index (`index_at`), the kind's record only where no index was written | rejected: REJECTED \| continuation.py:629-631 `run_state(pts, {"finished": record.finished, "unit": record.unit, "finish": (index or {}).get("finish")})`; runpoints.py:179-180 `empty_index`: `"unit": record.un… |
| D5-36 | docs-untrue | chemistry-correctness.md § 2a.2 ES10 and § 2a.5 — the read-back stated as a working rule | mark ES10 and its row "(not built — plan § 5s P5)" in place; drop the "code is behind" clause. | rejected: REJECTED \| chemistry-correctness.md:274-275 the section's status box already says "**Status: CONTRACT, built (M6, 2026-09-28/29) — all but the read-back (ES10, plan § 5s, P5)**" and :284-285 "Where … |
| U1-24 | words | Projects sidebar › New folder › the dialog says "Created inside claude-s28-U1." before anything is created | mutation-bar.js hint "A new folder inside <ctx>." — at a project's top level "…: one of its topics" (the listing's `topic` marks name them) | rejected: REJECTED \| lib/projects/mutation-bar.js:189-194 `hint: \`Created inside ${ctx}.\`` — a caption naming where the folder goes, the usual dialog idiom, not a report of an act done. The topic rule is en… |
| U2-5 | words | Results › spectra viewer, AuBDTAu/spectrum/siesta-vib run-0 — the line in the bar's place tells the person to run the vibration again with 2.36.0 to get an infrared the run never asked for | in `no_charges_reason`, the `PHASE_NOT_REQUESTED` branch before the schema branch. | rejected: **REJECTED** (premise and fix wrong; a smaller real disagreement remains, the other way round) \| `spectra/infrared.py:179-205` `no_charges_reason`: schema `< 11` branch before `phase_ir == PHASE_NOT… |
| I1-015 | inventory | (b) the engine's own memory analysis, (c) a memory formula | OBSOLETE | confirmed |
| I1-024 | inventory | a transport sweep's one start and finish | DONE | confirmed |
| I1-025 | inventory | `envs/recipes.py` ASE comment | DUP | confirmed |
| I1-030 | inventory | what a transmission round's milestone says (each frame's converged + G/G₀, the frames' average) | DONE | confirmed |
| I1-031 | inventory | a frame set at one voltage: one round, its end the run's finish | DONE | confirmed |
| I1-039 | inventory | one layout | DONE | confirmed |
| I1-040 | inventory | one run of points | DONE | confirmed |
| I1-046 | inventory | the floors and the objects | OPEN | confirmed |
| I1-048 | inventory | `_plan_group` walking a run-of-points member as one run | DONE | confirmed |
| I1-049 | inventory | the phases' warm files | DONE | confirmed |
| I1-050 | inventory | `Run.prepared` for an optical call | DONE | confirmed |
| I1-051 | inventory | the folder answer's status at a run of points' root | DONE | confirmed |
| I1-052 | inventory | the two `_newest_output` helpers | DONE | confirmed |
| I1-053 | inventory | a member copied in whole left unstamped | DONE | confirmed |
| I1-054 | inventory | the web reuse describe dropping a promoted column | DONE | confirmed |
| I1-055 | inventory | transport's `shared/` clause inside the shared done door | DONE | confirmed |
| I1-057 | inventory | `PointsOf.record_says` deciding nothing | DONE | confirmed |
| I1-058 | inventory | a point's way up to its run from two sources; its steps rendered two ways | DONE | confirmed |
| I1-063 | inventory | a point's prepared folder spelled by the walks and by `Run.prepared`'s climb | DONE | confirmed |
| I1-064 | inventory | transport's `shared/` spelled four times in launch | DONE | confirmed |
| I1-065 | inventory | the phase-name branches of `_phased_walk` | DONE | confirmed |
| I1-067 | inventory | a point the walk never ran is *not run*, never *failed* | DONE | confirmed |
| I1-068 | inventory | every point's start is its `.continued-from`, an optical call's written by its planner | DONE | confirmed |
| I1-069 | inventory | the force on a displaced call stays as it is | DONE | confirmed |
| I1-071 | inventory | `jobset migrate` on a SIESTA vibration described before 2.42.0 | OBSOLETE | confirmed |
| I1-078 | inventory | the vibration's `points_in` reading the plan | DUP | confirmed |
| I1-079 | inventory | the PySCF deck's mode records as text | DUP | confirmed |
| I1-080 | inventory | TBtrans decided by stage name in prep | DUP | confirmed |
| I1-081 | inventory | the fit's rung onto `to_wire()` in two doors | DUP | confirmed |
| I1-082 | inventory | `transmission_point` → `points_in` | DUP | confirmed |
| I1-095 | inventory | the transport e2e's relaxation made by hand | DONE | rejected: **REJECTED** \| the item is s25 r4 **R4-J F11** (2.54.0): "the relaxation `P/opt/J` is made with `_write_junction` + a raw `molbuilder describe` + `one_rank`, which restates `real_runs._described(…)`… |
| I1-102 | inventory | a point's record does not state the run it continued from (two sources, the user's) | DONE | confirmed |
| I1-104 | inventory | what a run of points' folder opens answered by `folder_answer`, not `openable` | DUP | confirmed |
| I1-106 | inventory | about a dozen file names by hand in the door, status and export | DUP | confirmed |
| I1-108 | inventory | support helpers and e2e building run folders and records by hand | DUP | partial: PARTIAL \| the § 1am S4 bullet (plan.md:525-530) is I1-091–094 **and I1-095** \| I1-095 is open (above), so the DUP set must include it \| |
| I1-110 | inventory | the run script's branches (`runwrap.py` … `run_script`, a part) | DUP | confirmed |
| I1-111 | inventory | the page's label palette | DUP | confirmed |
| I1-116 | inventory | a frame set's spread stated as a temperature | DONE | confirmed |
| I1-117 | inventory | seven frames | DONE | confirmed |
| I1-118 | inventory | the swing re-averaged after the run | DONE | confirmed |
| I1-119 | inventory | the proofs | DUP | confirmed |
| I1-120 | inventory | the polarizability table | DUP | confirmed |
| I1-123 | inventory | merge the two walks of a run of points, one report machinery, the walk a declared part | DONE | confirmed |
| I1-124 | inventory | a device or transmission point's `--cold` names its gathered inputs and `--force` removes them | DUP | partial: PARTIAL \| s26 VT-SW K1: "a **device or transmission** point's `--cold` names its gathered inputs (leads' .TSHS, seed .DM, device .TS.HSX), and `--force` removes them"; coldsim: transmission "names t… |
| I1-125 | inventory | the PySCF vibration's Mode shows WARM-RESTART/RESUME over `.chk`; `resumes` dropped, the kind sniffed from the deck text | DUP | confirmed |
| I1-126 | inventory | `rm -f` on geomeTRIC's prefix folder under `set -e` | DUP | confirmed |
| I1-129 | inventory | the device deck names the leads' GF folder from the engine's vocabulary, the job from the calculation's | DONE | confirmed |
| I1-130 | inventory | `transport_rung.py` picks the lead by the stage's name | DONE | confirmed |
| I1-131 | inventory | `transmission_point` → `point_of` (§ 1ak S3) | DUP | confirmed |
| I1-132 | inventory | transport's `shared/` spelled four times in launch, handed to every kind | DUP | confirmed |
| I1-133 | inventory | a call's prepared folder a phase-name branch in the walk (`pt.phase == OPTICAL`) | DONE | confirmed |
| I1-134 | inventory | a point's prepared folder spelled two ways (the walks, `Run.prepared`'s climb) | DUP | confirmed |
| I1-135 | inventory | the relaunch rule's words composed once | DUP | confirmed |
| I1-136 | inventory | the three road tests owed (warm twice after a wall; retry budget 0; fc SCF NO) | DUP | confirmed |
| I1-137 | inventory | the "states no cell" refusal: a test, or reached by no road | DUP | confirmed |
| I1-138 | inventory | `points_view`, `point_of`, `views_of` named, not built | DUP | confirmed |
| I1-139 | inventory | `_lead_orbitals`; missing member's next step; label cap; a point's record and its start | DUP | confirmed |
| I1-140 | inventory | the seven 1b deferrals (shared/ clause, `points_in`, `record_says`, two ways up, PySCF text, TBtrans by name, `transmission_point`) | DUP | confirmed |
| I1-141 | inventory | `--cold` leads, geomeTRIC, PySCF banner, `MB_RETRY_N`, one-second log | DUP | confirmed |
| I1-143 | inventory | the fc/optical Run panel's force at a displaced geometry, read as if a relaxation verdict | OBSOLETE | confirmed |
| I1-144 | inventory | a vibration call's Run panel says nothing of what it started from | DONE | confirmed |
| I1-145 | inventory | *failed* used in two senses (ended without running; ran and not done) | DONE | confirmed |
| I1-146 | inventory | Prep's answer does not say the optical container | OBSOLETE | confirmed |
| I1-152 | inventory | bump the transport record's schema for `order_way` | DONE | confirmed |
| I1-153 | inventory | `tests/test_task_setup_prep_e2e.py` `_rows` must read the setting's key, the rows now showing labels | DONE | confirmed |
| I1-158 | inventory | v_molecule's `fc` row reads *from what prep wrote* (a migrated folder with no `.continued-from`) | DONE | confirmed |
| I1-159 | inventory | `older_index` not added to `results.md` § 2.4 / `web-api.md` | OBSOLETE | confirmed |
| I1-161 | inventory | *ended, exit 0* for `job_completed`; a failed part's and a milestone's card titles | DONE | confirmed |
| I2-001 | inventory | tests read files around the owner doors | DUP of § 1ak S4 (plan line 179: "TEST-AUDIT (5)–(9) is § 1ak S4's") — still true | confirmed |
| I2-002 | inventory | expectations computed by the code under test | DUP of § 1ak S4 — still true | confirmed |
| I2-003 | **closed 2.67.8, for the file it names** | prep-only checks sitting in engine (e2e) files; engine runs no test needs | DUP of § 1ak S4 — still true (one example checked) | partial: PARTIAL \| `tests/test_vibration_e2e.py:849-900`: the refusal half drives `jobset prep` only (`"can't be read: 0 is below 1"`), but the same test then RUNS the engine (`d = _prep_and_run(bundle)`, `:… |
| I2-004 | inventory | copies that should be one table's rows | DUP of § 1ak S4 — not re-counted | confirmed |
| I2-005 | inventory | stale words in tests | DUP of § 1ak S4 — still true | confirmed |
| I2-007 | inventory | a machine-record copy naming no machine | DUP of W57-O's first item, I2-103 (DONE 2.41.0) | confirmed |
| I2-012 | inventory | the SIESTA benchmark field list is a second, hand-kept source | DUP of § 1am step 3 (the codebase review's E18 "SIESTA_BENCH_FIELDS" under *engine and kind records*) — still true | confirmed |
| I2-015 | inventory | `jupyter --help` claimed a restart keeps notebook state | DONE (the text now says what happens) | rejected: **REJECTED — still OPEN** \| `cli.py:2668-2670` "The server is parented to the molbuilder SUPERVISOR, not to the web server, so a code reload does not destroy notebook state"; against `serve_daemon.p… |
| I2-020 | inventory | the folder download carrying an engine's scratch | DONE 2.25.0 (§ 1v) | confirmed |
| I2-022 | inventory | `vibration.md` § 5.10's banner still listing ② and ③ as owed | DONE (but one stale word left) | confirmed |
| I2-036 | inventory | the engine offset's P4 | DUP of § 5q (rows in § 5q's table) — still true | confirmed |
| I2-037 | inventory | the charge and spin a run used, read back (P5) | DUP of § 5s P5 — still true | confirmed |
| I2-038 | inventory | the run record's remainder (P2, P4, P6) | DUP of § 5t — still true | confirmed |
| I2-040 | inventory | the Makov–Payne script is run by hand | OBSOLETE (by contract the person runs it) | confirmed |
| I2-045 | inventory | `project-layout.md` said the archive skips a trial's `.DM` | DONE | partial: **PARTIAL** \| invariant 16 corrected: `project-layout.md:2978-2983` "**Not held**: a save … archives a trial's `.DM` like any other". But § 6.1 still says the opposite: `:2801-2803` "A benchmark's t… |
| I2-046 | inventory | `checkpointing.md` § 12 read "needs the layout" | DONE | confirmed |
| I2-047 | inventory | a person's own deck text (§ 5e) | DUP of § 5e (SCIENCE) | confirmed |
| I2-048 | inventory | every parameter confirmed end to end (pointer) | DUP of § 5w (K-rows) and THE ORDER step 4 (M3 P6 = W50 step 3) | confirmed |
| I2-057 | inventory | `transiesta._compute_cell_from_extents` live against "a transport calculation derives no cell" | DONE 2.54.1 | confirmed |
| I2-058 | inventory | the structure-API audit's findings | container — the A1.x rows below; order § 5r.1 | confirmed |
| I2-070 | inventory | a partial metadata payload resets the cell | OPEN (now written as the method's rule in its docstring, no contract checked) | rejected: **REJECTED — not a finding (OBSOLETE)** \| full replace IS the contract: `model/structure-molstruct.md:333` "(v8) … applied **full-replace** on read", `:334` "(v9) … Applied **full-replace** like eve… |
| I2-080 | inventory | audit #2 — the rest of the tree | DUP of § 1am (the whole-codebase review of 2026-10-29 read every module; its ~355 findings are steps 2–7) | confirmed |
| I2-081 | inventory | `runwrap` branches on the engine | DUP of § 1am step 2 ("The run script names no engine") — still true | confirmed |
| I2-086 | inventory | a cold sweep on a finished sweep | DONE (seen on the road) | confirmed |
| I2-094 | inventory | a stage's `overrides` held by no road test | DUP of TEST-AUDIT (4)'s "raw edit of `task.json`" (§ 1ak S4) — held now, but by hand edits | partial: **PARTIAL** \| the overrides ARE held by a road test now: `tests/test_transport_on_a_real_junction_e2e.py:388-398` sets `st["overrides"] = {"mixing_weight": …}` and saves it through Task setup's save… |
| I2-097 | inventory | a warm retry's own launch record (`retry_of`) | OBSOLETE — the field is gone | confirmed |
| I2-103 | inventory | a record copy naming no machine skipped the conflict check | DONE 2.41.0 | confirmed |
| I2-115 | inventory | why SIGTERM was not recorded on Sol job 65999920 | SOL (the advance-signal half settled 2026-10-27) | confirmed |
| I2-117 | inventory | "takes nothing from another run" said by one side | DONE | confirmed |
| I2-118 | inventory | a benchmark continuation's label differs (*written at* vs *every trial's geometry*) | OBSOLETE | restatused: V9 § 7: **OBSOLETE as a parity item; a residue found** |
| I2-125 | inventory | `generator_mismatch`'s severity | OBSOLETE — the contract rules it a warning and the code follows | confirmed |
| I2-127 | inventory | Task setup acting on `onChange` | OBSOLETE | restatused: V9 § 7: **OBSOLETE** (located) |
| I2-130 | inventory | the error surface, unwritten | not located (archived § 11) | unsure: UNSURE, likely done in substance \| located: archive 2026-10-08 `:3054-3071`, "3DNA is not probed by running it … **What must improve instead is the failure**: when it does not work, say exactly what… |
| I2-132 | inventory | 11 documented routes absent, 9 live undocumented (2026-10-08) | DUP of § 7 step 2 (the generated API index) | confirmed |
| I2-135 | inventory | `create_app(config={})` reads the disk config through the snapshot | OBSOLETE | restatused: V9 § 7: **OBSOLETE** (located) |
| I2-136 | inventory | the label handed to the wrapper through a `Resources` field | USER (waits for a yes) | rejected: **REJECTED — OBSOLETE** \| the label now travels on the run's names, handed whole: `render_run_wrapper(script_path, *, names, …)` (`runwrap.py:2123-2125`), `_cold_restart_block(names.label)` (`:2707`… |
| I2-137 | inventory | `BlockSize` as a benchmark axis | DONE (a value axis) | confirmed |
| I2-138 | inventory | the catalogue's `kind` steers no reader | DONE in code; the contract row is stale | confirmed |
| I2-144 | inventory | V1.22's dropped warning half | OBSOLETE — withdrawn | confirmed |
| I2-146 | inventory | a SIESTA road run whose held-first sort reorders atoms | DONE | restatused: V9 § 7: **DONE** |
| I2-147 | inventory | the PySCF road's `info.calculation` | DUP of § 5w K18 | confirmed |
| I2-148 | inventory | the v1 reader's residue (`vibration.md`) | NOT LOCATED | restatused: V9 § 7: not located |
| I2-149 | inventory | nothing compares a trial's asked GPU with `gpu_used` | DUP of § 5t P2 (the launch rows asked against those run) | confirmed |
| I2-153 | inventory | `gpu.md` lists a test that does not exist | DONE | confirmed |
| I2-154 | inventory | a PySCF residual shown with no tolerance (`run-reports.md`) | DONE | restatused: V9 § 7: **DONE** (by reading) |
| I2-156 | inventory | five known gaps in `projects.md` | OBSOLETE | restatused: V9 § 7: **not findings** (re-checked one by one) |
| I2-161 | inventory | the calculation-wide block editable only by hand | DONE | restatused: V9 § 7: **DONE** |
| I2-180 | inventory | A1.15, the stale comments and rules to state | DUP of I2-075 (A1.15) | confirmed |
| I2-181 | inventory | A1.5, one home for the structure-path rule; rename | DUP of I2-059 | confirmed |
| I2-182 | inventory | A1.8, the fallbacks, the partial-charge door, the k-hint; A1.14 joins | DUP of I2-061–063, I2-072–074 | confirmed |
| I2-183 | inventory | A1.10, the builder defects | DUP of I2-064–066 | confirmed |
| I2-184 | inventory | A1.18, the line-number pass | DUP of I2-077 | confirmed |
| I2-185 | inventory | A1.13, each at its owner, starting with `replace()` (A1.6) | DUP of I2-060, I2-067–071 | confirmed |
| I2-186 | inventory | A1.17, the unpinned rules and the tests | DUP of I2-076 | confirmed |
| I2-187 | inventory | A1.20, residue | DUP of I2-079 | confirmed |
| I2-188 | inventory | the audit of the rest of the tree, after steps 1–4 | DUP of I2-080 (the 2026-10-29 review read every module) | confirmed |
| I2-189 | inventory | `frozen_atoms`'s shape is benign | settled (unchanged) | confirmed |
| I2-190 | inventory | "`resolve_cell` is not the one resolver": `_compute_cell_from_extents` | OBSOLETE — the second resolver is gone (2.54.1) | confirmed |
| I2-191 | inventory | the companion-lookup merge stays withdrawn | settled (not re-read) | confirmed |
| I2-192 | inventory | the torn-frame XYZ reader is forced, not a duplicate | settled (unchanged) | confirmed |
| I2-193 | inventory | no file claimed by two parsers | settled (not re-measured; `detect` refuses two claims by construction, `parse/registry.py:46-89`) | confirmed |
| I2-194 | inventory | none in `parse/`; transport numbers through `engine_atom_index` | settled (unchanged) | confirmed |
| I2-195 | inventory | dropped on the user's ruling | settled | confirmed |
| I2-209 | inventory | a PySCF run's level of theory read by the one reader | DUP of § 1am step 3 (the review's P25 = R33, "no engine reading part for *the same calculation*") — still true | confirmed |
| I2-212 | inventory | the wrapper writes *OMP-only* by the deck's `.py` suffix | DUP of § 1am step 2 — still true | confirmed |
| I2-224 | inventory | § 1am, one layout then every fact through its door (steps 1–8; § 1ak S2.4's rest, S3–S5) | OPEN — steps 1 (2.54.0) and 1b (2.55.0) done; 2–8 open | unchecked |
| I2-228 | inventory | messages in the wrong words | NOT LOCATED | restatused: V9 § 7: not located |
| I2-229 | inventory | unit spellings | NOT LOCATED | restatused: V9 § 7: not located |
| I2-230 | inventory | a node's two core counts | NOT LOCATED | restatused: V9 § 7: not located |
| I2-231 | inventory | dead whole-node code (`runwrap.py`, `jobset/model.py`) | DUP of I2-119 (EXCL) | confirmed |
| I2-232 | inventory | the sweep presenter's columns | NOT LOCATED | restatused: V9 § 7: not located |
| I2-233 | inventory | `read_data_file`'s atom map | NOT LOCATED | restatused: V9 § 7: not located |
| I2-234 | inventory | a copied device's provenance | NOT LOCATED | restatused: V9 § 7: not located |
| I2-235 | inventory | PySCF's two convergence doors | NOT LOCATED | restatused: V9 § 7: not located |
| I2-236 | inventory | `--one-job`'s nothing-to-write | NOT LOCATED | restatused: V9 § 7: not located |
| I2-238 | inventory | THE ORDER's steps 2–6 | container — the rows below | confirmed |
| I2-239 | inventory | the run record (W35, § 5t: P2's remainder, P4) | DUP of I2-200–205 | confirmed |
| I2-240 | inventory | the charge and spin a run used, read back | DUP of I2-197 | confirmed |
| I2-241 | inventory | one atomic writer | DUP of I2-207 | confirmed |
| I2-242 | inventory | each engine's parallel model as data | DONE 2.47.8 (`one_process` in `data/engine_table.toml`, § 5w "Done"); its residue DUP of I2-212 | confirmed |
| I2-243 | inventory | the wrapper fence | DUP of I2-208 | confirmed |
| I2-244 | inventory | a PySCF run's level of theory | DUP of I2-209 (archive 2026-10-28 `:1313`: "**Moved** — the ECP read back and a `-V` functional with D3 to K18") | confirmed |
| I2-245 | inventory | each engine default declared once with its source | DUP of I2-211 | confirmed |
| I2-246 | inventory | the named-queue check; `proxy.trust`; the bench grid | DUP of I2-120, I2-121, I2-122 | confirmed |
| I2-247 | inventory | the design-frame exports, the transport validation subject | DUP of I2-178, I2-179 (THE ORDER's "§ 5q.6" no longer exists — § 5q has items 1–3) | confirmed |
| I2-248 | inventory | checkpointing's owed half | DUP of I2-043–046 | confirmed |
| I2-249 | inventory | two calculation-root checks | DUP of I2-083 | confirmed |
| I2-250 | inventory | the lead rebuilt; the cell from extents | DUP of I2-056 (open), I2-057 (done 2.54.1) | confirmed |
| I2-251 | inventory | a SIESTA vibration's finish, the remainder | DUP of I2-039–041 | confirmed |
| I2-252 | inventory | the memory cap row; the kind defaults; old-data tolerances | DUP of I2-099, I2-106, I2-103–105 | confirmed |
| I2-253 | inventory | latent shapes; the GPU reduced Hessian untested | DUP of I2-072–074, I2-027 | confirmed |
| I2-254 | inventory | the tidy rows | DUP of I2-119; I2-220, I2-221; I2-084; I2-108–114; I2-090, I2-091, I2-089, I2-092, I2-100–102; I2-123 | confirmed |
| I2-255 | inventory | one CSS programme | DUP of I2-050–055 (W4, W5 archived) and the Q19 CSS rows below | confirmed |
| I2-256 | inventory | the structure-API cleanup | DUP of I2-180–188 | confirmed |
| I2-257 | inventory | a test writing into the checkout | DONE 2.24.1 — THE ORDER's word is stale | confirmed |
| I2-260 | inventory | rules no test holds; a held atom on PySCF; the kill line and the take-over hop | DUP of I2-093–098, I2-107, I2-085, I2-087 | confirmed |
| I2-261 | inventory | the leads-and-seed road test | DONE — judged 2026-10-18: kept (archive 2026-10-28 `:1317`) | confirmed |
| I2-263 | inventory | the documents | DUP of I2-214–219, I2-222, I2-213, I2-138–141, I2-142–167, I2-124–137 | confirmed |
| I2-264 | inventory | the engine-offset document sweep | DUP of I2-177 (the "§ 5q.5" cited no longer exists) | confirmed |
| I2-265 | inventory | the Makov–Payne script | DONE 2.43.2 — THE ORDER's word is stale | confirmed |
| I2-280 | inventory | `RunRef` for `written_at` and the continuation's fields | DONE (part) — the continuation carries a `RunRef`; `written_at` not checked | rejected: **REJECTED — still OPEN (one half OBSOLETE)** \| `jobset/continuation.py:25-32` `class Continuation: … stage: str … source: str` ("Its attempt, from the calculation folder (``01_coarse/run-0``)") — t… |
| I2-281 | inventory | one wrapper hook per engine | DUP of I2-081 (§ 1am step 2) | confirmed |
| I2-282 | inventory | one path for a single run and a swept stage | DONE | restatused: V9 § 7: **DONE in substance** (by reading) |
| I2-283 | inventory | `next_step`'s `s is None`, an always-false `bench=`, `_with_notify`'s `None` tests | OBSOLETE (part) — no `def next_step` in `molbuilder/`; no `bench=` in `jobset/prep.py`; `_with_notify` (`prep.py:1939`) now says "NO BLOCK IS ``None``" | confirmed |
| I2-284 | inventory | the layer index's missing rows and four upward imports | DUP of § 1am step 7 / § 1ak S3's import scan (not re-run here) | confirmed |

## Found by F0a's review rounds (2026-10-31), by the release that owns each

*(The user, 2026-10-31: "make sure your review on milestones will include holistic code evaluation on api and framework unification, and detect duplicated/redundant/handcrafted code".)* Seven readers on 2.56.2's change — the code and the documents changed, and five holistic readers each taking whole modules (`jobs_cli`, `ready`, `runstatus`; `prep` and the files beside a deck; `diagnostics`, the ending reader, `runrecord`, `migrate`; the build route; the trajectory viewer and the tests). Their findings for this release were fixed in it (the changelog's 2.56.2 row); below, the ones the release that owns the door takes, each read again in the code before it was filed.

| id | release | what | the door |
|---|---|---|---|
| F0a-H4-1 | F0b — **closed 2.57.0** | the builders' findings (a peptide's aldehyde, a duplex's clash, a backend's unenforced form or terminal) travel as Python warnings, which `validation.md` § 4.1 R5 forbids; four of the five builder panels show none, and a warning capture around the build is shared by the whole server (measured: two overlapping builds answered each other's) and records any library's notice (PubChemPy's deprecation) | each builder's findings as `Issue`s, said once by the generate chain the five panels share, in the one renderer (R2) — **its design the user's (F0b)**; 2.56.2 left the route as 2.56.1 had it |
| F0a-H4-12 | F0b / F10 — **closed 2.57.0** | `tabs.md` names the peptide builder AmberTools `tleap` (it is PeptideBuilder); the `/api/build/molecule` row names neither `backend_used` nor the findings; R5 cites `test_no_warnings_warn_in_emitters`, which no test file holds | written with F0b's design |
| F0a-H5-8 | **closed 2.65.11, in part** — the page's badge reads *Not run* for a point its run ended before starting, in the run's own words beneath, seen in a browser on a real failed vibration; the dead source word gone. The engine's name for the title and step word left: nothing goes wrong, one page, two engines | the trajectory page keeps copies of the server's words — the targets' source words, the run states — and branches on the engine's name for its title and step word | the engine's words from its record, served with `format`; the run states from their one home |
| F0a-H2-5 | **closed 2.65.7** | `materialize.bring_files` falls back to the calculation root for the deck and every file beside it, so a file prep took away could still reach a run from the root (latent: no road writes them there) | the deck and `beside_the_deck` from the prepared folder alone, as a vibration call's copy does |
| F0a-H2-6 | **closed 2.65.5** | whether a file travels beside a deck is half the catalogue's `bundle` column, half a hand list in `runfiles.beside_the_deck` — and since 2.56.2 that list decides what prep takes away | one catalogue column, read by engine and kind — `Artifact.travels`, `beside_the_deck(engine, calculation, finish, names)` |
| F0a-H2-8 | **closed 2.65.5** | some `runfiles.WRITTEN` rows' `writer` / `only` words say what the code does not do | the catalogue rows' words — eight rows' words, § 5 rendered again; the Task setup card lists the header with its row's condition |
| F0a-H3-3 | **closed 2.65.7** | `jobset migrate` keeps its own decide-then-write list beside `jobset.planned.Plan`, so it cannot call an owner's writer and the start record grew a text door of its own (`runrecord.continued_from_text`) | migrate's decisions in the `Plan` protocol; the text door then goes |
| F0a-H3-4 | **closed 2.65.7** | `runrecord`'s three one-line records each have their own path, text and reader (partly B2-15) | one row per run record, its file, codec and refusal |
| F0a-H3-11 | **closed 2.65.7** | `runrecord.record_ended` says *written once*, and the once is its caller's | the door holds its own once |
| F0a-H4-7 | **closed 2.65.7** | the source pair's geometry told from its sidecar by its suffix in the hand-over, otherwise than `describe` names it | `StructureCodec.source_files` answers it for both writers |
| F0a-H1-7 | **left 2.65.13** — the row's one predicate does not hold: the two questions differ | *this sequence job sent this run* is decided in `ready.sent_runs` and again in `continuation._own_job_s`; `jobs_holding` matches the address alone, so `jobset scratch` and a run of points' *between two points* miss a run the sequence sent and could not put on its address | one predicate beside the address (`joblist`), asked by all three |
| F0a-H1-8 | **left 2.65.11** — on purpose: `status` holds the differences and picks one line, `way_back` has none and offers both, the same lines underneath | `runstatus._steps` chooses and words a changed group's way on itself, beside `commands.way_back` | `way_back` takes the differences; the step asks it |
| F0a-H1-9 | **closed 2.65.13** — `prepared_with.changed_words`; the `jobs` lines one already | *changed since prepared* and *prepare it again* composed in `ready` and `runstatus` both; the `jobs` verb's lines composed by hand (B3-22's list, longer) | one words helper; a `commands` composer for `jobs` lines |
| F0a-H1-10 | **closed 2.65.11** — the *launched* row says `no`, and the detail beside it says *prepared, not launched*, once | `status <stage>` says a stage with no launch record *prepared, not started*, a missing one among them; *pending*'s detail spelled three ways | the row's own state and detail; the run door's one spelling |
| F0a-H1-11 | **left 2.65.13** — one reader of a literal; the words are what print | the next step's states are bare literals, and Task setup's page spells one again | named once beside `GroupStep` (B3-17's pattern) |
| F0a-H2-4 | **left 2.65.13** — each says what it took away | prep's *a file this prep does not write is taken away* written by hand five times (beside a deck, a walk's header, a dropped stage's header, the sequence's, a job formed again) | one plan step owns it |
| F0a-H3-6 | **left 2.65.13** — a list of what migrate brings on, each read by a person; for F10's documents | what `jobset migrate` brings on is listed in four places — its docstring, `CURRENT`, the command's help, `project-layout.md` § 1.5 — and they differ | migrate's steps as one table, read by the other three |
| F0a-H3-13 | **left 2.65.13** — a bare exit code in two modules that agree | the ending reader's *cannot read* exit is a bare `2` in two modules, its meaning written five times | one named code, rendered into the bundle's entry and the run script |
| F0a-H4-5 | **closed 2.65.11** — `/presets`, `/sweepable` and the analyze answer ask `_not_in_tables` | `/api/task-setup/presets` with a kind the tables do not hold answers 500, `/sweepable` answers settings for it, and `_analyze_response` writes its own refusals | `_not_in_tables`, the door already in `build.py` |
| F0a-H4-6 | **closed 2.65.11** — no `kind` is refused, as its contract says, and a benchmark's folder is opened guarded. Reached by a hand-typed address only, so no test (the user, 2026-10-29) | `/api/task-setup/commands` defaults a missing `kind` to *task* (its contract says 400) and opens a benchmark's folder unguarded (500) | refused as prep refuses; the guarded open |
| F0a-H1-12 | **closed 2.65.16**, the four sites that hold the plan; nine others hold another set or the dict once, and stay | a stage's folder composed by hand from the plan at eight sites (`Path(calc.base) / job_dir_names(...)[job.name]`) | one door, `naming` or the `Calculation` |
| F0a-H2-10 | **closed 2.65.17** | `base` assigned twice in a row in two prep functions — one assignment each | dead code |
| F0a-H3-8 | **its two sentences closed 2.65.16**; the move → F9 | `Capabilities.file_picker_roots` hangs a projects-tree answer on the machine snapshot; tests steer it by monkeypatching the class | the picker asks `projects.projects_root()` |
| F0a-H3-9 | **closed 2.65.15**, the true sentences | `diagnostics`' own statements are untrue (*never the reverse* import; *three* dicts; *the three* facts that travel — four) | with B3-28's import cycle; the counts |
| F0a-H3-12 | **closed 2.65.17** | `runrecord`'s section header over the launch record says *a run of points' index*; its `__all__` leaves out nine names other modules import — the header gone before this release; `__all__` whole | the header removed, `__all__` whole |
| F0a-H4-8 | **closed 2.65.17** | `/api/build/load`'s `fmt` state is dead and its *unknown format* branch unreachable; its module docstring restates shapes the route does not read — the route simplified | removed; the docstring points at web-api.md |
| F0a-H4-9 | **closed 2.65.17** | a bool `add_hydrogens` accepted for an older client no page has — the three words only, a bool refused naming them | the tri-state words only |
| F0a-H4-10 | **closed 2.65.17** | `build.py` imports one constant under two or three names — one name each | the top-of-module names, once |
| F0a-H5-9 | **closed 2.65.17, in part** | the trajectory module keeps fifteen backward-compatibility aliases, one state in two shapes; `state.targets` outside every bucket, so no transition clears it — `state.targets` moved into `fileState`, written by the transition; the aliases left, as `web/results.md` § 4.2 says they stand on purpose | the render code onto the buckets, with the module's ESM conversion |
| F0a-H5-6 | **closed 2.67.1** | the trajectory CSV keeps its own SCF columns: a SIESTA export's residual column is always empty | one column per residual the run declares |
| F0a-H5-7 | **closed 2.65.11** — brought forward from F8, since F3-T widened it to a prepared single point: no targets, no band; the hint gone (`web/trajectory.md` § 3) | *Convergence targets not found … Load the source .fdf* shown for a run that relaxes nothing, against trajectory.md § 3 | no targets, no band |
| F0a-H5-10 | **closed 2.67.1** | the running badge says *sim time* for the wall | § 4's word, one word for both badges |
| F0a-H2-9 | **closed 2.67.6**: `with_values_set` is the one helper, `set_template_value` gone | `template.with_values_set` has no production caller — test support in the product; B4-18 names it the one helper, and this reader argues the line edit is the person's road | F9 decides the one helper (2.56.2 moved `support.road.set_by_the_person` and the case runner onto `with_values_set`; `_road.set_template_value` left) |
| F0a-H3-7 | F9 | `Capabilities.conda_envs` bound as sets and lists in tests, and production keeps a branch for them | tests bind what `detect` makes; the branch deleted |
| F0a-H4-11 | F9 | `test_peptide`'s skip for a missing protonation kit cannot happen (the builder raises) | the skip and its sentence deleted |
| F0a-H5-5 | F9 | `test_trajectory_clocks_js.py` keeps its own source lifter and node runner beside `tests/_node_esm`, and its rows feed states the server never sends | `lifted` + `run_node`; the rows a state the server sends |
| F0a-R2e-3 | **closed 2.65.13** — gone already: `warmfiles` asks `runfiles.engines`, checked equal to the table at load | B3-31's other half: `warmfiles.py` keeps its own list of engines beside `engine_table.engines` | the engine table's list, read |
| F0a-R2a-7 | **closed 2.65.16** — one `task_state` per folder answer | Task setup's folder answer reads the ready door twice — its ladder and groups from one read, its next step from another (`build.py` `_folder_ladder`, `runstatus.jobset_status`) | `JobSetStatus` carries the `TaskState` it read; one read per answer |
| F0a-R2c-4 | F2 | a question asked of the wrong file (not a run's output, or `--stderr` beside an output that carries its own) is said as *the ending cannot be read*, not as usage — no rendered run script reaches it | `ending_of`'s two refusals one class of their own, said as usage |
| F0a-R2c-6 | F2 | the run script's ending chains ask one question twice (`relaxation-capped`, `stopped-by scf_not_conv`), each a full reading of the output | each question asked once, its answer kept, in F2's skeleton |
| F0a-R2b-6 | F9 | the clocks test's rows feed series no road writes (an `elapsed_s` with no epoch) and no row covers an ended run's total from the run's answer | with F0a-H5-5: rows a state the server sends, each with its `run` |
| F0a-R2d-7 | F9 | three callers of `template.with_values_set` (real_runs, test_vibration_e2e, test_siesta_vibration_deck) build the config class by hand, beside `support.road.set_by_the_person` | the one helper |
| F0a-R2d-8 | F9 | rows put run settings under `template`, which the runner moves onto the run card — so *...the run card's beats it* tests the runner's merge, and two gpu rows share one input | each row states a run setting where it lives |
| F0a-R2e-4 | F10 | access-control: § 2.1's diagram routes the notebook to 403 where a server with no sign-in serves loopback, asks one supervisor question where the restart and the notebook ask differently, and draws no public endpoint; § 5 says two subsystems ask the admin list (three do), and the notebook's 403 has its own sentence beside `admin.py`'s; `installation.md` § 1 says the host only installs and probes `molbuilder-MDtools` (it runs `tleap` there to build DNA and RNA); checkpointing's L8 row points at a plan § that no longer exists | each chapter against its code, once |
| F0a-R1-7 | F9 | the 2.56.2 fault-path fixes — the ending reader's crash, the diagnostics door, a sequence's group not on the job's address, a trial whose run is missing — have no test that fails before the fix | judged with F9's tests owed: a state against the design is said, and effort goes to what is allowed |

## Found by F0b's review rounds (2026-10-31), by the release that owns each

Two readers on 2.57.0's change: round 1 the change against its contracts (the species label, the junction's order, the *IR asked* reading, the refused citation's way out, the builders' findings); round 2 whole modules for unification, duplication and residue (the structure codec and its sidecar, the file catalogue and `pseudos`, the infrared reading and the spectra page, the builders, their backends, the build route and the command line). Every finding was fixed in 2.57.0 at the door that owns it, but those listed below; the measured ones are new.

| id | release | what | the door |
|---|---|---|---|
| F0b-R1-C1 | **closed 2.57.0** | a PySCF result before schema 11 that carries no infrared was told it lacks SIESTA's charges | `infrared.no_charges_reason`: a stated setting says *the run recorded none* |
| F0b-R1-E1 | **closed 2.57.0** | clash relief ran on every explicit duplex, a clash-free one changed (and a 500 without OpenBabel) | relieved only where `chemistry.steric_clash` finds one |
| F0b-R1-E2 | **closed 2.57.0** | every canonical duplex with a G·C pair reported as a mismatched-pair clash | the clash check reads the input as the builder does: explicit duplexes only |
| F0b-R1-E3 | **closed 2.57.0** | RDKit DNA asked `3P` / `PP` built OH / 5P, unsaid | each backend's own `built()` from the table its build reads |
| F0b-R1-E4 | **closed 2.57.0** | 3DNA `3P` said built 5′-P / 3′-OH; it is built 5′-OH / 3′-OH | `_threedna._TERMINAL_BUILT`, read by the build and by `built()` |
| F0b-R1-E5, R2-6 | **closed 2.57.0** | `engines/builders.md` and `backend-architecture.md` described Python warnings and a deleted helper | written with the build door |
| F0b-R2-1 | **closed 2.57.0** | the pseudopotential folder was keyed on the element each file declares: `Au1.psml` + `Au2.psml` failed `pseudo check DIR`, and `Au2` was told to take `Au1.psml`; the set's generator check compared nothing for a labelled species | `scan_psml_directory` by file name; `labels_in_directory`; `runfiles.fixed_fields` |
| F0b-R2-2, R2-3, R2-4, R2-5 | **closed 2.57.0** | a hand table of what each backend builds; the default form and terminal spelled per door; `auto` resolved twice; build → check → say assembled by the route and the CLI apart (the CLI's SMILES and name builds unchecked) | `backends.built`, `DEFAULT_FORM` / `DEFAULT_TERMINAL`, `resolved_backend`, the one build door `builders.build` |
| F0b-R2-3a | **closed 2.57.0** | the Molbuilder panels' own `|| "B"` / `|| "A"` fallbacks for an unset form (dna.js, rna.js), and the page's `selected` | the select's chosen form rendered from `backends.DEFAULT_FORM`; none chosen, none sent |
| F0b-R2-7, R2-8, R2-12 | **closed 2.57.0** | the IR column and its CSV column on an older SIESTA result; *still computing* read two ways; the sentence test hand-writing the door's answer | `_irColumn`; the load door's one reading; the test's rows are the door's answers |
| F0b-R2-9, R2-10 | **closed 2.57.0** | *one `.psml` per element* in the pages, refusals and documents; `<label>.psml` composed by hand in eight messages | species-label words; `fixed_name(PSEUDO_FILE, …)` |
| F0b-R2-11, R2-13, R2-14, R2-15 | **closed 2.57.0** | a constant inside another's doc comment; the builder panel's list named *what the build found* while it draws every notice; relief read two ways; sidecar-less writers dropping labels unsaid | moved; the list's words; the explicit-duplex rule; the command line says the labels lost |
| F0b-M1 | **closed 2.57.0** | a canonical duplex (`ds,GCGC`, `ds,ATGCAT`) carried a 0.88 Å H–H contact across C·G (`A2DC:H`–`B7DG:H`) — F0b-M8's cytosine imino H on N3 meeting G's N1–H; with M8 fixed the duplex is clash-free | F0b-M8 |
| F0b-M8 | **closed 2.57.0** | every cytosine 3DNA built (DNA and RNA single strands, a canonical duplex's first strand) came out the imino tautomer — an H on N3, one on N4, a different molecule — because OpenBabel perceived the bond orders from fiber's geometry (C4–N3 1.34 Å, C4–N4 1.33 Å); the H counts the cross-backend test pinned are the same for both forms (measured 2026-10-31) | each base's bond orders from its chemistry (`residues.NUCLEOBASE_BONDS`), set before OpenBabel places the hydrogens |
| F0b-M2 | **closed 2.57.0** — the user, 2026-10-31: "go with b" (said, never repaired unasked; `build.steric_clash` on every DNA and RNA build) | AmberTools single strands (`ATGC`, `ACGU`): every junction's `H3′(i)`–`H5′(i+1)` 1.14 Å — tleap's `sequence` sets ζ +60°, α −41° at every junction, putting `C3′(i)` and `C5′(i+1)` 2.66 Å apart (measured 2026-10-31); in raw tleap DNA two methylene H sat 0.06 Å apart, which the methylene fix separates into this contact | the user's: repair tleap's backbone (set the torsions, or relax it), or say it — the check after the build reads explicit duplexes only |
| F0b-M3 | **closed 2.57.0** | RDKit: every hydrogen its `AddHs` placed was labelled one residue `MOL` (50 of ATGC's), so a bonded O–H (0.99 Å) read as an inter-residue contact | `AddHs(addResidueInfo=True)`: each H in its anchor's residue |
| F0b-M4 | **closed 2.57.0** | the spectra and transport data files' `element` variable — "the chemical symbol" — held a species label (`Au1`) | `Structure.chemical_symbols()`; the label stays in the structure held whole. The transport writer's labelled case has no test: its one test, the real-junction e2e, is all hydrogen |
| F0b-M5 | **closed 2.57.0** | the sidecar's refusal of a label whose name is not the file's symbol (`MolstructPairingError`) had no test | a step of the pair round-trip test: the geometry edited alone and opened anyway |
| F0b-M6 | USER | PubChemPy deprecates `canonical_smiles` / `isomeric_smiles`; the fix needs `pubchempy>=1.0.5` — an environment change | the user's |
| F0b-M7 | field tier | `runfiles.find_fixed`'s scheduler-output reader is reached by the field suite alone | by the basic suite's rule — no scheduler text fabricated (`tests/test_a_sent_run_the_scheduler_forgot.py`) — the field tier's; `fixed_fields` itself is pinned by `pseudo check`'s rows |

### F0b round 3 (2026-10-31) — four whole-file readers and the test audit

*(The user, 2026-10-31: "validate the correct test for each topic is in present and perform static code review for each round within that scope in their full code/text"; "follow the test design discipline, avoid bloating up the total tests numbers, check for overlaps with old tests"; "e2e is only needed after f9".)* Four readers took the release's scope whole — the species label and the pseudopotential check, the builders and their panels, the infrared reading and its page, the two transport fixes — and the tests were audited against `process/testing.md`.

| id | release | what | the door |
|---|---|---|---|
| F0b-R3-1 | **closed 2.57.0** | the settings gate split the labels by folder, so `Au2` (library) was told to take `Au1.psml` (`Au1` from the folder) — the bug closed above, back through the gate; the set's generators compared per folder | `check_coverage` over the structure's whole label set, each label's folder given; the gate calls it once |
| F0b-R3-2 | **closed 2.57.0** | an unparseable `<label>.psml` answered `misnamed` (C6 says `parse_warning`); a label no file can be named for (`au`, `Au_1`, `CA`) crashed `pseudo check` and the gate with a traceback; the misnamed words said the file *will* be read | the element compared only where the file declares one (`chemistry.is_atom`); `pseudos.file_named_for`; the words |
| F0b-R3-3 | **closed 2.57.0** | `Structure.geometry_lines` (a record's atom keys and fingerprint) wrote the label, so a labelled structure's records would not match a run read by element | the element (`chemical_symbols`), as its docstring says; `Structure.labelled_species` the one "which labels are not their element" |
| F0b-R3-4 | **closed 2.57.0** | the page read "still computing" four ways (`done` twice, `_asked && !done`, a hand `empty or running`); the IR column's route clause reached only by a state no writer makes | `_running` everywhere; `_irColumn` asked and stated or running |
| F0b-R3-5 | **closed 2.57.0** | a builder panel without its findings list dropped the notices unsaid (R3); a plain `H` named in a clash; a default form said *asked*; the amber terminal advice untrue for 3P/PP; the DNA clash words blamed the backend for hydrogens molbuilder's methylene repair places; the Python API documented around the door | `panel-deps` answers an error naming the panel; `H on <anchor>`; `Asked.defaults`; the words; the docs point at `builders.build` |
| F0b-R3-6 | **closed 2.57.0** | the refusal outside the cell named the tab's save for a frame set, which the tab cannot save whole; the fixture's new calculation `P/transport/O` collided with the `reused` fixture's (14 e2e tests would have errored) | the frame set's way said; the fixture cites the one-geometry pair as `P/transport/outside` |
| F0b-R3-T | **closed 2.57.0** | the test audit: `tests/spectra/test_infrared_asked.py` duplicated the sentence test's door (deleted; its current-schema assertions folded in); rows building older releases' results (schema 6, 10) and a state no writer makes removed; `test_web`'s page-default test (a second copy is review's) removed; two scan tests, two build rows and the relieved-duplex row shown covered by hand mutants and removed; the U1-2 e2e test built a v13 sidecar and passed on the old code — removed; in round 4, the sentence test's schema-6 row (an older release's result), the two JS tests pinning that a panel spells no default form (a second copy is review's, as above) and the panel test's "no findings list" row (a page the app never serves) removed too | `process/testing.md` § 3b, § 6 |
| F0b-R3-7 | untested until after F9 | U1-2 (an older release's junction record — no test builds one, `testing.md`); the refusal's way out and the transport data file's `element` for a labelled junction — their only tests are the real-junction e2e, not run before F9 | — |
| F0b-R3-8 | not a finding (2.59.0) | a SIESTA finish whose charges were not recorded writes `phase_ir` *not requested*, so a run that asked for its infrared is told Compute IR was off | the finish states the ask and the absence apart — read again at F1: a deck asking charges whose output prints none is refused at the finish; `phase_ir` is *not requested* only where Compute IR was off |
| F0b-R3-9 | **closed 2.65.11, in part** — the RNA panel sends the text as typed, as the DNA panel does, its letters checked with the one `panelDeps.strandCore`; a one-sided label is a 400 in the parser's words. Left: Z under `auto` (the refusal names 3DNA and choosing a backend is the way); the CLI's hand list of options (a person's options, nothing wrong). `build.py`'s copy of `_normalise_h_mode`: 2.65.13 | the DNA and RNA panels' own notation checks disagree with `nucleic.parse_dna_notation` (a one-sided label reaches the server as a 500; Z refused under `auto` whatever it resolves to; `5'-ACGU-3'` refused); `build.py` re-implements `nucleic._normalise_h_mode`; the CLI's nucleic options a hand list (no `relax_clashes`, `add_hydrogens`) | the server's parser and the `GENERATORS` asks, read by the page and the CLI |
| F0b-R3-10 | **left 2.65.14** — *(the user, 2026-11-04: "why is this a problem? isn't openbabel already fucking included in the environment?")*: it is (`envs/host-env.txt`), so the fallback is not reached on the road we support | the RDKit hydrogen fallback (OpenBabel absent) reads the bases' bond orders from its residue templates (canonical, measured) but drops the hydrogens it cannot place — cytosine N4 and guanine N2 left one H each — stated as its limit | the fallback's hydrogens placed, or the fallback refused for nucleic acids |
| F0b-R3-11 | — | the IR phase dot reads the phase's record (`web/spectra.md` § 3), which differs from `infrared_asked` only on older files — no work on older files (`testing.md`) | — |

### F0b round 4 (2026-10-31) — the round-3 fixes, read whole again

| id | release | what | the door |
|---|---|---|---|
| F0b-R4-1 | **closed 2.57.0** | round 3's comment said the RDKit fallback perceives the bases' bond orders from geometry; RDKit's residue templates give the canonical orders (probed) — the fallback's limit is the hydrogens it drops | the chemistry header and `model/chemistry.md` § 2 |
| F0b-R4-2 | **closed 2.57.0** | `molbuilder dna --help` still said single-stranded; `build_dna`'s docstring promised every overlap named | the words |
| F0b-R4-3 | **closed 2.57.0** | `_irColumn`'s clause differed from `_irAsked` only on older releases' files (no work on them, `testing.md`) | the column and the CSV read the door's `infrared_asked`; a SIESTA result from before 2.36.0, asked by the schema rule, shows an IR column of dashes — an older file |
| F0b-R4-4 | — | "(the default)" reaches the command line and Python only: the page always sends the select's form | — |
| F0b-R4-5 | — | a panel clears its earlier findings at every build, a cancelled one included (`web/tabs.md`: cleared at every build); a throw inside the renderer is reported as the route unreachable | the panel's error words, with F8 |
| F0b-R4-6 | **closed 2.57.0** | `pseudo check DIR` never screened an unreadable `<label>.psml` (the folder's list was built from the files that parse) | `labels_in_directory` lists every `<label>.psml`, read or not |
| F0b-R4-7 | **closed 2.57.0** | `copy_pseudopotentials` and the mesh check composed the file name beside `file_named_for`; the gate's test compared a set of findings (a doubled one passed); the generator row could not see a file no label opens compared; `test_non_psml_files_ignored` failed for nothing; stale `.py:NNN` pointers and "eight statuses" in `pseudopotentials.md` | one door; counted findings; a row with an unopened file; retired; the doors named |
| F0b-R4-8 | **left 2.65.13** — the file is found or it is not; the search is the person's | a misnamed file in the calculation folder, for a label that falls through to the library, is not said misnamed (the search is the label's own folder); with `psml_lib` unset the person who put it there is told to | the gate's search over every folder the structure reads |
| F0b-R4-9 | **closed 2.65.11** — the pseudopotential check leaves a label naming no element to the species check, whose sentence now states the rule (`chemistry.names_an_element`) | a label no file can be named for (`Au_1`) is said twice — by the coverage check and `check_species_labels`; a label naming no element (`Xx1`) is told to download a file | the coverage check leaves a label the species check refuses to it |
| F0b-R4-10 | F5 | a label naming no element (`Xx1`, `Au_1`), written through the Structure API alone, is written into its pair and refused by its own reader | the pair writer refuses what its reader refuses |

## Found by F0c's review rounds (2026-10-31), by the release that owns each

Round 1: three whole-file readers — the launch side (submit, continuation, ready, ladder, points, the runs-of-points chapter), the job side and the tests (siesta_vibration, displacement_sweep, runfiles, the e2e and the protocol rows), the spin change (optical_data, the grammar, fdf, the EPSIMG reader); round 2: two readers over the round-1 fixes and the whole modules again.

| id | release | what | the door |
|---|---|---|---|
| F0c-R1-1 | **closed 2.58.0** | the rule *a later stage continues from the first* resolved by the ladder (floor 3) from the run-of-points declaration (floor 4) | a floor-1 part, `vibration_phases.continues_from`, named the kind table's `continues` and resolved as `inputs` is |
| F0c-R1-2 | **closed 2.58.0** | a run of points' state read two ways (by its index, by its declaration), a refused send's run read *failed*, a failed run said *stopped without its conclusion marker* | `continuation.points_run_state`, the one rule `read_run` and `stage_run_state` read by; a run of points concludes in its points' words |
| F0c-R1-3 | **closed 2.58.0** | `--from` with `--cold` on a run of points launched again taken, `--cold` ignored | `cannot_be_named` told `--cold` where `--from` is given |
| F0c-R1-4 | **closed 2.58.0** | the summary's way out printed a `--from` launch refused where the stages differ in `already_relaxed`; chosen from the description, not the results | the two results' `relaxation.enabled` choose between the two refusals |
| F0c-R1-5 | **closed 2.58.0** | the relaxing stage read off a folder — gone, the force-constant analysis crashed | `runrecord.RunRef.stage`, the record's text |
| F0c-R1-6 | **closed 2.58.0** | the remedy for a relaxation taken over named the wrong way back and an impossible `--from` | `nonstationary_remedy(relaxation_stage, stage)`: the relaxing stage's own way back, the later stage's first launch taking the new relaxation; the cold branch with `--from` |
| F0c-R1-7 | **closed 2.58.0** | a later stage judged its reference geometry by its own `relax_force_tol`, not the tolerance the relaxation ran to | the relaxation record's tolerance (`result_of`, § 5.5's one yardstick) |
| F0c-R1-8 | **closed 2.58.0** | `--from` naming a launched run of points refused *never launched* (the run door names no run at a container) | the launch record read at the run's root |
| F0c-R1-9 | **closed 2.58.0** | a refusal printed `<that run>`; the ledger's `continues` line recorded nothing copied for a take-over; a stage prepared stated relaxed took a relaxation over by today's description | a typed `--from` line for a run that has the points done; `copied` the points' words; `carried_points` among the stage's own prepared carried points |
| F0c-R1-10 | **closed 2.58.0** | stale words in nine documents and the module docstrings; the writer took a per-channel value of any length; `SPIN_COMPONENTS` said `nspin` | the words; the writer's refusal by name; `spin%H` |
| F0c-R1-S | AFTER-SCIENCE | under `Spin.Fix`, SIESTA's optical calculation fills both spin channels against one Fermi level, `ef` — its start value 0, or the up channel's — so a fixed-spin optical frame's ε₂ is likely not the right one (`optical.F:511`, `siesta_analysis.F90:514`, `diagg.F:223`, `m_energies.F90:83`, `save_density_matrix.F90:154`; read, not measured) | a measured check — an H₂O triplet's optical single point, its `.contrib.EPSIMG` band energies against `Ef_up`/`Ef_dn` — the user's |
| F0c-R1-C | **left 2.65.16** — narrowed to a SIESTA displacement sweep's later stage; nothing wrong is said | Task setup's *Continue from* reads every run of the first stage by its points on each folder open — the cost of `points_state` per run | the results views (F1) answer a run of points' state once per answer — the views read a run of points once per answer (`continuation.read_points`); Task setup's *Continue from* and the ready door reading again → F7 (F1-R1-11) |
| F0c-R1-T | untested until after F9 | the take-over on a real run and its renamed files; the remedy's words; the summary's two refusals; a fixed-spin output's Fermi levels; `--from` with `--cold` refused at a launch again — each a real run's | the e2e, from F10 |
| F0c-R2-L | — | a taken-over relaxation's record names its output as the computing stage wrote it (`…_01_freq-run0.out`) inside the later stage's run — provenance, true of the call that computed it | — |
| F0c-R2-F | **left 2.65.13** — works: run folders are shell-safe | `commands.launch_lines` takes switches only; `--from <run>` passes as one (works: run folders are shell-safe) | flag-value pairs, as `launch_with` takes them |

## Found by F1's review rounds (2026-10-31), by the release that owns each

Round 1: three whole-file readers — floors 2 and 4 (the run door, `point_of`, the declarations, `continuation`, `atom_index`), floor 6 and the route (`runstatus`, the results and Task setup routes, the digest, the transport record), the page (viewer, picker, Run panel, file card, the transport report); my own headless check of the page on real runs beside them. Round 2: two readers over the round-1 fixes and the whole modules again — the Python floors and the documents, the page and the route — and a focused check of round 2's own fixes.

| id | release | what | the door |
|---|---|---|---|
| F1-R1-1 | **closed 2.59.0** | a point a DIRECT run's walk never started stayed *queued*, live, forever — nothing writes a run here's end, and the run door read only *failed* | the run handle's state (`runs.state_of`) reads the walk over as the run's rows do: no job of this machine's list that holds the run going (`joblist.holding`, moved down from `continuation`; `joblist.going_of`) → *not run* (finished by F1-R2-1) |
| F1-R1-2 | **closed 2.59.0** | `point_of` let an older index's refusal escape: a 2.42–2.54 vibration call's folder answered 500 | `point_of` answers no point where its run's records do not read; the run's row says why |
| F1-R1-3 | **closed 2.59.0** | a run of points read five ways (`_points_reading`, `points_run_state`, the gather's, the digest's, the transport record's), the pending words differing; the newest run not launched read twice per answer | ONE reading at floor 4, `continuation.read_points` → `PointsReading`, every reader asking it; `every_run` reuses the stage row's reading always |
| F1-R1-4 | **closed 2.59.0** | a transmission run of points' root opened nothing, then its point at work's file named as such, and the report not at that point; its ladder row opened the report at no point | the run door answers it with the report (results.md § 0.1, `CalculationKind.opens_report`); the view's point at work says it opens the report; ladder rows carry the point their report opens at (`opens_at`) |
| F1-R1-5 | **closed 2.59.0** | picking a run of points' own row off the ladder found no row for its file (it lies in a point's folder) — no presenter, a whole path in the menu | `_pickRun` takes the view's point at work's row by the answer |
| F1-R1-6 | **closed 2.59.0** | a finished run's Run panel said *at work* | *at work* while it runs, *the last to run* after |
| F1-R1-7 | **closed 2.59.0** | the view asked `run_of` per point for *holds an output*; `every_run` composed the point at work's full record and threw it away | the rows carry `output`; the view's opening and its record two steps |
| F1-R1-8 | **closed 2.59.0** | `views_of` returned no view, silently, where the calculation does not open; one stage's runs failing blanked the ladder | said where the view would be (`{unread}`, `unreadable`); a stage's runs not listing said on its row (`runs_unread`) |
| F1-R1-9 | **closed 2.59.0** | *is this a run of points* spelled four ways in Python and three on the page; *climb out of `shared/`* three ways; *a rung's folder opens the report* twice | `calcdirs.Where.of_points` (served as the place's `of_points`), `calcdirs.read_with`, `CalculationKind.opens_report` |
| F1-R1-10 | **closed 2.59.0** | stale names (`runs._file_row`, `runs.run_answer` in comments and tests), `record_dir` sent and read by none, a base name cut for the loading cover, absolute paths shown, double imports | the names corrected; `record_dir` gone; the announcement's `name`; shown from the projects folder |
| F1-R1-11 | **closed 2.65.16** with F0a-R2a-7; the stage before's run read again in `jobset_status` costs a few ms and stays | the ready door reads the stage before's newest run again inside `jobset_status`; Task setup reads it more (`task_state` twice, *Continue from*) | with F0c-R1-C: one answer's reads carried with the calculation |
| F1-R1-12 | **closed 2.67.3, nothing changed**: nothing goes wrong: the report's one join is its folder with a file in it; Task setup's are a path split for its breadcrumb, a file's name shown, a folder joined with its own hand-over's name | the transport report joins its folder to the record's junction name (`transport.js`); Task setup's page cuts or joins three paths | a composed-on-read record's paths served whole while its written copy stays relative (`summarize task`'s) |
| F1-R1-13 | **closed 2.67.1** | Reload or a tab re-entry drops a run picked off the ladder (`_currentChoice` reads the *(showing …)* row) | the picked run kept across the scan |
| F1-R2-1 | **closed 2.59.0** | round 1's point rule sat inside `concluded`, returning before the scheduler was asked: reading a point never wrote its run's end, so once `jobs status` forgot the address the run's never-started points read *queued*, live, for good; and a direct run's, its address forgotten at its exit, read *queued* where its rows said *not run*; the jobs list read 2N times per answer inside the rows, the answer thrown away | the rule on the run handle's state (`runs.state_of`), after the end is asked and written once; the walk read over as its run's rows read it — no job holding the run going (`joblist.going_of`); the rows read the state raw, as 2.58.0's did |
| F1-R2-2 | **closed 2.59.0** | *is a job holding this run going* composed four ways over `holding` + `is_going` | one answer, `joblist.going_of` — `status` (going), the run door (not going), `jobset scratch` (not over) each reading it |
| F1-R2-3 | **closed 2.59.0** | a pick off the ladder announced the file the ladder named at its last scan — a running run of points' point at work moved on, the old point's file shown under the new point's record; the pick re-derived by hand what the picker decides | the picker's one reading of a folder's answer (`readAnswer`, `metaOf`), the pick reading the fresh answer through it |
| F1-R2-4 | **closed 2.59.0** | a return to the tab moved a transmission run of points' report to its new point at work, over the point picked since (results.md § 2) | the announced point kept per folder: reading the same folder again keeps the point you picked |
| F1-R2-5 | **closed 2.59.0** | the one reading raised where its stage left the description (500 on the folder) or a point's folder did not read; the transport record's rung-level catch narrowed; the DOS reader took a run not launched's finished-looking points; a run whose launch record does not read said *pending* on its own ladder row | `read_points` says each *unreadable*, never raises; the record keeps its catch; the DOS reader skips a run not launched; the run's row takes the stage row's state whole |
| F1-R2-6 | **closed 2.59.0** | the view's `at_work.opens_report` sent and read by none; the run's opening computed twice; a point row's `.copied-from` read again; `points_run_state` a pass-through; a path as shown with three redundant fallbacks; five sentences false (a transmission run of points' folder opening *its own result or nothing*, *one answer reads the description once*, the report's point at a run of points' root, the not-run rule's *a run here*, op 16's door) | dropped, read once, carried, removed; the projects door asked as it is; the sentences corrected |
| F1-R2-7 | **closed 2.59.0** | found by the focused check of round 2's fixes: before its walk started any point, a point's folder read alone said *not run* where this machine cannot tell or holds no job of the run (a run viewed from another host, a queue it cannot ask, a copy synced in), while its run's row said *queued* | the run's row's own test read from one place: where the index stamps no start of a point the run walks itself (`runpoints.walk_began`), the walk is over only once it ended before a point (`runs.ended_before_a_point`, moved down from `continuation`) |
| F1-R1-T | untested until after F9 | the direct run's never-started point read *not run*; a transmission run of points' root opening its report at its point at work; picking a run of points' row off the ladder; the Run panel's lead — each a real run's | the e2e, from F10 |

## Found by one start per run's review rounds (2026-10-31), by the release that owns each

*(The user, 2026-10-31: "automatic retry is a residue of flat dir design. explicitly remove this. retry need re-run. period.")* Released as 2.60.0 before F2. It closes by removal: F2's B1-2, B1-3, B1-4, B1-5, B1-10 and B1-16, and the inventory rows I1-083, I1-084, I1-086, I1-093, I1-101, I2-204, D1-PLb-9 and D2-RID-1. B4-14 closes for its cold-sweep half; its walk-script half stays F9's. Round 1 had four whole-module readers, one each for: file names and records; the run script and launch; the configuration chain and migrate; and the readers and pages. One helper brought the e2e files onto the new names; they stay untested until after F9. Round 2 had three readers over the round-1 fixes and a residue sweep of the contracts, which found none.

| id | release | what | the door |
|---|---|---|---|
| OS-R1-1 | **closed 2.60.0** | the failure text keyed on the kind's `resumes` while launch decides by `Job.relaunch_continues` and never launches a trial again: a transport electrode, a PySCF vibration and a benchmark trial were told to "continue warm" | `_launch_again(resumes, continues, restart_honoured, relaunch, trial)`, prep handing the launch's own facts; PySCF prints it too |
| OS-R1-2 | **closed 2.60.0** | the wall stop's line went with the retry arms | `stopped at its wall (MaxWalltime): …` in place of the failure line, every program that takes the wall |
| OS-R1-3 | **closed 2.60.0** | the run script read its arguments twice, after the session log opened: a mistyped flag left a log, and the corrected start was then refused as a second one; `-omp --dry-run` ran PySCF for real | each engine's arguments read once, first of all (`_mb_positive`), before the guard, the gate and the log; `env_bootstrap`'s help guard gone |
| OS-R1-4 | **closed 2.60.0** | the guard's way out was a bare `jobset launch`; a dry run with a broken env wrote a conclusion | `launch_as_typed(...)` (a trial: it measures its point once); a dry run concludes nothing |
| OS-R1-5 | **closed 2.60.0** | migrate answered "retired?" three ways, and the `task@1` path wrote the retired key into the new job request | `_retired_off` / `_cards_off`, both branches and every job's resources, by the template's own rule |
| OS-R1-6 | **closed 2.60.0** | a calculation prepared by 2.59 passed migrate and then launched run scripts that refuse on the node | migrate names each group's prep (`_prepare_again`; a benchmark, its prep anew) |
| OS-R1-7 | **closed 2.60.0** | the Run panel said *Ended: running* for a live run | the record's `ended` only for a run that concluded (parse.md § 5d.6) |
| OS-R1-8 | **closed 2.60.0** | the session log read as per-start sections; a run's file in a role chosen first in one reader and last in four | `read_wrapper_log` → the start's one dict; one rule, first by name, at the run door |
| OS-R1-9 | **closed 2.60.0** | `continued_from_text`, `summarize._at_run`, `point_currents`' dead parameter and second read, makov_payne's no-op lookup, `run_output`'s literal `.out` | each collapsed into its one door |
| OS-R1-10 | **closed 2.60.0** | catalogue help (`restart`, `write_md_history`, `write_md_xmol`, `mpi_np`), `GROUP_WORDS`, `migrate --help` and some 30 contract sentences stated the removed design | rewritten as the rule stands |
| OS-R1-11 | #361 | a start that dies in the machine's own setup step writes no conclusion and reads *queued*; a requeued job's second start refuses before any log (runs.py's requeue picture); a start by hand in a never-launched trial folder | the whole-jobset review |
| OS-R1-12 | **closed 2.67.3** | the prep answer's resources line: the terminal prints the memory cap, Task setup's line does not | the composed line served once |
| OS-R2-1 | **closed 2.60.0** | a count from the environment (`OMP_NUM_THREADS=4,2`) refused after the session log opened: the start used up, no conclusion, read *queued* for good; a missing program the same | the resolved count refused naming its source (`_mb_positive NAME VALUE SOURCE`), and the run concludes; a missing program concludes; a dry run concludes nothing |
| OS-R2-2 | **closed 2.60.0** | a trial's usage said "launch the stage again" beside its guard's "measures its point once" | `_one_start_usage(trial)` |
| OS-R2-3 | **closed 2.60.0** | the two engines' argument parsers written twice; the failure line rendered three times; the launch's facts defaulted where a missed pass would hide | one `_args_parse`, one `_again_line`; `resumes`, `relaunch_continues`, `trial` required |
| OS-R2-4 | **closed 2.60.0** | migrate named a benchmark's way back when every trial was launched, and before nothing undid the root's prep | named only where a trial is not yet launched (`_plan_trial`'s test), printed first |
| OS-R2-5 | **closed 2.60.0** | *is this item retired* in three shapes | `template.retired(name, engine=None)`, the one door migrate, `Resources.from_dict` and `why_not_an_item` ask |
| OS-R2-6 | **closed 2.60.0** | a requeued job's second start refuses, and no contract said so | running-a-job § 4.2, project-layout § 1.6.1 |
| OS-R2-7 | **closed 2.60.0** | the record's outputs spelled by hand and the speaking output chosen beside `run_status`'s; dead fallbacks; `log_of_run`'s label-only case; `.validation.txt` spelled three times; a dead `engine_version` reader | `ROLE_MOLWATCH` / `stdout_roles`, `active_source`; `status` and `engine` required; `stage` required; `runfiles.ROLE_VALIDATION`; the row gone |
| OS-R2-8 | left | a renamed or engine-scoped retired item on a run card or in resources promises what migrate does not do | only a handcrafted file reaches it (the user's rule: build for what molbuilder writes) |

## Found by the whole-jobset review (#361, 2026-11-01), by the release that owns each

*(The user, 2026-11-01: "remember code disciplines, and framework/api unification rules for any work"; on older runs, "Migrate renames single-start runs"; on a start by hand in prep's run, "Adopt it as a run", then "if this takes too much coding, then we can go with 3" — and 3, only `jobset launch` starts a run, was taken.)* Five read-only reviewers over the whole jobset after one start per run: prep; a run's life and its states; launch and continuation; structure and duplication; records and truth. Each finding was checked in the code before it was planned; its unit is the door that owns it. Release A (2.61.0) closes what decides how a run's end is known and how an older run is read; Release B (2.62.0) the rest of the defects and the ways out composed once; later F releases what their plan rows already own.

| id | release | what | the door |
|---|---|---|---|
| L1 | **closed 2.61.0** | a run here that died before its engine read *queued* for ever, one killed outright *running* for ever; a run waited on lost its address before its launch read how it ended | the run script's one exit trap concludes a failure of its own (`stopped before its end`), the stopping signals caught by one handler; `runs.concluded` asks a run here's process (`joblist.going_of`) and writes its end once (`record_ended`), read by `parse.dirs.job._job_ended` — *never started* for a run whose start never began; the launch removes a waited run's address once it has said how it ended, waiting first for the run script's close when stopped at its terminal (`submit._ended_within`) |
| L2 | **closed 2.61.0** | a requeued walk ran, in the same run, the points its first start never reached | the walk refuses where a point it runs holds a start (`submit._walk_script`'s `started`, `identity.next_run_words`) |
| L3 · L4 · L8 | **closed 2.61.0** | a point started and silent read *queued* (its rows, *not run*); the first point's step before its engine read *queued* | `parse.dirs.job._build_status`: a start with no output yet is *running* (*started, no output yet*), read from its session log; the point's start stamp follows its log, so its rows and its folder read alike |
| L5 | **closed 2.61.0** | past its wall never asked of a run with no output; prep's and launch's files counted as the job's writes | asked once a start began; `_the_jobs_own` by the catalogue's `when` and `output` |
| S3 | **closed 2.61.0** | the stage row decided a run of points by the description now, every other reader by the run's folder | `runstatus._job_status` by `calcdirs.where(run).of_points` |
| R1 | **closed 2.61.0** | a run named before 2.60.0 read *queued* for ever, its files invisible | `runfiles.older_named` (the one reading of the older names); the run door reads such a run `unreadable`, naming `jobset migrate` (`runs.older_run`), which names a run that started once anew, every byte kept, and leaves one it cannot with one of five reasons (`migrate._migrate_runs`, `project-layout.md` § 1.6.1) |
| R2 | **closed 2.61.0** | migrate dropped a pre-2026-10-02 copy's `script_generation` | carried into `env_init`, two values refused by name |
| R5 | **closed 2.61.0** | the plain-text run records read three ways | each reader refuses a record that does not read; a gather that took nothing writes none |
| R6 | **closed 2.61.0** | the monitor's closing record read two ways | `job.MONITOR_ENDED_LINE`, the one reading, which the instrument parser imports |
| OS-R1-11 | **closed 2.61.0** | a start that died in its setup read *queued*; a start by hand in prep's run | L1; A1 |
| A1 | **closed 2.61.0** | a start by hand in prep's never-launched run — a run no record of molbuilder's placed | only `jobset launch` starts a run: the run script's gate takes the launch's claim alone (`identity.started_by_launch`), the `manual` claim and the terminal's question removed; adopting a hand start was set aside as more code for the same end (the user: "if this takes too much coding, then we can go with 3") |
| R8 | closed, no action (2.62.0) | `joblist.holding` matches by run folder only | reached only by restoring a saved state while a job still runs in that folder and launching into it again — against the design, not molbuilder's road |
| P1, R3, R4, S4 | **closed 2.62.0** (S4's read per stage left: a small file, and a cache would be a second state) | `prepared_with` compares only keys both sides hold; settings at dataclass defaults; schema unchecked | `prepared_with.differences` over the run card's stated keys; run settings left to the run card (`template.run_settings`); the schema checked, a record that does not read the stage's *unreadable* answer and prep's refusal |
| A2 | F3/F4 (minor) | launch builds a transport rung's points from the description, not from what prep wrote | reached only by launching after `status` says *changed since prepared — prepare it again*; prep's record of the points comes with the record work |
| A4, A9 | **closed 2.62.0** | a sequence's header left stale by a prep of one of its groups; a run of points recorded as continuing from a run it took nothing from | `group.sequence_headers_holding`, taken away by any prep of the group but the header it writes; the run-level hand-over written by `submit._points_walk` where it takes something, R-L17 for points that would continue |
| A5, A8, A10 | closed, no action (2.62.0) | a launch again while the last run still runs; a warm `--gather` taking over points computed from the old inputs | A5, A8: the contract settles it — both run (`job-system.md`, *What stays yours*); A10: a rare corner, no failure on the road |
| A11, S7, S8 | **closed 2.62.0** (the bench way back and the next step already one source each) | ways out composed by hand | `identity.jobs_as_typed`, `migrate_as_typed`, through every refusal and line |
| S1, S11, R10, P6 | **closed 2.62.0** | the transport describe checked two ways; the earlier run searched twice; an unreadable machine copy swallowed at `summarize`; a prep refused after its plan without its notes | `transport.deck.describe_findings`; `continuation.earlier_run`; the copy's refusal said; prep's one `_refuse` |
| P4 | left (2.62.0) | `_computes_else` compares the stage folder's deck | the same calculation as the deck its runs copied on every road molbuilder writes; only a hand edit tells them apart |
| P3 | **closed 2.67.3** | the page composes its own resources line | the pages' words |
| S9, S10 | **S10 closed 2.65.16** (`materialize.runs_launched`; the rest read and left); **S9 left**: one speller, one number | the bench answer derived in four places; web routes re-deriving door answers | the floors and the objects |
| R11, R13, S13, P8, L9, A-doc | the jobset documents' consolidation (after 2.62.0; L9's code part closed 2.62.0) | document drift | the owning chapters |
| S2, S5, S6, S12 | F3 | a kind's code in shared modules; two deck renderers; floors | the kind table |
| R12, R7, R9, R14 | F4/F5 | role literals; records read past unknown keys | as planned — R12 **closed 2.65.5** (a `ROLE_*` at every site); R9 **closed 2.65.7** (the plan's four readers refuse a key they do not read; the six others stay as they read, the user's answer); R14 **closed 2.65.7** (`runrecord.RUN_RECORDS`); R7 **closed as designed** (`placed_on`: two facts, two sources) |
| S14 | later | unused parameters, long functions | review |
| S15 | **closed 2.62.0** (the waits → F9: three contracts) | `Kind.line(launched_by=)` has one caller, which passes the constant; `road.until` stands beside two private waits in the test files (the round-3 review) | the line filled from `LAUNCH_CLAIM`, as its variable is |
| L6, L7, A6, A7, P5, P2, R-design | listed for the user | design notes, no defect | — |

**Found by Release A's own review rounds**, fixed in it: the exit trap writing an exit code 0 for a hang-up, and stopping the monitor before its marker; a group's stop said as a broken pipe; a never-started run said *stopped*; a Ctrl-C read before the run script had closed; a finished run's launch words never shown; migrate's kept files of a second try, two files taking one name, a calculation reached through a link, and a failed rename written twice in the ledger; a failed write said on the state first read; a dry run a person starts dying on its verdict line; a sequence's runs left unread at a Ctrl-C; the stale words in the run script's comments, the catalogue's help and the chapters that described a start by hand.

**Found by Release B's review**, fixed in it: the R-L17 refusal asked whether a job has warm files, where a transport rung's points start over and take none (now: a walked point that would continue); a `prepared-with.json` that did not read led status to the way back of a changed launched group (now the stage's *unreadable* answer); a sequence's header left stale by another sequence's prep of a group of it; a run handed its last point back said to continue; the hand-spelled `jobs`, `migrate` and checkpoint lines it missed; the describe checks assembled three times. **Its second round**: the unreadable record said as a launch record that does not read, a never-launched group counted launched, a launched stage's *launched* dropped (now its own answer, `record`, the group's step *unreadable* with no line over it); a record that is not text named like any other; the `jobs stop` spelling in the folders' audit.

## Found by the jobset documents' consolidation (#364, 2026-11-01), by the release that owns each

*(The user, 2026-11-01: "holistically review the jobset module's related document to reduce overlaps,
redundancy, inconsistency and duplicated parts, consolidate to make them complete, logical, readable, and
developer and user friendly"; "a holistic revision … not just patching"; "review after the document is
revised to make sure information are consistent, correct and up-to-date by validating against code.")*
Twenty chapters rewritten to one home per fact under an ownership map agreed with the user, `overview.md`
made the one route in, the README's rows renewed; two review rounds against the code, each finding checked
before it was fixed. Docs only — the code text the review found is 2.62.1's.

| id | release | what | the door |
|---|---|---|---|
| R11, R13, S13, P8, L9, A-doc | **closed (docs)** | the drifting document sentences the whole-jobset review set aside | the owning chapters, rewritten: architecture § 3.2 (every door in use), project-layout § 1.6.3 / § 2 (the first prep's machine copy, run.json's moment, the session log at *running*), job-system § 5.0 and § 6.0 |
| D-SEQ | **closed 2.62.1** (the user, 2026-11-01: "yes, add it in 2.62.1") | a save made by a sequence's job says so in its note | `commands.launch_save_note`, read back by `commands.names_launch_of`; checkpointing § 9's note row |
| D-TXT | **closed 2.62.1** | code text the chapters now contradict: user-facing (`describe` / `jobset` / `prep --target` / `launch --mode` / `migrate` help, the catalogue's `use_gpu` and `gpu_count` help, the refusals at prep_inputs.py:665, group.py:231, engines.py:245, submit.py:322, the walk script's and every `.sbatch`'s cited section, runwrap's prep note on the GPU route, `_launch_again`'s "not done") and comments, docstrings and test cites (≈90 sites once swept: § 5.3 → § 5.0, job-system § 6 → scheduler § 6a, *The task* numbers → § 5.0 checkpoint 2a, running-a-job § 5.x → configuration § 4, project-layout § 2.3.x → generator, "fifteen" → sixteen, "task.json" → job-request.json, a killed trial "incomplete" → failed, `traits_for`, `AmbiguousTarget`, `identity` docstrings, `runfiles.WRITTEN` writer rows) | each owner, one release |
| D-CAP | **closed 2.62.1** (verified: 1–3 bytes over admitted) | `identity._cap_for` caps the label by the stage name, not its `NN_` token — 3 bytes too generous against the 255-byte name rule run-identity § 3 states | `identity._cap_for` |
| D-WORD | **closed 2.62.1** | a point's state said in more than one word across the printers — *not done* for a failed point, *queued -- queued* in `status`'s rows | `runpoints.state_word`, asked by every printer; runs-of-points § 1.3 |
| D-A7 | **the floors half closed 2.65.15** (B3-11; `mode_frames` left: only surfaces import it, and `structure_path` in `task` would make the core layer import a higher one; `describe`'s two asks are the contract's now, its sentence corrected); the engine-name half closed by F2 | upward imports and engine-name branches the architecture rules A7 / A18 forbid (`mode_frames`, `gather`, transport/stages, parse, runs, commands, submit; runwrap's engine names) | the floors (F7); the run script naming no engine (F2) |
| D-SENT | left | `notified.json`'s `sent` lists every event said, a channel taking it or not | run-reports § 2 says what the key records; a rename would be a schema change no road needs |
| D-BANNER | **closed 2.63.0** | the run script's banner, header comment and failure line still say *wrapper* (`runwrap.py`; `running-a-job.md` quotes the banner) while the deck's text says *the run script* | the run script F2 rewrites to name no engine |
| E2E-MON | **closed after 2.62.1** (the user, 2026-11-02: "those two failures seems to be an outdated tests") | `test_monitor_watches_a_live_run_e2e.py::test_every_file_that_ships_beside_a_job_reads_the_run_without_molbuilder` failed: `AttributeError: 'Run' object has no attribute 'run'` — it still handed the monitor the try number 2.60.0 removed (`WatchedRun(run=…)`) | the test: the monitor told the run by its names alone; passes |
| E2E-STOP | **closed after 2.62.1** (the same) | `test_siesta_stopped_run_e2e.py::test_the_run_panel_says_what_ran_and_why_it_stopped[chromium]` failed: `FileNotFoundError` on a path with the calculation's folder doubled — it read the ladder button's title as a path from the calculation, where since 2.59.0 the page names a path it only shows from the projects folder (`web/results.md` § 0.3) | the test: the run's folder from the projects folder; passes |

## Found by the codebase overlap review (2026-11-09), by the release that owns each

*(The user, 2026-11-09: "do another agent review … file/path access, structure access, persistence, result parser, template related operations, molview/other plot related work, UI module … screen through document/design contract to see if there are potential handcrafted code/duplicated code that repeat existing modules". Ten readers over the whole codebase — the seven areas named, a sweep from every owner the contracts declare (about 280, every one resolving), the execution side, and the engines and science; each reader's top claims read again against the code before they landed here. Rows are grouped by the door that should own the job; the readers' file-and-line detail is kept with each row's release notes.)*

**What is wrong now, not only duplicated** — each its own small release, first:

| id | where | finding | door |
|---|---|---|---|
| V-VAC | **closed 2.69.13** | `molbuilder describe --vacuum` sets the box past the one door (`jobset/_cli.py:651` `struct.replace`): no periodic-axis refusal, no non-negative check, no return to a derived box — a structure stating its own cell silently overrides the vacuum typed | `--vacuum` is the Cell page's `vacuum` op through `periodicity_gate.apply_edit`: refused while an axis is periodic, a typed cell given up and said (`structure-periodicity.md` § 6.2); as on the Cell page, a run-exported structure's recorded charge and spin are then not taken (ES7) |
| V-RELAX | **closed 2.69.13** | two answers to "did the relaxation converge": the continuation compares the largest free-atom force with the tolerance (`continuation._relaxed` → `relaxation_of`), every other reader the engine's own verdict (`verdict_of`) — they can disagree | the status column's words from the same reading — `RunStatus.converged_words`, a run of points' `PointsReading.converged` (the column's one rule, moved from `runstatus`) — (`parse.md` § 2b P-S4), the run's own row on the Results ladder by the same rule (it said nothing beside its stage row's); `relaxation_of` stays the `info.relaxation` record; the `verdict=` switch, which only spared that second parse, gone |
| V-HYD | **closed 2.69.13** | the nucleic builder's automatic hydrogens count `e == "H"` over labels (`nucleic.py:287`): a structure whose hydrogens are labelled `H1`… is counted as having none and gets hydrogens again | `chemistry.is_atom` (`model/chemistry.md` § 3). **Measured**: both callers hand it the builders' own output, every element a symbol, so no road reaches a labelled hydrogen; the change is the contract's one way to ask, untested; the docstring's *user-loaded structure* (no caller passes one) gone |
| V-LAT | **closed 2.69.13** | the TranSIESTA deck's geometry header is its own copy (`transport/transiesta.py:250`), the lattice at 8 decimals where the SIESTA deck writes 12, the incomplete-species refusal twice in two wordings | `siesta/deck_lines.py` writes the lattice, species and coordinate rows for both decks; an order that omits a species is refused by the one entry, `chemistry.species_order`, in one wording; `chemistry.md` § 3a's *caught at the deck gate* corrected (the gate checks a written deck's indices, which no writer can write without the species) |
| V-PSML | **closed 2.69.13** | one prep judges the pseudopotentials twice — the copy step's own screen (`siesta/seam.py:189`) omits the relativistic check the gate makes | the settings gate at step 3.3 of every deck, reading the folder as the plan leaves it, after the copy; `seam.screen_pseudos` and its three calls gone |
| V-TYPE | **closed 2.69.13** | a wrongly typed PySCF integer or triple reaches the range comparison and reports a "programmer bug": the type is checked in four places, SIESTA's by class validators and PySCF's by none | `config_from_template` judges every value by today's catalogue through `_type_refusal` and passes it on as the catalogue spells it (`template.md` § 5.4); the gate's own type checks gone — its enum membership, its *programmer bug* catch (a range now warned through `outside_range` alone), SIESTA's `_validate_kgrid`, `_validate_block_size`, `_validate_kgrid_displacement`. **Measured**: the road is a hand edit of an item's `type`, `value` and `default` together |
| V-REC | **closed 2.69.13** | stated values become a config two ways: the command line lays the recommendations OVER what was typed, the form UNDER it; whitespace is blank in one reader and "you set this" in the hand-over's map | `template.stated_config(cls, kind, (values, who), …)` → `(config, sources)`, asked by the form doors, the hand-over, `molbuilder describe` and a transport describe (its citation's answers a layer under the person's, its own map of who set each gone); `_shared.config_from_params` gone. **Measured**: the four builders agreed on every road — the command line states only `psml_lib` and the name, which no recommendation names, and the hand-over's own map read a field of spaces as the person's, which no page sends (`collectField` trims a text field; a select sends a member or null); the one door is the change, untested by a test of its own. Two differences on the citation, through a hand-made request alone: a `system_label` among the shared values is written as the calculation's label (one name, the identity field's), and a person's value equal to the cited one in another type (300 against 300.0) is the cited one, its source *cited* as before |
| V-PATH | **2.69.14, 2.69.15, 2.69.16** | a rename gives a calculation or project folder a name New folder refuses (the project list then skips it); `molbuilder describe` never asks "a calculation lives under a topic"; the structure save and the frame-set write skip the write-path checks (the root, a symlink); the browser's "inside the projects folder" is a bare prefix, so `projects2` counts | **2.69.14**: a folder renamed takes its depth's name (`projects.validate_folder_name`, mkdir's too); every describe asks the depth rule (`describe.refuse_above`), `molbuilder describe` among them. **2.69.15**: the structure save and the frame-set write ask the files layer's one write-path check (`files._writable`), as the upload and the write do. **2.69.16**: inside a folder is the folder or below it, never a bare prefix (`path-utils.relative`), asked by the sidebar's start, its restore, the notebook tab and the projects door's address; the tree picker never offers the projects root as an answer |
| V-READ | **2.69.19** | a record with a byte-order mark reads through `read_json` and is refused through the prep plan (`Plan.read_text`, plain utf-8); `sequence.read_record` lets a crash through on a non-object file; the workspace lets one tag's save land in another's slot | **2.69.19**: one decode of what molbuilder reads (`persist.text_of`, `json_of`), the plan's text and JSON reads asking it (`Plan.read_text`, `Plan.read_json`); a record file holds one JSON object (`persist.read_object`), the sequence's record read by it — the other ~10 record readers move to it with X-PERSIST; the workspace id one spelling per character, two tags never one id (`workspace.md` § 4) |
| V-PANEL | **2.69.19** | the run panel reads SIESTA's printed `T`/`F` itself (the grammar stores them raw); the transport report decides "not converged" from its display words; two readers of the progress log's atom-index line, one raising, one dropping | **2.69.19**: a printed logical read once (`parse.fdf.fdf_logical`), the grammar storing `parallel_over_k` as a bool and the panel's own reader gone (`elpa_gpu` is SIESTA's GPU word, `no-gpu`, shown as printed); the cited run's verdict served as data (`provenance.slot.relaxation.converged_ok`, its kept words read back by their writer, `_run_ending.verdict_read`); one reader of the atom-index line in the format's grammar, a stated map that does not read refused by both, the job monitor leaving that log's facts out of its wake and reading on (`run-reports.md` § 2.3) |

**Where two contracts disagree — the user's to settle before the code can follow** (each with the reader's recommendation in the release notes): the structure answer's shape (`web-api.md` § 1 vs `structure.md` § 1); a saved `.xyz` read alone or as a pair (`parse.md` § 4 vs `structure.md` § 2.4); the browser's own typed readers (`template.md` § 5.4 vs `form-schema.md` § 1.1); the force-constant sweep record read in the browser (`results.md` § 0.3 vs § 0.2); a one-note call for the workspace (O-STATE's door was the copies); a calculation folder's records written by one transaction or by save-then-two-writes; display preferences in browser storage or the workspace; a chart's picture export; the viewer's Selection/Cell/Metadata switch as a tab strip; the browser-built CSV of a lone log; the host's cores and GPUs (`architecture.md` names `diagnostics`, the code `scheduler.record`); `scripts/bench-siesta-blocksize.sh`, a second benchmark lane no live doc names.

**Settled by the user, 2026-11-09** ("Accept all, defects first"): (1) a structure answer is the one structure object — the shapes no page reads go; (2) a saved `.xyz` is read with its sidecar; (4) the form reads only a number's and a triple's syntax, the server judges the value; (5) the force-constant sweep record is read on the server and served; (6) the workspace gains one call for a page's one note; (7) a calculation folder's records are written by one all-or-none step, Save's too, after its checkpoint; (8) display preferences in browser storage through one helper, a page's state through the workspace; (10) Plotly's camera button off on the charts the pages draw; (11) the viewer's Selection/Cell/Metadata switch is the shared tab strip; (12) a trajectory's CSV through the server's export by kind; (13) the machine record owns the host's cores and GPUs (`architecture.md` corrected); (14) `scripts/bench-siesta-blocksize.sh` retired. The defects above go first. **(3) and (9) withdrawn the same day, my error in posing them:** they put VibrationView's export beside MolView's — its GIF/WebM/zip code to import shared writers, the Spectrum tab's export dialog to be MolView's. A sealed module carries its own tools whole (`vibrationview.md` § 4: it reaches nothing else in the app, and still exports with every other web module deleted; `molview.md` § 4: one entry point, its dialog internal). A copy across that wall is the seal, not a duplicate, and was never a disagreement to ask (the user: "two independent modules"). For the same reason (10) leaves a sealed chart module's own buttons to that module.

**Duplicates by owning door** — one release per door, the contract first, the copies deleted, a discipline reader before each commit:

| id | where | the duplicates | door |
|---|---|---|---|
| X-PATH | **2.69.14, 2.69.15, 2.69.16** | the tree address composed by hand ~8 times and its reverse ~7 (`projects.contain`/`address`); "must exist / be a folder" ~9 ways with 400 and 404 mixed; two names for the one root (`file_picker_roots` vs `projects_root`); the pair moved and copied by two mechanisms; the structure-file suffixes in five browser copies; basename, join and inside-root by hand ~20 times (`path-utils`, not importable today); `[A-Za-z0-9_-]` in five expressions | **2.69.14**: `projects.at`, and `projects.segments` with `address`, for the tree address both ways (the root resolved before joining); the fence's own reason said (`OutsideRoot.outside`); the one root by its one name (`projects_root`; `file_picker_roots` retired); the name characters' one home (`runfiles.NAME_CHARS`); a project made whole by one door, `projects.populate_project_skeleton`, which New project, `molbuilder describe` and `ensure_structure_dir` ask; every describe holds the folders it makes to their depth's name (`describe.refuse_place`). **2.69.15**: the one must-exist answer (`files._existing`: 404 not there, 400 the wrong kind) for every route; the one pair operation for rename, move and copy (`files._pair_to`). **2.69.16**: `lib/path-utils.js` the browser's one path door — an ES module the page header loads, mirrored on `window.molbuilder.path` — every page's name, folder, join and inside-a-folder through it (MolView's own kept: sealed); which files are structures the server's pairing rule, stated by the page header and asked through `projects.isStructureFile`, the seven suffix copies gone (the lattice picker's `.xyz` is its contract's own) |
| X-STRUCT | **2.69.20** (the reads), **2.69.21** (the structure sent once), **2.69.22** (the trajectory route), **2.69.23** (the run's record once), **2.69.24** (the PySCF continuation's read) — **closed** | the text read beside the codec (`_shared.structure_from_text`, 2.69.10's — it drops the comment-line rule) and two more; the pair re-read 2–4 times for one fact (`atom_index`, `postprocess`, prep); the sidecar's forward lookup rebuilt in seven places; one structure in up to four shapes per answer (decided, the user 2026-11-10: *one shape, sent once* — the answer carries `structure`, its notices and labels, and the viewer's translator reads only that); a saved pair's `.xyz` read alone by the registry's `xyz` reader — `postprocess.parse`, the command line's trajectory verbs, the trajectory route handed one; the Results tab's structure preview already reads it whole (decided, the user 2026-11-10: *whole there too* — a `.xyz` with its sidecar is read through the one structure read, a run's own `.xyz` alone); the trajectory route's side-blocks the frame-0 envelope carries; the run's `info` worked out twice | **2.69.20**: the codec's one text read (`StructureCodec.read_document`), which `load` and the paste and upload (`_shared.structure_from_text`) ask, so a record's order words never ride a title; a `.xyz` with its sidecar read whole by the registry (`parse/coords/pair.py`, `structure-pair`), the `xyz` reader keeping the files no sidecar is beside, and `postprocess` reading no pair again for its order record. **Measured, no change**: the CLI's stdin read goes through `load` (a temp file, its notice naming it); the atom-index door's pair read and prep's three reads ask one door one question each — the map, the witness hash, the order notice — and the door also answers a sidecar whose `.xyz` is gone, so a map handed back from `load` would add a branch, not remove one; the forward lookup closed 2.69.17 (`molstruct.paired_sidecar`), the remaining `sidecar_path_for` calls naming a destination. **2.69.21**: every answer carrying a structure is one shape (`_shared.structure_answer`): the structure's own dict once, beside it only what the server derives (`Structure.to_wire`: `resolved_box`, `atom_order_said`) and the kind table's labels; the per-atom rows, the flat box, channels and identity columns, the text, PDB copy, format, summary, issues and `extra` gone; MolView's translator reads the structure, the cell door answers the same shape. **2.69.22**: the trajectory route's load sent the run's labels, box and `info` twice — in frame 0's envelope and as three blocks beside it (`atom_metadata`, `periodicity`, `info`), of which the page read only the box, to say it came from the run; the envelope is the one carrier, the page asks it, the blocks gone. **Found by 2.69.22's review**: a load answered before a live run's first frame holds no envelope, and the polls that bring the frames leave the envelope out (keep), so the viewer says the geometry could not be assembled until the next load — older than this release. **2.69.23, decided** (the user, 2026-11-10: *build it*): the run's `info` worked out once — the codec's read of a file a run wrote (SIESTA's own `<label>.xyz`, PySCF's `_optimized.xyz`) applies the run's full record once, read from the run's output; the Results tab's structure preview stops asking `/api/results/contract` and writing over the viewer's `info`, so a person's saved pair keeps its own record; the route retires. **2.69.24**: a PySCF continuation takes its kept geometry's coordinates through the codec (`StructureCodec.geometry_of`) beside its own deck's labels (`pyscf.input.emit_continuation_read`) — found by its review: the old read took the file's bare elements for the labels, losing an ECP keyed by a label. **Found by 2.69.20's review**: a PySCF continuation read its `_optimized.xyz` pair with `Structure.from_xyz` (`pyscf/input.py`, the script's head) beside the codec, which travels in its bundle. *(2.69.23's review also said the kept pair's sidecar carries the input's `info.relaxation`; measured, it does not — `pyscf.input._sidecar_for` leaves the input's `relaxation` and `calculation` behind, saying why — so no change.)* `web/app.py` sets `JSON_SORT_KEYS = False`, which Flask 3 does not read, so every answer's object keys are sorted (found by 2.69.21's review: an atom's labels reach the page in name order) |
| X-PERSIST | **2.69.25** (one record reader, one writer with the plan), **2.69.26** (the workspace's calls), **2.69.27** (a page's own layout), **2.69.28** (everything a page keeps in the browser through one door) — **closed** | "the plan or the disk" chosen by hand in 11 writers and 3 readers; the record reader composed ~10 times; four ways to write several files into a calculation; the pair as one unit in four mechanisms; `constants.js`'s storage keys read by nothing; a hand copy of the contact-distance table in the viewer | one road through `Plan`/`persist`; one record reader in `persist`; the codec for the pair. **2.69.25**: `persist.read_object(path, schema=…)` (`object_of` its core, `Plan.read_object` the plan's) the one record reader — twenty-five readers on it, the review's ten and the rest found by measuring and by the release's two review rounds (the description and job request, the plan's door, the machine record, the channels file, the composition's provenance at any major, `molbuilder.json`, migrate's raw read, among them); `write_json`/`write_bytes` take the plan, the eleven hand branches and five hand-composed JSON writers gone (`sequence.record_text` removed, `calcdirs.record` writing into its plan), `Plan.read_json` removed. **Measured, no change**: the composition record's temporary folder (`transport/rung.written_and_read_back`) is the codec's own round trip — the codec writes the pair to disk, so its bytes go into the plan after; `pseudos.copy_pseudopotentials` is a copy (`Plan.copy` stamps its source), not a writer's choice. **Found by 2.69.25's reviews**: a deck's USER-CUSTOM zone is read two ways — through prep's plan strictly (a byte not UTF-8 refused), by the Build tab's regenerate and the transport *deck now* (`merge_user_custom_from_target`) with a replacement character for it. **2.69.26, decided** (the user, 2026-11-10: *a draft call, every call naming only its owner and step*): the workspace names every file from the tag — `persist(tag, bytes, step)`, `readState(tag, step)`, `pruneStatesAbove(tag, step)`, `persistDraft(tag, bytes)`, `readDraft(tag)` — so no caller composes an identity or reaches another tag's file, and a `:draft` tag is refused; the seven one-slot notes are their tags' drafts; the tail delete goes through the write queue (§ 9.2 — found by the release's review: a slow one could erase the next point). **Found by 2.69.26's review**: `test_molview_mount.py`'s module stand-in (`{read, write}`) has never matched the workspace's surface, so its mount tests run with history and the view lane off — with the stand-ins' collapse into one shared door, 2.67.12. **Decided too** (the user, 2026-11-10): display preferences split by what they are — how you look at a structure or a spectrum beside the work, a page's own layout in the browser through one helper and one key prefix (2.69.27); the contact table's one home the viewer's, the unread data file to go with the other dead code (not a release of its own: nothing is wrong for anyone, the user 2026-11-11). **Measured, no decision**: `constants.js`'s storage keys are unread copies of `projects/state.js`'s (2.69.27). **2.69.27**: a page's own layout through one door, `pagePrefs` (`lib/page-prefs.js`; `runtime.md` § 3 the rule) — the eight hand accessors in six files under five key prefixes on it; `constants.js`'s unread storage keys gone. **Settled** (the user, 2026-11-11: *keep persistency within each tab* — *unified API, no handcrafted reinvention*): the Structure-optimization tab's two forms are kept in each tab, through the one door (2.69.28). **Dropped 2026-11-11 as seen only in the code — wrongly**: a workspace save still waiting its turn when the page is left is never sent; 2.69.29's browser test saw it (below). **2.69.28** (the user, 2026-11-11: *unified API and no handcrafted reinvention of wheels*): the last two hand-written stores in the browser — the sidebar's place (`projects/state.js`) and the Structure-optimization tab's forms — go through `pagePrefs`, which keeps an empty value (a cleared pick never falls back to another tab's) and says when the browser refuses. **Taken out as nothing anyone needs** (the user, 2026-11-11: *detect any invented problem*): the pair moved and copied four ways — each works, no failure measured; and the review's low rows (three event logs, a README written then made private, a test seam, two conda readers, three zip makers, the channels file's writer beside its reader) — none breaks anything. **X-PERSIST closed with 2.69.28.** **2.69.29, decided** (the user, 2026-11-11: *same module, one set per tab*): the workspace names each browser tab's files for the tab, `ws-<tab>-<tag>`, the tab's id kept in the tab (`pagePrefs`); closing a tab ends its set, a duplicated tab shares the original's. **Found by 2.69.29's browser test**: since 2.69.26 (its one queue for every save, the delete of abandoned points in it) an edit's draft waited behind its point and that delete, and a reload right after dropped it — the Molbuilder tab showed the file, not the edit, two times in three (2.69.25: never); one queue per file fixed it (`workspace.md` § 9.2). A write still in flight as a page is left can still be cut; no test has seen it since. |
| X-PARSE | **2.69.30** (the progress log read by its one pass), **2.69.31** (every engine file through its registered parser), **2.69.32** (a run's state and engine through the run door), **2.69.33** (a call in one wording; a file's version by the ending reader's rule) — **closed**; **2.69.39** (§ 5a rule 4, left there) | PySCF's SCF cycles read from two files, the unit conversion twice; the progress log's header pass rebuilt; second doors onto files (`MolwatchLogParser`, a "legacy" re-wrap); "which engine ran" composed twice; a run's state asked past `runs.state_of`; the call words rebuilt on the page ; the `.XV` reader lets its own refusal (`SiestaXVError`) and an `OSError` escape unwrapped, an HTML 500 where the registry answers a `ParseError` (found by 2.69.20's review) | the grammars (`parse.md` § 4a); `runs`; served words; `parse/coords/_helpers`. **2.69.30**: the progress log read one way — its one read, `molwatch_reader.read_log` (the twin of `siesta_reader.read_output`), which the registered parser, geomeTRIC's trajectory beside the log (`pyscf._read_molwatch_metadata`, its header walked by hand) and the atom-index door (`atom_index._of_log`, its header walked by hand) read through; K269 had kept the two hand readings (nothing goes wrong), this review's *unified API* took them; a companion log that does not read leaves the trajectory read as itself alone (§ 5a, rule 2). **Found by 2.69.30's review — a defect against the pass's own rule**: a run stopped while its step's SCF table was being written left the table open, and the pass read the footer its exit hook wrote as an SCF row — the log read *running*, its error unsaid, while the ending reader said *stopped*; the footer now ends the table. **Kept, measured**: the timing rows (`scf_timing_rows.progress_log_timing_metrics`), on the grammar's row reader — the monitor asks them every wake, and the pass costs three times as much (38 against 116 ms on a 2.5 MB log the log's step writer wrote for a stand-in molecule, outside the suite — no run's); the engine sniff (`contract._declared_in_molwatch`), the log's first 40 lines, as the registry's sniff reads them; PySCF's SCF cycles from its own `.log` (`pyscf._read_scf_history`), the contract's companion for the trajectory's SCF history (`parse.md` § 4, the four logs' table). **Left, pre-existing**: § 5a rule 4 (*say what happened*) — the trajectory states no word of the log it took its facts from, nor why it took none. **2.69.31**: the second doors gone — `SiestaParser` folded into its registered parser, `MolwatchLogParser` deleted, the command line's `trajectory_to_legacy_dict` re-wrap and the route's alias of the converter's name gone; `openable_in(calculation=)` gone. **Dropped as invented** (the user, 2026-11-11: *`.xv` is only used in notebook code*): the `.XV` refusal as an HTML 500 — no route opens a `.XV`. **2.69.32**: the transport composer asks `runs.state_of` (its hand `older_run` and `run_state_of` gone), the Transport tab's words for a cited run with no conclusion the door's (*still running, or force-stopped (identical on disk)* was false once the job had been asked); *which engine ran* one door, `runs.engine_at`, the folder answer, the trajectory route and a run's view (its relaxation record's engine) asking it; the composer's refusal for a run with no conclusion and the exit-code refusal's way on (*run it to its end*, impossible — a run folder holds one start) the door's and § 3.1's. **Decided by the user from 2.69.32's review** (2026-11-11: *refuse at describe too*): a cited run whose job ended without concluding refused when described, as one that ended with an error — **2.69.35**. Describe and prep then share one rule (`compose.refuse_failed_citation`), so a point its walk never started and a conclusion whose exit code does not read are refused at describe too, as prep refused them. **Found by 2.69.35's review, and closed in it**: the rule asked two sources — the conclusion's exit code for *did it finish*, the run door for *is it going* — so a run concluding with exit code 0 while its output states a stop, which the door reads *failed*, was taken by both; the rule now asks the run door alone (§ 3.1: a citation is a relaxation *that has finished*). **2.69.33**, X-PARSE closed: the spectra load's own copy of a file's version (`spectra._version`) gone for the ending reader's rule (`_run_ending.version_of`); *the run's end read before the file's last read* stays applied by each route over its own reading — the trajectory route over its reading kept across polls, its token the time it trades with the page, the spectra load over one read; a door for both would take two callbacks for two callers. What the trajectory route's token misses, read in its code: a final write in the same timestamp tick as the one before it, after the poll's look — the viewer would stop a few lines short; a version token on the wire would change the page's poll, and nothing has been seen. The call's words served (found by 2.69.33's review: the page and the server rounded apart — a mode at −0.3 cm⁻¹ read *-0* in the run's index and `status`, *0* on the page; at an exact half, apart for half of them): the call's wire form carries `Call.words`, the page shows it, and a mode a hair below zero reads 0. **Kept, with why**: TBtrans's data-file version (`tbtrans_nc._version`) — its size and time as the digest records them, no path, the digest writing the two; the bench's launch facts beside the run record — one source in two places, the same lines of the same files through the same grammar readers and read window, the record stating no orbital distribution block size (`ProcessorY, Blocksize`); the two run-of-points branchings (`runstatus.run_answer`, `continuation.read_run`) — two answers' shapes over the same doors (`read_points`, `concluded`/`state_of`); PySCF's unit conversion in its two readers — one constant, `constants.HARTREE_EV`. Already fixed: the verdict from words on the transport page, SIESTA's logicals on the Run panel. A doc line, not a route: the displacement sweep's page reads its record whole once the registry has claimed it (`web/results.md` § 0.2, said so). Dropped: a foreign extended XYZ's cell (foreign input). **Left, placed in the plan**: § 5a rule 4, the trajectory saying which log its facts came from — **closed 2.69.39**: the progress log and PySCF's own log each named, the steps the second gave an SCF history counted, and each that did not read named with why. **Retired as handcrafted** (the user, 2026-11-11: *are you handcrafting tests again?*): 2.69.30's row of a log cut by hand inside a step's SCF table, an exit staged around it — the pass's footer read there untested; and 2.69.33's own row of calls with typed frequencies — the page's call words asked by the SIESTA vibration's real run alone, against the contract's words. |
| X-TMPL | **2.69.40** (an optional setting's blank in its own words; what is left of it in the changelog's row), **2.69.42** (what a stage may override, one door), **2.69.43** (the one default: the four readers onto `default_of`, `ecp_atoms`'s two answers made one; the words, `default_words`, read the same two sources in the same order — one source, left; a run's record of its parameters and `molbuilder validate` onto it, `Parameter.default` deleted; `jobset migrate`'s class default for an item an older file never had, left: an older release's files get no new work, the user 2026-10-16; the defaults panel's own `show` beside the form's, both older, left to the page doors; its regression -- a launched PySCF stage refused when prepared again, its deck's parameters record changed -- fixed by **2.69.44**, which compares a PySCF deck as Python reads it, what it only reports aside, the user 2026-11-12; its messages held in the deck's words and the report names spelled once, **2.69.45**), **2.69.46** (`select`'s per-kind test: a kind's marker asked by the kind, `Item.carried_by` the one answer of `calculations`), **2.69.47** (the control's kind decided once, the server's, which Task setup's cells read), **2.69.48** (ids and labels: the schema's fields through the renderer's one walk and one map of ids, the renderer's own readers included; the Build tab's live check listening per form, its prefix copies and id list gone; the server's prefix per route — `p`/`py`/`common` for the Build forms, `t`/`t-<rung>` for the Transport tab's — stays each route's own, the one home of its page's ids), **2.69.49** (the setting's one record, `_shared.item_record`, which the form field and Task setup's run cards, columns and shared values start from, the page keeping the run card's whole; the shared card's switch said in its words) — closed; *re-derived from the code 2026-11-12, the readers' detail lost: "what a stage may override" also by hand in the rung form (`_shared` rung surface), two copies — the rung form's giving the door's own fields on every rung, the describe's list refusing nothing its check lets through; the one default four ways and a words mirror, the class's own default for PySCF `ecp_atoms` (`[]`) another than the catalogue's (none); more blank words than four (Task setup's run cards and stage cells, the deck's table, the Build tab's own caption); `select`'s per-kind test written by hand at its callers (`resolve`, `template`, `_shared`); the control's kind two contracts' rules (`form-schema.md` § 4, `task-setup.md`'s `legalValues`), agreeing on every road input; the schema walked for ids by `chemistry.js` too, the server keeping its own engines' prefix table; three hand-built item records, not two. What reaches a person: the blank's words -- and, found by 2.69.43's reviews, a vibration's defaults in a run's record of its parameters and in `molbuilder validate`'s finding, the class's or the item's general default where the kind recommends another.* | "what a stage may override" a second time (`resolvable_override_names`); the one default bypassed four ways; four names for an optional blank; `select`'s per-kind test rewritten by every caller; the control's kind picked twice; three tabs walking the schema for ids and labels, one with the engines' prefixes hard-listed; the item record built by hand (structural, not words) | `template` (`why_not_in`, `default_of`, `null_label`, `select`); `_item_to_field`; a form-schema door for ids |
| X-EXEC | **2.69.50** (the hand-over's proposal the server's: the folder answer's `proposed`, a description and its job request as their writers write them, a description's missing job request too; the page's assembly, the hand-over's three keys written for it, and the tests' copies gone; a new calculation's job request one door, `describe.new_request`), **2.69.51** (a group's lines on Task setup: *needs a prep* the ready door's one answer, `GroupAnswer.needs_prep`, which `prep task`'s offer asks too, its restatements swept, and *no verb takes it*, `GroupAnswer.no_verb_takes`; a stuck group's lines its status step; verified: the other differences the reader named are lines the contract allows -- a launched group launched again whole, each member's own relaunch shown beside it, a group whose record of its prep does not read launched as prepared; a changed group sent by a bare `launch task` -- ruled by the user, **2.69.52**: a launch naming no group offers none needing a prep; left as its contract states: a launched changed group's tab), **2.69.53** (the held sentence: one composer, `ready.held_why`, for prep's refusal and launch's, the stop a line of its own, and the groups' listing's `jobs status` too; left: a background start's *to stop it:* line, the same rule; `GroupAnswer.said`, the step's words and the groups' listing keep the state's short name, *goes in job N -- after group K* (or *its first group*), read from it; verified and left: `_how_runs_ended` reads each stage's row as its contract states, *A run here, waited on* -- read: right after the launch each member's newest run is the one the job ran, so the two readings meet there), **2.69.54** (the GPU ask by `gpu_request` in a group's envelope, a sequence's request, the shelves and a benchmark's recommendation; left: the plan line's display spelling; verified and left: Save's launched-group check asks the one rule and composer, `launched_without` and `joined_launched`, its condition the contract's split between § 5's regrouping and § 8's stage taken out), **2.69.55** (`checkpoint restore`'s question through `ask.confirm`, saying why it stops with no terminal; left: `jobset probe --write`'s questions read answers as typed, a pipe's too -- M-6 silent on a pipe, its table answering through one; `molbuilder envs`' six questions through click directly, one, wiping builds, Yes by default), **2.69.56** (a record's unknown keys in one sentence, `persist.unknown_keys`, its rule in `job-contracts.md` § 6.1 -- asked by `check_keys`, the description, the job request, a warm-files row, a spectra result, `molbuilder.json`'s sections, the catalogue's items, the kind and engine tables; left, the finding's second part: the readers still composing it by hand -- `molbuilder.json`'s top level, the catalogue's keyed tables and `not_offered` rows, a spectra result's optical block, the warm-files sections, the structure's rows and record, the sidecar, the notebook settings, the atom index map, the spectra data file), **2.69.57** (2.69.56's regression: the result's key check made a relative import inside itself (`from ..persist`), which a SIESTA vibration's record step beside the job cannot make -- it stopped at the force-constant call's -- now `persist.check_keys` taken at the module's head, inside molbuilder or from the bundle, `_refuse_unknown_keys` removed; the bundle test calls the check, not only imports it. Measured 2026-11-13, the three bundles a job carries (`runwrap.monitor_companions()`, `VIBRATION_COMPANIONS`, `PYSCF_COMPANIONS`): 44 distinct imports of molbuilder's own modules inside functions with no fallback for the bundle (59 counted bundle by bundle). A call graph by name -- which over-reaches, never under, from the entries it is given -- from every entry a job uses: `siesta_vibration.main`; the monitor's `main` (its verbs, the `deck` verb's keyword table among them) and `_run_ending.main`; the readers `runfiles.reader` resolves beside a job through `runpoints.part` (the run-ending, live-pass, loop-word and timing readers); the Makov-Payne script's imports from the monitor's bundle (`siesta_reader.read_output`, `runfiles.RunNames`, `runrecord.run_output`); what the PySCF scripts import (`StructureCodec().write_moved`, `StructureCodec.geometry_of`, `AtomIndexMap.of_engine_record`, the functions `emit_bundle_imports` names, the progress hooks); and every step the tables name in a bundle's module (`runpoints.outcomes`, `run_words.count_words`/`round_outcome`, `vibration_phases`' -- `continues_from` resolved on molbuilder's side, which only widens the search). It reaches 9; each, read, is off a job's path: `parse.dirs.job._job_ended` returns before its import when no launch record is passed, and the monitor passes none; `RunRef.run_folder()`, `report_fields.for_run`/`refusal` and `transport.energies.held` are called by no module that travels; `Structure.frame_at` only by the codec's `frame_choice`, which only `StructureCodec.load` calls and no script, and `Structure.to_ase` by none (constructing a `Structure` reached every method, by name); `transport.energies`' deck readers only when `joined` is given no `window_of`, and `run_words` gives one. A second reader re-derived the 44 and each of the 9. Nothing moved; the batch's bundle tests and the engine's road tests stay the guard), **2.69.58** (the finding's second part: `molbuilder.json`'s top level, the warm-files sections, a spectra result's optical block and mode rows by `persist.check_keys`, the structure's customized rows and row sets (by `persist.check_keys`, no key required), its metadata and the structure a page sends, the sidecar's two gates through one door (`sidecars.molstruct.unread_keys`, which takes in the old `known_keys`: the retired keys accepted and never offered as known), its identity columns, the order record, the atom index map, the notebook settings, the spectra data file's three blocks and the optical calls' data file's, `not_offered` rows, a kind's description block, the cell's whole-block edit, the scheduler kinds table, a result's Raman and infrared asks -- each by `persist.unknown_keys`, the travelling modules taking it both ways; left by their kind: vocabulary-keyed tables, refused as a value outside their words; a function's own arguments; a calculation setting's name, its typo pointed at the schema's names (89 SIESTA, 62 PySCF: too many to list) and a retired or renamed one told so (`template.why_not_an_item`). Finding 7 closed: two sweeps of the hand-built refusals left, each read, found no other record's keys), **2.69.59** (the user's "suggest the closest" and "unify this": a name that is not a setting, where `jobset migrate` reaches -- the template reader, the description's `varies` and overrides, `resolve`'s overrides, a run card at the description's check and at prep through `job_request.why_not_on_card`, a migration -- by `template.why_not_items`: retired or renamed told so (on a run card a retired one alone, which migrate takes off it), misspelt with its near one among what its place may hold (`template.items_in` -- the one membership rule, the stage table's columns' too -- `settings_in`, `template_settings`), by `persist.unknown_keys` -- the settings unlisted but for a run card's few; a benchmark row's names and pins, which no migration reaches, by `persist.unknown_keys` alone, never a retired story; the description check's second near-miss rule, `validation.task._no_such_field`, removed: the rule is written once; left: a sweep's axes, a backstop that tells a machine fact, and a name the page itself sends; found, as at the old code, measured for X-EXEC's rest: a retired item's way out is the template's words wherever it is named -- a `varies` entry naming SIESTA's retired Raman selector is told the migration states Compute Raman from it, where the migration takes it out of `varies` -- what the migration does at each place is data the retired row does not hold), **2.69.60** (that data: `template.MIGRATE_AT` -- by where a name is written, the migration's words for a retired one and whether it renames a renamed one -- and `Retired.on_a_stage`, the way out of a stage's override the migration refuses, which `jobset.migrate`'s refusal reads too, its branch on the selector's names gone; `Retired.way_at` and `template.why_not_an_item(at=)` read them, the run card's switch folded in; `job_request.why_not_on_card` refuses only a name no setting of the engine; left, as at the old code: `jobset.model.Resources.from_dict` tells a retired key of `job-set.json` in its own words, a machine-written record whose error names the migration), **2.69.61** (the reuse route's own standing words: `JobSetStatus.stands`, the status door's line, *every group finished* by the next step's own rule, `runstatus._every_group_finished`, never `complete`; the route reads it; the Results tab's *every stage finished* is `complete`, another fact, in `web/results.md`'s words, kept). **X-EXEC CLOSED** -- each item of the re-derived list taken, or verified and left where its release's *left* says why; *re-derived 2026-11-12 by a reader, each to be verified by the release that takes it: the reuse route's own standing words; `GroupAnswer.said` a second standing sentence; Save's launched-group check written in the route beside `ready`'s; `_how_runs_ended` reading each stage's newest run where `job_members` reads the job's own, and deciding a sequence's walk; the GPU count read raw in `group.envelope`, `sequence_request`, the shelves, the plan line and `summarize`; the unknown-key refusal in two `persist` wordings and five of its own; `probe`'s and `checkpoint restore`'s questions past `ask`; the held sentence in `ready.held_why`, `submit._held_by`, `GroupAnswer.said`, `runstatus` and `groups_text`.* | the hand-over finished into `task.json` key by key in the browser; the scheduler's R6 refusals twice; a group's next-step lines worked out in a route; Save's rule composed in the route; two answers to how a job's members ended; the GPU count read past `gpu_request`; the unknown-key refusal three times; terminal questions past `ask.confirm`; the held-group sentence three ways | `describe`/`task`; `scheduler.place`; `runstatus._steps`; `ready`; `runstatus.job_members`; `gpu_request`; `persist`; `ask.confirm`; `ready.held_why` |
| X-ENG | re-derived 2026-11-13 at 2.69.61 by a reader, each to be verified by the release that takes it (fixed since: the vacuum flag past the cell door; not a duplicate: the device rung's own choice). The twentieth, verified: the inverted-junction warning is said on the Transport card alone, where `engines/transport.md` (*The label convention*) has *the preflight* warn too. A fix tried at the settings gate (2.69.62, withdrawn before commit) was refused by its review on two counts, measured: an extension's device and transmission would be told to swap the labels, the step § 2a.16 refuses there (the card holds the note back while extending) -- and the gate has no way to know it is extending; and `transport.gather.rung_deck_now`'s re-render for `ran_otherwise` reports the gate's findings, so `status`, `launch` and the server would print the warning on every call once a device run finished (the gate's other findings already do, rarely). The user chose (2026-11-13): prep says it, like the card. **2.69.62**: `sort.junction_notes`, what the junction says of itself, read off the junction, nothing while extending; the card and prep's rungs (`Rung.said`) ask it; the sort's own note path removed (`SortResult.notes`, `ComposedJunction.notes`); **2.69.63**, the monitor's GPU reading: what `nvidia-smi` says read once, `runtime_info.gpu_reading` and `gpu_models` -- the monitor's sample no longer dropped by one `[N/A]`; the load widget's reader and two listing parsers gone; **2.69.64**, the pseudopotentials judged twice: SIESTA's data-files step copies and refuses nothing, the settings gate refusing its three cases in the one door's words -- which now say the species, the full place and `psml_lib`; a transport cited run's own refusal kept, the citation's way out -- the rest planned | the empty span along an axis three ways; the box and its zero volume re-derived in the gate and the deck; the PySCF molecule built twice; engine seams that only re-expose the engine table; the range warning three ways; the neutral-junction rule restated; `is_dft` and the state's method both asked | `cell`; one PySCF writer; the engine table; `outside_range`; the kind record |
| X-UI | with O-STATUS … O-KEYS | the describe-a-structure page built twice (optimization and Spectrum: the Inspect card, the load controller, the live check, the Send) — one partial and one module, as the chemistry card is; Task setup's own message severities; ~20 hand wrappers around the status writer while `status.writer` has no caller; the element builder copied seven times and a second shape four; the five generator panels' wiring five times; four cached loaders; the next step drawn twice; a second token set; the card look and the action row restated six times each; every page's head repeating the header's assets | the partials; `lib/status.js`; `lib/dom.js`; `panel-deps`; `tokens.css`, `page-shell.css` |
| X-PLOT | with O-FIND | the bench and spectra charts re-setting the theme, the transport legends placed by hand; five behaviours for "Plotly is not loaded"; the read-only viewer's mount and its failure written six times; the data export: the kind table's for the terminal, a route per kind for the web, the browser's own CSV for a trajectory | `plot-theme`; `plot-box`; the viewer's handle; `kind_table.export_of` behind one route |

## Found by the test audit of 2.69.25–2.69.33 (2026-11-11), by the release that owns each

*(the user, 2026-11-11: "are you handcrafting tests again?", "how did you
execute the test?", "use agent to review and discipline your tests", "i
don't trust your memory": every test changed since 2.69.24 judged by a fresh
reader against `process/testing.md` alone, its scope taken from git, the
changelog's test claims checked against the code)*

| id | where | finding | door |
|---|---|---|---|
| TA-1 | **closed 2.69.34** | `test_calculation_door`'s two `job-set.json` rows cut prep's own file by hand — no person edits it | retired |
| TA-2 | **closed 2.69.34** | `test_auth_config`'s two non-object rows edited to the code's words | retired; the malformed file's one row asks `configuration.md` § 2.2's words |
| TA-3 | **closed 2.69.34** | `test_persist`'s TypeError loop over argument mixes no caller passes | retired |
| TA-4 | **closed 2.69.34** | the record reader's table on invented schemas | rewritten on the description `describe` writes, a plan falling through to the disk added |
| TA-5, TA-6 | **closed 2.69.37** | `test_workspace_tag_isolation`'s nine tests answered by a server written in the test (seven through `_run`, two MolView round trips), every one waiting on time (`process/testing.md` § 4, § 5) | every request answered by the real routes (`served(base)` under `support.live_server.serve`), each test's saves in its own store, each wait the harness's `__quiet()`; the prune held, never timed — `served` holds a request (`__hold`, `__release`; one never released, or a `match` that throws, fails the run; § 4); what was saved read back through the page's own calls and counted in the server's folder, the real server's refusal standing for the file-name check; MolView's load answered by the real load route; *same-step saves stay in order* newly asked, of a point and of the draft, a write held; the spelling row *two pages* retired, the overwrite test asking it |
| TA-7 | **closed 2.69.34** | the workspace's tag-refusal table: three wrong tags on five calls | one example per clause of the one rule kept (`process/testing.md` § 3b): a missing tag and a `:draft` one (`workspace.md` § 4), each refused |
| TA-8 | **the typed note closed 2.69.34; its store owed to 2.67.12** | a typed note read back on the structure page; the round trip's workspace a hand-made store | folded into the round trip, the page writing its own note; the store with TA-16 |
| TA-9 | **2.67.12** | the sidebar busy-guard test on its own node start and a typed answer | `run_node` with `served(base)` |
| TA-10 | **closed 2.69.36** | `test_in_progress_frames_stay_out_of_plots`'s typed frames (`process/testing.md` § 0, § 6: a run is made on the road) — no other test of the adapter's `in_progress` | the e2e written first, then the typed tests retired: the shared CO2 relaxation replayed as it stood inside its first move's SCF and inside a later one's (`support.replay`, its line-finder `Output` moved there from the monitor's e2e) and read through the converter the trajectory routes answer with — not the route itself, which adds nothing to the flag (`watch.py`) — its frames the whole output's up to that move, the last flagged, alone; a run that ended, finished or stopped inside a step by SIESTA, asked of the route's own last load, flagging nothing; left: the reader's placeholder for an SCF with no coordinates yet, reached by no output of the SIESTA here (5.4.2), the converter's `[]` told from a list of falses by no test, and the page's reading of the flag (`plottableFrames`) fed typed arrays — 2.67.12's, the page tests fed the server's answers; a step SIESTA stopped inside plotted at its last SCF row's energy, left as it is (the user, 2026-11-11: *include or leave out is fine … as long as the state of the run is correctly set as incomplete/error/whatever, it will not change how the run will be used*) |
| TA-11 | **closed 2.69.34** | the two strict-JSON adapter tests on a typed NaN history | retired; `persist.json_safe`'s own tests stay |
| TA-12 | **closed 2.69.34** | two tests on a progress log cut mid-JSON (2.69.19's) | retired with their helper |
| TA-13 | **closed 2.69.34** | the preview test on a typed structure handed to the writer | rewritten on the log prep seeds, read by the registry |
| TA-14 | **closed 2.69.33** | the vibration ladder test compared the page with the server's composer | the contract's words again |
| TA-15 | **closed 2.69.34** | `test_app_notifications_e2e`'s comment on the one queue | each workspace id's queue |
| TA-16 | **2.67.12** | eight hand-made workspace stand-ins in five files (`test_molview_model`, `test_molview_mount`, `test_molview_stores_history`, `test_molview_ui_context`, `test_structure_page_js`) | the real `dispatcher.js` under `served(base)`, as TA-5 and TA-6 |
| TA-17 | **closed 2.69.34** | false test claims in the changelog's 2.69.25, 2.69.26, 2.69.28 and 2.69.31 rows | corrected in place, each marked |
| TA-18 | **closed 2.69.38** | `tools/verify_subsumption.py`'s coverage plug-in writes outside the test sandbox, and the suite's machine guard refuses it — the tool `process/testing.md` § 3b names for a subsumption verdict cannot run | the coverage file made in a folder of its own in the system temp folder, which the run inherits and its guard lets it write; the tool's own scratch folder — `--tmp`, `SUBSUMPTION_TMP`, a session's `CLAUDE_JOB_DIR`, which put the file under the home — gone, nothing else using it; run in a session before (refused) and after (a verdict) |
| TA-19 | **left, found** | rows older than this session pin a reader's words where `job-contracts.md` § 6.1 states the condition (*text that is not UTF-8 … does not read*) and no refusal words: `test_calculation_door`'s *save it as UTF-8*, `test_auth_config`'s *not UTF-8* | the contract's: refused as not reading, the file named, a way out |

## Found by 2.69.14's work (2026-11-09)

| id | where | finding | door |
|---|---|---|---|
| K26914-1 | **closed 2.69.14** | `tests/test_monitor_watches_a_live_run_e2e.py`'s bundle probe asked the monitor for `SiestaReader`, which 2.65.5 moved behind the catalogue's row; failing since 2.65.5, unseen because no whole e2e suite ran after it | the probe asks each live reader's catalogue row, `runfiles.reader(role, "live")` |
| K26914-2 | **open** | the notebook's test (`test_notebook_through_one_port_e2e.py`, the http cases): JupyterLab under `conda run` did not start in two of five runs, its own error not captured | read its log on the next failure before any change |
| K26914-3 | **the user's, later** | a folder named `projects` inside the tree begins a tree of its own (`projects.find_projects_root`, the nearest wins): a calculation folder named so is a place, and one below it measures its depth and its neighbours from it | whether a folder below the root may take the tree's name |
| K26914-4 | **planned** | the test files' own trees: `test_web_files`' fixture puts files and `spectrum/BDT` at the projects root, a state no door makes; 21 other sites in 9 files make tree folders by hand | the tests' release: each tree made through the road |

## Found by the overlap audit of 2.68.0–2.69.9 and 2.69.10's review (2026-11-09), by the release that owns each

*(The user, 2026-11-09: "check all existing modules and against your previous codes to see overlap" — four readers over every release of the UI review, each finding read against the code before it landed here; then the plan the user agreed to: one release per owning door, the contract first, the copies deleted, a discipline reader over the whole touched modules before each commit. The rows below are grouped by that door, in the order agreed.)*

| id | where | finding | door |
|---|---|---|---|
| K26910-1 | **closed 2.69.10** | the sidebar's upload wrote a structure file's bytes as they came, and since 2.69.9 a replace left the old sidecar describing other atoms | an uploaded structure read as an import and written as a pair through the save's path; a sidecar alone refused (`model/structure.md` § 2.4) |
| K26910-2 | **closed 2.69.10** | the preview's editor wrote a structure file's text past the codec | the read answers `read_only`; the plain write refuses a structure file or a sidecar |
| K26910-3 | **closed 2.69.10** | the hand-over's pair was written by the browser as two writes, and a later failed write left the folder half written | the hand-over door writes every file or none on the server (`describe.publish`, the transaction `describe` already had) |
| K26910-4 | **closed 2.69.10** | three "file exists" answers in three wordings, the upload's and the write's duplicating the overwrite logic | one answer, `files.exists_answer`, `{needsOverwrite, replaces}` |
| K26910-5 | **closed 2.69.10** | the overwrite question asked in the sidebar's button (2.69.9), worded apart from the save's | the files door asks, one question (`dialogs.confirmOverwrite`), naming what a yes replaces (`projects.md` § 3) |
| K26910-6 | left | a rename, move or copy to a structure file's name makes one from bytes the codec did not write | the file operations by the pairing rule — a rename keeps bytes, so whether it is a write is the question |
| K26910-7 | the findings release | an upload's notices (what the read said, the box's check) are shown nowhere: the sidebar has no findings panel, and the notification bar is not a findings renderer (`science/validation.md` R2a) | where a sidebar action's findings are drawn |
| K26910-8 | left | a `.pdb` upload keeps what molbuilder reads of it — the first model's atoms — and the rest of the file is not kept, unsaid | what the reader says of what it did not read |
| K26910-9 | left | `describe.publish` moves file by file; one failing move leaves a folder part written (as `describe` did before) | — |
| O-ANS | **closed 2.69.11** (the browser half) | answers read in a second shape: Task setup's prep call flattened the reader's answer by hand; the sidebar asked `/api/files/zip_prepare` past the files layer; Task setup's settings-list fetch sent its reason to the console and the page guessed one; `/api/results/dir` asked from two places. *Kept, by the user's word (2026-11-09) and the contract*: the files layer answers in its own documented shape, made once in `projects/api.js` (`projects.md` § 5); the Spectrum tab's "no answer read" reads the reader's table (§ 12: `body` only when an answer was read); the Modify panels' `cancelled` is their own result, a no and a Cancel alike; `chemistry.analyze`'s `{ok, body}` is that module's documented result; "throw, else body" ×7 left as each caller's own two lines. *Fixed too*: the Transport report's Read rebuilt its answer by hand; Task setup's `fetchVocabulary`, a wrapper over the reader, gone — a list of tier presets no longer says the setting names are raw | `lib/server.js` `ask` (ui-contract § 12, a page reshapes nothing); the files layer (`projects.zipPrepare`); the results picker (`askFolder`) |
| O-ERR | **closed 2.69.12** | the server's failure answers: `web-api.md` § 1 names one error builder, `_shared.err`, which served the Modify routes alone, and 298 failure answers elsewhere were written by hand beside it — each route's inline `jsonify({"ok": False, …})`, the refusals built in a variable first (the checkpoint routes' advisory, the trajectory route's not-a-trajectory, the transport describe's with its findings), helpers that were the builder under another name (`checkpoint._protocol_error`, `_server_fault`, `workspace_storage._bad`), and the channel probe's refusal riding a `hint` the page finished. *Kept*: `jupyter._refuse` (the gate's three refusals) and `_signalled` (202 or a refusal), both through `err`; the folder answer's parts, which carry their own `error` inside one answer | `_shared.err(msg, code, **fields)`, the fields a refusal carries beside its sentence (`web-api.md` § 1) |
| O-STATUS | planned | status lines written by hand: the benchmark page's failed look, Documents, JupyterLab, the Transport plan line, Task setup's reports note, 2.68.5's folder-changed line, the preview's own; `dom.wordsEl` rebuilt ×4; the form tooltip's partial `wordsText`; `validation-findings` keeps a `textContent` fallback; `ladder.unread` drawn as text; the tests' Python copy of the renderer (`pages.drawn`, `findings._said`); "a part of this page did not load" ×4 | `lib/status.js`, `lib/dom.js` (ui-contract §§ 4–5, 12.1) |
| O-FIND | planned | the Molbuilder Save's notices as text (2.69.9); "outcome + notices" composed four ways; the hand-over's structure judged for its engine twice (the write gate and the settings gate, `task-handover.js` `beside` existing to drop the copies); the inspector-ready signal hand-built in five result views, and none sent on a first look that fails; three renderers for "this result view failed"; a bench refusal asked again every 15 s | `lib/validation-findings.js` (R2a); `lifecycle.announceReady`; `ctx.showError`; `structure-periodicity.md` § 2.1 |
| O-STATE | planned | a tab's form kept three ways — Transport's own note, the optimization tab's session store, the Spectrum tab's nothing; "unsaved" answered from two sources on the Modify page; a project save recorded twice (`savingThrough` and `save.js`); *measured 2026-11-12 (2.69.51's review)*: Task setup's Prep card line under its boxes reads the groups' states itself (`state`, `matches`), calling every group prepared while one is offered -- never launched and changed since its prep, or a member's run gone | the workspace, through its § 5 calls, as the panels use it |
| O-WORDS | planned | a value in words composed by three server and three browser composers; a Task setup item record built by hand in two routes and the form's (one record since 2.69.49, `_shared.item_record` -- measured then, still each page's own rule for saying a value: the Transport tab's locked switches *true* / *false* (`_paintFixed`, `_said`), a run card's inherited value as held (*— true (every stage)*) and its members' differing values (*stated apart — device: true*), the defaults panel's own rows (`diffFromDefaults`), the three-way select's *yes* / *no* (`form-schema.md` § 4) beside Task setup's *on* / *off* (§ 5.2)); a second wording of "default"; the frame words built by hand twice, beside `frameWords`/`toDisplay`, the hand-over taking a finished sentence and dropping it on its notices branch; the pseudopotential caption copying `SETUP_WORDS`; `chemistry._spinText`; `gf_files._volts_words` parsing the bias token away from its writer; launch words in four tables; the Preview's counts from two sources; the hand-over's "else the folder's name" worked out in the browser; the trajectory's word table keyed on server tokens; a tab's name in `TABS` and its `<title>`; the no-dots rule in three expressions; the `required` key with no item; This machine's form sending a kind nobody ticked (`kindOfNamed`, 2.69.9) | each word's one home, served |
| O-KEYS | **2.69.16 the folder of a path; the rest planned** | the folder of a path worked out four ways; `window.molbuilder.projects` reached past the imported door; the documents tree's own keyboard beside the tree picker's; the roving-focus core copied between `one-stop-list` and `tab-strip`; focus after a redraw kept by two mechanisms, and `afterTheEvent` beside the fence's; `button.plain` copied in `transport.css`; focus rings written by hand beside `--focus-ring`; the Transport tab re-deriving its strip's choice | `lib/path-utils.js`; `lib/one-stop-list.js` / `lib/tab-strip.js`; `lib/page-busy.js`; `page-shell.css`, `tokens.css` |
| O-TESTS | planned | the focused-control probe written twice; a calculation-folder fixture copied twice; typing a value then Tab beside `support.sidebar.committed`; Node DOM stand-ins per file beside `tests/support`; a Node runner of its own beside `_node_esm.run_node`; one case twice in `test_server_answer_js`; the reader's sentences checked again in callers' tests; made-up server answers where the real route answers; a run-script regex reader beside `runwrap.stated_counts`; a water molecule built by hand beside `support.road` | `tests/support` |

## Found by the UI review's group C, what a page says after an action (2026-11-09), by the release that owns each

*(Group C's items on the sidebar, the Molbuilder tab, the benchmark page and This machine, each checked against its contract before the fix)*

| id | where | finding | door |
|---|---|---|---|
| K2699-1 | **closed 2.69.9** | an upload whose name was already in the folder was refused with words for a script (*pass overwrite=true, or pass auto_rename=true*), which a person in the sidebar cannot do | the route answers `needsOverwrite`, as a save's does, and the sidebar asks *Replace water.xyz?*, sending again on a yes (`projects.md` § 5) |
| K2699-2 | **closed 2.69.9** | the benchmark page replaced the sweep with the error at the first look that failed and stopped looking — a server restart froze it | the last answer stays with its two clocks, the foot says the failed look, the next look comes on time, as the trajectory viewer's poll (`bench-summary.md` B4) |
| K2699-3 | **closed 2.69.9** | a label's chip was set in capitals, so `Au_tip` showed as `AU_TIP` — two different labels drawn alike | the chip spells the label as stored (`molview.md` § 1) |
| K2699-4 | **closed 2.69.9** | the Molbuilder tab's Save said *Saved water.xyz.* and dropped what the server said of the box, which the viewer's own Export said | the server's notices after the outcome, in the tone that says read this (`tabs.md` § 6) |
| K2699-5 | **closed 2.69.9** | on This machine, an edit of a listener's address under its name saved it as a Slack webhook — the form's first kind stayed ticked — so every report went unsigned to a listener that refuses it | a name typed that is a stored channel's ticks its kind (`this-machine.md` § 3) |
| K2699-6 | **closed 2.69.9** | *Issue a key to* asked for *your molbuilder user id*, a name molbuilder uses nowhere else | *the email you sign in with* — the name the report log reads by default |
| K2699-7 | 2.69.10 | the optimization and Spectrum tabs do not say where the Send writes before the click, and their live check asks with no folder (K2698-7) | each tab follows the sidebar's folder and says it beside the Send; the live check asks of that folder |
| K2699-8 | 2.69.10 | the Spectrum tab's form starts over after a switch to another tab and back | kept across a tab switch as the optimization tab's is |
| K2699-9 | 2.69.10 | Results' stage ladder clips a long stage name | the ladder's layout |

## Found by the UI review's group C, sending a structure (2026-11-09), by the release that owns each

*(Group C's items on the road from a structure tab to Task setup — the optimization and Spectrum tabs' Send — each checked against its contract before the fix)*

| id | where | finding | door |
|---|---|---|---|
| K2698-1 | **closed 2.69.8** | a calculation's name typed as the field's default (`siesta`) was replaced by the folder's, the page having shown it as the name that applies | what the person typed names it, the default too; the folder's name only for a field left blank (`run-identity.md` § 4) |
| K2698-2 | **closed 2.69.8** | the hand-over of a PySCF molecule said its vacuum was too thin for an accurate result — box advice about a calculation PySCF does not run | the hand-over's structure judged for its engine, `cell.box_findings_for`, as the describe door judges it (`structure-periodicity.md` § 2.1) |
| K2698-3 | **closed 2.69.8** | the SIESTA pseudopotential folder was drawn red as a field still to fill while the `.psml` files beside the calculation answered it and the Send wrote | the field not drawn required; the live check and the Send's refusal say what is missing ([`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) § 1) — the catalogue's `required` marker now has no item |
| K2698-4 | **closed 2.69.8** | the folder's caption said *nothing can be built until the folder is filled in*, which the files beside the calculation make untrue | the caption and the card's line say the calculation's own files are taken first |
| K2698-5 | **closed 2.69.8** | a file of several frames sent one of them without saying which | the Send sends the frame on screen and its line says so (*the structure at frame 3 of 12, the one on screen*); the tabs' hint says it before the click |
| K2698-6 | **closed 2.69.8** | the charge's reason read as a chemistry verdict (*no deprotonated phosphate groups*) and *open-d metal* as shorthand | *none of the charged groups molbuilder counts (deprotonated phosphates)*; *no metal with an open d shell* |
| K2698-7 | 2.69.9 | the live check on the optimization and Spectrum tabs asks about the pseudopotentials with no folder, so it warns *no pseudopotential folder is set* while the folder the Send writes into holds every file | the tabs know where the Send writes (the sidebar's folder, C14) and the live check asks of it |
| K2698-8 | left | the blank name field's caption reads *SIESTA's own default: siesta* although a blank field takes the folder's name | the field's caption by the name rule (`run-identity.md` § 4) |
| K2698-9 | the dead-code release | the catalogue's `required` key reads, validates and serves a list no item declares | delete the key and its reader, or keep it for the next item that needs one — the dead-code release decides |

## Found by the UI review's group C, Task setup's flow (2026-11-09), by the release that owns each

*(Group C's items, each checked against the code and its contract before any fix; those a contract or the user has decided are kept: the cards' order, a launch value refused at Preview, Results neither following the sidebar nor re-reading on a timer, This machine's scope)*

| id | where | finding | door |
|---|---|---|---|
| K2697-1 | **closed 2.69.7** | *What gets written* said *3 of 3 in the folder* after prep had written twenty files — it counts the files it lists | *3 of the 3 listed in the folder* |
| K2697-2 | **closed 2.69.7** | a plan that could not be worked out (a description mid-edit) hid the plan and the written files without a word | the list says why, in the reader's words, in its place (`ui-contract.md` § 12) |
| K2697-3 | **closed 2.69.7** | a queue once chosen could not be unchosen, though the field is optional | choosing the chosen queue again unchooses it |
| K2697-4 | **closed 2.69.7** | the Preview's *what this run will be launched with* gave the run script's variables (`_mpi_np_default=2`) | each count as data — its setting and value — named as the cards name it (*MPI ranks (np) 2*) |
| K2697-5 | **kept** | prep's provenance lines mark where a value came from with `<-` | the terminal's and the page's one notation for *from*, the same in every provenance table |

## Found by the UI review's group D, the settings' help (2026-11-09), by the release that owns each

*(Group D's items on the settings' own words in the catalogue, each read before any fix)*

| id | where | finding | door |
|---|---|---|---|
| K2696-1 | **closed 2.69.6** | a setting that is molbuilder's own showed a badge naming code (*comment-block control*, *the SCF class module*, *geomopt callback*, *per-mode selector parameter*, *orbital-window record size*, `.run.sh` and `.fdf` for the run script and the deck) | each badge says what molbuilder does with the value, in words (*the run script's `mpirun -np N`; not in the deck*) |
| K2696-2 | **closed 2.69.6** | two transport settings said *lead* where the others say *electrode*; one setting named a tool's old name (*the molwatch trajectory log*) | *electrode*, TranSIESTA's own word; *the trajectory log the live viewer reads* |
| K2696-3 | **closed 2.69.6** | the help named an internal rule (*job-layout v1*) and shouted (*does NOT*) | why no dot, said; plain case |
| K2696-4 | **kept** | the help cites SIESTA's source by file (*read_options.F90*), and each engine default carries its source line | a citation of where SIESTA's behaviour was read — evidence a scientist can check, not jargon |

## Found by the UI review's group D, the other pages' words (2026-11-09), by the release that owns each

*(Group D's items on the pages' own words, each read in the code before any fix; the user, 2026-11-09: one word *stage* on every page, the notebook tab *JupyterLab*)*

| id | where | finding | door |
|---|---|---|---|
| K2695-1 | **closed 2.69.5** | the Transport tab, the transport viewer and the Results ladder said *rung* (*Rungs and provenance*, *5 rungs, every rung finished*); a run opened from the ladder read *a run off its ladder*, as if it were not on it | *stage*; *a run picked from its ladder* |
| K2695-2 | **closed 2.69.5** | a part of a page that failed to load named a script and a template (*form-schema.js did not load — check the script order in transport_calculation.html*, *structure-load-save-contract*) | what to do: reload the page |
| K2695-3 | **closed 2.69.5** | Results' *Reload from current project dir*; its empty state sent a person to a *View button next to each file* (it is in the file's ⋯ menu) and spoke of *inspector flows* | *Reload from the sidebar's folder*; the ⋯ menu's View; *the files that hold a calculation's results* |
| K2695-4 | **closed 2.69.5** | the optimization tab: *Build accepts .xyz / .pdb only*, a pseudopotential caption *(unset -- SIESTA will fail…)* and the server's full path, *Tick all* when the click unticks, a footer linking `/api/health` (and on the Molbuilder tab) | plain words; the folder said within the projects folder; the button says what its click does; the footer link gone |
| K2695-5 | **closed 2.69.5** | the Spectrum tab named *the Optimization tab* and promised a feature (*is planned*) | the tab's name; no promise |
| K2695-6 | **closed 2.69.5** | the Molbuilder tab's edits said their past tense while running (*Deleted…*, *Deleted failed: …*) | each edit names what it is doing and what it did |
| K2695-7 | **closed 2.69.5** | the history pill *33 unsaved*, its *Tag HEAD…* and *Sensor error*; the file viewer showing the server's full path | *33 changed*, *Tag this state…*, *the folder's history could not be read*; the path within the projects folder |
| K2695-8 | **closed 2.69.5** | the notebook tab named *JupyterNB* over a page saying *JupyterLab*; its messages and the reload's spoke of a *supervisor* | the tab *JupyterLab* (the user); the process said in words |
| K2695-9 | **closed 2.69.5** | one spin state worded twice — the chip's *closed shell* / *moment free*, the card's *restricted (closed shell, 2S = 0)* / *the moment floats (free)* | one wording, the chip's, both places |
| K2695-10 | **closed 2.69.5** | the save warning named its internal step (*write-state*) and said *retract / crash-recovery history* | the reason alone; *Retract and a reopened tab may not bring back everything* |
| K2695-11 | **later** | the run panel shows a phase's token as written (*negf*) where the SCF plot says *NEGF*, and an unknown field by its name; *CG/MD step 6* beside 8 frames (whether the two counts should agree, to be read); a *-0.0000*; the trajectory viewer named *Trajectory + SCF history* where the picker's group says *SIESTA optimization* | each when its page is next touched |

## Found by the UI review's group D, the server's words on a page, and 2.69.4's review round (2026-11-09), by the release that owns each

*(Group D's items about how the server's sentences reach a page, mapped to where each is drawn before any fix; one review round over every changed file, every finding read again in the code)*

| id | where | finding | door |
|---|---|---|---|
| K2694-1 | **closed 2.69.4** | every page showed the terminal's dash as two hyphens (` -- `), and many places painted a server sentence as plain text, its backtick-quoted command shown with its backticks | one renderer for the server's words, `codeWords` (`lib/dom.js`), loaded once for every page by the shared header: a command as code, the terminal's dash as a dash; every place a page shows a server sentence draws it so (`ui-contract.md` § 12.1) |
| K2694-2 | **closed 2.69.4** | capitals as emphasis or state in the server's sentences and the chips: *CONCLUDED*, *NOT CONCLUDED*, *ENDED WITH EXIT CODE*, *a STRUCTURE PAIR*, *TRANSPORT CALCULATION*, *FRAME 0 IS THE JUNCTION … RAN*, *DERIVED*, *CONTINUES*, the seam's *ECLIPSED* and *TWIN*, the run log's *CLEAN START* and kin, the state chips set in capitals by style | plain case; the first frame said as *its first frame* |
| K2694-3 | **closed 2.69.4** | the file card's *written by* named code functions (*describe door (`describe.write_description`)*, *`materialize.open_scratch`*) — the field is *one line a person reads* (`job-contracts.md`) — and the page stripped the parenthesised ones by a rule of its own, which also took away folder names | every writer line in a person's words at the source; the page's rule removed |
| K2694-4 | **closed 2.69.4** | *24 domain(s)*; *the v0.2 pair, 2 files*; the saved state's sentence leading with its history id | *24 queues*; *the electrodes' Green's functions at 0.2 V*; the note first, which leads with its time, and the id last |
| K2694-5 | **later** | still drawn as plain text: the folder picker's refusals, the source viewer's, a restore's finding in its dialog, a file option's label; MolView's and VibrationView's few server sentences (sealed, by their contracts) | through the renderer, when each is next touched |
| K2694-6 | **later** | the notification bar and two inspectors draw a sentence that carries a path or a name through the renderer, where a backtick in a file's name would turn to code | the sentence and the name drawn apart |
| K2694-7 | **later** | the findings renderer keeps a plain-text path for a page without the renderer, which its test harness relies on | removed with the harness loading the renderer |

## Found by the UI review's group D, the server's two words, and 2.69.3's review round (2026-11-09), by the release that owns each

*(The user's word of 2026-11-09 — one word, *stage*, for one calculation's step on every page — carried to every sentence the server writes, since its sentences are what pages show; and *time limit*, 2.69.2's one word for the scheduler's limit, likewise; every person-facing string read first, a review round over every changed line, every finding read again in the code)*

| id | where | finding | door |
|---|---|---|---|
| K2693-1 | **closed 2.69.3** | the server's sentences said *rung* where Task setup says *stage* — refusals, prep's and launch's lines, a stage's file named in a refusal, the transport and PySCF decks' comments, the catalogue's help, a data file's coordinate name | *stage* in every person-facing sentence; a data key, an API parameter (`surface=rung`, `rung=`), a module path and a code name unchanged |
| K2693-2 | **closed 2.69.3** (and K2692-6) | the server said *wall* for the scheduler's limit — the Preview's *one wall for the whole run*, *walls summed*, a fit's *over debug's wall*, the run's *stopped at its wall*, the run script's banner *Wall :*, the slack setting's label *Wall-time slack* | *time limit*, the queue card's word; *wall time* for measured elapsed time is another thing and stays, as do the scheduler's `wall` directive and the log's `[WALL]` tag |
| K2693-3 | **later** (F10, F11) | the documents still say *rung* and *wall* in their own prose; only their quotes of what molbuilder prints were changed | the documents made one word with the documents pass |

## Found by the UI review's group D, Task setup's words, and 2.69.2's review round (2026-11-09), by the release that owns each

*(Group D's Task setup items, mapped to where each word is made before any fix; the user, 2026-11-09: "continue", and on the two names asked: one word *stage* on every page, the notebook tab *JupyterLab*; one review round over the whole touched files, every finding read again in the code)*

| id | where | finding | door |
|---|---|---|---|
| K2692-1 | **closed 2.69.2** | Task setup named a setting by its file key in status lines (*Removing mpi_np…*, *Adding a point to mpi_np…*), refusals (*mpi_np has no value in force*, *… was not written*), the plan's lines (*mpi_np=2 · omp_threads=1*), the benchmark's rows and hint, and every control's spoken name (*value for omp_threads*, *coarse mesh_cutoff*) — while the form beside it said its label | one label door, `labelOf`, the catalogue's label or the run's own ask's (`lane_asks`); every place the page names a setting asks it (`task-setup.md` § 5) |
| K2692-2 | **closed 2.69.2** | one setting, two words: the queue card's *Time limit* and the run card's *wall clock for this run* | *time limit* on the page and in the run's own ask (`job_request.LANE_ASK_ROWS`) |
| K2692-3 | **closed 2.69.2** | *rung* beside *stage* for the same thing on the page | *stage* (the user, 2026-11-09) |
| K2692-4 | **closed 2.69.2** | the transport card showed the treatment's token (*low-bias-approximation*) and the rename's key (*swap_electrodes: true*) | the record's one wording of the treatment (`treatment_note`, now in present tense, read before anything runs as after); the rename said in words |
| K2692-5 | **closed 2.69.2** | capitals as emphasis: three headings set in capitals by style, *NOT written*, *REPORTED*; *combination(s)*; the Preview's hover saying it *runs* prep | plain case; a count said as a count; the Preview's hover says it writes nothing |
| K2692-6 | **to 2.69.3** | the server's own sentences on the page still say *wall* (a run of points' *one wall for the whole run*, a Preview's *walls summed*, the launch words) | the server's sentences, in the next release with every other server word on a page |
| K2692-7 | **later** | a disabled cell names the catalogue's roles as stages, where the picker names the calculation's stages; the transport hint spells *five stages* where the folder answer holds the list | one rule for both, when the stage table is next touched |

## Found by the UI review's group B, its last two items (2026-11-09), by the release that owns each

*(B6 and B13 of the UI review's findings, read against `web/spectra.md` § 8, `web/molview.md` § 9.3 and § 9.4 before any fix; the user, 2026-11-09: "continue")*

| id | where | finding | door |
|---|---|---|---|
| K2691-1 | **closed 2.69.1** | the Spectrum tab's Send card said to set frozen atoms *"in the viewer above"*, a read-only viewer that changes no label; `spectra.md` § 8 said the same | the card and § 8: the atoms labelled `frozen_atoms` on the Molbuilder tab and saved with the structure there |
| K2691-2 | **closed 2.69.1** | the MolView demo page advertised `__molview.getSelection()`, `onChange(fn)` and `setFrame` — none exist on the handle | the page names the handle's own calls: `selection`, `subscribe(fn)`, `setCurrentFrame(i)`, and `reloadFrames` for its trajectory button |

## Found by the UI review's group B, its answer items, and 2.69.0's review rounds (2026-11-08), by the release that owns each

*(B11 and B12 of the UI review's findings, read in the code against `web/ui-contract.md` and `ops/access-control.md` § 3.3 before any fix; the user, 2026-11-08: "Go ahead with recommendation but I want framework level fix not patching or hacking", and during the release: "make sure you are not patching, but understand the framework and do this systematically"; two review rounds over the whole touched modules, every finding read again in the code)*

| id | where | finding | door |
|---|---|---|---|
| K2690-1 | **closed 2.69.0** | This machine painted a refused answer as its state — *"No channels are set up … They live in undefined."* — and Task setup's channels and report fields alike | the one reader of a server's answer, `ask` in `lib/server.js` (`ui-contract.md` § 12); each page says a refusal beside what asked |
| K2690-2 | **closed 2.69.0** | no page knew the person was signed out: the notebook said *status refused* for good, Documents *not authenticated*, This machine painted it as data | the reader says *signed out* once for the page, its **Sign in** link the page itself in a new tab, what the page holds kept |
| K2690-3 | **closed 2.69.0** | each question read its answer its own way: bodies parsed without asking whether the server said yes, an error page called a network error (`lib/fetch-error.js`), the browser's *Failed to fetch* shown | every question through the reader; `fetch-error.js` retired |
| K2690-4 | **closed 2.69.0** | MolView asked the server itself; handed the reader instead, one host forgot it and the Structure optimization tab loaded nothing (seen live by the user on the working tree, 2026-11-08) | the app's doors for a viewer one bundle, `molviewDoors` (`lib/projects/molview-doors.js`), every page handing it whole; `mount` refuses a viewer with no server door, as one with no owner (`molview.md` § 4) |
| K2690-5 | **closed 2.69.0** | a saved state that could not be read back was taken for nothing saved: the reopened tab came up empty without a word | `readState` still answers null and says the failure on `onPersistError`; the bar's own row (`notifications.md` § 3) |
| K2690-6 | **closed 2.69.0** | the notebook's control routes refused outside the envelope (`message`), so the page kept a reader of its own | the routes refuse as `{ok: false, error}` (`_signalled`) |
| K2690-7 | **closed 2.69.0** | Task setup hid its machine card when the machines could not be read | said in the card |
| K2690-8 | **closed 2.69.0** | sentences of our own where the server's were due: the history's Save *HTTP undefined*, the export *HTTP 200*, the hand-over *render failed (200)*, the save warning *HTTP n* in place of the server's words | the reader's sentence everywhere |
| K2690-9 | **closed 2.69.0** | `isCancelError` left with no caller; the file preview's error kind, read by nothing, keyed on the old *network* wording | retired |
| K2690-9b | **closed 2.69.0** (judged, not repaired) | `test_the_page_remembers_which_file_it_is_showing_across_a_reload` reloaded the moment the readout said *Loaded:*, before the page's save of its note was sent — a save is sent without waiting, one after another (`workspace.md` § 3) — and lost the race more often once each save's answer was read whole | the test waits for the note's save before the reload, its expectation unchanged; seen failing with the page made to forget its note |
| K2690-10 | **later** | a selection filter the server could not answer selects nothing, silently (`model-jobs.js` `resolveFilter`; `molview.md` § 6.9 governs the doors that change the structure) | a decision about § 6.9's reach |
| K2690-11 | **later** | still silent on a failed answer: Task setup's memory-need note and its plan (hidden on an unreachable server as on an unreadable description), the history list (its old rows stay), the transport inspector's reload | each says its refusal, when its page is next touched |
| K2690-12 | **later** | `projects/api.js`'s `_fetchEnvelope` flattens the body beside `ok` and `error` — a second answer shape — and its name says it fetches | the projects calls on the reader's shape, or the adapter named in § 12 |
| K2690-13 | **later** | a classic script reaches the reader three ways (`whenReady`, the global at call time, an import); `runtime.md` says `whenReady` | one rule in § 12 and `runtime.md` |
| K2690-14 | **later** | the reader knows the sign-in page by the path `/login` alone; behind a proxy with a path prefix a page answered by sign-in would read as unreadable (unmeasured) | the gate's own URL |
| K2690-15 | **later** | small leftovers: the store's `root.fetch` guards guard a call it no longer makes; Task setup's two flags for the machines' one answer; its vocabulary reads dropping the reader's sentence; the bar choosing its row by the operation's name; the spectrum's follow telling *nothing read* by `undefined` from `null`; every GET now `no-store` (speed only) | each when its module is next touched |

## Found by the UI review's group B, its folder item, and 2.68.5's review round (2026-11-08), by the release that owns each

*(B9 of the UI review's findings, read in the code against `web/projects.md` § 2 before any fix; the user, 2026-11-08: "Go ahead with recommendation but I want framework level fix not patching or hacking"; one review round over the whole touched modules, every finding read again in the code)*

| id | where | finding | door |
|---|---|---|---|
| K2685-1 | **closed 2.68.5** | the sidebar's rename, move, copy, upload and delete, and the file viewer's Save, announced no folder change: Task setup and the run history went on showing a folder that had changed | the file doors announce every folder they changed, once it has succeeded (`state.js` `_changed`), and re-list nothing by hand; the sidebar's deletes and the viewer's Save through the doors |
| K2685-2 | **closed 2.68.5** | a page writing through its own server door re-listed the sidebar by hand, or announced nothing — MolView's structure save, the hand-over, the spectrum's frame set, the transmission digest's Read | each announces its folder, the one call |
| K2685-3 | **closed 2.68.5** | a copy or a move re-listed the sidebar by opening the destination, moving the person to another folder | the sidebar re-lists the folder it shows when announced; a file door never moves it; a copy says where it went |
| K2685-4 | **closed 2.68.5** | had Task setup read its folder again on every announcement, a file renamed beside it would have dropped what it held unsaved without a word | unsaved, kept and said in the save bar (`task-setup.md` § 2.1) |
| K2685-5 | **closed 2.68.5** | a page skipped its own announcement by a flag set around a synchronous publish, which an awaited delete escapes | an announcement names its writer (`{dir, by}`), and a page skips its own |
| K2685-6 | **later** | the sidebar re-lists from the folder last stored as shown: an announcement landing while a folder click is still listing re-lists the old one, and the click is lost — as before 2.68.5, now on more roads | the re-list naming the folder being listed |
| K2685-7 | **later** | Task setup holding a proposal it has not saved (a hand-over opened) counts as unsaved, and follows no outside change until saved | the proposal's own state, when the hand-over road is next touched |

## Found by the UI review's group B, its keyboard items, and 2.68.4's review rounds (2026-11-08), by the release that owns each

*(B8 and B10 of the UI review's findings, each read in the code against `web/ui-contract.md` § 4.1 before any fix; the user, 2026-11-08: "framework level fix not patching or hacking"; review rounds over the whole touched modules, every finding read again in the code; the roads walked in Chrome on the dev server)*

| id | where | finding | door |
|---|---|---|---|
| K2684-1 | **closed 2.68.4** | the spectrum's modes table sorted by a click on a header cell alone and chose a mode by a click on its row alone | the column names are buttons (`button.plain`); the modes are the app's one list of choices — one stop, the arrows, Enter as a click (`lib/one-stop-list.js`) |
| K2684-2 | **closed 2.68.4** | the run history showed a state's *Restore to here* only after a click on its row, the row no stop of the keyboard; its pill re-read the state on a click alone | each state's words a button opening its actions (`aria-expanded`), the buttons one list of choices, *Restore* the next stop of an open row; the pill a button |
| K2684-3 | **closed 2.68.4** | a Documents folder opened and closed by a click on its row alone, saying neither | its arrow a button (`button.plain`) named for the folder, `aria-expanded` |
| K2684-4 | **closed 2.68.4** | the Transport engine strip, unlike every other strip, was not mounted by `lib/tab-strip.js` — its panel named by nothing | mounted |
| K2684-5 | **closed 2.68.4** | the sidebar's list keys were its own code, a second list would have been a copy | `lib/one-stop-list.js`, the one list, the sidebar on it |
| K2684-6 | **closed 2.68.4** | a button reading as its text was reset in each module that drew one, the shell's hover still changing its look | `button.plain`, the shell's, one rule |
| K2684-7 | **closed 2.68.4** | a Restore ran outside the page-wide cover, every other heavy click under it (`ui-contract.md` § 10) | `pageBusy.run`; a state's actions kept open across the redraw |
| K2684-8 | **closed 2.68.4** | Documents opened on a document more than one folder deep left the folders above its own closed — the climb searched its siblings; a click on a parent document's name folded its own documents | the climb by the page's nesting; a click inside a document's link opens the document only |
| K2684-10 | **closed 2.68.4** | a list drawn again under the keys -- the modes while a run is followed, the sidebar's folder opened by Enter -- left them on the page's body; a row a filter puts out of view could take the stop | `redraw`: the keys on the item carrying their id, else on the stop; an item out of view of its own passed over |
| K2684-11 | **later** | the sidebar's rows are `option`s, each holding its menu button -- a control inside an option, which assistive technology reads poorly (`nested-interactive`) | the sidebar's rows as the history's: a list of buttons, when the sidebar's menu is next touched |
| K2684-9 | **later** | the history's graph view shows a state's facts on a click of its node alone | the list view holds the same facts, by the keys; the graph's nodes when the graph is next touched |

## Found by the UI review's group B, its first item, and 2.68.3's review rounds (2026-11-08), by the release that owns each

*(B7 of the UI review's findings, reproduced on the dev server and logged event by event in Chrome before each fix; three review rounds over the whole touched modules and chapters, every finding read again in the code; the fix walked in Chrome on the dev server)*

| id | where | finding | door |
|---|---|---|---|
| K2683-1 | **closed 2.68.3** | in Task setup a value typed and Tab left focus on the page's body — a run card's field, a stage cell, the queue card's wall time: while the change of the field Tab leaves runs, Chrome names no control focused, so the fence's claim took the body for where focus was, took focus to the cover, and the browser abandoned its move; the write redrew the card, and the release had nothing to give focus back to | the fence's own record of where focus is; a claim made while focus is moving leaves the move to finish; focus given back once the operation's caller has ended it and the move has finished, never taken from a control that has it; a redrawn control found again by its id or by its name in its container, never guessed between two (`ui-contract.md` § 10) |
| K2683-2 | **closed 2.68.3** | the stage table was drawn again 400 ms after the page's own write, after the fence's release, taking focus from where the person had moved | the editor's read-back from a hand edit only, kept across a card's write inside its pause (`task-setup.md` § 7a, § 9a) |
| K2683-3 | **closed 2.68.3** | a stage renamed, then Tab or a click in its row: the write waits on nothing, so the table was drawn again inside the change, taking out the control the move was going to — Chrome abandoned the move, focus back on the name; and the row's controls, named by the stage's name, were not found again after the rename | `commitEdit` draws again once the event that made the edit has finished (`afterTheEvent`); a stage row's controls carry an id by the row's number (§ 7a; `ui-contract.md` § 10) |
| K2683-4 | **closed 2.68.3** | a blank run row went on saying the queue card's old wall clock after the queue card changed, until another edit redrew the card (measured on the dev server) | a block's write draws the run cards again, once the event that made it has finished — drawn inside it, a click into a run card's field was dropped and focus went back to the wall clock (`applyBlockToDoc`, § 7a) |
| K2683-5 | **closed 2.68.3** | controls told apart by nothing: a group's wall-clock rows all *"value for time"*, each member's bench box's *Preview bench* / *Prep bench here* | each named for what it edits, the words on it first — *"value for wall clock · seed"*, *"Preview bench — seed"* |
| K2683-6 | **closed 2.68.3** | `test_a_folder_opens_showing_what_it_already_states` expected the GPU switch's blank row to read *false*, the word 2.68.1 replaced with *off* (`task-setup.md` § 5.2) — a browser test outside 2.68.1's touched set, failing on 2.68.2's code too | the expectation the contract's words |
| K2683-7 | **later** | a cancellable operation started as a field is left — none today: `editTyped`, `commitEdit` and `applyBlockToDoc` pass no Cancel — would leave Cancel unreachable by the keyboard: the cover takes no focus while a move finishes, and every key outside it is fenced | the cover's Cancel taking focus whenever there is one, with the first such operation |
| K2683-8 | **later** | a hand edit followed within the read-back's pause (400 ms) by another folder opened reads the new folder back as a hand edit — as before 2.68.3 | the pause cleared by a folder's read |

## Found by the UI review's group A and 2.68.2's review round (2026-11-08), by the release that owns each

*(group A of the UI review's findings, each reproduced on the dev server before any fix; one review reader over the whole touched page modules, every finding read again in the code)*

| id | where | finding | door |
|---|---|---|---|
| K2682-1 | **closed 2.68.2** | the Molbuilder page's *Unsaved* came from the viewer's timeline flag, which every edit clears by laying down its own point: after a save and an edit the readout said only *"Target: x.xyz"*, and Load's add-or-clear question never said the view held unsaved changes (tabs.md § 2, § 6) | the page's own note compares what the view holds with what it last wrote or loaded (`page.js` `markSavedTo(path, structure)`, `_onCanvasChanged`); kept across a reload |
| K2682-2 | **closed 2.68.2** | after every reload the Save readout said *"Not saved yet"*, the note read back but its readers never told | `restorePanelNote` repaints its readers |
| K2682-3 | **closed 2.68.2** | the viewer's own Export → Data → Save to project wrote the file and left the page's note as it was (tabs.md § 6: one door out) | `page.savingThrough(files)`, the door the page hands its viewer |
| K2682-4 | **closed 2.68.2** | Add slab with the lattice-constant box emptied built with a table value the panel did not show (the user, 2026-09-07: "the box is never blank") | an empty box sends nothing and says so; any number is sent, the server refusing ≤ 0 itself |
| K2682-5 | **withdrawn at the user's word** | Results' file viewer offers Edit and Save on a finished run's deck — *"Why … would you assume the user is a … idiot who would modify the result folder … If he does so just say so that the dir is messed up"* | none: a folder the person changed is read and said as found |
| K2682-6 | **withdrawn on reading the code** | Transform turning the measuring tool on: it says so (*"Measuring is on: the Transform tab picks atoms with the ruler…"*, `modify/viewer.js:680-692`), as molview.md § 11.6 asks; staying on is the person's to turn off | none |
| K2682-7 | **later** | two meanings of *unsaved* side by side: MolView's corner badge *"Unsaved changes"* (the timeline) beside the Save readout's (the file); the Structure-optimization tab (`viewer.js:738-745`) and the projects parser's discard confirm (`parser.js:63`) read the timeline flag | each page's own note, or the badge's words |
| K2682-8 | **2.67.12** | the slab panel's empty-box answer has no page test; checked on the dev server | the page-side test doors |
| K2682-9 | **the dead-code release** | `page.markDirtyAfterModification` and `page.onCanvasChange` have no caller in the static JS | its list, shown first |

## Found by 2.68.1's review round (2026-11-08), by the release that owns each

*(K2680-4 and K2680-5, the user's word; one review reader over the whole touched modules and chapters, every finding read again in the code; the fast batch and the touched browser tests)*

| id | where | finding | door |
|---|---|---|---|
| K2681-1 | **closed 2.68.1** | a blank setting's default said *recommended* on every form and in Task setup's hover, an engine's own value among them | `template.default_words` — *recommended for a vibration* (`recommended`), *SIESTA's own default* (`engine_default` equal, an engine reading keywords), else *default* — served on every form field, Task setup's columns and run-card routes; `web/form-schema.md` § 1.1, `web/task-setup.md` § 5.1 |
| K2681-2 | **closed 2.68.1** | Task setup named a setting by its catalogue name and its values raw — the picker, the column header, the cells, a fixed cell, the hover of a cell its rung does not read, the remove question, the status lines, the run card's options | the catalogue's label and `template.value_labels` (a switch *on* / *off*), served with the columns; the name only in the hover beside the keyword it writes; `web/task-setup.md` § 5, § 5.2 |
| K2681-3 | **later** | a switch's words have two sources: the server's *on* / *off* (`template.choice_words`, `value_labels`) and `form-schema.js`'s tri-select *yes* / *no* for an optional switch (`scf_must_converge`, `parallel_over_k`) | the form's tri-select reads the served words |
| K2681-4 | **later** | a TBtrans setting's default is said *SIESTA's own default*, its source TBtrans's (`tbt_reinit_m.F90`; TBtrans ships with SIESTA 5.4.2) — imprecise, not false | the words name the program a rung's item is read by (`engine_table` `programs`) |
| K2681-5 | **2.67.12** | the pages' reading of the served words — a caption, the stage table's header, picker, cells and hover — has no page test; checked in the browser on the dev server | the page-side test doors |
| K2681-6 | **later** | a page-typed fallback, *default*, where a field arrives with no words — none of the server's routes sends such a field today, only `test_form_schema_setvalues_js.py`'s own schema; `diffFromDefaults` still names its field `recommended` | the fallback goes once every caller is the server's |

## Found by 2.68.0's review rounds (2026-11-08), by the release that owns each

*(the UI review's transport walk: a hydrogen-chain device whose charge never settled; the user, 2026-11-08: "expose the option as we planned because it is a valid parameter that the user should be able to setup"; two review readers over the whole touched modules and chapters, every finding read again in SIESTA 5.4.2's source)*

| id | where | finding | door |
|---|---|---|---|
| K2680-1 | **closed 2.68.0** | TranSIESTA's charge correction, `TS.dQ`, offered nowhere: a device whose periodic start set its reference energy 0.31 eV off held 0.18 % of its electrons too many and ran its 1000 cycles to `SCF_NOT_CONV`, with no setting a person could reach | the catalogue row `ts_dq` (device, `none` · `fermi` · `buffer`, TranSIESTA's `none` by default), its config field, the device's TranSIESTA section; `engines/transport.md` § 2a.12, § 2a.13 |
| K2680-2 | **refused at the user's word** | a draft wrote `TS.dQ` at the 0 V point only, through the catalogue's `active_when` stretched to decide each voltage's deck, with help saying when to choose it — the user: a scientific parameter is "entirely … left to user's judgement"; `active_when` marks only a setting the calculation the person built does not use | removed before the release; the setting is written as chosen at every voltage, its help states what each choice and the engine do |
| K2680-3 | **closed 2.68.0** | the review's corrections to the first help and chapter text: `fermi` starts only within 50 times the density and Hamiltonian tolerances and holds convergence while the charge is off by more than 0.01 electrons, whatever the charge check says (`siesta_forces.F90:432–452`) — so the check's own help was untrue under it; a wrong source for the periodic start's one k-point; "at its start" true of a frame's first voltage only; a document path in a help; the 0.8 factor, the 1.5 eV cap, the spline inside one cycle; a T(E) claimed for a run never made; a duplicate test row | each in its help or chapter; the duplicate row retired (the generated tab row covers it) |
| K2680-4 | **closed 2.68.1** (the user, 2026-11-08: "ok continue") | every form shows a catalogue default as *recommended* — *"not chosen · recommended No correction"* (`form-schema.js:193–200`; Task setup's hover, `task-setup/viewer.js:2530–2531`; asked by `form-schema.md`) — where the value is an engine's own default, not molbuilder's advice | a change of its own, every form |
| K2680-5 | **closed 2.68.1** (the user, 2026-11-08: "ok continue") | Task setup's stage-table column shows a choice setting's raw values (`none`, `fermi`, `buffer`) and its picker the item's internal name (`ts_dq — …`) | a change of its own, every choice column |
| K2680-6 | **later** | SIESTA's line writer learns a setting's `active_when` only through which sections a deck has: every SIESTA setting with one today sits in a section absent when it is unused (the relaxation's, the optical phase's), so no file is affected; a future one placed in a section always present would be written as if it ran | the writer asks `template.inactive_settings`, as PySCF's does — no file changes today |
| K2680-7 | **later** | no test checks a catalogue `engine_default` against the engine (`TS.dQ`'s read by hand: `ts_dq.F90:75–76` and SIESTA's own fdf log, `TS.dQ none # default value`); `buffer` is written in no test deck | the catalogue's engine defaults, as a whole |

## Found by 2.67.11's server-side test doors (2026-11-08), by the release that owns each

*(an inventory of F9-6 in two readers, split three ways at the user's word; the shared doors built first in one copy, then four fixers on disjoint files, then the 2026-11-02 audit's retire list judged by breaking each guarded line and applied; two discipline rounds of two readers over every touched file whole, each finding read again in the code, and their fixes; the fast batch, and the 27 browser and real-run files the release edited — the user's choice, 2026-11-08: "Only the 27 edited files")*

| id | where | finding | door |
|---|---|---|---|
| K27611-1 | **closed 2.67.11** | the projects tree pointed by hand: `Capabilities.file_picker_roots` patched in 10 files (the server's folder list and `projects_root()` disagreeing inside one test, which no real server does — with it made to stop following the root, 158 tests in 12 files now go red, where the old patches hid it) and `PROJECTS_ROOT_ENV` set by hand at ~50 sites | `conftest.point_the_one_door_at`, `isolated_projects_root(_module)`; only `test_projects.py`, `test_psml_anchor.py` set it (the root's resolution is their subject) |
| K27611-2 | **closed 2.67.11** | two makers of a described calculation stating the run card differently (`road.describe_calculation` always 2 ranks, `real_runs.described` none), ~25 calculations described or their records typed by hand | `support.road.described` (the run card `RUN_CARD` unless stated; `psml_lib=`), the tree by `support.road.road_tree` or `support.real_runs.road_at`; where a describe's printed answer is what a test reads, the describe stays typed, said in the file |
| K27611-3 | **closed 2.67.11** | private support names imported by other files (9) and helpers borrowed from test files (12) | public names in `support/` (`road`, `junction`, `pages`, `chemistry_card`, `live_server`); no test imports a test |
| K27611-4 | **closed 2.67.11** | one maker per real run: the displacement sweep's procedure written twice; the GPU sampler patched in two files (the load monitor's tests could not catch a broken sampler: with the sampler made to read stderr only, 2 of 3 now go red); the road's H2 optimization re-typed in two e2e files; a CPU-only machine never reached on this host (an RTX 3060 Ti): with the page taking an empty GPU list as a fault, the old test stayed green, the new one goes red; the shared junction run's digest moved away and back in a `finally` | `support.real_runs.swept`, `h2_optimization`, `copy_of`; `support.machines.a_gpu_tool`, `no_gpu_tool` |
| K27611-5 | **closed 2.67.11** | TA-RETIRE's rest — 6 whole files and 69 tests still standing of the audit's retire list, each judged by breaking its guarded line: keep 28, keep rewritten on road input 9, merge 27, retire 25 (17 differ from the audit, each on a measured break); TA-ISOLATE (ii): the `--clean` test ran on the real `sys.prefix` — now a recording env manager | the list below |
| K27611-6 | **closed 2.67.11** | tests that could not catch what they name, found on the way: the PySCF halt's words (since 2.67.1: asserted "relaxed on" absent, which both the product's branches say — now "against a target of … not reaching it", `engines/vibration.md` § 2.2); two transport describe refusals green with their rule removed; Task setup Save's no-stages refusal green with its rule removed; the solvated vibration green with the solvent removed (now its enthalpy, 9.4 mEh below the gas run's); a docstring keeping hand-made input on a false reason ("no road makes that folder": describe, prep and the zip door make it) | each test's own assertion |
| K27611-7 | **refused at the merge** | a fixer moved the workspace store's isolation onto the pointed tree, emptied at each test's end by the projects variable's value — which deletes under whatever tree that variable names, the person's own if it ever does | the store stays patched onto each test's own temp root (`conftest._isolated_workspace_store`) |
| K27611-8 | **unrun, named** | `tests/field/test_the_manager_answers.py`'s doctor test (replaces two retired basic tests; needs a probed target record) | the field pass on Sol |
| K27611-9 | **later**, by owner | K27611-L1 (2.67.12): `test_task_setup_tab.py`'s `_THE_SERVED_DOCUMENT`, a 150-line DOM stand-in this release added for the rewritten designs-parts test, beside `support/molview_dom_standin.js` and `test_notify_js`'s `_DOM` — the page tests' DOM, one door, 2.67.12's.; K27611-L2 (2.67.12): `support/molview_server_standin.js` (moved, not replaced: typed server replies) — fed the real server's answers in 2.67.12; its two "used to carry" tombstones go with it.; K27611-L3 (later): `test_task_setup_tab.py:~1177-1203` arranges the config by hand (`MOLBUILDER_CONFIG_DIR`, `molbuilder.json`, `environments/` by `mkdir`) where `machine_config`/`config_root` and `ensure_private_dir(environments_dir())` are the doors.; K27611-L4 (later): `test_engine_offset_reaches_every_deck.py:~50` keeps a local `_water()` (5 Å vacuum) beside `support.road.water_in_vacuum()` (8 Å).; K27611-L5 (later, low): `format_of_text`'s blank-line skip (`workingcopy_structure.py:~604-607`) caught by no test now — and the XYZ reader refuses such text ("XYZ holds no frames"): a product question (does a pasted XYZ with a blank first line read?), the dead-code release's to judge.; K27611-L6 (later, low): a door pointed twice (`test_where_an_item_binds.py:~92`, `test_task_setup_tab.py:~484` request `isolated_projects_root` and call `road_tree`); `test_transport_task.py:~205` writes `j.xyz` where `structure_saved` exists; `real_runs.py:~463` writes pseudopotentials `live_siesta` already copied; fixtures named `described` importing the maker (`test_checkpoint_sensor_js.py:~230`, `test_prep_from_the_browser.py:~36`, `test_notify_setup_api.py`); `fresh_calc_dir` in support with one user; `test_envs_recipes.py:~75`'s message reads the direction backward.; K27611-L7 (later): `test_workspace_storage_api.py:~17-27` rebinds the blueprint's `projects_root` to undo conftest's store patch — kept: conftest's store isolation stays a patch onto each test's own temp root; the alternative tried in this release (the store under the pointed tree, emptied at teardown) deletes under whatever tree a variable names when a test ends, which could be the person's own — refused at the merge.; K27611-L8 (later): `test_system_load_monitor_e2e.py:~135-141` runs the host's own `nvidia-smi` for the healthy case — on a host whose driver is broken it fails for the host (not new); `support.machines.DRIVER_MISMATCH` prints typed driver text (one wrong input, one rule; testing.md § 6 asks the reason be stated).; K27611-L9 (later): "a hand-over as the tab leaves one" made three ways (`test_task_setup_one_folder_e2e.py:~53`, `test_issues_card.py:~127`, `test_task_setup_tab.py`); `support.road.until` copied as `_until` (`test_notebook_through_one_port_e2e.py:~88`) and an inline wait (`test_sequence_of_groups_e2e.py:~116-124`); a tree spelled `"projects"` by hand (`test_inspector_registry_e2e.py:~141`), a bench folder spelled (`test_bench_summary_inspector_e2e.py:~52`) — older than this release.; K27611-L10 (a design question, the user's): a named target machine record has no door — "sol" written by hand ~12 times, two in `support/` (`road._road_target`, `real_runs._header_under_a_slurm_grant`).; K27611-L11 (later): `test_where_an_item_binds._overridden` a third way of writing an override refused at Save; `test_launch_ask_mode._no_scheduler_on_path` copies the road's PATH filter; `test_describe` calls `described()` where only the tree is needed; `in_a_folder_of_its_own` redundant in `test_hard_limits`, `test_routing_domains`; the engine passed to both `road_tree` and `described`; `support.junction.LAYERS_*` one user; `watch/test_api_load.client` copies `web_client`; `test_web._PATTERN_B_REGIONS` unused.; K27611-L12 (later, testing.md § 3b): enumerations of wrong input older than this release — `test_job_request_notify`'s seven tables, `spectra/test_blueprint.py:~49-82`.; K27611-L13 (2.67.12): `test_pages_no_js_errors` carries `subscriber_tabs`' failure as a browser test under a fast-tier name with a 250 ms wait; `test_task_setup_tab.py:~200` imports `tests._node_esm` beside `_node_esm`; `test_checkpoint_sensor_js._DOM` (with L1).; K27611-L14 (later, low): `test_what_a_kind_offers.py:~86-89`'s solvated half green with the solvent call removed (read, not run; the e2e covers it now).; K27611-L15 (later): `described()` then `one_rank()` restating the card (`test_transport_decks_e2e.py:~43-46`, `test_bench_summary_inspector_e2e.py:~43-44`); two makers of "H2 as a SIESTA vibration stated relaxed" (`test_spectra_no_spectrum_sentence_e2e.py:~62-80`, `real_runs.sweep`); more doors pointed twice (`test_bench_summary_inspector_e2e`, `test_task_setup_reacts_to_a_restore_e2e.py:~33-48`, `_described_h2`/`_describe` in the three Task setup e2e files; `test_results_blueprint.py:~174`) — with L6; a pair written into the shared session run and deleted in `finally` (`test_a_person_reads_a_pyscf_run_e2e.py:~65-73`; `copy_of` is the road); `capped`/`walled` yielding inside `on_the_road` (the module's tests run with its tree pointed — harmless, as in HEAD); 10 untouched e2e files keep the e2e marker; `described`'s docstring silent on a PySCF optimization's one stage; cosmetic indents.; additions to L3, L9 as round 2 found them | each where it belongs |
| K27611-10 | **the dead-code release** | product findings: `identity.continues`' branch for an object with no `restart` reached by no road; `convert_to_engine_numbering` has no product caller; `parse_manifest`'s BOM and CR checks removable with every test green; the default k-grid offset pinned by one kept test only; `format_of_text` skips leading blank lines that the XYZ reader then refuses ("XYZ holds no frames"); `cli._resolve_tls` and the bind check both gate an incomplete pair | that release, its list shown first |

### What 2.67.11 removed (the list the user saw before the commit, generated from the code against 2.67.10, 2026-11-08)

**From the product**: nothing.

**Test functions gone** — each retired (what then goes unseen, said in the verdict list K27611-5's judge recorded), merged (another test goes red on the same break), or rewritten under a new name on road input; generated from the code:

- `parse/test_registry.py`: `test_engine_parsers_registered`
- `parse/test_round2_fixes.py`: `test_frozen_dataclass_invariant_on_each_result_kind`
- `spectra/test_blueprint.py`: `test_app_header_includes_spectrum_tab`, `test_page_is_generate_only`, `test_page_loads`, `test_plotly_not_loaded_on_spectra_generator`
- `test_auth_config.py`: `test_tenant_id_defaults_to_common`
- `test_build_format_sniff.py`: `test_blank_leading_lines_skipped`, `test_leading_zero_line_is_pdb_not_xyz`, `test_pdb_header_before_first_atom_is_pdb`
- `test_checkpoint_cli.py`: `test_restore_returns_the_whole_folder`
- `test_cli.py`: `test_electrode_gap_is_refused_by_name_not_as_a_typo`, `test_modify_delete_only`
- `test_cli_tls.py`: `test_resolve_cli_flag_overrides_json`, `test_resolve_json_with_only_key_falls_back_to_http`, `test_serve_cli_cert_only_warns_and_falls_back`, `test_serve_cli_flags_engage_https`, `test_serve_json_only_engages_https`, `test_serve_no_tls_passes_none_ssl_context`
- `test_engine_atom_index.py`: `test_dispatch_matches_named_forward_functions`
- `test_envs_builds.py`: `test_buildspec_rejects_duplicate_component_names`, `test_buildspec_rejects_empty_components`, `test_default_jobs_capped_at_8`, `test_default_jobs_floors_at_1`, `test_default_jobs_honours_env_override`, `test_describe_phase_returns_action_and_cost`, `test_phases_in_canonical_order`, `test_resolve_paths_layout`, `test_strips_blas_fftw_hdf5_overrides`, `test_strips_compiler_binary_overrides`, `test_strips_compiler_driver_flags`, `test_strips_compiler_include_paths`, `test_strips_cuda_location_overrides`, `test_strips_lib_loader_paths`, `test_strips_mpi_location_overrides`, `test_strips_mpi_prefix_families`
- `test_envs_doctor.py`: `test_report_all_honours_envs_override`, `test_report_all_marks_envs_present_when_in_caps`
- `test_envs_recipes.py`: `test_no_two_recipes_share_a`
- `test_envs_siesta_gpu_recipe.py`: `test_siesta_clones_with_recurse_submodules`
- `test_envs_validate.py`: `test_the_long_upstream_suites_are_not_optional`
- `test_launch_ask_mode.py`: `test_the_table_says_the_time_is_an_estimate_that_moves`
- `test_restart_group.py`: `test_a_missing_restart_field_reads_as_the_catalogue_s_default`
- `test_script_emit.py`: `test_begin_end_markers_are_literal`, `test_blocksize_field_constrains_pow2_and_leaves_the_range_to_the_deck`, `test_emit_bench_marks_carries_version_tag`, `test_emit_bench_marks_uses_anchors_not_line_numbers`, `test_emit_provenance_includes_required_fields`, `test_siesta_bench_fields_cover_the_four_bench_knobs`, `test_user_custom_placeholder_is_well_formed`
- `test_smiles_and_siesta.py`: `test_smiles_bdt_has_two_sulphurs`
- `test_structure.py`: `test_default_is_empty`
- `test_structure_smiles_js.py`: `TestConfigurationErrors`, `test_generate_rejects_when_structurePage_unconfigured`
- `test_task_setup_tab.py`: `test_the_card_reuses_the_shared_choice_component`, `test_the_page_renders_the_designs_parts`, `test_the_runtime_loads_before_every_other_script`, `test_the_tab_is_in_the_roster_and_routes`
- `test_template_declarations.py`: `test_range_unit_and_group_come_from_the_CATALOGUE`
- `test_transport_task.py`: `TestInitCLI`, `test_slot_on_a_non_transport_init_is_refused`
- `test_web.py`: `test_build_load_response_includes_atom_metadata`
- `test_web_files.py`: `TestRootsContract`, `test_capabilities_returns_only_projects_root`, `test_subscriber_tabs_use_inquire_api`
- `test_workingcopy_structure.py`: `test_codec_writes_xyz_and_sidecar`
- `test_workspace_tag_isolation.py`: `test_the_workspace_does_not_open_what_it_was_given`

**Helpers, fixtures and constants gone or renamed** (each a copy of a door, a hand-made input, or a private name made public — `support.road.describe_calculation` and `support.real_runs.described` are now `support.road.described`; `support.real_runs._h2_optimization` is `h2_optimization`; `support.road._all_written`, `_description_with_overrides`, `_road_card_written`, `_road_points_before_prep` are `all_written`, `description_with_overrides`, `card_written`, `points_before_prep`; `support.junction._LAYERS_*` are `LAYERS_*`):

- `support/junction.py`: `_BRIDGE`, `_LAYERS_L`, `_LAYERS_R`
- `support/real_runs.py`: `_h2_optimization`, `described`
- `support/road.py`: `_all_written`, `_description_with_overrides`, `_road_card_written`, `_road_points_before_prep`, `describe_calculation`
- `test_api_bench_summary.py`: `_set_picker_root`
- `test_checkpoint_routes.py`: `_allowed_root`
- `test_chemistry_card_e2e.py`: `_answered`, `_blocks`
- `test_effective_parameters_record.py`: `the_fence_holds`
- `test_electronic_state.py`: `WATER`
- `test_engine_offset_reaches_every_deck.py`: `_assigned`
- `test_job_request.py`: `DESCRIPTION`, `EXAMPLE`, `REPO`
- `test_job_request_allocation.py`: `_BASE`, `_DESCRIPTION`, `_task`
- `test_job_request_notify.py`: `_BASE`, `_DESCRIPTION`, `_task`
- `test_jupyter_contract.py`: `_free_serve_port`
- `test_lattice_from_run.py`: `picker_root`
- `test_launch_ask_mode.py`: `_jobset`
- `test_launch_door.py`: `_states_its_wall_and_memory`
- `test_periodicity_gate.py`: `client_with_root`
- `test_prep_bench_fold.py`: `_describe_cpu`, `_state_the_run`
- `test_prep_from_the_browser.py`: `_cli`, `_make_preppable`, `_pyscf_calc`
- `test_pyscf_relaxation_outcome_e2e.py`: `_jobset`
- `test_restart_group.py`: `Bare`
- `test_sequence_of_groups_e2e.py`: `_calculation`
- `test_siesta_vibration_deck.py`: `FIXTURES`, `in_tree_psml`
- `test_spectrum_results_e2e.py`: `_load`
- `test_system_load.py`: `_a_tool`
- `test_system_load_monitor_e2e.py`: `_set_gpu_error`
- `test_target_machine_choice.py`: `_REC`
- `test_task_setup_tab.py`: `_fresh_calc_dir`
- `test_transport_compose.py`: `_junction`
- `test_transport_on_a_real_junction_e2e.py`: `_write_junction`, `described`
- `test_transport_task.py`: `_init`
- `test_vibration_e2e.py`: `WATER`, `_asks`, `_describe`, `_prep_and_run`
- `test_wall_reaches_the_engine_e2e.py`: `_write_h2`
- `test_web_files.py`: `_only_tmp_roots`
- `test_where_an_item_binds.py`: `_described`, `_template_beside`
- `watch/test_api_load.py`: `_TmpRootCaps`, `client_with_default_roots`

**Data file gone**: `tests/data/job_request_example.json` (read by no test). **The e2e marker** dropped from the 11 touched files that repeated conftest's.

## Found by 2.67.10's review rounds (2026-11-07), by the release that owns each

*(four fixers on separate copies, each fix's rule written first and its test seen failing on the old code; the work merged and the fast batch run; then two discipline readers, the server side and the page side, over every touched module read whole, each finding read again in the code before it was fixed or recorded; the fast batch and every browser and real-run test the release touches run again)*

| id | where | finding | door |
|---|---|---|---|
| K27610-1 | **closed 2.67.10** | K2679-2: the PySCF reader claimed every `.xyz`. Each XYZ is now read by the reader that is for it, by its name: geomeTRIC's `_geom_optim.xyz` by `pyscf`, a PySCF deck's two geometries by `pyscf-geom`, every other XYZ (a saved structure, SIESTA's `<label>.xyz` and `.ANI`, a calculation's `<label>.source.xyz`) by the new `xyz`; and a folder no calculation claims opens nothing. The review added: `parse/engines/__init__.py` said order decides among parsers, where `detect` refuses a file two claim; `presenters.md` § 2 still named `*_optim.xyz`; SIESTA's `.ANI` opens in no viewer, now said (`presenters.md` § 1) and a row | `parse/engines/xyz.py`, `runs._opens`; `tests/parse/test_registry.py` (one table), `tests/data/presenters.toml` |
| K27610-2 | **closed 2.67.10** | K2679-3, the user's decision (*"calculations can continue from another, not within another"*): describe refuses a folder inside another calculation's folder or holding one, at every describing door; a history never begins inside one; the transport describe's own narrower refusal is gone. The review added: the hand-over's refusal named the folder by the full path the page sent, where the command line named it by its address, and three callers composed the address by hand: one door now names a folder by its address (`projects.address`), which `bundle_flag` takes too; a folder inside a calculation that a person had run `git init` in was adopted as a history (the inside question came after the repair), now refused; the walk above stopped at the configured projects root, not the tree the folder lives in (`projects.find_projects_root`); the walk below went into `.git` and a describe's staging folder | `describe.refuse_described`, `checkpoint.Repo.init`, `calcdirs.calculation_above`/`calculation_below`, `projects.address`; `test_describe.py` rows, `test_checkpoint_states.py` L1 rows, `test_build_e2e.py` |
| K27610-3 | **closed 2.67.10** | K2679-4: a folder picker asked before the page knows the projects root waits for it, and says why when the root cannot be resolved. The review found two ways to wait for the root — the new answer beside the old `onProjectsRootResolved`, which said nothing on a failure — so the notebook tab waited 8 s on a failure the page already knew: one answer now (`whenProjectsRoot`: the root, or why not), the old door removed, and the notebook tab opens at the projects root at once when the root cannot be resolved. The browser test released the root on a timing it did not hold: now held by the picker's module having loaded | `lib/projects/state.js` `whenProjectsRoot`; `web/projects.md` § 5, `web/jupyter.md` § 4.0 |
| K27610-4 | **closed 2.67.10** | K2679-8's double scan: a page restored from the browser's cache scanned its folder twice. Coming back is the page turning visible; the picker and each viewer listen to that one event. The review: the test's docstring and the test maps claimed it caught a second ask by the viewers, which their own guards already stopped | `lib/results/file-picker.js`; `test_coming_back_to_the_tab_e2e.py` |
| K27610-5 | **closed 2.67.10**, found on the way | an uploaded geomeTRIC trajectory was read with no energies (six `None`): the upload was kept under a name the reader that is for it does not claim. It is kept under its own name now | `web/blueprints/watch.py`; `test_the_viewers_follow_a_real_run_e2e.py::test_an_uploaded_trajectory_is_read_as_the_file_it_is` |
| K27610-6 | unseen, named | that the sidebar hands its words to whoever waits on the root, and that the picker shows them when the root cannot be resolved: the server fails the root only on an error a test cannot make without breaking the page. Also untested: the walk above stopping at the folder's own tree, and the walk below skipping dot-folders | the dead-code release judges them with K2679-5 |
| K27610-7 | **later, a design question** | a folder stamped a run that lies in no stage's `run-<n>/` still opens the newest file of each role (`parse/dirs/rundir.py` `openable_in`), as `model/parse.md` § 5.2 states, while a folder no calculation claims now opens nothing; both are said to show "only what is in it" | the owning chapter first |
| K27610-8 | later, low | the fresh-load bind of the Results picker depends on the sidebar having answered before `load` (older than this release); two XYZ readers, `Structure.from_xyz` for building and `xyz` for results, differ in symbol case, extended XYZ and the frame-set check; `xyz.NAMED_FOR_ANOTHER` restates two readers' names (a drift is refused loudly by `detect`); Task setup's Save does not ask the holds question (the hand-over asked it); the command line's transport describe asks the folder question twice; `runs.folder_answer` calls `_opens` in both arms; the spectra viewer's coming-back load resets the picked mode; a person's `git init` inside a calculation leaves a hole in the outer history | each where it belongs |

### What 2.67.10 removed (the list the user saw before the commit, generated from the code against 2.67.9, 2026-11-07)

**From the product**:

- `molbuilder/web/static/lib/projects/state.js`: `projects.onProjectsRootResolved` (the old way to wait for the projects root, silent when the root could not be resolved) and its subscriber set — one caller, the notebook tab, now on `whenProjectsRoot`; K27610-3, the review's finding
- `molbuilder/web/static/jupyternb/index.js`: the *opening* state's 8-s budget and its `rootWaiter` — the budget existed because the old door said nothing on a failure; K27610-3
- `molbuilder/describe.py`: `refuse_described`'s `shown` argument — its three callers (`molbuilder describe`, the hand-over, `transport.describe`) each composed it by hand; the function names folders itself (`projects.address`); K27610-2, the review's finding
- `molbuilder/jobset/commands.py` `bundle_flag`: its own composition of a folder's address, now `projects.address`
- `molbuilder/transport/compose.py`: `refuse_inside_citation` — called by `transport/describe.py` and `transport/reuse.py` (its `ComposeError` import with it); covered whole by the user's rule of 2026-11-07, K27610-2
- `molbuilder/parse/engines/pyscf.py`: `_can_parse_xyz` and `_MAX_PLAUSIBLE_ATOMS` (the claim on every XYZ), and the imports only they used; K27610-1
- `molbuilder/web/static/lib/trajectory/core.js`, `lib/spectra/core.js`: `_onPageShow` and its `pageshow` listener — the second listener for coming back; K27610-4
- `molbuilder/web/static/lib/results/file-picker.js` `_onPageShow`: its scan on a restore (the bind after a fresh load kept); K27610-4
- `molbuilder/web/static/lib/tree-picker.js` `pickPath`: its silent `null` before the root is known; `lib/projects/dialogs.js` `chooseSavePath`: its throw *"the project tree is not available on this page yet"* and the import only it used; K27610-3
- `molbuilder/web/blueprints/watch.py`: an upload's temporary name made from its stem and suffix (`safe_stem`, `safe_suffix`); K27610-5

**Tests** (each change seen failing on the old code where it guards a change):

- `tests/parse/test_registry.py`: `test_a_pyscf_decks_geometry_pair_is_one_parsers` — its two rows are rows of `test_each_xyz_is_read_by_the_reader_that_is_for_it`
- `tests/test_projects_public_surface_js.py`: `TestOnProjectsRootResolved` (`test_subscriber_fires_when_setProjectsRoot_lands`, `test_subscriber_registered_after_resolution_fires_immediately`, `test_unsubscribe_works`) and the `onProjectsRootResolved` row of the duplicate-callback test — the door is gone; `TestWhenProjectsRoot` (two rows and one test) asks the one answer
- `tests/test_inspector_pageshow_refresh_e2e.py` renamed `tests/test_coming_back_to_the_tab_e2e.py`; `test_a_page_restored_from_the_cache_asks_the_server_at_once` renamed `…_asks_the_server_once`, counting each ask
- `tests/test_checkpoint_sensor_js.py::test_setup_does_not_ask_for_a_name_when_a_name_would_not_help`: no longer stands on a calculation inside another's folder (refused now) but on a folder where a person ran `git init` by hand
- `tests/test_molview_files_door_js.py`: the stand-in for `state.js` no module asks for any more

**New**: `tests/test_checkpoint_states.py::test_a_folder_inside_a_calculation_is_refused_a_history_of_its_own` (two rows); `tests/test_the_viewers_follow_a_real_run_e2e.py::test_an_uploaded_trajectory_is_read_as_the_file_it_is`; rows in `test_describe.py` (two), `tests/data/presenters.toml` (three); the order held in `test_molbuilder_e2e.py`'s lattice test; the address asked of the hand-over's refusal in `test_build_e2e.py`.

## Found by 2.67.9's page tests (2026-11-07), by the release that owns each

*(three fixers on separate copies; a fresh reader over the merged work and two fixers on what it found; then, at the user's word, two discipline rounds of six readers each over every touched file read whole, each finding read again in the code, and eleven fixers on what they found; the fast batch and every browser and real-run test the release touches run headless, in a browser the tests start)*

| id | where | finding | door |
|---|---|---|---|
| K2679-1 | **closed 2.67.9** | a spectrum put away while its 3D mode box was still being built left that box alive for the page's life, a drawing window and three listeners each time (the CO2 run's spectrum opened and put away five times: two windows live). The rule was nowhere written for the spectrum page | the box released as it arrives once the panel is gone (`lib/spectra/core.js` `_showMode`); the rule in `web/spectra.md` § 4; its test holds the box's stylesheet request until another file is picked (`test_inspector_registry_e2e.py::test_a_spectrum_put_away_while_its_mode_box_is_built_leaves_nothing`) |
| K2679-2 | **closed 2.67.10** (K27610-1) | the PySCF reader claims every `.xyz` (`parse/engines/pyscf.py` `can_parse`): SIESTA's own `H2.xyz` is answered as a PySCF file and folded into the run's `.molwatch.log` by the trajectory viewer (`lib/inspectors/trajectory.js` `absorbs`), so one of the two ways `web/results.md` § 6.0 gives to save a run's structure cannot be reached from the menu (`test_results_export_e2e.py::test_the_engines_own_geometry_file_saves_the_box_the_run_had` red, not hidden); a calculation's `<label>.source.xyz` is answered the same way; in a plain folder the route names a copied `.out` openable while no viewer claims a file with no role | design read first, then the fix |
| K2679-3 | **decided by the user 2026-11-07: refuse; closed 2.67.10** (K27610-2) | `describe` accepted a calculation inside another calculation's folder (`refuse_described` asks only the folder itself), where status already refuses any folder inside a calculation (`job-system.md`) and a history is one per calculation (`checkpointing.md` L1). The user: *"calculations can continue from another, not within another"*. With it: `checkpoint init` on a folder inside a calculation, against L1 as written, measured then refused; the history panel's nested-history test (`test_checkpoint_sensor_js.py::test_setup_does_not_ask_for_a_name_when_a_name_would_not_help`, standing on a nested calculation today) moved to the one road left, git run by hand in a calculation's folder, or retired with what goes unseen named | `describe.refuse_described`, `checkpoint.Repo.init`, the contract first |
| K2679-4 | **closed 2.67.10** (K27610-3) | the folder picker (`lib/tree-picker.js` `pickPath`) returns nothing, silently, when clicked before the page knows the projects root, where its sibling `chooseSavePath` says so (*"said, never a silent nothing"*): the slab panel's *From a bulk run…* clicked early does nothing (`test_molbuilder_e2e.py::test_a_measured_lattices_notes_are_rows_each_at_its_own_severity` may fail on it) | the picker waits for the root or says why, its rule written first |
| K2679-5 | **F9, dead code** | what no folder answer or route reaches, to judge with K2677-4 and K2678-5: `lib/projects/molview-doors.js`'s single-file branch (the codec always writes the pair; `web/projects.md` and `blueprints/build.py` describe it); `modify/structure/panel-deps.js`'s `!body.ok` under HTTP 200; the trajectory heading's *Trajectory* fallback (a role comes only with the engine's words and the noun); the error card when a viewer's partial cannot be fetched; `api.js`'s "server returned non-JSON" words; an unknown severity drawn as info; the file picker's fallbacks (no time, input not a list, ties by name, an empty filter, one stage's master taking another's file, a run without its master, a master in another folder); the badge's clock fallbacks; `chemistry.js` `analyze`'s AbortError branch and its `null` call; the projects sidebar's second root resolution; `serve(config)`, `page_free(timeout)` | that release |
| K2679-6 | **closed 2.67.9** | the tests' own discipline, found by two rounds of readers and fixed: node started by hand in nine page tests beside `run_node` (copies with timeouts of 15 to 60 s); the projects root pointed by hand where `isolated_projects_root` does it; `support.live_server.serve` holding a second copy of the suite's root door, which left a dead projects root behind (a red browser test); a test that could not fail (a secret no route hands the page); a test that looked before any answer had landed; a SIESTA run made twice in one pass; a Results binding, an editor edit, a fill-and-Tab, Task setup's opening and its editor read each written several times; a frame set typed as text; a topic no person can make; tests asking one bit twice; docstrings and documents naming tests that no longer exist or sections that never did | one door each: `support.sidebar`, `support.task_setup_page`, `support.real_runs.stopped`, `_node_esm.served`/`run_node`, `conftest.point_the_one_door_at`, `support.leftovers` |
| K2679-7 | **closed 2.67.9** | two tests already failing on 2.67.8, defects of the tests: `test_molbuilder_e2e`'s "2 atoms" where the page says *Deleted 1 atom: 2 left.* on purpose (U1-4); the run panel test (code words with their backticks, Reload bound to the sidebar, a wait on a selected row) | read against the contract |
| K2679-8 | left, said in each file | kept because no action makes them: `visibilitychange` dispatched (a page behind another stays visible in headless Chromium, measured; `pageshow` is now a real back-and-forward restore in the full Chromium, headless); the name panel (PubChem refused from here); the two transport-record rows typed (a finished junction is the 800-s module). Measured on the way, small: a restore asks the server twice (`results/file-picker.js`, `trajectory/core.js` each listen to both events) where `web/results.md` § 2.2 says one scan | the double scan: closed 2.67.10 (K27610-4) |
| K2679-9 | **server half closed 2.67.11** (K27611-1…4); the page half 2.67.12  | the tests' discipline outside this release's files, and what the readers found there: five files patching `Capabilities.file_picker_roots`; the projects root pointed at a real run by hand at ~31 sites; the `app_at`/`flask_server` fixture copied in ~40 files; module trees no test uses; five e2e modules and five fast files describing calculations by hand, and two makers of a described calculation (`road.describe_calculation`, `real_runs.described`); private road helpers used as doors and helpers imported from other test files; templates written by hand in two files; two files driving a browser in the fast tier; the hand list of pages in `test_pages_no_js_errors.py`; per-file markers repeating conftest's; time waits in Task setup prep, the stopped run, the preview and the sidebar tests; the bench summary's rewritten timers; two JS-error collectors; the bound `WAIT_MS` beyond the doors; the displacement sweep made twice; typed network failures; the GPU sampler patched; the `.XV` time set by hand; the digests moved away in the junction file | F9-6 |
| K2679-10 | unseen, named | what no run here reaches: the badge's *Queued* (a scheduler: the Sol pass) and *Record unreadable* (a record broken by hand); *Not run* (no session run leaves one); a live run's frame still being computed counted as a step (the faked test retired; no run is stopped inside a frame) | the Sol pass |

### What 2.67.9 removed (the list the user saw before the commit, generated from the code against 2.67.8, 2026-11-07)

**From the product**: nothing. One fix, K2679-1.

**Test functions gone, 157; new, 63** (4189 → 4095 in all). Each gone one is merged into a test or a row that goes red on the same break, moved onto a real run or real reply under another name, or retired with what then goes unseen named in its fixer's report (kept in the release's scratch notes) and, where it matters, in K2679-5 and K2679-10:

- `spectra/test_blueprint.py`: `test_no_cdn_fallback_in_spectrum_page`
- `test_app_notifications_e2e.py`: `test_persist_error_event_surfaces_one_error_row`, `test_persist_ok_clears_the_error_row`, `test_persist_ok_leaves_other_notifications_alone`, `test_repeat_persist_error_dedups_into_one_row_with_counter`
- `test_build_e2e.py`: `test_send_writes_and_STAYS`, `test_send_writes_the_handover_into_the_selected_folder`, `test_setShared_alone_does_NOT_load_structure`
- `test_checkpoint_sensor_js.py`: `test_a_calculation_is_named_by_its_description_not_by_its_depth`, `test_a_stored_open_preference_greets_a_run_dir_with_the_view`, `test_state_git_failure_is_structured_error_not_crash`, `test_the_panel_and_its_importer_actually_parse`
- `test_chemistry_module_js.py`: `test_an_abort_is_a_supersede_not_a_failure`, `test_an_error_page_says_so_instead_of_unexpected_token`, `test_no_structure_is_answered_without_calling_the_server`, `test_the_forms_and_the_kind_travel_with_the_structure`, `test_the_servers_own_message_is_what_the_person_gets`
- `test_inspector_pageshow_refresh_e2e.py`: `test_coming_back_to_the_tab_asks_the_server_at_once`
- `test_inspector_registry_e2e.py`: `test_a_viewer_whose_file_cannot_load_says_why_and_lets_go`, `test_opening_a_viewer_again_and_again_leaves_what_once_left`, `test_switching_files_in_the_dropdown_again_and_again_leaves_what_once_left`
- `test_layout_fits_e2e.py`: `test_content_fits_its_box`, `test_no_page_scrolls_sideways`
- `test_molview_files_door_js.py`: `test_a_single_file_downloads_bare`, `test_failures_and_nonsense_come_back_as_envelopes_never_throws`
- `test_molview_reserved_label_js.py`: `test_no_atom_carries_the_fact_twice`
- `test_projects_api_envelope_js.py`: `test_apiDelete_forwards_signal_into_fetch`, `test_apiList_forwards_signal`, `test_apiMkdir_forwards_signal`, `test_apiRead_forwards_maxBytes`, `test_apiRead_forwards_signal`, `test_apiRename_forwards_signal`, `test_apiRename_posts_with_correct_body`, `test_apiUpload_forwards_signal_into_fetch`, `test_apiWrite_forwards_signal_into_fetch`, `test_no_signal_is_safe`, `test_normalises_to_envelope_on_success`, `test_apiList_returns_server_body_verbatim`, `test_apiReadRange_returns_server_body_verbatim`, `test_apiRead_returns_server_body_verbatim`, `test_apiWrite_post_with_overwrite_flag`, `test_apiDelete_non_json_returns_envelope`, `test_apiList_passes_cache_no_store`, `test_signal_is_preserved_alongside_cache_default`
- `test_projects_public_surface_js.py`: `test_isCancelError_does_NOT_match_disposed`, `test_isCancelError_matches_aborted_code`, `test_isCancelError_matches_aborted_flag`, `test_isCancelError_matches_abort_error_name`, `test_isCancelError_matches_a_cancelled_envelope`, `test_isCancelError_rejects_null_and_other_errors`, `test_after_wiring_delegates_to_impl`, `test_impl_failure_envelope_returned_verbatim`, `test_mkdir_failure_does_not_refresh`, `test_duplicate_onCommit_throws`, `test_publishCommit_fires_setShared_when_pick_changes`, `test_publishCommit_fires_subscribers`, `test_only_fires_once_per_resolution`, `test_delegates_to_apiRead`, `test_abort_signal_threads_into_fetch`, `test_default_offset_and_max_bytes_omitted_from_url`, `test_explicit_offset_and_max_bytes_in_url`, `test_negative_offset_for_tail_read`, `test_network_drop_returns_uniform_envelope`, `test_createProject_refreshes_root`, `test_deleteEntry_refreshes_parent_dir`, `test_mkdir_refreshes_parent`, `test_rename_409_does_not_refresh`, `test_rename_refreshes_parent_dir`, `test_upload_refreshes_target_dir`, `test_onChange_throws_on_duplicate`, `test_onProjectsRootResolved_throws_on_duplicate`
- `test_results_blueprint.py`: `test_body_carries_required_inspector_ids`, `test_cache_control_is_private_and_revalidated`, `test_content_type_is_html`, `test_endpoint_returns_html_200`, `test_partial_does_not_carry_page_chrome`, `test_cache_control_is_private_and_revalidated`, `test_content_type_is_html`, `test_endpoint_returns_html_200`, `test_partial_does_not_carry_page_chrome`
- `test_results_file_picker_js.py`: `test_a_label_holding_a_stage_shaped_run_is_still_one_entry`, `test_a_master_in_another_directory_absorbs_nothing`, `test_a_rung_does_not_absorb_another_rungs_trajectory`, `test_the_master_log_absorbs_its_working_files`, `test_the_stage_token_is_stripped_before_matching`, `test_without_the_master_the_parts_still_list`, `test_empty_input_returns_empty`, `test_filters_by_pick_result`, `test_handles_missing_mtime`, `test_invalid_input_returns_empty`, `test_only_what_the_menu_shows_is_counted`, `test_sorts_newest_first`, `test_takes_the_whole_path_the_server_sends`, `test_tiebreak_by_name`, `test_category_is_built_from_the_servers_answer`, `test_without_mtime`
- `test_siesta_optimization_run_e2e.py`: `test_a_step_a_run_is_inside_or_stopped_inside_is_no_step`
- `test_siesta_run_in_the_viewer_e2e.py`: `test_opening_the_run_again_and_again_leaves_what_once_left`
- `test_slab_orthogonal_choice_js.py`: `test_an_unknown_surface_is_left_alone`, `test_a_surface_with_one_shape_gets_it_set_and_locked`, `test_it_reads_the_server_and_does_not_hardcode_the_planes`, `test_the_one_surface_with_a_choice_keeps_it`
- `test_spectra_phase_indicator_js.py`: `test_a_phase_the_run_did_not_ask_for_shows_no_dot`, `test_dot_classes_match_phase_status`, `test_dot_title_carries_human_readable_status`, `test_first_call_reveals_the_indicator`, `test_missing_element_silent_no_op`, `test_missing_phase_field_defaults_to_empty`, `test_relaxation_dot_is_data_driven`
- `test_spectra_transition_js.py`: `test_a_knob_write_schedules_the_save`, `test_a_non_knob_write_schedules_nothing`
- `test_structure_dna_js.py`: `test_default_backend_is_auto`, `test_explicit_backend_forwarded`, `test_unknown_backend_rejected_client_side`, `test_backend_failure_surfaces_message`, `test_canvas_cancel_passes_through_as_cancelled`, `test_empty_sequence_rejected`, `test_illegal_bases_rejected_client_side`, `test_invalid_form_rejected_client_side`, `test_z_form_with_gcgc_passes_guard`, `test_z_form_with_non_gc_sequence_rejected_client_side`, `test_z_form_with_rdkit_backend_skips_guard`
- `test_structure_peptide_js.py`: `test_backend_failure_surfaces_message`, `test_canvas_cancel_passes_through_as_cancelled`, `test_empty_sequence_rejected_without_fetch`, `test_illegal_codes_rejected_client_side`
- `test_structure_save_js.py`: `test_the_page_records_the_path_the_SERVER_answered`
- `test_structure_smiles_js.py`: `test_canvas_accept_returns_ok`, `test_server_returns_ok_false_envelope`, `test_empty_smiles_rejected_without_fetch`, `test_non_string_rejected`, `test_whitespace_only_rejected`, `test_generate_picks_up_late_structurePage`
- `test_task_setup_notify_js.py`: `test_a_channels_secret_never_reaches_the_description`, `test_the_hours_box_is_ignored_until_its_row_is_ticked`, `test_the_number_is_a_number_not_a_human_spelling`
- `test_trajectory_clocks_js.py`: `test_a_duration_is_never_offered_as_the_timestamp`, `test_an_epoch_is_never_offered_as_the_duration`, `test_a_run_with_no_clocks_at_all_says_nothing`, `test_each_clock_lands_in_its_own_slot`, `test_mtime_is_the_fallback_and_only_for_the_epoch`, `test_the_total_by_the_run`
- `test_trajectory_rebuild_js.py`: `test_a_rebuild_replaces_the_movie_in_a_read_only_viewer`
- `test_trajectory_settle_post_load_js.py`: `test_every_ending_the_parser_concludes_with_has_a_row`, `test_the_badge_reads_the_run_or_the_files_own_ending`, `test_the_viewer_follows_the_run`
- `test_trajectory_transition_js.py`: `test_an_answer_is_written_under_its_own_files_name_or_not_at_all`, `test_a_poll_whose_write_ended_the_run_settles_it`, `test_a_poll_with_new_content_keeps_the_runs_state_it_held`, `test_a_quiet_poll_on_a_live_run_keeps_watching`, `test_a_quiet_poll_that_finds_the_run_over_settles_it`, `test_loaded_stops_polling`
- `test_validation_findings_js.py`: `test_group_with_no_rendered_card_lands_in_residual`, `test_no_form_scope_sends_everything_to_residual`, `test_rendered_count_equals_received_count`, `test_server_order_is_preserved`, `test_unrecognised_severity_renders_as_info_never_dropped`

**Files gone whole**: `test_spectra_phase_indicator_js.py`, `test_trajectory_clocks_js.py`, `test_trajectory_rebuild_js.py`, `test_trajectory_settle_post_load_js.py` — their cases moved onto real runs (`test_the_viewers_follow_a_real_run_e2e.py`) or retired.

**Test helpers gone, 67** (copies of a door, or the hand-made inputs they built):

- `conftest.py`: `_point_the_one_door_at`
- `spectra/test_blueprint.py`: `_allowed_root`
- `support/leftovers.py`: `grew`
- `support/real_runs.py`: `_described`, `_left_to_end`, `_on_the_road`, `_road_at`, `_stopped_by_its_line`, `_taken`, `_typed`
- `test_app_notifications_e2e.py`: `_boot`
- `test_build_e2e.py`: `_register_tmp_as_picker_root`
- `test_checkpoint_sensor_js.py`: `_allowed_root`, `client`, `_js`, `_seed_with_archive`
- `test_chemistry_card_e2e.py`: `_ready_for_a_pick`, `_register_tmp_as_picker_root`
- `test_chemistry_module_js.py`: `_item`
- `test_inspector_pageshow_refresh_e2e.py`: `_register_tmp_as_picker_root`
- `test_inspector_registry_e2e.py`: `_open_results`
- `test_molbuilder_e2e.py`: `_register_tmp_as_picker_root`
- `test_page_ids_unique.py`: `_Ids`
- `test_preview_modal_edit_save_e2e.py`: `_cm_line_count`, `_register_tmp_as_picker_root`
- `test_projects_api_envelope_js.py`: `_run_node`
- `test_projects_public_surface_js.py`: `_run_node`
- `test_results_blueprint.py`: `web`
- `test_results_export_e2e.py`: `pyscf_run`, `siesta_run`
- `test_results_file_picker_e2e.py`: `_register_tmp_as_picker_root`
- `test_results_file_picker_js.py`: `_run_node`
- `test_sidebar_e2e.py`: `_register_tmp_as_picker_root`
- `test_siesta_optimization_run_e2e.py`: `_deck`, `_out`
- `test_siesta_run_in_the_viewer_e2e.py`: `tree`
- `test_siesta_stopped_run_e2e.py`: `after_the_stop`, `flask_server`, `_jobset`, `_set_item`
- `test_spectra_phase_indicator_js.py`: `_extract_fn_source`, `_make_phase_indicator_stub`, `_run_with_results`
- `test_structure_dna_js.py`: `_run_node`
- `test_structure_peptide_js.py`: `_run_node`
- `test_structure_smiles_js.py`: `_run_node`
- `test_task_setup_cell_types_e2e.py`: `_described`, `_open`, `out_of_set_dir`, `_request_beside`, `_type`
- `test_task_setup_notify_js.py`: `_controls`, `_run`, `_tick`
- `test_task_setup_one_folder_e2e.py`: `_open`
- `test_task_setup_prep_e2e.py`: `_came_back_to_the_files_value`, `_open`, `_page_free`
- `test_task_setup_reacts_to_a_restore_e2e.py`: `_editor_text`
- `test_trajectory_clocks_js.py`: `_run_cycle_clock`
- `test_trajectory_rebuild_js.py`: `_rebuild_model`
- `test_trajectory_settle_post_load_js.py`: `_badge`, `_run`, `_settle`
- `test_trajectory_transition_js.py`: `_applied`, `_parsed`, `_poll_once`

**Shared test doors renamed or narrowed** (no old name kept): `conftest._point_the_one_door_at` → `point_the_one_door_at`; `support.real_runs` `_road_at`, `_on_the_road`, `_typed`, `_taken`, `_described`, `_left_to_end`, `_stopped_by_its_line` → the same names without the underscore; `support.sidebar._WAIT_MS` → `WAIT_MS`; `support.live_server.serve` no longer makes a projects tree of its own and lost its `config` argument; `support.task_setup_page.page_free(timeout=)` and `wait_for_file(timeout=)` gone; `support.results_export._WAIT_MS` gone; `support.leftovers.grew` and `SPY.quietly` gone (one door, `what_grew`); `support.road.a_new_folder` takes the tree.

**Rows**: `presenters.toml` headings 11 → 10 (the two fallback rows no folder answer reaches gone; a run's file copied into a person's own folder, listed under no heading, added; the PySCF vibration's progress-log row now the SIESTA vibration's).

**New real runs, each measured**: the stopped H2 run made once for two files (19-24 s, `real_h2_stopped`); an H2 displacement sweep (33 s, `real_h2_sweep`); followed relaunches of the layered H2 (50-68 s), the stopped H2 (16 s) and PySCF water (7 s); a PySCF H2 under another label (7 s).

## Found by 2.67.8's real runs (2026-11-07), by the release that owns each

*(three fixers on separate copies; a fresh reader over the merged work, its findings read again before they were taken; the fast batch and every real-run test the release touches run on the real engines)*

| id | where | finding | door |
|---|---|---|---|
| K2678-1 | **closed 2.67.8** | 2.67.7 relaxed a real Au-BDT-Au junction, where `testing.md` § 6 says the smallest system that shows the mechanism, a hydrogen chain for a junction (the user: "a Au-BDT-Au is too much"). Mine: the passage was not read | a hydrogen chain with a two-element bridge, 19 s |
| K2678-2 | **closed 2.67.8** | two product defects real results showed, which a typed result had hidden: a mode's shares keyed by species label, labelled water reading "44% H1, 44% H2" where `web/spectra.md` § 4.2 says each element's; `read_spectra_data` giving `warning: ""` where the file's JSON states null, against the round trip of `engines/vibration.md` § 6.10.5 | `resolve_element`; a `text?` kind |
| K2678-3 | **closed 2.67.8** | three real-run tests had not been run since 2026-11-03 and held words the releases since changed: an engine's file taking a role from `ENGINE_FILES`; `jobset export --to` outside the projects root; a plan's line said as what it would do. Not defects of the product | each read against its contract |
| K2678-4 | **closed 2.67.8** | the reader's: a scratch-size test comparing numbers from runs of another day (now the estimate against what TranSIESTA printed in this pass); a run made twice in a pass; checks that depended on this machine's speed; four wrong examples of one refusal; a deck "as release 0.0.0 wrote it" edited by hand (now the writer's own, at another version); dead builders in `support/junction.py`; a typed parse result for a torn force block (now the real output, cut) | each fixed |
| K2678-5 | **F9, dead code** | what nothing reaches now, to judge with K2677-4: `spectra/results.py`'s ascending check, complex refusal, "supplied together"; the file version's second home (`results.py:1376` beside `sidecars/spectra.py:286`); `infrared_said` (`core.js`, `blueprints/spectra.py`), reached only by older files; the modes-without-geometry branch of the load door. `__post_init__`'s other refusals of a file no writer writes (`engines/vibration.md` § 6.7), their tests retired by § 3b | that release's list |
| K2678-6 | left, by the rule | readers of older files with no test now (`testing.md` § 6: "no work goes into them"): `sidecars/molstruct.py` `RETIRED_ENVELOPE_KEYS` and `READABLE_VERSIONS` below today's, `pyscf/script_input.py` `_by_what_it_holds`, `jobset/prepared_with.py` `_OLDER_FRAME` and `_words_now`, `task.py`'s `task@1` branch, `siesta/engine_input.py` `_as_siesta_reads` | `jobset migrate`'s own day |
| K2678-7 | left: a run cannot reach it | one util.csv row covering 300 s needs over five minutes of steady load (the longest gap measured 24 s): analytic numbers. A file with `[N/A]`: the writer never writes one | none |
| K2678-8 | **F10, first end-to-end pass** | browser tests, none edited here, and the end-to-end files this release's changes to `support/real_runs.py` reach but do not name, run in that pass | that pass |
| K2678-9 | left: B4-2, the user's decision | `test_a_sent_run_the_scheduler_forgot.py` keeps its hand-written launch record, this machine having no scheduler, its reason stated in it; the field test waits for the Sol pass (K2675-5) | that pass |

### What 2.67.8 removed (the list the user saw before the commit, generated from the code against 2.67.7, 2026-11-07)

**From the product**: nothing. Two fixes, K2678-2.

**Goes unseen from here** (each named so it can be judged; nothing a run reaches):
a spectrum file whose atom lists do not partition being refused; one row per free atom; a misspelled key refused rather than served as "not computed"; a complex vector refused; elements without positions refused; the second version gate, a file with no version key; a malformed file or bad field answering 400 rather than 500; a directory as a path; sizes no run reaches; a result with modes and no geometry; the no-spectrum sentence when `infrared_said`; the way-back wording when the history holds no state before a first launch; `[N/A]` in a util.csv; an older pair, record or deck being read as today's; a listed-only row dropped from the restart-file list; a block in a layout no writer writes; decks leaving scratch and the bias contour to the engine counting as the same input, an older deck's case.

**Test functions gone, 75**, each merged into a test or row that goes red on the same break, or moved onto a real run under another name, unless said above:

- `spectra/test_atom_index_contract.py`: `test_eigenvector_rows_must_match_n_free`, `test_free_frozen_overlap_raises`, `test_out_of_range_free_index_raises`, `test_uncovered_index_raises`, `test_valid_partition_with_frozen_atoms_constructs`
- `spectra/test_blueprint.py`: `test_a_frame_sets_temperature_not_stated_is_none`, `test_a_result_without_a_geometry_still_loads`, `test_load_says_which_atoms_each_mode_belongs_to`, `test_path_input`, `test_the_on_disk_format_does_not_grow_the_computed_field`, `test_what_a_frame_sets_temperature_means_is_served_per_rule`, `test_path_load_endpoint_works`, `test_load_still_reads_a_spectra_file_inside_the_roots`
- `spectra/test_datafile.py`: `test_the_data_file_names_the_json_it_was_written_with`
- `spectra/test_parsers_json.py`: `test_complex_dtype_on_input_array_rejected`, `test_full_feature_set_round_trip`, `test_atomic_replace_no_torn_file_on_failure`, `test_inf_in_array_rejected`, `test_nan_in_scalar_field_rejected`, `test_no_bom_in_output`, `test_empty_modes_in_progress_state`, `test_schema_error_carries_expected_and_actual_attrs`, `test_specific_errors_inherit_base`, `test_directory_path_raises_oserror_branch`, `test_geometry_omitted_back_compat`, `test_geometry_partial_rejected`, `test_geometry_round_trips`, `test_negative_frequency_with_has_imag`, `test_high_precision_scf_energy_round_trip`, `test_very_small_and_very_large_floats`, `test_electronic_structure_null_round_trips`, `test_ir_intensity_null_round_trips`, `test_raman_activity_null_round_trips`, `test_utf8_special_chars_round_trip`, `test_extra_engine_metadata_keys_round_trip`, `test_intermediate_phase_state_round_trips`, `test_round_trip_minimal_results`, `test_the_file_the_writer_wrote_is_read_by_its_parser`, `test_missing_file_message_names_path`, `test_missing_file_raises_not_found_error`
- `spectra/test_types.py`: `test_imaginary_mode`, `test_cross_mode_eigenvector_shape_consistency`, `test_free_fixed_partition_complete`, `test_free_fixed_partition_disjoint`, `test_zero_imaginary_part_complex_still_rejected`, `test_from_dict_rejects_missing_or_wrong_schema_version`, `test_the_reader_refuses_a_key_it_does_not_know_by_name`
- `test_checkpoint_states.py`: `test_s1_holds_over_a_real_staged_tree`
- `test_molstruct_json.py`: `test_a_pair_the_release_before_wrote_reads_whole`, `test_any_version_but_the_current_one_is_refused`
- `test_prep_calculation.py`: `test_a_record_from_before_it_held_folders_still_compares_its_points`
- `test_pyscf_stages.py`: `test_what_says_nothing_of_the_computation_is_no_difference`
- `test_script_emit.py`: `test_a_current_block_applies`, `test_apply_atom_metadata_normalises_indices`, `test_the_brace_walk_still_accepts_both_json_layouts`
- `test_structure_authority_roundtrip.py`: `test_a_box_edit_marks_the_contract_outdated`, `test_a_python_edit_marks_the_contract_outdated`, `test_append_takes_the_contract_from_the_structure_APPENDED_TO`, `test_every_atom_edit_carries_the_recorded_contract`, `test_the_contract_carries_its_OUTDATED_flag_through_an_edit`
- `test_structure_replace_door.py`: `test_a_strip_is_something_a_caller_says`, `test_info_travels_through_a_change_nobody_named_it_in`
- `test_transport_au_bdt_au_validation.py`: `test_fixture_pair_opens_through_the_read_door`, `test_fixture_regenerated_from_textbook_chemistry`
- `test_transport_decks_e2e.py`: `test_the_canonical_junction_prepares_clean`
- `test_transport_on_a_real_junction_e2e.py`: `test_what_a_deck_leaves_to_the_engine_is_read_as_siesta_reads_it`
- `test_util_csv_is_time_weighted.py`: `test_a_hole_in_one_column_does_not_shift_the_others_onto_it`, `test_a_single_row_still_answers`, `test_gpu_sm_is_weighted_the_same_way`, `test_the_window_is_not_truncated_by_a_short_column`
- `test_vibration_e2e.py`: `test_a_hartree_fock_deck_names_no_functional`, `test_a_setting_that_enters_nothing_is_said_to_at_prep`
- `test_warmfiles.py`: `test_pyscf_inventory_is_the_declared_vocabulary`, `test_siesta_inventory_is_the_declared_vocabulary`
- `test_where_an_item_binds.py`: `test_a_calculation_wide_value_is_taken_by_the_rungs_that_read_it`

**Files gone whole**: `tests/spectra/test_parsers_json.py`, `test_types.py`, `test_atom_index_contract.py`, `test_datafile.py`, for `tests/test_spectrum_results_e2e.py` on real results; `tests/test_spectra_no_spectrum_sentence_js.py`, for its `_e2e` on real files; `tests/test_transport_au_bdt_au_validation.py` with `tests/data/au_bdt_au.xyz`, `au_bdt_au.molstruct.json` and `au_bdt_au.README.md`, a committed geometry only its own tests read, its bond-length provenance with it.

**Test helpers gone**: `tests/spectra/_helpers.py` `_make_results`, `_make_mode`, `_make_es`; `_make_minimal_results`; `support.road._road_launched` and `_road_steps_written`, which wrote a launch record and steps by hand; `support.junction.au_bdt_au`, `build_junction`, `regions`, `frozen`; `test_warmfiles`'s two inventories. **Rows**: `prep_protocol` 90 → 83, `held_atoms` 10 → 9 (moved or merged), `a_stage_that_has_run` 6, new, `the_deck` 63 → 68, `the_transport_deck` 4 → 7.

**New real runs, each measured**: a hydrogen chain with a bridge 19 s; H2O on PySCF, oxygen held, 13 s; H2 two stages one job 20 s; water straight 13 s; a CO2 copy under a 20 s wall 18 s. Moved from one module to the session: three PySCF water vibrations.

## Found by 2.67.7's prep-side tests (2026-11-06), by the release that owns each

*(three fixers on separate copies, each break made and read; two fresh readers, over the merged work and over the real-run tests, their findings read again before they were taken. The user, on a record read off a deck that never ran: "real test is a real test, stop bending the rules"; "you have always fucking siesta related env why the fuck this cannot be tests")*

| id | where | finding | door |
|---|---|---|---|
| K2677-1 | **closed 2.67.7** | the reader's: a typed record on every transport pair; a typed deck and a named program handed to the run-script writer; two typed descriptions on an empty folder, so Save never ran its check with a template; two transport refusals called unreachable; a dead door, dead table keys, docstrings claiming what nothing checked; five and three wrong examples of one rule | a real relaxation, cited (`support.real_runs`); the road; one example each |
| K2677-2 | **closed 2.67.7** | nothing fast noticed a transport calculation losing what it takes from the relaxation it cites: level of theory, k-point mesh or the charge refusal could go and every non-engine test stayed green | `test_transport_decks_e2e.py`, on real relaxations, each break red |
| K2677-10 | **closed 2.67.7** | the second reader's, on the real-run tests: a mesh relaxed at 3 x 3 x 1 could not show the transport axis laid to one point; a lead's anchor checked by its word and not its number, so one atom off went unseen; an edit stood in by setting its flag; a voltage outside its range tested nowhere | a mesh of 3 x 3 x 2; the number, by the reader; a real translation; a test at Task setup's Save, where a person changes the list |
| K2677-3 | left: not observed to matter | `runwrap.py`, the exit trap's `_mb_signalled`: on TERM and HUP it changes nothing, the handler's earlier commands leaving `$?` 0; it could matter only on the PIPE trap, which no test reaches | the next review |
| K2677-4 | **F9, dead code** | checks that repeat another or that nothing can reach: `submit._plan_shelves`' two `--only` filters; the bench-grid route's type check; `task.py`'s two calculation shape checks; `describe.py`'s empty ladder; `migrate.py`'s second "does not read"; `rung.py`'s witness, reached by no test | that release's list |
| K2677-5 | left: low | a 0.5 Å displacement prints two warnings for one number (`validation/metadata.py`, `validation/spectra.py`), against one fact, one finding; `--psml-lib P/structure` is refused as in `structure`, naming the last folder only | the next review |
| K2677-6 | **F10, first end-to-end pass** | changed and unrun: the prep end-to-end fixtures in `test_task_setup_prep_e2e`, `test_task_setup_reacts_to_a_restore_e2e`, `test_task_setup_one_folder_e2e` and `test_bench_summary_inspector_e2e` (a stated 10 Å cell in place of a vacuum-sized box, `psml_lib` in the template, a run card on the binding folder and the PySCF folder, `named_environment_path("sol")`). The two transport files below did run | that pass |
| K2677-7 | **closed 2.67.7** | `test_transport_keywords_exist_in_the_binary.py`'s verdict is the machine's, and it needs a citation: it is an end-to-end test now, on the real relaxation and the real binary | the user's rule |
| K2677-8 | left: the subject is the hand-made thing | what could not move: `test_describe`'s two planted ladders, the command taking only a shipped one's name; `test_a_record_from_before_it_held_folders`, a typed older record no command writes; `test_submission_gates_the_cold_start`, a prepared deck edited by hand, the edit its subject; `test_two_blocks_of_a_deck_that_disagree`; `test_the_kind_gate_refuses…`, a config of no engine; `test_transport_wizard._au_device` | none |
| K2677-9 | **closed 2.67.8** | typed records and citations of a run that never was remain where this release did not reach: `test_structure_authority_roundtrip`, `test_structure_replace_door`, `test_where_an_item_binds._transport`, `test_transport_task`'s `_CITE` before the door reads it | that release, a real run each |

### What 2.67.7 removed (the list the user saw before the commit, generated from the code against 2.67.6, 2026-11-06)

**From the product** (the only product removal; mine, under the rule that a fallback kept for tests goes): `runwrap.render_run_wrapper`'s `else` branch reading the record's `conda_envs`, with `_rec_envs`, `_where = "this machine"`, `_there = "here"`. Its one product caller, prep, always hands a record; both tests that took the branch now hand one. The command help and `job-system.md` § 5.1 no longer say `describe` checks stage names and overrides a person types: it takes neither.

**Goes unseen from here** (each named so it can be judged): the words of the PySCF benchmark refusal leaving out `max_scf_iter`; a PySCF stage's own run card's threads (the calculation's card is held, by the same map); a description citing a file that is absent, which the command refuses first; `describe`'s own empty-ladder refusal, no caller handing it one; `Task`'s two shape checks, the kind table refusing the same input; a job whose deck names another stage, no prep writing one; a module re-created at an old path; four more spellings of one rule beside `--reuses` (`--psml-lib`, `--structure`, `--vacuum`, `--stage-strategy`) and two beside a citation (`--structure`, `--stage-strategy`), one example standing for each rule; prep's saying the seam on the gold slabs of the four-row table, the table now asked of the card's door and prep's saying it asked once, on the real gold junction.

**Test functions gone, 84**, each merged into a test or row that goes red on the same break, unless said above:

- `parse/test_contract.py`: `test_a_deck_stating_nothing_is_none`, `test_no_deck_is_none`, `test_the_record_speaks_the_catalogues_names`
- `test_bench_grid_card.py`: `test_a_bench_that_is_not_an_object_is_refused`, `test_the_folder_must_be_inside_a_picker_root`
- `test_deck_runner.py`: `test_check_passes_a_deck_the_runner_itself_wrote`, `test_check_reads_the_file_and_not_the_string_it_was_handed`, `test_the_runner_reports_what_it_emitted_so_check_can_close_the_loop`
- `test_describe.py`: `test_a_name_that_cannot_normalise_is_a_clean_refusal`, `test_a_siesta_describe_whose_species_has_no_pseudopotential_is_refused`, `test_a_value_past_its_hard_limit_is_refused`, `test_an_empty_ladder_is_refused`, `test_an_override_the_schema_does_not_know_is_refused_by_name`, `test_describing_writes_no_deck`, `test_init_refuses_a_folder_that_already_holds_a_calculation`, `test_it_writes_exactly_the_template_the_description_and_its_request`, `test_the_structure_is_a_reference_and_a_witness_never_a_copy`
- `test_emitted_program_namespace.py`: `test_no_emitted_function_rebinds_a_name_the_deck_imported`
- `test_every_form_field_reaches_the_deck.py`: `test_every_field_the_form_offers_changes_the_generated_deck`
- `test_jobset.py`: `test_a_number_resolves_to_the_seq_and_never_to_the_row`, `test_cli_errors_without_jobset_json`, `test_cli_prep_is_described_only`, `test_each_stage_lays_out_in_the_folder_its_deck_token_names`, `test_job_dir_name`, `test_plan_prints_the_seq_not_the_row`, `test_prep_says_reused_only_of_an_attempt_an_earlier_prep_opened`, `test_render_plan_surfaces_per_job_ranks_and_cores`, `test_render_plan_sweep_says_independent`, `test_resolver_refuses_a_number_a_sweep_cannot_have`, `test_stage_refs_carries_the_jobs_name_not_the_tokens`, `test_stage_refs_reads_the_seq_off_the_deck_not_the_row`, `test_the_declaration_is_now_the_only_rendering_of_the_rule`, `test_the_grammar_is_unambiguous_even_for_a_stage_named_3`, `test_the_machine_probe_is_molbuilders_not_the_benchmarks`, `test_the_progress_channel_ends_up_in_the_run_and_nowhere_else`, `test_validate_catches_empty_and_bad_kind`, `test_validate_refuses_a_condition_the_job_could_never_meet`, `test_warm_and_traits_survive_job_set_at_1`
- `test_prep_bench_fold.py`: `test_a_bad_enum_value_and_a_non_bool_are_refused_with_the_choices`, `test_a_charged_decks_promised_script_ships_with_it`, `test_a_declared_gpu_point_runs_the_declared_total_ranks`, `test_a_declared_pin_reaches_the_trial_deck`, `test_a_fine_tuned_vocabulary_copy_wins_and_is_named`, `test_a_multi_point_value_entry_is_a_value_axis`, `test_a_name_outside_the_execution_category_is_refused`, `test_a_pyscf_description_is_refused_by_name_at_the_bench_seam`, `test_a_pyscf_runs_threads_reach_the_launch_shape`, `test_cli_prep_bench_requires_a_stage`, `test_the_benchmarks_memory_reaches_every_shelf_header`, `test_the_benchmarks_prep_states_gpu_sharing`, `test_the_verb_renders_the_trial_decks_it_promises`, `test_trials_nest_inside_the_stage_they_measure`
- `test_prep_calculation.py`: `test_a_structure_that_changed_since_describing_is_refused`
- `test_routing_domains.py`: `test_a_record_the_reader_fully_understands_says_nothing_extra`, `test_get_routing_is_empty_on_a_workstation`, `test_get_routing_is_empty_without_a_record`, `test_get_routing_reads_the_calculations_record`
- `test_trajectory_log_stage_targets.py`: `test_per_stage_molwatch_log_carries_its_geometry_step_cap`, `test_per_stage_molwatch_log_carries_stage_target`
- `test_transport_au_bdt_au_validation.py`: `test_each_electrode_block_declares_its_REGION_SIZE`, `test_prep_refuses_a_spacing_that_leaves_the_bias_window_under_two_points`, `test_the_canonical_junction_validates_clean`, `test_the_device_deck_states_its_identity_and_method`
- `test_transport_wizard.py`: `test_the_seam_verdict_is_measured_and_said`, `test_without_a_starting_geometry_the_frozen_DECLARATION_still_holds`
- `test_validation_across_stages.py`: `test_the_sequence_warnings_reach_the_prep_surface`
- `test_value_axes.py`: `test_a_duplicated_allocation_point_refuses_at_prep`, `test_a_gpu_count_past_the_record_is_kept_with_its_fit`, `test_a_trial_asks_the_device_count_it_states`, `test_declared_gpu_count_is_exact`, `test_gpu_count_on_a_cpu_bench_is_refused_not_ignored`, `test_preflight_flags_a_typod_choice_per_point`, `test_the_worked_example_matrix_enumerates_as_the_doc_states`
- `test_vibration_form_honesty.py`: `test_every_shown_parameter_changes_the_deck_or_is_openly_pending`
- `test_vibration_render_gate.py`: `test_amplitude_advisory_reaches_the_person`, `test_an_ecp_deck_compiles_and_carries_one_ecp_kwarg`, `test_an_hf_raman_deck_never_mentions_the_dft_name`, `test_frozen_atoms_stay_frozen_through_the_relaxation`, `test_one_fact_one_finding_on_a_vibration_deck`, `test_soscf_reaches_the_relax_site`, `test_the_frozen_regime_is_said_out_loud`, `test_the_kind_door_runs_the_same_gate_body`, `test_the_level_of_theory_has_one_spelling_per_deck`
- `test_what_a_kind_offers.py`: `test_a_description_refuses_what_its_kind_does_not_offer`

Two files gone whole: `test_vibration_form_honesty.py`, its check now the form test's, comparing what the engine runs, where three time stamps had made it pass by accident (K2674-7); `test_trajectory_log_stage_targets.py`, three rows.

**Moved, not gone**: to `test_transport_decks_e2e.py`, on a real relaxation: the four deck tests of `test_transport_au_bdt_au_validation.py`, two of `test_species_order_is_one_rule.py`, two of `test_transport_task.py`, one each of `test_warm_file_inventory.py` and `test_electronic_state.py`; five rows of `the_deck.toml`, four into `the_transport_deck.toml` and one into the electrode test. `test_transport_keywords_exist_in_the_binary.py` is `…_e2e.py`. `test_the_decision_log_prints_channel_names_readably` to `test_pipeline_log.py`.

**Test helpers gone**, for the road's doors: `conftest._PSML_FIXTURE`, for real files; `conftest.prep_the_group_of`, no caller; `test_engine_offset_reaches_every_deck._prep`, built on describe's internals, with its four users on `support.road.described_and_prepared`; `test_transport_task._cited_transport`, its typed record with it, and `support.junction._junction_struct`, its one builder; `_declare` and `_declare_bench`, for `support.road.declare_bench`; `_first_rung_overrides`, for Task setup's Save; typed junctions and two typed waters, for real relaxations and `support.road.WATER`; `test_warm_file_inventory._wrapper`, for prep's run script. **Table keys gone**, read and used by no row: `answers` (a row's), `named_record`, `ledger_lacks`. **Rows**: `the_deck` 52 → 63, `prep_protocol` 78 → 90, `gpu_contract` 44 → 48, `launch_values` 46 → 47, `the_catalogue` 2 → 3, `the_transport_deck` 4, new. **Sentences withdrawn**: the test names and claims above, in `execution/scheduler.md`, `generator.md`, `project-layout.md`, `job-system.md`, `engines/vibration.md`, `engines/transport.md`, `web/spectra.md`.

## Found by 2.67.6's shared test doors (2026-11-06), by the release that owns each

*(a fresh reader over the runner and every door moved, each finding read again or measured before it was taken)*

| id | where | finding | door |
|---|---|---|---|
| K2676-1 | **closed 2.67.6** | `status` called a run's `done` true of whatever the checkout held: `[e2e] done | 543/543` from 2026-11-03 read beside today's | a digest of every file git keeps, recorded at a run's start and asked at `status` |
| K2676-2 | **closed 2.67.6** | `run e2e tests/x.py` left `[e2e] done | 12/12`, a part reading as the batch | its file is `custom`, naming what was asked |
| K2676-3 | **closed 2.67.6** | an expected failure counted as a skip, no reason kept for either, and a file skipping itself whole recorded nowhere: a missing package read as a whole run | each counted apart with its reason; `xfail_strict` |
| K2676-4 | **closed 2.67.6** | the reader's, on this release's first form: a commit reading as other code, a person's git settings making every fresh run read so, `status` needing pytest, `code_now` tested only through a stand-in | content, not the commit; git's settings left out; `code_now` in `testrun.py`, stdlib alone; a test on a real repository |
| K2676-5 | **closed 2.67.7** | B4-6, B4-11, B4-12 and K2675-2 move there: thirteen files build a calculation from describe's internals, plant a machine record, type a pseudopotential, and they change together with what they use. `in_a_folder_of_its_own` is asked for by four files, and two more keep a sandbox of their own that writes a record | through `describe_calculation` and the probe's record |
| K2676-6 | left | `testrun.py failed` defaults to the e2e batch, and after a part reads the older whole; `status --fails` is right. `.test-progress.jsonl`, a plugin's default no run of `testrun` uses, is not ignored | the next review |
| K2676-7 | left: by design | `molbuilder/data/jupyter_server_config.py`'s `c` is Jupyter's own global, the one pyflakes finding left | none needed |
| K2676-8 | left | a run reading `done, ON CODE NOT KNOWN` until each batch runs again: progress files from before this release record no code | the next run of each |

### What 2.67.6 removed (the list the user saw before the commit, generated from the code against 2.67.5, 2026-11-06)

No product name removed; `ledger.read`, `monitor_log.NOTIFY_STUB`, `NOTIFY_STUB_TAG`, `NOTIFY_STUB_LINE` and `parse.instruments.monitor.events_said` added. No test case removed. From the tests: `_road.set_template_value`, for `support.road.set_by_the_person`; `_isolated` in `support/junction.py`, `test_prep_calculation.py` and `test_pipeline_log.py`, for `conftest.in_a_folder_of_its_own`; the layers' positions in `test_transport_compose.py`, moved to `support/junction.py`; private ledger and plan reads in thirteen files, for `ledger.read` and `JobSet.of`; unused imports and variables across tests, tools and scripts.

## Found by 2.67.5's applying the rule (2026-11-06), by the release that owns each

*(the user's rule, `process/testing.md` § 3b, applied to the tests that typed a tool's output; a fresh reader over what was added, its findings confirmed before they were taken; the old and new `parse_psml_header` compared on all 150 real files on this machine: identical answers)*

| id | where | finding | door |
|---|---|---|---|
| K2675-1 | **closed 2.67.5** | the reader's: a docstring saying v0.5 sulfur's `d` is wholly zeroed, when one of its two projectors is real; `dirac` left with no check; a CUDA "variable over the driver" check that could not fail with a driver answering nothing, and a field test the session's cached recipes would have failed; a short pin's row lost; a field test checking a manager other than the product's; a job's field tests erroring where they should skip; a peak test passing for a copy; `z-pseudo` able to read as Z; stale words in `monitor.py`, `pseudos.py`, `installation.md`, `pseudopotentials.md`, `testing.md` | each fixed |
| K2675-2 | **closed 2.67.7** | `conftest.write_pseudos` still writes a small typed PSML for tests that only prepare a calculation, for many elements | the real files where they exist; the rest decided there |
| K2675-3 | **closed 2.67.8** | `test_util_csv_is_time_weighted.py` types the monitor's own file | a file the monitor wrote |
| K2675-4 | left: no real file shows it | the check that a wholly zeroed channel outside the valence is not named has no test: none of the 144 real files has such a channel | a file that does |
| K2675-5 | **the user, on Sol** | `tests/field/test_the_monitor_reads_its_job.py` is written and not run: it needs a scheduled job, `srun --mem=8G -c 4`. Until then nothing checks the cgroup v1 path, the one Sol uses | that run |
| K2675-6 | left | a manager reporting no details (mamba, micromamba) is tested only on a machine that has one; the LDA and PBEsol rows of `_LIBXC_MAP` have no test, all 150 files being PBE | a machine that has them; a real file |
| K2675-7 | left | the reader's suspected ones, not measured: the first monitor field test where a cgroup is named and unreadable; `AllowedRAMSpace` other than 100; `PsmlInfo.atomic_number` and `valence_config` have no product reader (with K2674-9); `_lib` and `_write_folder` overlap | the next review |

### What 2.67.5 removed (the list the user saw before the commit, generated from the code against 2.67.4, 2026-11-06)

**From `molbuilder/pseudos.py`**, used only by hand-typed test files, none of the 150 real files holding any of them:

| removed | whose decision |
|---|---|
| the `<header>` element in place of `<pseudo-atom-spec>` | the user, once checked (0 of 150) |
| the `element=` attribute; reading attributes off the root when there is no spec | mine, same rule |
| `z-pseudo` standing in for the atomic number | mine, the reader's finding |
| functional ids on `<libxc-info>`, `<exchange>`, `<correlation>`; the `xc`, `xc-functional`, `xc-id` attributes | mine, same rule |
| the `relativistic=` attribute; the spellings `fr`, `fully`, `fully-relativistic`, `spin-orbit`, `sr`, `scalar-relativistic`, `non`, `non-relativistic`, `nr` | mine, same rule |
| `generator=` on `<provenance>`; a `<generator>` element | mine, same rule |
| valence written as text | mine, same rule |
| `<pseudo-dojo rmax/mesh-cutoff/recommended-cutoff>` | mine, same rule |

**What goes unseen**: the cgroup v1 readings Sol uses, until K2675-5 runs; a manager reporting no details; the LDA and PBEsol table rows; the hint when a deck is loaded where an output belongs; the refusals of mismatched orbital arrays and an unknown phase word. No science rule is lost: the p-channel block, bismuth's warning and the cutoff rules stand on the real files.

**Tests gone (51 cases):**

- `test_monitor_allocation_basis.py` (9): `test_cpu_time_is_the_jobs_own`, `test_job_cgroup_strips_the_step_and_task`, `test_memory_is_the_jobs_own_not_the_nodes`, `test_the_basis_line_names_every_rung`, `test_the_limit_comes_from_the_JOB_cgroup_not_the_task`, `test_the_peak_is_not_a_copy_of_current`, `test_the_sentinel_reads_as_no_limit_stated`, `test_the_v1_layout_is_read_per_controller`, `test_the_v2_layout_is_read_as_one_path`
- `test_pseudos.py` (13): TestParsePsmlHeader: `test_basic_round_trip`, `test_fully_relativistic_normalised`, `test_malformed_xml_returns_parse_warning`, `test_pbesol_libxc`, `test_unknown_libxc_id_falls_back_to_unknown`; TestDeadProjectorC5: `test_absent_channel_is_not_flagged`, `test_dead_projector_is_error_status`, `test_dead_projector_maps_to_error_severity`, `test_good_pseudo_has_no_null_channels`, `test_null_channel_detected`; TestGeneratorVersionC4: `test_patch_difference_does_not_warn`; TestErrorStatusesSharedBySurfaces: `test_cli_screens_a_folder_by_the_files_it_holds` (1 row), `test_the_settings_gate_says_each_label_once` (1 row)
- `test_pseudo_requirements_layer.py` (3): `test_a_zeroed_valence_channel_the_semilocal_block_carries_is_named`, `test_an_UNOCCUPIED_zeroed_channel_is_not_named`, `test_the_two_tiers_do_not_overlap`
- `test_envs_doctor.py` (7 rows of `test_audit_detects_which_tree_is_installed`, the same test on real pip and git with 7 new)
- `test_envs_one_answer_about_an_env.py` (6): `test_a_manager_that_reports_no_details_hides_nothing`, `test_an_env_outside_envs_dirs_is_VISIBLE`, `test_an_env_the_snapshot_never_saw_is_still_looked_for`, `test_the_gate_and_the_state_machine_now_agree`, `test_the_probe_sees_an_out_of_tree_env_on_EITHER_kind_of_manager`, `test_the_registry_answers_with_prefixes_and_drops_the_installation_root`
- `test_system_load.py` (6): `test_a_field_the_card_lacks_reads_None_not_a_crash`, `test_a_parsed_row_carries_the_fields_the_widget_draws`, `test_a_timeout_is_reported_and_the_server_keeps_serving`, `test_a_tool_that_fails_is_loud_on_the_wire`, `test_the_query_always_carries_the_hard_timeout`, `test_the_reason_is_read_off_whichever_stream_carried_it` (back with two rows)
- `spectra/test_types.py` (5): `test_equilibrium_mos_reject_complex`, `test_es_mo_array_rejects_complex`, `test_invalid_phase_status_rejected`, `test_mo_arrays_must_match_size`, `test_mode_eigenvector_rejects_complex`
- `test_machine_identity.py` (1): `test_no_tool_and_no_device_read_the_same`, renamed for what it now does
- `watch/test_registry.py` (1, file gone): `test_unknown_format_fdf_suggests_out_file`

**Helpers and typed data gone**: `_make_psml`, `_psml_with_projectors`, `_psml`, `_make_pdojo_psml`, `_S_DEAD_P`, `_S_GOOD`, `_S_V04`, `_S_V05`, `_BI_V05`, `_S_HINTS`, `_S_SHELLS`, `_fake_dist`, `_registry`, `sol`, `SOL_V1`, `SOL_TASK`, `SOL_JOB`, `V2_LINE`, `_ROW`, `_smi_result`, class `TestPhaseStatus`.

**Sentences withdrawn**: `ops/installation.md`'s *each of its three tiers*; `science/pseudopotentials.md`'s *`z-pseudo` … parsed for sanity*; `monitor.py`'s comment that its two paths are named for a test to point at a fixture; document mentions of scratch folders no longer on disk (`claude-s28-onestart`, `claude-review-2-45-0`, `claude-transport-walk`, `claude-vib-ui`, `claude-w33`), each kept as what it showed.

## Found by 2.67.4's judging of the tests (2026-11-06), by the release that owns each

*(every test the audit named judged by breaking the code and reading what else failed: 581 rows in five readers' tables; a fresh reader over what was added. The user's rule of 2026-11-06 came out of this work: [`process/testing.md`](?doc=process/testing.md) § 3b, *A test validates what is correct*.)*

| id | where | finding | door |
|---|---|---|---|
| K2674-1 | **closed 2.67.5** | the user's rule reaches tests this release left: three typing `nvidia-smi`'s output and one typed system-load row, the cgroup-v1 tests, the typed pip file and `_fake_dist`, the typed PSML files and `_make_psml`, `test_envs_one_answer_about_an_env.py`'s typed manager answers, and 8 wrong-input checks in `spectra/test_types.py` and `watch/test_registry.py` that each caught a break nothing else in its set caught | what the tool prints read where it runs; correct input through the real writer; one test for *reported, no number invented* |
| K2674-2 | **closed 2.67.5** | `pseudos.py:371`'s `<header>` branch may read a shape no real file has, and only a typed test reaches it | read the real H, C and O files first; a branch no real file uses goes |
| K2674-3 | **closed 2.67.8** | a spectrum file of the wrong version gets the refusal `engines/vibration.md` § 6.7 promises, and removing `_validate_schema_version` turns no test red | one test on what the writer writes, with its version moved |
| K2674-4 | **closed 2.67.8** | spectra rules with no test: the canonical eigenvector read from the display form; `free_atom_idxs` not ascending; a complex canonical vector not refused; a probe's SCF-minus energy and amplitude | a run's result, correct input |
| K2674-5 | **closed 2.67.8** | a mutation letting T through the server's RNA alphabet left 248 tests green: only the page's own check is tested | one row on the server's refusal |
| K2674-6 | **closed 2.67.8** | older-release tests added after the audit, not judged by it: `test_a_pair_the_release_before_wrote_reads_whole` (2.65.8), `_as_2_65_7_wrote_it` in `test_pyscf_stages` (2.65.9), `test_a_record_from_before_it_held_folders...` (2.67.2) | `testing.md`'s rule on older files |
| K2674-7 | **closed 2.67.7** | `test_vibration_form_honesty` passes by accident; `test_sidebar_layout_opt_in_is_server_side`'s page half passes against a 404 | correct input, or retire |
| K2674-8 | **F9** | `_parse_lscpu`, `_parse_scontrol_node` and `sniff5` have no test; `-DCMAKE_IGNORE_PATH=/usr/local` and the § 6.3 Fortran fence (`_FORTRAN_ABI_CEILING`) neither | field tests for the first two; the rest decided by what the build shows |
| K2674-9 | **a release of its own, list shown first** | code no product path reads: `SIESTA_RESTART_GROUP`, `PYSCF_RESTART_GROUP` and their `RestartGroup` (though `run-identity.md` § 4 presents them as each engine's declaration); `annotations_fdf`'s register and report functions and `unregistered_channels`; `onCanvasChange` in `structure/page.js`; `category=` of `template.select`; `Frame.__post_init__`'s list coercion; `jobset/ask.py:333`; `/api/modify/meta`'s `lattice_error`; the response keys `extra`, `source_format`, `lattice` | delete, with what `run-identity.md` § 4 says of them |
| K2674-10 | **F10** | `parse.md` § 3 against `AmbiguousFormatError`; `web-api.md` § 3 says all 99 routes, there are 110 | the chapter |
| K2674-11 | **left** | two `ParseWarning` classes (`frame.py:35`, `parse/types.py:33`); silent bail-outs at `results/viewer.js:697` and `:816` and `if (!els.esPanel) return` in `spectra/core.js`; a *not bound* error naming `_bind()` | the next review of each module |
| K2674-12 | **closed 2.67.4** | a fresh reader over what this release added: the create bar's first click could beat the button's wiring; a test's name claimed a citation draws the rungs, drawn at load; a frame-set row named a channel and a record it never checked; a presence check, a stale store no product path reaches and a test helper's wrong rows added; a negative row asking other words than its positive; a second node launcher; a hand-written machine config beside the fixture; an fdf scan beside `read_fdf`; `engines/template.md` and `web/form-schema.md` saying six panels where every form draws five; `installation.md` § 6 not stating the MPI pins a test cites it for | each fixed or gone; the words on six panels measured on every engine and kind |
| K2674-13 | left | the reader's low ones: `test_setShared_answers_through_projects` asserts only `ok`; `_GCC_VERSION`'s minor asserted twice; `row()` defined twice in `test_sidebar_e2e.py`; `[cell.atoms_outside]`'s negative row has no positive in its table, and it did go red with the frame fix reverted in a scratch copy | the next review of each |

### What 2.67.4 removed (the list the user saw before the commit, generated from the code against 2.67.3, 2026-11-06)

Nothing in the product calls a test. Each removal rests on `process/testing.md` § 3b, default removal, and was judged by a break; those marked *older files* on the user's rule of 2026-10-16. The user read this list and answered *go ahead* to it, to the rule above and to keeping `test_delete_does_not_mutate_input`, my recommendation, which is why that test is not below. A fresh reader then found three of the tests added here should not be (a page presence check, a stale store no product path reaches, a test helper's wrong rows): those are gone, K2674-12, and the words below say so.

**`parse/test_coords.py`** (1). Moved: the same check now asks the file reader by the name a person's file has, and still fails if the `.XV` reader is not registered.
`test_coords_parsers_registered`

**`parse/test_parse_warning.py`** (2). File gone. It restated a small record's fields. A renamed field breaks the code using it at once. Nothing lost.
`test_parsewarning_allows_none_for_line_no_and_snippet`, `test_parsewarning_constructible_with_documented_fields`

**`parse/test_registry.py`** (1). Moved to a test over every registered format: a file nothing can read is told every format.
`test_detect_unknown_extension_raises`

**`parse/test_sidecars.py`** (5). Used a structure file from an older release, or restated a shape. Replaced by a test on a file the writer really wrote. Lost: nothing a person meets.
`test_detect_routes_to_molstruct_parser`, `test_molstruct_parser_claims_suffix`, `test_parse_molstruct_returns_sidecarresult`, `test_sidecar_parsers_registered`, `test_sidecar_result_is_frozen`

**`parse/test_tbtrans_out.py`** (1). **Real loss.** Empty files named from TBtrans's source, no run made them. If the lookup of a spin-polarized point's files breaks, that point is refused with wrong words. No wrong number. The real-run tests in F10 are where this belongs.
`test_a_points_averaged_files_are_found_by_tbtrans_own_names` (1 row)

**`spectra/test_parsers_json.py`** (4). Same call twice over. Replaced by reading a file the writer wrote. Lost: nothing.
TestDumpSpectraJson: `test_pathlike_accepted`, `test_round_trip`; TestExceptionHierarchy: `test_not_found_also_inherits_filenotfound`; TestParseSpectraJsonHappyPath: `test_accepts_pathlike_input`

**`spectra/test_types.py`** (27). **Small real loss.** Restated fields, and hand-built wrong results. Moved first: infrared read back, each phase's state, a stored display form. Lost: if someone deletes a check on an impossible result (a HOMO outside its window, orbital arrays of the wrong shape, modes disagreeing on window size), or the line making `a == b` on two results an error, or reading an older result file, no test says so. All writers state these values.
TestEqualityIsLoud: `test_error_message_points_at_compare_api`, `test_mode_data_eq_raises`, `test_mode_electronic_structure_eq_raises`, `test_spectra_results_eq_raises`; TestModeData: `test_ir_intensity_reserved_for_future`, `test_raman_activity_none_when_not_computed`, `test_round_trip_with_es`, `test_round_trip_without_es`; TestModeElectronicStructure: `test_json_serialisable`, `test_round_trip`; TestPhaseStatus: `test_independent_phase_state`, `test_running_status_round_trips`; TestPostInitValidation: `test_cross_mode_es_window_consistency`, `test_eigenvector_coerced_to_float`, `test_eigenvector_shape_validated`, `test_empty_modes_allowed_for_in_progress_runs`, `test_equilibrium_mos_coerced_to_float`, `test_homo_idx_validated`, `test_homo_index_in_range`, `test_mo_arrays_coerced_to_float`, `test_mo_arrays_must_be_1d`; TestSpectraResults: `test_engine_metadata_passes_through`, `test_from_dict_defaults_for_missing_optionals`, `test_modes_preserve_order_through_round_trip`, `test_round_trip_complete`, `test_round_trip_in_progress`, `test_schema_version_pinned`

**`test_annotations_fdf.py`** (2). File gone. **Close call:** a structure made in a notebook carrying an unknown annotation word would stop prep with a raw error instead of being carried. Loud, not silent. The second test covered an extension point nothing uses.
`test_no_strategy_is_carried_not_emitted`, `test_register_and_emit_channel`

**`test_atom_annotations.py`** (2). One restated a default; the other became one test, a channel of an unknown kind refused. Lost: nothing.
`test_annotations_default_empty_and_backcompat`, `test_channel_kind_validation_and_defaults`

**`test_auth_config.py`** (6). Defaults restated, and the code asks for them the same way either way. The two cookie tests became one with two rows, both still run. Lost: nothing.
TestGitHubSpecific: `test_allowed_organizations_defaults_empty`; TestGoogleSpecific: `test_hosted_domain_defaults_empty`; TestMixedConfig: `test_google_plus_cas_round_trip`; TestORCIDSpecific: `test_minimal_entry_valid`; TestSetupSessionSecurity: `test_session_cookie_flags_set_regardless_of_trust_proxy`, `test_session_cookie_flags_set_when_auth_on`

**`test_backends.py`** (4). Asked whether an answer is a bool. Lost: nothing a reader sees.
`test_amber_is_available_returns_bool`, `test_available_backends_returns_dict_of_bools`, `test_rdkit_is_available_returns_bool`, `test_threedna_is_available_returns_bool`

**`test_bench_result.py`** (2). Duplicates. Lost: nothing.
`test_choice_survives_a_json_round_trip_for_the_offer`, `test_from_dict_rejects_major_mismatch`

**`test_build_format_sniff.py`** (1). Duplicate. Lost: nothing.
`test_positive_count_is_xyz`

**`test_checkpoint_cli.py`** (1). Pinned the list of verbs. A seventh verb would land without this test objecting; adding one takes a deliberate edit.
`test_the_group_offers_exactly_the_contracts_verbs`

**`test_cli.py`** (10). One walk now asks every command of the real tree for help, instead of a typed list of eight. Lost: nothing.
`test_electrode_still_builds_one_slab_per_flag`, `test_subcommand_help_exits_cleanly` (8 rows), `test_top_level_help_exits_cleanly`

**`test_cli_tls.py`** (6). Private helper, or duplicates of the real `serve` tests. Lost: a flag hidden from `--help` would go unseen.
`test_check_tls_readable_noop_on_falsy_inputs`, `test_resolve_cli_flag_alone_without_pair_falls_back`, `test_resolve_json_supplies_both_when_no_flags`, `test_resolve_json_with_only_cert_falls_back_to_http`, `test_resolve_no_json_no_flags_returns_none_pair`, `test_serve_help_advertises_cert_and_key`

**`test_config_dir_has_one_home.py`** (1). Read back what it wrote. Lost: nothing.
TestOperationalStateFollowsTheVariablesOnly: `test_projects_stays`

**`test_convergence_targets_nested.py`** (6). Two tests that never ran in this suite (the engine package isn't in this environment). The file's other test stays, under a name that fits: `test_convergence_targets_card.py`.
`test_a_target_reads_as_a_person_reads_it` (4 rows), `test_emitter_writes_flat_lines_for_flat_input`, `test_emitter_writes_nested_lines_for_nested_input`

**`test_deck_helpers.py`** (18). Tested a test helper with no callers left, wrong rows typed into a deck prep never writes. A bare energy refused stays, the one example. One moved to `test_siesta_vibration_deck.py`. Lost: only how a failing test words itself.
`test_a_block_comes_back_tokenised_row_by_row`, `test_a_block_read_from_a_file_beside_the_deck_reads_as_one_deck`, `test_a_keyword_is_found_by_any_fdf_spelling` (5 rows), `test_a_keyword_the_deck_does_not_set_reads_as_None`, `test_a_malformed_row_names_the_block_and_the_problem` (5 rows), `test_an_absent_block_is_None_and_its_rows_are_empty`, `test_assert_fdf_says_which_of_the_two_failures_it_is`, `test_block_rows_key_on_the_first_token`, `test_repeated_leading_tokens_collapse_and_that_is_documented`, `test_spacing_is_not_part_of_the_value`

**`test_diagnostics.py`** (14). Typed a fake conda answer, or duplicates. Replaced by one test with a real config file in the sandbox. Lost: nothing, as long as `test_envs_one_answer_about_an_env.py` stands, and that file types manager answers too (K2674-1).
`test_detect_assembles_capabilities`, `test_detect_conda_failure_yields_empty_envs`, `test_detect_filters_out_conda_root_installation`, `test_detect_no_conda_gives_empty_envs`, `test_env_available`, `test_env_for_category_honours_config_override`, `test_env_for_category_returns_default`, `test_env_for_category_unknown_returns_none`, `test_env_for_tool_known`, `test_env_for_tool_playwright_not_routed`, `test_env_for_tool_unknown_returns_none`, `test_every_tool_routes_to_a_known_category`, `test_routed_env_none_when_env_missing`, `test_routed_env_returns_name_when_routed_and_available`

**`test_docs_tab.py`** (3). Duplicates; the image test was renamed for what it does. Lost: the Docs tab dropped from the menu would be seen at first use.
`test_documents_page_renders`, `test_documents_tab_registered_in_tab_order`, `test_img_serves_a_real_docs_image`

**`test_envs_abi.py`** (11). Restated the rule table, and 7 rows of version strings that real code paths already exercise. Kept the one that only it held: text with no number is no version.
`test_every_rule_carries_a_remedy`, `test_every_rule_has_a_unique_stable_code`, `test_every_rule_remediates_on_the_env_side`, `test_parse_version_handles_every_shape_this_module_receives` (8 rows)

**`test_envs_doctor.py`** (3). One could not fail, one was a duplicate, one pinned a report's length. Lost: a long block in a report.
`test_report_all_marks_every_env_missing_on_empty_machine`, `test_report_all_skip_verify_dispatches_nothing`, `test_verify_output_trimmed_to_2k`

**`test_envs_nfs_shmem_fix.py`** (2). File gone, both moved: the hook's fix is now checked by sourcing the real hook in a shell, psutil into the recipes test.
`test_activate_hook_sets_ompi_tmpdir_base`, `test_host_env_includes_psutil`

**`test_envs_recipes.py`** (6). Could not fail (the class refuses such a recipe), or restated. Two merged into one with two rows. Lost: nothing.
`test_every_recipe_has_required_fields`, `test_no_two_recipes_serve_one_category`, `test_recipe_by_name_finds_every_recipe`, `test_recipe_by_name_returns_none_for_unknown`, `test_recipe_names_are_unique`, `test_registry_is_non_empty`

**`test_envs_siesta_gpu_recipe.py`** (22). Restated recipe lines the build itself would show. Merged and kept, per your answer: SIESTA's features and MPI compilers, one test. The hooks are now executed. Lost: a lost `cmake>=3.30` floor, a flag naming `/usr/lib`, or a second channel, each needing a deliberate edit and each showing at the build.
`test_build_spec_activate_hook_publishes_paths`, `test_build_spec_artifact_subdir`, `test_build_spec_components_in_order`, `test_build_spec_cuda_required`, `test_build_spec_deactivate_hook_mirrors_activate`, `test_cuda_toolkit_in_conda_packages`, `test_elpa_pins_env_compilers_and_cuda_path`, `test_kernel_headers_is_not_double_pinned`, `test_libxc_present`, `test_no_elsi_component`, `test_no_system_paths_in_cmake_flags`, `test_pins_cmake_geq_3_30`, `test_pins_the_project_python`, `test_recipe_carries_build_spec`, `test_recipe_category`, `test_recipe_description_mentions_gpu_and_source`, `test_recipe_is_registered`, `test_recipe_uses_conda_forge_only`, `test_siesta_component`, `test_siesta_pins_cmake_prefix_path_to_env_and_elpa`, `test_siesta_pins_mpi_compilers_to_env`, `test_the_lua_build_step_is_the_one_carrying_the_search_paths`

**`test_envs_validate.py`** (6). Asked that a table's hint isn't empty. Lost: only a deliberate empty string.
`test_every_probe_row_carries_a_runtime_hint` (6 rows)

**`test_fcc_cell_shapes.py`** (3). Duplicate of two tests beside it. Lost: nothing.
`test_every_plane_has_an_entry` (3 rows)

**`test_fcc_lattice_table.py`** (1). A field no page reads. Lost: nothing.
`test_meta_endpoint_lattice_error_is_none_on_happy_path`

**`test_frame.py`** (1). File gone. No product code passes a list there. Lost: nothing.
`test_frame_post_init_coerces_list_inputs`

**`test_frame_sets.py`** (1). Not a removal: the same row reworded, because its record no longer pretends to be a vibration's.
`test_frame_set` (1 row)

**`test_geometry_pin.py`** (2). **Your rule, older files.** Two rows built v12 and v13 files. How a pre-v13 pair opens could change, unseen.
`test_the_pin_through_the_read_door` (2 rows)

**`test_gpu_answerers.py`** (6). Duplicates of the tests that stay. Lost: nothing.
`test_each_machine_answered_setting_really_carries_the_flag` (5 rows), `test_the_gpu_flag_is_the_persons_choice_not_the_machines`

**`test_junction_sidecar_roundtrip.py`** (4). Duplicates, a fixture check, an older schema's file. Lost: nothing.
`test_a_key_this_version_does_not_read_is_refused_rather_than_dropped`, `test_a_sidecar_from_an_older_schema_is_refused_not_half_read`, `test_the_junction_is_shaped_the_way_the_test_claims`, `test_the_pair_is_written_at_the_current_schema_version`

**`test_launch_ask_mode.py`** (1). A branch the command line never reaches. Lost: nothing.
`test_an_empty_ask_says_so_rather_than_printing_a_header`

**`test_machine_identity.py`** (1). Moved to the field tests: it typed what `nvidia-smi -L` prints, and now reads the real tool. Run once here and it passes; this machine has one GPU, so its "each model once" half had nothing to tell apart, and it does not run in the fast batch.
`test_gpu_models_dedup_and_never_a_count`

**`test_modify.py`** (3). **Small loss.** An added atom could get a blank name, its element untouched. A metal dropped from the electrode list gets a refusal naming the list. The table could load at import.
`test_add_atom_atom_name_defaults_to_element`, `test_electrode_supported_lists_match_table`, `test_modify_module_import_does_not_eagerly_load_fcc_table`

**`test_molbuilder_runtime_js.py`** (2). One pinned method names; the other kept, in its own class. Lost: nothing.
TestSurfacePresence: `test_double_load_is_idempotent`, `test_runtime_object_exposes_documented_methods`

**`test_molstruct_json.py`** (6). **Your rule, older files.** v2, v3, v6, v7 and v8 files and the retired-keys file. An older pair's read could change, unseen. No test now reads a pair into a structure already holding a store: no product path does. Rows for a future version and no version stay.
TestARetiredKeyPassesEveryGate: `test_every_retired_key_passes_all_three_gates`; TestInfoBlock: `test_a_v8_file_loads_with_an_empty_store`; TestSchemaVersioning: `test_a_v7_file_reads_whole_under_v8`, `test_any_version_but_the_current_one_is_refused` (3 rows)

**`test_molwatch_preview.py`** (3). Text of one writer's lines. `test_molwatch_emitter.py` went red on all 7 of the same breaks. Lost: nothing.
`test_preview_helper_marks_kind_and_nulls`, `test_preview_helper_one_block_with_all_atoms`, `test_preview_helper_writes_header`

**`test_notify_listener.py`** (2). Duplicate, and a private function. Lost: nothing.
`test_deny_actually_RAISES`, `test_the_user_is_STAMPED_never_accepted`

**`test_path_framework_doors.py`** (2). A table check. Lost: a new row with `*` in its role finds no file, shown by its author's first run.
`test_every_declared_field_has_a_declared_shape`, `test_no_role_in_the_catalogue_is_a_GLOB`

**`test_pdb_workflow_integration.py`** (2). Steps 2 and 3 merged into one: labels saved on the pair come back on the atoms. Lost: nothing.
TestPdbWorkflowEndToEnd: `test_step_2_save_writes_current_schema_sidecar`, `test_step_3_a_reread_picks_up_the_sidecar`

**`test_peptide.py`** (3). Shape only; the writer is Structure's. Lost: nothing.
`test_pdb_atom_count`, `test_pyscf_listing`, `test_xyz_round_trip`

**`test_periodicity_gate.py`** (5). **Your rule, older files,** on the two `v9` rows; a test that could not fail; the `v10` rows simply stop being rows of a table of one. Lost: a v9 corner's read could change, unseen.
TestLoaderGate: `test_the_load_door_serves_the_rules_box_corner` (2 rows), `test_the_read_seam_invents_no_corner` (2 rows); TestTheCellPageCanRepairWhatItRefuses: `test_a_mirrored_cell_is_admitted_so_it_can_be_corrected`

**`test_prep_calculation.py`** (10). **Your rule, older files.** `jobset migrate` is no longer seen taking retired settings out, renaming older runs, or renaming the key in an older machine copy. These rest on your 2026-10-16 ruling.
`test_a_retired_run_setting_comes_out_where_it_lies` (5 rows), `test_a_run_an_older_molbuilder_named_is_named_once` (4 rows), `test_an_older_machine_copy_keeps_how_a_shell_enters_its_environment`

**`test_projects.py`** (3). Duplicates or shapes. Lost: nothing.
`test_ensure_structure_dir_creates_full_chain`, `test_path_resolution_composes`, `test_projects_root_with_explicit_base`

**`test_projects_api_envelope_js.py`** (11). Same wrapper asked per caller. Kept one of each kind. Lost: a new function skipping the wrapper would throw on a dropped network or get a browser-cached answer. Review holds that.
TestAbortSignal: `test_apiDelete_aborted_returns_clean_envelope`, `test_apiUpload_aborted_returns_clean_envelope`; TestNetworkFailure: `test_apiMkdir_network_drop_returns_envelope`, `test_apiUpload_network_drop_returns_envelope`, `test_apiWrite_network_drop_returns_envelope`; TestNoStoreCache: `test_apiReadRange_passes_cache_no_store`, `test_apiRead_passes_cache_no_store`, `test_apiRoots_passes_cache_no_store`, `test_apiStat_passes_cache_no_store`, `test_post_endpoints_also_get_no_store`; TestNonJsonResponse: `test_apiList_non_json_returns_envelope`

**`test_projects_public_surface_js.py`** (3). Moved first: the one thing only it saw, `projects.setShared` exported, is now a test of its own. Lost: nothing.
TestReadFile: `test_failure_passes_through`; TestReadRange: `test_server_error_envelope_passes_through`; TestSurfacePresence: `test_all_M4_methods_are_callable`

**`test_pseudos.py`** (2). Moved to two deck rows, written the way the road writes them. Lost: nothing.
TestMetalAwareScriptTemplates: `test_spectra_pyscf_fe_emits_level_shift_template`, `test_spectra_pyscf_organic_skips_level_shift_template`

**`test_pubchem.py`** (2). **Small real loss.** It faked the call underneath. `optimize=False` dropped by that three-line pass-through would go unseen; a title is only a label.
`test_build_from_name_explicit_title_overrides_default`, `test_build_from_name_threads_through_build_from_smiles`

**`test_pyscf_convergence_knobs.py`** (3). File gone. Words on the form, seen on the form; one could not fail. Lost: nothing silent.
`test_memory_is_one_item_across_both_engines`, `test_soscf_field_exists_with_ui_metadata`, `test_the_field_exists_with_ui_metadata`

**`test_pyscf_stages.py`** (6). Duplicates, or a field that no longer exists. Lost: nothing.
`test_both_engines_build_a_ladder_through_the_same_shape`, `test_default_ladder_is_three_named_stages`, `test_default_ladder_matches_siesta`, `test_no_field_of_the_config_is_a_list_of_dataclasses`, `test_the_config_has_no_stages_field`, `test_the_pyscf_form_schema_emits_no_stage_table`

**`test_record_architecture.py`** (3). File gone; `arch` is now a row of the machine-record table, through `jobset probe`. Lost: nothing.
`test_env_arch_round_trips`, `test_the_field_is_absent_rather_than_null_when_unknown`, `test_topology_arch_survives_the_round_trip`

**`test_restart_group.py`** (6). No product code reads these two group objects. Lost: nothing, and that's a finding: those objects are dead (K2674-9).
`test_a_stage_sets_restart_like_any_other_field`, `test_every_shipped_engine_declares_a_group` (2 rows), `test_pyscf_also_says_it_with_one_field`, `test_the_members_cannot_be_set_individually_any_more`, `test_the_two_engines_mean_the_same_idea_by_different_mechanisms`

**`test_results_blueprint.py`** (12). Duplicates of tab tests, and absence of old names. Lost: the kind word beside a file's name could go blank, shown at once.
TestPartialSpectraInspectorEndpoint: `test_partial_does_NOT_carry_generate_side_ids`; TestPartialTrajectoryInspectorEndpoint: `test_endpoint_does_NOT_require_auth_gate_for_the_test_client`; TestResultsInTabNav: `test_other_pages_do_not_mark_results_active`, `test_results_link_present_on_every_page` (5 rows), `test_results_page_marks_itself_active`; TestResultsRoute: `test_page_has_status_header`, `test_page_includes_projects_sidebar`, `test_route_returns_200`

**`test_results_file_picker_js.py`** (2). One misnamed duplicate; one listed method names. The real picker has its end-to-end test. Lost: nothing.
TestAPISurface: `test_module_exposes_named_helpers`; TestFilterToResultFiles: `test_filters_out_directories`

**`test_runfile_names.py`** (12). **Your rule, older files.** 12 older names. Older runs could stop reading back, unseen.
`test_an_older_name_reads_back_to_the_name_it_takes_now` (12 rows)

**`test_runtime_config.py`** (2). Shape. Lost: nothing.
`test_empty_json_object_returns_empty_dict`, `test_nested_envs_section_parses`

**`test_shared.py`** (27). Fields of a reply restated, one by one; fields nothing reads (`extra`, `source_format`, `lattice`). Lost: nothing. Three that say something stay, as plain tests.
TestOkStructureResponse: `test_carries_canonical_shape_keys`, `test_carries_the_atom_columns`, `test_issues_array_present`, `test_ok_envelope`, `test_phase_2_extras_at_top_level_and_in_extra_subdict`; TestStructureToDict: `test_atoms_matches_workspace_payload_atoms`, `test_emits_every_key_a_viewer_reads`, `test_emits_the_canonical_keys`; TestStructureToDictExtraThreading: `test_extra_can_override_canonical_source_format`, `test_extras_also_appear_in_canonical_extra_subdict`, `test_extras_appear_at_top_level`, `test_issues_array_is_present_when_no_extra`, `test_no_extra_yields_empty_extra_dict_not_none`; TestWorkspacePayloadCanonicalKeys: `test_atoms_carry_coordinates_on_the_atom`, `test_atoms_matches_atoms_list_helper`, `test_extra_defaults_to_empty_dict_not_none`, `test_extra_is_a_copy_not_an_alias`, `test_extra_passes_through_keyword_dict`, `test_has_canonical_keys`, `test_issues_array_is_present_and_a_list`, `test_labels_are_the_kind_tables`, `test_lattice_is_none_for_non_periodic_structure`, `test_n_atoms_matches_structure`, `test_source_format_defaults_to_xyz`, `test_text_is_xyz_bytes`, `test_title_falls_back_to_empty_string_not_none`; `test_every_helper_carries_atoms_for_three_atom_water`

**`test_siesta_stages.py`** (5). Could not fail, or duplicates. Lost: nothing, because the one caller passes the strategy.
`test_default_ladder_is_coarse_and_medium`, `test_default_ladder_is_three_named_stages`, `test_no_stage_carries_a_nonconvergence_policy`, `test_the_siesta_form_has_no_stages_field_to_look_a_policy_up_in`, `test_the_siesta_form_schema_emits_no_stage_table`

**`test_siesta_use_gpu.py`** (3). File gone; its one live rule, ELPA-1STAGE as our default, is now a row of the deck table. Lost: the words of one help line.
`test_diag_algorithm_field_metadata`, `test_use_gpu_default_is_off`, `test_use_gpu_metadata_is_present`

**`test_smiles_and_siesta.py`** (1). Shape. Lost: nothing.
`test_smiles_xyz_header_count`

**`test_ssl_log_filter.py`** (1). Presence. Lost: a second copy of the filter, which answers the same.
`test_install_attaches_filter_to_werkzeug_logger`

**`test_structure.py`** (4). Substring checks. The `END` line moved to the PDB round-trip test. Lost: a banner's formula, seen at once.
`test_basic_construction`, `test_to_pdb_basic_structure`, `test_to_pyscf_string_form`, `test_to_xyz_renders_the_document`

**`test_structure_dna_js.py`** (1). Listed method names. Lost: nothing.
TestSurfacePresence: `test_methods_callable`

**`test_structure_envelope_protocol.py`** (2). One admitted it pins nothing; one checked old keys. Lost: nothing.
`test_a_response_carries_the_envelope_beside_todays_keys`, `test_export_reports_what_the_server_did_to_the_structure`

**`test_structure_name_js.py`** (8). File gone. Its one own fact, the panel asking for the right kind, is a row of the panel table. Lost: the words "Enter a name" could change.
TestErrorPaths: `test_canvas_cancel_passes_through_as_cancelled`, `test_network_failure_returns_envelope`, `test_pubchem_lookup_failure_surfaces_message`; TestHappyPath: `test_name_trimmed_before_post`, `test_successful_generate_routes_through_canvas`; TestInputValidation: `test_empty_name_rejected_without_fetch`, `test_whitespace_rejected`; TestSurfacePresence: `test_methods_callable`

**`test_structure_page_js.py`** (6). Tested a test helper, or a function with no product caller. Lost: Generate pressed before the viewer mounts fails in different words.
TestBind: `test_rejects_missing_canvas`, `test_rejects_missing_modal`; TestSnapshot: `test_onCanvasChange_is_a_passthrough`; TestSurfacePresence: `test_methods_callable`; TestUnboundErrors: `test_loadIntoCanvas_rejects_when_unbound`, `test_the_helpers_that_need_a_viewer_refuse_without_one`

**`test_structure_peptide_js.py`** (1). Listed method names. Lost: nothing.
TestSurfacePresence: `test_methods_callable`

**`test_structure_periodicity.py`** (1). A retired field's tombstone. Lost: nothing.
TestAxisKindIsTheOnePeriodicityField: `test_the_boolean_view_is_no_longer_a_field`

**`test_structure_replace_door.py`** (2). Tested Python's standard library. Lost: nothing; it never saw product code either.
TestTheInterpreterHook: `test_the_replace_hook_is_installed`, `test_the_stdlib_helper_carries_the_trap_on_every_version`

**`test_structure_rna_js.py`** (8). File gone. Strand direction and the whole request moved to the panel table, now asserting more. Thymine refused stays. Lost: blank text is told "must use ACGU" instead; a form the page can't offer reaches the server.
TestErrorPaths: `test_backend_failure_surfaces_message`, `test_canvas_cancel_passes_through_as_cancelled`; TestHappyPath: `test_a_strand_goes_to_the_server_as_typed`, `test_explicit_form_forwarded`; TestInputValidation: `test_dna_thymine_rejected_in_rna_panel`, `test_empty_sequence_rejected`, `test_invalid_form_rejected_client_side`; TestSurfacePresence: `test_methods_callable`

**`test_structure_smiles_js.py`** (2). Listed method names, and a browser with no `fetch`. Lost: nothing.
TestConfigurationErrors: `test_generate_rejects_when_fetch_unconfigured`; TestSurfacePresence: `test_methods_callable`

**`test_system_load.py`** (1). Duplicate. Lost: nothing.
`test_endpoint_returns_envelope`

**`test_task_description.py`** (5). **Your rule, older files,** on three. A flat description's refusal stays held by a prep-protocol row (I removed the refusal in scratch and it went red). Two were shape. Lost: the "name the way out" words for an older description could change.
`test_a_description_written_before_is_refused_naming_migrate`, `test_a_description_written_before_the_job_request_names_migrate`, `test_a_flat_description_is_refused_by_name`, `test_a_task_built_in_code_writes_the_same_shape`, `test_written_file_is_named_and_versioned`

**`test_template_declarations.py`** (2). Restated a value, weak check. Lost: nothing.
`test_pyscf_renders_a_template_at_all`, `test_the_kgrid_is_one_declaration_not_three`

**`test_template_roundtrip.py`** (5). An older template's row, a duplicate, and a filter no product code calls. Lost: nothing.
`test_a_template_missing_an_item_keeps_the_class_default`, `test_select_filters_compose`, `test_select_finds_an_item_by_its_SECOND_category`, `test_select_with_no_filter_is_every_item`, `test_the_machine_answered_items_are_flagged_and_nothing_else_is`

**`test_transport_blueprint.py`** (4). Duplicates, and old names. Kept: the PySCF tab disabled, with its words. The junction button's check moved to a click-through test, written and not yet run. Lost: Describe pressable for a moment before the page's script runs; the server still refuses.
TestTransportPageRendering: `test_active_tab_marker_set`, `test_page_includes_form_container`, `test_page_loads_at_canonical_route`, `test_send_button_is_disabled_until_a_junction_is_cited`

**`test_validation_delivery_contract.py`** (3). Moved, science kept: 4 rows of the prep table, through the real commands, with the planted record gone. The ion's dipole (measured 88.8 D) is one of them. Lost: nothing.
TestAChargedMoleculesDipoleIsItsOwn: `test_a_charged_molecules_dipole_does_not_measure_the_box`; TestTheDipoleAdvisoryAsksTheAxes: `test_a_molecule_at_four_k_points_is_told_and_a_crystal_at_gamma_is_not`; TestTheSubjectAndItsCoordinatesShareOneFrame: `test_the_deck_validates_the_frame_it_emits_and_leaves_the_structure_alone`

**`test_warmfiles.py`** (1). Guarded a list nothing reads. Lost: nothing. The comment claiming it is corrected.
`test_the_restart_group_object_is_not_a_second_declaration`

**`test_web.py`** (14). Text pins and retired buttons. Moved first: the keyword badge, the form's categories, and Add atom, Orient and Rotate, now 3 click-through tests. **These 3 are written and not yet run,** because no end-to-end runs before F9 ships.
`test_all_pages_serve_with_shared_tab_nav`, `test_build_load_pdb_overrides_canonical_source_format`, `test_build_load_returns_workspace_payload`, `test_build_molecule_returns_workspace_payload`, `test_engine_key_pins_load_bearing_pyscf_keywords`, `test_engine_key_pins_load_bearing_siesta_keywords`, `test_index_page_has_tab_markup`, `test_index_page_loads`, `test_modify_page_has_m3_edit_controls`, `test_modify_page_has_m4_orient_rotate_controls`, `test_molbuilder_page_loads`, `test_pyscf_form_schema_matches_documented_layout`, `test_siesta_form_schema_matches_documented_layout`, `test_the_tab_neither_generates_nor_saves`

**`test_web_files.py`** (6). Absence of retired pages; the create bar moved to a click-through test, **written and not yet run.** Lost: nothing silent.
TestSidebarCreateUI: `test_create_bar_in_partial`, `test_three_action_buttons_visible`; TestSidebarPartialAndShim: `test_projects_nav_entry_removed`, `test_projects_page_route_removed`, `test_projects_selection_shim_removed`; TestSidebarStubsUI: `test_preview_modal_starts_hidden`

**`test_workspace_tag_isolation.py`** (1). Absence of retired names. Lost: a retired name re-added by hand.
`test_there_is_no_second_way_to_save`

**`test_wrapper_notify_flags.py`** (1). API shape. Lost: nothing.
`test_a_list_of_names_is_accepted_and_normalised`

**`validation/test_aggregator.py`** (2). A duplicate, and a usage guard. Lost: a future caller gets an empty message.
`test_report_prints_warnings_to_stream`, `test_validation_error_rejects_empty_or_warn_only`

**`watch/test_molwatch_log_parser.py`** (1). File gone; the registry's tests show a reader that claims too much. Lost: nothing.
`test_can_parse_rejects_non_molwatch`

**`watch/test_pattern_matcher.py`** (38). File gone. Text of regexes and speed paths. What mattered is now 14 rows through the real matcher in `parse/test_section_rules.py`, and those rows caught breaks the old tests stayed green on. Lost: speed only.
TestAnyOfShape: `test_all_pattern_matchers_yields_pattern_matcher`, `test_empty_raises`, `test_mixed_yields_plain_callable`, `test_short_circuit_semantics`; TestCompileRules: `test_all_predicate_only_no_combined_regex`, `test_duplicate_name_raises`, `test_empty_rule_list_compiles`, `test_mixed_rules_combined_includes_only_pattern_rules`; TestContainsCiShape: `test_matcher_still_callable_directly`, `test_pattern_escapes_metachars`, `test_pattern_has_no_start_anchor`, `test_returns_pattern_matcher`; TestExistingTestsStillWork: `test_old_style_match_call_works`; TestFindMatchDispatchOrder: `test_no_rule_matches_returns_none`, `test_predicate_before_regex`, `test_predicate_exception_isolated`, `test_regex_before_predicate`, `test_three_rules_first_registration_wins`, `test_two_regex_first_wins`; TestFindMatchFastFilter: `test_fast_filter_skips_regex_rules_when_no_match`, `test_predicate_rules_still_run_when_fast_filter_misses`; TestHelperEquivalence: `test_contains_ci_equivalence_suite`, `test_metachar_heavy_substring`, `test_starts_with_ci_equivalence_suite`; TestMatchesRegexCiShape: `test_case_insensitive_via_compile_flag`, `test_matcher_callable_directly`, `test_pattern_is_raw_passthrough`, `test_returns_pattern_matcher`; TestSectionRuleValidation: `test_empty_name_raises`, `test_non_callable_start_raises`, `test_non_string_name_raises`, `test_pattern_matcher_start_accepted`, `test_plain_callable_start_accepted`; TestStartsWithCiShape: `test_matcher_still_callable_directly`, `test_pattern_escapes_metachars`, `test_pattern_has_no_inline_ignorecase_flag`, `test_pattern_includes_lstrip_anchor`, `test_returns_pattern_matcher`

**`watch/test_registry.py`** (4). Moved to `parse/test_registry.py`. Lost: one line of advice.
`test_detect_unknown_format`, `test_registry_lists_all_parsers`, `test_unknown_format_error_lists_supported`, `test_unknown_format_generic_hint_points_at_the_types_it_reads`

**`watch/test_section_rule.py`** (16). File gone, same as above. Lost: nothing.
TestAnyOf: `test_empty_combine_raises`, `test_first_matcher_wins`, `test_none_matches`; TestContainsCi: `test_case_insensitive_substring`, `test_no_match`, `test_substring_match`; TestSectionRule: `test_full_rule`, `test_minimal_rule_only_needs_name_and_start`; TestSentinels: `test_are_strings`, `test_distinct_values`; TestStartsWithCi: `test_empty_line`, `test_exact_case`, `test_leading_whitespace_stripped`, `test_mixed_case`, `test_no_match_on_non_prefix`, `test_uppercase`

**`watch/test_siesta_parser.py`** (1). File gone; subsumed. Lost: nothing.
`test_can_parse_rejects_non_siesta`

**Helpers gone, with no user left**: `tests/_deck.py`: `assert_fdf`, `fdf_sets`, `fdf_value`; `parse/test_sidecars.py`: `_need`; `test_config_dir_has_one_home.py`: `_write`; `test_diagnostics.py`: `_stub_conda_env_list`; `test_envs_siesta_gpu_recipe.py`: `_configure_flags`; `test_k_point_mesh.py`: `_prep`; `test_molstruct_json.py`: `_payload`; `test_molwatch_preview.py`: `water_struct`; `test_prep_refuses_before_writing.py`: `_ledger`; `test_pseudos.py`: `_fe`, `_water`, `_make_pseudodojo_psml`, `_names`; `test_shared.py`: `_payload`, `_h2o_structure_path`, `_h2o_with_regions`, `_h2o_xyz_text`; `test_slab_orthogonal_choice_js.py`: `_extract_fn_source`; `test_task_setup_notify_js.py`: `_slice`; `test_validation_delivery_contract.py`: `_junction_with_an_assigned_origin`; `test_web.py`: `_assert_canonical_workspace_shape`.

**Table rows**: `geometry_pin.toml` 2 rows gone; `frame_sets.toml` one record reworded; `restart_files.toml` 3 run-script checks loosened from whole shell lines to what matters; `machine_record.toml` `arch` added to a row; `prep_protocol.toml` 4 rows added; `the_deck.toml` 3 rows added; `tests/support/road.py`'s structure helper takes an engine offset.

**Sentences withdrawn**: `model/structure.md` and `model/structure-molstruct.md`, that every retired key is pinned by a test; `sidecars/molstruct.py`'s comment, that each readable version is read by a test; `config/siesta.py`'s comment, that the restart keywords are proven equal to the catalogue; `ops/installation.md` § 9, its test map rewritten, the claim that the § 6.3 fence is asserted being already false; `engines/siesta.md` and `science/chemistry-correctness.md`, naming removed tests.

## Found by 2.67.3's review rounds (2026-11-05 and 06), by the release that owns each

*(three independent readers over the release's whole diff, each finding read again or measured before it was taken; a walk on the real server, Preview only)*

| id | where | finding | door |
|---|---|---|---|
| K2673-1 | **closed 2.67.3** | a reuse's preview said the copy had happened (*copied in*, *copied now: 4.4 MiB*) once the page's own heading went; the terminal's question and the Transport tab's plan before Describe said the same with nothing written | `Reuse.lines(preview=)`, on every door that speaks before the writing |
| K2673-2 | **closed 2.67.3** | a pick mixing a group left as it stands with one prepared said a warning twice and its count sentence between them, the filter being opened again for the second group | the filter held once at the entry, `prep_task`; the benchmark's at `prep_bench` |
| K2673-3 | **closed 2.67.3** | a mix floated the one written stage's deck finding above the stages, as if the stage left as it stands said it too | the description's findings across every answer, a deck's across the stages written |
| K2673-4 | **closed 2.67.3** | a deck's finding on a group left as it stands reached the terminal and never the page (older than this release) | `kept_as_is` carries `deck_findings` |
| K2673-5 | **closed 2.67.3** | the cited pair read for every stage before a first prep, the seed and the leads too, though only a rung that runs per point uses the count | asked of a per-point rung only: 5 reads a folder answer in place of 8 |
| K2673-6 | **closed 2.67.3** | `prep_said`'s both-doors check accepted a page showing fewer lines; the served `group` was pinned by nothing outside the end-to-end tests | the check takes the whole run from the state saved to the next step; `preview_group_header` |
| K2673-7 | **closed 2.67.3** | a launched device's row read ` · tsgf/: empty`, a separator with nothing before it, on the terminal, Task setup and Results | the parts joined, an empty one left out |
| K2673-8 | **closed 2.67.3** | the bench's *see the cells left out above* with the list below it on the page | the word gone |
| K2673-9 | left: nothing goes wrong | a tab left open across the upgrade draws fewer lines until it is reloaded: the old page code on the new answer | a reload clears it; no kept answer crosses a reload |
| K2673-10 | left: measured | before a frame set's first prep the cited pair is read 5 times a folder answer: 2.2 s at 2000 frames × 22 atoms, unnoticed at the sizes a run per frame allows | a real frame set's size decides; none reported |
| K2673-11 | left: the page's | the page's status line above the answer (*Prepared for …*), and beside a preview the header line for line and the files it would write | the terminal has no such button and no preview; `task-setup.md` § 11.1 says so |
| K2673-12 | left: the reason written | `test_transport_task.py`'s cited pair states a level of theory through `contract_fields_of`, a record no run made: a run needs the engine, no e2e runs before F9, and the test's subject is the count | F9's frame-set run reads the same count on a real pair |
| K2673-13 | left | `Structure.to_wire()` still holds `elements` and `n_residues`, read by no route now: the structure's own view, not a response | — |
| K2673-14 | left | the Results note's two sentences for no role and for a run of points stay on the page: no role means no server answer, and a run of points' is read from its view | — |

### What 2.67.3 removed (the list the user saw before the commit, generated from the code against 2.67.2)

| # | removed | who used it at 2.67.2 | why safe | whose decision |
|---|---|---|---|---|
| 1 | `_cli._echo_prep_answer`, `_cli._echo_pipeline_log` | the prep command | moved to `PrepAnswer.lines` and `prep_said`, line for line, apart from the visible changes the changelog names | the user: *Page shows the terminal's lines* |
| 2 | `Reuse.to_wire` | the served `copied` alone | `Reuse.lines` gives the manifest's words | mine |
| 3 | `plan_stage(on_inputs=)` | `prep_bench`, to print a benchmark's grid before its decks | the grid prints with the answer, later in the terminal | mine |
| 4 | the second argument of the `on_found` callback (`notes`) | the terminal's `_show` | notes travel with the answer; a refusal still prints its notes before its sentence | mine |
| 5 | served `differs`, `saved`, `provenance`, `launched`, `attempt`, `points`, `takes`, `resources`, `pipeline_log`, `linked`, `placement`, `stated`, `need`, `shelves`, `copied` | the page's old wording; `saved`, `provenance`, `attempt` a route test, now reading the lines | on `PrepAnswer` still; the page alone calls this route | mine |
| 6 | served `notes` | the page's old wording and a route test; kept on a refusal | the lines carry them | mine |
| 7 | served `kind`, `deck`, `sources` | no page, no test of this route | on `PrepAnswer` still | mine |
| 8 | served `launch.said`, `launch.walk`, `group.header`, `group.need`, `group.launch.said`, `group.launch.fit` | the page's old wording | the page keeps `launch.header`, `launch.run_script`, `group.number`, `group.members`, `group.writes`, `group.launch.header` | mine |
| 9 | served `next` on the prep route | no page | the card lists its own; the terminal's next step stays | mine |
| 10 | `_showPrepAnswer` and its own wording (`Run`, `opened`, `read from:`, `its queues:`, its resources line, a branch nothing reached) | Task setup | the page draws the served lines | the user, the same answer |
| 11 | Task setup's import of `runPointWords` | its answer | the server's `words` in the line; the function stays for Results | mine |
| 12 | `PLACE_NOTE` in Results | its note for a container, a run | the same words, `runs.PLACE_SAYS` | mine |
| 13 | style rule `.ts-prep-answer-line[data-state="warn"]` | nothing set it | — | mine |
| 14 | root `xyz`, `elements`, `n_residues` on a structure response | no page, tool, notebook or end-to-end test; fast tests, now reading `text`, `structure.elements`, `residue_ids`, the same values | `atom_names`, `residue_ids`, `residue_names`, `chain_ids` stay, a viewer folds them in | mine |
| 15 | tests: `test_xyz_alias_equals_canonical_text` gone; `test_emits_every_legacy_key`, `test_carries_legacy_shape_keys` replaced; `test_emits_canonical_keys_alongside_legacy`, class `TestStructureToDictLegacyShim` renamed; `_LEGACY_ALIAS_KEYS` replaced | — | tests of what stays | mine |
| 16 | sentences: `web/molview.md`'s "which rows are worth offering is read from the structure" (the code offers four fixed kinds; `channelKinds` stays with no caller); `web/task-setup.md`'s point "named as the UI names a point"; `execution/job-system.md`'s `stated` row; "legacy aliases" in `web/web-api.md`, `backend-architecture.md`, `model/structure.md`; the prep route's fields and `next`; "the Results sidebar says" of a scratch folder | — | each now says what the code does | mine |

No refusal removed.

## Found by 2.67.2's review rounds (2026-11-05), by the release that owns each

*(an independent reader over the release's whole diff, each finding read again or measured before it was taken; the browser walk's on the real server)*

| id | where | finding | door |
|---|---|---|---|
| K2672-1 | **closed 2.67.2** | a frame of a different size in a set threw `toDisplay is not defined`: I had written the number's door into the model without importing it, and no test reached the line | the import; a row of the install test |
| K2672-2 | **closed 2.67.2** | the k-point sentences read *along x = 1 along x* and *= 4 k-points along an axis* after their names became words | the sentences; the name from the catalogue's label (`kmesh._grid_words`), not a table of my own |
| K2672-3 | **closed 2.67.2** | the low-bias refusal named a page's question alone, so the terminal was told of a question it does not have, and said *got None* | both spellings, and what each answer does; *got* only of a value given |
| K2672-4 | **closed 2.67.2** | citations left where the sweep skipped a file or matched no `.md`: what a failed run says it may mean, a name refused, the run script's refusals at prep, checkpoint's rule codes, three lines of `--help`; and the catalogue's help and refusal reasons, which the first sweep did not read at all | swept; the catalogue's `manual` rows stay, the engine's own manual |
| K2672-5 | **closed 2.67.2** | one printout said `f001` and `frame 2` of one frame | `frame_words` in both tables |
| K2672-6 | **closed 2.67.2** | raw backticks left in the run badge's detail, the Run panel's verdict and a point's line | `codeWords` |
| K2672-7 | **closed 2.67.2** | `model/overview.md` and `model/structure.md` still said Python makes no number | the rule where it lives |
| K2672-8 | **closed 2.67.2** | the unstated-order warning lost what can go wrong: the order may be an engine's, and a calculation described from the file is described again | said again, shortly |
| K2672-9 | **closed 2.67.2** | a line holding only `.`, doubled spaces, *1 unpaired electrons*, comments saying a Send clears nothing | each |
| K2672-10 | **closed 2.67.2** | my tidy step stripped spaces a joined string needed; caught by comparing every changed string with its old self before anything ran | redone from a saved state; the comparison kept as this release's check |
| K2672-14 | **closed 2.67.2** | my frame wording made every prepared frame-set calculation read *changed since prepared*, on the terminal, Task setup and the ladder, a finished one too: `prepared-with.json` compared a stage's points by their words. The tests and the reader both missed it (the reader's search for stored words did not reach this file); opening every page on a scratch frame set found it, when the user asked whether the revision was correct and complete | points compared by their folders (`prepared_with.point_folders`), so no rewording is a change again; a record written before this release by the words it holds, read as today words them, so a voltage added to an older calculation is still said; road rows and a table row, each seen failing |
| K2672-15 | **closed 2.67.2** | still printed after the reader: raw backticks in every long form help and the machine card's line, doubled ones among them; the run script's printed lines citing a chapter; `task:` leading the description's refusals; `cfg.xc_authors` in a help | `dom.codeWords` in the help and the card; the sentences; the prefix gone |
| K2672-11 | left | a run started before this release and continued after it keeps its index's older words for the points it takes over, so one line can read *frame index 1 (of 3) 0.02, frame 3 of 3 0.03* | only an older run's file; no work on older releases' files |
| K2672-17 | **closed 2.67.2** | a reader asked for one thing, what saves text and compares it: my first fix left a calculation prepared before this release silent about a changed voltage list, the launch then running the old points; and five end-to-end tests still pinned words this release removed, which no fast batch runs | the older record's words compared; the pins corrected; every string of the tests the fast batch skips scanned for the changed words |
| K2672-16 | left | a paper's title in a form's references keeps its bibliography's LaTeX quotes (```Level-Shifting''`) | the title as the one bibliography spells it |
| K2672-12 | left | the terminal tags a finding `[config.x]`: there it is the only name of the key a person edits | the page shows none |
| K2672-13 | left | `ncdata`'s writer refusals, which only a developer adding a variable can reach, and a file's own texts: deck and script comments, data-file definitions, a generated config's comments, the pipeline log's rule codes, the methods text's `references.bib`, cite their chapter | not messages (`science/validation.md` § 4.1 R7) |

## Found by 2.67.1's review rounds (2026-11-05), by the release that owns each

*(an independent reader of this release's own changes, each finding measured or read again before it was taken; the browser walk's two on the real server)*

| id | where | finding | door |
|---|---|---|---|
| K2671-1 | **closed 2.67.1** | keeping a picked run across a rescan made the Result file menu do nothing while a run was picked: the file chosen was overridden by the run again | only the picker's scan says `rescan`; a choice in the menu leaves the pick (`web/results.md` § 2.4) |
| K2671-2 | **closed 2.67.1** | three figures for every target printed a count as a rounded float: *MaxSCFIterations 1e+03*, a cap of 2345 as *2.34e+03* | each row of `keys.LEAF_ROWS` states how it reads |
| K2671-3 | **closed 2.67.1** | a picked run that had gone trapped Reload: *click Reload* said for ever, the ladder not redrawn, the gone run's viewer staying, only Back a way out | a rescan's run that no longer opens leaves the pick, the calculation shown and the reason said |
| K2671-4 | **closed 2.67.1** | every return to the browser tab remounted the picked run, against the same-file rule | the re-pick passes the scan's own `force` |
| K2671-5 | **closed 2.67.1** | the phone rule of This machine stood before the base rule it overrides and never applied; my "no sideways scroll" check could not see that | moved after it; 372 px, one column, measured at 390 px wide |
| K2671-6 | **closed 2.67.1** | the reduced-motion rule was a weaker selector than the pulse it should stop | the running dot's own selector; `none` with the preference, the pulse without, measured |
| K2671-7 | **closed 2.67.1** | a benchmark's queued trials were called running | `n_running` and `n_queued`, each counted and each said |
| K2671-8 | **closed 2.67.1** | the export question was a second dialog of its own, copying `lib/dialog.js`'s test of a click off its surface without its guard, so a drag out of a field closed it | built on the one component (`web/ui-contract.md` § 11): `newDialog`, `dialogField`, `dialogActions`, `openDialog`; its look and CSS rows gone from the viewer's sheet; a dependency named in `web/molview.md` § 4 |
| K2671-9 | **closed 2.67.3** | a frame set before prep: *its 2 points* said of a run of 4, and with one voltage nothing said, since the frame count is read from the record prep writes | `transport.rung.frames_had` asking `compose.cited_frames` when nothing is composed: read every caller of `rung_points_of` first, a first prep among them |
| K2671-10 | **closed 2.67.1** | a refused bias list cleared one card of the form, so a shared field's stale row stayed beside it | `_formParts`, one answer for both |
| K2671-11 | **closed 2.67.1** | `web/ui-contract.md` still named *sim time* | the sentence |
| K2671-12 | **closed 2.67.1** | a test of mine hand-built the relaxation record, against the rule that a test holds no record a run did not make | retired; the words read from a real run's record in end-to-end text: `test_a_relaxation_cut_short_is_said_as_it_stands`, the halted PySCF relaxation; F9 runs them |
| K2671-13 | **closed 2.67.1** | migrate's kept-copy rule and a finished run's skip had no test | a row of the older-run table, seen failing; end-to-end text on a finished PySCF run; `PointsOf.runs_points` stated in `runs-of-points.md` § 2.2 |
| K2671-14 | **closed 2.67.2** | a plain structure file chosen in the menu shows *Running, last result …* under the trajectory view (the walk, on `J.source.xyz`) | the badge says a state only of a file that has one |

## Found by 2.67.0's review rounds (2026-11-05), by the release that owns each

*(the user, on this release: "toc.json just need to be regenerated by the standard designed api and test should pin on that api rather than its content")*

| id | where | finding | door |
|---|---|---|---|
| K2670-1 | **closed 2.67.0** | a page a test served wrote the checkout's `toc.json`, or broke with an empty response when a document was not listed yet, so every new document needed a step by hand before a batch | a served page has a documents tree of its own (`support.live_server`); `test_docs_tab` pins `GET /api/docs/toc` on a tree of the test's own, never what the repository lists |
| K2670-2 | **closed 2.67.0** | a rung that died left no trace in the exported file and read as finished, its name in no variable | every run on the `run` axis, at its fill, `said` saying why |
| K2670-3 | **closed 2.67.0** | *converged* read as the run having converged, where it is a force test at the last step stating one, true on a real PySCF log cut after its first step | `force_within_tolerance` in the file; the words on `relaxation["converged"]` |
| K2670-4 | **closed 2.67.0** | *in the order they ran* was false: a stage can be launched again after a later one, measured on a real ladder | the ladder's order, said; `continued_from`, each run's own record (`runrecord.read_continued_from`) |
| K2670-5 | **closed 2.67.0** | a flag not stated read back as -1, which is true in Python; the reader's sentence said NaN | `ncdata.unflag`: `True`, `False` or `None`. Transport's reader keeps its -1, its sentence corrected |
| K2670-6 | **closed 2.67.0** | words: `source` omitting the `.MD.nc`; the energy's mixed precision, so two rungs can read the same to 0.0001 eV; `cell_ang` on an isolated axis, a box and no lattice; a charge's sign; what a step is | said, and `axis_kind` beside the cell |
| K2670-7 | **closed 2.67.0** | a rung spelled by its folder where the `rung` axis uses stage names; runs not checked against the first run's atom map; `log.name.split` in a test | `runs_of` yields the stage's name; the columns compared; `run_of(log).label` |
| K2670-10 | **closed 2.67.0** | a flag read through xarray is NaN where not stated, and `bool(NaN)` is true, so a rung that died read as passed; `xr.open_dataset` then a read of the same file crashed a process, a limit the document owed; *the run's input* named no file a run holds, the deck already holding the move; the force figure said to be what PySCF's test is, where geomeTRIC tests each atom's force length; *which came first*, where the file holds no time; the charges not said to be printed only where a deck asks | the document, and each variable's definition, say so |
| K2670-11 | **closed 2.67.0** | `test_docs_tab`'s first-render test still read the repository's index; the road's `continued_from` could not fail | a tree of the test's own; the record's one writer |
| K2670-12 | left | the flag's read-back lives in this file's reader; transport keeps -1 and spectra unflags by name: no -1 in 47 real transport files, so nothing goes wrong measured | `ncdata.read_variables`, when one of them asks |
| K2670-8 | left | a new folder, or a document at the top of `docs/`, is listed only once the tree has a group for it: the route adds to the groups it has. Said in `web/tabs.md` § 5.3 | nothing goes wrong: such a document needs a group |
| K2670-9 | left | an optimization's output unreadable by permission reads as *holds no step*, naming its output, not the permission | the person is sent to the right file |

## Found by 2.66.0's review rounds (2026-11-05), by the release that owns each

*(the user, on what this release is for: "assure the correct index etc conventions will not cause user to get the wrong data (such as atom index, mapping etc are all correctly concealed and exposed explicitly)")*

| id | where | finding | door |
|---|---|---|---|
| K2660-1 | **closed 2.66.0** | a `Structure` read by `parse` answered its own index questions as if its rows were the person's: `map_to_user_index([0])` said 0 on a run where the map says 13, and `restored()` did nothing | the order record on each structure the layer reads, the pair's or the run's |
| K2660-2 | **closed 2.66.0** | `map_rows_to_user_order` took any rows: a `(step, atom, xyz)` stack of as many steps as atoms was reordered by step, up to 19.8 Å off on a real run, and rows longer than the atoms were cut short with no word | the atom axis, said for more than two dimensions (`axis=`); an axis that does not hold the map's atoms refused by name |
| K2660-3 | **closed 2.66.0** | a queued run, a dead one or a transmission point answered one step, the input geometry prep put in the progress log, as if it were a result; a document's own vibration example had been measured on that | a log's opening geometry is no step (`Frame.preview`); a folder asks the run door's choice (`runs.openable`), not a walk of its own; a run with none is refused by name |
| K2660-4 | **closed 2.66.0** | the largest force was over every atom, so a converged run with held atoms read as far from it: 0.94 eV/Å against the 0.009 its tolerance tests, on a real 52-atom vibration; and it differed by engine, the largest component on SIESTA and the per-atom length on PySCF | `contract.force_figures`, the rule the relaxation record already judged by, on both engines; `max_force_free_ev_per_ang`; the run's own record on the table as `relaxation` |
| K2660-5 | **closed 2.66.0** | `final_energy_ev` could be an SCF row's number: on a running or killed run, and about 1000 eV off on a real run stopped inside a move, the reader flagging only a run still going | a step still being computed or stopped inside is no step (`Frame.cut_short`, read by the reader); `said` says one is |
| K2660-6 | **closed 2.66.0** | a force block the run was still writing answered as the step's: the largest of the rows written so far, `converged` True on a real run whose figure was 0.0185 against a tolerance of 0.01. Older than this release, and every door asking the relaxation record inherited it | `force_figures` states none unless the rows are the step's atoms, so the record stands on the step before |
| K2660-7 | **closed 2.66.0** | `read_steps` crossed a pair saved beside the run a second time, 13 Å off, the run's map asked for a file listed by user index; any `.xyz` passed for a run's steps | only a file the run writes its steps into (`runfiles.role_of`, `run_output_roles`); `parse` and `read_structure` read the rest |
| K2660-8 | **closed 2.66.0** | `held_atoms` took the structure's held label, 20 atoms on a seed that holds none; the force figure beside it took the run's own statement | the run's own output, one source, mapped by the run's map |
| K2660-9 | **closed 2.66.0** | `elements` held the engine's species labels, `Au1` for `Au` | `Structure.chemical_symbols`, the labels on `steps.last` |
| K2660-10 | **closed 2.66.0** | a device point's forces were all NaN with no word beside them; a mistyped file answered its run's steps; a run's `.XV` died on a bare attribute error; a torn forces block or a step of another atom count on a bare numpy error; `stated_by` empty on a log's own map; two rules for a reading's atom count; three reads of a deck for one answer | `forces_not_computed` in `said`; each refused by name; `of_reading` takes the reading; `said` handed down |
| K2660-11 | **closed 2.66.0** | the coordinates in a run's steps stand from its input by the offset prep applied, and nothing said so or gave the number | `applied_offset_ang`, the deck's record, measured exact on four real runs of both engines |
| K2660-12 | **closed 2.66.0** | documents: a pair's `.xyz` said to read as a structure; *on every reading that holds atoms*, *the registry states no map*, *the last step states no energy or forces*, *a step's energy is its E_KS*, *your structure = the first row less the offset* all false somewhere; the engine differences, `eigenvector_display`, `running`'s meaning, `pyscf` absent from the notebook's env unsaid; the vibration document SIESTA's only | corrected, each example run on real calculations of both engines |
| K2660-13 | **closed 2.67.0**, the charges and stress on `read_steps` and in the file | each step's charges and stress are read from a SIESTA run and kept nowhere a script can ask; a `.out` is the only place to read them | an optimization's data file (`model/datafiles.md` § 8), with K2617-10 |
| K2660-14 | **closed 2.67.1** | `jobset migrate` leaves a run as it is when it holds a `*.pre-2.55.0` copy an older migration kept, saying *names no file this molbuilder writes*: measured on a real vibration, so its output keeps its older name and its folder reads as no steps. Naming the file reads it right | the migration |
| K2660-15 | left, the user's word | `read_xv`'s `Structure` knows nothing of the run, so its own `map_to_user_index` and `restored` take its rows for the person's: *"the user should be aware that explicit xv is generated by engine"*. Its docstring and the documents say so, and `parse` reads the same file with the order on it | documented |
| K2660-16 | left | a map listed by user index leaves engine rows as they are, and a `(step, atom)` array of as many steps as atoms looks like `(atom, value)`: nothing in a bare array says which it is. The rule is the map of the file the rows came from, said in the docstring and `model/overview.md` | nothing can tell |
| K2660-17 | left | which of a person's atoms an electrode cell's atom is: an electrode rung is a structure of its own, `said` says its rows are its own, and nothing answers another atom | no door asked for it |
| K2660-18 | left | the same figure is `max_force_free_ev_per_ang` on the table and `max_force_free_ev_ang` in the relaxation record, which a saved pair already holds | nothing goes wrong |

## Found by 2.65.17's review rounds (2026-11-05), by the release that owns each

*(the user, on the first list of what it removed: "i have doubt if you actually did a document and tracking of use/design in full code at all". The audit that answer called for: [`archive/audits/2026-11-05-2.65.17-removals.md`](?doc=archive/audits/2026-11-05-2.65.17-removals.md), every item's role, history, promises and what stands in its place; [`archive/audits/2026-11-05-postprocessing-api.md`](?doc=archive/audits/2026-11-05-postprocessing-api.md), the post-processing layer tried on real sorted runs.)*

| id | where | finding | door |
|---|---|---|---|
| K2617-1 | **closed 2.65.17** | `_every_cited_item_is_read_back`, a guard never connected since 2.16.3, deleted by this release's draft as dead code. Nothing else caught a setting tagged for a citation and read back by nothing | two tests, each seen failing: the tag against the lists, the lists against what a deck the framework writes answers |
| K2617-2 | **closed 2.65.17** | caused failures nothing caught: two recipes on one environment category (this draft had deleted the only test), the Makov-Payne length in the script a run carries, a structure pair half written when the disk fills | a test each, seen failing; the script's length a function a test can call |
| K2617-3 | **closed 2.65.17** | `chemistry` asked `modify` for a deletion, a floor upward, and two functions remapped labels | `Structure.without` on `_of_atoms`, shared with the reorder; `modify.delete_atoms` marks the record |
| K2617-4 | **closed 2.65.17** | the fence's refusal translated at 34 sites and refused in three other shapes in the file routes | `_PickerError` answered once by `web.app`, with its status; `files._must_exist` |
| K2617-5 | **closed 2.65.17** | three tables for what a rule's field is; a 3DNA run's call written twice, one of them saying nothing of a hang | `selection._FIELD_KINDS`; `_threedna._pdb_made` |
| K2617-6 | **closed 2.65.17, by the user's word** | removed by this release's draft and brought back: `annotations_fdf` ("i agree to your #1"), `read_xv` and `read_xv_cell` ("the user should be aware that explicit xv is generated by engine so it is of course engine index based ... these functions should be well documented and exposed to user"), the seven project functions, `FirstN`, `selection.to_json`, `effective_L` | public, documented, in `molbuilder.postprocess` where a person reads a result |
| K2617-7 | **closed 2.65.17** | `list_projects` listed the tree's pseudopotential library as a project and warned of molbuilder's hidden folder | both left out, a test |
| K2617-8 | **closed 2.66.0** | reading a sorted run's own files through `parse` hands back the engine's order with nothing on the answer: measured 22 Å atom for atom against the person's structure, forces likewise | what the transmission reading and `watch parse` already do: the map on the answer; one reader of a run's steps in the user order |
| K2617-9 | **closed 2.66.0** | no worked example for an optimization or a vibration; `parse.md`'s prints `None`, the last frame of a real SIESTA run being its geometry with no energy | the post-processing documents, each example run first |
| K2617-10 | **closed 2.67.0** | an optimization has no data file and no export (the user: "the rest is fine", on its own release) | `model/datafiles.md` § 8 |
| K2617-11 | **closed 2.67.1** | `jobset migrate` on a template holding a retired word for `spin_treatment` ends in a bare traceback, on both old and new code | a refusal naming the item |
| K2617-12 | left | `projects_root(base)` (the contract names it); `FdfParams.xc`, `solution_method`, `z_len_ang` (a deck's facts nothing reads yet); the BENCH-MARKS `field` lines (the contract promises them to tools) | nothing goes wrong |

## Found by 2.65.16's review rounds (2026-11-04), by the release that owns each

| id | where | finding | door |
|---|---|---|---|
| K2616-1 | **closed 2.65.16** | **Task setup's folder answer was a bare 500 on a description its reader refuses**: one of its parts forgot to probe, so the sentence naming `jobset migrate` never reached the page | the route narrows to one `Task` or `None`; every part takes it |
| K2616-2 | **closed 2.65.16** | the run of points' family took the same arguments in three orders and carried `launch`, `names`, `stage` nothing read; `done` asked another state door than the one every reader asks | `calc, stage, run, …` throughout; `done` asks `runs.concluded`; `PointsOf` parts take what they read |
| K2616-3 | **closed 2.65.16** | four hand copies of `point_done` in a reuse of a finished calculation: a term the kind's part gained would not have been seen there | `continuation.point_done` |
| K2616-4 | **closed 2.65.16** | answers and objects carrying what their caller holds: `TaskState`, `JobSetStatus.calc`, `Resolved`'s three fields, `Described`'s, `base` beside `described`, `_one_run_status`, `run_folders`, `_prepare_again` | the calculation whole |
| K2616-5 | **closed 2.65.16** | doors spelled by hand: `index_at`, `rung_points_of`, `Calculation.job`, `Where.of_points`, `warm_carry`, `stage_name`; a stage resolved a second time in `prepare_attempt`; guards `attempts_in` already answers; a row of the catalogue's page form kept twice | each door asked; `runfiles.row_words` |
| K2616-6 | **closed 2.65.16** | a sequence's record states its scheduler and the run read another copy of it; a branch no road reaches; a refusal asked twice before a launch again | `sequence.Record.run_said` asks the record's kind; `first_launch_takes` forwards `cold` |
| K2616-7 | **closed 2.65.16** | `check_rules` was a second fdf reader, its atom-count and species rules a third and fourth scan | `read_fdf(repeated=)`; `check_deck` reads a file through the engine's `read_deck`. On a deck no road writes: a block spelled as SIESTA also reads it is now checked; a block left unclosed is no longer counted, and SIESTA stops on it |
| K2616-8 | **closed 2.65.16** | migrate's fold of a retired stage inside the description's step, `old` meaning two things; the machine record's step another shape | `_fold_retired_stage`; `_migrate_machine_record(rec, said, writes)` |
| K2616-9 | **closed 2.65.16** | sentences: *writes nothing* beside the one write a run's end takes; the fence's *several roots*, *cwd*; a door the contract names in neither § 2.1 nor § 3.2 | corrected |
| K2616-10 | **closed 2.65.17** | `migrate._split`'s repeat of its view, `_retired_here` and `_gone` over `template.retired`, the schema gate; the fence's refusal translated 30 times where an app handler would answer; `_shares`' `reuse`, a dict spelled three times, `plan_stage`'s repeats — `plan_stage`'s repeats left: nothing goes wrong | delete, or the one door |
| K2616-11 | F9 | the picker's plural shape for one root, with its move; `status_line(base, task, …)`, a part whose one direct caller is an e2e line | `projects.projects_root`; the calculation whole |
| K2616-12 | **closed 2.67.1** | a warm start warns *written as the job ended ... may be incomplete* on any file from the same clock second as the job's end: a fast H2 run's complete `J.chk` gets it. A person is told a good file may be bad, with the way out `--cold` | judged with the page's other words |
| K2616-13 | left | `names` as a parameter (a listing per point otherwise); `Path(calc.base)` at 67 sites (a no-op that cannot differ); row keys read by eight readers; each branch on a run of points wanting another answer; *what the run cannot be read* three times | nothing goes wrong |

## Found by 2.65.15's review rounds (2026-11-04), by the release that owns each

*(the user, on these rounds: "review should discipline your coding, unified api and framework level design, detect handcraft code that is not justified, duplicated code and redundant code should be avoided")*

| id | where | finding | door |
|---|---|---|---|
| K2615-1 | **closed 2.65.15** | three scans of a block's fences, a hand-written brace walk beside the JSON decoder, and readers left in `script_emit` only because of who asked them | `deck_record`, the one reader of a block: `_first_pair`, `read_json_block` through the decoder, `read_bench_marks_block`, `written_by` and `molbuilder_words` moved; the marker's retired two-word form gone. USER-CUSTOM's outermost pair, the parameters fence and `check_deck`'s count differ on purpose and stay |
| K2615-2 | **closed 2.65.15** | the run card's two halves split in two modules on one test, a wrapper with one caller, three lambdas computing one answer | `resolve.declared_run_shape` beside `card_settings`; `run_inputs` deleted; `prep_inputs._cell_resources` |
| K2615-3 | **closed 2.65.15** | three copies of a JSON block's frame; a wrapper with one caller | `script_emit._json_block`; `atom_metadata_block` deleted |
| K2615-4 | **closed 2.65.15** | `siesta/layout`'s padded keyword line written five times, the engine's filter by hand twice, a second predicate for *the block says nothing*, two imports for one module, a spacing written by hand | `layout._keyed`, `template.select`, `structure.atom_metadata_has_something`, `energies.spacing` |
| K2615-5 | **closed 2.65.15** | the transport record's version history, 32 lines of another module's fields, in its reader | two lines; the changelog holds the history |
| K2615-6 | **closed 2.65.15** | sentences the moves left wrong: `describe --help`'s *Floor 2 only*, *describing renders nothing* (a reusing calculation renders in memory, writes none), `script_emit`'s header, a test that does not exist, a function deleted, an empty banner, A7's list missing `transport/reuse` | corrected |
| K2615-7 | **closed 2.65.15** | `gpu_count`'s help still said prep proposes the counts, since 2.65.12 refused | the help says it states them, or prep refuses |
| K2615-8 | 2.65.16 | `siesta/layout`'s three scans of blocks `read_fdf` already answers, in the check gate; its first loop stays, its job being to see duplicates | `parse.fdf.read_fdf` |
| K2615-9 | **closed 2.65.17** | never asked: `record.deck_spin`; `decl_line`'s `value`, `indent`; `emit_bench_marks`' `version`; `emit_provenance`'s `form_config_hash`, one test; `parse/contract`'s case-blind key loop; `job-contracts.md` § 3.3's *a higher version makes an old parser refuse*, which no code does — `form_config_hash` stays out by the user's word (2026-11-05), its idea kept in the plan's *Decided* table | deleted, or the sentence corrected |
| K2615-10 | **left 2.65.17** | BENCH-MARKS' `field` lines: about 150 lines write what nothing in the package reads, and `job-contracts.md` § 3.3, § 3.6 promise them to outside tools — nothing in the package reads them, and the contract promises them to tools: a fixed declaration can be the contract | read then, judged by the contract |
| K2615-11 | left | the two schema names held by their readers on the lower floor, the writers importing them: the only two records whose reader sits below its writer, and one constant cannot drift; `extract_*` wrappers, each stating its payload; twin sidecar parsers; pseudopotential providers' shared opening; `deck_record` carried in the monitor's bundle through `energies`, which only real runs can prove (after F9) | nothing goes wrong |

## Found by 2.65.13's review round (2026-11-04), by the release that owns each

| id | where | finding | door |
|---|---|---|---|
| K2613-1 | **closed 2.65.13, in part** | F4-267-c, the plan's *one check of a record's keys* | `persist.unknown_key_words`, the two identical checks one. `persist.check_keys` keeps its words (records molbuilder writes, listing every unknown key) and `validation/task` its near-miss (a stage's override, its own sentence): they differ on purpose |
| K2613-2 | **closed 2.65.13** | documents and comments the merges made wrong: the sweepable row's `run_card`, Task setup's *marker*, a renamed function, the geomeTRIC table, *the config's own* default | swept |
| K2613-3 | **closed 2.65.13** | the merged rename and move sentence lost *which file is where* when both a sidecar's move and its undo fail (`web/projects.md`) | it names the sidecar's path |
| K2613-4 | **closed 2.65.13** | `atom_index`'s two spellings of the geometry suffix, the supervisor's import at each child exit, two provenance headers, a docstring promising a supervisor that outlives an import failure | one predicate; the import before the loop; corrected |
| K2613-5 | said | the release's commit and tag went out without its version line, ledger and changelog: my chain ran on past a failed step | the next commit holds them; the tag is left where it was, the code the same |

## Found by 2.65.12's review round (2026-11-04), by the release that owns each

| id | where | finding | door |
|---|---|---|---|
| K2612-1 | **closed 2.65.12** | the What to measure card labelled the rows the refusal calls *yours to state* as what the machine answers | *a shape to try*; the picker's note gone |
| K2612-2 | **closed 2.65.12** | a row's class was not painted while the grid was unstated: a wait mark that never ended | the refusal carries `rows` |
| K2612-3 | **closed 2.65.12** | the Measure hint and the fold's count spoke beside the refusal, in their own words | one neutral sentence; the fold counts settings |
| K2612-4 | **closed 2.65.12** | documents and two comments still said the machine proposes | swept |
| K2612-5 | **closed 2.65.12** | new cases written as a function beside two tables, and a test its claim had left | rows; retired |
| K2612-6 | **closed 2.67.1** | Results says *still running* of a benchmark prepared and never launched | not this release's page; found in passing |

## Found by 2.65.11's review rounds (2026-11-04), by the release that owns each

| id | where | finding | door |
|---|---|---|---|
| K2611-1 | **closed 2.65.11** | my first fix of F0b-R3-9 stripped the labels in the RNA panel and sent the letters: `3'-UGCA` built the strand read the other way, refused until then | the text sent as typed, as the DNA panel does; rows on the panel and on the route |
| K2611-2 | **closed 2.65.11** | my first *Not run* badge took a class styled `display: none`: nothing shown; its detail said *walk* | a style of its own, seen in a browser; *its run ended without starting it* |
| K2611-3 | **closed 2.65.11** | a Results ladder row's detail showed its backticks, with the steps beneath them rendered | through `dom.codeWords` |
| K2611-4 | **closed 2.65.11** | a schema line gone read *it says None* | *it names no schema* |
| K2611-5 | **closed 2.65.11** | the species sentence said a label *must start with* a symbol, and `Au_1` does | it states the rule: a symbol, with a number after it |
| K2611-6 | **closed 2.65.11** | This machine told a hand-edited bad channel name *it needs a url* | the name's rule first (`_why_skipped`) |
| K2611-7 | **closed 2.65.11** | what the release left behind: an unused `is_dft`, a second lookup in `prep`, *cheaper than ddCOSMO* in decks, two bibliography entries nothing cites | removed |
| K2611-8 | left | a launched, finished run of points is told *takes over every point done*; a run with no warm files *continues*: its next warm launch is refused, naming `--cold` | both sentences carry `--cold` beside them, and the refusal names it: nobody is stranded |
| K2611-9 | left | the vacuum hint misses a slab wrapped across a face and overstates a slanted wire: the same on 2.65.10 | a hint; the person's science |
| K2611-10 | left | Task setup's stage table does not grey `functional` or `grid_level` under Hartree-Fock: its columns carry no `active_when` | prep says it; a stage table greys nothing today |
| K2611-11 | → 2.65.13 | F3-M, which the design listed here: which relaxer moves atoms is still stated by the deck writer and the stepless check | nothing goes wrong; with one home |
| K2611-12 | left | a builder's `ValueError` that is no fault of the text (no conformer, a missing tool) now answers 400, not 500 | the page shows the same sentence for both |

## Found by the test audit (2026-11-02), by the release that owns each

*(The user, 2026-11-02, after a test helper run by hand wrote the suite's machine record into their real
config: "detect those shit and rogue tests", "give me a review of test design and detection of rogue tests";
then "keep the plans order on the conditions that (1) this would not happen in current/new tests (2) clean up is
scheduled".)* The audit, every file read whole, with a row per file:
[`audits/2026-11-02-test-audit`](?doc=archive/audits/2026-11-02-test-audit/README.md).

| id | release | what | the door |
|---|---|---|---|
| TA-GUARD | **closed (tests, before 2.63.0)** | test code ran outside pytest; module, session and collection-time code and child processes ran outside the per-test isolation; tests reached the network; the harness's writers took any folder they were handed | `testing.md` § 2b: test code loads only under pytest; one sandbox for the whole process, set before anything is collected; nothing leaves the machine; one write guard for the whole run; `test_the_suite_stays_in_its_sandbox.py` |
| TA-PREFIX | **closed (tests, before 2.63.0)** | eleven env tests handed the product fake prefixes at the machine's root (`/fake`, `/prefix`, `/scratch/me`), under which it creates folders | the prefixes under the sandbox home |
| TA-DOCS | **closed 2.67.9** | a page test asked the docs route for its index over the real `docs/`, and the route rebuilt `docs/toc.json` in the checkout; the write guard refuses it now, so the next new document fails the test until it is mended | the page tests pointed at a copy of `docs/` |
| TA-RETIRE | **in part 2.67.4; the rest closed 2.67.11** (K27611-5)  | the 22 files the audit retires, and the single tests and rows each group table names | the group tables |
| TA-REWRITE | **server files closed 2.67.11** (K27611-2); the page files' typed server replies 2.67.12  | the 50 files whose inputs are handcrafted (the audit's § 4: machine records, records of runs that never happened, calculation records typed instead of `describe`, older releases' files, typed engine and tool output, typed server replies, invented results) | through the road, or a recorded real run |
| TA-ISOLATE | **closed 2.67.11**: (ii) on a recording env manager (K27611-5); (iv) left by judgement — each file's own engine skip states which engine it needs  | the 56 "fix isolation" files: what the guards close by construction is closed; what is left — the validation tests' reads of the real `projects/pseudopotential/`, the `--clean` test's real prefix behind a recording door, a stand-in server with no projects tree of its own, collection-time engine checks — is F9's | per file, the group tables |

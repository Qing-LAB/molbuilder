# Plan rows closed, withdrawn and measured untrue — consolidated 2026-09-10

**Role:** history. **Not a source of truth.**

`plans/plan.md` held four separate tables of open items plus a 2026-09-01 fact-check and three already-closed sections. The user's instruction: *"consolidate plan and to do, archive finished things and untrue things, clean it up so we have one list"*. The live rows are now one table at [`plans/plan.md` § 2](?doc=plans/plan.md); everything below is what came out.

Under `archive/README.md`'s substance-first rule, a closed section must never be the only place a live invariant lives. Each block below either names where its rule now lives or carried none.

## 1. Rows measured UNTRUE — 2026-09-10

Four front-end rows described code that does not exist, and one stated a missing local file as missing work. They were killed with the evidence in their own commits (`097cbab7`, `1b3b4ac6`, `a8c9d4ed`):

| row | what it claimed | what the code says |
|---|---|---|
| **W8** | three caller-less endpoints, with line numbers, "confirmed still present 2026-09-07" | none of the three is registered. `docs.py` has `/api/docs/toc`, `/img/<path>`, `/read`; `checkpoint.py` has state, list, init, save, tag, restore; `selection.py` has `/api/selection/eval`. The one "JS caller" is a comment reading *"no /api/selection/atoms"* |
| **W17** | citing a flat folder "refuses as ambiguous" | no `cite` verb in the CLI, no flat-shape handling in `references.py`, and every "ambiguous" in our own code is an unrelated comment about valence sites, tag indices or parser markers |
| **W7**, **W12** | browser click-throughs not done | not defects — assertions about the user's time that no code can confirm or deny |
| **W9** | "transport viewer default orientation" | one sentence naming no behaviour, current or wanted. Nobody could act on it or close it |
| **E7** | *corrected, not killed* | it read "No `kind: run` submission exists on any Sol tree", derived from `projects/` on one box. A tree on Sol is not in this repo, and the user has tested submission there. Absence of a local record is not absence of the work |

**Killed by the user on 2026-09-10**, as not worth carrying: **N4** (the science-validation tail — three of its four checks were already somebody else's job or delegated to PseudoDojo) and **W14** (a web plan view, with no stated use).

## 2. Rows DONE, struck in place before this consolidation

Twenty-five, with the area they sat under. Their text is as it stood:

- **engine / science** — | ~~**E2**~~ | **DONE 2026-09-03 by your ruling — *use the exact one when it's there, fall back to the csv*.** `parse_utilisation` is the one door: the monitor's `[UTIL-SUMMARY]` means (averaged over EVERY tick) where it wrote them, `util.csv`'s time-weighted reconstruction (a ≥10%-change-gated subset) where it did not — which is what a KILLED trial leaves, the one a benchmark most needs to read. | `bench-and-junction` § 2.2 | done |
- **engine / science** — | ~~**E3**~~ | **DONE 2026-09-03 — named, with the tell that detects it.** | `bench-and-junction` § 3B | done |
- **engine / science** — | ~~**E4**~~ | **DONE 2026-09-03 — but this row overstated what shipped, corrected 2026-09-07.** The `[MACHINE]` line is written first, unconditionally, on every path. It carries `node`, `cores`, `mem_gb` and `gpu` (`monitor.py:555`) — **there is no CPU model field**, and `gpu` is a device model, not one. If the CPU model is wanted, it is a new item, not a done one. | `bench-and-junction` § 3D | done (narrowed) |
- **engine / science** — | ~~**E5**~~ | **DONE — the run already happened, 2026-08-28/30; the row outlived its own evidence.** 57 files under `projects/` carry `[MACHINE]` lines from live Sol jobs across 8 distinct nodes, in the exact format `machine_line()` emits today (`monitor.py:555`). And `lightwork`'s cap is answered: four probed `environment.json` files record `max_cpus_per_job: null` **with the key present**, which is the contract's *asked and unstated* — the suspected 8-core cap does not exist. **Why it survived:** the evidence is in the gitignored `projects/` tree, so no document could see it. A row whose proof cannot be checked by reading the repo needs its outcome written into a document at the time, or it stays "blocked" forever | `machine-identity` § 6 | done |
- **engine / science** — | ~~**E6**~~ | **DONE 2026-09-03 by your ruling — a reference, not a default.** | found 2026-09-01 | done |
- **engine / science** — | ~~**E8**~~ | **ALREADY DONE — verified 2026-09-03, not changed.** | roadmap § 1 | done |
- **engine / science** — | ~~**E9**~~ | **BOTH HALVES CLOSED 2026-09-03.** | roadmap § 6 | done |
- **config & ops** — | ~~**C2**~~ | **DONE 2026-09-03 — measured on a real built wheel: 90 of 141 static files.** | roadmap § 4 | done |
- **config & ops** — | ~~**C3**~~ | **DONE 2026-09-03 — deleted.** The package was 35 lines re-exporting `molbuilder.builders.backends`, kept *"for external callers"*; and `build.py` carried a comment saying in-tree code goes direct. | roadmap § 4 | done |
- **config & ops** — | ~~**C1**~~ | **DONE 2026-09-03 — and there were more than three.** | `config-access` § 5 | done |
- **front end** — | ~~**W7**~~ | **KILLED 2026-09-10 — not a defect, an assertion about the user's time.** "The browser walk of Results export → cite → describe → prep." Nothing in the code can confirm or deny that a click-through happened, and the user tests and says when they have not | `structure-info` I7 | killed |
- **front end** — | ~~**W8**~~ | **KILLED 2026-09-10 — all three endpoints it names DO NOT EXIST.** The row said "All three confirmed still present with no JS and no non-test caller, 2026-09-07" and gave line numbers. The route tables: `docs.py` registers `/api/docs/toc`, `/api/docs/img/<path>`, `/api/docs/read` — no `/list`. `checkpoint.py` registers state, list, init, save, tag, restore — no `/config`. `selection.py` registers `/api/selection/eval` — no `/atoms`; its one JS "hit" is a comment reading *"no /api/selection/atoms"*. The row describes code that no longer exists and the "confirmed present" was not a check anyone ran | `structure-info` § 3 | killed |
- **front end** — | ~~**W9**~~ | **KILLED 2026-09-10 — one sentence with no content.** "Transport viewer default orientation — a MolView camera-door contract question." It names no current behaviour and no wanted one, so nobody could act on it or close it | `structure-info` § 3 | killed |
- **front end** — | ~~**W12**~~ | **KILLED 2026-09-10 — same shape as W7.** "The live browser walk-throughs" — checkpoint swap at two widths, reload round-trips, a real export, click-selection. A list of things to click, carried as an open defect since the archived molview plan | roadmap § 0.3 | killed |
- **front end** — | ~~**W14**~~ | **KILLED 2026-09-10 by the user.** "A web plan view and a per-stage status roll-up." Task setup describes a staged calculation and Results watches a run; this row asked for a third page showing "the plan as a whole" and never said what a person would do with it. Not deferred: gone |  roadmap § 6 | killed |
- **front end** — | ~~**W16**~~ | **DONE 2026-09-10.** The eight knobs (mode filter, sort column and direction, broadening width, animation amplitude and speed, the amplitude pairing, the thermal temperature) now persist to sessionStorage under the key the contract already named. Hooked into the uiPrefs alias setter -- the one door every write already passed through -- so no render or event code changed. A restored value is taken only when its key is declared and its type matches the default's; storage that throws leaves the defaults standing. `spectra.md` § 7 rewritten, the stale TODO in the bucket replaced, 333 spectra tests pass | roadmap § 3 | done |
- **front end** — | ~~**W17**~~ | **KILLED 2026-09-10 — the behaviour it describes does not exist.** The row said citing a flat calculation folder "refuses as ambiguous" because several stage decks share one `.XV`. There is no `cite` verb in the CLI, no flat-shape handling in `references.py`, and every "ambiguous" in our own code (vendor excluded) is an unrelated comment about valence sites, tag indices or parser markers. Nothing refuses | `structure-info` § 5.6 | killed |
- **front end** — | ~~**W11**~~ | **DONE — verified 2026-09-06 across both halves.** | `structure-info` § 3 | done |
- **needs you** — | ~~**N1**~~ | **CLOSED 2026-09-03 — the rule is written and there is nothing to fix.** | audit 08-28 O4 | done |
- **needs you** — | ~~**N2**~~ | **ALL NINE RETIRED 2026-09-03** | roadmap § 4 | done |
- **needs you** — | ~~**N3**~~ | **ALL THREE WERE ALREADY FIXED — the row was stale, verified 2026-09-03.** | audit 08-28 O5 | done |
- **needs you** — | ~~**N5**~~ | **DECIDED AND BUILT 2026-09-07 — two flags, because they invalidate different things.** One `structure_modified` was set by geometry ops, cell ops AND label writes alike, which made it unusable by the only reader that wanted it: acting on it meant telling a person their mesh cutoff might not apply because they renamed an electrode, which on a junction is routine. Now `structure_modified` (geometry/cell — invalidates the inherited mesh cutoff and transverse k-mesh, both functions of the CELL) and `labels_modified` (a label write — the settings stand, but the electrode/device partition the categorical sort reads IS labels). Both warn, neither refuses. Surfaced at the citation line a person reads when choosing and again on the prep path through `validation.report`. Rule first, in `molview.md` § 8.4a; the source-text pin that counted the calls is now a lint over the edit-landed sites, and which flag each door raises is driven under node | `structure-info` § 5.5a | done |
- **needs you** — | ~~**N4**~~ | **KILLED 2026-09-10 by the user.** The "science-validation tail" -- valence-charge correctness, ghost states, transferability, basis-pseudo consistency. Three of the four were already recorded as someone else's job or deliberately delegated (PseudoDojo's delta-factor), and the row had been carried as a decision waiting on the user for weeks without anyone wanting it. Not open, not deferred: gone | roadmap § 5 | killed |
- **doc drift** — | ~~**D1**~~ | **DONE 2026-09-03 — closed, and now guarded.** | `consolidated-cleanup` § 7 | done |
- **doc drift** — | ~~**D3**~~ | **DONE — and guarded.** | `consolidated-cleanup` § 9 | done |

## 3. Rows WITHDRAWN on 2026-09-06

- | **S8** | **WITHDRAWN 2026-09-06 — this is not a work item.** `engines/overview.md` § 5 is titled **"Adding a new engine"**: it is the checklist a FUTURE engine must satisfy to join the sidecar contracts. It names no existing engine that fails it, and there is nothing to roll out until one is added. | `engines/overview.md` § 5 | **Withdrawn.** The roadmap bullet framed a how-to-add checklist as a rollout "with spectra the only fully-wired instance", which reads as pending work; spectra is simply the engine that has sidecar labels. I then compressed it further, into "four frozen-atom rules only one engine follows" — off one grep line that happened to mention `_seed_frozen_indices_from_sidecar`, which § 5 cites once as an example to mirror. Two layers of drift, the worse one mine. |
- | **S11** | **WITHDRAWN 2026-09-06 — the measurement was meaningless** *(user: "you should use framework and understand what the addition of those code is… instead of counting beans")*. A script generator SHOULD grow as engines gain parameters; line and f-string counts say nothing about whether it is sound. Asked properly instead: `render_run_wrapper` has **18 named section emitters** (`_cold_restart_block`, `_gpu_loadbalance_block`, `_phys_cores_probe_block`, …), uses the existing doors (`script_emit`, `warmfiles`, `identity`, `resolve`) and hand-rolls **zero** path joins. It grew by gaining sections. The framework holds. | roadmap § 6 | **Withdrawn.** What the right question *did* find is recorded rather than lost: the run's basename is read back by four hand-rolled parsers (Python and awk, per engine). Compared on twenty-odd inputs — they differ only where `_validate_basename` already refuses, so nothing is broken. The design behind that is now written where it is read (`job-contracts.md` § 2.5b, `_validate_basename`, both extractors, the wrapper's branch) because it had been re-derived from the regexes more than once. The two Python extractors have **no production caller** and are a deletion candidate. |

## 4. The 2026-09-01 fact-check

*Its one live finding — bench § 2.2's `parse_util_bound` takes the VERDICT and not the numbers, so the item is open — is a row in `plan.md` § 2.*

## 1. What the fact-check found

Nine plan documents, read in full and checked against the code on 2026-09-01.
Three headers were **flatly false** and are corrected in the archived copies:

| document | claimed | actually |
|---|---|---|
| `modify-redesign-plan` | *"items 3 and 4 designed"* | all five built — § 3 carried its own **Built 2026-08-30** marker three sections lower |
| `css-system-plan` | *"proposed, not started"* | steps A and B done **2026-08-02**, four weeks earlier, ticked in its own § 4 table |
| `config-access-plan` | header *"steps 1–4 built"*, body ***"No code yet"*** | one document, two answers; steps 1–4 built and step 5 mostly |

**Two were in no index at all** — `consolidated-cleanup-plan` and
`config-access-plan` existed on disk and appeared nowhere in `toc.json`, so
they were invisible in the Documents tab. Fixed 2026-09-01.

**One item I got wrong first, corrected by reading the function:** bench § 2.2
looked built on a grep for `cpu_mean_pct`; `parse_util_bound` takes only the
*verdict*, and its docstring says the numbers on that line **"are deliberately
NOT read here any more."** It is open.

---


## The former § 5d / 5i / 5j

## 5d / 5i / 5j — CLOSED, and archived 2026-09-07

`parse/scripts/` retired, the projects root made one door in tests as well as
production, and `parse/dirs/_assembler_helpers` deleted. All three shipped;
each earned a rule, and each rule now lives in the document that owns it —
`model/parse.md` § 1a, `process/testing.md` § 2a, and
`execution/running-a-job.md` § 4.2. The sections themselves are in
`archive/2026-09-07-plan-closed-sections.md`.

**§ 2a of the testing doc was written on the day of the archive move**, not
before: until then the projects-root rule existed only in the closed plan
section, which is exactly what the substance-first rule forbids.


## The former § 5k

## 5k — CLOSED 2026-09-08, and in git rather than a second file

The paths framework's first program, M1–M8: every name molbuilder composes got a
door that FINDS it, and every counter-keyed name got composed by its owner.
Twenty-four handcrafted searches → 0, eight hand-built counter names → 0, both
guarded (`tests/test_path_framework.py`). Shipped as `6822f639` and `981534b9`.

The rule it earned lives in the document that owns it —
[`execution/project-layout.md` § 4.5](?doc=execution/project-layout.md).

**Its METHOD did not survive its own last day.** Closing each asymmetry with a
door per question left ~40 public functions answering four questions four ways,
five of them added on 2026-09-08 to fit individual call sites. **§ 5l is the
live plan.** The 367 lines this section held are in git (the commits above and
the plan as it stood at `e9586c09`) — read them there to recover *why*, never to
decide what is open.


## The former § 6

## 6. Closed by consolidation — what was archived, and why

| document | why it is a record now |
|---|---|
| `modify-redesign-plan` | all five items built, plus § 3.4 and § 3.4a/b's removals. Nothing open |
| `transport-design` | *"graduates to a contract when built"* — it did; `engines/transport.md` is `Role: contract` |
| `machine-identity-plan` | all seven pieces built; the two remaining facts need a machine, and are **E5** above |
| `bench-and-junction-plan` | ~85% built history; its four open items are **E1–E4** |
| `structure-info-plan` | I1–I6 done; its open items are **W7–W11** |
| `config-access-plan` | steps 1–4 built, step 5 mostly; the remainder is **C1** |
| `css-system-plan` | steps A and B done; C–F are **W1–W5** |
| `editor-module-plan` | not started, and its design is worth keeping whole — **W6** points at it |
| `consolidated-cleanup-plan` | 10 of 12 items done; the rest are **D1–D3** |
| `roadmap.md` | R3's old home. 1770 lines, ~85% closed work never struck; its live items are **E7–E11, C2, C3, W12–W15, N1–N4** — **plus § 6's architecture seams, which the 2026-09-01 fact-check never reached and this merge dropped**; recovered 2026-09-06 as **§ 5f**, **S1–S14** |
| `archive/2026-09-01-audit-2026-08-21-fullstack-review.md` | its open list became roadmap § 7.5, and § 7.5 is now empty — verified |
| `archive/2026-09-01-audit-2026-08-28-full-review.md` | O1–O3 closed in the document; O4's general case is **N1**, O5 is **N3**, its uncovered lane is **E11** |

---



## 5. Rows DONE inside the design sections, struck in place

Archived by the same consolidation. `§ 5l`'s N1–N4 (the paths standard's first four) and `§ 5m`'s TS rows (the test screen), as they stood:

- **§5l** — | ~~**N1**~~ | **DONE 2026-09-08.** `segments()` in the same tool, wired into `--check` and the summary, with the override discipline the other two passes use. Corrected the row above: 10 sites, not 9. Five mutations, five killed — including the `GUARDED_UNDECLARED = ()` kill switch and the first-component rule (dropping it lets Flask routes back in) | shipped |
- **§5l** — | ~~**N2**~~ | **DONE 2026-09-08.** `FIELD_SHAPES` declares a field's shape once (as `QUALIFIERS` does for counters); `Artifact.fields` declares which rows carry one; `.runwrap-*.log` → `.runwrap-{stamp}.log` + `stamp`. `role_matches` and `_tail_of`'s pattern arm **deleted**, and both pattern branches in `find` / `find_by_role` are equality again. Six mutations, six killed | shipped |
- **§5l** — | ~~**N3**~~ | **DONE 2026-09-09.** `molbuilder/ref.py` — `Ref` (③'s label through ⑤, plus the bench qualifier, counters and fields) and `compose` / `find` / `parse` over whole paths, L1 and importing only its two floor-1 siblings. **Two layout rules had to move down first**: `bench_container` and the trial name were `jobset/materialize`'s, floor 4, and the address layer is floor 1 — `materialize` still answers, by asking. `paths` also gained `bench_containers_in`, the SEARCH half `bench_container` never had (§ 4.5's pairing), because the flat container carries its stage in its own name and a search with no stage in hand cannot compose it. **17 tests over the address, parametrised on both shapes**; seven mutants, seven killed — two only after the first pass left them alive (`parse`'s own round-trip check, and the flat container's stage qualifier, whose measured defect needs TWO stages to see) | shipped |
- **§5l** — | ~~**N4**~~ | **DONE 2026-09-09, and the count was a hypothesis as § 5a says.** Re-derived: the attempt functions (`attempts`, `latest_attempt`, `run_dir`, `resolve_attempt`) already delegate to `paths` — M4b did that — and `job_dir_name` / `bench_container` moved in N3. **Two sites were left, and each had a comment admitting it.** `sweep_set_paths`' docstring said outright *"one door to COMPOSE a path, none to FIND one — is what the paths framework is for; when it lands, this is one of its callers"*: it now asks `paths.bench_containers_in`, which gained a `shape=None` arm for exactly that caller (an error path with no `job-set.json` to read a shape from). `materialize.trials_in` spelled `startswith(TRIAL_PREFIX)` and now reads through `paths.trial_point`. **Two searches and two exemptions retired** — 55→52 and 16→14 — and `--check` failed on the stale entries, which is the guard working. Five mutants, five killed, one only after the first pass seeded every container and so never exercised the existence check. **The third item is BLOCKED, see § 5l.4b** | shipped |
- **§5m** — | ~~**TS1**~~ | **DONE 2026-09-09.** Both now fail on a mutant. The fence test builds a path genuinely outside the root and **excludes the echoed path from its own evidence**, which is what made the old assertion unfalsifiable; the axis test asserts `along a` AND that no axis which fits is named. |
- **§5m** — | ~~**TS2**~~ | **DONE 2026-09-09.** `_LEAF_KINDS` declares what each leaf field must hold — a third table beside the two `selection.py` already had — checked in `from_json`, at the boundary where untrusted JSON becomes a rule. All four shapes answer 400 with the envelope, and a bool is refused where an int is wanted because `True` as an index would silently evaluate to atom 1. |
- **§5m** — | ~~**TS3**~~ | **DONE 2026-09-09, rule first in both.** The topic refusal carries **no count at all** — the list is right there, and a number beside it is a second thing to keep in step. `web-api.md` § 1 now says a path escape is **400**: the roots are an addressable-space boundary, not a permission model, and 403 is the admin gate, which is genuinely authorization. Both restatements followed the rule, not the other way round. |
- **§5m** — | ~~**TS4**~~ | **DONE 2026-09-09.** `validation.task.config_class_for` gives the engine→config lookup one owner and both save-route arms ask it; the guard file written for this defect now covers the two sites it named and missed. The XSS allowlist matches **exact paths** and gained an artifact lint over the table — which immediately found **nine more dead exemptions** (**#77**), each a standing permission for whatever is written at that name next. |
- **§5m** — | ~~**TS5**~~ | **DONE 2026-09-09, six items, every one mutation-verified.** **#76** the chirality flip — a non-coplanar four-atom fixture and the SIGNED triple product, the only quantity that separates a rotation from an improper transform (an inversion and two reflections killed). **#78** the sidecar↔selection path, joined end to end through the real doors, asserting on element+position and never on the index, with REPEATED elements so a shift stays chemically plausible (off-by-one, bleeding labels, empty labels killed). **#66's gap** — the BROKEN env-state gate, the test the deleted one only promised (4 killed). **#70** — one artifact lint over every path-taking route, not six tests; the fence removed from each of the six in turn, six kills. **#72** retired (its contract lives in two documents, the JS source, and the class below it). **#69** deleted. **Residual, recorded not fixed:** the depth-2 canonical-topic rule is still written twice, live and inline, in `rename` and `delete` — merging them is a design change, not the removal of dead code |
- **§5m** — | ~~**TS7**~~ | **DONE 2026-09-09. The harness was broken TWICE and the audit's rate is ~50%, not 1 in 5.** (a) it loaded its coverage plug-in as `-p covplug`, never committed, and failed SILENTLY — every pair read `NO-COVERAGE`; (b) it had **no operator dropping a clause from an `or`**, the commonest shape in a validator, whose two tests reach one clause each. Both fixed. Re-run: 22 pairs / 19 candidates → **9 KEEP, 10 cuttable**; one of the ten was already absorbed by `TS13`, so **nine applied**. **Each was hand-checked for its distinguishing input rather than cut on CONFIRMED** — and one, `test_accepts_str_path`, proved a FALSE confirm (its defect is *rewrite the door to use `Path` methods*, which no operator generates); it is cut anyway on § 3b grounds, because both shipped callers pass a `Path` so the capability it guards is unused. **Remaining: the 4 subprocess pairs need another method** — `sys.settrace` does not follow a child | shipped |
- **§5m** — | ~~**TS11**~~ | **DONE 2026-09-09 — test the door the app calls, with a real file.** *(user: "prove api that actually used in app, not some bogus thing that has no relevance")* `read_config(path)` is the validator and `web/app.py:254` refuses to start on it; the auth tests called `_normalise(dict)`, the middle step only. Five dict tests became a seven-row table over real files, covering three cases a dict cannot reach (truncated JSON, top-level array, top-level string) **and the R10 defect** — the refusal naming WHICH file, whose disabling now fails four rows and previously failed nothing. It also keeps the three tests the broken harness would have thrown away, because `providers: []` is now a row you can read | shipped |
- **§5m** — | ~~**TS12**~~ | **DONE 2026-09-09 — the wizard validates its own output.** `auth_setup.emit_molbuilder_json` wrote a config and never asked the validator, so `molbuilder auth setup --output X` could only be checked by moving X into place and starting the server. It now reads the FILE back through `read_config` — the same door, so the wizard cannot drift from what the server accepts. Its duplicate rule went with it: `build_auth_block` raised `ValueError("at least one provider is required")` where the validator says `RuntimeConfigError("non-empty list")` — one rule, two implementations, two exception types, nothing checking they agreed | shipped |
- **§5m** — | ~~**TS13**~~ | **DONE 2026-09-09 — and the count lesson is the finding.** 23 refusal tests became ONE table of 50 rows through `read_config(path)` with real files: **80 → 58 test functions, 1,584 → 1,439 lines, collected 131 → 136.** Functions and lines fall; COLLECTED RISES, because the old tests hid several cases inside `for bad in (...)` loops and the table names each one. **Re-aiming does not shrink a suite — it makes it legible**, and legibility is what was actually missing: the day's whole error came from an auditor reading two tests, seeing one LINE, and calling one a duplicate when they reach one CLAUSE each. In a list you see both rows. Mutation-verified against every clause of the three two-clause guards, and the fifth mutant found a real gap in my own table — a TRUTHY NON-STRING (`id: 42`) slips past when the `isinstance` half goes, which no existing test covered either. Two rows added. Kept the two message-quality tests a table cannot express (the retired key must not read as a typo) | shipped |
- **§5m** — | ~~**TS14**~~ | **DONE 2026-09-09, and the answer was different for each.** The topic rule: the two copies are NOT the same rule — `rename` refuses outright, `delete` takes `force=true` so the sidebar kebab can proceed after a confirmation. That is why `_validate_op_target` could never centralise them: it took `(resolved, op)` and had no notion of `force`. What they DO share is the three-term CONDITION, now `_is_canonical_topic_dir`; each policy stays where it belongs. Three mutants on the predicate fail tests of both routes. The Makov-Payne formula: **left as two copies, with a note**, because the constants are already interpolated from one home so a copy can only drift on four branchless lines, and the machinery to splice a function's source is more moving parts than the thing it guards. The note names `runwrap._config_dir_source` as the pattern to switch to *if it ever grows a branch* | shipped |

## 6. Rows measured CLOSED — the second pass, 2026-09-10

*(user: "see if we still have any remaining issues, other than finishing the consolidation/clean up of tests")*

**Seven of § 2's rows were not open.** Three said so in their own state cell and had simply never been moved (**S2**, **S5**, **S12**) — which is § 5f's own warning about a summary disagreeing with the evidence beneath it, one level up: the *table* disagreed with its own cells. Four were measured closed today:

| row | measured 2026-09-10 | evidence |
|---|---|---|
| **R4** | **CLOSED** — and § 5a's 2026-09-06 line *"R4 is **not** done"* was wrong when it was written. Both halves of the row are in the code, and landed three and four days before that check | the buffer/model split is gone — `patchDoc` (`task-setup/viewer.js:2925`) ends `_task = task;  // the model moves with the buffer, never after it` (`cb4d9197`, 2026-09-02). The 400 ms window is closed by the page-wide fence — `applyBlockToDoc` wraps every card write in `underFence`, whose comment quotes the ruling that put it there (`0efe541d`, 2026-09-03). The fence is live on this page: `page-busy.js:139` registers `window.molbuilder.pageBusy` at import, and `task_setup.html:499` loads the projects sidebar, which imports it through `projects/state.js:31` |
| **S9** | **DONE 2026-09-06** — the row's own state cell said so, but the cell was filed under **S10** | `backend-architecture.md:99` now retracts *"vestigial wrapper"* in writing; `web/web-api.md` (which owns the HTTP API) and `model/structure.md` agree; `_shared.py:413` `structure_to_dict` is the composer, exported at `:1419` |
| **S10** | **CLOSED — M2a cannot happen through the placement door, and M3's `account` half has no door at all.** One fact, not two disagreeing | **M2a:** `scheduler/emit.py::Directives.of` reads partition and QoS **from a `Placement`**, never from the config block, and `place(..., named=…)` (`scheduler/place.py:199`) refuses a declared domain the record does not offer. Checked on real Sol output rather than a fixture: `projects/Au-BDT-Au/optimization/sol/AuBDTAu-slabcorrected/01_coarse/bench/launch/bench-group-gpu-G2K24C1.sbatch` carries `-p htc -q public -t 0-04:00:00`, and that tree's `environment.json` `htc` row reads `partition: htc, qos: public, max_time: 4:00:00`. **M3:** `runtime_config` has no `scheduler.account` key, nothing in the tree emits `#SBATCH -A` or an `-A` flag, and `configuration.md` § M-1 already records `Site.qos` / `Site.account` as fields *"nothing has ever written"* — so *"a declared account appears in no run record"* describes a declaration that cannot be made |
| **S14** | **CLOSED 2026-09-08.** Its state cell held **S12**'s pre-fix finding, so the row asserted a guard absent that S12's own cell shows written — and its real subject had been fixed two days before this consolidation carried it as open | `parse/dirs/job.py::run_status(run_dir, match)` takes the glob, `_enumerate_files(run_dir, match)` narrows the bucket to one rung, and `jobset/runstatus.py:267` passes **the same glob the existence check used**. The docstring carries the measurement: *"with a later stage's `.out` present a stale rung read 'running'; with that one file moved aside, 'stale'"* — which is precisely this row's *"two flat stages that have both run report the same state"* |

**Why these four and not others.** S10's and S14's state cells had been misfiled onto a neighbour since before the consolidation (`bdd87214^` carries the same pairing), and § 5f recorded that on 2026-09-07 without filling the hole. **The two rows nobody could close were exactly the two with no evidence of their own** — which is § 5a's rule seen from the other side: a row is evidence of when it was written, and a row holding somebody else's evidence is a row that cannot be checked at all.

The rows as they stood:

- | **R4** | run-decision round | *(priority P2)*  | **Three writers, one buffer, a 400 ms loss window.** `syncFromModel` writes from `_task`; `applyAsksToDoc` and `applyNotifyToDoc` patch the buffer and never touch `_task`. Type a memory value, blur, click "+ Add stage" inside the debounce and the `allocation` block is gone | verified |

- | **S2** | architecture seams | **`jobset/runstatus.py`'s warm-file table → producer-supplied inventory** | `backend-architecture.md` § 5 (**W2**) | **CLOSED — verified 2026-09-06.** `runstatus._warm_files()` calls `_carry_inventory(engine)`, the § 4.2a data file's own reader; its docstring records retiring the module-level dict that stood there |

- | **S5** | architecture seams | **`script_emit` re-filing** (its former sibling `bundle_writer` retired 2026-08-29) | `backend-architecture.md` § 5 (**W5**) | **CLOSED — verified 2026-09-06.** `script_emit` sits in `backend-architecture.md`'s **Data management** column now; W5's row is the stale half |

- | **S9** | architecture seams | **`structure_to_dict` disposition — two documents disagree.** `model/structure.md` calls it the retained web composer; `backend-architecture.md` § 2 calls it a vestigial wrapper to delete. **One decision, then align both** — this is a ruling, not a code change | both docs | **needs a decision from you** |

- | **S10** | architecture seams | **Capability and allocation reach `prep`.** **M2a** — capability is assembled twice and never reconciled: topology and the detected partition go to `environment.json`, the `molbuilder.json` `scheduler` block goes straight to the `.sbatch` emitter, and nothing compares them, so *the record can name one partition while the header submits to another*. **M3** — a declared `qos` or `account` appears in no run-directory record. M1, M4, M5, M6 hold | `execution/project-layout.md` § 2.3.1b | **CLOSED 2026-09-06 — there was no decision to make.** `web/web-api.md` owns the HTTP API and documents the legacy aliases as part of the response shape, with no deprecation; `model/structure.md` agreed. `backend-architecture.md` was the single restatement out of step, and is corrected. **I asked about this three times instead of opening the contract that owns the surface** — the rule is to find the owner first, and a disagreement between two documents is a question about which one owns the concept, not a question for the user. |

- | **S12** | architecture seams | **GPU detection is implemented twice** — Python at prep (`.sbatch` header) and awk at launch (after a person may have edited the deck). **Two implementations are required** — one runs on a login node, the other on a compute node hours later — and the truthy set is already one constant. **The fix is a test rendering both against one deck set, never a merge** | roadmap § 6 | **CLOSED 2026-09-06 — the guard is written.** Twenty decks (both spellings, every truthy and falsy value, last-wins including across spellings, case, leading whitespace, a longer token, a commented-out line, a bare keyword, a trailing comment) driven through **both** detectors: the Python one directly, and the wrapper's own awk + `case` block **extracted from a rendered wrapper**, so the test exercises what ships. They agree on all twenty — there was no live divergence, and now one cannot appear silently. Mutation-verified twice: dropping the older `Diag.ELPA.UseGPU` spelling from the awk alone, and making the awk take the FIRST occurrence, each fail with the deck they diverged on named. The row's own resolution held — a test, not a merge. |

- | **S14** | architecture seams | **Floor 6, flat layout: one stage's verdict is still read from the whole folder.** Now sharper than when it was written — the 2026-09-05 ruling made *stages separate runs*, so reading a verdict folder-wide is reading several runs as one | `execution/architecture.md` | **OPEN — verified 2026-09-06.** Python at `runwrap.py:1594` (`_fdf_requests_gpu`, called at 1786), awk at `runwrap.py:2320`, and the emitted comment states they share one rule: *'BOTH keyword spellings, SIESTA fdf_get's truthy set, LAST occurrence wins'*. **The guard this row proposes does not exist** — `test_siesta_use_gpu.py` drives the Python half alone |

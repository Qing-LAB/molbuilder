# § 1s — what is computed vs how it runs: the plan's record — 2026-10-21

**Role:** archive — the plan's § 1s as it stood when 2.27.0 shipped it (N5): the evaluation of 2026-10-20, the separation and its items, verbatim under the original heading. The rules live in [`execution/job-request.md`](?doc=execution/job-request.md); the user's four decisions of 2026-10-21 are its §§ 4–5 and § 2, and the release is the changelog's 2.27.0 row. Nothing of it is open.

---

## 1s. What is computed vs how it runs — the evaluation, and the separation *(2026-10-20; the user's yes: "build it as 2.24.0 after 2.23.2" — N5 in § 0a's queue)*

*(the user, 2026-10-20: "it is completely stupid to have this involved in a
job scheduling"; "carefully evaluate the jobset logic and see how much
mixing you already have between science request vs job request")*

**The finding, verified at the code** (one fresh audit plus a read of the
description, the plan, `resolve` and the catalogue): the two are mixed in
five places, and the first two cost a person a run.

| # | where | what is mixed | what it costs |
|---|---|---|---|
| 1 | the transport gather's *same engine input* check (`siesta/engine_input.py`, `gather.py`; `transport.md` § 6.1) | the deck compared as science while it carries `BlockSize` (prep derives it from the rank count), `Diag.ParallelOverK`, `Diag.Algorithm`, `Diag.ELPA.GPU`, `TS.SolutionMethod` (its own help: *"never the answer"*) | more ranks, another solver or the GPU switch makes a finished seed or device run "another calculation"; the rung after it refuses to gather — met here: an older device run differs in `Diag.ParallelOverK` alone |
| 2 | a launched stage is frozen whole (`job-system.md` § 5.0 rule 5, `submit._as_described`) | the deck (science) and its header and run script (queue, wall, memory, ranks) prepared, frozen and compared as one | a run out of memory or wall cannot continue on a bigger request; the only road is the state saved before its group's first launch, its runs left behind; `--time` is the one value a launch may restate |
| 3 | the description `task.json` | beside the science (structure or citation, kind, stages and what they vary, voltages): the queue, wall and memory (`allocation`), the run card (`execution`, each stage's own), the benchmark grid (`bench`), the notification policy (`notify`), which stages share a job (`stages[].group`) | a queue change is a description edit, saved and checkpointed as one; it then trips 1 and 2 |
| 4 | the template, the science file | 13 run settings (`execution`) carry template values beside the run card — `block_size`, `parallel_over_k`, `diag_algorithm`, `use_gpu`, `ts_solution_method`, `ts_elecs_out_of_core`, `wall_slack`, `continue_retries`, … (`template.py` writes them, `resolve` lays the card over) | two homes: a template solver or `use_gpu = true` is silently in force while the card is blank; the Build tab and the run card both show them |
| 5 | catalogue categories | `density_fit` is `method` + `execution` though it changes the energy (its help: up to 0.1 kcal/mol); `restart` — what a stage builds on — is `convergence` + `execution`, set on the run card; SIESTA's `continue_retries` is `execution`, PySCF's `geom_continue_retries` `procedure`, one step budget | a science choice sits on the resources card, a run setting decides a ladder link |

And the command line: `init`, which describes the science (`--bias`,
`--slot`, `--structure`, …), sits in the `jobset` group beside `prep` and
`launch`; it needs nothing of the job code but the next-step line it prints.
The docs state the boundary two opposite ways (`task.py`'s header and
`project-layout.md`: *"names no machine"*; `stages.md` § 6.8: the person's
asks belong in the description).

**The rule proposed**: *a calculation is two records with two editors — the
description says what is computed; the job request says how and where it
runs. Nothing in one is read as the other.*

1. **The job request leaves `task.json`** — `allocation`, `execution` (and
   each stage's), `bench`, `notify` and the stages' groups — into one file
   beside it (proposed: `job-request.json`), written by the run card and
   prep's flags, saved on its own; `task.json` holds the science alone.
2. **The run settings leave the template**: every `execution` item is
   valueless there, as the machine-answered ones already are, and lives on
   the run card alone. `density_fit` becomes `method`; `restart` a stage's
   own science (what it builds on), never on the card; both retry budgets
   `procedure`.
3. **"The same calculation" compares science only**: the same-input check
   leaves out every `execution` keyword, as it leaves out the engine's
   scratch (this closes the open question in the handover).
4. **A launched stage's job request may change**: prep writes its header and
   run script again; its deck — the science — stays frozen; the next launch
   sends the new request, and its `run.json` records it.
5. **The command line**: `init` leaves `jobset` (proposed:
   `molbuilder describe`); `jobset` prepares, launches and observes.
   `launch --from/--cold/--gather` and `prep --from` stay: each picks one
   run's input among runs that exist only after the earlier run, and is
   recorded with that run (`continued_from`, `written_at`).
6. **One statement of the boundary** in the contract, every opposite one
   removed; `Resources` the job request alone (`program` moves to the job's
   traits, the notify policy to the job request).

`shape` stays in the description: it is how the folder is laid out, fixed at
`init` for the folder's life — neither science nor a request that can change.

**The scope, measured** (2026-10-20):

| what | affected | how |
|---|---|---|
| calculations on disk | every one | the readers refuse an older `task.json`, naming `jobset migrate`, which moves the keys and the template's run-setting values onto the run card — decks render the same, so no prepared stage is refused (to check on a copy of a prepared calculation) |
| code reading the moved keys | `allocation` 31 lines in 7 files; the run card 16 in 8; `bench` 18 in 7; `notify` 6 in 4; stage groups 33 | through one reader of the new file |
| the Task setup page | 182 of `task-setup/viewer.js`'s 4892 lines name these keys; its Save | two records saved |
| tests | 67 files name these keys; 37 calls in 22 files drive `jobset init` | re-pointed |
| docs | 93 mentions of `jobset init` in 30 files; the boundary's statements | rewritten |

**Re-measured at 2.26.4** (2026-10-21, before the contract): row 1's premise
was wrong — prep no longer works `BlockSize` out from the rank count; a
person sets it or it is not written — and the four other keywords are still
compared; row 2's rule is § 5.0's agreement item 4 and checkpoint 2a; row 4
is 8 items written with a value, 10 that may hold one. Missing from the
table: the run card's `restart` decides the groups (`group.upstream_of`);
the description's writer and check import `jobset.group`; launch refuses a
rank count against the deck's own comment (`agreement`, BENCH-MARKS
`mpi_np`) though nothing in the deck is worked out from it now; run settings
written into the decks themselves (a SIESTA deck's PROVENANCE, a PySCF
script's `max_memory`).

**Decided** *(the user, 2026-10-21)*: **a launched stage's deck keeps its
science, never its run settings** — prep renders it again with the new run
settings and refuses a stage whose science differs from what its runs ran;
and **a speed setting nobody stated takes the catalogue default**, written
and recorded as not chosen (ranks and threads stay refused unstated, a
queue's wall, memory and queue too).

**When**: N5, contract first — § 0a's queue, *NOW*. **The contract is
written** (2026-10-21, not committed): [`execution/job-request.md`](?doc=execution/job-request.md),
new, and the chapters it moves rules out of or points into (`stages.md`,
`template.md`, `job-system.md`, `architecture.md` A3/A17, `project-layout.md`,
`job-contracts.md`, `task-setup.md`, `pyscf.md`, `transport.md` § 6.1, and the
`jobset init` → `molbuilder describe` sweep). It waits for the user's reading
before any code. Two automated review rounds (2026-10-21: three readers,
then two — documents, code, science) found and fixed: the same-calculation
door's floor (`deck_record`, each engine's reading registered by the
surfaces), gpu4pyscf's two limits behind PySCF's `use_gpu` (no stability
check, an unchecked solvent model), the conditions
each speed setting holds under, a launch again sending what prep last wrote
(R-L15 at a first launch, `--from`, `--gather`), the agreement's retirement
named whole, `jobset migrate`'s full moves, and some forty leftovers in the
neighbouring chapters. Where it differs from the table above:
`continue_retries` stays a run setting — a stop that launching again resumes
is the run's, as the wall is, and the door could not see it in a deck; the
program a job runs is the job's (`Job.program`). **Decided** *(the user,
2026-10-21, the four recommendations)*: the flat layout's launched stage is
not prepared again (restore only); the calculation-wide run card is written
from Task setup's *every stage* view; every run setting stands on the run card
from the start, off the parameter tabs; gpu4pyscf's limits are a hint, never
counted or refused — the run card's GPU switch and a preflight `warning` for an
unrestricted reference or a solvent model on the GPU (the user, 2026-10-21: "a
hint and information is enough"); the vibration's solvent-with-GPU `error`
(`validation/spectra.py`) becomes that warning. **For the code**:
`tools.manifest --check` differs until `runfiles.WRITTEN` carries the
`job-request.json` row, the `.pre-<rule>` copies `jobset migrate` keeps, and
says `molbuilder describe`; the catalogue's rows move (five into `staging`,
`run_card` on every run setting, `restart` and `density_fit` out of
`execution`); about 75 test files encode the old rules, most through
`tests/support/road.py`'s two builders.

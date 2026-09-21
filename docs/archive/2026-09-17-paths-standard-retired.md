# The paths STANDARD (§ 5l) — retired 2026-09-17, archived 2026-09-20

> **Not a source of truth, and deliberately not a plan.** This section was
> RETIRED by user ruling on 2026-09-17: `molbuilder/ref.py` and
> `tests/test_ref_address.py` were deleted with it. It is kept because the
> reasoning is the most useful thing in it — a module built, never adopted,
> and read as *done* from the L1 index while answering nothing.
>
> It moved out of `plans/plan.md` on 2026-09-20 because 380 lines of retired
> design in the live plan is 380 lines a reader has to work out is dead. The
> §§ 5k and 5m either side of it are live; this is not.
>
> **What stayed behind:** § 5k is the paths framework and stands. `plan.md`
> keeps a one-line pointer at § 5l so the twenty-one internal references and
> the citations in `architecture.md` and `process/testing.md` still land.

## The section as it stood

**The paths STANDARD — one address, three verbs — is retired, and
`molbuilder/ref.py` and `tests/test_ref_address.py` are deleted with it.**

**What it was.** After § 5k closed 24 handcrafted searches, the API had been
grown *"by adding a door for whatever question a call site happened to ask"* —
~40 public functions, four APIs answering *which files are this rung's*, three
answering *what is this file called*, sixteen answering *where does this thing
live*. The user's 2026-09-08 ruling was that the direction of fit must reverse:
*"don't twist the API design such that you can fit arbitrary kind of need, but
rather a standardized API that has reasonable extensibility."* § 5l's answer was
a single `Ref(label, stage, bench, attempt, role, counters, fields)` with
`compose` / `find` / `parse` over it.

**What actually happened.** `ref.py` was written — 444 lines, the address and
all three verbs, layers 2 and 5 of the six-layer stack § 5l.5 specifies. **The
migration never ran.** N5, N6 and N7 were never started, so in every revision it
existed the module's only importer was its own test. It was carried as L1 in
`architecture.md`'s index and in `test_layering.py`'s enforced set, so it read
as part of the framework while answering nothing.

**Why retiring is the right call and not a loss.** The standard's own premise was
that ~40 functions answering four questions is too many doors. Shipping a
seventh module implementing a *replacement* vocabulary, and then not migrating,
made it ~40 functions **plus** a parallel address nobody called — the exact
failure § 5l was written to end, arrived from the other side.
`process/testing.md`'s rule names this shape: *"a unification that ADDS has
usually not unified anything — it has added a layer and kept the old surface."*

**What stays, and is untouched by this.** § 5k's RULE — *for every name it
composes, the framework owns the search* — is closed, enforced, and green:
`tools/classify_path_finders.py` (three axes, in `--check`),
`tests/test_path_framework.py` (22 tests), and the `GUARDED_UNDECLARED` set that
holds the two undeclared segments to exactly the ten known sites. None of them
referenced `ref`. `runfiles` (catalogue + name grammar) and `paths` (layout)
remain the L1 path modules, and § 5l.7's constraints on them remain true
independently: **stdlib-only, the shape is DECLARED never inferred, `identity.stage_token`
is the one speller of `<NN>_<name>`, and `project-layout.md` § 2.6 is the
authority on the tree.**

### 5l.a What § 5l FOUND that outlives it — re-homed, not dropped

Retiring the standard retires its *answer*. It does not retire the defects the
inventory measured on the way, and one of them is live and user-facing. **These
are now `N5` in § 2's open list**, restated here because their original framing
— *"under § 5l that is `find(root, role=…)`"* — died with the section and a
reader must not go looking for it.

**① Every staged run loses its frozen atoms — ~~LIVE~~ FIXED 2026-09-17.**
`parse/engines/_sidecar.read_frozen_atoms(traj_path, label="")` needs the label
to strip the rung off an artifact stem, and **three of its four callers omit it**
(`parse/engines/molwatch.py:590`, `parse/engines/pyscf.py:626`,
`parse/engines/siesta.py:1877`; only `pyscf.py:402` passes one). So
*"Hide frozen atoms"* and `runtime_info["frozen_atoms"]` are empty for every
laddered calculation:

| artifact | no label | with label |
|---|---|---|
| `bdt.out` | `[5, 6, 7]` | `[5, 6, 7]` |
| `bdt_01_coarse.out` | `[]` | `[5, 6, 7]` |
| `bdt_01_coarse.molwatch.log` | `[]` | `[5, 6, 7]` |

The function's own comment states the mechanism and why the obvious fix is
wrong: *"THE LABEL IS REQUIRED TO STRIP THE RUNG, AND GUESSING IT IS A BUG"* —
`bdt_01_coarse.out` (rung `01_coarse` of `bdt`) and `sample_02_test.out` (an
unstaged calculation whose label reads that way) are the same shape, and guessing
handed the second one a **different calculation's sidecar**.

> **And retiring § 5l cost nothing here, which is worth recording because I
> predicted it would.** The fix § 5l proposed was *"ask by ROLE"* —
> `find(root, role=…)` — and I wrote that losing it was *"the one real cost of
> retiring"*. It was not. **The answer was already in the same file**:
> `_siesta_fdf_path_for` pairs a `.out` with its `.fdf` by trying the exact
> stem and then *"falling back to a single `*.fdf` in the same directory"*, and
> `model/parse.md` § 5.3 names that very function as the shape a companion
> lookup may legitimately take. `read_frozen_atoms` sat beside it for three
> months without the fallback.
>
> *The lesson is `feedback_read_the_code_path_before_probing`'s, in a new
> place: I reached for a retired abstraction to solve a problem its own module
> had already solved twice over. Read the neighbours before designing the
> door.* **Fixed 2026-09-17**, guarded by a prefix check and an
> ambiguity-declines rule, both mutation-tested.

**② The stage token has two readers and they disagree — latent.**
`identity.parse_stage_token` splits a filename inline, which `runfiles.py`'s own
header forbids (*"nothing composes or splits a run-file name inline"*), and the
two answer differently:

| filename | `identity.parse_stage_token` | `runfiles.parse` |
|---|---|---|
| `bdt_01_coarse_geom_optim.xyz` | stage `01_coarse_geom_optim` | `01_coarse` + role `_geom_optim.xyz` |
| `bdt_01_coarse.runwrap-…log` | no stage | `01_coarse` |

`runfiles` is right both times. Latent because the three callers
(`parse/dirs/job.py:81`, `jobset/materialize.py:394`, `:433` — re-measured
2026-09-17) feed it only deck and `.out` names, where the two agree. **The
duplication is the defect**, and deleting the second reader does not need the
standard.

**③ A phantom rung for an unstaged calculation.**
`parse_stage_token("run_01_setup.out")` returns `(1, "setup")` with no label and
`None` with it. `parse/dirs/job.py`'s `_detect_stage` omits the label, and it
feeds the sort that picks `active = sorted_outs[-1]` — *which file speaks for the
directory* in `run_status`. Fixed by ② rather than separately.

**Two decisions § 5l was holding, now standing on their own feet as `N6`:**

- **The group launcher.** `jobset/submit.py` builds `<container>/launch/<name>.run.sh`
  and `.sbatch`; `launch_dir` is spelled at `:1204` (`container / "launch"`) and
  `:1517` (`stage_dir / "launch"`). `<name>` is a **group**, not a calculation.
  `launch` is one of the two undeclared segments `GUARDED_UNDECLARED` holds, so
  **§ 5k's live guard already owns this**, and the question is § 2.6's, not
  § 5l's: does a group become a real qualifier on ④, or does the group launcher
  move to where a launcher belongs?
- **`web/blueprints/files.py`.** `_SIDECAR_SUFFIX = ".molstruct.json"` at `:260`
  against `sidecars.molstruct.SUFFIX` — re-measured 2026-09-17, still the only
  duplicated constant of its kind — plus a re-implementation of the pairing rule
  at `:271`. Its written reason is *"to keep its dependency graph narrow"*, which
  layering does not require (`sidecars` is L2, `web` L3). Either import the door
  and delete the copy, or keep the choice and add the parity test
  `process/code-audit.md` D4 then demands. **This is a code-audit item and never
  needed the standard.**

### 5l.c "What label does this deck carry?" — one fact, three readers, and where it belongs *(investigated 2026-09-17)*

**Why this section exists.** N5 ②/③ need the calculation's label to call
`runfiles.parse`, and one caller (`parse/dirs/job.py`) has none. I first
concluded that no door answered *"what calculation is this directory for?"* and
that I would have to add one. **That was wrong** — the question is answered in
three places already. The investigation is here because the ANSWER is a layer
question, not a missing function.

| # | reader | where it runs | how it reads | live? |
|---|---|---|---|---|
| 1 | `runwrap.py`'s awk, inside the generated wrapper | the **compute node**, at LAUNCH | lower-cases, strips quotes, takes `$2`, then sanitises to `[A-Za-z0-9._-]` or falls back to the basename | ✅ |
| 2 | `parse/dirs/rundir.py::openable_in` (via `parse.fdf.system_label` / `pyscf.input.job_name`) | the server, at read time | case-insensitive, allows the dotted spelling, restricts the value charset | ✅ — called at `rundir.py:109` and `:127`. *(Was `watch.py::_basename_from_fdf` / `_basename_from_py` at `:269` / `:285`; those two wrappers went with the chain on 2026-09-18, § 5c step 2.)* |
| 3 | `parse/dirs/_assembler_helpers.py::extract_system_label` | — | case-SENSITIVE, requires whitespace, accepts any non-space value | ❌ **deleted 2026-09-06**, "six helpers, zero callers" |

**Two of them disagreed**, and the one that survived is in a web blueprint.

#### The awk one is NOT irreplaceable — it is G7's surviving violation

**`execution/gpu.md` G7 already ruled on this exact class, 2026-08-23:**

> **"G7 — The value travels; the deck is not re-read for it."** *`use_gpu`
> declares `read_by = ["wrapper"]` precisely so the wrapper can be **handed**
> the value… The deck scan remains only for a caller that states nothing,
> which is not re-deriving: that path has no allocation to ask.*

**The label is the same case and was never converted.** Measured 2026-09-17:
`[item.system_label]` is already a catalogue item carrying
`anchor = "SystemLabel"` / `engine_key = "SystemLabel"`, and **exactly one row
in the whole catalogue declares `read_by = ["wrapper"]`** — `use_gpu`, G7's own
landing. `system_label` does not, which is the only reason the wrapper digs for
a value `prep` has in hand. G7's escape clause does not cover it: there is no
caller here that states nothing.

**So the earlier reasoning in this section is superseded, all of it.** I argued
in turn that the awk (a) could not be replaced because molbuilder is not
importable there — false, the env is activated 210 lines earlier; (b) had to be
lax because decks get hand-edited — false, the deck fences a `user-custom` zone
and warns against the rest; (c) was justified because the *value* can change
after prep — true but worthless, because the wrapper's own output naming is a
baked literal, so such a run is already inconsistent with itself. **The real
answer was a ruling already on the books.**

#### What the awk's comment gets right, and what it gets wrong

*Its stated reason:* it re-reads at launch rather than having the label baked in,
because the wrapper runs *"on the deck as it is at LAUNCH, after a person may
have edited it, and a cold-restart sweep keyed to the wrong label moves aside
files the engine will then not find."* **That premise still holds** — `prep`
writes, `launch` runs later, and the USER-CUSTOM zone plus the web edit-save
path exist precisely so a person may edit a deck in between. A sweep keyed to
the prep-time label would move the wrong files, and SIESTA looks up
`SystemLabel`, not the wrapper's filename. Its laxness follows from the same
premise: fdf is case-insensitive and mawk/BSD awk have no `IGNORECASE`, so
lower-casing is required, not sloppy; and the sanitiser downstream re-imposes
the charset the generator would have guaranteed.

**I claimed a structural reason here and it was FALSE — corrected 2026-09-17
after the user challenged it.** I wrote that *"molbuilder is NOT IMPORTABLE
where that code runs, so the awk cannot call a Python reader."* **Measured in a
real emitted wrapper** (`projects/claude-audit/optimization/benzene-flat/…run.sh`):
`conda activate` is at line **210**, the awk is at line **420**. The environment
is live when the label is read; Python is available. And molbuilder not being
importable in the ENGINE's env is precisely why `mb_monitor.py` and
`config_dir.py` ship beside a job — a mechanism that proves a standalone reader
*could* travel, not that it couldn't.

**So the second implementation is a CHOICE, and these are its real grounds:**

* **Blast radius, measured.** Adding a companion file has cost a production
  outage in this repo: two hand-kept lists of what travels with the monitor
  drifted, `config_dir.py` went into one and not the other, and *"every
  production run's monitor died at import, stderr to /dev/null: no [MACHINE],
  no status, no util.csv, no reports"* — found only by a bench run that happened
  to have all four files (`runwrap.py:4156-4164`). A third companion to recover
  one string is a poor trade against one line of awk.
* **It feeds shell control flow.** `$_warm_label` is consumed by a `case`
  pattern and a glob loop in the same script; a Python reader means spawning an
  interpreter to capture one word.
* ~~**They answer DIFFERENT questions, so they are allowed to differ.**~~
  **WITHDRAWN 2026-09-17, same day, user: *"i don't buy the reason for why it
  has to be forgiving and it's just a fucking pattern matching of names. what
  kind of shit argument is this?"*** — and that is correct. I argued the awk
  must be lax because it reads a possibly hand-edited deck. **molbuilder writes
  the keyword**, the deck fences a `user-custom` zone for edits and says *"Do
  not hand-edit it"* about the rest, and nobody renames `SystemLabel` to
  `System.Label` by accident. The laxness is not earned by that story; it is
  two regexes written on different days.

  **What survives is narrower and real: the VALUE can legitimately change.** A
  person may rename the job in the deck between `prep` and `launch`, and a
  sweep keyed to the baked-in old value moves the wrong files. *That* is why it
  re-reads. It says nothing about how the keyword is matched.

**Recorded so it is not "unified" later:** a future reader who sees only the
similarity will merge these and break the wrapper. The rule is the third bullet
— *the wrapper reads the deck AS LAUNCHED; the framework reads what it wrote* —
and it belongs in the awk's own comment (N5d).

**And its comment carries a claim that was false when written.** It says
*"This is the only reader of the label now"* — written 2026-09-06 at reader 3's
deletion, while reader 2 was live in `watch.py` and had been for months. Same
class as everything else the 2026-09-17 sweep found: a claim made AT a deletion,
about the thing being deleted, never checked against the rest of the tree.

#### Where the PYTHON reader belongs — `model/parse.md` § 1a settles it

> *"A reserved block in a molbuilder-generated script is not foreign.
> molbuilder writes it, into a file molbuilder generated... **A block belongs to
> its writer** — `script_emit` owns the emitting and the reading of what it
> emitted."*

We write `SystemLabel` and `JOB = "…"`. So reading them back is the **writer's**,
and `script_emit` already demonstrates the shape with five of them —
`_extract_header_text`, `_extract_provenance_dict`, `_extract_user_custom_inner`,
`_extract_atom_metadata_dict`, `_extract_bench_marks_dict`.

**This is also the retrospective explanation for reader 3.** It was deleted for
having zero callers; the deeper fact is that it was in `parse/` — the layer for
FOREIGN formats — reading something molbuilder itself had written. Re-creating
it there, which is what I was about to propose, would have rebuilt the same
mistake with a caller attached to excuse it.

#### The solution — an EXTENSION of two existing surfaces, no new module

| step | change | why it is not a patch |
|---|---|---|
| **N5a** | ~~give each emitter a label reader~~ — **WITHDRAWN: that would be a FIFTH reader.** Instead: **move the one correct fdf reader out of `transport/preflight.py`**, where it is a leftover of the deleted cross-deck comparison, to where reading an engine's format belongs | `_parse_fdf` + `_norm` is the ONLY reader in the tree that implements fdf's real keyword rule (lower-case AND strip `.`/`-`/`_`). `parse/contract.py` already reaches across a package boundary for it with an apologetic function-level import; `watch.py` and `runwrap._parse_fdf_n_atoms` each hand-rolled a narrower one |
| **N5b** | `watch.py` deletes `_FDF_SYSTEM_LABEL_RE` / `_PY_JOB_NAME_RE` and asks the moved reader; the label is then `scalars["systemlabel"]` — **no new regex anywhere** | four fdf readers become one, and the borrow in `parse/contract.py` stops crossing a package boundary |
| **N5c** | `parse/dirs/job.py` asks N5a for the label, then `runfiles.parse`; **`identity.parse_stage_token` is DELETED**, its `materialize.py` callers moving to `runfiles.parse` + a token splitter (the inverse of `identity.stage_token`, taking a TOKEN not a filename) | kills the second FILENAME reader, which is N5 ② — and ③ falls out, because the phantom rung came from parsing with no label |
| **N5d** | **apply G7 to the label — BUT IT IS NOT TWO DECLARATIONS.** ⚠ *I described this as trivial before running the review; the review says otherwise.* `read_by = ["wrapper"]` **documents** a dependency, it does not carry a value: G7 was reached *"by carrying the answer rather than by importing the catalogue into the wrapper"* — `resolve` puts `use_gpu` on `Resources`, which already travels. Measured: `render_wrappers` / `write_run_wrapper` receive `(script_path, resources, env, emit_sbatch, project_dir, machine_record)` and **not** the unsuffixed label; A8 forbids adding a loose kwarg (it exists because eleven were removed). So the label reaches the wrapper only by **a new `Resources` field** — and `Resources` is identity-free today, carrying allocation, retry, notify and `program`. **That is a design decision, not a mechanical application**, and it waits for a yes | `gpu.md` G7 + `architecture.md` A8. The alternative — prep reads the deck once with the consolidated parser and bakes a literal — needs no new field and no plumbing, and is what N5f makes possible |
| **N5f** | the remaining deck readers consolidate onto the one correct parser — **`watch.py`'s pair and `_parse_fdf_n_atoms` DONE 2026-09-17**; `_fdf_requests_gpu` **SETTLED AND DONE 2026-09-18, see below** | **eight readers of deck content measured** — four awk in the wrapper (label ×2, GPU flag, a `%block` line counter), four Python — and only `_parse_fdf` + `_norm` implements fdf's real keyword rule |

> **SETTLED 2026-09-18 — the evidence was already in the tree, on another
> row.** This said settling it *"needs SIESTA's own fdf source, which is
> not available in this checkout"*, and the table below records
> `_parse_fdf`'s first-wins as citing *"none stated"*. Both were true when
> written and neither is now: **`siesta/layout.py::check_rules` states the
> rule from libfdf's own source** — *"libfdf takes the FIRST match and
> ignores the rest (`fdf_locate` walks from the top and stops)"* — which is
> why the deck gate REFUSES a duplicate keyword instead of resolving one.
> `summarize.deck_value` and `script_emit._deck_answer` say the same. So
> three readers plus a libfdf citation say FIRST; `_fdf_requests_gpu` alone
> said last, citing `read_options.F90` — which is SIESTA's *consumer* of
> the value, not fdf's lookup. The lookup never returns the second line.
>
> **Migrated, and the launch-time twin with it.** `_GPU_TRUTHY`'s own
> comment says the Python check and the wrapper's launch-time awk are kept
> in step *"so the two rules cannot diverge (R6)"*. Converting only the
> Python half split them — a deck spelling `Diag_ELPA_GPU` took the GPU env
> at prep and CPU rank defaults at launch, which is the task-#36 OOM class.
> The awk now squashes `.`/`-`/`_`, is first-wins per keyword, and ORs the
> two keywords; six spellings measured identical across both readers.
>
> *(Process note: this row said STOPPED and the migration went in anyway,
> in `dada356a`, without updating it. The answer was right and the evidence
> was real, but a reserved decision was made silently — which is the thing
> this row existed to prevent.)*
>
> *The original disagreement, for the record* *(2026-09-17)*: `_fdf_requests_gpu` was next, and routing it
> through `_parse_fdf` would have **changed its answer**: the two readers
> disagree on what a REPEATED keyword means.
>
> | reader | repeated keyword | authority it cites |
> |---|---|---|
> | `parse/fdf.py::_parse_fdf` | **first** occurrence wins (`scalars.setdefault`) | none stated |
> | `runwrap.py::_fdf_requests_gpu` | **last** occurrence wins | *"matches SIESTA's `read_options.F90` semantics"* |
>
> Only one can be right, and the consequence is not cosmetic: a deck that
> states `Diag.ELPA.GPU` twice routes to a different environment depending on
> which reader answers. **Settling it needs SIESTA's own fdf source**, which is
> not available in this checkout — `labeleq` is called in the copy on hand but
> defined elsewhere. Recorded rather than guessed; `_fdf_requests_gpu` keeps
> its own scan until then, which is the honest state and not an oversight.
>
> *(It is also a second instance of § 6a's shape: two readers of one format,
> each stating a rule, neither checked against the other until something made
> them meet.)*
| **N5e** | `RunDirResult` gains `label`, reader named per § 5.0 | **§ 5c's, recorded not built** — the directory-level question is the door's, and N5c's call is the same one the door will make |

**What this deliberately does NOT do:** build a `label_of(directory)` door now.
That is a question *about a directory*, which § 5 says goes through
`JobDirParser`, and building it anywhere else is how a seventh consumer with its
own answer appears — the exact shape § 5l was retired for.

---

### 5l.d Reading back a deck we just wrote *(2026-09-17)*

**The question that closed this thread, and it was not the one I kept
answering.** I spent several rounds on *which reader wins when a keyword
appears twice*. The user's actual question was **why anything re-reads the deck
at all** — every value in it was known when it was written, and the layers had
already resolved each one to a single value with provenance. Reading it back to
recover what we were holding is the defect; who wins a tie is noise.

**Two different things, and I had been treating them as one:**

| | reads | verdict |
|---|---|---|
| a **cited** deck — someone's finished run from last week | `parse/contract.py`, `transport/compose.py`, `transport/citation_defaults.py` | **legitimate.** That file is the only record of what they ran; there is no description to ask |
| a deck **we just wrote, in this operation** | `runwrap._deck_label`, `_parse_fdf_n_atoms`, `_fdf_requests_gpu` | **re-derivation.** `gpu.md` G7: *"the value travels; the deck is not re-read for it"* |

**`_deck_label` is DONE, and it was mine, added the same day.** Told to bake the
label in as a literal, I implemented it by opening the deck at prep and reading
`SystemLabel` back out — the same defect as the awk it replaced, one step
earlier. The value was `task.label` the whole time. It now travels:
`prep` → `write_run_wrapper(label=…)` → `render_wrappers` → `render_run_wrapper`
→ `_cold_restart_block`, and `runwrap` reads no label from any deck.

> **A8 never forbade that parameter, which is why the first attempt went the
> wrong way.** I read A8 as *no new arguments* and designed around it. A8 says a
> door taking one of § 3's objects *"may not also name that object's FIELDS"* —
> it forbids destructuring `Resources`, not adding a parameter. `label` is not
> one of its fields. **The rule I invented to avoid was not the rule.**

**Still open, same class, each needing its own value threaded:**

* `_parse_fdf_n_atoms` reads `NumberOfAtoms` — known as `len(struct.elements)`;
* `_fdf_requests_gpu` reads `Diag.ELPA.GPU` — known as `resources.use_gpu`, and
  G7 already made that the preferred path; this is the documented fallback for
  *"a deck someone points at"*, which has no allocation. **Whether that fallback
  should exist at all is the open question** — not, as I claimed three times,
  which occurrence it picks.

**And the keyword rule is written twice more:** `siesta/layout.py::check_rules`
spells `lower().replace(".","")…` inline, and it is the same rule as
`parse/fdf._norm`. One of them should call the other.

---

### 5l.b The pattern this is the second instance of *(2026-09-17)*

§ 5l and `model/parse.md` § 5's `JobDirParser` are mirror images, and both were
surfaced by the same sweep on the same day:

| | `ref.py` | `JobDirParser` |
|---|---|---|
| specified | § 5l | `model/parse.md` § 5 |
| built | **yes — 444 lines** | no |
| adopted | **no — zero production callers** | no |
| named by a contract | **no** — only `architecture.md`'s L1 index, as a bare `ref` in a list | yes |
| outcome | **retired and deleted 2026-09-17** | still owed — § 5c |

**A parked deliverable is not free, and the two failure modes are different.**
`JobDirParser` unbuilt costs a gap a reader can see: `parse_dir()` raises, and
the Results picker guesses from filenames because there is no door to ask
(§ 5p.3p.3). `ref.py` built-and-unadopted cost something worse — it read as
*done* from the L1 index while answering nothing, so the four APIs it was meant
to replace kept growing beside it. **The lesson is the sequencing rule § 5l.6
already stated and did not follow: a step is "each step separately GREEN", and a
module with no caller is not a green step — it is an unmerged branch living in
`main`.**

---


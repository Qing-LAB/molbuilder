# molbuilder — the reuse map (task → tool)

**Role:** reference
**Domain:** *(root — the spine)*
**Companions:** [`backend-architecture.md`](?doc=backend-architecture.md) (the
same backend by *functional concern*, the paired lens to this layer index);
[`design.md`](?doc=design.md) (mission · principles · decisions — the narrative
sibling); [`README.md`](?doc=README.md) (the doc index + the rules);
[`plans/plan.md`](?doc=plans/plan.md) (open work);
[`process/package-layout.md`](?doc=process/package-layout.md) (where each file
lives); [`process/conventions.md`](?doc=process/conventions.md) (the L1/L2/L3
layering rule + the provenance header this map rests on).

> **Read this BEFORE building anything.** It is the single map of the major
> infrastructure, modules, and APIs that already exist — so we stop
> reinventing or patching around a tool we already own. It is an **index**:
> each entry gives the role, the layer, the public API entry point, and the
> **authoritative doc** to read for detail. When the linked doc and this
> summary disagree, the linked doc wins — keep this one thin and route to it.

---

## 1. The rule: find it here before you build it

Before writing a new helper, doctor, parser, launcher, config reader, or
persistence format, **find the capability in § 2 (task → tool) or § 3
(subsystem index) and reuse it.** If nothing fits, build the new thing as
**shared infrastructure** — a named module with its own doc and named
adopters — not a local patch. (This is the standing "don't reinvent wheels"
principle; the narrative rationale lives in [`design.md`](?doc=design.md).)

Two habits make the reuse rule work:

- **Read the provenance header first, never judge code dead from a glance.**
  The house style opens each code file with a `MODULE · ROLE · USED-BY`
  header naming who depends on it. It is **advisory today** (no test pins it;
  partial adoption — see [`process/conventions.md`](?doc=process/conventions.md)
  § 2), but it is still the first thing to read: trace the callers before you
  decide a function is removable. A surface glance is not evidence.
- **The two surfaces are the only top layer.** `cli.py` and `web/` (L3) call
  the *same* lower-layer verbs — never a private copy. If you are about to
  hand-roll logic in a blueprint or a CLI command, the verb you want almost
  certainly already exists one layer down. Review keeps this
  (`process/code-audit.md` § 1c (e)); § 3 is organised by that layer.

---

## 2. Task → existing tool (check here first)

| I need to… | Use | Do NOT | Detail |
|---|---|---|---|
| Check an env is present / has the right GPU · CUDA · ELPA codepath | `molbuilder envs {advise,bootstrap,clean,doctor,install,list,repair,validate}` (`envs/`); `scripts/install-env.sh bootstrap` is the shell wrapper that hands off to `envs bootstrap` | write a new doctor / checker | [`ops/installation.md`](?doc=ops/installation.md) |
| Read `molbuilder.json` (launch mode · env names · paths · the server's sections) — and a machine's facts (its queues, its activation) from its record | `runtime_config.get_launch_mode` / `get_envs` / `get_paths` / `write_config_scope`; the record through `scheduler.machine_for`, its queues through `runtime_config.get_routing` | re-parse the JSON yourself | [`configuration.md`](?doc=configuration.md) § 4–5 |
| Emit a run wrapper / `.sbatch` for a job | `runwrap.render_wrappers` / `write_run_wrapper` / `render_sbatch` | hand-write shell / sbatch | [`execution/running-a-job.md`](?doc=execution/running-a-job.md), [`execution/job-system.md`](?doc=execution/job-system.md) |
| Persist a versioned JSON artifact (`molbuilder/<name>@<major>`) | `persist.schema_major` / `check_schema_major` / `read_json` / `write_json` (atomic) | hand-roll the schema check + IO | [`execution/job-contracts.md`](?doc=execution/job-contracts.md) (data vocabulary) |
| Persist the user's *unsaved* browser edits across tab switches / reload | the **workspace's byte storage** (`lib/workspace/` → `/api/workspace-storage/*`) — format-blind opaque snapshots | auto-save to, or directly bond an edit to, the project `.xyz` / `.json` | [`web/workspace.md`](?doc=web/workspace.md) |
| Save the edited structure to a project file (choose a folder, then a name) | **`projects.molviewFiles.save("project", …)`** → `/api/structure/save` (server writes the `.xyz`+sidecar pair) | a bespoke second file stack | [`web/tabs.md` § 6](?doc=web/tabs.md) |
| Change the unit cell · vacuum · axis kinds · cell origin | **`POST /api/structure/periodicity`** (web; the ONE door, `molview.data.commitPeriodicityOp` client-side) or **`periodicity_gate.apply_edit`** (Python) | write `cell` / `cell_origin` / `vacuum` / `axis_kind` directly, or compute a box in JS | [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 6.1–6.2 |
| Ask what box a structure actually has (to draw it, or to emit it) | `struct.resolve_cell()` / `resolve_cell_origin()` — computed **views** (`expected_cell_corner` / `cell_contains_atoms` beside them) | read raw `cell` / `cell_origin` and assume `(0,0,0)`, or re-derive a bbox yourself | [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 3, § 6.1 |
| Parse an engine `.out` / sidecar into typed data | `parse.parse` / `parse.detect`; `runs.folder_answer` (a folder → what it is and holds, `model/parse.md` § 5) and `runs.run_of` (the run a file belongs to, `execution/architecture.md` § 3.2); `parse.dirs.job.run_status` (how a run dir is doing); an engine's output lines through its grammar and reading pass (`model/parse.md` § 4a) | write a bespoke parser, or a second pattern for a line the grammar already reads | [`model/parse.md`](?doc=model/parse.md); decoded-run view → [`execution/running-a-job.md`](?doc=execution/running-a-job.md) |
| Run a *set* of related jobs (stage ladder / sweep) | the `jobset/` framework + `molbuilder jobset {init,prep,plan,launch,summarize,status}` | reimplement dir isolation / sbatch chaining | [`execution/job-system.md`](?doc=execution/job-system.md) |
| Know what a **stage** is, and what may vary between two of them | [`engines/stages.md`](?doc=engines/stages.md) — a stage is molbuilder's device, not the engine's; SIESTA has no idea a deck is the second of three | invent a per-tab notion of "stage" | [`engines/stages.md`](?doc=engines/stages.md) |
| Lay out (or read) a whole **calculation directory** | [`execution/project-layout.md`](?doc=execution/project-layout.md) — the two shapes (flat / hierarchical), who writes each level, and what `prep` resolves on the target | assume one directory shape, or finish a deck on the laptop | [`execution/project-layout.md`](?doc=execution/project-layout.md) |
| Name a calculation, or decide whether a run **continues** | [`execution/run-identity.md`](?doc=execution/run-identity.md) — continuing is what the engine does when it finds warm files keyed by the id it was given | derive an id from anything a run produced | [`execution/run-identity.md`](?doc=execution/run-identity.md) |
| Benchmark GPU/CPU knobs on a target | `molbuilder jobset {prep,submit,summarize} bench` — benchmarking is `prep` whose parameters are a set (fold landed 2026-08-12; the legacy in-place `siesta-gpu` sweep was deleted 2026-08-13 -- the group itself was deleted 2026-08-17 and its one config helper is now `molbuilder jobset probe`) | reinvent the sweep / adapters | [`execution/generator.md`](?doc=execution/generator.md), [`execution/job-system.md`](?doc=execution/job-system.md) |
| Snapshot / restore / branch a run dir (with big-binary safety) | `checkpoint.Repo` + `molbuilder checkpoint {init,save,list,tag,restore,config}` — **six verbs, and there is no `branch`** (a fork is what happens when you save from a restored state, `execution/checkpointing.md` § 7.1) | shuffle files by hand | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 6 (how to drive it); [`execution/checkpointing.md`](?doc=execution/checkpointing.md) (what the history must guarantee — **read this before changing anything in `checkpoint.py`**) |
| See the whole thing done once, end to end, with a real molecule | [`execution/worked-example.md`](?doc=execution/worked-example.md) | infer the workflow from four contracts at once | [`execution/worked-example.md`](?doc=execution/worked-example.md) |
| Build a calculation on a finished run (relax → transport) | **cite the attempt** — `jobset init --slot junction=<dir>` (any directory whose files satisfy `transport-design.md` § 4.1b); `prep` composes from the citation (`transport/compose.py`).  The handoff bundle retired 2026-08-29 | copy geometry ad hoc; re-grow a bundle writer | [`archive/2026-09-01-transport-design.md`](?doc=archive/2026-09-01-transport-design.md) § 4.1; [`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 5 (the closure) |
| Build a structure (peptide / DNA / RNA / SMILES / name) | `molbuilder {peptide,dna,rna,smiles,name}` (`peptide.py` / `nucleic.py` / `smiles.py` / `pubchem.py` / `builders/backends/`) | new geometry code | [`engines/builders.md`](?doc=engines/builders.md) |
| Emit a SIESTA / PySCF input | `jobset prep`: the engine describes the deck — `siesta.input.spec_for` · `pyscf.input.spec_for`, from `SiestaConfig` / `PySCFConfig`, return a `script_emit.DeckSpec` — and `script_emit.prepare_deck` validates, renders (`render_deck`), writes and checks it | template strings; **or a CLI verb** — there is no `molbuilder fdf`, no `molbuilder run` (both deleted 2026-08-11) and, since 2026-09-17, **no engine verb at all**: `molbuilder pyscf` was the last, and the "one recorded exception" this cell used to note is closed ([`process/conventions.md`](?doc=process/conventions.md) § 3).  A deck is rendered by `jobset prep`, on the machine that will run it | [`engines/siesta.md`](?doc=engines/siesta.md) § 1.1 · [`engines/pyscf.md`](?doc=engines/pyscf.md) |
| Carry a calculation's **parameters** from a browser to the machine that runs it | `<label>.template.toml` — one TOML file, every parameter with its value and a `kind` saying which layer owns it | invent a second description file, or finish the deck early | [`engines/template.md`](?doc=engines/template.md) |
| Decide a calculation's charge and spin | `electronic_state.electronic_state` — the one answer for the form, the checks and every deck | reading `cfg.net_charge` / the spin fields raw, a per-engine rule | [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) § 2a |
| Validate chemistry / open-shell / pseudos before emit | `validation/` (the one `validate()` pass; `check_electronic_state` for the charge and spin), `chemistry.analyze_structure` (the facts), `pseudos.check_coverage` | ad-hoc checks | [`science/validation.md`](?doc=science/validation.md), [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md), [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) |
| Warn the user about something scientific | return an `Issue` from a validator — it reaches the web panel *and* the CLI report | `warnings.warn` (server stderr only — no web user ever sees it) | [`science/validation.md`](?doc=science/validation.md) § 4.1 (R5) |
| Put a validation finding on screen | **`molbuilder.validationFindings.render(issues, {panel, formScope})`** (`lib/validation-findings.js`) | a per-tab renderer (there were four, each losing findings) | [`science/validation.md`](?doc=science/validation.md) § 4.1 (R2), [`web/ui-contract.md`](?doc=web/ui-contract.md) § 5.1 |
| Send a structure to a validating / emitting endpoint | **`molview.data.exportFile()`** — coordinates + labels + cell, in the server's words, for the frame on screen, in ONE read | a page-local `state.xyz` mirror; a server-side disk re-read; **or a body that repeats the labels and the cell beside the structure** (that is a second read, however fresh) | [`web/molview.md`](?doc=web/molview.md) § 9.3a; [`science/validation.md`](?doc=science/validation.md) § 4.1 (F1/F2) |
| Detect host capabilities (cores / GPU / env-for-category) | `diagnostics.get_capabilities().env_for_category(...)` | probe by hand | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) (runtime resolution) |

---

## 3. Subsystem index (by layer)

**Layer key.** **L1** core types (import nothing above them) · **L2** domain
verbs (may import L1) · **L3** surfaces (cli / web; may import both).
**Which is which is read, never judged:** L1 is exactly the index below; L3
is `cli` and `web`, and `__main__`, the `python -m molbuilder` entry, which
calls `cli`; every other module is L2. **The package root**
(`molbuilder/__init__.py`) loads before every module inside it, so it imports
only L1 — `structure` — and re-exports no builder: each caller imports the
builder it uses from its own module *(the layer review, 2026-09-27: the root's
re-exported peptide and nucleic builders loaded under every core module).* This is
the load-bearing invariant — it is what stops the registry/abstraction tangle
from growing back — and **review keeps it** (`process/code-audit.md` § 1c (e)),
against the index below, *(user, 2026-09-27: "a static code review problem")*.
What it prevents at run time shows up by itself: an upward import at a module's
top fails every import of that module, and a file that ships beside a job and
reaches into molbuilder fails there (`tests/test_monitor_watches_a_live_run_e2e.py`
runs it). The proof that the current code complies is a workflow that finishes
end to end, its monitor's report included (`process/code-audit.md` § 1c (e)).

```mermaid
flowchart TB
  subgraph L3["L3 · surfaces — the only top layer"]
    CLI["cli.py"]
    WEB["web/ (Flask blueprints)"]
  end
  subgraph L2["L2 · domain verbs"]
    ENG["siesta/ · pyscf/ · transport/ · builders/"]
    EXE["jobset/ · bench/ · runwrap · runtime_config · envs/"]
    RW["parse/ · runs · sidecars/ · script_emit · validation/"]
  end
  subgraph L1["L1 · things, and how each is written"]
    GEO["geometry — structure · cell · selection · periodicity_gate<br/>chemistry · residues · engine_atom_index"]
    RES["results — frame · trajectory_log · runtime_info · issues<br/>report_fields"]
    JOB["the job as described — task · calcdirs · config · identity<br/>runfiles · wrapper_log · paths · warmfiles · annotations_fdf<br/>deck_record · atom_permutation"]
    MACH["the machine — scheduler (records · queues · admission<br/>· the quantities a job asks for)"]
    INF["infrastructure — persist · config_dir · constants · units<br/>pipeline_log · references · reload_protocol · serve_daemon"]
  end
  L3 -->|calls the same verbs| L2
  L2 -->|reads/writes| L1
```

**The L1 index, grouped by the object each module owns.** This is the one list
of L1 modules — the map an import is reviewed against — so it states no count.
*(Until 2026-09-27 `tests/test_layering.py` held a second copy, and a test
compared the two.)* The diagram above draws the same set: a picture that
disagrees with the index is how *"which layer does this go in?"* becomes a
guess.

| the object | modules | what they own |
|---|---|---|
| **geometry** | `structure` · `cell` · `selection` · `periodicity_gate` · `chemistry` · `residues` · `engine_atom_index` | atoms and positions, the cell, an atom selection, chemical facts, and how an atom is numbered for a given engine |
| **results** | `frame` · `trajectory_log` · `runtime_info` · `issues` · `report_fields` | a per-step physics record, the `.molwatch.log` format, the runtime facts a run reports, a validation finding, **what a run's report may carry and which runs can state each field** (`engines/stages.md` § 6.9 — L1 and stdlib because it travels inside `mb_monitor.pyz` with the monitor, and `task` checks a description against it) |
| **the job as described** | `task` · `calcdirs` · `config` · `identity` · `runfiles` · `wrapper_log` · `paths` · `warmfiles` · `annotations_fdf` · `deck_record` · `atom_permutation` | `task.json`, **a directory's own account of where it sits** (`calcdir.json` — `project-layout.md` § 1.4a, L1 for `task`'s reason: a calculation root needs no record because its description's `shape` already answers, read through `read_task` by the run door, `runs.place_of`), the engine-knob dataclasses, how a run id is written, **the run-file name grammar and the catalogue of what molbuilder writes**, **the wrapper's session log — its lines and their one reader, which travels beside the job**, **the two layouts and where every file sits in them, with the search for each name it composes** (`project-layout.md` § 4.5 — was `jobset/shape.py` on floor 4 until 2026-09-08), the warm-file rules, the fdf annotation strategies, **the molbuilder blocks a deck carries — their grammar and their one JSON reader** (`deck_record`; `script_emit` writes them), **the atom-permutation record — its class and its one reader** (`atom_permutation`; the sort writes it) — the last two stdlib (and numpy) and L1 because a job reads them beside itself (`engines/vibration.md` § 5.5) |
| **the machine** | `scheduler` | what a machine offers and what a job may ask of it — records, queues, admission, placement, emission, and **the quantities a job asks for and every dialect each is written in** (`quantities.py`) |
| **infrastructure** | `persist` · `config_dir` · `constants` · `units` · `pipeline_log` · `references` · `reload_protocol` · `serve_daemon` | versioned documents, the one per-user config directory, **the physical constants** and the words a unit may be written in, the prep pipeline's record, the bibliography, the two constants the supervisor and its child agree on — and the supervisor itself (daemon, pidfile, log roll), L1 because it must never import the application it restarts |

> **`constants` sits lower than everything, because it imports nothing at
> all** — which is the point of it. The Bohr radius was written out eight
> times in three different values, and two `.XV` readers using two of them
> gave the same file coordinates 4e-7 apart. A number every layer may reach
> for cannot itself reach anything, or asking for one would drag a
> dependency into the layer that asked.
>
> **`units` sits beside it and imports only it**, and holds the other half:
> the *dialects* — which words an energy, a length or a temperature may be
> written in, and the one door that turns one into a number. Same argument
> `scheduler/quantities.py` makes for a wall time and a memory size, and it
> is a codec on a basic unit for the same reason.

**The rule these two carry, and it is one rule.** A physical constant has one
home and a unit word has one vocabulary, because the cost of getting either
wrong is paid somewhere else entirely — a wrong factor is invisible in the
result and wrong by a fixed ratio in every number after it.

| | |
|---|---|
| **A value** | lives in `constants`, and is DERIVED from a sibling wherever it can be (`RYDBERG_EV = HARTREE_EV / 2`), so two spellings cannot drift. Where two conventions genuinely differ, the name carries which one — `HARTREE_BOHR_EV_ANGSTROM_ASE`, because the CODATA-derived value is 0.36 ppm away and force numbers must agree with the thresholds drawn over them |
| **A unit word** | lives in `units`, in the table for its quantity. No reader keeps its own word list; that is where a word goes missing, and a missing word is not an error but a number returned in the wrong unit |
| **What a MISSING unit means** | belongs to the reader, because it is a fact about the format: a bare energy in an `.fdf` is Ry, one in a tbtrans contour block is eV. The reader passes its `default=`; passing none means a bare value is refused |
| **An unknown word** | is refused, never passed through |
| **Bare arithmetic** — `x * BOHR_ANGSTROM`, where the unit is fixed and known and no word is being read | fine, and the common case. The rule is about where the NUMBER came from, not about wrapping every multiplication: `units` is for reading a unit WORD out of a file |

**A second spelling is legal in exactly two places, and each has one
answer.** These are mechanisms, not exceptions — the value still comes from
the one home:

* a **generated standalone script**, which runs where molbuilder is not
  installed → *interpolate* the value in from `constants`
  (`siesta/makov_payne.py` is the exemplar: it interpolates
  `HARTREE_EV` and `BOHR_ANGSTROM` in by name);
* the **browser**, which cannot import Python. A value is **served**, table
  or scalar — [`process/code-audit.md` § 3.6](?doc=process/code-audit.md)'s
  ruling (*"shipping a periodic table into JavaScript is the same mistake
  with a longer commute"*) — in the reply the page already fetches:
  `web/blueprints/modify.py` serves the lattice parameters, and
  `/api/spectra/load` carries the three constants the spectrum page converts
  with (`web/blueprints/spectra.py::_page_constants` — Hartree in eV and in
  kcal/mol, kelvin per cm⁻¹), so the browser holds no copy and nothing needs
  pinning *(user, 2026-09-27: one source per fact. Until then four were
  copied into `lib/spectra/core.js` and held equal by a test that read the
  file, which `testing.md` gives no place; Boltzmann in Eh/K went on
  2026-09-28, when the thermochemistry panel stopped deriving its reference
  energy and read the file's)*.

*(A third — a class body pasted into a script by `inspect.getsource`, its
literals pinned to `constants` by a test — went on 2026-10-05: the class,
`MolwatchEmitter`, travels beside the script as its own file and imports
`constants` like any module, [`engines/pyscf.md`](?doc=engines/pyscf.md) § 3.)*

A value enters `constants` **in the same change that routes its call
sites**, or not at all — three did not, on 2026-09-21, and the diff was
read; a second home standing beside an unrouted first one is the thing
this is meant to prevent.

Anywhere else, import it. There is no lint: `082ba979` retired
`test_one_home_for_a_constant.py` with the reasoning that *"the rule belongs
in the document that owns the concept, and the drift is caught by reading the
diff"* — this table is that rule.

Reading the table is how you answer *"where does this new function go?"* —
find the object it acts on, and that is the module. A function that acts on a
duration goes where durations live (`scheduler/quantities.py`), not where it
was first needed; `design.md` records what it cost the one time that rule was
not applied.

The four core types (`Structure`, `Frame`, `Config`, `Issue`) are the wire
between subsystems: construction emits a `Structure`; validation reads a
`Structure`+`Config` and returns `List[Issue]`; the engine emitters render a
job from `Structure`+`Config`; data management owns the round-trip of all of
them to and from disk. (For the same backend seen through the *functional-
concern* lens — data · construction · validation · execution, and where those
concerns leak into each other — see
[`backend-architecture.md`](?doc=backend-architecture.md).)

### Execution & scheduling

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `jobset/` | L2 | engine-agnostic **staged execution**: a set of related jobs sharing a package | `prep_stage` (the one prep entry, both doors — the five steps, `prep_calculation` and its tail `prep_jobset`, take only its answer); `plan_launch` → `send_launch` (the one launch entry: the plan, shown and asked, then sent as shown); `jobset_status`; `render_plan`; `JobSet.write` / `load`; CLI `molbuilder jobset {init,prep,launch,status,summarize,probe,machines,migrate}` — *the chaining producers (`stages_to_jobset` / `sweep_to_jobset`) died in the 2026-08-12 fold* | [`execution/job-system.md`](?doc=execution/job-system.md) |
| `bench/` | L2 | the two library modules the jobset sweep uses: the machine-probed grid (`sweep_grid`) and the `bench-result@1` reader (the legacy `siesta-gpu` stack was deleted 2026-08-13) | `sweep_grid` (shared grid) — the sweep itself is `molbuilder jobset prep bench` | [`execution/generator.md`](?doc=execution/generator.md), [`execution/job-system.md`](?doc=execution/job-system.md) |
| `runwrap` | L2 | **launcher** emitter: `.run.sh` + `.sbatch` (env activation, MPI/OMP, memory, GPU pinning), and the bundles that travel beside a job — the monitor's, a finish (`mb_vibration.pyz`, a SIESTA force-constant job's own last step, `engines/vibration.md` § 5.5), and the code a PySCF script imports (`mb_pyscf.pyz`, `engines/pyscf.md` § 3) | `render_wrappers`, `write_run_wrapper`, `render_sbatch`; `monitor_bundle`, `vibration_bundle`, `pyscf_bundle` | [`execution/running-a-job.md`](?doc=execution/running-a-job.md), [`execution/job-system.md`](?doc=execution/job-system.md) |
| `runtime_config` | L2 | reader for `molbuilder.json` (launch / envs / paths / the server's sections), and the door to a record's queues | `get_launch_mode`, `get_envs`, `get_paths`, `get_routing`, `write_config_scope` | [`configuration.md`](?doc=configuration.md) § 4 |
| `diagnostics` | L2 | host capability detection + env-for-category routing | `get_capabilities().env_for_category(...)` | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) |
| `monitor` | L2 | stdlib-only progress/utilization sampler for every engine, shipped next to each job in one file with the readers it reads through (`mb_monitor.pyz`, `runwrap.MONITOR_BUNDLE`) | `python mb_monitor.pyz …` (watch a run); `python mb_monitor.pyz ending OUTPUT [QUESTION]` (how it ended — the wrapper's `_mb_ending`); `molbuilder monitor` | [`execution/run-reports.md`](?doc=execution/run-reports.md) § 2–2.3 (what it reads and says), [`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 4.1 |

**The start-here map for *running* a molbuilder-generated job** on any target
(single-task everywhere · JobSet from the CLI · the browser job system as the
target) is [`execution/overview.md`](?doc=execution/overview.md) — the
current → target status matrix.

### Persistence, parsing, data exchange

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `persist` | L1 | shared **versioned-doc** schema check + atomic JSON IO | `schema_major`, `check_schema_major`, `read_json`, `write_json` | [`execution/job-contracts.md`](?doc=execution/job-contracts.md) |
| `parse/` | L2 | unified **read stack**: FileParsers → typed `ParseResult`; under the engine parsers, stdlib grammar tables and reading passes that travel with the monitor | `parse.{detect,parse}`; `parse.dirs.job.run_status`; `parse.engines._run_ending.ending_of` | [`model/parse.md`](?doc=model/parse.md) |
| `runs` | L2 | **the run a file belongs to, and its files** — what a folder is in its calculation, the run it speaks for, what each file is (the catalogue's row, read back with the run's label), and the directory door the Results tab asks | `runs.{place_of,run_of,about,run_answer,openable,folder_answer}` | [`execution/architecture.md`](?doc=execution/architecture.md) § 3.2; [`model/parse.md`](?doc=model/parse.md) § 5 |
| `sidecars/`, `script_emit` | L2 | write-side JSON sidecars + the reserved-block emitter | `sidecars.{to_dict,save,load,apply_to_structure}`; `script_emit.emit_*` | sidecar → [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md); blocks → [`execution/job-contracts.md`](?doc=execution/job-contracts.md) |
| `config/` | L1 | the engine-knob **dataclasses** (`SiestaConfig` / `PySCFConfig`) — the lingua franca | `config.siesta.SiestaConfig`, `config.pyscf.PySCFConfig` | [`engines/`](?doc=engines/overview.md); the JS form, built from the catalogue → [`web/form-schema.md`](?doc=web/form-schema.md) |

### Safety, checkpoints, validation

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `checkpoint` | L2 | git-backed **snapshot/restore of a whole calculation folder**; files over a size limit are stored beside git in a content-named archive (safety-critical) | `Repo.{init,save,restore,status,states,standing_at,resolve,tag,tags,classification,calculation}`; CLI `molbuilder checkpoint …`. **No `branch`** — a fork is what happens when you save from a restored state | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 6 |
| `validation/` | L2 | scientific-correctness analyzers + the per-engine `validate()` pass | `validation.validate(struct, cfg, prior=…)` (one gate per engine) | [`science/validation.md`](?doc=science/validation.md), [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) |
| `pseudos` | L2 | PSML pseudopotential parse + coverage/version checks (C1–C6) | `pseudos.check_coverage` | [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) |
| `chemistry`, `residues` | L1 | structure analysis (the metals, charge, residues) | `chemistry.analyze_structure` (→ `ChemistryAnalysis`, facts) | [`model/chemistry.md`](?doc=model/chemistry.md), [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) |
| `electronic_state` | L1 | the charge and spin a calculation carries — one class, every engine and kind | `electronic_state(struct, cfg, kind=)` (→ `ElectronicState`) | [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) § 2a |

### Environments, engines, builders

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `envs/` | L2 | the **environments toolkit** (presence + verify-cmd + GPU / CUDA / ELPA readiness) | `molbuilder envs {advise,bootstrap,clean,doctor,install,list,repair,validate}` | [`ops/installation.md`](?doc=ops/installation.md); NEVER build a new doctor |
| `siesta/`, `pyscf/` | L2 | per-engine input emitters + stage rendering | `siesta.input.spec_for` · `pyscf.input.spec_for` → a `script_emit.DeckSpec`, which `jobset prep` renders, writes and checks (`script_emit.prepare_deck`); `molbuilder.siesta` exports `SiestaConfig`, `copy_pseudopotentials`, `find_psml`, and `molbuilder.pyscf` exports `PySCFConfig` | [`engines/siesta.md`](?doc=engines/siesta.md), [`engines/pyscf.md`](?doc=engines/pyscf.md) |
| `builders/`, `peptide/`, `nucleic`, `smiles`, `pubchem` | L2 | structure synthesis | `build_peptide` / `build_dna` / `build_rna` / `build_from_smiles` / `build_from_name` | [`engines/builders.md`](?doc=engines/builders.md) |
| `transport/` | L2 | TranSIESTA multi-run workflow — the composite: one citation → five rungs | `molbuilder jobset …` (no calculation-kind verb since 2026-09-17) | [`engines/transport.md`](?doc=engines/transport.md) |

### Core types (L1) & surfaces (L3)

- **L1 core types**: `structure`, `frame`, `issues`, `selection`,
  `runtime_info`, `trajectory_log` — the data model.
  See [`model/overview.md`](?doc=model/overview.md) and
  [`model/structure.md`](?doc=model/structure.md).
- **L3 surfaces**: `cli` (`molbuilder …`; the thin-shell-over-the-web-API
  doctrine + the full command catalogue →
  [`process/conventions.md`](?doc=process/conventions.md) § 3) and `web`
  (Flask blueprints → [`web/web-api.md`](?doc=web/web-api.md); the whole front
  end → [`web/overview.md`](?doc=web/overview.md)).

---

## 4. Persisted artifacts & schemas

The concentrated registry of on-disk names + the `molbuilder/<name>@<major>`
schema strings + the config↔scheduler parameter vocabulary is the **data
vocabulary** in [`execution/job-contracts.md`](?doc=execution/job-contracts.md).
New persisted artifacts MUST use `persist.check_schema_major` and be registered
there. The structure save file itself (`.molstruct.json`, its envelope and
schema versions v3–v6) is [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md).

---

## 5. Where the deeper design lives

This index is deliberately thin — it routes you to the authoritative doc.

- **The narrative design** — mission, the L1/L2/L3 architecture, the design
  principles, the anti-patterns we refuse, and the decisions index — is
  [`design.md`](?doc=design.md), the narrative spine sibling.
- **The concern lens** — the same backend by *functional concern* (data ·
  construction · validation · execution), which concern owns each module, and
  where the concerns leak into each other — is
  [`backend-architecture.md`](?doc=backend-architecture.md), the companion to
  this layer index.
- **The execution domain's internal shape** — which of its floors owns which
  decision, the routes that cross them, and the objects that travel between
  them — is [`execution/architecture.md`](?doc=execution/architecture.md). It is
  a **finer** grouping than the L1/L2/L3 index above: `jobset` is one import
  tier here and spans four floors there.
- **The domain docs are the authoritative per-subsystem source** for every row
  above: [`model/`](?doc=model/overview.md) (the L1 data model),
  [`science/`](?doc=science/overview.md) (correctness),
  [`engines/`](?doc=engines/overview.md) (the emitters + builders),
  [`execution/`](?doc=execution/overview.md) (running jobs),
  [`web/`](?doc=web/overview.md) (the front end + web API),
  [`ops/`](?doc=ops/installation.md) (install + serve),
  [`process/`](?doc=process/package-layout.md) (conventions · testing · audit ·
  package layout).
- **The forward plan** — every open feature/backend workstream, including the
  execution↔engine decoupling items and the front-end ESM conversions — is
  [`plans/plan.md`](?doc=plans/plan.md). Closed decisions live in [`design.md`](?doc=design.md).

Keep this map in sync when a **major** subsystem or public entry point is
added; per-detail changes belong in the linked docs, not here.

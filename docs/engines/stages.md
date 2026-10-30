# Stages — a named parameter set over one system, and the file that describes it

**Role:** contract
**Domain:** engines
**Companions:** [`engines/tuning.md`](?doc=engines/tuning.md) — what *values* a
stage should carry and why (this doc says what a stage *is*, never what to put in
one); [`engines/siesta.md`](?doc=engines/siesta.md) +
[`engines/pyscf.md`](?doc=engines/pyscf.md) — the emitters that render an
effective config; [`execution/run-identity.md`](?doc=execution/run-identity.md) —
the id every stage in a folder shares, and the engine parameters that decide
whether a stage continues; [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
— the run directory the decks land in and the persisted-artifact registry;
[`engines/template.md`](?doc=engines/template.md) — the file the effective config
is resolved *from*, and the format every engine's parameters share;
[`archive/2026-08-19-staged-runs-implementation-plan.md`](?doc=archive/2026-08-19-staged-runs-implementation-plan.md)
— the plan this contract was built from, kept as a record; the open work is in
the one plan, [`plans/plan.md`](?doc=plans/plan.md).

**Status: landed.** This document was written first and the code built to it,
the way `web/spectrumchart.md` and `web/vibrationview.md` were: `task.json`
and `molbuilder describe` shipped 2026-08-11 (plan step 2), and `prep`'s five
steps — the effective config resolved from the template ⊕ a stage's
`overrides` and rendered per stage on the target — shipped 2026-08-12
(step 4). *(This line read "proposed, not built — nothing in `SiestaConfig`
matches it yet" until 2026-08-12.)* Remaining work stays in the plan, not
here ([`process/conventions.md`](?doc=process/conventions.md)'s R3 — *status
lives in the roadmap, never in a contract*; the `R1`–`R4` used inside § 4 below
are that section's own, and unrelated).

**This contract owns:** what a stage is, which fields are a stage's and which are
the shared schema's, how an effective config is formed, where a field
lands, and the shape of `task.json`.

---

## 1. A stage is ours, not the engine's

**No engine has a concept of a stage.** SIESTA reads a `.fdf`; PySCF runs a
`.py`. Neither knows the file it was handed is the second of three, or that
anything preceded it. The word exists only inside molbuilder:

> **A stage is a named set of the parameters a mission tunes, laid over the
> shared description of the system it does not.**

The base is *what the system is*. A stage is *how we are approaching it this
time*.

**Scope: one deck per stage, for every engine.** A three-stage calculation is
three decks and three jobs, whichever engine renders them.

> **The open question here is closed** *(2026-08-18, user)*. This paragraph used
> to say one deck per stage was *"SIESTA's shape"*, describe PySCF's staged
> relaxation as a loop inside one Python process writing a single unified log, and
> leave the choice between that and genuinely separate files explicitly undecided:
> *"that decision is not made here."* It is separate files, for both engines, and
> § 1.1a is where it is decided and why.

**A stage resolves completely at generate time.** What comes out is an ordinary,
complete engine input that does not need molbuilder to be interpreted and does
not refer to a stage it follows.

Precisely: the stage name survives in the **filename**, `<label>_<NN>_<name>.fdf`, as a
label — and nothing has to interpret it to run the file. The deck's *content*
carries no stage marker at all. Anything that would require a downstream reader
to understand the word "stage" in order to act correctly is outside this
contract.

### 1.1 No engine config carries a stage list

*Stated 2026-08-07 (user), because the shipped code does the opposite and § 4
only implied it.*

**An engine config is one parameter set.** `SiestaConfig` describes a single
calculation — a mesh cutoff, a basis, one relaxation tolerance. It has no
`stages` field, and the emitter that reads it never learns the word.

*(Amended 2026-08-17: the rule is **no AUTHORED stage list**. A structure
`prep` derives from the resolved description and hands to the emitter one step
later is not a declaration — it cannot disagree with the description, which is
what this rule protects. § 1.1a states the test: can a person put a value there
that the description does not say?)*

**The stage list lives in `task.json`, and nowhere else** (§ 6). It is not a
field of an engine's config, because a stage is not a property of a calculation
— it is a record of the user's intention to tune some parameters across a
sequence of them.

> **This was the sentence the shipped code contradicted, until 2026-08-07.**
> `SiestaConfig.stages` was a `List[SiestaStageSpec]`, so the engine config
> carried a list of stages, and § 4's *"the effective config is an ordinary
> instance of the engine's config dataclass"* could not be true of it:
> resolving a stage would produce a config that still contained the whole
> ladder. `SiestaStageSpec` was therefore **removed**, not reshaped, along
> with the field, its default factory, its validator and its parser.
> `Task.stages` (`molbuilder/task.py`) is the model, and it is
> engine-agnostic by construction. Each engine's shipped ladder is built
> from it by the one ladder rule, `jobset/ladder.py::engine_ladder`.
>
> PySCF was a deliberate exception **for now**: its ladder runs inside one
> process, so its stage list has a second life as engine behaviour. It was left
> alone until the SIESTA path works.
>
> **That gate is met, and the exception is closed** *(2026-08-17, user
> decision)* — see § 1.1a. The rule above now holds for **both** engines with no
> exception, which is what 2026-08-07 was reaching for: it deleted the SIESTA
> engine config's stage list, and this deletes PySCF's. Closing it is completing
> that direction, not reversing it.
>
> **And the second half of the exception closed on 2026-08-18**: the loop that
> gave the stage list its *"second life as engine behaviour"* is retired, so a
> PySCF ladder is N decks and N jobs like any other (§ 1.1a). There is no longer
> a sense in which PySCF's stages differ from SIESTA's.

### 1.1a Closing the PySCF exception — the ladder is declared once, and runs like SIESTA's

*Decided 2026-08-17 (user). § 1.1's exception named its own gate — "until the
SIESTA path works" — and `prep`'s five steps now run over SIESTA end to end.*

**Two things were conflated, and only one of them is PySCF's difference.**

| | SIESTA | PySCF |
|---|---|---|
| **where the ladder is declared** | `task.json` | `task.json` — *was `PySCFConfig.stages`* |
| **how the ladder executes** | N decks, N jobs, a person looks between them | **the same** — *was one deck, one job, a loop inside the process* |

The first row was never a consequence of the second. A ladder can be *declared*
in one place and *executed* in whichever shape the engine requires. Keeping the
declaration in the engine config meant a PySCF description and a SIESTA
description meant different things, and the workflow could not treat them alike.

#### And the second row went the same way *(decided 2026-08-18, user)*

**A PySCF ladder is N decks and N jobs, exactly as SIESTA's is.** The in-script
loop over rungs is retired.

**Why: a ladder exists so that somebody looks between the rungs.** That is the
whole reason stages do not chain — *"a run that continues on its own can spend a
week refining a geometry you would have rejected in a minute"*
([`project-layout.md § 1.6`](?doc=execution/project-layout.md)). Looking between
rungs requires a rung to have *ended*, and a single process running every rung
ends once, at the end. Everything the workflow offers between stages — open the
next run, say which run it continues from, read what happened, redo one rung
with different numbers — is per-job machinery, and an engine whose whole ladder is
one job can reach none of it.

**It works because PySCF already has the state a rung hands to the next one.** The
two things a relaxation must carry are the geometry and the converged density, and
PySCF writes both: `<JOB>_optimized.xyz` and `<JOB>.chk`, already declared in its
warm-file vocabulary ([`job-contracts.md § 4.2a`](?doc=execution/job-contracts.md))
and already read by the generated script, which prefers the optimized geometry over
the literal one and loads the checkpoint as its initial guess when it is there.
That is the same pair SIESTA carries as `.XV` and `.DM`. Nothing new has to be
invented for a rung to be able to end.

**What it costs, plainly.** Each rung starts Python again, rebuilds the molecule
and reads the checkpoint back from disk instead of keeping it in memory. That is
wrong when the rungs are many and each is seconds long, where the restart would
dominate what it is measuring. It is right when a rung is a real piece of work —
which is when a ladder is worth having at all, and the case this framework is
built for.

**Five things follow, and they are the whole of the change. All five are in
place; each names the file that holds it, so the claim is checkable.**

1. **The stage token reaches the deck's name, the engine's log and the trajectory
   log** — the same three names it suffixes for SIESTA — so two rungs cannot write
   to one file. It is a render *argument*, never a config field, so the emitter
   never learns the word *stage* (`pyscf/input.py`, `spec_for(..., names=)`, the stage's names).
2. **The `JOB` literal stays unsuffixed**, exactly as `SystemLabel` does and for
   the same reason: the engine finds the previous rung's files by that name, so a
   name that changed per rung would hide them (§ 1, decision 26).
3. **`restart` is both engines'.** One field
   ([`run-identity.md § 4`](?doc=execution/run-identity.md) rule 3) serves both:
   `continue` by default, so a rung reads what the rung before it left, and
   `clean` when somebody says so. It expands into three declared keywords for
   SIESTA and into generated control flow for PySCF, and no shipped ladder sets
   it — a rung's position says nothing about whether there is anything to
   continue from.
4. **PySCF's warm-file declaration carries the geometry.** `<JOB>_optimized.xyz`
   joins `.chk` as `carry = "when-continuing"` (`pyscf/warm-files.toml`). The
   geometry is what a rung hands the next one and the density is the other
   half — the same pair SIESTA carries as `.XV` and `.DM`. **geomeTRIC's
   trajectory is not warm state, and is not carried** *(plan § 5w K10,
   2026-10-01; the M11 review's PO-C16)*: geomeTRIC rewrites it from the first
   step of every run (geomeTRIC 1.1.1 `optimize.py`, `progress.write(xyzout)`
   at each step; `Molecule.write(append=False)`), so a copied one is overwritten
   before anything reads it — and each rung's carries the rung's token, which by `job-contracts.md`
   § 2.2a's rule never crosses rungs. **Nor is its scratch folder**, `<prefix>.tmp`: geomeTRIC
   makes it and PySCF's engine never writes into it — it computes in memory
   and has no `read_result` — and nothing reads it back, because the decks
   ask for no Hessian (geomeTRIC 1.1.1 `engine.py` `calc`, `normal_modes.py`
   `calc_cartesian_hessian`). Their rows left the rules file, with a third,
   `_geom_optim.tmp`, that named a file nothing writes; the trajectory's ROLE
   stays declared where it is written (`runfiles.ROLE_GEOM_TRAJ`), which
   is how its readers find it.
5. **`PySCFConfig.stages` and the in-script loop are gone.** The field outlived
   `SiestaConfig.stages` on the strength of one reader — the `for STAGE in
   STAGES:` loop that made the list engine BEHAVIOUR rather than a rival
   declaration. With the loop retired the field had no reader, and both went;
   `tests/test_pyscf_stages.py` asserts their absence directly.

#### What a PySCF rung varies

A `Stage` (`task.py`) carries `name` and `overrides`, and § 2 says
`overrides` may name **any field of the shared schema** except the few bound to
the whole calculation or fixed by the rung (below). So a PySCF rung is
declared exactly as a SIESTA one is:

```
Stage(name="coarse",
      overrides={"scf_conv_tol": 1e-7, "geom_gmax": 2e-3,
                 "geom_max_steps": 50})
```

*(No `restart`: the shipped ladders set none. Every rung takes the default,
`continue`, and a rung that should start over says so in its overrides,
`"overrides": {"restart": "clean"}` (§ 2) —
[`run-identity.md § 4`](?doc=execution/run-identity.md) rule 3.)*

**What that required was vocabulary, not a new shape** — a knob is only a legal
override if it is an item of the shared schema, and PySCF's convergence knobs
were not. They are:

| catalogue item | its `engine_key` |
|---|---|
| **`scf_conv_tol`** | `mf.conv_tol` |
| `geom_gmax` | `geomeTRIC convergence_gmax` |
| `geom_grms` | `geomeTRIC convergence_grms` |
| `geom_dmax` | `geomeTRIC convergence_dmax` |
| `geom_drms` | `geomeTRIC convergence_drms` |
| `geom_etol` | `geomeTRIC convergence_energy` |
| `geom_max_steps` | `geomeTRIC maxsteps` |
| `on_nonconvergence` | *(none — `kind = "produce"`)*, because it **is** generated control flow rather than a keyword |
| `geom_continue_retries` | *(none — same reason)*, and meaningless without the one above it |

All carry `engines = ["pyscf"]` and `group = "stage"`, and their per-tier values
are [`tuning.md` § 2.4 and § 2.5](?doc=engines/tuning.md)'s — stated there,
written down once in `PYSCF_STAGE_PRESETS`, and checked against the table on
every run.

**`scf_conv_tol` is one item, and that is the finding that decided the rest.**
The engine config used to declare an SCF tolerance of its own beside the
catalogue's, the two agreeing only by the coincidence of both naming
`mf.conv_tol`. That is the drift § 1.1 exists to prevent, sitting inside the
exception § 1.1 granted.

**None of them gets a `--flag`.** They are set per rung, in `task.json` — and
no config field has generated a command-line option since the dataclass →
click bridge was deleted (2026-09-17). That is not a new
policy: the flat `--geom-max-steps` family was **deliberately retired** when
these knobs became per-stage, and a catalogue row that regenerated them would
have undone that.

#### Why `group = "stage"` is the right home

§ 1.3's mechanism — *the default selection is a group each engine declares* — is
already live in the catalogue and already per-engine. Before this landed SIESTA
declared 11 items in `group = "stage"` and PySCF declared 3, so PySCF's group
described about a third of its own ladder. **It is 30 and 12 now** (the
vibration kind's six electronic-structure-probe selectors are not in it —
nothing steps them per stage; [`engines/vibration.md`](?doc=engines/vibration.md)
§ 3.1), and the same UI, the same `varies` machinery and the same resolver
serve both engines.

**No new mechanism is introduced by this decision.** The catalogue is the
master, `overrides` names schema fields, a group declares the default selection,
`prep` resolves — every part is the one already in use for SIESTA.

#### No engine config carries a stage list, and PySCF is not an exception to that

**The rule is about where a ladder is DECLARED**, and after this decision
neither engine's config declares one. `task.json` is the only thing anyone
authors, for either engine; `prep` resolves it into one config per rung, and a
rung's config carries THAT rung's values as ordinary flat fields — the same
shape SIESTA's per-rung knobs always had.

The practical test, and the one to keep: **can a person put a value there that
the description does not say?** For a config holding a list of rungs the answer
is yes, which is what makes it a second declaration. For a config holding one
value per knob it is no, because a rung's config is overwritten from the
resolved description on every render.

**The tier values are a table, not a field.** Each engine ships one —
`SIESTA_STAGE_PRESETS` and `PYSCF_STAGE_PRESETS`, both keyed by the same three
tiers — and `<engine>/stages.py::default_<engine>_stages` turns it into the
shipped ladder. A table of defaults is not a declaration: nothing reads it at
render time, and a description that names no tier gets no value from it.

#### What stays PySCF's own

The engine differs in **which parameters a rung varies** and in **how it
answers a question the shared schema asks** — never in the shape of the ladder,
the number of decks, or the number of jobs. Two such answers today:

- **`restart` is control flow rather than keywords.** SIESTA expands the field
  into three declared keys; PySCF's script reads `<JOB>.chk` and
  `<JOB>_optimized.xyz`, or does not. One field, two answers, and the mechanism
  is the engine's business ([`run-identity.md § 4`](?doc=execution/run-identity.md)
  rule 2).
- **`on_nonconvergence` is real control flow**, which is why § 3 keeps it out of
  the *shared* stage schema and why it is a PySCF-only item: on SIESTA the same
  word names a scheduler edge that no longer exists.

### 1.2 Which parameters may vary is the user's choice, not a class's

**The catalogue and the selection come from different places, and fusing them
is what limited a stage to four values.**

| | Question | Who answers |
|---|---|---|
| the **catalogue** | *What settings exist? What type, unit, range and label does each carry?* | **the catalogue itself** — `molbuilder/data/catalogue.template.toml`, authored as TOML ([`template.md`](?doc=engines/template.md) § 4.3). *(This row said "the engine's config class, through the generated form schema" until 2026-08-16. That is the direction `template.md` § 2.1 forbids: it makes the Python class the master and the file its printout. The reversal landed 2026-08-14 when `render_template` was deleted; a config class now **translates** the catalogue on the way out and defines nothing.)* |
| the **selection** | *Which of those settings vary per stage?* | **the user**, in the UI, recorded as `varies` in `task.json` (§ 6.2) |

Today's four relaxation values are a **default selection** over that catalogue —
and not a privileged class of parameter. Any field of the shared schema can be
selected, except the items bound to the whole calculation — the items the catalogue marks `shared` for the kind, the
electronic state first among them (§ 1.3's note; ES1) — the items the rung
itself fixes (`role`: SIESTA's per-step forces and coordinates, a transport
rung's solver and bias; `template.md` § 6.4) — and the run settings, the
catalogue's `execution` items, which each rung's run card states, in the job
request ([`execution/job-request.md`](?doc=execution/job-request.md) § 3.2).

### 1.3 The default selection is a group each engine declares, not a list in code

**Every item in the catalogue carries a `group`**, and the tag is exactly this
question. The vocabulary is closed and lives in `template.GROUPS`
([`template.md`](?doc=engines/template.md) § 5), in render order:

| group | meaning | SIESTA | PySCF |
|---|---|---|---|
| `setup` | what the run is called, and where its pseudopotentials come from — nothing can be built without these | *(2)* `system_label`, `psml_lib` | *(1)* `job_name` |
| `profile` | set once for the calculation | *(18)* `xc_functional`, `solution_method`, `spin_treatment`, `net_charge`, `species_order`, … | *(29)* `method`, `basis`, `functional`, `dispersion`, `solvent`, … |
| **`stage`** | **the settings that typically vary across a sequence** | *(30)* `basis_size`, `pao_energy_shift`, `mesh_cutoff`, `dm_tolerance`, `dm_energy_tolerance`, `scf_energy_converge`, `kgrid`, `kgrid_displacement`, `relax_type`, `relax_force_tol`, `relax_max_displ`, the transport rungs' own, … | *(12)* `scf_conv_tol`, `scf_conv_tol_grad`, `grid_level`, … |
| `budget` | what it is allowed to spend | *(3)* `max_scf_iter`, `scf_must_converge`, `relax_steps` | *(1)* `scf_max_cycle` |
| `output` | what the run writes | *(17)* the `write_*` set, `verbose_comments`, the transmission's `tbt_*` outputs and `tbt_progress`, `ts_hs_save` | *(10)* `verbose`, `chkfile`, `log_file`, `save_*`, `write_trajectory`, `verbose_comments`, `write_molwatch_log`, `es_n_*` |
| `staging` | answered by the staging surface, Task setup, not by a parameter form — every run setting on the run card ([`execution/job-request.md`](?doc=execution/job-request.md) § 2), and what a stage builds on in the stage table | *(13)* `mpi_np`, `omp_threads`, `use_gpu`, `gpu_count`, `max_memory_mb`, `wall_slack`, `continue_retries`, `diag_algorithm`, `block_size`, `parallel_over_k`, `ts_solution_method`, `ts_elecs_out_of_core`, `restart` | *(6)* `threads`, `use_gpu`, `gpu_count`, `max_memory_mb`, `wall_slack`, `restart` |

The counts are the whole group — an item the engine shares with the other
counted in both — the names a readable sample where the group is long; `max_memory_mb` is in `staging` for **both** engines, because it is one of
the five items that declare no `engines` list and therefore apply everywhere
(`verbose_comments`, `write_molwatch_log`, `gpu_count`, `max_memory_mb`,
`use_gpu`)
([`template.md`](?doc=engines/template.md) § 4.2).

> **So `varies` starts as what the proposed stages vary** (`task.varies_for` over their overrides — nothing for a fresh one-stage ladder; carried on the hand-over as `proposed_varies`, the one rule `molbuilder describe` and the Task setup page both write with, 2026-10-18), and the user adds to or
> removes from it — declared by each engine, beside the fields themselves, in the
> one place that already knows what a field *is*. No engine needs code in the
> shared machinery, and a new engine gets a working starting point by tagging its
> own fields.

**`group` says which panel asks about a setting. It does not say whether the
setting may differ between stages.** Those are two questions, and the answer to
the second is § 6.2's: any setting the description is allowed to hold may differ
between stages. The `stage` group is where the table *starts*, not what it is
limited to.

> **Corrected 2026-08-18 (user).** The paragraph above used to end *"That is the
> whole of which parameters may vary"*, and one tag was being asked both
> questions. A setting can only carry one group, so anything that is not a
> physics parameter — and therefore belongs on the staging panel — lost its
> ability to differ between stages at the same time.
>
> `restart` is the case that made this visible, and it is the worst one it could
> have been: **it is the setting that decides whether a ladder is a ladder.**
> `restart` says whether a stage starts from what the stage before it produced.
> It is not a physics parameter, so it sits in `staging`; and because the table's
> columns were being read off the `stage` group, it could not become a column at
> all. A ladder built anywhere except `molbuilder describe --stage-strategy` therefore
> had no `restart` on any stage — which, while `clean` was the default, meant
> every stage started clean and the stages were three unrelated runs. Nothing
> said so; the refusal only arrived later, at `prep --from`, as *"this stage
> declares no warm-restart files"*. **Flipping the default to `continue`
> (2026-08-18) is what removed that failure mode**: a ladder that says nothing
> now continues, which is what a ladder is. It is the description's — an
> override like any other, a column of the stage table (§ 2).
>
> ⚠ **The tag is a default, never a restriction.** Any field of the shared schema
> may be promoted, whatever group it carries — § 1.2's rule stands, with its
> exceptions: an item bound to the whole calculation (`shared`), one the rung
> fixes (`role`) and a run setting (the catalogue's `execution` items, each
> rung's run card in the job request, [`execution/job-request.md`](?doc=execution/job-request.md) § 3.2) are not columns
> at all. The group only decides what
> is *already ticked* when the tab opens.
>
> **What the tag was actually built for, since it is easy to over-read.** It is a
> **UI grouping**, added 2026-06-13 to fix a reported bug: the form used to mix
> stage, budget and system fields inside the same fieldsets, *so switching the
> stage preset silently rewrote budget and system fields too*. Two consumers, both
> in the surface — `form-schema.js` renders the cards in a fixed order so
> "switching the stage selector touches the stage card only" is visible at a
> glance, and `_shared.py::card_of_finding` routes a validation finding to
> the card whose fields it concerns (`web/ui-contract.md` § 5.1). **It has never
> been a model constraint and must not become one.** Under the checkbox design it
> keeps three honest jobs: ordering the form, routing findings, and deciding which
> boxes start ticked.
>
> `profile`'s line says what its settings are, never that they are set once —
> a stage may override any setting that is not shared
> ([`web/form-schema.md`](?doc=web/form-schema.md) § 1.3). When a `profile`
> field turns out to vary in practice, the honest answer is to fix the tag;
> `relax_type` was exactly that case and is now `stage` (below).
>
> **Four items are bound, and not by this tag** *(W34, decided 2026-09-25)*:
> the electronic state — `net_charge`, `spin_treatment`, `unpaired_electrons`,
> `method` — belongs to the calculation, and a stage override of it is refused,
> because every rung's `.DM` or `.chk` is a density for one state. The binding
> is declared on the items, the way transport's `shared` marker binds a value to
> every rung, so the `profile` tag stays what this note says it is
> ([`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) § 2a.2,
> ES1).
>
> **The groups may overlap, and that is not a defect** (user, 2026-08-07). They
> serve **user clarity and where a validation finding appears** — not a partition
> of the model. A field can belong to the run's identity *and* be something a user
> steps; `relax_type` is exactly that. Nothing breaks when the sets intersect,
> because nothing downstream reads the tag to decide anything.
>
> **Two decisions, and they are orthogonal:**
>
> | Question | Decided by | When | By whom |
> |---|---|---|---|
> | **where a value lives** — template only, or template + `task.json` | **the checkbox** | per calculation | the **user** |
> | **how a field is presented, and where its advice lands** | **the tag** | per engine, once | **us** |
>
> So `profile` does not mean *template-only*, and asking which file it "belongs
> to" is the wrong question. A `profile` field a user ticks simply gains a stage
> column, like any other.
>
> **Which is how the tag earns its keep under the rule that a tag must be
> meaningful structurally or functionally: it is functional, and it is not
> structural — and it does not need to be.** Its jobs are real and one of them is
> pinned by a wire-contract test.
>
> ### ⛔ The `section` half of this rule is RETIRED for the two engine configs
>
> **Superseded 2026-08-15.** The SIESTA and PySCF forms are built from
> `data/catalogue.template.toml`, which has no `section`
> ([`engines/template.md`](?doc=engines/template.md) § 5). An item is on the
> form because the catalogue carries it, and § 7 makes that *every parameter
> the schema declares*.
>
> **The passage that stood here got the facts backwards, and it is worth
> recording why rather than deleting it.** It said a field without a `section`
> is *"deliberately internal — `species_order`, `copy_psml` — and never
> rendered"*, and it withdrew an earlier finding on that basis: the untagged
> fields *"are not in the form and there are no findings to route"*.
>
> Both of those are now false, and the second was never a safe inference.
> `species_order` and `copy_psml` were on the form by then (`copy_psml`
> retired at 2.10.0: nothing read it), along with thirteen
> others that `section` had been hiding — ordinary parameters that reach the
> generated file, invisible because a presentation tag had become an opt-in
> gate. The original finding was right: they were half-integrated. It was
> withdrawn because `section` was read as evidence of intent, when all it
> recorded was that nobody had typed it.
>
> **What replaces it**, and it is not weaker: every catalogue item must declare
> a `group` from the closed vocabulary `template.GROUPS`, guarded by
> `tests/test_catalogue_agreement.py::test_every_catalogue_item_declares_a_panel`,
> with a second guard that the renderer knows every card the form asks for.
>
> `section` itself is gone *(2026-10-02)*: the last tab that read it moved onto
> the catalogue on 2026-09-24, and the dataclass form builder and the key were
> deleted with `TransportConfig` ([`web/form-schema.md`](?doc=web/form-schema.md) § 1a).
>
> **And the selection is made in place, one checkbox per parameter** (user,
> 2026-08-07) — **not** a separate list of stage-able settings anywhere. The form
> already lists every parameter; each one carries a *vary per stage* checkbox
> beside it, and what is ticked **is** `varies`. A second list would be a second
> copy of the field set, drifting from the first — the same duplication this whole
> correction removed. [`web/task-setup.md`](?doc=web/task-setup.md) § 5 and
> [`web/form-schema.md`](?doc=web/form-schema.md) § 1.3 carry the surface detail.
>
> **`relax_type` was tagged `profile`, that tag was wrong, and it is now
> `stage`** — ✅ corrected, verified against the catalogue 2026-08-16. The call
> was scientific rather than a naming preference (user, 2026-08-07): a ladder
> deliberately changes the optimizer between stages — CG to warm up, Broyden
> once the geometry is close — so it belongs in the `stage` group, with
> `engines/tuning.md`'s reasoning behind it. It also demonstrated the rule
> above: even while the tag was wrong, a user could tick the box.

**The four hard-coded values are historical residue, and the tagging proves it**
(2026-08-07). Of the four that `render_siesta_stage_fdfs` could vary *(the
shipped renderer this critique was written against; deleted 2026-08-12 with the
pre-resolve producers — `varies` + `overrides` through `prep` is the mechanism
now)*, **two were not even tagged as stage settings**:

| hard-coded as varying | tagged, when this was written | tagged today |
|---|---|---|
| `relax_force_tol` | `stage` ✓ | `stage` |
| `relax_max_displ` | `stage` ✓ | `stage` |
| `relax_type` | **`profile`** — read as a set-once choice | **`stage`** — the tag was the thing that was wrong, and it was corrected |
| `relax_steps` | **`budget`** — a resource | `budget` — still a resource, and correctly so: it caps how long you are willing to relax, not how well |

And **eight fields tagged `stage` could not be varied at all**: `basis_size`,
`pao_energy_shift`, `mesh_cutoff`, `dm_tolerance`, `dm_energy_tolerance`,
`scf_energy_converge`, `kgrid`, `kgrid_displacement`. So the shipped set
simultaneously admitted two fields the tags said were not stage settings and
excluded eight they said were. **The tagging already knew the right answer; the
stage mechanism never read it.**

*(The right-hand column is the 2026-08-16 recheck. One of the two mismatches was
a genuine mis-tag and is fixed; the other was never a mismatch, which is why the
count of stage-tagged-but-unvarying fields grew from six to eight rather than
shrinking — `scf_energy_converge` and `kgrid_displacement` joined the `stage`
group and the deleted renderer would not have varied them either. The argument
this passage makes does not depend on the arithmetic: a hard-coded list of four
names cannot track a catalogue that gains items.)*

### 1.4 One mechanism, engine-specific only where the science is

The same machinery serves every engine, and exactly three things are the
engine's own:

| | Shared, written once | The engine's |
|---|---|---|
| the description | `task.json` + its reader | — |
| the catalogue | **the file itself** — one authored TOML serving every engine, its schema, and the form-schema generator that reads it ([`template.md`](?doc=engines/template.md)) | **which items apply to it** — each item's `engines` list — and, per item, its `anchor` and `engine_key` |
| resolution | template ⊕ `overrides` → one config | — |
| the per-stage table | one tab, driven by `varies` | — |
| the deck | — | **the file the engine reads** (`.fdf` for SIESTA, `.py` for PySCF) |
| correctness | the *structural* preflight (§ 6.6) | **the science** — is this basis adequate for that cutoff, does this ladder loosen |

**Everything generic happens first; the engine-specific judgement happens last,
on the resolved config.** That is what R2 already requires — a stage is validated
as a resolved whole, never as a diff — and it is why the split works: by the time
an engine's validator is asked anything, it is looking at an ordinary complete
config of its own type, exactly as it would for a single run.

> **The science validator is the one that already ships, not a new one**
> (2026-08-07). `molbuilder/validation/` registers a validator **per config
> class** (`_register_engine_validator`) and exposes one door,
> `validate(struct, cfg)`. It is already both of the things a stage needs:
> **the gate before a script is written** — `siesta/input.py`, `pyscf/input.py`
> and `cli.py` all call it — **and the live advice in the tab**, through the web
> blueprints. Per-stage validation is therefore *that same call, once per
> resolved stage*, and nothing engine-specific is added to the staged machinery.
>
> **This also decides an argument elsewhere.** `validate` takes a **config
> object**. Any design in which a stage is resolved by rewriting lines of deck
> text has nothing to hand it, and would lose both R1/R2 *and* the live advice a
> user gets today — which is why the effective config must be a real config
> (§ 4).

**So the stage setting is a contract between the UI and `prep`, and the engine
sits downstream of both.** The browser asks the user which parameters to vary
and writes the answer down; `prep` reads it and resolves each stage into one
ordinary config; the emitter renders that config and never sees a stage. Because
the mechanism is *catalogue × selection*, it is engine-agnostic without being
written twice: every tab already has a schema, so every tab gets this.

> **What went wrong, named so it is not rebuilt.** `stages` was made a *field*
> of `SiestaConfig`, so the form generator walked into it and answered the
> **selection** question with the **catalogue** machinery — listing
> `SiestaStageSpec`'s own fields as the columns a user may vary. That is why a
> stage could vary exactly four things: they were the four somebody typed into
> a Python class. A generator that reads an engine class to discover *what the
> user is allowed to choose* has the arrow backwards.
>
> **Deleting the *field* is what closed it** (2026-08-07) — not deleting the
> generator, which is doing its own job correctly. `SiestaConfig` now has no
> `List[<dataclass>]` field at all, so the catalogue machinery has no route by
> which it can reach a stage. `tests/test_siesta_stages.py` asserts the
> *shape* and not merely the name, since a differently-named ladder would
> reopen it just as wide.

---

## 2. The object

```jsonc
{ "name": "coarse",
  "overrides": { "mesh_cutoff": 150, "relax_force_tol": 0.04 } }
```

**Two fields, and no others.**

| Field | Type | Meaning |
|---|---|---|
| `name` | `[A-Za-z0-9_]+` — letters, digits, underscore, **no hyphen**; **compared case-insensitively everywhere** (two names differing only in case are ONE name, refused as a duplicate) | becomes the deck's suffix, `<label>_<NN>_<name>` (`job-contracts.md § 2.3`). The hyphen is excluded because it is the separator *around* a name, never inside one: a run is `run-0`, a trial is `bench-G1N4C6`, and a try's stdout is `<label>_<NN>_<name>-run<N>.out`. A name free of hyphens means any of those can be split on one without knowing what it contains. Case folds because the name keys **filenames**, and the filesystems these run on include case-insensitive ones (macOS, some network mounts) — `Tight.fdf` and `tight.fdf` are one file there, so `Tight` and `tight` must be one stage everywhere (D6, 2026-08-12: the constructor compared exact strings while both parsers folded case; all three doors agree now). **One key, `identity.stage_key`**, compares a name the description holds with one from outside it — the duplicate check, the verbs' resolver (`launch task --stage TIGHT` is `tight`) and the role rule (a transport rung named `Device` is the `device` rung); until K12 the last two compared exact strings (plan § 5w K12, the M11 review's SS-C15). A verb resolves once, at its entry, and works with the description's own spelling after that |
| `overrides` | map | schema field name → that stage's value |

**There is no `enabled`** *(W38 F5, agreed 2026-09-27 — user: "remove on/off
for optimization and vibration", "yes, drop the seed switch too")*. A
stage is in the description or it is not. Removing one that left files leaves
them untouched and keeps its number taken
([`project-layout.md`](?doc=execution/project-layout.md) § 4.2); a transport
ladder keeps every rung, its seed among them — the leads and the device start
from its density (user, 2026-10-12; the kind table's `stages`, [`execution/architecture.md`](?doc=execution/architecture.md) § 3.3). A description written before carries
`"enabled"` on every stage — refused by name, naming `jobset migrate`, which
removes a `true`; a stage switched off it refuses by name, since whether that
stage now runs or leaves the description is the person's (one policy for a state against the design: an explicit error,
never a silent rewrite; until 2026-10-19 a `true` was read and dropped on
Save). A shipped ladder's
strategy writes the stages it runs; a tier it leaves out is added later from
the tier presets, its values filled in ([`web/task-setup.md`](?doc=web/task-setup.md)
§ 9). *(Until 2026-10-03 a stage carried `enabled`, and a strategy wrote a
switched-off third tier.)*

`overrides` may name **any field of the shared schema** and **never** `name`,
nor an item bound to the whole calculation — the items the catalogue marks `shared` for the kind (the
electronic state's four for every kind; the calculation's identity — its
`system_label`, `species_order` and `psml_lib` — for every SIESTA kind;
transport's electronic description) — nor an item the rung itself fixes (`role`:
SIESTA's per-step forces and coordinates on every kind, a transport rung's
solver and bias). A
description carrying a stage-field name inside `overrides` is refused: two homes
for one fact is how the previous model produced fields that lived in both places
and silently disagreed. A bound item there is refused by name at resolve, saying
why it is the calculation's (`template.why_shared`; ES1 for the state: every rung's
`.DM` or `.chk` is a density for one electronic state); a fixed one, saying why
it is not a choice (`template.why_role`) — and the same refusal meets a pin or a
sweep axis naming it (`template.md` § 6.4).

**`overrides` say what this stage computes, and nothing else.** A mesh
cutoff, a force tolerance — and **whether it builds on the stage before it**:
`restart`, `continue` (the template's value) or `clean`, an override like any
other, a column of the stage table once a stage sets it. How a rung *runs* —
its ranks, threads, device, solver, and which job it runs in — is its run card
and its group, in the job request ([`execution/job-request.md`](?doc=execution/job-request.md) § 3.2, § 3.3), which reads
`restart` from here to know what each stage builds on. A run setting may never
appear in `overrides`, and a field that changes the answer never on a run
card; validation refuses either by name.

---

## 3. Which fields are a stage's, and which are the schema's

Two questions, asked in order. It matters that they are two: either alone
mis-sorts a field.

> **1. Does it survive without a scheduler?**
> A setting that means nothing until something else queues the work does not
> describe a calculation. `execution/job-system.md` owns it.
>
> **2. Of what is left: can a single run mean it?**
> If yes, it is an ordinary field of the shared schema, which a stage may
> override like any other — *wherever that field happens to land* (§ 5) —
> unless it is a run setting, which says how the run goes, never what it
> computes, and is stated per stage on the job request's run card instead
> ([`execution/job-request.md`](?doc=execution/job-request.md) § 2).
> **A stage types only what a single run cannot mean.**

Question 2 deliberately does **not** ask where the field ends up. A promoted
field may become a deck line, a wrapper setting, or a scheduler request; sorting
fields by destination is what produces stage types that grow without limit.

Worked against the fields the deleted `SiestaStageSpec` carried — **the class
is gone (§ 1.1), and this exercise is why**: sorting its eight fields by
these two questions leaves exactly two on the stage, which is the same answer as
*a stage is not a property of a calculation*. The table is the derivation, not a
description of something that will still exist.

| Field | Survives without a scheduler? | Can a single run mean it? | Lands |
|---|:--:|:--:|---|
| `name` | yes | no — a single run is named by its id | **the stage** |
| `relax_type` | yes | yes | the shared schema |
| `relax_steps` | yes | yes | the shared schema |
| `relax_force_tol` | yes | yes | the shared schema |
| `relax_max_displ` | yes | yes | the shared schema |
| `continue_retries` | yes — `running-a-job.md § 3.5` | yes | a run setting, on the job request's run card, routed to the **wrapper** (§ 5) |
| `on_nonconvergence` | **no** — its only effect was the scheduler edge | — | **outside this contract** — and, for SIESTA, **retired with the edges on 2026-08-10** |

Two of those are worth stating explicitly, because both were on the stage type
and neither belonged there.

**`on_nonconvergence` fails question 1.** Its entire effect was the dependency
edge a JobSet threads (`proceed → afterany`, `halt → afterok`). Without a
scheduler there is nothing for it to mean.

> **And on 2026-08-10 that argument finished the job.** A staged ladder stopped
> emitting edges (`project-layout.md` § 1.6), so the policy had **no effect at
> all** — accepted, resolved, and dropped. It was removed from the SIESTA
> producer rather than left inert. Reinstating a per-stage policy means giving
> it a home in the description *and* a reader that does something with it.
>
> **PySCF kept its `on_nonconvergence`**, because its rungs ran in one process
> and the setting became generated control flow inside that loop — a real effect,
> and the reason the same word meant something on one engine and nothing on the
> other.
>
> **Settled with the loop's retirement** *(2026-08-18, § 1.1a)*: it survives as
> ONE rung's own policy — what that deck does when geomeTRIC reports its
> criteria unmet at the step budget — never an edge between two jobs, and
> [`pyscf.md` § 3](?doc=engines/pyscf.md) owns it.

Leaving the stage is not enough, though: if it stayed a field of the **shared
schema** it would be promotable through `overrides` like anything else, and § 2's
"any field of the shared schema" would quietly readmit it. So it is not a field
of the shared schema at all. It belongs to the JobSet producer's own input, which
is a different object with a different reader
([`execution/job-system.md`](?doc=execution/job-system.md) — `job-system.md § 4.1`).

**`continue_retries` passes both questions and is still not a stage field.**
`running-a-job.md § 3.5` is explicit: a *single* SIESTA run whose wrapper was
installed with a retry budget re-runs itself with `--continue` — the same deck
launched again, warm, as a launch again does — so it is a run setting, stated on
the run card, per stage or for the calculation ([`execution/job-request.md`](?doc=execution/job-request.md) § 2); what
made it look special is only where it lands. That is also why
`job-system.md § 4.1` records that the SIESTA ladder never implemented it — the
field sat on the stage, and the stage is not what honours it.

**One field arrives.** Whether a stage continues from what is already in the
folder or starts clean has to be sayable, and by question 2 it is a shared field:
a single run can mean it too. `restart` (`continue` | `clean`) joins the shared
schema; what the generator does with it is
[`execution/run-identity.md`](?doc=execution/run-identity.md) § 4.

**And `group` is not a stage field.** It says which stages one launch runs
— how the calculation runs, not what it computes — so it is the job request's
([`execution/job-request.md`](?doc=execution/job-request.md) § 3.3).

---

## 4. The effective config

> **effective config = the template's values ⊕ that stage's `overrides`**
> — and, laid on last, what the rung itself fixes (`role`,
> [`template.md`](?doc=engines/template.md) § 6.4), which no override may name.

The template (`<label>.template.toml`) is the science backbone the generating tab
wrote — **everything a script owns, with values**: what the user set, or the
default where they did not touch it. A stage supplies only the cells it changes.

**Its format is [`template.md`](?doc=engines/template.md)** — a TOML file with
one table per parameter, each carrying the value in force plus everything known
about it. That is what makes this section implementable: `prep` reads the values
with one `tomllib.load`, and **nothing has to parse an `.fdf`** — which nothing
in molbuilder can do. Together they make an ordinary instance of the engine's
config dataclass — a `SiestaConfig`, not a new type — so every default, every
bound and every `engine_key` mapping applies to it unchanged.

> **Corrected 2026-08-07 (user). This section used to say `base` ⊕ `overrides`,
> and `base` was a key in `task.json` holding "every schema field, one value".**
> That is the template's content, written a second time in a second file, with
> nothing saying which one `prep` reads — and § 7.1's own diagram never mentioned
> it: *template ⊕ the stage's row ⊕ this machine*. The document contradicted
> itself for three sections and I reported the overlap as an open question rather
> than as the duplication it is.
>
> **`base` is removed from `task.json`.** The file carries what *changes* —
> `varies` and the per-stage `overrides` — plus what identifies the calculation.
> What does not change is already in the template, once.

Two rules govern it, and both exist to stop a stage becoming a special case:

**R1 — one object is validated and rendered.** The config handed to validation is
the same object handed to the emitter. What was checked and what was written
cannot come apart.

**R2 — a stage is validated as a resolved whole, never as a diff.** Two overrides
can each be individually reasonable and jointly wrong: a mesh cutoff that is
fine, a basis that is fine, and a pair that is under-converged together. The
validator is handed a whole config plus the stage's name as a label — never an
overlay. The label travels beside `where`, never inside it
(`science/validation.md § 4.1`).

**§ 4 R3 — the sequence is checked as well as its members.** § 4 R2 makes
every stage
individually sound and says nothing about the order they are in, yet the order is
the whole point of having several. A ladder that *loosens* — stage 2 coarser than
stage 1 — passes R2 twice and is almost certainly a mistake, because the second
stage throws away what the first paid for. So a description is also read across
its stages, and a finding about the sequence carries **no** stage label: it is a
fact about the description, not about a member of it (the same rule that already
governs a shared-config complaint, § 6.2). What the checks *are* — which
parameters must not go backwards, and by how much — is `engines/tuning.md`'s to
say, not this contract's; **the catalogue carries it**, each item's `tightens`
(`"down"` for a tolerance, `"up"` for a mesh cutoff — `template.md` § 5), set
where `tuning.md` § 2 gives a tier table, and read on every engine
*(2026-09-30, plan § 5w K4: a table of SIESTA's four stood in
`validation/stages.py` until then, M11 PO-C14)*. **Only rungs of one role are
compared** (`template.stage_roles`, `template.md` § 6.4): a ladder tightens one
calculation as it goes, and rungs that are different programs on different cells
— transport's five — are not one calculation tuned twice, so a transport rung's own tolerance never reads as
a loosening (M11 T-F14). An optimization's rungs have no roles and are compared
whole.

**An `error` in any stage blocks the whole produce**, not just its own deck.
That is not a policy choice made here — it falls out of § 7.2: the folder appears
whole or not at all, so there is no such thing as writing the stages that passed.

---

## 5. Where a field lands — four destinations

A field is not always a line in the deck, and assuming it is writes decks that
are subtly wrong for the machine they run on. Where a field's value comes from
is another question: a run setting's — `diag_algorithm`, `use_gpu`, `mpi_np`,
`omp_threads` — from the job request's run cards, every other's from the
template and a stage's overrides ([`execution/job-request.md`](?doc=execution/job-request.md) § 2).

| Kind | Examples | Lands |
|---|---|---|
| an ordinary deck line | `mesh_cutoff` → `MeshCutoff`; `diag_algorithm` → `Diag.Algorithm` | the stage's deck, and nowhere else |
| **a deck line that is also a resource decision** | `use_gpu` → `Diag.ELPA.GPU` | the deck **and** the wrapper's env routing **and** a scheduler's `--gres` |
| a field the deck never carries | `mpi_np`, `omp_threads`, `continue_retries` | the **wrapper** — baked at prep (`continue_retries`) or resolved at run time (ranks, threads) — and a scheduler's `-n` / `-c` if one is asked |
| **what a rung needs from the runs before it** | a transport rung's `.DM`, `.TSHS`, `.TS.HSX` | **not a field**: the kind's DAG (`transport.stages.inputs_of`), checked and copied by the gather at the rung's first launch (`jobset/gather.py`; `job-contracts.md § 2.1`, *What a stage needs*) |

> **The second row is about where a value *lands*, not about who *chooses* it**
> (clarified 2026-08-07, because the wording invited the other reading).
> `use_gpu` is an **ordinary explicit option** — the user ticks it, and
> nothing derives it from the machine. What makes it a resource decision is only
> that the choice is *read* downstream as well as written into the deck, which
> is exactly what [`template.md`](?doc=engines/template.md) § 6.1's `read_by`
> records on the item itself. It is the **only** item in the catalogue that
> carries one, and the list is `["wrapper"]`.
>
> > ⚠ **This note argued from `diag_algorithm` until 2026-08-16, and the premise
> > had already been measured false on 2026-08-14** — one contract was corrected
> > and this one was not. Any ELPA solver does *not* need a different
> > environment: conda-forge's SIESTA carries ELPA through ELSI and runs it on
> > CPU (`engines/siesta.md` § 7.2, `running-a-job.md` § 2.3). So
> > `diag_algorithm` decides nothing outside its own deck, declares no `read_by`,
> > and belongs in the first row. `use_gpu` is the live case and the better
> > one: the GPU build, the `gres` ask, MPS, the NUMA pin and the rank/thread
> > budget all turn on it. [`template.md`](?doc=engines/template.md) § 6.1
> > carries the same correction.
>
> **And whether the engine can honour a solver choice is the engine's business**:
> a deck asking for an ELPA solver a build does not have fails when SIESTA runs,
> which is the right place to fail. The generator does not check.
>
> **`block_size`** is written verbatim when set, under every target; unset
> means SIESTA's own automatic and the keyword is not emitted at all
> ([`tuning.md § 2.11`](?doc=engines/tuning.md) owns the rule; under GPU ELPA
> SIESTA rounds the diagonaliser's block down to a power of two itself;
> [`template.md`](?doc=engines/template.md) § 12).

> **Why the fourth row is not the third one wearing a hat** *(added
> 2026-08-08; brought to what is built, 2026-10-18)*. The third row's fields
> are **values the wrapper uses**: a rank count becomes an `mpirun -np`, a
> retry budget becomes a loop. What a rung needs from the runs before it is
> not a value at all and not a field: it is the kind's DAG, and the gather at
> the rung's first launch checks each input (the upstream prepared, its newest
> run finished, the same engine input, the file there) and copies it into the
> run with its record — before anything is sent
> ([`job-contracts.md`](?doc=execution/job-contracts.md) § 2.1, *What a stage
> needs*; `jobset/gather.py`). The wrapper checks nothing of the kind.

**The routing is derivable, never a second list.** A field carries an
`engine_key` when it is a line in the deck; the config ↔ exchange translation for
the third row is already fixed by `job-contracts.md § 6.2` and applied by the
producer at its boundary. The fourth row needs no translation at all — the
value is read where it is checked. Nobody maintains a mapping table by hand.

**Walltime, memory and partition are deliberately absent from that table.** They
are not fields of the shared stage schema: they are the job's **scheduler
ask** — the job request's `allocation`, the run card's `time` and `domain`
([`execution/job-request.md`](?doc=execution/job-request.md) § 3.1, § 3.2a), or
`prep`'s flags — and a named `domain` resolves to a partition and QoS on the
target's record. Each is stated or refused; none has a default
([`architecture.md` § 5.2](?doc=execution/architecture.md)).

### 5.1 The middle row, and what it costs

`job-contracts.md § 6.2` lists the eigensolver as a config value that becomes a
`.fdf` keyword and the GPU request as one *derived from* the `.fdf`.
`running-a-job.md § 2.3` says which of the two actually moves the wrapper:
**only a job that uses a GPU re-routes**, to `molbuilder-siesta-gpu` — the
job's GPU request (`jobset.model.gpu_request`), the same `use_gpu` the deck
writes as `Diag.ELPA.GPU .true.`. CPU-ELPA
does not, because the packaged SIESTA carries ELPA through ELSI and runs it on
CPU. The two environments differ by **provenance** — one installs from packages
anywhere, the other has to be built from source — so this is not a hardware
split, and sending CPU-ELPA to the source build used to refuse a perfectly
runnable calculation on any machine where compiling is not allowed.

Two consequences:

- **Two stages in one folder may need two different environments.** A coarse
  stage on CPU and a tight stage on ELPA-GPU is an ordinary thing to want,
  and it works: routing is per script, so each deck's own wrapper activates its
  own environment. Nothing about the folder has to change.
- **A build the target lacks is said, not refused.** If a stage opts into a
  build whose environment the target's record does not list, prep says so in
  its notes — the env and its install command — and prepares the stage as
  asked ([`execution/gpu.md`](?doc=execution/gpu.md) G6); the run script
  enters that environment at its start and stops there without it. The
  reading belongs to *wrapper* rendering, after the decks, and § 6.6 does not
  duplicate it in the preflight.

### 5.2 A deck line may depend on the launch

**A deck carries values tied to the launch it was rendered for.** SIESTA's
BENCH-MARKS block (`job-contracts.md § 3.3`) records the rank count (`mpi_np`)
and, when the deck carries a `BlockSize`, the window that rank count allows —
**orbitals over ranks**, because the block distributes the Hamiltonian
([`tuning.md § 2.11`](?doc=engines/tuning.md)). The `BlockSize` value itself is
never derived: it is the person's, written verbatim, or absent.

**A deck is rendered with the launch it is prepared for**, by one prep, and
a launch sends what prep wrote: nothing compares the deck's record of its rank
count with a launch ([`execution/job-request.md`](?doc=execution/job-request.md) § 6). A launch at another rank count is a
prep again — for a launched stage too, whose deck keeps its science and takes
the new launch's lines ([`execution/job-request.md`](?doc=execution/job-request.md) § 5). A benchmark uses the same block,
because each of its trials is rendered at its own point.

---

## 6. `task.json` — the description on disk

```jsonc
{
  "schema": "molbuilder/task@2",

  "engine": { "name": "siesta" },

  // What kind of calculation this is -- stated by every description,
  // "optimization" included (it was left out for an optimization until
  // 2026-10-06; `jobset migrate` rewrites such a file).
  "calculation": "optimization",        // or "vibration", "transport"

  // How the calculation's files are kept apart on disk (§ 6.7).
  // Required; "hierarchical" is its one value.
  "shape": "hierarchical",

  // What identifies this calculation, and what the user called it.
  // The rules are execution/run-identity.md.
  //
  // `id` is DERIVED -- run.name + structure.formula, normalised once
  // (run-identity.md § 2.0a) -- and the reader CHECKS it rather than
  // accepting whatever string is here.  There is no `label` key: the
  // SystemLabel is the id's first half, and storing it would be a second
  // place for the same string to be wrong.
  "run": { "name": "BDT/Au relax",                    // typed, kept verbatim
           "id":   "BDT_Au_relax_Au38C6H4S2",         // = run_id(name, formula)
           "created": "2026-08-06T22:14:03-07:00" },  // for tracing, not identity

  // What this is a calculation OF: a reference into the tree, plus a witness of
  // what was there when it was written (§ 6.3).
  "structure": { "source": "projects/BDT-Au/structure/bdt_au.xyz",
                 "formula": "Au38C6H4S2", "atoms": 50 },

  // WHICH fields the user chose to tune. Intent — it cannot be inferred (§ 6.2).
  // There is no `base` key: everything that does NOT vary is in the template,
  // once (§ 4).
  "varies": ["mesh_cutoff", "relax_force_tol", "relax_type"],

  "stages": [
    { "name": "coarse",
      "overrides": { "mesh_cutoff": 150, "relax_force_tol": 0.04,
                     "relax_type": "CG" } },

    // tight continues from coarse -- its restart is the template's
    // `continue`; a stage that starts from the structure says
    // "restart": "clean" here.
    { "name": "tight",
      "overrides": { "mesh_cutoff": 300, "relax_force_tol": 0.01,
                     "relax_type": "Broyden" } }
  ]

  // HOW AND WHERE IT RUNS -- the queue, the run cards, the groups, a
  // benchmark's plan, the reports -- is job-request.json's, beside this
  // file (§ 6.8).
}
```

**The calculation kinds a description states**: `optimization`;
**`vibration`** — the described vibrational-spectroscopy job, whose template
the catalogue narrows by the `calculations` key (`template.md` § 6.3) and whose
warm files `pyscf/warm-files.toml` declares; and **`transport`**, the composite
with its own codec rules (`task.py`): exactly one `junction` slot, cited as a tree-relative directory path whose files satisfy
[`transport.md`](?doc=engines/transport.md) § 3.1, no `structure`, a `bias`
list — `siesta/warm-files.toml`'s `[transport]` section declares its warm
vocabulary.

### 6.1 Four rules

**The id is derived, and the reader proves it.** `run.id` is
`run_id(run.name, structure.formula)` and nothing else, checked on every parse
(`molbuilder/task.py::Task._check_id`). `run-identity.md § 3` rule 1 —
*"normalisation happens once, and the result is stored"* — only means something
if the stored value is compared against what made it; without the check, `id` is
a free string and this file can say two different things about which calculation
it is. Since hand-editing is supported (the plan's decision 3), the edit that
matters most is the cheapest to make: **rename `run.name`, leave `id` behind,
and every warm file on disk is orphaned** — § 1's second failure mode, whose
cost is a run that silently starts cold. A mismatch is **refused, not repaired**:
a corrected formula and a renamed calculation are indistinguishable from inside
the file, so guessing which one is right would be `§ 3` rule 3's *append a digit
and carry on* wearing a different hat.

*The `SystemLabel` is not a key here.* It is the id's first half, derived through
the same normaliser (`Task.label`), and what makes deriving it safe is precisely
that `id` **is** stored and checked (§ 2.0a).

**It names fields; it never defines them.** Every key in every `overrides` map,
and every name in `varies`, must resolve to a field the shared schema already
declares. A key
the schema does not know is **refused, not ignored** — an ignored key is a
calculation quietly different from the one that was asked for. This is what keeps
the file from becoming a second schema.

**It is parsed *into* the typed config, not around it.** The reader produces a
config object and stage specs; the emitters are unchanged. A reader that rendered
whatever keys the JSON happened to carry would throw away validation, defaulting
and the `engine_key` mapping, and re-implement all three badly.

**It carries the shipped schema convention.** `job-contracts.md § 6.1` fixes it:
`molbuilder/<name>@<major>`, checked **name + major** through the one shared
helper `molbuilder/persist.py` (`check_schema`), and *"New persisted artifacts
must use it."* That check is not a promise that somebody writes migrations — it
is *"refuses with a clear message rather than mis-parsing"*, which is the
behaviour this file wants. The artifact registry carries its row
(`job-contracts.md § 6.1`). *(Amended 2026-08-12 with § 6.1 itself, U9: this
repeated the "major-only" rule, which the check once implemented literally —
any `@1` artifact parsed as any other `@1` artifact.)*

### 6.2 `varies` declares the columns; `overrides` fills the cells it chooses to

`varies` is the set of fields the user chose to tune — the **columns** of the
table ([`web/task-setup.md`](?doc=web/task-setup.md) § 5). It is intent, and no artefact downstream
records it: a mesh cutoff that happens to be equal in every stage is
indistinguishable, in the decks, from one that was never promoted.

**A stage's `overrides` is a subset of `varies`, never a superset.**

- **No key outside `varies`.** A field nobody promoted must not carry a per-stage
  value, or a demoted parameter leaves a value hiding in a stage nobody can see.
- **A key may be absent**, and absent means **"this stage uses the template's
  value"** — the shared one, unchanged. That is a real state a user asks for: a
  column exists because *some* stage varies it, and the stages that do not are
  simply at the backbone value.

> **Corrected 2026-08-07.** This section used to require *exactly* the keys in
> `varies`, and that had two faults. It made `varies` **redundant** — with
> equality, `varies` is just the key set of any stage's overrides, derivable from
> the file, so the sentence above defending it as un-inferable was arguing about
> *decks* while stating a rule about *this file*. And it made the table's own
> design unbuildable: § 6 of the tab plan draws **a cell equal to the shared
> value quietly** so that progressive tightening reads as a shape, which requires
> a way to *be* at that value — and equality forbade it, forcing every cell to be
> filled with a copy.
>
> The subset rule fixes both. `varies` becomes load-bearing rather than a
> duplicate: it is the one place the column set is stated, and it cannot be
> recovered from the cells once a stage is allowed to leave one empty.

**And the fallback is the template, not a second copy in this file.** A stage
that omits a varied key renders with the template's value for it (§ 4).

#### Which settings may become columns

> **Any setting the description is allowed to hold may become a column. The ones
> it is not allowed to hold may not.**

There is no separate list of promotable settings, and there must not be one: § 1.2
already says a stage may name any field of the shared schema but the ones bound to
the whole calculation (the catalogue's `shared` marker, which the columns read
too), the ones the rung fixes (its `role` marker, read the same way), and the
run settings — the catalogue's `execution` items, the machine's answers
(ranks, cores per rank, memory) among them. Those rules together give the
column set with nothing left to decide.

*(A run setting is not column material because it is not a per-stage
**parameter** of the calculation: it is how the rung is run, and it has its
own home — the rung's run card in the job request ([`execution/job-request.md`](?doc=execution/job-request.md)
§ 3.2). A column would be a
second home for one value, and a second home is how a rung's `use_gpu`
reached its deck and not the scheduler's device ask (plan § 5w K5, SO-C1;
2026-09-30). The template refuses a value on one, too
([`template.md`](?doc=engines/template.md) § 7): a run setting is stated on a
run card and nowhere else.)*
It is the same membership `prep` already applies when it accepts or refuses an
override, a pin, or a benchmark axis, so a column the table offers is a column
`prep` will accept, by construction rather than by agreement.

Concretely, for SIESTA that means the physics settings — `restart` among
them, what a stage builds on — and **not** the run settings
(`template.run_settings`): `continue_retries`, `wall_slack`, `use_gpu`,
`diag_algorithm`, `block_size`, `parallel_over_k`, `ts_solution_method` and
`ts_elecs_out_of_core`, which a person answers on the run card, nor `mpi_np`,
`omp_threads`, `gpu_count` or `max_memory_mb`, which the machine answers. A surface that instead borrows the parameter form's list gets
a different and smaller answer, because that form filters out the whole staging
panel on purpose: it does not ask a person how many ranks the scheduler granted.
Filtering a panel and limiting a table are different jobs (§ 1.3).

> **The two files answer different questions, which is why neither duplicates
> the other** (user, 2026-08-07):
>
> | | |
> |---|---|
> | **the template** | *everything a script owns, with values* — what the user set, or the default where they did not touch it. **Including the parameters that vary**: the template holds their starting value. |
> | **`task.json`** | *which of those the user wants flexible*, so the calculation can be conducted stepwise — plus each stage's value for them. |
>
> So `mesh_cutoff` appears in both, and says something different in each: the
> template says **what it is**, the description says **that it steps, and to
> what**. A stage that overrides it wins; a stage that does not takes the
> template's. That is the whole relationship, and it is why there is no `base`.

### 6.3 `structure` is a reference plus a witness — and the FILE travels

Coordinates are what runs *produce*; a description that embedded them in
`task.json` would be a second copy drifting from the file the moment either
moved. So `source` records the reference, and `formula` and `atoms` travel
beside it as evidence of what was there when the description was written —
which is what the id was built from (`execution/run-identity.md § 2`) — with
`fingerprint`, the pair's own (`StructureCodec.fingerprint`: its two files'
bytes, their stamps masked), so every atom, coordinate, label, held atom and
the box are pinned. A description opened against a structure that has since
changed can therefore *say so*, rather than silently building a different
calculation under the same id (`run-identity.md § 5`): **every prep compares
the fingerprint and refuses an edit**, naming it — every stage of a
calculation is of the one structure it was described with, and describing
again is the way to build on an edit *(user, 2026-10-14: a person who edits
a calculation's structure between its stages breaks the agreement)*. A
description written before the fingerprint was recorded states none, and is
checked by `formula` and `atoms` alone.

**The structure FILE is copied into the calculation at `describe`, exactly
like the pseudopotentials** *(M9, 2026-08-12 — this section's title said
"never a copy", and the sentence conflated two different copies)*. What
§ 6.3 forbids is coordinates embedded **in `task.json`**; what M9 requires
is the data file **beside it**, because the description is the portable
package and a relative `source` recorded from another cwd was unresolvable
the moment the bundle moved. `prep` resolves the reference
beside-the-calculation FIRST, then at the recorded path; the file is the
calculation's data, the PATH stays the describing machine's.

### 6.4 What writing it down buys

Three things, and the first is why the file exists at all rather than the
description living only in a browser tab.

- **One producer for both surfaces.** The CLI and the browser stop being two paths
  to a staged calculation: each writes a description, and the same reader turns it
  into decks from either. That is what makes "the web is additive on top of the
  CLI" checkable — the two must produce the same bytes for the same description,
  and a single reader is how.
- **A deck can be traced back to what asked for it.** PROVENANCE
  (`job-contracts.md § 3.2`) already reserves an optional `form-config-hash` key
  and this is its use: the hash of the description that produced the deck. Any
  deck in a project then names its origin, and a deck someone edited by hand can
  be told apart from one the description would reproduce. PROVENANCE stays exactly
  what it is — a generation snapshot, not a config.
- **Descriptions diff.** Two calculations that differ can be compared as *intent*
  — one file against one file — rather than by reading two directories of decks
  and inferring what was deliberate.

### 6.5 A job always has at least one stage

> **Decided 2026-08-16 (user): *"Consistency and uniformity is better than all
> these implicit rules. A single stage is still a stage. You start with one, and
> that's it."*** This section said the opposite until then — that `stages` could
> be absent and absent meant one — and the paragraph below records what that
> cost.

**Every description carries `stages`, with at least one entry.** One stage is the
ordinary starting point, not a special shape: it gets a name, it gets the ordinal
token like any other (`01_coarse`), and its artifacts are named the way every
other stage's are. Adding a second stage later is adding a row.

**What the old rule cost, and why it is gone.** A stage-less description produced
artifacts with *no* token — `<label>.fdf`, `<label>.XV`, `<label>.out`, all at the
folder root. That is fine while it stays stage-less. The moment a second stage is
added, the ladder needs `01_` / `02_` tokens and **the existing run belongs to no
token at all**: nothing says whether it *is* stage one or is simply orphaned.
The fix for a transition nobody had specified is to make the transition
impossible.

**A description with no `stages` is refused, not repaired.** There is no
migration and no tolerated older form — accepting one would reintroduce the
second path this rule exists to delete, and the refusal names the fix.

`varies` travels with `stages` and is therefore always present too; empty is a
real state — several stages differing in nothing but their name.

Removing the last stage is refused.

**A group is named or picked, never guessed — including when there is only
one.** `prep task --group 1` names it; a bare `prep task` shows the groups
and asks which — what it offers is [`job-system.md`](?doc=execution/job-system.md),
*The task*, `prep task`, 2 — on one rung exactly as on three, and with nobody
to ask it is refused, naming the line that picks them. `prep bench
coarse`, not `prep bench`; `--from 01_coarse/run-0`, not `--from run-0`.
Taking the lone stage without asking would be a rule that holds only at length
one and silently stops holding when a second stage is added — the implicit kind
this section exists to delete. The cost is one answer, or one `--group`; what it buys is that the command a user
learns on their first calculation is the command that still works on their
tenth.

### 6.5a The hand-over — a partial description, and why it is a different file

*Added 2026-08-16 (user).* A parameter surface can collect the physics but
does not settle the ladder — the stages a description requires (§ 6.5) — so
what it produces is not yet a description. It writes **`task.1st.json`**
(`molbuilder/task-handover@1`) beside the template **and beside the structure
the calculation is of** — the `.xyz` + `.molstruct.json` pair its
`structure.source` names, folder-relative — and Task setup, which asks for the
stages, finishes the job: its Save writes the description whole, its `shape`
the one layout's (§ 6.7).

A hand-over carrying only a formula and an atom count is not one: the receiving
folder has to be readable on its own, and a k-grid without the lattice it
indexes is not a calculation anybody can check
([`handover-procedure.md § 7`](?doc=web/handover-procedure.md) records what that
cost).

**Why not just write `task.json` early and fill it in.** Its *presence* is a
claim, not a convenience: `checkpoint.py::_BUNDLE_DESCRIPTORS` treats a
`task.json` as the marker that a folder **declares itself the root of one
multi-directory unit of work** (`checkpointing.md` L1). A premature one makes a
folder claim to be a calculation root before it is one. The same shape as
`run.json`, which "cannot exist before launch — its presence marks the run
launched"
([`project-layout.md § 1.6.3`](?doc=execution/project-layout.md)), and which
answers with the same device: a private carrier between two steps of one act.

**Three rules, and each one is doing work:**

| | |
|---|---|
| **its own schema** | `molbuilder/task-handover@1`, never `molbuilder/task@2`. It has no `shape` and no settled `stages`, so it would fail that schema's reader — and a file claiming a schema it does not satisfy is worse than one that says what it is. `check_schema` refuses a wrong artifact **by name**, so nothing can read it as a description by accident (`job-contracts.md § 6.1`) |
| **the extension is last** | `task.1st.json`, not `task.json.1st`. Highlighting is chosen by suffix, so the second spelling renders as plain text in the editor a person is meant to read it in; and nothing that looks for `task.json` matches it |
| **it says what it is** | JSON carries no comments, so the file opens with a `_what` line and an `awaiting` list naming the keys Task setup still asks — `stages`; the `_what` line says the same, from the same list. It is read by a **person**, in an editor, and should not need a document open beside it |

**It resolves in one direction only.** On a successful save the real `task.json`
is written and the hand-over is **deleted**, so the next visit to that folder
finds one description and no ambiguity about which file is current. A folder
holding both is a save that did not finish; the description wins, because it is
the one that passed § 6.6's preflight.

### 6.6 The preflight

In order, and all of it before anything is written:

| Check | On failure |
|---|---|
| the schema string is `molbuilder/task@<known major>` | refuse — not a description, or not one this reader knows |
| the engine is one this backend has a generator for | refuse, naming what it has |
| every named field exists in the shared schema | refuse, naming the field |
| no `overrides` key names a stage field (§ 2) | refuse, naming the field |
| no `overrides` key names an item bound to the whole calculation (`shared` for the kind — the electronic state for every kind, ES1; the calculation's label, species order and pseudopotentials for every SIESTA kind) | refuse at resolve, naming the item and why it is the calculation's |
| no `overrides` key names an item the rung fixes (`role` for the kind — SIESTA's per-step forces and coordinates on every kind; a transport rung's solver and bias) | refuse at resolve, naming the item and why it is not a choice |
| every stage `name` matches `[A-Za-z0-9_]+` | refuse, naming the stage and the rule |
| **stage names are unique**, compared case-insensitively | refuse, naming the repeat |
| no `overrides` key names a run setting — the catalogue's `execution` items, the job request's ([`execution/job-request.md`](?doc=execution/job-request.md) § 2) | refuse, naming the item and its card |
| the job request beside it: every stage with its entry and its group, no stage in a group with one it builds on — the job request's own check ([`execution/job-request.md`](?doc=execution/job-request.md) § 3, § 3.3), asked here with the template read, since the ladder's links need it (`group.check_groups`; [`job-system.md`](?doc=execution/job-system.md) § 5.0, checkpoint 3) | refuse, in its reader's words — a description written before 2.27.0 (`task@1`) names `jobset migrate`; one at `task@2` holding a job key is refused, naming `job-request.json` |
| every value is one its item's declared type can hold — the catalogue's `type`, asked of the setting door in the value's home (`template.accepts`, [`template.md`](?doc=engines/template.md) § 5.4), never a config class's annotation | refuse, naming the field, the value and what the item declares |
| every value may stand for its item on this kind — not a component the kind fixes, a choice the kind offers (`offered`, § 6.3a — a vibration's relaxation a relaxer), above its hard limit — one door, `template.why_not` ([`template.md`](?doc=engines/template.md) § 5.3) | refuse, naming the stage, the value and the one reason, the first that holds |
| every value is inside its item's recommended `range` | **warn**, naming the field and both bounds — a recommendation, the same severity on every surface, and silent for a value refused above ([`template.md`](?doc=engines/template.md) § 5.3) |
| the job request's run cards' values and a bench's points may stand for their items too — the same door, in their home: the type, the choices, the limits, and `null` only for an item whose unset is an answer ([`execution/job-request.md`](?doc=execution/job-request.md) § 3.2) | refuse, as a stage's value, in the item's words — they become run settings and sweep points, and prep would otherwise refuse them first, on the machine that runs it |
| a transport description's bias list: numbers, starting at 0, no two points in one folder (`bias_token`) | refuse (the codec) — each point is one device run in its own `v<V>/` folder ([`transport.md`](?doc=engines/transport.md) § 2a.10) |
| each bias point inside `bias_voltage_v`'s recommended range | **warn**, as any value |

**Two things are deliberately not checked here.**

- **The engine's version.** Nothing in the shipped system records or gates one.
  The version is known where the binary is — `running-a-job.md § 4.1`'s run banner
  prints it — and the machine writing a description may not have the engine at
  all. Gating here would break `job-system.md`'s decision 3, *the machine's
  knowledge lives on the machine*.
- **Declared requirements** (MPI, a GPU build, a library). Already answered twice,
  at well-defined moments: env routing derives the requirement from the deck
  (§ 5.1), and the doctor verifies prerequisites on the target
  (`running-a-job.md § 2.2`). A third, hand-maintained list would only drift from
  what the deck actually asks for.

> ### ⛔ The schema-fingerprint row is RETIRED — deleted 2026-08-14
>
> It was the preflight's only non-refusal: it reported that a description had
> been written against a different schema, and proceeded. **The argument for
> deleting it is the sentence that used to stand here** — *"the fingerprint's
> claim is deliberately weak. One string can say* this was written against a
> different schema*; it cannot say which fields moved. The per-field rows do
> that work."*
>
> Every row below it names the parameter and the problem. The fingerprint
> announced that something had moved, more vaguely, immediately before the
> checks that said what. One writer, one reader, no refusal.
> [`template.md`](?doc=engines/template.md) § 10 records the deletion, and
> `task.json` no longer carries `schema_fingerprint`.

**Why two of those rows are about names.** A stage name becomes a filename
(§ 2), so a name outside the set or repeated between stages produces two decks
that collide — the second silently overwriting the first, in a folder whose whole
premise is that every file in it is accounted for. Refusing costs a message;
allowing it costs a calculation nobody knows is missing.

### 6.6a Two stages that resolve to the same thing

*Decided 2026-08-07 (user). This used to be an open question pointing at the
plan; it is now a rule, and it is not the blanket warning the question expected.*

**Two stages may resolve to identical settings, and that is allowed.**
Refusing would break a workflow people actually want: `tight` followed by
`tight` where the second **continues** is simply *more steps at these settings* —
the honest way to say *keep going* after a stage ran out of its step budget.
Forbid it and someone invents a token difference to get past the check, which is
worse than the thing being prevented.

**Warn on exactly one case: the later stage resolves identically *and* starts
`clean`.** Then it recomputes what the stage before it just produced and throws
that result away — always a mistake, and an expensive one.

> **What separates them is `start from`, not the overrides.** So the comparison
> is over the **resolved pair**: two stages whose effective configs are equal
> *and* whose second says `clean` — in its overrides, the one place a rung's
> `restart` lives (§ 2). Comparing the other overrides alone would flag the
> legitimate case and miss nothing, which is how a warning becomes noise people
> learn to click through.
>
> This is a **warning, not a preflight row.** § 6.6's table is refusals, all of
> them before anything is written; this one says *this is probably not what you
> meant* and proceeds if it is.
>
> **`restart` is the discriminator, so it is not part of the equality test**
> (settled while implementing this, 2026-08-07). A field cannot both separate
> two stages and be part of the test for whether they are the same. Read the
> other way, the second clause would be redundant — equal configs already agree
> about `restart` — and one real recompute would slip through: an earlier stage
> that **continues** followed by an identical one that **cleans**, which redoes
> the first from scratch. So: equal *in every field but `restart`*, and the
> later one says `clean`.
>
> **Adjacent pairs only** — *"the stage before it"*. Two identical stages with
> a different one between them do not recompute each other's output.
>
> Implemented at `validation/stages.py::check_identical_stages`; the finding
> carries **no stage label**, by § 4 R3's rule — it is a fact about a pair, not
> about a member of it.

---

### 6.7 `shape` — the layout the calculation uses

**`shape` is required, and its one legal value is `"hierarchical"`**: each
stage a directory, each run a directory in it
([`project-layout.md`](?doc=execution/project-layout.md) § 1). Every writer of
a description writes it — `molbuilder describe`, Task setup's Save, the
Transport tab's describe door — and nothing asks for it, since there is
nothing to choose. **The description's reader refuses any other value by
name** (`task.of`) — `"flat"` among them — its way out to describe the
calculation again; no description is rewritten.

**Why it is written down.** The description is the one artifact every `prep`,
and every reader of the folder, reads; it states how its files are kept, so a
description written for a layout molbuilder does not lay out is refused by
name rather than read as this one. The layer that names stage directories,
run directories and benchmark containers is the same for every engine and
kind, and names none of them.

### 6.8 What is not here — the job request

**How and where the calculation runs is its job request's**, a file of its own
beside this one, `job-request.json`
([`execution/job-request.md`](?doc=execution/job-request.md)): what it asks
the scheduler for (`allocation`), each run's card (`execution`), which stages
share a job (each stage's `group`), what a benchmark measures (`bench`) and
when the calculation reports (`notify`). **This file says what is computed, and
nothing else** — the rule both files keep is § 1 of that contract.

---

## 6b. Open questions about the description

*Migrated here 2026-08-11 from `archive/2026-08-11-staged-runs-architecture.md` § 9, which is
now archived. Both are about `task.json`, so they belong here.*

1. **Is `task.json` the right name?** It sidesteps the collision the word *plan*
   already has in this domain — *"Job-set plan"*, the registry label for
   `job-set.json`, and prep's `STAGE-PLAN.md` (and `jobset plan` the verb, until
   it was folded into `status`, 2026-10-01).
2. **Is a description editable by hand?** It is JSON sitting beside the decks,
   so it will be. If yes, the reader owes a person the same errors it owes the
   browser — which is an argument for § 6's refusal rule being **loud rather
   than tolerant**, since a hand-editor has no form to stop them first.

---

## 7. What the generator must produce

> **Scope, so this section does not drift into its neighbours' territory.** What
> a produce must *emit per stage* — decks, wrappers, the description, and the
> transactional rule that they all appear or none does — is this contract's, and
> that is § 7 proper and § 7.2. **Where those files land is not**: the levels of
> the tree, how each is named, and who may write at each one belong to
> [`project-layout.md`](?doc=execution/project-layout.md). § 7.1 below defers to
> that contract rather than restating it.
> **Nor is the saved history**: [`checkpointing.md`](?doc=execution/checkpointing.md)
> owns it, and § 7.4 is a pointer rather than a specification. What stays here in
> § 7.3 is only the part that is about a *stage* — that a stage's name is its
> identity, and what follows from a description that grows.

A folder whose decks are correct on their own. Concretely, per rendered stage:

- **the cell, explicit** — the description holds cell *parameters*; the generator
  computes the vectors and shifts the atoms into the frame the deck must carry
  (`model/structure-periodicity.md § 6.2`; design record `archive/2026-08-20-cell-plan.md`).
- **pseudopotentials resolved** per species, through the path that already
  refuses on `xc_family_mismatch`, and written into the folder. (`job-contracts.md
  § 2.7` says the layout does not *require* co-location; putting them there is
  what makes the folder self-contained.)
- **every value the description determined**, written rather than left to an
  engine default. A field the user set must appear in the deck; a field the
  description never touched may rely on the engine's own default, which is what
  engine defaults are for. The failure this rules out is *omit-and-hope* — leaving
  out a value the calculation depends on and discovering later which default
  filled it.
- **the engine's identity group set as one**, never key by key —
  [`execution/run-identity.md`](?doc=execution/run-identity.md) § 4.
- **BENCH-MARKS declaring every line derived from a launch quantity** (§ 5.2).
- **a run wrapper per deck**, built by the shipped builder
  (`job-contracts.md § 2.6`). A folder of decks with no wrappers is not something
  a user can run.
- **a distinct trajectory-log basename per deck.** Two decks resolving to one
  basename would interleave their frames into a single file, and no reader
  could separate them again; and the name says which stage a log belongs to
  wherever it is read.
  **The rule: a run's log is named for the deck that produced it** —
  `<label>_<NN>_<name>.molwatch.log` beside `<label>_<NN>_<name>.fdf`. One
  naming, derived rather than declared, so there is nothing to keep in step.
  **Landed 2026-08-10**: the stage's names (`runfiles.RunNames`; the helper
  `molwatch_log_basename` until 2026-10-06) take the stage's artifact token, the same one the deck carries, and every reader reads it back
  through the one grammar, `runfiles.parse`, with the run's label rather
  than keeping a second regex *(through `identity.parse_stage_token` until
  2026-10-04)*.

  **Cost, stated rather than hidden:** the run decoder's stage regex keyed on
  the `-stage<N>` form the log carried before, so it changed with this. That is code following a
  contract, which is the direction that is allowed. What it did **not** cost is
  the decoder's ordering: decision 27 keeps the ordinal in the token, so
  `_anchor_sort_key` still has a number to sort on and the Results tab keeps its
  notion of *the active stage*.

**The test:** the decks are portable — an engine with no molbuilder installed
runs them correctly, a PySCF deck with the bundle of molbuilder's code it imports
beside it (`mb_pyscf.pyz`, [`engines/pyscf.md`](?doc=engines/pyscf.md) § 3),
which is the same on every machine. The wrappers are not, and are not meant to
be: they are baked for a target (§ 8).

### 7.1 The layout: portable above, machine-specific below

**What this contract requires of the layout:** what every stage
shares *and any machine can read* is written once and kept apart from what one
stage produced **for one machine**. That separation is what makes the description
portable and the deck disposable.

**How that separation is realised is not this contract's to say** — it is
[`project-layout.md`](?doc=execution/project-layout.md) § 1: each stage gets a
directory, each run a directory in it.

The tree below is drawn here only because § 7.2–7.4 refer to it.
`project-layout.md` § 1 is the authority, and if the two ever disagree that
one wins.

```
projects/BDT-Au/optimization/bdt-relax/     ← the folder: the user typed this
├── <label>.template.toml               ← the science backbone
├── task.json                          ← what each stage tunes, and the run id
├── job-request.json                   ← how and where it runs: queue, run cards, groups
├── pseudos/  Au.psml  S.psml  …       ← shared, stored ONCE
├── 01_coarse/                         ← written by `prep`, on the target
│   ├── <label>_01_coarse.fdf             ← template ⊕ coarse ⊕ its run card
│   ├── <label>_01_coarse.run.sh          ← its wrapper, for this machine
│   ├── mb_monitor.pyz                 ← the monitor's one file, beside it
│   ├── Au.psml  …                     ← a real copy of the shared pseudopotential
│   └── run-0/  run-1/                 ← what each run produced
└── 02_tight/
    ├── <label>_02_tight.fdf
    └── run-0/
        ├── <label>.XV                 ← real copies, from the coarse run it continues
        └── <label>.DM                    (SIESTA names these, so they are bare)
```

**`<label>` is the `SystemLabel`, and it is the stem of every file here.** The
**run id** — the label plus the structure's formula — is a field in `task.json`
and never a filename (`run-identity.md § 2.0a`). A molbuilder-named file adds
`_<stage>`; an engine-named one cannot, because SIESTA looks for
`<SystemLabel>.XV` and nothing else.

**One template, one deck per stage, and the fan-out happens at prep:**

```mermaid
flowchart LR
    T["<b>&lt;label&gt;.template.toml</b><br/>functional · basis · k-grid<br/>every parameter — the hardware's<br/>named, not answered"]
    J["<b>task.json</b><br/>coarse: mesh 150, tol 0.04<br/>tight:  mesh 300, tol 0.01"]
    M["<b>its run card, or prep's flags</b><br/>ranks · solver · GPU<br/>for the machine it runs on"]
    DC["<b>01_coarse/&lt;label&gt;_01_coarse.fdf</b>"]
    DT["<b>02_tight/&lt;label&gt;_02_tight.fdf</b>"]
    T --> DC
    T --> DT
    J -->|"coarse's row"| DC
    J -->|"tight's row"| DT
    M --> DC
    M --> DT
```

Two decks come out, as before — what changed is **who renders them and when**.
The template is written once by the browser; each deck is produced by `prep`, in
its own stage directory, on the machine that will run it.

**The deck is rendered where the machine is known, and that is not deferral for
its own sake.** Some of what goes *inside* a `.fdf` is a fact about the hardware:
the BENCH-MARKS window a `BlockSize` may use is drawn from the rank count and
whether there is a GPU (`siesta/input.py` `_block_size_bounds`), and whether
the job uses a GPU picks both the numerics (`Diag.ELPA.GPU`) and the conda
environment the wrapper activates — the job's GPU request
(`jobset.model.gpu_request`), the same `use_gpu` the deck writes (§ 5). A deck finished on a
laptop is either wrong for the cluster or guessing. So the parent carries a
**template** and `prep` completes it — the same shape `prep bench` already ships,
where the bundle is portable until the target formats it
(`project-layout.md § 2.2`).

**The template is not a fill-in-the-blanks file.** It is the effective config
rendered with the machine-dependent keys left out; `prep` renders the stage's
deck from the same renderer with those keys resolved. One renderer, two moments —
not a text-substitution language.

**This is not a new layout.** It is what `job-system.md § 5.2`'s `prep` already
builds, and its place in the wider tree is
[`execution/project-layout.md`](?doc=execution/project-layout.md), and this contract reuses that materializer rather than writing a second
one. Two of its properties are the reason:

- **Shared files have one source and are copied into every stage that needs
  them** — the calculation's `pseudos/` is the one copy of a pseudopotential,
  and each stage holds a real copy beside its deck, never a link (user,
  2026-08-24: a run directory holds everything it needs, and a link holds
  nothing; [`project-layout.md § 1`](?doc=execution/project-layout.md)).
- **What a stage continues from is copied, not linked.** Stage 2 writes to
  `<label>.XV` itself; a link would send that write straight through into stage 1's
  directory and destroy the result it started from. A real copy closes it, and
  the stage's **first launch** makes it *then and there* (prep copies nothing
  from another stage's run), because the run it takes has already finished —
  which is the whole payoff of stages not being chained
  ([`project-layout.md § 1.6`](?doc=execution/project-layout.md)). The read-only
  files — the deck, the wrapper, the monitor, the pseudopotentials — are real
  copies too: **every file in a stage is a real file, never a link.**

**Why a directory per stage.** `.XV`, `.STRUCT_OUT`, `.EIG` and every other
engine output is keyed by `SystemLabel`, which is *identical* across stages by
design. Stages sharing one folder would keep one set of results — the last.
For a framework whose purpose is managing a mission across several parameter
sets, losing every intermediate result is a defect.

**Stages are not chained**, and what that means on disk —
no `depends_on`, no queued follow-on, nothing pointing at a file that does not
exist yet, and a **copy** made at the stage's first launch from the run it
builds on — is
[`project-layout.md § 1.6`](?doc=execution/project-layout.md)'s, which owns the
rule and the reasoning.

> **The rule is about a LADDER, and this document once stated it as though it
> covered everything** *(corrected 2026-08-11)*. It read *"nothing schedules a
> stage after another, **here or anywhere**"* — and the transport composite's
> bias walk runs a sweep's points in sequence under one submission
> (`engines/transport.md` § 2a.11; it was `transport bundle`'s
> `run-transport.sh` until the composite retired it, 2026-08-29).
>
> **The two are different relationships, and the difference is what the rule is
> protecting.** A ladder's stages are *attempts at one answer*: whether stage 2
> should start is a judgement about the geometry stage 1 produced, and a chain
> would spend a week refining something you would have rejected. Transport's
> three runs are *one answer assembled from three pieces*: the electrode run
> emits a `.TSHS` that is an **input**, not a result anybody evaluates, so there
> is no judgement between them to take away.
>
> **What is true without exception is the narrower sentence:** *no **stage** of a
> ladder is scheduled after another*. Whether a genuinely coupled set should get
> a representation — and stop needing a shell script — is
> [`job-system.md § 2`](?doc=execution/job-system.md) decision 6's recorded
> limit, not a rule this contract may extend to cover it.

**What this contract owns is only the part that is about a *stage*: which files
it declares.** `.XV` always, `.DM` when the config saves it, and `.CG` **only
when the run it continues from used the same relaxation method** — *"a CG state
is meaningless to a Broyden stage"*. The stage declares that as its `warm` list
with `requires_same: "optimizer"`, and the comparison is made at the launch
that takes the files, against the run it continues from (`job-system.md
§ 4.1`).

**And `restart` gets sharper.** In one directory, *continue* could only mean
"whatever ran here last" — order-of-execution dependent, and wrong if you re-ran
an earlier stage. With a subdirectory each, **`continue` means: carry from the
previous stage**, which is a fact about the description rather than about
what happened to run. `clean` carries nothing.

### 7.2 The folder appears whole, or not at all

Rendering a description can fail after it has started: a pseudopotential does
not resolve, a disk fills. **A half-written folder is worse than none**, because every rule in
this contract about what a folder contains stops being true of it, and the run
directory it half-occupies may already hold warm files from a previous
calculation.

So a produce is **transactional**: every deck, every wrapper and every other file
it writes is decided first, with nothing written, and written only when all of
them succeeded ([`execution/job-system.md`](?doc=execution/job-system.md) § 5.0,
rule 3). On a refusal nothing is written, and the message names the stage that
stopped it; a write that fails part way — a full disk — leaves the stage not
prepared, and the folder's state saved before the prep is the way back.
*(Until 2026-10-05 prep wrote as it went and put three files back when it
refused, leaving the decks, wrappers and data files behind.)*

This is the same discipline the sidecar and archive writers already use — build,
verify, then write ([`model/structure-molstruct.md`](?doc=model/structure-molstruct.md)'s
atomicity rule, which each file of a produce is still written by) — applied to a
calculation rather than a file. What it must **not** do is remove warm files that were already
there; producing twice is `execution/run-identity.md § 6`, and those files are
the point.

**And a produce that replaces an earlier one may make the folder match the
description exactly — because it checkpoints first.** Remove a stage, and the
deck it produced last time is still there, with a wrapper that still
runs it, describing a calculation the description no longer contains. Left alone
that breaks the premise every rule here rests on: that a folder's contents are
what its description says they are.

The answer is not to tiptoe around the orphans. It is
[`molbuilder checkpoint`](?doc=execution/running-a-job.md) § 6, which already puts
a run directory under a git-backed history — text tracked (including the small
`.XV` / `.CG`, *"so a restore brings back a resumable state"*), large binaries
archived by content and stored once, and the engine's scratch in neither
store ([`checkpointing.md`](?doc=execution/checkpointing.md) S1):

> **A replacing produce checkpoints the folder before it writes anything.**
> A stage the description no longer contains keeps its files, untouched
> ([`project-layout.md`](?doc=execution/project-layout.md) § 4.2), and
> nothing is lost, because the prior state is a commit — restore it, or
> branch from it.

A produce that only rewrites decks changes only text, so that half is cheap —
and so is the binary half: an archive is named by the sha256 of its own
MANIFEST, and a large file whose content an earlier archive already holds is
hard-linked into the new one, so a checkpoint of unchanged binaries costs no
disk ([`checkpointing.md`](?doc=execution/checkpointing.md) § 3.1, I1, § 12
*Disk cost*). That is what lets a mission be checkpointed at both of § 7.3's
boundaries without paying for its `.DM` set twice per stage.

The warm files are never removed by any of this: they belong to the calculation,
not to any one stage (`execution/run-identity.md § 6`).

### 7.3 A description grows, and a stage that has run is a record

A description is not written once and produced once. The ordinary way a mission
goes is **incremental**: run a stage, look at what came out, decide the next one
from what you saw, run that. So the stage list **grows over time**, and a produce
usually lands in a folder where earlier stages have already run.

**There is no fixed number of stages.** `job-system.md § 4.1`'s three-rung ladder
is a *default set* — a shipped strategy writes the stages it runs, and a tier it
leaves out is added later from the tier presets (§ 2) — a starting proposal, not
a bound. A fourth and fifth stage decided next month are ordinary, and each
continues from the stage before it, because `restart: continue` means *carry
from the previous stage* (§ 7.1). That is the payoff of the identity being
blind to everything a stage tunes (`execution/run-identity.md § 4`): appending
a stage does not change the id, so the state is still there to continue from.

Two rules make growth safe, and both follow from one observation.

> **A stage has run when the folder holds output keyed to its deck** — the run's
> stdout and its trajectory log (`job-contracts.md § 2.6`). That is a fact on
> disk, not a flag anybody maintains.
>
> *Keyed to* is the load-bearing part: the output sits in the stage's run
> folder (`project-layout.md § 1`), and that is what ties it to the deck.

**R4 — a stage that has run is a record, and the record is a checkpoint.** The
outputs beside a deck were made by that deck as it was, so replacing it without
keeping the old one leaves a folder whose results came from a file that no longer
exists. That is not a reason to refuse the edit — redoing a stage is ordinary
work. It is a reason for the history to exist, which § 7.2 already requires.

So the history keeps it: the folder's state is saved before the acts that
change it — which acts, and the note each writes, are
[`execution/checkpointing.md`](?doc=execution/checkpointing.md) § 9's. What a
launched stage computes is not prepared again — its way back is
[`job-system.md`](?doc=execution/job-system.md)'s *How the checkpoint supports
you*; how it runs may be ([`execution/job-request.md`](?doc=execution/job-request.md) § 5).

> **The obstacle here is cleared** (2026-08-06). `Repo.init` used to refuse any
> directory whose subdirectories held a working-dir marker — and § 7.1's layout
> is exactly such a directory, so the folder this contract specifies could not be
> put under checkpoint at all. It now permits them when the root carries its
> description (`task.json`), which is what says the subdirectories are this
> calculation's stages rather than rival jobs; a directory that declares nothing
> is still refused. See `execution/checkpointing.md` L1.

#### What each checkpoint is called

A history is only worth taking if you can find the point you want in it, and
the naming is small and sits where it belongs:

| | What it is now |
|---|---|
| **the note** | written by the act that saves — the time the state was taken, then what was about to change it, `2026-10-03 14:05:12 · before prep task group 1 (coarse), group 2 (medium, tight)` (`checkpointing.md` § 9); yours, and required, when you save by hand (L3) |
| **the calculation's name** | carried in the state's own `Calculation:` trailer, so a folder opened a year later still says which calculation its history is |
| **a tag** | *yours*, and only yours. Nothing tags on your behalf |

**Why the automatic tags went** (`checkpointing.md` L4): every state already
carried a note saying what happened, written by whoever took the save — so the
tags added no information, and they filled the one namespace you were meant to be
naming things in yourself. A history where most tags are machine-made is one
where your own are hard to find, which is the opposite of what a tag is for.

*(A save offered when a stage finished, its note drafted from the run's record,
was described here until 2026-10-04: no such moment is § 9's, and none is
built.)*

**And re-entering costs no new verb.** The folder stops being *the current state
of one calculation* and becomes a chain of states you can go back into: restore
the state you want, change what you like, and save. The new state's parent is the
one you restored, so both attempts stay listed and the list shows them as
alternatives — that *is* the fork (`checkpointing.md § 7.1`). There is nothing to
declare and nothing to name, which is why the "no branch route" this section used
to call the design's most consequential gap is not a gap at all.

**R5 — a stage's name is its identity, and renaming is rewriting.** The name is
in the deck's filename, in every output beside it, in the notes on the states
that recorded it, and in the detector above. Renaming a stage that has run moves all four at once, so it
is an R4 event and takes an R4 checkpoint.
This is also why the stage's **position in the list** must never appear in a
filename: insert a stage at the front, or reorder two, and every positional
number after it shifts — silently reassigning outputs that already exist to
stages that did not produce them. **Names are stable; positions are not.**

> **A list position and an assigned ordinal are not the same number, and this
> rule only forbids the first.** *Clarified 2026-08-10 with decision 27, which
> puts an ordinal in the filename and would read as violating this rule without
> the distinction.*
>
> | | shifts when the ladder grows? | may it be in a filename? |
> |---|:--:|:--:|
> | **position in the list** — where a row happens to sit today | **yes** | **no** — this rule |
> | **`seq`, assigned once by the produce** (`project-layout.md § 4.2`) | **no** | **yes** |
>
> § 4.2 is what makes the second row true: *"a `seq` is never changed, so a
> stage can only be added at the end"*, and *"insert something between 1 and 2
> is not an insertion — it is a new stage that happens to be coarser, and
> numbering it `03` is the truth."* A number that cannot shift cannot cause the
> failure this rule exists to prevent. So the artifact token is `<NN>_<name>`
> (`identity.stage_token`), carrying **both** halves: the ordinal so a
> listing of eight decks sorts into the order they ran, the name so every
> artifact of a stage can still be read back to it.

Where the shipped trajectory log used `-stage<N>` (§ 7, the log bullet), that
was the half of the naming question growth decides — **resolved 2026-08-10**: it
keys on the token, so a stage's deck and its own log now share a basename.

### 7.4 What the layout costs the checkpoint system

*Settled 2026-08-06. This section proposed three changes to the checkpoint side;
two were made and one came free. It is kept as a pointer rather than a proposal.*

| What this layout needed | Where it now lives |
|---|---|
| **the repository at the parent**, not in each subdirectory — a per-stage repository cannot restore a shared file above it, and cannot express *branch the workflow at stage 2*, because no repository contains the workflow | [`checkpointing.md`](?doc=execution/checkpointing.md) **L1** |
| **archive globs that match at depth** — `*.DM` does not match `<stage>/<label>.DM`, so every big binary would have gone into git as a blob, the exact outcome the archive exists to prevent | [`checkpointing.md`](?doc=execution/checkpointing.md) **L2** |
| **nothing points at a file that does not exist** — this came free from stages not chaining (§ 7.1): whatever a stage continues from was copied in as a real file at that stage's first launch, so a checkpoint always holds real files | [`project-layout.md`](?doc=execution/project-layout.md) § 1.6, invariant 4a |

**The general rule this leaves behind**, which is what belongs in *this* contract:
a layout decision is not finished when the folder is right. It has to be carried
to whatever reads the folder — here the checkpoint system, whose glob defaults
were written for files at one level and would silently have lost data in a tree.

---

## 8. What this contract does not own

- **The environment, activation, and how a wrapper finds its engine** —
  [`execution/running-a-job.md`](?doc=execution/running-a-job.md) §§ 2 and 5.
  Nothing here changes any of it. To restate only what a reader of this document
  needs: molbuilder must be installed on the machine that *generates*; the
  activation form (`conda activate` / `source activate`) and any module preamble
  come from the target machine's record, have **no default**, and generation of
  any wrapper refuses without them; environment *names* are configurable per
  category and must never be hard-coded; `.sbatch` is emitted unless the
  target's record names a scheduler with no queue. Everything site-specific is
  baked at generate/prep, and at run time the wrapper reads only the
  allocation -- what the scheduler granted, in molbuilder's names, said by the
  job's header; for a run here, the wall its launch tells it (`MB_WALL_S`,
  [`running-a-job.md`](?doc=execution/running-a-job.md) § 3.6) -- and the
  hardware.
- **The run id, its normalisation, and the engine's identity group** —
  [`execution/run-identity.md`](?doc=execution/run-identity.md).
- **The run directory, filenames, reserved script blocks, warm-restart files, the
  project tree** — [`execution/job-contracts.md`](?doc=execution/job-contracts.md).
- **What values a stage should carry** —
  [`engines/tuning.md`](?doc=engines/tuning.md).
- **`Job.warm`, `Job.traits`, `Job.resources`, and every scheduler concern** —
  [`execution/job-system.md`](?doc=execution/job-system.md). A producer reads
  this file and turns each stage into a `Job`; **it asks for nothing this
  file does not carry**, because there are no edges left to thread. *(That bullet
  read "the dependency chain, `Job.carry` … and asks for `on_nonconvergence`"
  until 2026-08-11; all three were deleted on 2026-08-10 — § 3.)*
- **Carrying a finished run into the next calculation** — **it is CITED, never
  bundled** (retired 2026-08-29; `job-contracts.md` § 5 holds the closure).  A
  calculation that builds on a finished result names the run explicitly
  (a directory whose files satisfy [`transport.md`](?doc=engines/transport.md) § 3.1) and `prep` composes from it —
  [`archive/2026-09-01-transport-design.md`](?doc=archive/2026-09-01-transport-design.md) § 4.1 is the
  shipped instance.  The which-script tie-break question the old handoff carried
  died with it: a citation names ONE run, so there is nothing to guess.
- **What a checkpoint history must always hold** —
  [`execution/checkpointing.md`](?doc=execution/checkpointing.md). This contract
  says *when* a checkpoint is taken and *what it is called*; that one says what
  must be true of it afterwards, in a form a test can assert.
- **Phasing, status, and what is built when** —
  [`archive/2026-08-19-staged-runs-implementation-plan.md`](?doc=archive/2026-08-19-staged-runs-implementation-plan.md) and
  [`plans/plan.md`](?doc=plans/plan.md) — the *status-lives-in-the-plan* rule,
  not § 4's R3.

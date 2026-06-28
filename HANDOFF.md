# HANDOFF — Job execution workflow & toolbox

You are continuing **the job-execution work** on molbuilder
(repo `/home/qqing/molbuilder`, branch `main`). This file is scratch; the
durable truth is the docs + task list it points at. Read this whole file,
then **obey the READ-FIRST gate before doing anything**.

---

## 0. THE JOB (one line) + scope

Make the **job-execution toolbox** ready: molbuilder generates a *self-running
script bundle* → you copy it to a target → a `prep` step detects the machine
and formats the run → the wrapper runs itself (activates its own conda env,
runs the engine, handles warm/cold restart). The core pipeline is BUILT; the
remaining work is closing the gaps in § 3.

**What job execution IS:** the *generic* machinery to **run any prepared
script on a target** — engine-agnostic AND calculation-agnostic. It does not
know or care what the script computes. It owns: `runwrap` (self-activating
wrapper), `prep` (detect the machine), scheduler adapters, the monitor, the
portable bundle, the benchmark sweep, and production runs.

**What job execution IS NOT (out of scope — do NOT pull these in):** the
*science* modules that PREPARE inputs — `transport` (NEGF; multi-backend:
SIESTA/PySCF/future VASP), `optimization`, `spectra`. They produce `.fdf`/`.py`
for a scientific purpose and are tracked under their OWN modules/roadmaps.
Note: "TranSIESTA" is not an engine — it is SIESTA run with a transport
`.fdf`, so a transport job is just a SIESTA job to the runner. **Reason about a
thing's role/layer, not its name** (see memory `reason-by-role-not-surface-terms`).

**LASER FOCUS — the only goal is a ready toolbox, in priority order:**
- **P1 = #25** prep tells the user if the target is ready (readiness pointer).
- **P2 = #24/#25** ONE bundle runs on workstation AND HPC without regeneration.
- **P3 = #5** hardening (only after P1+P2).

Do them in order. Do not start anything outside this list — no science
modules, no web UI, no new features. If a task isn't in § 3, it is out of scope.

---

## 1. READ-FIRST GATE (do this BEFORE proposing or writing anything)

Past sessions failed by acting while clueless. Do NOT. In order:

1. **`memory/MEMORY.md`** — the rules index. Especially: *read-before-claiming*,
   *reason-by-role-not-surface-terms*, *framework-first / no reinventing*,
   *static-review-first*, *align-before-act*, *assistant-not-nanny*,
   *design.md is source of truth*, *run under correct env*, *no pip install -e*,
   *commit author/trailers*.
2. **`docs/design.md`** — the Stance ("assistant, not nanny": easy but explicit,
   never push-button; don't twist the env/recipe) + the numbered Design
   Principles + Anti-patterns (no custom frameworks/registries).
3. **`docs/job-execution.md`** — THE SOLE SOURCE OF TRUTH for this work. Read it
   *in full*: §1 big picture, §2 workflow, **§3 the detection/standalone
   contract**, §4 cookbook, §5 the sub-doc map, §6 roadmap.
4. **The sub-doc that owns your task's detail** (from §5 map):
   - config schema + wrapper contract → `docs/config.md` §§1–8
   - self-running wrapper, warm/cold restart per engine → `docs/protocols/script-execution.md`
   - SLURM/sbatch/Sol facts, CUDA floor, GPU gate → `docs/protocols/slurm-integration.md`
   - benchmark workflow stages, probes/adapters, data formats → `docs/protocols/benchmark-workflow.md`
   - on-disk naming → `docs/protocols/job-layout.md`
5. **The actual code for your task** — grep + read it; run the relevant
   `molbuilder <group> --help`. Never claim a fact you have not just verified.

Then run `TaskList` and pick the highest-priority unblocked task.

---

## 2. DISCIPLINE (non-negotiable)

- **READ-FIRST / NO REINVENTING.** Check what exists first. The canonical trap:
  do **not** build a readiness/doctor/env-checker — it EXISTS as
  `molbuilder envs doctor` + `molbuilder envs validate`. Reuse it.
- **REASON BY ROLE, NOT NAME.** Before claiming two things are the same or one
  belongs to another, state each one's purpose/owner/layer; if they differ,
  they're different even when they share a word. (This session's whole mess
  came from collapsing transport-module / TranSIESTA / job-execution by name.)
- **STATIC-REVIEW-FIRST.** Read the code AND the generated product/script
  before executing anything.
- **ALIGN-BEFORE-ACT.** Propose the change in words, get an explicit "go", ship
  ONE thing. No trailing "want me to also…?". No same-day self-reversals.
  For tasks that change the **contract** (e.g. #24/#25), update
  `job-execution.md` in the SAME change as the code.
- **ASSISTANT, NOT NANNY.** Surface support/hints; don't own the recipe,
  auto-decide, auto-install, or silently twist the env. Easy but EXPLICIT.
- **ENV/REPO HYGIENE.** Run each command under its category's conda env (unit
  pytest → `molbuilder`; backend execution → its named env, e.g.
  `molbuilder-siesta-gpu`); never `pip install -e .` (use `python -m molbuilder`);
  no PYTHONPATH/sys.path hacks. **Commit only when asked**; author
  `Quan <qqing@asu.edu>`, NEVER `Co-Authored-By: Claude`; never push under
  `.github/workflows/*`.
- **VERIFY EVERY CLAIM — yours and any agent's.** ~half of subagent claims
  don't survive a check (this session a survey wrongly reported a feature
  "done"). Verify in the code before acting.

---

## 3. GROUNDED STATE — verified this session (do NOT re-derive)

**Vocabulary:** "deployment" = serving the molbuilder *app* (`deployment.md`).
Using the script-generator *module* to run calculations = "job execution"
(this work). Don't conflate.

**BUILT (the core pipeline):** `bench generate` (host, bundle) → `prep-bench`
(target detect+format) → run+monitor → `summarize` → `prep-run` → production.
Shared core: `bench/{generate,environment,adapters,result,prep,prep_run,
summarize}.py` + `runwrap.py` (self-activating, warm/cold restart) +
`monitor.py`. Scheduler adapters: slurm + workstation. Activation: baked at
generate time — workstation autodetected (`bench/generate.py::_ensure_activation`
→ writes `.molbuilder.json`), HPC explicit (`--activation`/`--preamble` or the
`asu-sol` example config). The *core* generator refuses-to-emit without
`activation` (config.md §2/§4); the bench front-end autodetects+writes it on a
workstation.

**REMAINING WORK — three open tasks (see `TaskList` for full detail):**
- **P1 · #25 readiness gate** (small): `bench/prep.py::_summary` does NOT yet
  surface `envs doctor` / `envs validate`. Add the pointer (surface only,
  reuse the envs toolkit). Owner doc: job-execution.md §3.4 + §6 item 1.
- **P2 · #24/#25 one portable bundle** (large, THE goal): activation is baked
  at generate → a bundle is locked to one target. Move activation resolution to
  **prep-time on the target** so ONE bundle works on workstation + HPC. Changes
  the contract (relaxes job-execution.md §3.3 row C for activation only) →
  design against `job-execution.md §3` and update the doc in lockstep.
- **P3 · #5 hardening** (after P1+P2): test the §3.5 activation defaults;
  end-to-end static-review + validation of prep→bench→run.

**Done this session:** consolidated all job-execution docs into the single
master `docs/job-execution.md` (moved config.md §9 → §3; folded+deleted the old
cookbook; reconciled the activation layering note), then **removed a
transport↔job-execution conflation** I had introduced (transport is a separate
science module, not job execution). See `git log` for the doc series. Task list
is clean: #1 done; #2 (#25), #3 (#24/#25), #5 (hardening) open.

---

## 4. START HERE

**#25 (P1) first** — small, self-contained, pure surface. Then design
**#24/#25 (P2)** in words and get a go before coding (it changes the contract).
**#5 (P3)** only after P1+P2.

After the READ-FIRST gate, propose the chosen task in words, wait for the
explicit "go", ship ONE thing. This file is scratch — delete it once absorbed,
or ask to keep it tracked.

# HANDOFF — Job execution workflow & implementation

You are continuing **the job-execution work** on molbuilder
(repo `/home/qqing/molbuilder`, branch `main`). This file is scratch; the
durable truth is the docs + task list it points at. Read this whole file,
then **obey the READ-FIRST gate before doing anything**.

---

## 0. THE JOB (what this is, in one line)

Build out the **job-execution system**: molbuilder generates a *self-running
script bundle* → you copy it to a target → a `prep` step detects the machine
and formats the run → the wrapper runs itself (activates its own conda env,
runs the engine, handles warm/cold restart). It serves **benchmark,
transport, and plain production** runs — the same machinery underneath. The
core pipeline is BUILT; the remaining work is closing four gaps (§ 3).

**LASER FOCUS — the only goal is a *ready job-execution toolbox*.** "Ready"
= the P1→P2→P3 chain done: (P1) prep tells the user if the target is ready,
(P2) ONE bundle runs on workstation AND HPC without regeneration, (P3)
transport rides the same toolbox. Do these IN PRIORITY ORDER. Do NOT start
P4 hardening, or any task outside this list, until P1–P3 land — and do not
wander into transport science, web UI, or new features. If a task isn't on
the §3 list, it is out of scope for this work.

---

## 1. READ-FIRST GATE (do this BEFORE proposing or writing anything)

The previous sessions failed by acting while clueless. Do NOT. In order:

1. **`memory/MEMORY.md`** — the rules index. Especially: *read-before-claiming*,
   *framework-first / no reinventing*, *static-review-first*, *align-before-act*,
   *assistant-not-nanny*, *design.md is source of truth*, *run under correct env*,
   *no pip install -e*, *commit author/trailers*.
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
- **VERIFY AGENT CLAIMS.** If you delegate, ~half of subagent claims don't
  survive a check (this session: an agent wrongly reported #37 "done"). Verify
  in the code before acting on any finding.

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
→ writes `.molbuilder.json`), HPC explicit (`--activation`/`--preamble` or
`asu-sol` preset). The *core* generator refuses-to-emit without `activation`
(config.md §2/§4); the bench front-end autodetects+writes it on a workstation.

**REMAINING WORK — the four open tasks (see `TaskList` for full detail):**
- **P1 · #25 readiness gate** (small): `bench/prep.py::_summary` does NOT yet
  surface `envs doctor` / `envs validate`. Add the pointer (surface only).
- **P2 · #24/#25 one portable bundle** (large, THE goal): activation is baked
  at generate → a bundle is locked to one target. Move activation resolution to
  **prep-time on the target** so ONE bundle works on workstation + HPC. Changes
  the contract (relaxes job-execution.md §3.3 row C for activation only) → design
  against `job-execution.md §3` and update the doc in lockstep.
- **P3 · #37 transport → shared runwrap** (medium; blocked by P2): `transport
  bundle` (`orchestrate.py::build_transport_bundle` → `render_driver:173`)
  emits its OWN `run-transport.sh` with manual `conda activate`, no scheduler
  adapter, no monitor, no warm/cold restart — it does NOT ride the system
  (job-execution.md §4.5 documents this gap). The single-job wrapper path
  (web `build.py` + `cli.py` + `bench/generate.py`) uses runwrap; the
  multi-run transport driver does not. Rebuild it on runwrap+adapter.
- **P4 · hardening** (optional): test the §3.5 activation defaults; end-to-end
  static-review + validation of prep→bench→run.

**Done this session:** consolidated all job-execution docs into the single
master `docs/job-execution.md` (moved config.md §9 → §3; folded+deleted the
cookbook; reconciled the activation layering note; trimmed the readiness
overlap). Commits: `40b63b2` (backup), `31d8ebd` (consolidation), `793d404`
(reconcile). Task list rebuilt (#1 done; #2–#5 open).

---

## 4. START HERE

Recommended order: **#25 (P1) first** — small, self-contained, pure surface.
Then design **#24/#25 (P2)** in words and get a go before coding (it changes
the contract). #37 (P3) follows P2. #5 (P4) opportunistically.

After the READ-FIRST gate, propose the chosen task in words, wait for the
explicit "go", ship ONE thing. This file is scratch — delete it once absorbed,
or ask to keep it tracked.

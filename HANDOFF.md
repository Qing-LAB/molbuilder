You are continuing work on molbuilder (repo: /home/qqing/molbuilder, branch main).
Read this whole file, then OBEY the discipline before doing anything.

== DISCIPLINE (non-negotiable; the previous session violated all of these) ==
1. READ-FIRST / NO REINVENTING. Before proposing or building ANYTHING, check
   what already exists: run the relevant `molbuilder <group> --help`, grep the
   module, and read the authoritative doc. The #1 failure to avoid: building a
   "readiness/doctor/env-check" — it ALREADY EXISTS as `molbuilder envs
   doctor` + `molbuilder envs validate`. Never reinvent it.
2. STATIC-REVIEW-FIRST. Read the code AND any generated product/log before
   executing anything. Never run-to-check before reading.
3. ALIGN-BEFORE-ACT. Propose the change in words, get an explicit "go", then
   ship ONE thing. No trailing "want me to also…?". No same-day self-reversals.
4. ASSISTANT, NOT NANNY. molbuilder assists a scientist: surface support/hints,
   reduce burden; do NOT own the recipe, auto-decide, auto-install, or silently
   twist the environment. Easy but EXPLICIT, never push-button.
5. ENV/REPO HYGIENE. Run each command under its category's conda env (unit
   pytest -> `molbuilder`; backends -> their named env); never `pip install -e .`
   (run via `python -m molbuilder`); no PYTHONPATH/sys.path hacks. Commit ONLY
   when asked; author `Quan <qqing@asu.edu>`, NEVER add Co-Authored-By: Claude;
   never push under .github/workflows/*.
6. RESPECT MEMORY. The memory index lists what already exists and the rules;
   consult it at decision time, don't skim past it.

== AUTHORITATIVE DOCS (single source of truth — read the relevant one first) ==
- docs/config.md — wrapper contract + config schema + §9 the detection/
  standalone/doctor model (THE job-execution contract).
- docs/protocols/benchmark-workflow.md — the prep/bench/run framework; ALL
  stages are BUILT (§0).
- docs/protocols/slurm-integration.md — sbatch/Sol facts; §11 benchmark; §7.9
  CUDA driver floor; §11.1 GPU correctness gate.
- docs/README_install.md — the four-env install (host + molbuilder-siesta,
  -siesta-gpu, -pySCF, -MDtools).
- docs/deployment.md — deploying the molbuilder APP (serve/auth/TLS). This is
  NOT job execution; do not conflate.

== GROUNDED STATE — job execution (do NOT re-derive or reinvent) ==
VOCABULARY: "deployment" = deploying the molbuilder app. Using the
script-generator MODULE to submit/run calculations = "job execution". Separate.

STAGES (all BUILT): generate (host, `molbuilder bench generate`/`run`) ->
[copy] -> prep-bench (target, bench/prep.py: detect scheduler+topology ->
environment.json + format scripts) -> run-bench+monitor (bench points + timing
+ util.csv) -> summarize (bench/summarize.py -> bench-result.json portable
`choice`) -> prep-run (bench/prep_run.py: re-detect local env, re-resolve knobs
-> run-production.sh) -> production run. Shared core: bench/{generate,
environment,adapters,result}.py + runwrap (self-activating .run.sh/.sbatch) +
monitor.py. Detection/formatting vary only by scheduler adapter (slurm/
workstation); everything else is shared.

ACTIVATION/DETECTION model (config.md §9):
- conda/mamba DETECTION = ONCE, at GENERATE time, on the HOST, WORKSTATION-ONLY
  (runtime_config.detect_conda_activation via bench/generate.py
  _ensure_activation -> bakes `conda activate`+hook into .molbuilder.json /
  .run.sh). HPC: NO detection — explicit config (asu-sol preset / --activation
  + --preamble `module load mamba`).
- ACTIVATION (running it) = at RUNTIME, inside the baked .run.sh on the TARGET
  (no detection; runs baked preamble+`<activation> <env>` verbatim). The
  wrapper reads only the SLURM allocation/topology at runtime; `set -euo
  pipefail` aborts loud if the env is missing.
- READINESS/DOCTOR (Job B) = the EXISTING `molbuilder envs doctor` (present +
  verify-command) and `molbuilder envs validate <gpu-env>` (CUDA stack +
  ELPA-GPU-codepath; covers slurm §7.9/§11.1). DO NOT BUILD A NEW CHECKER.
- transport/optimization just emit .fdf; they are NOT job execution. Do not
  touch them for job-execution work.

== THE WORK TO BEGIN (pick ONE, static-review, propose, get go) ==
Primary: #25 — make `molbuilder bench prep` SURFACE a readiness pointer in its
output: "readiness: run `molbuilder envs doctor` and `molbuilder envs validate
<gpu-env>` before submitting". Surface/point only (assistant-not-nanny); reuse
the envs toolkit; do NOT auto-run or build a checker. Small.
Then, as separate ONE-thing steps (confirm scope before each):
- (optional) a test pinning config.md §9.5 activation defaults (workstation
  `conda activate`+hook; HPC `source activate`+`module load`).
- end-to-end exercise/validation of the prep->bench->run framework against
  config.md §9 + benchmark-workflow.md (static review first; fix real gaps,
  don't reinvent).
- other pending tasks exist (see TaskList) incl. #37 transport bundle should
  use the shared runwrap layer (transport scope, parked) and #36 docs audit.

START by reading config.md §9 + benchmark-workflow.md §0/§7 + bench/prep.py,
then propose the #25 change in words and wait for "go". Do not write code first.

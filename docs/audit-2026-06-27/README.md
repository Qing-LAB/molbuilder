# Audit 2026-06-27 — SLURM/Sol session self-review + benchmark ground-truth

Self-review of the five commits shipped this session, plus the Sol
benchmark results that landed the same day. Three parallel adversarial
reviewers (general-purpose agents) produced candidates; **every candidate
below was verified by hand / by execution** before being recorded here
(per `docs/protocols/code-audit.md` § 1: agent findings are candidates).

**Commits under review** (all pushed, `bf7a6aa..eec7199`):

| SHA | Subject |
|---|---|
| `dfdfcd2` | estimator: recalibrate `--mem` against np=64 OOM/survival data |
| `01ba00d` | monitor: shorter wake + quiet/correct stall handling |
| `5c919fe` | bench: add `molbuilder bench generate` (CPU + GPU bundles) |
| `4c01037` | runwrap: runtime `[INFO]` memory estimate-vs-allocation audit (P2) |
| `eec7199` | runwrap: GPU↔CPU socket co-location in per-rank launcher (§ 7.5.2) |

---

## Part A — Sol benchmark ground-truth (2026-06-27)

The real run both **validated the memory model** and **exposed a bench
bug**. This data drives the fix decisions in Part B.

### A.1 The jobs

`sacct -u $USER -X` + step-level `TRESUsageInMax`. The four "extended"
jobs are the real SIESTA work; the rest were debug iterations.

| JobID | Name | Elapsed | ReqMem | Peak `mem=` | State | Meaning |
|---|---|---|---|---|---|---|
| 57519530 | cpu-np64 | 0:00:52 | 128 G | 128.00 G | OOM | demand **> 128** (killed at ceiling) |
| 57519775 | cpu-np64 | 4:00:27 | 240 G | 239.61 G | TIMEOUT | **throttled** — pinned at the 240 cap, crawled, never reached peak |
| 57522405 | cpu-np64-largemem | 0:00:48 | 320 G | 320.00 G | OOM | demand **> 320** (killed at ceiling) |
| 57522630 | cpu-np64-largemem | 4:00:24 | 480 G | **433.15 G** | TIMEOUT | **unthrottled survivor → true peak ≈ 433 G** |
| 57518705 | gpu-k8 | 2:08:48 | 64 G | 25.23 G | FAILED | ~5 iters, then non-zero exit (see A.3) |
| 57518706 | gpu-k4 | 2:42:39 | 64 G | 22.28 G | FAILED | ~5 iters, then non-zero exit |

**Reading `mem=` correctly:** for an OOM job, `mem=` shows the *ceiling it
was killed at* (`128.00`, `320.00`) — a *lower bound* on demand, not the
demand. Only the unthrottled survivor (480 G alloc, peak 433.15 G) gives
the real number.

### A.2 Memory model — VALIDATED, no coefficient change

np=64 true peak ≈ **433 G** (the 480 G survivor). Our estimator:

- raw model (pre-safety) np=64 = `base 2 + dense 135.4 + mesh 1.6 + repl
  (c_rank 3.5 × 64 = 224)` = **363 G**
- × `safety 1.3` = **472 G** (the `#SBATCH --mem` we generate)

So the **raw model under-predicts the real peak (363 < 433, −16 %)**, and
the **safety factor lifts it into the safe band: 433 < 472 < 480**
(survives) and 472 > 320 (the OOM). The estimator does exactly its job;
`safety=1.3` is load-bearing — without it we'd request 363 G and OOM.

**Decision:** do NOT change `c_dense`/`c_rank`/`safety`. Record 433 G as a
third real anchor in `test_siesta_memory.py` + memory. (Resolves Part B
finding #6 — the data shows the calibrated point is safe; the *scientific*
caveat about gamma/other-k remains a documented limitation, not a number
to tweak.)

**Throttle insight:** the 240 G run timed out *partly because it was
under-allocated* — memory pressure throttled it (RSS held at the 240 cap,
SCF crawled). The auto-estimator (which requests 472 G) prevents both the
OOM *and* this throttle-to-timeout. A manual `--mem` that is too low is
worse than the estimate.

### A.3 GPU — host RAM + the CPU-vs-GPU verdict

- **GPU host RAM is tiny:** gpu-k8 peaked 25.2 G, gpu-k4 22.3 G (of 64 G).
  The `gpu.mem: 64G` default is safe and generous; could trim to ~32 G.
- **Timing:** same 5 capped iters → gpu-k8 2:08:48 (~1546 s/iter) vs gpu-k4
  2:42:39 (~1952 s/iter). **More ranks/GPU wins; gpu-k8 is the production
  point.** (Exact iter counts to be confirmed from `*.scf-timing.log`.)
- **CPU never produced a clean measurement** at 444 atoms: it either OOM'd
  or (when given enough memory) timed out at the 4 h walltime. **Verdict:
  GPU decisively at this size.**

### A.4 Why every point read FAILED — a real bench bug

SIESTA's `SCF.MustConverge` defaults to `.true.`. The bench caps
`MaxSCFIterations 5`, so SIESTA hits the cap unconverged and **aborts with
a non-zero exit** → the wrapper faithfully propagates it → SLURM marks the
job `FAILED`. The wrapper is correct; the bench fdf is wrong to leave
`MustConverge` on for a deliberately-capped run. (→ Part B fix #BENCH-1.)

Secondary: the generated CPU bench walltime (4 h) is too short for a CPU
point at this size, and a too-low manual `--mem` throttles it (A.2).

---

## Part B — Verified review findings

Severity: BLOCKER / IMPORTANT / NIT. Each was reproduced. "Verified-FALSE"
items (agent flagged, hand-check disproved) are listed in § B.NOTES so we
don't re-litigate them.

### IMPORTANT

#### B-1 — Socket-pin over-binds to NUMA nodes the rank doesn't own
- **Where:** `molbuilder/runwrap.py`, `_gpu_socket_affinity_block`
  (commit `eec7199`).
- **Problem:** `_sock_numas` is built by scanning **all** system NUMA
  nodes whose `physical_package_id` == the GPU's socket — *not* restricted
  to nodes the rank's cpuset actually owns. The pin fires whenever the
  cpuset spans >1 socket, which the docstring equates with `--exclusive`
  but is **not** equivalent on Sol's 8-NUMA-per-2-socket A100 nodes.
- **Reproduction (agent, fake 8-NUMA sysfs):** rank owns cpu `0-7` (numa0,
  socket0) + `32-39` (numa4, socket1), GPU on numa5/socket1 →
  `_pin='numactl --cpunodebind=4,5,6,7 --membind=4,5,6,7'`. The rank owns
  only numa4 on socket1 but binds CPU+mem to numa 4,5,6,7 — three nodes it
  wasn't allocated.
- **Impact:** `--cpunodebind` is saved by the kernel cpuset intersection,
  but `--membind` is a **hard** bind to (possibly unowned) NUMA memory →
  isolation breach / contention / OOM risk. Bites the `exclusive:false`
  GPU sweep (the Sol preset's benchmark mode). Under true `--exclusive`
  (whole node owned) it is correct.
- **Secondary:** even on a legitimately exclusive node, `--membind=<one
  socket>` is hard — a rank whose host working set exceeds that socket's
  RAM (~250 G of a 503 G node) is OOM-killed with the other socket free.
  (GPU host RAM is ~25 G per A.3, so low real risk, but the strictness is
  latent.)
- **Fix:** build the *owned* NUMA set from the cpuset (the numa node of
  each owned cpu), intersect with the GPU socket's NUMA nodes, and bind
  only to that intersection; warn (no pin) if the intersection is empty.
  Consider `--preferred`-style binding for memory to avoid the hard-fail.

#### B-2 — `bench generate` directive surgery is not spelling-normalized
- **Where:** `molbuilder/bench/generate.py`, `_set_or_append` /
  `_remove_directive` (commit `5c919fe`).
- **Problem:** match anchors on the exact canonical spelling
  (`re.escape(anchor)`), but SIESTA treats `.` `-` `_` as interchangeable
  and labels as case-insensitive.
- **Reproduction:** input with `DM-UseSaveDM .true.` + `Diag-ELPA-GPU
  .true.` → CPU bundle keeps the original `DM-UseSaveDM .true.` AND
  appends a conflicting `DM.UseSaveDM .false.`; `Diag-ELPA-GPU` is **not**
  stripped → the "plain diagon" CPU job still asks for ELPA-GPU. SIESTA
  honors the first occurrence, so the cold-start override is silently
  ignored — the jobs are no longer cold/comparable, no warning.
- **Impact:** violates the "no silent absorption / warn on labels it can't
  consume" three-stage contract. Bounded by "molbuilder-generated
  (canonical) input," but `cmd_generate` accepts any `.fdf`.
- **Fix:** match `[._-]`-tolerant + case-insensitive in both the
  existence-check and the strip; or validate/warn on non-canonical
  spellings at the CLI boundary.

#### B-3 — `bench generate` copies only `*.psml`, not legacy pseudos
- **Where:** `molbuilder/bench/generate.py` (`*.psml` glob).
- **Problem:** `molbuilder/pseudos.py` states SIESTA accepts `.psml` *or
  legacy `.psf`*. A project using `.psf`/`.vps` produces a bundle missing
  its pseudos → runtime failure on the cluster, and the CPU `--mem`
  estimator loses its valence input — silently.
- **Reproduction:** source dir with `C.psf` → bundle does not contain it.
- **Impact:** bounded (molbuilder defaults to `.psml`), but a real
  run-time failure for `.psf` projects.
- **Fix:** widen the glob to `*.psml`, `*.psf`, `*.vps` (+ `.psp8` if used).

#### B-BENCH-1 — capped bench jobs exit non-zero → `FAILED` (from A.4)
- **Where:** `molbuilder/bench/generate.py`, `transform_fdf`.
- **Problem:** bench caps `MaxSCFIterations 5` but leaves
  `SCF.MustConverge` at its `.true.` default → SIESTA aborts unconverged
  with a non-zero exit → SLURM `FAILED`, and accounting is messy.
- **Evidence:** every Sol bench point read `FAILED`/`TIMEOUT` (Part A).
- **Fix:** emit `SCF.MustConverge .false.` in both bundles. Capped runs
  then exit 0 / `COMPLETED` with clean `MaxRSS`.

#### B-BENCH-2 — CPU bench walltime too short / not tunable (from A.4)
- **Where:** `molbuilder/bench/generate.py` + `_cli.py`.
- **Problem:** generated CPU bench inherits the 4 h default; a CPU point at
  444 atoms can't finish 5 iters in 4 h (both CPU survivors TIMEOUT'd).
- **Fix:** expose `--time` on `bench generate`; default the CPU bundle
  higher (and rely on the auto `--mem` so it isn't memory-throttled).

### NIT

#### B-4 — runtime mem-audit bakes `fixed_gb` from rounded components
- **Where:** `runwrap.py` `_build_mem_audit` → `_siesta_mem_audit_block`.
- **Problem:** `fixed_gb = est.base_gb + est.dense_gb + est.mesh_gb`, each
  `round(.,1)`, whereas `estimate_siesta_memory` sums the **unrounded**
  components before one `ceil`. Verified by a 188 k-combo sweep: **6.3 %
  of cases differ, worst case exactly 1 GB, never > 1 GB.** The 1 GB can
  land on the wrong side of the `est > alloc` WARN test exactly at the
  boundary, silently dropping a borderline OOM warning.
- **Fix:** bake `fixed_gb` from the unrounded sum (recompute dense/mesh
  without `round`, or add `fixed_gb_raw` to `MemEstimate`).

#### B-5 — geometry-move regex is loose
- **Where:** `monitor.py` `_GEOM_LINE = Begin\b.*\bmove\b\D*([0-9]+)`
  with `IGNORECASE`.
- **Problem:** `IGNORECASE` + greedy `.*` can match contrived narrative
  ("This will begin to move 8 things" → 8). Since the parser takes the
  last tail match as the current move, a stray false-positive after the
  real last move could momentarily flip a stalled job to "progressing."
  Low real-world probability; all real SIESTA move lines match correctly
  and `"Beginning the move to 9 atoms"` is correctly rejected.
- **Fix:** tighten to `^\s*Begin\b.*\bmove\b\s*=\s*([0-9]+)` (drop
  `IGNORECASE`).

#### B-6 — `SLURM_MEM_PER_CPU` branch is whole-job, not per-node
- **Where:** `runwrap.py` `_siesta_mem_audit_block`
  (`SLURM_MEM_PER_CPU * NTASKS * CPUS_PER_TASK`).
- **Problem:** correct for single-node (the target), but `SLURM_NTASKS` is
  whole-job, so on a multi-node alloc this is the total across nodes while
  the `SLURM_MEM_PER_NODE` branch is per-node — mismatched, would miss a
  warn. Also a minor GB-vs-GiB mix (estimate `/1e9` vs alloc `/1024`),
  within safety noise.
- **Fix:** use `SLURM_NTASKS_PER_NODE` (or derive per-node CPU count), or
  document single-node-only. Low priority (v1 is single-node, `-N 1`).

### Scientific caveat (documented limitation, not a code bug)

#### B-7 — dense term's full `×n_kpoints` is a single-point calibration
- **Where:** `molbuilder/siesta/memory.py`, dense term + `c_dense`.
- **Observation:** at np=4 the dense term is ~88 % of the pre-safety
  total. Under ScaLAPACK + `Diag.ParallelOverK`, SIESTA streams k-points
  through distributed buffers rather than holding all 16 at once — so the
  literal `N_orb²·n_kpoints` over-attributes the cost to k-points; the real
  ~162 G is also driven by k-independent sparse H/S/DM + mesh + workspace.
  `c_dense=2.4` is an *effective* coefficient fit at exactly one k-grid
  (16). Consequence: a **gamma-only** large system collapses the dense
  term ~32× and could **under-estimate** (np=4 gamma → ~28 G), since the
  k-independent baseline doesn't vanish.
- **Also:** `estimate_norb` has two offsetting Au errors — semicore
  same-`l` collapse (`_parse_valence_config` uses a `set`, merging 5s+6s →
  undercounts ~3 orbitals/atom) and polarization placed at `l_max+1=f`
  (overcounts ~4/atom vs the physical 6p). They nearly cancel for Au
  (net N_orb ≈ 14848, hand-verified), so the calibration holds *for
  Au-heavy systems*; the cancellation differs for other compositions.
- **Why not a number to tweak now:** Part A validated the *calibrated
  point* (k-sampled metal junction) against real RSS. The exposure is
  other regimes (gamma-only, very different composition / k-count), which
  the safety factor + node cap make conservative-or-safe in practice.
- **Future fix (optional):** split a k-independent baseline
  (`~c_sparse·N_orb·nnz` or mild `N_orb^1.x`) from a small genuinely
  per-k buffer scaling with `n_kpoints/ntasks` (floored at 1); count
  orbitals per `(n,l)` shell and polarize the highest *occupied* shell.
  Re-calibrate against ≥2 systems incl. a gamma run before changing.

### § B.NOTES — verified FALSE / actually-correct (do not re-flag)

- **`_build_mem_audit` GPU skip** — correct. Mirrors the real GPU-mode
  decision (`env is None and _fdf_requests_gpu`); an explicit `--env`
  runs CPU-mode and a CPU audit is consistent. GPU via `--gres` → skipped.
- **Affinity-block guard paths** — all degrade to no-pin/no-warn (GPU numa
  unknown, missing `numactl`, empty `node[0-9]*` glob).
- **`gpu-gpu-sweep.sh` math/submit** — `-n=K*G`, `-c=cores/socket/K`,
  validity gate, and the `${line%%   #*}` note-strip before `eval` all
  correct; no K∈{1,2,4,8} yields `-c 0` at 24 cores.
- **bench routing + mem** — CPU fdf → `molbuilder-siesta`, GPU fdf →
  `molbuilder-siesta-gpu`; CPU sbatch picks up the recalibrated `--mem`.
- **Monitor progress/stall classification** — `_progressed` + terminal
  `progressing=True` + `changed` predicate sound; `--stall-heartbeat 0`
  silences; terminal job reports the whole-run average. "Last tail match
  wins" gives the highest move (SIESTA prints in increasing order).
- **Endpoint-sampled cpu ranges** — correctly classify `0-47` as two
  sockets; >2-socket middle-socket miss is a documented dual-socket-only
  limitation (correct-but-slow, never a wrong pin... except via B-1).
- **mem-audit awk vs Python** — reproduces the floored/capped/ceil
  structure; only the rounding NIT B-4 (≤1 GB) differs.

---

## Part C — Action plan

Ordered; the Sol data already decided the estimator question (no change).

1. **Record the 433 G anchor** — `test_siesta_memory.py` comment + the
   SLURM memory note. (Done-style: documents validation, no model change.)
2. **B-BENCH-1** `SCF.MustConverge .false.` in `transform_fdf` + test.
3. **B-BENCH-2** expose `--time` on `bench generate`; higher CPU default.
4. **B-2** spelling-normalized directive surgery + test (variant-spelled
   input).
5. **B-3** widen pseudo glob + test.
6. **B-1** socket-pin owned-NUMA intersection + functional test (partial
   multi-NUMA cpuset → no over-bind).
7. **B-4** unrounded `fixed_gb`.
8. **B-5** tighten geom regex.
9. **B-6** — defer (single-node v1); leave a doc note.
10. **B-7** — defer; documented limitation, revisit with a gamma anchor.

Test suite green → commit per logical group → push.

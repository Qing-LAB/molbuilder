# Benchmark analysis — `point-G1K1C4` (single rank on one A100)

**Run:** ASU Sol, node `sg038`, SLURM job `57852378`, 2026-06-29.
**Verdict:** the GPU is **starved**, not the bottleneck — one MPI rank cannot
feed an A100. This is the empirical "K=1 is unusable" baseline; throughput
comes from **ranks (K)**, not threads (c). Details below.

---

## 1. The system

A **benzene-dithiol (BDT) molecule bonded to an Au(111) slab** — the canonical
molecular-junction test case.

| property | value |
|---|---|
| Atoms | **444** |
| Species | 4 — H, C, S, Au (`ChemicalSpeciesLabel` Z = 1, 6, 16, 79) |
| Basis | **TZP** (triple-ζ + polarization), PAO |
| **Orbitals (unit cell)** | **10,960** (`initatomlists`) — *note: ~2.5× the bench-marks header's `n_orbitals_est 4440`; the estimate was low for TZP on a metal slab* |
| Auxiliary supercell | 11,988 atoms / 295,920 orbitals (for the **sparse** real-space H/S; not the dense solve) |
| k-points | **4 × 4 × 2 Monkhorst–Pack = 32 k-points** |
| XC | GGA / PBE |
| Mesh cutoff | 400 Ry |

The cost driver: each SCF step must diagonalize a **dense complex 10,960 ×
10,960 Hamiltonian at *every one* of the 32 k-points.**

## 2. How it was set up (the benchmark transform)

The bundle's generator turns the source `.fdf` into a **timing probe** — a
cold, single-point, iteration-capped run so each point measures *wall-time per
SCF iteration*, not convergence (job-execution.md § 8.12; the transform is
`bench/generate.py::transform_fdf`):

| knob | value | why |
|---|---|---|
| `SolutionMethod` | `diagon` | dense diagonalization (the O(N³) path the GPU accelerates) |
| `Diag.Algorithm` | `ELPA-1STAGE` | same eigensolver for CPU and GPU points |
| `Diag.ELPA.GPU` | **`.true.`** | the GPU point: ELPA-CUDA on the A100 (the CPU baseline sets this `.false.`) |
| `Diag.ParallelOverK` | `.true.` | distribute the 32 k-points across MPI ranks |
| `MaxSCFIterations` | **5** | iteration cap (timing, not convergence) |
| `SCF.MustConverge` | `.false.` | finish the capped iterations cleanly, no abort |
| `DM.UseSaveDM` | `.false.` | **cold** start (no warm DM reuse) |
| `MD.NumCGsteps` | `0` | single-point (no geometry relaxation) |
| `BlockSize` | 256 | ScaLAPACK/ELPA block size |

Engine: **SIESTA 5.4.2**, MPI + OpenMP build, env `molbuilder-siesta-gpu`
(the ELPA-CUDA stack).

## 3. The run configuration — `G1K1C4`

| axis | value | meaning |
|---|---|---|
| G = 1 | `--gres=gpu:a100:1` | one A100 (of the node's 4) |
| K = 1 | `-n 1` | **one MPI rank** = one "app" feeding the GPU |
| c = 4 | `OMP_NUM_THREADS=4` | 4 host OMP threads for that rank |

MPS auto-disabled (1 rank/GPU → no concurrency to gain). Submitted to `htc`
(4 h wall). Launch (from the run log):
`mpirun -np 1 … siesta` , `1 rank × 4 OMP threads`, `CUDA_VISIBLE_DEVICES=0`.

## 4. What happened

- **One SCF iteration took 12,969.5 s (~3.6 h)** — from SIESTA's own timer:
  `timer: IterSCF 1 call 12969.515 s = 98.94 %` of the run. Setup was the
  other ~1 % (~140 s).
- The job hit htc's **4 h wall and was killed after that single iteration**
  (the cap was 5). `5 × 12,970 s ≈ 18 h` — K=1 cannot finish this benchmark
  in any reasonable window.

### Utilization (`job-gpu.util.csv`, 1497 samples @ 5 s)

| resource | mean | peak | reading |
|---|---|---|---|
| **GPU0 SM %** | **13.7 %** | 100 % | A100 **idle ~86 %**, bursting to 100 % only during each k-point's diagonalization |
| GPU0 VRAM | — | **6.2 GB** / 80 | one k-point's dense matrix at a time (16 B × 10,960² ≈ 1.9 GB × a few buffers) |
| **CPU %** | **2.5 %** | 4 % | ≈ 1 of 48 cores busy — even the 4 OMP threads sat mostly idle (SIESTA host work does not OMP-scale) |
| host RAM | — | **36 GB** | actual peak — the generator's 375 G/500 G estimate is ~10× conservative |
| GPU 1–3 | 0 | 0 | only GPU0 used (correct for G=1) |

## 5. Diagnosis — host-serial-bound, GPU starved

An SCF iteration = **host work** (real-space mesh Hartree/XC at 400 Ry, H/S
setup, density-matrix build) **+ 32 dense k-point diagonalizations** on the GPU.

With **one rank** and `Diag.ParallelOverK .true.`, the 32 k-points are solved
**one at a time** by that single rank, and the host work runs essentially
serially. The GPU does each k-point quickly (the 100 % bursts) and then **waits
~86 % of the time** for the lone rank to prepare the next one.

So the 3.6 h is *not* the GPU being slow — it's **the GPU idle, waiting to be
fed.** A single core/rank fundamentally cannot exploit the A100 here.

## 6. Extrapolating the A100's capacity

Two independent ceilings say how much the card can absorb:

1. **From idle time:** busy only **14 %** at K=1 → roughly **1 / 0.14 ≈ 7**
   well-pipelined ranks could run before the GPU itself saturates. That matches
   the documented **~4–8 ranks/GPU** optimum — *derived from this run, not
   assumed.*
2. **From VRAM:** ~2–6 GB per concurrent k-point into 80 GB → **well over a
   dozen** ranks fit memory-wise. VRAM is not the limit; **feeding** is.
3. **From the physics:** there are **32 independent k-points**. With
   `Diag.ParallelOverK`, adding ranks parallelizes directly over them — this
   work is embarrassingly parallel up to 32-way, and K=1 throws all of it away.

**The lever is K (ranks), not c (OMP).** SIESTA's host work parallelizes by MPI
domain decomposition, and the k-point diagonalizations parallelize across ranks
(each feeding the GPU via MPS). Adding OMP threads to one rank did ~nothing
(CPU 2.5 % at 4 threads).

## 7. Recommendations

- **Do not benchmark K=1** — confirmed unusable (the design already excludes
  it; this is the proof).
- **Next runs:** `K=4 c=6` (full 24-core socket, the published optimum) and
  `K=8 c=3`. With 32 k-points and MPS, expect a **large** speed-up — you are
  moving a 14 %-utilized GPU toward a fed one.
  ```bash
  ./prep-bench --gpus-per-node 1 --gpu-ks 4,8 --gpu-cs 6,3
  ./run-bench --domain htc        # or --domain public for slow points (7-day wall)
  ```
- **Walltime:** slow points may still exceed htc's 4 h for 5 iterations — use
  the `public` domain (7 d) or drop `MaxSCFIterations` to 2–3 for the probe.
- **Memory:** real peak 36 GB host / 6 GB VRAM → `--mem 64G` is ample in shared
  mode; the per-job estimate is ~10× high for this GPU run.
- **Next measurement:** sweep K at fixed c≈6 and watch `s/iter` fall; the
  **knee** is the real ranks-per-GPU capacity for *this* system (the util
  sampling, slurm-integration.md § 11.0e, tells you when the GPU finally
  saturates).

## 8. One data-quality note

The bench-marks header under-estimated orbitals (`n_orbitals_est 4440` vs the
actual **10,960** for TZP on this slab). It is only used for the memory model
and a display hint, but the gap is worth correcting in the estimator's
orbital-count heuristic for metal/TZP systems (it would also explain why the
memory estimate ran high).

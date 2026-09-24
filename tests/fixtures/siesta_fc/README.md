# A two-atom force-constant run, one atom held

SIESTA 5.4.2 (the packaged `molbuilder-siesta` binary), H2 at 0.741 Å in a
10 Å box, PBE/DZP, `MD.TypeOfRun FC` with `FC.First 2`, `FC.Last 2`,
`FC.Displacement 0.04 Bohr`, and `%block Geometry.Constraints / atom 1`.
Measured 2026-09-23 (design § 18 step 0.5):

* `h2.FC` — the force constants: a header line, then for each displaced
  atom (the FC range), each direction x, y, z, and each side (−, +), one row
  per atom of the structure holding the force-constant contribution.  Units
  **eV/Å²**: the z-block's two sides average to 41.713, and two single-point
  runs displaced by hand (`sp_plus`, `sp_minus`) give
  −ΔF/2Δ = 41.713 eV/Å² for the same element.
* `h2.FCC` — the same file with the HELD atoms' force rows zeroed (SIESTA's
  "constrained" variant).  The free block is identical, so the reader takes
  `.FC` and slices the free atoms.
* The geometry is not relaxed (0.44 eV/Å on the atoms), which is why the two
  turns of the free atom come back with negative curvature (−1.70 eV/Å²)
  rather than zero — the reason the surviving whole-body motions are
  projected out and never trusted.

**The relaxed pair of files, measured 2026-09-24** through the road — an
optimization calculation holding atom 1 to `MD.MaxForceTol` 0.001 eV/Å, its
final frame exported from the Results tab as a pair (H–H 0.7745 Å in the same
10 Å box, PBE/DZP), then the vibration description's `freq04` stage,
`FC.Displacement 0.04 Bohr`, `FC.First 2`, `FC.Last 2`:

* `h2_relaxed.FC` — its force constants (the free atom's z-block averages to
  33.859 eV/Å²; the mode comes out at 3022.3 cm⁻¹).
* `h2_fc.out` — the run's output, trimmed to the header and the first two
  `Begin FC step` blocks: step 0 is the reference geometry, whose forces
  (−0.003787 eV/Å on the held atom, −0.000057 on the free one) are what the
  read-back judges stationarity by (`engines/vibration.md` § 5.5); step 1 is
  the first nudge.

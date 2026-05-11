# Spec — chemistry: charge + protonation

**Module**: `molbuilder/chemistry.py` &nbsp;·&nbsp; **Tests**: `tests/test_chemistry.py`, `tests/test_review_fixes.py`

Two functions, both pure (no global state, no I/O):

```python
formal_charge_from_phosphates(struct: Structure) -> int
protonate_phosphate_oxygens(struct: Structure) -> Tuple[Structure, int]
```

## What it counts

The heuristic looks **only at phosphate groups**.  For each phosphorus:

1. Find non-bridging oxygen neighbours (O atoms whose only heavy
   neighbour is this P).  Adjacency is distance-based with cutoffs
   `_HX_CUT = 1.30` Å (X-H) and `_XX_CUT = 1.95` Å (heavy-heavy).
2. Of those non-bridging O's, the alphabetically-first **bare** one
   (no H attached) is the implicit `P=O` and contributes 0.
3. Each remaining bare non-bridging O contributes -1 to the molecular
   charge.

The function does NOT count:

* carboxylates (Asp, Glu side chains)
* protonated amines (Lys, Arg side chains)
* histidine pKa effects
* sulfonates, sulfates, nitrates
* metal coordination

These groups are **invisible to the heuristic**.  Users with such
systems must override via `cfg.charge` (PySCF) or `cfg.net_charge`
(SIESTA).  The implementation comment and the docstring say so
explicitly; the spec for the SIESTA / PySCF emitters propagates the
override.

## Idempotency

* `formal_charge_from_phosphates` is pure — same input, same output.
* `protonate_phosphate_oxygens` is idempotent: running it twice on
  the same input adds no extra hydrogens the second time.
* If no protonation is needed, the function returns the **same
  Structure instance** (via `is` identity check) and `n_added = 0`.

## Geometry of added H atoms

For each phosphate that needs protonation:

* Bond length: `0.96 Å` (canonical O-H)
* P–O–H angle: `109.47°` (sp3 tetrahedral)
* Direction: chosen so the O-H bond points *outward* from the
  centroid of P's other heavy neighbours, with a fallback to a
  perpendicular axis when the centroid is collinear with P-O.

## Edge cases (must not crash)

* Empty Structure → returns `(struct, 0)`, charge `0`.
* Phosphorus with only one non-bridging O (e.g. `P=O` alone): no
  protonation; charge contribution `0`.
* Mixed protonation (some non-bridging O's already have H, others
  don't): the alphabetically-first **bare** O stays as `P=O` (since
  protonating an already-protonated O would over-saturate); the rest
  of the bare O's get H added.

## Test reference

`test_chemistry.py` covers:

* fully deprotonated diester (charge -1, +1 H added)
* fully protonated diester (charge 0, no H added)
* partially protonated diester (charge 0, no H added — both halves)
* terminal phosphate dianion (charge -2, +2 H added)
* peptide / phosphate-free input (no-op)
* empty Structure (no-op)
* idempotency (second run adds 0 H)
* H placement geometry (0.96 Å, 109.47° within tolerance)

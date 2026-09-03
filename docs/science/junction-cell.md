# The junction cell along z — why the box needs one more layer spacing

**Role:** contract
**Domain:** science
**Companions:** [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md)
(the `cell` / `axis_kind` fields this decides a value for — § 4 `resolve_cell`,
§ 5 capture-at-construction); [`engines/transport.md`](?doc=engines/transport.md)
(the TranSIESTA invariants I8 / I10 / I12 that read this geometry);
the Modify tab's **Cell page** is where `c` is set (§ 6).

When you flank a molecule with two metal slabs, something has to decide how long
the box is along z. Get it wrong by one layer spacing and the crystal either
collides with itself or grows a contact no metal has. This document is the rule,
the numbers behind it, and what each choice costs.

**The rule in one line:** for a boundary that continues the crystal, the box
along z is **`z_span + d`** — the atoms' extent plus **one interlayer spacing of
the electrode crystal** — never the extent alone.

**Who applies it changed on 2026-08-31.** It is the number a person sets on the
Cell page after measuring `d` with the ruler (§ 6), not one a builder adds
behind them. This document still owns *what the right value is*; § 6 owns *how
it gets there*.

---

## 1. Why the extent alone is wrong

A periodic box repeats. If `c = z_max − z_min`, then the bottom atom's image in
the next cell sits at `z_min + c = z_max` — **exactly on top of the top atom**,
at zero distance. SIESTA stops with an error, and it is right to.

This is not a new class of mistake. `structure-periodicity.md` § 3 already states
it for the *in-plane* axes:

> **Wrong size.** For a slab with in-plane spacing `d` and `m` repeats, the true
> period is `m·d`, but the atoms' bbox is `(m−1)·d`-ish — short by ~one spacing,
> non-commensurate; tiling it overlaps/gaps atoms at the seam.

The same sentence applies verbatim to z. An atoms' extent is a **bbox**, and
§ 3's headline is that a bbox is "categorically not a lattice". The fix is the
one the same doc implies: add back the spacing the bbox dropped.

---

## 2. The spacing, per surface

`d` depends on which plane faces the molecule. For an fcc metal with cubic
lattice constant `a`:

| plane | interlayer spacing `d` | Au, `a = 4.0782 Å` (exp.) | Au, `a = 4.158 Å` (PBE) |
|---|---|---|---|
| (100) | `a / 2` | 2.0391 Å | 2.0790 Å |
| (110) | `a / (2√2)` | 1.4419 Å | 1.4701 Å |
| (111) | `a / √3` | 2.3545 Å | 2.4006 Å |

These are not asserted — they were checked against the slabs ASE actually
builds, and agree to six decimals. Note that ASE's surface builders return
`cell[2][2] = 0` for a slab (`pbc = [True, True, False]`), so **there is no
z-period to copy from ASE**; it has to be computed.

`a` itself is not a constant to hard-code: it comes from
`molbuilder/data/fcc_lattice.json`, which carries three references per metal
(`a_experimental`, `a_pbe`, `a_pbe_siesta_psml`). The gap must use **the same
`a` the slab was built with**, which is why the derivation reads the built slab
rather than a table.

---

## 2a. FCC only, and which metals

The builder makes **face-centred cubic** slabs and nothing else: `Ag`, `Au`,
`Cu`, `Ni`, `Pd`, `Pt`. The list is closed, so the UI dropdown and the Python
API cannot drift apart, and an element outside it is refused by name rather
than built wrong.

**Why the restriction is real and not a gap to fill in passing.** Everything
this document says about a seam is fcc arithmetic: the stacking period (§ 3.1)
is 3 on (111) and 2 on (100)/(110) *because* of the ABC stacking; the registry
step is `a/√6` on fcc(111); the spacings in § 2 are `a/√3`, `a/2`,
`a/(2√2)`. A bcc or hcp metal has a different period, different registries and
a different set of low-index surfaces, so it is not a table entry — it is a
second set of rules with its own seam conditions.

**What a bcc/hcp electrode would need**, if it is ever wanted: its own spacing
and period rows in § 2 and § 3.1, its own registry step for the seam classifier
(`cell.classify_seam`), and — for hcp — a `c/a` ratio, which fcc does not have.
Until those exist the honest answer is a refusal that says so.

## 2b. The cell shape is not a free switch

The slab builders take an `orthogonal` flag, but on two of the three surfaces
there is nothing to choose. Measured against ASE, not assumed:

| plane | `orthogonal = False` | `orthogonal = True` |
|---|---|---|
| (100) | **not available** — ASE raises `NotImplementedError` | always |
| (110) | **not available** — ASE raises `NotImplementedError` | always |
| (111) | always | only when `n` is even |

So the flag is a real choice on (111) alone, and even there it constrains
`(m, n)`. Offering it as a free checkbox on all three is how a default request
for a (100) slab came back a `400`: the panel's box starts unchecked, which is
the one setting (100) cannot build.

**The table lives in `modify.FCC_ORTHOGONAL_CHOICES`** and is served to the
browser by `/api/modify/meta`, the same anti-drift route `STACKING_PERIOD`
takes — the panel must not carry its own copy. `n`'s evenness is *not* in the
table: it depends on the size, so it stays where it already is, passed through
to ASE and surfaced verbatim (§ 5).

`tests/test_fcc_cell_shapes.py` builds every combination in the table and
asserts ASE agrees, so the copy cannot drift from the library it describes.

---

## 3. Filling the gap is necessary, not sufficient

Padding `c` by `d` fixes the *distance*. It does not fix the *registry* — which
atom sits over which. Two independent conditions have to hold before the seam is
a real crystal interface.

### 3.1 The stacking period — a layer-count condition

Close-packed layers repeat with a period that depends on the surface:

| plane | stacking | period | layers per side must be a multiple of |
|---|---|---|---|
| (111) | ABCABC | 3 | **3** |
| (100) | ABAB | 2 | **2** |
| (110) | ABAB | 2 | **2** |

Measured across the seam of a translated slab, with `c = z_span + d`:

| plane | layers/side | nearest metal across the boundary | verdict |
|---|---|---|---|
| (111) | 3, 6 | `a/√2` | continues the crystal |
| (111) | 4 | `d` | eclipsed |
| (100) | 4, 6 | `a/√2` | continues the crystal |
| (100) | 3 | `d` | eclipsed |
| (110) | 4, 6 | `a/√2` | continues the crystal |
| (110) | 3 | `d` | eclipsed |

`a/√2` is the bulk nearest-neighbour distance; `d` is the interlayer spacing
from § 2. "Eclipsed" means one atom sitting directly above another —
`Δxy = (0, 0)` — separated only by `d`, which on every surface here is far
shorter than any real metal bond. The relationship is what matters and it is
scale-free: it holds for any fcc metal at any `a`.

A caution about how to test this: **a distance check alone does not prove fcc**.
At (111) with 2 or 5 layers the across-seam distance is *also* 2.8837 Å, so a
distance test passes — but the lateral step is reversed. Comparing the step the
crystal takes inside the slab with the one it takes across the seam (allowing
for the triangular lattice's three equivalent directions):

Writing the in-slab step as **`s`** (magnitude `a/√6` on (111)):

| (111), layers per slab | seam step | |
|---|---|---|
| 2 | `−s` | **twin** — reversed |
| 3 | `s` rotated 120° | fcc continues |
| 4 | `(0, 0)` | eclipsed |
| 5 | `−s` | **twin** — reversed |
| 6 | `s` rotated 120° | fcc continues |

On a triangular lattice the three 120°-separated directions are the same step,
so the rotated form counts as a continuation while the reversed one does not.
A twin is a real defect carrying the *right* bond length — which is exactly why
a distance check misses it. **Only a multiple of 3 gives true fcc continuation
on (111).**

### 3.2 The `-z` slab must be *translated*, not flipped

This is the condition that is easy to miss, because every obvious symmetry
operation fails it, and fails it identically.

`add_slab(grow="-z", stacking="mirror")` places a slab by **mirroring**
(`z → −z`). A mirror maps the slab's outermost layer to the *other* slab's
outermost layer — the same layer index — so both faces meeting at the cell
boundary carry the **same in-plane registry** and are eclipsed. The deleted
`add_electrode_slab` did this *unconditionally* for `side="-z"`, with no way
to ask for anything else; `stacking="continue"` is the alternative the
redesign added, and it is what `--electrode` now uses.

Choosing a different symmetry does not help, and the reason is structural rather
than incidental: **each close-packed layer is itself a centrosymmetric 2-D
lattice**, so any point-group operation maps that layer's atom set onto itself.
Measured, at (111) with 3, 4 and 6 layers per side:

| operation for the `-z` slab | seam step | nearest across boundary | verdict |
|---|---|---|---|
| mirror `z → −z` (today) | `(0, 0)` | `d` | eclipsed |
| C₂ about x, `(x, −y, −z)` | `(0, 0)` | `d` | eclipsed |
| inversion `(−x, −y, −z)` | `(0, 0)` | `d` | eclipsed |
| **translation, no flip** | one lattice step | `a/√2` | continues (when § 3.1 holds) |

Only a lateral translation changes the registry. `c` cannot: it sets how far
apart the two faces are, never which sits over which.

**The trade this creates.** With a mirror, the molecule sees layer 0 on both
sides — the two contacts are equivalent, which is the point of a symmetric
junction. With a translation, the molecule sees layer 0 on one side and layer
N−1 on the other, so the two anchor sites differ by one registry step. A lateral
shift can restore contact symmetry, but it moves the seam by the same step.
With rigid slabs you cannot have both, so this is a real choice, not an
oversight to be fixed.

---

### 3.3 The arithmetic of the two placements is not symmetric

§ 3.2 established that only a lateral translation changes the registry. There is
one more difference between them, and it is easy to get wrong because the two
operations do not take the same reference point.

```
                        MIRROR                        TRANSLATE
                  (layer order flips)            (layer order kept)

   cell top  ─────────────────────────      ─────────────────────────
             layer N-1 ▪ ▪ ▪ ▪ ▪ ▪           layer N-1 ▪ ▪ ▪ ▪ ▪ ▪
             ...                             ...
   +gap/2 ▸  layer 0   ▪ ▪ ▪ ▪ ▪ ▪  ◂ face   layer 0   ▪ ▪ ▪ ▪ ▪ ▪  ◂ face
                        M O L E C U L E                M O L E C U L E
   -gap/2 ▸  layer 0   ▪ ▪ ▪ ▪ ▪ ▪  ◂ face   layer N-1 ▪ ▪ ▪ ▪ ▪ ▪  ◂ face
             ...                             ...
             layer N-1 ▪ ▪ ▪ ▪ ▪ ▪           layer 0   ▪ ▪ ▪ ▪ ▪ ▪
   cell bot  ─────────────────────────      ─────────────────────────
                      ▲                              ▲
         SEAM: layer N-1 meets itself       SEAM: layer 0 meets layer N-1
               same registry → ECLIPSED           one step → CONTINUES
```

A **mirror** reflects about the contact plane, so the layer that was nearest the
molecule is nearest again and the face lands where it should by construction.

A **translation** keeps the layer order, so the face meeting the molecule is the
slab's far end — and the shift must therefore be referenced to that end, not to
the end the mirror uses. Referencing the wrong one displaces the slab by its own
full thickness, which puts it on the far side of the molecule rather than beside
it. The slab's thickness enters the arithmetic; a mirror's does not.

Both correct forms give an identical gap and an identical slab. Only the
registry at the seam differs — which is the whole point of the choice.

---

## 4. What a bad seam actually costs

It depends entirely on whether the z axis is genuinely periodic in the run.

**A plain SIESTA run — periodic in z.** The seam is physically present. A zero
gap is a hard error; an eclipsed gap is a head-on metal contact at 1.4–2.4 Å
(the interlayer spacing) where the lattice wants an *offset* neighbour at
`a/√2` — 2.88 Å on the experimental `a`, 2.94 Å on the PBE one. So the
boundary layers carry spurious repulsion and
a perturbed density. Freezing those layers stops forces from propagating into
the molecule, so a **relaxation stays sound**, but the total energy is
contaminated — which matters if energies are compared across structures whose
boundary registry differs.

**A TranSIESTA run — z is open.** Three invariants from `engines/transport.md`
§ 5 change the picture:

- **I8** — the device runs at `kz = 1`. There is no Bloch periodicity along
  transport; the semi-infinite leads enter only as self-energies Σ built from a
  *separate pristine bulk-lead run*. What lies across the device cell boundary
  is **replaced** by Σ, so the seam is not part of the transport physics.
- **I12** — `z-vacuum ≈ 0 at the leads`: "a gap = severed lead, not a junction".
  `c = z_span + d` satisfies this; the padding is one lattice spacing, not
  vacuum.
- **I10** — the electrode block must map atom-for-atom onto the bulk lead.

So the seam matters most for the **bulk electrode cell**, which by **I9** *is* a
periodic run along z — and that cell is already derived correctly (§ 5).

### 4.1 And the tab says which of these you have

Every slab edit is answered with a seam verdict, carried back in the response's
receipts slot and shown as a panel notice. It is not a distance check: § 3.1's
point is that a twin has the correct bond length, so only the **registry**
separates a continuation from a twin. The question with an answer is *which
layer of this slab does the imaged bottom layer land on*, and in a crystal of
period `p` that is decidable without any lattice arithmetic:

| the image lands on | verdict | what you are told |
|---|---|---|
| layer `N − p` | `continues` | the crystal carries on — the spacing and the nearest-atom gap, as numbers |
| the top layer | `eclipsed` | stacking repeats instead of advancing — § 4's head-on contact |
| layer `N − 2` | `twin` | right bond length, reversed stacking — the case a distance check passes |
| — (an open face) | `vacuum` | **not a warning.** An open face is what a slab calculation wants |

One more verdict is a *warning about the slab rather than the seam*: a boundary
that continues but with a repeat shorter than `STACKING_PERIOD` says for that
plane (`too_thin`) — two layers of (111) are A,B, which is fcc and hcp alike,
so such a slab does not determine its own stacking. It says *add layers*.

**The period is measured, never passed in.** The plane never has to be stated,
and a strained or unusual slab is judged on what it is.

> **Why this is checked at the edit and not later.** Measured on a real
> `Au-BDT-Au` junction, the seam is bit-identical before and after relaxation —
> 2.4008 Å, step (0.000, 0.000) both times — because the layers that form it
> are exactly the ones `Geometry.Constraints` pins. **No later check can catch
> it, because nothing later changes it.**

---

## 5. Where this lives in the code

**One home.** The derivation is `cell.bulk_z_period(layer_z)`, returning
`(z_period, d_interlayer, n_layers)` with `z_period = z_span + median(Δlayer)`.
The median (not the mean) makes it robust to a slightly relaxed outermost layer,
and reading the spacing off the *built* slab means an `inter_layer_offset`
override is honoured automatically. It raises `ValueError` on a single layer,
where the repeat is genuinely underivable.

It began in `transport/wizard.py`, where the electrode wizard has always used it
to derive the bulk lead's z-period, together with a note telling the user to
confirm the layer count is a whole stacking period. Moving it to `cell` (L1)
lets the junction builder use the same function instead of growing a second
copy — the two callers are `transport.wizard.extract_electrode_model` and
`modify.add_slab`.

**The monolayer case.** A one-layer slab has no spacing to *measure* — there
is only one layer to measure between. This mattered while the builder PADDED
the cell for you; § 6 retired that on 2026-08-31 and `add_slab` sets
`c = z_extent` verbatim, so nothing needs the spacing at build time any more.
The probe below is how `add_electrode_slab` handled it, kept as the method
for anyone who needs the number: ask the same ASE builder for a two-layer
slab at the same `(m, n)` and
`orthogonal` and reading `d` off that; using the caller's own lateral size
matters, since a probe at `(1, 1, 2)` would trip ASE's "second number must be
even" constraint on orthogonal fcc(111).

For a slab built through § 6's route the case does not arise as a *builder*
problem: nobody is padding, so nothing needs a spacing it cannot measure. It
becomes a question for the person instead — a monolayer's repeat distance is
the crystal's `d` from § 2, which they can read off the table rather than off
their own structure.

| Concern | Home |
|---|---|
| the rule `z_period = z_span + d` | `cell.bulk_z_period` |
| grouping atoms into layers | `cell.detect_layers` |
| layers per stacking period, by surface | `cell.STACKING_PERIOD` |
| **what the boundary does to the crystal** | **`cell.classify_seam`** (§ 4.1) — the measurement; the wording is `blueprints/modify.py`, the same split `_lattice_notes` uses |
| applying it to the bulk lead | `transport.wizard.extract_electrode_model` |
| **setting it on a built slab** | **you**, on the Modify tab's Cell page (§ 6) |

`STACKING_PERIOD` is a table rather than a verdict function on purpose: the
panel's note re-renders as the user types a layer count, so `/api/modify/meta`
ships the crystallography once and the arithmetic happens client-side. A
surface absent from the table means *not known*, and the note then says nothing
rather than asserting a verdict.

---

## 6. The `c` axis is measured and set, never invented

*(Rewritten 2026-08-31 on the user's decision. This section described a
checkbox — "**Pad cell by one layer spacing**", on by default — that built
`c = z_span + d` for you. The switch is gone and so is the padding.)*

**The builder sets what it knows and nothing else.**

| | who sets it | why |
|---|---|---|
| `a`, `b` | the **builder** | they are the crystal's own in-plane vectors, straight from the slab ASE built. Re-deriving them by hand would be a second formula waiting to disagree with the atoms. |
| `c` | **you** | it is a decision about the calculation — how much vacuum, or which repeat distance makes the boundary continue — and nothing in the geometry answers it. |

A freshly built slab therefore gets `c = z_extent`, the atoms' span verbatim.
That is not a usable cell: the top layer sits on its own periodic image, and
`classify_seam` says so — **`collision`**. That is the point. The missing step
is visible in the tab you are already in, and stays visible until you take it.

### 6.1 How you set it, and what the tab owes you

The principle is *measure, calculate, set* — so the tab's job is to make the
two measurements cheap, not to make the decision:

1. **The layer spacing** — pick two atoms in adjacent layers with the ruler.
   The readout's signed `Δ = (Δx, Δy, Δz)` gives it directly.
2. **The span** — the Cell page shows `z range (max − min)` for the current
   structure. Displayed rather than left to arithmetic, because it is a fact
   about the atoms and a person should not be recomputing it.
3. **The gap** — a box beside the `c` axis: `c = z range + gap`. For a
   boundary meant to continue the crystal, the gap **is** the layer spacing
   from step 1. For a slab with a free surface, it is however much vacuum the
   calculation needs.

**Nothing checks that you typed the spacing rather than something else**, and
nothing should: a slab calculation wants vacuum there, and only you know which
you are running. What the program owes you is a verdict on what you set, and
`classify_seam` gives one on every build — `collision` if `c` is too small,
`vacuum` if it is much larger than a layer, `continues` if the crystal joins
(§ 3). That closes the loop without anyone guessing.

### 6.2 Why the switch went

It hid the decision it claimed to expose. On — the default — the builder chose
`d` for you and the note beside the switch deliberately *did not print the
number*, on the grounds that quoting it would mean a second formula in
JavaScript. So the one value that determined whether your junction was a
crystal was computed out of sight and never shown.

Measuring it with the ruler and typing it costs one gesture, puts the number on
screen, and makes `c` a thing you decided. The second formula the old note
feared is not needed either: the ruler is not a formula, it is the distance
between two atoms you chose.

**What still overwrites what.** Unchanged, and worth restating since the switch
was tangled in it: building a slab *establishes* the in-plane lattice, so `a`
and `b` are overwritten by a build — the slab is what defines them. A `c` you
commit through the Cell page is an explicit cell and stands (`resolve_cell`
branch 1, `structure-periodicity.md` § 4) until the next build replaces the
in-plane pair. Editing the Cell tab is now the only way `c` is ever set, which
is the simplification: there is one door instead of a switch and a door.

---

# Modify — the Molbuilder tab's five-item redesign

**Role:** plan (designed, none of it built)
**Domain:** web · science
**Started:** 2026-08-30
**Companions:** [`web/molview.md`](?doc=web/molview.md) §§ 8.5, 9.5, 11.2b, 11.6 —
the viewer ·
[`web/tabs.md`](?doc=web/tabs.md) § 2 — the Molbuilder tab, and the
Modify op-tabs inside it ·
[`science/junction-cell.md`](?doc=science/junction-cell.md) §§ 3, 5, 6 — the seam
and the padding switch ·
[`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 6.2 —
the one cell door ·
[`plans/bench-and-junction-plan.md`](?doc=plans/bench-and-junction-plan.md) § 2.3 —
subsumed by item 3 ·
[`web/ui-contract.md`](?doc=web/ui-contract.md) §§ 1, 4 — the stylesheet layers
and the rhythm item 5 has to obey ·
[`plans/css-system-plan.md`](?doc=plans/css-system-plan.md) §§ 3, 4 — the tier
rule, and the steps C/D item 5 partly discharges

**Why this file exists.** The user dictated four changes to the Molbuilder tab's
Modify surface in one sitting, then settled every open question in the exchange
that followed; a fifth — the tab's own placement and alignment — was added once
the first four were written down. This is
the record of what was asked, what the code actually does today, and what each
item changes — written before any code, so the contract is what gets reviewed
rather than a diff.

**Nothing here is a guess.** Every claim about current behaviour was read out of
the code or measured; the measurements are in § 6, with the numbers. Where a
first reading of mine turned out to be wrong, the correction is recorded rather
than quietly replaced (§ 2.2, § 3.3) — the wrong reading is the more useful half,
because it says where the code misleads.

---

## 0. State when this was written

| | |
|---|---|
| HEAD | `465e1e22`, tree clean, nothing unpushed |
| branch | `feature/generator-jobset-ui` |
| built | none of this |
| lane | last full run `7995 passed, 6 skipped, 1 xfailed` on this commit |

---

## 1. Measurement is its own selection track

**Asked** *(user, 2026-08-30)*: measurement gets its own selection track, capped
at **3 atoms**, reached by its own toggle beside the other rail switches. While
it is on, a click — **in the list or in the 3D window** — does not touch the real
selection. The track is **not written to the `.xyz`**, but persists the way the
camera does. The readout shows **each picked atom's coordinates**; at two atoms
the **distance together with Δx, Δy, Δz**; at three, the **angle**, as today.

Two follow-ups settled it: the **fourth pick drops the oldest** so measuring a
chain stays fluid, and clearing is a **Clear button on the readout**, not a
right-click.

### 1.1 What is there today

The layer already exists and is already separate — `molview.md § 11.6` says the
readout is its own layer, and `ui.js mountReadout` implements it. **Only its
input is wrong**: its atoms come from `selection`.

Everything else the item needs has a home already built:

| What the item needs | Where it goes | Today |
|---|---|---|
| a toggle beside the others | `RAIL` in `ui.js` | five switches + Reset |
| clicks stop moving the real selection | **two paths, and only two** — `mount.js`'s `embed.onPick`, and the panel's two row handlers in `ui.js` | both call `selection.toggle` |
| persistent, but not data | `ui-context.js`, tag `<owner>:ui` | already holds camera, frame, switches, view, under `molview.md` § 11.2b's *"looking is not changing"* |

### 1.2 What changes

- **A new store** in `stores.js` — an ordered list of at most 3, dropping the
  oldest on a fourth pick, with `clear()`. It is **not** a field on `selection`
  and does not appear in `selection`'s snapshot.
- **A sixth rail button.** It reads its lit state from the store like the other
  five, so nothing has to be kept in step.
- **The two click paths ask one question first** — *measuring?* — and route
  accordingly. One question, asked in two places, because there are exactly two
  places a click enters.
- **The readout reads the measurement track only**, and grows:

  | picked | shown |
  |---|---|
  | 1 | `Au #12 — (1.234, 5.678, 9.012) Å` |
  | 2 | both atoms' coordinates · `\|Au #12 – S #3\| = 2.401 Å` · `Δ = (0.412, −1.203, 2.010) Å` |
  | 3 | all three coordinates · `∠Au #12 – S #3 – C #4 = 104.7°` |

  The vertex stays the atom picked **second** — the chemist's convention
  `molview.md` § 11.6 already fixes.
- **Clear sits on the readout chip.** The selection panel already has a `Clear`
  (`ui.js:1568` → `selection.clear()`); a second button with the same word in the
  same card, meaning something else, is the confusion this item exists to avoid.
  On the chip it appears only while something is picked, and there is nothing
  else it could plausibly clear.
- **`ui-context.js` carries the track** in the lane it already writes, under the
  same `match` guard that protects the camera and the frame.

### 1.3 What is deleted

`orderedForMeasurement`'s no-trail branch, and `byGeometry` with it.

That guess exists only because a *selection* can arrive with no pick order — from
*All*, *Invert*, an applied filter, a restored session — and the readout then had
to invent a vertex from geometry. A measurement track is **only ever built by
clicks**, so there is no such case left. `molview.md` § 11.6 loses a paragraph
rather than gaining one.

### 1.4 The isolation rule, and what pins it

*(User: "make sure your code does not make the measurement selection
conflict/overlap with atom selection data in the molview that is used
elsewhere.")*

**The track is written by exactly one thing — a click while the toggle is on —
and read by exactly one thing — the readout.**

Where it must never appear, each line a test that fails if the wall breaks:

| Never | Why it would matter |
|---|---|
| `selection`'s snapshot | the panel, the halo, the count and isolate all read that one settled object |
| the structure | no coordinates, labels, regions or sidecar — hence never the `.xyz` |
| `history.js` | measuring is not an edit; no state, no draft, no badge |
| any export | Data, image, save |
| any request body | the track never leaves the browser |
| `selectedIndices()` in `modify/viewer.js` | the single door every op's group goes through |

Persisted in **exactly one** place: the `<owner>:ui` lane.

**The named hazard, and it is created by items 1 and 2 landing together.** Item 2
makes Center act on *the selection*. Every op resolves its group through
`applyOp`'s `handed.readSelection()`, which reads `selection`. If that ever became
the measurement track, clearing a measurement would silently change what an edit
operates on. It gets its own pin, separate from the wall above.

---

## 2. Center at origin follows the selection

**Asked** *(user, 2026-08-30)*: *"there seems to be a bug in the transform 'center
at origin' — when we have selected a group of atoms, the idea is that the center
operation would be about this selected group, not the whole structure."*

Settled in the exchange: **the group is the rigid part** — only those atoms move,
so their centroid lands at the origin, and nothing else moves. **Nothing selected
means everything is the group**, moving as one rigid body.

### 2.1 Where the bug is

**Not in the browser.** `OPERATIONS.translate` in `model-jobs.js` declares
`{emptySelection: "all", group: "indices"}`, and `applyOp` writes the live
selection into `body.indices`, omitting the key when the selection is empty —
with a comment saying why: *"so the server applies its own centring rather than
being handed an empty list."* `applyCenter` posts to **the same route as
Translate**, so Center's request has always carried the selection.

The bug is one branch in `blueprints/modify.py::api_modify_translate`:

```python
if bool(body.get("recenter", False)):
    new_struct = struct.centered()      # the mean over ALL atoms
    return _ok_response(new_struct)     # ← returns here
...
indices = body.get("indices")           # only the dx/dy/dz path ever reads this
```

**The `recenter` branch returns before `indices` is ever looked at.** The
selection arrives and is discarded one line before it would have been used.

That is precisely the inconsistency the user reported — Translate honours the
group, Center does not — and it is not because Center was not told.

### 2.2 A wrong reading of mine, recorded

I first reported that the Translate **button** had the identical gap, having read
`applyTranslate()`, seen a body with no `indices`, and stopped there. The group is
injected one layer down, in `applyOp`. The user had tested Translate and knew it
worked.

Worth keeping because it says where the code misleads: **the op body at the call
site is not the request.** Any future reading of a Modify op has to go through
`postOp` → `applyOp` → the `OPERATIONS` table before concluding what is sent.

### 2.3 The fix

Make Center **be** a Translate whose vector the server computes, so there is one
path and nothing to drift:

```python
group = body.get("indices") or None
centroid = (struct.positions[group] if group else struct.positions).mean(axis=0)
return _ok_response(_translate(struct, -centroid, indices=group))
```

Both of the user's rules then hold by construction, because they are already
`modify.translate`'s, stated in its docstring:

- **group named** → `_moved_subset`: only those atoms move, the box stays
- **nothing named** → `struct.translated`: everything moves rigidly, the box goes
  with it

### 2.4 A rule I proposed and withdrew

I proposed *"the box moves when, and only when, everything moved"*, to close the
edge where selecting **all** atoms behaves differently from selecting **none**
(box stays vs box travels).

**Withdrawn.** That edge already exists in Translate, which the user has tested
and called correct. Adding the rule would make Center differ from Translate,
which is the opposite of what was asked. **Center copies Translate, edge
included.**

---

## 3. A new slab tab, beside the old one

**Asked** *(user, 2026-08-30)*: a **new op-tab**, so the old UI and backend are
untouched and are removed only once the new design is proven. Keep element,
plane, lattice ref, m, n, layers, and the (dx, dy) offset. **Drop `gap` and the
symmetric/single mode.** Add **starting surface (A, B, or C if available)**, the
**z-offset of that starting surface**, and the **direction of growth** — one slab
per operation, specified by its lattice, its starting z, and how it grows. Keep
the orthogonal-cell option. **"Pad cell by one layer spacing" moves out**, into
the cell setup (item 4). Make **"Your bulk run"** work.

**Placement is absolute** *(user)*: dx, dy **and** the starting-surface z are all
measured from the origin of the 3D window's own coordinate system. There is no
selection-centroid anchor — **the new panel reads no selection at all**, which is
one fewer thing to keep apart from item 1.

### 3.1 A/B/C is buildable, and it is read off the slab

Measured on all three surfaces, not assumed (§ 6.1): an fcc(111) slab's layers
sit on exactly **three** in-plane registries, separated by `(a₁+a₂)/3`; (100) and
(110) have **two**, separated by `(a₁+a₂)/2`. So *"start on A, B or C"* is a
lateral shift of `k·(a₁+a₂)/period`, and the period is `cell.STACKING_PERIOD` —
`{111: 3, 100: 2, 110: 2}` — which already ships and is already **served to the
panel** by `/api/modify/meta` (`stacking_period`).

**The step is equal only modulo the lattice vectors**, and that is not a
technicality to skip: the raw difference between two consecutive layers comes out
as the *negative* of `(a₁+a₂)/period` on every surface measured (§ 6.1). Whoever
implements this will read that difference, see a sign that does not match, and
have to decide whether the formula or the slab is wrong. It is neither.

*"if available"* falls out of the period: three choices on (111), two on the
others. **No new table** — the shift is built from the slab's own in-plane cell
vectors, so a `lattice_constant` override is honoured with nothing passed in,
which is the same discipline `junction-cell.md § 5` applies to the z padding for
the same reason.

### 3.2 What this closes rather than adds

`bench-and-junction-plan.md § 2.3` carries an unbuilt `mirror | translate`
parameter. It exists only because `side="-z"` unconditionally mirrors, which flips
the layer order and breaks the seam — the failure `junction-cell.md § 3.2`
describes.

**Saying where a slab starts and which way it grows makes that flip unreachable.**
§ 2.3 should be closed into this item rather than built separately. Dropping
`gap` and the pair mode is the same argument: the pair is what forced the mirror.

`§ 2.4`'s seam detector survives, with a changed job: it stops being the only
thing that knows about registry and becomes the **check on a stated one**.

### 3.3 "Your bulk run" — dead by construction, not broken

Verified (§ 6.3): `a_pbe_siesta_psml` is `null` for **all six** metals in the
packaged `fcc_lattice.json`, and **nothing in the codebase writes it**. The radio
greys itself when the value is null — correctly — so it has been unreachable
since it shipped. Its only home is that packaged table plus the machine-wide
`MOLBUILDER_DATA_DIR` override. There is no per-run slot for it, which is the
shape the user rejected: *"of course it is for one optimization run."*

**The new door** *(user's design)*: extract it from a result the user points at —
*"a `.xyz` or `.XV` result where one single periodic lattice is correctly
optimized with the same pseudopotential/basis etc… the user needs to make sure
this setup is correct, and the backend just extracts the lattice from that
result."*

```
POST /api/modify/lattice-from-run   { path, element }
  → { a, d_nn, coordination, n_atoms, notes }
```

Both readers already exist: `transport.compose.read_xv` for `.XV`, and
`Structure`'s `Lattice="…"` header for extended `.xyz`.

**It measures the atoms, not the cell — and that is the load-bearing choice.**
The file's cell may be conventional cubic (edge `a`), primitive rhombohedral
(edge `a/√2`), or the user's own m×n×N layered lead cell (edge `m·a/√2`) — three
different relations to `a`, and **the file does not say which**. Reading the cell
means guessing the user's convention. The nearest-neighbour distance under
minimum image assumes nothing:

```
d_nn = min pair distance over periodic images,  within the named element
a    = √2 · d_nn
```

Verified on a 3×3×6 Au(111) supercell: all three routes recover `a` exactly
(§ 6.2 here). This is exactly the division of labour the user set — **they** guarantee
the pseudopotential, basis and cutoff; the backend does not have to guess how the
cell was set up.

**Two checks, reported as notes, never refusals** — the setup is the user's to
own:

- coordination at `d_nn` should be **12** for fcc; anything else means the file
  is probably not the bulk crystal intended (a slab with vacuum, a surface, a
  defect)
- the second shell should sit at **√2·d_nn** — the fcc signature, and the cheap
  catch for a distorted or non-cubic result

**Two refusals, because guessing would be worse:** a file carrying **no cell**
(no periodic images; on a small cell the measured minimum is simply wrong), and
**more than one element with none named**.

**The field stays typeable.** The picker fills it, the derived line shows what it
means, and the value can be overridden by hand:

```
Your bulk run    [ Pick a result… ]   Au₅₄ from AuBulk-relax/run-0/Au.XV
                 nearest-neighbour Au–Au = [ 2.9503 ] Å      12 neighbours ✓
                 → a = 4.1723 Å · d(111) 2.4089 · d(100) 2.0862 Å
                 ✓ 0.3% from PBE (4.158 Å)
```

The derived line is the cross-check. The one mistake anyone actually makes is
picking a **second-shell** pair, which in fcc sits at exactly `a` — a factor
1.414 out, landing 41% from both references, where the comparison says so
immediately.

**What is deleted:** `a_pbe_siesta_psml` leaves the packaged table — a column
nothing can ever fill — and with it `renderLatticeRefRadios`' greying, its
"is my pick still valid" check, and its silent fall-back to Experimental, all of
which exist only to cope with a value that is never present. The table keeps the
two literature references, which is what a shared table is for. `load_fcc_lattice_full`'s
v2 schema and `molbuilder/data/README.md` change with it; the loader reads the key
with `.get`, so its absence does not raise.

### 3.4 The scheduled removal

The user's instruction is explicit: build beside, replace after proving. So the
new tab does **not** get to leave the old one standing indefinitely. When it is
proven, these go:

- the Junction op-tab panel and its half of `modify/viewer.js`
- `modify.add_symmetric_electrodes`
- `POST /api/modify/symmetric_electrodes`
- `OPERATIONS.symmetric_electrodes` in `model-jobs.js`

---

## 4. Cell setup — axes and origin from the structure

**Asked** *(user, 2026-08-30)*: keep the 3×3 matrix, and add: a **drop-down for
the a, b or c axis**, then **pick two atoms in the 3D window** to define that
axis; an option to **change the length** of a, b and c, to set the spacing
between periodic images; and the same for the **cell origin** — pick an atom to
assign it, alongside the three existing text boxes.

### 4.1 No new backend

There is already **one door**: `POST /api/structure/periodicity`, with
`op ∈ {vacuum, axis_kind, cell, cell_origin}`. Both new gestures land on ops that
already exist — two atoms define a row of the `cell` matrix; a picked atom is a
`cell_origin`. **No new route, no new op.** The arithmetic is vector subtraction
on coordinates the browser already holds for the readout, and the length control
scales a row to a stated magnitude — the same act the nine text boxes already
perform, reached by a different gesture.

This makes item 4 the cheapest of the four.

### 4.2 Picking, without a third track

Item 1 diverts clicks to the measurement track. Item 4 wants picking too. **The
cell page does not get a track of its own**: it reads the ordinary selection
through a *Use selection* button — two atoms → that axis, one atom → the origin.

One picking mechanism, one diversion switch, no third mode to explain — and
nothing new to keep apart from § 1.4's wall.

### 4.3 Handedness — the refusal this gesture will actually hit

**Three axes picked independently from atom pairs have no reason to come out
right-handed**, and `periodicity_gate` refuses a left-handed cell outright —
`ValueError` → HTTP 400, by the rule *"right-handed cells only, det > 0"*. Typing
nine numbers rarely produces one by accident; **picking three atom pairs will
produce one about half the time**, so this stops being a rare refusal the moment
the gesture ships.

The panel should say so **before the request**, and say the way out — which is
unusually easy here, because a two-atom pick carries its own sign: **picking the
same two atoms in the other order flips that axis.** So the note is one sentence
with an action in it ("these three axes are left-handed — pick c's two atoms the
other way round"), not a 400 the user has to interpret.

Naming it here because it is a *predictable consequence of the new gesture*, not
a defect in the gate: the gate is right to refuse, and § 4 is where the panel
learns to expect it.

### 4.4 What it inherits from item 3

*"Pad cell by one layer spacing"* moves here.

That is a **contract rewrite, not a UI move**: `junction-cell.md § 5` names the
Junction panel as the switch's home, and § 6 explains why it defaults on — *an
unpadded box collides with itself, and no engine can use it*. That reasoning has
to survive the move intact, including the deliberate refusal to print `d` in the
panel (a second formula in JavaScript would be a second answer waiting to
disagree with the one that shapes the box).

---

## 5. The surface — one inset, one edge, one typeface

The user, 2026-08-30: *"your current modify tab have many elements not displayed
clearly or inconsistent from the rest. this is a good time to improve those
visual placement and alignment etc. make sure your css is designed
systematically rather than a patch."*

It is item 5 rather than an afterthought because items 3 and 4 each add a
**panel to this card**. Whatever decides where a row starts has to be settled
before two more panels are built on top of it, or they inherit the arithmetic
below and the problem doubles.

### 5.1 What is on screen, measured

Read out of the live page — `/molbuilder`, one 312-atom structure loaded, a
641 px content column — not out of the stylesheet.

**Three cards, three title positions.** Distance from the content column's left
edge:

| card | card's left edge | title's left edge | title inset |
|---|---:|---:|---:|
| Init structure | 0.0 | 20.8 | 20.8 |
| Structure & selection | 16.0 | 36.0 | 20.0 |
| Modify | 16.0 | 54.8 | **38.8** |

Two faults at once: the init card sits **flush** with the column while the two
below it are inset 16, and the Modify title is indented 19 px further than
either of the other two.

**Inside the Modify card, six left edges:**

| what | inset from the card's edge | where it comes from |
|---|---:|---|
| header rule · action row · message band | 18.8 | `.card`'s own padding |
| `h2` "Modify" | 38.8 | + `.card-header`'s 20 |
| op-tab row · panels · fieldsets | 34.8 | + `.modify-edit-panel`'s 16 |
| the tab buttons | 42.8 | + `.modify-op-tabs`' 8 |
| every legend, label and control | 51.6 | + the fieldset's 16 + the legend's 4 |

The card's title, the rule under it and its first form row start at three
different places, and **8.6 % of a 600 px card is left margin** accumulated four
containers deep.

**The rest of the app does not do this.** On `/task-setup`, five cards all wear
the shell's `.card` and every one of their titles sits at exactly **18.8 px** —
the card's own inset, one number, no exceptions. Modify is the outlier, not the
norm, which is why this is a repair rather than a redesign.

### 5.2 Why it happens — indentation by nesting

Each container adds padding without knowing what its ancestors already spent:
18 (`.card`) + 20 (`.card-header`) for the title; 18 + 16
(`.modify-edit-panel`) + 16 (the fieldset) + 4 (the legend) for a form row.
Every one of those is defensible alone. The sum is nobody's decision.

That is also why a patch cannot fix it. Subtracting 20 from `.card-header`
lines the Modify title up and **breaks the viewer card's**, because that
section is not a `.card` and has no 18 underneath to absorb it — the same rule
has to serve a header with a padded parent and a header without one.

### 5.3 The rule

> **One inset per card.** A card states its inset once, on itself. Everything
> inside it — title, rule, tabs, fieldsets, rows — begins at that inset and adds
> none of its own. A nested block may add *vertical* space; horizontal
> indentation is a deliberate signal of hierarchy, never a by-product of
> nesting.

Here that means `.card-header`, `.modify-edit-panel`, `.modify-op-tabs` and
`.modify-op-block` give up their horizontal padding, and one edge runs from the
title straight down through every control. Where a row genuinely *should* be
indented — a sub-option under the control it belongs to — it says so with one
step of the spacing scale, and the reader can tell, because it is then the only
indentation on the card.

> **One gutter per tab.** Every top-level block shares one left edge, one right
> edge and one max-width. Nothing sits outside the page's own container.

Here that means `.modify-init-card` moves **inside** `<main class="modify-main">`
— today it is a sibling of it (`modify.html:55` against `:279`), which is the
whole reason it is flush while everything below is inset — and the max-width
that only `.modify-grid` carries moves to the container both blocks sit in.

### 5.4 Three findings bigger than this tab

Each affects every page; none is fixed by item 5, and each is recorded so it is
decided once rather than a fourth time.

1. **The app's one card surface is off the 4 px grid.** `page-shell.css` sets
   `.card { padding: var(--space-md) 18px 18px }`. 18 is not a hairline (under
   3 px, exempt) and not a dimension (over 40 px, exempt), so by
   `ui-contract.md` § 4's own rule it should be on the scale. Because `.card` is
   the canonical surface, *everything* nested inside one is 2 px off the grid it
   is supposed to share. Fixing it shifts every page by 2 px and deserves its
   own commit.

2. **The page renders in two typefaces.** `page-shell.css` sets the body to
   `-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, "Inter",
   sans-serif`; six `font:` shorthands in `modify/style.css` write
   `system-ui, sans-serif`. **The shorthand resets the family**, so card titles,
   op-tab labels, hints and the timeline status come out in a different face
   from the body text around them — measured on this machine, the same string
   sets 292.61 px in the first stack and 290.93 px in the second. `tokens.css`
   has had `--font-sans` since it was written and **nothing in the page layer
   uses it**. The rule that closes it is one line: *a `font:` shorthand names
   `var(--font-sans)`, or it is not written.*

3. **Four answers to "how wide is a page".** `results` caps at 1200 px,
   `--ts-page-max` at 1240, `modify` at 1680, and three tabs have no cap at all.
   `modify/style.css`'s own preamble already defers this — *"reconciling those
   three is a design decision about the whole app, not a tidy-up"* — and that is
   still true. Named here, not settled here.

### 5.5 What the sheet gives up

`modify/style.css` declares selectors that are not its own vocabulary:
`html, body`, `header`, `header h1`, `footer`, `footer a`, `.status`,
`.card[hidden]`, `.card-header`, `.ghost`, and a `:root`. That is
`plans/css-system-plan.md` § 3's rule — *a page sheet may contain only T3* —
unmet, and its steps **C** and **D** are exactly this work.

Two of them are cheap and belong with item 5 because it touches them anyway:
`.card-header` and `.ghost` are written by `modify.html` and **by nothing else
in the app** (3 and 10 occurrences, all in that one file). They are page
vocabulary wearing a shared name, so renaming them `.modify-card-header` and
`.modify-ghost` states the ownership and changes no pixel.

### 5.6 What this is not

Not a restyle. No new colours, no new type scale, no change to what the tab
*looks* like beyond where things sit — `tokens.css` is the scale and it is
fine, which is what `plans/css-system-plan.md` § 6 already says.

And nothing here touches `molview.css`. The viewer card's interior belongs to
the module, in CSS exactly as in JavaScript; the page arranges the section that
hosts it and stops at its edge.

---

## 6. What was measured, and what it showed

Everything below was run, not recalled.

### 6.1 The registries, on all three surfaces

`ase.build.fcc{111,100,110}('Au', size=(1,1,6), a=4.158, vacuum=0)`, layers low → high:

| surface | a₁ | a₂ | distinct registries | raw step, layer 1 → 2 | `(a₁+a₂)/period` |
|---|---|---|---|---|---|
| (111) | (2.9401, 0) | (1.4701, 2.5462) | **3** | (−1.4701, 0.8488) | (1.4701, 0.8487) |
| (100) | (2.9401, 0) | (0, 2.9401) | **2** | (−1.4701, −1.4701) | (1.4701, 1.4701) |
| (110) | (4.1580, 0) | (0, 2.9401) | **2** | (−2.0790, −1.4701) | (2.0790, 1.4701) |

The last two columns agree **modulo the lattice vectors** — e.g. on (111),
`−1.4701 + a₁ₓ = −1.4701 + 2.9401 = 1.4700`. The raw step's sign is what a reader
will actually see in the positions, which is why both columns are printed here
rather than only the tidy one.

### 6.2 Three routes to `a` from a 3×3×6 supercell

Built at `a = 4.158`; every route recovers it exactly:

| read from | value | conversion | recovered |
|---|---|---|---|
| in-plane \|A\| | 8.8204 Å | `√2·\|A\|/m`, m=3 | 4.1580 |
| z-period c | 14.4036 Å | `√3·c/N`, N=6 | 4.1580 |
| **nearest neighbour** | **2.9401 Å** | **`√2·d`** | **4.1580** |

The first two need the user's own `m` or `N` and the surface; the third needs
nothing. That is why § 3.3's door measures `d_nn`.

### 6.3 The dead column

```
Au: a_experimental=4.0782  a_pbe=4.158  a_pbe_siesta_psml=None
Ag: 4.0853  4.147  None      Cu: 3.6149  3.635  None
Ni: 3.524   3.52   None      Pt: 3.9242  3.967  None
Pd: 3.8907  3.943  None
```

### 6.4 3Dmol's mouse map, and why Clear is a button

| button | gesture |
|---|---|
| left | rotate |
| middle | pan |
| **right** | **zoom** (drag) |

`contextmenu` fires on right **mousedown**, before any drag — so a plain
`contextmenu` listener would clear the measurement every time a right-drag zoom
began. Distinguishing them needs a release-without-drag threshold, which is real
code for an invisible gesture.

The user proposed a **Clear button** instead; it removes the threshold, the
gesture and the discoverability problem together. Recorded here because the
gesture will look like an obvious idea again later.

---

## 7. Order, and where the items collide

| | item | why here | touches |
|---|---|---|---|
| 1 | **Center follows the group** (§ 2) | one branch in one route; no browser change | `blueprints/modify.py` |
| 2 | **Measurement track** (§ 1) | settles *what a click does*, which item 4 depends on | `stores.js`, `ui.js`, `mount.js`, `ui-context.js`, `molview.css` |
| 3 | **Cell setup** (§ 4) | no new route; must be settled before item 3 finishes, since the padding switch moves into it (§ 4.4) | `periodicity.js`, `modify.html`, `junction-cell.md` |
| 4 | **The surface** (§ 5) | before the two new panels exist, so they are built on one inset instead of inheriting five | `modify/style.css`, `modify.html`, `page-shell.css` |
| 5 | **New slab tab** (§ 3) | largest — new panel, new builder, new route, plus § 3.4's deletion | `modify.py`, `blueprints/modify.py`, new panel, `tabs.md` |

Three collisions, all handled above: item 1 before item 4's picking (§ 4.2),
item 4 before item 3's padding move (§ 4.4), and **item 5 before the new
panels** — a panel added after the inset rule lands needs no rework, one added
before it needs redoing (§ 5.3).

---

## 8. Contracts to change, before any code

| Document | What changes |
|---|---|
| `web/molview.md` | § 11.6 — the readout reads its own track, and the geometry guess goes · § 8.5 — the rail gains a sixth control · § 9.5 — the selection is untouched while measuring · § 11.2b — the lane carries the track · § 12.4 — the worked example |
| `web/tabs.md` | § 2 — the op-tab list, and the new slab tab beside the old |
| `web/web-api.md` | the `translate` entry's recenter rule · the new `lattice-from-run` route · the catalogue count |
| `science/junction-cell.md` | § 5 and § 6 — the padding switch's home moves to the Cell page, its reasoning intact · § 3 — registry becomes a **chosen parameter**, not only a warned-about outcome |
| `plans/bench-and-junction-plan.md` | § 2.3 closed as subsumed (§ 3.2) · § 2.4 restated as a check on a stated registry |
| `model/structure-periodicity.md` | § 6.2 — the two new gestures reaching the existing `cell` / `cell_origin` ops, and the handedness refusal they will routinely meet (§ 4.3) |
| `molbuilder/data/README.md` | the `a_pbe_siesta_psml` column leaves the table (§ 3.3) |
| `web/ui-contract.md` | § 4 — **one inset per card** and **one gutter per tab** join the rhythm rules (§ 5.3) · the `font:`-shorthand rule that keeps `--font-sans` the only family (§ 5.4) |
| `plans/css-system-plan.md` | § 4 — steps **C** and **D** record what item 5 discharges for `modify`, and what it leaves (§ 5.5) |

---

## 9. Decisions on the record

All from the user, 2026-08-30, in the exchange that produced this file.

| Decision | Effect |
|---|---|
| a fourth measurement pick **drops the oldest** | measuring a chain stays fluid; no state to clear by hand |
| clearing is a **button on the readout**, not right-click | § 6.4 — kills the drag threshold and the invisible gesture |
| the measurement track must not overlap the real selection | § 1.4's wall, and the item-2 hazard it names |
| **the group is the rigid part** — only selected atoms move | § 2.3's `indices` path |
| **nothing selected = everything**, one rigid move | § 2.3's no-`indices` path, box travels |
| dx, dy **and** the starting-surface z are **absolute**, from the 3D window's origin | the new slab panel reads no selection |
| "Your bulk run" is **extracted from a result file**, the user owning the setup's correctness | § 3.3's door |
| the extracted number belongs to **one optimization run**, never a global table | § 3.3's deletion of the packaged column |
| the tab's placement and alignment are **designed, not patched** | § 5.3's two rules, applied everywhere at once rather than per element |

---

> **Nothing here is built.** The next step is § 8 — the contracts — and only then
> § 7's order.

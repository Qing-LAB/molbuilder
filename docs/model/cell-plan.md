# The unit cell, and how a user says what they want — a plan

**Role:** plan
**Domain:** model
**Started:** 2026-08-03
**Companions:** [`structure-periodicity.md`](?doc=model/structure-periodicity.md)
(the settled rules this reconciles against — § 6.1's state table, § 6.2's regime
model, § 8.2's one-way-in contract); [`web/molview.md`](?doc=web/molview.md)
§ 9.5 (the Cell page, which is where a user actually meets all of this);
[`science/validation.md`](?doc=science/validation.md) (the checks that reach the
user before a calculation is generated).

---

## 1. The one idea everything here serves

**A generated script sets up a unit cell and puts the molecule in it.** The cell
is the output. Everything on the Cell page is either *the cell itself* or *a
parameter used to work one out*:

```
    axis kinds  ─┐
    vacuum      ─┼─→  [ derive ]  →  the unit cell  →  the generated script
    the molecule ┘                        ↑
    an explicit cell ─────────────────────┘   (when you set one, it IS the cell)
```

So there are two regimes, and which one you are in is the single most useful
thing a user can know:

| Regime | What decides the box | What vacuum does |
|---|---|---|
| **derived** | the molecule's size + your vacuum + the axis kinds | **it makes the box** |
| **manual** | the cell you typed | **nothing** — it is reference-only |

That is not new: `structure-periodicity.md` § 6.2 already says it, in those
words. This plan does not invent a model. It **reconciles the code and the UI
against the model that is already written down**, and fixes the places where a
user cannot tell which regime they are in or what their edit will do.

## 2. What is already right — verified 2026-08-03, not assumed

Each of these was run, not read:

| Claim in § 6.2 | Verified |
|---|---|
| an explicit cell demotes vacuum to reference-only | ✅ a 20 Å cell with vacuum 5 gives a 20 Å box, not 2 + 2×5 |
| editing vacuum **resets to derived** and says so | ✅ the explicit cell is cleared, and the notice explains the regime change |
| an origin is only meaningful with an explicit cell | ✅ refused otherwise, with the reason |
| clearing the origin re-derives the corner (does not jump to 0,0,0) | ✅ and the notice says what the corner became |
| the check runs on every structure the server hands back | ✅ § 8.2's seam 2 |
| a thin gap is warned about before a script is generated | ✅ `cell.vacuum_thin` reaches the tab's live panel — though only since the preflight was repaired on 2026-08-03; it had been answering `400` to every request that tab made |

**The model is sound. The gaps are in what a user is told and what a user can
say.**

## 3. What is wrong

### 3a. A user cannot say "I did not choose a vacuum"

`vacuum` is the only field in its own block with no *unset* state:

```
cell:         Optional  = None      ← "the structure says nothing"
cell_origin:  Optional  = None      ←  same
pbc:          Optional  = None      ←  same
axis_kind:    Optional  = None      ←  same
vacuum:       Tuple     = (0,0,0)   ←  the odd one out
```

So *"I deliberately want no gap"* and *"I never touched it"* are the same value,
and no rule can tell them apart.

**MolView already has the vocabulary for the distinction, on both sides**, and
cannot use it: `getVacuum()` is documented as *"what the structure actually says,
**null where it says nothing**"* — it can never return null, because the model
cannot produce it. `commitPeriodicityOp("vacuum", null)` is documented as
*"null clears"* — there is nothing to clear *to*. The wire already carries the
pair `vacuum: [0,0,0]` beside `resolved_vacuum: [3,3,3]`, and nothing reads it.

**Decision (user, 2026-08-03): give vacuum an unset state**, exactly like its
three siblings. No new door on MolView; the API already says all three states.

### 3b. The minimum-thickness floor fires far more widely than intended

Today the guard is one line:

```
if  extent + 2 × your_vacuum  <  3.0:      →  vacuum = max(yours, 3.0)
```

Measured on a molecule 2 Å across:

| | you typed | box used | |
|---|---|---|---|
| flat (extent 0) | 0.4 | 3.0 | raised |
| **thin, not flat (extent 2)** | 0.4 | **3.0** | **raised — should not be** |
| thick (extent 5) | 0.4 | 0.4 | kept |
| **flat, you typed 1.0** | 1.0 | **3.0** | **raised — should not be** |

**The intended rule (user, 2026-08-03)** — the floor is a guard against a box
with no volume, nothing more. It applies only when **all three** hold:

1. the axis is **isolated** (vacuum is meaningless on a periodic or transport
   axis — the lattice or the device length sets those), **and**
2. the molecule has **zero length** along that axis, **and**
3. **you did not set a vacuum** for it.

Anything you typed is **kept**, however small, and warned about. *You dictate
what you want.* The code currently checks only a version of (1).

### 3c. "Vacuum is not respected under a manual origin" is suppressed exactly
when it matters

The warning is emitted only `if not conditions` — that is, only when the
validator found nothing else to say. So when the box also has a problem, the
sentence explaining *why your vacuum stopped mattering* is dropped. It is not a
duplicate of the containment warning; it is a different fact about a different
parameter.

### 3d. The UI promises a warning it does not give

§ 6.2's table says the vacuum op *"resets to derived (explicit cell + origin
cleared; the boundary moves — **the UI warns before committing**)"*. There is no
such warning: the Cell page has **no** pre-commit confirmation anywhere. A user
with a hand-typed 20 Å cell who edits vacuum loses that cell, and learns
afterwards.

### 3e. The Cell page never says which regime you are in

It shows four values with a `(default)` tag meaning *"derived, not set on this
structure"*. That answers *"is this box mine?"* — but not *"is my vacuum doing
anything?"*, which is the question that actually decides whether an edit will
have an effect.

### 3f. The floor, when it does fire, is announced almost nowhere

It is reported only from the vacuum / axis-kind edit path. Load a flat molecule
and generate, and nothing says the box grew. Under 3b's narrowed rule this
becomes rare — but rare and silent is worse than common and silent, because
nobody expects it.

### 3g. Two checks, two thresholds, one physics

`cell.vacuum_thin` warns when the vacuum you **set** is under 8 Å per side;
`cell.image_distance` warns when the gap actually **achieved** is under 6 Å. Both
fire on a thin box with different numbers. They measure genuinely different
things — a setting versus a result — so the answer is probably *make each say
what the other is*, not delete one.

## 4. The plan

**Step 1 — the model can say "unset".** `vacuum` becomes `Optional`, like its
three siblings. Round-trips through `apply_metadata_dict` / `metadata_to_dict`,
the `.molstruct.json` schema, and the wire. **Open question, § 5.**

**Step 2 — the floor becomes the three-condition rule** (3b), and
`vacuum_floor_axes()` reports only what it actually raised.

**Step 3 — the floor is answered by the check every hand-over runs**, not only
by one edit path (3f), so it arrives as a condition like containment and the
derived corner. It must be an **Issue** as well if it is to reach Generate: the
emit seam discards the gate's notices.

**Step 4 — stop suppressing the manual-origin warning** (3c).

**Step 5 — the Cell page tells you the regime, and warns before it changes it.**

- A line at the top of the page: *"The box is worked out from the molecule, your
  vacuum and the axis kinds"* or *"The box is the cell you typed — vacuum is not
  used"*. One sentence, and it changes when the regime does.
- The vacuum row greys with a hint when it is inert (manual regime), rather than
  showing an editable number that does nothing.
- Editing vacuum or axis kinds while an explicit cell exists asks first, naming
  the cost: *"this clears the cell you typed and works a new box out from the
  molecule"* (3d). The confirm module already exists and is already generalised.
- Where the floor applied: show it in the row — `0.0 → 3.0 used` — with the
  reason, instead of a number with no history.

**Step 6 — one story for the two thin-gap checks** (3g).

**Step 7 — the help text, consolidated.** Today the Cell page carries a
`(default)` tooltip and nothing else; everything a user would need to understand
the two regimes lives in a document they are not reading. Each row gets a short
hint saying what it *does*, not what it *is*:

| Row | Hint |
|---|---|
| Lattice | *the box the calculation runs in. Type one to fix it; leave it and it is worked out from the molecule, the vacuum and the axis kinds* |
| Origin | *which corner the box starts at. Only meaningful once you have typed a lattice* |
| Axes | *periodic repeats forever · isolated is a molecule in a box · transport is a device length. Vacuum only applies to isolated axes* |
| Vacuum | *the empty gap left on each side of the molecule when the box is worked out. Ignored when you have typed a lattice. ≥ 8 Å per side is the usual advice for an isolated molecule* |

## 5. The one question this plan cannot answer by itself

**Every `.molstruct.json` on disk today says `vacuum: [0,0,0]`.** Under the new
rule that reads as *"I explicitly chose zero"*, not *"unset"* — so a flat
molecule saved last week would stop getting its 3 Å guard, silently, on the next
load.

Three ways out, and it is a decision, not a detail:

1. **Treat a stored `[0,0,0]` as unset** on read. Old files keep behaving as they
   did; a user who genuinely means zero has to type something once the writer
   starts emitting `null`. Simple, and slightly dishonest for exactly one value.
2. **Migrate on read**: `[0,0,0]` → `null`, rewritten on the next save. Honest,
   and it changes files the user did not ask to change.
3. **Take `[0,0,0]` at its word.** Cleanest rule, and it silently changes the box
   for existing flat-molecule structures — which is the failure this whole plan
   exists to prevent.

Nothing in steps 1–7 should land before this is settled, because step 1 is what
makes it live.

## 6. What this plan does not do

- **It does not change the regime model.** § 6.2's two regimes and their
  transitions are the design; everything here reconciles code and UI *to* them.
- **It does not add a MolView door.** The API already says all three states; the
  model is what cannot supply them.
- **It does not make a thin vacuum an error.** A small gap is legal and sometimes
  deliberate. It is warned about, before a script is generated, and the user
  decides — which is the project's rule for anything that is not physically
  impossible.

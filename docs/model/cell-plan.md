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

### 3d. ~~The UI promises a warning it does not give~~ — WRONG, withdrawn 2026-08-03

**This entry was my error and is retracted.** It claimed *"the Cell page has no
pre-commit confirmation anywhere"*. There is one, and it is correct:
`modify/periodicity.js::_confirmReset` fires on the vacuum and axis-kind
updates, and only when `getUnitCell() !== null` — exactly the destructive case,
where a typed cell is about to be discarded. Switching an axis *to* `periodic`
keeps the explicit cell, and correctly skips the confirm.

**How I got it wrong, because the shape of the mistake matters.** I read
MolView's own Cell page (`lib/molview/ui.js`) and found no confirm. That page is
a **read-only readout** — a `<dl>` of four values plus the notice list; it has
no edit doors at all. The editing lives in the host's form,
`modify/periodicity.js`, which consumes MolView's public doors through the
viewer handle it was given (§ 5.6). Judging a behaviour absent from one of the
two surfaces that implement it is exactly the "surface glance" failure the
module-provenance rule exists to prevent.

What is genuinely open is smaller, and is wording: the confirm does not name the
cell being lost, and does not say the edit cannot be undone — and it cannot be,
since periodicity ops never enter MolView's history (`commitPeriodicityOp` →
`applyCell`, no history call). It also fails **open**: with no
`window.molbuilder.warningModal` it resolves `true` and commits unconfirmed.
`warning-modal.js` does ship on every page that mounts the form
(`_molview_scripts.html`), so this is a safety net rather than a live defect.

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

**Step 1 — the model can say "unset". ✅ landed 2026-08-03.** `vacuum` became
`Optional`, like its three siblings, and round-trips through
`apply_metadata_dict` / `metadata_to_dict`, the `.molstruct.json` schema, and
the wire (`null`, not `[0,0,0]`). A stored `[0,0,0]` reads as **unset** (§ 5),
so nothing on disk changes meaning — written down in
[`structure-molstruct.md`](?doc=model/structure-molstruct.md) § 1 where the
reader is.

*Also required, and found only by using it:* `commitPeriodicityOp("vacuum",
null)` has always been documented as *"null clears"*, and the op rejected it
("must be 3 non-negative floats") because until now there was nothing to clear
**to**. The door now accepts `null`.

**Step 2 — the floor becomes the default gap. ✅ landed 2026-08-03**, and it is
simpler than the three-condition rule sketched in 3b. The user's correction:

> *3 Å is the vacuum **distance**, not the size of the molecule.*

So the condition "the molecule is flat along this axis" fell away entirely. Two
states, not three conditions: **a vacuum is set → used verbatim, however small;
nothing is set → every isolated axis gets 3 Å per side, whatever the molecule's
size.** `vacuum_floor_axes()` was renamed `defaulted_vacuum_axes()` — nothing is
"raised" any more, so the old name described a mechanism that no longer exists.

**Step 3 — the default is answered by the check every hand-over runs. ✅ landed
2026-08-03.** `validate_periodicity` now reports on a *derived* box (it used to
return in silence the moment `cell is None`), so two facts finally reach a user
who merely loads a structure: the default gap is in use, and — separately — the
box has no volume. The old edit-path-only producer (`_floor_notices`) was
**deleted**: the door re-validates the result of every edit, so keeping it
delivered the same sentence twice.

*Not yet done in this step:* reaching **Generate** as an `Issue`. The emit seam
discards the gate's notices, so the disclosure is a notice only. `cell.vacuum_thin`
does already reach Generate and does fire on the default (see § 6.1a of
`structure-periodicity.md`), so the *thin-gap* half is covered; the *"this number
is a default"* half is not.

**Step 4 — stop suppressing the manual-origin warning** (3c). ✅ **landed** —
and not by unsuppressing a receipt. The fact became a CONDITION,
`cell.vacuum_ignored`, reported on every hand-over whenever it is true. A
receipt says what an edit did; this says what is *true until the regime
changes*, which is why it was in the wrong place to begin with.

**Step 5 — the Cell page tells you the regime, and warns before it changes it.**
✅ **landed** (user decisions, 2026-08-03).

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

**Step 6 — one story for the two thin-gap checks** (3g). ✅ **written down**
2026-08-03, in [`structure-periodicity.md`](?doc=model/structure-periodicity.md)
§ 6.1a; the user supplied the physical definition that resolves it:

> *The validation is testing the distance between atoms and the next periodic
> cell. In z: (top of the box − the highest atom) + (the lowest atom − the
> bottom of the box) is the closest approach between atoms in two adjacent
> cells.*

That is exactly what `cell.image_distance` measures (`_min_image_distance`
generalises it — a true minimum over all atom pairs and translations, so it is
also right for an off-centre molecule or a skewed cell). It gives the two checks
one currency: **vacuum is per side, so the image gap is twice it.**

| Check | Asks about | Warns below | The same number, other side |
|---|---|---|---|
| `cell.vacuum_thin` | the vacuum you **set** or defaulted to | 8 Å/side (25 charged) | a 16 Å image gap |
| `cell.image_distance` | the gap **achieved**, measured | a 6 Å image gap | 3 Å/side |

They are **nested, not contradictory** — advice versus alarm. Measured: a
4 Å/side box trips the advice only; the 3 Å default trips the advice only; an
8 Å/side box is quiet. One code change came with it: `cell.vacuum_thin` now
reads `effective_vacuum()` (it read `struct.vacuum[i]` and **crashed** on the new
`None`), and is **skipped in the manual regime**, where vacuum is reference-only
— a molecule in a hand-typed 30 Å box was being told its vacuum was thin.

**Step 7 — the help text, consolidated.** ✅ **landed** — each row of the Cell
readout carries a `title` saying what it *does*.
 Today the Cell page carries a
`(default)` tooltip and nothing else; everything a user would need to understand
the two regimes lives in a document they are not reading. Each row gets a short
hint saying what it *does*, not what it *is*:

| Row | Hint |
|---|---|
| Lattice | *the box the calculation runs in. Type one to fix it; leave it and it is worked out from the molecule, the vacuum and the axis kinds* |
| Origin | *which corner the box starts at. Only meaningful once you have typed a lattice* |
| Axes | *periodic repeats forever · isolated is a molecule in a box · transport is a device length. Vacuum only applies to isolated axes* |
| Vacuum | *the empty gap left on each side of the molecule when the box is worked out. Ignored when you have typed a lattice. ≥ 8 Å per side is the usual advice for an isolated molecule* |

## 5. Existing files — decided 2026-08-03

Every `.molstruct.json` on disk today says `vacuum: [0,0,0]`. Under the new rule
that would read as *"I explicitly chose zero"*, so a flat molecule saved last
week would silently lose its 3 Å guard on the next load.

**Decision (user): treat a stored `[0,0,0]` as unset on read.** Old files keep
behaving exactly as they do now — nothing on disk changes, and nothing silently
changes meaning. A user who genuinely wants a zero gap says so once the writer
starts emitting `null` for unset, and from then on the two are distinguishable
in new files.

It is slightly dishonest for exactly one value, and that is the price of not
rewriting files nobody asked us to touch. Write it down where the reader is:
the sidecar schema doc must say that `[0,0,0]` is read as *unset*, so nobody
later "fixes" the asymmetry without knowing what it protects.

## 5a. The periodicity truth must reach the check that acts on it

**Instruction (user, 2026-08-03): make sure `pbc` is used in the validation
path.** Chasing it turned up a live science-correctness defect, and clarified
what the instruction has to mean.

**What `pbc` is.** `axis_kind` is authoritative and is never `None` after
construction; `pbc` is its derived view — `periodic → True`, `isolated → False`,
**`transport → True`**. Validation keys on `axis_kind` everywhere, so it already
honours the periodicity truth: the two cannot disagree. Switching validation
*to* `pbc` would be a regression, because `pbc` collapses transport and periodic
into one and the k-grid check needs them apart (k>1 is *wasted* on an isolated
axis but *wrong* on a lead).

**The real gap: the check has to know what the ENGINE will do with it.** Fixed
2026-08-03:

> A structure periodic on all three axes generated a **gas-phase PySCF script**
> — molecular `gto.M()`, no lattice, no k-points — with the cell dropped and
> **nothing said**. A 5.6 Å NaCl cell produced a two-atom isolated cluster. Not
> a rough version of what was asked for: a different calculation, and a
> plausible-looking one. The PySCF validator's own comment said why — *"`cell` is
> unused (PySCF jobs are gas-phase)"* — an assumption never checked against the
> structure in front of it.

Now `cell.periodic_in_gas_phase` names the repeating axes, the lattice being
dropped, and what you get instead. **Warn, not error** (user): an
isolated-cluster run of a periodic input is legal, and only the physically
impossible refuses — an error would mean no script at all. It reaches the tab's
panel before the click, and the CLI through `report(validate(...))`.
Seven tests in `tests/test_pyscf_periodicity_check.py`.

**What this leaves open** — worth its own look, not part of this plan: a
transport structure loaded from an extended XYZ carries `pbc="T T F"` and no
axis kinds, so its axes come back `periodic`, never `transport`. The lossy
direction runs inbound too.

## 6a. One process line for the cell — ✅ SHIPPED 2026-08-03

**Instruction (user): one single data process line; not too many hands on the
process; a unified tool/API chain for cell setup and validation.**

### What the hands are today — counted, not estimated

**Setting the box.** Six methods on `Structure` (`resolve_cell`,
`resolve_cell_origin`, `effective_vacuum`, `expected_cell_corner`,
`cell_contains_atoms`, `defaulted_vacuum_axes`) plus two delegating wrappers in
the gate (`expected_corner`, `contains_atoms`). They call each other:
`resolve_cell_origin` → `expected_cell_corner` → `effective_vacuum`, and
`resolve_cell` → `effective_vacuum` again. Resolving one box computes the
effective vacuum three or four times, and a caller can enter at any of the six
and get a partial view of the answer.

**Judging the box.** Two parallel finding systems for one subject:

| System | Shape | Produces |
|---|---|---|
| `periodicity_gate` notices | `{level, message, about}` | 4 conditions + 8 receipts |
| `validation/` issues | `{severity, message, where}` | `cell.vacuum_thin`, `cell.image_distance`, `cell.periodic_in_gas_phase` |

…plus three bare `raise ValueError`s. **"This box has no volume" is decided in
four places at two different thresholds:**

| Where | Threshold | Mechanism | Covers |
|---|---|---|---|
| `structure.py:458` | `1e-8` | raises | an explicit cell, at construction |
| `periodicity_gate.py:142` | `1e-8` | `warn` notice | a derived box, on hand-over |
| `periodicity_gate.py:346` | `1e-6` | raises | a derived box, at the edit |
| `siesta/input.py:383` | `1e-6` | raises | the emitted cell |

Only the delivery contract keeps this coherent by hand — and
`science/validation.md` R1–R6 already says findings travel **one** path, so the
notice channel is a second one that predates the rule.

### The chain

```
   the pair (.xyz + .molstruct.json)      the truth — § 6.1 clause 1
              │
              ▼
   Structure                              holds it; resolves nothing itself
              │
              ▼
   cell.resolve(struct) ──► ResolvedCell  ONE resolver, computed ONCE
              │             .box .corner .vacuum
              │             .regime .defaulted_axes
              ▼
   cell.check(resolved) ──► [Issue, …]    ONE checker — every cell fact is an
              │                            Issue with a stable `where`
              ▼
   report(issues, intent)                 ONE verdict
        intent=generate  → raises on error severity
        intent=edit/load → returns them
```

Three named things replace the spread:

1. **`ResolvedCell`** — a value object built once per hand-over, carrying the
   box, the corner, the effective vacuum, the **regime** (derived / manual) and
   **which axes were defaulted**. Removes the recomputation and the "which of
   six methods do I call" question. Every consumer reads fields; nobody
   re-derives.
2. **`cell.check(resolved) → [Issue]`** — every cell condition becomes an Issue
   with a `where`. The gate's four *condition* notices retire into it.
   **Receipts stay separate**, and should: a receipt says what an edit just did
   and is meaningless a moment later, which is not what a finding is.
3. **`report(issues, intent)`** — already exists and already raises on error
   severity. Make it the only place refuse-vs-report is decided, per § 8.2.

### What it buys

* `cell.no_volume` becomes **one** check at **one** threshold, replacing four
  judgements at two.
* Cell conditions reach **Generate** for free — closing step 3's remainder with
  no special case, because Issues already travel there and notices never did.
* One vocabulary: `where` is the stable id the delivery contract already
  mandates, so the Cell page and the preflight panel show the same finding.

### What it costs — stated, not buried

The periodicity door's response shape changes: conditions move from `notices` to
`issues`, so it returns **both** (`receipts` + `issues`). MolView's
`applyCell(block, notices)` and the Cell page's `drawNotices` take issues
instead. Contained to `model-jobs.js` + `ui.js`, **no change to MolView's seal**
— `index.js` still exports `mount` and `formula`, and no door is added.

### What actually landed

`molbuilder/cell.py` (L1 — imports `structure` and `issues`, nothing else, which
is what lets the gate, the validators and the emitter all ask *it* instead of
each answering for themselves).

| Was | Is |
|---|---|
| 4 zero-volume judgements at 2 thresholds | one `cell.no_volume`, one `cell.ZERO_VOLUME_TOL` that all four sites import |
| gate notices `{level, message, about}` **+** validator `Issue`s | one `Issue` from `cell.check`; `notices_for_report` is the ONE serializer onto the wire |
| notices with no id, so tests matched prose | every notice carries `where`, the same stable id `Issue` uses |
| `validate_periodicity` walked the § 6.1 table in five branches | it calls `cell.resolve_and_check` and hands the findings on |
| `_require_right_handed` / `_too_small_axes` / `_derived_box_notices` / `_floor_notices` | deleted — the rules live once |
| `ok_structure_response` caught the gate's exception and rebuilt a notice from `str(exc)` | it asks the checker directly; § 8.2 says a modify door reports, so there is nothing to catch |
| "your vacuum is inert" emitted only `if not conditions` (§ 3c) | `cell.vacuum_ignored`, a CONDITION, said whenever it is true |
| cell facts never reached Generate | `validate()` folds `cell.check` in, so they arrive in the preflight panel |

Eight findings, one per cause, verified across ten structures — a zero-volume
box no longer also reports "not right-handed" and "atoms outside", which was
one cause wearing three names.

**MolView's seal is untouched**: `index.js` still exports `mount` and `formula`,
and the wire key stayed `notices`, so `where` is additive.

## 6. What this plan does not do

- **It does not change the regime model.** § 6.2's two regimes and their
  transitions are the design; everything here reconciles code and UI *to* them.
- **It does not add a MolView door.** The API already says all three states; the
  model is what cannot supply them.
- **It does not make a thin vacuum an error.** A small gap is legal and sometimes
  deliberate. It is warned about, before a script is generated, and the user
  decides — which is the project's rule for anything that is not physically
  impossible.

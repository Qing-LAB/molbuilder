# MolView — building the code from the document

**Role:** plan
**Domain:** web
**Started:** 2026-07-30
**Companions:** [`molview.md`](?doc=web/molview.md) — the contract this builds. Retired at
closeout, when the code has caught up and the contract stands alone.

[`molview.md`](?doc=web/molview.md) is what must be true. The code is what changes.

---

## 1. How this runs

**The old tree is frozen; the new one is built from the document.** Today's `lib/molview/`
moves to `lib/molview-old/`, which **nothing imports** — reference material, not code.
`lib/molview/` starts empty and is built file by file from the contract. A piece of the old
code crosses over only when it is deliberately worth reusing; nothing arrives because it was
already there.

That is a fix, not a preference. Every failure of the first attempt came from editing
*around* code too large to hold in view — a region cut that took a helper with it, a pattern
that stripped lines out of multi-line statements, a deleted block that left its closing
brace. Writing a file from a contract section has none of those failure modes, and there is
no half-migrated state: the new tree either has a file or it does not.

**One unit at a time, and its tests are written first, from the contract.** § 13.3's rows are
the test plan; § 13.2's first two levels — behaviour in node, boundary behaviour with
stand-ins that obey the *document* — are what a data structure and an API contract are
checked with. **Run only those.** § 13.2's third level, § 1.1 end to end, is a
finished-module check: a page-level test fails for anything on the page, so using one to
judge a low-level change says nothing about the change and throws away work that was correct.

**That is the verification before every commit:** the concentrated tests written from the
contract for the unit that just landed. Not the demo, not the package.

**The old tests are retired, not repointed.** All nineteen whose subject was MolView moved
to `lib/molview-old/_retired-tests/`, outside `testpaths`, and are deleted with the frozen
code at closeout. Repointing them would measure the new code against the old code's shape:
§ 13.1 rules out exactly that — a pinned list of names "passes for a surface that has
drifted away from this document, and fails for a rename that changed nothing". Tests
belonging to *other* modules that reach into MolView were left where they are; they break,
and that is not repaired from here.

**`window.molbuilder` is never typed.** MolView publishes nothing and reads nothing from the
app's global namespace — § 4 in both directions: *"it is imported by name, it reaches nothing
else by name, and nothing in the app can reach inside it … Nothing it needs comes from a
global."* The workspace arrives as an argument to `mount`. Building fresh makes this free: a
global only exists if someone writes one.

**Nothing outside this module is consulted, accommodated or repaired.** How anything else
uses any code is not this plan's business. If this leaves something outside broken, it stays
broken.

---

## 2. The tree

One file per layer, one per API — § 7's levels and § 9's surfaces.

| § | File | The API it is |
|---|---|---|
| 9.1 | `index.js` | `mount` + `formula`, and nothing else is importable |
| 11.5 · 9.5 | `_atom.js` | the numbering translation; what a filter row can match |
| 9.9 | `3dmol-embed.js` | the only file that names 3Dmol |
| 9.7 · 9.8 | `render-engine.js` | what to redraw and at what cost · the per-frame maths · the drawing commands |
| 9.3 | `model.js` | the master copy, the data API, the read-only gate |
| 7.3 | `model-jobs.js` | load a structure in · write it out · the geometry edits |
| 11.2 | `history.js` | the sequence and the position on it; the write machine; the badge |
| 9.5 · 9.6 | `stores.js` | `selection` (what is picked, the switches) · `view` (how it is drawn) |
| 8 · 9.2 | `mount.js` | assembles the viewer; the handle — lifecycle, playback, `data` |
| 1.1 · 11.4 · 11.6 | `ui.js` | the panel, click-to-select, the frame bar, the switches, the View menu, the Export menu, the readout, the badge |
| 13.4 | `demo.js` | the in-repo demo page |
| — | `molview.css` | one stylesheet, one link |

`_atom.js` stays its own leaf: the model, the stores, the render engine and the UI all read
it, so folding it into any one of them either duplicates the translation — breaking § 11.5's
single home — or makes a lower level import from a higher one.

`history.js` stays out of `model-jobs.js`: § 11.2's claim that *"the mechanism does not know
or care what is in it"* only holds while it is not sitting beside the serialiser it is handed.

---

## 3. The order

**Layers first, then the embed, then the data structures, then up.**

### A — The layer skeleton
Every file of § 2 created with nothing in it but its contract header: what it owns, who calls
it, and its **"never"** column from § 7. The shape exists before any behaviour does, so each
file that follows is written into a place that already says what it may not do.

### B — The 3Dmol embed
`3dmol-embed.js`, carried across and carved to § 9.9's job: the movie, the camera, the
styles, the picking, the highlight — and **two questions answered upward**, both self-checks
(§ 10.10). This is the one file NOT written from the document: 6,500 lines of hard-won
knowledge about a library that punishes guessing — that restyling one atom rebuilds the whole
model's geometry, that shapes need re-placing every frame, the batching behind § 10.7's
measurements. The contract records the conclusions, not the calls.

Out of it and into the layers above: the card scaffold and info line (`mount.js`), the knob
bar (`ui.js`), the frame strip and its animation interval (`ui.js` + mount's one timer), the
export menu, snapshot and GIF encoder (`ui.js`, § 11.4). Deleted outright: the
`molbuilder.projects` reach and the `/api/files/*` calls — a file route at the bottom of the
stack (§ 6.7), and task **#39**.

### C — The data structures
§ 6.2's shapes as the module's vocabulary — `Structure`, `Coordinates`, `DisplayedFrame`,
`Selection`, `Switches`, `ProcessedFrame`, `Scene`, `ViewSettings` — plus `_atom.js` and the
per-frame maths of § 10.3 and § 6.5. Pure, no browser, tests first.

### D — The model
`model.js` and `model-jobs.js`: the master copy and the displayed frame with its range
(§ 6.3, § 6.4's ordering), § 9.3's fourteen needs with one main way in each, and the
read-only gate — one question asked of every truth-changing door, a no-op that does not throw
(§ 9.4).

### E — The stores and the history
`stores.js` — `selection` with the switches (§ 9.5) and `view` with the four drawing settings
and **no camera anywhere** (§ 9.6). `history.js` — § 11.2 whole: point 0, a Retract that
spends unsaved work first, the badge, and SETTLED / CHANGING / WRITING.

### F — The render engine
`render-engine.js`: § 10.5's four costs chosen by what changed and never by atom count,
§ 10.9's rebuild window where nothing that arrives is dropped, § 10.10's self-checks. Handed
the master copy; holds nothing.

### G — Mount, the handle, the UI
`mount.js` assembles the card, the panel, the controls and MolView's own menu, and returns a
handle that is lifecycle, playback and `data` (§ 9.2). `ui.js` is every control MolView
draws, each a caller of the model. § 1.1 end to end lands here — the first browser check of
the program.

**The card panel layout and its stylesheet come across as they are.** That design is
validated — it works — so it is carried from the frozen tree, not redesigned. What changes
is what sits *behind* each control: every one becomes a caller of the model, through the
same doors and the same read-only gate as anything else. This step is a rewiring, and it
should be hard to see that it happened.

### H — Closeout
`lib/molview-old/` deleted. Every **Transition** note in the contract whose code has caught up
removed — it should end with none. § 11.1's route sentence, which says three routes and omits
the filter's two. § 15's file map rewritten to § 2's tree.

---

## 4. What the first attempt established

Findings worth keeping, not code.

- **A namespace rename left a runtime global lookup behind** — `mount.js` still asked for
  `mvApi.engine.create` after the module renamed what it published, and the guard around it
  turned a missing factory into a no-op, so **every viewer mounted and then never drew**.
  Invisible to node tests, which stub the engine.
- **Per-frame data carried appearance.** Every force arrow shipped a `color` and a `radius`
  on every frame of every trajectory, against § 6.5. Appearance belongs beside the highlight
  constant in the embed.
- **The master copy lived in the render engine**, which called itself *"the source of truth we
  own"* while the model called itself *"a thin coordinator"* — § 7 level 5 holding everything,
  and § 6.4's ordering rule with nothing to stand on.
- **A held cell edit or force update was dropped on the floor** — the replay matched command
  names a rename had already changed, so an op arriving during a rebuild was held exactly as
  § 10.9 requires and then matched nothing on the way out.
- **Forces arriving on an append were dropped when the load carried none**, so a run caught at
  its first geometry never grew arrows however many polls arrived.

---

## 5. Open

1. **Canvas-state** — a third home for the structure (§ 6.3 allows two). Its structure fields
   belong in the model, per owner; the rest is not MolView's to hold.
2. **`factsForRequest`** — § 9.3 retires it in favour of one read plus one translation. It is
   named in another document's contract, so retiring it is a decision there.
3. **§ 11.1's route sentence** — one line, your wording.

# Persistency on the Molbuilder tab — what is kept, by whom, and what is missing

**Role:** investigation
**Domain:** web
**Started:** 2026-08-03
**Companions:** [`molview.md`](?doc=web/molview.md) § 11.2 · § 11.2a · § 11.3 —
the timeline, the restore, and the four kinds of saving.
[`workspace.md`](?doc=web/workspace.md) — the module that moves the bytes.

---

## 1. Five things on this tab look like saving. Only four are named anywhere.

molview.md § 11.3 lists four: **a saved point**, **the draft**, **Export → Data**,
**Export → Image**. It is a good list and it is MolView's list — every one of the
four is read out of the viewer.

The Molbuilder tab has a fifth that the contract never names, and a sixth that
nothing keeps at all:

| # | What the user is doing | Who owns it today | Survives a reload |
|---|---|---|:--:|
| 1 | **come back to this later** — Save state | MolView's timeline (`history.js`) | **yes** (since 2026-08-02) |
| 2 | **do not lose this if the tab closes** — the draft | MolView's timeline, written on every edit | **yes** (same change) |
| 3 | **take the structure away** — Export → Data | MolView's Export menu | n/a |
| 4 | **take a picture away** — Export → Image | MolView's Export menu | n/a |
| 5 | **put this in my project** — **Save to project** | **the tab**, via `projects.parser.saveMolecule` | n/a |
| 6 | **where I was working** — which file, where it saved, which panel was open | **the tab, in memory only** | **NO** |

Rows 5 and 6 are the subject of this document. Rows 1–4 are settled and correct.

## 2. Row 5 — "Save to project" is Export → Data, arrived at from the other side

Both write **the same pair** — `<name>.xyz` + `<name>.molstruct.json` — and both
read the same truth. They differ only in who chooses the name and where the bytes
land:

| | Export → Data | Save to project |
|---|---|---|
| door | MolView's Export menu | the tab's Save panel |
| reads | `exportFile(range)` | `exportFile(range)`, through `projects.parser.saveMolecule` |
| destination | a download | a path in the project tree |
| names it | the export stem (§ 11.4) | the user, in a dialog |

**So there is one act with two front doors, and § 11.3 documents one of them.**
That is not necessarily wrong — "download it" and "put it in my project" are
genuinely different intents — but the contract should say so, because today a
reader of § 11.3 would conclude the Export menu is the only way the truth leaves
the viewer, and it is not.

**This is where task #39 already lives** ("give MolView its own menu surface;
Export moves there and stops dropping the sidecar"). Worth resolving together:
if the sidecar is dropped by one path and written by the other, the two doors
disagree about what "the structure" is.

## 3. Row 6 — the tab keeps its own context, and keeps it nowhere

**The structure comes back after a reload. The context around it does not.**

| The tab knows | Held in | After a reload |
|---|---|---|
| which file is on the canvas (`_loadedFrom`) | `selection-bootstrap.js`, a closure variable | **lost** |
| where it last saved (`_lastSavedTo`) | `structure/page.js`, a module variable | **lost** |
| which op sub-tab is open (Atom / Transform / Junction / Cell) | a DOM class, `is-active` | **lost** — back to Atom |
| which Init source is open (Load / SMILES / DNA / …) | a DOM class | **lost** — back to Load |
| every op form field (element, dx/dy/dz, gap, electrode side…) | the DOM | **lost** |

Nothing under `modify/` writes to the workspace. It has no tag of its own.

### 3.1 Why this is worse than untidy

**The loader readout lies after a genuine restore.** `_loadedFrom` is set only by
`_commitFile` — the path that reads a file. When the reload is served by the
*restore* instead (which is now the normal case), `_loadedFrom` is empty while a
structure is plainly on the canvas. The readout falls to **"Picked: X"** and the
Load button **re-enables** — inviting the user to press Load and discard the very
work that was just restored for them.

**And the two paths are indistinguishable on screen.** Whether the reload
restored your session or silently re-read the file from disk, the page reads the
same: *"Loaded RAW_BDT.xyz — 14 atoms."* Those are different events with
different consequences — one preserved your unsaved edits, the other threw them
away — and nothing tells them apart.

**The Save readout forgets its target.** `markSavedTo` is set on a successful
save, so the readout says *"Target: wire.xyz"* — until a reload, after which it
says *"Save as… into structure/"* again, as though the structure had never been
saved anywhere.

### 3.2 The rule — it is already written, in two places

Nothing here is new design. Both halves are stated in the contracts:

| | Where |
|---|---|
| A page may have several savers; each says its tag on every call, decides what goes in the bytes, and decides when to save. **`"modify:panel"` is named as the example** — *"the Modify tab has a viewer holding a molecule AND its own panel state"* | workspace.md § 4 and § 6 |
| MolView asks for nothing until the first structure is in; `installMolecule` lays down point 0; edits rewrite the draft and lay down no point; `save(1)` and `load(-1)` are the user's | molview.md § 11.2a, *"Starting, in an editable viewer"* |

So the viewer saves under `modify`, the page saves under `modify:panel`, and two
tags are two slots. **They cannot disagree**: each writes what *it itself did*, at
the moment it did it. The page writes `loadedFrom` because the page performed the
load, and writes `null` when the user generates instead, because it performed that
too.

### 3.3 What was built (2026-08-03)

`structure/page.js` owns both facts and is the single writer of the slot — two
writers on one state file is how one silently drops the other's field. It gained
`markLoadedFrom`, `restorePanelNote`, `getLoadedFrom` and `onPanelChange`
alongside the existing `markSavedTo`.

The note is written at the two gates every install passes through:
`loadIntoCanvas` (all six generators and the upload — it already knows whether a
file was involved) and `_commitFile` (the sidebar load). It is read at mount,
before the seed decision.

**One thing the build turned up that reading could not.** `installMolecule`
announces to subscribers from *inside* the call, before the promise resolves — so
a readout listening on the viewer re-renders while the note still holds the
previous filename, and nothing tells it to look again. A molecule generated from
SMILES sat under *"Loaded: water.xyz"*. The note therefore has its own change
channel (`onPanelChange`): the page's own state, the page's own readers, and
nothing asked of the viewer, which has no business knowing a file was involved
(§ 6.7). Pinned by `test_a_generated_structure_claims_no_file`.

## 4. What is already correct, and should not be touched

- **The timeline** (`history.js`) — MolView's submodule, private, reached only
  through `save` / `load` / `undo`. Verified: nothing outside `lib/molview/`
  imports it.
- **The draft** — rewritten on every edit, carrying the position and the badge
  since 2026-08-02, so a reopened page comes back where it was.
- **The split between them** — an edit rewrites the draft and adds no point; a
  save adds a point and leaves the draft's job alone. § 11.3's central rule.
- **Failure surfacing.** A failed write is not silent: the dispatcher reports it,
  `app-notifications.js` listens, and the bar shows *"Couldn't save state to disk
  (write-state · HTTP 404). Your edits are kept in memory, but retract /
  crash-recovery history may be incomplete."* Confirmed live, with a repeat count
  rather than a stack of identical rows.

## 5. What is left

**One piece of work, and it is small** (§ 3.4): the page writes its own two facts
under its own tag, and reads them at mount. No new API, no change to MolView, no
change to the workspace, no change to the server. It closes the Load-button bug
in § 3.1 by construction.

**One contract question**, which is not this document's to answer: is *Save to
project* a fifth kind of saving, or Export → Data with a destination (§ 2)?
molview.md § 11.3 names four and this is a fifth door onto the same act. It
belongs with task #39, which is already about those two doors disagreeing over
the sidecar.

**One nicety, worth a sentence if it is wanted:** a restore and a re-read of the
file currently read identically on screen. They have different consequences, and
saying which happened is one line of status text.

---

### A note on how this document went wrong first

An earlier draft raised an "ordering problem" between the page's note and
MolView's draft — two independent writes, no atomicity, so they might disagree —
and proposed reconciling them.

**There is no such problem.** Each saver writes what *it itself did*, at the
moment it did it: the page writes `loadedFrom` because the page performed the
load, and writes `loadedFrom: null` when the user generates instead, because it
performed that too. Neither is guessing at the other's business. Their data is
orthogonal by tag, which is what workspace.md § 4 means — a tag is a wall, and
walls do not need locks between them.

The invented problem produced an invented requirement (some coordinating
mechanism), which would have produced code nobody needed. Worth leaving in the
record: **when a design starts asking for a mechanism the contract never mentions,
the likely fault is in the reading, not in the contract.**

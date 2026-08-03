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

### 3.2 What the rule should be

§ 11.2 already gives the test, for MolView's own state:

> **State is the truth. What you are looking at is not state.**

The same line sorts the tab's context, and it does not put everything on one
side:

| The tab's fact | Truth, or a way of looking? | Keep it? |
|---|---|---|
| which file is on the canvas | a fact about **what you are working on** — the same class as the structure itself | **yes** |
| where it last saved | a fact about **what you did** | **yes** |
| which op sub-tab is open | where the user's attention was — a **view** | no |
| which Init source is open | a view | no |
| op form fields (dx, gap, element…) | arguments to an act not yet performed — a view | no |

The first two are the ones whose absence produces a wrong readout. The last three
are genuinely "what you were looking at", and § 11.2 is right to drop them: a
reopened tab opening on the Atom panel is correct, not a bug.

### 3.3 The door already exists

The workspace is public and the **tag** is the mechanism for several savers to
share one page (workspace.md § 4). MolView writes under `modify`. The tab would
write under a tag of its own — `modify-page` — and the two would not touch.

```js
ws.persist("modify-page", { loadedFrom, lastSavedTo },
           { workspace_id: ws.workspaceId("modify-page"), state_index: 0 });
```

That is exactly the shape `lib/inspectors/structure.js` already uses for
`SHOWING_TAG`, which is the same problem — *which file is this panel showing* —
solved once already, one directory away. **The tab should not invent a mechanism;
it should copy that one.**

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

## 5. Open questions, and one that is a bug

1. **Is "Save to project" a fifth kind, or Export → Data with a destination?**
   § 11.3 should say either way. Bundle with task #39.
2. **Which tag does the tab write under?** `modify-page` keeps it clearly apart
   from MolView's `modify`. One tag per *saver*, not per page, is the rule that
   already holds.
3. **Should a restore and a re-read look different on screen?** They have
   different consequences and currently read identically. A restore could say so
   — *"Restored your unsaved work"* versus *"Loaded wire.xyz"* — which is one
   sentence and removes a real ambiguity.
4. **BUG, independent of all of the above:** after a genuine restore the Load
   button re-enables against the file the structure came from, because
   `_loadedFrom` was never set. Pressing it discards restored work through the
   dirty-canvas gate. Fixing row 6 fixes this; until then it is reachable.

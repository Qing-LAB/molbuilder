# MolView — wiring the tabs to the finished module

**Role:** plan
**Domain:** web
**Started:** 2026-08-02
**Companions:** [`molview.md`](?doc=web/molview.md) — the contract the tabs are brought to.
[`tabs.md`](?doc=web/tabs.md) — the pages this reaches. Retired when the last page is
wired and its § 8 table describes the finished state.

[`molview-rework-plan.md`](?doc=web/molview-rework-plan.md) built the module from its
contract and said, deliberately: *"Nothing outside this module is consulted,
accommodated or repaired. If this leaves something outside broken, it stays broken."*
It did. This is that bill.

---

## 1. The rule this plan runs under

**The contract and the ESM design are the standard. Every difference in a tab is
abandoned and rewritten to the contract — never accommodated, never met halfway.**
(Your decision, 2026-08-02.)

So there is no design step here and no negotiation. Where a tab calls a name the
module does not have, the answer is not "add the name" — it is § 9.3's table, which
already says what a caller may ask for and which one call answers each need. This
plan is a translation and a rewiring, and every entry in it is read off the document.

---

## 2. What is actually broken

**MolView publishes nothing to `window.molbuilder`** — § 4, and it is true of the
code. **Thirteen files outside the module still read `window.molbuilder.molview.data`
across 31 sites**, and nothing has written that name since the rebuild.

**`/molbuilder` does not mount a viewer at all.** `modify/selection-bootstrap.js:62`
tests the dead global *before* it calls `mount`:

```js
if (typeof mount !== "function" || !_mvdata()) {
    _renderFailure(host, "molview module missing");   // ← always
    return;                                            // ← never mounts
}
```

The other pages mount and then read null.

**And some of it fails silently, which is worse.** The modify tab's cell editor
(`modify/periodicity.js`) calls a `{value, isDefault}` accessor family. **Three of the
four names are on no MolView surface at all** — `getVacuumInfo`, `getAxisKindInfo`,
`getUnitCellOriginInfo` — and the fourth, `getUnitCellInfo()`, returns
`{cell, cell_origin, axis_kind, vacuum}`, so reading `.isDefault` off it gives
`undefined`.

Every call is written `w.getVacuumInfo ? w.getVacuumInfo() : fallback`, so the page
renders **"(default)"** on every row instead of failing. A feature-detection guard in
front of a name that was never exposed does not protect the caller; it hides that the
caller is wrong.

**The wrapper is a client-side re-statement of something the metadata already says.**
The cell is resolved **once, on the server** — `Structure.to_wire()`, *"the ONE
resolver, run once, here"* — and the block it sends carries both halves beside each
other: `cell` / `resolved_cell`, `cell_origin` / `resolved_cell_origin`, `vacuum` /
`resolved_vacuum`. MolView stores that block verbatim and hands the two halves back
through two reads: `getUnitCellInfo()` is resolved-first, `getUnitCell()` /
`getUnitCellOrigin()` / `getVacuum()` / `getAxisKind()` are the raw fields.

So the editor reads the same block through the same doors, and `commitPeriodicityOp`
changes it. **Derived-versus-explicit is a fact in the data; neither surface computes
it.** A UI that decides for itself which values are defaults is a second resolver, and
the reason there is one on the server is that two of them disagreed once already —
the panel said a structure had a cell while the drawing found none, and neither failed.

(`axis_kind` has no resolved twin: `null` is unset, and that is all the block says
about it. The floor case is the same shape — `resolved_vacuum` differs from `vacuum`
when the minimum-thickness floor raised it, and nothing displays that yet: task #36.)

---

## 3. The host seam was already written

The rework plan's **F½** — *pin the seam before anything is built to fit it* — was run
for the UI edge and never for the host edge. It does not need running now: **§ 9.3's
needs table _is_ the host seam**, written out call by call with parameters and
answers. The work is to check all 31 sites against it.

### What gets rewritten at the call site

**No adapter, no shim, no wrapper, and nothing that could be implemented as one.** The
names in the left column below are not translated — they **cease to exist**, and the
call site is written against the contract's surface. A caller passes the format the
door takes; that is the caller's job, and it is the whole of what "the contract is the
standard" means here. The column exists so nobody has to rediscover which door answers
a need, not so anything can be mapped onto anything.

| Ceases to exist | Called from | What the call site is written as |
|---|---|---|
| `factsForRequest()` | `structure-optimization/viewer.js` | **`getStructure()`.** § 9.3 retires the second door: the master copy carries the elements, the per-atom facts, the cell block, every frame and its forces, so *"one read holds everything a request needs"*. The guarantee is the shape of the read, not the caller's discipline |
| `isDirty()` `markDirty()` `markSaved()` | `structure/page.js`, `projects/parser.js`, `structure/save.js` | **`uncommitted`** — whether there is unsaved work (§ 9.3). It is not marked from outside: the badge is raised inside the gate, after a change lands (§ 11.2) |
| `getLastSavedTo()` | `structure/page.js` | **Deleted — the page keeps it.** A project save is an *export of the truth* (§ 11.3); where the bytes landed is the host's fact, and the host is what chose the destination |
| `getSource()` | `structure/page.js`, `structure-optimization/viewer.js` | **`exportFile(range)`** returns `{name, structure}` — *"plus the name it came in under"* (§ 9.3). The name travels with the export; there is no standalone read, and "one need, one main way in" rules out adding one |
| `isEmpty()` | `transport/core.js`, `projects/parser.js`, `structure/page.js` | **`getStructure() === null`.** With nothing loaded a read answers nothing, *"which is a different answer from a structure with no atoms"* (§ 9.3) |
| `getVacuumInfo()` `getAxisKindInfo()` `getUnitCellOriginInfo()` | `modify/periodicity.js` | **`getVacuum()` · `getAxisKind()` · `getUnitCellOrigin()`** — what the structure states, `null` where it states nothing; `getUnitCellInfo()` for the value as it will be used |
| `getUnitCellInfo().isDefault` | `modify/periodicity.js` | **nothing — the block already says it.** The server sends `resolved_*` beside the raw values; a value is derived when the raw field is `null` and the resolved one is not. No flag is invented on either side |
| the `{ok, error, notices}` envelope `commitOp` expects back | `modify/periodicity.js` | **three outcomes, per § 6.9**: the cell block when it worked · a **throw** carrying the server's sentence when it was refused · `null` when there was nothing to do. So the handler is a `try`/`catch`, and the catch is where the tab says what stopped it. Notices are not in the answer at all — they are delivered inside the door and read with `getNotices()` (§ 6.8), so pushing them to the notification bar as well would put one fact in two places |
| `setFrame(i)` | `trajectory/core.js` | **`setCurrentFrame(i)`** — resolved against the range, never taken on trust |
| `setViewFlag(...)` | `trajectory/core.js` | **`view.set(...)`** (§ 9.6) |
| `selection.getState()` | `selection-bootstrap.js`, `modify/viewer.js` | ~~already matches — § 8.4's one whole snapshot~~ **Wrong, and it cost two of § 6.6's seven defects.** The call matches; **two of the keys read off it never existed**. `.indices` → `selection.get()`, and `.sourceFile` → the page's own note, because the viewer tracks contents, not files (§ 6.7). "Already matches" was written by looking at the call and not at what was read from the result |

Nothing on the left is added back, and nothing stands between the two columns. A file
that still has to ask "what did this used to be called" has not been rewritten yet.

---

## 3b. What changed underneath, on 2026-08-02

Saving was rebuilt while this plan was being written, and every page here touches it.
None of it is optional reading: three of the five pages ask the workspace a question
at mount, and all five hand a workspace to `mount`.

**One place your work is kept.** Files on the server, under the project directory —
`<projects_root>/.molbuilder_workspace/states/<workspace_id>.<step>.wc.json`. There
used to be a second copy in the browser's own storage; nothing ever restored from it,
so it cost a write on every edit and bought nothing. It is gone, and `snapshot-io.js`
with it.

**Every save and load names a tag** — who the bytes belong to. It is an argument on
each call, not something set beforehand. **The tag is the viewer's `owner`**, so a
page passes the same string to `mount` and to any workspace call it makes; declare it
once per file and use it for both, or the two drift into naming different slots.

**The workspace no longer answers questions about what is in your work.**
`hasRestorableSnapshot` and `mountRestoreTarget` are gone. They opened the saved bytes
looking for a molecule, and one of them got it wrong — a molecule built from SMILES
was never in a file, so it answered "nothing saved" and the tab wiped it. A page that
wants to know reads its own bytes:

```js
const saved = ws.readState({ workspace_id: ws.workspaceId(TAG), state_index: 0 });
const haveWork = !!(saved && saved.state && saved.state.structure
                    && (saved.state.structure.elements || []).length);
```

Two things about that snippet, both of which bit somebody already:

- **atoms, not text.** MolView holds no coordinate document (§ 11.7), so a page
  looking for `structure.text` finds nothing in a perfectly good save, every time.
- **`readState` is async**, where the old question was not. The three mount-time
  checks are all already inside `async` functions, so this is an `await` and not a
  restructure — but it *is* a change in shape.

**A page's own file name is its own business.** A saved state carries `source.file`
— the file it came out of, or `null` for a molecule that was never in one — and the
inspector compares that against the file clicked in the sidebar. Do not compare
against MolView's export name: that is a *stem* (`mine`), deliberately stripped of
folder and extension, and it never matches a path.

**An edit is saved; only Save state adds a point.** Nothing a page does needs to
arrange that, but it changes what "is there work here?" means: there may be work in
the draft that is on no point, and that is exactly the case a mount-time check exists
to protect.

---

## 4. How a tab reaches its viewer

**One owner per page. It mounts, and it hands the handle down.**

§ 5.6: *"there is no global to look one up in — so a tab cannot reach the wrong viewer
by accident."* That is not tidiness on `/results`, where **two viewers mount on one
page**: a page-level handle global would be a live bug there, not a style violation.

Every consumer today is a self-starting IIFE that looks the viewer up on
`DOMContentLoaded`. Each becomes something the owner starts:

- **module controllers** — `export function init(viewer, deps)`; the owner imports and calls it.
- **classic controllers** — keep the IIFE, gain an explicit bind door called by the
  owner. `modify/structure/page.js` already has `_bind`, used today only by tests;
  production stops looking up a global and starts being told.

Nothing publishes a handle anywhere, on either side. `lib/molview/demo.js` is the
reference shape — it mounts and uses what it was given, and it is the one consumer
that works today.

**Every call to a door that changes the structure is wrapped.** § 6.9 gives those
doors three outcomes — the thing, a **throw** carrying the server's own sentence, or
`null` meaning there was nothing to do — and the throw is where the tab tells the
user what stopped it. A page that skips the `catch` gets an unhandled rejection and a
control that goes quiet, which is the same silence this whole plan exists to end, one
level up. So each page's rewrite is not done until its door calls have somewhere for
that sentence to land.

---

## 5. The order, and where it stands

Whole pages, easiest first, each independently verifiable. **A page is done when it
reads no global, calls only § 9.3's surface, and has been looked at in a browser.**

| | Page | State |
|---|---|---|
| **a** | `/transport-calculation` | walked 2026-08-03 (mounts, no console errors); asked for `mode: "modify"`, a mode MolView has never had — fixed. Needs a walk that actually generates |
| **b** | `/spectrum-calculation` | walked 2026-08-03 — **its whole Generate side was dead** (`useViewer` exported from a scope it was not in, so the module never published); fixed. Needs a second walk with a real spectra run |
| **c** | `/structure-optimization` | ✅ **walked in a browser 2026-08-03.** Restores the structure it was showing (the TAB saves it, under its own tag); Generate + preflight fixed — all three doors were answering `400 no xyz provided` |
| **d** | `/results` | wired — the structure inspector and `lib/trajectory/core.js` both (§ 5a); needs a browser, and the run's lattice needs a decision |
| **e** | `/molbuilder` | ✅ **done** — all six steps of § 6.5 walked in a browser, five defects found and fixed (§ 6.6); 192 of its own tests pass |

**What "wired" was worth, measured on the one page that has now been walked:** `e`
was called wired on the same evidence `a`–`c` are, and a browser found five live
defects in it — one of which had frozen every control on the page. Read
`a`–`c`'s state as *the file-level rewiring is done*, not as *the page works*.

### What each page turned out to be

**a — transport.** It already held its handle in `_mvHandle` and never read it: every
read went to the dead global, so **Generate shipped no labels at all**. It also
loaded a file *before* mounting the viewer to load it into, which only worked while
the load door could find a viewer by name.

**b — spectrum.** Checked for the dead global *before* calling `mount`, so the page
**never mounted a viewer**. Generate asked the viewer for an XYZ document, got an
empty string, and the server answered "no structure provided" with a molecule on
screen — so the route now takes the structure as data and the text path is gone
(with the 15 tests that carried it; two had no subject left, since the route parses
no documents at all now). `_equilibriumGeometry` was fetching that same text and
parsing it back into atoms to reach the atoms the viewer already held.
`spectra/viewer.js` mounts and hands the handle to `lib/spectra/core.js`.

**c — structure-optimization.** `factsForRequest` → one `getStructure`; `getSource` →
gone with file tracking; `isDirty` → `uncommitted`; four sends stopped posting an XYZ
document. Its mount was fire-and-forget *after* the load; it is awaited before.

**d — results.** `lib/inspectors/structure.js` is wired and keeps **its own note** of
which file it is showing, under its own tag — the viewer tracks contents, not files
(§ 6.7). `lib/trajectory/core.js` is the remaining work; see § 5a.

**e — molbuilder.** `selection-bootstrap.js` is the page's owner: it mounts the one
viewer and starts the rest. `viewer.js` and `periodicity.js` became
`export function init(viewer)` and lost their `<script>` tags — they used to load
*before* the file that mounts, so even a global would have been empty when they woke.
The classic `structure/page.js` and `structure/save.js` are handed the viewer through
a `useViewer` door. The Cell panel's `{value, isDefault}` family is replaced by the
two real reads, and its Update button now catches a refusal and says what stopped it.
`init` now hands its restore back, so the owner can wait for it before deciding
whether the sidebar's file may be seeded. See § 6.5 for the browser walk and § 6.6
for what it cost.

### Cross-cutting, and the one that was breaking every tab

`lib/projects/parser.js` — the one door that reads a file into a viewer — looked the
viewer up by name. **So every file load on every tab failed**, with
*"projects.parser.openMolecule: molview.data.installMolecule unavailable"*. It takes
the viewer now: `openMolecule(viewer, path)`, `saveMolecule(viewer, path)`. Found by
running the real page, not by any test.

## 5a. `lib/trajectory/core.js` — wired 2026-08-03, one decision left

It is not like the other three read-only tabs. It watches a **running**
calculation and feeds frames in as they arrive, so it is the data source for
what the viewer shows.

**It never mounted a viewer at all.** `_mvdata()` read the dead global, and the
mount guard tested it *before* mounting — so the guard's answer was always "no
viewer", and none was ever built. Every later call went to null too, each inside
a try/catch or behind a `typeof` guard. It reads through the handle its own
mount returns now.

| It called | What landed |
|---|---|
| `setFrame(i)` | **`setCurrentFrame(i)`.** Both seeks called the missing name, so the playhead never moved — not after a rebuild, not to follow a growing run |
| `selection.setViewFlag("forceScale", …)` | **`selection.setSwitch(…)`** — `forceScale` is one of the switches beside the selection (§ 9.5). The original throw took the frame load with it (*"a trajectory showing ONE frame with no frame bar"*); it was then wrapped in a `typeof` guard rather than fixed, which stopped the crash and left the knob doing nothing |
| `installMolecule({atomMetadata})` | **carried now.** The server has taken `atom_metadata` at `/api/build/load` all along — a trusted block applied through `apply_to_structure` with no file envelope to satisfy. MolView's request builder simply never forwarded it, so a run's region labels and frozen tags were dropped **in the browser**, at HTTP 200 |
| `installMolecule(frame 0)` then `reloadFrames(the rest)` | **one call.** The contract names this shape as broken (§ 9.3): the one entrance stops being one, a subscriber sees a single-frame structure that never existed (§ 6.4), and point 0 is anchored on that one frame — so **a Retract threw the trajectory away** (§ 11.2). Found while fixing the row above |

### What a read-only tab cannot get from MolView

Three of these four pages mount `mode: "readonly"`, and **a read-only viewer
holds no session** — § 9.4: "a history exists to get back to a state you left,
and in a read-only viewer nothing can leave one", so `save`, `load` and `undo`
are no-ops and point 0 is never written. Driven directly against the model: **0
writes to the workspace, `load(0)` → `null`.**

So "the tab still shows what it had when you come back" cannot be delivered by
MolView on these pages, and two generations of code have tried anyway —
`readPersistedSnapshot` (a deleted door, behind a `typeof` guard) and then
`data.load(0)` (copied from the **editable** Modify tab, where it is the whole
restore). Both looked like they worked. Neither did.

If that promise is to be kept, it belongs to the **tab**, under the tab's own
tag, exactly like the panel note on `/molbuilder`. That is a decision, not a
fix: **task #51**.

### The run's lattice — decided 2026-08-03, then corrected on review

`installMolecule({periodicity})` was never forwarded either, and unlike
`atomMetadata` there was **no field on the route to apply it with**.

**The decision the user made:** *show the box, and mark it as the run's.*

**The first implementation of that decision was wrong, and a fresh-eyes review
caught it.** It folded the cell into the metadata block beside the labels —
which works, because `apply_to_structure` applies `cell` along with the
atom-scoped fields — and it cost three things:

1. **The cell went around the periodicity gate.** Every other structure door
   runs `apply_periodicity_from_body`, so a left-handed cell comes back 400 with
   the gate's sentence. Through the metadata block it was **accepted at 200**,
   on the one door a viewer actually loads through.
2. **The browser opened a document it does not own** — parsing the block,
   inserting a key, re-serialising. That block is a format the server writes.
3. **Worst: it re-stated `n_atoms_total`**, using the count of the geometry
   being loaded. That field is the guard that stops a label set written for one
   structure landing on another, and a count copied from the target makes the
   comparison `n == n` — **the guard could never fire again**.

**What landed instead.** Two facts from two places travel as two named fields,
and `/api/build/load` gained the one line every other door already has:

```
labels  (run's INPUT script)  → atom_metadata → apply_to_structure     (its own guard intact)
cell    (run's OUTPUT logs)   → periodicity   → apply_periodicity_from_body  (the gate)
```

The block is passed through **untouched**, as bytes. And because the gate now
runs on the structure rather than on the field it arrived in, the smuggling
route is closed too: a refusable cell is refused whichever way it reaches the
door — the same "the check lives in the return path" property § 6.8 gives the
modify ops.

**No `pbc` and no `axis_kind` are sent**, and the result is better than the
periodic-on-all-axes the first version produced by accident. A run reports a
box, not a statement about which axes repeat, and `axis_kind` is the one field
MolView will not default (§ 9.5). The axes stay `isolated`; **the box is still
drawn**, because the drawing uses the cell *as it will be used*
(`resolved_cell`) and a stated cell resolves to itself. So an isolated molecule
that ran in a large SIESTA box shows that box without anything claiming it
repeats.

The tab's status line still says *"Unit cell taken from the run output, not set
by you"* — the **tab** says it, because the Cell page deliberately answers *"is
this box mine?"* and not *"where did it come from"* (§ 9.5), and the tab is what
performed the load.

(Task **#35**, "Results-tab trajectory first load animates a single frozen
frame", was the symptom the `setViewFlag` comment describes and should be
re-checked against the two fixes above.)

---

## 6. Molbuilder, from the contract down

The other four tabs read a viewer. **This one edits.** It is the only page where a
user changes a structure, so it is the only page that exercises the write half of
§ 9.3 and the whole of § 11.2's timeline. It is also seven files. Both reasons to
work it from the contract rather than file by file.

### 6.1 What the contract says a host does

Four things, and nothing else:

| | |
|---|---|
| § 8 | **`mount(host, workspace, {owner, mode, …})`** — one call builds the whole card. It always resolves: on failure `ok` is false, `error` says why, `dispose` still works. |
| § 5.6 | **The handle is the way in.** A viewer belongs to whoever mounted it; there is no registry, so a page cannot reach the wrong one and cannot look one up. |
| § 9.2 | **The handle is lifecycle, playback, and one route to the model** — `viewer.data`. It does not mirror the model: a read the model answers is never duplicated onto the handle. |
| § 9.3 | **The model is sixteen needs**, one main way in each. Every read returns a copy; `selection` and `view` are doors, not values. |

Everything a host wants is one of those. If a page needs something that is not,
that is a contract question — not a place to reach past the seal.

### 6.2 What the tab needs, and the contract's answer

The complete list, from reading all seven files. Nothing here is a shape the tab
invents; every row is a name § 9.3 prints.

| The tab needs to | § 9.3's answer | Where |
|---|---|---|
| put a structure in | `installMolecule(input)` | `structure/page.js` (every generator and the file loader go through it) |
| edit the geometry | `applyOp(name, args)` | `viewer.js` |
| edit the cell | `commitPeriodicityOp(op, payload)` | `periodicity.js` |
| show the cell as it will be used | `getUnitCellInfo()` | `periodicity.js` |
| show what the structure itself states | `getUnitCell` · `getUnitCellOrigin` · `getAxisKind` · `getVacuum` | `periodicity.js` |
| know what is loaded | `getStructure()` — `null` when nothing is | `viewer.js`, `page.js`, `save.js` |
| know if there is unsaved work | `uncommitted` | `page.js`, `save.js`, `viewer.js` |
| know where you are on the sequence | `state_index` | `viewer.js` |
| save a point / step back / restore | `save(1)` · `load(-1)` · `load(0)` | `viewer.js` |
| write the structure out | `exportFile(range)` | through `projects.parser.saveMolecule` |
| reach the selection | `selection` | `viewer.js`, `selection-bootstrap.js` |
| hear that it changed | `subscribe(fn)` | all four |

**Two things the tab wanted that are not on that list**, and both are answered
without widening it:

- *"which file is this?"* — nobody's but the page's. The viewer tracks contents,
  not files (§ 6.7).
- *"the atom number the user sees"* — § 11.5 keeps that translation inside the
  module, and § 4 exports two names. The anchor readout adds one rather than
  importing a third name.

### 6.3 What that makes each file

`selection-bootstrap.js` is the **owner**: it mounts the one viewer and starts
everything else with it. `viewer.js` and `periodicity.js` are
`export function init(viewer)` and have no `<script>` tags — they used to load
*before* the file that mounts, so nothing could have handed them a viewer even if
one had existed. The classic `structure/page.js` and `structure/save.js` take it
through a `useViewer` door, because they load before any module runs and cannot
import. The five generators and the file loader touch no viewer at all: they go
through `page.js`, which owns the load gate.

### 6.4 The carve-out — every old use, and what it became

Nothing on the left exists anywhere in the tab any more. This is the list, so a
later reader can tell a deliberate removal from an oversight.

| Old use | Why it had to go | What replaced it |
|---|---|---|
| `window.molbuilder.molview.data`, in 5 files | MolView publishes nothing there (§ 4); every read was `undefined` | the handle `mount` returned, passed down by the owner |
| `viewer.js` / `periodicity.js` waking on `DOMContentLoaded` | they loaded *before* the file that mounts, so no viewer could exist yet | `export function init(viewer)`, started by the owner; both `<script>` tags gone |
| the guard `if (!mount \|\| !_mvdata())` before mounting | tested for a viewer before making one — **it is why the tab never mounted at all** | test the import only |
| `useNamespace("modify")` at page load | one shared setting two files raced over | the tag goes in every workspace call (workspace.md § 4) |
| `<body data-workspace-owner="modify">` | covered the window before that setting was made | no window to cover |
| `isEmpty()` | not on the surface | `getStructure() === null` — nothing loaded reads as nothing (§ 9.3) |
| `isDirty()` | not on the surface | `uncommitted` — a value, not a question |
| `markDirty()` | the edit already raised the badge, inside the gate (§ 11.2) | nothing; the call is a no-op kept only while the panels are rewired |
| `markSaved(path)` | set a viewer flag from outside, and told it a file path | nothing on the viewer; the page keeps its own note |
| `getSource()` · `getLastSavedTo()` | the viewer tracks contents, not files (§ 6.7) | the page's own note; both were `undefined` |
| `getVacuumInfo()` · `getAxisKindInfo()` · `getUnitCellOriginInfo()` · `.isDefault` | **never existed**; each call was feature-guarded, so the Cell page showed "(default)" on every row | `getUnitCellInfo()` for the value as used, plus the raw read; a default is the raw read being `null` |
| `commitOp` expecting `{ok, error, notices}` back | the door answers the cell block, or throws, or `null` (§ 6.9) | `try`/`catch`; the catch shows the server's own sentence |
| `import { toDisplay }` | § 4 exports two names — **a hard link error that killed the whole page** | the readout adds one; the translation stays in the module (§ 11.5) |
| `parser.openMolecule(path)` · `saveMolecule(path)` | the door looked its viewer up by name — **this broke file loading on every tab** | `openMolecule(viewer, path)`, `saveMolecule(viewer, path)` |
| load-then-mount | loaded a file into a viewer that did not exist yet | mount first; a viewer mounts before it has a structure (§ 8) |

### 6.5 How it is verified — in the browser, in this order

Each step is a thing a user does, and each has a way to be wrong that a test
cannot see.

1. **The page mounts.** ✅ — card, rail, panel, Modify controls all render.
2. **Load a structure from the sidebar.** ✅ — `RAW_BDT.xyz`, 14 atoms, drawn, and
   the `BDT` label from the sidecar on every row.
3. **The Cell tab shows real numbers.** ✅ — the tab's editor and MolView's own
   panel agree, both resolved-first, and `(default)` marks the raw read being
   `null` rather than every row.
4. **Edit an atom.** ✅ — Delete took 14 → 13, the drawing and the formula
   followed, and the **Unsaved changes** badge appeared in the corner.
5. **Save state, then Retract.** ✅ — `✓ saved #1` on disk as
   `ws-modify.1.wc.json`, Retract came back to `#0` with the atom restored.
6. **Save to project.** ✅ — `WIRED_CHECK.xyz` **and** `WIRED_CHECK.molstruct.json`
   written together (schema 7, real hash, `BDT` region intact); a load → save
   round trip is byte-identical to the original but for the timestamp. The page
   now records the target too: `Target: WIRED_CHECK2.xyz` in the readout.

*(`WIRED_CHECK.*` and `WIRED_CHECK2.*` are this walk's artefacts, sitting in
`projects/BDT-Au/structure/`. Delete them when you like — they are kept only so
the round-trip comparison above can be re-run.)*

### 6.6 What steps 2–6 turned up, and what it cost

The page mounted (step 1) but **nothing in `viewer.js` had updated since the page
was built** — no op button, no anchor readout, no Save state, no Retract, no
timeline indicator. One line:

```js
return s ? s.getState().indices.slice() : [];   // `indices` is on no snapshot
```

The selection snapshot carries `selection`, never `indices` (§ 9.5), so this was
`undefined.slice()` — **a TypeError on the first line of every refresh**. Both
paths that call it swallow what a subscriber throws (`catch (_) {}`), so there
was nothing in the console and nothing on screen: the buttons simply sat at the
disabled state the template ships and stayed there. It reads as "the tab is
half-wired" and is one wrong key.

Six more, all found by walking steps 2–6 rather than by reading:

| What was wrong | What it looked like | What it is now |
|---|---|---|
| `getState().indices` | every control in `viewer.js` frozen at its template state | `selection.get()` — the door's own read (§ 9.5) |
| `readPersistedSnapshot(tag)` in the owner | a door the workspace no longer has, called behind a `typeof` guard, so "no saved work" every time — and the sidebar's file free to overwrite restored work | `await` the restore, then `getStructure()` — atoms are what "there is work here" means |
| `r.n_atoms` off `applyOp` | **"Deleted: undefined atoms."** after every edit | `r.elements.length` — the door answers the structure (§ 6.9), not a report about it |
| `r.issues` off `applyOp` | the advisory region fed `undefined`, so server findings reached nobody | `getNotices()` (§ 6.8) — this is § 9's open item #2 coming due |
| the status line | `No structure loaded.` beside a drawn molecule; and `Restored 0-atom structure (unnamed)` written even when nothing was restored | says what landed on a load; says nothing when `load(0)` answers `null` |
| `markSavedTo` called by nobody | after saving, the readout still said `Save as… into structure/` — the page never learned where its own bytes went | the save calls it on success; the readout reads `Target: <name>.xyz`, and `Unsaved — ` once you edit again |
| `getState().sourceFile` | the loader readout said `Picked:` even with that file on screen, and Load never went dead — so a no-op click looked identical to a real one | the page's own `_loadedFrom`, set where the load happens; `Picked:` → `Loaded:` and the button disables |

**The pattern in most of them: a name that no longer exists, read through a guard
that turns its absence into a plausible answer.** `typeof x === "function"` became
"nothing saved"; a swallowed subscriber became "nothing changed"; `undefined`
became "(default)" and "undefined atoms". § 6.4's rule against feature-guarding a
moved door is not tidiness — the guard is what keeps these off the console.

**The last two are the same failure with the guard on the other side: a carve-out
half-finished.** § 6.4 correctly took `markSaved` off the viewer and said *"the
page keeps its own note"* — and then the call was **deleted rather than
redirected**, so `markSavedTo` sat there with no caller and `targetPath()`
answered `null` forever. `sourceFile` is the same shape: the viewer rightly stopped
tracking files (§ 6.7), and nothing took over the tracking. **A carve-out row is
not finished until the right-hand column has a caller.**

Seven defects on the page the plan called wired, and **six of the seven are
invisible to a test**: they are a control that stays at its template state, a message that
describes nothing, a readout that never learns. Only the browser walk finds them,
which is why § 6.5 is a list of things a user does rather than a list of asserts.


---

## 7. Residue to delete

- The `_mvdata()` helper — all thirteen copies.
- `_molview_scripts.html`'s standalone `<script type="module" src=".../molview/index.js">`.
  `index.js` only exports now, so the tag does nothing; the comment above it still
  describes a global (`atomIndexModel`) the module has not published for weeks.
- `structure-optimization/viewer.js`'s header, which says *"No `window.molbuilder.molview`
  global reads — those are the transitional shims we are dumping"* five lines above the
  function that does exactly that.
- Every feature-detection guard in front of a door that has moved (§ 2). A guard
  belongs in front of something genuinely optional, and none of these is.

  **Not done, and now enumerated for `/molbuilder` rather than left as a
  category.** These seven guard doors that § 9.3 always provides, so each one is a
  future rename turned into a silent no-op — the exact shape that cost six defects
  on this page:

  | | |
  |---|---|
  | `viewer.js:101` | `typeof d.getElements === "function"` |
  | `viewer.js:298` | `typeof d.getStructure === "function"` |
  | `viewer.js:912` | `typeof _d.subscribe === "function"` |
  | `periodicity.js:40` | `typeof w.getStructure === "function"` |
  | `periodicity.js:306` | `typeof w.subscribe === "function"` |
  | `save.js:373` | `typeof _model.subscribe === "function"` |
  | `selection-bootstrap.js:289` | `typeof store.subscribe === "function"` |

  Held back deliberately: several of this tab's tests hand in partial stub models,
  so removing the guards is a change to what those tests must provide. It is
  hygiene with a test bill attached, not part of making the page work — and every
  one of them is in front of a door that is present today.

## 7. Documents to repoint

- [`science/validation.md`](?doc=science/validation.md) **F1** and its diagram — the
  obligation is unchanged; the door is `getStructure()`.
- [`architecture.md`](?doc=architecture.md)'s reuse row for sending a structure to a
  validating endpoint.
- `tests/test_validation_delivery_contract.py`, which asserts the string
  `factsForRequest` in the source.
- [`tabs.md`](?doc=web/tabs.md) **§ 8**, which describes the half-migrated state as the
  current one.
- Any **Transition** note in `molview.md` this closes — the rework plan's step H says
  it should end with none.

## 8. How each page is verified

Per the program's rhythm: **one page, then only that page's tests.** Never the suite.

| Page | Its tests |
|---|---|
| `/transport-calculation` | `test_transport_blueprint.py`, `test_transport_cell.py`, `test_transport_generate_e2e.py` |
| `/spectrum-calculation` | `tests/spectra/`, `test_spectrum_generate_e2e.py`, `test_spectra_phase_indicator_js.py` |
| `/structure-optimization` | `test_build_e2e.py`, `test_validation_delivery_contract.py` |
| `/results` | `test_inspector_registry_e2e.py`, `test_trajectory_*`, `test_structure_inspector_measurement_e2e.py` |
| `/molbuilder` | `test_structure_page_js.py`, `test_atom_selection.py`, `test_modify.py`, `test_molbuilder_e2e.py` |

**And a browser, per page.** The rework plan's own record is that a stubbed test stayed
green while every viewer mounted and never drew. The tests here are stand-ins for a
handle; only a browser shows a page that got one.

### 8a. The e2e suites tested the architecture that was removed

Running `/molbuilder`'s own e2e on 2026-08-02: **106 of 139 failed**, every one with
`Cannot read properties of undefined (reading 'data')`. They drove the page through
`window.molbuilder.molview.data` — 103 lines of it — and read keys off the snapshot
that have never existed (`.atoms`, alongside § 6.6's `.indices` and `.sourceFile`).

**A test may not reach past the seal, and this is the argument.** § 4 exports two
names; § 5.6 says a viewer belongs to whoever mounted it and there is no registry. A
test holding the model is asserting on something the page's own controls do not use —
so it passes while every control on the page is dead. That is not a hypothetical: it
is precisely what happened. Seven live defects, 139 green tests.

**Retired** (they drove the dead global as their harness, so there was nothing to
repoint):

| File | Tests |
|---|---|
| `test_molbuilder_e2e.py` | 139 |
| `test_structure_inspector_measurement_e2e.py` | 7 |
| `test_transport_generate_e2e.py` | 6 |
| `test_workspace_dispatcher_mount_e2e.py` | 1 (asserted `molview._canvasState`, a private that is gone) |

**Rewritten from the contract:** a new `test_molbuilder_e2e.py` — § 6.5's six steps,
DOM in and DOM out, nothing reaching past the seal. Each test names the defect it
would have caught. Three more files needed only their readiness gate repointed
(`test_build_e2e.py`, `test_spectrum_generate_e2e.py`,
`test_no_legacy_persistence_keys.py`), and four had stale `factsForRequest` prose.

**Coverage not yet rewritten, named so it is a known hole and not a silent one:** the
electrode/junction ops, the transform sub-tab, the by-residue and by-label filters,
the measurement readout, the DNA/RNA/peptide generators, the narrow-viewport layout,
and the transport + structure-inspector walks. None of it can come back by
un-deleting.

One guard was itself corrected in passing: `test_validation_delivery_contract.py`
pinned the string `factsForRequest` in the source, so it fired on the comment
explaining that door's removal. **A guard that punishes documentation trains people to
delete it** — it now strips comments and searches the code.

## 9. Open

1. **`lib/projects/parser.js` is not a page** — it is the projects package, reached from
   several tabs. **Settled by e:** it takes the viewer as its first argument, from the
   caller that mounted it — `openMolecule(viewer, path)`, `saveMolecule(viewer, path)`.
   A page can hold more than one viewer, so "the viewer" was never a question this file
   could answer.
2. **`getNotices()` is on the model and not in § 9.3's table** (§ 6.8 covers it).
   **CLOSED, the other way.** It does not belong in § 9.3 and no row was owed. § 6.8
   already assigns the DISPLAY to MolView — a cell notice under the Cell rows, everything
   else on one line above the tabs — so no host needs to read notices, and none does.
   `getNotices()` is the model's read, used by MolView's own panel, which is a sibling
   inside the module.

   Recorded because I got it wrong in both directions and the second was worse: I
   rewired the Modify tab to read it and draw the notices a second time, then added a
   § 9.3 row saying "a host that shows findings reads them here" to justify it. **That is
   editing the contract to match code rather than the other way round**, and it is the
   failure this whole plan is a bill for. Both are reverted: the tab draws no notices and
   § 9.3 is back to sixteen needs.

3. **Coming back to work without re-opening the file — YOUR DECISION.** § 11.2a
   already names this and calls it a decision rather than an oversight, and **e** is
   where it stops being hypothetical.

   What happens today: MolView writes a draft on every edit and a numbered point on
   every Save state, both correctly, under this tag (`ws-modify-draft.0.wc.json`,
   `ws-modify.0.wc.json`, `ws-modify.1.wc.json` — all confirmed on disk). Reopening
   the page **cannot read any of them**: `load(0)` refuses before there is an anchor,
   and only installing a molecule anchors one. So the canvas is filled by re-loading
   the file the sidebar still has highlighted, and looks restored because usually
   that is the same molecule.

   Two consequences, and the second is the one that bites:
   - A structure you **generated** — SMILES, RNA, peptide, name lookup — has no file
     to re-load. Leave the tab and it is gone, though its bytes are on the server.
   - Edits you made and did not save to a project file are gone the same way.

   The two readings, and they are genuinely different products:

   | | What reopening the page means | What it needs |
   |---|---|---|
   | **take the file again** | you come back to the file, not to your session; unsaved editing is deliberately not kept | nothing — this is today's behaviour. Then the draft write should **stop**, because writing bytes nothing can read is worse than not writing them |
   | **take back the work** | you come back to what was on screen, whatever produced it | a way for a fresh viewer to **adopt** a sequence that already exists — the one thing `load` cannot do, by § 11.2a's own words |

   Until you choose, the tab behaves as the first and pays for the second.

4. **A save drops per-atom residue names.** `structureForServer` sends
   `regions · cell · cell_origin · axis_kind · vacuum` and **not `annotations`**, so
   the server fills residues with its default (`MOL` for every atom). Labels survive
   — they travel as `regions` — which is why a load → save round trip of an
   `.xyz` is byte-identical and this stays hidden. Open a **PDB** with real residue
   names, save it, and they come back as `MOL`. Not verified in a browser yet, and it
   is inside MolView's export door rather than the tab, so it is recorded here rather
   than fixed in passing.

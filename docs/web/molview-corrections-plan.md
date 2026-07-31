# MolView — the corrections plan

**Role:** plan
**Domain:** web
**Started:** 2026-07-31
**Companions:** [`molview.md`](?doc=web/molview.md) — the contract every item below is
measured against. [`molview-rework-plan.md`](?doc=web/molview-rework-plan.md) — the
rebuild this follows; that plan builds the module, this one repairs what the
finished module got wrong. Retired when the last item lands.

---

## 1. What this is

The rebuilt module was reviewed against its contract, layer by layer, and against
the frozen tree it replaced. The review found defects the tests could not see —
several of them features that were **complete, correct and connected to nothing**.

This is the list, and the order it will be executed in. **One item is settled at a
time**: what is wrong, what the evidence is, what the previous implementation did,
and the agreed change to the code AND to the document. An item is not written here
until it has been discussed and its shape agreed; an item written here is ready to
execute without re-deciding anything.

**Why a plan and not a task list.** Every item below is a *design* correction with a
document half. Recording only "fix the sidecar" loses the reasoning that makes the
fix the right one, and the reasoning is the part that stops it being re-broken.

### How an item is written

| Field | What it holds |
|---|---|
| **Symptom** | what a user or a caller sees, in plain terms |
| **Evidence** | what was actually read or measured, with file and route names |
| **The old code** | what the frozen tree did, and whether it was right |
| **Code** | the change, in the order it lands |
| **Document** | the sections that change, because the contract is the thing being complied with |
| **Open** | what is deliberately left undecided, and who decides it |

---

## 2. Status

| # | Item | State |
|---|---|---|
| 1 | The save / load pair, and where the saved bytes come from | **agreed — ready to execute** |
| 2 | The cell's private spelling | **agreed — ready to execute** |
| 3 | The coordinate document is rewritten when the server already sent one | raised, not discussed |
| 4 | The browser writes a file format at all | raised, not discussed |
| 5 | The label list is flipped in three places | raised, not discussed |
| 6 | `applyOp` sends a body the route does not read | raised, not discussed |
| 7 | What a load drops: identity columns and annotation channels | raised, not discussed |
| 8 | The coordinates' in-memory shape | raised, not discussed |
| 9 | The movie is rebuilt from one giant string | raised, not discussed |

Items 2–9 are recorded here so nothing is lost between sessions. Each gets its
section below when it is settled, in the shape § 1 describes.

---

## 3. Item 1 — the save / load pair, and where the saved bytes come from

**AGREED. Ready to execute.**

### Symptom

Export → Save to project writes `wire.xyz` and `wire.molstruct.json`. The second
file is not a `.molstruct.json`. Reopen the pair and the sidecar is refused, taking
every label with it — the frozen set, the electrodes, all of it — with no error
anywhere. The `.xyz` looks fine, so the structure loads and is quietly wrong.

### Evidence

A sidecar the codec writes carries **fourteen** keys:

```json
{"schema_version": 6, "n_atoms_total": 2, "structure_hash": "<sha256>",
 "regions": {…}, "frozen_atoms": [1],
 "cell": null, "cell_origin": null, "pbc": [false,false,false],
 "axis_kind": ["isolated","isolated","isolated"], "vacuum": [0.0,0.0,0.0],
 "annotations": {}, "selection_rules": {},
 "created_by": "molbuilder", "created_at": "2026-07-31T17:30:07Z"}
```

MolView produces **seven** of them. `schema_version` is the first thing the reader
checks, and it is one of the seven that are missing.

**Python already has the normalized pair**, and they are exact inverses — the
blueprint's own docstring calls the save "the symmetric inverse of the file-only
load":

| | route | what Python does |
|---|---|---|
| load | `POST /api/build/load {path}` | `StructureCodec.read` — reads the `.xyz` **and** its paired sidecar, applies the metadata |
| save | `POST /api/structure/save {path, blob}` | `from_scratch` + `write` — writes **both**, stamping the version and a real content hash |

**The projects sidebar already has the matching front door**: `openMolecule(path)`
and `saveMolecule(path, {overwrite})`, and it calls *into* MolView. That direction
is correct — the sidebar decides *which file*, MolView decides *what a structure
is*.

**Three generators inside Python, not one.** `files()`, `write()` and
`scratch_blob()` each produce the geometry-plus-sidecar pair. `files()` and
`scratch_blob()` share one helper; `write()` recomputes it inline. They agree today
by coincidence, not by construction — so save and download would be two generators
before MolView is involved at all. `files()`, which returns both files as bytes and
is exactly what a download needs, **has no callers**.

### The old code

It never authored the sidecar. `saveMolecule` read the structure out of the model
and POSTed it to the save route; the server wrote both files. The frozen tree left
a note saying why: *"a browser-authored sidecar had no schema_version, so the
file-only load door rejected the pair on the next open"*. The fix then was to stop
authoring it — not to add the missing key.

The rebuild lost that division by inventing a door, `files.save(destination,
filename, contents)`, which asks for a **filename and bytes**. That shape is what
forces MolView to author a format it cannot author.

### Code

**Python first, so the generator is in place before anything depends on it.**

1. **One generator, three consumers.** `write()` stops recomputing and writes what
   `files()` produces. `files()` carries the existing rule that a structure with no
   metadata worth keeping gets *no* sidecar, and a stale one is deleted — so that
   behaviour stays in one place rather than being duplicated into a second.
2. **`POST /api/structure/export {blob}` → `{ok, xyz, sidecar}`.** The same pair
   `write()` would put on disk, returned instead of written. This is what makes
   "save and download produce identical bytes" true by construction.

**Then MolView.**

3. `exportFile()` returns the structure as data and stops. It is a read.
4. The `files` door is removed from `mount`.
5. Save to project → the sidebar's `saveMolecule` → the save route.
6. Download → the host asks the export route and puts the two files on the user's
   disk. MolView still decides *what* leaves and *where it goes* (§ 11.4); it stops
   assembling bytes it gets wrong.

**`exportFile()` stays synchronous.** It reads the module's own memory and stops;
the round trip belongs to whoever is putting bytes somewhere — the sidebar for a
project save, the host for a download. Making it async would have bought a new
"the server was unreachable" failure at the moment a user expects a file, and a
gap between sending and answering in which the structure could change underneath
(the race § 11.2 built the CHANGING state for). Neither is worth taking on, and
neither is necessary.

**One consequence to expect:** the demo page's stand-in `files` door is replaced by
a small host-side save.

### Document

7. **§ 11.3** — "MolView produces two files" becomes: MolView produces the
   structure as data, and the files are written from it by the codec. "Both
   destinations produce identical bytes" stops being a promise and becomes a
   consequence of one generator.
8. **§ 8** — the `files` door leaves the mount options.
9. **§ 11.7** — gains the download half: one generator, two destinations. *(The
   section itself already landed, commit `5ff1d79`.)*

### Open

Nothing. The one question — how Download is served — was settled: the backend grows
export support so it shares the generator, rather than the browser assembling bytes
on a second path.

---

## 4. Item 2 — the cell's private spelling

**AGREED. Ready to execute.**

### Symptom

The unit-cell button drew nothing, for any structure, ever. The axes always fell
back to the Cartesian x/y/z triad instead of following the cell vectors. An
exported sidecar carried no cell. Nothing failed and nothing was logged, because
every one of those is the correct behaviour for a structure that is not periodic —
which is what the module believed about all of them.

### Evidence

The server sends the block as `{cell, cell_origin, axis_kind, vacuum}`. The module
renamed it on the way in to `{lattice, origin, axis_kind, vacuum}` and back on the
way out. The reading code then asked for `.lattice` — a key that has never existed
on the wire — and got nothing.

**`{cell, cell_origin, axis_kind, vacuum}` is the format everything else already
speaks**: the load payload, the `.molstruct.json` on disk, the modify routes'
`periodicity` block, and the frozen tree's `_setPeriodicity`, `factsForRequest` and
`scratchBlob`. The rebuilt module is the only thing in the system that ever called
it something else.

The name `lattice` does appear elsewhere in the front end, and neither instance is
this block: `mol-viewer-embed.js`'s `cellBox.lattice` is **the drawing's parameter
for a box to render**, and `trajectory/core.js`'s `state.data.lattice` is the
trajectory module's own parsed run-file data.

### The old code

It never renamed. The server's four names went in and came out unchanged, and every
reader used them.

### Code

1. The structure carries `periodicity: {cell, cell_origin, axis_kind, vacuum}` —
   the backend's names, unchanged. `cellFromServer` and `cellForServer` are deleted:
   with nothing to translate there is nothing to keep in step.
2. `getUnitCell()` returns `periodicity.cell`, `getUnitCellOrigin()` returns
   `periodicity.cell_origin`, and so on — the narrower cuts of § 9.3's table keep
   their names, because those are MolView's read surface and are unaffected.
3. **`lattice` survives in exactly one place, legitimately**: the box handed to the
   drawing. `sceneFor()` maps `periodicity.cell` onto the `{lattice, origin}` shape
   the drawing takes — a translation into the drawing's vocabulary, at the layer
   that already does that for `style` → `rep` and `radius` → `radiusScale` (§ 9.8).

### Document

4. **§ 6.2** — the structure's field is `periodicity`, and it carries the server's
   four names. The row already says MolView "carries the block and interprets none
   of it"; renaming is interpretation's first step, and this makes the row literally
   true.
5. **§ 11.1** — the inbound translation's job shrinks: the server's *atom* names
   still become the module's, and the cell block is now carried verbatim.

### Open

Nothing.

---

## 5. Items not yet settled

Recorded so they survive the session. Each becomes a section above when discussed.

**3. The coordinate document is rewritten when the server already sent one.** A load
returns the canonical text; the module ignores it and writes its own — no title
line, raw float formatting — so exporting an unedited file gives different bytes
from the file on disk. The old code carried the server's text and rebuilt it only
for a scrubbed trajectory frame, with matching six-decimal formatting.

**4. The browser writes a file format at all.** Every structure door takes geometry
as `.xyz` text, so a viewer holding numbers must write a document to ask any
question about them. A door accepting `elements` + `positions` would remove the
writer from the browser entirely. Backend change; recorded as Open in § 11.7.

**5. The label list is flipped in three places** — once for callers, twice for
payloads, with one deliberate difference (whether the frozen set is split out).

**6. `applyOp` sends a body the route does not read.** It sends
`{structure, selection, params}`; `/api/modify/*` reads `{xyz, …metadata…}` plus a
per-op field, so **every geometry edit is refused with a 400 today**. The per-op
field names exist in the frozen tree's registry and are absent from § 11.1's table.

**7. What a load drops.** The server returns atom names, residue ids, chain ids and
the annotation channels; the module keeps none. Invisible until an edit — a
structure that round-trips comes home flattened.

**8. The coordinates' in-memory shape.** Held as nested three-element arrays: one
object per atom per frame, 800k objects for a 400-frame, 2000-atom run. A flat
typed array indexed `[frame][atom][xyz]` is the alternative, and it is a change to
what § 6.2 says the coordinates are.

**9. The movie is rebuilt from one giant string.** Every rebuild concatenates the
whole trajectory into one XYZ document for the drawing library. The library also
accepts frames as atom objects — the append path already uses that — so a movie
could be built from numbers. Needs measuring before claiming, in the one file the
rebuild carried rather than rewrote.

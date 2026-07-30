# MolView — the 3D structure viewer

**Role:** contract
**Domain:** web
**Companions:** [`overview.md`](?doc=web/overview.md) (the web start-here map);
[`workspace.md`](?doc=web/workspace.md) (where a saved session's bytes are kept);
[`projects.md`](?doc=web/projects.md) (the file browser that hands structures to
MolView); [`web-api.md`](?doc=web/web-api.md) (the server routes MolView calls);
[`model/structure.md`](?doc=model/structure.md) +
[`model/structure-annotations.md`](?doc=model/structure-annotations.md) (the
structure and the region/frozen tags MolView carries);
[`model/structure-molstruct.md`](?doc=model/structure-molstruct.md) (the
`.molstruct.json` sidecar it reads and writes). How a user *builds* a structure —
the Modify tab's source panels and its save dialog — belongs to
[`tabs.md`](?doc=web/tabs.md), not here.

MolView is the one 3D molecular viewer used everywhere in the browser. The Modify
tab edits a structure in it; the Results, Spectra and Transport tabs show one in
it without letting you edit; the trajectory inspector plays an optimization in
it. Every tab embeds the same component with the same controls.

**This document is the design of that component** — what it is for, what it
refuses to do, what it holds, how it is layered, what each API is for, how the
pieces fit together, and how its tests are derived. It is not a tour of the
current code.

> **Words used in this document.** A few terms come up constantly. Each is
> defined here in plain terms and used consistently below.
>
> - **A viewer** — one mounted MolView: one 3D window, its panel, its controls,
>   and everything it holds. Two viewers on a page are two of these.
> - **The handle** — the object you get back when you make a viewer. Think of it
>   as the viewer itself; it is the only thing a tab holds.
> - **The model** — the part of a viewer that holds the structure and answers
>   questions about it. Everything that changes the structure goes through it.
> - **The master copy** and **the drawing copy** — the structure exists in two
>   forms, on purpose: the master copy is the real one (every atom, every frame),
>   the drawing copy is what the 3D library was handed to paint. § 6.3.
> - **A frame** — one set of atom positions. A trajectory is many frames of the
>   same molecule; a plain structure is one frame.
> - **Isolate** — the "show selected atoms only" switch.
> - **The renderEngine** — the part that decides what to redraw and works out
>   what each frame should look like. *Not* a calculation engine: SIESTA and
>   PySCF are engines in a completely different sense, and this document never
>   means that one.
> - **The sealed layer** — the only code in the whole app allowed to talk to
>   3Dmol.js, the third-party library that does the actual drawing. Nothing above
>   it knows that library exists.
>
> Atom numbering: **0-based in code, 1-based on screen**, translated in exactly
> one place (§ 11.3).

---

## 1. The goal

**One 3D molecular viewer, learned once, used everywhere.**

Every place in the app that shows a molecule embeds the same component with the
same controls. A user who learns to select, isolate, measure and scrub in the
Modify tab can do all of it in the Results tab without learning anything new. A
developer who needs a viewer does not build one.

### 1.1 What that looks like in use

This is the behaviour the rest of the document exists to protect. It is the same
in every tab.

**Moving around.** Drag to rotate, scroll to zoom, right-drag (or shift-drag) to
pan. `⟲` re-centres and re-fits the camera. The View menu offers **Perspective**
(natural depth, the default) or **Orthographic** — pick orthographic when you are
eyeballing bond lengths, because it removes the foreshortening that makes distant
atoms look closer together.

**Appearance.** The View menu holds style (stick, ball & stick, sphere, line), a
radius slider from 0.2 to 2.5 that scales stick thickness / sphere size / line
width, and a background colour with preset swatches plus a picker. One preset is
transparent — choose it before exporting a snapshot to drop onto a slide.

**The toolbar switches.** Six icon buttons sit down the left edge, always
outside the canvas, never on top of the molecule:

| Button | What it does |
|---|---|
| `⟲` Reset view | re-fit the camera |
| `✚` Show axes | draw the x/y/z axis widget |
| `#` Show atom labels | label every atom with its number |
| `➤` Show force vectors | draw the per-atom force arrows |
| `▦` Show unit cell | draw the periodic cell box |
| `◉` Show selected only | **isolate** — hide every unselected atom |

Isolate turns itself off when the selection becomes empty, since there would be
nothing left to show.

**Selecting.** Click an atom and a soft amber sphere appears over it. The atom
keeps its own element colour and the geometry is not rebuilt, so selecting is
instant even on a large system and the highlight follows the atom as a
trajectory plays. Click again to deselect; selections accumulate. Clicking is
off while isolate is on — with everything else hidden there is no unambiguous
atom to pick — so turn isolate off to select again.

The panel beside the viewer is the other way to select. **Click** mode picks by
hand; **Filter** mode selects everything matching a rule — by element (`Au,C`),
by atom number (`1-4, 6, 10-11`), by residue (`ALA,DA`), or by label
(`L-electrode`). Several filters combine with AND / OR. The panel shows a live
count and measurement of what is currently selected.

**Measuring.** Measurements come from what you selected, in the order you picked
it:

- one atom → its coordinates, `Au #3 — (0.000, 0.000, 0.000) Å`
- two atoms → the distance, `|H #5 – O #1| = 0.957 Å`
- three atoms → the angle, with the **middle-picked** atom as the vertex,
  `∠H #5 – O #1 – H #6 = 104.5°`

**Tagging atoms.** In an editable view the panel tags the selected atoms with a
region label — L-electrode, R-electrode, bridge, interface, frozen atoms, or a
name you type. Tags show as chips you can remove. These are not decoration: they
are written into the structure's sidecar file and into the generated input
script, so the calculation and the results view both see the regions set here.

**Playing a trajectory.** When a structure has more than one frame, a playback
bar appears under the viewer: `‹` / `›` step, `▶`/`❙❙` play-pause, `⟳` loop, a
speed box in milliseconds per frame (20–3000, default 150), and a slider with an
`i / N` counter. One frame shows no bar. As it plays, the force arrows animate
with the frames — the largest force is drawn gold, the rest shade dim-red to
orange-red by relative size, so converging forces visibly shrink.

**Getting things out.** The Export menu is organised by what you are exporting,
each with a *Save* (into the project) and a *Download* row: **Data** — the
structure as text (xyz / pdb); **Snapshot** — a PNG of the current view,
transparent if you chose that background; **Animation** — the trajectory as a
gif or webm, shown only when there is an animation to export.

Note what "Save into the project" means here, because it crosses a boundary:
**MolView produces the bytes and stops there.** Putting them in the project is
the projects module's job (§ 2), wired up by the host — the viewer holds no file
route and never learns where anything landed.

---

## 2. What MolView is not

Five jobs a viewer could plausibly grow into, and does not. Each boundary is
here for a reason, and every one of them has been crossed at some point by
something that seemed convenient at the time.

| Not | Whose job it is | Why the boundary is there |
|---|---|---|
| a structure **parser** | the server | one parser, one set of chemistry rules. A parser in the browser would be a second, weaker opinion about what a file means, and the two would disagree on the awkward cases |
| a structure **generator** | the server, driven by the Modify tab | building a molecule from a SMILES string, a name, a sequence or a file is a server call the tab's own panels make. A viewer that dispatched generation would have to know its host's modules by name — the exact inversion § 4 exists to prevent |
| a **file manager** | the projects module | MolView produces and consumes bytes. Where those bytes live on disk is not a viewing concern, and MolView holds no file route |
| a place to **keep** a saved session | the workspace module | MolView decides *when* to save and *how far* to step back; the workspace only knows *where the bytes go* (§ 11.2) |
| an **animator of vibrations** | a separate module, with its own document | animating a normal mode is a different job with different data; it is not this module's business and is not described here |

---

## 3. The overall shape

One picture, then the rest of the document fills it in.

```mermaid
flowchart TB
    subgraph outside["Outside MolView"]
      TAB["a tab<br/>its own UI, its own run file, its own plots"]
      SRV["the server<br/>parses structures, performs geometry edits"]
      WS["the workspace<br/>stores the session bytes"]
    end
    subgraph viewer["One viewer — everything below belongs to one owner"]
      H["the handle<br/>make it, drive it, tear it down"]
      M["the model<br/>holds the MASTER COPY,<br/>the selection, the displayed frame"]
      RE["the renderEngine<br/>decides what to redraw,<br/>works out what each frame looks like"]
      SEAL["the sealed layer<br/>holds the DRAWING COPY,<br/>the only code that talks to 3Dmol"]
    end
    TAB -->|"holds"| H
    H --> M
    M -->|"load / edit a structure"| SRV
    SRV -->|"the structure it made"| M
    M -->|"save / restore session bytes"| WS
    M -->|"here is what to draw"| RE
    RE -->|"here is this frame, painted"| SEAL
```

Three things to read off it:

**Data flows down.** The model hands the renderEngine what to draw; the
renderEngine hands the sealed layer a finished frame. Nothing lower ever answers
a question about what the structure *is*. If you want to know where an atom is,
you ask the model — never the drawing. (One narrow thing does travel back up: the
renderEngine asking the drawing whether its own last instruction landed. That
answer goes no further — § 10.10.)

**A tab owns none of it.** A tab has a handle. It does not have the structure,
the renderer, the camera, or a way to reach 3Dmol. What it does own is its own
business: its plots, its parsed run file, its own layout.

**Everything inside the box belongs to one owner.** Mount twice and you get two
of these, sharing nothing — two structures, two selections, two displayed
frames, two cameras. § 5.6.

---

## 4. MolView is a self-contained module

This is worth stating on its own, because everything else depends on it.

**MolView is one ES module, sealed at every edge.** It is imported by name, it
reaches nothing else in the app by name, and nothing in the app can reach inside
it.

**One entry point, and nothing else is importable.**

```js
import { mount, formula } from "/static/lib/molview/index.js";
```

That is the whole surface. `mount` makes a viewer; `formula` turns a list of
elements into a Hill formula and needs no viewer. Every other file in the module
is internal — the model, the stores, the renderEngine, the panel, the sealed
layer. A consumer that imports any of them directly has broken the module, not
found a shortcut.

**Nothing it needs comes from a global.** A viewer needs two things at mount: an
element to live in, and a workspace to save through. Both are handed in. It does
not look up the tab it is inside, does not read app configuration, and does not
consult any other module by name.

**Nothing leaks out.** No 3Dmol object, no store, no DOM node, and no internal
function ever appears on the handle. Within this module the name `3Dmol` occurs
in exactly one file — the sealed layer — which is also the only place that fails
with a clear error if the library is missing. Swapping the drawing library
touches that one file and reaches no consumer of MolView.

**It contains its own state.** The structure, the selection, the switches, the
displayed frame, the undo history and the camera all live inside the viewer. It
keeps nothing in a global, and nothing outside it keeps a copy of anything it
holds.

**The test of all of this:** delete every other web module and MolView still
loads, mounts, draws, selects, measures and exports. The only things it would
miss are the server routes it calls and the workspace it saves through — and
both of those are reached through named routes and an injected accessor, not by
importing anything.

> **Transition.** The module currently also publishes some
> `window.molbuilder.molview.*` values. They are live seams — node-test entry
> points and readiness signals used by the browser tests (§ 13.4) — plus one
> shared model left from before viewers were owned (§ 9.2). None of them are part
> of this contract, and retiring them is [`roadmap.md`](?doc=roadmap.md)'s
> business.

---

## 5. The ideas everything else follows from

Six of them. They are the reason every rule below exists, and a design choice
that breaks one is wrong no matter how convenient it is.

### 5.1 What you see is what you save

The structure on screen and the structure that goes to a calculation are the same
structure, at the same frame. Scroll to frame 40, click Save, get frame 40.

"Which frame am I looking at" and "which frame am I working on" must not be able
to have different answers — which is why there is exactly one number saying which
frame is displayed, held in one place and read by everyone (§ 6.4).

### 5.2 One place holds each fact

Every fact — the atoms, the cell, the selection, which frame — has one home. Two
copies of a fact are two things that must be kept in step, and every mechanism
for keeping them in step is a place they can fall out of step.

Where two forms genuinely must coexist — the real structure and the filtered
thing the graphics library draws (§ 6.3) — one of them is unambiguously the real
one, the other cannot be reached as an answer to anything, and a single number
joins them.

### 5.3 The graphics library is invisible

Nothing above the sealed layer knows the viewer is 3Dmol — not a tab, not a
panel, not the renderEngine. This makes "the same viewer everywhere" a property
of the code rather than a convention people have to remember.

### 5.4 A host needs to know nothing

Hand MolView somewhere to live and a workspace, and get back a viewer. No
knowledge of rendering, of how structures are parsed, or of how sessions persist.
The handle is deliberately small (§ 9.2) so that embedding a viewer is never a
project.

### 5.5 The user's intent is data, and it travels

Region labels, frozen flags, the selection, the frame in view — these are not
decoration. They are written into the structure's sidecar and into the generated
input script, so the calculation and the results view both see what the user set
here. The viewer is where scientific intent gets expressed, so that intent has
to survive the trip.

### 5.6 A viewer is owned

**Every mounted viewer is its own MolView, and it belongs to its owner.** Two
viewers on one page are two structures, two selections, two displayed frames,
two cameras — not one set of facts they fight over. The `owner` given at mount is
what makes that true, and it names *everything* the viewer holds.

This is what makes § 5.2 mean anything: "one place holds each fact" is only a
real rule once *which viewer's* fact you mean is unambiguous. A single structure
shared behind two viewers is not one home for a fact — it is one home for two
facts that happen to collide.

It follows that **the handle is the way in**. A tab holds a handle, a handle *is*
a particular viewer, and there is no global to look one up in — so a tab cannot
reach the wrong viewer by accident.

---

## 6. The data a viewer holds

A viewer holds **one structure**. If it came from a trajectory, it holds **many
frames** of that same structure and **one number** saying which frame you mean.
It holds **what is selected** and **which switches are on**. Everything else you
see — the drawn atoms, the arrows, the highlight — is worked out fresh on each
redraw and never stored.

Everything in this section belongs to one owner (§ 5.6). Two viewers hold two of
all of it.

### 6.1 The four things

**The structure — the same for every frame.** One element symbol per atom, one
set of per-atom tags (a region label, a frozen flag), and optionally a unit cell.
A frame never carries any of these; they are fixed when the structure loads and
are identical for frame 0 and frame 400. That is exactly what makes a trajectory
*one molecule moving* instead of a sequence of different molecules.

**The coordinates — one set per frame.** `frames[f][a]` is atom `a`'s position in
frame `f`. Per-atom forces, when a calculation produced them, sit in a matching
list of the same shape. A structure that is not a trajectory is simply a list of
**one** frame.

**Which frame, and how many there are.** One number saying which frame is meant,
and the range it is allowed to be in. They are held together because neither is
usable alone (§ 6.4).

**What is selected, and which switches are on.** The selected atom numbers with
the order they were picked in — a three-atom angle has to know which one was
picked second — the filter settings, and the on/off switches: isolate, atom
labels, force arrows, unit cell, axes, and the arrow scale. Window settings
(camera, style, background) are held separately, for the reason in § 9.6.

### 6.2 The shapes

```mermaid
classDiagram
    class Structure {
      +string[] elements
      +AtomTag[] annotations
      +Cell cell
    }
    class Coordinates {
      +Frame[] frames
      +FrameForces[] forcesPerFrame
    }
    class DisplayedFrame {
      +int index
      +int count
    }
    class Selection {
      +int[] selection
      +int[] pickOrder
      +Filter[] filters
    }
    class Switches {
      +bool isolate
      +bool showIndex
      +bool showForces
      +bool showCell
      +bool showAxis
      +number forceScale
    }
    class ProcessedFrame {
      +Vec3[] positions
      +int[] sourceIndex
      +string[] elements
      +Label[] labels
      +int[] selection
      +Arrow[] arrows
    }
    class Scene {
      +CellBox cellBox
      +Axes axes
    }
    Structure --> ProcessedFrame : identity, every frame
    Coordinates --> ProcessedFrame : the frame at the index
    DisplayedFrame --> ProcessedFrame : picks which frame
    Selection --> ProcessedFrame : what to highlight
    Switches --> ProcessedFrame : what to draw at all
    Structure --> Scene : the cell, if there is one
    note for ProcessedFrame "worked out per redraw, never stored"
    note for Scene "the same every frame unless the cell changes"
```

**The structure and its coordinates:**

| Field | Shape | What it is |
|---|---|---|
| `elements` | `string[]` | element per atom. **Shared by every frame.** |
| `annotations` | `{label?, frozen?}[]` | per-atom region label and frozen flag. **Shared by every frame.** These are facts about the molecule, not switches — the panel reads them; the drawing does not use them |
| `cell` | `{lattice: [Vec3,Vec3,Vec3], origin: Vec3}` or `null` | the a/b/c vectors, plus the corner the box is anchored at |
| `frames` | `Vec3[][]` | `frames[f]` = the coordinates of frame `f`. At least one. **Coordinates only** — no elements, no tags |
| `forcesPerFrame` | `Vec3[][]` or `null` | `forcesPerFrame[f]` = the forces of frame `f` |

Atom **count**, `elements` and `annotations` are fixed when the structure loads
and are identical for every frame. That *is* the same-atoms rule of § 10.8.

**The switches.** Only `selection` and `isolate` change which atoms are drawn at
all; every other switch adds or removes something drawn alongside them. That is
why turning labels or the cell on never rebuilds the geometry (§ 10.5).

### 6.3 Two copies, and which one answers what

The coordinates exist in **exactly two** forms, and only two.

**The master copy — the real structure.** Every atom, every frame, in the
original order. This is what gets measured, exported, saved, and handed to a
calculation. It is kept clean — never overwritten with a filtered or reduced
list — so every redraw starts from it rather than from whatever is currently on
screen. That is what lets the whole structure come back the moment isolate is
turned off.

**The drawing copy — what the graphics library was handed.** Under isolate the
unselected atoms are gone from it entirely, the survivors are renumbered, and
force arrows are baked in. It can answer exactly one question: what is currently
painted on screen.

**There is no third copy, and adding one is a design error.** A tab may hold its
own parsed run file — but that file carries *different* facts (energies, forces
per step, SCF history), not another copy of the coordinates. It feeds MolView; it
is not one of MolView's two.

Each question routes to the copy that can answer it, and the displayed frame
number is what routes them:

| The question | Answered from | How |
|---|---|---|
| Which frame is displayed — and which one will be saved? | the displayed frame number | `currentFrame()` |
| Where is **every** atom at that frame — to measure, export, save | **the master copy** | `getFrameAllAtoms(i)` |
| What is on screen right now? | **the drawing copy** | nothing outside the renderEngine asks it, and nothing ever asks it for coordinates. It is output, not a source — the one exception is the renderEngine checking its own work (§ 10.10) |
| What was the energy / SCF history at that step? | not MolView's data at all — the tab's own run file | the tab reads its own file, at the same frame number |

### 6.4 The displayed frame and the range it lives in

**They are one fact, kept in one place.** A frame number without the range it is
valid in cannot be used for anything — you cannot draw a slider, clamp a seek, or
follow the end of a growing run without both. So they sit together, are updated
together, and are read through one API. Splitting them is how a slider comes to
offer a frame that nothing can draw.

**Nothing anywhere keeps its own copy of either.** Not the renderEngine, not the
sealed layer, not a tab, not the frame bar. There is nothing to gain: it is one
integer, read a handful of times per redraw, so a private copy saves no
measurable work. What a copy *does* buy is a second thing to keep in step — and
this number answers two questions that must never disagree (what is on screen,
what gets saved). A copy is precisely how they would come to disagree.

**Both answer to the master copy, in this order:**

```mermaid
flowchart LR
    T["the master copy<br/>is updated first, and fully"] --> R["the range<br/>is recomputed from it"]
    R --> I["the frame number<br/>is checked against the range"]
    I --> N["everyone is told once,<br/>and sees a matching pair"]
```

1. **The master copy is updated first, and completely.** A load, an append, an
   edit, a restore — it reaches its final state before anything else moves.
2. **The range is recomputed from it.** Not from the drawing copy, and not from
   what the caller said it was adding. The master copy is the only thing entitled
   to say how many frames exist.
3. **The frame number is checked against that range** and moved if it no longer
   fits.
4. **Only then is anyone told**, and what they see is a matching pair.

No one ever observes a half-updated state. There is no moment when the range
belongs to the new structure and the frame number still belongs to the old one,
because nobody is notified until both have settled. This is what makes "what you
see is what you save" survive a structure changing underneath a user.

**Three kinds of access, and no fourth:**

| | Who uses it, and why |
|---|---|
| **read** — `currentFrame()`, `frameCount()` | anyone who needs to know which frame is meant and how many there are: the frame bar, the measurement readout, export and save, a tab |
| **write** — `setCurrentFrame(i)` | anyone who moves it: the frame bar, playback, a tab following the end of a growing run, a session being restored. A number outside the range is resolved against the range, never taken on trust |
| **subscribe** — `onFrameChange(fn)` | anyone who must react: the slider and its counter, the measurement readout, the renderEngine |

**Every UI reads and sets the displayed frame through exactly this API** — the
frame bar under the viewer, a tab's own scrubber, a keyboard shortcut, playback,
a restored session. There is no privileged writer and no back channel. A UI that
tracked the frame itself would be a second answer to a question that must have
one, and it would be the stale one the moment anything else moved it.

The write is the only way it moves, and it tells **every** subscriber regardless
of what did the moving. A subscriber never has to know which; nothing anywhere
needs its own "did it change?" check.

**It is deliberately not one of the switches.** Playback moves it many times a
second, and pushing that through the switch store would re-render the selection
panel that often and steal focus from a filter box mid-play. A switch is
something a user sets; this is something a movie drives.

### 6.5 Worked out fresh on every redraw, never stored

One frame, after the switches have been applied, becomes a **processed frame** —
and that is the only thing the sealed layer ever receives.

| Field | Shape | What it is |
|---|---|---|
| `positions` | `Vec3[]` | the atoms **actually drawn** — cut down to the selection when isolate is on |
| `sourceIndex` | `int[]` | `sourceIndex[m]` = the **original** number of drawn atom `m`. This map from drawn back to original is why labels still show the right number under isolate |
| `elements` | `string[]` | element per **drawn** atom |
| `labels` | `{position, text}[]` or `null` | the atom-number labels, when that switch is on. `text` is the **1-based original** number (§ 11.3) |
| `selection` | `int[]` or `null` | which drawn atoms to highlight. `null` means *draw no highlight* — which happens both when nothing is selected and under isolate, where every drawn atom is selected and a highlight would say nothing |
| `arrows` | `{start, end}[]` or `null` | force vectors for **this** frame, where `end = start + force × scale` |

The cell box and the axes are **not** in here. They are the same for every frame
unless the cell itself changes, so they are worked out once as scene-level data
and are not recomputed per frame.

**`selection` here is content, not styling.** It says *which* atoms are
highlighted. What the highlight looks like is a fixed constant owned by the
sealed layer — a translucent sphere, amber `#ffd54a`, radius 0.7, opacity 0.5 —
re-placed each frame so it follows moving atoms. Keeping the appearance out of
the per-frame data is what keeps every frame's data identically shaped, and it
means a selected atom keeps its own element colour with the highlight simply
sitting over it.

**Under isolate, the 3D window is display-only.** In-window clicking is off,
because a drawn atom's number no longer equals its real number. The panel curates
the selection instead and always speaks in original numbers. Measurement is a
separate thing from drawing: it takes its atoms from the panel's selection and
reads their coordinates from the **master copy** at the current frame, so it
stays correct and frame-aware without touching the drawing at all. Clicking
returns when isolate is off, where drawn number and real number agree again.

### 6.6 What a viewer does not hold

| Not held | Whose it is | Why |
|---|---|---|
| parsed structure text | the **server** | MolView never parses. It posts bytes and adopts the structure that comes back (§ 11.1) |
| files on disk | the **projects** module | `exportFile()` returns bytes; MolView owns no file route |
| the saved session bytes | the **workspace** module | MolView decides when and how far; the workspace knows where (§ 11.2) |
| trajectory frames, across a session | *nobody yet* | saving more than one frame is planned, not built — see [`roadmap.md`](?doc=roadmap.md) |

---

## 7. The layers

Seven levels, read from outside in: a tab at the top, the drawing library at the
bottom. Each level owns one thing, offers one surface, and has exactly one kind
of caller. **The "never" column is what stops a fact quietly acquiring a second
home** — it is the enforceable half of § 5.2.

| | The level | What it offers, and who calls it | Never |
|---|---|---|---|
| **1** | **the tab** | *No API — it is the caller.* Owns its own UI, its own run file, its own plots. Holds a handle and reaches its viewer only through it | keeps its own copy of the displayed frame, the range, or anything else the viewer holds; reaches past the handle; consults its own file to answer a question about the viewer |
| **2** | **the handle** | Making, driving and tearing down a viewer (§ 9.2). A handle *is* a viewer: one owner, one structure, one of everything. Called by a tab | holds structure data of its own — it passes every read and write down to the model |
| **3** | **the model** | The data API (§ 9.3), one per owner. Called through the handle, and by every level inside the same viewer. **Holds the master copy**, the selection, the displayed frame and its range. This is where the rules are enforced and where read-only is applied, so nothing may go around it | touches the drawing library; exists as one shared instance behind several viewers |
| **4** | **the stores** | Change-and-subscribe. Called only by the model, which assembles them. They exist so state has a home that knows nothing about drawing | draw anything; hold the displayed frame — that is not a switch (§ 6.4) |
| **5** | **the renderEngine** | Commands only — "draw this", "add these frames", "the forces changed", "throw it away". **Called only by the model.** It is *handed* the master copy and the switches, works out what each frame looks like, and passes the result down. It holds nothing of its own | keep its own copy of the displayed frame; answer a question about what the data is; run a change notification of its own |
| **6** | **the drawing commands** | Small, decision-free operations that translate a processed frame into calls the library understands, plus two questions the renderEngine asks *about the drawing* to check its own work (§ 10.10). **Called only by the renderEngine.** It owns exactly one fact: the multi-frame format the library expects | decide how much work a change needs; hold state; answer anything upward |
| **7** | **the sealed layer** | The only code that names 3Dmol. **Called only by level 6.** **Holds the drawing copy** — the movie, the camera, the styles, the picking, the highlight spheres | keep its own frame number, or be a source of truth about coordinates. It draws the frame it is handed and offers no way to read coordinates or frames back out |

### 7.1 Where the two copies sit

The model (level 3) holds the **master copy** — that is what a save, an export, a
measurement and a server request all read. The renderEngine (level 5) is *handed*
it, works it through the switches, and produces the **drawing copy**, which lives
in the sealed layer (level 7) and exists only to be looked at.

Data goes down. Nothing comes back up. The one thing that crosses levels in both
directions is a *question the renderEngine asks about its own work* (§ 10.10), and
that answer never reaches a user.

### 7.2 The shape of each surface tells you its job

The handle is nouns and lifecycle, because a tab wants *a viewer*. The model is
reads and writes with rules attached, because it is the only place truth changes.
The renderEngine is commands only, because a renderer is told what to do and not
consulted. Level 6 is operations with no decisions in them, because deciding is
level 5's job.

When a surface starts growing the wrong kind of thing — a read appearing on the
renderEngine, a decision appearing at level 6 — that is the signal a
responsibility has leaked, and it is visible before anything breaks.

### 7.3 Inside the model: one file, and helpers it hands work to

The model is not one big file. **One central file holds the structure, and hands
each real job to a small helper that does only that job.**

**When the central file builds a helper, it hands over exactly the functions that
helper is allowed to call.** The helper never reaches out on its own. The undo
helper is the clearest case: it has to save and restore the structure, but it
does not need to know the file format — it is simply handed a "make a snapshot"
function and a "put a snapshot back" function. That keeps each helper small,
testable on its own with stand-in functions, and replaceable without disturbing
anything else.

```mermaid
flowchart TB
    DM["the central file<br/>holds the master copy"]
    subgraph subs["Helpers — all internal, none reachable from outside"]
      IN["load a structure in"]
      SE["write the structure out"]
      ST["undo / redo history"]
      OP["the geometry edits"]
      SS["what is selected + the switches"]
      CS["the cell / working state"]
    end
    DM -->|"hands it: make a snapshot, put a snapshot back"| ST
    DM -->|"hands it: read the atoms, apply the server's result"| OP
    DM -->|"hands it: read everything needed to write out"| SE
    DM -->|"hands it: where to put a loaded structure"| IN
    DM -->|"builds it"| SS
    DM -->|"builds it"| CS
```

| Helper | Its job | What it is handed |
|---|---|---|
| load | put a loaded structure into the model | where to put it; how to announce a change; a way to record the first state |
| write out | turn the structure into text, for export and for snapshots | read-only access to the atoms, cell, selection, window state and history position |
| history | undo / redo (§ 11.2) | "make a snapshot" + "put a snapshot back"; where the bytes go |
| edits | the geometry operations (§ 11.1) | read the atoms; apply the structure the server sends back |
| selection | what is selected + the switches (§ 9.5) | *(an optional starting selection)* |
| cell state | the cell and working state | *(nothing — it stands alone)* |

Two more groups sit beside the model and are built the same sealed way: the
**renderEngine**, which turns the master copy into what you see, and the
**selection panel**, which is the panel you click in, the click-to-select wiring,
and the distance/angle maths. The panel reads what is selected; it never talks to
the 3D window directly.

---

## 8. Making and tearing down a viewer

```js
import { mount, formula } from "/static/lib/molview/index.js";

const viewer = await mount(hostEl, workspace, {
  owner: "results-structure",
  mode: "readonly",
});
if (!viewer.ok) {
  console.error("viewer failed to mount:", viewer.error);
  return;                    // dispose() is safe to call anyway
}
// … use it …
viewer.dispose();            // tears down in reverse order of assembly
```

`mount` assembles a complete viewer in one call — the 3D window, the
selection/cell panel and the switches. The frame bar is not part of that
decision: a viewer mounts before it has a structure, and the bar appears once a
structure with more than one frame is loaded into it.

**`owner` names the viewer, and therefore everything in it.** It is not a prefix
on a settings key; it is the identity of an instance. The structure, the
selection, the switches, the displayed frame and its range, the camera, the undo
history, the renderEngine and the sealed layer all belong to that owner. Two
mounts with different owners share nothing (§ 5.6). A viewer with no owner has no
identity, which is why one is always given.

**`mode: "readonly"`** freezes the master copy and changes nothing else — § 9.4.

**Mount always resolves; it never rejects and never returns nothing.** It always
gives back an object with `ok` and a real `dispose`. On success `ok` is true; on
failure `ok` is false and `error` says why, and `dispose` still works. A viewer
that cannot fit — a host narrower than the card's minimum width — renders a blank
card with the error written in it, rather than a half-built viewer.

> **Transition.** MolView can also wire itself into a pre-built card that the
> host laid out, instead of building its own. That path exists for the Modify
> tab's template and is being removed; new hosts hand over an empty element.

---

## 9. The APIs, and who each one serves

Eight surfaces, outermost first — each with who calls it and what it is for —
plus one rule (§ 9.4) that cuts across all of them.

### 9.1 The module entry point

`mount` and `formula` (§ 4). Nothing else in the module is importable, and this
is the only import a consumer ever writes.

### 9.2 The handle — for a tab that wants a viewer

**What it must be able to do, and what it refuses:**

| A tab must be able to | Refused, deliberately |
|---|---|
| find out whether it got a viewer, and tear it down | reaching the 3Dmol object, the stores, or the DOM |
| reach everything this viewer holds — the structure, the selection, the frames, the window state | reaching *another* viewer's; or reaching any of it without going through the model, which is where the rules and the read-only gate live |
| run the movie — play, pause, ask whether it is playing | owning the timer. Playback lives in the mount layer and moves the frame through the same write everyone else uses (§ 6.4) |
| hear that something changed | polling for it |
| set what the viewer *shows* — by changing the data or a switch (§ 10.1) | set how it *looks*. There is no "set the arrows", "set the labels", "show a busy state", "add a toggle". Arrows and labels are **worked out from the data** by the renderEngine, never handed in by a consumer |

**The handle contains the model; it does not mirror it.** This is what being
owned forces. When there was one shared model, a handle that also carried
`getStructure`, `getFrameAllAtoms`, `currentFrame` and the rest was a convenience
— two ways to the same object. Now that each viewer owns its model, a mirrored
read is a *second surface over the same fact*, and one of the two is the one
somebody forgets to update. So the handle carries lifecycle, playback, and one
route to the model; the model carries the data API with the selection and window
stores beneath it.

Adding a read to the handle that the model already answers is the specific move
this rule forbids.

In the examples below that one route is written `viewer.data` — the handle is the
viewer, and `viewer.data` is that viewer's model. There is no other way to it,
and no other viewer's model is reachable from it.

> **Transition.** Today's handle carries fifteen names: lifecycle and playback,
> plus eight reads and writes forwarded from the model. Those eight are what the
> rule above retires — they are how a tab reached a viewer back when there was
> only one to reach. Also today, the module publishes a single shared model as a
> global; while it exists the old rule applies to it — look it up at the moment
> you read, never hold on to it, because its contents change underneath you.

### 9.3 The model — the one place the structure lives

Every read returns a **copy**, so changing what you were given can never change
the viewer.

The surface is organised by **what a caller needs**, not by how the internals
happen to be split. **One need, one main way in.** Where several names serve one
need, exactly one is the main one and the rest are narrower cuts of it; a cut may
disappear, but it must never grow into a rival.

The last column is the read-only rule (§ 9.4), read straight off this table
rather than maintained separately.

| The need | The main way in | Narrower cuts of it | Changes the master copy |
|---|---|---|:--:|
| Get the whole structure | `getStructure` | `getAtoms`, `getElements`, `getCoordinates`, `getSource`, `getFrozen`, `getRegions` | — |
| Get the cell | `getUnitCellInfo` — the resolved cell, always answerable | `getUnitCell` (the raw 3×3 or `null`), `getUnitCellOrigin`, `getAxisKind`, `getVacuum` | — |
| Get one frame's coordinates | `getFrameAllAtoms(i)` — **every** atom, original order, before any filtering | | — |
| Know / move / follow the displayed frame | `currentFrame()` · `frameCount()` · `setCurrentFrame(i)` · `onFrameChange(fn)` (§ 6.4) | | — |
| Build a server request | `factsForRequest()` — the one payload a request is built from | | — |
| Get the structure out as text | `exportFile()` | | — |
| Hear that the structure changed | `subscribe(fn)` — the structure only; the frame has its own channel | | — |
| Reach the selection / the window state | `selection` (§ 9.5) · `view` (§ 9.6) | | — |
| Put a structure in | `installMolecule(input)` | | **yes** |
| Edit the geometry | `applyOp(name)` (§ 11.1) · `discard` | | **yes** |
| Edit the cell | `commitPeriodicityOp` — the one way the cell changes | | **yes** |
| Load or extend the frames | `reloadFrames` · `addFrame` · `addFrames` · `setForces` | | **yes** |
| Move through the session history | `save` · `load(delta)` · `undo` · `state_index` · `uncommitted` (§ 11.2) | | **yes** — `load` and `undo` restore a different structure |

Thirteen needs. That count is the honest measure of the surface; everything else
is a narrower cut, and a cut earns its place only by being what a caller actually
asks for.

**`getFrameAllAtoms(i)` is named for exactly what it promises:** every atom of
frame `i`, in the original numbering, before any selection or isolate filtering.
That is what its callers want — measurement resolves panel numbers against it,
and export writes the frame from it. Naming the promise means no call site has to
restate it, and there is no rival: reading coordinates back out of the drawing
would give the isolated subset under its own renumbering, which is a different
thing and one MolView does not offer.

**The two structure primitives.**

- **`installMolecule(input)`** — the only way a structure gets in. It sends the
  text (and an optional sidecar) to the server and, on the structure that comes
  back, replaces the whole model at once and resets the undo history. Everything
  upstream converges here: a generator builds text and installs it; the file
  browser reads bytes and installs them. One entrance means one place the rules
  are checked and one place the history is anchored.
- **`exportFile()`** — its exact inverse. Returns the structure text plus its
  sidecar, written from **the frame currently displayed** (§ 6.4). It **refuses**
  to produce anything when the geometry and the per-atom tags disagree about how
  many atoms there are, returning nothing rather than writing a corrupt
  structure. It is not a disk write and not the session save.

> **Transition.** Cell writes that predate `commitPeriodicityOp` still exist on
> the object. They go around the gate, which is exactly why there is one way in.
> Do not call them; removing them is a cleanup, not a design question.

**The encapsulation rule.** Consumers go through these. They never parse
structure text, and never reach past this surface into a store or into the
drawing. That is what lets the model stay the single source of truth.

### 9.4 Read-only — one rule, not a list of disabled buttons

**A read-only viewer freezes the master copy. Nothing else changes.**

That is the whole rule, and it is one sentence on purpose: every previous attempt
to describe read-only turned into a list of disabled controls that had to be
maintained by hand and drifted. There is no list. There is one question asked of
every entry in § 9.3's table: *does this change the master copy?* If yes, it is a
**no-op** in a read-only viewer — it returns without effect and without throwing.
If no, it behaves exactly as it does anywhere else.

The line falls exactly where the two copies do. Read-only is a statement about
**the master copy**; the drawing copy is the picture, and looking at the picture
is what a read-only viewer is *for*. Somebody studying a finished calculation can
still select atoms, isolate them, measure them, scrub the trajectory, turn on
force arrows, spin the camera and export what they see — none of that touches the
structure. What they cannot do is change the structure the calculation ran on.

Two consequences that are easy to get backwards:

- **Isolate is not an edit.** It hides atoms from the drawing; the master copy
  still has all of them, which is why the whole structure comes back when isolate
  goes off. A read-only viewer isolates freely.
- **Export is a read.** Getting bytes out of a viewer you cannot edit is the
  point of a read-only viewer, and it changes nothing.

### 9.5 `selection` — the facts about the molecule

What is selected and which of the molecule's features are drawn, all in one
place. The panel, the highlight and the measurements are all **readers** of it;
none of them keeps its own answer.

- **The switches live here** (all off by default), not in the renderEngine and
  not in the panel.
- **What it offers:** turn isolate on or off, set a switch, apply a filter, write
  a region label onto the selected atoms (which replaces that label's previous
  set), adopt a restored session's selection, the click-selection operations
  (toggle, add, remove, all, invert, clear), and the filter builder (mode,
  filters, and how they combine).
- **All of it stays live in a read-only viewer** — selecting and isolating change
  the drawing, not the structure (§ 9.4).

**One selection per owner, and that is the whole of how viewers stay out of each
other's way.** A read-only inspector's selection cannot disturb an editable tab's,
because they are not the same selection — not because something copies one aside.
When every viewer owns its state, "don't let them collide" stops being a
mechanism and becomes a fact.

### 9.6 `view` — the facts about the window

Camera position and zoom, projection, draw style, radius, background colour.

**Why these are separate from § 9.5, when both are things a user switches on.**
The test is: *does working out what a frame contains require reading it?*

| | **Facts about the molecule** — `selection` | **Facts about the window** — `view` |
|---|---|---|
| Examples | what is selected, isolate, atom labels, force arrows and their scale, the cell box, the axes | camera and zoom, perspective vs orthographic, draw style, radius, background |
| What they change | **what is in a frame** — which atoms, and what is drawn beside them | **how the same frame is painted** |
| Who reads them | the renderEngine, when working out a processed frame (§ 6.5) | nobody in that calculation; they pass straight through to the sealed layer |
| If one changed and nothing was recomputed | the picture would be *wrong* | the picture would be *correct, painted differently* |
| They belong to | the structure being worked on | the window it is being looked at through |

That line is checkable, not a convention to remember: a switch that reaches the
processed frame is a molecule fact; a setting the sealed layer applies without
the frame calculation ever seeing it is a window fact. Nothing is in both, and
there is no third kind.

**Neither store mirrors the other.** This is the rule that stops them collapsing
back into two homes for one fact: when a session snapshot is written it **reads**
each store at that moment, and on restore it **puts each value back through the
store that owns it**. It never keeps a parallel set of switches of its own. A
snapshot carrying its own copy of `isolate` would be a second answer to "is
isolate on", and the first symptom would be a restored session that draws one
thing and reports another.

The sealed layer is attached at mount. Window settings applied before it exists
are held and applied when it arrives, so a restore that lands early is not lost.

### 9.7 The renderEngine — commands only

"Here is the data", "here is the cell", "add these frames", "the forces changed",
"show this frame", "draw", "throw it away". Every one of them is an instruction.
None of them is a question, because the renderEngine is told what to draw and is
never consulted about what the data is.

Inside, it is split in two: a **maths half** that works out what to draw with no
drawing library anywhere near it, and an **I/O half** that is the only code
allowed to issue drawing commands. That split is why the interesting part — how
much work a change needs, and what each frame turns into — can be exercised with
no browser at all (§ 13.2).

### 9.8 The drawing commands — small operations with no decisions in them

Load frames, swap to a frame, append frames, apply the overlays, set this frame's
arrows, set the cell geometry, show or hide the "Updating view…" cover, and batch
a group of changes so the screen updates once. Each one translates finished data
into something the layer below can act on. None of them decides anything — which
operation to use is the renderEngine's call, made one level up.

Plus **two questions, asked only by the renderEngine, only about the drawing
itself**: *is there a movie loaded at all?* and *how many frames does it have?*
Both exist for one purpose — so the renderEngine can check whether its own last
instruction actually landed (§ 10.10). Neither is reachable from outside, and
neither answer ever reaches a user.

### 9.9 The sealed layer — the only code that knows about 3Dmol

Beneath the commands sits the one piece of code that names the drawing library.
It **holds the drawing copy** — the movie, the camera, the styles, the picking,
the highlight spheres — and it draws the frame it is handed.

Its surface faces **downward only**. It offers no way to read coordinates back
out, and no way to ask which frame is showing. That is not an oversight: a viewer
that could be asked what it is displaying would be a second answer to a question
the model already owns (§ 6.4), and the wrong one whenever the two had drifted.

This is the layer that makes § 5.3 true. Everything above it — the commands, the
renderEngine, the model, the panel, the tab — could be read end to end without
learning which library draws the molecule.

---

## 10. How a frame gets drawn — the render pipeline

This is the heart of the module: how the master copy plus a handful of switches
becomes what 3Dmol paints. It is one path. There is no second one.

### 10.1 The governing principle

**Every interaction is the same three steps:**

1. change the **data**, and/or
2. change a **switch**, then
3. ask for **one render**.

That is all any button, toggle, panel or streamed update ever does. **There is no
hand-crafted render anywhere.** No control builds its own view, pokes the drawing
library directly, or produces a picture on the side. Given the current data and
the current switches, one piece of code produces the finished frames and hands
them over.

This is why the module can promise "the same viewer everywhere". A second render
path would be a second answer to "what does this structure look like", and the
two would diverge the first time somebody fixed a bug in one of them.

### 10.2 What goes in, and what comes out

The whole pipeline is a **function of two inputs**: the **data** (the master
copy) and the **switches** (§ 6.1). Both are plain values — no drawing-library
objects, no DOM anywhere in it.

**Everything is treated as multi-frame.** There is no single-structure path. One
frame is a set of length one, and it runs through exactly the same steps as four
hundred. The only thing that changes is that the frame bar does not appear.

**The output is fully-ready data.** What reaches 3Dmol is finished — every frame
already filtered, with its arrows baked in. The pipeline does not micro-manage
the library frame by frame; it hands over the complete set and 3Dmol then uses
its own GPU acceleration to display and animate it. (Labels and the highlight are
the exception, and § 10.6 explains why: they are free-standing objects, re-placed
per frame rather than carried inside it.)

That division of labour is the whole performance story: **we do the derivation
once, up front; the library does the fast part, repeatedly.**

### 10.3 The per-frame steps, in order

For **each** frame, two steps, and the order matters:

```mermaid
flowchart TD
    subgraph PF["for every frame f"]
      C0["start: the master copy's frames[f]<br/>+ the shared identity (elements, tags)"]
      SEL["STEP 1 — filter by selection<br/>isolate ON and something selected?<br/>keep only those atoms, renumber them,<br/>and record where each came from"]
      OV["STEP 2 — add the overlays<br/>keyed to the atoms that survived step 1"]
      OUT["the finished frame f"]
      C0 --> SEL --> OV --> OUT
    end
    OUT --> LOAD["load ALL finished frames into 3Dmol at once (§ 10.4)"]
```

**Step 1 — filter by selection.** If isolate is on *and* something is selected,
the frame keeps only the selected atoms and drops the rest. Otherwise it keeps
everything.

Dropping atoms renumbers them, so this step also records **where each drawn atom
came from** — the map back to its original number. Everything downstream depends
on that map existing; it is what lets a label still show `#47` for an atom that is
now third in the list.

**Step 2 — add the overlays**, keyed to whatever survived step 1:

| Overlay | Produced when | From what | Note |
|---|---|---|---|
| **atom-number labels** | the labels switch is on | one label per drawn atom | the text is the atom's **original** number, recovered through the map from step 1, and converted to 1-based by the one shared translation (§ 11.3). Never its position in the filtered list |
| **the selection highlight** | something is selected **and isolate is off** | the list of drawn atoms to highlight | under isolate this is deliberately empty: the drawn set already *is* the selection, so highlighting all of it would say nothing. The pipeline emits only *which* atoms — never what the highlight looks like |
| **force arrows** | the forces switch is on and the data carried forces | frame `f`'s forces, times the scale | frame `f`'s arrows come from frame `f`'s forces. Getting this wrong shows converged forces on an unconverged frame |
The result, for every frame, is the finished data of § 6.5.

**Two things are deliberately not in that table, because they are not per-frame.**
The **unit cell box** and the **axes** are scene-level: they are worked out once
from the cell and the origin, and are the same for every frame unless the cell
itself changes (§ 6.5). Recomputing them per frame would be work that produces an
identical answer four hundred times.

**Measurement is deliberately not in this list.** The position / distance / angle
readout is the result of a user *interacting* with the view, not part of
producing a frame. It lives on its own (§ 11.3), takes its atoms from the panel's
selection, and reads coordinates from the master copy — which is exactly why it
stays correct under isolate, where the drawn numbering has changed.

> **The cell's geometry and the cell's visibility travel separately.**
>
> The box's vectors and its anchor corner are **structure** data. They are handed
> down **unconditionally** whenever they change — at load, and again when the cell
> is edited, through their own command (§ 9.8) — **even while the cell is
> hidden**. So the anchor is always current.
>
> The visibility switch carries **only a boolean**: show it or don't. It never
> carries geometry.
>
> Keeping those two apart is what makes a cell edit an overlay refresh rather than
> a rebuild (§ 10.5): the atoms did not move, so only the box and the axes need
> re-applying.
>
> The reason it is stated as a rule: if geometry is gated behind the visibility
> switch, it only ever arrives while the cell is already shown — so turning the
> cell **on after a hidden load** draws the box from the world origin instead of
> the structure's corner. A test of this must assert **where the wireframe is
> drawn**, not what the cell data says. The cell data is right the whole time;
> that is exactly why checking it proves nothing.

### 10.4 Load once; playing is a frame swap, not a redraw

Every finished frame is handed to 3Dmol **up front**, in one multi-frame load.

Then, when the user steps or plays, nothing is re-derived and nothing is
re-processed. The pipeline asks the library to **switch to a frame it already
holds** — one call. Playback is a native swap at the library's own speed, not a
render.

This is the single largest performance decision in the module, and it is why
scrubbing a 400-frame optimization is smooth: the per-frame work was paid once,
at load.

### 10.5 How much work a change costs

A render is **not** always a rebuild. Given what changed since the last one, the
pipeline does the least work that still produces the correct result — still one
place and one decision, not a second path.

**The cost is chosen by *what changed*, never by how big the system is.** There
is no atom-count threshold and no magic number anywhere in this decision.

| What changed | What it costs | What actually happens | "Updating view…" cover |
|---|---|---|---|
| the displayed frame only — scrubbing or playing | **frame swap** | ask the library to show a frame it already has (§ 10.4) | no |
| an overlay switch, with the same atoms drawn — the highlight while *not* isolating, labels, forces or their scale, the cell, the axes, **or a cell edit** | **overlay refresh** | re-derive and re-apply just the overlays. The coordinates are not rebuilt | no |
| new frames arrived from a running job | **append** | process only the new frames and extend the movie. The displayed frame does not move | no |
| **the set of drawn atoms changed** — isolate toggled, or the selection changed *while* isolating — or a whole new structure loaded | **rebuild** | re-process every frame and reload the movie | **yes** |

Two entries are worth reading twice, because both are easy to get wrong:

- **A cell edit is an overlay refresh, not a rebuild.** The atoms did not move;
  only the box and the axes changed.
- **A streamed append extends the movie.** It is not a reload. Only isolate,
  selection-while-isolating, and a full load rebuild.

### 10.6 Why the costs split exactly there — how overlays ride the movie

The movie the library holds contains **coordinates only**. Overlays attach to it
in two different ways, and that difference is what decides which change costs
what.

**Force arrows are baked into the movie, per frame, at load.** Frame `i` carries
frame `i`'s arrows, so a frame swap shows the right arrows for free — no
recompute. That is why changing the forces switch or the arrow scale re-derives
arrows for every frame and **re-bakes them in place**, without touching the
coordinates: an overlay refresh, not a rebuild.

**Labels, the highlight and other marker shapes are re-placed for the shown
frame on each swap.** They are free-standing objects sitting at atom coordinates.
A frame swap moves the atoms but not those objects, so each swap must repaint
them at the new positions. That is a handful of shapes — one frame's worth — not
a movie rebuild, and critically it **never restyles the molecule's geometry**.

A rebuild is the only cost that reparses coordinates: it rebuilds the movie, and
also re-bakes the arrows and re-applies the shown frame's labels and highlight.
It is the only one that raises the cover.

> If those shapes are not re-placed on a swap, they stay where frame 0 put them
> and visibly drift off the atoms as a trajectory plays. That is the failure this
> mechanism exists to prevent, and it is what a test of it must actually check —
> that the shapes move with the frames, not merely that they exist.

### 10.7 Why the highlight is a shape and not a restyle

This is a performance decision with a measured basis, and it explains a
constraint that runs through the whole design: **the highlight says *which* atoms,
never *what it looks like*.**

Selecting an atom has to do three things: show the atom clearly, be cheap on a
click, and keep up with a playing trajectory. Three mechanisms were measured:

| Mechanism | One click costs | During playback | Verdict |
|---|---|---|---|
| a translucent **sphere shape** over the atom, re-placed each frame | ~2–8 ms, and **flat** — independent of atom count | re-place a few shapes per frame | **this is what ships** |
| a **second model** drawn over the first | ~35 ms | rides the frame swap | hides the atom underneath; the fix for that resets on every frame swap, so it needs per-frame patching anyway |
| **dimming everything else** and popping the selected atoms | ~24–109 ms | rides the frame swap | washes out the scene — and see the hard fact below |

Two facts decided it:

1. **The drawing library has no partial model update.** Restyling *one* atom
   rebuilds the entire model's geometry. Measured: styling 1 atom costs the same
   as styling all of them — tens to hundreds of milliseconds on a 2000-atom
   model. So *any* highlight living in the atom styles pays a cost proportional
   to the whole structure on every single click.
2. **The library renders shapes translucently, but rebuilds their material every
   frame.** So a shape highlight needs re-placing each frame — cheap, a few
   objects — but it renders correctly with the atom's own colour showing through,
   with no tricks.

A shape touches nothing in the model. So a click is a few objects added or
removed, a frame swap re-places a few objects, and the molecule's geometry is
**never** rebuilt because of a selection. That is what "pay only for what
changed" means in practice.

Its one weak spot, stated honestly: a very large selection — hundreds of atoms —
on a *playing* trajectory re-places many shapes per frame and will drop the frame
rate. That is rare, and the answer if it ever bites a real workflow is to handle
that case then, not to make every click more expensive now.

### 10.8 The two ways data changes

These are different and must not be confused.

**A full load replaces everything.** It establishes the atoms' identity — count,
elements, order — from frame 0, resets to frame 0, and rebuilds every frame from
scratch. The structure that arrives has already been validated upstream; the
pipeline does not read files and does not re-check what the parser guaranteed.

**An append adds frames to what is already there.** A running job produces steps
one or a few at a time, and they extend the existing data rather than reloading
it. The rules:

1. **Something must already be loaded.** Appending with nothing loaded is a hard
   error — there is no atom identity to append to.
2. **Each new frame is checked against that identity before anything reaches the
   drawing.** Same atom count. Elements are not re-sent — a streamed frame
   carries coordinates only, because identity was fixed at load.
3. **A mismatch is a hard error.** Never padded, never truncated, never guessed
   into fitting.
4. **New frames go through the same two steps** as every other frame (§ 10.3),
   and their forces are taken from the correct new frame.
5. **The displayed frame does not move.** A user watching frame 12 keeps watching
   frame 12 while the run grows past it.

### 10.9 What happens during a rebuild

A rebuild takes long enough to be visible, so it shows the "Updating view…" cover
and locks the viewer while it works. That leaves a window in which other things
can arrive — a user click, or a timer-driven poll delivering new frames that no
amount of disabled buttons could stop. **Nothing that lands in that window is
silently dropped.**

- **Switch changes** are honoured by the rebuild reading the switches *when it
  runs*, not when it was scheduled. The latest values win.
- **Pushed operations** — new forces, a seek, appended frames — are held and
  replayed in arrival order once the rebuild finishes. A seek and a set of forces
  are latest-wins, since only the last one matters. **Appended frames
  accumulate**, because each poll tick's frames are a distinct piece of the run
  and losing one would leave a hole.
- **A full load cancels the held operations.** It replaces the atom set, so
  anything queued refers to atoms or frames that no longer exist. A full load is
  never itself refused: it supersedes a rebuild already under way, because it is
  the more authoritative statement about what the structure is.

Otherwise: one update at a time. Nothing races a half-built movie.

### 10.10 Keeping the offered frames drawable

**The range comes from the master copy; the drawing is made to match it.** The
master copy says how many frames exist (§ 6.4), the frame bar offers exactly that
many, and the pipeline's job is to make the drawing able to show every one.

This is what the two questions of § 9.8 are for. After acting, the pipeline can
ask the drawing whether a movie exists and how many frames it has, and compare
that with the master copy. **A check is only worth making against something that
could disagree** — asking the copy you just grew how big it is confirms nothing,
because it agrees with itself by construction. The only informative question is
whether the *drawing* ended up with as many frames as the *structure* has.

Two rules follow, both on append:

- **Appending to a structure with no movie yet becomes a rebuild.** A movie is
  only built once there is more than one frame, so a run caught at its very first
  geometry has none — and appending to a movie that does not exist quietly does
  nothing at all. This is the case the "is there a movie?" question exists to
  catch.
- **A drawing found short of the master copy is rebuilt from it.**

So `frameCount()` is the master copy's length and **the only count anyone is ever
offered**. The drawing's own count is not reachable from outside the pipeline and
exists only to catch a redraw that silently failed. A mismatch is never shown to
anybody; it triggers a rebuild.

### 10.11 Planned: doing even less work on an overlay refresh

Today an overlay refresh re-derives the overlays. It could re-derive only the
ones that actually changed.

The scene is a fixed stack of independent layers — atom style, labels, force
arrows, the highlight, the cell box, the axes — drawn in that order, each a
function of a **declared** set of inputs and of nothing else. Two rules would
follow: a change dirties only the layers that declare it as an input (a click
dirties the highlight and nothing else; the labels switch dirties labels and
nothing else; a frame swap dirties no layer's content at all, only positions),
and a dirty layer applies **only its difference** rather than clearing and
rebuilding.

The independence is the point beyond speed: because no layer's output depends on
another layer's state, any combination of switches produces the correct scene and
no switch can corrupt another. Cross-layer coupling — one layer's repaint
depending on a different mechanism having fired — is exactly the shape of the
drift bug described in § 10.6.

The highlight already works this way. The rest is planned, not built
([`roadmap.md`](?doc=roadmap.md)), and it changes none of § 10.5's four costs —
it makes the overlay-refresh one do less.

---

## 11. The other connections

### 11.1 Geometry edits go to the server

MolView never changes coordinates itself. An edit is **data describing what to
do**: `applyOp(name)` posts to the matching server route and applies the
structure that comes back, all at once. One small table declares each operation's
shape, rather than each one being hand-coded:

| Operation | The selection is | With nothing selected | Needs exactly | Effect on atom count |
|---|---|---|:--:|---|
| `translate` | the thing being moved | act on all atoms | — | unchanged |
| `rotate` | the thing being rotated | act on all atoms | — | unchanged |
| `orient` | a reference the move is defined against | refuse | 2 | unchanged |
| `add_atom` | a reference the new atom attaches to | refuse | 1 | grows |
| `electrode` | a reference | fall back to centring on the origin | — | grows |
| `symmetric_electrodes` | a reference | fall back to centring on the origin | — | grows |
| `delete` | the atoms to remove | refuse | — | shrinks |
| `calibrate` | the thing being mapped | act on all atoms | — | unchanged, whole-structure only |

Those columns drive one generic piece of code. The count requirement is checked
**before** the request goes out — `orient` with one atom selected never reaches
the network. `calibrate` always takes the whole-structure path even with a
partial selection, because it rigidly maps every atom into the cell and clears
the cell origin.

```js
await viewer.data.applyOp("delete");                 // delete the selected atoms
await viewer.data.applyOp("symmetric_electrodes");   // electrodes anchored on the selection
// in a read-only viewer both do nothing — they would change the master copy (§ 9.4)
```

The operation name **is** the server route segment. The delete operation is
`delete`, not `deleteAtoms`; the add operation is `add_atom`. Use these names
exactly.

**The three routes MolView calls** are: load a structure, perform one geometry
edit, and resolve a cell. On the way in, the server's payload is normalised into
the shapes of § 6.2 (regions become labels, `is_frozen` becomes `isFrozen`).

> The **field-level** JSON of those payloads — the structure envelope, the atom
> row, the error envelope — belongs to [`web-api.md`](?doc=web/web-api.md). This
> document names the routes and the direction data flows; copying the schemas
> into two documents is how two documents come to disagree.

### 11.2 Session history, and what the workspace owns

`save`, `load` and `undo` are the viewer's undo/redo history, and the mechanism
is internal to MolView.

It owns the position in the edit sequence (0 is the state the structure opened
at), a flag saying whether there are unsaved changes, and the save/restore chain
itself: **save** takes a snapshot of the current state; **load(delta)** moves
along the history, so `load(-1)` steps back; **undo** is exactly `load(-1)`.

**There is no automatic write.** Only installing a structure — the one anchoring
write — and an explicit save or load touch storage, and each moves the history
position only after its round trip has finished. The history mechanism is blind
to the file format: the model hands it a way to make a snapshot and a way to put
one back, and nothing else.

**MolView owns the whole mechanism and the policy** — when to save, what to
prune, how far back to step. The **workspace** module owns only what sits
underneath: where the bytes actually go, reached through an accessor handed in at
mount. That is the entire division. See
[`workspace.md`](?doc=web/workspace.md).

A snapshot carries the window state (§ 9.6) but **not** trajectory frames; saving
more than one frame is planned, not built.

### 11.3 One atom-numbering translation, in one place

Atom numbers are **0-based in code** and **1-based on screen**, and MolView never
writes a bare `+1` of its own anywhere.

One shared piece of code owns the translation in both directions: the number a
user reads, and the reverse — turning a typed 1-based input like `1-4, 6` in the
"by atom number" filter back into the 0-based numbers the server expects.

**Every** surface that shows or accepts an atom number goes through it: the
measurement readout, the atom-list column, the labels in the 3D window (§ 10.3 step 2),
the filter panel. That is why they cannot drift apart, and why the first atom
reads as `#1` everywhere even though the code sees `0`.

This is the browser end of a rule that spans the whole application — the same
translation exists on the server side, and the number a user reads must equal the
atom number in the generated input file. Its single home is
[`model/overview.md`](?doc=model/overview.md) § 2, and MolView defers to it
rather than restating it.

**Measurement is its own layer, not part of drawing.** The readout in the 3D
window repaints on a selection change **or** a frame change. Its maths comes from
pick order — which is why the vertex of a three-atom angle is the atom picked
second — and its coordinates come from the master copy at the current frame, so
it is correct while a trajectory plays and correct under isolate.

---

## 12. Worked examples

### 12.1 Delete two atoms, then undo

A user selects two atoms in the panel and clicks Delete.

1. The tab calls the delete operation on its handle.
2. The model passes it to the **edits helper**, which sends the selected atoms to
   the server and applies the smaller structure that comes back — all at once.
3. Because the structure changed, the model tells the **history helper** to
   record the new state.
4. The user clicks Undo. The model asks the history helper to step back one
   state, and hands the previous snapshot to the **load helper** to put back.

Every step crosses exactly one helper, and each helper only ever calls the
functions the model gave it. That is why the module stays sealed: the outside
sees the handle, and never any of this.

### 12.2 A trajectory that is still growing

The Results tab is watching a geometry optimization that is still running. It
polls, gets more frames, and hands them to the viewer:

```js
viewer.data.addFrames(newFrames, { forces });
```

What happens, in order (§ 6.4): the master copy grows; the range is recomputed
from it; the displayed frame is checked against the new range; everyone is told
once. If the user was watching the newest frame, the tab moves the displayed
frame to the new end through the ordinary write, and the slider and counter
follow because they are subscribers, not trackers.

The renderEngine appends to the drawing — and then checks whether the drawing
actually grew. The interesting case is the very first poll: a run caught at one
geometry has no movie yet, so appending would silently do nothing. The check
catches it and the drawing is rebuilt instead (§ 10.10), with the "Updating view…"
cover appearing for that one redraw.

Note what the tab does **not** do: it does not track the frame count, it does not
decide whether a rebuild is needed, and it never asks the drawing anything.

### 12.3 A read-only viewer in the Results tab

Mounted with `mode: "readonly"` (§ 9.4). The user selects two atoms, gets a bond
length, turns on isolate to see them alone, turns on force arrows, scrubs to the
last frame, spins the camera, and exports the structure as text — every one of
which works normally.

They then try an edit. Nothing happens: no error, no exception, no change. The
structure that the calculation ran on is exactly as it was.

### 12.4 Measuring an angle while a trajectory plays

Three atoms are selected in the panel, in order. The middle-picked one is the
vertex.

On every frame change the measurement readout recomputes: it takes the three
atom numbers from the selection, asks the model for **all** atoms at the current
frame, and computes the angle from the master copy — never from the drawing. So
the angle stays correct while the movie plays, and it stays correct under
isolate, where the drawn numbering no longer matches the real one.

---

## 13. How the tests are designed

**Every test is derived from this document, never from the source.** That is the
project's testing rule applied here, and for this module it is a correction — the
suite that came before was largely the opposite.

### 13.1 What that rules out

- **Reading the implementation to build the assertion.** A test that picks apart
  a file's returned object and asserts the names it finds can only ever confirm
  that the code still says what it said. It passes for a surface that has drifted
  away from this document, and it fails for a rename that changed nothing.
- **Lists of names copied out of the code.** A pinned list of method names is a
  transcription, not a contract. The contract is *what a surface must be able to
  do and must refuse to do* — the "never" column of § 7, not a spelling.
- **Stand-ins that copy how the code happens to work.** A stand-in takes the
  place of a level, so it must obey **that level's rules from this document**. A
  stand-in for the sealed layer that claims a two-frame movie loaded while also
  reporting that no movie exists describes something this design forbids — and a
  suite built on it will confirm behaviour that cannot happen while missing
  behaviour that does.

### 13.2 Three levels of test

| Level | Runs | Derived from | Shows |
|---|---|---|---|
| **Behaviour, no browser** | node | § 6, § 10.3 | the model's rules, and the per-frame calculation — values in, values out |
| **Boundary behaviour** | node, with stand-ins that obey this document | § 7, § 10.5–10.10 | how much work a change takes, and that each level refuses what its "never" column forbids |
| **End to end** | a real page | § 1.1 | what a user does: select, isolate, measure, scrub, play, export |

### 13.3 What each rule obliges a test to show

This table is the test plan. **A rule with no row here is a rule nothing guards.**

| The rule | A test must show |
|---|---|
| § 4 — the module is self-contained | nothing outside is importable but the entry point; the module mounts with only a host element and a workspace |
| § 5.6 — a viewer is owned | two mounts hold two structures, two selections, two displayed frames, two cameras; changing one leaves the other untouched, and neither is reachable from the other's handle |
| § 6.1 — one frame is not a special case | no read, edit, export or save path treats a single frame differently from four hundred |
| § 6.3 — saving writes the displayed frame | exporting at frame *N* yields frame *N*'s coordinates, not frame 0's |
| § 6.4 — nothing keeps its own copy | exactly one place answers "which frame"; one write reaches **every** subscriber, whatever moved it |
| § 6.4 — master copy, then range, then frame, then notify | after a load that shortens a trajectory, no subscriber ever sees a range from the new structure beside a frame number from the old one; an out-of-range write is resolved against the range, not accepted |
| § 6.5 — the drawn-to-original map holds | under isolate, labels carry original numbers and measurement resolves panel numbers against the master copy |
| § 6.5 — the highlight is content, not styling | per-frame data carries no colour, radius or opacity |
| § 6.6 — no file route | the module reaches no file endpoint |
| § 8 — mount always resolves | a mount that cannot fit still returns `ok === false` **and** a working `dispose`; nothing rejects, nothing returns nothing |
| § 9.2 — the handle refuses appearance | there is no way through the handle to push arrows, labels, a busy state or a toggle |
| § 9.3 — one need, one main way in | a narrower cut returns exactly what the main way in holds for that field — the two cannot disagree |
| § 9.4 — read-only freezes the master copy and nothing else | every change to the master copy is a no-op **and does not throw**, while select, isolate, scrub, camera and export all work normally |
| § 9.5 — one selection per owner | a read-only viewer's selection changes leave an editable viewer's selection untouched |
| § 9.6 — the two stores do not mirror each other | a session snapshot carries no second copy of a molecule switch; restoring puts each value back through the store that owns it |
| § 9.7 — the renderEngine answers nothing | it offers no read of the data and no read of the displayed frame |
| § 9.8 — the sealed layer answers nothing upward | it offers the renderEngine its two self-check questions and nothing else — no coordinates, no frame read-back |
| § 10.1 — one render place | no control produces a picture on the side; every interaction is a data or switch write followed by one render |
| § 10.3 — the two steps, in that order | the filter runs before the overlays, and the overlays are keyed to the atoms that survived it |
| § 10.3 — a label carries the original number | under isolate, a drawn atom's label shows where it came from, not its position in the filtered list |
| § 10.3 — frame *f*'s arrows come from frame *f* | arrows on a played trajectory match their own frame's forces |
| § 10.3 — cell geometry and cell visibility travel separately | turning the cell on **after** a hidden load draws the box at the structure's corner, and a cell edit while the cell is hidden still updates the anchor. The assertion is where the wireframe is drawn, not what the cell data says |
| § 10.3 — the cell box and the axes are worked out once | they are not recomputed per frame, and playing a trajectory does not re-derive them |
| § 9.9 — the sealed layer faces downward only | there is no way to ask it for coordinates and no way to ask it which frame is showing |
| § 10.4 — playing does not re-process | stepping or playing issues no per-frame derivation; the frames were finished at load |
| § 10.5 — the cost matches what changed | flipping a switch does not reload coordinates; an isolate does; the choice never consults the atom count |
| § 10.6 — shapes move with the frames | after a swap, labels and the highlight sit on the atoms' new positions, not where frame 0 left them |
| § 10.7 — a selection never restyles the model | a click adds or removes shapes and issues no model restyle, and its cost does not grow with atom count |
| § 10.9 — nothing is lost during a rebuild | frames that arrive mid-rebuild all appear afterwards; a seek and a force update keep only the last; a full load cancels what was queued and supersedes the rebuild |
| § 10.10 — the offered frames are drawable | appending to a structure with no movie rebuilds instead of extending nothing; a short drawing heals |
| § 10.10 — only the master copy's count is offered | the count a consumer reads never comes from the drawing |
| § 10.8 — same atoms, every frame | a frame with a different atom count is a hard error, never coerced |
| § 10.3 — forces in, arrows out | handing in ready-made arrows draws nothing |
| § 11.1 — the count requirement is checked first | `orient` with one atom and `delete` with none are refused locally, with no request sent |
| § 11.2 — there is no automatic write | nothing persists except through installing, saving or loading, and each moves the history position only after its round trip finishes |
| § 11.3 — one translation, one place | every surface agrees with the shared translation; none computes its own `+1` |

### 13.4 What makes this testable at all

The per-frame calculation and the selection store both run with no browser,
because neither touches the drawing library. There is an in-repo demo page that
exercises a multi-frame structure end to end. The browser tests use readiness
signals the module publishes for that purpose.

---

## 14. Every section, and what it is for

The check that keeps this document honest: each section exists because one of
§ 5's ideas needs it. A section that serves none of them is either describing an
accident of the implementation or documenting something that belongs elsewhere.

| Section | Serves | Because |
|---|---|---|
| § 1 the goal, § 1.1 what it looks like in use | 5.1, 5.5 | one set of controls, learned once; measuring and tagging are how intent gets expressed |
| § 2 what MolView is not | **all six** | the ideas say what MolView owns; this says what it must refuse, which is the harder half |
| § 3 the overall shape | 5.4 | one picture of what a host does *not* own |
| § 4 a self-contained module | **5.3, 5.4** | sealed at every edge is what makes the other five enforceable rather than aspirational |
| § 5 the ideas | — | the source everything else is checked against |
| § 6 the data | **5.1, 5.2** | the whole of what-you-see-is-what-you-save, and one-home-per-fact |
| § 7 the layers | **5.2, 5.3** | the "never" column is how one-home-per-fact and an invisible library get enforced instead of remembered |
| § 8 making a viewer | **5.4, 5.6** | embedding is one call, and `owner` is what makes it a viewer of its own |
| § 9 the APIs | 5.2, 5.4 | each surface named with who it serves, so nothing grows a second way to the same fact |
| § 9.4 read-only | 5.1, 5.2 | one rule instead of a list of disabled buttons that drifts |
| § 10 the render pipeline | **5.1, 5.2, 5.3** | the one path from the master copy to the picture: what each switch produces, in what order, at what cost |
| § 11 the other connections | 5.2, 5.5 | the server, the workspace, and the one atom-numbering translation |
| § 12 worked examples | — | the concepts above, in the order they actually happen |
| § 13 the tests | all six | a test derived from the source cannot defend an idea |
| § 15 the file map | — | for when you open the code |

Two parts of this document earn their place by what they **exclude**: § 2 refuses
five jobs by name, and § 11.1 keeps field-level JSON out. Boundaries are
load-bearing — most of what has gone wrong in this module was a fact quietly
acquiring a second home.

---

## 15. The file map

For reading the code. The helpers of § 7.3 are not repeated here; everything
below lives under `lib/molview/`.

| File / directory | Owns |
|---|---|
| `index.js` | the entry point — `mount`, `formula` (§ 4) |
| `mount.js` | assembling a viewer, the handle, the playback timer (§ 8) |
| `data-model.js` | the model — the master copy and the data API (§ 6, § 9.3) |
| `render-engine/` | the renderEngine: what to redraw, the per-frame maths, the drawing commands (§ 9.7–9.8) |
| `selection/` | the panel, the click-to-select wiring, the distance/angle maths (§ 9.5) |
| `frame-controls.js` · `measurement-overlay.js` | the frame bar and the measurement readout |
| `_atom-index.js` | the one atom-numbering translation (§ 11.3) |
| `_atom-channels.js` | the per-atom channels behind the selection store |
| `_viewer-overlay.js` | the internal corner-overlay framework |
| `demo.js` | the in-repo multi-frame demo page (§ 13.4) |

**Deliberately not listed:** the sealed layer. No consumer names its file and
neither does this document (§ 4).

---

> **Planned, not built.** Saving more than one frame of a trajectory, and finer
> control over exactly which parts of a drawing need refreshing, live in
> [`roadmap.md`](?doc=roadmap.md).

> **Where the code has not caught up.** Being owned (§ 5.6) and its consequences
> — one model per viewer instead of one shared model, the handle as the way in
> rather than a mirror of the model (§ 9.2), the frame number and its range held
> together and updated after the master copy (§ 6.4), read-only as a rule about
> the master copy (§ 9.4) — describe where the module is going. The places marked
> **Transition** say what the code does today. The design was settled first, so
> the change has something to be measured against; everything not marked
> Transition describes what ships.

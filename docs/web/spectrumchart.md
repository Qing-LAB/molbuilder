# SpectrumChart — drawing a vibrational spectrum, and picking a mode out of it

**Role:** contract
**Domain:** web
**Companions:** [`spectra.md`](?doc=web/spectra.md) — the Spectrum tab, which is
what drives this chart: where the modes come from, what a Raman activity is, and
what else on the page moves when a mode is picked;
[`vibrationview.md`](?doc=web/vibrationview.md) — the viewer beside it on that
same tab: the mode a click here selects is the mode it animates, and the two never
speak to each other directly; [`overview.md`](?doc=web/overview.md) — the register
of web modules.

SpectrumChart is the one component in the browser that draws a vibrational
spectrum. Hand it a list of modes and it draws them as sticks with an optional
broadened envelope over them — and when the user clicks near a peak, it says
which mode they meant.

**This document is the design of that component** — what it is for, what it
refuses to do, what it holds, what each API is for, and how its tests are derived.
It is not a tour of the current code; the code it replaces is 345 lines inside a
3,671-line tab controller. § 13 maps the parts to files and says which are built.

> **Words used in this document.**
>
> - **A chart** — one mounted SpectrumChart: one drawing surface and everything
>   it holds. Two on a page are two of these, sharing nothing.
> - **The handle** — the object you get back from `mount`. It *is* the chart;
>   there is no other way to reach one.
> - **The box** — the rectangle the chart lives in: how wide and how tall it is
>   on screen. Whoever puts the chart on the page decides it (§ 5.4).
> - **The sealed layer** — the one file in this module allowed to name Plotly.
>   Nothing above it knows that library exists.
> - **A mode** — one row of the spectrum: a wavenumber, optionally a Raman
>   activity, and whether it is imaginary. The chart holds a list of them.
> - **A stick** — one mode drawn: a vertical line at its wavenumber, as tall as
>   its activity.
> - **The envelope** — the smooth curve laid over the sticks, a sum of
>   Lorentzians of a chosen width. It is what a measured spectrum looks like.
> - **The band** — the invisible region around each stick within which a click
>   counts as clicking that mode (§ 6.3). It is the reason picking a mode is not
>   a test of aim.
> - **The two pictures** — with strengths known, a stick's height means its
>   strength; with none known, every stick is the same height (§ 6.2).
> - **An imaginary mode** — one whose frequency came out imaginary, meaning the
>   structure is not sitting at rest (§ 6.4).

---

## 1. The goal

**One component that draws a spectrum, and turns a click into a mode.**

Here is the whole job in one picture. A calculation produces a list of **modes** —
each one a way the molecule can vibrate, with a frequency and a strength. The
chart draws each as a vertical line, a **stick**, at its frequency and as tall as
its strength, then lays a smooth curve over them so the picture looks like a
spectrum you would measure rather than a barcode:

```
  strength
  (Å⁴/amu)
      │                        ╭─╮  ← the smooth curve: what a real
   14 │                       ╭╯ ╰╮    spectrum looks like, because
      │        ╭╮             │   │    real lines have width
   10 │       ╭╯╰╮            │ ┃ │
      │       │  │           ╭╯ ┃ ╰╮
    6 │      ╭╯  ╰╮      ╭─╮ │  ┃  │
      │      │ ┃  │     ╭╯ ╰╮│  ┃  │
    2 │     ╭╯ ┃  ╰╮   ╭╯ ┃ ╰╯  ┃  ╰╮
      │  ╭──╯  ┃   ╰───╯  ┃     ┃   ╰──╮
      └──┴─────┸──────────┸─────┸──────┴────────────▶  frequency (cm⁻¹)
              323        826   1131
                                ▲
                                └── the selected mode, drawn in its own colour

              each ┃ is one mode;  the curve is their sum
```

**And clicking it is not a test of aim.** A stick is one pixel wide. Around each
one the chart keeps an invisible **band**, roughly as wide as the curve you asked
for — § 6.3 says exactly — so a click anywhere inside it means *that* mode:

```
        the sticks               what you may click
        ─────────                ──────────────────
             ┃                     ▒▒▒▒▒┃▒▒▒▒▒
             ┃                     ▒▒▒▒▒┃▒▒▒▒▒        ← invisible; the
        ─────┸─────                ▒▒▒▒▒┸▒▒▒▒▒          user just finds
                                                        the peak easy to hit
```

So: a user opens a result, sees where the modes are and how strong they are,
broadens them until it looks like the spectrum they would measure, clicks at a
peak, and the rest of the tab follows to that mode.

**Drawing it honestly means one thing in particular:** nothing is invented. A mode
whose strength has not been computed yet is shown as *not computed*, never as
*weak* (§ 6.2).

---

## 2. What SpectrumChart is not

Each boundary is here because something on the other side of it would plausibly
drift across.

| Not | Whose job it is | Why the boundary is there |
|---|---|---|
| a **mode viewer** | VibrationView | it draws no molecule and holds no geometry. It knows a mode by its index and its frequency, and it never learns where the atoms are |
| a **spectroscopy calculator** | the server, and the Spectrum tab | what a Raman activity is, which normalization produced it, what temperature was assumed. SpectrumChart is handed numbers and draws them |
| a **mode list** | the tab | the sortable table, the filter box, the CSV export. A chart shows a distribution; a table answers "what is mode 22?" — two jobs, two surfaces, and the tab owns the one made of rows |
| the **selection itself** | the tab | it reports a click and is *told* what is selected. Two components that each believed they owned the selection is how the highlight and the viewer drift apart |
| a **control panel** | the tab | the broadening box, the mode filter, the export buttons. SpectrumChart draws no controls (§ 5.3) |

---

## 3. The overall shape

```mermaid
flowchart TB
    subgraph outside["Outside SpectrumChart"]
      TAB["the Spectrum tab<br/>the mode table, the broadening box,<br/>the viewer and the ES panel"]
    end
    subgraph sc["One chart — everything below belongs to one owner"]
      H["the handle<br/>the modes, the selection,<br/>the broadening, and every draw"]
      M["the maths<br/>the envelope, and how wide<br/>a click band may be — pure"]
      SEAL["the sealed layer<br/>the only code that names Plotly"]
    end
    TAB -->|"holds"| H
    TAB -->|"mode 22, please"| H
    H -->|"the user clicked mode 22"| TAB
    H -->|"what does the envelope look like?"| M
    H -->|"draw this picture"| SEAL
    SEAL -->|"a click, at this x"| H
```

Four things to read off it:

**A tab owns none of it.** A tab has a handle. It does not have the plotting
surface, the marks on it, or a way to reach Plotly.

**The selection is a loop through the tab, not a shortcut inside the chart.** A
click goes *out* as an event and comes *back* as `setSelected`. The chart never
decides that a click changed the selection — because the table, the viewer and
the electronic-structure panel must move together, and only the tab can say when
they have. A chart that highlighted its own stick first would be right most of the
time and out of step exactly when a selection is refused.

**The maths knows nothing about drawing.** It answers two questions — what curve
does this broadening make, and how wide may each band be — with numbers, for
anyone who asks.

**Nothing flows back up past the seal** except a click. The sealed layer is never
asked what is drawn, what the axes are, or where a point landed on screen (§ 8.4).

---

## 4. SpectrumChart is a self-contained module

**SpectrumChart is one module — one folder the browser loads by name — and it is
sealed at every edge.** It is imported by name, it reaches nothing else in the app
by name, and nothing in the app can reach inside it.

**Why this module in particular has to say so.** The code it replaces lives inside
a 3,671-line tab controller where the drawing, the selection, the mode table and
the watch poller all reach into one another's variables — which is why selecting a
mode currently redraws the whole spectrum (§ 5.1). Cutting 345 lines out of that
is only worth doing if the cut holds, and a cut holds because the boundary is
written down and tested, not because the files were moved to a new folder.

**One entry point, and nothing else is importable.**

```js
import { mount } from "/static/lib/spectrumchart/index.js";
```

That is the whole surface. Every other file in the module is internal — the
maths, the sealed layer, the stylesheet. A consumer that imports one of them
directly has broken the module, not found a shortcut.

**A folder is not a seal, so the boundary is made three ways:**

- **The entry point exports one name.** Importing `index.js` hands back `mount`,
  and there is nothing else on it to find.
- **Every internal file is underscore-prefixed** — `_maths.js`, `_seal.js`. An
  import of `_maths.js` reads as a violation where it is written.
- **A guard test asserts the boundary** — nothing outside the package may name any
  path inside it but `index.js`.

**Nothing it needs comes from the app's globals, and nothing it holds is published
to one.** It does not read `window.molbuilder` and it does not write to it. A chart
needs one thing at mount: an element to live in.

**Nothing leaks out.** No Plotly object, no DOM node, and no internal function ever
appears on the handle. Inside this module the name `Plotly` occurs in exactly one
file.

**And the module brings what it needs.** Nothing else on the page loads Plotly and
no template links a stylesheet: the sealed layer adds both at mount, once (§ 11).
A dependency the host has to remember is a dependency some host will forget, and
the result — an empty box or an unstyled one, no error, far from its cause — is
found only by someone looking at the screen. If the library will not load, the
mount fails with a message (§ 7). And if it arrives as a script that publishes
`window.Plotly`, that name is read in the one file already allowed to say it and
nowhere else: no page put it there, and no page can rely on it being there.

**No special door for tests.** Tests import the module and drive the handle
exactly as a page does (§ 12): there is nothing a test can reach that a page
cannot.

**The test of all of this:** delete every other web module and SpectrumChart still
loads, mounts, draws and reports clicks.

---

## 5. The ideas everything else follows from

Four of them. A design choice that breaks one is wrong no matter how convenient.

### 5.1 The door says the cost

Opening a different result and clicking a different stick are not the same amount
of work, so they are not the same call:

```
   setModes()      rare — you opened a different result
                   → redraw every mark, recompute the curve,
                     and fit both axes to the new numbers

   setSelected()   often — you clicked another stick, or another table row
                   → recolour one mark, and nothing else
                   → no curve, no axis change, no refit
```

**This is the rule the current code breaks, and the measurement is why the rule is
here.** Today one function draws everything, and selecting a mode calls it: on the
benzene-dithiol result, at the default broadening, a single click recomputes an
806-point envelope from 36 Lorentzians — **29,016 terms** — to change the colour of
one stick. The waste is not the point; the *coupling* is. A chart that cannot
recolour without rebuilding has no way to be told a cheap thing, so every caller
pays for the expensive one and nobody can tell which they asked for.

### 5.2 One place holds each fact

The modes, the selection, the broadening: each has one home inside one chart,
written through one door. Two homes for a fact means a mechanism to keep them in
step, and that mechanism is where they fall out of step.

The **selection lives in the tab** and is mirrored here for drawing — one
direction only, so "mirrored" stays true (§ 3).

**The broadening is the same shape**: the tab owns the control the user types
into, and the chart holds the width it was last told. The mirror is one-way for
the same reason — a chart that changed the width itself would be a second author
of a number the tab displays.

### 5.3 The controls are not ours

SpectrumChart draws **no controls** — no heading, no broadening box, no export
button. It fills the element it was given with a spectrum. The drawing library
brings a small toolbar of its own (zoom, reset, save); that is the library's, not
a control this module offers, and what appears on it is decided in the sealed
layer and nowhere else.

### 5.4 The host owns the box

The host sizes the element and the chart fills whatever it is given: **the box
states its size, the module never sets one.** And the module watches *its own box*
rather than the window — a panel opening beside it, a sidebar collapsing, a tab
becoming visible: each changes the box while the window sits perfectly still.

Three separate bugs came from breaking this rule, all on 2026-08-05. They are
written down where they can be enforced, as the box rules in § 11.

---

## 6. The data a chart holds

### 6.1 The one shape a caller must know

Everything the chart draws comes from a list of these, and nothing else:

```js
mode = {
    index:     22,          // required — the caller's own number for this mode
    freq:      1131.8,      // required — cm⁻¹, may be negative
    activity:  12.86,       // optional — Å⁴/amu; null or absent = not computed
    imaginary: false,       // optional — default false
}
```

| Field | Required | Unit | Absent means | Refused when |
|---|---|---|---|---|
| `index` | **yes** | — | *(the record is refused)* | missing, not a finite number, or repeated within the list |
| `freq` | **yes** | cm⁻¹ | *(the record is refused)* | missing, or not a finite number |
| `activity` | no | Å⁴/amu | **not computed yet** — drawn as pending, never as zero (§ 6.2) | present but not a finite number |
| `imaginary` | no | — | `false` | never — it is read as yes-or-no, so anything that is not `false`, `0`, `null` or missing counts as yes |

*A finite number* above means an ordinary number — not text, not `NaN`, not
infinity. A caller that hands over `"1131.8"` where a frequency belongs has a bug
one line earlier, and the chart says so rather than drawing at a place it made up.

**A record that must be refused takes the whole call with it.** `setModes` draws
the list it was given or it draws none of it; it never quietly skips the bad rows
and renders a spectrum missing modes without saying so. A caller with one
malformed record has a bug one line earlier, and a chart that hides it is a chart
that helped.

**And a refused list leaves an empty chart, not the last one.** The old spectrum
staying on screen is exactly the hiding this rule is against: the caller sees
something plausible and never learns the call did nothing. So the chart empties —
visible, on screen — and the reason for the refusal goes to the console, where a
developer will look, because a caller's bug is not something a user can act on.

**`index` is the caller's numbering, carried and handed back untouched.** The
chart does not renumber and does not sort — it draws the modes in whatever order
they arrive and hands back exactly the number it was given. So a caller may use
1-based mode numbers, row numbers, or any other finite number it can recognise
later; the chart has no opinion about what the number means, because it means
something only to whoever wrote it.

**But the numbers must differ from each other.** A list carrying the same index
twice is refused whole. `setSelected(22)` names one mode and a click hands one
number back; with two modes numbered 22 neither sentence has an answer, and the
chart would have to invent a tie-break no caller could predict.

**Nothing else travels.** No structure, no eigenvector, no run metadata, no
label, no colour. If a future need seems to want a fifth field, the question to
ask first is whether it is a fact about *the picture* or about *the science* —
the second belongs to the tab, and this list is how the boundary in § 2 stays
real rather than aspirational.

### 6.2 Two pictures, decided by the data, not by a setting

A run computes frequencies before it computes intensities, so a result can
legitimately arrive with no activities at all:

```
  SOME STRENGTHS KNOWN                  NONE KNOWN YET
  height means strength                 every stick is height one,
                                        and the picture says why

      │      ┃                              │  ┃ ┃    ┃  ┃ ┃   ┃
      │      ┃   ┃                          │  ┃ ┃    ┃  ┃ ┃   ┃
      │  ┃   ┃   ┃                          │  ┃ ┃    ┃  ┃ ┃   ┃
      │  ┃   ┃   ┃  ×   ×                   │  ┃ ┃    ┃  ┃ ┃   ┃
      └──┸───┸───┸──────────▶               └──┸─┸────┸──┸─┸───┸──▶
                 ↑                           "strengths not computed"
          × = computed as zero?
              No — NOT COMPUTED, and
              marked so it cannot be
              mistaken for silent
```

**The right-hand picture is not a chart with nothing in it.** Every stick stands
at height one, and the curve over them is the sum of those — so the picture
answers *where are the modes, and where do they crowd together*, which is a real
question and the only one the data can answer yet (§ 9).

**Both markings are drawn in the plot**, as the pictures show: the `×` on the
axis, the words along the top. Neither is a label beside the chart or a banner
above it — a chart made of one drawing surface stays a chart made of one drawing
surface, and § 5.3's "no controls" holds without an exception carved for it.

**Which picture you get is derived, never configured.** A flag would be a second
place to believe something the data already says, and the failure mode is a flat
axis with nothing on it and no explanation.

### 6.3 A click lands on a band, and the nearest band wins

Each mode gets an invisible band centred on it, and its half-width is the
**broadening width** — *the same number that draws the envelope*, so the region a
user aims at is the region they see. A tolerance set independently of the picture
would be a second fact about the same thing (§ 5.2), and it drifts the moment
either is changed alone.

**One thing bends that, and only upward.** Below **8 cm⁻¹** the band is wider
than the curve, because a target that narrow cannot be hit with a mouse. What
never happens is the picture being changed to match the band: the curve is the
claim about the science, the band is only about aiming, and where they must
differ it is the invisible one that gives way.

**Bands overlap where modes crowd, and a click takes the nearest one.** That is
the whole rule. "The nearest" is an answer that never needs a tie-break, so
nothing has to be moved or shrunk to keep it unambiguous.

> **This replaced a clamp, and the reason is worth keeping.** Each band used to
> be cut back to half the distance to its nearer neighbour so that no two could
> ever overlap. That was necessary while a click had to land on a *drawn mark* —
> two marks over the same spot meant the answer depended on which was drawn last.
> Reading a click as a position (§ 8.4) made "nearest" available, and the clamp
> then cost exactly what it was meant to protect: in a crowded region it shrank
> targets to a fraction of a pixel, so some peaks were easy to click and others
> could not be hit at all. A rule that survives a change of mechanism should be
> re-argued, not inherited.

**Two modes at the same frequency** cannot be told apart by any band. Both keep
theirs, so both stay reachable; which of the two a click reports is undefined —
they are the same distance away — and separating them is the table's job.

### 6.3.1 The band, made visible

An invisible target is a guess. So **the mode a click would pick lights up as the
pointer moves**, which is the band shown without drawing it: slide across a
crowded region and the answer changes under you, and what you see is what you
would get.

Two rules keep it honest and cheap:

- **What you picked outranks what you are pointing at.** A chosen mode keeps its
  colour under the pointer, or a selection would appear to flicker away while you
  reach for something else.
- **Nothing is redrawn until the answer changes**, and then only the colours
  (§ 5.1). A pointer crossing the plot reports hundreds of times a second;
  sliding along inside one band costs nothing, and hovering the mode that is
  already chosen costs nothing either, because the picture would come out the
  same.

### 6.4 An imaginary mode is drawn, marked, and left out of the curve

A mode can come back **imaginary**. That is not a very slow vibration: it means
the structure is not sitting at rest, and the "vibration" is the direction it
would fall away in. No measured spectrum contains such a line.

So the chart does three separate things with one, and each is its own sentence of
the contract:

- **It is drawn**, at the frequency the caller gave, and it is **clickable like
  any other mode** — an imaginary mode is very often the one the user opened the
  result to look at.
- **It is marked apart**, so it can never be read as an ordinary weak line. *How*
  it is marked is the seal's business (§ 8.4); *that* it is marked is this
  contract's.
- **It never joins the envelope** (§ 9), because the envelope is what a
  measurement would look like and no measurement contains this.

**`imaginary` and a negative `freq` are two separate facts, and neither implies
the other.** Runs differ over whether they report these frequencies as negative
numbers. Which convention produced a number is science, and science belongs to the
tab (§ 2): the chart draws the number it is handed and marks the mode the flag
names, and it never infers one from the other.

---

## 7. Making and tearing down a chart

```js
const chart = await mount(hostEl, { onSelect: (index) => …, broadening: 20 });
```

The options are written out in § 8.2; this section is about what `mount` *gives
back* and how it ends.

**`mount` is asynchronous and always resolves.** The handle it returns is live:
the surface is built, and every door works from the first call. There is no
readiness flag — a chart that is not ready yet is a state a caller can get wrong,
so it is not offered.

**Failure is a handle, never a rejection and never `null`:**

```js
{ ok: false, error: "…", dispose() {} }
```

so a caller branches on `ok` and can call `dispose()` unconditionally. A live
handle carries `ok: true`. Same mount contract as MolView and VibrationView, for
the same reason: teardown must never have to ask whether setup worked.

**A failed handle carries those three keys and no others** — no inert `setModes`,
no `setSelected` that quietly does nothing. A caller that skipped the `ok` branch
then fails at its first door, loudly and at the line that made the wrong
assumption, instead of feeding data to a chart that was never there.

**What that costs a caller is one line, and the shape is worth naming:** keep the
handle only if it is live — `this.chart = chart.ok ? chart : null` — and reach it
with `?.` from then on (§ 10 shows it). The alternative, doors that accept every
call and do nothing, would spread the same branch invisibly across the module and
turn a dead chart into a silent one.

**A library that will not load is a failed mount, not a broken page.** The module
fetches Plotly itself (§ 4); if that fails — offline, blocked, the file not
there — the mount resolves with `ok: false` and a message the tab can show. The
mode table beside it still works, which is the behaviour the current code has and
is worth keeping.

**The module owns the inside of its host, and nothing else about it.** `mount`
empties the element it is given before it draws, so whatever was in there — a
placeholder, a spinner, a chart from a previous result — is gone. The element
itself, its size and its place on the page remain the host's (§ 5.4).

`dispose()` releases the drawing surface, disconnects the box watcher and leaves
the host element empty. Calling it twice is safe.

---

## 8. The APIs, and who each one serves

### 8.1 The entry point

```js
import { mount } from "/static/lib/spectrumchart/index.js";
```

One name. § 4.

**And that name is all a page needs.** Here is a complete, standalone use — no
framework, no tab, nothing else from this app:

```html
<div id="chart" style="width: 640px; height: 320px"></div>

<script type="module">
import { mount } from "/static/lib/spectrumchart/index.js";

const chart = await mount(document.getElementById("chart"), {
    onSelect: (index) => console.log("the user picked mode", index),
});

if (chart.ok) {                   // the one branch every caller makes (§ 7)
    chart.setModes([
        { index: 1, freq:  323.5, activity:  2.9, imaginary: false },
        { index: 2, freq:  826.6, activity:  5.4, imaginary: false },
        { index: 3, freq: 1131.8, activity: 12.9, imaginary: false },
        { index: 4, freq: 3175.4, activity:  0.0, imaginary: false },
    ]);
    chart.setBroadening(20);      // draw the curve 20 cm⁻¹ wide
    chart.setSelected(3);         // colour mode 3 as the chosen one
}
</script>
```

That is the whole surface: **five doors and one callback.** The page above styles
a box, hands over four numbers per mode, and gets a spectrum it can click. There
is no stylesheet `<link>` and no library `<script>` — the module brings both
(§ 4) — so this really is the whole page.

### 8.2 What `mount` takes

```js
mount(host, options) -> Promise<handle>
```

| What is passed | What it is | Required | Default |
|---|---|---|---|
| `host` | the element the chart lives in. The **host sizes it** (§ 5.4); the module never writes a width or a height onto it | yes | — |
| `options.onSelect` | `(index) => void`, called when a **user** clicks a mode. Never called by anything the tab does (§ 8.3) | no | nothing is reported |
| `options.modes` | the first mode list, exactly as `setModes` takes it | no | empty — an empty chart, not an error |
| `options.selected` | the first selection, exactly as `setSelected` takes it | no | none chosen |
| `options.broadening` | line width in cm⁻¹, exactly as `setBroadening` takes it | no | `0` — bare sticks, no curve |

**`mount` refuses exactly one thing: a host that is not an element.** Everything
else it is given it can work with, and a missing or wrong host is the one mistake
that leaves it nowhere to draw. It refuses the way every failure here is reported
— a handle with `ok: false` and a message (§ 7), never a throw.

**A mount option is the first write through the door of the same name, not a
second way to hold the fact.** `mount({ broadening: 20 })` and
`setBroadening(20)` reach the same one place; there is no separate "initial"
value kept anywhere, so nothing can disagree with anything (§ 5.2).

Everything else about the chart — its size, its position, when it is visible — is
the host's, and is expressed in CSS rather than passed here.

### 8.3 The handle — the only way data goes in

```
setModes(list)               a different result: redraw, refit the axes (§ 5.1)
setSelected(index | null)    recolour one mark — nothing else moves
setBroadening(width)         how wide to draw each line, in cm⁻¹ — and therefore
                             how wide it is to click (§ 6.3)
refit()                      re-measure the box and redraw at its size
dispose()                    release, disconnect, empty
```

Five doors, and **every fact the chart holds arrives through exactly one of
them.** There is no property to assign, no options object to mutate after the
fact, no global to publish into, and no second path that "also" sets something.
Data goes in one way; nothing comes back out but a click.

**Each door, precisely:**

| Door | Takes | Refuses | Returns |
|---|---|---|---|
| `setModes` | an array of mode records (§ 6.1). An empty array is legal and means *an empty chart* | anything that is not an array; a record missing `index` or `freq`; the same `index` twice — the whole call, and **the chart empties** rather than leaving the last spectrum standing (§ 6.1) | nothing |
| `setSelected` | one `index`, or `null` for *nothing chosen*. It is **remembered whether or not the current list holds it**, and drawn as soon as a list does | nothing — an index no list holds is not an error (below) | nothing |
| `setBroadening` | a width in cm⁻¹, `≥ 0`. Zero means *no curve*: bare sticks, and click bands at their floor (§ 6.3) | a negative number or a value that is not a number — the width already set stands, because substituting a default would hide a caller's bug | nothing |
| `refit` | — | — | nothing |
| `dispose` | — | — | nothing; safe to call twice |

**Order does not matter.** Any door may be called before any other, any number of
times, in any order: selecting before there are modes, broadening an empty chart,
re-selecting the same index. Each call writes one fact and redraws what that fact
affects. A caller that has to remember a sequence is a caller that will get it
wrong.

**Which is why `setSelected` records rather than refuses, and the highlight is
derived from what it recorded.** An index the current list does not hold
highlights nothing; if a later `setModes` brings that index in, the highlight
appears without the tab having to say it again.

> A mirror that refused would be worse than one that shows nothing. The tab does
> `state.selected = 3` and then `chart.setSelected(3)`: a refusal leaves the tab
> believing 3 is selected while the chart still highlights 7 — the two out of step,
> which is the exact drift § 5.2 exists to prevent. And the mistake a refusal was
> meant to catch, choosing a mode that is not there, still shows: as no highlight.

**Nothing is read back.** There is deliberately no `getModes`, no `getSelected`,
no `getBroadening` and no accessor of any kind: every one of those values was
written by the caller, so a second copy inside the module would be a second place
to believe something (§ 5.2). What the module knows that the caller does not is
*where the marks landed on screen*, and that is the seal's business and never
leaves it (§ 8.4).

**`onSelect` is given once, at mount.** It is how a click leaves (§ 3). A chart
with no `onSelect` draws normally and reports nothing — a legitimate thing to be,
for a figure nobody is meant to click.

**`setSelected` never emits `onSelect`.** The event means *a user clicked*, not
*the selection changed*; a caller that wires the two together in the obvious way
would otherwise get an endless round trip. Nothing the tab does to the chart ever
comes back out of it.

**A click that lands in no band reports nothing at all.** `onSelect` is not
called, and nothing on screen changes. It does not report `null`: the event means
*the user picked a mode*, and there is no mode to name — a `null` would oblige
every caller to handle a case whose meaning is "nothing happened". Whether an
empty click should clear the selection is a decision about the whole tab, and the
tab is where the selection lives (§ 5.2).

**`refit` exists because of a rule, not because of a library's behaviour.**

> **The module never draws into a box it cannot measure**, and it cannot know on
> its own when an unmeasurable box becomes real.

A chart in a hidden tab has no box. The module watches its own box (§ 5.4), and
that covers a box that *changes size*; what it cannot be relied on to cover is a
box that did not exist. `refit` is how the host says **the box is real now**.

Justifying this door by what the browser's box-watching machinery happens to do
would put somebody else's behaviour inside this contract, where it cannot be
tested and cannot be relied on.

### 8.4 The sealed layer — commands down, clicks up

```
draw(picture)                   put this picture on the surface
recolour(marks)                 change the colours already drawn — no rebuild
resize()                        the box changed; fill it
onClick(cb)                     the user clicked: hand up where it landed on
                                the frequency axis, and nothing else
onHover(cb)                     the pointer moved: the same number, continuously,
                                or null over nowhere in particular
purge()                         release the surface
```

**A picture is what goes down.** The handle hands the seal a plain object, and
nothing in it is a colour or a mode:

```js
picture = {
    sticks: { x: [...], y: [...], width: [...],  // where and how tall each is drawn
              state: ["plain" | "chosen" | "hovered" | "pending" | "imaginary", …] },
    curve:  { x: [...], y: [...] } | null,       // null when there is no curve
    xTitle, yTitle, xUnit,                        // the words on the axes
    note,                                         // one line inside the plot, or nothing
}
```

**Nothing in the picture says where a click may land**, and that is deliberate:
the bands are not drawn and not handed down. § 8.4 says where they are used
instead.

**A state is a word, never a colour** — the layer above says a mark is *chosen*
and the seal decides what chosen looks like (§ 5.1's cheap door hands down the
same words and nothing else). **And every word the user reads comes down in the
picture too**: the axis titles, the unit beside a hovered value, the "not
computed" line. The seal writes none of them, because "cm⁻¹" is a fact about
spectroscopy and this layer knows none.

It answers **no** question about what is drawn, and it does not know what a mode
is. **A click comes up as one number** — a position on the frequency axis — and
the handle turns that into a mode through the bands (§ 6.3).

That is deliberately the only mechanism. A drawn mark carrying its own label back
would be the obvious alternative and it cannot work here: the click a band exists
to catch lands in *empty space* beside a peak, where there is no mark to have
carried anything. So the seal reports a position, which is a number, and stays as
ignorant of chemistry as before — "mode" is a word this layer never learns.

What a selected stick *looks like* is decided here too. The layer above says
*which mark* is the chosen one, never what colour it should be: a colour riding on
data is how a drawing decision ends up in three places.

**The palette is read from the stylesheet, not written in code.** Plotly takes
colours as JavaScript values, so a chart cannot inherit them the way an element
does — which is precisely how two tab controllers ended up carrying private copies
of the same nine hex literals. The sealed layer reads **this module's own tokens**
(§ 11) off the module's own outermost element and hands the library the answer,
so the sheet stays the one source of truth for how the chart looks.

**`recolour` is the cheap door § 5.1 is about.** It exists so that the expensive
one does not have to be called for a click.

**A click, and the pointer, are read off the surface rather than off a mark.** The drawing library reports
a click only when the pointer is over one of *its own* points — so nothing it
offers can hear a click in the empty space beside a peak, and that space is the
whole purpose of a band. So the seal takes the click from the surface itself and
converts it: where the pointer was, across the plot area, read against the axis
range. That conversion is the one place in this module that reads inside the
library, and this is the file whose job is to know it.

> **Why not draw the bands as invisible marks instead?** Because then the *seal*
> would be deciding which mode you clicked — the mark it reports would already be
> the answer — and the handle's band lookup would be a rubber stamp. The division
> this section states would be true on paper and false in the code. A raw
> position keeps it true: the seal knows where, the handle knows which.

**The seal takes the library as it finds it, and never assumes it is ready.**
Two consequences, both found by building against this contract rather than by
reading it:

- **A page that already carries the library is used as it stands** — the Results
  page loads Plotly for other figures — and only a page without it is served the
  fetch. Either way one page holds one copy and one stylesheet link, however many
  charts are mounted on it.
- **A click asked for before the first draw must still arrive**, because § 8.3
  promises any door may be called in any order. Listening on the surface rather
  than on the library's own event machinery is what makes that free: the surface
  exists from the moment the chart is mounted, drawn on or not.

---

## 9. How a spectrum gets drawn

The sticks are the data. The envelope is a sum of Lorentzians — the narrow peak
with long tails that a real spectral line has — one per mode with an activity:

```
                              γ²
     y(x)  =  Σ   A_i  ·  ───────────────      γ = half the width you asked for
              i           (x − x_i)² + γ²
```

so each mode contributes a peak of height `A_i` at `x_i`, and the sum is what a
measured spectrum looks like when lines have width.

**Where the curve is sampled follows the width, not a fixed setting.** The curve
is worked out at a list of frequencies — the grid — and a fixed grid is either too
coarse for a narrow line or wasteful for a broad one; the width is the only number
that knows which. Two requirements say how far that goes, and
they are stated as accuracy rather than as constants so a test can check the
result instead of the arithmetic:

- **A peak is smooth** — at least **eight samples across the full width** of the
  narrowest peak drawn. Fewer and a Lorentzian shows its corners.
- **The curve ends, rather than being cut off** — the grid extends past the
  outermost mode until the envelope has fallen below **1% of the tallest peak**.

Both are consequences of the width alone, so neither is a number anyone tunes.

**And the grid is a window around each mode, not one ruler laid end to end.**
Both draw the same curve where a curve is worth drawing; the difference is what
they cost. One ruler at a sharp broadening spends nearly all its points on empty
space between modes — at 0.05 cm⁻¹ across a 3,000 cm⁻¹ spectrum that is millions
of points and the browser stops — while windows cost about eighty points per
mode at *every* width. The consequence worth stating: between two distant modes
the curve is drawn as a straight line across a region where it is already near
zero. That is a sampling choice, and it is the one place this module draws
something it did not compute.

**What the sum runs over follows the picture you are in** (§ 6.2), because the sum
has to mean the same thing the stick heights mean:

- **Strengths known** — the sum runs over the modes that have one. A mode whose
  strength has not been computed contributes **nothing**, not a zero-height peak:
  it is missing, not weak.
- **None known** — every mode contributes a peak of height one, and the curve is
  a genuine frequency distribution: where it is high, modes are crowded. Drawing
  unit sticks with no curve over them would leave exactly the barcode § 1 says
  this component exists to avoid.

**No imaginary mode is ever in the sum**, in either picture (§ 6.4).

---

## 10. Who drives it

The Spectrum tab. It hands the chart a result, tells it what is selected, and
listens for clicks:

```js
const chart = await mount(host, { onSelect: (index) => tab.selectMode(index) });
if (!chart.ok) showMessage(chart.error);   // the mode table still works without it
tab.chart = chart.ok ? chart : null;       // a dead handle is not kept (§ 7)

// a result arrives
tab.chart?.setModes(modes);      // `?.` = do nothing if there is no chart

// the user clicks a stick, or a table row, or arrows through the list
tab.selectMode = (index) => {
    state.selected = index;          // ONE home for the fact (§ 5.2)
    tab.chart?.setSelected(index);   // the chart mirrors it
    viewer.showMode(…);              // and so does everything else
    renderEsPanel();
};
```

Read the loop: a click enters the tab and comes back to the chart as an
instruction. The chart never highlights on its own, so the highlight cannot
disagree with the viewer (§ 3).

---

## 11. The module's own stylesheet

The module ships one stylesheet and links it itself at mount, once — the drawing
library arrives the same way and for the same reason (§ 4), so a host provides one
thing: an element. `mount` waits for the sheet to load, because the sealed layer
reads the chart's colours out of it (§ 8.4) and a stylesheet loads asynchronously;
appending it and carrying on would give the first chart fallback colours,
invisibly, for exactly as long as the two agreed.

Five rules, and they are the ones peculiar to *this* module:

- **Its names are its own.** Every class starts `spectrumchart-`. Nothing else on
  the page styles those names, and this sheet styles nothing else — the same
  boundary as `mount`, in another language.
- **Its values sit in one block at the top** — the colours, the spacing, the
  type — and every rule below reads them by name. Changing how the chart looks is
  changing that block, not hunting through the rules.
- **It inherits nothing from the page.** Those values are the module's own, so
  deleting every page stylesheet leaves the chart looking right. It matches the
  app because its values were *chosen* to, not because it reads the app's — and
  the app has one palette, with no light/dark switch to follow. If that ever
  changes, what changes here is this one block.
- **The frame states its size; the surface fills it, carries no padding, and
  never spills past it.** All three come from real breakage on one day: a height
  that came from the plot being sized to it collapsed to a ten-pixel strip; a
  padded surface drew wider than its frame, because the width the library asks for
  counts padding in; and the drawing then overflowed across the panel beside it.
- **It reacts to its own box, never to the window.** The chart can sit in a
  full-width panel, a half-width tab, or beside a sidebar that collapses; all
  change the box while the window sits still. There are no size rules in the sheet
  at all — one surface filling one frame has nothing to re-arrange, so responding
  is one thing: redraw at the box's new size.

**The repository's CSS conventions apply here as they do everywhere else** — one
class per rule, no ids, no `!important`, no value written twice — and they are
*not* restated in this contract. They live in
[`css-system-plan.md`](?doc=web/css-system-plan.md), which is also where the
page-versus-module boundary is argued. A rule that would be true of any stylesheet
in this repository does not belong in the design of a spectrum chart.

The sheet stays a real `.css` file rather than a string inside JavaScript, so the
repository's existing CSS audits can read it.

---

## 12. How the tests are designed

**Every test is derived from this document, never from the source.** A test that
reads the implementation to build its assertion can only confirm that the code
still says what it said.

| Level | Runs | Derived from | Shows |
|---|---|---|---|
| **Behaviour, no browser** | node | § 6, § 9 | the envelope and the band widths — values in, values out |
| **Boundary behaviour** | node, with a stand-in that obeys § 8.4 | § 5, § 8 | what each door costs, and that nothing above the seal names the library or a colour |
| **End to end** | a real page | § 1, § 6.3 | clicking beside a peak selects that mode; clicking in a gap reports nothing |

**The stand-in replaces the drawing library, not a file of this module.** Nothing
inside the module is swappable, and no door exists for a test that does not exist
for a page (§ 4). The stand-in stands where Plotly stands; the module cannot tell
the difference, because § 8.4 is the whole of what it asks of that library.

### What each rule obliges a test to show

**A rule with no row here is a rule nothing guards.**

| The rule | A test must show |
|---|---|
| § 2 — the tab keeps its own | the module contains no table, no filter, no CSV, no poller, and no form |
| § 4 — self-contained | the module mounts given only a host element; it reads no global of the app's and writes none, and the only global name it reads at all is the library's, in the seal |
| § 4 — nothing else is importable | the entry point exports exactly `mount` |
| § 4 — no hidden way in | list everything on the handle, hiding nothing: not one entry leads to the drawing surface or to the module's own state |
| § 4 — the module brings what it needs | a page holding only a host element and one `import` draws a **styled** chart: no `<script>` and no `<link>` of its own, and mounting twice adds one link, not two |
| § 4 — the library name appears once | `Plotly` occurs in exactly one file of the module, and no file above it names the library at all |
| § 5.1 — the door says the cost | `setSelected` computes no curve and changes no axis; `setModes` does both |
| § 5.1 — recolouring is not redrawing | after `setSelected`, the picture handed to the seal is the same picture, with only its colours changed |
| § 5.2 — the selection is mirrored, not owned | a click emits `onSelect` and changes **nothing** on screen until `setSelected` is called; a chart mounted without `onSelect` still draws |
| § 5.3 — no controls | a mounted chart contains no heading, button, input or control of any kind |
| § 5.4 — the host owns the box | the module writes no width or height onto its host — not at mount, not at any door — and a box that changes size redraws while the window sits still |
| § 6.1 — the two required fields | a record without `index`, or without `freq`, is refused |
| § 6.1 — a refusal takes the whole call | one malformed record among many draws **nothing**, rather than a spectrum quietly missing that mode |
| § 6.1 — a refused list empties the chart | after a refused `setModes`, the previous spectrum is gone from the screen rather than left standing as though the call had worked |
| § 6.1 — absent is not zero | a mode with no `activity` field, and one with `activity: null`, are treated the same and neither is drawn as a strength of zero |
| § 6.1 — the caller's numbering is carried, not interpreted | an unsorted, sparse, 1-based list draws in the order given, and the index a click reports is the index that was handed in |
| § 6.1 — the indices must differ | a list containing the same index twice is refused whole, rather than resolved by a rule the caller cannot see |
| § 6.2 — the picture is derived | a result with no activities anywhere draws every stick at height one, with a curve over them; one with some activities draws the rest as pending — and neither picture is reachable by a setting |
| § 6.2 — the markings are in the plot | the pending marks and the "not computed" wording are drawn on the surface: the module's markup carries no label, banner or text node beside the chart |
| § 6.3 — one width, two uses | changing the broadening changes both the envelope and the band widths, and there is no way to set either alone |
| § 6.3 — every mode keeps a full-size target | modes 1 cm⁻¹ apart each keep the whole band; no arrangement of neighbours shrinks one |
| § 6.3 — the floor applies below it | at any broadening under 8 cm⁻¹, zero included, a band is 8 cm⁻¹ half-wide |
| § 6.3 — the nearest band wins | where bands overlap, a click reports the mode whose frequency is closest to it |
| § 6.3 — the band bends, never the picture | where the floor overrides, the envelope drawn is still the envelope for the width that was set |
| § 6.3.1 — the pointer shows what a click would take | moving over a band recolours that mode, and only it |
| § 6.3.1 — pointing costs nothing until the answer changes | sliding inside one band, and hovering the mode already chosen, each make no call at all |
| § 6.3.1 — what you picked does not flicker | a chosen mode keeps its colour under the pointer |
| § 8.4 — a drag is not a pick | a press and a release far apart report nothing; a click a pixel or two from the press still reports |
| § 8.4 — the toolbar and the labels are not the plot | a click above or below the plot area reports nothing, at any horizontal position |
| § 6.4 — an imaginary mode is drawn and clickable | it appears at its own frequency and a click on it reports its index, exactly like any other mode |
| § 6.4 — an imaginary mode is marked | it is drawn differently from an ordinary mode of the same height |
| § 6.4 — an imaginary mode is not in the curve | adding one to a list leaves the envelope unchanged, in both pictures |
| § 6.4 — the flag and the sign are independent | a negative `freq` without the flag is drawn as an ordinary mode; the flag with a positive `freq` is drawn as imaginary |
| § 7 — mount always resolves | a mount with no library still resolves with `ok === false` **and** a working `dispose`; nothing rejects and nothing returns null |
| § 7 — a failed handle has three keys | its keys are exactly `ok`, `error`, `dispose`: calling any other door throws rather than silently doing nothing |
| § 7 — the module owns the inside of its host | a host with content in it is empty of that content after mount, and empty again after `dispose`; the watcher stops (a resize after `dispose` draws nothing) and a second `dispose` neither acts nor throws |
| § 8.2 — a host that is not an element is the one refusal | `mount(null)` and `mount("#chart")` each resolve with `ok === false` and a message, and neither throws |
| § 8.2 — a mount option is the first write | `mount({ broadening: 20 })` and `mount()` then `setBroadening(20)` leave the chart in the same state, and no "initial" value survives a later write |
| § 8.3 — one way in | the handle exposes no settable property and no options object; changing a fact is possible only by calling its door |
| § 8.3 — nothing is read back | there is no `getModes`, `getSelected` or `getBroadening`, and nothing else hands those values back either |
| § 8.3 — order does not matter | selecting before any modes exist, broadening an empty chart, and re-selecting the same index each leave a coherent chart and no error |
| § 8.3 — the selection is recorded, not refused | `setSelected(3)` before any modes exist, followed by a `setModes` containing 3, draws 3 as chosen with no second call; an index no list ever holds highlights nothing and raises nothing |
| § 8.3 — a new list keeps the recorded selection | `setModes` with a list that still holds the selected index leaves it drawn as chosen; with a list that does not, nothing is drawn as chosen and the recorded value stands |
| § 8.3 — a bad width keeps the old one | a negative width, and a value that is not a number, each leave the width already set standing |
| § 8.3 — an empty list is a state, not a failure | `setModes([])` draws an empty chart and does not error |
| § 8.3 — an empty click is silent | a click landing in no band calls `onSelect` **not at all** — not with `null` — and changes nothing on screen |
| § 8.3 — refit redraws at the box's size | a chart whose box was zero-sized when drawn fills its box after `refit`, with no other call |
| § 8.4 — the seal faces downward | what is drawn, how it is laid out and what the axes span cannot be read out of it |
| § 8.4 — a picture carries words, never colours | the object handed down names states (`chosen`, `pending`, `imaginary`) and carries every axis title, unit and note; no colour and no mode index appears in it |
| § 8.4 — a click asked for before the first draw still arrives | `onClick` then `draw` reports a click, in that order as well as the other |
| § 8.4 — a click anywhere in the plot area is a position | a click over empty space, over the curve and over a stick each report the frequency under the pointer; a click in the margins reports nothing |
| § 8.4 — a click comes up as a position | the seal reports where the click landed on the frequency axis and nothing more; a click in empty space beside a peak still reaches the handle |
| § 8.4 — appearance is the seal's | nothing above the seal names a colour, and the palette the seal uses comes from the module's own values (§ 11) — never from a page variable, never from a literal in the module |
| § 9 — the envelope is the sum | the curve at a mode's frequency equals that mode's activity plus the tails of the others, computed independently of the implementation |
| § 9 — the sum follows the picture | with strengths known, a mode without one adds nothing; with none known anywhere, every mode adds a peak of height one |
| § 9 — a peak is smooth at every width | at any broadening, the narrowest peak drawn carries at least eight samples across its full width |
| § 9 — the grid is bounded by the modes, not by the width | a 36-mode spectrum at a 0.05 cm⁻¹ broadening draws from a grid of a few thousand points, not millions, and its peaks are still smooth |
| § 9 — the curve ends rather than being cut off | at any broadening, the envelope at both ends of the grid is below 1% of the tallest peak |
| § 11 — the sheet inherits nothing from the page | with every page stylesheet removed, the chart's own values still resolve and the colours handed to the library are unchanged; no page variable is named anywhere in the module |
| § 11 — the values are declared, not sampled | every colour the seal hands the library was read from a `--spectrumchart-` value on the module's own element |
| § 11 — the sheet stays this module's | every class it writes begins `spectrumchart-`, it styles the frame and the surface and nothing else, and its values are declared in one block rather than written into rules |
| § 11 — the box traps | the frame declares a height and never `auto`; the drawing surface carries no padding; nothing it draws spills outside the frame or scrolls the page sideways |
| § 11 — the chart responds to its own box | no rule in the sheet is keyed to the window's size, and a box that changes size while the window does not causes a redraw at the new size |
| § 11 — the mount waits for the sheet | the first chart on a cold page draws with its own colours, not with fallbacks |

---

## 13. The file map, and where the code stands

Everything below lives under `lib/spectrumchart/`. **It is built and the Spectrum
tab draws through it** (2026-08-05). This document was written before any of it
moved, so the door was reviewed before anything depended on it — and the build
then found five things prose could not: § 8.4's picture shape, how the library
arrives, that a click cannot be read off a mark, that the grid must be bounded by
the modes rather than the width, and that this app has no theme signal to follow.

| File | Holds | Status |
|---|---|---|
| `index.js` | the handle, the state, and every draw it causes | **built** |
| `_maths.js` | the envelope, the band widths — pure | **built**, from `spectra/core.js` (`_lorentzianEnvelope`, `_clickBandWidths`, `_clickTolerance`), **then satisfies rules those functions predate** |
| `_seal.js` | the only file naming Plotly — it fetches the library, links the sheet, and reads the palette from the tokens | **built**, moving (`chartTheme`, the `Plotly.react` / `purge` / `Plots.resize` calls, `_watchChartWidth`), plus the two loaders, which are new |
| `_style.css` | the module's own namespaced sheet | **built** |

**Which parts of this are new specification, not relocation** — worth knowing
before anyone sizes the work from the line counts below:

| New | Where |
|---|---|
| the four-step band rule, with the floor and the minimum as fixed numbers | § 6.3 |
| the sum following the picture — unit heights when no strength is known | § 9 |
| imaginary modes marked, and kept out of the envelope | § 6.4, § 9 |
| the grid stated as accuracy: eight samples a peak, ends below 1% | § 9 |
| the selection recorded rather than refused | § 8.3 |
| a refused list emptying the chart | § 6.1 |
| the module fetching its own library and sheet | § 4, § 11 |

**What came out of the tab: 287 lines**, leaving `lib/spectra/core.js` at 3,384.
`renderSpectrumChart` (242 lines),
`_lorentzianEnvelope` (37), `chartTheme` (22), `_watchChartWidth` (16),
`_clickBandWidths` (15), `_onChartClick` (7), `_clickTolerance` (6) — 345 lines of
`lib/spectra/core.js`, which is 3,671 lines today.

**What stays.** The mode table, the filter, the CSV export, the selection and its
consumers, the watch poller, the load path, the generator form, the
electronic-structure panel.

**What has been shown to work**, on the 36-mode benzene-dithiol result, in a real
browser rather than against a stand-in:

- the chart mounts, brings its own stylesheet, uses the Plotly the page already
  carries, and draws the sticks and the envelope;
- **a click 6 cm⁻¹ beside the 1131.8 peak — over empty space, on no mark —
  selects mode 22**, which is § 6.3's whole purpose and the one thing a stand-in
  could not have told us;
- a click in a gap between bands changes nothing;
- **a selection costs one recolour**, attributed per element: the spectrum is
  never rebuilt to move a highlight. That is § 5.1, and it is what the extraction
  was for — the old path recomputed an 806-point envelope from 36 Lorentzians,
  29,016 terms, for every click;
- the chart follows its own box: collapsing the panel beside it re-fits the
  drawing, with the window never moving (§ 5.4, confirmed on screen — a
  backgrounded tab gets no resize callbacks at all, so this one cannot be
  measured from a driven page).

**The electronic-structure level diagram is not part of this module** and is not a
module. It is a small figure belonging to the electronic-structure panel, with
different data and a different job; it stays where it is.

**Sequencing.** This extraction comes **before** the planned conversion of
`lib/spectra/core.js` into a module of its own (roadmap #102): converting first would convert the mess and
then re-cut it, while extracting first leaves #102 a smaller file with its hardest
part already sealed.

**No general charting layer.** `_seal.js` is this module's seal over Plotly, not a
shared abstraction waiting for consumers. If another module ever needs the same
seal it gets promoted then, with a real second caller rather than a guessed one.

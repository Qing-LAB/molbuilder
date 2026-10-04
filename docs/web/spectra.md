# Spectra — computing and reading a vibrational spectrum

**Role:** contract
**Domain:** web
**Companions:** [`engines/vibration.md`](?doc=engines/vibration.md) — **the
calculation this tab describes and displays**: the road, the parameters, the
two engines' routes, and the result file's contract (what every key is and
where it comes from); [`results.md`](?doc=web/results.md) — the Results-tab shell that
hosts the spectra *presenter*; [`presenters.md`](?doc=web/presenters.md) — the
registry that picks it for a `.spectra.json`; [`molview.md`](?doc=web/molview.md)
— the read-only 3D viewer the standalone tab uses to inspect the input structure;
[`web-api.md`](?doc=web/web-api.md) — the `/api/spectra/*` routes;
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) — the run that
produces the `.spectra.json`.

> **Migration status (2026-08-21).** The compute side of this surface is
> **framework-native**: the Send button hands over to Task setup, and the
> vibration deck is a calculation KIND of either engine
> ([`engines/vibration.md`](?doc=engines/vibration.md)), rendered by
> `render_deck` through the same gates as an optimization deck and run
> through `prep`/`launch`. (The old standalone-script path retired at the
> spectra migration's P3.) The *viewing* half — the presenter, the chart,
> the mode table — is unchanged.

The spectra surface computes a **vibrational spectrum** for a molecule —
frequencies and mode shapes, and on PySCF the infrared and Raman strengths
beside them ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.5) — and then shows it as an interactive chart: two stacked
panels of sticks over one frequency axis, a sortable table of the vibrational
modes, and — when you click a peak — a **3D animation of that normal mode**. You reach it two ways, but it is the same code
both times. **A result with no strength computed draws its mode
positions, not a spectrum**: every SIESTA result, whose route computes none,
and a PySCF run with infrared and Raman both off draw one line at each mode's
wavenumber — no heights, no broadening — which is still where a mode is
picked for its animation and its electronic structure, and one sentence says
why there are no heights (§ 2, § 9b.3).

## 1. Two surfaces, one engine

Everything below is drawn by a single module, `lib/spectra/core.js`. It is
mounted by two different pages, each a thin wrapper:

- **The standalone Spectrum tab** (`/spectrum-calculation`) — the
  *describe* half: inspect a structure, set parameters, and Send the
  description to Task setup (this tab renders no deck — § 5). It shows no
  result; the finished spectrum, or one still being computed, is read on the
  Results tab (§ 7). Its controller is `spectra/viewer.js`.
- **The Results-tab presenter** — the *view-only* half. When you open a
  `*.spectra.json` result on the Results tab, the presenter
  (`lib/inspectors/spectra.js`, registered as **"Spectra results"**) mounts the
  same engine to display it. It never shows the generate-side form.

Because both wrappers call into one engine, a fix to the chart or the mode table
lands on both pages at once.

```mermaid
flowchart TD
  ENG["lib/spectra/core.js — the whole spectra viewer<br/>form · chart · mode table · excited-state panel · 3D mode animation"]
  TAB["/spectrum-calculation<br/>(spectra/viewer.js — compute + view)"] --> ENG
  RES["Results tab<br/>(inspectors/spectra.js — view a *.spectra.json)"] --> ENG
  ENG --> API["POST /api/task-setup/handover — hand the description to Task setup<br/>POST /api/spectra/load — read a finished .spectra.json"]
```

## 2. The spectrum chart

> **Where this is going.** The chart is drawn today by ~345 lines inside
> `lib/spectra/core.js`, the tab controller. Its design as a sealed module —
> one importable `mount`, one file naming Plotly, a click that leaves as an
> event and returns as an instruction — is written in
> [`spectrumchart.md`](?doc=web/spectrumchart.md). Nothing is built yet: the
> contract was written first so the door could be reviewed before the code
> moves. What this section describes is what the tab does now.


The chart plots **frequency (cm⁻¹) on a shared x-axis against every channel the
run computed** — one stacked panel each, its own y-axis on the left, one
vertical stick per vibrational mode *(two panels since 2026-09-11; it was Raman
alone before, which is why the y-title used to be a constant)*. A slider
broadens each stick into a smooth **Lorentzian** peak (a single FWHM control,
default 20 cm⁻¹) so overlapping modes read as one band — the broadened envelope
is drawn over the sticks.

Below the sticks, **a rug of ticks carries every mode**, coloured by whether it
is active in one channel, both, or neither; a mode dark to **both** techniques
also gets a dotted line across the full height of every panel, because being
invisible to spectroscopy is not the same as not vibrating. A **display floor**
hides bands below a percentage of the strongest peak *in each channel* — the
rug is never filtered. The chart's own contract is
[`spectrumchart.md`](?doc=web/spectrumchart.md).

**Picking a mode does not mean hitting the line.** Each mode carries an invisible
band, as wide as the broadening you set (with a floor of 8 cm⁻¹ so bare sticks
stay reachable), and a click anywhere inside it selects that mode. Each band is
clamped to half the distance to its nearer neighbour, so no two overlap and the
band you are in is always the *nearest* mode — crowded regions get tighter
targets, which is where being off by one mode matters. Two modes at the same
frequency (BDT's 31 and 32 are 0.011 cm⁻¹ apart) cannot be separated this way;
that is what the table is for.

Two special cases are worth knowing:

- **Imaginary modes** (a negative frequency — a sign the geometry is a saddle
  point, not a true minimum) are drawn in **red at their negative frequency**, so
  a bad optimization is visible at a glance.
- **No strengths, mode positions** *(user, 2026-09-28: "there should be no
  width or spectrum for siesta result. it is just mode of frequency, and
  animation of the mode"; and, the same day: "we can still have the modes
  ploted as one straight bar in a wave-number plot. this visualization can be
  useful in selecting different modes for showing animation and electronic
  structure, just like the pyscf case the only differene is that there is no
  'peak fitting with lorentz shape or peak height', just bar/line to show
  where the mode is")*. A result in which no mode carries a strength draws
  **one line of one height at each mode's wavenumber** under the heading
  *Mode positions* — no broadened curve, no height scale, and **no width or
  display-floor control**, since there is no height for either to act on
  ([`spectrumchart.md`](?doc=web/spectrumchart.md) § 6.2). A click on a line
  picks the mode exactly as on a spectrum. One sentence above the lines says
  which of four cases it is: **the route computes none** (SIESTA — *this route
  computes the frequencies and the mode shapes, not infrared or Raman
  intensities*); **the run asked for none** (a PySCF run with both strengths
  off); **they are still being computed** (a PySCF run before its first
  strength lands — its `phase_raman` or `phase_ir` still to finish, § 7 —
  which draws the spectrum the moment one does); or **the run asked and
  recorded none** (its log says why). *(Until 2026-09-28 the chart drew every
  mode as a unit-height stick with a broadened curve over them — a frequency
  distribution, which read as an intensity spectrum wherever modes crowded;
  and for a few hours that day it drew nothing at all.)*

## 3. The three views of one mode

Under the chart sit **three tabs — Modes table, Mode visualisation, Electronic
structure —** all describing the *same* selected mode. They were three bands
stacked down the page, so comparing what a mode looks like against what its
levels do meant scrolling between two places and holding one in memory.

**Selecting a mode never switches tab.** All three update underneath and you stay
where you were looking. Becoming visible is an event rather than a style change:
a 3D canvas and a chart both take their size from a box, and a box in a hidden
tab has none — so opening a tab re-fits the viewer and re-sizes the chart.

The **Modes table** is a table of every vibrational mode — its number,
frequency, Raman activity, IR intensity, whether it is imaginary, and whether
the per-mode orbital check reached it (*ES?*) — that you can **sort and
filter**, export to CSV, and click a row to select a mode. Four more columns
give that check's numbers: the HOMO, the LUMO, the gap and the gap's largest
shift along the mode. **The columns follow the file's route, never its data**:
a column the route can compute stays in the header and its cells fill in as
the numbers land, and a column the route cannot compute — infrared, Raman and
the four orbital columns on SIESTA — is not shown, nor exported to CSV
(§ 9b.3).

### 3.1 The level diagram — and why it has to zoom

Under the table, a selected mode with excited-state data draws a **molecular
orbital correlation diagram**: the same levels at three geometries — pushed to
−A, at equilibrium, and pushed to +A — with **every orbital joined across the
three**, so the eye follows one level rather than comparing three separate stacks
of dashes.

```
        E ▲     ─────  ╲___  ─────      each dash is one orbital, in one geometry
          │                              the joins are the same orbital, moving
          │     ═════ ═════ ═════   ← LUMO
          │
          │     ═════ ═════ ═════   ← HOMO
          │     ─────  ___╱  ─────
          └──────  −A    eq    +A
```

**Zoom is not a convenience here, it is the point.** The shifts this panel exists
to show are *small*: in the BDT result, mode 1's HOMO moves **0.018 meV** between
−A and +A, against an 11.4 eV span of drawn levels. That is 1/4000 of a pixel —
literally invisible at full scale. So the figure opens on the whole picture and
then **scroll to zoom the energy axis, drag to pan, double-click to fit**. The
energy axis is free; the horizontal axis is locked, because it is three
geometries rather than a quantity and there is nothing between them.

It is drawn by **Plotly**, the same library as the spectrum above it, and that
choice replaced hand-rolled SVG on 2026-08-05. The SVG was right while the figure
was static; once it needed zoom, pan, a reset and a hover readout, writing those
four by hand beside a chart library already loaded on the page would have been
inventing a wheel in view of the wheel. Plotly takes colours as values rather
than inheriting CSS, so `chartTheme()` in `lib/spectra/core.js` reads the tokens
off the document and hands them over — the tokens stay the one source of truth
for both charts.

Beside the diagram sits a **panel of numbers**, grouped by the question
each answers: where the levels sit, how the gap moves, and the electron–phonon
coupling ΔE/(2A). The displacement A itself is in the panel header, since it is
the input every number below depends on rather than a result among them.

**A fourth, run-level tab: Thermochemistry** *(v5 artifacts)*. Beside the
three per-mode views sits a tab that appears when the results carry a
`thermo` block ([`archive/2026-08-20-spectra-migration-plan.md`](?doc=archive/2026-08-20-spectra-migration-plan.md)
§ 2b): the headline numbers with the regime sentence, curves over the deck's
temperature grid, and the decomposition bar at the headline temperature. The
**deck computes, the viewer draws, and the viewer derives nothing**: the
electronic reference is the file's own equilibrium energy — PySCF's SCF
energy; SIESTA's route carries none, and its numbers are vibrational
contributions alone ([`engines/vibration.md`](?doc=engines/vibration.md)
§ 4.7). **The labels follow the regime**:

| `thermo.regime` | the headline | the curves | the bar |
|---|---|---|---|
| `rrho` — PySCF, nothing held | T, P, ZPE, `H − E_elec`, S, `G − E_elec` (kcal/mol), *full RRHO* | `H − E_elec`, `G − E_elec`, `T·S` | ZPE · U_vib · translation + rotation + pV · −T·S · `G − E_elec` |
| `vibrational-only` — atoms held, and every SIESTA result | T, ZPE, `ZPE + U_vib`, `S_vib`, `F_vib`, *vibrational contributions only* — no pressure, which enters only the gas-phase translation | `ZPE + U_vib`, `F_vib = ZPE + U_vib − T·S_vib`, `T·S_vib` | ZPE · U_vib · −T·S_vib · `F_vib` |

The writer's own `thermo.note` says what the numbers are good for and where
they stop, beside them. *(Until 2026-09-28 the viewer recovered E_elec from the
grid as `h − zpe − u_vib − k_B·T` — a `k_B·T` the vibrational sums do not
contain, and without the translational and rotational energies the RRHO sums
do — so both regimes' curves were shifted and neither bar summed to its total;
and it printed `P = 1 atm` over results no pressure entered.)* The phase
indicator gained a **Relaxation** dot the same way (data-driven off
`phase_relaxation`), and an **Infrared** dot off `phase_ir` (2026-09-28 — the
flag existed and was waited on, and no dot showed it); it shows no dot for a
phase the file's route does not have (Raman, infrared and the probe on
SIESTA), nor for infrared in a file written before its flag (`""`, no
record); a v4 file simply shows it empty and hides the thermo tab.

## 4. Clicking a mode — the 3D animation

Click a stick in the chart or a row in the table, open the **Mode visualisation**
tab, and that **normal mode animates in 3D**: the atoms oscillate along the mode's displacement vectors so you can see
which bonds stretch or bend. This 3D box is **not** MolView — it's the concealed
**VibrationView** module ([`vibrationview.md`](?doc=web/vibrationview.md)), a
self-contained viewer built for exactly this one job (it owns the oscillation loop
and greys out frozen atoms).

> Two different 3D viewers live on the Spectra tab and it's easy to conflate
> them: the **inspect card at the top** is a read-only *MolView* showing the
> static input structure (§5); the **mode box** is *VibrationView* showing an
> animated eigenvector. Different modules, different jobs.

### 4.1 How big to draw the motion — and why the choice is not cosmetic

The eigenvector fixes the **shape** of the motion: which atoms move, in which
direction, and how far *relative to each other*. It does not fix the **size**.
An eigenvector's overall scale is arbitrary until something sets it, so the
animation is always a shape multiplied by a number — and where that number comes
from is the whole of this control.

```
   what the diagonalisation gives        what you choose
   ─────────────────────────────         ───────────────
   direction, and relative size     ×    absolute size    =   the animation
```

**Three answers, and only two of them are physics.**

#### exaggerated — a drawing convention

The eigenvector is rescaled so the largest per-atom vector has length 1
(`eigenvector_display`, dimensionless), and the slider states how many ångström
that largest swing is. The default **0.15 Å** is roughly a tenth of a bond
length: visible without looking dislocated.

This is a **visualisation choice and not a result.** Jmol, Avogadro and GaussView
all exaggerate mode animations, for the reason given below — genuine amplitudes
are too small to read on screen. It is the default here because the first
question a user asks of a mode is "which atoms move", and that is a question
about shape.

#### physical, zero-point — the molecule at absolute zero

A quantum harmonic oscillator cannot be at rest. Localising a particle in a
potential well costs kinetic energy, so the lowest state of every mode sits
½ħω above the minimum, and the nuclei retain a spread even at T = 0. That spread
is the **zero-point amplitude**:

```
    ⟨Q²⟩ = ħ / 2ω           Q_rms = √( ħ / 2ω )
```

where **Q** is the mass-weighted normal coordinate. In the units a spectroscopist
works in — a wavenumber ν̃ in cm⁻¹, since ω = 2πcν̃ — this evaluates to

```
    Q_rms = 4.106 / √( ν̃ [cm⁻¹] )        in √amu · Å
```

**The tab does not compute this number.** Every mode in the file carries it
as `zero_point_amplitude_amu12_ang`, derived at every serialisation from the
one constant in `constants.py` ([`engines/vibration.md`](?doc=engines/vibration.md)
§ 6.3, § 6.6); the animation reads it and applies only the thermal factor
below. *(A second spelling of the constant lived in the tab until
2026-09-24.)*

and the Cartesian displacement of atom *i* is `Q_rms × L_i`, with **L** the
mass-weighted eigenvector normalised so `Σᵢ mᵢ|Lᵢ|² = 1` (`eigenvector_canonical`).
The mass carried in that normalisation is what makes the amplitude **atom-specific**:
a heavier nucleus in the same mode moves as 1/√m.

Two consequences, both visible in the animation:

| ν̃ | H (1 amu) | C (12) | Au (197) |
|---|---|---|---|
| 100 cm⁻¹ — a torsion | 0.41 Å | 0.12 Å | 0.029 Å |
| 1000 cm⁻¹ — a ring breath | 0.13 Å | 0.037 Å | 0.0093 Å |
| 3000 cm⁻¹ — a C–H stretch | 0.075 Å | 0.022 Å | 0.0053 Å |

*(the swing of an atom carrying the entire mode; a real mode distributes it, so
per-atom values are smaller)*

**Stiffer bonds move less** — amplitude falls as 1/√ν̃, so a C–H stretch is half a
torsion. **Heavier atoms move less** — as 1/√m, so gold barely moves in a mode a
hydrogen dominates. Both are why an honest animation of a stretching mode looks
almost still, and why the exaggerated default exists.

#### physical, thermal — the same thing at a temperature

Above absolute zero the excited vibrational states are populated too, and the
thermal average over them raises the mean-square displacement by a factor:

```
    ⟨Q²⟩_T = ( ħ / 2ω ) · coth( ħω / 2k_BT )
```

The amplitude therefore grows by **√coth(x)**, `x = ħω / 2k_BT`. As T → 0,
coth → 1 and this becomes the zero-point expression exactly — the two are one
formula, not two, which is the check that the limit is right.

**The useful question is which modes it changes.** One wavenumber is worth
**1.44 K**, so room temperature is only `k_BT ≈ 207 cm⁻¹`. A mode much stiffer
than that has ħω ≫ k_BT, sits in its ground state whatever the temperature, and
is said to be **frozen out**. At 298 K, as a multiple of the zero-point swing:

| ν̃ (cm⁻¹) | 50 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|
| ×  zero-point | 2.9 | 2.1 | 1.27 | 1.01 | 1.00 |
| at 500 K | 3.7 | 2.7 | 1.57 | 1.06 | 1.00 |

So temperature is worth setting for **soft modes** — torsions, librations,
metal–ligand bends, lattice modes — and is invisible on a C–H stretch. This is
the same physics that makes the vibrational heat capacity of a stiff mode
approach zero: a mode that cannot be thermally excited neither stores energy nor
moves further.

#### Why the two normalisations must never be crossed

The two physical answers use `eigenvector_canonical`; the exaggerated one uses
`eigenvector_display`. **They are not in the same units**, and the amplitude that
pairs with each differs accordingly:

| | eigenvector | its amplitude is in |
|---|---|---|
| exaggerated | dimensionless (max = 1) | **Å** |
| physical | 1/√mass (`Σ mᵢ\|Lᵢ\|² = 1`) | **√amu·Å** |

Each pairing multiplies out to ångström of motion — but only within itself.
Feeding a display eigenvector into a physical amplitude, or the reverse, produces
a picture that is wrong **without looking wrong**, which is precisely why the
backend ships two fields: schema v1 carried one `eigenvector_free` used for both
animation and Raman projection, and that was recorded as a correctness bug when
the two uses were separated (see the SCHEMA_VERSION history in
`molbuilder/spectra/results.py`).

For the same reason an **export records which pairing produced it**
([`vibrationview.md`](?doc=web/vibrationview.md) § 12.2): the amplitude alone
does not say what it means, so it travels beside its normalisation and neither is
written without the other.

#### Where each piece is computed

The physics is the **tab's**, not the viewer's. VibrationView holds no frequency,
no temperature and no physical constant — it receives a displacement array and a
number and multiplies them ([`vibrationview.md`](?doc=web/vibrationview.md)
§ 12.2). `lib/spectra/core.js` turns a mode's wavenumber into the amplitude above
and chooses the eigenvector that pairs with it.

### 4.2 The sentence under the viewer — which atoms the mode belongs to

Under the 3D box is one line describing what you are watching:

```
the motion is 91% C, 9% H · nothing moves further than 0.173 Å from rest,
16% of that atom's bond · drawn exaggerated
   └── whose mode ──┘        └──── how big, against a yardstick ────┘   └ real? ┘
```

Each part answers a question a bare number leaves open.

**"the motion is 91% C, 9% H" — whose mode is it.** Each element's share of the
mode, computed as its part of the mass-weighted motion:

```
        share of atom i  =   mᵢ|Lᵢ|²  ⁄  Σₖ mₖ|Lₖ|²
```

This is the **kinetic-energy distribution**, the standard way a mode is assigned
to the atoms that carry it. For a harmonic mode the ratio is the same at every
phase of the oscillation, so it is a property of the mode and not of the instant
you look at it. Either stored eigenvector gives the same answer: the two forms
differ by one scalar per mode, and a scalar cancels in a ratio.

> **Why mass-weight it at all — the trap this replaced.** The line used to name
> the atom with the largest displacement. In the benzene-dithiol result that is a
> hydrogen in **32 of 36 modes**, including the 1648 cm⁻¹ ring stretch, where the
> hydrogens move 18% further than the carbons (|L| = 1.15 against 0.98) and carry
> **9%** of the motion. A light atom travels further for the same energy, and
> hydrogen is the lightest thing in most molecules — so "which atom moves
> furthest" answers *hydrogen* almost regardless of the mode, which is a true
> sentence that tells you nothing. Weighting by mass asks the question worth
> asking, and the answers agree with textbook assignments: 92% H for the C–H
> stretches at 3175 cm⁻¹, 91% C for the ring stretch, 49% S for the 205 cm⁻¹ mode.

**"nothing moves further than 0.173 Å from rest" — a ceiling, not a subject.**
Every atom in the picture stays inside it. It is measured to *one extreme* of the
swing, so an atom covers twice this between extremes. The atom concerned is
deliberately **not named**, for the reason above.

**"16% of that atom's bond" — the yardstick.** 0.173 Å is a number; "a sixth of a
bond" is a picture. The comparison is against that atom's own nearest-neighbour
distance, since a C–H bond is short and an Au–Au contact is long. Beyond 3.0 Å
the line says *nearest contact* rather than *bond*, because calling it a bond
would be a claim rather than a label.

**"drawn exaggerated" — is it real.** § 4.1: exaggerated is a drawing convention;
the two physical settings are measurements. This is the one thing a reader must
not get wrong about a number quoted from this panel.

#### Where the composition is computed, and why there

**The server**, in `/api/spectra/load` — not the browser, and not the file.

The share needs atomic masses. The `.spectra.json` stores none, and the browser
has none. The **table already exists**: ASE ships the IUPAC standard atomic
weights, and `chemistry.py` already reaches into `ase.data` for atomic numbers,
so `chemistry.atomic_mass` is a *name for that table* rather than a second copy
of it. Typing 118 masses into this program — in Python or, worse, into
JavaScript — would have been a second source of truth that no test would notice
going stale.

It is computed **when a result is opened, not when it is written**:

| | what it costs | what it means for the results already on disk |
|---|---|---|
| computed at load *(chosen)* | one pass over each eigenvector | every existing result gains the line the moment it is opened |
| written into the file | a schema bump | nothing shows until each result is re-run |

`SpectraResults.to_dict()` is the **on-disk format** and round-trips through
`from_dict`; the share is derived, so it is added to the reply and never to the
file. A result whose shares cannot be worked out — an element ASE does not know,
or no stored geometry — is served without the field, and the panel simply drops
that clause.

The maths lives in `spectra/results.py :: motion_share_by_element`; the sentence
is assembled in `lib/spectra/core.js :: _reportSwing`.

## 5. The standalone tab — from a structure to a script

On `/spectrum-calculation` the page is a short vertical workflow:

1. **Inspect the structure.** The top card mounts a **read-only MolView**
   ([`molview.md`](?doc=web/molview.md)). Pick a `.xyz`/`.pdb` in the Projects
   sidebar and load it; the card shows the 3D structure plus its atom count and
   formula, read straight off the loaded model.
2. **Charge and spin.** The chemistry card shows the charge and spin each
   engine form's vibration will carry — the one electronic-state class's answer
   for exactly what the form says, about the structure the viewer holds, each
   value with its reason (`POST /api/structure/analyze`,
   [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md)
   § 2a). It fills nothing in: leave a field blank and it is worked out; type a
   value to state it, and the card follows.
3. **Pick the engine, set the parameters.** An engine strip — the same
   widget the Structure-optimization tab mounts, `lib/tab-strip.js` over
   the shared sheet, two buttons over two mounted forms — offers **PySCF**
   and **SIESTA**. Both forms are **built from the
   CATALOGUE**, narrowed to (engine, vibration)
   (`GET /api/build/schema/<engine>?calculation=vibration`), through the
   same door and the same renderer as the Build tab, so a parameter is
   defined once and rendered the same everywhere
   ([`engines/template.md`](?doc=engines/template.md) § 6.3; the items each
   engine shows and why are [`engines/vibration.md`](?doc=engines/vibration.md)
   § 3.1). The strip's choice is the **description's engine** — what the
   live checks validate against and what the hand-over sends — and never a
   parameter of the deck (the one-choice `engine` form item retired
   2026-09-24). **The structure decides the default**: a structure that
   repeats or continues along an axis switches the strip to SIESTA and says
   why, because PySCF computes a repeating structure as an isolated cluster
   — a note, not a refusal (`engines/vibration.md` § 3.3); a click on the
   strip beats the default. The chemistry card answers for both forms.
4. **Send to Task setup.** The Send button runs the general hand-over
   ([`handover-procedure.md`](?doc=web/handover-procedure.md), via
   `lib/task-handover.js` — the same door `/structure-optimization` uses):
   the server renders the parameter template, the structure pair and
   `task.1st.json` (carrying `calculation: "vibration"`), the browser writes
   them into the selected folder, and Task setup finishes shape and stages.
   **This tab renders no deck** — the deck is written by `prep`, on the
   machine that runs it
   ([`execution/script-preparation.md`](?doc=execution/script-preparation.md)).

When the job runs it writes a `.spectra.json`; picking that on the Results
tab is what fills the chart, and a run still going is followed there (§ 7). On SIESTA the force-constant run leaves
`<label>.FC` and the same job then derives the modes from it — its finish,
`mb_vibration.pyz`, run by the wrapper — and writes the same file beside the
run, so the launch is the last step on both engines
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.5).

## 6. The API door

One spectra route remains — the ARTIFACT reader (full shape in
[`web-api.md`](?doc=web/web-api.md)); the compute half goes through the
catalogue schema + hand-over doors (§ 5):

| Route | Does | Returns |
|---|---|---|
| `POST /api/spectra/load` | parses an existing `.spectra.json` into display data | `{ok, results}` — or a **typed** error carrying a `kind` string (missing → 404, wrong schema version → 422, malformed or bad-field → 400) so the UI can react without reading the message |

Both follow the app's `{ok: …}` envelope convention. The form schema comes
from the catalogue door (`GET /api/build/schema/<engine>?calculation=vibration`, once per engine);
(the old `GET /api/build/schema/spectra` route retired at P3; frozen atoms
travel with the structure now — § 8).

## 7. Live updating, and what a refresh does

The file is the one picked in the Results tab's dropdown — the only route; a
path box with *Load once*, *Start watching* and *Stop* stood above the panel
until 2026-09-28, a second way to name a file, and a run picked while still
going was shown as a snapshot unless someone pressed *Start watching*
*(user, 2026-09-28: the dropdown is the only route, and a still-running result
is followed until its phases finish)*. If you pick a spectrum whose calculation
is still running, the viewer **follows it by itself — it polls
`/api/spectra/load` every 2 seconds** and redraws as new modes arrive, and the
status line above the phase dots says it is following. (That's a faster
cadence than the trajectory viewer's 15 seconds — a spectrum job produces its
phases in quicker bursts.) The poll stops when the run is no longer live —
finished, failed or never launched, the state the server sends with the file
from the one door ([`results.md`](?doc=web/results.md) § 4.1) — when the file
goes away or keeps failing to load, and with the viewer — another pick, or
leaving the tab. The phase dots say how far each phase got — the infrared
intensities with their own flag, `phase_ir`; a phase it never asked for reads
`not requested` from the first write
([`engines/vibration.md`](?doc=engines/vibration.md) § 4.9) — and a run that
stopped between phases says so on the status line, in the run's own words.
*(It stopped once every asked-for phase reported complete until 2026-10-03, so
a run killed mid-phase was followed until the page closed.)*

Like every Results-tab viewer, the spectra viewer is a small **state machine**
(idle → loading → loaded / watching → error), and **Refresh is a clean reload**
of the same file — the same reset a file-switch does. The full "which state
resets what" rules live in [`results.md § 4`](?doc=web/results.md). One thing
specific here: your view **preferences** (the broadening width, the animation
amplitude and speed and its pairing, the thermal temperature, the table's sort
and filter) **survive a page reload**, not just a file-switch (shipped
2026-09-10; a documented follow-up until then).

**They are kept where everything else is kept** — the workspace
([`workspace.md`](?doc=web/workspace.md)), under the tag
`results:spectra:ui`, one slot at step 0. That is the same lane MolView uses
for a viewer's camera, frame and switches
([`molview.md`](?doc=web/molview.md) § 11.2b, "looking is not changing"), and
it matters that it is not a second store of its own: `workspace.md` § 4
promises *one way to save and one way to load*, and a private one here would
be a second home for Results-tab state — which is what the tag exists to
prevent. So these knobs follow the project, not the browser tab.

Two properties, because both are visible when they fail. **A stored value is
not trusted**: a slot outlives a rename and the file is editable, so a knob is
restored only when its key is one the bucket declares *and* its type matches
the default's — a `broadeningFWHM` of `"abc"` is dropped rather than handed to
the broadening maths, and a payload whose version is not the current one is
ignored whole. And **writes are disarmed until the restore has finished**,
because the defaults are announced while the tab initialises and a write fired
then would overwrite the very value being read back. If the workspace is
unreachable the knobs stay at their defaults and the viewer works; a
preference is a convenience, not the result.

## 8. Frozen atoms travel with the structure

Frozen atoms are **structure-side facts**: they live in the model as a region
and ride the structure's own files — the `.molstruct.json` half of the codec
pair the hand-over writes — never a form field
([`archive/2026-08-20-spectra-migration-plan.md`](?doc=archive/2026-08-20-spectra-migration-plan.md)
§ 2). Set them in the viewer, or load a structure whose sidecar already
carries them; the Send button exports the model **in one read**
(`exportFile()`), so what you see frozen is what the calculation holds fixed.

**Frozen means frozen through every phase** (user ruling 2026-08-21). When
the deck relaxes, the relaxation holds the same set fixed — it writes
geomeTRIC's `$freeze` constraints file, the same mechanism the optimization
deck uses; under `already_relaxed` there is no relaxation and no constraints
file — and the Hessian is built over the free atoms only either way. Which atoms to freeze is
the user's own call: the tab never second-guesses the set, and nothing warns
you off a choice you made on purpose. What the calculation DOES say, out
loud, is what the freeze means for the numbers: the deck states the regime
(the free atoms' block of the true Hessian, *vibrational-only*
thermochemistry), the preflight names the frozen count **and how many
whole-body motions survive it**, the artifact records what was removed
(`removed_motions`, § 9b), and the Methods paragraph spells out that the
reported frequencies are those of the free atoms moving in the field of the
fixed ones.

**How many vibrations that leaves is not `3N−6`, and it is not free either.**
Freezing removes the whole-body motions that would move a frozen atom — and
leaves the ones that would not. Two frozen atoms still let the rest of the
molecule turn about the line through them — unless the surviving turn moves no
free atom, as with both oxygens of CO₂ held; one frozen atom leaves all three
rotations about it; three non-collinear frozen atoms leave nothing. Those
leftovers are not vibrations and must not reach a spectrum or a thermochemistry
sum. **[`science/normal-modes.md`](?doc=science/normal-modes.md) owns that
rule** — the count, why the leftovers are projected out before diagonalisation
rather than spotted afterwards, and the acceptance test. Nothing here or in the
emitters re-derives it.

*(This paragraph used to end "— an anchored molecule does not rotate". A
measured BDT run on 2026-09-21, both sulfurs frozen, reported a 36th mode that
was exactly the ring turning about the S···S line: 100 % of that motion, at
−0.93 cm⁻¹. An anchored molecule does not **translate**; whether it rotates
depends on where its anchors are.)*

**And the two lists must PARTITION the atoms — checked, not assumed.**
`SpectraResults` refuses at construction unless `free_atom_idxs` and
`frozen_atom_idxs` are disjoint *and* their union is exactly
`range(n_atoms_total)`, and unless every mode's eigenvector carries one row
per free atom. A result that fails either is a parser or programmer error,
and catching it where the object is built beats meeting it when the viewer
tries to draw a displacement.

> **A count-only check is not the same test.** It passed `free=[0,1,5]`,
> `frozen=[]`, `n=3` — three indices for three atoms, one of them not an atom
> — and the frontend then silently dropped that displacement from the
> scatter. Counting says *how many*; a partition says *which*.

The old form field (`frozen_indices`, pre-filled by the schema route from the
sidecar) retired with the P2 substitution: a form default was a **second
copy** of a structure fact, editable into disagreement with the structure it
described.

**The relaxation record travels the same way** *(2026-09-24)*. A pair exported
from the Results tab of a finished relaxation carries `info.relaxation` beside
`info.calculation` ([`model/parse.md` § 5b.1](?doc=model/parse.md)); the tab
reads nothing of it itself — the gate does, and its findings land on the
`already_relaxed` card like every other finding about that field, so the
record is displayed where the choice is made
([`engines/vibration.md` § 2.2](?doc=engines/vibration.md), the record table).
The Metadata pane of the viewer shows the raw store.

## 9. Where the module stands — ESM status

The design goal for every front-end module is a **concealed, independently
reusable ES module**. Spectra is **partway there**:

| File | Role | ESM today |
|---|---|---|
| `static/spectra/viewer.js` | the standalone-tab controller | **yes** — a real ES module (it imports MolView) |
| `lib/spectra/core.js` | the shared engine (chart, table, animation, API) | **no** — still a classic global-registered script (`window.molbuilder.spectraInspector`) |
| `lib/inspectors/spectra.js` | the Results-tab presenter | **no** — classic; registers via `molbuilder.inspectors.register` |

So the engine and its Results-tab presenter still load as plain scripts and
publish themselves on `window.molbuilder`, relying on the runtime registry to
sequence them. Converting both to ES modules is a tracked follow-up: they convert
together with the **`inspectors` → `presenters` rename** — one pass (task #102)
that does the file-viewer registry *and* the heavy engine cores it mounts (this
`lib/spectra/core.js` among them), since converting them rewrites those files
anyway. See [`presenters.md`](?doc=web/presenters.md) and
[`plans/plan.md`](?doc=plans/plan.md) **W15**.

## 9a. Which modes get the expensive treatment, and what the paper says

**Two things the deck decides that nothing else can, and both end up in what
you publish.**

### 9a.1 The three selectors

Electronic structure at a displaced geometry is an SCF per mode, so which
modes get one is a real cost decision. `es_mode_selection` — a PySCF item:
SIESTA's route has no probe (§ 9b.3) — takes one of three answers:

| | picks |
|---|---|
| `skip` | none |
| `all` | every mode |
| `explicit` | exactly the indices you list |

*(`top_n` and `threshold` were retired by decision on 2026-09-23 and removed on 2026-09-28, V1.6. Both ranked modes by **Raman activity**, and the probe measures how the gap moves along a mode — ∂ε/∂Q, which follows its own selection rule: in a centrosymmetric molecule the Raman-bright modes are exactly the infrared-dark ones, so the filter kept one symmetry class and dropped the other every time; and for an engine that computes no strengths they were undefined rather than empty. The frequency window is the cost control, and `all` is cheap where it matters — 8 SCFs for CO₂.)*

**The frequency window filters `all` (`skip` selects nothing) and is IGNORED by `explicit`** —
naming a mode by index is saying *that one*, and a window that silently
dropped it would answer a question you did not ask. So the form locks the
window outside `all`, as it locks the explicit list outside `explicit`
*(user, 2026-09-28)*.

**A mode that already has its electronic structure is skipped on a resume**,
whatever the selector says: the result persists, so re-running it buys
nothing.

### 9a.2 The Methods paragraph is composed, not written

`render_methods_md` builds the Markdown that ships **in the emitted script's
header and beside the finished result** — one composer, so the two cannot
describe different calculations. Every prose decision comes off the config:
the level of theory (`PySCFConfig.is_dft` — under Hartree–Fock no functional),
basis, the dispersion correction on either method with its own papers, the
selector above, the amplitude convention (§ 4.1) and the frequency window.

It is composed **once, before the run**, and says *what will be done*; the
mode count it states is the run's by construction, and the load path adds the
one sentence only the run can settle — which infrared route ran
(`with_ir_route`, [`engines/vibration.md`](?doc=engines/vibration.md) § 4.10).
**The engine fragment** is passed IN by the caller, because this composer is
engine-ignorant on purpose; the one producer that has an engine knows which
it is. *(An **after** form that re-composed the paragraph from the parsed
results, which nothing in production called, was deleted on 2026-09-28 —
V1.18; so was the preview modal it once also shipped in.)*

`extract_citation_keys` reads the bibliography keys back out of the rendered
prose, so what is cited is what was actually said rather than a second list
kept beside it.

## 9b. Where every number comes from — the file's contract

**This is a quantitative result, so the chain from the engine's output to the
number on screen is written down** — and it is written with the calculation,
not with the tab: [`engines/vibration.md`](?doc=engines/vibration.md) § 6
holds the result file's contract — every key and who writes it (§ 6.2), a
mode (§ 6.3), the provenance table saying what molbuilder only passes through
and what it **derives** (§ 6.4, with the two rules that table enforces: a
passed-through number is not ours to test, a derived number's rule is
callable and never only script text), what a SIESTA file looks like beside a
PySCF file (§ 6.5), the activity classes and the element shares derived at
serialisation and at load (§ 6.6), and what the reader refuses (§ 6.7). This
tab reads that file and draws it; the one thing it derives at load is the
element-share sentence of § 4.2, computed on the server for the reason given
there.

### 9b.3 A second engine, and what the file says when a number is missing *(user, 2026-09-23)*

The calculation has **two engines** (chosen 2026-09-21;
[`engines/vibration.md`](?doc=engines/vibration.md) § 1.3): PySCF for isolated
molecules, SIESTA for slabs and junctions — where the point is consistency,
the same pseudopotentials, orbitals and functional as the transport the modes
will be displaced for. SIESTA yields **frequencies and mode shapes only**;
infrared and Raman are not offered there (`science/normal-modes.md` § 4a.6).
So the file has to carry a result with pieces missing, and say so.

**One file, every engine; a missing number is ABSENT, never zero.** `engine`
names who produced the file. A key an engine cannot produce is absent or
`null`, and a reader treats that as *not computed* — a different statement
from `0.0`, which is a measured absence (`ModeData`'s own rule, and the
`partial` activity class). The consequences, for an engine that cannot
compute a channel and for a run that did not ask for one:

| where | what a reader sees |
|---|---|
| the chart | **mode positions, not a spectrum**, when no mode carries a strength — every SIESTA result: one line of one height at each mode, no curve, no height scale, no width or display-floor control, and one sentence above them: *not computed on this route* (§ 2) *(user, 2026-09-28)*. With some strengths computed, a lane whose channel is missing is titled *not computed* |
| the rug | part of the chart: every mode, in the `partial` colour where nothing was computed for it — never `silent`, because nobody looked |
| the modes table | **a column the file's route cannot compute is not shown** — IR, Raman and the per-mode orbital columns on SIESTA, read BY ROLE: the file's own config carries no switch for them. Within a route, a `—` in a cell, not a `0.00`; the CSV export follows the table |
| the electronic-structure tab | on SIESTA: the electrons' response along a mode is the projected density of states at structures displaced along it — **planned** ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.10, plan W42), not built — and the tab says so; it never names PySCF's `es_mode_selection`, which a SIESTA run has no way to set. On PySCF: *not requested* under `skip`, *not in the selection* for a mode the selector left out |
| the run summary | *not computed on this route* for every channel the route lacks — infrared, Raman, the per-mode orbital energies — by the same rule; *not requested* only where the run could have asked |
| the thermochemistry tab | labelled by its regime (§ 3): on SIESTA the vibrational contributions alone, with what they are good for and where they stop ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.7) |
| the write-up | names what was computed; the file and the viewer name what was not, so an absence cannot be read as a zero |

**What the file carries for each engine** — the molecular-orbital block
optional as a whole, the intensities `null` on every SIESTA mode, the routes
recorded per run, the SIESTA metadata naming the `.FC` file and its range —
is [`engines/vibration.md`](?doc=engines/vibration.md) § 6.5. *(What this
tab had to learn from it — the equilibrium energy drawn as a dash rather than
`0.00000000` when it is `null`, the Raman line said by route rather than by a
phase flag, the change fingerprint, and an engine-neutral label in
[`presenters.md`](?doc=web/presenters.md) — was done on 2026-09-24, V1.3 and
V1.13; the fingerprint reads `phase_ir` since that flag has been written,
2026-09-28.)*

## 10. Test map

The calculation's own tests — the rank rule, the decks, the artifact's gates,
the end-to-end runs on both engines — are mapped in
[`engines/vibration.md`](?doc=engines/vibration.md) § 9. This tab's: `test_results_state_contract_spectra_js.py` (the state
buckets), `test_spectra_phase_indicator_js.py` (the phase indicator, the
relaxation dot included), `test_task_setup_tab.py` (the send flow: the shared
door, the kind, and the browser-vs-CLI byte-compat pin),
`test_vibration_render_gate.py` (the deck runs the science gate — and it
refuses), `tests/test_vibration_e2e.py` (the live water runs),
`tests/test_siesta_vibration_results_e2e.py` (a SIESTA result on the Results
tab: the modes without a spectrum, the columns and dots by route, the
vibrational-only thermochemistry and its bars), `tests/test_spectra_from_a_real_run_e2e.py`
(a PySCF result computed and read back, its RRHO bars summing to the headline),
`test_spectra_no_spectrum_sentence_js.py` (the four cases of § 2, by role,
and when the viewer stops waiting — infrared's own flag included, § 7),
`test_vibrationview_maths_js.py` (the animation's eigenvector math).

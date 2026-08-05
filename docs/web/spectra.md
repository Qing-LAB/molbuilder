# Spectra — computing and reading a vibrational spectrum

**Role:** contract
**Domain:** web
**Companions:** [`results.md`](?doc=web/results.md) — the Results-tab shell that
hosts the spectra *presenter*; [`presenters.md`](?doc=web/presenters.md) — the
registry that picks it for a `.spectra.json`; [`molview.md`](?doc=web/molview.md)
— the read-only 3D viewer the standalone tab uses to inspect the input structure;
[`web-api.md`](?doc=web/web-api.md) — the `/api/spectra/*` routes;
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) — the run that
produces the `.spectra.json`.

The spectra surface computes a **Raman vibrational spectrum** for a molecule and
then shows it as an interactive chart: a stick spectrum of frequencies, a
sortable table of the vibrational modes, and — when you click a peak — a **3D
animation of that normal mode**. You reach it two ways, but it is the same code
both times.

## 1. Two surfaces, one engine

Everything below is drawn by a single module, `lib/spectra/core.js`. It is
mounted by two different pages, each a thin wrapper:

- **The standalone Spectra tab** (`/spectrum-calculation`) — the full
  *compute-then-view* workflow: inspect a structure, set parameters, generate a
  runnable script, and watch the spectrum appear as the calculation runs. Its
  controller is `spectra/viewer.js`.
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
  ENG --> API["POST /api/spectra/render — build the run script<br/>POST /api/spectra/load — read a finished .spectra.json"]
```

## 2. The spectrum chart

The chart plots **frequency (cm⁻¹) on the x-axis against Raman activity
(Å⁴/amu) on the y-axis**, one vertical stick per vibrational mode. A slider
broadens each stick into a smooth **Lorentzian** peak (a single FWHM control,
default 20 cm⁻¹) so overlapping modes read as one band — the broadened envelope
is drawn over the sticks.

Two special cases are worth knowing:

- **Imaginary modes** (a negative frequency — a sign the geometry is a saddle
  point, not a true minimum) are drawn in **red at their negative frequency**, so
  a bad optimization is visible at a glance.
- **"Density" mode.** Early in a run — or if the calculation was set up without
  Raman intensities — no mode has an activity value yet. Rather than a blank
  chart, the viewer draws every mode as a **unit-height stick** so you can still
  see *where* the frequencies are while the intensities are still being computed.

## 3. The mode table and the excited-state panel

Below the chart is a **table of every vibrational mode** — its number,
frequency, Raman activity, whether it's imaginary, and whether it carries
excited-state data — that you can **sort and filter**, export to CSV, and click a
row to select a mode. When the run also computed **excited states**, four more
columns appear (HOMO, LUMO, gap, and the gap's shift), and a small bar panel
draws the molecular-orbital levels. Those columns are always present in the
header; their cells simply fill in once there's excited-state data.

## 4. Clicking a mode — the 3D animation

Click a stick in the chart or a row in the table and that **normal mode animates
in 3D**: the atoms oscillate along the mode's displacement vectors so you can see
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
2. **Auto-detect the chemistry.** The page asks the server
   (`POST /api/structure/analyze`) for a sensible charge, spin, and method for
   this molecule and fills them into the form, with a one-line rationale. You can
   override anything.
3. **Set the parameters.** The rest of the form is **built from a schema** the
   server hands back (`GET /api/build/schema/spectra`) — so the form always
   matches the calculation's real options without being hand-maintained here.
4. **Generate.** Clicking generate posts to `POST /api/spectra/render`, which
   reads the structure *off the loaded model* and returns a **runnable script**
   plus a human-readable "Methods" summary and its citations. **Render does not
   run anything** — it writes the script. **Save** then writes a
   `<job>.spectra.py` into your project and installs a run wrapper so you can
   launch it like any other job.

When the job runs it writes a `.spectra.json`; loading that (here, or on the
Results tab) is what fills the chart.

## 6. The two API doors

The engine talks to exactly two spectra routes (full shapes in
[`web-api.md`](?doc=web/web-api.md)):

| Route | Does | Returns |
|---|---|---|
| `POST /api/spectra/render` | turns *structure + parameters* into a run script (does **not** execute it) | `{ok, script, methods_md, bibliography_keys, job_name, issues}` |
| `POST /api/spectra/load` | parses an existing `.spectra.json` into display data | `{ok, results}` — or a **typed** error carrying a `kind` string (missing → 404, wrong schema version → 422, malformed or bad-field → 400) so the UI can react without reading the message |

Both follow the app's `{ok: …}` envelope convention. A third route,
`GET /api/build/schema/spectra`, returns the form schema (and can **pre-fill the
frozen-atom list** — see §8).

## 7. Live updating, and what a refresh does

If you load a spectrum whose calculation is still running, the viewer **polls
`/api/spectra/load` every 2 seconds** and redraws as new modes arrive. (That's a
faster cadence than the trajectory viewer's 15 seconds — a spectrum job produces
its phases in quicker bursts.) It considers the run done once the result reports
all its phases complete.

Like every Results-tab viewer, the spectra viewer is a small **state machine**
(idle → loading → loaded / watching → error), and **Refresh is a clean reload**
of the same file — the same reset a file-switch does. The full "which state
resets what" rules live in [`results.md § 4`](?doc=web/results.md). One caveat
specific here: your view **preferences** (the broadening width, the animation
amplitude and speed, the table's sort and filter) survive a file-switch but are
held **in memory for the life of the mount only** — a page reload starts them
back at their defaults. (Persisting them to session storage is wired as a
follow-up, not shipped.)

## 8. Frozen atoms come from the structure's sidecar

If the structure you loaded carries a `.molstruct.json` sidecar with a
frozen-atom set (say, atoms you fixed in a previous relaxation), the schema route
reads it and **pre-fills the "frozen indices" field** so those atoms are held
fixed in the spectrum calculation too. This only sets the *default* — the form
stays authoritative, so you can still edit the list. If the sidecar can't be read
the field is simply left blank and a short notice explains why; it never fails
the page.

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
[`roadmap.md § 3`](?doc=roadmap.md).

## 10. Test map

Engine + backend (`tests/spectra/`): `test_blueprint.py` (the three routes +
dispose contract), `test_config.py` (the schema shape), `test_engine.py` (the
PySCF engine), `test_methods.py` (the Methods prose + citations),
`test_parsers_json.py` (the `.spectra.json` round-trip), `test_script.py` (the
emitted script), `test_selection.py` (which modes get computed),
`test_atom_index_contract.py` (the free-atom invariant).

Viewer + integration: `test_results_state_contract_spectra_js.py` (the state
buckets), `test_spectra_phase_indicator_js.py` (the phase indicator),
`test_spectrum_generate_e2e.py` (the end-to-end render flow),
`test_vibrationview_mode_math_js.py` (the animation's eigenvector math).

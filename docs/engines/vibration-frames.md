# Frames from the modes — the motion toolkit

**Role:** contract — the API and framework that turn a vibration's modes into
frames: how the molecule moves, its snapshots, a mode's own frames, the frame
set they are written as
**Domain:** vibration → frames
**Companions:** [`engines/vibration.md`](?doc=engines/vibration.md) (the
foundation: the vibration calculation and every mode's data, § 6.3, § 6.11);
[`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
(the theory: every state, the whole molecule, why each recipe is right, with
worked numbers); [`model/structure.md`](?doc=model/structure.md) § 2.2f (the
multi-frame structure, the container); [`engines/transport.md`](?doc=engines/transport.md)
§ 2a.9, § 2a.12 (a calculation over frames);
[`engines/vibration-postprocessing.md`](?doc=engines/vibration-postprocessing.md)
§ 2a, § 2b (the calls, worked).

*(The user, 2026-11-23: "this is entirely user's research and i created
molbuilder to support this which does not mean molbuilder does a fully
packaged rigid result generator"; "molbuilder should expose the correct api
to choose the mode, and provide a random number/weight calculator/generator";
"transport calculation should not overreach to design what the multiframe
is. we need to cleanly separate the logic and design"; "vibration
calculation is the foundation that exposes the fundamental data of all
modes".)*

**To build.** What this chapter marks *(to build)* is the plan's R3d2b2c
([`plans/plan.md`](?doc=plans/plan.md) § 2) — the probe's use of the nodes
(§ 6) is R3d2c's; § 9, decided 2026-11-25, is the plan's R3d2b2b4, before
it; what each release built is its row in
[`process/changelog.md`](?doc=process/changelog.md).

---

## 0. Where it sits — three parts, one direction

```mermaid
flowchart LR
    V["the vibration calculation<br/>(engines/vibration.md)<br/>every mode: frequency, pattern L,<br/>masses, Q_zp, the record's values"]
    K["the motion toolkit<br/>(this chapter)<br/>a state for each mode →<br/>snapshots, one mode's frames,<br/>or the person's own frames"]
    S["a multi-frame structure<br/>(model/structure.md § 2.2f)<br/>frames + each frame's weight<br/>+ the maker's rows, as written"]
    T["a calculation over frames<br/>(engines/transport.md § 2a.9)<br/>each frame computed and kept,<br/>Σ weight × result"]
    V --> K --> S --> T
```

| part | owns | does not |
|---|---|---|
| **the vibration calculation** | computing the modes and recording every mode's data in the result ([`engines/vibration.md`](?doc=engines/vibration.md) § 6) | decide how the molecule moves |
| **the motion toolkit** | describing how the molecule moves — a state for each mode — and turning it into frames with weights, by the person's choice of modes, states, recipe and count; writing them as a frame set with rows that say what was done | run a calculation, average one, or choose the model for the person |
| **the frame set** | the frames, each frame's `weight`, and every other row its maker wrote, carried as written ([`model/structure.md`](?doc=model/structure.md) § 2.2f) | mean anything to the calculation but the weights |
| **a calculation over frames** | computing each frame as an ordinary static calculation, keeping every frame's result, and stating $\sum_f w_f\,X_f$ ([`engines/transport.md`](?doc=engines/transport.md) § 2a.9, § 2a.12) | know where the frames came from, or design, interpret or re-weight them |

**The dependency runs one way.** The toolkit reads the vibration's result
and writes a frame set; a calculation reads the frame set. Transport — its
record, its data file, its composition — imports nothing of the toolkit.
**Of a structure it cites, transport reads what it reads of every structure**
— its atoms, labels and box, frame 0's `info.calculation` (the settings it
takes its own from, required) and a relaxation record of its geometry
(`info.relaxation`) ([`engines/transport.md`](?doc=engines/transport.md)
§ 3.1) — and **none of the maker's own rows or `info` keys**: not
`vibration_run`, not `info.sampling`, not an older set's `info.vibration`,
nothing of § 5.1 but each frame's `weight`. The one check of the frames that
needs transport's own knowledge — that no electrode atom moves beyond the
leads' tolerance — stays transport's, asked when a calculation cites the set
([`engines/transport.md`](?doc=engines/transport.md) § 2a.9); the toolkit's
writer checks what it owns (§ 5). The pages, which sit above both, ask the
toolkit for a row's plain words and a cited set's provenance (§ 6).

**The model is the person's.** Which modes move, in which state, with which
recipe and how many frames, is a research decision; the toolkit gives each
piece correctly and lets them be combined, states every choice in the frames
it writes, and never decides one. Transport is the same calculation whatever
the frames are.

---

## 1. Where each piece lives, and what it reads

| module | layer | holds |
|---|---|---|
| `vibration` | L2, **travels** beside a job in both vibration bundles — numpy, the constants, the generic tools and the frequency-window rule it imports two ways (`oscillator`, `sampling`, `distribution`, `spectra.selection` — beside a job, the file `mode_selection.py` — its `TRAVELS_WITH`) | the mechanics of every mode, and the molecule's motion built from them: `state`, `State` (`pdf`, `cdf`, `moment`, `draw`, `frames`, `weights_at`), `Frames`, `Mode`, `ModeState` (each of `State`'s calls handed on, `draw` and `weights_at` among them, and its fields — `quanta`, `thermal_quanta`, `zero_point_amplitude_amu12_ang`, `classical_spread_amu12_ang`), `Vibration` — its `displaced({...})`, the one rule $\mathbf R^0 + \sum_k Q_k\mathbf L_k$ ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.11 § 2), and its `motion()` — the molecule's motion and its snapshots (`Motion`, `Snapshots`), the modes that move chosen in steps (`ModeSelection`, `v.mode_selection()`), the analyses of a snapshot set's results (§ 7), `bose_einstein_temperature` |
| `oscillator` | L2, travels with `vibration` — numpy, `math` and the constants, imported two ways | **one harmonic oscillator of a frequency** in its own unit $x = Q/Q_{\mathrm{zp}}$ (§ 9.1 module 1): `Oscillator(frequency_cm1)` — `zero_point_amplitude_amu12_ang`, `quantum_ev` ($\hbar\omega$), `bose_einstein_mean_quanta(T)` and `bose_einstein_temperature(mean_quanta)` (Bose–Einstein), `classical_spread(T)` in $x$ — the levels' wavefunctions, densities and averages (a level's distribution function is `distribution.Level`'s), the ladder and position operators on a window of levels; the mechanics every mode's numbers take, `zero_point_amplitude_amu12_ang`, `bose_einstein_mean_quanta`, `bose_einstein_temperature`, `spread_of_quanta`, `thermal_spread_amu12_ang`, `classical_spread_amu12_ang`, `checked_temperature_k`, which `vibration` hands on under their names, read there by the record's writers (`spectra.derived`, `spectra.normal_modes`), the frame-set doors (`vibration_frames`, `cli`) and the spectrum tab's door (`web.blueprints.spectra`) |
| `oscillator_state` | L2, numpy, `math` and `oscillator`, imported two ways — it travels with `vibration` once `vibration` builds on it (R3d2b2b4b, § 9.3) | **chains of steps, and states** (§ 9.1 module 2; [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 2e): the starts `ground`, `level`, `thermal`, `classical_thermal`; the events `add_quantum`, `remove_quantum`; `Chain.reduced()`, `Chain.on(oscillator)` → an `OscillatorState` — its list, its mean quanta and words |
| `distribution` | L2, travels with `vibration` — numpy, `math` and `oscillator`, imported two ways | **one coordinate's distribution** in $x$ (§ 9.1 module 3): `of_state(state)` by the routes of § 9.1, or a person's own (`Gaussian`, `QuantumAdded`, `Level`, `LevelList`); its density, distribution function, averages, quantile, Gauss rule, draws and weights for points placed elsewhere (`points_weights`, which `vibration`'s states ask) |
| `sampling` | L2, travels with `vibration` — numpy, `math` and `oscillator`'s whole-number rule, imported two ways | **the seed's keyed streams and the statistics of arrays** (§ 9.1 module 4): `normal_stream`, the snapshots' streams' one home (§ 3.3), which `vibration` draws through; a mean and its error (`mean_with_error`), independent or in pairs (`pair_means`), a paired set's even and odd parts (`even_odd_parts`) |
| `vibration_frames` | L2, the **host's** — it imports `vibration`, the structure, `frameset`, the result's reader and the run door, never transport; it imports with `molbuilder-jupyternb`'s packages, numpy beside molbuilder's own modules, so the example notebook runs it there ([`web/jupyter.md`](?doc=web/jupyter.md) § 7) | built: the rows' vocabulary (`ROWS`, § 5.1), a mode's own set's `assumptions` row (`assumptions_of_a_mode`, its thermal state in `thermal_state_words`) and a set's own assumptions as its maker states them, an older set's said for it (`assumptions_of_set`, § 6), the pages' words and provenance from it (`said_of`, § 6), the weights for a set's rows at a state (`weights_for_set`, § 7), **a run's folder read as a `Vibration`** (`vibration_of_run`, below), the one frame-set writer and its reader (`write_frame_set` → `Written`, `read_frame_set` → `Frames`, `Snapshots` or `FrameSet`, § 5), a mode's state's outermost move for any state (`outermost_move`, its rule `vibration.outermost_move_ang`) — asked by the web layer, `postprocess` and the command line; the doors' set — the command line's and the page's: a mode's own set in any state asked every time (`frameset_from_mode` → `ModeFrames`, `doors_state` and the page's ask `read_state_ask` — `STATE_ASK_KEYS`, refusals `StateRefused` — the frame counts they offer `OFFERED_NODES`) and what a state means for a mode before its set is written (`readout`, the load door's `frameset` block), its quanta said beyond `HARMONIC_LIMIT_EV`, § 6 |

**No call of `vibration` reaches `vibration_frames`**: the foundation's
module never depends on the module built on it. The writer and the reader are
functions of `vibration_frames`, which `molbuilder.postprocess` hands on — the
one import of a person's script ([`model/parse.md`](?doc=model/parse.md) § 3):
`from molbuilder.postprocess import vibration, write_frame_set,
read_frame_set` — beside `frame_rows`, `weights_for_set`,
`reaveraged` and `outermost_move`. `postprocess.reaveraged` joins two calls — the
toolkit's weights for a set's rows and transport's own averaging door — and
adds no rule of its own (§ 7).

**What `v` knows.** `v = vibration(path)` takes the result file — its JSON
or its `.nc`, one answer — or the vibration run's folder, read
through the run door as `read_steps` takes one
([`engines/vibration-postprocessing.md`](?doc=engines/vibration-postprocessing.md)
§ 7). It carries what every frame set needs, each under its own name; the
`Vibration` holds every field below. The result's reader fills the file's
— the modes, the shared fields, `v.structure` (the structure the result
states) and `v.read_path`; the run's — `v.run`, `v.relaxation_calculation` —
are filled where a run's folder is read, by `vibration_of_run` (`None`
else), and `v.result_sha256` at the read door, whichever file:

| attribute | what | from |
|---|---|---|
| `v.modes`, `v.mode(n)` | each mode — its number (`index_1based`), frequency, whether it is imaginary, its canonical eigenvector over the free atoms ($\sum_A m_A\lvert\mathbf L_A\rvert^2 = 1$), and the record's stored values ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.11 § 6) | the result's modes ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.3) — a result before schema 7, which states no masses, read with the table's, as its reader reads it |
| `v.free_atom_idxs`, `v.masses_amu`, `v.element_symbols`, `v.positions_ang`, `v.species_labels`, `v.regions`, `v.mode_temperatures_k` | what every mode shares — the free atoms, every atom's mass, element (its species label's name) and position at the geometry the modes were taken at, the labels and regions as written, and the modes' temperatures the file states its spreads at; a mode reaches each through its vibration (`m.masses_amu`, …) | the result's `equilibrium.masses_amu`, its structure, its `modes_thermal` |
| `v.structure` | the geometry the modes were taken at, in the user order, its labels, held atoms and box — frame 0, and every frame's atoms | the result's `structure` ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.10.1) |
| `v.record_temperature_k` | the vibration record's temperature — the one a mode's thermal or classical state given none takes, said so (§ 2) | `thermo.temperature_K` |
| `v.read_path` | the path read, as given | the read |
| `v.run`, `v.result_sha256` | the run's folder and **the JSON's sha256** — the `.nc` states it as `json_sha256`, so it is the same whichever file was read: the set's provenance rows (§ 5) | the run door; the file |
| `v.relaxation_calculation`, `v.relaxation_said` | frame 0's `info.calculation`, the settings a transport citation takes its own from ([`engines/transport.md`](?doc=engines/transport.md) § 3.1, § 2a.7) — **the record of the relaxation frame 0's geometry came from** *(the user, 2026-10-24, on the recommendation)*: where the vibration's own run relaxed it — its `relax` call ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 1) — that run's settings, through the run door's view (`runs.view_of`, `parse.contract.contract_of`); where the structure was stated already relaxed, the record that structure states — its `info.calculation`, written when it was saved from its optimization *(measured on AuBDTAu, 2026-10-24: its vibration ran at a 1×1×1 k-grid and 300 Ry, its relaxation at 5×5×1 and 400 Ry; a transport citing a set stamped with the vibration's settings defaulted to the cheaper ones)*; where there is none, `v.relaxation_said` says why, in words a set written of it says | the run door's view |

**A run's folder → its `Vibration`, one door**:
`vibration_frames.vibration_of_run(run_folder)` reads the run's result through
its reader and gives the run, the JSON's sha256, the structure and frame 0's
relaxation record, refusing by name what is no run of ours and a run with no
result (`FrameSetError`, naming the run's folder, the one form it reads) and
what the result's reader refuses — a result before schema 7 is read as that
reader reads it, with the table's masses. `molbuilder frameset from-mode` and
*Write frame set…* ask it, and `read_frame_set` reopens a set's vibration
through it (§ 5). **Either form → its `Vibration`, one door**:
`vibration_frames.vibration_of(path)`, which `postprocess.vibration` hands on,
tells a run's folder (to the run door) from a result file (either of its two,
to the result's reader, the JSON's sha256 with it) and refuses a path that is
neither naming both (`TWO_FORMS`): not there (`SpectraJsonNotFoundError`, a
`FileNotFoundError`), a file of another name (`SpectraJsonError`), a folder
that holds no run's result (`FrameSetError`). `vibration` never asks the run
door. A vibration not read from its run's folder — its result file, an export, a copy — writes a set with no
`vibration_run` and no `info.calculation`, said with the way on (read the run's folder); a PySCF deck's, which
records no relaxation, and a stated-relaxed structure that states none, write
one with its `vibration_run` and no `info.calculation`, said so.
Such a set is no transport citation until its relaxation is recorded — said,
never refused.

**Each object knows where it came from**: a `Mode` carries its result
(`m.vibration`, left out of the mode's repr and comparison, so `Vibration` →
`Mode` → `Vibration` never recurses), and `Frames` the state it was made for
(`f.state`, a `ModeState`), so the writer (§ 5),
`write_frame_set(path, f)`, reaches the run, the structure and the state's
rows; a `Frames` made from a bare `State`, of no mode, has no geometries and
is refused by name by the writer.

**Modes are named by their number**, `index_1based`, the number
the result, the spectrum tab and a frame set's `mode_index_1based` use,
wherever a call takes one: `v.mode(7)`, `with_mode(7, …)`, `modes=[7, 8]`,
`mode_selection().add([7])`, `displaced({7: q})`. `v.modes` stays the list in the
result's order, so `v.modes[6]` is mode 7 — the one place a Python index
appears.

**Every frequency scaled at once**: `v.with_frequency_scale(0.96)` — each
mode at 0.96 × its frequency, its pattern kept, the result's own frequency
kept beside it (`m.result_frequency_cm1`), the factor for the harmonic
picture's overestimate
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2b.1); the scaled vibration states the factor (`v.frequency_scale`), and
every set made from it (`frequency_scale`, § 5.1); scales compound, and a
total of 1 is the result's own modes, as read. One mode's own,
`m.with_frequency(...)`.

---

## 2. One mode's state — the building block

A state says where one mode's coordinate $Q$ is, as a probability density
$P(Q)$ ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2b.3, which also says which state fits which experiment). Each is asked of
a mode, `m.state(...)`, by one of these sets of arguments — one table in the
module (`vibration.STATES`, its check `vibration.checked_state`); any other set is refused, naming them:

| the state | arguments | where the molecule is | science |
|---|---|---|---|
| **thermal** — equilibrium at a temperature, or quanta added at random moments | `temperature_k`, or `mean_quanta` — the state's mean number of quanta, what an anti-Stokes/Stokes ratio measures | a bell curve of spread $Q_{\mathrm{zp}}\sqrt{2n+1}$, zero-point motion included | § 2, § 2a.1 |
| **n quanta exactly** | `quanta` | the oscillator's $n$-th standing wave, $\lvert\psi_n\rvert^2$ | § 2b.3 |
| **classical**, for comparison | `classical=True` with `temperature_k` | the classical bell curve, variance $k_BT/\omega^2$, no zero-point motion — at 0 K a point at rest | § 2b.3 |

`mean_quanta` has one meaning as an argument — a thermal state's mean
number of quanta — and a state's `.mean_quanta` is its mean always; a
classical state counts none (`None`).

**A temperature not given** — the user's rule
([`engines/vibration.md`](?doc=engines/vibration.md) § 6.11 § 9 item 6); this
is the rule's one table. A state says where its temperature came from, `s.temperature_from`,
in one of three words — `"given"`, `"the record's"`, `"the motion's"`
(`vibration.GIVEN`, `THE_RECORDS`, `THE_MOTIONS`; the weights for a set's
rows say `"the set's"`, `THE_SETS`), one function
(`vibration._temperature_of`) answering every call below, its refusal naming
what states none (*name a temperature: the record states none*, *… the set
states none*):

| call | a temperature not given |
|---|---|
| `m.state(...)`, `v.motion(...)`, `postprocess.outermost_move(...)` for a thermal state, or a classical one at a temperature | **the vibration record's** (`v.record_temperature_k`), said (`s.temperature_from`, `motion.temperature_from`: `"the record's"` — and each of a motion's states says it too; `outermost_move` answers a number, its state's as `m.state` takes it); a record stating none asks for one |
| `motion.with_mode(k, ...)` for a thermal state, or a classical one at a temperature | **the motion's** (`motion.temperature_k`), said (`s.temperature_from`: `"the motion's"`, and in `motion.set_apart_modes`) |
| `m.state(...)` for $n$ quanta, mean quanta | none is needed |
| the weights for a set's rows at a state (`vibration_frames.weights_for_set`, which `postprocess.reaveraged` asks, § 7), for a thermal or classical state | **the frame set's own `temperature_k` row**, as the data file carries it in its structure, said as the weights' first words, the state named (*the thermal state at the set's temperature, 300 K (its temperature_k row)*); a set stating none — of $n$ quanta, of mean quanta, a person's set — is refused by name, naming the temperature to give |
| `molbuilder frameset from-mode`, *Write frame set…* | asked every time — the page offers the vibration record's in its box, the person's to change |

**What a state answers**, each by the same module:

| call | answers |
|---|---|
| `s.pdf(q)`, `s.cdf(q)` | the density and the chance of $Q \le q$ — arrays in, arrays out |
| `s.moment(k)`, `s.spread_amu12_ang`, `s.mean_quanta` | $\langle Q^k\rangle$ exactly; its r.m.s.; its mean number of quanta |
| `s.atom_rms_ang` | each atom's r.m.s. move in the state |
| `s.draw(size, *, seed, start=0, mode=0)` — a mode's state, `ModeState.draw`, passes its own number | `size` coordinates drawn from $P(Q)$ — snapshots `start` … `start + size − 1` of the seed's sequence — from the mode's own stream of the seed, `numpy.random.default_rng([seed, mode, 0])`, `mode` the mode's number (0 for a bare `State` of no mode), drawn in one block. Every state turns one standard normal number $z$ per draw: a bell curve, its spread times it; $n$ quanta, the $Q$ where `cdf` reaches $\Phi(z)$, $\Phi$ the standard normal's distribution — so a mirror $-z$ is the mirror $-Q$ and a shared seed couples every state alike ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 5.5). The motion draws every mode through it (§ 3.3) |
| `s.frames(nodes)` | the mode's own deterministic frames (§ 4) |
| `s.weights_at(q_amu12_ang, *, written=None)` | the weights for this state of frames at `q_amu12_ang`, `written` the weights the frames were written with — a `Weights(weights, beyond, exact_to_degree, said)`: the degree they reproduce the state's averages to, and what is said of frames placed for another motion (§ 7) |

---

## 3. The molecule's motion, and its snapshots

### 3.1 A state for every mode

In the harmonic picture the modes move **independently** — the probability of
one set of coordinates $(Q_1, \dots, Q_N)$ is the product of each mode's own
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2c.1). So the molecule's motion is a state for each mode, each drawn on its
own:

```python
from molbuilder.postprocess import vibration, structure_dir
v = vibration(structure_dir("P", "spectrum", "V") / "01_freq/run-0")   # the run's folder (§ 1)
motion = v.motion(temperature_k=300.0)                  # every real mode thermal at 300 K
motion = motion.with_mode(7, quanta=1)                  # mode 7 one quantum exactly
motion = v.motion(temperature_k=300.0, modes=range(7, 31))   # only modes 7 to 30 move
chosen = v.mode_selection().window(50.0, None).add([3]).drop([12, 13])
motion = v.motion(temperature_k=300.0, modes=chosen)    # the modes chosen in steps
motion.states                                           # {mode number: its state}
motion.said                                             # what the motion says (below)
```

| choice | how | said where |
|---|---|---|
| which modes move | `modes=` — mode numbers, or a **mode selection built in steps**; every real mode when unsaid. `v.mode_selection()` starts as every real mode, and each step acts on what the step before left, in the order written: `window(low_cm1, high_cm1)` keeps the modes whose frequency is inside the window, either end `None` for open — the rule molbuilder's per-mode steps choose by (`spectra.selection`; an imaginary mode's frequency negative); `only([...])` keeps, of the modes the steps before left, only those listed; `add([...])` and `drop([...])` put modes in and take them out; `final()` the list, in the result's order. Which modes to trust — a soft mode from a held-atom calculation, say ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 2c.5) — is the person's: the selection only carries it out. An imaginary mode among them is left out, said; a selection or `modes=` holding no mode, or only imaginary ones, is refused by name — nothing to move; a number the result holds no mode of, a window backwards, and a selection of another vibration, refused by name | the set's `modes` row; the selection's steps in `info.sampling`'s `mode_selection` and in what the motion says |
| each mode's state | thermal at `temperature_k` (the record's when none, said), quantum — zero-point included; `classical=True` for the classical bell curves | `temperature_k`, `quantum` rows |
| a mode set apart | `with_mode(k, ...)`: any state of § 2 by its arguments, or a `ModeState` of mode $k$ — never both, a state and a state's arguments together refused by name; a mode at a scaled frequency (`m.with_frequency(...)`) reaches the motion so; a `ModeState` of another mode, or of another vibration — two vibrations are one by their content: the modes, the masses, the geometry, the free atoms and the frequency scale, so a second read of the same result is the same vibration — is refused by name; a thermal or classical state given no temperature takes the motion's, said (§ 2) — **replacing** that mode's thermal state: a mode is in **one** state, its own; a mode outside `modes=` is added. In a classical motion, `with_mode(k, temperature_k=600)` alone makes that mode classical at 600 K; every included mode set apart is a motion too | `set_apart_modes` (§ 5.1) |
| degenerate modes | **each mode is drawn on its own. A quantum two modes share is not treated by this path** — a photon raising a combination of a degenerate pair, or of two modes closer than the photon can tell apart: the two coordinates then move together, which a state per mode loses. A degenerate pair is set up by hand with the generic tools — turned to the light's axes, the combination the photon raises drawn excited and the one across it thermal, then turned back ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 2c.2) — the frames written as a person's own (§ 3.4); two modes of nearby frequencies need a joint state over both modes' levels, which no tool here builds | — |
| a classical motion with a mode set apart | the mode set apart keeps its own state (§ 2) — `with_mode(k, quanta=1)` stays quantum; the others are classical | `quantum` (the background's), each set-apart mode's own in `info.sampling` |

`with_mode` returns a new motion, as `Mode.with_frequency` returns a new
mode. A held atom never moves: the modes are over the free atoms
([`engines/vibration.md`](?doc=engines/vibration.md) § 6.3), and the
electrodes' own vibrations are not among them.

**`motion.said`**, shown in its repr, summary sentences:

- how the modes were chosen, where a selection chose them — its steps in
  order (*every real mode; 50 cm-1 and up; mode 3 added; modes 12 and 13
  dropped*);
- the modes left out as imaginary — those still out of the motion, so a mode
  added later at a measured frequency (`with_mode(k, state=...)`) is no longer
  said left out;
- the temperature taken from the record;
- a mode set apart again, the later call's state replacing the earlier.

### 3.2 A snapshot, and a geometry from several modes

One snapshot is one possible geometry of the molecule. Each mode's coordinate
is drawn from its own state, and every atom moves by all of them at once:

$$
Q_k \sim P_k(Q) \ \text{for every mode } k, \qquad
\mathbf R_A = \mathbf R_A^0 + \sum_k Q_k\,\mathbf L_{k,A} .
$$

The motion composes each mode's draw from the state's own (`s.draw`, § 2)
with the numbers § 3.3's streams give it
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2c.3, the steps worked through to the average).

**The geometry for coordinates the person gives** is the same
rule, one call of the mechanics module: `v.displaced({7: 0.12, 9: -0.05})` —
every atom at $\mathbf R^0 + \sum_k Q_k\mathbf L_k$, the modes by number, $Q$
in amu^½·Å. It is what every recipe is built on, and what a person's own
model uses (§ 3.4).

### 3.3 Many snapshots — the frames, each weighted equally

```python
from molbuilder.postprocess import write_frame_set, topic_dir
snaps = motion.sample(64, seed=20261123, antithetic=True)
snaps.mode_index_1based          # (n_modes,): the columns' modes, in the result's order
snaps.q_amu12_ang                # (65, n_modes): frame 0 (all zero), then each snapshot's coordinates
snaps.atom_moves_ang             # (65, n_atoms, 3): each atom's move — the displacement map
snaps.positions_ang              # (65, n_atoms, 3): each frame's geometry, frame 0 the relaxed one
snaps.max_atom_displacement_ang  # (65,): each frame's largest single-atom move
snaps.max_atom_mode_index_1based # (65,): the mode moving that atom most (0 for frame 0)
snaps.weights                    # (65,): 0 for frame 0, then 1/64 each
snaps.said                       # summary sentences (below)
written = write_frame_set(topic_dir("P", "structure") / "bdt-300K-64.xyz", snaps)  # § 5
```

`atom_moves_ang[i]` is every mode's `Mode.displacement` at frame $i$'s
coordinate, summed — $\sum_k Q_k\mathbf L_k$ — and `positions_ang[i]` that
added to frame 0.

| choice | rule | why ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)) |
|---|---|---|
| how many, `n` | the person's; with pairs, even — an odd `n` with `antithetic=True` is refused by name, the two settings conflicting; `sample(2, antithetic=True)` is one pair, whose error has nothing to be taken from, said | the average's error falls as $1/\sqrt N$; the spread of the results says how many are needed, and a small effect needs many (§ 5.5) |
| `seed` | **required**, a whole number, 0 or more, written with the frames. **One stream per mode, keyed by the mode's number**, never its place among those included: `numpy.random.default_rng([seed, mode, 0])` its normal numbers; `[seed, 0, j]`, `j` 1 or more, is no mode's (no mode is numbered 0, and a bare state of no mode draws `[seed, 0, 0]`), free for a person's own draws. **Each stream is drawn in one block** | a `modes=` subset or a mode set apart leaves every other mode's draws as they were — so a set and its variation in one mode, one seed, differ by that mode's draw alone (§ 5.5); the same seed gives the same frames **on the same numpy** (numpy keeps its bit streams across versions, not every distribution's algorithm), so the set states the generator and numpy's version (`rng`) |
| `antithetic` | **the person's explicit choice, no default**: on, each snapshot is followed by its mirror, every normal number $z \to -z$, so every $Q_k \to -Q_k$ | within each pair the odd part of a response cancels exactly; whether that helps the set depends on which part dominates, which a paired set's own results say (§ 5.5) |
| `start` | snapshots `start` … `start + n − 1` of the seed's sequence, 0 when unsaid — even with pairs | **more frames later**: the streams are drawn in blocks, so `sample(2N)` begins with `sample(N)`'s frames for every kind of mode, pairs included, and `sample(N, seed=s, start=N)` draws the next $N$, written as their own set: two sets, their two calculations' results put end to end and averaged as § 7 says (`sampling.mean_with_error`, from pair means where paired) |

Each snapshot's weight is $1/N$: the probabilities are in how the
snapshots were drawn, not in the weights. **One indexing rule**: a
`Snapshots`, like a `Frames`, stands for the set as written — in memory as
read back, item $i$ is frame $i$, frame 0 the relaxed geometry at weight 0
(§ 5) included. The snapshot drawn $j$-th (from 0) is frame $j + 1$; with
pairs, frames $2j-1$ and $2j$ are a pair ($j = 1 \dots N/2$). Every analysis
of § 7 takes **one value per frame, frame 0 first** — a calculation's data
file's values whole — and leaves out the reference itself, which the object
knows by its weight 0; the person never slices.

**What is said with them, never refused** (`snaps.said`, shown in its repr):
summary sentences — the largest single-atom move over the set, its snapshot
and the mode moving it, beside each mode's spread; a soft mode at a high
temperature can push an atom further than the harmonic picture holds
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 7), which the person reads against a bond length — the arrays above give
every snapshot's.

### 3.4 A person's own model

The recipes are two of many. A two-mode grid to see a cross term, one
mode's own frames on top of snapshots of the rest, frames from a model of the
person's own — each is built from the two calls every recipe uses: the
geometry (`v.displaced`, § 3.2) and the writer (§ 5). A
two-mode set by the product of the two modes' own Gauss rules, exact to
degree 5 in each mode, its centre first:

```python
from molbuilder.postprocess import write_frame_set, topic_dir
f7 = v.mode(7).state(temperature_k=300.0).frames(3)
f9 = v.mode(9).state(temperature_k=300.0).frames(3)
pairs = [(a, b) for a in f7.q_amu12_ang[:, 0] for b in f9.q_amu12_ang[:, 0]]
grid = [v.displaced({7: a, 9: b}) for a, b in pairs]          # (0, 0) first
w = [wa * wb for wa in f7.weights for wb in f9.weights]       # (0, 0) at 4/9
write_frame_set(topic_dir("P", "structure") / "bdt-m7xm9.xyz", grid, w, vibration=v,
                rows=[{"name": "sampling_rule", "value": "gauss-hermite x gauss-hermite",
                       "note": "modes 7 and 9 thermal at 300 K, each its own 3-point rule"}],
                frame_rows=[[{"name": "q7_amu12_ang", "value": a,
                              "note": "position along mode 7"},
                             {"name": "q9_amu12_ang", "value": b,
                              "note": "position along mode 9"}] for a, b in pairs])
```

The writer adds what every set needs (§ 5) whatever made the frames: the
provenance, the masses, frame 0's `info.calculation`, each frame's measured
rows. Its first frame is the relaxed geometry here, so no frame 0 is added.

---

## 4. One mode's frames — a deterministic set

For one mode on its own — how much exciting this mode changes the result — a
few deterministic frames do far better than snapshots: the **Gauss rule of
the state's own distribution**, $m$ frames that reproduce the state's
averages $\langle Q^k\rangle$ for every $k \le 2m-1$
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2a.3, the steps and the three- and five-frame formulas worked; § 2b.4) —
every state's, in the order below; the writer is § 5's:

```python
f = v.mode(7).state(temperature_k=300.0).frames(nodes=3)
f.q_amu12_ang, f.weights          # (3, 1): (0, +q1, -q1); (2/3, 1/6, 1/6) for a bell curve
f.positions_ang                   # (3, n_atoms, 3): each frame's geometry, R + Q L
f.max_atom_displacement_ang, f.said, f.exact_to_degree
written = write_frame_set(topic_dir("P", "structure") / "bdt-m7-300K.xyz", f)  # § 5
```

| state | the rule |
|---|---|
| thermal | Gauss–Hermite: three frames at $0, \pm\sqrt3\,\sigma$, weights $\tfrac23, \tfrac16, \tfrac16$ |
| $n$ quanta | the Gauss rule of $\lvert\psi_n\rvert^2$ — a weight the frame's share of the average, not the chance of finding the molecule there (science § 2b.4) |

**One mode a set** (D6, [`engines/vibration.md`](?doc=engines/vibration.md)
§ 5.10's decisions): another mode is another set, cited by its own
calculation, and how the modes' results are weighed against each other is the
person's; many modes at once are snapshots (§ 3), or a model of the person's
own (§ 3.4). **A thermal set is stated by its temperature**
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5, with the user's ruling of 2026-10-28) — the junction's, or for a mode
pumped at random moments its effective one, `vibration.bose_einstein_temperature(
frequency_cm1, mean_quanta)` turning a measured number of quanta
into it. **What the frames move is said**: each frame's largest single-atom
move, and the outermost frames' on the set's rows — the check a person reads
against a bond length; `postprocess.outermost_move(path, *, mode,
nodes=None, temperature_k=None, quanta=None, mean_quanta=None,
classical=False)` gives it for any state before a set is
written — § 2's arguments, `nodes` any whole number (3 when unsaid), asking
the toolkit's `vibration_frames.outermost_move_of(v, mode, *, nodes=None,
**state)` (a mode named by its whole number; an imaginary mode's `None`; a
refusal `FrameSetError`), of a state `vibration_frames.outermost_move(s,
nodes=None)`, their one rule `vibration.outermost_move_ang`
($\lvert q_{\mathrm{out}}\rvert\max_A\lvert\mathbf L_A\rvert$, the outermost
node of the state's own rule), the set's `max_displacement_ang` row the
same number.

**The order and the symmetry**: the node at $Q = 0$ first,
then each pair nearest first, its positive side first — $(0, +q_1, -q_1,
+q_2, -q_2, \dots)$ — one order for every set. Every state the toolkit offers
has an even $P(Q)$, so each rule is made exactly symmetric: an odd rule's
middle node is 0 exactly and each pair $\pm q$ exactly, whatever the
eigensolver returns — $q = (x_+ - x_-)/2$ from the pair's two nodes, the
pair's two weights averaged — and frame 0 of an odd set is the relaxed
geometry bit for bit. An even rule's `Frames` stands for its set as written
(§ 3.3's rule): the relaxed geometry as frame 0 at weight 0, then the rule's
nodes (§ 5).

**The excitation's own effect** is a difference of two states at one
temperature ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 6, with the numbers): best from two sets, each by its own rule; one set's
results re-weighted to the other state (`weights_at`, § 7) are exact only to
degree $m - 1$ — $m$ for an odd number of symmetric frames and a symmetric
state — so that route needs five frames at least.

---

## 5. The frame set it writes

**One writer**, a function of `vibration_frames` handed on by
`postprocess`, in two forms:

- `write_frame_set(path, frames)` for the recipes — `frames` a `Frames` or a
  `Snapshots`, which bring their vibration, their state and their rows;
- `write_frame_set(path, positions_ang, weights, *, vibration, rows=(),
  frame_rows=(), frame0_rows=None)` for a person's own frames — `rows` a
  list of `{name, value, unit, note}`, the structure model's row,
  `frame_rows` one such list per frame given
  ([`model/structure.md`](?doc=model/structure.md) § 2.2d–2.2f:
  `set_customized`, `with_frames`), and `frame0_rows` the person's rows for
  a frame 0 the writer prepends; each row's unit in one place
  ([`model/structure.md`](?doc=model/structure.md) § 2.2d: blank where its
  name carries it, any spelling for a number whose name carries none — `"1"`
  for a count — none for a text). A person's
  frame row — in `frame_rows` or `frame0_rows` — named like one of the
  writer's own — `weight`,
  `displacement_ang`, `max_atom_displacement_ang` — or a person's structure
  row named like its provenance rows — `vibration_run`,
  `vibration_result_sha256`, `mass_source`, `frequency_scale` — is refused
  by name. So, by name (`FrameSetError`), are a pair already at the path
  (asked before anything is read), frames of a bare state, a set read back
  without its vibration (given its vibration, it is written again), a
  person's set read back (a `FrameSet`: written again as a person's frames,
  its positions and weights, leaving out the writer's own rows), a
  person's frames given without their vibration or their weights, or with
  weights or frame rows not one per frame, a row that is no mapping
  `{name, value, unit?, note?}` (in `rows`, `frame_rows` or `frame0_rows`),
  a set of one frame (a frame set holds two frames or more,
  [`model/structure.md`](?doc=model/structure.md) § 2.2f: one frame is the
  relaxed geometry alone, the structure itself), and a recipe's frames given
  the weights, vibration or rows it brings. A vibration read from its result
  file rather than its run's folder writes a set with no `vibration_run`,
  said with the way on — read the run's folder, `vibration(run_folder)`.

Both go through the structure's one writer. `path` is a path as Python takes
one — from the projects tree's doors (`topic_dir`, `structure_dir`), so the
set lands in the structure topic a transport citation reads and a mistyped
topic is refused by name; the command line and the page read theirs from the
projects tree (§ 6). It returns `Written(path, n_frames, said)`: the pair
written, its frame count — a prepended frame 0 among them — and what it said.
A recipe's object already stands for the set as written (§ 3.3), so its
items are the file's frames one for one; a person's frames given without the
relaxed geometry first are the file's frames 1 … after the prepended one.

| it writes | how |
|---|---|
| **frame 0, the set's reference** ([`model/structure.md`](?doc=model/structure.md) § 2.2f) | the relaxed geometry — this rule's one home: a mode's own set with an odd rule has its node there, at the rule's weight; an even rule's set and a snapshot set get it prepended at weight 0; a person's frames keep their first frame as frame 0 where it **is** the relaxed geometry as a pair holds positions — within the structure's one rule of two places (`structure.SAME_PLACE_ANG`, $10^{-6}$ Å: a pair's six decimals, so a set read back and written again is the set it was; the user, 2026-11-24) — the relaxed geometry written there exactly, as `v.displaced({})` or a node at $Q = 0$ gives it (§ 4: every rule's middle node is 0 exactly); a first frame farther off is no frame 0 and gets it prepended at weight 0. Exact in memory; the file holds every position to $10^{-6}$ Å, as every pair does |
| **a prepended frame 0's rows** | for a recipe, every frame row its other frames state, at its value there: `q_amu12_ang` 0 (each mode's, for snapshots), `node_sigma` 0, `displacement_ang` 0, `max_atom_displacement_ang` 0, `weight` 0; for a person's frames, the writer's own measured rows computed there, `weight` 0, and the person's `frame0_rows` — a person's frame row absent at frame 0 said |
| **each frame's `weight`**, the one row a calculation reads | as given; weights not summing to 1 within the container's tolerance written as given and said ([`model/structure.md`](?doc=model/structure.md) § 2.2f) |
| **each frame's measured rows**, for every set | `displacement_ang`, `max_atom_displacement_ang` — computed from the coordinates alone |
| the provenance, as rows | `vibration_run`, `vibration_result_sha256` (the JSON's, § 1), `mass_source`, `frequency_scale` where one was applied |
| **the maker's assumptions**, one row | `assumptions`: the harmonic modes, each mode's state, their independence, for snapshots the statistics of a finite set — science § 7's items, the sentences it asks to be stated beside the numbers, carried as written by any calculation like every row; a person's set carries the person's own `assumptions` row, or none, said |
| the masses | the pair's `mass_amu` channel, every atom's |
| what made the frames, as rows with their meanings | § 5.1's; for a set the person built, the rows and frame rows the person gives |
| per-frame coordinates along every mode | `info.sampling`, from the snapshot recipe only — everything that made its frames: `modes` (included), `temperature_k`, `temperature_from` and `classical` (the background), `frequency_scale`, `states` (each mode's state by `vibration.State`'s fields, with its `temperature_k`, `temperature_from` and `frequency_cm1`), `set_apart`, `mode_selection` (the steps that chose the modes, where a selection chose them), `left_out`, `notes`, `antithetic`, and `q_amu12_ang`, each frame's coordinate along each mode, indexed as the set is written (frame 0's all zero); read back by `read_frame_set`, its motion rebuilt whole |
| frame 0's `info.calculation` | `v.relaxation_calculation` (§ 1) |

The pair's `info` and channels travel with it wherever a structure travels —
into a composed junction, and with the junction's structure into a transport
calculation's sidecar and data file (`structure_json`), cited whole as the
structure ([`model/structure.md`](?doc=model/structure.md) § 2.2f).
Transport reads of it what § 0 says, and of the maker's nothing.

**What it checks is what it owns**: the same atoms, order, labels and box in
every frame, and every held atom where frame 0 has it — true by
construction for the recipes, every $\mathbf L$ zero on every held atom; a
person's frames moving a held atom are written and said, never refused.
Which atoms are the leads is transport's to know: a calculation citing the
set says which electrode atoms its frames move beyond the leads' tolerance
([`engines/transport.md`](?doc=engines/transport.md) § 2a.9), running as set
— the writer never says it.

**A mode at a scaled frequency** (`m.with_frequency(...)`,
`v.with_frequency_scale(...)`) writes the frequency and $Q_{\mathrm{zp}}$ its
frames were placed for, and the result's own frequency beside them
(`result_frequency_cm1`, § 5.1).

One pair, `.xyz` and `.molstruct.json`; **a pair already at the path is
refused**, never written over: a calculation citing it is pinned to it.

**Reading a set back** — `read_frame_set(path, *, vibration=None)` — returns the set **as written, every frame, frame 0 included**: a
`Frames` (a mode's own set; its `state` rebuilt from its rows and its
vibration — the one given, or the run its `vibration_run` names reopened
through `vibration_of_run` (§ 1); `None` where that run does not read, said),
a `Snapshots` (a sampled set, from `info.sampling`), or a `FrameSet` (a set a
person built: its positions, weights and rows as written). Read back as in
memory, item $i$ is frame $i$ (§ 3.3). **One shape for all three**: `q_amu12_ang` an
array (frames × modes — one column for a mode's own set),
`mode_index_1based` (modes,), `positions_ang` (frames, atoms, 3), `weights`,
`max_atom_displacement_ang`, `max_atom_mode_index_1based` (frames,), `said`;
`exact_to_degree` on `Frames`, `seed`, `antithetic` on `Snapshots`. A set
is a `Snapshots` only where it holds `info.sampling`, a `Frames` only where
it states the toolkit's own rule (§ 4) — `sampling_rule` `gauss-hermite` or
`gauss` and `sampling_nodes`, its degree $2m - 1$ — its mode's number,
frequency and zero-point amplitude read, and every frame states its
coordinate; any other — a person's set written in the vocabulary's names —
a `FrameSet`, said. A toolkit set whose rows name a state the table of § 2
refuses is read without its state, said. A vibration given that is no
vibration, or not the set's — its result's sha256 not the set's
`vibration_result_sha256`, or an older set's `info.vibration` — is refused by
name, whichever kind the set is, as a motion refuses another vibration's
state (§ 3.1); a vibration given at another frequency scale than the set's
is brought to the set's (its `frequency_scale` row); a run reopened whose
result is not the set's reads the set without it, said with the way on —
give the vibration it was written from. A mode's own set read back takes the
temperature from its own row (`s.temperature_from`, `"the set's"`). Read
without its vibration — none given, and the run its `vibration_run` names
not read, or none named — a set still gives its geometries, weights, moves
and analyses, its words saying why it has no vibration (first in `said`). A set that states no
frame set's rows, or a pair that does not read, is refused by name
(`FrameSetError`).

### 5.1 The vocabulary, row by row

The vocabulary is the toolkit's — one table in `vibration_frames` (the names,
kinds, units, meanings and page words) — and a person's script may write the same
names. Every row the toolkit writes carries its meaning as its `note`, the
meaning for the kind of set it is. A row's unit is said in one place
([`model/structure.md`](?doc=model/structure.md) § 2.2d): a name that carries
its unit (`frequency_cm1`, `temperature_k`, `displacement_ang`,
`zero_point_amplitude_amu12_ang`) leaves `unit` blank, and a number whose name
carries none states `1` in it (—); each row's unit in the table is its
name's (`Row.unit`, `units.unit_of_name`), as a page shows it
([`model/datafiles.md`](?doc=model/datafiles.md) § 2c's last column,
`Row.page_unit`). A set an earlier molbuilder wrote
is read as written — its counts stating no unit, a unit-named row's unit
written again (not kept) — and read the same. A page says a row in
the vocabulary's words where its unit is the vocabulary's **or it states
none** — the name itself is a claim about the unit — and otherwise by its
name and unit as written; the calculation checks nothing of it, and a row the vocabulary
names, written without a meaning, is said so by the toolkit when it reads its
own sets. Two rows are prose — the maker's `assumptions` and a snapshot set's
`set_apart_modes` — marked so in the table (`Row.prose`), and a page draws
each on a line of its own ([`engines/transport.md`](?doc=engines/transport.md)
§ 3.8.1).

| where | name | kind | meaning | unit | on a page |
|---|---|---|---|---|---|
| the `.xyz` | every frame's coordinates | given | frame 0 the relaxed geometry the modes were taken at, the set's reference; frames 1… displaced along the modes | Å | — |
| the `mass_amu` channel | each atom's mass | given | the masses the modes' coordinates are weighted by — the vibration result's own, `equilibrium.masses_amu`, which the vibration's deck stated and its analysis weighted the Hessian by; every atom, a positive finite number | amu | — |
| the structure's rows | `vibration_run` | given | the vibration run the modes were read from — its folder in the projects tree; absent for a vibration not read from its run's folder, said | — | Vibration run |
| | `vibration_result_sha256` | given | the sha256 of that run's result JSON | — | Result sha256 |
| | `mass_source` | given | which masses the `mass_amu` channel holds, by name, and so which $Q$ is measured with; `standard atomic weights`, a molbuilder vibration's: the isotope-averaged standard atomic weights (`chemistry.atomic_mass`) the vibration's deck stated, as its result states them; every $\mathbf L$ is normalised with them, $\sum_A m_A \lvert\mathbf L_A\rvert^2 = 1$ | — | Masses |
| | `frequency_scale` | given | the factor every mode's frequency was scaled by (`v.with_frequency_scale`), each pattern kept; absent when none | — | Frequency scale |
| | `assumptions` | given | the maker's assumptions, in words — what the weights' average holds under: harmonic modes, each mode's state, the modes' independence, for snapshots every mode not included at the relaxed geometry (an imaginary one named: no harmonic state) and a finite set's statistical error ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 7); a person's set, the person's own | — | Assumptions |
| | `sampling_rule` | given | the recipe, by name, that placed the frames and gave their weights: `gauss-hermite` — a bell curve's Gauss rule, a thermal or classical state's; `gauss` — another state's own Gauss rule ($n$ quanta), its state in the rows below; `monte-carlo` — snapshots of every included mode drawn from its state, each frame of weight $1/N$; a set the person built, the person's words | — | Sampling rule |
| | `sampling_nodes` | given | a mode's own set: $m$, the rule's number of nodes, one frame at each (for example 3: a thermal set's $q = 0, \pm\sqrt3\,\sigma$, weights $\tfrac23, \tfrac16, \tfrac16$) | — | Nodes |
| | `samples` | given | a snapshot set: $N$, the snapshots drawn — the frames after frame 0 | — | Snapshots |
| | `seed` | given | a snapshot set: the seed, a whole number 0 or more, each mode's stream is keyed by (with the mode's number) | — | Seed |
| | `start` | given | a snapshot set: the first snapshot of the seed's sequence drawn — 0, or where a set continues another (`sample(..., start=)`), its snapshots `start` … `start + samples − 1` | — | First snapshot |
| | `rng` | given | a snapshot set: the generator, `numpy PCG64, SeedSequence([seed, mode, 0])`, and numpy's version — the same seed gives the same frames on the same numpy | — | Random generator |
| | `antithetic` | given | a snapshot set: 1 when each snapshot is followed by its mirror (frames $2j-1$ and $2j$ a pair), 0 when every snapshot is drawn alone | — | Mirrored pairs |
| | `modes` | given | a snapshot set: the modes included, by number ("every real mode" when all) — how a selection chose them in `info.sampling`'s `mode_selection` | — | Modes |
| | `set_apart_modes` | given | a snapshot set: each mode given a state of its own, in words — its state and its frequency where it is not the result's, as `Motion.set_apart_modes` says them, e.g. "mode 7: 1 quantum exactly; mode 7 at 672 cm-1, the result's 700; mode 15: thermal at 77 K (the motion's)"; absent when none | — | Modes set apart |
| | `quantum` | given | 1 when the set's motion is quantum — zero-point motion included — 0 when classical: a mode's own set its state's; a snapshot set its background's, where a mode not set apart keeps it (absent where every mode is set apart), each set-apart mode's own in `info.sampling` | — | Quantum |
| | `mode_index_1based` | given | a mode's own set: which mode, counted as the vibration result counts it (`index_1based`) | — | Mode |
| | `frequency_cm1` | given | a mode's own set: the wavenumber its frames were placed for — the result's own, or a scaled one, the result's then in `result_frequency_cm1` | cm⁻¹ | Frequency (cm⁻¹) |
| | `result_frequency_cm1` | given | a mode's own set at a scaled frequency: the mode's wavenumber as the result states it; absent when `frequency_cm1` is the result's | cm⁻¹ | Result's frequency (cm⁻¹) |
| | `zero_point_amplitude_amu12_ang` | given | a mode's own set: the zero-point amplitude $Q_{\mathrm{zp}} = \sqrt{\hbar/2\omega}$ at `frequency_cm1` | amu½·Å | Zero-point amplitude (amu½·Å) |
| | `temperature_k` | given | the temperature of the motion the frames sample: a thermal state's — for a mode pumped at random moments, its effective temperature, the one whose thermal state holds the same mean number of quanta ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 2a.1; `vibration.bose_einstein_temperature`); a snapshot set's background, where a mode not set apart keeps it. Absent for a set named by its mean quanta, or of $n$ quanta | K | Temperature (K) |
| | `quanta` | given | a mode's own set of $n$ quanta exactly: $n$ | — | Quanta n |
| | `mean_quanta` | derived | a mode's own set: the state's mean number of quanta — at a temperature $n = 1/(e^{\hbar\omega/k_BT} - 1)$; given, for a set named by it; absent for a classical state, which counts none | — | Mean quanta n |
| | `spread_amu12_ang` | derived | a mode's own set: the r.m.s. of the state the frames sample, $\sigma = Q_{\mathrm{zp}}\sqrt{2n+1}$ for a quantum state ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 2a); a set written before 2.81.0 names it `sigma_amu12_ang`, read the same | amu½·Å | Spread σ (amu½·Å) |
| | `max_displacement_ang` | measured | a mode's own set: the outermost move — $\max_A\lvert\Delta\mathbf R_A\rvert$ at the outermost frames, $= \lvert q_{\mathrm{out}}\rvert\max_A\lvert\mathbf L_A\rvert$ for any state — the check against a bond length | Å | Outermost move (Å) |
| each frame's rows | `displacement_ang` | measured | the frame's plain displacement from frame 0, one number per frame, $\sqrt{\sum_A\lvert\Delta\mathbf R_A\rvert^2}$ over every atom — no masses; the atoms' move vectors themselves are the toolkit's `atom_moves_ang` | Å | Displacement (Å) |
| | `max_atom_displacement_ang` | measured | its largest single atom's, $\max_A\lvert\Delta\mathbf R_A\rvert$ | Å | Largest atom move (Å) |
| | `q_amu12_ang` | derived | a mode's own set: the frame's position along the mode, the mass-weighted normal coordinate — one number for the whole mode, every atom displaced by $q\,\mathbf L_A$: its size $\sqrt{\sum_A m_A\lvert\Delta\mathbf R_A\rvert^2}$, its sign the side of frame 0 it sits on; a node of the rule, or 0 for a prepended frame 0. A snapshot set's per-mode coordinates are its `info.sampling`'s | amu½·Å | Position along the mode, q (amu½·Å) |
| | `node_sigma` | derived | a mode's own set: the same position in units of the state's spread, $q/\sigma$ | — | q in units of σ |
| | `weight` | given | the frame's share of the average: `Σ_f weight_f X_f` is the recipe's estimate of the average of `X` over the motion the set samples — a Gauss rule's weight at its node, $1/N$ for a snapshot, 0 for a prepended frame 0. A share of the average, not the chance of finding the molecule at the frame ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 2b.4) | — | Weight |

*On a page* is how a page says the row: the plain words, the stored name on
hover ([`engines/transport.md`](?doc=engines/transport.md) § 3.8.1). *Given*
is stated by the vibration result, the person or the recipe; *measured* is
computed from the coordinates alone; *derived* is computed from the others by
the formula shown. A set written before 2.81.0 states `info.vibration`
(`run`, `result_sha256`) where a new one writes the two provenance rows, and
names its spread `sigma_amu12_ang`; the toolkit reads either, and those are
the marks it tells such a set by (§ 6).

---

## 6. Who calls it

| caller | what it asks |
|---|---|
| a person's script or notebook | any of it, through `molbuilder.postprocess` |
| `molbuilder frameset from-mode RUN --mode M (--temperature T \| --mean-quanta N \| --quanta N) [--nodes 3\|5\|7] --out PAIR.xyz` — `RUN` the vibration run's folder, read by `vibration_of_run` (§ 1), and `PAIR.xyz` the pair, each read from the projects tree as `--bundle` is; it prints the mode's state in the set's words, its mean quanta and $\sigma/Q_{\mathrm{zp}}$, each frame — named from 1 — its $q/\sigma$, largest atom move and weight, and where the pair is | a mode's own set in any state of § 2, asked every time — the user's decision ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.11 § 9 item 2) — through the doors' set, `vibration_frames.frameset_from_mode`: the frame count one of `OFFERED_NODES` (three, five or seven), the state named — one set of the state table's, never a classical one, each value by the table's rule (`doors_state`, `vibration.checked_state`), never the record's temperature assumed — the pair's name, each refused by name before the run is read, a state refused a usage error (exit 2); many quanta — beyond 1 eV in the mode (`HARMONIC_LIMIT_EV`, the user's choice of 2026-11-24) — said in a note, the set written as asked |
| the spectrum viewer's *Write frame set…* on the selected mode: its state, the number of frames, and the readout under them ([`web/spectra.md`](?doc=web/spectra.md) § 4.3, the page's own design) | a mode's own set in any state (§ 4) through the doors' set, `frameset_from_mode`, its run read by `vibration_of_run`; its readout, the load door's, `readout` for that state (asked as `frameset_state`, read by `read_state_ask`); many quanta — beyond 1 eV in the mode, ⟨n⟩ h c ν̃ (`HARMONIC_LIMIT_EV`, the user's choice of 2026-11-24) — said in its note and asked to confirm before writing, never refused |
| the web layer — the Transport tab's frame-set card and a calculation's record, through the transport blueprint; Task setup's card, through the build blueprint's folder answer; the Results tab's frame-set card, through the record's answer — each through the blueprints' one helper (`_shared.frame_set_said`) ([`engines/transport.md`](?doc=engines/transport.md) § 3.8.1; [`web/results.md`](?doc=web/results.md) § 2.5) | `vibration_frames.said_of(pair_path, recorded_files=None, *, structure=None)`: of `structure`, the frames the page draws — a calculation's composed junction on the Results tab and in the reuse answer, the citation as read at the describe door — else of the pair at `pair_path` read now (Task setup): each row's plain words, where its unit is the vocabulary's or it states none, and the set's provenance — its `vibration_run` and `vibration_result_sha256` rows, or an older set's `info.vibration`; given the `files` mapping a calculation's record states — the `.xyz`'s and the `.molstruct.json`'s sha256 — the pair at `pair_path` is read only for whether its files are as pinned: either not (changed since, gone, or never pinned: `workingcopy_structure.pair_files_changed`, the one comparison, which a transport calculation's composition refuses on) is said with its way on — *the cited pair has changed since this calculation cited it — set.xyz is not as its record pinned: the frames shown are the ones the calculation ran; cite the pair as it is now in a new calculation, or put it back* — so a pair moved or deleted after the run takes nothing from the card's words; what the set is (*frames of one mode*, *frames, the relaxed geometry and its snapshots*), by the one rule `weights_for_set` asks too — a snapshot set by its `samples` row or its `monte-carlo` rule, a mode's own by its mode's number, or its frequency and zero-point amplitude; for a mode's own set written before 2.81.0 — told by its own marks, `sigma_amu12_ang` or `info.vibration` (§ 5.1), every one a thermal set — which states no `assumptions` row, its harmonic assumptions in the words a set states them now, at its temperature (`assumptions_of_set`, the one function `postprocess.reaveraged` asks too; science § 7: never dropped), and nothing for a person's set that states none, its state not known; a pair that is not there or does not read, where it is read, said with its way on (*the cited pair set.xyz is not there now: its rows are shown without their words or where they came from -- put it back, or cite the pair as it is now in a new calculation*); a vocabulary row written without its meaning, said once, in its page words (*the set states no meaning for Displacement: it is shown with the vibration frame sets' own*). This row is the one home of what is said of an older set and of a changed pair; the other chapters point here. Answered `{structure, frames, what, provenance, assumptions, said}` — `structure` and `frames` each row's `{words, page_unit, kind, means, prose}` by name, only the rows the vocabulary speaks for, `means` the set's note else the vocabulary's, `prose` a row whose value is sentences (§ 5.1); `provenance` `{run, result_sha256, stated_as, rows}`, `rows` the rows it was read from, so a page says them once, on its provenance line; `assumptions` `{words, value}` — the vocabulary's words for the row (*Assumptions*) and the older set's — else null. The answers carry it as `frame_set_said` beside their `frame_set` (`/api/transport/describe_attempt`, `/api/transport/record`, `/api/task-setup/folder`) |
| the probe at the frame rule's nodes ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.11 § 4) *(R3d2c)* | a mode's own nodes (§ 4) |
| the example notebook (§ 6.1) | every piece |

### 6.1 The example notebook

`docs/engines/vibration-frames.ipynb` *(to build)*, saved with its outputs,
with a copy at `notebooks/vibration-frames.ipynb` in the projects tree for the
Jupyter tab, as transport's notebook is
([`engines/transport.md`](?doc=engines/transport.md), *The script, and the
notebook*). Its kernel is `molbuilder-jupyternb`, molbuilder's source on its
path as [`web/jupyter.md`](?doc=web/jupyter.md) § 7 states — frame sets are
made there through the toolkit — and the *(to build)*
`tests/test_vibration_frames_notebook_e2e.py` runs it on that env's Python,
named by its path (`nbclient` is that env's), so it is a check as well as a
demonstration *(the user, 2026-11-23: "an example jupyternotebook that can
open a calculation result folder and generate some mixing/plots as the
demonstration/example/tests to validate these calls")*:

- **its input — a vibration run's folder**: a PySCF methanol — twelve modes,
  among them a soft methyl torsion that is warm at 300 K — made down molbuilder's
  road by a test helper (`support.real_runs`), its run folder (description,
  `01_freq/run-0` with the result pair) carried beside the notebook and
  copied into the notebook's temporary projects tree; the first cell opens it
  with `vibration(run_folder)` and shows its provenance (`v.run`,
  `v.result_sha256`, `v.relaxation_calculation`), where the person names a run
  of their own;
- **its output**: every frame set into that temporary tree, so *Run All* runs
  twice;
- **its numbers computed, never typed**: each mode's $\sigma_T/Q_{\mathrm{zp}}$
  and thermal quanta at 300 K printed from the run — the torsion's warm width
  shows a temperature where the stiff modes sit at their zero-point motion;
- **a model response, said as a model**: $X(Q) = e^{\sum_k a_kQ_k}$ with
  chosen $a_k$, standing for a property computed at each frame — a curved,
  lopsided response like the contact bond of
  [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
  § 4.3, whose average over every state has a closed form;
- **asserts as well as plots**, each against the science (§ 2b.5's rows),
  each tolerance from the science — for the model response
  $X(Q)\,X(-Q) = 1$, so a paired set's pair-mean variance is
  $\tfrac12(\langle X^2\rangle + 1) - \langle X\rangle^2$, its error known
  before the draw — and each plot named:
  - *each mode's draws against its state's density* — thermal, $n$ quanta;
  - *two kinds of quanta side by side* at the same mean — `quanta=1`,
    `mean_quanta=1`: two distributions, two Gauss rules;
  - *one snapshot's displacement map*, and the snapshots' displacement of
    each atom against § 2b.5's covariance;
  - *the average against its closed form as $N$ grows* — a motion with the
    torsion set apart at one quantum, each mode's own factor, the merge the
    user asked for checked end to end, with its error from the pair means;
  - *an antithetic pair's cancellation*, and `parts` on an odd and an even
    response;
  - *a set against its variation in one mode*, one seed: every other mode's
    column identical, and the difference's error;
  - *more frames later*: the same seed drawn further (`start=`), written as a
    set of its own, the two calculations' results put end to end and their
    error taken together;
  - *a set written and read back*: `read_frame_set`, its frames and weights as
    written, the results paired with its frames by the file, item $i$ frame
    $i$;
  - *a mode's own set*: two sets against one re-weighted, and `weights_at`
    against frames placed for the state.

---

## 7. What it does not do — and the analyses of a set's results

It runs no calculation and averages none: the result of a calculation over
the frames, and its average at the frames' weights, are that calculation's
([`engines/transport.md`](?doc=engines/transport.md) § 2a.12). These
analyses of those results are the toolkit's:

**The weights for another motion — a mode's own set.** The same frames'
results averaged at another temperature, with $n$ quanta: the
weights from each frame's coordinate and the state asked
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.4):

```python
w = v.mode(7).state(temperature_k=600.0).weights_at(q_amu12_ang)
                   # a Weights(weights, beyond, exact_to_degree, said);
                   # written=the frames' own weights, where they have them
w.weights          # each frame's weight for that state, summing to 1
w.beyond           # the share of the state's motion beyond the outermost frames, both sides
w.exact_to_degree  # the degree the weights reproduce the state's averages to:
                   # m - 1 at least, one more for an odd number of symmetric frames
                   # and a symmetric state, 2m - 1 for the state's own Gauss rule
w.said             # frames placed for another motion: the share beyond, three frames
                   # blind to the shape; any frames: a weight below zero
```

`exact_to_degree` is the degree to which the weights reproduce the state's
averages of $1, Q, Q^2, \dots$, each checked: at least $m - 1$ — the $m$
averages the weights are solved for — one more for an odd number of
symmetric frames and a symmetric state (the odd average $m$ vanishes on
both), and $2m - 1$ where the $m$ frames are the state's own Gauss rule
(§ 4; science § 5.1) — each average checked within the one epsilon science
§ 2e.6 gives and derives.
**Frames placed for the state asked** are not said to
reach short or to be blind (science § 5.4, items 1 and 2): the $m$ frames
its own Gauss rule, or — given the
weights the frames were written with, `weights_at(q, written=…)`, which
`weights_for_set` passes from a set's rows — frames whose weights for this
state are those, an even rule's set with its prepended frame 0 at weight 0
among them: a set re-weighted to its own state is the rule it was written
with. **For frames placed for another motion** — counted on every frame,
so frames whose weights happen to fall on a shorter rule of this state, a
node at weight 0, still are — the share beyond the frames and, for three
frames, which carry only the curvature, that the re-averaged number is
blind to the state's shape
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.4, items 1 and 2; one or two frames are said to be exact only to
their degree) are said with the weights, never refused; a weight below zero is
said for any frames, past rounding (a weight within the one epsilon of zero, science § 2e.6, as
an even rule's prepended frame 0 comes out, is no weight below zero), the
frame named from 1, as every message names one (*frame 1 of 7's weight is
−1.005*; `model/structure.md` § 2.2e). Two frames at one position are
refused by name — no weights separate them.

**The weights for a set's rows at a state** —
`vibration_frames.weights_for_set(rows, *, temperature_k=None,
quanta=None, mean_quanta=None, classical=False)`,
handed on by `postprocess` as `postprocess.weights_for_set`: the frame
set's rows (a `frameset.FrameRows`), as `postprocess.frame_rows` reads them
from a calculation's data file ([`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md)
§ 3.2); the state named as § 2's table names it; each frame's coordinate
and the mode read from the rows, frame 0's included; the weights by
`weights_at`, a `Weights`. **Said first**, where the set decides it: a state
that takes a temperature, given none, takes the set's own `temperature_k`
row, the state named — *the thermal state at the set's temperature, 300 K
(its temperature_k row)*, the classical one likewise; each row the weights read is taken in the unit its name
carries ([`model/structure.md`](?doc=model/structure.md) § 2.2d: a row is
made so), a set written before 2.81.1 whose row states another read the
same, unsaid. **Refused by name**:
what is no frame set's rows (a path, say — naming what it was given by its
type, and `postprocess.frame_rows` and `frameset.read`); no rows — one
structure, or frames that state none (a family); a snapshot set, its frames
having no single mode's coordinate and its weights equal; a set stating no
mode's frequency and zero-point amplitude, or a frame stating no
coordinate; a state that takes a temperature, given none, of a set that
states none, naming the temperature to give. What a set is — a snapshot
set, a mode's own — is one rule, asked here and by `said_of` (§ 6). **`postprocess.reaveraged(path, *,
temperature_k=None, quanta=None, mean_quanta=None, classical=False)` joins
two calls and adds nothing**: those weights, and
transport's own door that averages a data file's frames at weights given,
`frame_average.averaged_at(data_file, weights)`
([`engines/transport.md`](?doc=engines/transport.md) § 2a.12), each voltage
by its own rule — answered with averaged_at's keys, `weights` the `Weights`,
`said` the weights' words then the average's, and `assumptions` transport's
own, then the set's, named for the motion it was made for, not the one
asked — its own row, *the set's own, for the state it was written for: …*,
or, for an older set that states none, molbuilder's words for it
(`assumptions_of_set`, § 6), *molbuilder's for the set, which states none,
for the state it was written for: …* (science § 7, never dropped).

**A snapshot set is not re-weighted** for another motion: the weights for
another distribution of independent draws are likelihood ratios, and the
frames needed grow by their mean square, with every mode whose state
changes ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.5, the derivation). Another motion is another set.

**The error of a snapshot set's average**, by the rule of
[`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.5, from values one per frame, frame 0 first — a calculation's data
file's values whole (§ 3.3's rule) — any trailing axes, a T(E) per frame
giving its error per energy:

- `snaps.standard_error(values)` → `Estimate(value, error, n)`: the average
  over the snapshots, its error and the snapshots counted — for independent
  frames from the frames, for pairs from the pair means, frame 0 left out. A
  value missing (NaN, a frame whose calculation did not finish) drops that
  frame — with pairs, its pair — and says it (`Estimate.said`); none left,
  or one, is said, its error NaN;
- `snaps.parts(values)` → `Parts(even, odd, said)`: a paired set's even and
  odd shares of the spread, from the pair means and the pair
  half-differences — whether pairs helped; a pair with a value missing
  dropped and said; a set drawn without pairs refused by name;
- the arithmetic itself is `sampling`'s (§ 9.1 module 4) —
  `mean_with_error`, `pair_means`, `even_odd_parts` — which a person's own
  arrays take directly: **one value per snapshot, in the order the
  snapshots were drawn, frame 0 left out**, any trailing axes kept (a value
  per energy). With mirrored pairs each snapshot is followed by its mirror,
  so values 1 and 2 are the first pair, 3 and 4 the second; `pair_means`
  and `even_odd_parts` **group them two by two themselves** — the pair
  means $(v_1 + v_2)/2, (v_3 + v_4)/2, \dots$, the half-differences
  $(v_1 - v_2)/2, (v_3 - v_4)/2, \dots$ — an odd count refused by name (a
  set's frame 0 left in, most often). Two sets of one motion — the seed
  drawn further, `start=` — put end to end keep their pairs, each set
  holding an even count:

  ```python
  import numpy as np
  from molbuilder.postprocess import sampling
  g1 = s1.conductance_g0.sel(bias_v=0.0).values   # set 1: frame 0, then its 64 snapshots
  g2 = s2.conductance_g0.sel(bias_v=0.0).values   # set 2, the seed drawn further: frame 0, then 64
  both = np.concatenate([g1[1:], g2[1:]])          # frame 0 of each left out: 128 values, 64 pairs
  means = sampling.pair_means(both)               # (both[0]+both[1])/2, (both[2]+both[3])/2, …: 64 means
  value, error = sampling.mean_with_error(means)  # the average and its error, from the pairs
  even, odd = sampling.even_odd_parts(both)       # the pair means' spread, the half-differences' spread
  ```

- `read_frame_set(path)` (§ 5) gives the `Snapshots` of a set
  written before, so a calculation's results pair with it by the file.

---

## 8. How it is validated

The science's expectations — every closed form a test takes, the
displacement covariance and each state's average of $e^{aQ}$ among them —
are
[`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2b.5's, never molbuilder's own output. The tests that hold the toolkit to
them: `tests/test_oscillator_states.py` (the generic tools of § 9.1 — each
chain's list, the events' means, the cut, the routes against each other and
against the science's closed forms, the draws, the refusals and the words,
the search from deliberately wrong starts, the precision, the statistics of
snapshots), `tests/test_vibration_mechanics.py` (each state's draw,
`weights_at`, `bose_einstein_temperature`, the motion and its snapshots, the
modes chosen in steps, beside the states' own rows), `tests/test_vibration_frames.py` (the vocabulary's words and its unit
rule, `said_of`'s provenance and a changed pair, the words of the frames
drawn whatever became of the pair
(`test_the_words_are_of_the_frames_drawn_whatever_became_of_the_pair`),
what a set is, an older mode set's assumptions, `weights_for_set` — a set's
own state its own rule, an even rule's set at its own state saying nothing
of another (`test_an_even_rules_set_at_its_own_state_says_nothing_of_another`),
the state named, the set's temperature taken and said; a row's unit in one
place,
`tests/test_row_units.py`; the writer and the reader — each recipe's kind of
set — a mode's own set in each state, a scaled one, snapshots with modes set
apart, classical, scaled, of some modes, of modes chosen in steps — written with
its rows and read back as written, a snapshot set's motion whole (its words,
the modes set apart, the steps that chose its modes, drawn again the same), a mode set's derived rows against § 2b.3's closed
forms, a person's frames and a person's set in the vocabulary's names, the
refusals and what is said, the outermost move the written row's and its
door by a mode's number; the doors' readout for each state against the
closed forms and the doors' state rule), `tests/spectra/test_what_a_temperature_means.py`
(the readout at a temperature, the science's § 5 table), `tests/spectra/test_blueprint.py`
(the doors' refusals before a run is read), `tests/test_vibration_mechanics_e2e.py`
(a real run's folder read as its vibration, the read door's sha256 from either
file, a set's provenance and relaxation record, the set reopening its run; the
command line in each state and its refusals),
`tests/test_spectrum_results_e2e.py` (the load door's readout for each state
against the script's door; *Write frame set…* in the browser), and the example
notebook's run (§ 6.1):

| what | the test |
|---|---|
| each state's drawing | its sample moments and distribution against the state's closed form — a two-sided Kolmogorov test against `cdf`; one normal number per draw for every state |
| the motion's draws | § 2b.5's covariance of the snapshots' coordinates and atom moves |
| antithetic pairs | each pair's coordinates sum to zero; a response odd in $Q$ averages to zero exactly; the pair-mean error against the true error of an even and an odd response; `parts` splitting them |
| the seed | the same seed, the same frames; `sample(2N)` beginning with `sample(N)`; `start=`; a mode set apart leaves every other mode's column unchanged |
| the modes chosen | a selection's steps in order — a window, only, add, drop — each acting on what the step before left, `only` keeping of those only the modes listed; the same steps in another order the same list where they commute; an empty selection, a mode the result has none of, a window backwards, and another vibration's selection refused by name (`test_the_modes_that_move_are_chosen_in_steps`, `test_a_mode_selection_that_moves_nothing_is_refused_by_name`) |
| re-weighting | frames for one bell curve re-weighted to a wider one (science § 5.4): its averages reproduced to the degree $m$ for an odd number of symmetric frames, the share beyond its closed form $\operatorname{erfc}(q_{\max}/\sigma\sqrt2)$, each weight below zero said with its frame named from 1; a state's own frames weighted for it its own rule — the rule's weights, exact to degree $2m - 1$, nothing said of an odd rule's, no zero weight said below zero (`test_a_states_own_frames_weighted_for_it_are_its_own_rule`); an even rule's set, its written weights given, saying nothing of another motion |
| a mode's own frames | each rule exact to its stated degree; the bell curve's three at $0, \pm\sqrt3\sigma$; the order; each rule symmetric bit for bit — an odd rule's middle node exactly 0 and its middle frame bit for bit $\mathbf R^0$, $q_- = -q_+$ for every pair, on the thermal and $n$-quanta rules; an even rule's set with the relaxed geometry as frame 0; `outermost_move` for each state |
| the indexing | every analysis on one value per frame, frame 0 first, the same in memory and read back; `standard_error` and `parts` leaving frame 0 out |
| the frame set | written and read back through the structure's door: frames, weights, every row with its meaning, a prepended frame 0's rows at 0; a person's frames whose first is the relaxed geometry written without a second copy |

---

## 9. The toolkit as steps *(decided and cut to lists 2026-11-25; modules 1–4 in place, the switch-over to come)*

*(The user, 2026-11-25: "i'd like the api to be more close to generic
question/math/quantum related calculation, but the solution to the question
we are interested in presented in a template/example jupyter notebook such
that user can easily tune/re-design it for other general problems".)* Today
a mode's state is one of three fixed kinds — thermal, $n$ quanta and
classical (§ 2) — each with its own density, distribution, averages, frames
and draw written for it, chosen by keyword switches (`temperature_k=`,
`quanta=`, `mean_quanta=`, `classical=`). A case outside the three — one
photon on a warm mode ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2d.4) — has no state, and adding it would mean a fourth kind. This section
makes the toolkit **generic tools** — the quantum mechanics of one
oscillator, the statistics of one coordinate, random sampling — and makes
**every case a chain of steps** written in the example notebook (§ 6.1),
which a person copies and changes. The theory, with figures and worked
numbers, is
[`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2e. **Modules 1–4 are in the package and `postprocess` hands them on;
`vibration` and `vibration_frames` move onto them, and § 9.3's other
changes follow, in the plan's R3d2b2b4b. On 2026-11-25 the user accepted the
recommended answers to § 9.4's four questions, and then cut the toolkit to
lists (§ 9.4, item 5).** § 9.1, the generic tools' manual, is in place
and stays; § 9.2–§ 9.4 are the design to come, and once it is built their
rules move into § 1–§ 8 in their places and they go.

```mermaid
flowchart TB
    subgraph G["generic — numpy, math and the constants; nothing molecular"]
        O["1 · oscillator<br/>one oscillator of a frequency:<br/>its levels, ħω, Q_zp, Bose–Einstein"]
        S["2 · oscillator_state<br/>chains of steps, and states:<br/>a start · the events · its list"]
        D["3 · distribution<br/>one coordinate's distribution:<br/>density · F · averages · frames · draws"]
        R["4 · sampling<br/>keyed random streams;<br/>a mean and its error"]
        O --> S --> D
        O --> R
    end
    V["5 · vibration<br/>the molecule's modes; a chain attached to a mode;<br/>the motion and its snapshots"]
    F["6 · vibration_frames<br/>frame sets: write, read, rows, words;<br/>the page's and the command line's choices"]
    P["molbuilder.postprocess<br/>the one import of a person's script"]
    G --> V --> F
    R --> V
    F --> P
    V --> P
```

Modules 1, 3, 4 and 5 **travel** with a job in the vibration bundles (§ 1)
— numpy, `math` and the constants, and for `vibration` the frequency-window
rule (`spectra.selection`) — module 2 with them once
`vibration` builds on it (R3d2b2b4b), and module 6 is the host's. **Each module
imports only the ones above it** in the drawing: 1–4 know nothing of
molecules or frame sets, `vibration` builds on them, `vibration_frames` on
it. A person reaches every tool through `molbuilder.postprocess`
([`model/parse.md`](?doc=model/parse.md) § 3), which hands on `Oscillator`,
the state module as `states`, the distribution module as `dist`, the
`oscillator` and `sampling` modules under their names, and the vibration's
reading and writing functions.

### 9.1 The generic tools — a manual

Four modules hold the toolkit's mathematics, and nothing in them knows of
molecules or frame sets: **`oscillator`**, one harmonic oscillator of a
frequency; **`oscillator_state`**, a mode's state built in steps;
**`distribution`**, one coordinate's distribution; **`sampling`**, random
streams and the statistics of results. A person's script reaches them all
through one import:

```python
from molbuilder.postprocess import Oscillator, states, dist, oscillator, sampling
```

**Three facts shape every call.**

- **The unit.** A state lives in the oscillator's own unit,
  $x = Q/Q_{\mathrm{zp}}$ — the coordinate $Q$ (amu½·Å) over the mode's
  zero-point amplitude. A **spread** is the root-mean-square
  $\sqrt{\langle x^2\rangle}$; the ground state's is 1 in $x$. A number in
  $x$ times $Q_{\mathrm{zp}}$ is a coordinate in amu½·Å, and a coordinate in
  amu½·Å over $Q_{\mathrm{zp}}$ is one in $x$.
- **A quantum state is its list.** A snapshot is one geometry: each mode's
  coordinate is one random draw from its state's density $P(x)$, at a
  moment of its own motion that is random too. Averaged over that moment,
  every quantum state is a list $p_n$ — the chance of finding $n$ quanta in
  the mode — and its density is $P(x) = \sum_n p_n\,\psi_n(x)^2$,
  symmetric, $P(-x) = P(x)$ ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
  § 2e.1). The classical bell curve, kept for comparison, counts no quanta:
  it is a distribution of the coordinate alone (science § 2e.5).
- **A count is an integer.** A level, an average's order, a number of
  frames, a size, a start and a stream key's entries are Python or numpy
  integers, 0 or more (`nodes` 1 or more); `2.0` is refused by name, as a
  truth value is.

**The road from a state to frames and snapshots:**

```mermaid
flowchart LR
    NU["a frequency<br/>ν̃ in cm⁻¹"] --> OSC["Oscillator(ν̃)"]
    CH["a chain of steps<br/>states.thermal(...).add_quantum()"] --> ON[".on(oscillator)"]
    OSC --> ON --> ST["a state<br/>its list p₀, p₁, p₂, …"]
    ST --> OF["dist.of_state(state)"] --> D["its distribution P(x)"]
    D --> GR["gauss_rule(m)<br/>m frames and their weights"]
    NS["sampling.normal_stream<br/>(seed, mode, 0)"] --> DR["draw(z)<br/>one coordinate per normal number"]
    D --> DR
    D --> WA["weights_at(points)<br/>frames placed for another state"]
    GR --> QZ["× Q_zp<br/>amu½·Å"]
    DR --> QZ
```

**A first example, end to end** — one quantum added to a 200 cm⁻¹ mode at
300 K (an absorbed photon on a warm mode). Every output in this section was
printed by molbuilder 2.84.0; numpy prints its arrays to four decimals here.

```python
>>> import numpy as np
>>> np.set_printoptions(precision=4, suppress=True)
>>> from molbuilder.postprocess import Oscillator, states, dist, oscillator, sampling
>>> osc = Oscillator(200.0)
>>> chain = states.thermal(temperature_k=300.0).add_quantum()
>>> state = chain.on(osc)
>>> state.words
'thermal at 300 K (0.621 mean quanta); one quantum added'
>>> round(state.mean_quanta, 4)
2.2426
>>> state.populations[:6]
array([0.    , 0.3804, 0.2916, 0.1676, 0.0856, 0.041 ])
>>> d = dist.of_state(state)
>>> type(d).__name__
'QuantumAdded'
>>> x, w = d.gauss_rule(3)
>>> x, w
(array([-3.2718,  0.    ,  3.2718]), array([0.2562, 0.4876, 0.2562]))
>>> x * osc.zero_point_amplitude_amu12_ang
array([-0.9499,  0.    ,  0.9499])
>>> z = sampling.normal_stream((7, 1, 0), 4)
>>> z
array([ 0.3474, -2.3379, -0.3389, -0.6831])
>>> d.draw(z)
array([ 1.3019, -4.5589, -1.2805, -2.0086])
```

Read it in order: the chain is written without a frequency; placed on the
oscillator it becomes a state with a list (no level 0 — a quantum was
added); its distribution gives three frames and their weights; the frames
times $Q_{\mathrm{zp}}$ are coordinates in amu½·Å; four normal numbers of
seed 7's stream for mode 1 give four drawn coordinates. Each call is
explained below — **what it is**, its **equation**, **what it does** step
by step, what it **refuses** by name, and an **example**.

---

#### 1 · `oscillator` — one harmonic oscillator of a frequency

**Assumed throughout:** the harmonic picture — a parabolic potential and
equally spaced levels $\hbar\omega$ apart. It holds while a mode carries
well under about 1 eV (80 quanta at 100 cm⁻¹, 8 at 1000 cm⁻¹); beyond,
the bond is breaking (science § 7, item 1).

##### `Oscillator(frequency_cm1)`

**What it is:** one harmonic mode of wavenumber $\tilde\nu$ (cm⁻¹), the
oscillator a chain is placed on.

| asked of it | equation | unit |
|---|---|---|
| `.frequency_cm1` | $\tilde\nu$, as given | cm⁻¹ |
| `.zero_point_amplitude_amu12_ang` | $Q_{\mathrm{zp}} = \sqrt{\hbar/2\omega}$, $\omega = 2\pi c\tilde\nu$ — the unit of $x$ | amu½·Å |
| `.quantum_ev` | $\hbar\omega = hc\tilde\nu$ | eV |
| `.bose_einstein_mean_quanta(temperature_k)` | $\bar n = 1/(e^{\hbar\omega/k_BT} - 1)$, 0 at 0 K — the mean quanta in equilibrium at $T$ | quanta |
| `.bose_einstein_temperature(mean_quanta)` | $T = \hbar\omega \big/ \big(k_B\ln(1 + 1/\bar n)\big)$, 0 K at 0 — the temperature whose thermal state holds $\bar n$: a mode pumped at random moments, its effective temperature | K |
| `.classical_spread(temperature_k)` | $\sqrt{2k_BT/\hbar\omega}$ — the classical bell curve's width, no zero-point motion ($k_BT/\omega^2$ its variance in $Q$) | $x$ |

**Refuses:** a frequency that is not one positive number — an imaginary
mode has no oscillator; a truth value, a text or an array is no frequency;
a temperature that is not a number of kelvin, 0 or more; a mean below 0.

```python
>>> osc = Oscillator(200.0)
>>> round(osc.zero_point_amplitude_amu12_ang, 4)
0.2903
>>> round(osc.quantum_ev, 5)
0.0248
>>> round(osc.bose_einstein_mean_quanta(300.0), 4)
0.6213
>>> round(osc.bose_einstein_temperature(1.0), 1)
415.1
>>> round(osc.classical_spread(300.0), 4)
1.444
>>> Oscillator(-50.0)
Traceback (most recent call last):
  ...
ValueError: an oscillator needs a positive frequency, in cm-1; got -50.0
```

##### The mechanics on a wavenumber

The same formulas as module functions, in amu½·Å, which `vibration` hands
on under their names:

| call | what it is | equation | refuses |
|---|---|---|---|
| `zero_point_amplitude_amu12_ang(frequency_cm1, has_imag=False)` | a mode's zero-point amplitude | $\sqrt{\hbar/2\omega}$; `None` for an imaginary mode (`has_imag`) or a frequency not above 0, which has no harmonic width | — |
| `bose_einstein_mean_quanta(frequency_cm1, temperature_k)` | the mean quanta in equilibrium | $\bar n = e^{-y}/(1 - e^{-y})$, $y = \hbar\omega/k_BT$ — no overflow for a stiff mode in the cold, no digits lost for a soft one in the heat; an array of frequencies gives an array | a temperature that is not a number of kelvin, 0 or more; a frequency not above 0 |
| `bose_einstein_temperature(frequency_cm1, mean_quanta)` | the temperature of a mean | $T = hc\tilde\nu \big/ \big(k_B\ln(1 + 1/\bar n)\big)$ | a frequency not above 0; a mean below 0 |
| `spread_of_quanta(zero_point_amplitude, quanta)` | the spread a mean number of quanta gives, in every state | $Q_{\mathrm{zp}}\sqrt{2\bar n + 1}$, in the unit of the amplitude given | a mean below 0 |
| `thermal_spread_amu12_ang(zero_point_amplitude, frequency_cm1, temperature_k)` | the thermal bell curve's spread | $Q_{\mathrm{zp}}\sqrt{2\bar n(T) + 1} = Q_{\mathrm{zp}}\sqrt{\coth(\hbar\omega/2k_BT)}$ | as `bose_einstein_mean_quanta` |
| `classical_spread_amu12_ang(frequency_cm1, temperature_k)` | the classical bell curve's spread | $\sqrt{k_BT}/\omega = Q_{\mathrm{zp}}\sqrt{2k_BT/\hbar\omega}$, 0 at 0 K | as `bose_einstein_mean_quanta` |
| `checked_temperature_k(raw)` | the temperature check every call here uses | the number, in kelvin | anything but a number of kelvin, 0 or more |

```python
>>> oscillator.bose_einstein_mean_quanta(np.array([50.0, 200.0, 1600.0]), 300.0)
array([3.6902, 0.6213, 0.0005])
>>> qzp = oscillator.zero_point_amplitude_amu12_ang(200.0)
>>> round(oscillator.thermal_spread_amu12_ang(qzp, 200.0, 300.0), 4)
0.4348
>>> round(oscillator.classical_spread_amu12_ang(200.0, 300.0), 4)
0.4192
>>> print(oscillator.zero_point_amplitude_amu12_ang(200.0, has_imag=True))
None
```

##### `level_wavefunction(n, x)`, `level_density(n, x)`

**What they are:** level $n$'s wavefunction and its density — where the
coordinate is when the mode holds exactly $n$ quanta.

$$
\psi_n(x) = \frac{He_n(x)\,e^{-x^2/4}}{(2\pi)^{1/4}\sqrt{n!}},
\qquad \int\psi_n(x)^2\,dx = 1,
\qquad \text{density } \psi_n(x)^2 ,
$$

$He_n$ the probabilists' Hermite polynomials ($He_0 = 1$, $He_1 = x$,
$He_2 = x^2 - 1$). **What it does** — the stable three-term recurrence in
$\xi = x/\sqrt2$, never a factorial or a power that overflows:

```text
h_0 = π^(-1/4);   h_1 = √2 ξ h_0
h_(k+1) = √(2/(k+1)) ξ h_k − √(k/(k+1)) h_(k−1)
    whenever |h| passes 1e100: divide h_k and h_(k−1) by 1e100, add 100 ln 10 to a kept log-scale
ψ_n(x) = h_n · exp(log-scale − ξ²/2) / 2^(1/4)
```

**Refuses:** $n$ not a whole number, 0 or more.

```python
>>> oscillator.level_density(1, np.array([0.0, 1.0, 2.0]))
array([0.   , 0.242, 0.216])
>>> xs = np.linspace(-12.0, 12.0, 4801)
>>> round(float((oscillator.level_density(5, xs) * (xs[1] - xs[0])).sum()), 10)
1.0
```

##### `level_moment(n, k)`

**What it is:** an average of level $n$, $\langle n\rvert x^k\lvert n\rangle$,
exactly (to rounding) — 0 for an odd $k$, and for even $k$

$$
\langle x^2\rangle_n = 2n+1, \qquad \langle x^4\rangle_n = 3(2n^2+2n+1), \quad\dots
$$

**What it does:** $x = \hat b + \hat b^\dagger$ applied $k/2$ times to the
level, on the window of levels it reaches, then the squared length:

```text
v = |n⟩ on the levels max(0, n − k/2) … n + k/2
repeat k/2 times:  v ← x v          (x moves each level one up and one down: ⟨j|x|j+1⟩ = √(j+1))
⟨n|x^k|n⟩ = v · v
```

**Refuses:** $n$ or $k$ not a whole number, 0 or more.

```python
>>> [round(oscillator.level_moment(1, k), 9) for k in (1, 2, 4, 6)]
[0.0, 3.0, 15.0, 105.0]
>>> round(oscillator.level_moment(10, 4), 9), 3 * (2 * 10**2 + 2 * 10 + 1)
(663.0, 663)
```

##### `lowering_matrix(lo, hi)`, `raising_matrix(lo, hi)`, `position_matrix(lo, hi)`

**What they are:** the ladder operators and the position on the levels
`lo` … `hi − 1`, as matrices:

$$
\hat b\lvert n\rangle = \sqrt n\,\lvert n-1\rangle,\qquad
\hat b^\dagger\lvert n\rangle = \sqrt{n+1}\,\lvert n+1\rangle,\qquad
\hat x = \hat b + \hat b^\dagger .
$$

Exact on every vector they move without leaving the window. **Refuses:**
a window not upward (`hi ≤ lo`), or an end that is no whole number.

```python
>>> oscillator.position_matrix(0, 4)
array([[0.    , 1.    , 0.    , 0.    ],
       [1.    , 0.    , 1.4142, 0.    ],
       [0.    , 1.4142, 0.    , 1.7321],
       [0.    , 0.    , 1.7321, 0.    ]])
>>> np.allclose(oscillator.raising_matrix(0, 4), oscillator.lowering_matrix(0, 4).T)
True
```

---

#### 2 · `oscillator_state` (`states`) — a mode's state, built in steps

**What it is:** a **chain** is a start and the events after it, in the
order they happen, written without a frequency — it has no numbers until
`.on(oscillator)` places it, and then it is a **state** with a list. Each
method returns a new chain; nothing is changed in place.

```mermaid
flowchart LR
    S["a start<br/>ground() · level(n)<br/>thermal(...) · classical_thermal(...)"] --> E1[".add_quantum()"]
    E1 --> E2[".remove_quantum()"]
    E2 --> E3["… in the order they happen"]
    E3 --> ON[".on(Oscillator(ν̃))"]
    ON --> ST["OscillatorState<br/>.populations · .mean_quanta · .words"]
```

##### The starts

| call | what it is | assumes | its list | refuses |
|---|---|---|---|---|
| `ground()` | the ground state, the zero-point motion alone | — | $p_0 = 1$ | — |
| `level(n)` | $n$ quanta exactly — the oscillator's level $n$ | the mode holds exactly $n$, as after $n$ detected quanta on a cold mode | $p_n = 1$ | $n$ not a whole number 0 or more |
| `thermal(temperature_k=T)` | the mode in equilibrium at $T$ | equilibrium with the surroundings | $p_n = \bar n^{\,n}/(\bar n+1)^{n+1}$, $\bar n$ Bose–Einstein's at $T$ on the oscillator | both a temperature and a mean; a temperature that is not a number of kelvin, 0 or more; a mean below 0 |
| `thermal(mean_quanta=n̄)` | many quanta added at random moments and relaxed, named by its mean — an effective temperature | that steady random excitation | the same list, of that $\bar n$ | as above |
| `thermal()` | a thermal state whose temperature is given where the chain is placed (`.on(..., temperature_k=)`) | as above | as above | as above |
| `classical_thermal(temperature_k=T)`, `classical_thermal()` | the classical bell curve, for comparison — at 0 K a point at rest; `classical_thermal()` its temperature given where the chain is placed | no zero-point motion | none — a classical state counts no quanta | a temperature that is not a number of kelvin, 0 or more |

##### The events: `.add_quantum()`, `.remove_quantum()`

**What they are:** one quantum gained or lost, **detected** — an absorbed
photon, a Stokes photon scattered or a tunnelling electron losing
$\hbar\omega$ (added); an anti-Stokes photon or an electron gaining
$\hbar\omega$ (removed). **Assumed:** the event is quicker than the mode
relaxes (its energy lifetime), so it acts on the state as it stands.

$$
\text{added: } \hat\rho \to \frac{\hat b^\dagger\hat\rho\,\hat b}{\mathrm{Tr}},\quad
p'_n = \frac{n\,p_{n-1}}{\langle n\rangle + 1};
\qquad
\text{removed: } \hat\rho \to \frac{\hat b\,\hat\rho\,\hat b^\dagger}{\mathrm{Tr}},\quad
p'_n = \frac{(n+1)\,p_{n+1}}{\langle n\rangle}.
$$

A detected event picks each level by the chance it happens there —
$n+1$ for a quantum added, $n$ for one removed — so the mean moves by
$1 + \mathrm{Var}(n)/(\langle n\rangle+1)$ when a quantum is added and by
$-1 + \mathrm{Var}(n)/\langle n\rangle$ when one is removed: a thermal
state of $\bar n$ goes to $2\bar n + 1$ and to $2\bar n$ — a detected
removal raises its mean, the event being likelier where more quanta are.

**Refuses:** either event on a classical start (it counts no quanta); a
quantum removed from a state that holds none — the ground state, a thermal
start typed at exactly 0 K or of no mean quanta. A removal such a start
cannot make after other events
(`thermal(temperature_k=0).add_quantum().remove_quantum().remove_quantum()`)
is refused where its list is first asked.

##### `.reduced()`

**What it is:** the chain in its one form, so two chains that make one
state compare equal: the ground state or a level with quanta added or
removed straight after it is that level; `level(0)` is the ground state; a
thermal start keeps its temperature, 0 K too. **Refuses:** the ground state
with a quantum removed.

```python
>>> states.ground().add_quantum().add_quantum().reduced() == states.level(2)
True
>>> states.level(1).remove_quantum().reduced() == states.ground()
True
>>> states.ground().remove_quantum()
Traceback (most recent call last):
  ...
ValueError: a quantum cannot be removed from a state that holds none -- the ground state, or a thermal state at exactly 0 K or of no mean quanta
```

##### `.on(oscillator, temperature_k=None, precision=1e-16)` → an `OscillatorState`

**What it is:** the chain on a mode of that frequency. **What it does:**

```text
refuse an oscillator that is no Oscillator, a precision not between 0 and 1
start ← the chain's first step
if temperature_k is given and the start (thermal or classical) names its own temperature:
    refuse unless they are equal
if temperature_k is given and the start names its own mean:
    refuse — a mean and a temperature would name the start twice
if the start (thermal or classical) names neither: start.temperature ← temperature_k   (refused if none is given)
any other start (the ground state, a level): temperature_k is not used
if the start is thermal: start.mean ← n̄(T) on this oscillator, or the mean it names
chain ← its reduced form
refuse a quantum removed straight after a start that holds none
→ the state; its list is built the first time it is asked (below)
```

**Refuses:** the steps above name each.

**How the list is built and cut** (`.populations_for(order)`, science
§ 2e.2, § 2e.4). A thermal start and every event after it come from one
formula, $p_n \propto c(n)\,t^n$ with $t = \bar n/(\bar n+1)$, so nothing
is divided by the mean (a 3000 cm⁻¹ mode at 4 K, $t = e^{-1079}$, gives its
limit, the ground state's list):

```text
if the start is a level n:          p = (0, …, 0, 1)                       p_n = 1
elif it holds no quanta (the ground state, a thermal start typed at 0 K or of no mean quanta):
    p = (1)
    for each event, in order:
        added:   q_n = n p_(n−1)          removed:   q_n = (n+1) p_(n+1)
        refuse if q is all 0;  p = q / Σq
else (thermal, mean n̄, t = n̄/(n̄+1)):
    c(n) = 1 for n ≥ 0
    for each event, in order:
        added:   c(n) ← n c(n−1)          removed:   c(n) ← (n+1) c(n+1)
    p_n ∝ c(n) tⁿ, built out (doubling the levels) until the levels beyond hold
        less than 1e-4 × precision of ⟨x^order⟩
cut:  T_n = p_n (2n+1)^(order/2);   keep levels 0 … N−1, N the first with Σ_(n≥N) T_n < precision · Σ T_n
renormalise; read-only
```

$k$ removals from a thermal state alone are $\binom{n+k}{k}(1-t)^{k+1}t^n$.
The default order is 12, $\langle x^{12}\rangle$, the highest even average
seven frames reproduce; science § 2e.6 says why each number.

| asked of an `OscillatorState` | answers |
|---|---|
| `.populations` | the list, cut for $\langle x^{12}\rangle$ — a removal its start cannot make after the events before it refused by name here, and wherever the list is asked (`.mean_quanta`, `dist.of_state`) |
| `.populations_for(order)` | the list cut for $\langle x^{\mathrm{order}}\rangle$ — a classical state refused, counting no quanta |
| `.mean_quanta` | $\bar n$ — in closed form where known: a level $n$; the ground state 0; a thermal state $\bar n$; with one quantum added $2\bar n + 1$, removed $2\bar n$ — else $\sum_n n\,p_n$; `None` for a classical state |
| `.temperature_k` | its start's temperature, or `None` |
| `.words` | each step in words |
| `.chain`, `.oscillator`, `.precision`, `.steps`, `.is_classical` | its reduced chain, the thermal start's mean and temperature resolved; the oscillator; the share of the averages its list may leave out; the chain's steps; whether it starts classical |

```python
>>> warm = states.thermal(temperature_k=300.0).on(osc)
>>> warm.populations[:4]
array([0.6168, 0.2364, 0.0906, 0.0347])
>>> warm.populations.size, warm.populations_for(0).size
(57, 39)
>>> round(warm.mean_quanta, 4)
0.6213
>>> s = states.thermal(mean_quanta=1.0).add_quantum().remove_quantum().on(osc)
>>> s.words
'thermal, 1 mean quantum; one quantum added; one quantum removed'
>>> round(s.mean_quanta, 4)
3.3333
>>> states.thermal().on(osc, temperature_k=77.0).words
'thermal at 77 K (0.0244 mean quanta)'
>>> states.thermal(temperature_k=300.0).on(osc, temperature_k=77.0)
Traceback (most recent call last):
  ...
ValueError: the chain's thermal start names its own temperature, 300 K; 77 K given here conflicts with it -- give one
```

##### `Step`, `Chain`

A `Chain(steps)` holds a tuple of `Step`s, the start first — `Step(name,
values)` one start or event by its name (`ground`, `level`, `thermal`,
`classical_thermal`, `add_quantum`, `remove_quantum`) and its values as
`(key, value)` pairs sorted by key, so two steps that are one compare equal
(`step.get(key)`, `step.with_values(...)`); `chain.start` its start,
`chain.is_classical` whether it starts classical, `chain.temperature_open`
whether its thermal start names neither a temperature nor a mean.

```python
>>> states.thermal(temperature_k=300.0).add_quantum().steps
(Step(name='thermal', values=(('temperature_k', 300.0),)), Step(name='add_quantum', values=()))
```

---

#### 3 · `distribution` (`dist`) — one coordinate's distribution

**What it is:** the statistics of one coordinate, in the unit it is given —
a state's in $x$. Every distribution is a `Distribution`, the base each
family fills in — its own `pdf`, `cdf`, `moment` and `gauss_rule`, a
family's own `quantile` and `draw` where it has one — and every one here is
symmetric about 0. Today `vibration`'s states ask this module for their
Gauss rules, the bell curve's functions and the weights for points placed
elsewhere; at the switch-over every state is one of these distributions
(§ 9.3). **Every distribution answers:**

| call | equation | what it does | refuses |
|---|---|---|---|
| `pdf(x)`, `cdf(x)` | $P(x)$; $F(x) = \int_{-\infty}^x P$ | the family's closed form, or its table (below) — arrays in, arrays out | — |
| `moment(k)` | $\langle x^k\rangle = \int x^k P(x)\,dx$, 0 for odd $k$ | exactly, in closed form or level by level | $k$ not a whole number 0 or more |
| `quantile(p)` | the $x$ with $F(x) = p$ | the bell curve's closed, $\sigma\Phi^{-1}(p)$; every other's by a bracketed Newton search (below) | — |
| `gauss_rule(nodes)` | $m$ = `nodes` points $x_i$ and weights $w_i$ with $\sum_i w_i x_i^k = \langle x^k\rangle$ for every $k \le 2m-1$ | the distribution's own Gauss rule (below) | `nodes` not a whole number 1 or more |
| `draw(z)` | the $x$ with $F(x) = \Phi(z)$, $\Phi$ the standard normal's distribution | one coordinate per standard normal number (below) | — |
| `weights_at(points, written=None)` | the weights with which `points` reproduce the averages | `points` in the distribution's own unit — a state's $x$, so a written set's coordinates in amu½·Å over $Q_{\mathrm{zp}}$; `points_weights` (below) with this distribution's `moment` and `cdf` | as `points_weights` |

**The families** — each a closed form, and the list for everything else:

| family | what it is | $P(x)$ and $F(x)$ | its Gauss rule | refuses |
|---|---|---|---|---|
| `Gaussian(sigma)` | a bell curve — a thermal state ($\sigma = \sqrt{2\bar n+1}$) or a classical one | $P = e^{-x^2/2\sigma^2}/\sqrt{2\pi}\,\sigma$, $F = \Phi(x/\sigma)$; $\sigma = 0$ a point at 0 | Gauss–Hermite scaled by $\sigma$; `quantile` $\sigma\Phi^{-1}(p)$, `draw` $\sigma z$ | a spread below 0, or a truth value |
| `QuantumAdded(nbar)` | one quantum added to a thermal state of $\bar n$ | with $s = 2\bar n + 1$, $y = x/\sqrt s$: $P = \big(\tfrac{\bar n}{s} + \tfrac{\bar n+1}{s}\,y^2\big)\,\phi(y)/\sqrt s$, $F = \Phi(y) - \tfrac{\bar n+1}{s}\,y\,\phi(y)$; $\langle x^k\rangle = s^{k/2}(k-1)!!\,\big(\tfrac{\bar n}{s} + \tfrac{\bar n+1}{s}(k+1)\big)$ | the bell curve's and the one-quantum shape's $(m{+}1)$-point rules pooled, then that measure's $m$-point rule | $\bar n$ below 0 |
| `Level(n)` | $n$ quanta exactly | $P = \psi_n(x)^2$; $F$ from the level's table | Lanczos on $\hat x$ from $\lvert n\rangle$ | $n$ not a whole number 0 or more |
| `LevelList(populations)` | a list over the levels — any quantum state, or a person's own model | $P = \sum_n p_n\,\psi_n(x)^2$, every level from one sweep of the recurrence; $F$ from one table of the whole list; $\langle x^k\rangle = \sum_n p_n\langle x^k\rangle_n$ | each level's $(m{+}1)$-point rule pooled with its $p_n$, then that measure's $m$-point rule | a chance below $-10^{-12}$ of the largest (a speck within it taken as 0); none above 0 |

$\phi(y) = e^{-y^2/2}/\sqrt{2\pi}$. Each family keeps what it was made of:
`Gaussian.sigma`; `QuantumAdded.nbar`, with `.var` $= s$ and the two
shares, `.w_bell` $= \bar n/s$ and `.w_quantum` $= (\bar n+1)/s$;
`Level.n`; `LevelList.populations`, the list given divided by its sum — so
a person's own model need not be normalised. A level's or a list's table is built
the first time $F$, a draw or a quantile asks it: the density on
$\pm(\sqrt{2n_{\max}+1} + 9)$ in $\xi = x/\sqrt2$, cut in $2(n_{\max}+20)$
panels, each by 40-point Gauss–Legendre, the running integral kept at every
panel's edge (science § 2d.5, § 2e.6).

##### `of_state(state)`

**What it is:** a state's distribution, by the route its reduced chain
takes — the closed form where the chain is a family, the list otherwise
(each the other's check, science § 2e.3). **Refuses:** a chain alone, its
numbers not yet known — place it first.

```mermaid
flowchart LR
    C["a state's reduced chain"] --> H{"which?"}
    H -->|"ground, or thermal"| B["the bell curve<br/>(Gaussian)"]
    H -->|"thermal → one quantum added"| Q["bell curve × quadratic<br/>(QuantumAdded)"]
    H -->|"n quanta"| N["its level<br/>(Level)"]
    H -->|"classical thermal"| K["the classical bell curve<br/>(Gaussian; at 0 K a point)"]
    H -->|"any other quantum chain"| L["its list<br/>(LevelList)"]
```

```python
>>> for c in (states.thermal(temperature_k=300.0), states.thermal(temperature_k=300.0).add_quantum(),
...           states.level(2), states.classical_thermal(temperature_k=300.0),
...           states.thermal(temperature_k=300.0).remove_quantum().remove_quantum()):
...     print(type(dist.of_state(c.on(osc))).__name__)
Gaussian
QuantumAdded
Level
Gaussian
LevelList
>>> from math import exp, factorial
>>> poisson = dist.LevelList([exp(-3.0) * 3.0**n / factorial(n) for n in range(60)])
>>> round(poisson.moment(2), 9)
7.0
>>> dist.LevelList([2.0, 2.0]).populations
array([0.5, 0.5])
>>> a = dist.of_state(states.thermal(temperature_k=300.0).add_quantum().on(osc))
>>> b = dist.LevelList(states.thermal(temperature_k=300.0).add_quantum().on(osc).populations)
>>> round(a.moment(4), 10), round(b.moment(4), 10)
(58.7174854798, 58.7174854798)
```

##### `gauss_rule(nodes)` — a distribution's own frames

A **frame** is one geometry of a set, with its weight; a distribution's own
$m$ frames are the points and weights of its Gauss rule — the fewest points
that reproduce its averages $\langle x^k\rangle$ up to $k = 2m-1$.

**What it does** — the $m$-point Gauss rule of a measure from its
three-term recurrence, Golub and Welsch's step [GolubWelsch1969]:

```text
Lanczos on the measure (its points x and probabilities, or x̂ on the levels from |n⟩):
    q₀ = the start;  for j = 0 … m−1:
        α_j = q_j · (x q_j);   w = x q_j − α_j q_j − β_(j−1) q_(j−1)   (fully reorthogonalised for a discrete measure)
        β_j = |w|;   q_(j+1) = w / β_j
J = the tridiagonal matrix with α on its diagonal, β beside it
the rule's points = J's eigenvalues; its weights = the squares of each eigenvector's first component
```

A list's rule pools every level's own $(m{+}1)$-point rule, weighted by its
$p_n$: that discrete measure has the list's averages to degree $2m+1$, so
its $m$-point rule is the list's own, exact to degree $2m-1$ (science
§ 2e.3).

```python
>>> lvl = dist.Level(1)
>>> x3, w3 = lvl.gauss_rule(3)
>>> x3, w3
(array([-2.2361,  0.    ,  2.2361]), array([0.3, 0.4, 0.3]))
>>> [round(float((w3 * x3**k).sum()), 9) for k in (2, 4)], [round(lvl.moment(k), 9) for k in (2, 4)]
([3.0, 15.0], [3.0, 15.0])
```

##### `quantile(p)` and `draw(z)` — one coordinate from one normal number

**What they do:** a draw turns one standard normal number $z$ into one
coordinate, the point where the distribution's chance reaches $z$'s. The
lower tail is taken always and the sign restored, so $-z$ gives exactly
$-x$ (a mirrored pair is exact) and the far tails keep their digits:

$$
x = -\operatorname{sign}(z)\;F^{-1}\!\Big(\tfrac12\operatorname{erfc}\big(\lvert z\rvert/\sqrt2\big)\Big)
\qquad(\text{the bell curve: } x = \sigma z).
$$

```mermaid
flowchart LR
    Z["z, a standard<br/>normal number"] --> T["p = ½ erfc(|z|/√2)<br/>its lower-tail chance"]
    T --> F["x₋ = F⁻¹(p) ≤ 0<br/>the quantile"]
    F --> X["x = −sign(z) · x₋"]
```

The quantile, for every family but the bell curve, is one bracketed Newton
search (science § 2d.5, § 2e.6):

```text
bracket [lo, hi] holding the answer (a table's panel; else ±12 spreads), start x inside it
repeat at most 200 times:
    miss = F(x) − p;   if miss = 0: x is the answer, kept
    narrow the bracket by the miss's sign
    newton = x − miss / P(x)
    x ← newton if it is inside the bracket and at most half the last step, else the bracket's middle
    stop once the step is within four units of the last digit of max(1, |x|)
```

```python
>>> g = dist.Gaussian(1.5)
>>> g.draw(np.array([-1.0, 0.0, 1.0]))
array([-1.5,  0. ,  1.5])
>>> from statistics import NormalDist
>>> one = dist.Level(1)
>>> z = np.array([-0.05, 0.05, 1.2])
>>> xs = one.draw(z)
>>> xs
array([-0.5473,  0.5473,  2.0754])
>>> np.round(one.cdf(xs), 8), np.round([NormalDist().cdf(v) for v in z], 8)
(array([0.4801, 0.5199, 0.8849]), array([0.4801, 0.5199, 0.8849]))
>>> float(one.quantile(0.5))
0.0
```

##### `points_weights(points, moment, cdf, scale, written=None)` → `PointWeights`

**What it is:** the weights with which frames placed **elsewhere** — for
another state, or by hand — reproduce this distribution's averages
(science § 5.4). **What it does:**

```text
m points x_i in the distribution's unit (refused: none, or two at one position)
moment(k) and cdf(x): the distribution's functions;  scale: a length near its spread
    (weights_at passes √⟨x²⟩) — the points are divided by it inside the solve only, so the
    system stays well scaled; the weights do not depend on it
solve  Σ_i w_i x_i^k = ⟨x^k⟩   for k = 0 … m−1            (the Vandermonde system)
exact_to_degree = the highest d, m−1 … 2m−1, for which every k ≤ d is reproduced
                  within ε = 1e-12 of max(Σ_i |w_i x_i^k|, |⟨x^k⟩|, 1)
own         = (d = 2m−1) or the weights equal `written` within ε  — the points are the distribution's own rule
beyond      = F(min x) + 1 − F(max x)   — the share of the distribution outside the outermost points
below_zero  = the i with w_i < −ε
```

Frames re-weighted are exact only to degree $m-1$ ($m$ for an odd number of
symmetric frames), against $2m-1$ for a state's own frames — so a change of
state read from one set needs five frames at least (science § 6).

```python
>>> hot = dist.of_state(states.thermal(temperature_k=300.0).on(osc))
>>> cold = dist.of_state(states.thermal(temperature_k=77.0).on(osc))
>>> frames, own_w = hot.gauss_rule(5)
>>> pw = cold.weights_at(frames)
>>> pw.weights
array([-0.002,  0.136,  0.732,  0.136, -0.002])
>>> pw.exact_to_degree, pw.own, round(pw.beyond, 8), pw.below_zero
(5, False, 2.946e-05, (0, 4))
>>> hot.weights_at(frames, written=own_w).own
True
>>> q_written = frames * osc.zero_point_amplitude_amu12_ang
>>> np.allclose(cold.weights_at(q_written / osc.zero_point_amplitude_amu12_ang).weights, pw.weights)
True
```

---

#### 4 · `sampling` — random streams, and the statistics of results

`vibration` asks it for every mode's stream and the arithmetic of
`Snapshots.standard_error` and `parts`.

##### `normal_stream(key, size, start=0)`

**What it is:** standard normal numbers `start` … `start + size − 1` of the
stream `key` — a mode's key is `(seed, mode, 0)`, `mode` its number from 1
(§ 3.3; science § 2d.6). Keys that differ only by trailing zeros are one
key — numpy pads a seed with zeros — so `(7, 1)` draws mode 1's numbers;
a person's own streams beside a set's take `(seed, 0, j)`, `j` 1 or more —
no mode is numbered 0, and a bare state of no mode draws `(seed, 0, 0)`.
**What it does:**

```text
generator = numpy.random.default_rng(list(key))      # PCG64, seeded through SeedSequence: another key, an independent stream
return generator.standard_normal(start + size)[start:]   # drawn in one block from the first
```

So a larger set begins with a smaller one's numbers, and a set continued
from `start` is the same numbers a longer set would have drawn. **Refuses:**
a key that is not a sequence of integers 0 or more; a size or start that
is not an integer 0 or more.

```python
>>> sampling.normal_stream((7, 1, 0), 5)
array([ 0.3474, -2.3379, -0.3389, -0.6831,  0.6629])
>>> sampling.normal_stream((7, 1, 0), 2, start=3)
array([-0.6831,  0.6629])
>>> sampling.normal_stream((7, 2, 0), 2)
array([ 0.1751, -0.5107])
>>> np.array_equal(sampling.normal_stream((7, 1), 5), sampling.normal_stream((7, 1, 0), 5))
True
```

##### `mean_with_error(units)`

**What it is:** the mean over independent units and its statistical error,
over the first axis, any trailing axes kept (a value per energy, say):

$$
\bar v = \frac1N\sum_i v_i,\qquad
\sigma_{\bar v} = \frac{s}{\sqrt N},\quad
s^2 = \frac{1}{N-1}\sum_i (v_i - \bar v)^2 .
$$

The mean is NaN with no unit, the error with fewer than two. **Refuses:**
nothing of its own.

##### `pair_means(values)`, `even_odd_parts(values)`

**What they are:** for a set of mirrored pairs — one value per snapshot,
each snapshot followed by its mirror, frame 0 (the relaxed geometry a
written set starts with, at weight 0) left out — the pairs' means, the
independent units of a paired set,
$\big((v_1+v_2)/2,\ (v_3+v_4)/2,\ \dots\big)$; and the set's scatter split
in two: the sample standard deviation $s$ (above) of the pair means — the
response's even part — and of the half-differences
$(v_1-v_2)/2,\ (v_3-v_4)/2,\ \dots$ — its odd part, which the pairs cancel —
each NaN with fewer than two pairs (science § 5.5). These are the scatter
of the units, not the mean's error: that is
`mean_with_error(pair_means(values))`.
**Refuses:** an odd count of values (most often frame 0 left in).

```python
>>> v = np.array([1.30, 0.90, 1.12, 1.04, 0.98, 1.10])
>>> sampling.pair_means(v)
array([1.1 , 1.08, 1.04])
>>> mean, err = sampling.mean_with_error(sampling.pair_means(v))
>>> round(float(mean), 6), round(float(err), 6)
(1.073333, 0.017638)
>>> even, odd = sampling.even_odd_parts(v)
>>> round(float(even), 6), round(float(odd), 6)
(0.030551, 0.131149)
>>> sampling.pair_means(v[:5])
Traceback (most recent call last):
  ...
ValueError: mirrored pairs come two by two -- each snapshot followed by its mirror, frame 0 left out; got 5 values
```

### 9.2 The cases, as chains

**Today**, a chain is placed on a mode's oscillator by hand, its frames
scaled by the mode's zero-point amplitude, and the geometries written as a
person's set (§ 3.4) — one photon on warm mode 7, five frames:

```python
import numpy as np
from molbuilder.postprocess import vibration, Oscillator, states, dist, write_frame_set

v = vibration(run_folder)
m = v.mode(7)
chain = states.thermal(temperature_k=300.0).add_quantum()      # one photon on the mode at 300 K
d = dist.of_state(chain.on(Oscillator(m.frequency_cm1)))
x, w = d.gauss_rule(5)                                         # in x = Q / Q_zp, smallest first
first = np.argsort(np.abs(x), kind="stable")                   # the middle frame first: the writer's frame 0 (§ 5)
x, w = x[first], w[first]
q = x * m.zero_point_amplitude_amu12_ang                       # amu½·Å
geometries = [v.displaced({7: float(qi)}) for qi in q]         # every atom at R + Q L (§ 3.2)
write_frame_set(out_xyz, geometries, w, vibration=v,           # out_xyz: the pair's .xyz in the project
                rows=[{"name": "assumptions", "value": "one photon on mode 7 at 300 K",
                       "unit": "", "note": "the chain, placed by hand"}])
```

The middle frame goes first because the writer takes a set's first frame
as its relaxed geometry, frame 0, where it stands there; left in the rule's
own order (smallest first), the writer prepends frame 0 and the set computes
the relaxed geometry twice. **At the switch-over** (§ 9.3) a mode takes a
chain itself, and a snapshot set can set one mode apart in a chain's state;
until then `v.motion` and `with_mode` take § 3.1's arguments, a set-apart
mode in one of § 2's states:

```python
# after the switch-over (§ 9.3)
warm = states.thermal(temperature_k=300.0)              # the mode at 300 K
photon = m.bind(warm.add_quantum())                     # one photon on the warm mode
photon.frames(5)                                        # its own 5-frame rule, frame 0 first
photon.moment(4)                                        # <Q^4>, amu^2·Å^4
photon.pdf(0.05), photon.draw(64, seed=1)               # the density at Q = 0.05 amu½·Å; 64 draws

motion = v.motion(states.thermal(temperature_k=300.0)).with_mode(7, warm.add_quantum())
snaps = motion.sample(64, seed=20261125, antithetic=True)
```

The chains below are `states.` calls, written without it:

| case | chain | a number to check (cold mode where not said) |
|---|---|---|
| thermal at $T$ | `thermal(temperature_k=T)` | 200 cm⁻¹ at 300 K: mean 0.62 |
| thermal at a temperature given where it is placed | `thermal()` — `.on(oscillator, temperature_k=T)`; after the switch-over, attached to a mode, the vibration record's temperature, said | |
| an effective temperature, mean $n$ | `thermal(mean_quanta=n)` | |
| $n$ quanta exactly | `level(n)` | |
| one photon on a cold mode | `ground().add_quantum()` — the same state as `level(1)` | mean 1 |
| one photon on a warm mode | `thermal(temperature_k=T).add_quantum()` | $\bar n = 1$: mean 3 |
| two photons on a warm mode | `thermal(temperature_k=T).add_quantum().add_quantum()` | $\bar n = 1$: mean 5 |
| a detected anti-Stokes event | `thermal(temperature_k=T).remove_quantum()` | $\bar n = 1$: mean 2 |
| a photon, then an anti-Stokes event | `thermal(temperature_k=T).add_quantum().remove_quantum()` | $\bar n = 1$: mean $10/3$ |
| the classical bell curve | `classical_thermal(temperature_k=T)`, or `classical_thermal()` — `.on(oscillator, temperature_k=T)`; after the switch-over, attached to a mode, the vibration record's | |
| every mode thermal, one set apart (after the switch-over) | `v.motion(thermal(temperature_k=T)).with_mode(7, chain)` | |
| a classical motion (after the switch-over) | `v.motion(classical_thermal(temperature_k=T))` | |

Each step can be swapped: a colder mode is another `temperature_k`, a third
photon one more `.add_quantum()`. A quantum two modes share is no chain of
one mode's steps: a degenerate pair is turned by hand to the light's axes,
one combination excited and the other thermal (§ 3.1; science § 2c.2); two
modes of nearby frequencies need a joint state over both modes' levels
(science § 2e.5).

**From today's calls to the steps:**

| today (§ 2, § 3.1, § 7) | after |
|---|---|
| `m.state(temperature_k=T)` | `m.bind(states.thermal(temperature_k=T))` |
| `m.state(mean_quanta=n)` | `m.bind(states.thermal(mean_quanta=n))` |
| `m.state(quanta=n)` | `m.bind(states.level(n))` |
| `m.state(temperature_k=T, classical=True)` | `m.bind(states.classical_thermal(temperature_k=T))` |
| `v.motion(temperature_k=T)` | `v.motion(states.thermal(temperature_k=T))` |
| `motion.with_mode(k, quanta=1)` | `motion.with_mode(k, states.level(1))` |
| `postprocess.reaveraged(path, temperature_k=77)` | `postprocess.reaveraged(path, states.thermal(temperature_k=77))` |
| `postprocess.reaveraged(path, classical=True)` | `postprocess.reaveraged(path, states.classical_thermal())` — the set's temperature |

### 9.3 What it changes

- **The molecule's calls** — `vibration` keeps the molecule: `Vibration`,
  `Mode` (its oscillator `m.oscillator`, its rows, `displaced`,
  `coordinate`), the motion and its snapshots, and the module functions the
  spectra code imports (`thermal_row`, `displaced`, `shares_by`, …) — its
  states built on § 9.1's tools:
  - **`m.bind(chain_or_distribution)`** → a `ModeDistribution`, the chain or
    distribution **attached to mode `m`**. Attaching gives it the mode's
    frequency (a temperature becomes a list), its $Q_{\mathrm{zp}}$ (every
    argument and answer from then on in amu½·Å), an open temperature — the
    record's, the motion's or the set's, by § 2's rule, said in
    `temperature_from` — and the mode's own random stream. A distribution
    made for another mode's scale is refused by name. It answers `pdf(q)`,
    `cdf(q)`, `moment(k)` (amu^{k/2}·Å^k), `spread_amu12_ang`,
    `atom_rms_ang`, `mean_quanta`, `quantile(p)`, `words`,
    `draw(size, *, seed, start=0)`, `frames(nodes)` — the rule in the set's
    order, frame 0 at 0 (§ 4) — `outermost_move(nodes)`, and
    `weights_at(q, written=None)` with § 7's words.
  - **The motion**: `v.motion(chain)` — the chain attached to every real
    mode, or a function of a mode giving its chain or distribution;
    `modes=` a list or a mode selection built in steps (§ 3.1). **The
    motion's temperature** is its chain's thermal start's — quantum or
    classical — where it has one; where it has none (a function, mean
    quanta, the ground state), an open thermal start set apart later takes
    the record's, said. `with_mode(k, chain_or_distribution)`;
    `sample(n, *, seed, antithetic, start=0)`.
  - `Snapshots`, the frame-0 rule, the analyses over § 9.1's `sampling`, and
    `Motion.said`, whose sentences give the way on in the new calls.
- **Retired** (§ 9.4, question 4): `State`, its three kinds, `STATES`,
  `checked_state`, `state()`, `Mode.state`, `ModeState`, and the keyword
  forms of `with_mode`, `v.motion`,
  `postprocess.outermost_move`, `postprocess.weights_for_set` and
  `postprocess.reaveraged` — the last three take a chain:
  `outermost_move(path, *, mode, chain, nodes=None)`,
  `weights_for_set(rows, chain)` and `reaveraged(path, chain)`, a thermal
  start left open taking the set's own `temperature_k` row, said (*the
  set's*), as today.
- **What a set records.** A mode's own set records its chain's steps —
  `info.state`, the steps in order, each `{step, values}` — and a snapshot
  set each mode's, in `info.sampling`'s `states`. The rows follow from them:
  `temperature_k` the temperature of the chain's start — a thermal one's, a
  classical thermal one's — the steps after it in
  `info.state` and the words, § 5.1's row rewritten so (today it reads *the
  temperature of the motion the frames sample*, untrue of a set whose
  temperature was followed by a photon); `quanta` from `level`;
  `mean_quanta` derived; `quantum` 1 for a quantum chain, 0 for a classical
  one, and none for a person's own distribution, said as theirs;
  `sampling_rule` `gauss-hermite` for a bell curve, `gauss` for another
  state's own rule; `assumptions` in the steps' words; a snapshot set's
  `set_apart_modes` each set-apart mode's words. *Example*, one photon on a
  200 cm⁻¹ mode at 300 K, `thermal(temperature_k=300).add_quantum()`:
  `info.state` =
  `[{"step": "thermal", "values": {"temperature_k": 300.0}}, {"step": "add_quantum", "values": {}}]`;
  the rows `temperature_k` 300, `mean_quanta` 2.24, `quantum` 1,
  `sampling_rule` `gauss`, no `quanta` row, and `assumptions` beginning
  *thermal at 300 K (0.621 mean quanta); one quantum added*. A thermal set
  at 300 K also writes `temperature_k` 300; the two are told apart by
  `info.state`, `sampling_rule` and the words a page shows. **A mode's own
  set written before the move is read as written** (§ 5.1): its kind's rows
  are read as the chain they name — a thermal row a thermal start, a quanta
  row `level`, a classical row the classical start — the same set as
  before. A snapshot set written before the move is not read back: the
  toolkit's own snapshot files get no handling (the user, 2026-11-25).
- **`vibration_frames`, rebuilt on the steps**: the page's and the command
  line's state (`doors_state`, `read_state_ask`, `STATE_ASK_KEYS`) building
  their chains from the person's choices; `readout`; `frameset_from_mode`;
  `outermost_move`, `outermost_move_of` and `vibration.outermost_move_ang`,
  on an attached distribution; `state_words` and `thermal_state_words`
  becoming the steps' words, and with them `said_of`'s words for a set
  written before 2.81.0; the writer's rows and records (above); `_sampling_of`
  and `_motion_of`; the reader rebuilding a mode set's state (`_mode_set_of`,
  `THE_SETS`); `weights_for_set`. **Unchanged**: the streams' keys, the
  vocabulary's names, and transport's reading of a set (§ 0).
- **Other callers**: `scripts/science_figures.py` (Figure 2d-4 draws through
  `state()`); `spectra/derived.py` and `spectra/normal_modes.py` (module 1's
  two names, handed on by `vibration`); `Motion.said`'s way-on sentences.
- **The architecture**: `docs/architecture.md`'s `vibration` and
  `vibration_frames` rows rewritten (their retired names out), and
  `vibration`'s `TRAVELS_WITH` naming `oscillator_state` beside the three
  generic tools it names today — the four modules' own rows and `model/parse.md` § 3's rows
  for what `postprocess` hands on are in place.
- **The documents**: this chapter's § 1–§ 8 (§ 9.2–§ 9.4's rules moved into them; § 9.1, the manual, stays)
  and the notebook's list (§ 6.1), which gains the new cases — one photon
  on a warm mode, a detected anti-Stokes event, two events in a row; the science chapter's § 2a.3, § 2c.4, § 2d.3–§ 2d.8 and Figure
  2d-4, where *`vibration.state` holds no such state* becomes false, and
  § 5.4's item 1, which names `State.cdf`;
  [`web/spectra.md`](?doc=web/spectra.md) § 4.3 (the dialog's note, in the
  steps' words); [`engines/vibration.md`](?doc=engines/vibration.md)
  § 6.11 § 6 and the sections naming `m.state`;
  [`engines/vibration-postprocessing.md`](?doc=engines/vibration-postprocessing.md)
  § 1 (its `outermost_move` row) and § 2a–§ 2b; the file map of
  [`engines/vibration.md`](?doc=engines/vibration.md) § 8.1;
  [`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md)
  § 3.2; the plan's R3d2b2c bullets.
- **The tests**: retired first — every call of the retired forms in
  `tests/test_vibration_mechanics.py`, `tests/test_vibration_frames.py`,
  `tests/test_vibration_mechanics_e2e.py`,
  `tests/spectra/test_what_a_temperature_means.py`,
  `tests/test_spectrum_results_e2e.py` and
  `tests/test_transport_on_a_real_junction_e2e.py` — then each step, each
  chain and each route held to the science's closed forms (science § 2b.5,
  § 2e.1's lists and their $P(Q)$, § 2e.2's events and their order, § 2e.3's
  worked chain), never to molbuilder's own output; the closed form and the
  list each other's check. `compare_head.py` against today's toolkit is the
  move's evidence, once, not a test: today's three kinds and their chains the
  same distributions — their frames to the digits the list's cut leaves
  (science § 2e.4) — and the same draws (a bell curve, a number state).

### 9.4 The user's decisions *(2026-11-25: "go with your recommendations on all four")*

1. **How a state is held.**
   - (a) **The list alone** (science § 2e.1): one route for every state. Its
     cost grows with the levels — 154 for a 50 cm⁻¹ mode at 300 K, 769 for
     a 10 cm⁻¹ mode at 300 K (science § 2e.4).
   - (b) **The list, and the closed forms where a family has one** — the bell
     curve (thermal), the bell curve times a quadratic (one photon on a warm
     mode), a level, the classical bell curve — recognised from the reduced
     chain, the list for everything else; the same distribution by both,
     each the other's test.
   - **Decided: (b).** It keeps today's exact, fast thermal and level states
     as they are, and adds the list for the states that have no closed form.
2. **A quantum two modes share** (science § 2d.4, § 2e.5) — two modes of
   nearby frequencies, or a degenerate pair under circularly polarised light:
   a density matrix over two modes' levels, the same steps on it — now, or
   when a case needs it. **Decided: when a case needs it**; the notebook
   states the theory. A degenerate pair under linearly polarised light needs
   no joint density matrix, only the pair's mode turned to the field's axes
   — built, too, when a case needs it.
3. **The spectrum page's *Write frame set…* dialog and `molbuilder frameset
   from-mode`** (§ 6): keep their named choices — thermal (by a
   temperature, or the command line's mean quanta) and $n$ quanta — built
   from chains underneath, or offer thermal alone and send every other state
   to the notebook. **Decided: keep them**, no new one until a person asks;
   their sets, their `assumptions` row and the page's note say the state in
   the steps' words.
4. **A person's script**: retire the keyword forms (`m.state(temperature_k=…)`,
   `with_mode(k, quanta=1)`, and the `postprocess` functions taking them) for
   the chains, or keep them beside the chains. No person's script calls them
   yet — the notebook is not written (§ 6.1); molbuilder's own callers are
   the page's and the command line's state and the re-weighting, which build
   chains. **Decided: retire them**, as the user's rule says (*"we are not
   exposing api for special case direct results"*), the named choices of
   question 3 living in the page and the command line alone.
5. **What a state is for** *(the user, 2026-11-25)*: a snapshot is one
   geometry, each mode's coordinate one random draw, its moment in its own
   motion random — so a state is its list, and nothing fixed in time has a
   place in it (*"a snapshot is not a fucking average by its fucking
   definition"*; *"why the fuck do we consider two pulses? we are generating
   one fucking snapshot"*). **Decided: cut it, and clean it all up** (*"yes,
   clean it all up and fucking check for consistency and completeness"*):
   the push and its phase, a state kept at one moment, the time average, two
   pushes adding, the density-matrix route, and the steady drive — the
   coherent state a phase-steady force makes, which no process on
   molbuilder's road (a DC conductance; incoherent excitation one quantum
   at a time) represents (science § 2a.2) — gone from the generic tools and
   from the doors and the page (the *driven* and *warm driven* states of
   2.78.0–2.83.0). Every call documented with what it stands for, assumes
   and computes (§ 9.1: *"it's best to explain the assumption of the
   calculation and the actual operation for each api"*).

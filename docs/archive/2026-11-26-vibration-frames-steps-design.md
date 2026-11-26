# The toolkit as steps — its design and the user's decisions (2026-11-25/26), archived

**Role:** record · **Domain:** archive · **Companions:** `engines/vibration-frames.md` (the toolkit's contract: the
binding § 2, the generic tools § 9), `science/vibrational-averaging.md` § 2e (the theory), `process/changelog.md`
(2.84.0, 2.85.0)

**Built — in 2.84.0 (the generic tools) and 2.85.0 (the switch-over to one layer).** `engines/vibration-frames.md`
§ 9.2–§ 9.4 as they stood on 2026-11-26, before the switch-over was built, moved here whole: their rules now live in
the contract's § 1–§ 9 (§ 2 the binding and its table of chains, § 5 what a set records, § 6 the doors' choices as
chains, § 9 the generic tools), as finished contracts hold rules only. What stays the record here is **the user's
decisions with their words** (§ 9.4 below): the closed forms kept beside the list (1(b)), a quantum two modes share
built when a case needs it (2), the doors keeping their named choices (3), the keyword forms retired (4), a state is
its list (5), and one layer (6) — with, at 2.85.0's first review round, a snapshot set 2.82–2.84 wrote refused by name when read (the user's choice, 2026-11-26). § 9.2's *Today* route and § 9.3's change list are the plan as it stood before the
switch-over, kept with them.

---

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
6. **One layer** *(the user, 2026-11-26, at the switch-over: "no obsolete
   design, no obsolete api, no twist new design to fit old needs, make sure
   we have one consistent api layer/framework design to support generic
   needs, and our special cases are built on the combination of these tools
   clear logic and solid foundation")*. **Decided** (the user's yes on the
   design put to them): the generic tools do every piece of mathematics —
   with two generic additions, a distribution in another unit
   (`d.scaled(s)`) and one weights result carrying the words a frame set
   says — and the molecule adds only what needs atoms or a mode's number:
   `m.bind(chain or distribution)`, the mode's distribution in amu½·Å with
   its draws on the mode's own stream, its frames as geometries, its weights
   for frames placed elsewhere, each atom's move; `v.motion(chain)`,
   `with_mode(k, chain or distribution)` — no second form taking a function
   of a mode. A set records what it was bound from — a chain's steps or a
   distribution's family and numbers — and is read back by binding it again;
   a distribution of a person's own class is drawn from in memory, and its
   set refused by name, since it could not be read back. The page's and the
   command line's choices are chains, re-weighting and the outermost move
   take a chain or a distribution; every old state type, the keyword forms,
   the doors' state dictionary, the second weights type and the separate
   state words go, nothing kept beside them. **A mode's own set written by
   2.81–2.84**, which records no state, is read without it — its geometries,
   weights and coordinates whole, said once; no reader of its old rows as a
   state (the user's choice, 2026-11-26).

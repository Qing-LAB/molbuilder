# The plan — one file

**Role:** plan — **the only one** *(user, 2026-09-01: "We don't need ten plan
files scattered")*. Every open item lives here, said once; a document that
finds work records the evidence and sends the item here (R3,
[`README.md`](?doc=README.md)). Finished work is archived, every archived file
listed in [`archive/README.md`](?doc=archive/README.md); the plan as it stood
at 2.77.23 is [`archive/2026-11-22-plan-consolidation.md`](?doc=archive/2026-11-22-plan-consolidation.md).
**Domain:** all · **Last consolidated:** 2026-11-22.

---

## ▶ Where we are *(2026-11-23)*

**2.81.0.** F0–F11 and F7a are closed (2.56.2–2.77.2). F12, *Proven*,
walked its three roads in the browser on real runs and ran the full batch
(6106 fast and 805 e2e tests, none failed, at 2.77.2); its two review rounds
([`archive/audits/2026-11-18-f12-review/`](?doc=archive/audits/2026-11-18-f12-review/TRIAGE-round1.md))
sorted every finding into the releases below. **The finish plan** *(the
user, 2026-11-19: "a consolidated plan to get things finished, not to drag
things along")*: five releases, R1–R5, each with its review rounds. R1 and R2
are done; R3, one home per fact on the server side, grew from one release
into parts as reviews and the user's decisions added work, fourteen of them
shipped:

| version | part | what it did | record |
|---|---|---|---|
| 2.77.10–2.77.12 | fix 6 (2.77.11) and the user's two asks | Plotly's toolbar back on every chart; MolView keeps what you picked; a mode's animation in view beside the spectrum | `fix6-review.md`, changelog |
| 2.77.13–2.77.15 | R1 | the viewers: the animation's export, the spectrum's picked mode, the spectra page keeps your view | `fix7a`–`fix7c-review.md` |
| 2.77.16 | R2 | your own lines in the run script run where you put them; the machine says what it does | `fix8-review.md` |
| 2.77.17 | R3a | one owner for each rule of the jobs and the run files | `fix9-review.md` |
| 2.77.18 | R3b | one owner for each rule of the run script, the envs and the server process | `fix10-review.md` |
| 2.77.19 | R3b2 | every bundle beside a job gathered from its members' own word | `fix11-review.md` |
| 2.77.20 | R3b3a | one record per bundle beside a job | `fix12-review.md` |
| 2.77.21 | R3b3b | a run keeps its files (a run of points' container holds its points' prepared files) | `fix13-review.md` |
| 2.77.22 | R3c1 | what runs in an environment is its own ([`ops/installation.md`](?doc=ops/installation.md) M6) | `fix14-review.md` |
| 2.77.23 | R3c2 | every printed command is one you can type, from any folder (the host env's path file, [`ops/deployment.md`](?doc=ops/deployment.md) § 0) | `fix15-review.md` |
| 2.77.24 | R3c3 | templates and engines, each fact in one home; every env command by the checkout's full path, in molbuilder's checkout in a file for another machine and in every run script | `fix16-review.md` |
| 2.77.25 | R3d1a | transport, each fact in one home; every data file names its engine in the engine table's words, a voltage said one way | `fix17-review.md` |
| 2.77.26 | R3d1b | the spectra, each fact in one home; the spectra page names its engine in the engine table's words | `fix18-review.md` |
| 2.78.0 | R3d2a | the vibration framework's module `vibration`, a mode's states and their frames, and the script door `postprocess.vibration` | `fix19-review.md` |
| 2.79.0 | R3d2b1 | a vibration's record states each mode's mechanics: each atom's share of the motion, the largest zero-point move, Gaussian's reduced mass and force constant, the spread and mean quanta at the modes' temperatures (schema 15, data file @11) | `fix20-review.md` |
| 2.80.0 | R3d2b2a | the molecule's motion in `vibration`: every state's own frames, draws and re-weighting; modes set apart and drive groups; snapshots, their error and pooling | `fix21-review.md` |
| 2.81.0 | R3d2b2b1 | transport reads each frame's weight alone and carries every other row as written (data file @14, record @19); a frame set's words and provenance the motion toolkit's, on every page that shows one, the Results tab's frame-set card new; `reaveraged` for any state | `fix22-review.md` |

Records: `archive/audits/2026-11-18-f12-review/fix<N>-review.md`. Since
2.77.23 the host env holds `molbuilder-source.pth` on the development machine
(bootstrap run once, the user's yes, 2026-11-22); doctor: every env OK.

---

## ▶ What runs next — to the finish line, in order

Each release: its owning chapters first, the code, a test seen failing on the
last release, the review rounds, the fixes, one e2e run (§ *How a release
closes*). Sources: the findings files in
[`archive/audits/2026-11-18-f12-review/`](?doc=archive/audits/2026-11-18-f12-review/TRIAGE-round1.md)
(`r2-<n>-*.md`), named per item as `<file> <row>`.

### 1. The vibration framework — designed, the user's decisions taken (2026-11-22); built in R3d

*(the user, 2026-11-21: "why not make sure these calculation results are part
of the netcdf results?"; "Design now, then R3c"; "you can use a module either
called vibration or something to unify these vibration related
calculation")*. Written into the chapters, no code, then brought to the user;
built in R3d with the vibration's data files.

- **What a person can already read** (2026-11-22, read in the code): the run's
  `<label>.spectra.nc` — per mode its frequency, imaginary flag, mass-normalised
  and display eigenvectors, IR intensity, Raman activity, and (data file v7)
  its zero-point amplitude √(ħ/2ω) and each atom's zero-point displacement,
  and (schema 15, data file @11) each free atom's share of the motion, the
  largest zero-point move, the reduced mass and force constant, and each
  mode's spread and mean quanta at the modes' own temperatures;
  the thermochemistry (ZPE, G, a temperature grid); the per-displacement
  charges (SIESTA, IR on); `<label>.optical.nc`, `<label>.raman.nc`,
  `<label>.steps.nc`; through the Results tab's Export and
  `molbuilder.postprocess` (`read_spectra_data`, `infrared_view`,
  `raman_view`, `read_optical_data`, `read_raman_data`, `read_steps`,
  `outermost_move`, `reaveraged`, `atom_index_map`), every row in the user's
  atom order.
- **What is computed on request and kept nowhere**: a mode's thermal spread
  and mean quanta at a temperature other than the modes' own, the shares
  summed by element, label or region, the mode over all atoms and the
  displaced geometry (`vibration`, 2.78.0),
  a frame set's outermost move (`spectra/mode_frames.py`, moving into the
  motion toolkit, `vibration_frames`, R3d2b2b2), the weights for a set's rows
  at a state (`vibration_frames.weights_for_set`, by `State.weights_at`,
  2.81.0).
- **The design, written 2026-11-22** ([`engines/vibration.md`](?doc=engines/vibration.md)
  § 6.11; the theory [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
  § 2b): one `vibration` module owning every per-mode quantity and its
  definition, the rules today written two or more times folded into it; a
  per-mode record in the data file — each free atom's share of the motion,
  the largest zero-point move, the reduced mass and force constant, the
  spread and mean quanta at the modes' own temperatures — each derived,
  stored and checked on read (built, schema 15); `postprocess.vibration(path)`;
  **the probe on D7/D8's rule** (the user's, 2026-09-28: the frame set's
  Gauss–Hermite nodes at the stated temperature), which settles `r2-4` P2 —
  its slope, the coupling $g$, the Huang–Rhys factor, the level's average and
  shift over the state, the even part's share, the mode's harmonicity and the
  SCF's noise, each named as an estimate, a degenerate or crossing orbital
  marked; and *(the user, 2026-11-22)* the zero-point amplitude recalculated
  from any frequency, the distribution of a mode's coordinate in each state
  ($n$ quanta, thermal, a swing, classical) with its moments and frames, each
  atom's move for a coordinate given and back, all validated by tests whose
  expectations are the science's closed forms. A science reader's review
  (2026-11-22) corrected the first draft (D8 missed, estimates called exact,
  the noise yardstick, an imaginary mode the person named); a second its
  revision.
- **The user's decisions, 2026-11-22** (§ 6.11 § 9): `jobset migrate` removes
  `displacement_amplitude_ang`, saying so, and writes the probe's temperature;
  the frame-set maker offers $n$ quanta and a swing beside the thermal bell
  curve, the probe following — the transport calculation reads the weights
  alone; the result stores the force constants, `with_masses` recomputing for
  isotopes; SIESTA's Raman step a choice (the force-constant step, or a
  fraction of the zero-point width) and its view's per-mode verdict by the
  relative rule; the probe's record holds the weights' numbers only, its
  slope, $g$ and $S$ views; every temperature explicit, the API's default the
  record's — narrowed by the user on 2026-11-23 for a driven mode, whose
  coherent state takes no temperature ([`engines/vibration.md`](?doc=engines/vibration.md)
  § 6.11 § 9 item 6, quoted there).
- **Its documents**: [`engines/vibration.md`](?doc=engines/vibration.md)'s
  two diagrams redrawn — they stop at the Results tab and show neither the
  Raman optical calls (with the run's own copy of their files, 2.77.21), nor
  the data files, nor the ways out (Export, the Python API, a frame set into
  transport); [`engines/vibration-postprocessing.md`](?doc=engines/vibration-postprocessing.md)
  given a diagram and the sections it lacks (the zero-point fields, frame
  sets, `reaveraged`, an `outermost_move` example); every quantity's
  definition in the science chapters
  ([`science/normal-modes.md`](?doc=science/normal-modes.md),
  [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md),
  [`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)).

### 2. R3d2 — the vibration framework

R3d1 shipped (2.77.25, 2.77.26). R3d2 builds the framework (§ 1) in five
releases, each read whole. **The design was re-cut on 2026-11-23** *(the
user: "vibration calculation is the foundation that exposes the fundamental
data of all modes"; "transport calculation should not overreach to design
what the multiframe is"; molbuilder exposes "the correct facility/api/tools
to help user to generate the correct multiframe data")*: the vibration
calculation and its record are the foundation; the motion toolkit
([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)) makes
frames from the modes; transport reads frames and weights alone. **Before any
code: three review rounds of the documents, the design, the API, the
framework and the examples, each with new readers among them a scientific
and an API reviewer** *(the user, 2026-11-23)*.

- **R3d2a** (2.78.0, done) — the module `vibration`, its states and
  `postprocess.vibration`'s mechanics.
- **R3d2b1, the record** (2.79.0, done; schema 15, data file @11; `engines/vibration.md`
  § 6.11 § 3, § 5), the user's choices of 2026-11-23: each free atom's share
  of every mode's motion, $m_A\lvert\mathbf L_A\rvert^2$, stored on the
  `free_atom` axis (the shares by element, by species label and by region its
  sums, on request; the spectrum tab's element shares from them); the largest
  zero-point move; each mode's reduced mass and force constant by Gaussian's
  convention, $\mu = 1/\sum\lvert\mathbf L\rvert^2$, $k = \mu\omega^2$
  (`reduced_mass_amu`, `force_constant_mdyn_per_ang`, the definition stated);
  each mode's spread and mean quanta on the modes' own temperature axis (its
  temperatures: § 6.11 § 3's table, the user's 2026-11-23 decision) — in
  `/modes/thermal`, a group of its own over its own `temperature_k` axis, so
  the probe's group inherits no temperature axis (the group repeating the
  `mode` coordinate); an imaginary mode's spread, mean quanta and largest move
  `null` in the JSON (whose writer refuses a number that is not finite) and
  NaN in the data file; its force constant positive, $\mu\lvert\omega\rvert^2$,
  as Gaussian prints it; `vibration.mean_quanta` refusing a frequency that is
  not positive; `mdyn/angstrom` a row of the data files'
  units (`model/datafiles.md` § 2c, `ncdata.UNITS`); every value derived by
  the module, written by both engines, checked on read; each atom's element
  its species label's name (`Structure.chemical_symbols`, travelling), prep
  having made it an element; `Mode`'s fields by the record's names; the load
  door summing the shares the result states — checked on read and worked out
  again by the one derivation — by element.
- **R3d2b2, the motion toolkit** ([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md);
  its § 9 is the one list of each piece and each visible change of a built
  call, transport's own in `engines/transport.md` § 2a.12; the design's three
  review rounds, `design-round{1,2,3}`, and its last fixes decided it).
  **Built in releases**, each with its own review rounds: **R3d2b2a
  (2.80.0, done), the motion in `vibration`** — the travelling module's items (the pointer
  below: the states, every state's Gauss rule and its order, `draw`,
  `weights_at`, `temperature_of_quanta`, the `Vibration`'s shared fields,
  `Motion`, `Snapshots`, `pooled`) and the callers of a changed name;
  **R3d2b2b1 (2.81.0, done), transport made generic and the vocabulary the
  toolkit's** — made first, so a set the new writer writes is never laid out
  by transport's own table of rows; its pieces and visible changes are
  [`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 9's
  *Built (2.81.0, R3d2b2b1)* list and the changelog's row; **R3d2b2b2, the host toolkit's writer and reader** —
  `vibration_of_run`, `write_frame_set`, `read_frame_set`, `postprocess`'s
  doors, `frameset from-mode` and *Write frame set…* for any state, the load
  door's block, `spectra.mode_frames` retired; **R3d2b2c, the example
  notebook** and its test:
  - **`vibration`** (L2, travels; numpy and the constants alone) — R3d2b2a,
    built in 2.80.0: each piece and each visible change is
    [`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 9's
    *Built (2.80.0, R3d2b2a)* list, which the plan does not restate.
  - **`vibration_frames`** (L2, the host's; imports `vibration`, the
    structure, `frameset`, the result's reader and the run door, never
    transport; imports with `molbuilder-jupyternb`'s packages). Carried from
    R3d2b2a's review: the read door states `v.result_sha256` whichever file
    is read (the JSON's bytes, the `.nc`'s `json_sha256`); a `Snapshots`
    read back prints and analyses without a live vibration; § 5.1's
    `quantum` row stated for a mode's own set too (a classical state);
    `info.sampling` records each drive group's key (the modes it was made
    with), each set-apart mode's `temperature_from`, each scaled frequency
    and the modes left out as imaginary, so `read_frame_set` rebuilds the same
    motion and its words;
    and `postprocess` hands on `vibration.pooled` (below) —
    `vibration_of_run(run_folder)`, the one door from a run's folder to its
    `Vibration` (the run, the JSON's sha256, the structure, frame 0's
    relaxation record — `mode_frames._relaxation_record`'s rule, moved here),
    asked by `postprocess.vibration`, `frameset from-mode`, *Write frame set…*
    and `read_frame_set`; the one frame-set writer in its two forms,
    returning `Written(path, n_frames, said)` (frame 0
    the relaxed geometry — prepended at weight 0 for an even rule, snapshots,
    and a person's frames whose first is not exactly it, a prepended frame 0
    stating every frame row at 0; each frame's measured rows; the provenance
    as rows; the `assumptions` row; `info.sampling` from the snapshot recipe;
    rows and frame rows `{name, value, unit, note}`, a row with no unit `"1"`;
    a person's `frame0_rows` for a prepended frame 0, a person's frame row
    named like the writer's own refused by name; a snapshot set's `seed`,
    `start` and `samples` rows, a pooled set's `pooled` 1 and its parts in
    `info.sampling`; the `assumptions` row with science § 7 items 8–9 for a
    driven set;
    weights not summing to 1 written and said; a person's frames moving a
    held atom said; a scaled mode's `result_frequency_cm1`,
    `frequency_scale`) and its reader `read_frame_set(path, *,
    vibration=None)` (every frame, frame 0 included: `Frames`, `Snapshots` or
    `FrameSet`); `outermost_move` for any
    state (behind `postprocess.outermost_move`, which takes any state's
    arguments — a visible change) and `at_temperature`, folded from
    `spectra/mode_frames.py`; each row's meaning for its kind of set (the
    vocabulary built in 2.81.0 states one meaning a row); `frameset from-mode` and *Write frame set…* writing a mode's own
    set in any state (`--quanta`, `--mean-quanta`, `--swing-quanta`; the
    dialog's state — the user's decision, `engines/vibration.md` § 6.11 § 9
    item 2). Carried from R3d2b2b1's review (`fix22-review.md`): one
    wording of a state for the writer's `assumptions` row and `said_of`'s
    older-set words (today three: `vibration._state_words`,
    `vibration_frames.thermal_state_words`, `weights_for_set`'s first
    words); one reader of a mode set's rows into its state, asked by
    `weights_for_set` and `read_frame_set`; a public lookup of a vocabulary
    row (the generator and `said_of` each index `ROWS` today); `FrameRows`'
    rows beside the reader's arrays; `frame_rows` the function beside the
    writer's `frame_rows=` keyword; `quanta=` named on a warm set's rows
    leaving its temperature unsaid, and naming nothing on a warm-driven or
    classical set's rows giving the quantum thermal state — said where the
    set states its own state. **No call of `vibration` reaches it.**
  - **`postprocess`** — hands on `write_frame_set`, `read_frame_set` and
    `pooled`; `vibration(path)` asking `vibration_of_run` for a run's folder;
    `outermost_move` for any state.
  - **The electrode check stays transport's** (`compose.frames_moving_the_leads`,
    its tolerance `wizard.FROZEN_TOL_ANG`): the toolkit's writer checks only
    its held atoms. **A visible change**: `frameset from-mode` and *Write
    frame set…* no longer say which electrode atoms move — a transport
    calculation citing the set says it; `compose.frames_moving_the_leads`'
    docstring follows.
  - **The load door's frame-set block** `{mean_quanta, spread_amu12_ang,
    spread_over_zero_point, outermost_move_ang}` per state (`sigma_amu12_ang`
    and `sigma_over_zero_point` renamed — a page change listed with the
    deletions; `web/web-api.md`, `web/spectra.md` § 4.3).
  - **Documents the build brings into line**: `web/web-api.md`'s
    `/api/spectra/load` and `/api/spectra/frameset` rows
    (`spectra.mode_frames.NODES`); `web/spectra.md` § 4.3 (the state choice,
    the writer, the electrode clause) and its `spectra.mode_frames` mentions;
    `engines/vibration.md`'s `spectra.mode_frames` mentions (§ 5.11's phases
    table, § 6.6); `science/vibrational-averaging.md` § 5's
    `spectra.mode_frames.outermost_move`; `science/normal-modes.md`'s status
    row (§ 4c) naming `frameset_from_mode`; every mark *(to build,
    R3d2b2b2)* and every "the motion toolkit's one writer" sentence marked
    to build until then.
  - **Tests** (R3d2b2a's rows of `tests/test_vibration_mechanics.py` and
    `tests/test_vibration_mechanics_e2e.py`, and R3d2b2b1's of
    `tests/test_vibration_frames.py`, `tests/test_frame_set_meanings.py` and
    the transport e2e, are built): `tests/test_vibration_frames.py`'s writer
    and reader rows (R3d2b2b2) and the notebook's run
    (`tests/test_vibration_frames_notebook_e2e.py`, on
    `molbuilder-jupyternb`'s Python, R3d2b2c); `tests/spectra/test_blueprint.py`
    (`test_a_frame_set_asked_without_its_temperature_or_rule_is_refused` — the
    write door now takes a state's arguments, so `{quanta: 1}` with no
    temperature is a valid ask); `tests/spectra/test_derived.py`
    (`mode_frames.at_temperature`, moved to `vibration_frames`, and the
    `sigma_over_zero_point` key, renamed);
    `tests/test_transport_on_a_real_junction_e2e.py`'s `from-mode
    --temperature` and the electrode clause;
    `tests/test_spectrum_results_e2e.py` (`mode_frames.at_temperature` and
    `NODES`, folded; its `sigma_amu12_ang`).
  - **The example notebook** `docs/engines/vibration-frames.ipynb` on a PySCF
    methanol run folder the repository carries beside it — made down the
    road by a test helper (`support.real_runs`) — its kernel
    `molbuilder-jupyternb` ([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)
    § 6.1; `web/jupyter.md`, `ops/installation.md`).
  - **The page's state choice**: *Write frame set…*'s state — thermal, $n$
    quanta, driven (warm optionally) — its readout per state, and the load
    door's `frameset` block per state ([`web/spectra.md`](?doc=web/spectra.md)
    § 4.3).
  - **The copies `fix19-review.md` named**: the maker's `_ordered_nodes` and
    its inline largest move, `frame_average`'s swing moments, the page's
    `onWriteFrameSet` largest zero-point move and `_swingReport`, the load
    door's own reading of the record's temperature for a frame set, the
    frame rule's default three nodes beside `mode_frames.NODES`, and
    `tests/spectra/test_derived.py` renamed for its rows.
- **R3d2c** — the probe at the frame rule's nodes (§ 6.11 § 4; § 9 items 1,
  5, 6) with the PySCF deck changes below.
- **R3d2d** — the force constants stored and `with_masses` (§ 9 item 3),
  SIESTA's Raman step and its verdict (§ 9 item 4), the data files on the
  shared writers and the diagrams (§ 6.11 § 8).

- **The vibration framework, built** (§ 1, after the user's yes), with the
  SIESTA vibration's data files on the shared writers
  (`ncdata.copy_stated`, `ncdata.json_attrs`, as the transport file and a
  chart's use) — today they copy a record's values by their own `key_of`
  loops and encode their attributes by hand; and the status door's
  unreadable copy note in its own field, where the transport record says
  `unreadable`; the record's and the data file's builders are long functions.
- **With the PySCF decks' change, said as its visible change**
  (`fix18-review.md` round 1 #2): the progress log's writer told its engine
  and writer by the deck that builds it (`r2-6` F15's third part), and the
  PySCF script's result stamped by `runrecord.stamp` (F9's PySCF part) —
  each a statement a launched PySCF stage's deck is compared by
  ([`execution/job-request.md`](?doc=execution/job-request.md) § 5), so the
  release that makes them says a launched PySCF stage is started again.

### 3. R3e — parse, structure and builders

- **Carried from R3c** (`fix16-review.md`): `cell.py`'s note and
  `structure.py`'s two restate the vacuum floors `validation.siesta`
  holds (`VACUUM_MIN_NEUTRAL_ANG`, `VACUUM_MIN_CHARGED_ANG`) — the floors
  move below `validation` so the structure's notes ask them.
- **Carried from R3d1a** (`fix17-review.md` round 2 #16): the electrode
  builder in `modify.py` types its axis kinds, `("periodic", "periodic",
  "transport")`, beside `structure.AXIS_KIND_TRANSPORT` and
  `ACROSS_TRANSPORT`.
- **Two migrations** *(the user, 2026-11-21: "do the run dir and sidecar
  migration in R3e")*: a calculation's records from before 2.60.0 — a
  `job-set.json` holding the retired `continue_retries`, as
  `AuBDTAu/transport/lowbias2`'s does; the v6 structure sidecars (`hemeC`,
  `hemeC-dithiol`), whose refusal's way out, *open the structure in Modify and
  save it*, cannot be followed, since Modify reads the pair through the same
  reader.
- **The user's R2**: a structure title pasted with a line separator (U+2028)
  cannot be saved — the title is one line, at its door. `builders` resolves
  `module:NAME` itself (left by R3a). H-U4 (= H-D1, modified residues'
  placement) and H-U5 (= H-D5, DNA input drops dashes and digits): check
  whether done.
- **Recommendations, no question needed** (`r2-7`, `r2-8`): #5 keep refusing
  an `.FC` overflow, one D/d exponent reader; C-6 MolView's single-field cell
  operations stay `block`s; C-4 a labelled O/N gets the same dipole warning
  (a visible correction, said).
- **Parse** (`r2-7`): #1 the parsers build `TrajectoryResult` directly (+ T3);
  #18 one `iso_z`; #2 the recorded contract set from the catalogue's citation
  tag, lazily (+ T4 retired); #3 four unread `FdfParams` fields deleted
  (shown first); #4 `targets_of(stage=)`; #5 `fortran_float` with D/d; #10
  the FAILED tuple; #7 one series tuple; #8 key tuples; #19 a narrow except;
  #9 `ENVELOPE_KEYS`; #14 `siesta_reader`'s two methods.
- **Structure and builders** (`r2-8`): C-3 symbols from ASE; C-4 one
  neighbour function; C-6 one reader per field and C-7 `Issue.to_json`
  (`periodicity_gate`); C-10, C-24, C-19 concat (`structure`); C-12, C-23
  (`atom_permutation`); C-13 one plane table, C-20 `detect_layers`, C-22,
  C-16 the thresholds in `cell`; C-14 `pyscf_numbering` deleted (shown
  first); C-21 residues.

### 4. R3f — the web server

- **The page's Prep runs `jobset prep` as its own process**, so it is never a
  server's stale code — designed and shown to the user first.
- **The side rail's height**: a held rail's cap is the window less a margin
  (`form-components.css` `.page-aside`, Task setup and Transport) while the
  pane under the header is shorter, so a rail taller than the pane holds its
  top and its end is reached with the page's end (measured 2026-11-19). A cap
  true to the pane needs the shell's content pane as a size container — a
  change to every page's shell: measured on the real pages, then the concrete
  change brought to the user before it is made *(the user, 2026-11-20)*.
- **Deletions, shown first** (`r2-9`): #6 `/api/files/write`'s
  overwrite/auto_rename/merge branch (its page side in R4); #26 the blueprints'
  `watch` re-export and its stale docstring line.
- **One door each** (`r2-9`): #1 checkpoint routes classify by
  `checkpoint.InputError` / NoSuchRef 404 / other 500, their blank checks
  and "set up" dropped (`Repo.init`'s default); #2 `modify` exports `_AXES`
  and the centre tuples; #3 the lattice's reference columns named once,
  `_lattice_notes` returning Issues; #4 the selection route reads only the
  labels and names it uses (moving it onto the structure envelope is R4's,
  with MolView); #7 one `notebook_env(caps)`, one `ask_start(port)`,
  `serve_daemon` naming its START/STOP signals; #9 + #30
  `parse.types.trajectory_refusal`, one directory sentence; #10
  `engine_table` carries the form prefix; #11 `scheduler.record`
  `label_of`/`name_of`; #12 `runtime_config.email_key` everywhere; #13 the
  rate-limit row's type and default in one place; #14 the monitor exports
  its envelope keys, `notify._FIELDS` from them and `report_fields.NAMES`;
  #15 `notify.listener_path(seg)`; #16 `references.cited_keys(text)`; #17
  `watch`'s `_answer`; #18 OAuth, one record per kind; #19
  `build_google_entry`'s dead checks dropped; #21 `bibliography_keys` once;
  #23–#25, #27 the low ones (`is_loopback`, `request_hostname`, the auth
  callback path, the config folder's shell spelling, `bounded_int`,
  `said_yes`).

### 5. R4 — one home per fact, the pages and the tests

- **Carried from R3c** (`fix16-review.md`): the path tool's
  (`tools/classify_path_finders.py`) stale `launch` row and its `--check`
  failure on `jobset/submit.py`'s `rglob`, both older than R3c.
- **Carried from R3d1b** (`fix18-review.md` round 2 #5, #6, the full-text
  readers' D7/B2): the spectra page types the charge partitions' options and
  default (`_spectra_inspector.html`, `lib/spectra/core.js`) and compares the
  route words as literals in its run summary; a point's words typed by the
  pages (`lib/state-chip.js`, `lib/trajectory/core.js`, the transport
  inspector's report head) and by the walk script's log lines
  (`jobset/submit._walk_script`); `transport/record`'s *not ended* beside
  `raman.NOT_ENDED`.
- Round 2's items for the pages, MolView, the viewers and the test harness
  (`r2-10`…`r2-14`) — the Results tab's file-menu pick copied in ten test
  files, its door now `support.sidebar.result_picked`; every T row the
  verdicts send here: `r2-1` T2–T7, `r2-2` T1–T3, T6, T7, `r2-3` F12, F13
  and its T rows, `r2-5` transport T1–T7 and `r2-6` spectra T1–T11 with the
  `gf_files.estimate_from` test, `r2-7`/`r2-8` T1, T2, T-2…T-9 and
  `TestJsonSafe`, `r2-9` T1–T4.
- The warning that renaming a file to a name starting with a dot leaves the
  sidebar; the scratch-env maker written twice in the tests
  (`test_envs_install.py`'s `scratch_host_env`, the M6 file's
  `real_manager`/`_made_env`) made one suite door (`fix15-review.md`).
- **The user's questions**: `r2-9` #4, the selection route onto the
  structure envelope (a design, with MolView); MolView's atom filter.
- The readers' proposed deletions, each shown to the user before it goes.
- F9's last items: four page stand-ins without a named need (the Task setup
  save bar's refusal, the export door's answers, the save door's page, the
  checkpoint Cancel's signal) and the installer's repair audit.

### 6. R5 — the documents and the proof

- One docs pass, no version: the ~80 chapter sentences that need no code
  (`TRIAGE-round1.md` § I), `r2-1` F2's CLI sentence, F3 (SIESTA keywords in
  a shared refusal, only a typed `--from`) and F14, `r2-9` #22, `r2-5` #14,
  `r2-7` #15; § 7 step 4 of the document survey — what merges, nothing moved
  until what it says is written where it goes.
- **The finish line's proof**: the full batch, and the three roads walked
  again in the browser on real runs.

---

## Waiting on the user

- **Next**: nothing — R3d1b is shipped (2.77.26); R3d2 runs.
- At their releases: R3f's Prep-as-a-process
  design, the side rail's change after measuring, and deletions #6, #26; R4's
  selection-route design and MolView's atom filter; every deletion, listed
  before its commit.
- Tooling, not molbuilder: the contract-discipline plugin
  (`Qing-LAB/claude-plugins`) made public once proven (the user's yes);
  its briefs still invite reading "compare with the last release" as an exact
  match — to be reworded to *correct against the chapters* *(the user,
  2026-11-20: "check for correctness not for exact match")*.

## Rules in force

- **Nothing stops a calculation the person set** *(the user, 2026-11-18)*: a
  scientific finding is said and the run goes as set; an engine's own stop is
  said as information and the deck written as asked *(2026-11-02)*. What may
  still be an error is [`science/overview.md`](?doc=science/overview.md) § 4's.
- **Every warning in plain words** *(2026-11-18)*: what happens to the
  calculation, then what to do, in the pages' words.
- **A fix is framework-level** *(2026-10-31, again 2026-11-19)*: one door, one
  API, one data shape per fact; a defect seen at several places is fixed in
  the door they share; a needed mechanism is extended where it lives, never
  patched beside it.
- **What runs in an env is its own** *(2026-11-21)*: named by its path, never
  `PATH`'s or the host's ([`ops/installation.md`](?doc=ops/installation.md)
  M6); **every printed command is typable as printed**, from any folder
  ([`execution/job-system.md`](?doc=execution/job-system.md) § 5.3); an env
  command by the checkout's full path, and in a file for another machine or
  a run script as typed in molbuilder's checkout *(2026-11-22: "Launcher by
  full path", "Full path here, words there", "Run script: words
  everywhere")* ([`ops/installation.md`](?doc=ops/installation.md) § 2).
- **The job side and the result's reader change together**, with their
  documents and diagrams *(2026-11-21)*.
- **Off the plan** *(2026-10-18)*: every feature the three roads do not need;
  any check, warning or format the contracts do not state. The AuBDTAu
  workflow is the user's, on Sol.
- **The test suite never reaches the person's machine**
  ([`process/testing.md`](?doc=process/testing.md) § 2b).

## Open beside the finish line — low harm, waiting

- `r2-4` P24(d), the SIESTA layout's two-line pair, and P24(e), its raw
  `systemlabel`/`numberofatoms` keys — the R3 triage left them (low harm).
- The page doors' rest — O-STATE's one save door, O-WORDS, O-KEYS with X-UI
  and X-PLOT — internal tidying no page shows; the test framework's TF3–TF10
  and the page tests on jsdom.

---

## The finish line *(the user, 2026-10-30)*

molbuilder is finished when (1) every engine and kind fact is asked of its
record, every name and record goes through its owner, one reader and one
writer per format, one home for each vocabulary, default and rule (§ 1am
steps 2–8, F2–F12); (2) the three roads — an optimization, a vibration (SIESTA
and PySCF), a transport calculation — are proven start to finish in the
browser on real runs; (3) the documents are true and findable. New science,
and what only the user's Sol runs can show, come after it.

## How a release closes *(the user, 2026-09-27, 2026-10-31; as practised 2.77.16–2.77.23)*

1. Its owning contracts read end to end and its code whole; the design
   written into the contract first (a design change waits for the user's
   yes); the code; each new test seen failing on the last release at its
   assertion.
2. The fast batch (`tools/testrun.py run none2e` where a targeted set cannot
   bound the change); the package pyflakes-clean.
3. **Review**: round 1 and round 2 by new readers each (correctness, one home
   and the tests, a sweep of the whole repository where the rule is
   repository-wide, the pages where a page changed); then two new
   independent readers of every touched module and chapter section in full
   text, and a discipline reader; then a reader of the fixes. Every finding
   judged against the chapters — *does it break molbuilder's own workflow?* —
   and re-read in the code before it is fixed; a design change goes to the
   user first.
4. One e2e run, last: the e2e files whose roads the code touches.
5. The deletions listed for the user before the commit; the plan, the
   changelog row and the review record (`fix<N>-review.md`, listed in
   `archive/README.md`) brought current; a secret scan of the staged lines;
   the user's yes; the release tagged `v<version>` and pushed; the dev server
   restarted.

---

## After the finish line

The design notes named here are in
[`archive/2026-11-19-plan-consolidation.md`](?doc=archive/2026-11-19-plan-consolidation.md),
each under its section; each new piece of science comes to the user as a
design with its science first.

- **New science**: § 1ag corrections to a junction's electronic structure (a
  framework before any one correction); § 1ah which modes get Raman (after
  the run, and by symmetry); M10 from a mode to the current; § 5e a person's
  own engine text as an input; § 9.2 ATOM as an optional extra layer; S13
  transport's convergence sweep; K22 per-atom starting spins; LEAD1; the V1
  rows; § 1ac's D-h, B and the measured intensity's factors — the
  finish-line ledger's *After — science* (28 rows,
  [`archive/2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md));
  F0c-R1-S, a fixed-spin optical frame's ε₂ against one Fermi level.
- **Designed, not built**: § 5t's and § 5s's run-record pieces (the ledger's
  *Waits — run record*); N11 release 2, the page split; a deck naming the
  description that produced it (a hash in its PROVENANCE block); PDB and
  later `.cif` reading, a plan of its own (U-PDB).
- **Only the user's Sol runs can show** — E13, SCHED-HDR, BENCH-SHELF,
  SCHED-R, § 1ac's proofs 3a and 3b, the PySCF environment's `netcdf4`, the
  gcc 15 pin (the ledger's *After — Sol*, 13 rows), and the monitor's cgroup
  v1 readings (`tests/field/test_the_monitor_reads_its_job.py`), in one pass
  on Sol; and **each rank's cores** (`fix10-review.md`, round 2 #14): under
  `mpirun` with no binding flags — a GPU job under a scheduler, and every CPU
  launch — OpenMPI binds a rank to one core, so a rank's `-c`/`-omp` threads
  may share it; a Sol GPU run's session log says each rank's cores
  (`molbuilder[rank i/n] … cpus=`), and a hybrid CPU build has no such line
  yet. The binding is decided on that evidence (`running-a-job.md` § 3.3).
- **When needed**: a real fully relativistic pseudopotential — take one from
  PseudoDojo's `nc-fr` set into `tests/fixtures/psml/` and read
  `relativity="dirac"` as spin-orbit on it.

**Still the user's words** (no work waits on them): a result written before
2.49.0 keeps its stored Methods sentence beside the free atoms' sentence;
`PLACE_NOTE_UNKNOWN`'s words; `envs/recipes.py`'s comment calling ASE a
monitor reader.

**Do not "fix"** the structure API's confirmed non-findings (re-read
2026-10-08): they are listed in
[`archive/2026-11-19-plan-consolidation.md`](?doc=archive/2026-11-19-plan-consolidation.md),
*Confirmed non-findings*.

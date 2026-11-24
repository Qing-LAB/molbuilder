# The motion toolkit's built lists (2.78.0–2.81.1), archived

**Role:** record · **Domain:** archive · **Companions:** `engines/vibration-frames.md` (the toolkit's contract),
`process/changelog.md` (each release's visible changes), `plans/plan.md` (what is to build)

`engines/vibration-frames.md` § 9 as it stood on 2026-11-24, moved here whole when the documents were
consolidated *(the user, 2026-11-24: "consolidate document and plan and archive correctly. stop using multiple
sources without good reason")*: its *Built* lists repeated the changelog's rows of 2.78.0, 2.80.0, 2.81.0 and
2.81.1, and its *To build* list the plan's R3d2b2b2 item. The changelog's rows and the plan are their one home.

---

## 9. Built, and to build

**Built** (2.78.0, [`engines/vibration.md`](?doc=engines/vibration.md)
§ 6.11): one mode's states — thermal, $n$ quanta, a swing, classical — their
densities, moments and spreads; `Mode`, `ModeState`, `Vibration` and
`postprocess.vibration(path)` of a result file.

**Built** (2.80.0, R3d2b2a — the module `vibration`), each a visible change
where it changes a built call:

- **a driven mode named by `swing_quanta` alone is the coherent state** (in
  2.79.0 it took the record's temperature for a bath); `mean_quanta` beside
  `swing_quanta` left the table (a warm driven mode by `temperature_k`); the
  classical swing carrying the classical bell curve (`classical=True` with
  `swing_quanta` and `temperature_k`);
- every state's frames its own Gauss rule — the swing's included (in 2.79.0
  its equal phases folded) — in the order $(0, +q_1, -q_1, \dots)$, each rule
  exactly symmetric; `Frames.q_amu12_ang` (frames × modes; in 2.79.0
  `nodes_amu12_ang`, one row — a bell curve's and $n$ quanta's ascending, a
  swing's by its phases), `Frames.state`, an even rule's `Frames` with the
  relaxed geometry as frame 0 at weight 0; `Frames.weights` and
  `Frames.positions_ang` arrays (in 2.79.0 tuples); a `Frames` compares as
  one object, its arrays not compared (in 2.79.0 by value);
- `s.draw`, `s.weights_at` (`Weights`), `s.swing_amplitude_amu12_ang`,
  `s.carried_spread_amu12_ang`, `cdf` on arrays — $n$ quanta's from a
  tabulated distribution, its last digits moved —
  `vibration.temperature_of_quanta`;
- `v.read_path`, the path read (in 2.79.0 `v.source`, the file's name);
  `v.structure`, the structure the result states; the fields every `Mode`
  carried in 2.79.0 — its free atoms, masses, elements, positions, the
  record's temperature, species labels, regions and the modes' temperatures
  — the `Vibration`'s, reached through `m.vibration` (a `Mode` is built
  through a `Vibration`; two modes compare by number, frequency, whether
  imaginary and the result's frequency, their arrays and vibration not
  compared, and a `Mode` is hashable; two `Vibration`s compare as objects,
  two reads of one result unequal, where in 2.79.0 comparing them raised);
  a `ModeState`'s repr its state alone, no longer its mode, and its state's
  fields handed on (`swing_quanta`, `quanta`, `bath_quanta`,
  `zero_point_amplitude_amu12_ang`, `classical_spread_amu12_ang`); `v.mode(n)`,
  `v.displaced({...})`, `v.with_frequency_scale(...)`
  (`m.result_frequency_cm1`); `v.motion()`, `Motion` with `with_mode` and
  `with_drive` — a drive group's phase stream keyed by the modes it was made
  with (`Drive.key`), kept when a member leaves — `Snapshots`
  (`standard_error` → `Estimate`, `parts` → `Parts`), `pooled`.

**Built** (2.81.0, R3d2b2b1 — the module `vibration_frames` in part, and
transport made generic), each a visible change where it changes a built call
or a written set; transport's own — the data file at `transport-data@14`
and its CSV columns, the record's keys at `transport-result@19`, the describe
answer, `summarize`'s rows — are
[`engines/transport.md`](?doc=engines/transport.md) § 2a.12's and § 3.8.1's,
and the changelog's row:

- the rows' vocabulary is the toolkit's (`vibration_frames.ROWS`, § 5.1),
  with `MASS_CHANNEL` and an older set's `INFO_KEY`, and what a page says
  of a cited set is its `said_of` (§ 6, the one home of its words) — the
  words, what the set is and its provenance, of the frames the page draws,
  and a pair changed since its record pinned it;
- `vibration_frames.weights_for_set` (§ 7), handed on as
  `postprocess.weights_for_set`, and `postprocess.frame_rows`;
  **`postprocess.reaveraged` for any state** — `reaveraged(path, *,
  temperature_k=None, swing_quanta=None, quanta=None, mean_quanta=None,
  classical=False)` (in 2.80.0 a temperature required and a swing on it):
  a thermal or classical state given no temperature takes the set's own
  `temperature_k` row, the state named; a swing named alone on a set
  stating a temperature above 0 K says that temperature is not used; a row
  in another unit said; a set stating none refused by name; a swing
  reaching past the outermost frame averaged and the share beyond said (in
  2.80.0 refused); answered `{bias_v, energy_ev, transmission,
  conductance_g0, conductance_change_percent, weights, said, assumptions}`,
  `weights` the `Weights`, `assumptions` transport's own then the set's,
  for the state it was written for (§ 7) (in 2.80.0 `{motion, frames, weights, reach, …}`,
  `weights` a list, no assumptions);
- `vibration.THE_SETS` (`"the set's"`), and the temperature refusal naming
  what states none;
- **`s.weights_at(q, *, written=None)`'s degree and words** (2.80.0's
  call): `exact_to_degree` the degree the weights reproduce the state's
  averages to, each checked — $2m - 1$ for the state's own frames, $m$ for
  an odd number of symmetric frames and a symmetric state (in 2.80.0
  $m - 1$ always); the share beyond and three frames' blindness said only
  for frames placed for another motion — not for the state's own rule, nor
  for frames whose written weights, given, are this state's (in 2.80.0 for
  any); a weight below zero only past rounding; every frame named from 1
  (in 2.80.0 *frame 0*, *frames 1 and 2*); the share beyond at its
  significant figures;
- a mode's own set written before 2.81.0, which states no `assumptions`
  row, has its harmonic assumptions said beside it (§ 6) and drawn on its
  card;
- **the sets the frame generator writes** (`frameset from-mode`, *Write
  frame set…*): the spread row `spread_amu12_ang` (in 2.80.0
  `sigma_amu12_ang`), the provenance as the rows `vibration_run` and
  `vibration_result_sha256` (in 2.80.0 `info.vibration`), the `assumptions`
  row (`assumptions_of_a_mode`), the rows in the vocabulary's order, a row
  of no unit written with `1` — each read the same by the toolkit from an
  older set.

**Built** (2.81.1, R3d2b2b2a — a row's unit said in one place,
[`model/structure.md`](?doc=model/structure.md) § 2.2d), each a visible
change of a written set or a built call:

- the frame generator's sets write no unit beside a row whose name carries
  it (`frequency_cm1`, `temperature_k`, `displacement_ang`, …; in 2.81.0 the
  unit in both places), `1` beside a count, none beside a text (in 2.81.0
  `1`); the vocabulary's unit of a row is its name's (`Row.unit` read by
  `units.unit_of_name`, its page unit from the units table,
  `units.UNIT_ON_A_PAGE`);
- a set's rows written through a script's door hold the unit rule
  (`set_customized`, `with_frames`' frame rows; in 2.81.0 any unit kept);
  `weights_for_set` takes each row in its name's unit and says nothing of
  another (in 2.81.0 said): no row is made so now, and a set written before
  is read as written (the user, 2026-11-24: no work for older records).

**To build**, R3d2b2b2b and R3d2b2c ([`plans/plan.md`](?doc=plans/plan.md)
§ 2), each a visible change where it changes a built call or a written set —
this list and the ones above are the one home of each of the toolkit's
changes:

- `v` from a run's folder (`vibration_frames.vibration_of_run`), filling
  `v.run` and `v.relaxation_calculation`, and `v.result_sha256` at the read
  door;
- `vibration_frames`: `vibration_of_run`, `write_frame_set` (→ `Written`),
  `read_frame_set`, `outermost_move` for any state and `at_temperature`;
  and `postprocess` handing them on;
- **`postprocess.outermost_move` for any state** (§ 4) — today
  `outermost_move(path, *, mode, temperature_k, nodes=None)`, a thermal
  state's alone, its temperature required;
- **the sets the doors write**: a snapshot set's `seed`, `start` and
  `samples` rows (a pooled one's `pooled`), `info.sampling`, an even rule's
  frame 0 prepended;
- **the frame-set doors no longer say which electrode atoms the frames move**
  — `frameset from-mode` and *Write frame set…* say it today; from
  R3d2b2b2 a transport calculation citing the set says it, as it already does;
- `molbuilder frameset from-mode` and *Write frame set…* for any state; the
  load door's `frameset` block per state, its keys `spread_amu12_ang` and
  `spread_over_zero_point` (today `sigma_amu12_ang` and
  `sigma_over_zero_point`; [`web/spectra.md`](?doc=web/spectra.md) § 4.3);
  `spectra.mode_frames` retired;
- the example notebook and its test.

# F12 round 1 — slice E: the transport calculation (TranSIESTA / TBtrans)

Reviewer read, end to end: `docs/engines/transport.md` (all 6927 lines),
`docs/engines/transport-postprocessing.md`, `docs/science/junction-cell.md`;
and every function of `molbuilder/transport/*.py` (all 22 modules),
`molbuilder/ncdata.py`, `molbuilder/postprocess.py`, `molbuilder/chartdata.py`.
Callers/callees followed where a claim needed it (`validation/transport.py`,
`validation/siesta.py`, `web/blueprints/transport.py`, `jobset/_cli.py`
summarize, `cli.py` main, `task.py` bias checks, `deck_record.py`,
`continuation.point_rows`, `runpoints.rows_of`, `cell.entering`,
`structure.restored`, `data/catalogue.template.toml`, `data/chart_table.toml`).

Checked read-only with the molbuilder python (nothing written): the k-point
counts of § 0.3 (4×4→10, 12×12→74, 18×18→164, 30×30→452, 50×50→1252, 5×5→13),
pole counts (10 eV→123, 5 eV→61, 4 eV→49, 1.5 eV→18, least for 20 at 300 K
1.63 eV), the bias-window line (0.2 V: 46 points at 0.01 eV, 23 at 0.02 eV),
the window fit (−3…3 on −2…2/401 → −3.00748…3.00748 at 603; on the
catalogue default → −3.005…3.005 at 601), the sharing contours (601-point
extension: 2 lines, 401 held, 200 computed; cut three: 1203, 802 computed;
`in_common` 1 for −3…3/601, 0 for −3…3/600), the electrode block emitted for
a sorted 22-atom chain (L/R listed by label, `elec-pos begin 1` /
`elec-pos end -1`, GF path `../tsgf/T.v0.2.TSGFL`), and a sharing deck's
SHARED block round-tripping through `Contours.of_deck` / `deck_window`.
All agree with the chapter.

Totals: **2 defects, 5 untrue sentences** — all of low to moderate weight.
No defect was found on the main roads (describe → prep → gather → walk →
record → data file → export → PDOS); the ones below are on edges.

---

## Defects

### D1 — an unreadable composed junction crashes `summarize task` and the Results report (traceback / HTML 500), instead of a refusal in its reader's words

- **Where**: `molbuilder/transport/record.py:1007` (`collect_record`),
  `record.py:1492` (`selection_pdos`), `molbuilder/transport/datafile.py:662`
  (`data_file_bytes`) each call `recorded_junction(base)` unguarded;
  `compose.py:969-970` raises
  `ComposeError(f"the composed junction, {path.name}, does not read: {exc}")`,
  and `compose.py:43` defines `class ComposeError(Exception)` — not a
  `ValueError`.
- **Who catches what**:
  - `transport/sum.py:28-35` (`_run`): `except RecordError` … `except DataFileError` only;
    `jobset/_cli.py:1458-1461` catches `SumError` only; `cli.py:3793-3819`
    translates `UsageError`, `RuntimeConfigError`, `ClickException`, `Abort`,
    `_BackendUnavailable` — so `molbuilder jobset summarize task` ends in a
    Python traceback.
  - `web/blueprints/transport.py:507-510` (`/api/transport/record`):
    `except (RecordError, OSError, ValueError)`; `:611-614`
    (`/api/transport/pdos`): `except RecordError` — `ComposeError` falls
    through to Flask's HTML 500 (no app-level handler for it in
    `web/app.py:474-511`).
  - Only `transport/export.py:429` lists `ComposeError` among what it turns
    into a refusal.
- **Rule**: `transport.md` § 2a.12, *Composed on read*: "**A record that does
  not read is said, never the whole report refused**"; the docstring of
  `recorded_junction` itself (compose.py:948-950): "a pair that does not
  read is refused in its reader's words, :class:`ComposeError`". Prep honours
  it (`load_compose_record` turns the same error into a *why*,
  compose.py:1103-1106); the readers of a finished calculation do not.
- **Scenario**: a transport calculation whose `junction.xyz` /
  `junction.molstruct.json` the codec now refuses (the pair edited by hand so
  its geometry pin no longer matches, a sidecar damaged) → opening it on the
  Results tab gives an HTML 500 the page can only call unreadable; `jobset
  summarize task` prints a traceback; a selection's PDOS gives 500. The prep
  of the same folder says the reason in words.
- **Sureness**: propagation read-and-traced end to end; the trigger needs the
  composed record damaged or edited (not a normal road) — moderate/low weight.

### D2 — a transmission point whose deck's SHARED record does not read refuses the whole record (and crashes `summarize task`)

- **Where**: `record.py:1100` — `sharing = sharing_of(where)` sits outside
  the `try` that guards the rest of the point (`record.py:1125-1139`).
  `sharing_of` → `Contours.of_deck` → `deck_record.read_json_block`
  (deck_record.py:157-172) raises `DeckRecordUnreadable(ValueError)` for a
  block whose fences or JSON do not read.
- **Rule**: `transport.md` § 2a.12: "A record that does not read is said,
  never the whole report refused: a `.gathered-from`, a run's own records or
  its copy note that does not read is said on its rung (`unreadable` …)".
- **What happens**: the exception leaves `collect_record`; the record route
  answers 400 for the whole report (the `ValueError` catch,
  web/blueprints/transport.py:509), and `summarize task` (sum.py catches
  `RecordError`/`DataFileError` only) ends in a traceback. The same
  exception in the `dos = joined(where, _dos)` call (record.py:1150) would
  escape too, since only `TbtError` is caught there.
- **Scenario**: one sharing point's deck (prep-written, but a person's
  USER-CUSTOM edit or a damaged file can break the `# === molbuilder shared`
  fences) → the Results tab shows no report at all for the calculation, every
  other point included; `summarize task` traces back.
- **Sureness**: read-and-traced; trigger is a damaged deck — low weight.

---

## Untrue sentences

### U1 — the data file's point groups are not "named as its folders are" where a rung has no voltage level

- **Chapter**: `transport.md` § 2a.12, *The calculation as groups*:
  "`/points/<point>` — **each point the description asks for, done or not**,
  named as its folders are (`/points/v0.2`; a frame set's `/points/f003/v0.2` …)".
- **Code**: `datafile.py:1051-1057` builds every group as
  `Point(frame=f if n_frames > 1 else None, volts=volts, …).rel` with `volts`
  always a number, so the group always carries a voltage token. Measured:
  `Point(volts=0.0).rel == 'v0'`, `Point(frame=0, volts=0.0, of=3).rel ==
  'f000/v0'`, while the transmission's real point under the low-bias
  approximation is `Point(frame=0, of=3).rel == 'f000'`, and a single-bias
  transmission has no point folder at all (its run).
- **Scenario**: a single-bias calculation's data file holds `/points/v0`
  although no `v0/` folder exists; a frame set under the low-bias
  approximation holds `/points/f000/v0` beside folders `05_transmission/run-0/f000`.
  (`model/datafiles.md` § 9.1 lists `v0` correctly; the transport chapter's
  wording is what is off; `point_folder` in each group names the real folder.)
- **Sureness**: read-and-traced, measured.

### U2 — "No usable record is one of three" — the code has more causes

- **Chapter**: `transport.md` § 1.1, *The first prep*: "**No usable record**
  is one of three: none yet …, one composed for another citation or with the
  leads named the other way round (`swap_electrodes`), or an older one whose
  junction composed again is another".
- **Code**: `compose.load_compose_record` also answers *no record* for a
  record naming no citation kind (compose.py:1088-1092), an incomplete record
  — a file of `record_files` missing (1093-1098) — a composed pair that does
  not read (1103-1106), and an order-unstated record checked with no projects
  tree, or whose citation no longer composes (1115-1126). Each then composes
  afresh when nothing is launched (`gather.no_usable_record`).
- **Sureness**: read-and-traced; wording only.

### U3 — the lead seam's "undetermined" is "fewer than three layers", not "a block thinner than one period"

- **Chapter**: `transport.md` § 7.1: "*eclipsed*, *twin*, or undetermined (a
  block thinner than one period) as a warning"; `junction-cell.md` § 5: "a
  warning where it does not or a block thinner than one period cannot say".
- **Code**: `wizard.seam_finding` (wizard.py:206-223): `if n_layers < 3:`
  → warning "the periodic seam is UNDETERMINED …", whatever the plane.
- **Scenario**: a two-layer fcc(100) or (110) lead — exactly one stacking
  period (junction-cell.md § 3.1 table: period 2) — is reported UNDETERMINED
  on the Transport card and in the lead deck's `.validation.txt`, though it is
  not thinner than one period (the code's own comment says the lead path
  cannot know the plane). The rule actually applied is "fewer than three layers".
- **Sureness**: read-and-traced.

### U4 — the chart export of a selection's PDOS is built, and does not go "through the same function"

- **Chapter**: `transport.md` § 2a.12, after `postprocess.pdos`: "A chart's
  own export of the report's PDOS of a selection, when it is built, writes it
  through the same function."
- **Code**: it is built — `data/chart_table.toml:51-53` (the `dos` chart's
  tags include `pdos`), `:223-225` (`[tags.pdos]`, `pdos_per_ev`), served by
  `POST /api/export/chart` (web/blueprints/results.py:132) — and
  `chartdata.data_file_bytes` (chartdata.py:263-370) writes the curves the
  page sends, never calling `tbtnc.summed_pdos` / `postprocess.pdos`.
  (The page's numbers come from `record.selection_pdos` → `summed_pdos`, so
  they agree; the sentence's "when it is built" and "through the same
  function" are what is untrue.)
- **Sureness**: read-and-traced.

### U5 — an extension's refusal is not placed "beside the control it names"

- **Chapter**: `transport.md` § 2a.16, *Extending it*: "**The Transport tab's
  Describe holds the limits**: … a changed fixed value, or its rename or its
  label changed is refused by name beside the control it names, nothing
  written (`transport.reuse.Extension.refusal`)".
- **Code**: `Extension.refusal` returns `Refused(message)` with no place
  (reuse.py:464-476); `transport/describe.py:351-356` wraps it as
  `Issue("error", kept.message)` — no `where` — so the finding the route
  returns (web/blueprints/transport.py:740-745) names no control to land on
  (§ 3.8.2a: "a finding's `where` lands on one control").
- **Scenario**: a Describe whose posted bag changes a seed/lead value or the
  rename → the refusal is said, by name in its text, but not beside any
  control. (On the extension layout those controls are not drawn, so the
  case is rare; the sentence still claims placement the code cannot give.)
- **Sureness**: read-and-traced to the route; the page's drawing is outside
  the slice — plausible as to what the page shows.

---

## Also noted (not findings — wording on a page, below the bar)

- `record._iv` (record.py:1573-1589): under the low-bias approximation a
  frame whose 0 V transmission point *failed* (or finished without its
  transmission) is noted "the 0 V transmission … has not run" — the point
  ran. The chapter's own phrase (§ 2a.12: "a frame whose 0 V point has not
  run") matches the code, so this is the record's wording, not a
  contradiction; it says *not run* of a point the same record lists under
  `failed`.
- `stages.rung_notes()["electrode_R"]`: "the Results tab compares their
  Fermi levels" beside § 0.2's "shows both leads' Fermi levels … and judges
  neither" — reads as *side by side*; left out.

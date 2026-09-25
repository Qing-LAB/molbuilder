# The engine offset — scope and plan *(2026-09-25)*

*The rule, the name, the operations and the checks are the contract's:
[`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 6.0.
This document is the scope and the order of work, and restates no rule. Plan row
**W33**.*

## 0. Why — four measured facts, one cause

All four were measured on 2026-09-25, on the fake-junction ladder
(`projects/claude-vib-ui`).

1. **TranSIESTA refused the device deck** — *"Electrode: L lies outside the
   unit-cell"*, 4 s in, before any SCF. The junction's box was anchored at its
   lowest atom by the electrode builder, deliberately (`modify.py:872–881`,
   *"the padding opens at the TOP"*). The stored corner was `−17.355` against an
   atom at `−17.355016`, so that atom sat 16 fm below the face in every deck. The
   relaxation and the seed ran with the same atom, because plain SIESTA treats z
   as periodic; only TranSIESTA checks. molbuilder's own `.validation.txt`
   passed the deck.
2. **The Results tab drew a box the engine never had.** `watch.py::_run_periodicity_json`
   sends no corner — so the server derives one — and takes the axis kinds from a
   `.source` pair searched in the **run** directory. A ladder keeps that pair at
   the calculation root, so none was found, the axes defaulted to `isolated ×3`,
   and the derived corner (`−0.48, −0.48, −1.177`) drew the atoms centred.
   SIESTA's corner was `(0,0,0)`, and every file SIESTA wrote is flush.
3. **The same kind of junction was centred on 2026-08-29 by accident.** Before
   `20f1cca0` (2026-09-21), transport form A left every axis `isolated`, so the
   derived corner centred it (the `Au-BDT-Au` device deck: 1.2004 Å each side).
   That commit correctly stated z = `transport`, and a transport axis's corner
   rule is `bbox_min`, which is flush. Nothing caught it because no device had
   run since.
4. **The provenance never existed, and the trajectory mixes frames.** `frame_shift`,
   the stamp § 6.1 clause 5 promises, is written and read by no code. The molwatch
   log's step 0 is in the design frame while the deck from the same prep call is
   in the engine frame (audit X4: a 45 Å jump between step 0 and step 1).

**The cause, once:** placement is re-derived by each reader, with rules that
depend on the axis kind, instead of computed once by one rule and recorded
where it was applied.

## 1. Data structure

| | today | becomes |
|---|---|---|
| `Structure.cell_origin` | a stored field (`structure.py:132` `METADATA_FIELDS`, the check `:585`, to/from dict `:896`/`:979`, copy `:1979`, the view block `:1078–1095`, the periodicity blocks `:2140`, `:2255`) | **removed** |
| the corner derivers | `resolve_cell_origin` `:679`, `expected_cell_corner` `:780`, `_derived_corner_under_explicit_cell` `:840` | **removed**. `resolve_cell` and `effective_vacuum` stay: they size a box nobody typed |
| `cell.py` | composes box + corner, and `ResolvedCell` carries `origin_is_user_owned`, `corner_was_derived`, `contains_at_world_origin` and two fractional projections | **gains** `engine_offset`, `EngineFrame`, `to_engine`, `engine_frame` (contract § 6.0); `ResolvedCell` carries `engine_offset` / `box_corner`, and the three origin fields and the second projection **go** |
| `periodicity_gate.py` | the `cell_origin` op (`:370–:624`), `BLOCK_KEYS`, the manual-origin regime and its notices | the op, the key and the regime **go**; the gate has two box states (typed, derived), not four |
| `modify.py` | the electrode builder states the flush corner (`:872–:902`); `calibrate_to_cell` (`:1137–:1170`) | the builder states **no** origin; calibrate per decision **D3** |
| a frame set | — | **one** offset, from frame 0, for every frame (contract § 6.0) |

## 2. File access

| file | today | becomes |
|---|---|---|
| `.molstruct.json` sidecar — `sidecars/molstruct.py` (writer), `parse/sidecars/molstruct.py` (reader) | schema v9, carries `cell_origin` | **v10**, without it. The writer writes v10; the reader accepts v9 and ignores its `cell_origin` (**D2** — the projects tree is real data). `engine_offset` is never stored in a sidecar |
| every deck molbuilder writes (SIESTA `.fdf`, PySCF `.py`) | an `atom-metadata` block only when there are labels; no placement record | + a **`molbuilder engine-offset` block** for every deck (cell, offset applied, axis kinds). One writer beside `script_emit.emit_atom_metadata`, one reader beside `_extract_atom_metadata_dict` |
| engine output (`.XV`, `.out`, `.STRUCT_OUT`, `.ANI`, `.MD`, the PySCF logs) | read as bare coordinates plus a lattice | read into `engine_frame()`, offset 0 stated |
| the molwatch log (`trajectory_log/format.py` ← `jobset/prep.py`, audit X4) | step 0 written in the design frame | step 0 written from `to_engine()` — closes X4 |
| exports — the Results export, and `cli.py:1163–1174` (`.XV` → pair) | set `cell_origin = None`, which then derives | engine coordinates + cell, v10, through `engine_frame()` |
| transport artifacts | form A composes the `.XV` with `cell_origin: None` (`compose.py:623`) and derives | the `.XV` through `engine_frame()`, each rung through `to_engine()`. `atom-permutation.json` and `slot-provenance.json` are unaffected (indices, provenance) |

## 3. Protocol agreement — the wire

* **One periodicity view**, built by one server function from an `EngineFrame`
  / `ResolvedCell`: `{cell, axis_kind, vacuum, engine_offset, box_corner,
  coordinates: "design" | "engine"}`. Every door that sends a structure to the
  browser spreads it, and none composes its own.
* **The doors**: `/api/structure/periodicity` (the Cell page;
  `build.py:480–494`), the structure load/save payload (`_shared.py:134–160`),
  `/api/watch/load` and `/api/watch/data` (the Results door — from the deck's
  record, or `engine_frame()` for a run made before it), the transport citation
  viewer, and the calculation pages' viewers. § 6.2 calls the last an
  "engine-calibrated view", but no door by that name exists; what those pages
  draw is established at the start of phase 3.
* **MolView** draws from `box_corner`, verbatim: `render-engine.js:329` (today
  `used.cell_origin`), `model-jobs.js:323` (today `resolved_cell_origin ||
  cell_origin`), `ui.js:2353` / `:2397` (the Cell panel's Origin line → the
  offset, read-only), `model.js:898`. The browser never computes a corner, which
  § 6.2 already requires — this keeps it true.
* **The Cell page** (`modify/periodicity.js`, `templates/modify.html`): the
  origin inputs and the Reset-origin button go (**D1**); the offset is shown.
  `demo.js:270` commits a `cell_origin` op and goes with them.
* `/api/modify/calibrate` (`web/blueprints/modify.py:656–675`) follows **D3**.

## 4. Validation

**The checks** (contract § 6.0): *the atoms fit* (fractional span `< 1`) at the
edit; *transport clearance along c* in the transport kind validator
(`validation/__init__.py`, beside the `TBT.k` checks).

**The tests** — through the road, and fewer than today:

| | what it drives | what it asserts |
|---|---|---|
| **T1** | per engine, `jobset init → prep`: a SIESTA relaxation, a SIESTA single point, PySCF, the five transport rungs | the deck's coordinates are design + `engine_offset`, every atom is inside with the rule's margins, and the deck's record equals the computed offset |
| **T2** | the Results door on a finished run | `box_corner = 0` and the coordinates are verbatim from the output. The fixture is `claude-vib-ui/optimization/au333bdt-loose`, a real flush run — it must be drawn **flush** |
| **T3** | export from Results → reload | `engine_offset = 0` (idempotent) |
| **T4** | `prep device` on the fixture | clearance at both faces along c; the lead's and the device's L blocks agree (risk **R1**) |
| **T5** | the rule itself, API-level on a measured fixture — the docstring says so | the hexagonal junction's `[6.3684, 3.746, 18.5325]`. A Cartesian implementation fails it (x-extent 10.093 > \|a\| 8.651); a re-wrapping one fails it (the 2.399 > 2.355 cut) |

Each is mutation-tested: break the rule, watch it fail.

**Retired**: the tests that pin the obsolete designs — the derived corner per
axis kind, the user-owned origin, the `cell_origin` op, the containment regimes,
the `resolved_cell_origin` wire shapes. Today that is 24 files and 210
references (`test_periodicity_gate.py` alone has 82, then
`test_structure_authority_roundtrip.py` 22, `test_web.py` 18,
`test_structure_periodicity.py` 16, `test_molview_model.py` 13). Unifying must
lower the count.

**"Nothing translates by hand"** is checked by a code-text review of § 5's
inventory at the end of phase 4, not by a lint test.

## 5. The inventory — every code site, and its phase

| site | role today | phase |
|---|---|---|
| `structure.py` | stores the corner; derives it three ways | P1 |
| `cell.py` | composes box + corner (`:186`, `:208`, `:234`) | P1 |
| `periodicity_gate.py` | the Cell-page door, the origin op and regime | P1 |
| `sidecars/molstruct.py`, `parse/sidecars/molstruct.py` | schema v9 | P1 |
| `modify.py` | builder's flush corner; `calibrate_to_cell` | P1 |
| `siesta/input.py` | hand translation (`:871`); validators get a re-derived struct (`:990`) | P2 |
| `transport/transiesta.py` | a second hand translation (`:401–404`) | P2 |
| `pyscf/input.py` | coordinates verbatim into `gto.M` (`:137`, `:547`) | P2 |
| `script_emit.py` | the metadata block writer/reader; + the record's | P2 |
| `trajectory_log/format.py` ← `jobset/prep.py` | molwatch step 0 in the design frame | P2 |
| `transport/compose.py` | form A composes with no corner, then derives | P3 |
| `cli.py` | the `.XV` export (`:1163–1174`) | P3 |
| `web/blueprints/watch.py` | the Results periodicity block | P3 |
| `web/blueprints/_shared.py` | the payload's periodicity block | P3 |
| `web/blueprints/build.py` | the periodicity door's origin op | P3 |
| `web/blueprints/modify.py` | `/api/modify/calibrate` | P3 |
| `lib/molview/render-engine.js`, `model-jobs.js`, `ui.js`, `model.js`, `demo.js` | draw / choose / show / commit an origin | P3 |
| `modify/periodicity.js`, `templates/modify.html` | the origin inputs, Reset-origin | P3 |
| `validation/__init__.py` | no transport clearance rule | P4 |
| `transport/sort.py`, `config/siesta.py` | comments naming the old field | P4 |

**The documents to sweep** in P4 (restatements; the owner is
`structure-periodicity.md`, whose superseded clauses are deleted then):
`web/molview.md` (11), `model/structure.md` (11), `web/web-api.md` (6),
`model/structure-molstruct.md` (5), `engines/transport.md` (3),
`engines/vibration.md` (2), `architecture.md` (2), `science/normal-modes.md`,
`README.md`, `model/overview.md`, `backend-architecture.md` (1 each). Dated
plans and handovers are history and are not rewritten; `plans/plan.md`'s live
rows are.

## 6. Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0** | the name, contract § 6.0, this plan | — *(2026-09-25, this commit)* |
| **P1** | data structure + file access: §§ 1–2's model and sidecar rows | T5 passes; the codec reads v9 and writes v10 |
| **P2** | every emitter through `to_engine`, each deck carrying its record | T1 passes for every engine |
| **P3** | readers and the wire (§ 3), MolView and the Cell page | T2 and T3 pass, and on the dev server the browser draws what the deck says |
| **P4** | the checks, the test retirement, the document sweep | T4 passes; the review of § 5 finds no hand translation |
| **P5** | **acceptance** — resume the fake-junction ladder (`claude-vib-ui/transport/au333bdt-t`): re-prep and re-run the seed (≈ 70 min measured), both leads (≈ 3 min each), then the device (never yet run), the transmission, and `summarize run` | the device reaches its SCF; the record is written; the Results tab shows each rung's engine frame |

The ladder's rungs 1–3 concluded on flush decks. Re-prepped under the rule their
decks differ, so strict composition will not reuse them — that is correct, and
is why P5 re-runs them.

## 7. Risks

* **R1 — the lead and device L blocks.** Their half-gaps differ by 0.23 mÅ
  (1.17725 vs 1.1775), because `c` was typed as 37.065 while span + spacing is
  37.0645. Whether TranSIESTA's electrode matching tolerates that is not
  knowable by reading; the first device run measures it. If it does not, the
  fix is one consistent gap number, not a placement special case.
* **R2 — existing runs.** A re-prep of any existing calculation renders shifted
  coordinates, so its old concluded attempts no longer count. Intended for
  transport; for a finished relaxation it means its runs do not match a re-prep.
* **R3 — egg-box.** Moving atoms against SIESTA's real-space mesh shifts total
  energies at the meV level, so an energy from before this change and one from
  after are not bit-comparable for the same structure.
* **R4 — frame sets** need one offset from frame 0, or the electrode-identity
  gate of `engines/transport.md` § 2a.9 breaks between frames.

## 8. Decisions

* **D1 — the manual origin: retired.** Settled by the rule itself — the offset
  is *"never chosen"*; the user: *"always automatically calculated based on the
  cell unit and all the atoms"*.
* **D2 — existing sidecars** *(proposed)*: read v9, ignore `cell_origin`, write
  v10. The xyz is untouched.
* **D3 — calibrate** *(proposed)*: keep it as *apply `engine_offset` to the
  stored coordinates* — the explicit save-in-engine-frame, after which the
  offset is 0. *Calibrated-then-emit ≡ emit* still holds and pins the rule.
* **D4 — calculation pages** *(proposed)*: show the design coordinates with the
  box at `−engine_offset` — the same geometry the engine gets, in the numbers
  the person authored — rather than engine coordinates.
* **D5 — the Results tab's axis kinds** *(proposed)*: those recorded in the
  deck, i.e. what molbuilder asked for, rather than SIESTA's own treatment
  (periodic on all three axes, always).

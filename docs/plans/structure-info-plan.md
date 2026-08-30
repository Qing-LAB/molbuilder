# Structure info — free metadata that travels with the pair

**Role:** plan (the board below is the live status; rows flip to Done as
they land)
**Domain:** model · web · execution
**Started:** 2026-08-29 (user rulings, same day: the § 1 store, the
Metadata pane, and the recorded electronic contract as its first
consumer)
**Companions:** [`web/molview.md`](?doc=web/molview.md) (the pane + the
doors) · [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md)
(persistence) · [`plans/transport-design.md`](?doc=plans/transport-design.md)
§ 4.1b (the consumer that motivated it)

## 1. What is being built, in one paragraph

A structure gains a **free-form, non-structural metadata store** —
`info`, a JSON dict of key → value — supplied by host tabs through ONE
MolView API (`viewer.data.info.set / remove / get`), displayed in a new
**Metadata** pane beside Selection and Cell, and persisted in the
`.xyz + .molstruct.json` pair so it travels wherever the structure
goes.  Its first consumer closes a real gap: when the Results tab
exports a structure from a finished run, the engine's own parser
records the **electronic contract** (basis, XC, mesh, k, temperature)
into `info.calculation` — and a transport citation of that pair then
runs with the relaxation's contract **sealed**, exactly as if the deck
itself had been cited (`transport-design.md` § 4.1b gains the third
shade: pair-with-recorded-contract).

Design stances (user, 2026-08-29):

* **The contract block's keys are one agreed vocabulary** — exactly
  `TransportConfig`'s `CONTRACT_FIELDS` names (`transport/stages.py`),
  never a spelling invented at a call site.  The extractor is pinned to
  the set by test (`tests/parse/test_contract.py`), the sealed fill
  reads only names in it, and any new field joins by joining the
  constant — so the recorder, the pane, the pair and the seal cannot
  drift apart on a name.

* `info` is **not structural**: it never enters `structure_hash`, the
  frozen/region machinery, or any deck emission — it *describes* the
  structure.  That is also why the read-only gate does not apply to it
  (`molview.md` § 9.4's one question — "does this change the structure
  the calculation ran on?" — answers no), which is what lets the
  read-only Results viewer attach the contract before export.
* The pane **displays**; the API **mutates**.  A person reads the
  metadata in the viewer; which keys exist is the host tabs' business.
* A recorded contract **seals** the transport fields (fdf-is-truth
  transfers to the recorded copy); a hand-made pair with no
  `info.calculation` stays the open lane.  Consistency with the
  relaxation is the point of the contract, and the refusal already
  names the way out (re-export, or cite something else).

## 2. The board

| # | leg | steps | proof | status |
|---|---|---|---|---|
| **I1** | **The contract text** | `molview.md`: the Metadata pane, the `data.info` doors, the § 9.4 reasoning, install/export ride.  `model/structure.md` + `structure-molstruct.md`: the `info` field, sidecar schema 9 (additive; 8 stays readable), hash invariance.  `transport-design.md` § 4.1b: the recorded-contract shade. | doc guards green (`test_docs_structure`, `test_doc_claims`) | **Done 2026-08-29** — doc guards green |
| **I2** | **Server core** | `Structure.info: dict` (JSON-refusing setter path at the dict doors), `to_dict`/`from_dict` top-level `info`, `structure_hash` untouched by it; sidecar `SCHEMA_VERSION` 9 — write `info` when non-empty, read 8 and 9, stray-key rule updated. | round-trip tests: pair→load→`info` intact; a v8 file loads with empty `info`; hash equal with/without `info` | **Done 2026-08-29** — `TestInfoBlock` pins the five invariants (ride-whole, empty-writes-no-key, hash invariance, v8 full-replace, JSON refusal at write) |
| **I3** | **The wire** | `/api/build/load` answers `info` (path + `{structure}` envelope branches); `struct_from_body` carries it; the save door writes it through the codec. | endpoint round-trip test | **Done 2026-08-29** — `to_wire` + the `{structure}` envelope gate carry it; the web battery green |
| **I4** | **MolView** | model `info` store + announce; `viewer.data.info.set/remove/get` (ungated, no unsaved badge, rides the editable draft); `structureFromServer`/`exportFile` carry it; `ui.js` third panel page `Metadata` (read-only key/value rendering, honest empty state). | js pins (doors exposed, third tab present); browser check: set a key → pane shows it → export → sidecar holds it | **Done 2026-08-29** — `test_molview_info_js` pins the ungated doors, the page, the two-way wire, the sheet; browser-proven (the demo: `info.set` → the Metadata tab renders the contract JSON) |
| **I5** | **The recorded contract (producer)** | one `contract_of(run_dir) -> dict \| None` interface with per-engine extractors (SIESTA `parse_fdf_params`; PySCF its deck parser); the Results-tab structure load attaches `info.calculation` (engine, contract, source deck + sha) at the same server-side compose that already attaches periodicity. | load a finished run in Results → pane shows the contract → export → pair carries it | **Done 2026-08-29** — `parse/contract.py::contract_of` (SIESTA real; a PySCF deck answers None until its extractor lands), `/api/results/contract`, the structure inspector records through the `info` door; the Export menu's indicator + the tab's presence dot shipped with it.  Residue: the TRAJECTORY inspector's export does not record yet (its load rides /api/watch, not the structure door) — § 3 |
| **I6** | **The sealed pair (consumer)** | `transport/compose.py`: a form-B citation whose sidecar carries `info.calculation` fills the config from it and **seals** `CONTRACT_FIELDS` (same refusal as form A, wording names the recorded deck); `describe_attempt` answers `contract: "cited"` with "contract recorded from the <engine> deck" in the summary; the tab's lane logic already follows `contract`. | compose + describe tests both lanes; **mutation:** drop the seal → the stays-sealed test fails; browser: cite an exported pair → contract fields hidden, meta line says recorded | **Done 2026-08-29** — `recorded_contract_of` (one reader for compose + both web doors), the fill forces kz=1, the seal's refusal names the record, the travel copy keeps it (`TestTheRecordedContract`, seal mutation-killed); live: the recorded pair answers 'contract RECORDED from the siesta deck', and `prep run seed` renders SZ/150 Ry from the record, not the defaults |
| **I7** | **Close-out** | full battery; browser walk of the whole chain (Results export → cite → describe → prep); this board flipped to Done; README/toc rows already indexed. | battery green (the 9 § 3 pre-existing failures excepted); walk screenshots | open |

Order is the dependency order; nothing in I5–I6 starts before I2–I4
hold, because the store must exist before anything records into it.

## 3. The parked backlog (so one file answers "what is open")

* ~~Nine pre-existing test failures on main~~ — **diagnosed and fixed
  2026-08-29**, each read whole and classified:
  `test_admin_reload` ×2 = test staleness (the probes invoked the
  retired `serve --port` spelling; `serve` became a group — all four
  probes retargeted at `serve foreground`);
  `test_catalogue_agreement` = REAL DRIFT, the catalogue was the wrong
  side (it still taught the retired `./` psml_lib spelling the code
  refuses — aligned to the code's truth, said here as the test
  demands);
  `test_docs_tab` toc dedupe = passes on current toc (the duplicate it
  saw is gone);
  `test_http_status_contract` = stale doc header (route count 85 → 88,
  the contract endpoint's row added);
  `test_launch_ask_mode` = test-logic bug (a first-occurrence slice
  between two anchors inverted to empty when the transport chain added
  an earlier `if mode == "direct"`; the code itself was correct — the
  end anchor is now searched after the start);
  `test_negative_body_assert_lint` = a missing status guard in a test
  body (added);
  `test_molview_mount` ×2 (plus two more the same cause) = stale pins
  from two legitimate features (the `buffer` label joined the
  predefined set with the transport composite; the Metadata page made
  the panel three pages).
* **CSS polish batch** from the 2026-08-29 fresh-eyes review (wrong
  `var()` fallbacks documenting stale values; raw sizes in
  `molview.css` against its own tokens; the inspector vocabulary
  living in `results/style.css`; `modify.html`'s three hand-written
  `molviewer-*` names; `.hint` restyled by three page sheets;
  `rec-diff`/`btn`/`secondary` template classes with no rules).  Some
  entries sit on `web/ui-contract.md`'s recorded keep-list
  (2026-08-25) — touching those reopens decisions, deliberately not
  done in the review round.
* **Two caller-less endpoints** (`/api/checkpoint/config`,
  `/api/docs/list`) and `/api/selection/atoms` (test-pinned as a
  vocabulary contract — confirm the role before deleting).
* **Flat-shape citation ergonomics**: a flat calculation folder holds
  several stage decks and one shared `.XV`, so citing it refuses as
  ambiguous (never guess which deck is the contract).  Fine per
  § 4.1b; a future refinement could disambiguate by the concluded
  marker's stage.
* **Transport viewer default orientation**: every junction's transport
  axis is z, and the default camera looks down z — a 1-D chain reads
  as a single ball until rotated.  A host camera-orientation door is a
  `molview.md` contract question, not a patch.
* **The trajectory inspector does not record the contract yet** — its
  structure arrives through `/api/watch/load`, not the structure door,
  so an export from the trajectory view carries no `info.calculation`;
  the recording call it needs is the same one the structure inspector
  makes (I5).
* **The Results-tab transmission inspector** (reads the shipped
  `<label>.transport.json`) and `TransiestaEngine.parse_output`
  (raises by design, pointing at the record) — roadmap § 2 items 3–4.

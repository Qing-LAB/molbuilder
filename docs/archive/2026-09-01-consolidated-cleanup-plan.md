# Cleanup — the drift left by the Junction removal and the config unification


> **ARCHIVED 2026-09-01.**  Its open items moved to the one plan,
> [`plans/plan.md`](?doc=plans/plan.md); what stays here is the record of
> what was decided and built.  Nine plan documents were consolidated that
> day *(user: "We don't need ten plan files scattered")*, and a fact-check
> against the code found three of the nine headers stating the opposite of
> what had shipped.

**Role:** plan
**Domain:** web · execution · model
**Started:** 2026-08-31
**Companions:** [`web/molview.md`](?doc=web/molview.md) §§ 9.5, 11.6 — the
selection/measurement split this plan finishes applying ·
[`web/ui-contract.md`](?doc=web/ui-contract.md) §§ 1, 4, 5, 8 — the stylesheet
layers and the one-owner rule ·
[`configuration.md`](?doc=configuration.md) §§ 2, 2.1, 2.1a — the owner of the
lookup rule item 6 repairs ·
[`plans/modify-redesign-plan.md`](?doc=plans/modify-redesign-plan.md) § 3.4 —
the removal whose sweep this finishes

Two large changes landed and neither swept all the way: the **Junction panel
removal** and the **configuration unification**. What is left is not a set of
open questions — every item below is settled by a rule that already exists in a
contract. This plan names the rule, the sites, and the order.

**The ordering rule for every item: fix the RULE in the document that owns the
concept, then sweep the restatements.** Fixing a restatement first creates fresh
drift, which is how three of these items were made in the first place.

---

## 1. Ordered gestures read the ordered track *(user, 2026-08-31)*

**The rule already exists.** `molview.md` § 11.6: *"It is the ordered-pick track,
and the ruler is its first reader — not its owner. Order and a small count limit
are what it promises; measuring is one use of that promise."* The Cell page's
axis gesture is named there as the second reader. Every other ordered gesture was
left on `selection`, which is a **set** — `selection.add()` sorts — so "the first
atom you picked" is a fiction.

**What it costs today.** `modify/viewer.js` reads `sel[0]`, `sel[1]` for Orient
and `sel[0]` for the add-atom anchor. Orienting two atoms picked by clicking and
the *same two* picked by shift-range give **opposite directions**, silently. With
`tilt ≠ 0` that is a wrong structure; at `tilt = 0` it is a 180° flip.

**Done 2026-08-31, as a column rather than a per-op decision.** § 11.1's table
gained `ordered`; `orient` sets it, one line in the resolver picks the track,
and `model.js` hands `readPicks` beside `readSelection`.

**`add_atom` was tried and reverted, and the reason is worth keeping.** It has
arity **1** — a set of one has no ambiguous first, so there was no order to get
wrong — and it shares the Atom tab with Delete, a *set* gesture. Locking the
ruler on there would have taken Delete's clicks away to serve an op with no
defect. Unifying it would have cost a working gesture and bought nothing; the
existing e2e test that pins Delete's gesture is what caught it.

Set operations (Delete, the slab's `center_indices`, which is a centroid) stay
on `selection`; order is meaningless to them, and an ordered track would cap
them at three atoms.

The Transform tab locks the ruler on when reached, with the message § 11.6
already requires, the same as the Cell page.

## 2. The ordered track is DRAWN as ordered *(user, 2026-08-31)*

**Why this belongs with item 1.** The Orient bug was invisible *because* an
ordered pick and an unordered one look identical — both a glow. A drawing that
shows direction makes the defect unmissable and the gesture self-explaining.

**Do:** replace the ruler's second glow with marks that carry the order.

| picks | drawn |
|---|---|
| 1 | a marker on that atom, distinct from the selection's glow |
| 2 | one arrow, first → second |
| 3 | two arrows, first → second and second → third |

The measurement readout (§ 11.6's coordinates, distance, signed `Δ`, angle) is
unchanged; this is what is drawn *on the molecule*.

**Two facts that make this small.** The pipeline already emits `measured` in
**pick order** (`render-engine.js` maps `track.picks` through the seat map, in
order), so no plumbing changes — only what the sealed layer draws. And arrows
are already a primitive there, used by the force vectors and the axis triads.

**The trap that is already recorded, and must be honoured:** a 3Dmol GLShape
carries a **single colour**, so batching arrows paints them all the colour of
whichever was first. Each arrow gets its own shape. `3dmol-embed.js` states this
and both of its existing arrow users were broken by it once.

Per § 10.3 the split is unchanged: the pipeline emits *which atoms in what
order*; the sealed layer owns what a mark looks like.

## 3. Any change to the molecule clears the track *(user, 2026-08-31)*

**The rule, with no "except" in it.** The picks are MolView's internal state,
tied to the molecule in the window. `model.js` clears at the edit door and the
load door. Two places did not:

**Done 2026-08-31, and not where it was first put.** The clear went to a third
call site before the design was right. It is now ONE rule in `settle` — the
function that already declares itself *"every change to the structure … settled
HERE, once"* — with a single exemption for the four doors that **prove** the
atoms are unchanged (`requireSameAtoms` runs before they land): a running job's
frames arriving. Those must not clear, or § 12.4's *measuring while a
trajectory plays* would delete the measurement it exists to show.

- **`commitPeriodicityOp`** — a cell edit moves no atom but is still an edit; it
  calls `history.edited()`, and it was the door that had already forgotten.
- **`ui-context.js`** — the restore adopted saved picks back *after* the install
  had cleared them, guarded only by an **atom count**, so two different 3-atom
  molecules matched and the readout quoted a bond length for atoms nobody
  picked. **Done 2026-08-31:** the picks are no longer saved or restored. The
  ruler's on/off is a way of *looking* and still persists.

**Sweep:** § 11.6 says the track "persists in the `<owner>:ui` lane"; that is now
the ruler's on/off only. Fix that sentence.

## 3a. The measurement API is the module's, and nothing escapes it *(user, 2026-08-31)*

*"All the measurement API has to be within MolView, and all the users have to
call through it. Nothing escapes."*

**Two boundaries.** The import boundary already held: `index.js` exports `mount`
and `formula`, and all seven outside imports use it. The **runtime** boundary did
not — `mount` returns `data: model`, and `measurement:` was **the store**, so
every internal write door was public. A consumer could call `toggle()` and add a
pick without going through `pickAtom`, which is the one place that decides
measuring-vs-selecting and the one place a drawn index has been translated.

**Done 2026-08-31.** The model hands out a surface, not the store:
`getState · positions · subscribe · setActive · requestPicking · clear`.

| escaped | where it lives now |
|---|---|
| `modify/ruler-lock.js` — a page-side file holding "is the ruler on, turn it on" | `measurement.requestPicking()`, inside. It answers *whether it turned the ruler on*; the page supplies only the wording, because MolView reads no global (§ 4) |
| `periodicity.js` walked the current frame for the picked atoms' coordinates — the same walk, staleness guard included, that the module's readout does | `measurement.positions()` |
| `toggle` / `adopt` public on the handed-out object | off the surface; `adopt` deleted outright — it restored picks guarded by an atom count, and the picks no longer persist |
| two spellings of one read (`get()` vs `getState().picks`) | one |

**The complete consumer set outside the module is two files** — `modify/viewer.js`
and `modify/periodicity.js`. Every other `.js` under `web/static` matching
"measurement"/"picks" is the English word in unrelated prose. Neither holds
measurement state.

**Pinned, both guards mutation-tested:** one test fails if any outside file names
a member that is not on the surface, another if any outside file imports a MolView
internal. The first also asserts its corpus still contains the two known
consumers, so it cannot pass by scanning nothing.

**Still handed out raw: `selection` and `view`.** Outside consumers use three
doors of `selection`'s eighteen (`get`, `subscribe`, `setSwitch`) and one of
`view`'s four (`set`). The other fifteen — `toggle` among them — are reachable
and unused. The right fix is the facade at `handle.data` in `mount.js` rather
than inside the model, so "inside uses internals, outside uses the API" is
structural; that is its own pass.

## 4. An operation says whether it happened, and blocks what conflicts

**The rule.** Every op on the Modify panel reports its outcome and owns an
in-flight lock — `viewer.js` does this for delete, translate, rotate, orient and
add-atom. **`slab-panel.js` does neither**: it awaits `applyOp("slab", …)` and
discards the return, which is `null` both when another edit is in flight and
when a precondition is refused. A success and a silent no-op are indistinguishable,
and a second click during a slow build does nothing without saying so.

**Do:** route the slab through the same wrapper the other ops use — disable the
button while in flight, write the outcome to the status line, and surface a
refusal as a refusal.

## 5. Two Cell-tab defects that make the panel state something untrue

- **The z fallback is applied to whichever axis is chosen.** When the chosen row
  has no direction yet the code substitutes `z`. That is right for `c` and wrong
  for `a` and `b`: the readout then reports a span measured along z while
  labelled as `a`'s, and Use writes a row that collapses the cell to no volume.
  **Do:** apply the fallback only when the chosen axis is `c`; for `a` and `b`
  refuse with the reason, as the length control beside it already does.
- **The pick buttons stay live when the ruler is off**, with a title reading
  "turn measuring on, then pick two atoms". Pressing one stages a row from picks
  that are no longer marked on the molecule. **Do:** one readiness rule for the
  buttons and their titles.

## 6. `molbuilder.json`: the documents still teach the deleted lookup

**The owner is `configuration.md` §§ 2 / 2.1 / 2.1a**, and it is the document
that is wrong: it says a `./molbuilder.json` "still wins today". It does not —
`machine_config_path()` has one branch, and `machine_config_shadow()` exists to
say the working-directory file is **not read**.

**Fix the owner first, then the restatements:** `running-a-job.md` § 5.1 + § 5.5,
`access-control.md` § 3.4, `deployment.md` § 5 + § 6, the shipped
`docs/ops/examples/molbuilder.json.example` header, and `cli.py`'s
`auth-setup --output` help text — which is user-facing and describes the
first-found-wins search by name.

**Then the citations.** Sixteen test files cite § 2.1a as authority for the
*opposite* of what it says. They become correct the moment the owner is fixed;
they are the reason to fix the owner rather than each site.

**Also `architecture.md` § 8.2** documents `paths.logs` / `paths.run` /
`paths.reports` as live and names three functions that do not exist. A reader
who follows that row writes a config file that **refuses to load**.
`run-reports.md` § 6 repeats it.

## 7. Citations that resolve to nothing

About twenty citations **in the source** name sections that do not exist. Their
targets, written here without the section marks so this plan does not trip the
guard below: `structure-periodicity.md` sections 3a/3b/3c (the content is 4, 6,
6.2 and 7); `siesta.md` section 13 (it is 7); `job-system.md` section 10;
`generator.md` section 12.1 (it is `template.md`'s); `submission.md` sections 5
and 8. **Two are user-facing error strings**, so a person hits an error, follows
the pointer, and finds nothing: `modify.py`'s FCC-only refusal points at an
"Off-scope" section of `tabs.md` that exists only in the archive, and
`siesta/input.py`'s ELPA refusal points at `siesta.md` section 13.

**Do:** repoint each to the section that carries the content.

**And extend the guard that already exists.**
`test_docs_structure.py::test_every_cross_document_section_citation_resolves`
already resolves every **doc → doc** citation and passes today — it is why this
plan had to write the list above in prose. What it does not cover is
**code → doc**, which is where all twenty live. Widening its corpus to the
source tree is the whole fix, and it would have caught every one of them.

**And write down the two rules that no live document states**: the
`--electrode ELEM:PLANE:MxNxL@contact=DIST:±z=I,J` grammar, and the FCC-only
restriction the error message points at.

## 8. Residue from the Junction removal

`transport.md` and `structure-periodicity.md` §§ 5, 6 still teach
`add_symmetric_electrodes` and flanking slabs at `z = ±gap/2`; `tabs.md` § 2
describes a Junction op-tab and an electrode dropdown that no longer exist and
never mentions the Slab tab; `web-api.md` § 4 lists `symmetric_electrodes` and a
`selection-panel` partial that are gone. Code comments in `modify.py`,
`web/blueprints/modify.py`, `slab-panel.js` (whose header asserts three removed
things and holds the last `__elc` reference in the tree) say the same.

**And the dead route:** `/api/modify/electrode` had no browser caller — the
Junction panel was its only client, and the CLI reaches `add_electrode_slab` as
a Python function. **Done 2026-09-01** *(user: "delete it, clean up all obsolete
residue so we don't have overlap or bugs related duplication or redundancy")* —
the route, `_parse_electrode_common`, `OPERATIONS.electrode`, the now-unused
`emptySelection: "origin"` value, and its tests. The record is
`modify-redesign-plan.md` § 3.4a, which also names the one overlap this did
**not** close: `modify.add_electrode_slab` stays, because the CLI's
`--electrode` flag is its live caller and § 3.4's promised re-expression over
`add_slab` has not been done.

## 9. Tests with no target

The rule *(user, 2026-08-31)*: **tests pin what should exist.** Pinning an
absence is legitimate only where the forbidden thing is a design invariant new
code could violate — not something that was deleted.

Twelve were removed on 2026-08-31. The rest, all verified: a loop whose body can
never execute (`test_doc_claims.py`'s retired-type pin); an assertion that cannot
fail (`callable()` on a `def`); a `STILL_OPEN = {}` iterated for unplanned
entries; and a dozen greps for identifiers deleted with the MolView migration.
Two files are **empty tombstones** — a docstring saying "retired" and no tests.

**Also:** `test_task_setup_tab.py` writes into the real `projects/` tree, and the
guard meant to stop that does not match the path syntax those tests use.

## 10. CSS the contract already forbids

§ 5's line is *read the properties, not the selector* — position is composition
and stays; appearance is a second owner and goes.

- **Done 2026-08-31:** two rules in `modify/style.css` restated page-shell's
  number-input appearance and had already drifted from it (`--border-soft` for
  the shell's `--border-strong`, a different focus ring, no focus background),
  so the Transform boxes were the only number inputs in the app that looked
  unlike the rest. Now composition only.
- **Open:** a 19-line block is byte-identical in `structure-optimization` and
  `transport`'s sheets but for one selector — both are MolView mounts, so it
  belongs on `.molviewer-card`. `html, body { height: 100% }` sits in three page
  sheets. `modify/viewer.js` writes the same tab-switching loop twice, and the
  op-tab panels are hidden by `display:none` alone, so they stay exposed to
  assistive tech.

## 11. `molview.md` still declares what the code deleted

§ 6.2's shape diagram lists `+int[] pickOrder` and § 9.5 says the snapshot
"still carries" it — it does not; `stores.js` deleted it and a test now pins its
absence. Three code comments restate the old rule, and one of them
(`modify/viewer.js`'s "IN THE ORDER THEY WERE PICKED") is what makes item 1's
bug look correct.

Fix § 6.2 and § 9.5 first, then the comments.

---

## 12. Run reports: named channels, and a tab for the secrets *(user, 2026-08-31)*

> *"why not make the setup of key/secret/webhook a separate tab, and here user
> just checkboxes to select where the notification should go?"* … *"a new tab
> for setup secrets etc (not viewing them but just overwrite or add/merge),
> while the task can just check and use those channels (without secret listed,
> just name/availability)"*

Three defects, one shape. The destination was **one per machine**, so pointing
it at Slack silently replaced the listener already in use. Its form lived
**inside a per-calculation card**, where a machine-wide setting had to be
explained away in a hint and the two-files rule was held by a template
comment. And the other half — issuing a listener key — had **no UI at all**,
so setting this up end to end always dropped to a shell, which is what a
person hit and reported as *"it seems the code is missing"*.

A fourth found on the way: `GET /api/notify/destination` returned every stored
URL in full on the strength of *"an address, not a secret"* — true of the
listener URL it was written for, **false of the Slack webhook actually in the
file**, and readable by anyone signed in.

The contract is `run-reports.md` §§ 1, 3, 3.0, 3.1 and the new
`this-machine.md`.

## Order of work

1 → 2 → 3 (the ordered-track group: one contract, one visualisation, one rule).
Then 4 and 5, which are user-visible and small. Then 6 → 7 → 8, the document
sweep, owner-first. Then 9, 10, 11.

## Status

| # | item | state |
|---|---|---|
| 1 | ordered gestures read the ordered track | **done** — `model-jobs.js` grew an `ordered` column; **`orient` alone** declares it (`add_atom` was tried and reverted: arity 1 has no ambiguous first, and it shares the Atom tab with Delete, a set gesture). `viewer.js`'s Orient readout and enablement follow the picks, the Transform tab asks `measurement.requestPicking()`, and the comment that made the bug look correct is gone |
| 2 | the track is drawn as ordered (arrows) | **done** — 1 pick a mark, 2 one arrow, 3 two; its own shapes bucket so a long measurement cannot steal the gold from the largest force arrow; a pick that is off-screen falls back to marks rather than asserting a step the user never made. Two tests (drawing layer + whole chain), both mutation-tested against "marks only" and "arrow reversed" |
| 3a | the measurement API is concealed; nothing escapes | **done** — surface not store; `requestPicking` and `positions` moved inside; two static guards, both mutation-tested. `selection`/`view` facade still open |
| 3 | any change clears the track | **done** — one rule in `settle`, one exemption for the doors that prove same-atoms; § 11.6 states it; both halves mutation-tested |
| 4 | an op says whether it happened | **done** — the slab goes through `viewer.js`'s `runOp`, the page's one op wrapper (in-flight lock, status line, both refreshes) |
| 5 | Cell tab: z fallback, pick-button readiness | **done** — z is now `c`'s fallback only; `a`/`b` with no direction answer `—` instead of reporting a z-measured span under `a`'s label and writing a zero-volume row. The pick buttons fixed themselves once the ruler-toggle clears the picks. Verified in the browser |
| 6 | the config lookup documents | **done** — owner first (`configuration.md` §§ 2 / 2.1 / 2.1a, which had named itself as the section to edit when the step landed), then all five restatements: `running-a-job.md` §§ 5.1 + 5.5, `access-control.md` § 3.4, `deployment.md` § 5, the shipped `molbuilder.json.example` header, and `cli.py`'s user-facing `auth-setup --output` help. Plus `architecture.md` § 8.2 and `run-reports.md` § 6, which documented three keys the reader refuses |
| 7 | citations that resolve to nothing | open |
| 8 | Junction residue + the dead electrode route | **done** — route, helper, OPERATIONS row, the orphaned `"origin"` value and three tests deleted; swept `molview.md` (which still carried BOTH electrode ops in its table and one in its example), `web-api.md` (the op list, the route count, and a `/partials/selection-panel` whose route no longer exists), `tabs.md` (a Junction op-tab that is gone, and no Slab tab where one has shipped for weeks), `transport.md`, `structure-periodicity.md`, and four orphaned continuation lines the *previous* removal left inside `modify.py`'s own route list. Recorded in `modify-redesign-plan.md` § 3.4a. **Open, and named there:** the CLI's `--electrode` still calls `add_electrode_slab`, so two slab builders coexist until that flag is re-expressed over `add_slab` — a behaviour change (relative vs absolute placement), not residue |
| 9 | tests with no target | partly done (12 removed). The two files called *empty tombstones* on 2026-08-31 were read in full on 2026-09-01 and are **not** empty: each is a signpost recording where the retired coverage now lives, which is a service to whoever greps the old name. Left alone; the characterisation was wrong, not the files |
| 10 | CSS second owners + duplication | **done** — the MolView host's `overflow-x` rule was 18 identical lines in two tab sheets under two different ids, which is why a per-selector check never saw it. One owner now (`molview.css`, `.molviewer-host`), pages opt in with the class, and `test_no_long_block_is_copied_between_stylesheets` fails on ten identical non-blank lines shared by two sheets — mutation-tested by pasting the block back. `ui-contract.md` § 4 records the shape |
| 11 | `molview.md`'s `pickOrder` | **done** — off § 6.2's diagram and § 9.5; the three code comments that described it as live are swept |
| 12 | named channels + the This-machine tab | **done**, then reviewed in full text, which found eight more: the submit-machine recipe was built from a form the page had **hidden**, so it could only emit its own placeholders (the form stays and the button changes meaning); it concatenated JSON rather than serialising it; a save left the retired top-level `url`/`key` in the file, a credential nothing reads and nobody would rotate; `NotifyPolicy.__bool__` disagreed with the twin it mirrors about `channels=()`; the `[NOTIFY]` log line was written twice, so stdout got a double prefix; **`Resources` stopped round-tripping through JSON** the moment it gained its first sequence field (a tuple out, a list back, and only equality lies — so the symptom would surface far from the cause), fixed where `time` and `mem` are already canonicalised; the decision ledger printed `notify_channels=('slack', 'lab')` and a bare `()` into the one file whose value is being readable; and my own deletion of `_NOTIFY_USER_RE` left an orphaned comment fragment in `cli.py`. All eight pinned and mutation-tested. Originally: `notify` became `{channels: {name: …}}`, so a run reaches a Slack *and* a listener; `task.json` carries `notify.channels` as **names only**, with absent / `[]` two states rather than two spellings (the one field written when falsy, in `task.py`, `prep`, the wrapper flag and the JS alike — mutation-tested at all four, and the `task.py` gap was found *by* the mutation). New tab holds the channels and the listener, masks **every** address (masking only webhooks is a rule mislabelling defeats), and shows an issued key exactly once. Task setup keeps ticks + names and can no longer reach a second file. `issue_notify_key` is one door for the CLI and the tab. Swept the two settings retired 2026-08-31 out of `configuration.md`, `architecture.md` § 8.2 + § 8.3, `deployment.md` § 5 and `notify.py`'s own header, which all still taught them as live |

*Verdict: closed in 2.77.6.* The discipline review of F12 fix 2b, read before its release: every finding was fixed in 2.77.6, except the generic range sentence for one lead k-point (finding 13), which stays general. Kept as the record of what the review found.

# Discipline review — F12 fix 2b ("nothing stops a calculation the person set")

Reader: discipline lens (plain language, one home per fact, dead code, tests, docs).
Read: the whole diff; the touched modules whole where they carry the changed doors
(`transport/wizard.py`, `transport/compose.py`, `transport/citation_defaults.py`,
`validation/transport.py`, `validation/sidecar.py`, `validation/geometry.py`,
`validation/stages.py`, `validation/spectra.py` § solvent, `kmesh.check`,
`pseudos.check_coverage`, `cell.bulk_z_period`/`uneven_layers`, `transport/sort.py`,
`web/blueprints/transport.py` describe-attempt and schema routes, `transport/record.py`
low-bias I-V, `validation/task.py` preflight, `template.py` limits); every touched test
file; `docs/process/testing.md` § 3a/3b; every changed doc hunk and the chapters around
it; the plan, archive README and the inventory. pyflakes over all touched files: clean.
Nothing in the checkout was written; measurements were read-only Python.

**21 findings.** Ordered by harm.

---

## 1. The relaxation warning about unfrozen leads now tells the person something false — and in code names

`molbuilder/validation/sidecar.py:100-109` (`check_electrode_labels_are_frozen`):

> "Add them to \"frozen_atoms\" in /modify before running this, or the relaxation will
> move the leads and the junction cannot be composed afterwards (the composer refuses
> it, by which point this run has been paid for)."

After this release the composer does not refuse it: `wizard._lead_findings` says it and
the junction composes. Its docstring says the same thing three times (`:66` "There is
already a gate for that", `:77` "the compose-time refusal is what makes a bad junction
impossible", `:84` "The refusal belongs at compose … and it is already there"), and so
does `tests/test_electrode_labels_are_frozen.py:9-13` and `:68-73` ("The REFUSAL belongs
at compose, where transport IS the intent, and it is there").

Scenario: a person labels the leads on the Molbuilder tab, forgets to freeze them, and
starts a relaxation. They read that the junction "cannot be composed afterwards", cancel
or re-run a paid relaxation — when the truth is that it composes, with a warning, on
leads that moved. The message also speaks in code names (`"frozen_atoms"`, `/modify`,
"the composer").

How sure: traced (the warning's text; `wizard.py:279-287` returns the finding instead of
raising; `compose.py:769-779` records it).

Fix: say what now happens — the leads move with the bridge, and a transport calculation
citing this relaxation will say so and run on those leads — and the way on in page words
(freeze the lead atoms on the Molbuilder tab before relaxing). Correct the docstring and
the test file's two docstrings to "said at compose, as a warning". The test that pins
`"frozen_atoms" in msg` (`test_electrode_labels_are_frozen.py:41`) pins the field name;
pin "freeze" instead.

## 2. A Fermi cutoff below 2.5 kT is said twice, from two homes of the one 2.5

`molbuilder/data/catalogue.template.toml:2874` `range = [2.5, 20.0]` and
`molbuilder/transport/transiesta.py:98-100` `MIN_NEQ_FERMI_CUTOFF_KT = 2.5`, asked by
`molbuilder/validation/transport.py:123-139`.

Measured (`validate_config_metadata` on `negf_neq_fermi_cutoff_kt = 2.0`):

> `warn config.negf_neq_fermi_cutoff_kt Bias window's reach past each Fermi level (in kT) (TS.Contours.nEq.Fermi.Cutoff) = 2.0 is outside the recommended range [2.5, 20.0]`

and the transport validator adds, on the device, the `info` "the bias window reaches 2
kT past each Fermi level; TranSIESTA stops the device stage under bias when this is
below 2.5 kT." Before the release the `at_least` refusal made the range warning stand
aside ("a value refused draws that refusal alone"); removing it left both.

Scenario: a person sets 2.0 on the device tab and sees a yellow warning that names a deck
keyword and says only "outside the recommended range", plus a grey note saying the
engine will stop — the louder message is the less useful one, and the floor lives in two
places that can drift. `science/validation.md` § 7's "One fact, one finding. … Two
findings for one fact is the failure this rule closes" is broken, and the docs that say
"info" for this case (`science/overview.md` transport table, `engines/transport.md`
§ 3.8 device table) do not describe what the person sees.

The precedent for the fix is in the same release: `validation/task.py:175-180` passes the
`relax_steps = 0` warning as `refused=` so the range warning stands aside for it.

How sure: measured (range warning) + traced (the info).

Fix: one home for 2.5 (the constant read from the catalogue row, or the row's bound from
the constant), and the range check standing aside where the engine's stop is said, as it
does beside the step warning.

## 3. The window sentence says "the current is cut short" where the record leaves the current blank

`molbuilder/transport/energies.py:73-78` — the one sentence said by the transmission
gate, the description's preflight and the record:

> "… but the energies computed reach only [lo, hi] eV, so the current is cut short.
> Widen the transmission's energy range."

The record (`transport/record.py:308-313`, `low_bias_iv`) does not cut the current short:
it leaves that voltage's current `None` and attaches this note. Measured:
`current_a = [0.0, 1.55e-05, None]`, `notes = {'2': "at 1 V … so the current is cut
short.  Widen …"}`. The description's preflight under the low-bias approximation speaks
of that same record. Only a sweep point's own TBtrans current is truly cut short.

Scenario: on the Results tab a voltage shows no current, and the note beside it says the
current was cut short — the person looks for a number that is not there.
`tests/test_transport_record.py:49` now pins exactly that pair
(`current_a[4] is None and "cut short" in notes["4"]`).

How sure: measured.

Fix: word the consequence so it holds on all three roads (TBtrans's current at that
voltage is cut short; the record leaves it blank), or drop the consequence clause and
keep the reach and the way on. "Widen the transmission's energy range": the form calls
it "Transmission window, lower edge / upper edge" — use the form's words.

## 4. The `at_least` limit is dead code, still documented as a mechanism

`molbuilder/template.py:483` (`Item.at_least`), `:932` (wire key list), `:989-990`
(served to the form), `:1800-1803` (parser branch), `:2073` (parse), `:2463-2467`
(`why_not` branch). No catalogue row declares `at_least` any more and no test uses one
(grep: none in `molbuilder/data`, none in `tests`). `docs/engines/template.md:680`
keeps a field-table row for it ("*No item declares it now*" — a history word in a
contract), and § 5.3 (`:859-865`) keeps a paragraph for it that restates transport's
2.5 kT fact (owned by `engines/transport.md` § 3.8 and `transiesta.MIN_NEQ_FERMI_CUTOFF_KT`).

Scenario: a later author reaches for `at_least` to hold an engine floor — exactly what the
user ruled against (2026-11-02) — and the parser, the wire and the door are all waiting;
meanwhile three copies of the 2.5 kT fact can drift.

How sure: traced.

Fix: remove `at_least` from `template.py` (field, parse, wire, door, comment) and its row
and paragraph from `template.md`; § 5.3 points to transport for the cutoff instead of
restating it.

## 5. Comments and docstrings still describe the refusals this release removed

All traced; each is text a reader now acts on wrongly.

- `transport/compose.py:302-308` (`_classify_pair`): "and the per-frame promise only the
  citation can check … every frame keeps the electrode atoms where frame 0 has them" —
  the check left `_classify_pair`; it is in `compose_junction` and is a warning.
- `transport/compose.py:763-766`: "the § 3 lead gates, which is the same step: a block
  that is not frozen bulk does not become a model" — it now does, with `said`.
- `transport/compose.py:602-604` (`_extract_and_gate_electrodes`): "A refusal names
  atoms in the user order" — the warnings do; the remaining refusals name no atoms.
- `validation/transport.py:149-152`: "each rule reads the lead's own spacing, and both
  are refusals"; `:189-190`: "the electrode rung that sees it alone refuses the same
  calculation". The function docstring (`:26-33`) lists its science without the two new
  infos (spin, cutoff).
- `web/blueprints/transport.py:261-266`: "a reason a junction cannot be BUILT -- labels
  missing, an electrode that moved, …" — a moved electrode no longer stops it;
  `:298`: "as prep does (`sort.junction_notes` …)" — the card and prep now ask
  `compose.junction_said`; `:948-950` (`_panel_template`): "Raises ValueError -- … a
  citation the template's door refuses (a cited run that carried a net charge, ES7)" —
  it no longer raises for that.
- `cell.py:791-793`: `UNIFORM_SPACING_TOL_ANG` is how far spacings may differ "before it
  is refused as a bulk lead"; `:808-809` "the smallest defect worth refusing" (the doc,
  `junction-cell.md`, was changed to "worth saying"; the code comment was not);
  `:823-825` `bulk_z_period`: "`d_interlayer` is the spacing every adjacent pair of
  layers is checked to share" — it is the first spacing, checked elsewhere.
- `validation/task.py:139-141`: "one of them refuses -- so it is asked before the range
  check" — the step check now warns; its result is still passed as `refused=` (`:178`).
- `transport/sort.py:166` and `:187-190`: `junction_notes` is "every surface alike -- the
  Transport card and prep" — `compose.junction_said` now claims that role in its own
  docstring (`compose.py:628-634`); two docstrings claim one door.
- `jobset/rung.py:165-168`: points at "`transport.sort.junction_said`", a name that does
  not exist — it is `transport.compose.junction_said` (the rename edited the module path
  wrong), and the field now carries the leads' findings too, not only the inverted pair.

Fix: correct each to what the code does now; keep `sort.junction_notes`' docstring to the
inverted pair and let `junction_said` be the one that names the surfaces.

## 6. `bulk_z_period` keeps a parameter it no longer reads

`molbuilder/cell.py:817-820`: `def bulk_z_period(layer_z, tol_ang=UNIFORM_SPACING_TOL_ANG)`
— the body no longer uses `tol_ang` (the check moved to `uneven_layers`), and no caller
passes it (grep: `wizard.py:253` and the tests call it with `layer_z` alone).

Scenario: a caller passes a tolerance expecting it to matter; nothing changes.
How sure: traced. Fix: remove the parameter.

## 7. The lead warnings state a rule instead of what happens to the calculation

`transport/wizard.py:280-287`, `:298-305`, `:326-331`, and `cell.py:864-870`:

> "A lead stands for the bulk metal on its side, so its atoms must stay where the crystal
> puts them while the rest relaxes -- freeze them …"
> "… so its atoms must not move: relax the junction again …"
> "the L-electrode lead: its atomic layers are not evenly spaced: 2.3500, … A apart (…),
> so repeating them does not give the bulk metal. …"

The user's rule for this release: every message says what happens to the calculation,
then what to do. These say what "must" be and what to do; none says what the run does if
the person goes on (the leads are computed from these atoms as they stand, so each
self-energy is built from a lead that is not the bulk metal). The uneven one does not
say which spacing the lead is built on (`bulk_z_period` takes the first). "Must" in a
warning that never stops reads as a refusal. Compare the charge warning, which does it
right: "So the junction is computed at +0 on a geometry relaxed at -1."

How sure: traced (message text). Fix: one clause each — "the leads are computed from
these atoms as they are, …" / "the lead is built on the first spacing, d = 2.350 Å" —
and "should"/"is meant to" for the rule.

## 8. The two new information notes give no way on

`validation/transport.py:111-113` (spin) and `:127-129` (cutoff):

> "spin treatment non-collinear: TranSIESTA handles at most two spin channels, so it
> stops at the start of the device stage -- after the first stage and both leads have run."
> "the bias window reaches 2 kT past each Fermi level; TranSIESTA stops the device stage
> under bias when this is below 2.5 kT."

The neighbouring engine-stop notes in the same function each say what runs
("At least 10 K runs; 300 K is the default"; "at 300 K the least energy is 1.57 eV. The
default is 10 eV"), and so does the stripe note (`kmesh.py:337-338`, "1 point along y
runs"). These two do not.

How sure: traced. Fix: "restricted or unrestricted runs on the device" / "2.5 kT or more
runs; 5 is the default".

## 9. The close-atoms warning names neither the atoms nor the way on

`validation/geometry.py:46-52`: "closest atom pair is 0.100 Å apart -- this is too short
for any real bond: two atoms overlap". It replaced the error below 0.3 Å but was not
reworded; it reads the same at 0.1 Å as at 0.69 Å, names no atom (the codebase's
`Structure.atoms_named` exists for this), and says nothing of what the run does or what
to do.

Scenario: a 400-atom junction with one duplicated atom: the person is told a distance and
must hunt for the pair.
How sure: traced. Fix: name the two atoms (`atoms_named`), and one clause on what to do
(move or delete one of them on the Molbuilder tab).

## 10. Five pseudopotential messages were not brought to plain words

`molbuilder/pseudos.py` — the brief names every `check_coverage` warning; three were not
touched and two carry leftovers:

- `:853-859` `relativistic_mismatch`: "Au.psml is scalar; calc expects spin-orbit. For
  most non-spin-orbit work, scalar (SR) is correct. Fully-relativistic (FR) is …" —
  "calc", "SR"/"FR", and no word on what happens to this calculation.
- `:865-867` `parse_warning`: "S.psml present but header had issues: …" — "header", no
  consequence, no way on.
- `:907-915` `generator_mismatch`: "… confirm the odd-one-out (S) is intended -- a stray
  version is how a defective pseudo usually sneaks in." — "pseudo", no consequence.
- `:826` `xc_family_mismatch`: "-- another XC family (XC-FAMILY mismatch): …" — the status
  code in capitals inside the sentence.
- `:713-715` (error, a missing file): "or point `psml_lib` at a library that has it" — a
  field name; the form calls it "Pseudopotential directory (.psml)"
  (`catalogue.template.toml:1039`). The same in `validation/siesta.py:86-87, 123-124`
  ("set `psml_lib` to …").

How sure: traced. Fix: reword each in the form's words, consequence first.

## 11. A refusal now carries the warning's "every stage computes …"

`transport/compose.py:688-702` (`_refuse_outside`, kept as an error) embeds the collision
warning from `validation/sidecar.py:184-192`. The person now reads:

> "the junction, centred in its cell, has … outside it along z, the transport direction:
> along the transport direction (z) the cell leaves only … -- so the end layers of the two
> leads overlap, and every stage computes the two leads run into each other.  On the Cell
> page, set …"

In a refusal nothing runs, so "every stage computes" is false here; "along z, the
transport direction: along the transport direction (z)" says the place twice; and
"computes the two leads run into each other" is not a sentence.

How sure: traced. Fix: let the refusal take the measurement and the way on only (the
sidecar could hand the `closes` sentence and the numbers separately), or reword the
warning's consequence so it reads true in both places ("so the two leads' end layers
overlap").

## 12. Page words that are not the pages'

- `transport/wizard.py:285` "select them on the Modify tab" — the tab strip says
  "Molbuilder" (`web/tabs.py:43`); `compose.py:570` and other messages say "the
  Molbuilder tab".
- `kmesh.py:337` and `validation/transport.py:113` "after the first stage" — the page
  names it "seed" (`transport/stages.py` `rung_notes`), and the same sentence says "the
  device stage" by its page name.
- `validation/stages.py:142` "its relaxation steps are 0" — the field is "Max
  geometry-optimisation steps" (`catalogue.template.toml:542`).

How sure: traced. Fix: the tab strip's and the form's labels.

## 13. A lead at one k-point along z is told only "outside the recommended range", with a deck keyword

With `above` lowered to 0, `electrode_kz = 1` is said only by the generic range sentence
(`validation/metadata.py:101-104`, `_keyword_suffix`). Measured: "Electrode k-points along
z (%block kgrid_Monkhorst_Pack) = 1 is outside the recommended range [50, 200]". The deck
keyword is in the sentence a person reads (the brief: no deck keywords), and the cutoff's
range warning in finding 2 carries "(TS.Contours.nEq.Fermi.Cutoff)" the same way.
How sure: measured. Fix: the range sentence names the setting by its label alone.

## 14. A lead now has two channels for what is said of it, and one is computed and dropped

`transport/wizard.py:92-101`: `ElectrodeModel.notes` (Issues with a severity and a
`where`, "what was measured of this lead") and the new `ElectrodeModel.said` (bare
strings, always warnings). On every prep after the first, `compose.py:1173`
(`_extract_and_gate_electrodes(dev)` in `load_compose_record`) recomputes `said` — the
frozen and spacing checks — and nothing reads it: `junction_said` reads the record's
`lead_findings` instead. On the Transport card the period and seam come labelled
"L-electrode: …" with a `where` (`blueprints/transport.py:310-314`); the lead checks come
with no `where`, so the page cannot place them by lead.

How sure: traced. Fix: one channel — the checks as `notes` entries at `warn` with a
`where` (e.g. `lead.frozen`), their messages what the record keeps; or, if `said` stays,
build it only where it is recorded.

## 15. The record-kept lead findings and the frames check have no test

What `lead_findings` exists for — a later prep saying what the first composition found
(`compose.py:793` written, `:637` read back) — is not tested on the road that reads it
back; `tests/test_transport_on_a_real_junction_e2e.py:1371` asks `junction_said` of the
fresh composition only. `compose.frames_moving_the_leads` is asked by no test at all (grep:
no test names it or its words), neither by the frame-set builder (`mode_frames.py:360`)
nor at composition.

Scenario: a later change writes the provenance without the key, or the frames check
returns `None` early; every later prep stops saying that a lead moved, and nothing goes
red. How sure: traced (grep). Fix: one row each where the road already runs —
prep after the first, saying the held-by-relaxation finding; and a frame set moving a lead
atom, said at composition. If the release ships without them, say so in its record.

## 16. The two `said_too` switches change what a maker returns

`tests/support/road.py:297/354` and `:415/475`: a second boolean beside `refused`, each
turning a single return into a tuple; one caller each
(`test_transport_decks_e2e.py:460`, `test_what_a_kind_offers.py:83`). With both flags set,
`refused` wins and `said_too` is ignored without a word.
How sure: traced. Fix: one shape — the makers return what was said always (e.g. the
output as a second value every caller unpacks, or an object with `.folder` and `.said`) —
or one switch that names what is returned.

## 17. One rule, two rows

`tests/validation/test_geometry.py:25-41`: rows 0.1 Å and 0.5 Å test the one rule
(`< 0.7 Å`, warn). The 0.1 Å row alone pins "however close"; the 0.5 Å row is the old warn
test kept. How sure: traced. Fix: keep the 0.1 Å row (testing.md § 3b: one example per
rule).

## 18. Test docstrings and names still speak of refusals

- `tests/test_cell.py:411-412` "the tolerance must not refuse real data"; the tests named
  `test_bulk_z_period_tolerates_…` / `…_admits_…` (`:411`, `:423`) now assert
  `uneven_layers`.
- `tests/validation/test_transport_kind.py:8-16`: "§ 0.4's whole argument for guards over
  advice", "a validator that refuses everything".
- `tests/test_electrode_labels_are_frozen.py` (finding 1).

How sure: traced. Fix: the words of what each now asserts (testing.md § 3b: a test states
its contract in its first lines; a stale one hides its own obsolescence).

## 19. Docs: a stale plan line, a rule in two homes, a sentence restated three times, a contrast that no longer holds

- `docs/plans/plan.md:44-46`: "in the tree (2.77.6): the code and its tests done, the
  contract sentences, the end-to-end batch and the two reviews to come" — the contract
  sentences are in this tree.
- `plan.md:63-67` states the error rule ("Kept as errors: nothing to compute, a file …,
  what a deck … cannot express, and bookkeeping …") that `science/overview.md:132-138`
  now owns — point to it.
- The kind table's charge reason (`kind_table.toml:235`) is restated word for word in
  `execution/architecture.md:672-673` and `science/chemistry-correctness.md:503-508`
  (which itself says "in the reason the kind table states" and then restates it) — the
  release had to edit all three.
- `engines/template.md:867-877`: "A value the engine takes is not a limit … What a
  DESCRIPTION states is another matter: … warns of `relax_steps = 0`" — both halves are now
  "taken, said"; the contrast reads as a difference that no longer exists.
- `engines/stages.md:1033`, the preflight table's range row: "silent for a value refused above" —
  the code also silences it beside the step warning (`validation/task.py:178`).

How sure: traced. Fix: correct the plan line; point instead of restating; fold the § 5.3
paragraph into one sentence; add "or warned of" to the stages row.

## 20. The one lead refusal left tells the person to do something no page offers

`cell.py:837-840`, reaching the person through `wizard.py:245-255` and
`compose.py:614-622`: "cannot derive a bulk z-period from a single atomic layer; pass an
explicit z_period (the lead's bulk lattice repeat)". No page, command or record takes a
`z_period` (grep: none outside `cell.py` and `wizard.py`). This release names it as one of
the two refusals `extract_electrode_model` keeps (`wizard.py:370-372`).
How sure: traced. Fix: the way on a person has — label at least two whole layers of the
metal as each lead.

## 21. The frames sentence, said by the frame-set builder, does not say whom it is about

`transport/compose.py:377-382`: "set.xyz: frame 2 (of 5) moves atom index 3 (Au) by
0.0123 A; frame 3 … , from where the first frame has it.  Every frame uses the leads
computed once from the first frame, so a lead atom should stay …". Said on the Spectrum
page when a frame set is written (`mode_frames.py:360-362`), where no transport
calculation exists yet: "every frame uses the leads" is true only of a transport
calculation citing the set, which the sentence never names. "from where the first frame
has it" follows a list of several atoms and frames.
How sure: traced. Fix: "A transport calculation citing this set computes the leads once,
from the first frame, so …", and "from where the first frame has them".

---

Checked and clean: the 13-row count and 170-row count in the plan match the inventory;
the archive README row and the two `fix2-review-*.md` verdict lines; `charge_finding`'s
wording (says what happens, then what to do); the PCM warning; the transport boundary and
transverse warnings; the stripe note; the revised tests assert outcomes (decks, reports,
routes, door answers) — none reads source or doc text; the new setting-door rows and the
`_ENGINE_STOPS` table are rows; every renamed function's old name is gone from code
(`frames_keep_the_leads`, `_refuse_unless_frozen_bulk`, `severity=` on
`check_junction_boundary`).

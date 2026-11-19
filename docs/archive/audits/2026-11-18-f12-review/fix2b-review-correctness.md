*Verdict: closed in 2.77.6.* The correctness review of F12 fix 2b, read before its release: every finding was fixed in 2.77.6, the cited spin (finding 7) kept as cited by decision (a cited relaxation's treatment is the junction's, TranSIESTA's stop said beforehand). Kept as the record of what the review found.

# F12 fix 2b — correctness review

Reader: correctness lens. Read only; no file under the checkout touched. Measurements are read-only
Python run from the checkout with config, state and projects pointed at
a scratch folder. "Measured" means I ran it,
"traced" means I read the code path through, and "plausible" means I read the code but did not run it.

What holds (checked, no finding): no scientific refusal from the inventory's 13 (S) rows still
stands. I checked `validate`, `refuse_on_error`, the description preflight, the transport describe,
`transport_parts` and the compose gates. Every remaining `"error"` is one of the inventory's kept
(a)–(d) rows. No caller maps a changed `where` code back to error. The lead findings survive
later preps: `compose_junction` → `written_and_read_back` → `load_compose_record` reads
`slot-provenance.json` whole, and `junction_said` reads `lead_findings` from it. They are not
carried into another junction, because a re-pointed citation or a changed rename means no record,
and an edited pair is refused. Prep says them once above several stages (`prep_said`). The card adds
them once beside the leads' `notes`. Non-collinear spin on transport goes through the gate with info
only. I measured this on the seed, the leads, the device and the transmission: no error.

## Findings (12)

### 1. The PySCF Methods paragraph says the solvent is in the Hessian for runs where it is not (measured)
`molbuilder/pyscf/vibration_emitters.py:1481-1489`:
`f"... at every self-consistent field of the run, the Hessian including the solvent's own response, in the equilibrium-solvation approximation ..."`.
This is written for every solvated run. The new warning (`validation/spectra.py:383-395`) and
`engines/vibration.md` § 4.6 both say the held-atom Hessian (`hess_elec(atmlst=)` + `hess_nuc`,
`spectra/pyscf_vibration.py:157-164`) does not carry it.
- **Scenario.** Water molecule, atom 0 held, solvent water, IR on. The warning says "the
  frequencies with 1 atom(s) held … computed without the solvent's response". The Methods text
  written with the results says "the Hessian including the solvent's own response". A person who
  copies the Methods into a paper states something the run did not compute. These runs were refused
  until this release, so the sentence was true for every run that got this far.
- **Also (plausible).** On the analytic-IR route (IR only, nothing held, `pyscf.prop.infrared`
  installed), the Hessian is `proc_hessian_`'s `hess_elec + hess_nuc`
  (`pyscf_vibration.py:200-217`), so the frequencies miss the solvent term too. The warning lists
  only "the IR intensities".
- **Fix.** Make the clause conditional, in the same terms the warning uses: say "including the
  solvent's own response" only for the frequencies-only route with nothing held, and otherwise name
  what was computed without it.

### 2. The relaxation-time lead warning still promises a refusal that is gone (traced)
`molbuilder/validation/sidecar.py:101-108`:
`"... or the relaxation will move the leads and the junction cannot be composed afterwards (the composer refuses it, by which point this run has been paid for)."`
The docstring at 66-85 says the same ("refuses a lead whose atoms are not frozen", "The refusal
belongs at compose … and it is already there").
- **Scenario.** A person relaxes a labelled junction with the leads not frozen. Prep and the web
  preflight warn that the junction "cannot be composed afterwards". Composition now succeeds and
  only warns (`wizard._lead_findings`), so they are told the opposite of what happens.
- **Also.** The message uses internal names: `"frozen_atoms" in /modify`.
- **Fix.** Say what now happens: the transport calculation will be composed from where the
  relaxation left the lead atoms, and it will say so. Name the Modify tab's selection instead of
  `"frozen_atoms"` and `/modify`.

### 3. An uneven lead is built on the bridge-side spacing for R and the outer spacing for L (measured)
`molbuilder/cell.py:842` `d_interlayer = float(diffs[0])` via `transport/wizard.py:407`. The
layers are ascending in z, so the "first spacing" is at L's outer end but at R's bridge end, which
is the spacing a relaxation moves.
- **Measured.** Bulk spacing 2.35 Å. On each side, the lead layer next to the bridge moved
  0.15 Å toward it:
  - L lead: gaps [2.35, 2.35, 2.50], seam 2.350, period 9.55.
  - R lead: gaps [2.50, 2.35, 2.35], seam 2.500, period 9.70.
  - Bulk period: 9.40.
- **Scenario.** A person forgets to freeze the leads; the relaxation moves the layers next to the
  bridge. The R lead's semi-infinite continuation is then built on the perturbed spacing, while
  the L lead's uses the bulk one. The "not frozen" finding is said, but the uneven-layers finding
  is skipped (`said` is non-empty, `wizard.py:409`), so nothing says which spacing the leads were
  built on.
  Before this release such a lead never got built.
- **Fix.** Read the spacing at the lead's end away from the bridge: the first for L, the last for
  R, by the junction's orientation. Correct `junction-cell.md`'s "the first" to match.

### 4. On a lead's own stage, the boundary warning talks about "the two leads" and the Cell page (measured)
`molbuilder/validation/sidecar.py:159, 176, 187-191`: `"between the two leads across its edge"`,
`"the end layers of the two leads overlap"` and `closes` = `"On the Cell page, set the cell's
length along z to …"`. These are asked on every stage (`validation/transport.py:168`,
`whole_structure_is_the_lead=True`).
- **Measured.** A frozen L block with layers at 0, 1.0, 3.35 and 5.70 Å (a stray part-layer) is
  now built (z_period 6.70). The `electrode_L` stage's gate says: "the cell leaves only 1.00 Å
  between the two leads across its edge … the end layers of the two leads overlap … On the Cell
  page, set the cell's length along z to 8.050 Å".
- **What is wrong.** A lead stage has one lead. Its length along z is the one molbuilder builds
  from the spacing, not one the Cell page edits. The same cause is said three times: the uneven
  layers at composition, the seam finding, and this. Uneven leads used to be refused at
  composition, so this stage was never reached.
- **Fix.** When `whole_structure_is_the_lead`, word it as the lead's own seam and point at its
  layers and labels, not at the Cell page. Or leave it to the uneven-layers finding and the seam
  finding, which already say it.

### 5. The window sentence says "the current is cut short" where the current is left out (measured)
`molbuilder/transport/energies.py:73-78`:
`"... but the energies computed reach only [lo, hi] eV, so the current is cut short.  Widen …"`.
- **Measured.** `record.low_bias_iv(E∈[-0.3, 0.3], V=[0, 0.4])` gives `current_a = [0.0, None]`
  with this note. The record leaves that voltage's current out rather than cutting it short.
- **Same on the description preflight.** Under the low-bias approximation
  (`validation/transport.py:261-280`) the record integrates every listed voltage over the one 0 V
  slice, and a short window leaves the voltage blank.
- **Fix.** Let each caller say the consequence: cut short for a point's own TBtrans run, left out
  of the record otherwise. Or word the shared sentence to cover both.

### 6. A charged citation is said once, at describe, and never at prep (traced)
`molbuilder/transport/describe.py:387-391` is the only caller of `charge_finding`.
- **What is missing.** Nothing keeps the finding with the composition. Prep (first or later), the
  card's citation pick and Task setup never say it. The lead findings, by contrast, are said at
  every prep through the record.
- **Scenario.** A person describes from the command line. Days later they prep: the junction is
  computed at 0 on a geometry relaxed at +1, and nothing at prep says so. Until this release the
  calculation could not be described at all.
- **Fix.** Compute it where the citation is read for the composition and keep it in the record
  beside `lead_findings`, so `junction_said` says it on every surface. Describe can still say it
  first.

### 7. A cited non-collinear or spin-orbit relaxation now sets the transport calculation's spin (measured)
`molbuilder/transport/citation_defaults.py:145-153`. Since `siesta.transport` now offers
`non-collinear` and `spin-orbit`, `_can` drops nothing.
- **Measured.** A cited pair whose record states `non-collinear` (or `spin-orbit`) gives
  `values = {'spin_treatment': 'non-collinear', …}`. At 2.77.5 it was left blank and worked out
  on the junction as restricted or unrestricted.
- **Scenario.** A person cites their non-collinear relaxation and changes nothing. The device stage
  stops after the seed and both leads have run. The only word is an info line at prep.
- **Not stated.** The brief does not name this change to the default, and neither does
  `engines/transport.md` § 3.1's spin note.
- **Fix.** Either keep the citation from defaulting a treatment in `STOPS_ON_SPIN` (it stays
  offered for the person to pick), or state in the contract that the cited treatment is carried.
  This is a call for the user, and I have not decided it.

### 8. The frames-moving warning does not say that TranSIESTA stops that frame's device stage (traced)
`molbuilder/transport/compose.py:377-382`: `"... a lead atom should stay where that frame has it,
within 0.001 A, or the frame no longer matches its leads."`.
- **Traced in SIESTA's source.** TranSIESTA compares the device's electrode atoms with the lead's
  at start and dies above `TS.Elecs.Coord.Eps`, default 0.001 Å (`m_ts_options.F90:380, 808-810`,
  `ts_electrode.F90:1080-1110`).
- **Scenario.** The device stage of every moving frame (after frame 0, whose geometry the leads
  were built from) stops after the queue wait, and the warning never said that would happen.
- **Fix.** Add what happens to the calculation: TranSIESTA stops the device stage for each frame
  named.

### 9. A bias-window reach under 2.5 kT is said twice on the device, and as a warning on the other stages (measured)
`molbuilder/data/catalogue.template.toml:2874` `range = [2.5, 20.0]` beside
`validation/transport.py:117-130`.
- **Measured** with `negf_neq_fermi_cutoff_kt = 2.0`:
  - On the device: `warn` "… = 2.0 is outside the recommended range [2.5, 20.0]" and `info` "…
    TranSIESTA stops the device stage under bias when this is below 2.5 kT".
  - On the seed, both leads and the transmission, which never write it: the same `warn`.
- **What is wrong.** The range's lower end is TranSIESTA's stop, so that stop is said as a
  warning, against the ruling that an engine's stop is information.
- **Fix.** Say it once. For example, let the range warning stand aside where the info speaks, as
  the preflight already does for `relax_steps`.

### 10. `relax_steps = 0` is said twice at prep (measured)
`validation/task.py:156-182` puts the stepless finding into `refused=`, so the description's range
warning stands aside. The deck's settings gate does not.
- **Measured.** `validate(H2, relax_type=CG, relax_steps=0, rung=relaxation)` gives `warn` "Max
  geometry-optimisation steps (MD.Steps) = 0 is outside the recommended range [1, 10000]".
- **At prep.** Prep shows both `findings` (preflight) and `deck_findings`, so one setting gets
  the plain warning and the one that uses the deck keyword. Before this release the first was a
  refusal and the second was never reached.
- **Fix.** The same stand-aside in the deck gate.

### 11. The refusal for a junction longer than its cell now says "every stage computes" (measured)
`molbuilder/transport/compose.py:695-703` embeds the collision warning's words in a ComposeError.
- **Measured.** "the junction, centred in its cell, has atom indices 0 (Au), 7 (Au) outside it
  along z …: … the cell leaves only -0.70 Å between the two leads … so the end layers of the two
  leads overlap, and every stage computes the two leads run into each other.  On the Cell page …".
- **What is wrong.** Nothing is computed, because the composition is refused. The room is also
  given as a negative number.
- **Fix.** In this refusal use only the measurement and the length that closes the gap, without
  the "every stage computes" clause.

### 12. Code text that still states the old refusals, and new messages missing a way forward (traced)
None of these changes behaviour. Each is a sentence that now contradicts the code:
- `compose.py:303-312` `_classify_pair`'s docstring still lists the per-frame check, which moved
  to `compose_junction`.
- `web/blueprints/transport.py:949-950` `_panel_template`: "Raises ValueError … a cited run that
  carried a net charge, ES7".
- `validation/transport.py:149-151, 189-191`: "both are refusals", "the electrode rung … refuses
  the same calculation".
- `validation/task.py:139-142`: "one of them refuses".
- `modify.py:160-162`: close atoms "enforced by the GENERATION gate … at emit time".
- `cell.py:817-825`: `bulk_z_period` keeps an unused `tol_ang` parameter, and its docstring says
  "the spacing every adjacent pair of layers is checked to share".

Plain words, against the brief's "what happens, then what to do":
- The two new info lines (`validation/transport.py:111-114` spin, `:126-129` cutoff) say the stop
  but not what runs (two spin channels; 2.5 kT or more, 5 by default). The stripe and pole lines
  do say it.
- `pseudos.py`'s `relativistic_mismatch` ("calc expects …"), `parse_warning` ("header had
  issues") and `generator_mismatch` were not reworded, though the brief says the 2.77.5
  pseudopotential warnings were.

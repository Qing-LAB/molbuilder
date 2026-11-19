*Verdict: closed in 2.77.5.* The discipline review of F12 fix release 2, read before its release: every finding was fixed in 2.77.5, the plan sentence (finding 6) answered by the sweep that followed (fix 2b). Kept as the record of what the review found.

# Discipline review: F12 fix release 2 ("the deck writes the value the person set")

Reviewer lens: discipline (one home per fact, unified API, residue, rules the
contracts do not state, test discipline, the plan consolidation). Read only.
I read every touched module in full where a finding depends on it, and the
contract sections each fix cites. Nothing here asks for a new warning, refusal
or threshold.

Checked and clean:
- No reader is left of `_XC_FAMILIES`, `IDENTITY_ITEMS` or `render_deck(verbose=)` / `prepare_deck(verbose=)`. The only copies are in the untracked `build/lib/`.
- No new door exists only for a test, and no new branch on an engine's name.
- `solvent_of` now has every reader of `cfg.solvent`.
- Every line taken out of `plan.md` is word for word in the new archive file or in the new plan. One heading was renamed.

17 findings, worst harm first.

---

## 1. A frame set judges "stated already relaxed" from today's description, not from the run's own record

`molbuilder/spectra/mode_frames.py:107-109`
```python
calc = open_calculation(run.place.root)
cfg = stage_config(calc.task, command_stage(run.stage), calc.template_text)
if not getattr(cfg, "already_relaxed", False):
```
**What goes wrong.** Whether a finished run's structure was stated relaxed is
now read from the calculation's template as it is *now*. The run's own result
already states this, on both engines:
- PySCF: `vibration_emitters.py:588`, `'already_relaxed': ALREADY_RELAXED`
- SIESTA: `vibrational_analysis.py:358`, `"already_relaxed": bool(already_relaxed)`, "the person's statement, as made"

`frameset_from_mode` has parsed that result (`res`, line 263) before it calls
`_relaxation_record` (line 303).

The contract says the generator "reads the result alone"
(`engines/vibration.md` § 5.10 ③ and § 4, line 3487). So there are now two
sources for one fact, and the live one can disagree with what the run did.

**Scenario.** A person runs a vibration with *already relaxed* ticked. Then
they untick it to try a relaxing run, and make a frame set from the first run.
The generator reads `False` and stamps the frame set with the vibration's own
deck record (its cheaper k-grid and mesh) as the relaxation's. That is exactly
the transport-citation error § 5.10 ③ was written to prevent (the AuBDTAu
measurement). A second case: if the run's stage has since been renamed or
removed from `task.json`, `stage_config` raises, and the frame-set door shows a
traceback.

The module header (`MODULE spectra.mode_frames (L2; the run door, …)`) also no
longer lists what the module imports (`calculation`, `jobset.ladder`).

**How sure:** traced, by reading the code and the contract.

**Smallest fix:** pass `res.relaxation.get("already_relaxed")` into
`_relaxation_record` and drop the `open_calculation` / `stage_config` lookup.

---

## 2. The by-hand `finish` cites a Raman data file that is not beside its result, and it repeats the run's finish step by step

`molbuilder/spectra/siesta_vibration.py:582-588`
```python
here = _here(deck)
index = _rp.read_index(here.index)
res = result_of(run, **_relaxation_of(index))
res = _with_raman_phase(res, run, _write_optical(here, index, res),
                        PHASE_COMPLETE)
out = Path(deck).parent / compose(run.label, ROLE_SPECTRA)
```
**What goes wrong.**
- `_write_optical` writes `<label>.optical.nc` at the **run's root**
  (`here.run_dir / compose(here.label, ROLE_OPTICAL)`, line ~1120). The result
  that cites it is written **beside the force-constant deck**, a folder below.
  The citation holds only the file's name, and the one reader resolves it
  beside the result: `raman.optical_data_of`, `path.parent / result.optical["data_file"]`, which refuses with "cites …, which is not beside it".
- The contract for the by-hand verb (`engines/vibration.md` § 5.5, *When the
  finish fails*) says it "writes the spectrum beside that call's deck … its
  root holds no result". The verb now also rewrites a file at the root.
- `_finished` is a second copy of `finish_run`'s composition: `_here`,
  `read_index`, `result_of(**_relaxation_of)`, `_write_optical`,
  `_with_raman_phase`. Only the source of the output and the destination
  differ. The next phase added to one copy will be missed in the other. That
  is how F-D1 happened in the first place.

**Scenario.** A run that asked for Raman fails in its finish. The person runs
`mb_vibration.pyz finish <deck> <output>` as § 5.5 says. The new
`<label>.spectra.json` says `phase_raman: complete`, but opening its Raman view
is refused because the data file it names is not in its folder. The new e2e
test compares only `phase_relaxation` and `phase_raman`, so it does not see
this.

**How sure:** traced (`_write_optical`, `optical_data_of`, § 5.5). Not run.

**Smallest fix:** one composition, used by both `finish_run` and `_finished`,
that takes the folder the result is written to and writes the optical data
file beside the result that cites it. Also decide in § 5.5 whether the by-hand
verb may touch the run's root.

---

## 3. D12's new stability verdict uses the test the contract calls wrong

`molbuilder/pyscf/input.py:1024-1031`
```python
"        else:",
"            _internal = _mb_stability()[0]",
"            _MB_STABLE = bool(_mb_np.allclose(_mb_host(_internal),",
"                                              _mb_host(mf.mo_coeff)))",
```
**What goes wrong.** After the third restart, the added check decides
stable or unstable by comparing orbital coefficients. `engines/pyscf.md` § 7.3
(line 792), *Why the energy and not the orbitals*, says: "Comparing orbital
coefficients … looks obvious and is wrong. A degenerate shell … `stability()`
returns numerically different coefficients for a physically identical state,
for ever … a coefficient test called all three unstable and ended the run with
a false warning." The loop just above follows that rule and judges by energy.
So the block now has two rules for "stable".

The contract's § 7.3 and § 7.4 also do not mention this extra check.

**Scenario.** An open-shell radical with a degenerate pair, where each of the
three restarts lowers the energy. The extra check finds rotated coefficients
at a true minimum and prints `stability: WARNING -- still internally unstable
after 3 restarts`. That is the false warning § 7.3 was written to remove, and
it says the geometry and frequencies are provisional.

**How sure:** traced against the contract. Not run.

**Smallest fix:** judge the final check by § 7.3's rule (energy: a restart from
the suggested orbitals that does not lower the energy is stable), not by
coefficients. State the final check in § 7.3 and § 7.4.

---

## 4. D10 is fixed in one of the two decks that save `_optimized.xyz`, and its rule is in no contract

`molbuilder/pyscf/vibration_deck.py:344`
```python
out.append(emit_save_call("mol", PYSCF_OPTIMIZED, names, indent="    "))
```
**What goes wrong.** `emit_save_call` is "one spelling … for both decks". Only
the optimization deck passes `converged="_GEOM_CONVERGED"` (`input.py:696`).
The vibration deck's relaxation runs the same `_mb_relax` with
`on_nonconvergence`. Under `proceed` it keeps the geometry with
`_geometric_converged` False, yet still saves it under
`_MB_WORDS['optimized_geometry']`. `save_optimized_xyz` applies to vibrations
too (catalogue row, default true).

Separately, the new test
`test_the_saved_geometry_is_called_optimized_only_when_it_is` cites
"`engines/pyscf.md` § 3: the geometry … says what it is on its comment line --
optimized where geomeTRIC's criteria were met". Neither § 3 nor
`vibration.md` holds that sentence. The test's expectation ("Optimized" in the
title) comes from the code's own `_SAVED_AS` words.

**Scenario.** A PySCF vibration with *already relaxed* unticked and
`proceed`. The relaxation stops at `geom_max_steps`, and
`<job>_optimized.xyz` is titled "Optimized geometry (PySCF)". That is D10
exactly, on the vibration road.

**How sure:** traced.

**Smallest fix:** pass `converged="_geometric_converged"` at
`vibration_deck.py:344`. Write the comment-line rule into `engines/pyscf.md`
§ 3. Add a `proceed` row for the vibration deck to the D10 test.

---

## 5. G-D1: "the box did not change" is judged at 1e-9 Å, but the Cell page re-sends at 1e-6 Å; the test's stand-in skips the page's rounding

`molbuilder/periodicity_gate.py:385-393` (and the same inline rule at `:321-322` for `offset_by`)
```python
return np.allclose(np.asarray(x, dtype=float),
                   np.asarray(y, dtype=float), rtol=0.0, atol=1e-9)
```
`molbuilder/web/static/modify/periodicity.js:64`
```js
function round(n) { return Math.round(Number(n) * 1e6) / 1e6; }
```
The page fills every cell box (`:495`), vacuum box (`:451`) and origin box
(`:524`) through `round`, and Apply sends back what is in the boxes.

**What goes wrong.** The two halves of "unchanged" are stated in two places
and already disagree. Measured on a number, not on the page: an in-plane
length of 7.492851793542962 Å (2.884·√3/2·3), written to `.XV` in Bohr to 9
decimals and read back, gives 7.49285179343 Å. The page re-sends 7.492852,
which differs by 2.1e-7 Å. `_same_box` is then False and the record is marked
modified.

The test helper `tests/test_an_edit_on_a_recorded_structure_e2e.py:98`
`_its_own_box` says "each field as the page fills it" but sends full
precision (`np.asarray(s.cell).tolist()`). Its fixture is a 10 Å cube, where
the page's rounding is exact. The test cannot see the gap.

**Scenario.** A hexagonal Au(111) junction is relaxed on SIESTA and saved
with `xv2xyz --from-run`. The person opens the Cell page and presses Apply
with nothing changed. The structure is marked modified, and a later blank
charge drops the recorded one. That is G-D1, unchanged for any cell not
written to six decimals.

**How sure:** the mechanism is traced and the numbers are measured. The page
itself was not driven.

**Smallest fix:** state the page's precision once, as the tolerance both
`_same_box` and `_apply_block`'s `offset_by` test compare at. Have the test
build its block at the page's rounding, on a cell that has more than six
decimals.

---

## 6. The plan states a rule wider than the user's words and than the code, including this release's own code

`docs/plans/plan.md:60-63`
> **No check or warning stops a build or a calculation** *(the user, 2026-11-18: "None of these things should stop any calculation being constructed or being run. It is totally user's decision")*

**What goes wrong.** The quoted words were about the pseudopotential findings
("these things"). The headline turns them into a rule about every check, and
the code does not keep that rule:
- `pseudos.ERROR_STATUSES` still stops prep on a missing or misnamed file.
- The settings gate still has about 25 error-severity findings that stop prep
  (`validation/*`).
- This release **adds** a stop. D3 makes PySCF's `job_name` `shared`
  (`catalogue.template.toml:1245`), so a stage's own job name is now refused
  at prep.

The memory rule *trace a rule to the user's words* applies: a reader of the
plan cannot tell which statement is the user's.

**Scenario.** The next release, following the plan, takes out a refusal the
user never asked about. Or a reviewer flags D3's new refusal as breaking the
plan's rule, and nobody can tell which one is right.

**How sure:** measured (grep of the refusals; read the D3 diff).

**Smallest fix:** state the rule's scope as the user gave it (pseudopotential
findings: only a file SIESTA cannot open stops). Or, if the rule is meant to
be general, list the stops that remain, D3's among them, as an open question
for the user.

---

## 7. Two writers for the PySCF header's solvent line, and the `'?'` fallback behind D9 is kept

`molbuilder/pyscf/input.py:269-271`
```python
out.append(f"Solvent   : {solvent_of(cfg)} ({cfg.solvent_method}, "
           f"eps={_SOLVENTS.get(solvent_of(cfg), '?')})")
```
`molbuilder/pyscf/vibration_emitters.py:161-163`
```python
out.append(f"Solvent   : {_solvent}, "
           f"{cfg.solvent_method} "
           f"(dielectric constant {SOLVENTS[_solvent]:g})")
```
**What goes wrong.** D9 was "Water written with `eps=?` in the header". The fix
unified the name reader but left two header writers, each with its own
spelling and its own behaviour for an unknown name. The optimization deck
silently prints `?`; the vibration deck raises `KeyError`. The `?` is the
fallback that hid the original disagreement.

**Scenario.** The next change to the solvent table or its reader goes into one
header only. The optimization deck again writes `eps=?` in silence while the
vibration deck crashes.

**How sure:** traced.

**Smallest fix:** one `solvent_header_line(cfg)` in `scf_setup`, next to
`solvent_of`. It indexes `SOLVENTS` directly, since the validator has already
refused unknown names. Both headers call it.

---

## 8. The rows for D2 and D4: one tests a branch the named failures do not take; the other cannot fail once the help is reworded

`tests/data/the_deck.toml:103-106`
```toml
name = "a value is written as it was set, never rounded"
template = { dm_tolerance = 0.000125 }
deck_reads = { dm_tolerance = 0.000125 }
```
**What goes wrong (D2).** The row runs only the `"e"` branch of `layout._exact`
(`format_float_scientific`). The failures it names, a 0.00625 V bias and a
5e-7 eV broadening, go through the `"f"` branch (`format_float_positional`).
No test row exercises that branch (`tests/data/*.toml` has no
`bias_voltage_v` or `*_eta_ev` row). Breaking the `"f"` spelling, for example
by giving it a precision, passes every test.

`tests/data/the_deck.toml:112-115`
```toml
template = { verbose_comments = false }
deck_lacks = ["# Real-space integration grid (Ry)."]
```
**What goes wrong (D4).** The row checks that a sentence hand-copied from the
catalogue's `mesh_cutoff` help is absent, and there is no positive twin. If the
help is reworded, the row passes whatever `verbose_comments` does.
`tests/test_deck_runner.py:118` already checks the same rule through the same
`render_deck` code path, taking the note from its source
(`se.parameter("mesh_cutoff","siesta").note()[1]`). That is two tests for one
rule, and the weaker one is the new one.

Smaller: `tests/test_the_setting_door.py:101` now parametrizes one `row`
argument, so every row's test id is `row0…rowN`. A node id no longer says
which row it runs.

**How sure:** traced. Mutation not run, since running tests was not allowed.

**Smallest fix:**
- D2: add a transport or bias row for the `"f"` branch, such as `bias_voltage_v = 0.00625` read back as 0.00625.
- D4: keep one test. If the road row stays, derive the note text the way `test_deck_runner` does, or pair it with a `verbose_comments = true` twin that contains the same line.
- Setting-door table: give the table `ids=`.

---

## 9. The pseudopotential tests: one rule tested twice at one surface, a hand-typed config, and assertions on the constant's contents

- `tests/test_pseudos.py:394-398` (GATE_ROWS, "a-gga-pseudo-on-an-lda-calc-is-said") and `tests/validation/test_siesta.py:387` (the "LDA" row) both check that an LDA run on GGA pseudopotentials is a warning in the preflight. The GATE_ROWS copy extends a hand-typed stand-in, `class _Cfg: … xc_functional, xc_authors = xc` (`:429`). The new test uses the real `SiestaConfig` through `validate`.
- `tests/test_pseudos.py:286`, `assert ERROR_STATUSES == frozenset({"missing", "misnamed"})`, and `tests/test_pseudo_requirements_layer.py:101`, `assert "semilocal_only" not in ERROR_STATUSES`, pin the constant's contents, not what prep does. The outcome, that the preflight says it as a warning, is tested for `dead_projector` (`TestDeadProjectorC5`) and for the family mismatch, but there is no outcome row for `semilocal_only`.

**What goes wrong.** A change to how severity is mapped (for example a
severity read somewhere other than `ERROR_STATUSES`) passes the set-membership
assertions while prep stops. Meanwhile the duplicate rows double the upkeep
for one rule.

**How sure:** traced.

**Smallest fix:** drop the GATE_ROWS family row in favour of the
`SiestaConfig` test. Replace the two set assertions with a preflight row (the
S v0.5 file through `_check_siesta_pseudo_coverage`, severity `warn`, no
`error`).

---

## 10. The "convention folder" hint relies on `ERROR_STATUSES`, whose meaning changed underneath it

`molbuilder/validation/siesta.py:181`
```python
if any(e.status in ERROR_STATUSES for e in found):
    return ""
```
**What goes wrong.** `_the_convention_covers` names the tree's
`pseudopotential` folder only when "the coverage check refuses none". Until
now, "refuses" covered a wrong XC family, a dead projector and a
zero-strength channel. Now it covers only files SIESTA cannot open. So the
hint changed meaning without anyone deciding it. The same module also builds
the calculation's expectations (`xc_authors`, `expected_xc_family(cfg.xc_functional)`)
twice: `:132-133` and `:174-177`.

**Scenario.** An LDA calculation with no `psml_lib`, while the tree's folder
holds GGA files. The gate now says "set `psml_lib` to `pseudopotential`": it
recommends a folder whose every file it then warns is the wrong family.

**How sure:** traced.

**Smallest fix:** state the hint's question in the code. If it is coverage
only, say so in the docstring. If not, read `e.status == "ok"`. Build the
calculation's expectations once, in one helper that both functions call.

---

## 11. `stages.md` still says the identity items are SIESTA's alone

`docs/engines/stages.md:466`
> the calculation's identity — its `system_label`, `species_order` and `psml_lib` — for every SIESTA kind;

**What goes wrong.** `template.md` § 5 was updated for PySCF's `job_name` (D3),
but this restatement of the same fact was not. The two chapters now disagree.

**Scenario.** A reader of `stages.md`, the chapter on overrides, expects a
PySCF stage's own `job_name` to be accepted, and prep refuses it.

**How sure:** measured (grep).

**Smallest fix:** add "and PySCF's `job_name`", or point to `template.md` § 5
instead of restating it.

---

## 12. The transport deck still writes reals at six significant figures, while D2's exact spelling is private to `siesta/layout`

`molbuilder/transport/deck.py:412-413`
```python
f"    from - {kt:g} kT - |V|/2 to |V|/2 + {kt:g} kT",
f"      delta {float(delta.value):g} eV",
```
**What goes wrong.** The D2 rule (a real is written exactly as set) lives in
`layout._exact`, which is private. The transport writer's block lines spell
values the person set (`negf_neq_spacing_ev`, `negf_neq_fermi_cutoff_kt`) with
`:g`, which rounds to six significant figures. The TBT window uses `!r`. That
makes three spellings of a real across one family of decks.

**Scenario.** A spacing of 0.0012345678 eV runs at 0.00123457 (measured:
`f"{0.0012345678:g}"`). It is low harm at usual values, but it is the same
defect D2 fixed, in a second writer.

**How sure:** measured (the format). Which values people use in practice is
not known.

**Smallest fix:** make `_exact` the SIESTA-family deck's one public spelling of
a real, and use it in `transport/deck.py`.

---

## 13. `why_unread` re-reads the stage-role rule's data inline

`molbuilder/template.py:2311-2313`
```python
held = (set(_roles_of(kind)) if rule is None or rule["otherwise"] is None
        else {r for rs in [*rule["named"].values(), rule["otherwise"]]
              for r in rs})
```
**What goes wrong.** `stage_role_rule` calls itself "the one statement of the
rule", and `stage_roles` is the function that reads it. `why_unread` now
expands the rule's shape by hand. Its `named` part is always `{}` in
`stage_role_rule`. If that shape changes, this inline copy silently goes
stale.

**Scenario.** A kind gains named stage roles. `stage_roles` follows the change
and the refusal's "which stages read it" does not, so it names the wrong
rungs again (D13's symptom).

**How sure:** traced.

**Smallest fix:** a function `roles_held(engine, kind)` beside `stage_roles`,
called by `why_unread`.

---

## 14. The seam finds a phase by its role, and the `-1` default is stated three times

`molbuilder/siesta/seam.py:48-50`
```python
energy_step=next(
    (p.energy_step for p in PHASES
     if p.role == role), -1))
```
**What goes wrong.** The rung gets a role from a phase
(`phase(point.phase).role`), and the seam then looks the phase back up by
role, with a silent `-1` for anything else. `vibration_phases` already has the
lookup function, `phase(name)`, but only by name. The default "the last step"
is stated three times: here, in `render_correction_script(energy_step=-1)` and
in `emit_correction_script(energy_step=-1)`.

**Scenario.** A role spelled differently, or a new phase whose role is shared,
silently gets the last step. The Makov–Payne script then prints a displaced
geometry's energy again (D6) with no error.

**How sure:** traced. Low harm today.

**Smallest fix:** look the phase up through `vibration_phases` (a `phase_of_role`,
or hand the seam the phase), and keep the default in one signature.

---

## 15. The energy shift's help text restates the engine default that the same catalogue row declares

`molbuilder/data/catalogue.template.toml:79`
> SIESTA's own default is 0.01 Ry too (atom.F), the value this catalogue starts at: …

and `:67`, `engine_default = { value = 0.01, source = "SIESTA 5.4.2 atom.F:57,3374: …" }`.

**What goes wrong.** One fact in two places in one row. The prose copy is the
one that drifted: D5 was this sentence saying 0.02.

**How sure:** measured.

**Smallest fix:** drop the number from the help (keep the reason why the
catalogue starts where it does), or have the help say it from `engine_default`.

---

## 16. Stale or doubled comments, and a history note in a contract

- `molbuilder/pseudos.py:580-603`: two consecutive comment paragraphs state the `ERROR_STATUSES` rule twice ("values that BLOCK -- a file SIESTA cannot open …" and "WHAT STOPS PREP IS A FILE SIESTA CANNOT OPEN …"). Between them, "Everything else — `xc_mismatch`, `relativistic_mismatch`, `generator_mismatch`, `parse_warning` — is advisory" lists four of the eight warning statuses.
- `molbuilder/parse/engines/pyscf.py:143`: "a list of runs, one per geom-opt step". `_scf_by_frame` exists because that is not true (the log holds runs that belong to no frame).
- `molbuilder/pyscf/input.py:1616-1617`: `emit_words` "opens with the comment line of each geometry … (`_SAVED_AS`, `_REACHED_AS`)", which no longer mentions `_STOPPED_AS`.
- `molbuilder/cli.py:242-244`: `--xc-functional` help hand-types the catalogue's choices "(LDA, GGA, VDW)".
- `docs/science/pseudopotentials.md:185-187`: "(until 2.77.5 every such run was refused, an authors table naming a `VDW` family no file declares)". This describes the removed code inside a contract; contracts should state rules only.

**What goes wrong.** A reader is told something the code no longer does, or the
same rule is spelled twice and the copies will drift.

**How sure:** measured.

**Smallest fix:** delete the duplicate paragraph and complete the warning list.
Correct the three docstrings. Take the CLI's choices from the catalogue item.
Drop the history clause from C2.

---

## 17. The plan moved standing guidance for reviewers into the archive, which calls itself "history, not policy"

`docs/archive/2026-11-18-plan-consolidation.md:306-312`: § 5r.2, "Do NOT 'fix' these — confirmed non-findings … so nobody fixes them by analogy". `docs/archive/README.md:3-4` says: "**Not a source of truth.** If you are reading a date-prefixed file, you are reading history, not policy."

**What goes wrong.** The list exists to stop a reviewer re-raising those six
items. Once it is archived, it no longer binds. The plan points to it only
under *Archived sections*.

**Scenario.** A later review flags `frozen_atoms`'s descriptor or the
multi-frame reader again and "fixes" it by analogy. Reading it with `ase`
would break live viewing.

**How sure:** measured.

**Smallest fix:** keep § 5r.2 in the plan (or in the structure chapter it is
about) as rules, and archive only the record of how it was found.

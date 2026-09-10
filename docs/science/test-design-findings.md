# Scientific-validation tests — design findings, 2026-09-08

**Role:** record — a review queue, not a contract
**Domain:** science
**Companions:** [`process/testing.md § 3b`](?doc=process/testing.md) — the rule these were
gathered under; [`science/validation.md`](?doc=science/validation.md) — what the gates
are for; [`process/test-audit-findings.md`](?doc=process/test-audit-findings.md) — the
same audit's NON-science half (two real code defects, four tests owed a redesign,
and the unapplied subsumption verdicts).

## Why this file exists

A four-partition agent audit gated ~3,000 of the suite's 4,448 test functions
against `testing.md` § 3b. **Scientific validation was a protected class** (user
ruling, 2026-09-08): *"these must be correct and rigid — but how they should be
designed can be discussed."* So no science test was cut, and roughly **120
drafted cuts were withdrawn** when the exception landed.

What the auditors returned instead is this: **cases where a science test would
pass for a physically wrong reason.** Every entry below is a design weakness in
a test that is currently green. None is a proposal to delete.

> **Nothing here has been fixed.** Each item names the test, what it asserts,
> and why that is weaker than it looks. Verify before acting — these are
> readings, and § 5 records one subsumption claim from the same audit that was
> mutation-tested and proved **false**.

---

## 1. The sharpest one: `test_pyscf_smoke.py`

Three problems compound, and together they mean the file could pass while the
numbers are wrong.

1. **The reference is self-referential.** All three "literature" values are
   quoted as *"PySCF 2.x prints …"* — so the reference is PySCF's own output. A
   PySCF-side change moves the reference rather than failing the test.
2. **The tolerance is ~50× too loose.** 5–10 mHa, against a stated cross-code
   reproducibility of **0.1 mHa**. A wiring error of that size passes.
3. **It never runs in the default environment.** A module-level
   `pytest.importorskip("pyscf")` and `pyscf` lives only in `molbuilder-pySCF`.
   The file's own comment records this trap rotting a retired frequencies test.

**A real external reference** (a published value, or a number from another code)
with a tolerance derived from the 0.1 mHa figure would make it fail for the
right reason. The env-skip is separate and worse: a science gate that never
executes is not a gate.

## 2. Tolerances with no stated basis

| test | asserts | why it is weak |
|---|---|---|
| ~~`test_makov_payne.py::test_q_plus_one_vacuum_L15`~~ | ~~`1.35 < dE < 1.40`~~ | **CLOSED 2026-09-09 — and the original finding was the wrong call.** It said to assert against the imported constants at `rel=1e-9`. Tried, and it is CIRCULAR: the expected value is built from the same constants the code uses, so a Madelung constant 1% wrong moves both sides and the test stays green (measured). And the deeper objection, from the user: **this is fixed algebra over three constants feeding a warning printed to two decimals** — a tighter pin buys nothing, and the derivation in the comment said the same thing the window said. The comment is gone; one scale assertion remains. |
| ~~`…::test_script_runs_on_synthetic_out`~~ | ~~`-122.5 < corrected < -121.5`~~ | **CLOSED 2026-09-09 — this one was never a tolerance.** The window's real job is the **SIGN**: added gives −122.09, subtracted gives −124.82, and subtracted is the defect that shipped until 2026-07 (the energy moved the wrong way by 2·ΔE). Filing it as a loose tolerance is what made it read as sloppy. It now asserts the direction outright, and that the script's own three printed numbers add up. Three mutants killed: the shipped script subtracting, printing a ΔE it did not apply, and a silently zeroed correction. |
| `test_backends.py::test_threedna_a_form_differs_from_b_form` | `diff.max() > 0.1` Å | A- and B-DNA differ by ~2–3 Å in rise/diameter over a 4-mer, so **0.1 Å also passes for two B-form builds differing by numerical noise.** Derive from the known rise difference (2.56 vs 3.38 Å per base pair). |
| ~~`test_add_slab.py::test_the_same_holds_on_the_two_period_surfaces`~~ | ~~`moved > 0.1`~~ | **CLOSED 2026-09-09, and this row's premise was wrong.** It asked for the steps to be derived (`a/2`, `a/(2√2)`). They are ASE's crystallography, and this layer computes no distance — it asks ASE for a taller slab and keeps a WINDOW of it. So the expectation is now MEASURED from the superset ASE builds, and the mechanism itself is covered structurally and element-agnostically by `test_the_slab_is_the_contiguous_window_ASE_built`. (For the record: (100) is `a/2` and (110) is `a√6/4`, not `a/(2√2)` — but neither belongs in a molbuilder test.) |
| `test_cell.py` — `detect_layers` | `abs=1e-3` | `junction-cell.md` states the median rule but no tolerance. |
| `test_cell.py` — `bulk_z_period` | `pytest.approx` defaults | Same. |
| ~~`test_modify.py::test_electrode_lattice_constant_override`~~ | ~~`e_extent > d_extent + 0.5`~~ | **CLOSED 2026-09-09, after one wrong turn.** This row said to assert the ratio; the ratio is `5.0/a`, which is ASE's LINEARITY, not ours. `add_slab` forwards `a` to `builder(element, size=, a=)` and computes nothing, so the only property it owns is that the value is FORWARDED. The expectation now comes from calling `_build_ase_slab` directly at each lattice. A dropped override, an override 2% off, and a WRONG DEFAULT are all killed — the last by neither earlier version. |
| `…::test_junction_stepped_contacts_via_two_calls` | atom counts only | Hard-codes "ASE's fcc(111) Au inter-layer spacing ≈2.355 Å" into its arithmetic and never asserts it. If ASE's lattice constant moved, the stacks would overlap and the test would still pass. Measure `a/√3` from the built geometry. |
| `…::test_add_atom_zero_offset_is_advisory_not_blocked` | a finding is raised | Neither the threshold nor the value. A validator flagging every pair at any separation passes. |
| — spectra | `n_h >= 25` for ARNDC | A floor that **over**-addition passes, when over/under-deletion is the measured failure mode. |
| `test_periodicity_gate.py::test_translating_the_whole_molecule_keeps_the_box_with_it` | `span_after > span_before + 15.0` | Exactly computable (the translation is a known +20 Å); asserted as an inequality. **Filed under "spectra" until 2026-09-09 and therefore unfindable** — `grep -rn span_after tests/` returns `test_periodicity_gate.py:756` and nothing else. |

## 3. Assertions that cannot fail, or fail for the wrong reason

- ~~**`test_smiles_and_siesta.py::test_cell_volume_just_above_threshold_passes`**~~ — **CLOSED 2026-09-09.** Re-measured and confirmed: the fixture put 10 atoms in a 2.29 Å box, so `render_fdf` refused with `cell.unfittable` and the final assert was never reached. All three volume tests collapsed into one parametrised boundary over the gate molbuilder owns (`vol < n * 1.0`), plus a render-through and the message test. The exact-boundary row needed `diag(v,1,1)` rather than a cubic cell: `10.0**(1/3)` cubes back to `10.000000000000002`, two ULP above, which let the `<` → `<=` mutant survive the row written to catch it. Four mutants killed. *Original finding:*
  measured: `render_fdf` raises *"The molecule is longer than the cell along a,
  b, c"*, so the `except ValueError: … return` branch is taken and the final
  `assert "BlockSize" in fdf` is **never reached.** It proves only *"the volume
  gate was not the refusal"*, never that a 1.2× cell renders.
- ~~**`test_pyscf.py::test_python_api_ecp_none_sentinel_disables_ecp`**~~ — **CLOSED 2026-09-09.** The four fixed-width literals were strictly subsumed by the `^\s*ecp\s+=` line match beside them — and were the same literals the test's own comment named as the original defect, left in place when the regex replaced them. Removed. *Original finding:* four of
  five assertions are vacuous *by its own docstring* (fixed-width literals the
  script never contains). Only the `re.search(r'^\s*ecp\s+=')` line checks
  anything.
- ~~**`test_a_charged_decks_promised_script_ships_with_it`**~~ — **CLOSED 2026-09-09.** The premise is asserted now, not assumed. And sharpened past the original finding: matching the bare filename still passed with the run line renamed, because the header names the script in prose one line above. It now reads the promised filename OUT of the deck (`python3 (\S+\.py)`) and checks that exact file — two mutants killed. *Original finding:* its only assertion
  is inside `if "makov_payne_correction.py" in deck:`. **Reword the header and
  this net-charge test silently asserts nothing.**
- ~~**`test_makov_payne.py::test_script_header_warns_about_slab_crystal`**~~ — **CLOSED 2026-09-09, by removal.** Confirmed: `render_correction_script(system_label, q, epsilon_r)` never receives the structure, so it cannot condition on slab-ness even in principle; the decision molbuilder makes is `emit iff q != 0`, already pinned by `test_charged_writes_script` / `test_neutral_skips_script`. The caveat is documentation and a keyword probe never protected it. **Now unchecked, deliberately** — recorded at the removal site. *Original finding:*
  measures keyword presence in generated prose. **A header stating the opposite
  — that Makov-Payne is right for a slab — passes.** Makov-Payne is the *wrong*
  correction for a charged slab, so this is an applicability domain checked as
  vocabulary.
- ~~**`spectra/test_parsers_json.py::test_negative_zero_round_trip`**~~ — **CLOSED 2026-09-09, by removal.** Field survival is covered by the general round trip (`equilibrium_scf_eh = -76.4123`), and NaN/±Infinity have their own tests. Measured aside: `-0.0` *does* survive JSON with its sign bit — the test disclaimed the one thing that was true and asserted the one thing that could not fail. *Original finding:* asserts
  `loaded.equilibrium_scf_eh == 0.0` after writing `-0.0`. True by IEEE-754 for
  any value comparing equal to zero.
- ~~**`spectra/test_motion_share.py::test_shares_account_for_all_of_the_motion`**~~ — **CLOSED 2026-09-09.** Confirmed arithmetic (`w / total` where `total` is that sum). Replaced by the two things a caller can rely on: the element set, and the largest-first ORDER the docstring promises and the panel renders in — which nothing tested. Two mutants killed, including one that made the ordering alphabetical. *Original finding:*
  `sum(share.values()) == 1.0` is true by construction (`results.py:425` divides
  by the total). Only its key-set half can fail.
- ~~**`spectra/test_spectrumchart_maths.py::test_it_does_not_depend_on_where_the_modes_are`**~~ — **CLOSED 2026-09-09, by removal.** `bandHalfWidth(w)` takes only `w`, so mode positions cannot enter it — a fact about the SIGNATURE, read in one line. And the test could not have caught a reintroduced clamp anyway: `band(20) == 20` would still hold. *Original finding:*
  `band(20) == 20`, character-identical to its sibling; `band()` takes only `w`,
  so the test **cannot express** its stated claim.
- ~~**`test_modify.py::test_rotate_around_z_default_no_op`**~~ — **CLOSED
  2026-09-09.** Replaced by `test_a_rotation_is_rigid_and_proper`, parametrised
  over three axes × five angles: every interatomic distance survives (rigid) and
  the signed tetrahedron volume survives (proper, det = +1). `angle=0` remains as
  one row, where it now means something because the other four can fail. Four
  mutants killed — degrees read as radians, a reflection in place of the z
  rotation, a 1% scale, and a sign flip. The non-coplanar fixture is what makes
  a reflection visible, for the same reason as the antiparallel case.
- ~~**`test_fcc_lattice_table.py`**~~ — **CLOSED 2026-09-09.** Confirmed: `load_fcc_lattice_full` rebuilds each entry with four fixed keys, so the two `not in e` assertions against loader output could not fail. Removed; the raw-JSON guard stays and is marked as the only place a re-add is visible. Verified by re-adding the column to `fcc_lattice.json`: exactly one test fails. *Original finding:* three tests carry `assert "a_pbe_siesta_psml"
  not in e`, which cannot fail: `load_fcc_lattice_full` rebuilds each entry with
  four fixed keys. Only `test_every_metal_has_the_two_literature_references`,
  which reads the raw JSON, can catch a re-add.
- ~~**`test_junction_sidecar_roundtrip.py::test_the_junction_is_shaped_the_way_the_test_claims`**~~
  — **WITHDRAWN 2026-09-09: THIS FINDING WAS WRONG.** It asked for the fixture's
  "outermost layer is frozen" claim to be checked. That was done, from z — and
  the user's objection retired it: **`frozen_atoms` is the USER'S choice.**
  molbuilder never derives it; `structure.FROZEN_LABEL` is a reserved label and
  every reference in the package reads, converts or writes it. Asserting which
  atoms are frozen pins a decision this tool has no opinion about — the same
  fault as asserting ASE's crystallography in `add_slab`'s tests, which §§ 2
  above also had to withdraw. **We are designing a tool: it carries the user's
  setup, it does not grade it.** The file's subject is that whatever they chose
  survives write → read → write, for any set; the count assertion stays only
  because the round trip needs a non-empty list to lose.

## 4. The claim is not where the test looks

> **Worked through 2026-09-09: three closed, four WITHDRAWN.** More than half
> of this section was wrong, and in one recurring way — **a suite-wide absence
> asserted from one file's contents.** The `in_progress` family and both
> GPU-routing findings each said "nothing anywhere asserts X" about an X that
> another file asserts directly; the rank-agreement finding asked for a rule
> the user had deleted as unscientific six days earlier; the netCDF one asked a
> test to validate a third-party FORMAT across a byte-for-byte copy. This is
> the same error the mutation campaign measured in the audit proper (nine of
> nineteen verdicts wrong), wearing different clothes: there, a verdict was
> issued against the whole suite instead of the named coverer; here, an absence
> was inferred from a single file. **Re-measure before acting** is the rule
> either way.


- ~~**`test_render_siesta_emits_propor_diagnostic`**~~ — **CLOSED 2026-09-09.**
  Confirmed by measuring the rendered wrapper (pseudo at 40588, np at 41070,
  spin at 41474). The three cause markers are now read with `.index` and their
  positions asserted sorted, so presence and sequence are one assertion.
  Mutation: swapping cause blocks 1 and 2 in `runwrap.py` — the exact
  pre-2026-06-26 framing — now fails, and passed before. *Original finding:*
- ~~**`test_the_cap_is_clean_scf_must_converge_is_pinned_off`**~~ — **CLOSED
  2026-09-09.** The deck is asserted now (`SCF.MustConverge .false.`), and the
  measurement sharpened the point: with the pin absent the emitter writes **no
  keyword at all**, so SIESTA falls back to its own must-converge default —
  the failure is silent, and only the deck can show it. Two mutants killed;
  the second (dropping the field from `siesta/layout.py`) is one the pins-only
  test could not see, since the pin can be correct and still not reach SIESTA.
  *Original finding:*
- ~~**The rank-agreement family (jobset)**~~ — **WITHDRAWN 2026-09-09: THIS
  FINDING WAS WRONG, twice.**

  It asked for "ranks must not exceed what the system size supports" to be
  checked. **That rule was deliberately deleted by user ruling on 2026-09-03**,
  and `runwrap._orbitals_per_rank_notice`'s docstring records why: the wrapper
  used to clamp an auto rank count to `n_atoms` and warn above it, citing the
  `propor IMAX=0` abort — but that abort comes from a PSML problem, not from
  system size, so the clamp *"helped by accident on the systems where it fired
  and refused perfectly good rank counts on the others."* Asking a test to
  enforce it is asking the tool to grade a choice it has no basis to grade —
  the same fault as the frozen-atoms finding below.

  What molbuilder DOES decide here is a NOTICE, never a limit: `n_orbitals /
  mpi_np` with both numbers shown, so the claim is checkable rather than a
  verdict. That is tested (`test_runwrap.py`, the message and the empty case
  when the deck states no atom count).

  And the equality it dismisses is not nothing: `mpi_np` in the deck and
  `mpi_np` at launch are two numbers **we** write in two places, and their
  agreement is ours to keep — the same arrangement as
  `test_contact_distance_reference.py`'s JSON-vs-JS copy, which this same audit
  approved.
- ~~**Both GPU-routing tests**~~ — **WITHDRAWN 2026-09-09: THE CLAIM IS FALSE,
  and no test matches the description.** Nothing in the tree asserts a boolean
  `"gpu: yes/no"`. What the routing tests assert is exactly the agreement this
  finding says is missing:

  * `test_runwrap_pair.py::test_the_pair_agrees_about_the_gpu_when_one_is_asked_for`
    puts `Diag.ELPA.GPU .true.` in the deck and asserts the launcher activates
    `molbuilder-siesta-gpu` **and** the sbatch carries `--gres=gpu:a100:1`;
  * its sibling asserts the other direction, so it cannot pass by never
    emitting a GPU header at all;
  * `test_wrapper_preamble_preflight.py` goes further — a GPU deck on a machine
    whose record lacks the GPU env is REFUSED at prep, with the machine named.

  Deck-vs-environment solver agreement is the subject of all three. Recorded
  rather than deleted because the pattern matters: this is the second § 4
  finding asserting an absence that a file the auditor did not open contains.
- ~~**`test_in_progress_frames_stay_out_of_plots.py` (×3)**~~ — **WITHDRAWN
  2026-09-09: THE CLAIM IS FALSE.** It says nothing asserts that a real
  mid-write `.out` produces `in_progress=True`. `test_siesta_in_progress_first_scf.py`
  does exactly that, four times, feeding real `.out` text through
  `SiestaParser` — `test_fresh_first_scf_emits_in_progress_frame` asserts the
  True case and `test_clean_completed_run_does_not_emit_in_progress_frame` the
  False one. **From the file onward**, in a file this finding did not open.

  The three tests it names are the OTHER half and are right to hand-build the
  flag: they test the ADAPTER, and the flag is that layer's input. The contact
  point between the halves is one boolean on one dataclass (`Frame.in_progress`),
  with nothing for the two sides to disagree about — unlike `#78`'s
  sidecar↔selection seam, where the halves used different identity keys and a
  join test earned its place.

  **The auditing error is worth keeping**: a suite-wide absence was asserted
  from one file's contents. That is the mirror of the rule the mutation
  campaign settled — a verdict is against the NAMED coverer, never the whole
  suite — and it is how several of the nine wrong verdicts happened.
- ~~**`test_a_continue_carries_the_accumulative_records_too`**~~ — **WITHDRAWN
  2026-09-09.** The carry is `shutil.copy2` — a byte-for-byte binary copy — so
  a real `.MD.nc` and a twenty-byte stand-in are handled identically and the
  format is not a variable molbuilder controls. Asking the test to show netCDF
  stays appendable after a byte-identical copy is asking it to validate the
  FORMAT, which is the same fault as asserting ASE's crystallography or the
  user's frozen set. And the failure the docstring names — a truncated record
  silently shortening a continued stage — is exactly what byte equality rules
  out, so the existing assertion is the right one.
- ~~**`test_atom_identity_end_to_end.py`'s PySCF half**~~ — **CLOSED
  2026-09-09.** Both halves fixed. `_distinct_struct` now carries REPEATED
  elements (`C C N C O`), so an index shift lands on a different carbon and
  stays chemically plausible — the element half of each assertion is doing work
  instead of decorating a fixture that could only fail on coordinates. And
  `_ATOM` takes any decimal width, so a coordinate-format change fails as a
  named parse miss rather than as *"expected 5 atom lines, got 0"*. Off-by-one
  in `engine_atom_index` still dies.

  Note these two tests are NOT the frozen-atoms fault withdrawn in § 3: they
  assert an INDEX MAPPING (0-based internal → SIESTA/geomeTRIC 1-based), which
  is entirely molbuilder's, not which atoms the user chose to freeze.

  *Original finding:*

## 5. Constants and references retyped instead of imported

A test that hard-codes a physical constant is measuring **two** constants and
cannot see them drift apart.

| where | retyped |
|---|---|
| `tests/test_parsers_siesta_struct.py` | `_BOHR = 0.5291772108` — and `test_one_home_for_a_constant.py` scans only `molbuilder/`, so `tests/` is outside its net |
| `tests/parse/test_coords.py::test_xv_cell_round_trip_against_synthetic_file` | `ang_per_bohr = 0.529177249`, tolerance `1e-3` unexplained |
| two engine test files | `HA_BOHR_TO_EV_ANG = 51.42208619`, `_HARTREE_TO_EV = 27.211386245988` |
| `test_add_slab.py::test_a_and_b_are_still_the_crystals_own_vectors` | `4.078` — against the module's own warning at line 32 that *"a test that retypes 4.078 and compares against it is measuring the two constants"*; it also re-derives its expectation with the same door the implementation uses |
| `test_cell.py::TestOneThreshold::test_the_constant_is_shared_not_copied` | pins `ZERO_VOLUME_TOL` **by source text** in two modules — passes if the name survives in a comment beside a restored literal, and fails on a correct rename |

**Citation fragility:** `test_lattice_from_run.py::test_it_compares_against_the_literature_and_says_by_how_much`
asserts the PBE reference 4.158 Å only through the substring `"-1.9%"` — **a
Unicode minus fails while the physics is right.** Its sibling
`test_the_second_shell_mistake_reads_as_a_big_offset` uses a three-way
alternation `"+38" or "+39" or "+41"` standing in for the √2 factor → +41.4%,
which should be derived.

## 6. Coverage gaps found while auditing — not test defects

- **`docs/model/overview.md:122` cites `test_frontend_display_matches_engine_atom_number`
  as binding the atom-index invariant. That test does not exist anywhere in
  `tests/`.** The front-end half of a protected invariant is unbound.
- **`test_structure_save_endpoint.py`'s docstring lists three pinned properties;
  property 1** (*"a BROWSER-shaped payload … lands on disk as a VALID pair the
  load door reads back"*) **has no test in the file.**
- **`spectra/test_selection.py::test_prior_without_es_does_nothing`** promises
  two cases and tests one. Selection decides **which modes get an ES
  calculation**, so the missing half is worth adding.
- **`TestOpsPreservePeriodicity`**'s class docstring claims the k-grid is
  covered; `_assert_lattice_preserved` checks only cell, axis_kind and vacuum —
  the k-grid moved to `SiestaConfig`.
- **`test_isolated_axis_keeps_pbc_false_despite_cell`** pins the flag, not the
  condition: two H at 0 and 0.74 Å in a 10 Å box, with nothing checking the
  vacuum makes "isolated" mean anything.
- **`modify.py::_finish_slab` carries a stale comment** — the pre-2026-08-31
  *"z is the atoms' extent PLUS ONE INTERLAYER SPACING"* sits directly above
  code that now sets `z_len = z_extent`. The test is right; a reader of the
  source gets the retired rule.
- **`test_annotations_fdf.py`** registers strategies into a process-global
  registry and never removes them. `test_render_fdf_unchanged_without_annotations`
  is meaningful *because* of that leakage — a real property, currently
  accidental.

## 7. Redundancy inside protected files — recorded, not acted on

Kept because the class is protected; listed so the redundancy is visible if the
file is ever redesigned.

- `test_cell.py` — `test_a_zero_volume_box_is_not_also_uncontained` is fully
  implied by `test_a_zero_volume_box_is_not_also_left_handed`;
  `test_a_box_with_no_volume_is_an_error` by that pair plus
  `test_generating_refuses`; `test_every_notice_carries_its_id` has a vacuous
  loop body if no notice is emitted.
- `test_atom_metadata_results_bridge.py::test_no_block_returns_none` is a
  byte-for-byte duplicate of the second half of `test_empty_block_returns_none`
  — whose **name is wrong**: it never writes an empty block.
- `spectra/test_parsers_json.py` — 86 tests over a door with six decision
  branches. ~15 land on one `except (TypeError, ValueError, AttributeError)`;
  another 7 assert that CPython's JSON lexer rejects non-JSON. **The four
  carrying actual physics** — free∩frozen overlap, `len(free)+len(frozen) ==
  n_atoms_total`, `homo_idx` range, cross-mode ES-window uniformity — **are
  buried among them** and deserve to be visibly separated.
- `test_siesta_runtime_info_build.py` — 12 of 19 are one-key assertions over two
  fixture headers. A table-driven form collapses 12 into 2 with no loss.
- `test_modify.py::test_passthrough_op_carries_annotations_verbatim` tests
  through a forwarder (`modify.translate` with no `indices` is
  `struct.translated(vec)`). The version that would earn its place is the
  `indices=` branch, where annotation carriage could actually be lost.

## 7a. From the 2026-09-09 header pass — nine more, one of them measured

Recorded under the same rule as everything above: **none is a proposal to
delete.** Companion record for the non-science half:
[`process/test-audit-findings.md § 6`](?doc=process/test-audit-findings.md).

### The one that was measured, and it is the sharpest in this file — **FIXED 2026-09-09**

**`test_modify.py::test_orient_handles_antiparallel_case` — an INVERSION passes.**
The antiparallel branch of `_rotation_matrix_from_a_to_b` (`modify.py:267`) is
the singular case, `cross(a, -a) == 0`. Replacing its `2·nnᵀ − I` (a proper
rotation, `det = +1`) with `-np.eye(3)` (an inversion, `det = −1`) was run on
2026-09-09:

- the named test passes,
- **all 101 tests in `test_modify.py` pass**,
- and **all 320 tests in every file that touches `orient` pass** — including
  `tests/validation/test_geometry.py`.

**An inversion flips the chirality of any real molecule.** The shipped code is
correct; nothing in the suite would notice if it stopped being. The cause is the
fixture: a **two-atom** structure, in which a reflection and a rotation are
indistinguishable.

> **CLOSED 2026-09-09.** The fixture is now four atoms, deliberately
> **non-coplanar** — three would still span a plane, and a reflection through
> that plane is undetectable. The test asserts the **signed** triple product is
> unchanged, which is the only quantity that separates a rotation from an
> improper transform: distances, angles and the pair's final direction are all
> invariant under both. An inversion, a reflection through xy, and a reflection
> in the axis plane are each killed.

**And the second half of that gap is closed too.** `science/test-design-findings.md`
recorded that *nothing joins the sidecar to the selection evaluator* — `ByRegion`
lived only in the selection tests, `apply_to_structure` in eight other files, and
neither met. `tests/test_atom_identity_end_to_end.py` now walks the whole path
through the real doors: labels → `metadata_to_dict` → `save` → `load` →
`apply_to_structure` → `evaluate(ByRegion(...))`, asserting on **element and
position, never on the index** — an index that round-trips while pointing at a
different atom is the entire failure. The fixture carries **repeated elements**,
which the audit had flagged as missing, so an index shift lands on a different
carbon and stays chemically plausible. An off-by-one on read, bleeding labels
and empty labels are each killed.

### Assertions weaker than they read

- **`test_modify.py:434::test_delete_preserves_metadata_in_lockstep` asserts
  LENGTH, not contents.** Four `len(...) == out.n_atoms` checks. `delete_atoms`
  builds six comprehensions over one `keep` set, and a copy-paste error in one
  of them yields the right length and the wrong contents — every atom after the
  deletion point wearing its neighbour's name and residue. That is an
  atom-identity error that renders as a valid structure and reaches both
  emitters. *Fix:* assert the survivors equal `[orig[i] for i in keep]`.
- **`test_atom_selection.py:342::test_not_is_complement` asserts a relation, not
  the atoms.** It compares `evaluate(Not(au))` against
  `evaluate(Minus(All(), au))` — but in `_evaluate` both are literally
  `frozenset(range(n)) - _evaluate(operand)`, so a mutation breaking `n` breaks
  both and stays green. It is `Not`'s **only** evaluation test. *Fix:* assert the
  named set, and keep the equivalence as a second line.
- **`test_atom_selection.py:370::test_first_n_more_than_available` asserts
  Python's slice semantics.** `ordered[:99]` is a language guarantee. The
  property it means to protect is a design decision — a saved rule whose
  structure has since shrunk degrades rather than refusing — which would be
  observable by evaluating a rule written for 11 atoms against a structure of 5.
- **`test_transport_prep.py:337::test_the_transmission_deck_carries_the_tbt_window`
  checks presence, not the window.** `"TS.TBT.NumE" in text and "TS.TBT.Emin" in
  text` is satisfied by any values, including a range entirely above the Fermi
  level — and a window that does not bracket `E_F` is the failure that matters
  for `T(E)`. This one line is an outlier in an unusually strong file (which
  elsewhere asserts species columns, `elec-pos` offsets, `kz = 1`, and the
  `.TSHS` stem two writers share). *Fix:* derive `Emin`/`Emax` from the cited
  deck's `E_F` and assert the sign relation.

### Gaps, not weaknesses

- ~~**`spectra/test_parsers_json.py:1063` pins the impossible `homo_idx`.**~~
  **CLOSED 2026-09-09, and the read side was never where it could be fixed.**
  It refused `homo_idx = 99` against a 5-orbital array. In-range-but-wrong is
  the real failure: `web/spectra.md` § 3.1 has the level diagram, the gap and
  the gap SHIFT all reading from this index, at shifts of ~0.018 meV, so an
  off-by-one looks like a different answer rather than an error.

  **But the sidecar carries `homo_idx` WITHOUT the occupations it came from**,
  and `SpectraResults` holds only `equilibrium_mo_energies_eh` — no occupancies,
  no electron count, no elements. So the read side cannot cross-check it, and
  `results.py:642`'s range test is the most it can do. This finding asked the
  wrong layer.

  **Where it lives is the emitter, and it is OURS**: PySCF reports occupation
  numbers and molbuilder derives the index. It existed only as text inside the
  generated script, where nothing could call it — **and it has a branch**: 1-D
  `mo_occ` for RHF/RKS, 2-D `(alpha, beta)` for UHF/UKS, which must be summed.
  Miss the sum and every OPEN-SHELL calculation reports one spin channel's HOMO.
  It is now `vibration_emitters.homo_index`, spliced into the script from its own
  source so there is one implementation, with nine cases and five mutants —
  the dropped sum, the wrong axis, `min` for `max`, the threshold at 0, and the
  empty-reference refusal.

- **Nothing joins the sidecar to the selection evaluator.** `ByRegion` appears
  only in `test_atom_selection.py`; `apply_to_structure` appears in eight other
  files but never with a rule evaluation. So the end-to-end path that decides
  *which atoms are computed* — labels written to `.molstruct.json`, applied to a
  `Structure`, then a `ByRegion` rule re-selecting exactly those atoms — is
  covered as two halves that never meet. `model/overview.md` § 2 names index
  translation as where an off-by-one happens, and both halves use hand-built
  0-based fixtures, so a shift introduced between them is invisible. **A test to
  write.**
- **`TestOpsPreservePeriodicity` is asymmetric in the dangerous direction.** The
  two rotation tests assert only `axis_kind` and `vacuum` — correctly skipping
  the lattice check, since rotation legitimately rotates the vectors — but
  nothing in that class then asserts anything about the rotated cell, and the
  check that does lives in another class with its own two-atom fixture. So for
  the fixture carrying a real `("periodic","periodic","transport")` cell, a
  rotation could return `cell=None` and both tests pass. *Fix:* assert
  `out.cell is not None` and that `det(out.cell)` is unchanged — the volume is
  invariant under any rotation and is the cheapest statement of it.

### The sidecar pairing — protected, and two notes

- **The fixture hand-builds the artifact, two schema versions stale.**
  `test_web_files.py:1596 _seed_paired` writes `{"schema_version": 7, …}` by
  hand while `sidecars/molstruct.py:92` is `SCHEMA_VERSION = 9` (2026-08-29).
  The artifact under test is not the artifact the codec writes, and the existing
  door — `molstruct`'s own writer — is bypassed. Nothing fails today because the
  file operations only move bytes, which is exactly why it can drift unnoticed.
- **The pairing tests assert file presence, not scientific survival.** After
  rename/move/copy the strongest assertion is
  `json.loads(new_sidecar)["n_atoms_total"] == 3`. **`regions` and
  `frozen_atoms` — the fields whose silent loss is the harm the pairing exists
  to prevent — are never read back**, and no pairing test re-opens the moved pair
  through `StructureCodec.read`. A future coherence check that a rename
  invalidated would leave all of them green. *Fix:* one line of the existing
  door — load the moved pair and assert the labels come back.

---

## 8. Two tests worth copying

- **`test_runwrap_retry.py::TestMarkerGroundTruth`** pins the wrapper's grep
  strings against **real frozen SIESTA output**, so wording drift fails there
  before it silently disables SCF-abort detection.
- **`test_siesta_keyword_smoke.py::test_the_kgrid_displacement_reaches_siesta_and_changes_the_sampling`**
  asserts **44 vs 32 irreducible k-points** — the physical consequence, not the
  echoed keyword.

Runner-up, for a different reason: `test_contact_distance_reference.py::test_both_are_non_empty_and_keyed_by_the_PAIR`
states the physical reason in the test (Pt–N 2.05 vs Pt–S 2.30 are different
bonds) *and* guards its sibling from passing on two empty dicts.

---

## Appendix — the audit's own error rate

One CUT-SUBSUMED claim from the same audit **was** mutation-tested, and it was
**wrong**: `test_a_capped_benchmark_reads_ended_through_BOTH_doors` looked
redundant, but no frozen fixture carries both `SCF_NOT_CONV` and `>> End of
run`. That is the error rate to expect from unverified subsumption reasoning,
and the reason nothing in this file should be acted on without re-deriving it
first.

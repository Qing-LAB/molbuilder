# Region-label convention for transport

This doc codifies how molbuilder names + interprets the per-atom
region labels that drive the TranSIESTA emitter.  The convention
landed with the 2026-06-18 modernization commit; before that the
emitter hardcoded a closed 2-terminal topology with literal
`L-electrode` / `R-electrode` / `bridge` strings.

The convention is shared across three surfaces:

* The **modify tab UI** lets users assign labels to atoms (free-
  text input with the charset `^[A-Za-z][A-Za-z0-9_\-]*$`).
* The **molbuilder dataclass** (`Structure.regions`, a
  `Dict[str, List[int]]`) carries the labels in-memory.
* The **TranSIESTA emitter** discovers electrode regions from
  `struct.regions` and emits one `%block TS.Elec.<name>` per
  electrode.

## 1. The convention in one line

> **Any region whose label ends with `-electrode`, `_electrode`,
> or bare `electrode` (case-insensitive) is treated as a TranSIESTA
> lead.**

The defaults the modify tab ships — `L-electrode` and
`R-electrode` — fit this convention.  Users who need extra leads
(STM-tip junctions, multi-terminal devices, gate electrodes) add
regions with the same `-electrode` suffix and the emitter picks
them up without code changes.

The Python helper that pins this is
`molbuilder.config.transport.is_electrode_label(label)`.

The JS mirror is
`window.molbuilder.regionLabelDefinitions.isElectrodeLabel(label)`
(in `lib/region-label-definitions.js`).  The two are pinned to
agree by `tests/test_region_label_definitions_js.py::
test_electrode_convention_helper_is_consistent_with_python`.

## 2. The canonical labels

What's currently used + interpreted:

| Label | Role | Atoms that belong here | Common practice |
|---|---|---|---|
| `L-electrode` | Left semi-infinite lead (periodic bulk) | The slice of lead metal that SIESTA will replicate as a semi-infinite bulk lead. | 3-4 atomic layers, deep enough that the electronic structure matches bulk at the boundary.  Use only the BULK portion (the part SIESTA will replicate); surface terminations / dangling-H caps go in `bridge`.  The lattice along the transport direction (z) MUST match the bulk lead's periodicity. |
| `R-electrode` | Right semi-infinite lead | Mirror of L-electrode. | Same metal + orientation as L for the canonical 2-terminal case.  Asymmetric leads are an advanced case. |
| `bridge` | Scattering region | The molecule(s) under test + any lead-side atoms that break periodicity (surface terminations, tip atoms, chemisorbed contacts). | For Au-BDT-Au: the entire benzene-1,4-dithiol molecule INCLUDING the two S anchors.  Surface layers below the molecule but above the periodic bulk go here too. |
| `interface` (optional) | Contact atoms (sub-label) | A SUB-label that flags atoms still chemically inside `bridge` but participating in the metal-molecule bond.  Useful for projected-DOS and charge-transfer analysis; does NOT change the TranSIESTA partition. | The 2 S anchors in Au-BDT-Au.  The N atoms in Au-PDA-Au. |
| `<name>-electrode` | Additional electrode (multi-terminal / asymmetric) | Any region whose label ends with `-electrode`.  The label stem (before the suffix) becomes the SIESTA block name. | Use for STM-tip leads (`tip-electrode`), 3-terminal devices (`gate-electrode`), or asymmetric junctions where `L`/`R` doesn't capture the topology. |

## 3. References

* **Brandbyge et al., Phys. Rev. B 65, 165401 (2002)** — the
  canonical TranSIESTA paper.  §III defines the L/R/scattering
  partition; §IV covers the NEGF contour.
* **Stokbro et al., Comp. Mat. Sci. 27, 151 (2003)** —
  Au-BDT-Au geometry + the 350 Ry MeshCutoff convention for
  Au surfaces.
* **Reed et al., JACS 128, 14328 (2006)** — asymmetric STM-style
  junctions; canonical reference for the `interface` sub-label.
* **Solomon et al., J. Chem. Phys. 129, 054701 (2008)** —
  projected-DOS analysis at metal-molecule contacts; canonical
  reference for the `interface` convention.

## 4. The emitter's behavior

Given `struct.regions`, the TranSIESTA emitter
(`molbuilder/transport/transiesta.py::_emit_transiesta_block`):

1. Calls `_find_electrode_regions(struct)` which iterates
   `struct.regions.items()`, filters with `is_electrode_label`,
   and sorts the matches by atom z-centroid (lowest first).
2. The leftmost electrode (lowest z) gets the SIESTA chempot
   `Left` and `semi-inf-direction -A3`; the rightmost gets
   `Right` and `+A3`.  These are CONVENTIONAL SIESTA names — they
   don't need to match the user's region labels.
3. Each electrode emits a `%block TS.Elec.<sanitized-block-name>`
   block, where the block name is the label with the
   `-electrode` suffix stripped.  Examples:
   * `L-electrode` → `L`
   * `tip-electrode` → `tip`
   * `electrode` (bare) → `electrode`
4. The TBtrans block + the `TS.Voltage` value + the
   `%block TS.ChemPots` + per-chempot `%block TS.ChemPot.<name>`
   land after the per-electrode blocks per the modern SIESTA
   4.1+ / 5.x syntax (verified against SIESTA 5.4.2 binary; see
   `tests/test_transiesta_siesta_smoke_l4.py`).

The `bridge` region is NOT a TranSIESTA block — it's implicit
("the atoms that aren't in any electrode region").  The
preflight cross-checks that L + bridge + R atoms are contiguous
in the AtomicCoordinates block order, because TranSIESTA
identifies the electrode atoms by their POSITION in the
coordinates list (first N atoms = first electrode, etc.), not
by region label.  An out-of-order structure produces silently
wrong physics.

## 5. UI affordances

* The **modify tab's selection panel** has an ⓘ button next to
  the `Target:` dropdown.  Clicking it opens a popover with the
  scientific definitions of each canonical label (this doc's §2
  table) + a bias-direction reminder.  The popover highlights
  which labels are present in the current structure vs.
  "available default".  Source:
  `molbuilder/web/static/lib/region-label-definitions.js`.
* (Planned, Phase 2b) The **transport tab embedded viewer** will
  carry a per-region filter toolbar so the user can hide every
  region except `bridge` (or any single label) to visually verify
  the partition.  Tracked as task #501.

## 6. Bias direction (cross-reference)

The `Bias direction` paragraph in the popover restates the
convention the form's `Electrodes` section description carries:

> Bias is `V_left - V_right` in volts.  POSITIVE bias raises μ_L
> above μ_R; electrons flow from the higher chemical potential to
> the lower (L → R for positive V), conventional current flows
> R → L.  Pick the L electrode to be the more negative reservoir
> in your forward-bias measurement.

This is restated in the popover (rather than only on the form's
section description) so the user has the reminder visible at the
moment they're labeling atoms — which is when the bias-direction
choice happens, not when they later configure the form.

## 7. Tests that pin the contract

| Surface | Test |
|---|---|
| Convention (Python helper) | `tests/test_region_label_definitions_js.py::test_electrode_convention_helper_is_consistent_with_python` |
| Convention (JS helper) | same — checks both halves agree |
| Popover HTML + button | `tests/test_region_label_definitions_js.py::test_popover_html_landed_in_template` |
| JS module loads in modify.html | `test_modify_template_loads_the_js_module` |
| CSS styles present | `test_css_has_popover_styles` |
| Modern TranSIESTA syntax (end-to-end against SIESTA 5.4.2) | `tests/test_transiesta_siesta_smoke_l4.py` |
| Emitter's used-atoms count from region sizes | `tests/test_transport_au_bdt_au_validation.py::test_render_script_emits_correct_atom_counts` |

## 8. Open work

| Task | Status |
|---|---|
| Definitions popover (this doc + UI) | ✓ shipped (2026-06-18) |
| Transport-tab embedded viewer + label-filter toolbar | Planned (#501 + #502); requires mounting the viewer on the transport tab first |
| Multi-terminal UI (per-chempot mu form field) | Planned; the emitter scaffold accepts arbitrary electrode counts but the form is 2-terminal only today |
| Atom-ordering reorder affordance in modify tab | Planned; today out-of-order labels are caught at preflight and the user must re-export |

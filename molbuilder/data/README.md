# `molbuilder/data/` — fundamental physical / chemical data

This directory holds **fundamental data** that the molbuilder code reads at
import time but is otherwise treated as immutable input.  Keeping it here
rather than in the source modules has two purposes:

1. **Auditable provenance.**  Every value is cited in this README; reviewers
   can check the entry against the cited source without reading code.
2. **User-updatable.**  An end user can edit a file in this directory to
   override a value (for example, to use a strained-lattice constant or a
   non-room-temperature value) without touching molbuilder's Python source.

Each file in this directory is paired with a section below that lists every
value, its source, and any caveats.  When you add or change a value, update
this README in the same commit.

If you reinstall the molbuilder package, your in-place edits will be
overwritten by the package's pristine copy.  To make local overrides
durable, see *§ Local overrides* at the bottom of this file.

---

## `catalogue.template.toml` — the parameter catalogue

**The master list of every parameter both engines declare**, in the template
format ([`engines/template.md`](../../docs/engines/template.md)). 82 items:
45 apply to SIESTA, 40 to PySCF, 3 to both.

**It sits here because it is data, and because it is not one engine's.** The
per-engine `warm-files.toml` files live inside `siesta/` and `pyscf/`; this one
spans both, so it belongs in the shared directory beside the other tables a
reviewer can check without reading code.

**Why it exists.** The direction of flow is **template → per-engine config →
that engine's input file** ([`template.md`](../../docs/engines/template.md)
§ 2.1). A parameter is *defined* here; a config class only *carries* it on the
way out. Until 2026-08-14 the catalogue lived as metadata on the dataclass
fields and this file was generated *from* them — the inverted direction, which
meant enriching the catalogue required editing Python and two engines' items
could never share one file.

**Values here are DEFAULTS.** A calculation carries its own template with its
own values; this is what one starts from.

**Editing it.** It is TOML precisely so a person can. `read_template` refuses a
file that breaks § 3's required keys and names what is missing, so a bad edit
fails loudly at the next `prep` rather than silently producing a different
calculation.

---

## Files

| File | Schema | Read by | Status |
|---|---|---|---|
| `fcc_lattice.json` | v1 | `molbuilder.modify._load_fcc_lattice` | live |
| `catalogue.template.toml` | `molbuilder/template@2` | `molbuilder.template.read_template` | **new 2026-08-14 — the parameter catalogue** |

---

## `fcc_lattice.json`

Lattice constants of the FCC metals supported by the Modify-tab
electrode builder (`molbuilder.modify.add_electrode_slab`).  The list is
**closed** at six entries — only the canonical metal-electrode materials
used in single-molecule junction / NEGF transport DFT work.  Adding a
new entry here is intentionally a deliberate act, not a casual
convenience.

| Symbol | Element  | *a* / Å | *T* / K | Crystal system | Citation |
|--------|----------|---------|---------|----------------|----------|
| Au | Gold      | 4.0782 | 298 | FCC | Wyckoff 1963 vol. 1 p. 10 |
| Ag | Silver    | 4.0853 | 298 | FCC | Wyckoff 1963 vol. 1 p. 10 |
| Cu | Copper    | 3.6149 | 298 | FCC | Wyckoff 1963 vol. 1 p. 10 |
| Ni | Nickel    | 3.5240 | 298 | FCC | Wyckoff 1963 vol. 1 p. 10 |
| Pt | Platinum  | 3.9242 | 298 | FCC | Wyckoff 1963 vol. 1 p. 10 |
| Pd | Palladium | 3.8907 | 298 | FCC | Wyckoff 1963 vol. 1 p. 10 |

### Primary citation

> **R. W. G. Wyckoff**, *Crystal Structures*, 2nd edition,
> volume 1 (Wiley, New York, 1963), p. 10 — "The structures of the
> elements: Cubic close-packing, A1 type."

Wyckoff is the textbook source for room-temperature (~298 K, atmospheric
pressure) experimental lattice constants of the elemental metals;
research-code lattice tables (ASE, GULP, VASP examples, …) generally
trace back to it.

### Cross-checks

The values above were also verified against:

* **CRC Handbook of Chemistry and Physics**, 95th edition (CRC Press,
  Boca Raton, 2014), Table "Properties of the Elements" → "Lattice
  parameters of the elements."  Agreement is to within the last cited
  decimal in every case.
* **ASE's `ase.data.reference_states`** — same source (Wyckoff), but
  rounded to 3 decimal places.  Our values keep the 4th decimal where
  Wyckoff's table provides it.

### When to override

These are room-temperature *experimental* values.  Override (either by
editing the JSON or by passing `lattice_constant=` to
`add_electrode_slab` per call) when:

* You want a strained-lattice slab (epitaxial growth on a different
  substrate).
* You want a non-room-temperature value (low-T transport experiments).
* You want the *DFT-equilibrium* lattice constant for your specific
  XC functional.  Plain LDA underbinds gold by ~3 %, GGA-PBE overbinds
  by ~1 %; if you optimise the slab cell self-consistently in your
  production run, use that value here for consistency between the
  initial geometry and the relaxed one.

The JSON `_units.temperature_K` field carries the temperature at which
the published values were measured; if you replace a value with a
DFT-equilibrium one, change `_units.temperature_K` to `null` (or to the
temperature your DFT calculation targets) and update this README.

### Schema (v1)

```json
{
    "_format": "molbuilder.data.fcc_lattice v1",
    "_units":  { "a": "angstrom", "temperature_K": 298 },
    "_source": "<one-line citation; full chain in this README>",
    "metals":  {
        "<symbol>": { "a": <float>, "system": "fcc", "name": "<full name>" }
    }
}
```

Adding a new entry: add a `"<symbol>": {"a": ..., "system": "fcc",
"name": "..."}` row, append a citation line in the table above, and add
a paragraph explaining why this entry was added (typical use-case,
literature reference, anything readers should know).

---

## Local overrides

If you want your edits to survive `pip install --upgrade molbuilder`,
export an environment variable pointing at a directory that mirrors
this layout:

```bash
export MOLBUILDER_DATA_DIR="$HOME/.config/molbuilder/data"
mkdir -p "$MOLBUILDER_DATA_DIR"
cp /path/to/installed/molbuilder/data/fcc_lattice.json \
   "$MOLBUILDER_DATA_DIR/"
# edit the copy as needed
```

molbuilder will look there *first*, then fall back to the bundled
`molbuilder/data/` files.  When you read this README from a deployed
copy and the override directory exists, log lines should mention the
override path.

---

## Adding new data files

1. Drop the file in this directory with a clear, lower-case-snake name.
2. Document its schema and every value in this README.
3. Cite the source for every numeric value you add.
4. Update the `## Files` table at the top.
5. Wire it into the relevant Python module via a `_load_*()` helper that
   reads through the `MOLBUILDER_DATA_DIR` override path before falling
   back to this directory.

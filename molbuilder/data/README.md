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
| `fcc_lattice.json` | v3 | `molbuilder.modify.load_fcc_lattice_full` (and `_load_fcc_lattice`, the experimental-only shim) | live |
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
  production run, use that value.  **Do not put it here** — that is the
  mistake v3 undid: it is a fact about one run, and this file is shared
  by every project.  The Modify tab reads it from the run's own result.

The temperature the published experimental values were measured at lives
in `_sources.experimental.temperature_K`; the PBE column is a
zero-temperature calculation and carries no temperature of its own.

### Schema (v3)

```json
{
    "_format":  "molbuilder.data.fcc_lattice v3",
    "_units":   { "a": "angstrom" },
    "_sources": { "experimental": {...}, "pbe": {...} },
    "metals":   {
        "<symbol>": {
            "a_experimental": <float>,
            "a_pbe":          <float>,
            "system": "fcc", "name": "<full name>"
        }
    }
}
```

**Two references per metal, and they are both from the literature** —
that is what a shared table is for. `a_experimental` is Wyckoff's
room-temperature measurement; `a_pbe` is the all-electron PBE value.
Which one to use is a question about your calculation, not about this
file: match the one your run was built with.

Adding a new entry: add the two values, `system` and `name`, append a
citation line in the table above, and add a paragraph explaining why the
entry was added.

#### What v3 removed, and where it went (2026-08-30)

v2 carried a third column, `a_pbe_siesta_psml` — "the lattice constant
**your** SIESTA and **your** pseudopotential produce". It was `null` for
all six metals, and **nothing in molbuilder could ever write it**: its
only homes were this packaged file and a `MOLBUILDER_DATA_DIR` copy, so
the "Your bulk run" control that read it greyed itself out — correctly —
from the day it shipped.

The shape was wrong, not just the value. A lattice constant measured in
your own setup belongs to **one optimisation run**, not to a table every
project on the machine shares. So it is read from that run's result
instead: point the Modify tab at a relaxed bulk `.xyz` or `.XV` and it
measures the nearest-neighbour distance and reports
`a = √2 · d`, along with what looks wrong (`POST
/api/modify/lattice-from-run`).

**A v2 file still loads.** If you have an overriding copy, it keeps
working; the extra column is ignored. Only the v1 `"a"`-only schema is
refused, because a single number with no exchange-correlation functional
attached is the ambiguity v2 existed to end.

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

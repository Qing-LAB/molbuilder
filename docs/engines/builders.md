# Builders — sequence / SMILES / name → a 3-D `Structure`

**Role:** contract
**Domain:** engines
**Companions:** [`model/structure.md`](?doc=model/structure.md) (the `Structure`
every builder returns); [`model/chemistry.md`](?doc=model/chemistry.md) (the
hydrogen-placement + clash-relief helpers the builders call);
[`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) (the
charge / spin science, and the amber methylene-H fix's chemistry);
[`engines/siesta.md`](?doc=engines/siesta.md) + [`engines/pyscf.md`](?doc=engines/pyscf.md)
(the emitters that consume a built structure — they need clean element strings).

A **builder** is any function that takes a human-friendly description — a peptide
sequence, a DNA/RNA sequence, a SMILES string, a compound name — and returns a
fully-formed 3-D [`Structure`](?doc=model/structure.md) ready to feed a SIESTA or
PySCF job. This is the **server-side synthesis contract**: what each builder
guarantees about its output, how the pluggable nucleic-acid backends are chosen,
and how molbuilder papers over each external tool's quirks so the API stays uniform.

> **The uniform output contract.** Every builder returns a `Structure` with
> `elements` + `positions` set, and `atom_names` / `residue_ids` / `residue_names`
> / `chain_ids` populated **wherever the source format provides them** (peptides and
> nucleic acids carry full PDB-style residue metadata; a SMILES molecule has
> per-atom names like `C1`, `O2` but no residues). Element strings are always
> whitespace-stripped — BioPython hands back space-padded elements like `" C"`, and
> the SIESTA/PySCF species detection downstream needs clean strings.

> **Vocabulary.** A **residue** is one monomer of a chain — an amino acid or a
> nucleotide. **PDB** metadata = the atom / residue / chain names from the Protein
> Data Bank file format. A **heavy atom** is any non-hydrogen atom. **SMILES** is a
> compact text notation for a molecule's atoms and bonds. A **conformer** is one 3-D
> shape a flexible molecule can adopt; a **force field** (MMFF94s, UFF) is a fast
> classical energy model used to tidy a conformer before the real quantum run. The
> builders make a *starting* geometry for a **DFT relaxation** (the quantum-chemistry
> geometry optimization molbuilder ultimately runs) — whose first **SCF**
> (self-consistent-field) step blows up if two atoms sit on top of each other. For
> nucleic acids: the two chemically distinct ends of a strand are **5′** and **3′**
> (sequences read 5′→3′); a **duplex** is two base-paired strands; **Watson-Crick**
> pairing is the standard A·T / G·C base pairs; **B / A / Z** are the three standard
> helix geometries (B = the common form, A = compact, Z = left-handed). (Charge /
> spin and the hydrogen-placement mechanism are in
> [`model/chemistry.md`](?doc=model/chemistry.md).)

---

## 1. The five builders

| Builder | Input | Module | Backends |
|---|---|---|---|
| `build_peptide(sequence, *, title=None, add_hydrogens=True)` | amino-acid sequence | `peptide.py` | PeptideBuilder (extended chain) |
| `build_dna(sequence, *, backend="auto", form="B", terminal="OH", add_hydrogens="auto", protonate_phosphates=True, relax_clashes=False, title=None)` | DNA sequence / duplex notation | `nucleic.py` | threedna / amber / rdkit |
| `build_rna(sequence, *, backend="auto", form="A", terminal="OH", add_hydrogens="auto", protonate_phosphates=True, title=None)` | RNA sequence (single strand) | `nucleic.py` | threedna / amber / rdkit |
| `build_from_smiles(smiles, *, title=None, optimize=True, seed=0xF00D, return_backend=False)` | SMILES string | `smiles.py` | RDKit (OpenBabel fallback) |
| `build_from_name(name, *, title=None, **smiles_kwargs)` | common / IUPAC name | `pubchem.py` | PubChem lookup → SMILES |

```mermaid
flowchart LR
    subgraph IN["you describe the molecule"]
      P["peptide sequence<br/>ARNDC"]
      D["DNA / RNA sequence<br/>ATGC · ds,ATGC · ATGC,GCAT"]
      SM["SMILES<br/>c1ccccc1"]
      NM["name<br/>caffeine"]
    end
    P --> BP["build_peptide"]
    D --> BD["build_dna / build_rna"]
    SM --> BS["build_from_smiles"]
    NM --> BN["build_from_name"] --> BS
    BP --> ST["Structure<br/>(elements + positions +<br/>PDB metadata)"]
    BD --> ST
    BS --> ST
    ST --> ENG["SIESTA / PySCF emitter"]
```

Everything a builder returns is a starting geometry, not a converged one — a
subsequent DFT relaxation is what produces the final structure. The builders' job is
to hand that relaxation something chemically correct and free of atomic near-overlaps.

**One build door** *(decided with the user, 2026-10-31, (b))*. Every surface —
the build route, the command line — builds through `builders.build(kind, text,
**asked)`: the builder its kind names (`builders.GENERATORS`, one row per kind:
the builder, what it takes beside its input, whether a nucleic-acid backend builds
it), `auto` resolved once before the build (`backends.resolved_backend`), and one
check after it (`validation.builders.check_built`,
[`science/validation.md`](?doc=science/validation.md) § 4.1 R5). The answer
(`builders.Build`) is the structure, the backend that built it, and the build's
**findings** — `Issue`s, never Python warnings: the structure's own, read off what
was built (a peptide's aldehyde end `build.peptide_aldehyde`, a nucleic acid's
steric clash `build.steric_clash`), and what the backend that ran does not build of what
was asked (`build.form`, `build.terminal`, § 5). The route answers them as notices;
the command line prints them as `validation.report` prints every finding.

---

## 2. The sequence & notation contract

Peptide and nucleic-acid sequences share one tiny grammar (`residues._parse`):

- **Single letters**, case-insensitive — `ARND` (amino acids), `atgc` (DNA).
- **`[XXX]` brackets** open a 3- or 4-letter PDB code for a modified or non-standard
  residue: `AR[SEP]C` is Ala-Arg-phospho-Ser-Cys. A standard 3-letter code works in
  brackets too (`[ALA]`); the brackets just disambiguate.
- **Whitespace is ignored** anywhere (`A R N D` == `ARND`).
- Anything else — dashes, parentheses, `+` — **raises `ValueError`** with the
  position, so the syntax stays unambiguous.

**Modified amino-acid residues** (`residues.MODIFIED_RESIDUES`) currently
cover six codes — each defined as a patch on a standard parent (atoms to remove +
atoms to add, in the parent's local frame):

| Code | Residue | Parent |
|---|---|---|
| `SEP` | phospho-serine | SER |
| `TPO` | phospho-threonine | THR |
| `PTR` | phospho-tyrosine | TYR |
| `MLY` | N-methyl-lysine | LYS |
| `M3L` | N,N,N-trimethyl-lysine | LYS |
| `ALY` | N6-acetyl-lysine | LYS |

**5′/3′ directionality.** Bare letters follow the biology convention (5′ on the left).
Explicit end-labels are accepted, and `build_dna` and `build_rna` apply the **same
strict parser** (`residues._strip_directionality`; `build_dna` reaches it via
`nucleic._strip_direction`):

| Input | Interpretation | Result |
|---|---|---|
| `ATGC` / `5'-ATGC-3'` | 5′→3′ (implicit / explicit) | as written |
| `3'-CGTA-5'` | reverse-direction | reversed → 5′→3′ (== `ATGC`) |
| `5'-ATGC` / `ATGC-3'` | one-sided label | `ValueError` |
| `5'-ATGC-5'` / `3'-ATGC-3'` | self-contradictory | `ValueError` |

A validation check guards this from the other side: `_check_polymer_orientation`
(`validation/geometry.py`, surfaced as a `polymer.orientation` issue — see
[`science/validation.md`](?doc=science/validation.md)) flags a structure whose
*structural* 5′ end (the residue with no incoming O3′–P bridge) doesn't match
`residue_ids[0]`, protecting against a future backend that lists residues 3′→5′.

---

## 3. Nucleic acids — single strand, duplex, and mismatches

`build_dna` reads three notations from its `sequence` argument
(`nucleic.parse_dna_notation`, which the check after the build reads too):

1. **Single strand** — a bare sequence: `"ATGC"`.
2. **Canonical duplex** — prefix `ds,`: `"ds,ATGC"` lays down the strand plus its
   auto-generated Watson-Crick complement (via X3DNA `fiber`).
3. **Explicit / arbitrary duplex** — two comma-separated strands: `"ATGC,GCAT"`.
   Both are 5′→3′ and paired **antiparallel** (base pair *i* couples `strand1[i]`
   with `strand2[N-1-i]`). A non-complementary pair (A·G, T·T, …) is a **mismatch**,
   placed at the standard base-pair frame as a B-form starting model. Built via X3DNA
   `rebuild` from an idealized Arnott B-DNA base-pair-step parameter file.

   > *Worked pairing.* For `"ATGC,GCAT"` (N = 4) the antiparallel pairs are A·T,
   > T·A, G·C, C·G — all Watson-Crick, a canonical duplex. For `"AAAA,TTTG"` the
   > first pair is **A·G**, a mismatch that lands at the standard frame as the
   > clash the next step says: adenine's amino hydrogen 0.91 Å from guanine's O6
   > (measured 2026-10-31).

**Gates**: any duplex requires the **`threedna` (X3DNA)** backend — refused as
a tool this install lacks (`chemistry.BackendUnavailable`, `missing="threedna"`,
which the build route answers as advice, not a fault); the rest raise a clear
`ValueError`. Canonical duplexes support **B / A / Z** form
(Z needs an alternating poly-d(GC) sequence); explicit two-strand duplexes are
**B-form only** and must be **equal length** (no overhangs/bulges yet). `build_rna`
is **single-strand only** (no duplex notation) and defaults to A-form.

### Steric clashes — said on every build, relieved on request for a mismatch

Atoms of two residues far too close make the first SCF step of a later DFT
relaxation explode. On **every DNA and RNA build** the closest inter-residue contact
is measured (`chemistry.steric_clash`, over `min_nonbonded_contact`) against two
thresholds, and the check after the build (`validation.builders`,
`build.steric_clash`) says a clash, the pair named — never repaired unasked
*(decided with the user, 2026-10-31, (b))*. Two ways one is built:

- **A mismatched pair in an explicit duplex** (two strands typed): a non-Watson-Crick
  pair placed at the standard frame *interpenetrates*. `build_dna` relieves it when
  `relax_clashes` asks, and the check says what is left.
- **A strand as built**: tleap's `sequence` sets each link's ζ/α torsions to
  +60°/−41°, putting C3′ and the next residue's C5′ 2.66 Å apart; their
  hydrogens, as placed — tleap's own on RNA, the methylene repair's
  (`_fix_methylene_hydrogens`) on DNA — end with H3′ and the next H5′ 1.14 Å
  apart (measured 2026-10-31, every link of `ATGC` and `ACGU`). Said, with a
  force-field minimization before the optimization as the way on.

For the explicit duplex:

```mermaid
flowchart TD
    M["explicit duplex built<br/>(post H-add + protonation)"] --> C{"closest inter-residue<br/>contact d?"}
    C -->|"d ≥ 1.3 Å"| OK["clash-free — emitted as built, nothing said"]
    C -->|"d < 1.3 Å"| CL{"relax_clashes?"}
    CL -->|"False (default)"| W["finding build.steric_clash names the pair<br/>(says 'near-coincident' if d < 1.0 Å)"]
    CL -->|"True"| R["relieve_clashes: short OpenBabel UFF push-apart<br/>→ a finding only if still < 1.0 Å"]
```

The finding a user sees (`relax_clashes=False`) names the clashing atoms and, when
they are near-coincident, the explosion risk:

```text
warn [build.steric_clash]: This duplex has a steric CLASH at B8:O6<->A1:H on N6
(0.91 A) -- a non-Watson-Crick (mismatched) base pair at the standard frame.
These atoms are NEAR-COINCIDENT -- relaxing as-is risks an explosive first SCF
step. Build it again with clash relief (Relieve clashes on the DNA panel;
relax_clashes=True through builders.build) -- a short force-field minimization
that removes the near-coincidence, SIESTA relaxing the rest -- or minimize
externally before a geometry optimization.
```

A hydrogen OpenBabel placed is named by the atom it is bonded to (`H on N6`):
the engine names every one plain `H`.

`chemistry.CLASH_CONTACT_A = 1.3 Å` (a real steric overlap — below the tightest
legitimate canonical contact) and `chemistry.NEAR_COINCIDENT_A = 1.0 Å` (near-
coincident nuclei — the actual explosion risk). The relief
([`model/chemistry.md`](?doc=model/chemistry.md) owns the mechanism) first deletes
every mis-perceived inter-residue bond except the real O3′–P backbone link, then runs
steepest-descent + conjugate gradients so the overlapping atoms repel instead of
staying pinned. The goal is only to clear the near-coincidence — the DFT relaxation
does the rest.

---

## 4. The backend registry

The nucleic-acid builders don't build geometry themselves — they dispatch to a
pluggable backend (`builders/backends/__init__.py`):

```mermaid
flowchart TD
    REQ["build_dna(..., backend='auto')"] --> DISP["dispatch(kind, seq, backend, form, terminal,<br/>double_strand, strand2)"]
    DISP --> AUTO{"backend == 'auto'?"}
    AUTO -->|yes| ORDER["resolved_backend → auto_backend_name():<br/>walk _AUTO_ORDER = [threedna, amber, rdkit]<br/>pick first installed"]
    AUTO -->|"no (named)"| NAMED["use the named backend"]
    ORDER --> NONE{"any available?"}
    NONE -->|no| ERR["BackendUnavailable<br/>(install instructions, not a cryptic ImportError)"]
    NONE -->|yes| RUN["backend.build(...)"]
    NAMED --> RUN
```

- **`_AUTO_ORDER = ["threedna", "amber", "rdkit"]`** is the
  single source of truth for auto-selection — **best geometry first**: canonical
  helix → extended chain → folded conformer.
- **`auto_backend_name()`** is a read-only resolver — it reports which backend
  `auto` *would* pick without running anything; **`resolved_backend(backend)`**
  asks it once, before the build, for every door that builds (`builders.build`),
  so the build runs and the check after it names the same backend.
- **`available_backends()`** maps each backend → runnable-here.
- **`DEFAULT_FORM`** (`{"dna": "B", "rna": "A"}`) and **`DEFAULT_TERMINAL`**
  (`"OH"`) are the form and terminal a build uses when none is asked — every door
  reads them here.
- **`built(backend, kind, form, terminal)`** — what a backend builds of an ask
  (`Built`: the form, `None` where it shapes no helix, and the terminal, each with
  the words that say why), answered by each backend's own module beside the table
  its build reads; the check after the build says what differs (§ 5).
- **`BackendUnavailable`** (defined in `chemistry`, the lowest layer that raises
  it) is raised when no installed backend can satisfy the request, with install
  hints for all three -- and by `chemistry.add_hydrogens` when neither hydrogen
  engine (OpenBabel, RDKit) is installed. Its `missing` names what is absent
  (`"threedna"`, `"amber"`, `"rdkit"`, `"hydrogens"`), which the Build route
  reports as `backend`.
- **Adding a backend** is two steps: drop a `_<name>.py` defining `build()`,
  `is_available()` and `built()`, and name it in `_modules()`.

| Backend | Helix shape | Install | Notes |
|---|---|---|---|
| **`threedna`** (X3DNA `fiber`/`rebuild`) | canonical B/A/Z (+ A-RNA) | x3dna.org — **license-gated**, non-commercial; not pip/conda-installable | the only true-helix backend; molbuilder detects it (in-tree dir → `$X3DNA` → `fiber` on PATH) but never auto-downloads |
| **`amber`** (AmberTools `tleap`) | extended chain | `conda install -c conda-forge ambertools` | clean OL15/OL3 force-field chemistry; a `timeout=120` guards a hung `tleap` |
| **`rdkit`** | folded conformer (not helical) | already a dependency | fine for short oligos a DFT run will fully optimize; poor for 10+ mers |

---

## 5. Backend quirks & how molbuilder compensates

Each tool gets something wrong; molbuilder repairs what it can so the
`build_dna`/`build_rna` contract is uniform across backends, and **says** what a
backend does not build — built all the same, as the backend makes it, with a
finding (`build.form`, `build.terminal`) from the backend's own `built()`.
What each produces raw:

| Backend | H atoms | terminal phosphate | residue names |
|---|---|---|---|
| `threedna` | **none** (heavy-only) | **always 5′-P** from `fiber`; stripped for `OH`/`3P` | DA / DT / DG / DC |
| `amber` | included | **OH only** (a 5P/3P/PP asked is built OH, `build.terminal`) | DA5 / DT / DG / DC3 (5′/3′ suffixes); every link's H3′/H5′ 1.14 Å apart, said (`build.steric_clash`) |
| `rdkit` | included (`Chem.AddHs`) | OH or 5′-P (DNA: a 3P asked is built OH, a PP 5P — `build.terminal`) | per-residue DA/DT/DG/DC, each H in its anchor's residue (`AddHs(addResidueInfo=True)`) |

Of the helical forms: `amber` builds an extended chain and `rdkit` a folded
conformer — neither shapes a helix, so a form asked of them is said
(`build.form`, "was not built").

The X3DNA path needs the most repair (`builders/backends/_threedna.py`):

1. **Heavy-atom output → routed through `chemistry.add_hydrogens`** at the
   `build_dna`/`build_rna` layer (see § 7).
2. **Mandatory 5′-phosphate → `_strip_5prime_phosphate`** removes the terminal
   phosphate atoms (both modern `OPx` and legacy `OxP` names) when the terminal
   built is `OH` (`_TERMINAL_BUILT`: asked `OH` or `3P`); the bridging O5′ stays.
3. **3′-phosphate cannot be added → said.** `fiber`'s output is 5′-P / 3′-OH; a
   `3P` is built 5′-OH / 3′-OH and a `PP` 5′-P / 3′-OH, each said
   (`build.terminal`, "was built OH" / "was built 5P").
4. **Z-form is poly-d(GC) only; RNA is A-form only** — a Z-form sequence that is
   not poly-d(GC) is refused; RNA asked in another form is built A-form and said
   (`build.form`).

The **amber** backend post-processes `tleap`'s output with
`_fix_methylene_hydrogens` (`_amber.py`): `tleap`'s residue library copies H
positions verbatim, which is wrong at the C5′ methylene between O5′ and C4′, so this
recomputes *only* the -CH₂- hydrogen positions from tetrahedral sp3 geometry —
touching no heavy atoms and adding/removing none.

---

## 6. SMILES & name builders

`build_from_smiles` (`smiles.py`) tries **RDKit first** — an ETKDGv3 conformer +
an MMFF94s force-field cleanup — because that is RDKit's recommended path for typical
organics and preserves stereochemistry reliably. **OpenBabel is the fallback** for the
two things RDKit can't do:

- **parse failures** — metal-organics / non-kekulizable aromatics like heme (a ring
  RDKit can't assign alternating single/double bonds to), where `MolFromSmiles`
  returns `None`;
- **embed failures** — cages like C60, where ETKDG can't place a conformer.

```mermaid
flowchart TD
    S["SMILES string"] --> P{"RDKit parses it?"}
    P -->|"no (heme, metal-organic)"| OB["OpenBabel make3D<br/>(lower-fidelity geometry)"]
    P -->|yes| E{"ETKDGv3 embeds<br/>a conformer?"}
    E -->|"no (C60, cages)"| OB
    E -->|yes| MM{"MMFF94s<br/>parameterises?"}
    MM -->|"yes (rc 0 or 1)"| KEEP["keep MMFF geometry"]
    MM -->|"no (rc -1 / raises)"| UFF["UFF cleanup"]
    KEEP --> ST["Structure (+ backend tag)"]
    UFF --> ST
    OB --> ST
```

The result records which engine produced the geometry (`return_backend=True` →
`(Structure, backend_str)`) so a caller can warn the user when they're on the
lower-fidelity OpenBabel path.

The optimizer (`smiles._rdkit_optimize`) keeps MMFF94s when it parameterises and falls
back to UFF only when MMFF returns `-1` (could not parameterise) or raises — a
max-iterations result (`1`) is still a valid MMFF geometry and is kept. The ETKDG
seed is pinned (`0xF00D`) for reproducibility.

`build_from_name` (`pubchem.py`) looks the name up in PubChem over the network
(`smiles_for_name`, a 30-second socket timeout; unresolved name → `ValueError`,
timeout → `RuntimeError`), pulls the canonical SMILES, and defers to
`build_from_smiles`.

---

## 7. Hydrogens & protonation

Two independent controls decide the H content of a built nucleic acid:

- **`add_hydrogens`** — tri-state (`"auto"` / `"on"` / `"off"`,
  `nucleic.normalise_h_mode`). `"auto"` is a size-aware
  heuristic: **skip if H/heavy ≥ 0.5** (amber ≈ 0.63, rdkit ≈ 0.72 arrive complete),
  otherwise add (X3DNA's fiber skeleton sits at ≈ 0.05, so it gets protonated). The
  0.5 cutoff is a convenience default, **not a correctness check** — for a structure
  in the gray zone (~0.3–0.6) pass `"on"`/`"off"` explicitly.
- **`protonate_phosphates`** (default `True`) — adds an H to each deprotonated
  non-bridging phosphate oxygen so the molecule is formally **neutral** (the easier
  DFT starting point — no `NetCharge` to set). Set `False` to keep the deprotonated
  state; the SIESTA/PySCF emitter then auto-sets the net charge.

The actual H placement lives in [`model/chemistry.md`](?doc=model/chemistry.md):
`add_hydrogens` tries **OpenBabel first** (geometric, no ghost-coordinate failure
mode; each base's bond orders given it from `residues.NUCLEOBASE_BONDS`, so every
base is its canonical tautomer), **RDKit as fallback** (which can leave "ghost" H at parent coordinates on
exocyclic -NH₂ sites — but only on the PDB-parse→`AddHs` path that the X3DNA
heavy-skeleton takes; SMILES-built molecules are unaffected), then a
`_drop_overlapping_hydrogens` pass strips any H that landed < 0.05 Å from another
atom. AmberTools' `reduce` isn't used even though it's already a transitive
dependency: keeping H-placement uniform and in-process across the peptide and nucleic
flows is worth more than its marginal, protein-only tautomer edge.

---

## 8. Surfaces — CLI and web

**CLI** (`molbuilder/cli.py`) — one command per builder, each writing `.xyz`
with its sidecar (`--out`) or a PySCF atom block (`--pyscf-atom-block`); a
`.pdb` is read, never written ([`model/structure.md`](?doc=model/structure.md)
§ 2.4):

```bash
molbuilder peptide "ARND[SEP]C" --out pep.xyz          # [SEP] = phospho-Ser
molbuilder dna "ATGC" --backend auto --out dna.xyz
molbuilder dna "ds,ATGC" --backend threedna --out duplex.xyz   # canonical duplex
molbuilder smiles "c1ccccc1" --out benzene.xyz
molbuilder name "caffeine" --out caffeine.xyz          # PubChem lookup
```

The nucleic commands add `--backend`, `--form`, `--terminal`, and
`--no-protonate-phosphates`; the duplex/mismatch notation flows through the
`sequence` argument. Every command builds through `builders.build` and prints the
build's findings on stderr (`warn [build.peptide_aldehyde]: …`). The two controls
the CLI doesn't expose — `relax_clashes` and the `add_hydrogens` tri-state — are
reachable in Python through the same door, which answers the structure and what
the build found (a builder called directly returns the structure alone):

```python
from molbuilder.builders import build

# explicit mismatched duplex; relieve the steric clash for a clean DFT start
made = build("dna", "AAAA,TTTG", backend="threedna", relax_clashes=True)
made.structure, made.findings

# force full protonation regardless of what the backend produced
made = build("dna", "ATGC", add_hydrogens="on")
```

**Web** — the builder panels call `POST /api/build/molecule`, which builds through
the same door (`builders.build`) and answers the build's findings as notices, each
panel drawing them under its status; the result goes into the workspace via the
load door (`/api/build/load`). That client-side
flow is [`web/tabs.md`](?doc=web/tabs.md) § 2; this doc is the
synthesis contract behind it.

---

## 9. Regression pins

The protonation and directionality contracts are guarded by tests
(`tests/test_nucleic.py`, `tests/test_peptide.py`, `tests/test_residues.py`,
`tests/test_smiles_and_siesta.py`). Load-bearing ones:

- `test_dna_default_protonation_yields_simulation_ready_h_count` — H/heavy ≥ 0.55
  across all installed backends (catches the X3DNA path silently falling through to
  "no H added").
- `test_protonation_chemistry_matches_across_backends` — pins the canonical
  `5 O-H / 37 C-H / 8 N-H` breakdown (catches the RDKit-fallback regression that
  drops Watson-Crick H-bond donors), every base's canonical tautomer on DNA, RNA
  and a canonical duplex (catches cytosine read as its imino form), and each H in
  its anchor's residue.
- `test_threedna_strips_5prime_phosphate_for_terminal_oh` — pins P count = 0 for one
  nucleotide, 3 for ATGC (catches a strip-helper or fiber-format regression).
- `test_dna_add_hydrogens_false_returns_heavy_skeleton` — pins that `add_hydrogens`
  off is honored.

If any of these go red, the synthesis contract has drifted — re-derive what changed;
don't relax the thresholds.

---

## Tools & references

- **X3DNA** (`fiber` / `rebuild`, canonical B/A/Z helices + arbitrary duplexes) —
  x3dna.org (Lu & Olson, *Nucleic Acids Res.* 2003); non-commercial license,
  not auto-fetched.
- **Idealized B-DNA fiber geometry** (the `rebuild` base-pair-step frame the mismatch
  path starts from) — Arnott & Hukins, *Biochem. Biophys. Res. Commun.* **47**, 1504 (1972).
- **AmberTools** (`tleap`) — `conda install -c conda-forge ambertools`; the DNA/RNA
  force fields are OL15 (Zgarbová et al., *J. Chem. Theory Comput.*, 2015) / χOL3
  (Zgarbová et al., *JCTC*, 2011).
- **RDKit** conformer generation — ETKDG (Riniker & Landrum, *J. Chem. Inf. Model.*
  **55**, 2562, 2015); the **v3** variant adds small-ring / macrocycle torsion terms
  (Wang, Witek, Landrum & Riniker, 2020).
- **Force fields** used for the geometry cleanup — MMFF94 (Halgren, *J. Comput. Chem.*
  **17**, 490, 1996) and the **MMFF94s** static variant actually used (Halgren, *J.
  Comput. Chem.* **20**, 720, 1999); UFF (Rappé et al., *J. Am. Chem. Soc.* **114**, 10024, 1992).
- **OpenBabel** (`make3D`, MMFF94/UFF cleanup; the SMILES + H-placement fallback) —
  O'Boyle et al., *J. Cheminform.* **3**, 33 (2011).
- **PeptideBuilder** (extended-chain polypeptides) — Tien et al., *PeerJ* **1**, e80 (2013).

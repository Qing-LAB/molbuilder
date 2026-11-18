# F12 review, round 1 — slice H (builders, validation, science pages, ops, top-level design chapters)

Checkout `the checkout` at f578482db (2.77.2). Read-only. Nothing was run that writes; the measurements below are in-memory Python calls (`build_peptide` / `builders.build("peptide", …)`, `validate_geometry`, `parse_dna_notation`, `rdkit.Chem.MolFromSequence`) and reads of installed library and data files (AmberTools leap libs, X3DNA `fiber/str15`, pubchempy 1.0.5).

The work was split in two. I read the builders, validation and science chapters, the four top-level chapters, and their code. A forked reader with the same brief read the five `ops/` chapters against `envs/`, `auth_setup.py`, `serve_daemon.py`, `reload_protocol.py` and `scripts/install-env.sh`. Its findings are marked **O-…** below, as it reported them. I did not re-run its traces.

**Totals: 17 defects (H-D1…H-D8, O-D1…O-D9) and 10 untrue sentences (H-U1…H-U6, O-U1…O-U4).**

---

## DEFECTS — builders / validation / science (H)

### H-D1 — A modified amino acid's added atoms ignore the residue's orientation; phospho-Ser's P can land 0.24 Å from CB
- **Code:** `molbuilder/peptide.py:196-197`
  ```python
  anchor_pos = positions[atom_index_by_name[anchor_name]]
  new_pos = anchor_pos + np.array([dx, dy, dz], dtype=float)
  ```
  The offsets in `residues.MODIFIED_RESIDUES` (`residues.py:56-125`, e.g. SEP `("P","P","OG",1.59,0.00,0.00)`) are added in the lab frame. The residue's own frame (the CB→OG direction) is never used.
- **Chapter:** `engines/builders.md` § 2: *"each defined as a patch on a standard parent (atoms to remove + atoms to add, in the parent's local frame)"*.
- **Failure (measured, in memory):**
  - `builders.build("peptide", "[SEP]")`: the build route's `validate_geometry` notice is `error geometry.min_distance: closest atom pair is 0.235 Å apart`. P sits on CB.
  - `"AR[SEP]C"`: 0.577 Å, P3–CB3.
  - With `add_hydrogens=False`: PTR's O3P is 1.16 Å from CE2, MLY's HM1 0.95 Å from CD, ALY's OAC 0.97 Å from CD.
  - Whether a patch clashes depends on where the residue falls in the extended chain (`A[SEP]A` happens to be clean).
  - The route turns the error into a warn notice saying *"check for failed protonation or backend output corruption"*. Prep then refuses below 0.3 Å; between 0.3 and 0.7 Å the run goes ahead from near-coincident atoms.
- Also: every `remove_atoms` entry names a hydrogen (HG, HG1, HH, HZ1-3), and PeptideBuilder writes heavy atoms only (measured H/heavy 0/32), so the removal never removes anything.
- **Sureness:** read-and-traced, and measured.

### H-D2 — Every phosphopeptide gets a false "disconnected backbone" warning
- **Code:** `molbuilder/validation/geometry.py:232-268`. Any atom named `"P"` makes the structure count as a nucleic acid (`if nm == "P": P_pos[rid] = …`, then `if not P_pos and not O3_pos: return []`). With no O3′ present, every residue is a "5′ end", and a single chain gives `warn polymer.orientation: polymer has N residues with no preceding O3'-P bridge … Possible disconnected backbone or unintended branching`.
- **Chapter:** `science/overview.md` § 4, Geometry table: *"polymer residue listing reversed / no preceding O3′–P bridge | warn | reversed or disconnected backbone — a likely backend regression"*. `engines/builders.md` § 2 scopes the check to the nucleic-acid 5′ end.
- **Failure (measured):** `builders.build("peptide", "AA[TPO]A")` and `"AR[SEP]C"` both show this warning beside the structure on the Build panel (`ok_structure_response` → `validate_geometry`). The same warning recurs at preflight and prep. A loaded PDB containing SEP/TPO/PTR residues (atom name `P`) gets it too.
- **Sureness:** read-and-traced, and measured.

### H-D3 — A Z-DNA sequence starting with C is built starting with G, with no finding
- **Code:** `molbuilder/builders/backends/_threedna.py:324-330`
  ```python
  if form == "Z":
      n_repeats = len(seq) // 2
      cmd = [found.fiber] + _FIBER_FLAGS[flags_key] + [f"-rep={n_repeats}"]
  ```
  The sequence is never passed to fiber. `_is_alternating_gc` (`:244-261`) accepts both `GCGC…` and `CGCG…`. fiber's Z model (`-z, -15`, `x3dna-v2.4/fiber/str15/S1.pdb`) has repeating unit residue 1 = G, residue 2 = C. So the strand built is always 5′-(GC)n-3′.
- **Chapter:** `engines/builders.md` § 3: *"Canonical duplexes support B / A / Z form (Z needs an alternating poly-d(GC) sequence)"*. § 2: input is built *"as written"*.
- **Failure:**
  - `molbuilder dna CGCGCG --form Z` (or the DNA panel with Z) returns 5′-GCGCGC-3′ — a different 5′ base and a different molecule. Nothing compares the built sequence with the one asked for.
  - The panel's own refusal text offers "CGCGCG" as a valid example (`web/static/modify/structure/dna.js:166-168`).
  - `ds,CGCG` with Z gives d(GCGC)·d(GCGC) the same way.
- **Sureness:** read-and-traced. fiber was not run; its Z template file was read.

### H-D4 — A single nucleotide built by AmberTools is missing its 3′-OH hydrogen (a radical), and nothing is said
- **Code:** `molbuilder/builders/backends/_amber.py:296-310`. With `n == 1`, the `i == 0` branch gives suffix `"5"`, so one residue is written as `DA5` / `A5`.
- **Library read:**
  - In `molbuilder-MDtools/dat/leap/lib/DNA.OL15.lib`, `DA5`'s atoms end at `"O3'"` with no `"HO3'"`. The both-ends residue `DAN` carries `"HO3'"`.
  - `RNA.lib` (loaded by `leaprc.RNA.OL3`) is the same: `A5` has no `HO3'`, while `AN` exists.
- **Chapter:** `engines/builders.md` § 5 table: amber *"OH only"*. `built()` (`_amber.py:281-293`) answers `terminal="OH"` (5′-OH / 3′-OH). § 1: a builder returns a *"fully-formed 3-D Structure"*.
- **Failure:**
  - `molbuilder dna A --backend amber`, or the panel with amber, or with auto when 3DNA is absent, gives deoxyadenosine short one H.
  - H/heavy is 12/18 ≈ 0.67 ≥ 0.5, so `add_hydrogens="auto"` skips it (`nucleic.py:289-292`).
  - No `build.terminal` finding is raised, because `built()` says OH.
  - The odd electron count is then detected downstream as a radical (`unrestricted`, 2S = 1).
- **Sureness:** read-and-traced (the code and both leap libs). tleap was not run.

### H-D5 — DNA input silently drops characters the grammar refuses, and handles brackets differently from RNA
- **Code:** `molbuilder/nucleic.py:38-40`
  ```python
  body, direction = _strip_directionality(s.strip())
  core = "".join(c for c in body if c.isalpha()).upper()
  ```
  This filter runs before `parse_dna_sequence` / `residues._parse` ever sees the text.
- **Chapter:** `engines/builders.md` § 2: *"Peptide and nucleic-acid sequences share one tiny grammar (`residues._parse`) … Anything else — dashes, parentheses, `+` — raises `ValueError` with the position"*. Also: *"A standard 3-letter code works in brackets too"*.
- **Failure (measured):**
  - `parse_dna_notation` turns `"AT-GC"`, `"A(T)G+C"`, `"AT3GC"`, `"ATGC."` and `"5ATGC3"` all into `('ss','ATGC',None)`. A typo such as `ATG3C` builds ATGC without a word.
  - RNA's `"AU-GC"` is refused (`Unexpected character '-' at position 2`).
  - DNA's `"[DA]TGC"` becomes `"DATGC"` → `Unknown DNA base 1-letter code: 'D'`, while RNA's `"[A]UGC"` is accepted.
- **Sureness:** read-and-traced, and measured.

### H-D6 — The RDKit backend says it cannot build a DNA 3′-phosphate, but the installed RDKit can
- **Code:** `molbuilder/builders/backends/_rdkit.py:34-39`
  ```python
  ("dna", "3P"): 6,  ("dna", "PP"): 7,    # rdkit DNA only has 6 / 7
  ```
  and `:53`: `terminal_words="RDKit's DNA templates carry no 3'-phosphate"`.
- **Library check:** RDKit 2026.03.6 (molbuilder env), `MolFromSequence('ATGC', flavor=8)` gives 4 P, including a 3′ cap residue `2PO` (residue 5). `flavor=9` gives 5 P (both caps).
- **Chapter:** `engines/builders.md` § 5 table, rdkit: *"(DNA: a 3P asked is built OH, a PP 5P — `build.terminal`)"*. That describes the code; the reason the code gives the person is the untrue part.
- **Failure:** `molbuilder dna ATGC --backend rdkit --terminal 3P` builds 5′-OH/3′-OH, with a finding that says *"RDKit's DNA templates carry no 3'-phosphate"*. A PP ask is built 5P the same way. AmberTools' finding (`_amber.py:290-292`) also steers DNA 3′-phosphate users away from RDKit.
- **Sureness:** read-and-traced, and measured. Low severity: the difference is said, but in false words.

### H-D7 — Building by name drops stereochemistry
- **Code:** `molbuilder/pubchem.py:277-278`
  ```python
  smi = (compounds[0].canonical_smiles
         or compounds[0].isomeric_smiles)
  ```
- **Library read:** in pubchempy 1.0.5 (installed), `canonical_smiles` returns `connectivity_smiles` — *"excludes stereochemical and isotopic information"* (`pubchempy.py:1287-1318`). `isomeric_smiles` is only the fallback.
- **Chapter:** `engines/builders.md` § 6: RDKit first *"because that is RDKit's recommended path … and preserves stereochemistry reliably"*. The name builder *"pulls the canonical SMILES, and defers to `build_from_smiles`"*.
- **Failure:** "L-alanine", "(R)-carvone" or "cholesterol" from the Name panel or `molbuilder name` reaches RDKit with no `@` marks. ETKDG then embeds an arbitrary stereoisomer at each unspecified centre (seed 0xF00D), possibly the other enantiomer, with no finding.
- **Sureness:** the code and pubchempy were read and traced. The ETKDG behaviour on unassigned centres is RDKit's documented behaviour; I did not run it, because it needs the network.

### H-D8 (low) — A PubChem failure that is not a timeout is misreported, or answered as a server fault
- **Code:** `molbuilder/pubchem.py:265-272`. Every `OSError` (DNS failure, connection refused — immediate) becomes `RuntimeError("… did not complete within 30.0 s …")`. pubchempy's `PubChemHTTPError` (e.g. a 503 "server busy"; `pubchempy.py:380-382` and `:2059`) is neither `OSError` nor `ValueError`, so it escapes. The route answers it 500 (`web/blueprints/build.py`, last `except Exception`) and the CLI shows a traceback.
- **Chapter:** `engines/builders.md` § 6: *"unresolved name → `ValueError`, timeout → `RuntimeError`"*.
- **Sureness:** read-and-traced for the wrong timeout wording. The 5xx path is plausible: it needs PubChem to answer 5xx.

---

## DEFECTS — ops (O), from the forked reader

- **O-D1 — The documented activation override, and `--projects` on a re-run, are silently ignored.**
  - Code: `envs/initconfig.py:479-483` returns "kept" when the file exists and already declares an activation. `:487-489` writes only `env_init` and never `projects`. `envs/_cli.py:2109-2139` asks both questions before knowing whether the file exists.
  - Chapter: `ops/installation.md` § 2.1, *"Override it without being asked: `… init-config --activation "source activate" --yes`"*. § 2: *"`--projects PATH` declares the tree non-interactively."*
  - Failure: after `bootstrap --yes`, the override prints "left as it is (env_init.activation = "conda activate")". `bootstrap --projects /scratch/x` on a re-bootstrap is dropped without a word.
  - Sureness: read-and-traced.
- **O-D2 — `serve status` calls a live server wedged.**
  - Code: `cli.py:3035` probes `127.0.0.1:{port}` whatever the bind; on failure `:3044` → `:3122` `_note_wedge` → exit 4.
  - Chapter: `ops/deployment.md` § 1.0b/§ 1.0c.
  - Failure: `serve start --host 10.0.0.5` (or `127.0.0.2`) gives "answering: NO", exit 4, a false `DETECTED` line in the log, and advice to restart a healthy server.
  - Sureness: read-and-traced.
- **O-D3 — `serve start --host ::1` is refused.**
  - Code: `cli.py:2735-2737` pre-binds with `AF_INET`; `:2626` then says the address is the problem.
  - `::1` is listed as loopback at `cli.py:1345-1347`, and `serve foreground --host ::1` works.
  - Sureness: read-and-traced, plus a stdlib check.
- **O-D4 — The remedy launcher ignores `envs.manager`.**
  - Code: `scripts/install-env.sh:585-640` (`detect_env_mgr`) never reads `molbuilder.json`. Every remedy names this launcher (`envs/hints.py:54`).
  - Chapter: `ops/installation.md` "The package manager", resolution order 1: `envs.manager`; the Sol `module load` case.
  - Failure: on Sol with no module loaded, doctor finds the manager, but the printed `repair` command exits 2 with "no conda-compatible env manager found". Or it uses a different on-disk manager than Python does.
  - Sureness: read-and-traced.
- **O-D5 — The shim's `--help` says 3.11.**
  - Code: `install-env.sh:424` and `:518-519` say the floor is 3.11; it is enforced at 3.12 (`:218`; `pyproject.toml:50`).
  - Sureness: read-and-traced.
- **O-D6 — A hand-written `env_init.preamble` is overwritten.**
  - Code: `envs/initconfig.py:487-489` with `runtime_config._deep_merge` (scalar replaces).
  - Chapter: `ops/installation.md` § 2.1, *"A file you wrote is never overwritten"*.
  - Failure: a `"preamble": "module load mamba/2.6.2"` is replaced when "conda activate" is chosen.
  - Sureness: read-and-traced.
- **O-D7 — The printed fix for a missing host env is one the shim refuses.**
  - Code: `envs/_cli.py:217-219` and `:488` print `install molbuilder --yes`; `install-env.sh:1277-1283` refuses it and points at bootstrap.
  - Sureness: read-and-traced. Low.
- **O-D8 — Preflight remedies print a literal `<recipe>`.**
  - Code: `envs/builds.py:1280`, `:1469`, `envs/abi.py:380`.
  - Pasted, bash reads `<recipe` as a redirect.
  - Sureness: read-and-traced. Low.
- **O-D9 (plausible) — doctor's verify may point TMPDIR at a directory that does not exist.**
  - Code: `envs/install.py:267` (`make_dirs` off for VERIFY) and `envs/builds.py:740` (TMPDIR = `<prefix>/var/tmp`).
  - The host env is created by the shim, so that directory never exists.
  - Not traced: whether mamba 2.x / micromamba `run` refuses a nonexistent TMPDIR. If it does, verify reads FAILED on a healthy host env.

---

## UNTRUE SENTENCES — builders / validation / science / top level (H)

- **H-U1 — 3DNA's output has no hydrogens to replace.**
  - Chapter: `science/chemistry-correctness.md` § 3.2, *"(X3DNA's raw output has stylised hydrogens that need replacement)"*.
  - Code: fiber's models are heavy-atom only. `x3dna-v2.4/fiber/str01`, `str04` and `str15` hold C, N, O and P only, and `config/Atomic_A.pdb` has no H. `_threedna.py:15-22` says *"Heavy-atom output"*, and `engines/builders.md` § 5 says *"none (heavy-only)"*.
  - Sureness: read-and-traced.
- **H-U2 — The user can pick a backend.**
  - Chapter: `science/chemistry-correctness.md` § 1, *"The user doesn't pick a backend — they choose `--backend auto` (or accept the form default)"*.
  - Code: the CLI offers `--backend` with choices `auto/rdkit/amber/threedna` (`cli.py:311-314`), the DNA panel has `dna-backend-select`, and the route forwards `backend` (`web/blueprints/build.py`).
  - Sureness: read-and-traced.
- **H-U3 — Adding a backend takes more than two steps.**
  - Chapter: `engines/builders.md` § 4, *"Adding a backend is two steps: drop a `_<name>.py` … and name it in `_modules()`"*.
  - Code: `auto` walks only `_AUTO_ORDER` (`builders/backends/__init__.py:217`). The CLI's `--backend` is a hard-coded `click.Choice` (`cli.py:312`), and the panel checks `VALID_BACKENDS` (`dna.js:144`). A backend added in the two steps is never picked by auto and cannot be chosen from the CLI or the panel.
  - Sureness: read-and-traced.
- **H-U4 — Modified-residue offsets are not in the residue's local frame.**
  - Chapter: `engines/builders.md` § 2, *"(atoms to remove + atoms to add, in the parent's local frame)"*.
  - Code: lab-frame offsets from the anchor (`peptide.py:196-197`; `residues.py:54-55`, *"relative offsets from the named anchor atom"*). The consequence is H-D1.
- **H-U5 — DNA does not share the grammar.**
  - Chapter: `engines/builders.md` § 2, *"Peptide and nucleic-acid sequences share one tiny grammar (`residues._parse`) … Anything else … raises `ValueError`"*.
  - Code: DNA goes through `nucleic._strip_direction`'s non-letter filter first (`nucleic.py:39`). See H-D5.
- **H-U6 — An L1 module imports a module outside the L1 index.**
  - Chapters: `architecture.md` § 3, *"L1 is exactly the index below … every other module is L2"*, *"L1 core types (import nothing above them)"* and *"review reads every import, a function's included"*. `design.md`, Architecture: *"lower layers must never import a higher one"*.
  - Code: `runfiles.py:1852` (`runfiles` is in the L1 index) does `from .runpoints import part`, and `runpoints` is not in the index. `runpoints` itself imports `parse.engines._run_ending` (L2, `runpoints.py:40`), although its header calls it "floor 1 + stdlib".
  - Either the index omits `runpoints` (whose own imports would still be upward), or L1 `runfiles` imports L2.
  - Sureness: read-and-traced.

## UNTRUE SENTENCES — ops (O), from the forked reader

- **O-U1 — The Python floor is 3.12, not 3.11.**
  - Chapter: `ops/installation.md:667-669`, *"The floor is 3.11 and it is real — `pyproject.toml` declares it …"*.
  - Code: `pyproject.toml:50` says `>=3.12`, and the shim refuses below 3.12.
- **O-U2 — `seed_document` omits `auth`.**
  - Chapter: `ops/installation.md:224-226` and `:278`, *"a TEMPLATE: every section present and empty"*.
  - Code: `seed_document` omits `auth` on purpose (an empty `auth` is refused).
  - `getting-started.md`'s wording is the accurate one.
- **O-U3 — `MOLBUILDER_LOG` appears nowhere in the code.**
  - Chapter: `ops/deployment.md:587`, *"`MOLBUILDER_LOG` appears in a code comment but is not wired"*.
  - Code: no code or comment mentions it.
- **O-U4 — The shim does not refuse every line it cannot read.**
  - Chapter: `ops/env-framework.md:396-397`, *"A line neither reader understands is refused … by the shim before it creates anything"*.
  - Code: the shim (`install-env.sh:771-781`) refuses only an unknown kind; `pip Foo` with no reason passes it. `recipes.py:233-247` refuses the same line at import, so every `envs` verb then fails with a traceback.

---

## Checked and consistent (not reported)

**Builders:**
- The `builders.build` door: stray asks refused, `auto` resolved once, `Asked` defaults.
- `GENERATORS`, and the signatures in the § 1 table.
- The 5′/3′ label parser.
- The duplex gates (X3DNA only; B-only and equal length for an explicit duplex).
- The clash thresholds 1.3 / 1.0 Å, and relief only for an explicit duplex.
- The `build.steric_clash`, `build.form` and `build.terminal` wording.
- The X3DNA detection chain and the per-chain 5′-P strip.
- The amber timeout.
- The SMILES RDKit→OpenBabel fallback and the MMFF rc rule.
- The CLI options and `report` printing.
- Every test file named in `builders.md` § 9, `validation.md` §§ 4.1/9, `chemistry-correctness.md` § 3 and `overview.md` § 7 exists.

**Validation:**
- F4 box declaration and the `cell.unresolved` info.
- `report()` prints warn and info, and raises on error.
- `check_electronic_state` order matches ES4/ES5/ES6/ES3/ES9/ES8/ES7 and overview § 4's severities.
- The metal sets match `validation.md` § 2.1, the `ChemistryAnalysis` fields, and the detection table (`electronic_state.recommend`).
- SIESTA validator: pseudo coverage severities (`ERROR_STATUSES`), mesh cutoff (normal hint, max over elements, 150 Ry floor), Makov–Payne and vacuum keyed on the state, dipole only when finite.
- PySCF validator: periodic-in-gas-phase warn, ECP hint, basis adequacy, the shared grid gate.
- The transport kind's thresholds (1.5 seam factor, 20 poles, 10 K).
- The `validation/` package tree in `validation.md` § 7.
- `references.py` parses every `.bib` field (all brace-delimited).
- `issues.py` and `reload_protocol.py` match.
- The `workflow.md` § 4 door functions all exist.

## Not counted — docstrings, not chapters

- `nucleic.build_dna`'s docstring says *"``True`` is read as "auto"; ``False`` as "off""*, but `normalise_h_mode` refuses booleans with a ValueError.
- `residues.py`'s module docstring promises `"ATG[5MC]C"` for DNA; DNA never reaches the bracket grammar (H-D5).
- `validation/task.py`'s docstring calls `template` L2, while `architecture.md` lists it as L1.

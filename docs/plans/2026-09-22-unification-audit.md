# Unification audit — what the seven reviews found, and the order to clean it up

**Role:** plan — audit + cleanup sequence
**Domain:** model · parse · transport · web · cli · tests · docs
**Opened:** 2026-09-22
**Why it exists:** several sessions unified the structure/metadata data model —
one periodicity field, one codec, one serialization authority, one index API —
and nobody had checked the result as a whole. Seven full-text reviews did.

**The evidence basis.** Seven reviews, each reading its scope END TO END, each
forbidden from citing `projects/` and required to measure on fixtures. Between
them they read every line of `structure.py`, `workingcopy_structure.py`,
`cell.py`, `periodicity_gate.py`, both `molstruct.py` files, all of `parse/`,
all of `transport/`, `modify.py`, and eight contract documents.

**The verification record.** Every finding acted on below was re-measured by
the primary session before it entered this plan. Of the findings checked:

| checked | held up | corrected |
|---|---|---|
| 26 | 24 | 2 (both *understated* — a gate count of 2 that is 3, a "five docstrings" that was nine) |

One agent claim was **wrong** and is not in this plan: that all four `info`
doors accept a non-finite float. `set_info` refuses it (`TypeError`); only
`apply_info_dict` lets it through. One of my own first measurements was wrong
too (a truncated pipe read as "no output"); re-run, the finding is worse than
first reported, not better.

---

## 0. The one thing to read before deleting anything

**A comment that says code is dead is not evidence that it is.** Three of the
findings below are comments pointing the wrong way, and each one would cause a
regression if a cleanup pass believed it:

- `parse/dirs/job.py:311` — **"NO PRODUCTION READER TODAY"** about
  `RunStatus.concluded`. `web/blueprints/results.py:326` serves it in
  `/api/results/dir`.
- `docs/plans/plan.md:100` — X1 lead ① is **"unreachable"**. It is reachable,
  by two paths, and the line numbers cited as proof point at different
  functions.
- `docs/process/code-audit.md:439` — two audit invariants are **"now an
  enforced test"**, and the reader is told *"Run it"*. Both test files were
  deleted twelve days ago.

This is why the previous session's ordering rule — **fix the map before
navigating by it** — generalises: §§ 1 and 2 below come before § 5.

---

## 1. Correctness holes that reach a user's data or science

Ordered by what they cost.

### 1.1 A half-written pair loses labels, cell and frozen atoms, silently

`apply_info_dict` accepts a non-finite float (`_json.dumps` default
`allow_nan=True`); `molstruct.dumps` refuses it (`allow_nan=False`).
`StructureCodec.write` `os.replace`s the geometry **first**, then writes the
sidecar. Measured:

```
apply_info_dict  ACCEPTED nan
codec write      RAISED ValueError    files left: ['nan.xyz']
reopening it     regions {} · frozen [] · cell None · axis_kind isolated×3
```

No error on reopen — a label-free pair is legitimate. Two stated invariants
break at once: `workingcopy_structure.py:12` promises *"both-or-neither
atomicity on write"*, and `load` reads *"no .json == empty metadata"*.

**Fix:** one `allow_nan=False` at every `info` door, and a rollback (or
sidecar-first ordering) in `write`.

### 1.2 `molbuilder validate` with no `--engine` runs no cell check at all

```
validate lh.xyz --exit-on-error                 → 0 issues, exit 0
validate lh.xyz --engine siesta --exit-on-error → cell.left_handed (det=-125), exit 2
```

The documented pipeline is
`molbuilder validate run.xyz --exit-on-error && molbuilder jobset init …`. A
geometrically impossible box passes to job submission. Two independent causes:
the seam passes `cell=struct.cell` (the raw field, not `resolve_cell()`), and
the geometry-only branch never enters `validate()`, where `cell.check` lives.
`cmd_validate`'s own docstring advertises *"image distance, cell volume"*.

### 1.3 A form-B citation's recorded electronic contract is silently dropped

Measured on a pair whose sidecar carries `info.calculation.contract`:

```
recorded contract seen by compose : TZP / 275 Ry / revPBE / 150 K / (3,3,4)
the template the live doors write : DZP / 300 Ry / PBE   / 300 K / (1,1,1)
```

Five settings lost — while **three live surfaces assert the inheritance**,
including a warning that the values *"come from the deck and were converged for
a cell that is no longer there"*, about values that did not come from it. So a
relaxed junction cited from the Results tab runs a transport ladder with a
*different electronic description from the relaxation it annotates*, which is
the one thing the composite exists to keep single.

Cause: the recorded contract's vocabulary is `TransportConfig`'s; TR4 moved the
road to `SiestaConfig`; the only reader that could translate is `config_for`,
which lost its production caller on 2026-09-16 — **18 days after the branch was
built.** See § 5 lead ④.

### 1.4 The documented CLI pipe destroys the pair and asserts the loss

```
dev.molstruct.json    L-electrode, frozen_atoms · cell set · origin (-4,-4,-1) · periodic,periodic,transport
piped2.molstruct.json regions [] · cell NULL · origin None · isolated,isolated,isolated
```

`modify`'s own `--help` gives this two-stage pipe. The stdout destination
writes `to_xyz()` — bare XYZ — in the same two functions whose *file* branches
were fixed to use the codec. The loss is then **materialised as a positive
claim**: the second stage writes a sidecar saying the junction is a vacuum box.

### 1.5 `/api/structure/analyze` 500s on a file that is not UTF-8

`build.py:236` reads the file one line *above* the `try`, so
`UnicodeDecodeError` escapes to Flask. A `REMARK angle 90°` PDB written
latin-1 — what external tools emit — gives an HTML 500 where
`/api/build/load` gives a clean 400 JSON. Live from the browser via
`auto-detect.js:163`.

### 1.6 A false containment warning, on a knob that changes nothing

`validation/siesta.py:798` combines `struct.positions` with a cell anchored at
`(0,0,0)`, ignoring `resolve_cell_origin()` and `axis_kind`. Measured: the
preflight panel warns *"1 of 4 atoms have fractional coords outside [0,1)"*
while `cell_contains_atoms(resolve_cell_origin())` is `True` and the emitted
deck has no such complaint. **And `cfg.wrap_into_cell` is inert** — no
production caller passes `cell=` to `spec_for`, so the decks for
`wrap_into_cell` True and False are bit-identical. A browser checkbox that
produces a warning contradicting the deck and no change to the deck.

**This is the origin rule's sixth independent site.** Five were found and
fixed; `handover.md` § 2.1 is the record.

### 1.7 A malformed annotations channel escapes as a bare `KeyError`

`AtomChannel.from_json` does `obj["kind"]`; every layer catches
`(ValueError, TypeError)`. So a hand-edited sidecar gives
`KeyError: 'kind'` with **no file path**, through a door whose whole stated
purpose is *"fails loudly at load time with the file path in the error"*. The
deck format makes the opposite mistake and drops the channel silently.

---

## 2. Documentation: one policy, not forty-four edits

**4 of 44 `file.py:NNN` references in the contract documents resolve — a 9%
hit rate.** Exhaustive, not sampled. `structure.md` 0/16,
`structure-periodicity.md` 0/10. Plus seven symbol names that have never
existed, and five retired concepts written as current.

The fix is already written down in this repo, by this repo —
`docs/model/parse.md:355`:

> *"A line number is a pin: it measures where a thing sits rather than what it
> does, and it rots on the next edit of a file this document does not own. The
> function name is the anchor and it is greppable."*

`parse.md` acted on it and scores 2/4. The four model documents did not and
score **2/38**. So this is **one policy to propagate**, not 44 corrections:
strip the line numbers, keep the symbol names.

Separately, and not covered by that policy — documents asserting **behaviour**
that is not true. Highest cost first:

1. `code-audit.md` §§ 5.1 / 3.1 — two invariants "enforced by" deleted tests,
   with *"Run it"* (§ 0 above).
2. `structure.md` § 2.4 and `workingcopy_structure.py:13` — *"the periodicity
   gate on read"*. The codec's own body says **"READING DOES NOT JUDGE"** 162
   lines later. Four sites, two documents.
3. `structure.md:173` — `apply_metadata_dict` *"ignores unknown keys"*. It
   refuses them, so the document's "to remove a key" recipe is incomplete.
4. `conventions.md` — an enforced parse-purity gate whose file is deleted, and
   `molbuilder pyscf` described as surviving three paragraphs after being
   described as deleted.
5. `structure-annotations.md` — **three different sidecar schema versions in
   one document.**
6. `testing.md` — four stale counts, in the sections that say not to quote a
   count from a document.
7. `docs/engines/transport.md`'s *"deleted | why"* table — header says
   **deleted** (past), the sentence above says *"stop existing"* (future);
   four of six rows name live code; `:1585` asserts
   `dataclass_to_form_schema` *"has no callers"* and it is called at
   `web/blueprints/transport.py:547`.

### 2a. Vocabulary that actively misleads

- **"the gate"** names four mechanisms (`validate_periodicity`, `cell.check`,
  `_refuse_on_error`, the metadata key gates). Two findings above turn on which
  was meant.
- `structure.md:75` says `frozen_atoms` *"is not a field"* — meaning not a
  storage site. It **is** a declared dataclass field, and `structure.py` spends
  45 lines on the `dataclasses.replace` trap that follows. A reader taking it
  literally reaches for the one function that section forbids. The code's own
  phrase is the clear one: *"a constructor door, not a storage site."*

---

## 3. One rule, several enumerations

Each of these is complete **today**; each fails silently the day a field is
added. They are listed together because the fix is the same shape.

| the rule | where it is spelled | the claim that is false |
|---|---|---|
| the metadata field set | `metadata_to_dict`, `apply_metadata_dict`, **`to_wire`**, + 2 in the sidecar stack | `structure.py:864`: *"these TWO methods … and nowhere else"* |
| the identity columns | 5 places | — |
| the `Structure` field list | `replace()`, **`affine()`**, `concat()` | `copy()`: *"ONE implementation, in `replace()`"* |
| legal sidecar keys | 2 spellings of one 5-set union | `RETIRED_METADATA_KEYS` was hoisted to stop exactly this; the union was not |
| `n_atoms_total` / `structure_hash` validity | 3 spellings, **2 different messages** | — |
| containment + the fractional solve | `structure.py` and `cell.py`, 2 eps constants | 3 comments say `periodicity_gate` delegates — to functions deleted in August |
| "empty space on this axis" | `axis_vacuum` (triclinic-safe) vs `validation/siesta.py` (not) | measured to disagree by 7.1 Å on a hexagonal Au(111) lead — the stack's central case |
| the number 40 | 4 value homes + 1 prose literal | the handover says three |
| "transverse vacuum" | 4 numbers: 3.0, 5.0, 8.0, 15.0 | two "canonical floors" 2× apart; the transport one is unsourced |

**And the guard behind the most load-bearing of these does not work.**
`test_replace_carries_every_field_the_dataclass_declares` promises *"a field
added tomorrow is covered the moment it is declared."* Measured with a 16th
field: `replace()` dropped it and the test flagged nothing. It iterates
`dataclasses.fields()` but compares against a **hand-built fixture**, so a
field the fixture does not set is default on both sides and passes either way.
The hand-listed set moved from the assertion into the fixture.

---

## 4. One rule, two strictnesses

- **negative `vacuum`** — the gate refuses it; `Structure` and
  `apply_metadata_dict` accept it. `resolve_cell` then returns a negative box
  length and `cell.check` reports `cell.left_handed` with the advice *"swap any
  two of the three rows"* — for a derived box with no rows — and
  short-circuits, so the real findings never run.
- **`annotations` vs `regions`** — `regions` refuses a non-list and a non-`int`
  index by name; `annotations` refuses neither. And `set_channel` installs the
  channel **before** validating, so a refused call leaves a structure that
  cannot be saved.
- **`files()` vs `write()`** — `write` derives `fmt` from the target suffix;
  `files` never passes it, so the save door can produce a PDB pair and the
  export door answers `mol.pdb.xyz` with XYZ inside.
- **form A vs form B** — A states `axis_kind` outright (twice, each under a
  comment recording the regression a sidecar-supplied kind caused); B takes
  whatever the pair says. Measured: a pair claiming `periodic×3` composes as
  `periodic×3`, the deck labels the open boundary `periodic`, and I8's warning
  cannot fire. A checks its two statements of the cell against each other
  twice; B zero times.
- **`apply_metadata_dict` is full-replace** and `apply_to_structure` takes a
  partial silently. Every production caller passes a complete block today; the
  trap is one line from any new one, and the rule lives in a comment at a call
  site rather than at the door.
- **`schema_version` is rewritten on load.** On disk `7`, `molstruct.load`
  returns `9`. So the parse layer's discriminator is always `molstruct/v9`, and
  an electrode swap restamps a v7 file as v9. Nothing is lost today; the
  damage is to the evidence.

---

## 5. Residue — safe to delete, once §§ 1–2 are done

X1's five leads, each with the § 1d step-0 read. **The plan's own verdict on
lead ① was wrong**, which is the warning in § 0 coming true.

| lead | verdict |
|---|---|
| ① `_compute_cell_from_extents` | **NOT RESIDUE — LIVE, and the reachability claim is false.** `load_compose_record` has no `_unusable_cell` call, and `as_structure()` wraps a *fabricated* box in an explicit `cell=`, so `_lattice_block` prints *"Explicit lattice preserved from the structure (NOT recomputed from atom extents)"* about a recomputed one. Measured: `resolve_cell()` says `[7.44, 6.00, 17.22]`, the deck writes `[31.44, 30.00, 20.00]` — the transverse pair off by 4×. Two tests pin the fabricated box **as the contract**, so this is a contract decision, not a cleanup. |
| ② `DEFAULT_ELECTRODE_KZ` | **RESIDUE.** The commit that deleted its last two readers edited `__all__` to keep it. Deletion orphans nothing. |
| ③ `SEALED_TRANSPORT_FIELDS` | **RESIDUE.** Production builds a *different* union inline (three members, not two). Its only reader is a test weaker than the code it guards. |
| ④ `config_for` | **PART RESIDUE, PART LOST CALLER.** One rule inside it — filling the config from a form-B pair's recorded contract — stopped running on 2026-09-16 and nothing took it over. That is § 1.3. |
| ⑤ `num_threads` | **NOT A DEFECT** — a dead field behind a working structural guard. **But `log_level` is the same shape and *does* reach the deck**: `TBT.Verbosity 5` in every transport deck, unconditionally, from a field no description can set. |

Other residue, each with the step-0 read done: five `_enumerate_files` buckets
with no reader anywhere, built on the Watch polling hot path with four full
directory scans; three legacy shim classes and the five test-only names that
depend on them (one decision, not five); `sha256_of_file`, whose docstring
calls it *"the `structure_hash` invariant pin"* — **nothing in the tree
verifies a `structure_hash`**, so one of two documented integrity guards is
prose; `selection_rules`, a format field with no producer and no consumer;
`sidecars.molstruct.load_text`, zero callers, docstring naming a capability
deliberately deleted; the `*-electrode` convention two live modules advertise,
which `sort.PARTITION_LABELS` cannot compose.

---

## 5a. The tests — the count must come DOWN, and three rules are unpinned

*(The seventh review. Every duplicate cluster below is backed by a mutant: one
change to production and the listed tests go red **together**. All mutations
ran in an isolated `git worktree`; the eight taken before that was arranged
were re-run there and all eight reproduced.)*

**~28 of 441 in-scope tests are removable** — 24 duplicates and 4 that cannot
fail or assert a shape. `docs/process/testing.md` already says unifying an API
must REDUCE the count; it has been going up.

Twelve clusters, each with the bit it carries and the one test to keep. The
largest: **16 tests carry the single fact "the default isolated vacuum gap is
3 Å"** (mutant: `3.0 → 5.0`), three of them byte-identical assertion triples in
three files. Three of the sixteen are thin-wrapper tests on `cell.resolve()`,
which just forwards to `Structure` and decides nothing — `testing.md` puts
those on the `Structure` side. Next largest: **9 tests assert the same literal
corner `[7.5, 7.5, 7.5]` on the same fixture**; keep three, one per layer.

### Three cannot-fail tests, each measured

- **`test_periodicity_gate.py:1126`** is the *inverted* test `testing.md` § 3a
  names. Its assertion is
  `assert "Thin vacuum" not in texts or "4.0" not in texts`. Measured: the
  first disjunct is **False** today — the answer *does* carry
  `cell.vacuum_thin` — and the test passes only because
  `validation/siesta.py:409` formats with `{v:g}`, rendering `4.0` as `"4"`.
  Mutating the number format **failed** it; deleting the whole check
  **passed** it. Fails on cosmetics, passes on deletion.
- **`test_cell.py:202`** asserts `"a " in found.message` to mean "it names the
  axis letters". `"a "` occurs three times in the prose. Dropping the axis
  letters while keeping the clearances left all 46 tests in the file green.
  This is the identical defect `test_periodicity_gate.py:222` records having
  fixed on 2026-09-09 — *"`\"a\" in str(exc.value)` stood here until
  2026-09-09 and could not fail: the letter is in 'than', 'cannot', every
  English sentence."*
- **`test_cell.py:588`** asserts a **signature** via `inspect.signature`. The
  same ruling is observable as a result: `interplanar_spacing("fcc","111")`
  raises `TypeError`. Assert that instead.

And **`TestDocMatchesTheDoor`** (3 tests) reads shipped text with
`pathlib.read_text()` — the shape `testing.md` retired 18 files of on
2026-09-10. It iterates `OPS`, so **a shrinking `OPS` is invisible to it**:
removing the `block` op left all three green while the six `TestTheBlockOp`
tests failed. It is blind in the one direction that loses a user a button.

### Three documented rules that nothing pins

| the rule | where it is stated | the mutant, and what happened |
|---|---|---|
| `write(struct, "x.pdb")` must produce a readable PDB pair | `structure.md:446` records it as a **measured defect fixed 2026-09-07** — *"the door could not read back what it had just written"* | restored the pre-fix behaviour → **460 passed, 1 skipped.** And suite-wide: **0** tests write a `.pdb` through the codec, **0** pass `fmt=` |
| geometry is replaced **before** the sidecar, and the write is both-or-neither | `structure.md:354`, § 2.4 clause 4 | reversed the order the document forbids → **441 passed.** `atomic=` has no caller in the suite |
| an explicit cell that fits the structure but not structure + vacuum **centres** it | `structure-periodicity.md:485`, the corner table's fourth row | deleted the centring branch → **441 passed** |

The first is the sharpest: a regression that actually happened, with its
numbers written down, is re-introducible today with the suite green.

### And 15 fixtures hand-build a `structure_hash` no writer can emit

`sha256_of_file` returns 64 hex chars; the write gate only asks
`isinstance(str) and len >= 16`. So `"b"*32`, `"0"*32` and
`"sha256:" + "0"*64` all pass and none is a value the codec produces.
Tightening the gate to the real shape failed **15** in-scope tests — and
three of them failed with *"Regex pattern did not match"*, meaning their
`pytest.raises` was matching a **different** error than the one they name.
That is the sharper risk: they would pass and fail for reasons unrelated to
their subject. Same shape as the sixteen sidecar fixtures already fixed, one
layer down.

---

## 6. Needs a ruling, not a fix

1. **X2 ①** — a frame-range `.xyz` read back without its sidecar loses
   `transport`, because the extxyz `pbc=` header is boolean. Either molbuilder
   writes its own `axis_kind` key into the comment line, or the loss stands and
   the pair remains the only faithful carrier.
2. **Lead ①** — deleting `_compute_cell_from_extents` changes a contract two
   tests pin. Delete the fabrication and refuse instead, or keep it and fix
   the false *"Explicit lattice preserved"* claim?
3. **`TBT.Contour` energies** — absolute or E_F-relative? `record.py:326`
   writes `"energies_relative_to_ef": True` unconditionally and
   `conductance_g0` reads T at E = 0; `config/transport.py:202` says the
   question is *"unresolved against SIESTA 5.4.2"*. Settled by reading one
   real `AVTRANS` beside its device `.fdf` — not derivable from source.
4. **Does a CLI edit owe the user a notice?** The web's eight ops all report;
   `cmd_modify` reports nothing. § 8.2 assigns the CLI only the *generation*
   guard. (The CLI *seam verdict* was already raised and dropped on a ruling —
   do not re-propose that one.)
5. **`set_info` / `drop_info`** have no Python caller — deliberate parity with
   `molview.data.info`. Keep as a declared extension point, or delete?

---

## 7. Do NOT "fix" these

Confirmed non-findings, recorded so nobody fixes them by analogy.

- **`frozen_atoms`'s shape is benign.** Measured: there is no instance
  attribute at all — a data descriptor sends every read and write through
  `regions[FROZEN_LABEL]`, so the two cannot disagree in any order at any
  time. `replace()` handles it correctly by not re-passing it. **Not the same
  defect as `pbc`.**
- **`resolve_cell` really is the one resolver.** Nothing else in the tree
  computes an effective cell.
- **The companion-lookup merge stays withdrawn** — but for the right reason.
  `parse.md` § 5.3 does *not* forbid it; what forbids it is a test pinning a
  deliberately more permissive guard. A shared helper with the guard as a
  parameter would keep both. And there is a **third** copy of the shape
  (`sibling_md_nc`) with neither guard.
- **The second multi-frame XYZ reader is forced**, not duplicated: it
  tolerates a torn final frame, which every live geomeTRIC run has and which
  ase refuses. Unifying it would break live-run viewing.
- **Registry overlap: none.** 14 parsers against a 29-file synthetic run
  directory — 0 files claimed twice, 0 sniffers raised.
- **Bare atom-index arithmetic in `parse/`: none data-carrying.** Every `±1`
  classified by reading; the hits are range ends later routed through
  `from_engine_index`, frame indices, and six refusal-message strings.
  **But `transport/transiesta.py` has four real ones** (§ 3's index row) in
  the block whose own docstring says *"an off-by-one … computes transmission
  through a region that is not the molecule, and converges while doing it."*

---

## 7a. COVERAGE — what this audit read, and what it did not

*(Recorded 2026-09-22 at the user's request, and then **validated** rather than
asserted. The scope of this audit is **structure handling and metadata
handling**. Everything else is audit #2, § 7c.)*

### How the scope was established

A reference scan located every production module that imports `Structure`,
`StructureCodec`, `sidecars.molstruct`, or `metadata_to_dict` /
`apply_metadata_dict` — **59 modules**. A scan locates candidates and settles
nothing, so each was then checked against the read-depth table the owning
review declared. That comparison is what § 7b is.

### Read END TO END, by review

| review | files |
|---|---|
| structure data model | `structure.py` (2232) · `workingcopy_structure.py` (379, ×3 reviews) · `cell.py` (1086) · `periodicity_gate.py` (626) |
| sidecar / serialization | `sidecars/molstruct.py` (705) · `parse/sidecars/molstruct.py` (315) |
| parse layer | **all of `parse/`** — `__init__`, `base`, `registry`, `errors`, `types`, `contract`, `fdf`, `ion`, `_log`; all of `coords/`, `engines/`, `sidecars/`, `instruments/`, `dirs/`; plus `engine_atom_index.py`, `runfiles.py`, `constants.py` |
| transport | **all of `transport/`** — `compose` (1184), `transiesta` (776), `stages` (442), `wizard` (440), `deck` (624), `sort` (339), `record` (384), `citation_defaults` (118) · `config/transport.py` (671) |
| tests | 18 files, **441 tests** · `conftest.py` (1039) · `testing.md` |
| contracts | `structure.md` · `structure-periodicity.md` · `structure-molstruct.md` · `structure-annotations.md` · `parse.md` · `code-audit.md` · `testing.md` · `conventions.md` |
| consumer seams | `modify.py` (1310) · `web/blueprints/modify.py` (1100) · `validation/geometry.py` (301) |
| spectrum (4 reviews) | `pyscf/vibration_emitters.py` (1883) · `pyscf/vibration_deck.py` (696) · all of `spectra/` (1950) · `sidecars/spectra.py` · `parse/sidecars/spectra.py` · `web/blueprints/spectra.py` · `lib/spectra/core.js` (3617) · `lib/spectrumchart/index.js` · `lib/vibrationview/{index,_maths}.js` · `config/siesta.py` (1648) · `siesta/stages.py` · `pyscf/stages.py` |

### Read at REGION depth only

`script_emit.py` (2250 — the atom-metadata block and the emit call sites) ·
`cli.py` (3663 — every structure-related region) · `siesta/input.py` (1925 —
the deck writer's structure and cell paths) · `web/blueprints/build.py` (2579 —
1-1300 end to end, the rest by region) · `web/blueprints/watch.py` (906) ·
`web/blueprints/_shared.py` (three regions) · `jobset/prep.py` (the engine
seam and dispatch) · `validation/siesta.py`, `validation/pyscf.py`,
`validation/__init__.py` (the periodicity and kind-gate regions) ·
`pyscf/input.py` (the deck assembly and `emit_save_helper`) ·
`config/pyscf.py` (the vibration section) · `chemistry.py` (`symbol_for_z` only)

---

## 7b. THE COVERAGE GAP — 11,888 lines inside this audit's own scope

**The validation found a hole, and it is not small.** Of the 59 modules that
touch the structure/metadata surface, **eighteen were not opened by any of
the seven reviews this audit ran.** That is a fact about this audit's reach,
not about the code. Sized
and grouped by what they are:

### The structures are BORN here, and nothing audited it — ~2,200 lines

`peptide.py` (262) · `smiles.py` (199) · `nucleic.py` (380) · `pubchem.py`
(90) · `builders/backends/{__init__,_common,_rdkit,_amber,_threedna}.py`
(1,413)

Every one of these **constructs a `Structure`**. This audit is about one door
per operation, and the door a structure comes *through at birth* was not
looked at once. That matters concretely: the front-page docstring example
taught `s.to_xyz("out.xyz")` until this session, and the builders are the
other half of that story — whatever they do with regions, `axis_kind`,
`vacuum` and identity columns on the way out is unexamined. Four separate
backends is also the exact shape §§ 3 and 4 are about.

### A sidecar validator, in a session about sidecar unification — 153 lines

`validation/sidecar.py`. Never opened. Its name says it validates the thing
three reviews spent their time on.

### Two structure-touching web doors — 2,677 lines

`web/blueprints/selection.py` (214) and `web/blueprints/files.py` (2463).
`files.py` is mostly the file-picker and not structure work, but it is named
in `structure-molstruct.md`'s own pairing rule (`_paired_sidecar_path`,
`_existing_paired_sidecar`) — and those are two of the stale line references
§ 2 counted.

### The element and mass table — 1,947 lines

`chemistry.py`, read only for `symbol_for_z`. `spectra.md` § 4.2 names
`chemistry.atomic_mass` as the sanctioned mass source, and the worst defect
this project has had was a mass-convention bug wrong by 1823×. The table
itself was never audited.

### Smaller, all unread — ~750 lines

`describe.py` (320) · `frame.py` (250) · `trajectory_log/format.py` (180) ·
`validation/chemistry.py` (303)

### Out of scope even though it appeared in the scan

`jobset/_cli.py` (3714) imports `Structure` but is the **execution** layer —
audit #2. Same for the non-structure bulk of `files.py`.

**So the honest coverage statement is:** this audit covered the *core* of
structure and metadata handling — the model, the codec, the serialization
stack, the parse layer, the transport consumer and the spectrum path — and did
**not** cover where structures are **created**, nor the sidecar validator, nor
the chemistry table, nor two web doors. The findings in §§ 1-5a stand on what
was read; they are not a statement about the eighteen files that were not.

### Closing the gap

One targeted pass, three reviews, ~5,000 lines of genuinely in-scope code:

| review | scope | the question |
|---|---|---|
| **G1 — birth** | the five builders + four backends | what does a newly built `Structure` carry, and does every backend agree? Do any of them write a file, hand-build metadata, or set fields the codec owns? |
| **G2 — the unaudited validators** | `validation/sidecar.py`, `validation/chemistry.py`, `chemistry.py`'s mass/element tables | is `atomic_mass` one table? does the sidecar validator agree with the three gates? |
| **G3 — the remaining seams** | `web/blueprints/selection.py`, the structure parts of `files.py`, `describe.py`, `frame.py`, `trajectory_log/format.py` | the same one-door and origin-rule questions §§ 1 and 4 asked of the other seams |

This belongs to **this** audit, not the next one — it is inside the scope the
audit claims.

---

## 7c. AUDIT #2 — the rest of the tree, planned not started

Roughly **70,000 of ~110,000 lines** are outside this audit's scope
altogether. Grouped as they would be reviewed, largest first:

| area | lines | why it is its own audit |
|---|---|---|
| `web/static/lib/` | **33,364** | the browser half — molview, editor, inspectors, projects. The unification rules cross a **language boundary** here, so a violation looks different: a JS module re-deriving a fact the wire payload already carries cannot be found by reading Python |
| `jobset/` | **11,895** | prep, materialize, submit, runstatus, summarize, ledger, model, ask, `_cli`. How a calculation becomes a job. Crosses a **process boundary** — the wrapper runs on a cluster node with no molbuilder |
| `web/blueprints/` (remainder) | ~9,000 | results, transport, `_shared`, app, projects |
| `runwrap.py` | 4,950 | the wrapper generator — the other side of the process boundary |
| `runtime_config.py`, `template.py`, `monitor.py`, `checkpoint.py`, `task.py` | ~9,000 | the catalogue and config resolution; `template.py` is where a knob's identity is decided |
| `config/`, `validation/` (remainder) | ~7,800 | |

**Two reasons to keep it separate rather than extend this one.** The
principles are different: this audit checked *one door per operation* over a
data model, while the browser and the wrapper are about *one fact per side of
a boundary* — a different failure shape needing different questions. And the
evidence is different: a JS finding cannot be measured with a Python fixture,
and a wrapper finding cannot be measured without submitting a job, which is
one-at-a-time and manual here.

**Sequencing.** After §§ 1-2 of this audit's order and after § 7b's gap pass.
Running it sooner would mean navigating the browser layer by a map this audit
has already shown to be 9 % accurate on line references — and the browser
contracts were not among the eight documents checked.

---

## 8. The order, and why

1. **§ 0's three misleading comments.** One commit. Nothing else is safe until
   the map is right — that is `handover.md` § 5's D5-before-X1 rule, and lead
   ① is what happens when it is skipped.
2. **§ 1.1, § 1.2, § 1.5, § 1.7** — the data-loss and uncaught-exception set.
   Independent of each other, each small, each user-visible.
3. **§ 1.6 + the `wrap_into_cell` knob** — the origin rule's sixth site. Do it
   with the rule in front of you, from `handover.md` § 2.1.
4. **§ 1.3 + § 5 lead ④ together.** The lost caller and the residue are one
   decision: either revive the recorded-contract read on the live road, or
   delete the branch and stop three surfaces claiming it works.
5. **§ 1.4** — the CLI stdout destination. Needs § 6 decision 1 first, because
   what a single stream *can* carry is the same question.
6. **§ 2's line-number policy** — one pass over four documents, mechanical.
   Then § 2's behavioural list, which is the part that needs reading.
7. **§ 3 and § 4** — fix each at its owner, never at the instance. Start with
   the `replace()` guard, because it is what makes the rest safe to touch.
8. **§ 5a's three unpinned rules** — write them before §§ 3/4 touch the code
   they guard. The `.pdb` one first: it is a regression with numbers already
   written down.
9. **§ 5a's four cannot-fail tests** — retire or rewrite. They are worse than
   absent, because they read as coverage.
10. **§ 5a's duplicate clusters** — ~24 tests out, one keeper per bit, each
    cluster's mutant re-run afterwards to confirm the keeper still goes red.
11. **§ 5's residue** — last, and only the four with a clean step-0 verdict.

**Before step 11, and after step 2:** § 7b's gap pass. Residue cannot be
deleted from a surface where eighteen modules were never opened by this
audit — the
builders in particular, since they are where a `Structure` is created.

**Then, separately:** audit #2 (§ 7c).

**Standing on its own, not part of the order:** the 15 hand-built
`structure_hash` fixtures (§ 5a). They agree with the writer today only
because the gate is loose, so they are a latent break rather than a defect —
convert them as each file is touched for another reason.

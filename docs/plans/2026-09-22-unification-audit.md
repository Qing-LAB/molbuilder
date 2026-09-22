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
| 24 | 22 | 2 (both *understated* — a gate count of 2 that is 3, a "five docstrings" that was nine) |

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
8. **§ 5's residue** — last, and only the four with a clean step-0 verdict.

**What this audit did not cover:** the tests review was still running when
this was written; its duplicate-cluster analysis slots into § 3 and § 4, and
its verdict on the three legacy shim classes decides § 5's last row.

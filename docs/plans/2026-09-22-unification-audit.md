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

## 0c. THE HARNESS COULD REPORT A GREEN SUITE THAT WAS NOT GREEN — and what that does to this plan's evidence

*(2026-09-23. Three commits from the other machine — `9d586758`, `93fb090c`,
`3ee27d9c` — fixed it. Reviewed and re-verified here.)*

**`FAIL 0` was never evidence. `(exit 0)` is.** For **62 days**
(2026-07-22 → 2026-09-22) a session-scoped fixture raising in **teardown**
produced no test record, so `status` printed `FAIL 0`, and `cmd_status`
returned **0**. The exit code was in the head line the whole time as
decoration beside the counts. Same for an internal error or a plugin error:
anything that fails outside a test's `call` phase.

**It happened, once, and it is dated.** 2026-09-05:

```
[none2e] done (exit 1) | 8234/8234 ran | pass 8231  FAIL 0  skip 3 | 1399.4s
```

Complete run, non-zero exit, `--fails` printed nothing, the session moved on
seven seconds later. Under the fixed harness that reads `UNEXPLAINED (exit 1)`
and returns 1.

### This plan's own evidence is SOUND, and here is the check

`.test-progress/none2e.jsonl` still holds the run every "suite green" claim
here rests on. Re-verified 2026-09-23:

```
{"event": "done", "exitstatus": 0, ...}
9527 test records · 9518 passed + 9 skipped = 9527 = collected
```

**Exit 0 rules out the entire hidden class** — pytest assigns
`session.exitstatus` *before* calling `pytest_sessionfinish`, so a teardown
error cannot hide behind a zero. The counts close. Sound.

**The unsafe form is the SUMMARY, not the raw line.** This session wrote
`9527/9527 ran | pass 9518 | FAIL 0 | skip 9 | 2762s` — exit code dropped.
That is exactly the shape that would have been wrong on 2026-09-05. **Quote
the exit code or quote nothing.**

### Three regressions the fix introduced — open, in `tools/`, NOT held

1. **`run lf` with nothing to rerun now shouts NOT GREEN.** Verified:
   `--last-failed --last-failed-no-failures none` exits **5** when the last
   run was green, and the new branch fires on *any* non-zero exit with zero
   failures — printing a message blaming a teardown canary. `exit 4 | 0/0 ran`
   appears 302 times in this repo's history; that whole class becomes noise.
   **This is how the new guard gets trained away**, which is the failure the
   series exists to stop.
2. **`testrun.py failed` emits invalid node-ids.** The `[teardown]` suffix
   rides into the output; fed back to pytest it is a usage error, while the
   docstring promises "feed back to pytest".
3. **The head line stops summing.** A failing call plus a failing teardown
   gives `2/2 ran | pass 1  FAIL 3`.

### Two same-class holes still open

* `progress_plugin.pytest_configure`'s `except OSError: _STATE["path"] = None`
  silently disables the writer for a whole run, and `cmd_status` returns **0**
  for `no-data`. A batch that measured nothing exits clean.
* The env canary's *"DISARMED"* notice goes through `warnings.warn`, and the
  plugin has no `pytest_warning_recorded` hook — so in `status`, a run where a
  canary proved **nothing** is indistinguishable from one where it proved
  everything. `conftest.py`'s own comment demands the opposite.

### One note for the other machine

**The commit messages' cited artifacts do not reproduce.** `all.jsonl`'s first
record is 2026-09-04, not 2026-09-11, and it ends `done exit 0` — so it reads
`done` and never reaches the liveness code the commit fixes. The "7812 of 9530
records, no `done`" file holds 9530 records ending `done exit 0`, mtime
*before* the commit. **The defects are real and were reproduced on fixtures;
the artifacts named as evidence are not the ones that show them.**

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

## 0a. Most of § 1 is ONE condition, not twelve findings

*(Added 2026-09-23, after a systematic read replaced a piecemeal one. The
first five items were walked individually and each produced a local fix.
Read together they are instances of a single condition, and the local fixes
were in several cases the wrong fix.)*

**THE CONDITION: a rule has an owner, and the call site re-derives it.**

| § | the rule | its owner | what the call site does instead |
|---|---|---|---|
| **1.1a** | how a sidecar becomes bytes | `StructureCodec.pair()` | returns `sidecar: dict` (unrendered) beside `document: str` (rendered), so three consumers finish the job at three different moments — one of them after the `.xyz` is already on disk |
| **1.5** | reading a stored structure from a path | `StructureCodec.load` | `/api/structure/analyze` reads and parses the file itself: no `encoding=`, its own suffix dispatch, and the sidecar never read |
| **1.6** | is the cell around the atoms | `cell_contains_atoms(resolve_cell_origin())` — **2 callers** | `inv(cell); positions @ inv` with no origin term |
| **1.8** | what a structure path is, and where its companion lives | the codec | three spellings (`files.py:257`, `selection.py:99`, an inline error string), and rename gates the source end only |
| **1.12** | four rules at once — element from a label, the `axis_kind` default, image distance, cell derivation | `resolve_element`, `Structure.__post_init__`, `_min_image_distance`, `resolve_cell` | **eleven** hand-written `axis_kind` fallbacks plus three hand-rolled geometries |

**Why it keeps happening, in the words of one of the sites.** `transiesta.py:235`
carries a careful comment explaining why its `axis_kind` fallback was changed:

> *"The fallback AGREES WITH ITS TEN NEIGHBOURS. It arrived as the literal
> swap … and said `("periodic",) * 3`, while every other `axis_kind or …` in
> the tree … answers `("isolated",) * 3`."*

The author compared the copy against **ten other copies** and never against
the owner. `__post_init__` resolves a missing `axis_kind` to `periodic×3`
**when a cell is present** — and this site sits inside `if cell is not None:`.
The original value was right for its branch and was changed to the wrong one
to match the neighbours. **Careful work, wrong reference point.** Nothing
breaks today only because all eleven are unreachable.

**What follows for the fixes.**

1. **Fix the owner, then delete the copies.** Never fix the instance — that
   is how a code-vs-contract finding gets closed by creating a
   code-vs-contract disagreement. (The documentary form of this rule is
   already standing practice; this is its code form.)
2. **Prefer deletion.** § 1.12b's eleven fallbacks cannot fire; removing them
   is the whole fix. *delete > one home > parameter > abstraction.*
3. **Before writing any arithmetic, ask whether the rule has a door.**
   § 1.12c nearly gained a hand-written `|a·(b×c)| / |b×c|` — a sixth copy —
   while `_min_image_distance` already answers that question exactly and is
   already imported two files away.
4. **CHECK THAT A SYMBOL EXISTS BEFORE BUILDING ON IT.** An earlier draft
   of this bullet said *"`cell.image_distance` has zero callers — either the
   fix gives it one, or it is residue"*. **There is no such function**
   (`git log -S "def image_distance" --all` is empty; `cell.py` has 17
   `def`s and that is not one). It is an Issue `where` **id**, emitted by
   `_min_image_distance` at `validation/geometry.py:164` — and `cell.py:38`
   says so in the same breath it is quoted from. A whole dilemma was built
   on a phantom, from reading a `where` id as a call. This belongs with § 2's
   never-existed symbol names, not here.

**Two more conditions, smaller, named where they were found:**

* **§ 1.13 — STATED STATE OVERWRITTEN BY DERIVED STATE.** `load()` returning
  `SCHEMA_VERSION` instead of the version the file states; `write()` deleting
  a sidecar nobody asked it to delete. The code substitutes its own answer
  for a fact the file or the user stated.
* **§ 1.15 — A PLACEHOLDER SPELLED AS A LEGAL VALUE.** rdkit's per-atom
  fallback to residue `1` / `"MOL"` / `"A"`, which are also perfectly
  ordinary values, so nothing downstream can tell "unknown" from "residue 1"
  — and `_DEFAULT_EN = 2.20`, which is **hydrogen's exact Pauling value**, so
  an unrecognised species label is silently treated as hydrogen. A sentinel
  must not be a value the data can legitimately hold.

**The genuinely separate items** — instances of none of the three — are
§ 1.1's half-written pair (an ordering and atomicity question) and § 1.11's
fail-open registry. Those are ordinary bugs.

---

## 0b. ON HOLD — `transport/` and `web/` are being worked on another machine

**Another machine is actively changing the transport module and the web UI**
*(user, 2026-09-23)*. Findings whose FIX lands in `molbuilder/transport/` or
`molbuilder/web/` are **KNOWN GAPS, not work items**, until that lands. They
stay recorded here; nobody fixes them from this side.

This is not caution for its own sake — a push from this session was rejected
on 2026-09-23 because that machine had already moved `main` six commits.
Fixing into those two trees is how the same collision becomes a merge
conflict in code instead of in git.

| § | the fix lands in | status |
|---|---|---|
| **1.3** | `transport/citation_defaults.py` — the dropped form-B recorded contract | **done on the other machine** (`764addd3`) |
| **1.5 / 1.5a** | `web/blueprints/build.py` — analyze onto the codec | **HOLD** (§ 1.5a's `structure.md` § 2.4 doc fix is NOT held — see below) |
| **1.12d** | `transport/transiesta.py` — the fabricated box, `_compute_cell_from_extents` | **HOLD**; the fabricator was rewritten there, the origin skip remains, and reachability is a definition question settled in § 1.12d item 3 → § 6 item 2 |
| **1.6** | split: the OWNER is `validation/__init__.py` clause F4; the INSTANCE is `validation/siesta.py`, reached from `web/blueprints/build.py:1088` | **owner fixable now**, web-side behaviour to be re-checked after |
| **1.8a** | `web/blueprints/files.py` + `selection.py` — rename structure vs rename file | **HOLD** |
| **1.12b** | ten of the eleven `axis_kind` fallbacks are outside those trees; **`transiesta.py:235` is inside** | **do the ten, leave that one** |

**The hold worked.** The merge of 2026-09-23 brought 29 files from that
machine against 22 changed here with zero overlap; every disagreement it left
is semantic and is recorded in § 1.17.

**What is NOT held, and can proceed:** § 1.1 / 1.1a (the codec generator and
the PySCF deck writer), § 1.2 (the CLI + `validate`), § 1.4, § 1.7, § 1.8b–d
(the codec's delete rule, the hex gate, `to_xyz`'s title), § 1.11 (the
validation registry), § 1.12a and § 1.12c, § 1.13, § 1.15 (the builders), and
**every document fix** — including § 1.5a's correction to `structure.md`
§ 2.4, which is a contract error that would otherwise instruct a regression
whoever reads it.

**A note on § 1.8's shape.** Its resolution splits *rename file* (generic,
web) from *rename structure* (owns the pairing rule). The **rule's home** —
one predicate for "is this a structure path", replacing the three spellings —
is in the codec and is NOT held. Landing that first makes the web-side change
smaller and is the half that stops the rule drifting again.

---

## 0d. The goal of this revision *(user, 2026-09-23)*

**Every parameter explicit at every step** — `engines/template.md` § 6.6.
Declared once with its source and scope; recorded with its source; shown
everywhere and edited in one place; explicit in every deck; traceable on
every road. § 1's findings and § 6's decisions read against it: D-1 is
obligation 1, the shared panel and the per-rung echoes are obligation 3, the
fabricated box and the silent `None` items are obligation 4, W20 is
obligation 5.

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

**The OVERWRITE case is worse, and was missed on the first pass.** Re-measured
2026-09-23 on an existing pair (atom moved, one region deleted, a NaN in
`info`):

```
v1 regions: {'L': [0], 'R': [1]}
v2 write:   RAISED ValueError: Out of range float values are not JSON compliant
reopened positions[0]: [5. 5. 5.]             <- the move STUCK
reopened regions:      {'L': [0], 'R': [1]}   <- the DELETED region is back
sidecar hash 463e2ab1…  vs  actual xyz b231ba5…   MATCH? False
```

Half the edit survived. The save was loud; the reopen was silent.

**Why the second guard does not fire.** `structure-molstruct.md` § 3 specifies
**two** guards, "deliberately kept separate": the atom count, and
`structure_hash`. Measured — `count changed → REFUSED MolstructPairingError`,
`atoms moved → ACCEPTED`. The count guard is wired; the hash guard is assigned
to "the caller" by `sidecars/molstruct.py:598` and **no caller exists**. A
responsibility named, delegated, never picked up.

### 1.1a The root cause is the GENERATOR, not the write door

`StructurePair` promises *"what a Structure looks like OUTSIDE MEMORY … ONE
shape for every consumer"* and then declares `document: str` (rendered) beside
`sidecar: dict` (**not** rendered). A dict is not an outside-memory form, so
every consumer finishes the rendering itself, at a different moment:

| consumer | renders the sidecar | outcome |
|---|---|---|
| `files()` | immediately, before returning | correct |
| `write()` | inside `molstruct.save`, **after** the `.xyz` is swapped | the half-write above |
| spliced deck writer (`pyscf/input.py:1551`) | at runtime, with its own settings | see below |

**The error handling is not uniform because the PRODUCT is not uniform.**

The deck writer is a third implementation of the pair write, which § 2.4's
*"every structure↔bytes translation goes through the codec"* forbids. It omits
`ensure_ascii=False`, `allow_nan=False` and `encoding=`; it has no
temp-and-rename; and it carries JSON through a **Python-literal** channel
(`f"_MB_SIDECAR = {sidecar!r}"`). Verified: `repr({'a': float('nan')})` is
`{'a': nan}`, and `nan` is not a Python name — so a non-finite value kills the
deck at **import**, with `NameError`, before any SCF.

**RESOLUTION (user, 2026-09-23) — one change at the generator:**

1. **`pair()` renders BOTH halves.** `StructurePair.sidecar` becomes the
   rendered text. Every serialisation failure then happens once, inside the
   generator, before any consumer can touch the disk. This is the uniform
   error handling, and it arrives by construction rather than as a check added
   at each door.
2. **`write()` stages both temps, then does both renames.** No pre-check
   needed — both halves arrive final. Disk-full and permission errors land at
   the temp write, before either rename. The residual window is a crash
   between two adjacent `os.replace` syscalls, which POSIX cannot close
   without renaming a directory and is not worth the machinery.
3. **Delete the `atomic=False` branch.** Zero callers in `molbuilder/` or
   `tests/` — and it is the branch that would keep the defect.
4. **The deck writer takes the codec's own JSON text**, and `molstruct.dumps`
   is spliced for the re-emit it needs after patching `structure_hash` and
   `title` — the `homo_index` / `dipole_derivatives` pattern, and what
   `dumps`'s own docstring demands: *"Two serialisers is two answers to 'what
   does this sidecar look like'."*
5. **Correct `structure.md` § 2.4's sentence** in the same commit. It states
   the order exists *"so a reader never sees new geometry with a stale
   sidecar's atom indices"* — which is exactly what geometry-first produces.
   The code's own docstring is honest (*"the only visible interleaving is
   OLD-sidecar + NEW-geometry"*). Nothing pins the order: reversed, 95
   targeted tests stayed green, so add the test with the fix.

**Rejected, and why.** A *rollback* after a failed sidecar write (do the
damage, undo it) needs the process to survive; pre-rendering means the damage
never happens. *Sidecar-first* ordering does not remove the window, it moves
it, and yields stale coordinates with nothing announcing them. A
*`structure_hash` check on read* would refuse a hand-edited `.xyz` whose
labels are still perfectly valid, which `structure-periodicity.md` § 8.2
protects (*reading does not judge*: the file opens, the doors that ACT
refuse, and the report has to arrive). If that guard is ever built it belongs
at the doors that ACT, not at `read`. *(An earlier draft cited a
`structure.md` § 8.2 that does not exist, quoting a sentence found nowhere;
the consolidation pass of 2026-09-23 caught it. The nearest real words are
`science/junction-cell.md`'s "the box is the author's to set", which is about
the box.)*

**This one change also closes** the design document's § 15.6 items: the deck's
escaped non-ASCII region labels, its `NameError` on a non-finite value, and its
missing `encoding=`. They are not separate fixes — they are the same
half-rendered product, finished wrongly by a third consumer.

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

**The mechanism, read 2026-09-23.** The CLI picks between two different
functions on whether `--engine` was given:

```python
if engine == "siesta":   validate(struct, SiestaConfig(), cell=_cell)
elif engine == "pyscf":  validate(struct, PySCFConfig())
else:                    validate_geometry(struct, cell=_cell)   # <- a DIFFERENT function
```

`validate_geometry`'s own docstring: *"Cell-dependent checks (volume / image
distance / determinant) are **skipped when `cell is None`**."* `validate()`
would have derived the cell itself under clause F4. So the no-engine branch
misses the cell checks twice over — it never reaches F4, and it may be handed
`None`.

**`validate_geometry` is not the problem and stays.** It is a COMPONENT of
`validate` (`validation/__init__.py:248` — `issues += validate_geometry(...)`),
and the web's `_shared.py:408` uses it to build the structure DESCRIPTION
payload with `cfg=None` passed deliberately. That is a description surface,
and `structure-periodicity.md` § 8.2 says descriptions do not judge. *(Its docstring's claim that it is
for "the web Build page … before they even pick SIESTA vs PySCF" is stale —
that is not what the live caller is.)* The defect is the CLI reaching for it
as a GATE.

**RESOLUTION (user, 2026-09-23) — the fallback is fine; the SILENCE is the
defect.** `science/validation.md` clause F4 already states the rule: *"A check
that cannot run says so as `info`; **silence is never the answer**."*

1. `--engine` is the normal input; the documented pre-submit pipeline always
   has one.
2. Without it the command still runs, and runs **everything
   engine-independent** — including the cell checks, because F4's
   `resolve_cell()` derivation needs no engine.
3. It emits an `info`: *"no engine given — engine-specific configuration
   checks (pseudopotentials, basis, k-grid) did not run."* A green exit can
   then not be mistaken for a full pass.

**The `info` is emitted INSIDE `validate()`, on the `cfg is None` path** (user,
2026-09-23), not at the CLI seam — F4's words are *"never an argument a caller
can forget"*, so every caller that omits a config gets the same treatment.
Note this reaches `_shared.py:408`'s wire payload, pinned by
`tests/test_workflow_group_wire_contract.py::test_cfg_none_path_correctly_omits_workflow_group`
— the test named for exactly this path. The line is true there too, and a
caller that does not want it filters `info`; see § 5b.

**Rejected:** requiring `--engine` (blocks a legitimate "is this geometry
sane" run on a file whose engine is undecided), and a bare
`validate(struct, None)` with no `info` (keeps the silence, which is the
actual defect).

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

**Fixed on the other machine, 2026-09-23** (`764addd3`): `citation_defaults.py`
reads a saved structure's recorded contract through `_FROM_RECORD` beside
`_FROM_DECK`, both through one `_apply_kgrid`; `tests/test_citation_brings_its_settings.py`
pins it. Their measurement of the consequence is the one to keep: a pair
recording 400 Ry / TZP / 4×4 produced a template of 300 Ry / DZP / Γ-only,
which gives a poorly defined E_F and puts every transmission feature at the
wrong energy. The cause above stands as the record of *why* it was lost.

### 1.4 The documented CLI pipe destroys the pair and asserts the loss

```
dev.molstruct.json    L-electrode, frozen_atoms · cell set · origin (-4,-4,-1) · periodic,periodic,transport
piped2.molstruct.json regions [] · cell NULL · origin None · isolated,isolated,isolated
```

`modify`'s own `--help` gives this two-stage pipe. The stdout destination
writes `to_xyz()` — bare XYZ — in the same two functions whose *file* branches
were fixed to use the codec. The loss is then **materialised as a positive
claim**: the second stage writes a sidecar saying the junction is a vacuum box.

### 1.5 `/api/structure/analyze` 500s on a file it should describe

**What the user does.** Picks a structure in the browser. molbuilder
auto-detects what it is — the detection badge (`lib/detection-chip.js`) and
the **pre-filled calculation form** (`lib/form-schema.js`, reading
`suggested.<engine>`) both come from this one reply (`lib/auto-detect.js`).

**What happens.** A PDB saved in latin-1 (a degree sign in a REMARK is enough)
returns **HTTP 500 with an HTML body** where the page expects JSON. The
sibling door `/api/build/load`, given the same bytes, returns a clean JSON
400 naming the problem. `web-api.md` § 1 is explicit — *"400 | bad body /
validation / parse failure"* — so the route disagrees with the contract AND
with its neighbour.

**The immediate cause**, `build.py:236`:

```python
    text_in = p.read_text()          # OUTSIDE the try
    ...
    try:
        struct = Structure.from_pdb(text_in) / from_xyz(text_in)
    except (ValueError, IndexError) as exc:
        return ... 400
```

`UnicodeDecodeError` ⊂ `UnicodeError` ⊂ `ValueError`. **The route's own
handler would have caught it.** The read is one line too high. The read was
never inside the `try` — `18ad15ad`, 2026-05-23, the route's first commit.
Original placement, not a refactor artifact.

**The real cause is § 1.8's missing door, a third time.** The route holds a
PATH and parses it itself, so it re-derives three things the codec owns and
gets each slightly wrong:

1. reads with **no `encoding=`**, where `StructureCodec.load` uses
   `utf-8-sig`. Measured: a BOM'd UTF-8 `.xyz` gives a confusing
   *"invalid literal for int() … '﻿3'"* here and opens fine there.
2. dispatches `.pdb`-vs-else — a **fourth** spelling of § 1.8's
   structure-suffix rule.
3. **never reads the `.molstruct.json`** — so the answer that pre-fills your
   form is computed while ignoring the electrode regions and the cell in the
   file beside it.

It is the only route in the web layer that takes a path and parses it itself
(`build.py:825` takes upload TEXT, which is correct — `from_xyz` refuses a
path by design). One site, not a pattern.

**RESOLUTION (user, 2026-09-23): route it through the codec.**

```python
    try:
        struct = StructureCodec().load(p)
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False, "error": f"could not load structure: {exc}"}), 400
```

One line replacing four, and the existing handler is already the right one:
`MolstructJsonError(ValueError)` and `MolstructPairingError(MolstructJsonError)`
are both `ValueError` subclasses, so a bad sidecar becomes a 400 rather than a
second 500.

**This costs nothing** — see § 1.5a. An earlier draft of this section warned
that routing through the codec would make `analyze` inherit a periodicity
refusal. That refusal does not exist.

### 1.5a Three places claim `load` refuses a bad cell. It has not since 2026-08-03

`StructureCodec.load` contains **no periodicity gate**. The two mentions in
the module are the header claiming one, and the comment recording its removal:

> *"READING DOES NOT JUDGE (`structure-periodicity.md` § 8.2, decided
> 2026-08-03). … It used to raise here, and that made such a file unopenable
> and therefore UNFIXABLE: the Cell page is the one place the box can be
> corrected, and it cannot be reached without the structure on screen. …
> NOTHING IS LEFT UNGUARDED BY THIS … the box is stopped at every door that
> would ACT on it, and at none that would merely show it."*

Three statements are stale:

| where | what it claims |
|---|---|
| `workingcopy_structure.py:13` | *"…both-or-neither atomicity on write, **the periodicity gate on read**"* |
| the `read` docstring | *"Runs the periodicity gate to **REFUSE** a cell nothing can be done with"* |
| **`docs/model/structure.md` § 2.4** | quotes that docstring verbatim |

**Why this one matters more than an ordinary stale comment.** It describes a
guard as present when it was removed *on purpose*, with the trap it caused
written down. A reader comparing § 2.4 against the code concludes the gate is
MISSING and restores it — re-creating an unopenable, unfixable file. The
contract is actively instructing a regression.

**This is the SECOND wrong statement in § 2.4**, after § 1.1a's write-order
sentence. Both are in the four-clause "what it owns" block. When § 2.4 is
edited for § 1.1a, fix both, and re-read the other two clauses rather than
assuming they are sound.

### 1.6 A false containment warning, on a knob that changes nothing

> **This is a fifth instance of § 1.12's shape** (noted 2026-09-23).
> `validation/siesta.py:798-810` is `inv = np.linalg.inv(cell); frac =
> struct.positions @ inv` — geometry on a RAW cell with **no origin
> term** — while `Structure.cell_contains_atoms(resolve_cell_origin())`
> is the door for exactly that question and has **no external callers at
> all** — its two call sites are both inside `structure.py`, in
> `_derived_corner_under_explicit_cell`. That makes this finding stronger
> than "2 callers" suggested. Fix
> it by calling the door, not by adding an origin subtraction: under
> clause 2a a derived corner contains the atoms by construction, so
> bare arithmetic makes the check nearly unfirable. Read with § 1.12
> and fix in the same pass.

`validation/siesta.py:798` combines `struct.positions` with a cell anchored at
`(0,0,0)`, ignoring `resolve_cell_origin()` and `axis_kind`. Measured: the
preflight panel warns *"1 of 4 atoms have fractional coords outside [0,1)"*
while `cell_contains_atoms(resolve_cell_origin())` is `True` and the emitted
deck has no such complaint. **And `cfg.wrap_into_cell` is inert** — no
production caller passes `cell=` to `spec_for`, so the decks for
`wrap_into_cell` True and False are bit-identical. A browser checkbox that
produces a warning contradicting the deck and no change to the deck.

**This is a further site of the origin rule that the 2026-09-21 sweep did not
reach.** `handover.md` § 2.1 — the record — tabulates **three** fixed defects
(`transport/compose.py` form A, the `siesta/input.py` deck writer, the Results
tab), and this session added a fourth in `cli.py::_apply_run_metadata`. An
earlier draft of this section said "the sixth site; five were found" — a count
that does not reconcile with the record it cites, and is withdrawn rather than
replaced by a second guess. Why the sweep missed it: every swept site *applies*
a `cell_origin`, so a reference scan finds them; this site **omits** it, and the
line contains no such token.

### 1.7 A malformed annotations channel escapes as a bare `KeyError`

`AtomChannel.from_json` does `obj["kind"]`; every layer catches
`(ValueError, TypeError)`. So a hand-edited sidecar gives
`KeyError: 'kind'` with **no file path**, through a door whose whole stated
purpose is *"fails loudly at load time with the file path in the error"*. The
deck format makes the opposite mistake and drops the channel silently.

---

### 1.8 The pair's OTHER doors — one rule, reinvented three times

Same shape as § 1.1a. There the codec handed out a half-rendered product and
three consumers finished it differently. Here the codec never exposes
**"which files are structure files, and where is a structure's companion"**,
so everyone who needs it writes their own copy:

| where | what it spells |
|---|---|
| `web/blueprints/files.py:257` | `_STRUCTURE_SUFFIXES = (".xyz", ".pdb")` |
| `web/blueprints/selection.py:99` | `_SUPPORTED_STRUCTURE_SUFFIXES = (".xyz", ".pdb")` |
| `workingcopy_structure.py` `load` | `"expected .xyz or .pdb"`, inline in an error string |

`_paired_sidecar_path`'s docstring caught half of this on 2026-09-18 and fixed
the duplicated **constant** (`.molstruct.json`), delegating to
`sidecar_path_for`. The **policy** stayed in the blueprint. § 1.8a is what
that costs.

#### 1.8a Rename strands the labels — the destination is not gated

`/api/files/rename` already pairs the sidecar, with rollback, and its comment
states the purpose: *"Otherwise renaming `water.xyz` to `bridge.xyz` orphans
`water.molstruct.json` — next load can't find the sidecar from the new stem;
the user's labels silently"* [disappear]. Measured:

```
structure suffixes the SOURCE is gated on: ('.xyz', '.pdb')
  rename -> notes.txt   sidecar would go to: notes.molstruct.json
source .xyz has a paired sidecar?                True
after rename to .txt, is it still pair-tracked?  False
```

`_existing_paired_sidecar(src)` gates on `_STRUCTURE_SUFFIXES`;
`_paired_sidecar_path(dst)` does not. Rename `water.xyz` → `notes.txt` and the
sidecar is moved to `notes.molstruct.json`, where **nothing will ever read
it** — the same outcome the feature exists to prevent, by a route the fix did
not cover.

**RESOLUTION (user, 2026-09-23) — two menu items, two layers.**

* **Rename file** stays generic. It renames one file and knows nothing about
  structures. *"It is not a xyz/json serving tool, it is a file writing
  tool."* Teaching it the pairing rule would be a second implementation of
  that rule, which `model/structure.md` § 2.4 forbids by name.
* **Rename structure** owns the pairing rule: **load through the codec, write
  the pair at the new name, remove the old.** Both ends gated by the ONE
  structure-path predicate the codec now exposes. Validation comes free — the
  read path already refuses out-of-range regions and empty labels (measured).

**Why load-and-write and NOT "stamp a default sidecar when none exists."** A
default sidecar is not a blank file; it is six statements, including
`"cell": null` and `"axis_kind": ["isolated","isolated","isolated"]`. Measured
against an extended XYZ carrying its own `Lattice=`:

```
no sidecar:            cell = set    axis_kind = ('periodic','periodic','periodic')
with default sidecar:  cell = None   axis_kind = ('isolated','isolated','isolated')
```

A "default" sidecar **destroys the box** the geometry file declared. Loading
and re-writing does the right thing on both inputs: the periodic file gets a
real pair with its cell recorded, and a plain molecule with nothing to say
gets the `.xyz` alone.

#### 1.8b `write` deletes a sidecar the user never asked it to delete

```python
keep_sidecar=(not _metadata_is_default(meta) or bool(identity) or bool(struct.info))
...
if made.keep_sidecar:        molstruct.save(sidecar_path, made.sidecar)
elif sidecar_path.exists():  sidecar_path.unlink()          # <- removes a user file
```

The stated reason (`write`'s docstring) is *"so the pair can't disagree"*: if
you clear every label and save, skipping the write would leave the old sidecar
to re-apply them.

**That reason justifies REWRITING the sidecar, not deleting it.** A sidecar
saying `{"regions": {}, …}` records "no labels" equally well and destroys
nothing. The only thing deletion adds is canonical form — and the rule it
serves has two halves, of which only one is forced:

* **"absent means empty"** — FORCED. A plain `.xyz` from ASE or VMD has no
  sidecar, so absence must mean something, and empty is the only sane reading.
* **"empty must be written as absent"** — NOT forced. A tidiness preference,
  and the half that deletes a file.

**The hazard is not theoretical.** The branch fires on what the in-memory
Structure holds, and the codec cannot distinguish *"the user cleared the
labels"* from *"this code path loaded the `.xyz` without its sidecar, or
dropped the metadata in transit"*. Same branch, opposite meanings — and this
audit is largely a catalogue of metadata lost in transit between layers.

**RESOLUTION (user, 2026-09-23): NEVER DELETE AN EXISTING SIDECAR.** Two
clauses, and an earlier draft of this section stated only the first, which
contradicted § 1.8a:

* **A sidecar that exists is rewritten, never removed** — with its current
  content, empty if that is what the metadata now is. The pair then still
  cannot disagree (both halves say "no metadata"), which is the invariant the
  deletion was defending; it is met by rewriting instead of by unlinking.
* **A sidecar that does not exist is not created for a structure with nothing
  to say.** Absence correctly means "empty" — foreign `.xyz` files from ASE or
  VMD depend on that reading, and § 1.8a's load-then-write rename relies on it
  (`plain input -> ['plain2.xyz']`, measured).

So the codec stops **removing** a file the user did not mention, without
starting to **litter** one beside every bare `.xyz`.

#### 1.8c The hash gate claims hex and checks length

```python
if not isinstance(structure_hash, str) or len(structure_hash) < 16:
    raise ...(f"structure_hash must be a hex string (got {structure_hash!r})")
```

**Three** spellings, not the two an earlier draft named:
`sidecars/molstruct.py:371` (`to_dict`), `parse/sidecars/molstruct.py:79`
(`_normalised_dict`) and `parse/sidecars/molstruct.py:210` (`load_text`,
which words it differently — *"hex string of >= 16 chars"*). § 3's own table
had the right count (*"3 spellings, 2 different messages"*) and this section
under-reported it.
Measured: a sidecar whose hash is `"not a hash at all!!!"` **loads**. So does
`"/etc/passwd\n\n\n\n\n\n"`. `structure-molstruct.md:67` documents the field
as *"hex, ≥16 chars"*, so **adding the hex check needs no document change** —
it makes the code do what its own error message and the contract already say.

**But only ONE of the three can fire on real data** (consolidation pass,
2026-09-23). `_normalised_dict` has exactly one caller, `load_text:251`,
which already ran the same test at `:209` — a dead gate behind a live one.
`to_dict` has exactly one production caller, `StructureCodec.pair`
(`workingcopy_structure.py:270`), which always passes a real digest; its gate
fires only for hand-built callers. So *"add the hex check at three sites"*
would triplicate a check that fires at one. **Fix shape: one predicate, two
doors** — the write door (`to_dict`) and the read door (`load_text`) call it;
the copy inside `_normalised_dict` goes.

Tightening to exactly 64 characters is a **separate contract decision** and
is not proposed. *(A count is a hypothesis: this section said 13 fixtures
conform to "≥16 hex"; the consolidation pass re-derived 22 failures + 1 error
across 18 files under a 64-hex gate. Different file sets — re-derive before
acting on either.)*

**Consolidated with the other machine's handover § 5.1** (2026-09-23), which
is the same field and a *different* finding: the value is **compared
nowhere** — every read of `structure_hash` in `molbuilder/` is a writer, a
shape gate or prose, and the only comparison in the tree is a fixture pin in
one test. Proved orthogonal by mutation: with a hex check at every gate, a
valid-but-stale digest still loads with its labels applied; a comparator
would not make `"z"*16` refuse at build time. Verified beside it:

* *"refused, never mis-applied"* occurs once in the tree —
  `structure-molstruct.md` § 1's envelope row for `n_atoms_total`. § 3,
  which owns the hash, says the caller compares it *"to detect"*. The
  `MolstructPairingError` docstring (`sidecars/molstruct.py:156`) cites the
  phrase as § 2's and extends it to the hash — comment-vs-contract, and the
  comment is the wrong one. So their *"detect, never refuse"* and § 3 already
  agree, and agree with `structure-periodicity.md` § 8.2 and with § 1.1a's
  *"not at `read`"* above. The two guards are not equivalent: a count
  mismatch is binary evidence of misaligned indices; a hash mismatch is
  weaker (a tool that reformats identical coordinates changes it) **and** the
  only detector for a row swap, which keeps the count equal and re-labels
  different atoms.
* Their premise *"the pair is written atomically, so molbuilder cannot leave
  the halves out of step"* is **false at HEAD** — that is § 1.1, re-measured.
  So § 1.1a's generator fix lands **before** any attestation door, or the door
  asks a person to attest to damage molbuilder did.
* *Attest by re-saving* is not an option: `to_xyz` writes six decimals, so a
  hand-written `1.2345678901` comes back `1.234568` — re-saving truncates a
  relaxed geometry. An attestation door writes the **sidecar only**
  (`molstruct.save` under `with_lock` exists for it).
* `spectra/results.py:523` carries the same *"so the parser can refuse"*
  sentence for the spectra hash, and nothing refuses there either.
* `Structure.mark_contract_outdated` rules *"never cleared"* while a re-stamp
  would clear the hash; both hold only if the document says why —
  `structure_modified` is a statement about **history**, the hash about
  **identity**.
* Their own `plan.md` X4 ⑤ says *"the likely answer is DELETE"*; their later
  handover § 5.1 says detect + attest, *"the shape the user specified"*. Their
  two documents disagree; the handover is the later one.

**The decision is the user's** (§ 6): delete the field, or build detect-in-
`StructureCodec.load`-as-a-notice plus a sidecar-only attestation door. If the
latter: the RULE goes into `structure-molstruct.md` § 3 first (name the door,
the channel, and that a hash mismatch *reports* while a count mismatch
*refuses*, with the reason), then the two comments above follow it.

#### 1.8d `title` absorbs the extended-XYZ header, and `to_xyz` writes it back

`to_xyz` emits `self.title` as the comment line:

```python
buf.write((comment or self.title or "Built by molbuilder").strip() + "\n")
```

`pair()` calls `struct.to_xyz()` with no comment, and loading an extended XYZ
leaves the whole header — `Lattice="…" Properties=… pbc="T T T"` — in `title`.
So the codec can write a `.xyz` whose comment line advertises a 10 Å periodic
lattice **beside a sidecar it wrote in the same call saying `"cell": null`**.

molbuilder reads it correctly (the sidecar wins). ASE, VMD and every other
tool read the header and see a periodic box. One file, two answers — and the
whole reason for keeping `.xyz` as the geometry format is that other tools
read it.

**Traced 2026-09-23, and it is bigger than this section.** The reader is
`from_xyz:1424` (`comment = lines[1].strip()`) feeding `:1459`
(`title=comment`) — the whole comment line, no keyword stripping. And the
sidecar MASKS it: same file, sidecar present → `'my junction'`, sidecar moved
aside → `'my junction Lattice="10.000000 …'`. So it is invisible in normal
use and shows up only when the pair is separated, or when another tool reads
the `.xyz`.

**FIXED 2026-09-23** — `model/structure.md` § 2.2c. The owner is settled
(the geometry file), `from_xyz` now cuts the structural keys, the sidecar
stopped carrying `title`, and older files still open via
`RETIRED_IDENTITY_KEYS`. All six defects close, pinned by
`TestTitleBelongsToTheGeometryFile`.

### 1.9 Withdrawn: `/api/files/write` is not a finding

An earlier list had this as *"the browser can author a sidecar, bypassing every
gate."* Both halves are wrong.

* It is a **file writing tool**. A generic text-write endpoint that special-cased
  `.xyz`/`.molstruct.json` would be the second implementation of the pairing
  rule that § 2.4 exists to prevent.
* And the gates are not bypassed. Measured, hand-authoring a sidecar through
  it and loading the result: *region index out of range* → **REFUSED**, *empty
  label* → **REFUSED**, *nonsense hash* → LOADED (which is § 1.8c, not this).
  The read path re-validates independently of who wrote the file.

Recorded so it is not re-raised.

### 1.11 Two more checks that silently do not run — the registry seam

Same family as § 1.2, different mechanism. `science/validation.md` § 7 owns
the rule: *"A registry keyed on a class is only as live as the callers that
construct that class — which is the difference between `_ENGINE_VALIDATORS`
and `_KIND_VALIDATORS`."*

#### 1.11a `_register_default_engines` fails open

```python
try:
    from ..siesta import SiestaConfig
    _ENGINE_VALIDATORS[SiestaConfig] = _validate_siesta
except ImportError:
    pass                       # <- the row is silently not registered
```

Both rows. If the import fails, `validate()` finds no engine validator and
runs only the generic checks — **every engine-specific check gone, nothing
said.**

**This is not an optional-dependency guard.** The function's own docstring
says it is *"Late binding to avoid an **import cycle**"*, and neither import
reaches an optional package: `molbuilder/pyscf/__init__.py` imports its own
`.input`, and `..siesta` is always present. The handler can therefore only
fire when something is genuinely broken, and it responds by hiding it.

The same failure SHAPE is recorded twice in the comments beside it —
`SpectraConfig` retired 2026-08-22 (*"a class nothing in production ever
constructed"*), `TransportConfig` retired 2026-09-17 (*"it dispatched for
nothing"*). Three routes to one outcome; two were closed.

**RESOLUTION (user, 2026-09-23): register a STUB**, not a raise and not a
`pass`. The stub returns one error Issue — *"the SIESTA validator could not be
loaded"* — so a broken import surfaces at the door that refuses, and the seam
stays visible in the framework for further development rather than being
erased by an exception.

#### 1.11b `_validate_vibration_kind` puts class-keyed dispatch back inside the kind registry

```python
from ..config.pyscf import PySCFConfig
if not isinstance(cfg, PySCFConfig):
    return []          # the seam refuses non-PySCF vibration by name
```

`_KIND_VALIDATORS` exists **because** class-keyed science silently skips — the
comment directly above this registration retires the `SpectraConfig` row for
exactly that. The kind validator's first line reinstates it.

**Inert today** (nothing constructs a non-`PySCFConfig` reaching vibration;
`science_view` is a `PySCFConfig` adapter) and **already scheduled**: design
§ 18 **step 0a** — make the guard `raise`. Recorded here only so the three
members of this family are in one place; do not fix it twice. `validation.md`
§ 7 gains one sentence with it: a kind validator does not gate on a config
class.

**Provenance note.** `validation/spectra.py:20-27` records this gate *"silently
skipping between P1 and P3"* for the adapter reason — so this failure mode has
already been paid for **in this very gate**, not merely in the two retired
rows.

### 1.12 ONE SHAPE: a rule with an owner, re-derived at the call site

*(Rewritten 2026-09-23 as a systematic item. It began as two local findings;
the user stopped the piecemeal pass — "read full code for a holistic
systematic fix rather than local patching without knowing what's going on" —
and the systematic read gives a materially different answer.)*

**What was read.** All 41 sites that touch `cell` / `cell_origin` / `vacuum` /
`axis_kind` outside their owning modules (`structure.py`, `cell.py`), across
12 files, plus the caller counts of the seven resolution doors:

```
resolve_cell 15 · resolve_cell_origin 18 · effective_vacuum 8
cell_contains_atoms 2 · resolve_element 10 · _min_image_distance 1
(`image_distance` was counted here as a door with 0 callers. It is not a
function at all — see § 1.12c's correction.)
```

**Most bare reads are correct** and must stay: asking *"is the cell
explicit?"* (the manual-regime test, `validation/siesta.py:396`), serialising
`axis_kind`/`vacuum` to the wire (`watch.py:267-270`), reporting the user's
own lattice lengths in a message (`validation/pyscf.py:157`),
`_structure_declares_a_box`. Those ask about **stated** state, which is the
raw field's job.

Four sites do **geometry** with raw fields, or re-derive a rule that has an
owner. They are one shape.

#### 1.12a `partial_charges` ignores `resolve_element` — a labelled structure reports 0.0 D

`chemistry.py::partial_charges` looks the element up by its RAW string:

```python
en_i = _PAULING_EN.get(ei, _DEFAULT_EN)      # _DEFAULT_EN = 2.20
```

`O H H` → 1.80 D and the polar-in-vacuum warning. `O1 H2 H3` → every atom
misses the table, all get 2.20, every `delta_en` is 0,
`ionic = 1 − exp(0) = 0`, every charge **exactly 0.0**, dipole **0.0 D**,
warning never fires. Label-blind twice in the same loop: `if "H" in (ei, ej)`
does not recognise `H2`, so a labelled hydrogen gets the heavy-atom cutoff.

**The owner is 1600 lines above in the same file**, with ten other callers:

> `resolve_element` — *"**A species label is the user's; the element is ours
> to derive.** … `Au1` / `Au2` — two gold species carrying different basis or
> pseudopotential — is **ordinary input, not a typo.**"*

`0.0 D` does not read as *"could not compute"*. It reads as *"not polar"* —
suppressing the warning about a real dipole–image artifact.

**FIX: route through `resolve_element`**, and derive the hydrogen test from
the resolved element.

#### 1.12b The `axis_kind` default is re-spelled ELEVEN times, and all eleven contradict the owner

`Structure.__post_init__` owns it, and the rule is **cell-dependent**:

```python
if self.axis_kind is None:
    # A stated cell means a lattice; no cell means a vacuum box.
    self.axis_kind = (("periodic",) * 3 if self.cell is not None
                      else ("isolated",) * 3)
```

Every call site hardcodes `isolated×3` **unconditionally** —
`cell.py:168`, `structure.py:628` and `:750` (**the owning file, twice**),
`periodicity_gate.py:296` and `:465`, `validation/siesta.py:388`,
`validation/pyscf.py:151`, `validation/geometry.py:151`,
`siesta/input.py:855`, `transiesta.py:235` — that is **ten** `isolated×3`
spellings — plus `validation/__init__.py:144` (`or ()`), for **eleven in
total**. An earlier draft called the last one "a twelfth", which made the
heading and the list disagree; the list was right. *(Line numbers drift: the
two in `structure.py` are now `:642` and `:764`.)*

So they are **dead** (`__post_init__` always fills the field, so `or` never
fires) **and wrong if they ever fired**: they would call a structure with an
explicit lattice *isolated* — a periodic calculation silently becoming
gas-phase.

**The failure mode is documented at one of the sites, in the reasoning that
produced it.** `transiesta.py:235`:

> *"The fallback AGREES WITH ITS TEN NEIGHBOURS. It arrived as the literal
> swap for the old `struct.pbc or (True,)*3` field read and said
> `("periodic",) * 3`, while every other `axis_kind or …` in the tree …
> answers `("isolated",) * 3`. All eleven are unreachable (`__post_init__`
> always fills the kinds), but this is the one whose waking would relabel
> every axis in an emitted deck and silence the warning below, so it is the
> one that must not disagree."*

That reasoning checks the **copies** and never the **owner**. And the site
sits inside `if cell is not None:` — precisely the branch where
`__post_init__` says `periodic×3`. **The original value was right for its
branch, and it was changed to the wrong one to match the neighbours.** This is
what re-derivation costs even when everyone is being careful.

**FIX: delete all eleven.** They cannot fire; if one can, that is a bug to
find, not to paper over. A deletion, which is the top of the preference order
(*delete > one home > parameter > abstraction*).

#### 1.12c The k-sampling hint measures the gap in two frames at once

`validation/siesta.py:726-748`. On a `periodic` axis with `k > 1` it fires:
*"periodic images sit ~{gap} Å apart … if a weak image interaction is
deliberate, **carry on**."*

```python
diag_lengths = [float(np.linalg.norm(cell[i])) for i in range(3)]          # LATTICE norms
atom_extent  = struct.positions.max(axis=0) - struct.positions.min(axis=0) # CARTESIAN spans
gap = max(0.0, length - float(atom_extent[axis]))
```

**Two independent frame errors:** (1) `norm(cell[i])` exceeds the
perpendicular interplanar distance on a non-orthogonal cell; (2)
`atom_extent[axis]` is a **Cartesian** x/y/z span indexed by a **lattice**
axis — it subtracts the x-span from `|a|`. The variable name `diag_lengths`
records the assumption; nothing enforces it. Measured: a reported **5.9 Å**
gap where the true perpendicular gap is **0.61 Å** — a reassurance about
images that are practically touching.

**The contract licenses error 1 and does not mention error 2**
(`science/validation.md:410`): *"a skewed cell's axis norm **overstates** the
perpendicular image distance … **Both err on the quiet side**."* Overstating
makes the hint fire MORE, and firing means reassuring — the loud side, and a
false all-clear rather than noise.

**The geometry is the contract's own canonical case.**
`structure-periodicity.md` § 4: *"a periodic sub-block (**e.g. a hexagonal
in-plane pair**) orthogonal to the non-periodic axis."* Two periodic axes at
120° is exactly what breaks it.

**FIX: use the existing door.** `validation.geometry._min_image_distance`
asks the hint's question verbatim and is already imported at
`validation/__init__.py:76`:

> `cell.py:644-655` — *"Distances are taken under the **minimum image
> convention** … checked against the 27 surrounding translations, which is
> **exact rather than merely usually right on a skewed cell**. … it asks
> **"how close does this molecule sit to its periodic copies"** — an artefact
> question."*

An earlier draft proposed writing `|a·(b×c)| / |b×c|` inline. That would have
been a **sixth** hand-rolled copy — the same mistake this section is about.

**Correction, 2026-09-23.** An earlier draft added *"and
`cell.image_distance` has ZERO callers — either this fix gives it its caller
or it is residue"*. **That function does not exist and never has.**
`validation/siesta.py:394` names `cell.image_distance` as *"the right tool"*
for a typed box, and that string is an Issue **`where` id**, not a symbol —
`_min_image_distance` emits it at `validation/geometry.py:164`. So the tool
the comment points at IS `_min_image_distance`, which is this fix. There is
no second door and no residue question.

**The word must change too.** The code comment (`:716`) and the
`validation.md` table call the gap *"the real vacuum, whether or not the
vacuum field was ever set."* `vacuum` is defined (§ 2 line 43) as *"meaningful
only on an `isolated` axis"*, and this hint fires on **periodic** axes. Say
**image separation**.

#### 1.12d The transport no-lattice branch fabricates a box and skips the origin

`transiesta.py:401`:

```python
resolved_cell = cell if cell is not None else struct.cell     # never calls resolve_cell()
origin = (struct.resolve_cell_origin() if resolved_cell is not None else None)
positions = struct.positions - origin if origin is not None else struct.positions
```

A variable named *resolved* that resolves nothing. When `struct.cell` is None
the origin shift is **skipped**, and `_lattice_block` then fabricates a box
with `_compute_cell_from_extents`.

**The fabricator changed shape in the merge of 2026-09-23**
(`transiesta.py:151-197`, other machine): it is now per axis by `axis_kind`,
honours a stated `vacuum`, and refuses a `periodic` axis. It is
`resolve_cell`'s rule re-spelled, and its docstring says why it does not call
the owner: *"its default where nobody chose is 3 Å — right for a molecule in
a box and five times too thin for a lead."* That is § 0a's condition exactly:
one rule, one owner, a copy that differs only in a default — and the fix
shape it implies is a vacuum-default parameter on the owner, not a copy
beside it. **What did not change:** the box is still emitted **diagonal and
anchored at (0,0,0)** while the atoms go out **unshifted in the world
frame**, because the origin line above never runs when `struct.cell` is None.
A structure sitting at x ∈ [50, 56] still gets every atom outside its box
(re-measured on the merged tree by the consolidation pass).

This is the class the 2026-07-29 finding named, quoted in this function's own
docstring: *"Emitting the cell at zero with world-frame coordinates
mistranslated a junction by its origin."* The branch warns loudly that the box
is **fabricated**; it says nothing about the atoms not being **in** it.

**REACHABILITY ESTABLISHED 2026-09-23, and the earlier guess was wrong in
both directions.**

An earlier draft said *"likely inert (transport structures carry explicit
cells) — if unreachable, it is residue and the branch goes, and
`_compute_cell_from_extents` with it."* Both halves are wrong.

**It is unreachable through the production ladder** — `compose_junction`
refuses a cell-less structure on both citation forms (`compose.py:904`,
`:959`, via `_unusable_cell`), electrode rungs get
`ElectrodeModel.as_structure()` which always states a cell, and
`build.py:1193` refuses `calculation="transport"` outright.

**But it IS reachable through the engine seam.** `siesta/input.py:759-770`
routes `calculation == "transport"` to `transport.deck.transport_spec`,
which has **no cell gate of its own** (verified by reading its head). With
`axis_kind` at `isolated×3` — what `__post_init__` gives any structure
stating no cell — `resolve_cell()` SUCCEEDS on the bounding box, validation
passes, and the deck renders with `_compute_cell_from_extents`'s fabricated
box anchored at (0,0,0) while the atoms go out unshifted. For atoms at
x ∈ [50, 56] that is every atom outside the emitted box: the exact
2026-07-29 class quoted in the function's own docstring. *(With
`("periodic","periodic","transport")` instead, `resolve_cell()` raises and
`render_deck` does refuse — so the dangerous case is the DEFAULT one.)*

**And the deletion advice would have broken a live caller.**
`_compute_cell_from_extents` has a second one at
`transport/wizard.py:424`, inside `extract_electrode_model`'s
`device.cell is None` branch — and `as_structure()` then wraps that box in an
explicit `cell=`, so the lead's deck says *"Explicit lattice preserved from
the structure (NOT recomputed from atom extents)"* about a recomputed one.
The helper survives either way.

**So the fix is a cell gate in `transport_spec`** — or routing
`_emit_geometry` through `resolve_cell` / `resolve_cell_origin` so the box
and the coordinates come from one frame — **not a deletion.**

*(Verified here: the route exists, `transport_spec` has no cell gate at its
head, and the second caller is real. The end-to-end deck render with atoms
outside the box is the review's measurement, not independently reproduced.)*

#### The systematic fix, in place of four local patches

1. **Delete the eleven `axis_kind` fallbacks** (§ 1.12b). Pure deletion.
2. **Route § 1.12a through `resolve_element`** and § 1.12c through
   `_min_image_distance`. *(No `image_distance` door to settle — that was a
   phantom; see the correction in § 1.12c.)*
3. **Decide § 1.12d's branch — and the decision is whether a cell-less lead
   is a supported input.** The other machine's handover § 4 puts it plainly:
   `compose` refuses a cell-less citation on both forms, so if the branch is
   unreachable, *"the isolated-electrode path the user asked to keep cannot
   currently be used — a gap rather than a reason to delete."* Supported →
   the branch resolves cell and origin together through the owner, with the
   lead's vacuum default as a parameter of `resolve_cell`; not supported →
   delete the branch and the fabricator with it. **The two machines do not
   disagree about the code**: both say the production ladder cannot reach
   it. They disagree on whether a route that bypasses `compose` — the engine
   seam, or `load_compose_record` without `_unusable_cell` — counts as
   reachable, which is a definition neither document states. Their `plan.md`
   X1 ① and X4 ③ each say this audit's verdict was *"measured wrong"*; it was
   measured on a different definition.
4. **Three document sentences** ride with § 1.12c: `validation.md:410`'s
   *"both err on the quiet side"*, its k-sampling table row calling the gap
   *"the real vacuum"*, and `validation/siesta.py:716`'s comment repeating it.

**The `vacuum` field's meaning does not change.** §§ 4 / 6.2a resolve the
three fields **together, keyed on `axis_kind`** — and § 4's block-orthogonal
scope guarantees an isolated axis is orthogonal to the periodic block, so for
a vacuum axis "extension along the axis" and "perpendicular separation
between images" are already the same number. These fixes give the quantities
the **same** meaning computed correctly in both places, not different ones.

| `axis_kind[i]` | cell extent | origin | `vacuum[i]` |
|---|---|---|---|
| **isolated** | `bbox[i] + 2·vacuum[i]` | `bbox_min[i] − vacuum[i]` | the only kind it applies to |
| **periodic** | commensurate lattice vector (**never** bbox-derived — raises) | `0` | **N/A** |
| **transport** | captured device length + one interlayer spacing | `bbox_min[i]` | `0` |

### 1.13 A SECOND condition: stated state overwritten by derived state

§ 0a names one condition (*a rule with an owner, re-derived at the call
site*). The shape-check of the remaining findings (2026-09-23) turned up a
second, smaller one, with two instances already in this plan: **the code
substitutes its own answer for a fact the file or the user stated.**

* **`parse/sidecars/molstruct.py:124`** — `load` reads the file's
  `schema_version` into `sv` (`:174`), validates it, and then returns
  `"schema_version": SCHEMA_VERSION` — **the module constant, not the file's
  value**. A v7 sidecar is reported as v9. The fact the file stated is
  discarded and replaced by the reader's own.
* **§ 1.8b** — `write` removing a sidecar the user never asked to remove, on
  the reader's judgement that its content is not worth keeping.

Both are defensible one line at a time and wrong as a rule: a reader reports
what it read, and a writer does not delete what it was not asked to. Fix
§ 1.13's restamp by returning `sv`; § 1.8b is already resolved above.

### 1.14 The shape-check of the remaining findings

*(2026-09-23, at the user's instruction: shape-check before walking each item
individually. Four of six collapse into conditions already named.)*

| finding | verdict | home |
|---|---|---|
| **the `replace()` completeness guard over-claims** | **instance of § 0a's condition**, test-side | § 1.12, and § 5b |
| **`load()` restamps `schema_version`** | **instance of § 1.13** | § 1.13 |
| **the electrode check is skipped on vibration decks** | **instance of § 1.11** (a check that silently does not run) | § 1.11 |
| **`n_atoms_total` unbounded → `MemoryError` → HTTP 500** | neither condition; its OUTCOME is § 1.5's family (the `web-api.md` four-bucket contract) | with § 1.5 |
| **no test for the `.pdb` write round trip** | not a code condition — a missing pin | § 5b |
| **the two builder defects** | **not yet read** | — |

**The `replace()` guard, in detail — partial delegation is the trap.** The
test's docstring claims *"COMPLETE BY CONSTRUCTION, not by memory … the check
iterates the LIVE field list rather than a copy of it. A field added tomorrow
is covered the moment it is declared."* It does iterate
`dataclasses.fields()` — the **names** come from the owner. But the fixture is
a hand-written constructor call, so the **values** come from memory. A field
the fixture leaves at its default passes whether `replace()` carries it or
not. Measured: dropping the already-declared `annotations` field from
`replace()` leaves this test **green**, and a different test in another file
catches it. It looks complete-by-construction because one half is.

**The electrode check, in detail — a check behind a gate written for its
neighbour.** `validation/pyscf.py:248-256` puts both
`check_unconsumed_region_labels` and `check_electrode_labels_are_frozen`
inside `if not vibration:`. The comment justifies the gate for the **first**:
*"The vibration kind runs its own copy over the deck's view … so this defers
there."* True for that one — `validation/spectra.py:415` calls it. The second
was placed *"beside it because it is the same question one step further"* and
inherited the gate — but `check_electrode_labels_are_frozen` has exactly two
callers, `validation/siesta.py:533` and this one, and **neither is in the
vibration path**. So on a vibration deck it does not run, and nothing defers
it. A moved lead goes unremarked until the compose-time gate, *"which runs
after the relaxation is paid for"* — the code's own words for why this check
is asked early.

### 1.15 The two builder defects — and a THIRD condition

*(Read 2026-09-23, the last two unscreened findings. One is an instance of
§ 0a; the other names a third condition.)*

#### 1.15a A duplex with `5P` is refused, and the blame is pointed at the user's X3DNA

`build_dna("ds,ATGC", terminal="5P")` raises. Measured: a connectivity
failure at **16.74 Å**, with the message appending `($X3DNA={found.root})` —
telling the user to look at their X3DNA installation.

**It is not X3DNA. It is our own checker, and it does not know chains exist.**
`builders/backends/_common.py:96-127`:

```python
for i in range(struct.n_atoms):
    rid = struct.residue_ids[i]          # chain_ids is NEVER read
    if   n == "P":    P_pos[rid]  = struct.positions[i]
    elif n == "O3'":  O3_pos[rid] = struct.positions[i]
for r in sorted(P_pos):
    if r - 1 in O3_pos:                  # "the previous residue" = rid − 1
        d = norm(P_pos[r] - O3_pos[r - 1])
```

Backbone adjacency is pure `rid − 1` arithmetic. On a **duplex** that reaches
across the strand boundary and measures from the end of one strand to the
start of the other — a real distance, and nothing to do with a broken
backbone.

**Why `OH` and `3P` work and `5P` does not:** `_threedna.py:639` strips the
5′ phosphate for those two terminals, so the spurious pair has no `P` to
measure to. `5P` keeps it, and the check fires. The terminal state is a
red herring; the duplex is the condition.

**`select_chain` sits FOUR LINES BELOW in the same file** and reads
`chain_ids`. The file knows.

**Instance of § 0a's condition:** the rule *"which residues are consecutive
in a backbone"* is owned by the chain, and the checker re-derives it from
residue-number arithmetic alone.

**Two fixes, both needed.** Key adjacency on `(chain_id, residue_id)`. And
stop appending `$X3DNA` to a failure of our own self-check — the variable is
right for a *fiber/rebuild* failure and wrong for a connectivity verdict
computed here.

#### 1.15b rdkit files added hydrogens under residue 1 / `MOL`, and the sidecar records it as fact

`builders/backends/_common.py:33-44` converts RDKit atom-by-atom:

```python
info = atom.GetPDBResidueInfo()
if info is not None:
    residue_ids.append(info.GetResidueNumber() or 1)
    residue_names.append(info.GetResidueName().strip() or "MOL")
    chain_ids.append(info.GetChainId().strip() or "A")
else:
    residue_ids.append(1); residue_names.append("MOL"); chain_ids.append("A")
```

**Per-atom**, so a molecule where RDKit populated residue info for only some
atoms — added hydrogens carry none — comes out **MIXED**: real residues on
the heavy atoms, placeholder `1` / `MOL` / `A` on the rest. Measured: 50 of
129 atoms.

**And the sidecar then records the placeholders as real identity.**
`structure.py:914-921` decides whether identity is worth persisting by an
**all-or-none** comparison:

```python
if self.residue_ids is not None and list(self.residue_ids) != [1] * n:
    out["residue_ids"] = [int(v) for v in self.residue_ids]
```

A mixed list is not all-`1`s, so the test says *"this is real identity"* and
writes the whole list — placeholders included. Your ligand's hydrogens are
persisted as belonging to residue 1 of chain A. Any selection by residue
picks them up; PDB output puts them in the wrong residue.
`selection.py:89` already records the symptom: *"named `"MOL"` and this rule
degenerates to all-or-none."*

#### THE THIRD CONDITION: a placeholder spelled as a legal value

`1`, `"MOL"` and `"A"` mean *both* "no information" and a perfectly ordinary
residue, name and chain. Nothing downstream can tell which it is, so the
all-or-none test is the best anyone can do — and a mixed structure defeats
it.

**The same condition explains § 1.12a's zero dipole.** `chemistry.py:1860`:

```python
_PAULING_EN = { "H": 2.20, "Li": 0.98, ... }
_DEFAULT_EN = 2.20
```

**`_DEFAULT_EN` is hydrogen's exact value.** So an unrecognised species label
is not merely defaulted — it is silently treated as *hydrogen*. That is why
`O1 H2 H3` returns `0.0 D` rather than raising: every atom becomes hydrogen,
every electronegativity difference is zero, and the answer looks like a
verdict instead of a failure.

**The rule:** a sentinel must not be a value the data can legitimately hold.
Where it already is, the fix is to carry "unknown" separately (`None`, a
parallel mask, or an explicit raise — `resolve_element` already raises, which
is why § 1.12a's fix is to route through it) rather than to pick a different
magic number.

### 1.16 `structure.py` read whole — what a file-level read found that section reads did not

*(2026-09-23, at the user's request. Three of this session's own errors came
from reading fragments of this file. All five below are invisible from any
one section.)*

#### 1.16a The module header DENIES a guard that exists

The header said `frozen_atoms` is *"consumed by … the Build SIESTA / PySCF
emitters (**warn-only today** pending the design.md 'fully respected'
rollout)"*. Both emitters now emit real constraints — `%block
Geometry.Constraints` (`siesta/input.py:1317`) and `$freeze`
(`pyscf/input.py:861`). **A reader would conclude that held atoms do not
hold.** Also stale: it described `frozen_atoms` as loaded from a sidecar,
when it has been a designated read of `regions[FROZEN_LABEL]` since the
reserved-label unification. **FIXED.**

This is the inverse of § 1.5a, where a document claimed a guard that had
been removed. Both are the same failure — a contract describing the code of
a different date — and they point opposite ways, so neither would be caught
by a sweep looking only for "missing" guards.

#### 1.16b A constant's comment named a member it no longer has

`IDENTITY_FIELDS`'s docstring opened *"The per-atom IDENTITY columns **+ the
title**"* after `title` left the tuple. **FIXED**, with the retirement named
in place.

#### 1.16c FOUR live citations pointed into `docs/archive/`

`§ 3c` was cited four times — at `cell_origin`'s field comment,
`resolve_cell_origin`'s docstring, `affine` and `translated`, i.e. **every
site that decides the corner**, the cluster this audit spent the most effort
on. It is a heading in **no live document**: only
`docs/archive/old_docs/protocols/structure-periodicity.md:192`.

The live rule is `structure-periodicity.md` **§ 6 / § 6.1a**, a different
number, and it reads the opposite way on the decisive point — *"`null` =
**derive the corner**, not 'the corner is zero'"*. A reader following the
citation lands in the archive on the pre-correction text and gets exactly
the reading `handover.md` § 2.1 records as the origin-rule bug. **FIXED** to
§ 6 / clause 2a / clause 2b per site.

**Not a defect, checked:** the three `cell-plan.md § 3a/3b` references are
deliberate provenance to an archived PLAN, which is legitimate. Two of them
omitted the `docs/archive/` prefix, so a reader would hunt for a live file;
path-qualified.

#### 1.16d Thirty-three of fifty-eight section references name no document

58 `§` references; **25 name a document on the same line**. The file's
sections span at least five (`structure.md`, `structure-periodicity.md`,
`structure-molstruct.md`, `structure-annotations.md`, `molview.md`).

And the ambiguity is real, not theoretical: **`§ 6.1` is a heading in three
documents** (`structure-periodicity.md`, `spectrumchart.md`, `results.md`)
and is cited seven times here, bare.

**This is the mechanism behind two of this session's own errors** — a bare
`§ 8.2` written into `structure.md` that read as *that* document's § 8.2,
and § 1.16c above. Not proposed as a sweep: 33 edits with no test is how a
different error gets introduced. Proposed instead as a **review question** in
`code-audit.md` — *does this reference name its document?* — and fixed at
whatever site is being touched for another reason. (`residue is review, not
tests`.)

#### 1.16e The header's load-bearing contract rests on the broken guard

The header states the strongest rule in the file: *"Any emitter that drops
them silently (rather than warning) violates the contract. `Structure.copy()`
/ `.translated()` MUST carry them through."*

Both do — correctly, and by construction: `copy()` is `return self.replace()`
with a docstring saying *"Two copies of the field list is how a field comes
to be duplicated in one and missing from the other"*, and `translated()`
routes through the one `affine` primitive.

**So the whole claim reduces to `replace()` being complete — and
`replace()`'s completeness guard is the one measured broken today** (§ 1.14:
field NAMES from `dataclasses.fields()`, field VALUES from a hand-built
fixture, so dropping the already-declared `annotations` left it green). The
file's top-level contract and its weakest test are the same fact.

#### The placeholder condition, at its source

`__post_init__:525-528` is where § 1.15's third condition is born:

```python
if self.atom_names    is None: self.atom_names    = list(self.elements)
if self.residue_ids   is None: self.residue_ids   = [1] * n
if self.residue_names is None: self.residue_names = ["MOL"] * n
if self.chain_ids     is None: self.chain_ids     = ["A"] * n
```

Four synthesized values, every one of them also a legal user value — and
`atom_names`'s placeholder is *relative* (equal to `elements`) where the
other three are absolute constants. Nothing downstream can distinguish
"synthesized" from "stated" except by re-running the comparison, which is
what makes `identity_to_dict`'s all-or-none test the best available answer
and why a MIXED structure (§ 1.15b) defeats it.

### 1.17 Consolidation with the other machine — merge `145f4001`, 2026-09-23

Sixteen commits from the other machine against fourteen here since base
`3ee27d9c`. **The two changed-file sets are disjoint**, which is why the merge
was textually clean and why everything below is semantic. Both sides' targeted
tests pass together on the merged tree: 263 passed (exit 0). Every claim here
was re-checked against the merged code, not taken from either side's record.

**Their work closed or moved items of this audit:** § 1.3 with § 5 lead ④
(`764addd3`); § 2 item 7's table row (the prose line remains); § 1.12b's
count (eleven → twelve, by a fallback they added); § 1.12d's fabricator (now
per axis; the origin skip and the false "preserved" claim remain); § 5 lead ①
(same code, two definitions of reachable — see § 1.12d item 3).

**This audit's own errors, found by the pass:** § 1.1 cited a `structure.md`
§ 8.2 that does not exist; § 1.8c proposed a check at three sites of which one
can fire. Both corrected in place.

**Their documents' errors, verified here:**

* Handover § 3 and `plan.md` W21 say `spectra.json` carries no activity
  classification and the viewer shows only Raman. Built 2026-09-11:
  `spectra/activity.py` is the one home, every serialisation stamps
  `activity_class`, the mirror plot / rug / display-floor slider / IR column
  landed in `b337f91c`, `tests/spectra/test_activity.py` pins the CO2 case.
  W21's *"designed, not started"* is twelve days stale. What remains of D-2:
  the per-mode ES probe still selects by Raman brightness
  (`spectra/selection.py`, `config/pyscf.py:818` *"Raman-activity
  threshold"*), and no document states the classification rule.
* Handover § 5.2 asks the next person to *"fix the comment in
  `wizard.as_structure()` that now states"* the info-strip rule. No comment
  states it; the rule's text exists only in the handover, and the docstring
  records the question as open. `plan.md` X4 ② calls the same strip a
  *defect*; the code calls it an open question.
* Handover § 5.1: *"the pair is written atomically"* is false at HEAD
  (§ 1.1); and the quoted *"NOT verified here (the caller compares it)"* is
  `apply_to_structure`'s docstring (`sidecars/molstruct.py:605`), not `load`'s.
* Handover § 4 *"the bias is asked twice"*: `bias_voltage_v` is a catalogue
  row (`catalogue.template.toml:2449`, `stages = ["device"]`) beside
  `TransportConfig.bias_voltages_v`; plausible, unverified in the browser
  (held).
* `plan.md` X4 ⑤ (*"the likely answer is DELETE"*) and handover § 5.1
  (detect + attest, *"the shape the user specified"*) disagree about the
  hash; the handover is the later document.
* `transport/wizard.py:476` is the twelfth `axis_kind` fallback, written by
  the very reasoning § 1.12b quotes — reconciled against *"its eleven
  neighbours"*, never against the owner.

**Same finding, two records — keep one:** § 0a and handover § 6 (*"changing
what a value is called means opening every reader"*), found independently on
two machines within a day, which is the strongest evidence either document
has that the condition is real · § 1.12b and the `wizard.py:469` comment ·
§ 5 lead ⑤ and handover § 4 `TBT.Verbosity` (theirs names the owner, § 5o.5
step 3) · § 1.16d / § 2's stale line pins and W19 / W20 / D5 (this audit owns
the policy, theirs the re-measured instances) · § 5a's blind tests and
handover § 4's weak assertions (disjoint sites, one list) · § 1.2 / § 1.11
and handover § 4's describe-door vocabulary (one family: a guard whose
vocabulary moved while the guard did not).

**Not the same — keep two:** § 1.14 (the electrode-frozen check is skipped on
vibration decks) vs X4 ④ (the unconsumed-label warning's kind, fixed
`ddfcb8bd`); § 1.12a (`partial_charges`' element lookup) vs X4 ① (species
*order*, fixed — the owner gained callers; this site still is not one).

**Gaps nobody owns, because each side assumed the other had them:** § 1.5
`/api/structure/analyze` (held here, untouched there); § 1.8a rename; § 1.6's
web half; `structure.md` § 2.2a's missing test for when a strip is correct
(their § 5.2 derived one and asked *"the next person"* to write it into a
document on this side); **D-1** — `jobset/prep.py:1381` builds its shared set
from `select(citation=True)`, so a stage override of `species_order` has never
been refused, and `jobset/` was never in this audit's scope;
`validation/sidecar.py` grew from 153 to 194 lines after § 7b sized it, so
G2's scope statement is stale.

**The comment rule meets their practice.** Their handover points the next
person at dated, narrative comments as the durable record
(`blueprints/transport.py:535-559`, `transiesta.py:283-295`,
`wizard.py:169-215`, `compose.py:583-620`). Under the rule as stated —
transform, never delete — the reasoning in those comments survives and the
dates and *"was reverted"* narrative go, so the pointer still lands. Those
files are held; the pass over them waits for the hold to lift.

## 2. Documentation: one policy, not forty-four edits

**6 of 39 `file.py:NNN` references in the contract documents resolve — 15%.**
*(Corrected 2026-09-22. This section first said 4 of 44 / 9%; both numbers
were wrong. Re-derived exhaustively over both reference forms — `path.py:NNN`
and the bare `` `:NNN` `` — the total is 39, and `parse.md`'s 4 and
`code-audit.md`'s count match the first pass exactly, so the error was
entirely in the four model documents. Direction and magnitude stand; the
figures did not.)* Plus seven symbol names that have never existed, and five
retired concepts written as current.

The rule is written down in this repo — `docs/model/parse.md:355`:

> *"A line number is a pin: it measures where a thing sits rather than what it
> does, and it rots on the next edit of a file this document does not own. The
> function name is the anchor and it is greppable."*

**But "one policy to propagate" was the wrong read, and that matters more than
the count.** Three things the provenance pass established:

1. **The pre-policy pins are a DECLINED sweep, not a missed one.** The day the
   rule was written, a second commit fixed three pins and said so in writing:
   *"Measured tree-wide first: 100 such pins live outside `docs/archive/` …
   these were the ones this work actually touched — **the rest are unverified,
   not endorsed.**"* That is a measured, scoped, declared non-sweep. The
   remedy is a decision about whether the debt is paid, not a sweep.
2. **The rule has three homes and no owner** — `parse.md:355`,
   `plans/plan.md:1103` § 5h, and `execution/project-layout.md:2285`, the last
   citing `plan.md` rather than `parse.md`. This is the repo's own rule
   inverted: *fix the RULE in the document that owns the concept* has no
   answer when no document owns it.
3. **The consequence is observable.** The rule's own author added three fresh
   pins to `parse.md` itself thirteen days later, and one to `code-audit.md`
   on 2026-09-22.

**So the first move is giving the rule one home, not sweeping 33 references.**
And no sweep cadence can fix it anyway: the pins in `plan.md`'s X1 row were
stale **68 minutes** after they were written, by the same author on the same
file.

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
   four of six rows name live code. **Half closed on the other machine
   (2026-09-23):** the table row (`:1755`) now says **LIVE**, called from
   `blueprints/transport.py`. The prose above it (`:1635`) still opens
   *"has no production caller and is to be deleted"*, with a correction
   bolted underneath — a reader scanning for the claim still finds it stated
   as fact.

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
| ① `_compute_cell_from_extents` | **NOT RESIDUE — LIVE; reached only by routes that bypass `compose`.** `load_compose_record` has no `_unusable_cell` call and the engine seam has no cell gate (§ 1.12d), while the production ladder refuses a cell-less citation at `compose` — so their *"unreachable"* (X1 ①, X4 ③) and this *"live"* describe the same code under two definitions of reachable. `as_structure()` wraps a *fabricated* box in an explicit `cell=`, so `_lattice_block` prints *"Explicit lattice preserved from the structure (NOT recomputed from atom extents)"* about a recomputed one. Measured before the merge: `resolve_cell()` said `[7.44, 6.00, 17.22]`, the deck wrote `[31.44, 30.00, 20.00]` — the transverse pair off by 4×. *(The fabricator's padding changed in the merge; the false "preserved" claim did not, re-measured on a fresh fixture.)* Two tests pin the fabricated box **as the contract**, so this is a contract decision, not a cleanup. |
| ② `DEFAULT_ELECTRODE_KZ` | **RESIDUE.** The commit that deleted its last two readers edited `__all__` to keep it. Deletion orphans nothing. |
| ③ `SEALED_TRANSPORT_FIELDS` | **RESIDUE.** Production builds a *different* union inline (three members, not two). Its only reader is a test weaker than the code it guards. |
| ④ `config_for` | **PART RESIDUE, PART LOST CALLER.** One rule inside it — filling the config from a form-B pair's recorded contract — stopped running on 2026-09-16 and nothing took it over. That is § 1.3 — **and it was taken over on the other machine** (`764addd3`, in `citation_defaults.py`, not by reviving this function). What remains here is the residue half. |
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

**~30 of 441 in-scope tests are removable** — 24 duplicates, plus **6 test
functions across 4 sites** that are blind in one direction or assert a shape.
(An earlier draft said "~28 … and 4", conflating the 4 *sites* with their 6
*functions*; 24 + 6 = 30. And "cannot fail" is too strong for two of the four —
each bullet below carries the correction.) `docs/process/testing.md` already
says unifying an API must REDUCE the count; it has been going up.

Twelve clusters, each with the bit it carries and the one test to keep. The
largest: **16 tests carry the single fact "the default isolated vacuum gap is
3 Å"** (mutant: `3.0 → 5.0`), three of them byte-identical assertion triples in
three files. *(The 16 is measured over this audit's file set and is a FLOOR,
not a total: a re-measurement over a wider shortlist put it at 17, and a test
asserting a derived consequence with no `3.0`/`6.0` literal is invisible to
either shortlist. Re-derive with the mutant over the full suite before step 16
acts on a number. Two triples are byte-identical; the third is the same triple
through `cell.resolve()` with renamed accessors.)* Three of the sixteen are
thin-wrapper tests on `cell.resolve()`,
which just forwards to `Structure` and decides nothing — `testing.md` puts
those on the `Structure` side. Next largest: **9 tests assert the same literal
corner `[7.5, 7.5, 7.5]` on the same fixture**; keep three, one per layer.

### Three of the four sites, each measured

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

## 5b. The test screen for § 1's fixes — what each one retires, breaks, or unpins

*(Added 2026-09-23 at the user's instruction: "tests need to be screened to
see which are related to these and retire correctly as needed.")*

Screened per decided fix. **Net effect: 1 rewrite, 5 mechanical updates, 0
retirements, 2 added cases** — and one fix with zero test impact at all.
Nothing here grows the function count; the two additions are cases on
existing tests, per `testing.md`'s REDUCE rule.

### Pins the OLD behaviour — rewrite, do not retire

**`test_structure_pair_one_generator.py::test_a_stale_sidecar_is_removed_rather_than_left_disagreeing`**
— § 1.8b changes what it asserts. Its docstring states the concern:

> *"A structure that loses its metadata must lose its sidecar, or the pair on
> disk says two different things about the same atoms."*

**That concern is correct and survives.** Only the mechanism changes: the
sidecar is **rewritten empty** rather than unlinked, so the pair still cannot
disagree — both halves say "no metadata". Rewrite the assertion to
*"the sidecar still exists AND carries no regions"*. Retiring it would lose a
real invariant.

### Breaks MECHANICALLY under § 1.1a (`pair().sidecar` becomes text, not a dict)

This is a cost of § 1.1a that the item did not track. **Five call sites**, one
of them production:

| site | what it does |
|---|---|
| `tests/test_structure_pair_one_generator.py:167` | `made.sidecar["schema_version"]` — indexes it as a dict |
| `tests/test_periodicity_gate.py:930` | `_json.dumps(made.sidecar)` — would double-encode |
| `tests/test_periodicity_gate.py:1259` | same |
| `tests/test_siesta_constraints_from_out.py:368` | `molstruct.save(d/name, pair(struct).sidecar)` — `save` takes a dict |
| **`molbuilder/pyscf/input.py:1502`** | **production** — `return StructureCodec().pair(struct).sidecar`, the deck splice's payload |

The production one is § 1.1a's own subject (the deck writer), so it is in
scope either way. The four tests are one-line changes. **Decide the shape
before doing any of them**: `sidecar` as text with a `sidecar_dict` beside it,
or text only with callers parsing. Four test sites and one caller is small
enough that text-only is viable.

### Left UNPINNED — the fix cannot be verified without adding a case

**§ 1.12c (the k-sampling hint) has NO test.** Searched
`tests/validation/test_siesta.py` for the hint's message and its condition:
nothing. So the two frame errors were never pinned, and the fix would be
unverifiable. **Add one case**: a skewed cell (hexagonal in-plane pair, the
contract's own example) where `norm(a)` says ≥ 5 Å and the true perpendicular
separation is < 5 Å. The current code fires the hint; the fixed code must not.

**§ 1.12a (label-blind `partial_charges`)** — three dipole tests exist
(`tests/validation/test_geometry.py:271, 288, 300`) and **none uses a labelled
species**. Add the labelled spelling as a **case on the existing polar test**,
not as a new function: `O1 H2 H3` must give the same verdict as `O H H`.

### Touched, needs re-reading rather than changing

**`tests/test_workflow_group_wire_contract.py::test_cfg_none_path_correctly_omits_workflow_group`**
— § 1.2 puts an `info` on `validate()`'s `cfg is None` path, and this test is
named for exactly that path. It asserts the `workflow_group` **enrichment** is
omitted, which stays true; whether it also asserts the issue LIST is empty
decides if it changes. Read it before editing.

### Zero test impact

**§ 1.12b (delete the eleven `axis_kind` fallbacks).** Searched `tests/` for
`axis_kind=None` / `axis_kind = None`: **no matches.** The fallbacks are
unreachable in production *and* untested. Clean deletion — and **do not add a
test**: it would assert an unreachable branch, which `testing.md` § 3a
forbids. If a test for it seems necessary, that is evidence the branch IS
reachable, and the reachability is the finding.

### Not yet screened

§ 1.5 (analyze onto the codec), § 1.8a (rename structure), § 1.8c (the hex
check), § 1.11 (the registry stub). Screen these before their steps, the same
way — the § 1.1a result shows the screen finds costs the item itself missed.

## 6. Needs a ruling, not a fix

1. **X2 ①** — a frame-range `.xyz` read back without its sidecar loses
   `transport`, because the extxyz `pbc=` header is boolean. Either molbuilder
   writes its own `axis_kind` key into the comment line, or the loss stands and
   the pair remains the only faithful carrier.
2. **Lead ① / § 1.12d — is a cell-less lead a supported input?** The
   production ladder refuses one at `compose`; the other machine's handover
   § 4 says that makes the isolated-electrode path *"the user asked to keep"*
   unusable today. Supported → the branch resolves cell and origin through
   the owner, with the lead's 15 Å vacuum default as a **parameter of
   `resolve_cell`** (a design change: § 0a's *parameter* rung, replacing the
   copy their rewrite left beside the owner); not supported → delete the
   branch and `_compute_cell_from_extents`, and retire the two tests that pin
   the fabricated box as the contract.
3. **`TBT.Contour` energies** — absolute or E_F-relative? `record.py:326`
   writes `"energies_relative_to_ef": True` unconditionally and
   `conductance_g0` reads T at E = 0; `config/transport.py:202` says the
   question is *"unresolved against SIESTA 5.4.2"*. Settled by reading one
   real `AVTRANS` beside its device `.fdf` — not derivable from source.
4. **Does a CLI edit owe the user a notice?** The web's eight ops all report;
   `cmd_modify` reports nothing. `structure-periodicity.md` § 8.2 assigns the CLI only the *generation*
   guard. (The CLI *seam verdict* was already raised and dropped on a ruling —
   do not re-propose that one.)
5. **`set_info` / `drop_info`** have no Python caller — deliberate parity with
   `molview.data.info`. Keep as a declared extension point, or delete?
6. **D-1 (their handover, W30 ①) — what declares that a value binds every
   rung?** The catalogue's `citation` marker means *who supplies the
   default*; the shared class is larger (`species_order`, `spin_treatment`,
   `spin_total`), and both doors that refuse a per-rung override of a shared
   value filter on `citation` (`jobset/prep.py:1381`, the transport form).
   **A:** a sibling marker `shared = ["transport"]` — one new axis on `Item`,
   `select()`, the TOML rows, and both doors read it. **B:** widen `citation`
   — three TOML rows and no code, but the marker's name then lies on rows
   with no run to cite, and the shared panel's provenance line (*"from the
   run you cited"*) would need a second distinction anyway. Recommended: A.
   `system_label` needs `role = ["transport"]` either way.
7. **D-2 (their handover, W21) — two parts, and the first is already
   built.** (a) The activity classifier (`spectra/activity.py`, 2026-09-11:
   widest log-gap ≥ 2 decades and ≤ 1e-3 of the peak, else 1e-6 of the peak;
   an absolute presence floor per channel; "partial" when a channel was not
   computed) — endorse as the rule, then give it a document home; no
   document states it today. (b) The per-mode ES probe's `top_n` and
   `threshold` selectors rank by Raman activity, which in a centrosymmetric
   molecule keeps one symmetry class and drops every IR-active mode. Retire
   both and keep `skip` / `all` / `explicit` plus the frequency window
   (recommended; the plan row's own suggestion), or re-key them on the
   activity class, or rank by measured gap change after computing. **(c)
   Ruled** *(user, 2026-09-23)*: the file is one format for both engines; a
   number an engine cannot produce is absent, never zero; a result with no
   strengths is drawn as lines; the PySCF-only MO block becomes optional
   before the SIESTA arm — `web/spectra.md` § 9b.3.
8. **`structure_hash` — delete, or detect-and-attest?** § 1.8c has the
   consolidated facts. Detect: compare in `StructureCodec.load` and report
   as a notice, never refuse; attest: a separate door that re-stamps the
   **sidecar only**; § 1.1a's generator lands first. Or delete the field, as
   their `plan.md` X4 ⑤ leans. If detect: the rule goes into
   `structure-molstruct.md` § 3 before any code.
9. **`structure.md` § 2.2a needs the test it lacks — when is a strip of
   `info` correct?** Their § 5.2 derived a rule from one case (*"`info`
   travels with a structure that may LEAVE this calculation"*); the
   consolidation pass tested it on a second (the in-script atom-metadata
   block, which carries no `info` and whose return path re-derives the
   contract from the deck) and found the decidable form: **`info` travels
   when the derived structure becomes a `.xyz` + `.molstruct.json` pair on
   disk; where the artifact states the settings itself, the artifact is the
   record.** Merges are § 2.2b's, not this rule's (`concat` keeps the first
   input's `info`, measured). Adopt into § 2.2a, then the restatements
   (`structure.py`'s `info` comment, `wizard.as_structure`, `script_emit.py:448`,
   `parse/dirs/atom_metadata.py:47`) cite it.
10. **The three mechanisms under `template.md` § 6.6.** (a) The source key on
   a template item — proposed `source` with the vocabulary `cited` · `record`
   · `person` · `default`, written by `init` / describe and read by every
   surface. (b) Where a deck carries its per-parameter table — proposed: the
   PROVENANCE reserved block grows it, because the deck is what is opened
   months later, `.validation.txt` is never read back, and G4's text
   comparison already tolerates that block's rendering moment. (c) The deck's
   *not set* line for an `optional` item at `None` — proposed: a comment
   naming the engine default that applies. The browser road's missing
   pipeline log (W20) is obligation 5's and needs plumbing, not a decision.

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
2. **§ 1.1 + § 1.1a** — the generator renders both halves. Do this one FIRST
   of the set: it is the only structural change here, and it closes the
   design document's § 15.6 deck-writer items in the same commit rather than
   leaving them to be fixed a second time at the third writer.
3. **§ 1.2, § 1.7** — the rest of the data-loss and uncaught-exception
   set. Independent of each other, each small, each user-visible.
4. **§ 1.8** — one home for the structure-path rule (**not held**), then
   rename structure vs rename file (**ON HOLD, § 0b — `web/blueprints/`**), the no-delete rule, the hex check, and § 1.8d. Group them:
   they are one missing door and four things that grew where it should be.
5. **§ 1.5a's DOC fix only** — § 2.4's second wrong statement. **§ 1.5's
   code fix is ON HOLD (§ 0b): it lands in `web/blueprints/`.** The doc
   error is not held; it instructs a regression whoever reads it. Do § 1.10a's doc fix in the SAME commit as
   § 1.1a's: both are in § 2.4's four-clause block, and a half-corrected
   contract is what produced them.
6. **§ 1.11** — the registry seam. Small, and it belongs beside § 1.2: all
   three are "a check that does not happen, and the absence is invisible".
   § 1.11b is design § 18 step 0a — do it there, not twice.
7. **§ 1.12 — the systematic item.** Start with § 1.12b's eleven deletions:
   it is pure removal and it makes the rest safe to read. Then § 1.12a and
   § 1.12c onto their doors. § 1.12d needs a reachability answer first. Three document
   sentences ride with § 1.12c.
8. **§ 1.13 + § 1.15** — the second and third conditions. § 1.13 is two
   one-liners. § 1.15a is a chain-aware adjacency key plus dropping a
   misdirected `$X3DNA` from a self-check failure; § 1.15b needs the
   placeholder carried separately from the data, which is a shape decision
   before it is a fix.
9. **§ 1.6 + the `wrap_into_cell` knob** — a further origin-rule site the sweep missed. Do it
   with the rule in front of you, from `handover.md` § 2.1.
10. **§ 1.3 + § 5 lead ④** — **done on the other machine** (`764addd3`): the
   recorded-contract read is alive on the live road, in `citation_defaults.py`.
   Only the residue half of `config_for` remains, and that is § 5's ordinary
   step-0 pass.
11. **§ 1.4** — the CLI stdout destination. Needs § 6 decision 1 first, because
   what a single stream *can* carry is the same question.
12. **§ 2's line-number policy** — one pass over four documents, mechanical.
   Then § 2's behavioural list, which is the part that needs reading.
13. **§ 3 and § 4** — fix each at its owner, never at the instance. Start with
   the `replace()` guard, because it is what makes the rest safe to touch.
14. **§ 5a's three unpinned rules** — write them before §§ 3/4 touch the code
   they guard. The `.pdb` one first: it is a regression with numbers already
   written down.
15. **§ 5a's 6 blind/shape-asserting tests across 4 sites** — retire or rewrite. They are worse than
   absent, because they read as coverage.
16. **§ 5a's duplicate clusters** — ~24 tests out, one keeper per bit, each
    cluster's mutant re-run afterwards to confirm the keeper still goes red.
17. **§ 5's residue** — last, and only the four with a clean step-0 verdict.

**Before step 17, and after step 3:** § 7b's gap pass. Residue cannot be
deleted from a surface where eighteen modules were never opened by this
audit — the
builders in particular, since they are where a `Structure` is created.

**Then, separately:** audit #2 (§ 7c).

**Standing on its own, not part of the order:** the 15 hand-built
`structure_hash` fixtures (§ 5a). They agree with the writer today only
because the gate is loose, so they are a latent break rather than a defect —
convert them as each file is touched for another reason.

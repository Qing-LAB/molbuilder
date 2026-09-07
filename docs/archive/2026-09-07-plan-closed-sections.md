# Plan sections closed and archived — 2026-09-07

**Role:** history. **Not a source of truth.**

Three sections of `plans/plan.md` whose work shipped and whose live rules now
live in the contract documents that own them. Archived under the substance-
first rule (`docs/archive/README.md`): a closed plan section must never be the
only place a live invariant lives.

| section | the rule it earned | where that rule lives now |
|---|---|---|
| **5d** `parse/scripts/` retired | `parse/` reads foreign formats; a reserved block in a script molbuilder generated belongs to its writer | `model/parse.md` § 1a |
| **5i** the projects root, one door | a test never builds inside the real `projects/` tree; the two isolating fixtures and when each applies | `process/testing.md` § 2a *(written 2026-09-07 — until then this section WAS the only home, which is why it could not be archived)* |
| **5j** `_assembler_helpers` deleted | a re-export reads as a caller — to a grep and to a reviewer | `execution/running-a-job.md` § 4.2 (the corrected dead-code figure) and `model/parse.md`'s `dirs/` tree |

---

## 5d. `parse/scripts/` retired — CLOSED 2026-09-05

`parse/` reads foreign formats; a reserved block in a script molbuilder
generated is its writer's. Steps 1-3 and 5 shipped (`5f742911`, `69de4f3b`,
`e05d3c65`): `parse/scripts/` is gone, with six `TextParser` classes and
`ScriptResult`.

**Step 4 is abandoned, not pending.** Routing the private `_extract_*`
importers onto a `read_script` door was never done, and the door built for it
sat with **zero callers for three weeks** while carrying a version gate the
live readers do not have — two answers for one block. It was deleted
(`1325ca18`), not adopted. The lint step 4 proposed — *no `_extract_` name
imported across a package boundary* — is therefore not owed.

**The rule this earned now lives where it belongs**, not here:
[`model/parse.md` § 1a](?doc=model/parse.md). The 144 lines of argument that
stood here were the case for a decision already taken.

---

## 5i. The projects root — one door in production, and now in tests too

**CLOSED 2026-09-06.** Audited after a fixture assumed the tree's location and
silently opened the wrong folder.

**Production was already one door**, verified hop by hop:
`projects.projects_root()` is the single definition and the only reader of
`$MOLBUILDER_PROJECTS`; it feeds `Capabilities.file_picker_roots()` →
`GET /api/files/roots` → `setProjectsRoot()` (one caller) →
`projects.getProjectsRoot()`. No hand-rolled `/ "projects"` join anywhere in
`molbuilder/`. `find_projects_root(start)` is not a second answer — it answers
which tree a *calculation* lives in.

**The tests were where it was not.** Thirteen sites in seven files built
`ROOT / "projects/_t_…"` **inside the developer's real tree**. All are
converted: each takes `isolated_projects_root`, or
`isolated_projects_root_module` — its module-scoped sibling, added because
`monkeypatch` and `tmp_path` are function-scoped and an e2e builds one tree
per file. **Both go through one implementation**, so there is one policy and
two scopes rather than two policies.

**The guard had never caught any of them, and now cannot miss them.** It was a
regex, and it was wrong three ways at once while reading green: it scanned one
LINE at a time, so a path split across two was invisible — which is how
`test_siesta_keyword_smoke.py` came to read a real calculation's `H.psml`
(`projects/BDT/optimization/TJ-BDT-Au111/`) for months; it required the literal
`"projects"`, so the `"projects/_t_…"` spelling every one of the thirteen used
never matched at all; and once those were patched it flagged the docstrings
that explain the rule.

Each patch fixed a symptom, so the detector is **an AST walk** now *(user:
"why are we still using string to guard python code?")*: a division chain
whose base is the checkout and whose first string segment is `projects`. Line
breaks are not a concept in a syntax tree, a segment's value is its value, and
a docstring is an `Expr` rather than a `BinOp` — all three problems stop
existing rather than being caught. It matches the house practice
(`test_architecture_rules.py`, `test_layering.py` are AST-based too), and it
carries **a ten-row truth table**, every row a shape that really occurred
here, because a checker with no known-good and known-bad inputs is how this
one stayed blind.

The `H.psml` it was missing is checked in at `tests/fixtures/psml/`. Swapping
it for `conftest.write_pseudos` was tried first and **measured to fail** —
that PSML satisfies prep's screening but SIESTA does not start on it.

**Still open, and deliberately not urgent:** seven sites hand-roll
`monkeypatch.setenv(PROJECTS_ROOT_ENV, …)` instead of requesting the fixture
(`test_target_machine_choice.py`, `test_structure_authority_roundtrip.py`,
`test_periodicity_gate.py` ×2, `test_transport_record.py` ×2,
`test_transport_prep.py`). All already point at `tmp_path`, so nothing is at
risk — it is one practice written two ways. `test_projects.py`'s own setenv
calls are **not** in that list: they are the unit tests *of* the variable.

---

---

## 5j. `parse/dirs/_assembler_helpers` deleted — six helpers, no callers

**Found by asking the right question** *(user: "there must be duplicated code…
do you need to cross check if there are already other passes that does a
similar job")*. The cross-check settled it, and not the way it looked.

**All six had zero callers**, not two: `read_fdf_initial_coords`,
`extract_system_label`, `check_xv_handedness`, `check_fdf_handedness`,
`read_py_initial_coords`, `extract_pyscf_job`.

**Why they survived two cleanups: a re-export reads as a caller.**
`coords/siesta_xv.py` and `coords/pyscf_geom.py` both imported the names and
listed them in `__all__` — *"so tests + future callers have one import path per
file type"* — which made six dead functions look like a maintained API with a
tidy front door. It fooled a prior audit in writing: `running-a-job.md` § 4.2
excluded this module from a dead-code count on the grounds that it *"was never
dead and is still read by both `coords/` parsers."* It was read by neither.

**Why nothing needed them.** Their real consumer,
`script_bundle.assemble_from_run_dir`, went on 2026-06-21; the module was kept
on a claim about serving *"the bundle + job DirParsers"*. No bundle DirParser
exists or is specified, and § 5.0 later specified `RunDirResult` as seven
fields — none a geometry, a label or a diagnostic — under *"no field is added
without naming its reader in this table."*

**And the app never wanted a deck's geometry.** The starting structure reaches
a viewer as the trajectory's **frame 0** — *"the `.out`'s own frame 0, the
structure the user submitted"* — out of the same file every later frame comes
from. A deck is never re-read for coordinates, which is why a unique parser of
`AtomicCoordinatesAndAtomicSpecies` could sit unused: unique **parser of a
format** is not the same as unique **source of a fact**, and mistaking the two
is what made this look load-bearing at first pass.

Gone with it: both re-export blocks, 26 tests, the dangling
`tests/parse/dirs/test_bundle.py` pointer in `tests/support/junction.py`, and
the corrected line in `running-a-job.md`. `read_xv` and `read_optimized_xyz`
are untouched — they are the live readers.

---

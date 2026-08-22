# Audit 2026-08-21 — jobset · engines · execution · the two tabs, full-text

**Role:** review record — THE LIVE PLAN (consolidated 2026-08-21 evening,
user: "archive finished items so we have a concise list").
**Domain:** jobset/execution, engine emission, validation, the
structure-optimization and spectrum tabs, and the documentation set —
transport excluded (its workflow is designed separately, user ruling).
**History:** everything delivered — round 1's five-reader findings in
full, U0/U1/U7 (2β + its ruling wave), round 2's fixed list — is
archived verbatim in
[`archive/2026-08-21-review-delivered.md`](?doc=archive/2026-08-21-review-delivered.md).
Never act from the archive; this file is the only open list.
**Rule:** nothing below is fixed without a yes per item — except
same-day regressions of in-flight work, fixable on sight.

---

## ON SOL (the user's own checklist)

1. `git pull` — main carries everything through the 2026-08-21 waves.
2. Fix the two enum spellings in the Relax `task.json`:
   `ELPA-1Stage`/`ELPA-2Stage` → `ELPA-1STAGE`/`ELPA-2STAGE`
   (the preflight now names them; the UI dropdown offers only legal
   values).
3. Once: `molbuilder jobset probe --write` on the login node — it now
   records each partition's GPU inventory AND `max_cores` (the GPU node
   group's own row).  Nothing to hand-edit.
4. If `prep bench` still refuses over pseudopotentials after the
   pull, the message now names the real place: Sol's tree has no
   `pseudopotential/` at its top — copy your `.psml` set to
   `~/molbuilder/projects/pseudopotential/` (Relax needs H, C, S, Au),
   or set `psml_lib` to an absolute path.
5. Then: `./jobset.sh prep bench coarse` → `submit bench coarse` (one
   exact-fit job per resource shelf, biggest first) → `summarize bench
   coarse` any time, mid-flight included.

---

## OPEN — in priority order

*(Re-consolidated 2026-08-21 on the user's ruling: the path framework
comes first.  Transport locks none of the rest.  Everything below is
independent of transport unless its own line says otherwise.)*

### O1 · file and directory knowledge: one anchor rule, one door *(framework; user 2026-08-21 — TOP)*

**What broke.** On Sol, `./jobset.sh prep bench coarse` refused with
*"the library they should come from is not a directory:
`…/optimization/Relax/projects/pseudopotential`"* — a folder fabricated
from wherever the user was standing.  Two layers under one symptom:

1. The walk-up stage (83d84b88) reached `origin/main` only 2026-08-21
   13:32, and the eight commits after it were unpushed until 9f5995f6,
   so no Sol run had it.  *(Shipping, not design — closed by the push.)*
2. Even with it, `resolve_psml_lib` chooses its ANCHOR **by what
   happens to exist** — try the calculation dir, try the walked-up
   tree, else cwd`/projects` — so a total miss still reports the cwd
   form and never names the tree the calculation lives in.

**The rule that is missing.** An anchor must be declared by the
SPELLING the user wrote, not discovered by probing the disk.  Probing
means the same string means different folders on different machines,
and the error message names a place nobody chose.

**Inventory — where file/dir knowledge is handcrafted** *(full sweep,
2026-08-21)*:

| # | duplicate knowledge | sites |
|---|---|---|
| a | "where is the projects tree" — cascade + inline walk-up | `pseudos.py:42–113` (tree discovery homed outside `projects.py`, which owns the tree) |
| b | "where is the repo root" — private parent-chains | `references.py:25` · `web/blueprints/docs.py:46` · `runwrap.py:4042` · `script_emit.py:691` · `builders/backends/_threedna.py:155,613` |
| c | `job-set.json` spelled as a literal | `jobset/prep.py:878,884,891` · `checkpoint.py:358` · `jobset/_cli.py:36` (a private `_JOBSET_FILE` — a near-home nobody imports) |
| d | `task.json` spelled beside its own constant | `checkpoint.py:358` re-spells it while `task.FILENAME` exists |
| e | two vocabularies for one field | hints say the repo-root form `projects/pseudopotential/` (`jobset/prep.py:494`, `validation/siesta.py:54`, `config/siesta.py:1592`) while the sidebar backend (`files.py`) exchanges projects-root-relative paths — and the prefixed spelling **breaks** the walk-up by joining `projects/projects/…` |

`envs/builds.py:403` also walks a parent chain — that one finds the
**nvcc toolchain's** root, not ours.  It stays.

**Proposed rules — the spelling declares the anchor; nothing guessed:**

- **R1** · `psml_lib` resolution: absolute or `~` → as-is.  Leading
  `./` or `../` → relative to the **calculation folder** (the
  Save-to-current-dir form).  A bare name → relative to the **projects
  tree the calculation lives in**, found by walk-up.  A miss errors
  naming that one candidate — never a cwd form.  Callers with no
  calculation dir (server-side validate) anchor at the server's own
  `projects_root()`; repo-root cwd is that process's contract.
- **R2** · tree discovery moves to its owner: `projects.find_projects_root(start)`;
  `pseudos.py` calls it.
- **R3** · one `repo_root()` (`molbuilder/__init__.py`); inventory-b's
  five copies call it.
- **R4** · one spelling per molbuilder-owned filename: `job-set.json`
  gets a module constant beside the model it belongs to, `checkpoint.py`
  imports both it and `task.FILENAME` instead of re-spelling them.
- **R5** · every hint and help string speaks the **bare** spelling
  (`pseudopotential`); the long explanation stays in the catalogue text
  only.
- **R6** · the psml refusal names what it looked for and where, in the
  tree's own terms — the message a user acts on without reading code.

No new mechanism: R1 collapses a cascade, R2–R4 re-home, R5–R6 are text.

**What the sweep found already correct — leave it alone.** Topic and
structure directories go through `projects.py`'s helpers at all 11
call sites with zero hand-joins; stage-directory naming has one home
(`identity.py:297`); the "what molbuilder wrote" pattern list has one
home with its second reader documented in place; the workspace store's
`SCRATCH_DIR` is a single constant.  The framework is sound — these
five leaks are where callers went around it.

**Test matrix owed** (R1 is a behaviour change, so it is pinned):
absolute · `~/x` · `./x` · `../x` · bare-hit via walk-up · bare-miss
error text · no-dest-dir server path · the `projects/`-prefixed
spelling that used to double-join.

### O2 · retire SpectraConfig — a re-homing, not a delete *(spectrum work; unblocked)*
No production constructor; the runtime object is the `_LiftView` over
PySCFConfig.  The class survives only as the VOCABULARY carrier for
three readers — the vibration kind's science duck-types its field
shape, the reference selector's parity tests construct it as their
fixture, and the Methods fragment reads the same fields.  The work:
make the view the one shape, point those three at it, drop the class,
its registry row and the "all four engines registered" pin (a one-line
test edit; nothing of transport's changes).  *(An earlier note claimed
a transport lock; that was wrong and is corrected here.)*

### O3 · extract the auto-detect panel for the two describing tabs *(unblocked)*
Structure-optimization and Spectrum carry near-verbatim copies of the
auto-detect panel; extract the shared module with those two as its
callers now.  Transport's third copy joins in transport's own round —
a shared module with one recorded hold-out beats three copies drifting.

### O4 · one home for the relax retry loop *(spectrum work; small)*
The optimization deck's retry budget and the vibration relax block's
`continue` arm spell the same loop twice; an emitted helper both
compose ends it.  Sized during the close as structural-but-small.

### O5 · residue the close did not reach *(sweep-scale, low risk)*
The unenumerated leftovers: C-jobset's remaining stage-less residue
branches + the duplicated read-API comment + two submit.py residues;
T2/T3 stale test-module docstrings; R2-9's last present-tense pre-fold
narration.  Plus one ruling to record: `Issue.stage` is write-orphaned
(its one stamper retired with validate_ladder) — field + serialization
arm are retirement candidates.

### Transport's own round *(excluded from all of the above, by ruling)*
The `transport bundle` migration, its engine-base `render_checks`
misnomer, its copy of the auto-detect panel, and its blueprint's
stale `/api/spectra/render` citations — all deferred together to the
transport design round.

---

### For the user, found during the close *(no action owed to the code)*
Your real workspace store — `projects/.molbuilder_workspace/states/` —
holds TEST junk mixed with your own July workspaces (e.g.
`ws-structure-opt-panel` is a one-atom "Fe atom for auto-analyze test").
Tests can no longer write there (isolated per-test since 2026-08-22),
but the existing files are yours to keep or clean; deleting
`ws-structure-opt-*.wc.json` and `ws-modify*.wc.json` costs you any
saved panel state on those tabs.

---

## Delivered — one line per wave *(detail in the archive)*

| wave | commits |
|---|---|
| U0 same-day regressions · U1 launcher/notes/E-A1/E-T4/E-J2 | 49a0c15f · 346b58b7 |
| U7 · 2β value axes, family axis, split submission | e9cae2bf |
| § 2.12 background + references + diagrams | 56e77541 · 3e6bd37e · f923ca59 · bc847643 · 29d09696 · 5924e6ca · dd962cfe |
| widest-first · cap 3 · resource shelves · mid-flight pin | 28802e4b · 9d13cbfb · a5d65db0 · 98255f30 |
| gpu_count declared, never derived (+ the full diagram) | e8eadbe8 |
| max_cores probed · psml walk-up fix | 83d84b88 |
| plain-language rewrite for scientists | 71b5f88f |
| review round 2: 38 fixed, 9 recorded | 1cd42734 |
| the #N stage grammar | 9d6d3fcd |
| submission owns the run's state: cold verified · continue by default | 8d51662a |
| bench prep asks only about launched trials | c38ffd3f |
| U2 sixteen correctness bugs + U3 frozen-means-frozen + dedup | 92646a59 |
| P2 engine-keyed caches · P6 census/pins/gap-tests | e7733bc2 |
| U4 documentation back-sweep, every named spot | 6ad9707d |
| U5 retirements part 1 (zero-caller deletions + Pattern-B re-home) | 46906470 |
| U5 retirements part 2 (one home per fact in the vibration deck) | df992287 |
| U6 the close: 4 readers, ~80 verified findings, workspace-store isolation | 615dbbc7 |
| span-cut restore (four innocents + `elx`) · e2e census: 2 stale suites retired, the Send witness added | 79148338 · 78bb0941 |

**Verified 2026-08-21:** e2e 90/90 · none2e 6974 ran, 4 FAIL — all four
stale-in-sweep (the retired-paths lookbehind + the presets drift-guard
rewrite landed mid-sweep; both re-proven green on the final tree).

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
5. Then, from anywhere (the verb takes the bundle):
   `molbuilder jobset prep bench coarse --bundle Au-BDT-Au/optimization/Relax`
   → `... launch bench coarse --bundle ... --mode submit` (one exact-fit
   job per resource shelf, biggest first) → `... summarize bench coarse
   --bundle ...` any time, mid-flight included.  From inside the folder,
   drop `--bundle`.

---

## OPEN — in priority order

*(Re-consolidated 2026-08-22 after the directory/verb design round.
Transport locks none of this.  Everything below is independent of
transport unless its own line says otherwise.)*

### O1 · orphan letter-citations to an archived plan *(small, mechanical)*
Roughly eight code comments cite `(A3…A8, 2026-08-12)` from the
2026-08-12 staged-runs plan, while `architecture.md` § 7 now runs A1–A11
with different meanings — `prep.py:67`'s "(A8)" is error translation, the
live A8 is "an object travels whole". The dates disambiguate for a
careful reader, which is the only reason this is recorded rather than
swept. The three that collided with letters created in the path round —
two in `prep.py` / `plan.py`, one in `engines/siesta.md` — are already
de-referenced.

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
| the path framework: A10 · A11, contract first, five resolvers folded into one | a691fce1 |
| jobset.sh deleted · env verified in the wrappers · scheduler header from the probed record · `submit`→`launch` | ac04b06a · 47e5958b · ba0ecc00 · 0f489861 · 127e8e57 |
| the directory/verb design: one projects-root door · `--bundle` uniform + fenced · one containment fence (8 copies → 1) · `describe`→`init` · the `.run.sh` environment contract | *(this commit)* |

**Verified 2026-08-21:** e2e 90/90 · none2e 6974 ran, 4 FAIL — all four
stale-in-sweep (the retired-paths lookbehind + the presets drift-guard
rewrite landed mid-sweep; both re-proven green on the final tree).

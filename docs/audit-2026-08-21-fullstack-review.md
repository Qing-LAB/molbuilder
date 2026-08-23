# Audit 2026-08-21 — jobset · engines · execution · the two tabs, full-text

**Role:** audit report — **evidence, not a plan.** Held the open list until
2026-08-22, when rule R3 was applied: open work lives in
[`roadmap.md`](?doc=roadmap.md) and nowhere else. Its remaining items are
roadmap **7.5**; the findings below stay here as the record of what was
measured and why.
**Domain:** jobset/execution, engine emission, validation, the
structure-optimization and spectrum tabs, and the documentation set —
transport excluded (its workflow is designed separately, user ruling).
**History:** everything delivered — round 1's five-reader findings in
full, U0/U1/U7 (2β + its ruling wave), round 2's fixed list — is
archived verbatim in
[`archive/2026-08-21-review-delivered.md`](?doc=archive/2026-08-21-review-delivered.md).
Never act from the archive; this file is the only open list.
**Rule:** nothing below is fixed without a yes per item — except
same-day regressions of in-flight work, fixable on sight. *(Scheduling now
happens in the roadmap; this rule still governs how an item is worked.)*

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

## OPEN — moved to the roadmap *(2026-08-22)*

O1, O4 and O5 are **roadmap 7.5**. They are not restated here: two copies of
one list is how the 2026-08-14 day of re-derivation started.

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
| O2 · `SpectraConfig` retired — `VibrationConfigView` is the one shape; class, registry row and engine-count pin gone | *(2026-08-22)* |
| O3 · auto-detect extracted: one card partial, one `lib/auto-detect.js` (panel + supersede protocol), audit §§ A2/C1/C2 closed, transport recorded as the hold-out | *(this commit)* |
| the Task-setup id collision: the bench card was unreachable behind a duplicate `ts-machine-card`; picker renamed `ts-target-*`, and a duplicate-id sweep added over every served page | *(this commit)* |

**Verified 2026-08-21:** e2e 90/90 · none2e 6974 ran, 4 FAIL — all four
stale-in-sweep (the retired-paths lookbehind + the presets drift-guard
rewrite landed mid-sweep; both re-proven green on the final tree).

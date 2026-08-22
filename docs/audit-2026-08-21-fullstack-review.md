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
4. Then: `./jobset.sh prep bench coarse` → `submit bench coarse` (one
   exact-fit job per resource shelf, biggest first) → `summarize bench
   coarse` any time, mid-flight included.

---

## OPEN — in priority order

*(P1–P6 delivered 2026-08-21/22 — see the ledger below.  What remains
is the deferred remainder, each with its reason, plus the close.)*

### D1 · the SpectraConfig class retirement *(deferred: transport-adjacent)*
No production constructor; the view is the runtime object.  BUT the
class rides the four-engine validator registry
(`_ENGINE_VALIDATORS` — `test_all_four_engine_configs_are_registered`
pins SpectraConfig and TransportConfig together), so retiring it
reshapes a contract transport shares.  Do it WITH transport's round.
(The safe half — `validate_selection`, `spectra/selection.py`'s dead
preview path — is already retired; `select_modes` stays as the parity
reference for the deck's inlined selector.)

### D2 · the auto-detect panel trio *(deferred: one copy is transport's)*
Three near-verbatim copies (structure-optimization / spectra /
transport).  The extraction meets the ≥2-caller bar, but doing it
without transport leaves a shared module plus one hold-out — worse
than three copies.  Extract in transport's round.

### D3 · the relax retry-loop two-home copy *(U6 sizes it)*
The optimization deck's retry budget and the vibration relax block's
`continue` arm spell the same loop twice.  Structural (an emitted
helper both compose), not mechanical — sized during the close.

### D4 · the unenumerated residue *(U6's readers re-find)*
The round-1 categories whose item lists did not survive consolidation:
C-jobset's stage-less residue branches + the duplicated read-API
comment + two submit.py residues; the D-series deleted-machinery
citations (~8); T2/T3 stale test-module docstrings; R2-9's remaining
present-tense pre-fold narration.  Also noted for a ruling:
`Issue.stage` is now write-orphaned (its one stamper retired with
validate_ladder) — field + serialization arm are retirement
candidates.

### U6 — the close *(now)*
The final full R×3 round over everything P1–P5 changed, plus a clean
`none2e` and the live E2Es.  Transport excluded throughout.

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

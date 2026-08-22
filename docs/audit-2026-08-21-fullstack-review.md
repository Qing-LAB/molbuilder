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

### P1 · U2 — the fourteen verified correctness bugs *(awaiting one yes)*
Spectra/engine side, each recorded with its fix shape in the archive's
bucket 2 + engine-reader section.  Headline: **HF + Raman crashes AFTER
paying for the Hessian** (NameError in the generated script, E-M4.7);
also the charge-heuristic bypass (E-M3.1/V-3c, silent wrong science),
vibration decks skipping the `ast.parse` gate optimization decks pass
(E-M4.6r), the CPU-fallback lie in help + emitted decks (E-M1.5),
newton-at-relax (M1.3), RKS+spin refusal ownership (G-1c), pyscf parity
(G-1d), `is_dft` on the vibration line (E-M6.3), two broken `__all__`s
(E-M4.1+E-V4a), wrapper header None arm (E-J3), handover `_what` sender
(E-B9), notice `level` key (E-B10), PySCF bench refusal by name (E-J1 —
today's review found the current gate is ACCIDENTAL), validate_ladder
(G-1a), gate ③ at save (G-1b).

### P2 · R2-1 — engine-blind UI caches *(awaiting yes; small, real risk)*
`loadSweepChoices` and `loadPresets` in the Task-setup page cache the
first engine's answers for the whole page-load: a PySCF folder opened
after a SIESTA one shows SIESTA's machine rows and tier presets, and
applying a preset writes SIESTA values into a PySCF description.  Fix
pattern is settled in the same file (the columns cache keys by
engine:kind, viewer.js:748).

### P3 · U3 — the dedup family + the frozen-atoms science decision
*(awaiting ruling)*
(a) One rule, one home: the double-fire validation family — the engine
copy vs the kind's, one charge resolver, `where` ids as catalogue names
— now joined by round 2's R2-5 (the bench per-point shape checks live
in the preflight AND `_declared_execution_pins`, with named
divergences: allocation-item duplicates caught only by the preflight;
Issue-list vs first-refusal).
(b) **The science decision only the user can make**: when the vibration
workflow relaxes geometry before frequencies, do atoms frozen in the UI
stay frozen?  Today the relaxation ignores the freeze list and the
documents contradict.  Options: honor the freeze / refuse frozen-atom
vibration jobs / always relax free and document why.

### P4 · U4 — the documentation back-sweep *(awaiting one yes)*
~40 verified stale spots predating the 2026-08-21 waves (listed in the
archive's bucket 4 + engine-reader additions), plus round 2's R2-9
(older test docstrings narrating pre-fold designs in present tense).

### P5 · U5 — retirements *(batch with the list, or per item)*
The archive's bucket 5 list (SpectraConfig + selection.py dead half,
`_shared.py`'s zero-caller helpers, two orphaned doors, JS/state dead
lists, jobset S1-S7, `density_fit_line`, doubled `_mb_outfile`, N² IR
loop, DFT dresser M1.2, one GPU mechanism M1.4, refs render-through,
T1-T3 test retirements, auto-detect trio) — joined by round 2's R2-4
(the dead next-unlaunched picker arm + its imports).

### P6 · small, each its own quick yes
- R2-2: summarize's "every timed trial ran something other than asked"
  fires when nothing was timed at all (condition scans all points).
- R2-7: test_pseudos' `../foo` assertion cannot fail; assert the exact
  anchored path.
- R2-8: consolidate the three near-identical shelf-grouping pins; write
  the two gap tests (a grouped bench on a FLAT calculation; a GPU-side
  winner riding run-config on a mixed sweep).

### U6 — the close *(after P1–P5)*
The final full R×3 round over whatever P1–P5 changed, plus a clean
`none2e` and the live E2Es.  (Round 2 — 2026-08-21 evening — covered
the bench/execution/UI span and is archived; U6 is the post-fix close.)

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

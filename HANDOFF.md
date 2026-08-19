# Handoff — the preparation layer

## Read these first, in this order. They are the source of truth; the code is not.

1. `docs/execution/script-preparation.md` — **the contract.** The three
   questions (space / time / the seam), the seven floors, `prep`'s five steps,
   the thirteen sub-steps of step 3, and W1–W10.
2. `docs/execution/generator.md` — the other half of the layer: what every value
   *is* and where it came from. § 7 owns the catalogue half of the engine seam;
   § 4 of the contract owns the code half.
3. `docs/engines/stages.md` § 1.1a — a ladder is N decks and N jobs for **both**
   engines. Its five consequences are all in place and each names the file that
   holds it.
4. `docs/engines/tuning.md` § 2.4 / § 2.5 — the per-tier convergence tables.
   **These tables are the authority for those numbers, not any code**, and
   `test_doc_claims.py` checks the presets against them. § 4's ladder table is a
   convenience restatement and loses to them.
5. `docs/roadmap.md` § 6 — **status lives here, never in a contract.** The
   preparation-layer debts are P1–P6 and all six have landed.

The programme that built this is archived at
`docs/archive/2026-08-18-preparation-backend-plan.md`. Open it for history, not
to decide what is open now.

## Where the work stands

The layer is built and both engines are on it. `prep`, `siesta convert` and
`pyscf convert` all reach the deck through one call —
`script_emit.prepare_deck(spec, struct, cfg, path)` — which runs **validate →
render → write → check** in that order, once, for every route.

**What the seam carries is the engine's FORM, not finished text.** `spec_for`
returns a `DeckSpec`: the deck's layout as an ordered table of `Section`s and
`Block`s, plus how this engine spells one setting. The framework walks it, so it
knows what the deck was supposed to contain and can compare that against the
file it just wrote — which is what the check gate needs and what no validator in
this tree could do before.

**Two rules make that work, and they are the ones to keep in mind when touching
either engine's writer:**

- **W9 — the layout's MEMBERSHIP is settled when the spec is built; each
  member's TEXT is settled when the framework walks it.** A section only some
  calculations have is *left out* of the layout for the others, never chosen
  inside a `Block`. `spec_for` holds `(struct, cfg)` and can answer that.
- **W10 — an engine keeps ONE per-render context** (`_derived`), and every
  reader takes it whole: the layout for membership, the syntax door for a
  derived value's spelling, the record blocks to quote one.

## What a review on 2026-08-19 found and fixed

The full findings are in `docs/execution/script-preparation.md`'s own history
notes; three are worth carrying forward because they are the shapes that recur.

- **Both engines rendered their sections from inside a `Block`**, by calling the
  framework's section walk themselves — nine times in one SIESTA deck. So the
  layout could not name them, `render_deck` could not collect what they wrote,
  and the check gate's loop-closing rule ran on an empty list and passed: a
  728-line SIESTA deck reported **zero** written keywords. It reports 24 now,
  PySCF 13, and the walk has one owner (`_render_sections` is private).
- **SIESTA's wrapper `--help` promised a timestamped backup directory** that the
  launcher had stopped creating the day before, while citing the section that
  says the opposite. PySCF's copy had been corrected and SIESTA's had not. The
  entry has one writer now (`runwrap._cold_usage_entry`) and a test reads the
  *generated* help.
- **A test that proves the framework against a stub proves nothing about the
  engines.** `test_deck_runner.py` used a stub whose layout was already the
  table the contract describes, so every promise held for it while both real
  engines violated them. It now asks the real forms too.

## How to work (these were the repeated failures)

- **Code top-down from the contract. Do not read the old implementation for
  guidance and do not preserve its behaviour by default.** Three convergence
  values in PySCF's tight rung were wrong because they were copied from
  `config/pyscf.py` instead of read from `tuning.md` § 2.4.
- **The contract may itself be stale — sweep it, don't work around it.** Fix the
  document that owns the concept, then sweep the restatements. A retraction
  stapled over a live table is harder to read than either version.
- **Prove a restructure output-neutral before believing it.** Render a matrix of
  decks across both engines and several configurations, normalise the timestamp
  and the git sha, and hash it. Every step of the 2026-08-19 restructure was
  gated on that digest; the one deliberate change (naming a section that had
  none) was measured at exactly one added line per deck.
- **Mutation-test every new guard.** Break the code, watch the test fail, put it
  back. A green test proves nothing until you have seen it go red.
- **Static analysis, not poking.** Read the code and the contract side by side.
  Tests would not have caught the wrong convergence numbers.
- **Targeted test runs**, not the 6800-test suite: `python tools/testrun.py run
  none2e <files>` then `status --fails`. Never edit source while a batch runs —
  it silently corrupts the result.
- **No obsolete claims in comments.** A comment describing the old mechanism is
  worse than none.
- SIESTA and PySCF must end up identical except their parameters and their
  engine — same stage names (`coarse`/`medium`/`tight`), same ladder shape, same
  framework for prep/bench/submit, both directory shapes. A rule that holds for
  one and is untested on the other is where they drift apart.

# E2E review, 2026-11-02/03 — does each e2e test still pin the right detail of the contract?

*(The user, 2026-11-02: "all e2e tests should be re-evaluated and reviewed in their full code since many details in
contract and design has been refined/revised. e2e tests may no longer pin the correct details. they need more work.
so make sure this happens before you run e2e tests to validate 2.65.4".)* The e2e batch had last run on 2026-10-31,
at about 2.56.2; 2.57.0 … 2.65.3 shipped without one (`plans/plan.md`, *How a release before F10 is checked*).

**What was read.** Every e2e file — 40 files, 565 tests — read whole against the contract as it reads at 2.65.3, by
nine read-only readers under [`brief.md`](brief.md), one file read by hand: a verdict per test (RIGHT, WRONG DETAIL, STALE ROAD, WEAK,
HANDCRAFTED, TEXT-PINNING, DUPLICATE, NOT E2E, RETIRE) and the checks the releases 2.57.0–2.65.3 named untested
that a run the pass already makes could carry (OWED). Every finding acted on was checked in the code by hand first:
[`verified.md`](verified.md).

| group | files | report | revisions |
|---|---|---|---|
| P | PySCF vibration, the live monitor, the PySCF relaxation outcome, the notebook, the monitor's start | [`P.md`](P.md) | [`fix-P.md`](fix-P.md) |
| S1 | SIESTA vibration, SIESTA relaxation run | [`S1.md`](S1.md) | [`fix-S1.md`](fix-S1.md) |
| S2 | SIESTA optimization, stopped run, the road on real runs, sequence of groups, keyword smoke, wall, GPU | [`S2.md`](S2.md) | [`fix-S2.md`](fix-S2.md) |
| T1, T2 | the real junction (fixtures and tests 1–2585; 2586–5230) | [`T1.md`](T1.md), [`T2.md`](T2.md) | [`fix-T.md`](fix-T.md) |
| W1 | Task setup (prep, cell types, one folder, restore), the bench summary | [`W1.md`](W1.md) | [`fix-W1.md`](fix-W1.md) |
| W2 | Build, this machine, the chemistry card, the spectrum form | [`W2.md`](W2.md) | [`fix-W2.md`](fix-W2.md) |
| W3 | the app, MolView, VibrationView, results export | [`W3.md`](W3.md) | [`fix-W3.md`](fix-W3.md) |
| W4 | the Results shell, presenters, sidebar, preview, notifications, load strip, layout, docs pane | [`W4.md`](W4.md) | [`fix-W4.md`](fix-W4.md) |
| X | the spectrum chart's box | [`X.md`](X.md) | (by hand) |

**How it was revised.** Eight fixers under [`fix_brief.md`](fix_brief.md): a wrong detail revised to the contract,
never to the code's output; a stale road moved onto the live one; a test on a faked failure, a duplicate or a test of
something no page mounts retired; an owed check added on a run the pass already makes. Handcrafted inputs in tests
that are otherwise right were left for F9 and listed. Where a finding was a defect of the product or of the
contract, it was fixed there (the ledger's E2E-P).

**Read again.** Round 1 — four readers of the revisions and the release's code, under
[`review_brief.md`](review_brief.md): [`rev-R1.md`](rev-R1.md) (the real junction and option B),
[`rev-R2.md`](rev-R2.md) (SIESTA, PySCF, the monitor, the notebook), [`rev-R3.md`](rev-R3.md) (Task setup, the
bench, Build, the chemistry card and the method's source), [`rev-R4.md`](rev-R4.md) (the Results pages, the GPU
notice, the documents). Round 2 — one fresh reader of the whole release, told round 1's triage:
[`rev-round2.md`](rev-round2.md).

**Where it went.** The finish-line ledger, *Found by the e2e review and 2.65.4's review rounds*
([`2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md)); the changelog's 2.65.4 row.

# Retired — MolView's old tests

These nineteen files tested the module in `../`, the frozen tree. They are here,
outside `testpaths = ["tests"]`, so pytest does not collect them: **retired, not
adapted and not repointed.**

They are kept for the same reason the frozen code is — as reference while the
module is rebuilt — and they are deleted with it at closeout (plan step H).

## Why they could not come across

Every test is derived from the contract, never from the source
([`molview.md`](../../../../../../docs/web/molview.md) § 13). These were largely
the opposite, which § 13 names as the thing being corrected: they pin the names
a returned object happens to carry, so they pass for a surface that has drifted
away from the document and fail for a rename that changed nothing (§ 13.1).

Repointing them at the new tree would carry that inversion into the rebuild —
the new code would be measured against the old code's shape instead of against
the contract. So each layer's tests are written fresh, from § 13.3's rows, as
that layer lands.

## What replaces them, and when

§ 13.3 is the test plan: *"a rule with no row here is a rule nothing guards."*
Each step of the rebuild writes its own rows' tests first and runs only those
(plan § 1). The demo and the full suite verify nothing about a single layer —
a page-level test fails for anything on the page.

## What was left behind on purpose

Tests belonging to **other** modules that merely reach into MolView — the
workspace dispatcher, the results blueprint, the atom-index guard, the XSS and
CSS audits — are not MolView's tests and were not touched. They break while the
module is rebuilt. That is expected and is not repaired from here (plan § 1).

"""`prep` refuses before it writes -- through the road: `jobset init`, `prep`.

PINS: ``docs/execution/job-system.md`` § 5.4 (*what cannot be taken is
refused before anything is written too: a run that is not an attempt of this
calculation, `--from` with `--cold`, either on the flat layout or a bias
scan, either on a bench -- and the attempt is opened once, with what it
carries, so a refusal leaves an attempt an earlier prep set up as it was*)
and § 5.3 (every decision the entry makes lands in the ledger; both doors
refuse through the one entry); plan W52.

PREVENTS, read in the code before 2026-10-01 (the W52 review): a mistyped
`--time` dumping a traceback.

Nothing here launches an engine.
"""
from __future__ import annotations

import json

from support.road import describe_calculation, jobset


def _prep(bundle, stage, *more):
    return jobset("prep", "run", stage, "--bundle", bundle,
                  "--target", "this", *more)


def _ledger(bundle):
    from molbuilder.jobset.ledger import LEDGER_FILE
    return [json.loads(ln) for ln in
            (bundle / LEDGER_FILE).read_text().splitlines()]


def test_a_mistyped_amount_is_refused_by_its_flag(tmp_path, monkeypatch):
    """`--time 4x`, `--mem 12Q`, `--gpus a100x2`: each refused in the verb's
    voice, naming the flag -- never a traceback.

    MUTATION THIS MUST FAIL AGAINST: the allocation built from the raw
    flags (the record's own refusal escapes as an exception)."""
    bundle = describe_calculation(tmp_path, monkeypatch)
    for flag, said in (("--time", "4x"), ("--mem", "12Q"),
                       ("--gpus", "a100x2")):
        r = _prep(bundle, "coarse", flag, said)
        assert r.exit_code != 0 and f"{flag}:" in r.output, (flag, r.output)

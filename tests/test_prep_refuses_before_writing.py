"""`prep` refuses in the verb's voice -- through the road: `molbuilder describe`, `prep`.

PINS: ``docs/execution/job-system.md``, *Every refusal, and its way on* (a
`--time`, `--mem` or `--gpus` that is no amount is said where it is typed);
``docs/execution/run-identity.md`` § 3 (a label longer than its engine keeps is
refused at prep).  What `--from` and `--cold` cannot take (`job-system.md`
§ 5.4) is `tests/data/prep_protocol.toml`'s.

PREVENTS, read in the code before 2026-10-01 (the W52 review): a mistyped
`--time` dumping a traceback.

Nothing here launches an engine.
"""
from __future__ import annotations

import pytest

from support.road import describe_calculation, group_words, jobset


def _prep(bundle, stage, *more):
    return jobset("prep", "task", *group_words(bundle, stage), "--bundle", bundle,
                  "--target", "this", *more)


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


#: A LABEL LONGER THAN ITS ENGINE KEEPS WHOLE (`execution/run-identity.md`
#: § 3, the engine record's ``longest_label``): SIESTA cuts its file names
#: past 48 characters, so prep refuses a longer label in SIESTA's words;
#: PySCF cuts none, and keeps it.  Until 2.64.0 the 48 was SIESTA's limit
#: asked of every engine.
LONG_LABEL = "H2_" + "a" * 46


@pytest.mark.parametrize("engine, refused", [("siesta", True),
                                             ("pyscf", False)])
def test_a_label_longer_than_its_engine_keeps_is_refused_at_prep(
        tmp_path, monkeypatch, engine, refused):
    """A 49-character label: refused at prep for SIESTA, naming its length
    and the 48 SIESTA keeps; prepared for PySCF.

    MUTATION THIS MUST FAIL AGAINST: the limit read as a constant for every
    engine (the PySCF row is refused)."""
    bundle = describe_calculation(tmp_path, monkeypatch, name=LONG_LABEL,
                                  engine=engine)
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    if refused:
        assert r.exit_code != 0, r.output
        assert "49 characters" in r.output and "SIESTA" in r.output \
            and "48" in r.output, r.output
    else:
        assert r.exit_code == 0, r.output

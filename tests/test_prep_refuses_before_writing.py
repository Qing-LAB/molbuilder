"""`prep` refuses before it writes -- through the road: `jobset init`, `prep`.

PINS: ``docs/execution/job-system.md`` § 5.4 (*what cannot be taken is
refused before anything is written too: a run that is not an attempt of this
calculation, `--from` with `--cold`, either on the flat layout or a bias
scan, either on a bench -- and the attempt is opened once, with what it
carries, so a refusal leaves an attempt an earlier prep set up as it was*)
and § 5.3 (every decision the entry makes lands in the ledger; both doors
refuse through the one entry); plan W52.

PREVENTS, each read in the code before 2026-10-01 (the W52 review):

* a mistyped `--from` refused only after the five steps had rendered --
  and, on a re-prep, after they had undone what an earlier prep had carried
  in (re-preps are refused since 2026-10-02, `job-system.md` § 5.0);
* `--cold` on the flat layout, and `--from` / `--cold` on a bench, refused
  (or ignored) only after the decks were written;
* `--from` with `--cold` silently taking `--cold`, and a path out of the
  calculation copied from, at the terminal -- while the browser refused both;
* a refusal leaving the ledger's last line for the stage at `prepped`;
* a mistyped `--time` dumping a traceback.

A finished run is the measured H2 relaxation (`support.road`).  Nothing here
launches an engine.
"""
from __future__ import annotations

import json

from support.road import a_finished_run, describe_h2, jobset


def _prep(bundle, stage, *more):
    return jobset("prep", "run", stage, "--bundle", bundle,
                  "--target", "this", *more)


def _ledger(bundle):
    from molbuilder.jobset.ledger import LEDGER_FILE
    return [json.loads(ln) for ln in
            (bundle / LEDGER_FILE).read_text().splitlines()]


# `test_a_mistyped_from_leaves_the_prepared_attempt_as_it_was` retired
# 2026-10-02 with re-preps: a prepped stage is refused before anything is
# read (`job-system.md` § 5.0), so a mistyped `--from` is a FIRST prep's,
# below.


def test_what_cannot_be_named_is_refused_before_anything_is_written(
        tmp_path, monkeypatch):
    """Each before a single file of the stage is written: `--from` with
    `--cold`, a path out of the calculation, and a run that does not exist
    -- refused in its words and in the ledger -- at the terminal as in the
    browser; `--cold` on a bench; `--cold` on the flat layout, which says
    how a flat stage starts clean.

    MUTATIONS THIS MUST FAIL AGAINST: `--from` with `--cold` taking
    `--cold`; the flat refusal after the five steps (a deck written)."""
    bundle = describe_h2(tmp_path, monkeypatch)
    assert _prep(bundle, "coarse").exit_code == 0
    a_finished_run(bundle / "01_coarse" / "run-0")
    for more, said in ((("--from", "01_coarse/run-0", "--cold"),
                        "two answers to one question"),
                       (("--from", "../elsewhere"),
                        "names a run of this calculation"),
                       (("--from", "01_coarse/run-9"), "no such attempt")):
        r = _prep(bundle, "medium", *more)
        assert r.exit_code != 0 and said in r.output, (more, r.output)
        assert not (bundle / "02_medium").exists(), (more, "written")
        last = _ledger(bundle)[-1]
        assert (last["verb"], last["decision"], last["stage"]) == (
            "prep", "refused", "medium") and said in last["reason"], last
    r = jobset("prep", "bench", "coarse", "--bundle", bundle,
               "--target", "this", "--cold")
    assert r.exit_code != 0 and "a bench trial" in r.output, r.output
    assert not (bundle / "01_coarse" / "bench").exists()

    flat = describe_h2(tmp_path / "flat", monkeypatch, shape="flat")
    r = _prep(flat, "coarse", "--cold")
    assert r.exit_code != 0, r.output
    assert "restart: clean" in r.output, r.output
    assert not list(flat.glob("H2_01_coarse.*")), (
        "the flat refusal came after the deck was written: "
        + ", ".join(p.name for p in flat.iterdir()))


def test_a_mistyped_amount_is_refused_by_its_flag(tmp_path, monkeypatch):
    """`--time 4x`, `--mem 12Q`, `--gpus a100x2`: each refused in the verb's
    voice, naming the flag -- never a traceback.

    MUTATION THIS MUST FAIL AGAINST: the allocation built from the raw
    flags (the record's own refusal escapes as an exception)."""
    bundle = describe_h2(tmp_path, monkeypatch)
    for flag, said in (("--time", "4x"), ("--mem", "12Q"),
                       ("--gpus", "a100x2")):
        r = _prep(bundle, "coarse", flag, said)
        assert r.exit_code != 0 and f"{flag}:" in r.output, (flag, r.output)

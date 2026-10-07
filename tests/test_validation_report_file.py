"""The deck's companion validation report — what the checks said, on disk.

**The gap this closes.** A generated deck carried its header, provenance,
benchmark declarations, atom metadata and the user's own section — and no
trace of what the checker said about it.  Warnings scrolled past in a terminal
and were gone, so a `.fdf` opened six months later said nothing about the
advice its author was given.  `info` findings reached no surface at all.

**A separate file, not a block inside the deck** (user, 2026-08-23), and the
choice removes a real problem: the artifact gate's subject is the file on
disk, so findings written INTO that file would mean the bytes that were
checked are not the bytes that ship.  Beside it, the deck is final the moment
it is checked.
"""
from __future__ import annotations


import pytest

from molbuilder.issues import Issue
from molbuilder.script_emit import VALIDATION_SUFFIX, write_validation_report


def test_a_clean_deck_still_gets_a_report_that_claims_nothing(tmp_path):
    """Absence of findings is not a certificate, and the empty form says so —
    otherwise a silent file reads as approval."""
    out = write_validation_report(tmp_path / "JOB.fdf", [])
    body = out.read_text()
    assert "nothing to say" in body
    assert "not a claim that the calculation is right" in body


def test_the_report_names_the_deck_it_is_about(tmp_path):
    out = write_validation_report(tmp_path / "JOB_01_coarse.fdf", [])
    assert out.name == "JOB_01_coarse" + VALIDATION_SUFFIX
    assert "JOB_01_coarse.fdf" in out.read_text()


def test_every_severity_reaches_the_file(tmp_path):
    """`science/validation.md` R4 -- all three, no surface quietly dropping
    one.  `info` was the one being dropped, everywhere."""
    out = write_validation_report(tmp_path / "JOB.fdf", [
        Issue("error", "the deck disagrees with itself", "deck.x"),
        Issue("warn", "mesh cutoff is low", "config.mesh_cutoff"),
        Issue("info", "high-spin Fe(II) is likely", "chemistry.spin")])
    body = out.read_text()
    for level in ("ERROR", "WARN", "INFO"):
        assert level in body, f"{level} findings never reach the file"
    assert "[chemistry.spin]" in body, "the id a reader needs is dropped"


def test_the_report_is_declared_as_a_file_molbuilder_wrote():
    """Conflict C of the plan.  `identity.OUR_FILE_PATTERNS` has a second
    reader — `runwrap._cold_restart_block` derives `--cold`'s *"except what
    molbuilder wrote"* exception from it — so an undeclared file reads as
    ENGINE OUTPUT, and `prep` greets a fresh calculation with *"already under
    way, warm files at the root"*.  That trap sprang once already, on the
    `.source.xyz` pair."""
    from molbuilder.identity import OUR_FILE_PATTERNS
    assert "{label}.validation.txt" in OUR_FILE_PATTERNS
    assert "{label}_*.validation.txt" in OUR_FILE_PATTERNS, (
        "the per-rung form is missing, so a staged deck's report would read "
        "as engine output")


# --------------------------------------------------------------------- #
#  R4 -- the CLI prints the same three                                   #
# --------------------------------------------------------------------- #

def _printed(*issues):
    import io
    from molbuilder.validation import report
    buf = io.StringIO()
    report(list(issues), stream=buf)
    return buf.getvalue()


def test_an_advisory_reaches_the_terminal():
    """**`info` reached no surface at all until 2026-08-23.**

    `report` looped on ``severity == "warn"`` and dropped the rest, so an
    advisory -- the class written precisely for a person to weigh against
    their own system -- was computed on every render and printed nowhere,
    while `science/validation.md` R4 said *"the CLI prints the same three"*.
    """
    out = _printed(Issue("info", "high-spin Fe(II) is likely", "chemistry.spin"))
    assert "high-spin Fe(II) is likely" in out, (
        "an info finding is printed nowhere -- R4 says the CLI prints all "
        "three severities, and dropping one to keep the screen quiet is the "
        "downgrade R4 forbids by name")
    assert "info" in out, "the severity is not named, so it reads as a warning"


def test_a_warning_still_reaches_it_and_says_which_it_is():
    out = _printed(Issue("warn", "mesh cutoff is low", "config.mesh_cutoff"))
    assert "warn [config.mesh_cutoff]: mesh cutoff is low" in out


def test_an_error_raises_rather_than_printing():
    """R4's own distinction, not a downgrade: `error` blocks generation and
    says why, so it travels as the exception rather than a line on stderr."""
    from molbuilder.validation import ValidationError
    with pytest.raises(ValidationError):
        _printed(Issue("error", "impossible spin",
                       "config.unpaired_electrons"))

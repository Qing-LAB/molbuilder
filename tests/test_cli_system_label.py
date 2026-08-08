"""The `SystemLabel` a deck carries is the calculation's id.

Contract: ``docs/execution/run-identity.md`` § 2 — *"That one id **is** the
`SystemLabel` / `JOB` literal. There is no second name"*, and the note beneath
it: SIESTA is fed the deck on **stdin**, so the *filename* never reaches the
engine and every file it writes is named from the `SystemLabel` line inside.
Also ``docs/execution/job-contracts.md`` § 2.1 Rule 2 (one basename per job)
and § 3 rule 2 there (the normalised result is shown, not hidden — by every
surface, the terminal included).

**What was wrong until 2026-08-08.** The two branches of ``molbuilder fdf``
disagreed. The staged branch aligned the label to the output stem; the plain
branch left the dataclass default, so ``molbuilder fdf w.xyz clean.fdf`` wrote
a deck whose every output was called ``siesta.*``. Two such runs in one folder
silently shared restart state — § 1's first failure mode, reached by doing
nothing unusual — and no filename revealed it, because filenames are not what
SIESTA used.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from molbuilder.cli import cli


_XYZ = textwrap.dedent("""\
    3

    O  0.000  0.000  0.000
    H  0.760  0.590  0.000
    H -0.760  0.590  0.000
""")


@pytest.fixture
def xyz(tmp_path):
    p = tmp_path / "water.xyz"
    p.write_text(_XYZ)
    return p


def _run(*args):
    return CliRunner().invoke(cli, [str(a) for a in args],
                              catch_exceptions=False)


def _label_of(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.split()[:1] == ["SystemLabel"]:
            return line.split()[1]
    raise AssertionError(f"no SystemLabel in {path.name}")


# --------------------------------------------------------------------- #
#  The defect itself                                                     #
# --------------------------------------------------------------------- #

def test_the_plain_path_names_the_deck_after_the_calculation(xyz, tmp_path):
    """The regression this whole change exists for. This used to be
    ``siesta`` no matter what the user called the file."""
    out = tmp_path / "bdt-relax.fdf"
    assert _run("fdf", xyz, out).exit_code == 0
    assert _label_of(out) == "bdt-relax"


def test_the_two_branches_agree(xyz, tmp_path):
    """Asserted as an equality between the branches, not as two separate
    expected strings: the failure was that they *disagreed*, and pinning
    each one's answer separately is how that stayed invisible."""
    plain = tmp_path / "same.fdf"
    _run("fdf", xyz, plain)

    staged_root = tmp_path / "staged"
    staged_root.mkdir()
    staged = staged_root / "same.fdf"
    _run("fdf", xyz, staged, "--stage-strategy", "loose-only")

    produced = sorted(staged_root.glob("same_*.fdf"))
    assert produced, "the staged branch emitted no per-stage deck"
    assert {_label_of(p) for p in produced} == {_label_of(plain)}


def test_rule_2_one_basename_per_job(xyz, tmp_path):
    """`job-contracts.md § 2.1` Rule 2: *every file the generator writes and a
    reader later opens carries the same basename, and for SIESTA the basename
    IS the SystemLabel.* A deck called `x.fdf` whose outputs are `siesta.*`
    breaks it."""
    out = tmp_path / "junction.fdf"
    _run("fdf", xyz, out)
    assert _label_of(out) == out.stem


# --------------------------------------------------------------------- #
#  Where the name comes from                                             #
# --------------------------------------------------------------------- #

def test_an_explicit_label_wins_over_the_filename(xyz, tmp_path):
    """Both are things the user *typed*, which is § 2's "built from inputs";
    the explicit one is the more specific statement of intent."""
    out = tmp_path / "ignored.fdf"
    _run("fdf", xyz, out, "--system-label", "BDT_Au_relax")
    assert _label_of(out) == "BDT_Au_relax"


def test_the_name_goes_through_the_one_normaliser(xyz, tmp_path):
    """§ 3 rule 1 — normalisation happens once, in one place. A `/` and a
    space would break a shell line, a glob and a scheduler argument, so a
    raw value must never reach the deck."""
    out = tmp_path / "ignored.fdf"
    _run("fdf", xyz, out, "--system-label", "BDT/Au relax")
    assert _label_of(out) == "BDT_Au_relax"


def test_case_survives_because_a_formula_carries_meaning_in_it(xyz, tmp_path):
    """§ 2.0 (decided 2026-08-08): lowercasing `Co` would yield the token
    `CO` also lowercases to."""
    out = tmp_path / "ignored.fdf"
    _run("fdf", xyz, out, "--system-label", "magnet_Co4")
    assert _label_of(out) == "magnet_Co4"


# --------------------------------------------------------------------- #
#  Shown, and refused                                                    #
# --------------------------------------------------------------------- #

def test_the_resolved_name_is_printed(xyz, tmp_path):
    """§ 3 rule 2, and decision #20: shown by **every** surface, the terminal
    included. It is the thing that decides whether the next run continues, so
    hiding it hides that."""
    out = tmp_path / "bdt-relax.fdf"
    r = _run("fdf", xyz, out)
    assert "SystemLabel: bdt-relax" in r.output


def test_a_name_that_cannot_normalise_is_a_clean_error_not_a_traceback(
        xyz, tmp_path):
    """§ 3 rule 3 says *say so and ask*; a stack trace is neither. Same
    house rule `--stages-json` is already pinned to."""
    out = tmp_path / "ignored.fdf"
    r = CliRunner().invoke(
        cli, ["fdf", str(xyz), str(out), "--system-label", "Über"])
    assert r.exit_code != 0
    assert "Traceback" not in r.output
    assert "Ü" in r.output
    assert "--system-label" in r.output


def test_a_refused_name_writes_nothing(xyz, tmp_path):
    """The refusal has to arrive *before* the deck exists, or it is a report
    about a file that was already written."""
    out = tmp_path / "Über.fdf"
    CliRunner().invoke(cli, ["fdf", str(xyz), str(out)])
    assert not out.exists()
    assert not list(tmp_path.glob("*.fdf"))

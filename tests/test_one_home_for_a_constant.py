"""A physical constant is spelled in exactly one place.

Before 2026-08-30 the Bohr radius was written out **eight times in three
different values** — `0.529177210903`, `0.5291772108` and `0.529177` — and the
consequence was not theoretical: two modules read the same SIESTA `.XV` file
using the first and the third, so **the same file gave coordinates 4e-7 apart
depending on which reader was asked**.  It surfaced as a test comparing the two
answers and failing by 1.6e-6 Å on a gold lattice constant.

The size of the discrepancy is not the point.  A physical constant is a fact
about the universe, not about a module; eight copies are eight things that can
be edited apart, and nothing would have told anyone.

So: `molbuilder/constants.py` holds them, everything else imports.  This test is
what makes that a rule rather than a wish — `execution/architecture.md` § 7's
own phrasing, and the reason it exists.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "molbuilder"
HOME = PKG / "constants.py"

#: The digit strings that mean a constant `constants.py` owns.  Written as
#: prefixes so a truncation (`0.529177`) is caught as readily as the full value.
_OWNED = {
    "0.529177": "BOHR_ANGSTROM",
    "27.21138": "HARTREE_EV",
    "13.60569": "RYDBERG_EV",
}

#: The one file allowed to retype a value, and why.
#:
#: `trajectory_log/emitter.py`'s class body is copied VERBATIM into the
#: standalone script the user runs on a cluster, where molbuilder is not
#: importable — an import there becomes a NameError several machines away from
#: the edit.  (Confirmed by running it: the generated preview script died on
#: exactly that line.)  The literal carries a comment saying so.
_ALLOWED = {
    "trajectory_log/emitter.py": "emitted verbatim into a standalone script",
}

#: Values that are NOT copies of anything here.  `51.42208619` is the
#: ASE / NIST-historical Hartree·Bohr⁻¹ → eV·Å⁻¹ figure, deliberately chosen
#: over the CODATA-derived 51.422067476 so emitted forces line up with what
#: users read in ASE / VASP / QE logs.  Both call sites say so.
_NOT_A_COPY = ("51.42208619",)


#: The suite is inside the net too, since 2026-09-09.  It was not, and by then
#: `tests/` held FOUR spellings of the Bohr constant across five files --
#: `0.5291772108` twice, `0.529177249` (CODATA 1986) and `0.529177` -- none
#: matching `constants.py`'s `0.529177210903`.  A guard that policed only the
#: package while the suite drifted was measuring the wrong half: a test that
#: retypes a constant asserts against a second definition, which is the exact
#: failure this file exists to prevent.
#:
#: Fixtures may still be BUILT from the constant -- importing to write input is
#: not circular, because the assertion is on the value that comes back.
_ROOTS = (PKG, REPO / "tests")


def _python_files():
    me = Path(__file__).resolve()
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
            # `constants.py` is the home; THIS file spells the digit prefixes
            # in `_OWNED` because that is how it recognises them.
            if path == HOME or path.resolve() == me:
                continue
            if "__pycache__" in path.parts:
                continue
            yield path


def _rel(path: Path) -> str:
    """Package paths stay bare (`trajectory_log/emitter.py`) so the existing
    `_ALLOWED` keys keep working; suite paths carry their `tests/` prefix."""
    try:
        return str(path.relative_to(PKG))
    except ValueError:
        return str(path.relative_to(REPO))


def _code_only(text: str) -> str:
    """The file with comments and docstrings blanked.

    A constant NAMED in prose is not a second definition — several modules
    explain which value they use and why, and a guard that could not tell the
    difference would forbid the explanation along with the duplication.
    """
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    return re.sub(r"#.*$", "", text, flags=re.M)


@pytest.mark.parametrize("digits,name", sorted(_OWNED.items()))
def test_no_module_retypes_a_constant(digits, name):
    offenders = []
    for path in _python_files():
        rel = _rel(path)
        if rel in _ALLOWED:
            continue
        code = _code_only(path.read_text(encoding="utf-8"))
        for other in _NOT_A_COPY:
            code = code.replace(other, "")
        for lineno, line in enumerate(code.splitlines(), 1):
            if digits in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"{name} is spelled outside molbuilder/constants.py:\n  "
        + "\n  ".join(offenders)
        + f"\n\nImport it instead: `from molbuilder.constants import {name}`. "
        f"If a call site genuinely cannot import — the only case so far is "
        f"source text emitted into a standalone script — add it to _ALLOWED "
        f"with the reason.")


def test_the_allowance_is_still_needed():
    """An exception nobody re-checks becomes a habit.

    This fails if the emitter stops carrying the literal, so the allowance is
    deleted at the moment it stops describing the code rather than years later.
    """
    for rel, why in _ALLOWED.items():
        text = (PKG / rel).read_text(encoding="utf-8")
        assert any(d in text for d in _OWNED), (
            f"{rel} is allowed to retype a constant ({why}) but no longer "
            f"does. Delete its entry from _ALLOWED.")


def test_the_guard_can_actually_see_a_violation():
    """A lint whose pattern never matches stays green over a regression."""
    assert "0.529177" in HOME.read_text(encoding="utf-8")
    assert _code_only('x = 0.529177  # a comment\n"""0.529177"""\n').count(
        "0.529177") == 1, "the blanking removed a real definition, or kept prose"

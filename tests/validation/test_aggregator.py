"""Tests for molbuilder.validation.__init__.

Per docs/process/testing.md (test layout mirrors source
layout).  Split from the pre-2026-06-13 flat tests/test_validation.py
on 2026-06-13; no test body was modified.  Shared fixtures
(``water_struct``, ``_vacuum_cell``) live in tests/validation/conftest.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate
from ._helpers import _vacuum_cell
from tests.spectra._helpers import _spectra_cfg


def test_issue_severity_accepts_error_warn_info():
    """Severity is restricted to error / warn / info.  Info was
    added 2026-05-22 for advisory hints (e.g. 'Fe + spin=4 implies
    high-spin Fe(II)') that don't add to the warn count; renamed
    from the old "error or warn only" pin."""
    Issue("error", "fine")
    Issue("warn",  "fine")
    Issue("info",  "fine")
    with pytest.raises(ValueError, match="severity"):
        Issue("debug", "not allowed")


def test_validation_error_carries_issues():
    issues = [
        Issue("warn", "minor", "x"),
        Issue("error", "fatal", "y"),
    ]
    with pytest.raises(ValidationError) as exc:
        raise ValidationError(issues)
    assert exc.value.issues == issues
    # Message lists the error but not the warn.
    assert "fatal" in str(exc.value)
    # The warning should NOT be in the formatted error message;
    # warnings get their own stderr path via report().
    assert "minor" not in str(exc.value)


def test_validation_error_rejects_empty_or_warn_only():
    with pytest.raises(ValueError, match="error-severity"):
        ValidationError([])
    with pytest.raises(ValueError, match="error-severity"):
        ValidationError([Issue("warn", "just a warning")])


# --------------------------------------------------------------------- #
#  report() helper: warnings to stderr, raise on errors                 #
# --------------------------------------------------------------------- #


def test_report_prints_warnings_to_stream():
    buf = io.StringIO()
    report(
        [Issue("warn", "watch out", "test.case")],
        raise_on_error=False, stream=buf,
    )
    out = buf.getvalue()
    assert "watch out" in out
    assert "[test.case]" in out


def test_report_raises_on_error_by_default():
    with pytest.raises(ValidationError):
        report([Issue("error", "fatal", "x")])


def test_report_emits_warnings_even_when_also_raising():
    """A run with both warnings and errors should surface BOTH -- the
    user wants to see all the warnings even if the error blocks
    emission."""
    buf = io.StringIO()
    issues = [
        Issue("warn", "minor first", "a"),
        Issue("warn", "minor second", "b"),
        Issue("error", "fatal", "c"),
    ]
    with pytest.raises(ValidationError):
        report(issues, stream=buf)
    out = buf.getvalue()
    assert "minor first" in out
    assert "minor second" in out


# --------------------------------------------------------------------- #
#  Wire-in: render_fdf and render_script call validate()                #
# --------------------------------------------------------------------- #


def test_render_fdf_raises_on_overlapping_atoms():
    """A structure with atoms < 0.3 Å apart triggers a min-distance
    error from validate(), which render_fdf surfaces as
    ValidationError before emitting any FDF text."""
    from molbuilder.siesta import render_fdf
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]),
        vacuum=(10.0, 10.0, 10.0),   # non-degenerate box so validate() is reached
    )
    with pytest.raises(ValidationError) as exc:
        render_fdf(s, SiestaConfig())
    assert "min_distance" in str(exc.value)


def test_render_fdf_emits_warnings_to_stderr(capsys, water_struct):
    """A spin_total without spin_treatment warning surfaces on stderr;
    the FDF still gets emitted (warnings don't block)."""
    from molbuilder.siesta import render_fdf
    cfg = SiestaConfig(spin_treatment="non-polarized", spin_total=1.0)
    fdf = render_fdf(water_struct, cfg)
    err = capsys.readouterr().err
    assert "spin_total" in err
    # FDF was still generated:
    assert "SystemName" in fdf


def test_render_script_raises_on_negative_spin(water_struct):
    """spin = -1 -> error from validate(), render_script raises
    ValidationError before emitting any Python text.

    Use UKS so the gate's restricted-method refusal (error-level since
    G-1c) doesn't pre-empt the negative-spin catch this test is about.
    """
    from molbuilder.pyscf import render_script
    cfg = PySCFConfig(spin=-1, method="UKS")
    with pytest.raises(ValidationError) as exc:
        render_script(water_struct, cfg)
    assert "spin" in str(exc.value)


def test_render_script_warns_on_open_shell_with_rks(capsys, water_struct):
    """Open-shell spin with closed-shell RKS / RHF method emits a
    warning to stderr but doesn't block emission."""
    from molbuilder.pyscf import render_script
    # (The old pyscf.input ValueError guard for RKS+spin retired with
    # G-1c, 2026-08-21 -- the GATE owns that refusal now, error-level.)
    # A LEGITIMATE open-shell system: the water cation (9 electrons,
    # one unpaired).  The old fixture said spin=1 on NEUTRAL water --
    # 10 electrons, an impossible pair -- and passed only while nothing
    # on this route checked parity (closed by G-1d, 2026-08-21).
    cfg = PySCFConfig(spin=1, method="UKS", net_charge=1)
    # For UKS + spin=1 on an odd-electron system the validator has
    # nothing to flag; this test documents that a *legitimate*
    # open-shell config doesn't warn.
    render_script(water_struct, cfg)
    err = capsys.readouterr().err
    # No "config.method" warn for a properly-set UKS config.
    assert "method" not in err or "warn [config.method]" not in err


# --------------------------------------------------------------------- #
#  ONE validation gate per engine (V1/V2 -- backend-architecture.md)    #
#                                                                       #
#  Every engine registers a validator so validate(struct, cfg) is the   #
#  single pass.  If a future edit drops Spectra/Transport from the      #
#  registry, those tabs silently skip their science again (the cross-   #
#  tab divergence this fixed) -- so pin the registration + the routing. #
# --------------------------------------------------------------------- #


def test_the_two_registries_hold_what_the_contract_says():
    """`science/validation.md`: an ENGINE row keys on a config class, a KIND
    row on `task.calculation` -- and which registry a science belongs in is
    decided by whether production constructs that class.

    TWO engine rows, not four.  Both retirements were the same finding a year
    apart in code time: `SpectraConfig` 2026-08-22 and `TransportConfig`
    2026-09-17 each keyed on a class no production caller validates, so the
    row dispatched for nothing.  A vibration's science and a junction's are
    the KIND's.

    Asserted by EQUALITY, not containment.  The version before 2026-09-17
    used `<=`, so a row could be added and never noticed -- and the row this
    one lost had been dead for a day under exactly that subset check.
    """
    from molbuilder.validation import _ENGINE_VALIDATORS, _KIND_VALIDATORS
    assert {c.__name__ for c in _ENGINE_VALIDATORS} == {
        "SiestaConfig", "PySCFConfig"}, (
        "the engine registry no longer matches `science/validation.md`; a row "
        "added here must name a config class production actually validates")
    assert set(_KIND_VALIDATORS) == {"vibration", "transport"}, (
        "a calculation kind lost its science -- this is the road a transport "
        "or vibration prep actually travels")


def test_the_render_gate_carries_the_science_but_not_the_selector(water_struct):
    """The render gate runs the SCIENCE (the hybrid grid_level advisory)
    but NOT the selector-availability check -- a top_n script is valid to
    emit, so the gate must not block on it (V1/V2).

    Called the way production calls it.  This went through
    `validate(struct, cfg)` and a `SpectraConfig` registry row until
    2026-08-22; the row keyed on a class nothing constructed, and the
    live door is the vibration KIND's."""
    from molbuilder.validation.spectra import spectra_render_checks
    cfg = _spectra_cfg(functional="B3LYP", grid_level=3,
                       es_mode_selection="top_n")
    issues = list(spectra_render_checks(water_struct, cfg))
    wheres = [i.where for i in issues]
    # render-gate science reached it (hybrid + low grid):
    assert any(w == "config.grid_level" for w in wheres), wheres
    # selector-availability (top_n soft-dep) is NOT a render gate, so it
    # must be absent here -- it lives in engine.selector_checks (preflight).
    assert not any(i.severity == "error"
                   and i.where == "config.es_mode_selection"
                   for i in issues), (
        "the top_n selector soft-dep leaked into the render gate")


# `test_validate_dispatches_transport_preflight` deleted 2026-09-17 with
# `TransiestaEngine`.  It called `validate(water_struct, TransportConfig())`
# and asked only that SOME transport-flavoured word appeared -- its own
# comment said so.  **No production caller ever made that call**: every
# transport rung resolves a `SiestaConfig`, and the two sites that build a
# `TransportConfig` build it as a projection for the NEGF block emitter and
# never validate it.  So the test pinned a dispatch that existed only for
# the test, and kept the dead registration alive for a day after the last
# real caller (`POST /api/transport/render`) was deleted.
#
# What a transport prep actually sees is `_KIND_VALIDATORS["transport"]`,
# covered by `tests/validation/test_transport_kind.py` -- each check there
# carrying a discriminating half and mutation-tested.


# `test_preflight_report_to_issues_bridge` deleted 2026-09-17 with
# `Check`/`PreflightReport`.  They were a CHECKLIST type for the
# cross-deck comparison -- distinct from `Issue` because they could
# report a PASSING gate -- and the comparison lost its subject when
# both decks came to be derived from one citation.


"""Spec invariants for the generated PySCF script.

These tests are deliberately decoupled from the implementation: they
encode the contract documented in ``docs/spec/pyscf-script.md``.  When
a spec change requires the code to deviate, both the spec doc AND
these tests must be updated in the same commit.

Style: each invariant test reads the *generated script's text* and
asserts properties of that text.  We don't import or call any
implementation helpers (e.g. _emit_preopt_block) -- that would defeat
the point.  If the generator ever switches to AST-based emission,
these tests still apply unchanged.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from molbuilder.pyscf_input import PySCFConfig, render_script
from molbuilder.structure import Structure


@pytest.fixture
def small_struct():
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0, 0, 0], [0.957, 0, 0], [-0.24, 0.927, 0]]),
        title="water",
    )


# --------------------------------------------------------------------- #
#  Output-file inventory: the docstring "Outputs:" block must match     #
#  the spec table exactly for each config variant.                      #
# --------------------------------------------------------------------- #


def _outputs_block(text: str) -> str:
    """Extract the 'Outputs:' section of the script's header docstring."""
    after = text.split("Outputs:", 1)[1]
    end = after.split("Dependencies:")[0]
    return end


@pytest.mark.parametrize("cfg, must_list, must_not_list", [
    # Default config: log + chk + initial + optimized + geom traj/log
    (
        PySCFConfig(),
        ["pyscf_relax.log", "pyscf_relax.chk",
         "pyscf_relax_initial.xyz", "pyscf_relax_optimized.xyz",
         "pyscf_relax_geom_optim.xyz", "pyscf_relax_geom.log"],
        ["_preopt"],
    ),
    # With pre-opt: also list the pre-opt trajectory files
    (
        PySCFConfig(preopt=True),
        ["pyscf_relax_preopt_optim.xyz", "pyscf_relax_preopt.log",
         "pyscf_relax_geom_optim.xyz", "pyscf_relax_geom.log"],
        [],
    ),
    # No optimization: no trajectory files, no _optimized.xyz
    (
        PySCFConfig(optimize=False),
        ["pyscf_relax.log", "pyscf_relax.chk",
         "pyscf_relax_initial.xyz"],
        ["_optim.xyz", "_optimized.xyz", "_geom.log"],
    ),
    # No trajectory: no streaming xyz, but still _initial / _optimized
    (
        PySCFConfig(write_trajectory=False),
        ["pyscf_relax.log", "pyscf_relax_optimized.xyz"],
        ["_optim.xyz"],
    ),
])
def test_header_outputs_block_matches_spec(small_struct, cfg, must_list,
                                            must_not_list):
    text = render_script(small_struct, cfg)
    block = _outputs_block(text)
    for needle in must_list:
        assert needle in block, (
            f"Outputs: block missing {needle!r} for cfg={cfg}\n"
            f"block was:\n{block}"
        )
    for forbidden in must_not_list:
        assert forbidden not in block, (
            f"Outputs: block lists {forbidden!r} but the spec says it "
            f"shouldn't appear for cfg={cfg}\nblock was:\n{block}"
        )


# --------------------------------------------------------------------- #
#  Logging contract: pre-opt -> production must NOT call gto.M(),       #
#  and stages must share the same stdout file handle.                   #
# --------------------------------------------------------------------- #


def _strip_comments(text: str) -> str:
    """Drop comment-only lines from generated python.  Inline comments
    on real code lines stay so we still match constructs accurately."""
    return "\n".join(
        ln for ln in text.splitlines()
        if not ln.lstrip().startswith("#")
    )


def test_preopt_does_not_truncate_log(small_struct):
    """Spec: 'It must NOT call gto.M(output=...) a second time -- that
    call opens the file in 'w' mode and truncates pre-opt log entries.'
    Test: between pre-opt's optimize() and the production mf=, no
    gto.M() call may appear in real code lines (comments may mention it).
    """
    text = render_script(small_struct, PySCFConfig(preopt=True))
    code = _strip_comments(text)

    # Slice between "Pre-opt done" print and the next "mf =" line.
    after_preopt = code.split('print("Pre-opt done', 1)[1]
    before_main_mf = after_preopt.split("mf = ", 1)[0]

    assert "gto.M(" not in before_main_mf, (
        "Forbidden by spec: post-preopt rebuild calls gto.M(...) which "
        "truncates <job>.log when output=<job>.log is passed.  Reuse "
        "mol_pre instead."
    )


def test_preopt_reuses_mol_pre(small_struct):
    """Spec: 'The transition must reuse mol_pre... only call mol.build()
    if the production basis differs from the pre-opt basis.'"""
    text = render_script(small_struct, PySCFConfig(preopt=True))
    code = _strip_comments(text)
    assert "mol = mol_pre" in code


# --------------------------------------------------------------------- #
#  Trajectory contract: every optimize() with write_trajectory=True     #
#  and optimizer="geometric" must include a prefix= kwarg.              #
# --------------------------------------------------------------------- #


_OPTIMIZE_BLOCK_RE = re.compile(
    r"^\s*\w+\s*=\s*optimize\s*\(\s*\n"   # x = optimize(
    r"(?P<body>(?:[^()]*\n)+?)"           # ... body lines
    r"^\s*\)",                            # closing paren on its own line
    re.MULTILINE,
)


def _optimize_calls(text: str):
    """Yield each ``... = optimize(...)`` block's body text."""
    code = _strip_comments(text)
    for m in _OPTIMIZE_BLOCK_RE.finditer(code):
        yield m.group("body")


@pytest.mark.parametrize("cfg, expected_prefixes", [
    # Default: just the production-stage optimize().
    (PySCFConfig(),                   ['JOB + "_geom"']),
    # Pre-opt enabled: BOTH optimize() calls must have prefix=.
    (PySCFConfig(preopt=True),        ['JOB + "_preopt"', 'JOB + "_geom"']),
])
def test_every_optimize_call_has_prefix_when_traj_on(small_struct,
                                                      cfg, expected_prefixes):
    """Spec: 'every call to optimize(...) in the generated script must
    include a prefix= kwarg' when write_trajectory=True."""
    bodies = list(_optimize_calls(render_script(small_struct, cfg)))
    assert len(bodies) == len(expected_prefixes), (
        f"Expected {len(expected_prefixes)} optimize() calls, "
        f"found {len(bodies)}"
    )
    for body, expected in zip(bodies, expected_prefixes):
        assert "prefix" in body, (
            f"optimize() block missing prefix= kwarg.  Body was:\n{body}"
        )
        assert expected in body, (
            f"optimize() prefix= mismatch.  Expected {expected!r}, body:\n{body}"
        )


def test_no_prefix_when_trajectory_off(small_struct):
    """Spec: when write_trajectory=False, no optimize() call should set
    prefix= (we're not asking geomeTRIC for a streaming file)."""
    cfg = PySCFConfig(write_trajectory=False)
    for body in _optimize_calls(render_script(small_struct, cfg)):
        assert "prefix" not in body, (
            f"optimize() set prefix= even though write_trajectory=False:\n"
            f"{body}"
        )


def test_no_optimize_calls_when_optimization_disabled(small_struct):
    """Spec: 'cfg.optimize=False produces a single-point script:
    mf.kernel() is called, no optimize(...)'."""
    cfg = PySCFConfig(optimize=False)
    bodies = list(_optimize_calls(render_script(small_struct, cfg)))
    assert bodies == [], (
        f"Found {len(bodies)} optimize() call(s) even though "
        f"cfg.optimize=False"
    )
    text = render_script(small_struct, cfg)
    assert "mf.kernel()" in text


# --------------------------------------------------------------------- #
#  Optimizer-import safety: missing dep -> friendly SystemExit          #
# --------------------------------------------------------------------- #


def test_optimizer_import_wrapped_in_try_except(small_struct):
    """Spec: 'imports it inside a try/except ImportError that raises
    SystemExit with an actionable message, not a 6-frame traceback'."""
    text = render_script(small_struct, PySCFConfig())
    # The geomopt import must be inside a try/except, with the except
    # raising SystemExit and instructing pip install.
    assert "from pyscf.geomopt.geometric_solver import optimize" in text
    assert "except ImportError" in text
    assert "raise SystemExit(" in text
    assert "pip install geometric" in text


# --------------------------------------------------------------------- #
#  Charge contract                                                      #
# --------------------------------------------------------------------- #


def test_charge_default_uses_phosphate_heuristic(deprotonated_diester):
    """Spec: 'Otherwise, fall back to formal_charge_from_phosphates'."""
    text = render_script(deprotonated_diester, PySCFConfig())
    # Diester missing both HOPs -> heuristic returns -1
    assert "charge     = -1," in text


def test_charge_explicit_zero_overrides_heuristic(deprotonated_diester):
    """Spec: 'If cfg.charge is not None, it wins (including
    cfg.charge=0)'."""
    text = render_script(deprotonated_diester, PySCFConfig(charge=0))
    assert "charge     = 0," in text


# --------------------------------------------------------------------- #
#  Cross-check: every script we generate must `compile()` cleanly       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("cfg", [
    PySCFConfig(),
    PySCFConfig(preopt=True),
    PySCFConfig(optimize=False),
    PySCFConfig(method="UKS", spin=1, charge=1),
    PySCFConfig(write_trajectory=False),
    PySCFConfig(solvent="water"),
    PySCFConfig(dispersion=None),
    PySCFConfig(threads=4),
    PySCFConfig(verbose_comments=False),
    PySCFConfig(preopt=True, basis="def2-TZVP", preopt_basis="def2-SVP"),
])
def test_every_variant_compiles(small_struct, cfg):
    text = render_script(small_struct, cfg)
    compile(text, "<spec-test>", "exec")

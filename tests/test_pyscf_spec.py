"""Spec invariants for the generated PySCF script.

These tests are deliberately decoupled from the implementation: they
encode the contract documented in ``docs/engines/pyscf.md``.  When
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

from molbuilder.pyscf import PySCFConfig, render_script
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
    # Default config: log + chk + initial + optimized + per-stage
    # geom traj/log placeholders (post-#534 commit 4: stages
    # generate one set per enabled stage).
    (
        PySCFConfig(),
        ["pyscf_relax.log", "pyscf_relax.chk",
         "pyscf_relax_initial.xyz", "pyscf_relax_optimized.xyz",
         "pyscf_relax_geom_<stage>_optim.xyz",
         "pyscf_relax_geom_<stage>.log"],
        ["_preopt"],
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
#  Logging contract: stages share the same stdout file handle.          #
# --------------------------------------------------------------------- #


def _strip_comments(text: str) -> str:
    """Drop comment-only lines from generated python.  Inline comments
    on real code lines stay so we still match constructs accurately."""
    return "\n".join(
        ln for ln in text.splitlines()
        if not ln.lstrip().startswith("#")
    )


def test_no_second_gto_M_call(small_struct):
    """The script must call ``gto.M(...)`` exactly once: building the
    initial mol.  Earlier preopt-era versions sometimes rebuilt via a
    second ``gto.M(output=...)`` call which truncated <JOB>.log.  Post-
    #534 commit 4 there's no preopt block, so any second gto.M() in the
    final script is a regression."""
    text = render_script(small_struct, PySCFConfig())
    code = _strip_comments(text)
    # Single gto.M(...) call -- the initial mol build.
    assert code.count("gto.M(") == 1


# --------------------------------------------------------------------- #
#  Trajectory contract: every optimize() with write_trajectory=True     #
#  and optimizer="geometric" must include a prefix= kwarg.              #
# --------------------------------------------------------------------- #


_OPTIMIZE_BLOCK_RE = re.compile(
    # Match either ``x = optimize(`` (loop body) or ``return
    # optimize(`` (inside the _mb_run_stage_opt helper introduced
    # in #534 6c).  Both shapes have the same multi-line arg list
    # we want to inspect.
    r"^\s*(?:\w+\s*=|return)\s*optimize\s*\(\s*\n"
    # Body lines: anything that ISN'T a bare ``)`` line.  Negative
    # lookahead lets body lines contain balanced parens (e.g. the
    # 2026-05-27 ``prefix = _mb_outfile(JOB + ".."),`` wrapping)
    # while still stopping at the function's closing paren below.
    r"(?P<body>(?:(?!^\s*\)\s*$).*\n)+?)"
    r"^\s*\)\s*$",                        # closing paren on its own line
    re.MULTILINE,
)


def _optimize_calls(text: str):
    """Yield each ``... = optimize(...)`` block's body text."""
    code = _strip_comments(text)
    for m in _OPTIMIZE_BLOCK_RE.finditer(code):
        yield m.group("body")


def test_optimize_call_has_per_stage_prefix_when_traj_on(small_struct):
    """Spec: when write_trajectory=True the single per-stage
    ``optimize()`` call inside the stages loop must pass a per-stage
    prefix= (the STAGE['name'] gets concatenated into the path) so
    each enabled stage's geomeTRIC trajectory lands in its own file.
    """
    cfg = PySCFConfig()
    bodies = list(_optimize_calls(render_script(small_struct, cfg)))
    assert len(bodies) == 1, (
        f"Expected exactly 1 optimize() call inside the stages loop, "
        f"found {len(bodies)}"
    )
    body = bodies[0]
    assert "prefix" in body
    assert "_mb_outfile(JOB + '_geom_' + STAGE['name'])" in body, (
        f"optimize() prefix must concat STAGE['name'] so each stage "
        f"gets its own trajectory file.  Body was:\n{body}"
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
    # raising SystemExit and directing the user to the managed backend.
    assert "from pyscf.geomopt.geometric_solver import optimize" in text
    assert "except ImportError" in text
    assert "raise SystemExit(" in text
    assert "bash scripts/install-env.sh bootstrap --yes" in text


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
    PySCFConfig(optimize=False),
    PySCFConfig(method="UKS", spin=1, charge=1),
    PySCFConfig(write_trajectory=False),
    PySCFConfig(solvent="water"),
    PySCFConfig(dispersion=None),
    PySCFConfig(threads=4),
    PySCFConfig(verbose_comments=False),
    PySCFConfig(basis="def2-TZVP"),
])
def test_every_variant_compiles(small_struct, cfg):
    text = render_script(small_struct, cfg)
    compile(text, "<spec-test>", "exec")

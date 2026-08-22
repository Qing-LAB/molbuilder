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
    # Default config: log + chk + initial + optimized + THIS RUNG's geom
    # trajectory and log.  A ladder is N decks and N jobs (`stages.md`
    # § 1.1a), so one deck lists one set -- the per-stage placeholders the
    # in-script loop needed are gone with it.
    (
        PySCFConfig(),
        ["pyscf_relax.log", "pyscf_relax.chk",
         "pyscf_relax_initial.xyz", "pyscf_relax_optimized.xyz",
         "pyscf_relax_geom_optim.xyz",
         "pyscf_relax_geom.log"],
        ["_preopt", "geom_<stage>"],
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


def test_optimize_call_carries_this_rung_s_trajectory_prefix(small_struct):
    """Spec: with ``write_trajectory=True`` the single ``optimize()`` call passes
    a ``prefix=`` naming **this rung**, so two rungs of one ladder cannot write
    into the same geomeTRIC trajectory.

    A PySCF ladder is N decks and N jobs (`stages.md` § 1.1a), so the separation
    that used to come from ``STAGE['name']`` inside an in-script loop now comes
    from the stage token the deck was rendered with -- the same token that
    suffixes the deck's own filename.
    """
    cfg = PySCFConfig()
    bodies = list(_optimize_calls(render_script(small_struct, cfg,
                                               stage_token="02_tight")))
    assert len(bodies) == 1, (
        f"Expected exactly 1 optimize() call, found {len(bodies)}")
    body = bodies[0]
    assert "prefix" in body
    assert "_mb_outfile(JOB + '_geom_02_tight')" in body, (
        f"optimize() prefix must carry this rung's token so two rungs get "
        f"separate trajectory files.  Body was:\n{body}"
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
    cfg.charge=0)'.  The neutral diester has an odd electron count
    (ΣZ = 59), so asserting charge 0 requires the open-shell spelling
    -- the parity gate (G-1d) rightly refuses charge 0 + spin 0 as an
    impossible pair."""
    text = render_script(deprotonated_diester,
                         PySCFConfig(net_charge=0, method="UKS", spin=1))
    assert "charge     = 0," in text


# --------------------------------------------------------------------- #
#  Cross-check: every script we generate must `compile()` cleanly       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("cfg", [
    PySCFConfig(),
    PySCFConfig(optimize=False),
    PySCFConfig(method="UKS", spin=1, net_charge=1),
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


# --------------------------------------------------------------------- #
#  The emitted program fills the directory it was LAUNCHED in           #
# --------------------------------------------------------------------- #


def test_emitted_outputs_stay_in_the_attempt_directory(small_struct, tmp_path):
    """A hierarchical attempt addresses the deck through a link
    (``run-0/<job>.py -> ../../<job>.py``) and owns everything the run
    produces (``project-layout.md``, the "attempt" row: *a run,
    immutable*).  So the emitted path anchor must NOT follow the link:
    every ``_mb_outfile(...)`` product belongs beside the path that was
    invoked, not beside the link's target.

    Executed against the EMITTED prelude itself -- the ``_MB_SCRIPT_DIR``
    assignment and ``_mb_outfile`` definition are cut out of a rendered
    deck and run with ``__file__`` set to the link -- because the defect
    this pins (``resolve()`` walking out of the attempt, found by the
    2026-08-19 E2E run) lived in emitted text no import ever executes.
    """
    import ast

    text = render_script(small_struct, PySCFConfig(job_name="w"))
    tree = ast.parse(str(text))
    keep = [n for n in tree.body
            if (isinstance(n, ast.ImportFrom) and n.module == "pathlib")
            or (isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "_MB_SCRIPT_DIR"
                        for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "_mb_outfile")]
    assert len(keep) == 3, "the deck no longer carries the one path anchor"
    prelude = ast.Module(body=keep, type_ignores=[])

    bundle = tmp_path / "bundle"
    attempt = bundle / "01_coarse" / "run-0"
    attempt.mkdir(parents=True)
    real = bundle / "w.py"
    real.write_text("# the deck\n")
    link = attempt / "w.py"
    link.symlink_to("../../w.py")

    ns = {"__file__": str(link)}
    exec(compile(prelude, str(link), "exec"), ns)
    out = ns["_mb_outfile"]("w.chk")
    assert out == str(attempt / "w.chk"), (
        f"emitted anchor left the attempt directory: {out}")

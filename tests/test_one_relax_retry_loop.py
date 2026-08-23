"""One retry loop, composed by both PySCF decks that relax a geometry.

**O4.** The optimization deck and the vibration deck both honour
``on_nonconvergence``, and its ``continue`` arm is a retry loop: a budget, a
`try`, a *"is this a convergence failure or a real error"* test, an
exhausted-raise, and a countdown in the warning.  Each deck spelled that loop
out in full until 2026-08-23, so a fix to either reached one deck of the two.

The rule this file pins is not *"these look alike"* — it is **a change to the
loop must reach both decks**, which is only true while there is one emitter.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import render_script, spec_for
from molbuilder.pyscf.relax_policy import emit_retry_loop
from molbuilder.script_emit import render_deck
from molbuilder.structure import Structure


def _water() -> Structure:
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.119],
                            [0.0, 0.757, -0.477],
                            [0.0, -0.757, -0.477]]))


def _cfg(**kw) -> PySCFConfig:
    return PySCFConfig(on_nonconvergence="continue",
                       geom_continue_retries=2, **kw)


def _optimization_deck() -> str:
    return render_script(_water(), _cfg(optimize=True))


def _vibration_deck() -> str:
    s, cfg = _water(), _cfg()
    return render_deck(spec_for(s, cfg, calculation="vibration"),
                       s, cfg, verbose=False)


def _loop(text: str):
    """The retry loop's lines, dedented and with the CALL removed.

    The call is each deck's own — one runs `_mb_run_optimization`, the other
    `_geom_opt` — so it is stripped before comparing.  What must match is
    everything around it.
    """
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "_budget = 1 +" in l)
    end = next(i for i, l in enumerate(lines[start:], start)
               if "left)\")" in l)
    body = lines[start:end + 1]
    pad = len(body[0]) - len(body[0].lstrip())
    out = [l[pad:] for l in body]
    # drop the call lines: everything between `try:` and `break`
    t = out.index("    try:")
    b = out.index("        break")
    out = out[:t + 1] + out[b:]
    # ...and normalise the TWO deliberate differences, both parameters of the
    # shared emitter rather than drift, each asserted on its own below:
    #
    #   * the message names WHAT failed ("relaxation " in the vibration deck,
    #     which also runs an SCF and a Hessian; nothing in the optimization
    #     deck, which has one phase);
    #   * the step-budget CONSTANT follows each deck's own naming convention
    #     for emitted constants -- the optimization deck is underscore-
    #     prefixed (12 of its 14), the vibration deck bare (61 of its 71).
    return [l.replace("WARN: relaxation did not", "WARN: did not")
             .replace("{_GEOM_MAX_STEPS}", "{STEPS}")
             .replace("{GEOM_MAX_STEPS}", "{STEPS}")
            for l in out]


def test_both_decks_emit_the_same_loop():
    """**The assertion that would have caught the duplication.**

    Strip each deck's own call and the two must be line-for-line identical --
    which is only achievable while one function emits both.
    """
    a, b = _loop(_optimization_deck()), _loop(_vibration_deck())
    assert a == b, (
        "the two decks' retry loops have drifted apart:\n"
        + "\n".join(f"  opt: {x!r}\n  vib: {y!r}"
                    for x, y in zip(a, b) if x != y))


def test_each_deck_names_what_failed_to_converge():
    """The one difference the loop is allowed to carry, asserted rather than
    normalised away: the vibration deck's message says *relaxation*, because
    that deck also runs an SCF and a Hessian and the reader needs to know
    which phase gave up.  The optimization deck has one phase and names none.
    """
    assert 'f"WARN: relaxation did not converge in "' in _vibration_deck()
    assert 'f"WARN: did not converge in "' in _optimization_deck()
    assert "relaxation did not converge" not in _optimization_deck()


def test_each_deck_names_the_step_budget_in_its_own_dialect():
    """The other allowed difference.  Each deck has a settled convention for
    the constants it emits -- the optimization deck underscore-prefixes them,
    the vibration deck does not -- so the loop must report the budget under
    the name that deck actually defines, or the message would raise
    `NameError` at the moment it tries to explain a retry."""
    assert "f\"{_GEOM_MAX_STEPS} steps; retrying \"" in _optimization_deck()
    assert "f\"{GEOM_MAX_STEPS} steps; retrying \"" in _vibration_deck()
    # ...and each name is one the deck really defines.
    assert "_GEOM_MAX_STEPS =" in _optimization_deck()
    assert "GEOM_MAX_STEPS =" in _vibration_deck()


@pytest.mark.parametrize("render", [_optimization_deck, _vibration_deck])
def test_the_deck_still_parses_as_python(render):
    """A deck is a runnable program, so it must at least parse.

    **This is weaker than it looks, and the docstring used to overclaim.**  It
    said a wrong indent is a syntax error; it is not.  Dropping the caller's
    indent dedents the loop out of `if not ALREADY_RELAXED:` and the file
    still parses cleanly -- verified by mutation, 2026-08-23.  What that
    mutation actually costs is in the test below.
    """
    ast.parse(render())


def test_the_relaxation_loop_stays_inside_the_already_relaxed_guard():
    """**The mutation `ast.parse` cannot see.**

    The vibration deck emits its retry loop inside
    ``if not ALREADY_RELAXED:``.  Lose the caller's indent and the loop
    dedents to module level -- still valid Python, and now it relaxes the
    geometry on EVERY run, including the one where the user stated the
    structure was already relaxed.  A silent wrong answer, not a crash, which
    is exactly the class this project refuses.

    So the guard is checked structurally: find the `if not ALREADY_RELAXED`
    node and assert the loop is in its body, not beside it.
    """
    tree = ast.parse(_vibration_deck())

    def _is_guard(node):
        return (isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == "ALREADY_RELAXED")

    guards = [n for n in ast.walk(tree) if _is_guard(n)]
    assert len(guards) == 1, (
        f"expected exactly one `if not ALREADY_RELAXED:` in the vibration "
        f"deck, found {len(guards)}; the premise of this test changed")

    def _has_retry_loop(body):
        return any(isinstance(n, ast.For)
                   and isinstance(n.target, ast.Name)
                   and n.target.id == "_attempt"
                   for n in ast.walk(ast.Module(body=body, type_ignores=[])))

    assert _has_retry_loop(guards[0].body), (
        "the retry loop is not inside `if not ALREADY_RELAXED:` -- a deck "
        "that dedented it would re-relax a geometry the user declared "
        "already relaxed, and still parse")
    # ...and nowhere else: one loop, inside the guard.
    top = [n for n in tree.body
           if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
           and n.target.id == "_attempt"]
    assert not top, "a retry loop escaped the guard to module level"


def test_a_real_error_is_not_swallowed_by_the_retry():
    """The loop retries a CONVERGENCE failure and re-raises anything else.
    Getting that test backwards would turn a crash into a silent retry
    loop, which is why it is asserted rather than read."""
    for text in (_optimization_deck(), _vibration_deck()):
        body = "\n".join(_loop(text))
        assert "if 'not converged' not in str(_e).lower():" in body
        assert re.search(r"raise\s+#\s*a genuinely different error", body)
        assert "if _attempt == _budget - 1:" in body


def test_the_budget_is_one_plus_the_retries():
    """`geom_continue_retries` is EXTRA attempts, not the total -- the deck's
    own help says the budget is `max_steps x (1 + retries)`."""
    assert emit_retry_loop(["x()"], retries=0, steps_var="S")[0] == \
        "_budget = 1 + 0"
    assert emit_retry_loop(["x()"], retries=3, steps_var="S")[0] == \
        "_budget = 1 + 3"


def test_the_caller_states_its_own_call_shape():
    """A multi-line call keeps its relative alignment, so a deck whose call
    wraps still emits valid Python."""
    got = emit_retry_loop(["mol = f(a,", "        b)"],
                          retries=1, steps_var="S", indent="    ")
    assert "            mol = f(a," in got
    assert "                    b)" in got

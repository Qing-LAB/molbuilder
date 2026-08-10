"""The Save pipeline's stage lookup, RUN rather than grepped.

``collectFdfParams`` in ``structure-optimization/viewer.js`` picks one row out
of the stages table so the selected stage's non-convergence policy can be
lifted to the top level -- that is how ``continue_retries`` reaches the
``.run.sh`` wrapper (`job-contracts.md § 6.2`: it is the one Resources field
that becomes no sbatch flag at all, so if it does not ride along here it is
simply lost).

**Why this file exists.** Decision 27 changed ``params.stage`` from a stage
NUMBER to the artifact TOKEN (``01_coarse``). The selection line kept doing
``(params.stage || 1) - 1`` -- a string minus one, which is ``NaN`` -- so the
lookup returned ``undefined``, ``selStage`` fell back to ``{}``, and every
staged save from the browser quietly dropped the retry policy. Nothing failed;
the wrapper was just generated without it.

A regex over the source would not have caught that: the line *looked* right and
still parses. So this extracts the shipped selection expression **out of
viewer.js itself** and runs it in Node against a stub ``params``. What is
asserted is the behaviour of the real source text, not a copy of it kept in
step by hand (`docs/execution/checkpointing.md` § 13.3 -- *run the thing and
look at what moved*).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VIEWER = (REPO / "molbuilder" / "web" / "static" / "structure-optimization"
          / "viewer.js")

#: The shipped expression, delimited by the two statements it lies between.
#: Anchored on code rather than comments so a reworded comment does not
#: silently empty the extraction.
_SELECT_RE = re.compile(
    r"(const _seq = .*?const selStage =.*?;)", re.S)

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is not installed")


def _select_stage(stage, stages):
    """Run viewer.js's own selection expression with these inputs."""
    src = VIEWER.read_text(encoding="utf-8")
    m = _SELECT_RE.search(src)
    assert m, ("could not find the stage-selection expression in viewer.js -- "
               "if it was renamed, update this test rather than deleting it")
    js = (f"const params = {json.dumps({'stage': stage, 'stages': stages})};\n"
          f"{m.group(1)}\n"
          "process.stdout.write(JSON.stringify(selStage));\n")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)


_LADDER = [{"name": "coarse", "on_nonconvergence": "continue",
            "continue_retries": 2},
           {"name": "medium", "on_nonconvergence": "continue",
            "continue_retries": 5},
           {"name": "tight", "on_nonconvergence": "halt"}]


def test_the_token_selects_its_own_stage_row():
    """``01_coarse`` must reach the coarse row.  It reached ``{}`` for every
    token from decision 27 until 2026-08-10, because the token was being used
    in arithmetic."""
    assert _select_stage("01_coarse", _LADDER)["name"] == "coarse"
    assert _select_stage("02_medium", _LADDER)["continue_retries"] == 5
    assert _select_stage("03_tight", _LADDER)["on_nonconvergence"] == "halt"


def test_the_ordinal_and_not_the_word_is_what_selects():
    """A stage's name is user-editable (`engines/stages.md` R5 makes it the
    stage's identity, not a fixed word), so a renamed row must still be found
    by the token that names its files."""
    renamed = [dict(_LADDER[0], name="rough"), _LADDER[1], _LADDER[2]]
    assert _select_stage("01_rough", renamed)["name"] == "rough"


def test_no_stage_selected_yields_an_empty_row_rather_than_a_wrong_one():
    """`custom` gives ``stage: null``.  The lift must then not happen at all --
    quietly borrowing stage 1's retry policy would attach a number the user
    never chose."""
    assert _select_stage(None, _LADDER) == {}


def test_the_token_is_never_used_in_arithmetic():
    """The bug in one line: ``(params.stage || 1) - 1``.

    The tests above run the *current* expression, so they prove it behaves --
    but they find it by shape, and a future rewrite that reintroduced
    arithmetic could move the anchor with it. This asks the question that does
    not depend on shape: nothing in this function may do maths on the token.
    It parses, it reads like an index, and it is ``NaN`` for every real token.
    """
    src = VIEWER.read_text(encoding="utf-8")
    body = re.search(r"function collectFdfParams\(\).*?\n    \}", src, re.S)
    assert body, "collectFdfParams not found in viewer.js"
    # Comments are stripped first: the code below explains the old bug by
    # quoting it, and a scanner that reads prose finds offences in the very
    # note warning against them.
    code = re.sub(r"//[^\n]*", "", body.group(0))
    # `\b` matters: `params.stage` is a prefix of `params.stages`, and the
    # correct indexing expression subtracts 1 from an ordinal off that list.
    offenders = re.findall(r"params\.stage\b[^;\n]*?[-+*/]\s*\d", code)
    assert not offenders, (
        f"arithmetic on the stage TOKEN: {offenders}. Since decision 27 "
        f"`params.stage` is `01_coarse`, not a number; parse the ordinal off "
        f"it (project-layout.md § 4.2) instead of subtracting from it.")


def test_a_token_past_the_end_does_not_wrap_or_throw():
    """A ladder can be shorter than the token says (rows disabled and dropped
    before collection).  Missing is ``{}``, never the last row."""
    assert _select_stage("09_extra", _LADDER) == {}

"""`results.md` § 4's single entry point: a write to the tab's state goes
through `transition()`.

**`test_applyNewData_routes_writes_through_transition`** reads the function's
body for a family of forbidden writes (`state.mtime =`, `state.data =`, …); it
stays until a test that runs `applyNewData` with a stale path replaces it
(plan, TST).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_STATIC = (Path(__file__).resolve().parent.parent
           / "molbuilder" / "web" / "static")
_LIB = _STATIC / "lib"


# --------------------------------------------------------------------- #
#  Trajectory: bucketed state shape                                     #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def core_body():
    return (_LIB / "trajectory" / "core.js").read_text()


def _unused_braced(src: str, open_idx: int) -> str:
    """The object literal that starts at ``src[open_idx] == '{'``, brace
    to brace -- so a pin over a literal covers the whole literal however
    long it grows, rather than a fixed character count that a new field
    can push a bucket out of."""
    assert src[open_idx] == "{", "not an opening brace"
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx: i + 1]
    raise AssertionError("unbalanced braces from the state literal")


class TestSettlePostLoad:
    """The follow rule (`results.md` § 4.1) lives in ``_settlePostLoad()``.
    Called from both loadByPath and pollOnce; reads the run's state the
    server sent with the file, decides which transition to invoke."""


    def test_applyNewData_routes_writes_through_transition(self, core_body):
        """applyNewData's two write blocks (noNewContent + full
        rebuild) MUST both go through transition('APPLY').  Direct
        ``state.mtime = ...`` / ``state.data = ...`` writes are
        forbidden -- the alias bridge would route them to fileState
        but bypass the single-entry-point contract."""
        m = re.search(
            r"function\s+applyNewData\s*\(\s*r\s*\)\s*\{(.+?)\n\s{4}\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "applyNewData function not found"
        body = m.group(1)
        # No direct legacy-alias writes of fileState fields.
        forbidden = [
            r"\bstate\.mtime\s*=",
            r"\bstate\.data\s*=",
            r"\bstate\.format\s*=",
            r"\bstate\.label\s*=",
            r"\bstate\.path\s*=",
        ]
        for pat in forbidden:
            assert not re.search(pat, body), (
                f"applyNewData still has a direct fileState write "
                f"matching ``{pat}``.  PR 2.3 routed these through "
                f"transition('APPLY', ...); the regression brings "
                f"back the contract § 2 'sole-writer' violation.")
        # AT LEAST two transition('APPLY', ...) call sites in the
        # function body (one per write block).
        apply_calls = re.findall(
            r"transition\s*\(\s*[\"']APPLY[\"']", body
        )
        assert len(apply_calls) >= 2, (
            f"applyNewData has {len(apply_calls)} transition('APPLY') "
            f"call(s).  Expected >= 2 (noNewContent path + full-"
            f"rebuild path).  Either a write block was removed or a "
            f"direct write was reintroduced.")

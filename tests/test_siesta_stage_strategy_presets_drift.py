"""Drift gate: JS ``STAGE_STRATEGY_PRESETS`` in form-schema.js must
match the Python preset dicts in both engine modules (#542 / C1.6).

Three sources carry the same data:
  * ``molbuilder/web/static/lib/form-schema.js``
      const STAGE_STRATEGY_PRESETS = { ... }
    (the JS dropdown that the user picks from in the form's
    stage-table widget)
  * ``molbuilder/config/siesta.py``
      SIESTA_STAGE_STRATEGY_PRESETS = { ... }
    (read by ``siesta/stages.py::default_siesta_stages``)
  * ``molbuilder/config/pyscf.py``
      STAGE_STRATEGY_PRESETS = { ... }
    (the PySCF-side ``apply_stage_strategy`` driver)

If they drift, the user picks "Publishable" in the form and a
DIFFERENT enable mask gets applied than the one the CLI's
``--stage-strategy publishable`` would have produced.  The bug
class is silent: the form renders fine; the wrapper renders fine;
the user just gets the wrong number of stages.

The drift gate parses the JS file via regex and asserts the parsed
``enables`` arrays equal both Python dicts value-for-value.  Update
all three sources in the same commit when adding / renaming a preset.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_JS_PATH = _REPO_ROOT / "molbuilder" / "web" / "static" / "lib" / "form-schema.js"


# --------------------------------------------------------------------- #
#  JS parser                                                             #
# --------------------------------------------------------------------- #


_PRESETS_BLOCK_RE = re.compile(
    r"const\s+STAGE_STRATEGY_PRESETS\s*=\s*\{(?P<body>.*?)\n\s*\};",
    re.DOTALL,
)
# Per-preset entry: ``"name": { label: "...", enables: [true, false, true] }``
# Matches whitespace and trailing-comma variants.  Names are quoted strings
# with ASCII letters, digits, dashes, underscores.
_ENTRY_RE = re.compile(
    r'"(?P<name>[A-Za-z0-9_\-]+)"\s*:\s*\{'
    r'[^{}]*?'                              # any non-brace fields before enables
    r'enables\s*:\s*\[(?P<enables>[^\]]+)\]'  # enables: [true, false, ...]
    r'[^{}]*?'                              # any trailing fields
    r'\}',
    re.DOTALL,
)


def _parse_js_presets(js_path: Path = None) -> Dict[str, Tuple[bool, ...]]:
    """Parse ``STAGE_STRATEGY_PRESETS`` from form-schema.js into a
    ``{name: (bool, bool, ...)}`` dict.

    Raises ``AssertionError`` with a precise message when the regex
    can't locate the block or any single entry -- so a future JS
    refactor that re-shapes the constant (e.g. moves it to a separate
    module) surfaces as a clear test failure rather than a silent
    drift.
    """
    src = (js_path or _JS_PATH).read_text()
    block_m = _PRESETS_BLOCK_RE.search(src)
    assert block_m, (
        f"failed to locate ``const STAGE_STRATEGY_PRESETS = {{ ... }};`` "
        f"block in {_JS_PATH}.  If the constant was renamed or moved, "
        f"update _PRESETS_BLOCK_RE in this test."
    )
    body = block_m.group("body")
    out: Dict[str, Tuple[bool, ...]] = {}
    for m in _ENTRY_RE.finditer(body):
        name = m.group("name")
        raw_enables = m.group("enables")
        # ``[true, false, true]`` -> [True, False, True]
        bools = []
        for tok in raw_enables.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok == "true":
                bools.append(True)
            elif tok == "false":
                bools.append(False)
            else:
                raise AssertionError(
                    f"unexpected token {tok!r} in JS enables array for "
                    f"preset {name!r}: expected only ``true`` or "
                    f"``false``"
                )
        out[name] = tuple(bools)
    assert out, (
        "_ENTRY_RE matched zero presets inside the STAGE_STRATEGY_"
        "PRESETS block -- either the block is empty or the regex no "
        "longer matches the entry shape.  Inspect form-schema.js and "
        "update _ENTRY_RE."
    )
    return out


# --------------------------------------------------------------------- #
#  Drift assertions                                                     #
# --------------------------------------------------------------------- #


def test_js_parser_finds_three_presets():
    """Sanity check: the JS file currently has exactly three named
    presets (publishable / loose-only / vib-quality).  If a future
    commit adds a fourth, this test fires and the operator can
    decide whether the new name should be added to both Python
    sources too."""
    js_presets = _parse_js_presets()
    assert set(js_presets.keys()) == {"publishable", "loose-only",
                                       "vib-quality"}, js_presets


def test_js_presets_match_siesta_presets_value_for_value():
    """The SIESTA-side ``SIESTA_STAGE_STRATEGY_PRESETS`` must agree
    with the JS-side ``STAGE_STRATEGY_PRESETS`` value-for-value."""
    from molbuilder.config.siesta import SIESTA_STAGE_STRATEGY_PRESETS
    js_presets = _parse_js_presets()
    assert set(js_presets.keys()) == set(SIESTA_STAGE_STRATEGY_PRESETS), (
        f"preset name set drifted between JS + SIESTA:\n"
        f"  js:     {sorted(js_presets)}\n"
        f"  siesta: {sorted(SIESTA_STAGE_STRATEGY_PRESETS)}"
    )
    for name in js_presets:
        js_val = js_presets[name]
        py_val = SIESTA_STAGE_STRATEGY_PRESETS[name]
        assert js_val == py_val, (
            f"preset {name!r}: enable pattern drifted between JS + "
            f"SIESTA.\n  js:     {js_val}\n  siesta: {py_val}\n"
            f"Update molbuilder/config/siesta.py "
            f"SIESTA_STAGE_STRATEGY_PRESETS and "
            f"molbuilder/web/static/lib/form-schema.js "
            f"STAGE_STRATEGY_PRESETS in the same commit."
        )


def test_js_presets_match_pyscf_presets_value_for_value():
    """The PySCF-side ``STAGE_STRATEGY_PRESETS`` must agree with the
    JS-side ``STAGE_STRATEGY_PRESETS`` value-for-value -- the form's
    preset dropdown surfaces the SAME options regardless of which
    engine's stage-table is being edited."""
    from molbuilder.config.pyscf import STAGE_STRATEGY_PRESETS as PYSCF
    js_presets = _parse_js_presets()
    assert set(js_presets.keys()) == set(PYSCF), (
        f"preset name set drifted between JS + PySCF:\n"
        f"  js:    {sorted(js_presets)}\n"
        f"  pyscf: {sorted(PYSCF)}"
    )
    for name in js_presets:
        js_val = js_presets[name]
        py_val = PYSCF[name]
        assert js_val == py_val, (
            f"preset {name!r}: enable pattern drifted between JS + "
            f"PySCF.\n  js:    {js_val}\n  pyscf: {py_val}\n"
            f"Update molbuilder/config/pyscf.py "
            f"STAGE_STRATEGY_PRESETS and molbuilder/web/static/lib/"
            f"form-schema.js STAGE_STRATEGY_PRESETS in the same commit."
        )


def test_siesta_and_pyscf_presets_match_value_for_value():
    """Belt-and-braces: SIESTA + PySCF must also agree with each
    other.  The JS↔Python checks above catch most cases, but if
    BOTH Python sources drift in the same direction together
    (without updating the JS), the JS gate above can still fail
    while leaving the two Python sources internally consistent.
    This check catches the inverse: Python sources mismatching
    each other regardless of what JS says."""
    from molbuilder.config.siesta import SIESTA_STAGE_STRATEGY_PRESETS
    from molbuilder.config.pyscf import STAGE_STRATEGY_PRESETS as PYSCF
    assert set(SIESTA_STAGE_STRATEGY_PRESETS) == set(PYSCF), (
        f"preset name set drifted between SIESTA + PySCF:\n"
        f"  siesta: {sorted(SIESTA_STAGE_STRATEGY_PRESETS)}\n"
        f"  pyscf:  {sorted(PYSCF)}"
    )
    for name in SIESTA_STAGE_STRATEGY_PRESETS:
        s_val = SIESTA_STAGE_STRATEGY_PRESETS[name]
        p_val = PYSCF[name]
        assert s_val == p_val, (
            f"preset {name!r}: enable pattern drifted between "
            f"SIESTA + PySCF.\n  siesta: {s_val}\n  pyscf:  {p_val}"
        )


# --------------------------------------------------------------------- #
#  Self-tests for the JS parser                                          #
# --------------------------------------------------------------------- #


def test_js_parser_rejects_unknown_boolean_token(tmp_path):
    """If a future maintainer writes ``enables: [1, 0, 0]`` instead of
    ``[true, false, false]``, the test must fail loud rather than
    silently coerce -- the JS engine reads it as truthy 1/0, but
    Python's ``true``/``false`` parser would interpret it
    differently (or not at all)."""
    fake_js = tmp_path / "form-schema.js"
    fake_js.write_text(
        'const STAGE_STRATEGY_PRESETS = {\n'
        '    "broken": {\n'
        '        label: "Broken",\n'
        '        enables: [1, 0, 0],\n'
        '    },\n'
        '};\n'
    )
    with pytest.raises(AssertionError, match="unexpected token"):
        _parse_js_presets(js_path=fake_js)


def test_js_parser_fails_loud_when_block_missing(tmp_path):
    """If a future refactor removes / renames the const, the test
    must fail with a clear message naming the regex to update."""
    fake_js = tmp_path / "form-schema.js"
    fake_js.write_text("// no presets here\n")
    with pytest.raises(AssertionError, match="STAGE_STRATEGY_PRESETS"):
        _parse_js_presets(js_path=fake_js)

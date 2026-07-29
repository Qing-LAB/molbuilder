"""Tests for molbuilder.validation.metadata.

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




# --------------------------------------------------------------------- #
#  Generic config-metadata pass                                         #
#                                                                        #
#  These tests pin the metadata-reading contract using a synthetic     #
#  dataclass so the test stays focused on the reading code itself.     #
#  Production-config coverage lives further down.                      #
# --------------------------------------------------------------------- #


def test_metadata_range_check_flags_out_of_bounds(water_struct):
    """When a config field has metadata={'range': (lo, hi)}, an
    out-of-range value produces a 'warn' Issue with the field name."""
    from dataclasses import dataclass, field

    @dataclass
    class _ToyConfig:
        cutoff: float = field(default=10.0,
                              metadata={"range": (50.0, 600.0),
                                        "label": "Cutoff", "unit": "Ry"})

    issues = validate(water_struct, _ToyConfig())
    msgs = [i for i in issues if i.where == "config.cutoff"]
    assert len(msgs) == 1
    assert "Cutoff" in msgs[0].message
    assert "Ry" in msgs[0].message
    assert msgs[0].severity == "warn"



def test_metadata_validate_callable_emits_issue(water_struct):
    """A custom validate=callable can return an Issue directly."""
    from dataclasses import dataclass, field

    def _check_pow2(v, _cfg):
        if v & (v - 1):
            return Issue("warn", f"value {v} is not a power of 2",
                         "config.block_size")
        return None

    @dataclass
    class _ToyConfig:
        block_size: int = field(default=7,
                                metadata={"validate": _check_pow2})

    issues = validate(water_struct, _ToyConfig())
    assert any(i.where == "config.block_size" for i in issues)

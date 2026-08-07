"""Versioned-document helpers — the shared ``molbuilder/<name>@<major>``
schema convention + JSON IO (data-vocabulary.md § 1).

The major-version check was hand-rolled identically in three persisted
artifacts (``bench/environment.py``, ``bench/result.py``, ``jobset/model.py``),
with a subtle inconsistency in how a missing ``@`` was handled.  This is the
one place that logic lives now.

L1: pure stdlib, no molbuilder deps -- any layer may use it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def schema_major(schema: str) -> str:
    """The major token of a ``molbuilder/<name>@<major>`` string, or ``""``
    when there is no ``@`` (so a bare/garbage value never matches a real
    major).

    A dotted version keeps only what precedes the first dot, so ``@1.4``
    has major ``1``.  ``job-contracts.md`` § 6.1 states the rule the whole
    convention rests on -- *"checked major-only, tolerating same-major minor
    bumps, rejecting a different major"* -- and until 2026-08-07 this
    function did not implement its first half: it compared the entire token,
    so ``@1.4`` was rejected against ``@1`` with a message saying the major
    differed when it did not.  Nothing shipped had a minor, so nothing had
    exercised it; ``task@1`` is the first artifact whose reader tests the
    claim.  The change only ever widens acceptance -- no string that passed
    before can fail now.
    """
    s = str(schema or "")
    if "@" not in s:
        return ""
    return s.rsplit("@", 1)[-1].split(".", 1)[0]


def check_schema_major(got: str, want: str, *, label: str = "") -> None:
    """Raise :class:`ValueError` unless ``got`` and ``want`` share the same
    major version (tolerate same-major minor bumps, reject a different
    major).  ``label`` prefixes the message so the artifact is obvious.

    The single enforcement point for the schema convention -- adopted by
    ``environment@1`` / ``bench-result@1`` / ``job-set@1`` (was duplicated)."""
    if schema_major(got) != schema_major(want):
        prefix = f"{label} " if label else ""
        raise ValueError(
            f"{prefix}schema mismatch: got {got!r}, need major "
            f"{schema_major(want)} ({want}).")


def read_json(path) -> Any:
    """Read + parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj: Any) -> Path:
    """Write ``obj`` as pretty JSON (2-space indent, trailing newline).
    ATOMIC: writes a sibling ``.tmp`` then ``os.replace`` into place, so a
    crash mid-write never leaves a truncated/corrupt config (which readers
    would then fail to parse).  Returns the path."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)                 # atomic on the same filesystem
    return p


__all__ = ["schema_major", "check_schema_major", "read_json", "write_json"]

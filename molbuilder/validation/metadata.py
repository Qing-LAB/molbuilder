"""Generic dataclass-field-metadata-driven validation pass.

Reads ``range`` and ``validate`` off dataclass field metadata and
produces Issues.  This is what makes design.md Principle #1 load-
bearing: field metadata IS the source of truth; CLI / web /
validators all read from the same place.

Split from the pre-2026-06-13 flat ``molbuilder/validation.py`` per
docs/protocols/scientific-validation.md § 10.  Function body +
public signature are identical to the pre-split version.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import List

from ..issues import Issue


def _validate_config_metadata(cfg) -> List[Issue]:
    issues: List[Issue] = []
    if not is_dataclass(cfg):
        return issues
    for f in fields(cfg):
        meta = f.metadata or {}
        value = getattr(cfg, f.name)
        # range = (lo, hi) inclusive
        rng = meta.get("range")
        if rng is not None and value is not None:
            lo, hi = rng
            try:
                if value < lo or value > hi:
                    label = meta.get("label", f.name)
                    unit = f" {meta['unit']}" if meta.get("unit") else ""
                    issues.append(Issue(
                        "warn",
                        f"{label} = {value}{unit} is outside the "
                        f"recommended range [{lo}, {hi}]{unit}",
                        f"config.{f.name}",
                    ))
            except TypeError:
                # 2026-06-14 G5: pre-fix this branch silently
                # swallowed every TypeError, which hid a real bug
                # class -- a Tuple-typed field (e.g. SIESTA's
                # ``kgrid: Tuple[int,int,int]``) with a scalar
                # ``range = (lo, hi)`` metadata is uncomparable and
                # the field's range goes unenforced.  Surfacing
                # this as an error-severity Issue makes the metadata
                # bug visible at preflight time instead of leaving
                # the user with a silently-permissive validator.
                # Use a ``validate`` callable on the field metadata
                # to range-check each component (see the kgrid
                # field's ``validate`` callable in config/siesta.py
                # for the pattern).
                issues.append(Issue(
                    "error",
                    (f"Field metadata for ``{f.name}`` declares "
                     f"``range = {rng}`` but the value "
                     f"{value!r} is not scalar-comparable.  "
                     f"This is a programmer bug: drop the scalar "
                     f"``range`` and add a per-component "
                     f"``validate`` callable instead."),
                    f"config.{f.name}",
                ))
        # Optional callable: meta["validate"] -> Issue or None
        validator = meta.get("validate")
        if validator is not None:
            try:
                result = validator(value, cfg)
            except Exception:
                continue
            if isinstance(result, Issue):
                issues.append(result)
            elif isinstance(result, list):
                issues.extend(i for i in result if isinstance(i, Issue))
    return issues

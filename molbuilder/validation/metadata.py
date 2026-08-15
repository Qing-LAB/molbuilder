"""Generic dataclass-field-metadata-driven validation pass.

Reads ``range`` and ``validate`` off dataclass field metadata and
produces Issues.  This is what makes design.md Principle #1 load-
bearing: field metadata IS the source of truth; CLI / web /
validators all read from the same place.

Split from the pre-2026-06-13 flat ``molbuilder/validation.py`` per
docs/science/validation.md  Function body +
public signature are identical to the pre-split version.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import List

from ..issues import Issue


def _keyword_suffix(meta) -> str:
    """`` (MeshCutoff)`` — the engine's own spelling, beside the human name.

    **Both, always** (user, 2026-08-15: *"always include the keyword relevant
    to it in addition to the meaning of that, such that it is easy to detect,
    and meaning is still clear"*).  A warning has two jobs and one word cannot
    do them: *"Real-space grid cutoff"* says what is wrong and cannot be found
    in the input file; *"MeshCutoff"* can be searched for and says nothing.

    This is a REGRESSION FIX, not a new idea.  The labels used to BE the
    keywords — ``MeshCutoff``, ``DM.Tolerance``, ``MD.MaxForceTol`` — and were
    replaced with prose on 2026-08-14 when the catalogue became the master.
    That was right for a form control and silently wrong here: seventeen
    warnings lost the only word in them a person could act on.

    The BARE keyword, not the full ``engine_key``: a warning is scanned, and
    ``MD.MaxCGDispl (universal for CG / Broyden / FIRE)`` is a sentence.  The
    full spelling belongs on the form's badge, where there is room for it.

    **Nothing is added for a setting that is not an engine keyword** — the
    molbuilder-side ones (``psml_lib``, ``copy_psml``, ``mpi_np``) whose
    ``engine_key`` is a parenthesised note.  There is no word to offer, and
    inventing one would make a search fail rather than merely not help.
    """
    from ..template import _bare_anchor
    kw = _bare_anchor(str(meta.get("engine_key", "") or ""))
    return f" ({kw})" if kw else ""


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
                        f"{label}{_keyword_suffix(meta)} = {value}{unit} is "
                        f"outside the recommended range [{lo}, {hi}]{unit}",
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
            except Exception as exc:
                # 2026-06-14 I3 round-3: pre-fix this branch
                # silently swallowed ANY exception from a
                # validator-callable.  Same anti-pattern G5
                # retired for the comparable-range path; this is
                # the other door it could leak through.  A future
                # validator that raises (regex .match() on a non-
                # string, attribute-access on None, etc.) used to
                # silently disappear from preflight.  Surfacing as
                # error-Issue makes the metadata bug visible at
                # preflight time so a contributor fixes the
                # callable instead of leaving the validator dead.
                issues.append(Issue(
                    "error",
                    (f"Field metadata for ``{f.name}`` has a "
                     f"``validate`` callable that raised "
                     f"{type(exc).__name__}: {exc}.  This is a "
                     f"programmer bug: the callable should "
                     f"return Issue / list[Issue] / None instead "
                     f"of raising."),
                    f"config.{f.name}",
                ))
                continue
            if isinstance(result, Issue):
                issues.append(result)
            elif isinstance(result, list):
                issues.extend(i for i in result if isinstance(i, Issue))
    return issues

"""A name that ends in a newline is not that name.

`re` gives `$` a carve-out: without ``re.MULTILINE`` it matches at the end of
the string **or just before a final newline**.  So ``^[A-Za-z0-9_-]+$`` with
``.match()`` accepts ``"proj\\n"`` — and every validator in this repo that
spelled it that way was letting a trailing newline through.  ``.fullmatch()``
has no such carve-out.

**Why that matters here rather than in the abstract.**  These are not opaque
tokens: a project name becomes a DIRECTORY, an upload filename becomes a FILE,
a workspace id becomes a STORAGE PATH, a stage name becomes part of every
filename a run writes (`job-contracts.md` § 2.2a).  A value that passes
validation and then lands on a filesystem with an embedded newline is a value
the rest of the system cannot round-trip: the readers that take it apart again
are anchored patterns that do NOT have the carve-out, so what was written can
no longer be recognised.  Measured 2026-09-08 across 32 call sites.

**This file is one place on purpose.**  The rule is one rule, so it is checked
once over the doors it governs rather than beside each of them — and it asks
the DOORS, never the private patterns behind them, so it keeps holding if a
validator is reimplemented without a regex at all.

The counterpart rule, for the record: the ~23 `.match()` calls left in the
parsers and `script_emit` are LINE scanners, and there `$`'s carve-out is
exactly right — a line pulled from a file still carries its ``\\n``.  Those must
not be "fixed"; converting them would break every one.
"""
from __future__ import annotations

import pytest


def _doors():
    """(name, callable that ACCEPTS a good value and REFUSES a bad one).

    Each returns True/None for good input and raises or returns False for bad.
    Normalised to a predicate so one table drives them all.
    """
    from molbuilder import projects, checkpoint, monitor, selection
    from molbuilder.web.blueprints import workspace_storage

    def _raises(fn):
        def ok(v):
            try:
                fn(v)
                return True
            except Exception:
                return False
        return ok

    from molbuilder.web.blueprints.files import _validate_upload_filename

    # Three refusal shapes in one table, because the doors genuinely differ:
    # raise, return False, or return an error STRING (None when fine).  The
    # adapter is what varies; the rule asked of each is identical.
    def _msg(fn):
        return lambda v: fn(v) is None

    return [
        ("project name",     _raises(projects.validate_name),            "proj"),
        ("calculation name", _raises(checkpoint.check_calculation_name), "calc"),
        ("channel name",     monitor.is_channel_name,                    "chan"),
        ("workspace id",     workspace_storage._valid_ws_id,             "ws1"),
        ("upload filename",  _msg(_validate_upload_filename),            "a.txt"),
    ]


@pytest.mark.parametrize("what,accepts,good", _doors(),
                         ids=[d[0] for d in _doors()])
def test_a_trailing_newline_is_not_part_of_a_name(what, accepts, good):
    assert accepts(good) is True, f"{what}: rejected a legitimate value"
    assert accepts(good + "\n") is not True, (
        f"{what}: accepted {good + chr(10)!r} -- `$` matches before a final "
        f"newline, so the value that passes is not the value that was checked")


def test_a_stage_name_with_a_newline_cannot_reach_a_filename():
    """The stage name is the one that travels furthest: it becomes the token in
    every per-rung filename, so a newline in it would be written to disk and
    then be unreadable by the anchored patterns that take names apart."""
    from molbuilder.task import Stage
    Stage(name="coarse")                      # the ordinary case still works
    with pytest.raises(ValueError):
        Stage(name="coarse\n")


def test_the_line_scanners_are_deliberately_left_alone():
    """The other half of the rule, stated so nobody 'finishes the job'.

    A parser matching a line from a file relies on `$`'s carve-out, because
    iterating a file hands back lines that still end in `\\n`.
    """
    from molbuilder.parse.engines import molwatch
    assert molwatch._ERROR_RE.match("# error: boom\n") is not None
    assert molwatch._ERROR_RE.fullmatch("# error: boom\n") is None

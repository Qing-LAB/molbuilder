"""Exceptions raised by the parse module.

Per ``docs/model/parse.md`` § 3.

TRAVELS in ``mb_vibration.pyz`` beside each call of a SIESTA vibration
(`runwrap.members`, `engines/vibration.md` § 5.11 § 3), so it imports of
molbuilder only the mark every reader's refusal carries, `persist`'s, which
travels with it (:data:`TRAVELS_WITH`).
"""

from __future__ import annotations

try:                                        # inside molbuilder
    from ..persist import FileRefused
except ImportError:                         # beside a job, in a bundle
    from persist import FileRefused  # type: ignore[no-redef]


class ParseError(Exception):
    """Base class for parse-module exceptions: a file parser's refusal of
    the file it was handed (:class:`FileUnreadable`), and the registry's
    answers that no parser, or two, claim a path."""


class FileUnreadable(ParseError, FileRefused):
    """A FILE PARSER'S REFUSAL OF THE FILE IT WAS HANDED -- a version it does
    not read, or a content -- naming the file: it carries the mark of a file
    its reader cannot read (`persist.FileRefused`, `execution/job-contracts.md`
    § 6.1), which the command line and the web server say at their edge.
    The registry's two answers below are no file's refusal and carry none."""


class UnknownFormatError(ParseError):
    """No registered parser claims to understand the given path.

    The error message lists every registered parser with its hint
    so the user can pick the right one OR install the missing
    plugin.
    """


class AmbiguousFormatError(ParseError):
    """More than one registered parser claims to understand the
    given path.  Should be rare — typically signals a registry
    misconfiguration (two parsers with overly-permissive
    ``can_parse``)."""


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.persist",)

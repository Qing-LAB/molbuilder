"""Exceptions raised by the parse module.

Per ``docs/model/parse.md`` § 3.

TRAVELS in ``mb_vibration.pyz`` beside a SIESTA force-constant job
(`runwrap.VIBRATION_COMPANIONS`, `engines/vibration.md` § 5.5), so it
imports nothing of molbuilder at module level.
"""

from __future__ import annotations


class ParseError(Exception):
    """Base class for parse-module exceptions."""


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

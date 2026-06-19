"""Exceptions raised by the parse module.

Per ``docs/protocols/parse-module.md`` § 4.
"""

from __future__ import annotations


class ParseError(Exception):
    """Base class for parse-module exceptions."""


class UnknownFormatError(ParseError):
    """No registered parser claims to understand the given path.

    The error message lists every registered parser with its hint
    so the user can pick the right one OR install the missing
    plugin.  Mirrors the legacy ``parsers.UnknownFormatError``
    so existing handlers keep working during migration.
    """


class AmbiguousFormatError(ParseError):
    """More than one registered parser claims to understand the
    given path.  Should be rare — typically signals a registry
    misconfiguration (two parsers with overly-permissive
    ``can_parse``)."""

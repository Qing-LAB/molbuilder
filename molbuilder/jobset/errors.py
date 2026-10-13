"""The refusals every floor of the job system raises -- floor 1
(`execution/architecture.md` § 2.1).

A plain type, so a lower floor can refuse in the person's own words without
importing the conductor that collects the refusal.
"""
from __future__ import annotations

from contextlib import contextmanager


class PrepError(Exception):
    """A prep refused -- in the reader's own words, which each surface shows
    as they are.

    What the one prep entry had already found when it refused rides with it
    (`prep.prep_task`): ``findings`` (the description's preflight notes) and
    ``notes`` (what its inputs said -- a bench's grid with every crossed-out
    cell, a run's sizing).  A refusal that says *see the crossed-out list
    above* is only honest if the list is shown with it; the command line
    prints these before the error, the Task setup route returns them beside
    it.  Empty on a refusal raised anywhere else.
    """
    findings: tuple = ()
    notes: tuple = ()


def user_errors() -> tuple:
    """The USER-FIXABLE refusal classes the steps raise, which every prep
    door says as a :class:`PrepError` (:func:`user_error_as_prep`)."""
    from ..issues import ValidationError
    from ..runtime_config import RuntimeConfigError
    from ..runwrap import WrapperError
    return (ValidationError, RuntimeConfigError, WrapperError)

def step_errors() -> tuple:
    """What a prep step raises for the person to fix, said as a refusal by
    each door that plans a stage (prep's two, launch's check that a stage is
    still as described): the machine named nowhere or twice
    (``UnknownTarget``, ``AmbiguousTarget``), the plain ``ValueError`` /
    ``KeyError`` the steps raise for a template naming an item its schema
    does not declare or a bundle written before a rename, and
    :func:`user_errors`.  A ``TypeError`` is not one: it is a bug, and
    should look like one."""
    from ..scheduler import AmbiguousTarget, UnknownTarget
    return (UnknownTarget, AmbiguousTarget, ValueError, KeyError,
            *user_errors())


def as_prep_error(exc: BaseException) -> PrepError:
    """``exc`` said as a refusal: its own words, with the hook boundary's
    attribution carried across (`issues.calling`)."""
    from ..issues import notes_of
    note = notes_of(exc)
    return PrepError(str(exc) + (f"\n  ({note})" if note else ""))

@contextmanager
def user_error_as_prep():
    """Translate the USER-FIXABLE refusal classes the steps below raise into
    :class:`PrepError`, so every caller of the two prep entries has ONE class
    to catch.  Without this the described route's two most likely
    first-contact failures -- missing pseudos (``ValidationError`` out of the
    deck render) and an unset activation
    (``RuntimeConfigError`` out of the wrapper render) -- escaped the CLI as
    raw tracebacks (2026-08-12 plan A8).  Only the NAMED classes translate: a
    ``TypeError`` here is a bug and should look like one.

    **The hook boundary's attribution is carried across**
    (`issues.calling`).  ``str(exc)`` does not include an exception's notes,
    so a refusal that came out of a named engine hook would arrive at the CLI
    saying what went wrong and not whose it was -- and the note would reach a
    traceback nobody sees."""
    try:
        yield
    except user_errors() as exc:
        raise as_prep_error(exc) from exc


__all__ = ["PrepError", "user_errors", "step_errors", "as_prep_error",
           "user_error_as_prep"]

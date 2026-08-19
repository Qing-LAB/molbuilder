"""Issue dataclass + ValidationError -- validator output type.

A validation pass produces a list of Issues.  Each Issue describes a
single problem with a structure or configuration; the consumer
decides what to do with the list (errors usually raise, warnings
usually print to stderr).

Spec: docs/design.md § "Pre-emission geometry validation" + §
"Validation pass (pre-emission)".
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

# The closed set of severities a validator may emit.  Pinned in one
# place so the runtime check + the static type stay in lock-step
# (previously the docstring promised a Literal but the annotation
# was a bare str -- static analysers couldn't catch typos).
Severity = Literal["error", "warn", "info"]
_SEVERITIES: tuple = ("error", "warn", "info")


@dataclass(frozen=True)
class Issue:
    """A single validation finding.

    Fields
    ------
    severity : "error" | "warn" | "info"
        Errors block emission (render_fdf / render_script raise);
        warnings print to stderr but the run proceeds; info entries
        are advisory (e.g. "Fe + spin=4 -> high-spin Fe(II)") and
        don't add to the warn count.
    message : str
        Human-readable, single-line, no trailing punctuation.  Should
        include any actionable advice in-line (e.g. "increase the
        structure's vacuum above 25 Å" rather than "cell too tight").
    where : str
        Locus of the problem, dotted-namespace style:
          "geometry.min_distance"   -- structural finding
          "cell.no_volume"          -- about the cell
          "config.mesh_cutoff"      -- about a config field
          "stages.loosens.mesh_cutoff" -- about the SEQUENCE of stages
        Used by the CLI / web UI to highlight the offending field.
    stage : Optional[str]
        Which stage of a ladder this finding is about, or ``None``.

        **Beside ``where``, never inside it** (``engines/stages.md`` § 4 R2).
        A stage is validated as a RESOLVED WHOLE -- the validator is handed a
        whole config plus the stage's name as a label -- so the same check
        that fires for a single run fires for a stage, and produces the same
        ``where``.  Folding the stage into the id instead would give one
        check as many ids as a ladder has stages, and the UI binds behaviour
        to the id.

        ``None`` means the finding is not about a member of the ladder:
        either an ordinary single-run finding, or a fact about the
        SEQUENCE (§ 4 R3) -- a ladder that loosens is a property of the
        description, not of any one stage in it.
    workflow_group : Optional[str]
        Optional workflow-card binding -- one of "profile", "stage",
        or "budget" -- so the web UI can attach the finding to the
        card whose fields it concerns rather than dumping every
        issue into one panel below the Generate button (per
        docs/web/ui-contract.md Rule 2).  Defaults to
        None; usually set by ``_shared.resolve_workflow_group(where,
        cfg)`` at serialization time so individual ``_check_*``
        functions don't need to think about UI structure.  Findings
        whose ``where`` doesn't map to a config field (geometry,
        cell, polymer) keep ``workflow_group = None`` and render in
        the residual "structure" panel.
    """
    severity:        Severity
    message:         str
    where:           str = ""
    workflow_group:  Optional[str] = None
    stage:           Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError(
                f"Issue.severity must be one of {_SEVERITIES}; "
                f"got {self.severity!r}"
            )


class ValidationError(ValueError):
    """Raised when a validation pass found one or more error-severity issues.

    The full list of errors (and any warnings collected at the same
    time) is available on the ``.issues`` attribute.  The exception
    message is the multi-line formatted version of the errors so a
    bare ``except ValidationError as e: print(e)`` shows the whole
    failure picture.
    """

    def __init__(self, issues: Iterable[Issue]):
        self.issues: List[Issue] = list(issues)
        errors = [i for i in self.issues if i.severity == "error"]
        if not errors:
            # If callers raise this without any errors, that's a usage
            # bug -- be loud about it rather than silently misreporting.
            raise ValueError(
                "ValidationError requires at least one error-severity Issue"
            )
        lines = ["validation failed with the following errors:"]
        for i in errors:
            tag = f" [{i.where}]" if i.where else ""
            lines.append(f"  *{tag} {i.message}")
        super().__init__("\n".join(lines))


# --------------------------------------------------------------------- #
#  The HOOK BOUNDARY -- an engine's callable, and whose it was when it    #
#  raises (`execution/script-preparation.md` § 4.6)                       #
# --------------------------------------------------------------------- #


@contextmanager
def calling(hook: str, *, engine: str = "", where: str = "", log=None):
    """Run an ENGINE-SUPPLIED callable, and make any exception say whose.

    **The failure this exists for is not a crash; it is an UNATTRIBUTED
    crash.**  Nearly every step of the preparation pipeline is a hook the
    engine supplied, so a mistake in one arrives as
    ``TypeError: __init__() got an unexpected keyword argument`` from a
    traceback twelve frames deep -- and the first question, *which engine,
    which hook, which item?*, costs a debugging session.  It cost two on
    2026-08-19.

    **The exception is ANNOTATED, never replaced.**  Its type and message are
    what every ``except`` clause and every test already match on, and a
    wrapper class would break all of them.  Worse, it would bury a refusal an
    engine raised deliberately -- SIESTA refuses GPU without an ELPA
    diagonaliser from inside a hook, and that message is written for a person.
    ``add_note`` attaches the attribution and leaves the exception alone;
    nested hooks each add one, so the chain reads innermost-first.

    **It never swallows.**  A hook that raised did not do its job, and a deck
    written anyway is the defect class this layer exists to prevent.  The
    re-raise is unconditional -- the only thing gained here is knowing who.

    ``log`` is the pipeline log when one is open (§ 4.5): the failure lands in
    the file under the phase it happened in, with its traceback, so a run that
    died is explained by the same file that explains a run that worked.
    """
    try:
        yield
    except Exception as exc:
        owner = f"{engine}.{hook}" if engine else hook
        exc.add_note(f"raised inside {owner}" + (f" -- {where}" if where else ""))
        if log is not None:
            log.failed(owner, exc, where=where)
        raise


def notes_of(exc: BaseException) -> str:
    """An exception's attribution chain as one string, or ``""``.

    A conductor turns a hook failure into a user-facing refusal with
    ``str(exc)``, and ``str`` does not include notes -- so without this the
    attribution reaches a traceback and never reaches the person who ran the
    command.
    """
    return "\n".join(getattr(exc, "__notes__", ()) or ())


__all__ = ["Issue", "ValidationError", "calling", "notes_of"]

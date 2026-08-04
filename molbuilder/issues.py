"""Issue dataclass + ValidationError -- validator output type.

A validation pass produces a list of Issues.  Each Issue describes a
single problem with a structure or configuration; the consumer
decides what to do with the list (errors usually raise, warnings
usually print to stderr).

Spec: docs/design.md § "Pre-emission geometry validation" + §
"Validation pass (pre-emission)".
"""

from __future__ import annotations

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
        Used by the CLI / web UI to highlight the offending field.
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


__all__ = ["Issue", "ValidationError"]

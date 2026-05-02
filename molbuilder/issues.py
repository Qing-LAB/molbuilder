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
from typing import Iterable, List


@dataclass(frozen=True)
class Issue:
    """A single validation finding.

    Fields
    ------
    severity : "error" | "warn"
        Errors block emission (render_fdf / render_script raise);
        warnings print to stderr but the run proceeds.
    message : str
        Human-readable, single-line, no trailing punctuation.  Should
        include any actionable advice in-line (e.g. "increase
        cell_padding above 25 Å" rather than "cell too tight").
    where : str
        Locus of the problem, dotted-namespace style:
          "geometry.min_distance"   -- structural finding
          "cell.determinant"        -- about the cell
          "config.mesh_cutoff"      -- about a config field
        Used by the CLI / web UI to highlight the offending field.
    """
    severity: str
    message:  str
    where:    str = ""

    def __post_init__(self) -> None:
        if self.severity not in ("error", "warn"):
            raise ValueError(
                f"Issue.severity must be 'error' or 'warn'; got {self.severity!r}"
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

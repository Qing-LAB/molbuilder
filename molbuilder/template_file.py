"""THE TEMPLATE FILE'S DOOR -- floor 2 (`execution/architecture.md` § 2.1,
§ 3.2, rule A16).

A calculation's parameters lie in ``<label>.template.toml`` at its root:
this module forms that name (:func:`template_filename`, :func:`template_path`)
for the writers, and reads the folder's one template (:func:`of`) for every
reader.  What a template SAYS -- the catalogue of every parameter and the
template format -- is `template`'s, floor 1.
"""
from __future__ import annotations

from pathlib import Path as _Path
from typing import NamedTuple, Optional

from .template import SUFFIX


def template_filename(label: str) -> str:
    """``<label>.template.toml`` — the NAME, for a caller that has no directory.

    ``describe`` returns ``{filename: text}`` and never touches the filesystem
    itself, so it needs the name rather than a path.  Splitting the rule in two
    would put the naming convention in two places, which is what this pair
    exists to prevent: :func:`template_path` is this joined to a directory.
    """
    return f"{label}{SUFFIX}"


def template_path(base, label: str) -> "_Path":
    """``<base>/<label>.template.toml`` — **the one place this name is formed.**

    The naming rule is `job-contracts.md` § 6.3's.
    """
    return _Path(base) / template_filename(label)


class TemplateFile(NamedTuple):
    """The calculation's template as read: where it lies, and its text."""
    path: "_Path"
    text: str


def of(base, label: str) -> Optional[TemplateFile]:
    """THE TEMPLATE'S DOOR (`execution/architecture.md` § 3.2, A16): this
    calculation's template in *base*, read -- the folder's one template,
    named for the calculation's *label* -- or ``None`` when there is none.

    **A folder is one calculation** (`project-layout.md` § 1.4), and its
    template is ``<label>.template.toml`` -- the name :func:`template_path`
    forms for the writers (`jobset init`, the hand-over, the Transport tab,
    `jobset migrate`).  So two templates, or one under another name, is a
    folder molbuilder did not leave, and it is refused by name rather than
    read (W38 M1).  *Nothing here* is ``None``, an answer.
    """
    # `.template.toml` IS a role `runfiles.WRITTEN` declares -- the answers
    # file, written once at the bundle root -- so the search is the
    # catalogue's too (`project-layout.md` § 4.5).
    from .runfiles import find_by_role
    found = find_by_role(base, SUFFIX)
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"{base} holds {len(found)} templates "
            f"({', '.join(p.name for p in found)}), so which one describes "
            f"this calculation is not answerable from the folder alone. A "
            f"folder is ONE calculation (project-layout.md § 1.4); this one's "
            f"is {template_filename(label)} -- remove the other.")
    if found[0].name != template_filename(label):
        raise ValueError(
            f"{found[0]} is not this calculation's template: a calculation "
            f"labelled {label!r} keeps its parameters in "
            f"{template_filename(label)}, the name it was described with.")
    return TemplateFile(found[0], found[0].read_text(encoding="utf-8"))


__all__ = ["TemplateFile", "of", "template_filename", "template_path"]

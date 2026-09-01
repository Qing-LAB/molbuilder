"""Run-directory ``info`` composition — what a run says ABOUT itself.

Module: ``parse/dirs`` (directory-level composers — the ONE parse layer
allowed to touch the filesystem), beside
``atom_metadata_json_for_run_dir``.

``info`` is a structure's free store (`archive/2026-09-01-structure-info-plan.md`,
`web/molview.md` § 8.4a): a dict of key -> value that DESCRIBES a
structure without being part of it.  The tab a viewer sits in is the one
that knows what describes the run it is showing (user, 2026-08-30: *"it
is always the tab it resides in that provides that information"*), and
every such tab asks the same question — **what does this run directory
say about itself?**  This is that question's one answer.

**A new metadata category is a new KEY here, and nowhere else.**  That is
the whole reason ``info`` is a free dict rather than a field per
category: it rides ``installMolecule`` in and ``exportFile`` out already
(§ 8.4a), so a key added here reaches the viewer, the Metadata pane and
the exported ``.molstruct.json`` pair without another line changing.

Today one key:

* ``calculation`` — the electronic contract the directory's deck records
  (``parse.contract.contract_of``), in the field names
  ``TransportConfig`` speaks, so a cited pair fills a transport config
  1:1 (`transport-design.md` § 4.1b).

Callers:
  * ``web/blueprints/watch.py::_run_metadata`` — the block every
    ``/api/watch/load`` answer carries.
  * ``web/blueprints/results.py::api_results_contract`` — the structure
    inspector's door, which reads the ``calculation`` key out of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


def run_info_for_dir(
    directory: Union[str, Path, None],
) -> Optional[Dict[str, Any]]:
    """The ``info`` block *directory* answers for itself, or ``None``.

    ``None`` rather than ``{}`` when the directory says nothing, so this
    reads exactly like its two siblings on the same load response
    (``atom_metadata``, ``periodicity``): a field that is absent when
    there is nothing to say.  A caller that wants a dict either way
    writes ``or {}``; one that hands the answer to the viewer passes it
    through, and the viewer's own load door substitutes ``{}``.

    Never raises: a directory that cannot be read is a directory with
    nothing to say, not a failed load.
    """
    if not directory:
        return None
    from molbuilder.parse.contract import contract_of

    out: Dict[str, Any] = {}
    try:
        calculation = contract_of(directory)
    except Exception:                                       # noqa: BLE001
        calculation = None
    if calculation is not None:
        out["calculation"] = calculation
    return out or None

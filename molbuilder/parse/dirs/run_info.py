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

Two keys:

* ``calculation`` — the electronic contract the directory's deck records
  (``parse.contract.contract_of``), in the catalogue's own names
  (``parse.contract.RECORDED_FIELDS``),
  so a cited pair defaults a transport calculation's template
  (`transport-design.md` § 4.1b).
* ``relaxation`` — what the run did to the geometry it left, read from
  the output the viewer has open, which the caller hands down
  (``parse.contract.relaxation_of``, `model/parse.md` § 5b.1): its force
  tolerance, the largest force left on the atoms it moved, the held set
  and a fingerprint of the final geometry, so a structure exported from a
  finished relaxation can be checked against its own record when it is
  stated relaxed (`engines/vibration.md` § 2.2).

Callers, each handing the output it has open:
  * ``web/blueprints/watch.py::_run_metadata`` — the block every
    ``/api/watch/load`` answer carries: the file the load opened.
  * ``web/blueprints/results.py::api_results_contract`` — the structure
    inspector's door: the run door's choice for the structure's folder
    (``runs.openable``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


def run_info_for_dir(
    directory: Union[str, Path, None], *,
    output: Union[str, Path, None] = None,
    traj: Any = None,
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

    ``output`` is the run output the viewer has open -- the run door's
    choice for the directory, or the file a person pointed at -- handed
    down because `parse/` cannot ask the door: the relaxation is ITS
    record, and there is none without one (`contract.relaxation_of`).
    ``traj`` is its parse when the caller holds one -- the viewer's load
    -- so the file is not parsed again.
    """
    if not directory:
        return None
    from molbuilder.parse.contract import contract_of, relaxation_of

    out: Dict[str, Any] = {}
    try:
        calculation = contract_of(directory)
    except Exception:                                       # noqa: BLE001
        calculation = None
    if calculation is not None:
        out["calculation"] = calculation
    try:
        relaxation = (relaxation_of(output, traj=traj)
                      if output else None)
    except Exception:                                       # noqa: BLE001
        relaxation = None
    if relaxation is not None:
        out["relaxation"] = relaxation
    return out or None

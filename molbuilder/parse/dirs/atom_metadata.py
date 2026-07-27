"""Run-directory ATOM-METADATA recovery — the results-side bridge.

Module: ``parse/dirs`` (directory-level composers — the ONE parse
layer allowed to touch the filesystem).  The text-level extraction
stays in ``parse/scripts/atom_metadata`` (``AtomMetadataTextParser``),
which is memory-only by contract (parse-module.md § 9 forbidden #2,
enforced by ``test_text_parsers_do_no_io``) — that contract is WHY
this glob/read helper lives here and not next to the TextParser.

Callers:
  * ``web/blueprints/watch.py::_atom_metadata_json`` — the Results-tab
    load adapter (``/api/watch/load``).
  * ``tests/test_atom_metadata_results_bridge.py`` — the end-result
    seam tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from molbuilder.parse.scripts.atom_metadata import AtomMetadataTextParser


def atom_metadata_json_for_run_dir(
    run_dir: Union[str, Path, None], n_atoms: Optional[int] = None
) -> Optional[str]:
    """Recover a run directory's embedded per-atom metadata block as a JSON
    string, ready to apply onto a loaded structure.

    The Build tab writes region labels / frozen tags / annotation channels
    into the ATOM-METADATA block of the input script it emits (``.fdf`` for
    SIESTA, ``.py`` for PySCF).  A results-side consumer that holds only the
    run's *output* geometry -- e.g. the Results-tab trajectory inspector,
    which loads coordinates from ``.molwatch.log`` / ``.out`` -- calls this
    to recover the metadata and re-apply it through
    ``sidecars.molstruct.apply_to_structure`` (the lenient, atom-count-only
    seam), so the loaded view carries the same regions/frozen/annotations.

    This is a TRUSTED FRAGMENT, not a standalone ``.molstruct.json`` file:
    the block is molbuilder's OWN emit and, by design, omits the sidecar
    file envelope's integrity fields (``structure_hash``).  It must therefore
    be applied via ``apply_to_structure``, NOT validated through
    ``molstruct.load_text`` (which demands the full untrusted-file envelope).

    The block carries ONLY atom-scoped keys (schema_version / n_atoms_total /
    regions / frozen_atoms / selection_rules / annotations -- never cell /
    axis_kind / vacuum), so applying it never disturbs the consumer's own
    geometry or lattice.

    ``n_atoms``, when given, guards the block against the consumer's
    structure: a mismatch means the block's 0-based indices no longer point
    at the same atoms, so ``None`` is returned rather than metadata that
    would make ``apply_to_structure`` raise.

    Returns ``None`` when ``run_dir`` is falsy / not a directory, no input
    script carries a non-empty block, or the atom count disagrees.  Never
    raises -- a results view must still show coordinates when metadata
    recovery fails.
    """
    if not run_dir:
        return None
    try:
        d = Path(run_dir)
        if not d.is_dir():
            return None
        # SIESTA (.fdf) first, then PySCF (.py).  Staged SIESTA runs share
        # one metadata block across stages, so the first script carrying a
        # non-empty block wins; later scripts are consulted only if earlier
        # ones are absent or empty.
        scripts = sorted(d.glob("*.fdf")) + sorted(d.glob("*.py"))
    except OSError:                                         # pragma: no cover
        return None
    for script in scripts:
        try:
            text = script.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:                                    # pragma: no cover
            continue
        md = AtomMetadataTextParser.parse(text).atom_metadata
        if not md:
            continue
        if not (md.get("regions") or md.get("frozen_atoms")
                or md.get("annotations")):
            continue                       # empty block -> try the next script
        if n_atoms is not None and md.get("n_atoms_total") != n_atoms:
            continue                       # indices no longer match -> skip
        try:
            return json.dumps(md)
        except (TypeError, ValueError):                    # pragma: no cover
            return None
    return None


__all__ = ["atom_metadata_json_for_run_dir"]

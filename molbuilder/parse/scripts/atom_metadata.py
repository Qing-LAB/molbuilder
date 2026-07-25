"""ATOM-METADATA block TextParser.

H1 of parse-module.md migration (was Phase F wrapper around
``script_contract.extract_atom_metadata_dict``): absorbed the
extractor body directly so this module no longer imports from
``molbuilder.script_contract``.

ATOM-METADATA carries the regions + frozen_atoms (the
``.molstruct.json`` schema v3 payload embedded in the .fdf /
.py).  Inside the BEGIN/END markers each line is comment-prefixed
(``# `` or ``#``); the extractor strips the prefix and then walks
the brace-balance of the JSON to support both pretty-printed
multi-line and compact single-line payloads.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from ._helpers import empty_script_result
from .markers import BLOCK_ATOM_METADATA, MARKER_RE


def _extract_atom_metadata_dict(text: str) -> Optional[Dict[str, Any]]:
    """Find the ATOM-METADATA block in ``text`` and return its JSON
    payload as a dict.

    Returns ``None`` when:
      * No ATOM-METADATA block is present.
      * The block markers are unbalanced.
      * The JSON between markers fails to parse.

    Comment-prefix-per-line is stripped before JSON parsing.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != BLOCK_ATOM_METADATA:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    # Inner lines: strip leading "# " (or "#") to recover JSON.
    inner: List[str] = []
    for raw in lines[begin_idx + 1: end_idx]:
        if raw.startswith("# "):
            inner.append(raw[2:])
        elif raw.startswith("#"):
            inner.append(raw[1:])
        else:
            inner.append(raw)
    # Brace-balance walk so the extractor accepts BOTH pretty-printed
    # JSON (molbuilder's emit_atom_metadata via json.dumps indent=2)
    # AND compact / single-line JSON.  The contract on the wire is
    # "valid JSON inside the block"; how the writer formatted it isn't
    # load-bearing.
    json_lines: List[str] = []
    saw_open = False
    brace_depth = 0
    for line in inner:
        stripped = line.strip()
        if not saw_open:
            if not stripped or not stripped.startswith("{"):
                continue
            saw_open = True
        json_lines.append(line)
        brace_depth += stripped.count("{") - stripped.count("}")
        if brace_depth <= 0:
            break
    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


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
        md = _extract_atom_metadata_dict(text)
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


class AtomMetadataTextParser(TextParser):
    """Extract the ``ATOM-METADATA`` block as the embedded
    molstruct/v3 dict.  Returns a :class:`ScriptResult` with
    only the ``atom_metadata`` field set; the dict's
    ``schema_version`` is also surfaced into the result's
    ``block_schema_versions["atom-metadata"]`` for
    cross-block-version auditing."""

    name   = "fdf-atom-metadata"
    label  = "molbuilder ATOM-METADATA block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        atom_md = _extract_atom_metadata_dict(text)
        schema_versions = {}
        if atom_md and isinstance(atom_md.get("schema_version"), int):
            schema_versions["atom-metadata"] = atom_md["schema_version"]
        return replace(
            base,
            atom_metadata=atom_md,
            block_schema_versions=schema_versions,
        )

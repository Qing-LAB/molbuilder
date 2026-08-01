"""molbuilder.script_emit — write-side of the script-contract blocks.

H2 of parse-module.md migration: absorbed from the legacy
:mod:`molbuilder.script_contract` module's write-side surface.  The
read-side (per-block extractors) lives in :mod:`molbuilder.parse.scripts`
per the parse-module contract.

This module is the canonical home for the shared building blocks of
the script-contract reserved blocks (HEADER / PROVENANCE /
BENCH-MARKS / ATOM-METADATA / USER-CUSTOM, plus the marker regex
and the BenchField declarations).  The read-side
:mod:`molbuilder.parse.scripts.markers` re-imports the
:data:`BLOCK_*` constants and :data:`MARKER_RE` from here so the
two sides stay in lock-step.

Public surface
--------------

* Constants — :data:`BLOCK_HEADER`, :data:`BLOCK_PROVENANCE`,
  :data:`BLOCK_BENCH_MARKS`, :data:`BLOCK_ATOM_METADATA`,
  :data:`BLOCK_USER_CUSTOM`, :data:`MARKER_RE`.
* Marker helpers — :func:`begin_marker`, :func:`end_marker`.
* Bench declarations — :class:`BenchField`,
  :data:`SIESTA_BENCH_FIELDS`.
* Block emitters — :func:`emit_header`, :func:`emit_provenance`,
  :func:`emit_bench_marks`, :func:`emit_atom_metadata`,
  :func:`emit_user_custom_placeholder`.
* In-body application — :func:`apply_inbody_atom_metadata` (mutates
  a Structure from the embedded ATOM-METADATA JSON; mirrors the
  sidecar protocol minus the structure_hash check).
* USER-CUSTOM round-trip — :func:`replace_user_custom_inner`,
  :func:`merge_user_custom_from_target`.
* Provenance helpers — :func:`molbuilder_git_sha`,
  :func:`generated_at_now`.

Pure functions (no I/O) except :func:`molbuilder_git_sha` (one
``git rev-parse`` subprocess) and
:func:`merge_user_custom_from_target` (reads the existing target
file).  Callers stitch block strings together themselves.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------- #
#  Block markers + names                                                #
# --------------------------------------------------------------------- #

# Block names used in the markers.  Centralised so a typo doesn't
# silently produce a file the parser refuses.
BLOCK_HEADER        = "header"
BLOCK_PROVENANCE    = "provenance"
BLOCK_BENCH_MARKS   = "bench-marks"
BLOCK_ATOM_METADATA = "atom-metadata"
BLOCK_USER_CUSTOM   = "user-custom"


def begin_marker(name: str) -> str:
    """Return the literal BEGIN marker line for a reserved block."""
    return f"# === molbuilder {name} BEGIN ==="


def end_marker(name: str) -> str:
    """Return the literal END marker line for a reserved block."""
    return f"# === molbuilder {name} END ==="


# Regex matching either marker for any block.  Group 1: block name;
# group 2: BEGIN | END.
MARKER_RE = re.compile(
    r"^#\s*===\s+molbuilder\s+([a-z-]+)\s+(BEGIN|END)\s+===\s*$"
)


# --------------------------------------------------------------------- #
#  BENCH-MARKS field declarations                                       #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenchField:
    """Declaration of one bench-overridable parameter.

    Per the contract: tools may override this field; the anchor
    locates the override site in ENGINE BODY by greping for the start
    of a code line.
    """
    name: str                                 # human-readable label
    anchor: str                               # what bench greps for in engine body
    type_: str                                # "int" | "float" | "str" | "pow2"
    range_: Optional[Tuple[float, float]] = None
    unit: Optional[str] = None


# Static field list for SIESTA .fdf.  PySCF and future engines get
# their own list when their bench subcommands land.
#
# Anchor (post-2026-06-23 SIESTA keyword fix): the generator
# emits ``MD.NumCGsteps`` UNIVERSALLY (CG / Broyden / FIRE) because
# it's the only step-count keyword SIESTA 5.4.2 recognizes -- the
# per-type aliases ``MD.NumBroydenSteps`` / ``MD.NumFIRESteps``
# don't exist in 5.4.2 and were silently dropped pre-fix.  So the
# bench anchor is the same regardless of cfg.relax_type and Step 4
# (molbuilder bench siesta-gpu) needs no per-type dispatch.  See
# decision-log 2026-06-23 in design.md.  Task #486 is closed by
# this realization.
SIESTA_BENCH_FIELDS: List[BenchField] = [
    BenchField("BlockSize",        "BlockSize",        "pow2",  (16, 256)),
    BenchField("MaxSCFIterations", "MaxSCFIterations", "int"),
    BenchField("MD.NumCGsteps",    "MD.NumCGsteps",    "int"),
    BenchField("MeshCutoff",       "MeshCutoff",       "float", None, "Ry"),
    # ELPA solver variant.  ELPA-1STAGE: direct dense.
    # ELPA-2STAGE: tridiagonalises through a banded form; typically
    # wins on N>1000 orbital problems and on GPU because the band
    # step parallelises well.  Bench sweeps both.
    BenchField("Diag.Algorithm",   "Diag.Algorithm",   "enum"),
]


# --------------------------------------------------------------------- #
#  Block emitters                                                       #
# --------------------------------------------------------------------- #


def emit_header(lines: List[str]) -> str:
    """Wrap the given comment-prefixed lines in a HEADER block.

    Caller is responsible for prefixing each line with the engine's
    comment character (``#`` for .fdf / .py / .run.sh — the only
    comment char in scope here).  Empty list produces an empty block.
    """
    body = "\n".join(lines)
    return (
        begin_marker(BLOCK_HEADER) + "\n"
        + (body + "\n" if body else "")
        + end_marker(BLOCK_HEADER)
    )


def emit_provenance(generator_version: str,
                    generated_at: str,
                    resolved_defaults: Optional[Dict[str, str]] = None,
                    form_config_hash: Optional[str] = None) -> str:
    """Emit the PROVENANCE block.

    ``resolved_defaults`` is a flat dict of "field -> description";
    e.g. {"BlockSize": "auto -> 256 (10 * 212 atoms / mpi_np)"}.
    Caller decides what to include.  The block has no version tag
    (per the contract — keys are additive and forward-compatible).
    """
    out: List[str] = [begin_marker(BLOCK_PROVENANCE)]
    out.append(f"#   generator-version    {generator_version}")
    out.append(f"#   generated-at         {generated_at}")
    if form_config_hash:
        out.append(f"#   form-config-hash     {form_config_hash}")
    if resolved_defaults:
        out.append("#   resolved-defaults:")
        max_key = max(len(k) for k in resolved_defaults.keys())
        for key in sorted(resolved_defaults.keys()):
            out.append(f"#     {key:<{max_key}}  {resolved_defaults[key]}")
    out.append(end_marker(BLOCK_PROVENANCE))
    return "\n".join(out)


def emit_bench_marks(metadata: Dict[str, Any],
                     fields: List[BenchField],
                     defaults: Dict[str, Any],
                     version: str = "v1") -> str:
    """Emit the BENCH-MARKS block.

    ``metadata`` — informational top-level keys (n_atoms,
        n_orbitals_est, gpu_mode, numa_pin, ...).
    ``fields`` — field declarations (per-engine static list).
    ``defaults`` — name → resolved default value at generation time;
        appended to the field line as ``default=...``.
    """
    out: List[str] = [begin_marker(BLOCK_BENCH_MARKS)]
    out.append(f"#   version {version}")
    if metadata:
        max_key = max(len(k) for k in metadata.keys())
        for key in metadata.keys():
            out.append(f"#   {key:<{max_key}}  {metadata[key]}")
        out.append("#")
    # Align the field-decl columns so the block is human-readable.
    max_name   = max((len(f.name)   for f in fields), default=0)
    max_anchor = max((len(f.anchor) for f in fields), default=0)
    for f in fields:
        line = (
            f"#   field {f.name:<{max_name}}  "
            f"anchor={f.anchor:<{max_anchor}}  type={f.type_}"
        )
        if f.range_ is not None:
            lo, hi = f.range_
            line += f"  range=[{lo},{hi}]"
        if f.unit:
            line += f"  unit={f.unit}"
        if f.name in defaults and defaults[f.name] is not None:
            line += f"  default={defaults[f.name]}"
        out.append(line)
    out.append(end_marker(BLOCK_BENCH_MARKS))
    return "\n".join(out)


def emit_atom_metadata(regions: Dict[str, List[int]],
                       n_atoms_total: int,
                       created_by: str = "molbuilder",
                       created_at: Optional[str] = None,
                       selection_rules: Optional[Dict[str, Any]] = None,
                       annotations: Optional[Dict[str, Any]] = None
                       ) -> Optional[str]:
    """Emit the ATOM-METADATA block, or return ``None`` when there is
    nothing to emit.

    This is **data-model persistence** (atom-annotations.md § 3) — the
    script's engine-agnostic, round-trippable copy of the per-atom data model,
    the same shape as the ``.molstruct.json`` sidecar.  It is NOT engine setup
    (frozen -> Geometry.Constraints etc. is a separate translation, § 4).

    Per the contract's emission rule, the block is emitted ONLY when at least
    one of ``regions`` / ``annotations`` is non-empty.  Absence is the honest
    signal that this generation had no labels.

    ``regions`` is the whole label store, so a reserved label -- ``frozen_atoms``
    -- is IN it rather than beside it.  The block is the sidecar's shape and
    follows it (``structure-annotations.md`` § 2); a key of its own would put
    the same fact in the generated script twice.

    Index convention: 0-based throughout the JSON payload, matching
    ``molstruct_json`` schema v4 and the in-Python ``Structure`` model.
    ``annotations`` are the extensible channels (schema v4+), serialised the
    same way the sidecar serialises them.
    """
    regions = regions or {}
    # Serialise extensible channels (AtomChannel -> JSON) exactly as the sidecar.
    normed_annotations: Dict[str, Any] = {}
    if annotations:
        from molbuilder.structure import AtomChannel as _AtomChannel
        for name, ch in annotations.items():
            if isinstance(ch, _AtomChannel):
                normed_annotations[name] = ch.to_json()
            elif isinstance(ch, dict) and "kind" in ch:
                normed_annotations[name] = _AtomChannel.from_json(ch).to_json()
    if not regions and not normed_annotations:
        return None
    # THE VERSION THE SIDECAR STAMPS, from the one constant -- never a literal.
    #
    # This said 4 while the block was written in the CURRENT shape (the rule
    # above: `regions` is the whole label store, reserved names included). A
    # block that claims a version it is not written in is worse than one with no
    # version at all, because a reader cannot refuse what it cannot recognise:
    # a script generated before the label store was unified and one generated
    # after it BOTH said 4 while holding different shapes, so nothing could tell
    # them apart -- and a real run's fifty frozen electrode atoms came back as an
    # empty list with the file looking perfectly fine.
    #
    # It is the sidecar's constant because it is the sidecar's shape ("the same
    # shape as the .molstruct.json sidecar", above); two numbers for one format
    # is the drift this whole block exists to avoid.
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION as _SCHEMA_VERSION
    payload: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "n_atoms_total":  int(n_atoms_total),
    }
    if regions:
        payload["regions"] = {
            k: sorted(set(int(i) for i in v))
            for k, v in regions.items()
        }
    if selection_rules:
        payload["selection_rules"] = selection_rules
    if normed_annotations:
        payload["annotations"] = normed_annotations
    payload["created_by"] = created_by
    if created_at:
        payload["created_at"] = created_at
    out: List[str] = [begin_marker(BLOCK_ATOM_METADATA)]
    out.append(f"# format: molstruct-json/v{_SCHEMA_VERSION}")
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    for line in body.splitlines():
        out.append(f"# {line}")
    out.append(end_marker(BLOCK_ATOM_METADATA))
    return "\n".join(out)


def emit_user_custom_placeholder() -> str:
    """Empty USER-CUSTOM block emitted on every fresh generation.

    The round-trip (:func:`merge_user_custom_from_target`) replaces
    this placeholder with the existing target file's user-custom
    content byte-for-byte across regenerations.
    """
    return "\n".join([
        begin_marker(BLOCK_USER_CUSTOM),
        "# Your own additions go here.  molbuilder will preserve",
        "# this section verbatim across regenerations.",
        end_marker(BLOCK_USER_CUSTOM),
    ])


# --------------------------------------------------------------------- #
#  In-body atom-metadata: apply to Structure                            #
# --------------------------------------------------------------------- #
#
# Step 3 (audit finding A1, 2026-06-16): molstruct_json.from_dict
# validates ``structure_hash`` (>=16 chars).  The in-body
# atom-metadata deliberately omits structure_hash per the contract
# (metadata + coordinates are written by the same generator pass and
# cannot drift apart by construction).  So molstruct_json's loader
# is the wrong entry point for in-body payloads — we need a small
# local apply that doesn't require the hash.


def apply_inbody_atom_metadata(struct: Any, text: str) -> bool:
    """If ``text`` carries an ATOM-METADATA block, apply its
    ``regions`` and ``frozen_atoms`` to ``struct``.

    ``struct`` is duck-typed: any object with mutable ``regions``
    (dict) and ``frozen_atoms`` (list) attributes will do.  Mirrors
    the protocol that :func:`molbuilder.sidecars.molstruct.apply_to_structure`
    uses for the sidecar path, minus the structure_hash check.

    Returns ``True`` when labels were applied, ``False`` otherwise
    (no block, or block carried empty regions + frozen_atoms).
    """
    # Local import — avoids a circular import via parse.scripts.
    from molbuilder.parse.scripts.atom_metadata import (
        _extract_atom_metadata_dict,
    )
    payload = _extract_atom_metadata_dict(text)
    if payload is None:
        return False
    regions = payload.get("regions") or {}
    annotations = payload.get("annotations") or {}
    if not regions and not annotations:
        return False
    if regions:
        # Normalise: sort + dedupe per label; coerce to int.  Reserved labels
        # are in here with the rest -- there is no second key to read.
        struct.regions = {
            str(k): sorted({int(i) for i in v})
            for k, v in regions.items()
        }
    if annotations:
        # Extensible channels (v4) -> struct.annotations, same round-trip as
        # the sidecar (§ 3 data-model persistence).
        from molbuilder.structure import annotations_from_json
        struct.annotations = annotations_from_json(annotations)
    return True


# --------------------------------------------------------------------- #
#  USER-CUSTOM round-trip preservation                                  #
# --------------------------------------------------------------------- #
#
# When the generator writes a fresh render over an existing target
# file, preserve the user-custom block content byte-for-byte.  Callers
# (typically the /api/files/write endpoint) chain
#
#     final_text = merge_user_custom_from_target(rendered, target_path)
#
# before actually writing.  ``rendered`` is what render_fdf /
# render_script / render_run_wrapper produced (carries the empty
# placeholder); ``target_path`` is where the file will live.  If the
# existing target carries a user-custom block, its inner lines splice
# into the new render's placeholder.  Edge cases (no existing file,
# no markers on either side, corrupt markers) all degrade to "return
# rendered unchanged" — the merge never throws.


def replace_user_custom_inner(text: str, inner_lines: List[str]) -> str:
    """Return ``text`` with its USER-CUSTOM block's inner lines
    replaced by ``inner_lines``.  If ``text`` has no USER-CUSTOM
    block, return it unchanged.
    """
    lines = text.splitlines(keepends=False)
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != BLOCK_USER_CUSTOM:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        return text
    new_lines = lines[: begin_idx + 1] + list(inner_lines) + lines[end_idx:]
    # Preserve trailing newline policy of the input.
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + trailing


def merge_user_custom_from_target(rendered: str,
                                  target_path: Path) -> str:
    """High-level merge: splice the existing target file's USER-CUSTOM
    block content into ``rendered`` before write.

    Safe in every degenerate case:
      * Target doesn't exist → return rendered.
      * Target has no USER-CUSTOM block → return rendered.
      * Rendered has no USER-CUSTOM placeholder → return rendered.
      * Target is unreadable → return rendered.
    """
    # Local import — avoids a circular import via parse.scripts.
    from molbuilder.parse.scripts.user_custom import (
        _extract_user_custom_inner,
    )
    try:
        if not target_path.exists():
            return rendered
        old_text = target_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rendered
    old_inner = _extract_user_custom_inner(old_text)
    if old_inner is None:
        return rendered
    return replace_user_custom_inner(rendered, old_inner)


# --------------------------------------------------------------------- #
#  Git / time helpers                                                   #
# --------------------------------------------------------------------- #


def molbuilder_git_sha() -> str:
    """Return the molbuilder git SHA (short form), or ``"unknown"``.

    Best-effort: 2 s subprocess timeout, returns ``"unknown"`` on any
    failure (no git, not a repo, stdin closed in a packaged install).
    Caller may also pass a literal SHA from elsewhere if a more
    authoritative source exists.
    """
    try:
        # Resolve repo root via this file's path (works under both
        # editable and packaged installs as long as the source is
        # actually present).
        repo_root = Path(__file__).resolve().parent.parent
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            timeout=2.0,
            stderr=subprocess.DEVNULL,
        )
        return f"git {out.decode('ascii').strip()}"
    except Exception:  # noqa: BLE001 — best-effort metadata
        return "unknown"


def generated_at_now() -> str:
    """ISO-8601 timestamp with timezone, seconds precision."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


# --------------------------------------------------------------------- #
#  Public re-exports of the per-block extractors                        #
# --------------------------------------------------------------------- #
#
# The block extractors live next to their TextParser definitions in
# ``molbuilder/parse/scripts/`` (Phase F per parse-module.md § 8) +
# the umbrella ``extract_script_source`` in ``parse/dirs/bundle.py``
# (Phase G).  Surface them here under their legacy unprefixed names
# so the emit/extract pair is reachable from a single module —
# write- and read-side of the same on-disk format.
#
# Resolved via module-level ``__getattr__`` so the imports happen
# AFTER ``parse/scripts/markers.py`` finishes initialising; an
# eager top-level import would deadlock because markers.py
# re-exports BLOCK_* + MARKER_RE from this module.
_LAZY_EXTRACTORS = {
    "extract_atom_metadata_dict": (
        "molbuilder.parse.scripts.atom_metadata",
        "_extract_atom_metadata_dict",
    ),
    "extract_bench_marks_dict": (
        "molbuilder.parse.scripts.bench_marks",
        "_extract_bench_marks_dict",
    ),
    "extract_header_text": (
        "molbuilder.parse.scripts.header",
        "_extract_header_text",
    ),
    "extract_provenance_dict": (
        "molbuilder.parse.scripts.provenance",
        "_extract_provenance_dict",
    ),
    "extract_user_custom_inner": (
        "molbuilder.parse.scripts.user_custom",
        "_extract_user_custom_inner",
    ),
    "extract_script_source": (
        "molbuilder.parse.dirs.bundle",
        "_extract_script_source",
    ),
}


def __getattr__(name):
    target = _LAZY_EXTRACTORS.get(name)
    if target is None:
        raise AttributeError(
            f"module 'molbuilder.script_emit' has no attribute {name!r}")
    import importlib
    mod_name, attr = target
    value = getattr(importlib.import_module(mod_name), attr)
    globals()[name] = value  # cache for next access
    return value


__all__ = [
    # Block names + markers
    "BLOCK_HEADER", "BLOCK_PROVENANCE", "BLOCK_BENCH_MARKS",
    "BLOCK_ATOM_METADATA", "BLOCK_USER_CUSTOM",
    "MARKER_RE", "begin_marker", "end_marker",
    # Bench declarations
    "BenchField", "SIESTA_BENCH_FIELDS",
    # Emitters
    "emit_header", "emit_provenance", "emit_bench_marks",
    "emit_atom_metadata", "emit_user_custom_placeholder",
    # In-body application
    "apply_inbody_atom_metadata",
    # USER-CUSTOM round-trip
    "replace_user_custom_inner", "merge_user_custom_from_target",
    # Git / time
    "molbuilder_git_sha", "generated_at_now",
    # Per-block extractors (read-side; re-export from parse/scripts/)
    "extract_atom_metadata_dict", "extract_bench_marks_dict",
    "extract_header_text", "extract_provenance_dict",
    "extract_user_custom_inner", "extract_script_source",
]

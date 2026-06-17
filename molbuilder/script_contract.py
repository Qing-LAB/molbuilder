"""molbuilder.script_contract -- emit and parse the script-contract blocks.

Single source of truth for the format defined in
``docs/protocols/script-contract.md``.  Other generators
(``siesta/input.py``, ``pyscf/input.py``, ``runwrap.py``) use the
``emit_*`` functions here so the contract is consistent across every
generated file.  Parsers (Step 2b user-custom round-trip, Step 4
bench subcommand) use the regex constants here too.

The functions are pure: no I/O, no subprocess except the optional
``molbuilder_git_sha`` helper.  They return string-shaped output;
the caller is responsible for stitching block strings together.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------- #
#  Block markers + names                                                #
# --------------------------------------------------------------------- #

# Block names used in the markers.  Centralised so a typo doesn't
# silently produce a file the parser refuses.
BLOCK_HEADER = "header"
BLOCK_PROVENANCE = "provenance"
BLOCK_BENCH_MARKS = "bench-marks"
BLOCK_ATOM_METADATA = "atom-metadata"
BLOCK_USER_CUSTOM = "user-custom"


def begin_marker(name: str) -> str:
    """Return the literal BEGIN marker line for a reserved block."""
    return f"# === molbuilder {name} BEGIN ==="


def end_marker(name: str) -> str:
    """Return the literal END marker line for a reserved block."""
    return f"# === molbuilder {name} END ==="


# Regex matching either marker for any block.  Group 1: kind
# ("BEGIN" | "END"); group 2: block name.
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
    locates the override site in ENGINE BODY by greping for the
    start of a code line.
    """
    name: str                                 # human-readable label
    anchor: str                               # what bench greps for in engine body
    type_: str                                # "int" | "float" | "str" | "pow2"
    range_: Optional[Tuple[float, float]] = None
    unit: Optional[str] = None


# Static field list for SIESTA .fdf.  PySCF and future engines get
# their own list when their bench subcommands land.
#
# Anchor caveat (caught 2026-06-16 post-2a audit): the
# ``MD.NumCGsteps`` anchor is only emitted by ``siesta/input.py``
# when ``cfg.relax_type == "CG"``.  For Broyden the engine body
# carries ``MD.NumBroydenSteps``; for FIRE, ``MD.NumFIRESteps``;
# etc.  Step 4 (molbuilder bench siesta-gpu) must either:
#   * select the correct anchor per ``cfg.relax_type``, OR
#   * filter SIESTA_BENCH_FIELDS by which anchors are actually
#     present in the engine body before emitting BENCH-MARKS.
# Today the bench would silently fail to find the anchor on
# non-CG runs.  Tracked under task #486.
SIESTA_BENCH_FIELDS: List[BenchField] = [
    BenchField("BlockSize",        "BlockSize",        "pow2",  (16, 256)),
    BenchField("MaxSCFIterations", "MaxSCFIterations", "int"),
    BenchField("MD.NumCGsteps",    "MD.NumCGsteps",    "int"),
    BenchField("MeshCutoff",       "MeshCutoff",       "float", None, "Ry"),
]


# --------------------------------------------------------------------- #
#  Emitters -- pure functions, return block-shaped strings              #
# --------------------------------------------------------------------- #


def emit_header(lines: List[str]) -> str:
    """Wrap the given comment-prefixed lines in a HEADER block.

    Caller is responsible for prefixing each line with the engine's
    comment character (``#`` for .fdf / .py / .run.sh -- the only
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
    (per the contract -- keys are additive and forward-compatible).
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

    ``metadata`` -- informational top-level keys (n_atoms,
        n_orbitals_est, gpu_mode, numa_pin, ...).
    ``fields`` -- field declarations (per-engine static list).
    ``defaults`` -- name -> resolved default value at generation
        time; appended to the field line as ``default=...``.
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
                       frozen_atoms: List[int],
                       n_atoms_total: int,
                       created_by: str = "molbuilder",
                       created_at: Optional[str] = None,
                       selection_rules: Optional[Dict[str, Any]] = None
                       ) -> Optional[str]:
    """Emit the ATOM-METADATA block, or return ``None`` when there is
    nothing to emit.

    Per the contract's emission rule, the block is emitted ONLY when
    at least one of ``regions`` / ``frozen_atoms`` is non-empty.
    Absence is the honest signal that this generation had no labels;
    a downstream sidecar (.molstruct.json) may then still apply.

    Index convention: 0-based throughout the JSON payload, matching
    ``molstruct_json`` schema v3 and the in-Python ``Structure`` model.
    """
    regions = regions or {}
    frozen_atoms = frozen_atoms or []
    if not regions and not frozen_atoms:
        return None
    payload: Dict[str, Any] = {
        "schema_version": 3,
        "n_atoms_total":  int(n_atoms_total),
    }
    if regions:
        payload["regions"] = {
            k: sorted(set(int(i) for i in v))
            for k, v in regions.items()
        }
    if frozen_atoms:
        payload["frozen_atoms"] = sorted(set(int(i) for i in frozen_atoms))
    if selection_rules:
        payload["selection_rules"] = selection_rules
    payload["created_by"] = created_by
    if created_at:
        payload["created_at"] = created_at
    out: List[str] = [begin_marker(BLOCK_ATOM_METADATA)]
    out.append("# format: molstruct-json/v3")
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    for line in body.splitlines():
        out.append(f"# {line}")
    out.append(end_marker(BLOCK_ATOM_METADATA))
    return "\n".join(out)


def emit_user_custom_placeholder() -> str:
    """Empty USER-CUSTOM block emitted on every fresh generation.

    Step 2b will replace this with a round-trip that preserves an
    existing user-custom block byte-for-byte across regenerations.
    """
    return "\n".join([
        begin_marker(BLOCK_USER_CUSTOM),
        "# Your own additions go here.  molbuilder will preserve",
        "# this section verbatim across regenerations.",
        end_marker(BLOCK_USER_CUSTOM),
    ])


# --------------------------------------------------------------------- #
#  Helpers used by emitters' callers                                    #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  USER-CUSTOM round-trip preservation                                  #
# --------------------------------------------------------------------- #
#
# Step 2b: when the generator writes a fresh render over an existing
# target file, preserve the user-custom block content byte-for-byte.
# Callers (typically the /api/files/write endpoint) chain
#
#     final_text = merge_user_custom_from_target(rendered, target_path)
#
# before actually writing.  ``rendered`` is what render_fdf /
# render_script / render_run_wrapper produced (carries the empty
# placeholder); ``target_path`` is where the file will live.  If the
# existing target carries a user-custom block, its inner lines splice
# into the new render's placeholder.  Edge cases (no existing file,
# no markers on either side, corrupt markers) all degrade to "return
# rendered unchanged" -- the merge never throws.


def extract_user_custom_inner(text: str) -> Optional[List[str]]:
    """Return the inner lines of the USER-CUSTOM block in ``text``, or
    ``None`` if there is no well-formed USER-CUSTOM BEGIN/END pair.

    Inner lines are everything STRICTLY between the BEGIN and END
    markers (markers excluded).  Trailing/leading whitespace inside
    the block is preserved.
    """
    lines = text.splitlines()
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
        return None
    return lines[begin_idx + 1: end_idx]


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
      * Target doesn't exist -> return rendered.
      * Target has no USER-CUSTOM block -> return rendered.
      * Rendered has no USER-CUSTOM placeholder -> return rendered.
      * Target is unreadable -> return rendered.
    """
    try:
        if not target_path.exists():
            return rendered
        old_text = target_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rendered
    old_inner = extract_user_custom_inner(old_text)
    if old_inner is None:
        return rendered
    return replace_user_custom_inner(rendered, old_inner)


def molbuilder_git_sha() -> str:
    """Return the molbuilder git SHA (short form), or "unknown".

    Best-effort: 2 s subprocess timeout, returns "unknown" on any
    failure (no git, not a repo, stdin closed in a packaged
    install).  Caller may also pass a literal SHA from elsewhere
    if a more authoritative source exists.
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
    except Exception:  # noqa: BLE001 -- best-effort metadata
        return "unknown"


def generated_at_now() -> str:
    """ISO-8601 timestamp with timezone, seconds precision."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


__all__ = [
    "BLOCK_HEADER", "BLOCK_PROVENANCE", "BLOCK_BENCH_MARKS",
    "BLOCK_ATOM_METADATA", "BLOCK_USER_CUSTOM",
    "begin_marker", "end_marker", "MARKER_RE",
    "BenchField", "SIESTA_BENCH_FIELDS",
    "emit_header", "emit_provenance", "emit_bench_marks",
    "emit_atom_metadata", "emit_user_custom_placeholder",
    "extract_user_custom_inner", "replace_user_custom_inner",
    "merge_user_custom_from_target",
    "molbuilder_git_sha", "generated_at_now",
]

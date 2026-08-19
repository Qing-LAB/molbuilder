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
import dataclasses as _dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Tuple)

if TYPE_CHECKING:                       # annotations only -- `issues`
    from .issues import Issue           # is L1 and imports nothing


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
#: The parameters the ENGINE actually holds, recorded into the run log at
#: startup.  Shared between engines on purpose: the deck says what was asked
#: for, this says what was heard, and one reader should be able to compare
#: them without knowing which engine wrote it.
BLOCK_PARAMETERS    = "effective-parameters"
#: § 3.7 item blocks: the marker is ``item <field>``, so the NAME reaches
#: the marker and prep can rebuild a config by scanning.


def begin_marker(name: str) -> str:
    """Return the literal BEGIN marker line for a reserved block."""
    return f"# === molbuilder {name} BEGIN ==="


def end_marker(name: str) -> str:
    """Return the literal END marker line for a reserved block."""
    return f"# === molbuilder {name} END ==="


# Regex matching either marker for any block.  Group 1: block name;
# group 2: BEGIN | END.
#
# The name is one lowercase word (``header``, ``bench-marks``) OR two words
# (``item mesh_cutoff``) -- the second form is job-contracts.md § 3.7's item
# block, whose marker carries the FIELD's name.  That is what lets prep walk
# a template and rebuild a config without an .fdf parser, so the name has to
# reach the marker; underscores are allowed there because field names have
# them.  Every consumer already filters on ``group(1) != BLOCK_<x>``, so
# widening the name pattern cannot make an item block look like a reserved
# one (checked across all six consumers, 2026-08-07).
MARKER_RE = re.compile(
    r"^#\s*===\s+molbuilder\s+([a-z-]+(?:\s+[A-Za-z0-9_]+)?)"
    r"\s+(BEGIN|END)\s+===\s*$"
)


# --------------------------------------------------------------------- #
#  BENCH-MARKS field declarations                                       #
# --------------------------------------------------------------------- #


#: The declaration line's ``type`` vocabulary.
#:
#: § 3.3 defined five for BENCH-MARKS, whose override surface is numeric.
#: § 3.7 reuses the same grammar for a TEMPLATE's item blocks -- and a config
#: is wider than a benchmark's knobs, so three names were missing.  Measured
#: against ``SiestaConfig``'s 39 exposed fields on 2026-08-07: 7 booleans, 1
#: integer triple (``kgrid``) and 1 optional boolean had no type at all.
#:
#: ``pow2`` stays BENCH-MARKS-only: it is a constraint a benchmark puts on an
#: override, not a type any config field has.
DECL_TYPES = ("int", "float", "str", "bool", "int3", "enum", "pow2")


@dataclass(frozen=True)
class BenchField:
    """Declaration of one overridable parameter — `job-contracts.md § 3.3`.

    Per the contract: tools may override this field; the anchor locates the
    override site in ENGINE BODY by greping for the start of a code line.

    **The same shape serves a template's item blocks** (§ 3.7), which is why
    the last three fields exist. § 3.7 is explicit that its declaration is
    *"the grammar § 3.3 already defines … extended with ``group=`` and
    ``choices=``. Not a parallel notation: the same shape, in the same file,
    parsed the same way."* So it is one class, not two.

    ``optional`` marks a field whose **unset** state is real and distinct from
    every value it could hold (``Optional[int]`` and friends — 11 of them on
    ``SiestaConfig``). Without it a reader cannot tell "the user left this
    alone" from "the user chose the default", and those mean different things
    to an engine that omits the line entirely for the first.
    """
    name: str                                 # human-readable label
    anchor: str                               # what a tool greps for in engine body
    type_: str                                # one of DECL_TYPES
    range_: Optional[Tuple[float, float]] = None
    unit: Optional[str] = None
    group: Optional[str] = None               # workflow_group (§ 3.7)
    choices: Optional[Tuple[str, ...]] = None  # the enum's members (§ 3.7)
    optional: bool = False                    # unset is a distinct state

    def __post_init__(self) -> None:
        if self.type_ not in DECL_TYPES:
            raise ValueError(
                f"field {self.name!r}: type {self.type_!r} is not one of "
                f"{', '.join(DECL_TYPES)} (job-contracts.md 3.3). A field "
                "whose type has no name cannot be read back, so the type is "
                "added to the grammar rather than left off the declaration.")
        # NOT checked here: ``type=enum`` with no ``choices``.  § 3.7 adds
        # ``choices=`` and an item block needs it -- a reader cannot validate
        # an enum whose members it was not told -- but § 3.3's BENCH-MARKS
        # shipped without it, and ``SIESTA_BENCH_FIELDS``' own
        # ``Diag.Algorithm`` is exactly that case.  So the rule is enforced
        # where the contract states it (the item-block emitter) rather than
        # here, where it would refuse a block that ships today.
        #
        # ⚠ That shipped block IS thin: a bench tool reading it learns the
        # field is an enum and not which values are legal.  Recorded
        # 2026-08-07 rather than fixed, because changing an emitted artifact
        # is not this unit's business.


# Static field list for SIESTA .fdf.  PySCF and future engines get
# their own list when their bench subcommands land.
#
# Anchor (post-2026-06-23 SIESTA keyword fix): ONE step-count keyword serves
# CG / Broyden / FIRE -- the per-type aliases ``MD.NumBroydenSteps`` /
# ``MD.NumFIRESteps`` do not exist and were silently dropped pre-fix.  So the
# bench anchor is the same regardless of cfg.relax_type, so a trial's deck
# needs no per-type dispatch.  (This named ``molbuilder bench siesta-gpu``
# until 2026-08-17; that verb was deleted 2026-08-13 and its group with it --
# a trial is rendered by ``jobset prep bench``.)  See
# decision-log 2026-06-23 in design.md.  Task #486 is closed by
# this realization.
#
# The keyword is ``MD.Steps`` since 2026-08-15.  The 5.4.2 manual marks
# ``MD.NumCGsteps`` deprecated (``\fdfdeprecates``, Docs/tex/sections/
# Relaxation_phonons_md/Structural_relaxation.tex) and keeps it only "for
# historical reasons".  A BENCH-MARKS block written before that date names
# the old one; ``parse/dirs/job.py`` reads either.
SIESTA_BENCH_FIELDS: List[BenchField] = [
    # NO range here, and that is the declaration.  BlockSize is the one field
    # in this list derived from a LAUNCH quantity (``engines/stages.md``
    # § 5.2), so its legal window is a fact about one deck's rank count, not
    # about the engine -- ``siesta/input.py`` supplies it per deck through
    # ``_block_size_bounds``.  It carried ``(16, 256)`` until 2026-08-10, a
    # constant that disagreed with the emitted default routinely rather than
    # exceptionally (under the ATOMS-era derivation of the day,
    # ``_auto_block_size(200, mpi_np=16)`` was 8; U18's orbital derivation
    # gives 64 -- the history keeps the old number because it is what
    # motivated the fix), so the block declared its own value out of
    # bounds.  Leaving it None means a renderer that forgets emits NO
    # range rather than a wrong one.
    BenchField("BlockSize",        "BlockSize",        "pow2"),
    BenchField("MaxSCFIterations", "MaxSCFIterations", "int"),
    BenchField("MD.Steps",         "MD.Steps",         "int"),
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
    # Through decl_line -- THE one renderer (R11, 2026-08-12: this loop
    # hand-rolled its own dialect, silently dropping choices/optional/
    # group -- a bench tool reading Diag.Algorithm's enum saw no legal
    # values at all, the drift decl_line's own docstring says one
    # renderer exists to prevent).  Column alignment went with the
    # hand-roll; the parser never needed it and § 3.7 says one shape.
    for f in fields:
        out.append(decl_line(
            f, default=(defaults[f.name]
                        if f.name in defaults
                        and defaults[f.name] is not None else None)))
    out.append(end_marker(BLOCK_BENCH_MARKS))
    return "\n".join(out)


def decl_line(f: "BenchField", *, value=None, default=None,
              indent: str = "#   ") -> str:
    """Render one ``field …`` declaration line — `job-contracts.md § 3.3`/§ 3.7.

    **One renderer, so BENCH-MARKS and a template's item blocks cannot drift
    into two dialects** — § 3.7 is explicit that its declaration is not a
    parallel notation but the same shape, in the same file, parsed the same way.

    ``value`` is the item's current value and ``default`` what it would be
    untouched. Both are rendered when given: the pair is what tells a surface
    whether the user set this field or left it alone, without a second marker
    saying so. ``value`` is also **what the reader reads** — never the payload,
    which may be absent, several lines, or a ``%block`` (§ 3.7 property 2).
    """
    line = f"{indent}field {f.name}  anchor={f.anchor}  type={f.type_}"
    if f.range_ is not None:
        line += f"  range=[{f.range_[0]},{f.range_[1]}]"
    if f.unit:
        line += f"  unit={f.unit}"
    if f.choices:
        line += f"  choices={'|'.join(f.choices)}"
    if f.optional:
        line += "  optional=true"
    if f.group:
        line += f"  group={f.group}"
    if default is not None:
        line += f"  default={_scalar(default)}"
    if value is not None:
        line += f"  value={_scalar(value)}"
    return line


def _scalar(v) -> str:
    """A value as one unambiguous token — no spaces, so the declaration line
    stays splittable on whitespace by the reader."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(_scalar(x) for x in v)
    return str(v)


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
    out.extend(f"# {line}" for line in _compact_json_lines(payload))
    out.append(end_marker(BLOCK_ATOM_METADATA))
    return "\n".join(out)


def _compact_json_lines(payload: Dict[str, Any]) -> List[str]:
    """The payload as JSON, ONE LINE PER TOP-LEVEL KEY.

    WHY NOT ``indent=2``.  Indented JSON puts every array element on its own
    line, and these arrays are ATOM INDEX LISTS.  A junction with 100 atoms in
    each electrode and 50 frozen produced a ~280-line comment block, and the
    whole of it sat between the reader and the physics.  A generated input is
    read by scientists checking the science; the machine record should not be
    the first three screens of it.

    WHY NOT ONE LONG LINE EITHER.  Fully compact JSON is one unbroken ~1500
    character line: small, and unreadable and undiffable.  Per top-level key is
    the middle -- eight or nine lines, each naming what it holds, and a changed
    region shows up as a one-line diff instead of a moved block.

    WHY IT WRAPS ONLY AT KEYS.  The reader rejoins these lines and parses the
    result, so a break may never fall inside a string literal -- wrapping at a
    fixed column could split a hash and produce JSON that cannot be read back.
    Breaking only between top-level keys cannot: a key boundary is always
    outside a string.
    """
    keys = list(payload)
    lines: List[str] = ["{"]
    for i, key in enumerate(keys):
        value = json.dumps(payload[key], separators=(",", ":"),
                           ensure_ascii=False)
        comma = "," if i < len(keys) - 1 else ""
        lines.append(f'  {json.dumps(key, ensure_ascii=False)}:{value}{comma}')
    lines.append("}")
    return lines


def machine_record_banner() -> str:
    """The line between the science and molbuilder's own record.

    Everything above it is the calculation: a scientist reads it, edits it, and
    it is the reason the file exists.  Everything below is how molbuilder reads
    the file BACK -- provenance, the benchmarking anchors, and the per-atom
    labels that reconstruct the structure (parse/scripts).  Those are data, not
    settings, and hand-editing them does not change the calculation; it makes
    the file unreadable to the tool that wrote it.

    So it is marked, loudly, and placed at the END: a generated input opens on
    the physics now instead of on three screens of index lists.
    """
    rule = "# " + "=" * 70
    return "\n".join([
        rule,
        "#  MOLBUILDER RECORD -- everything below this line is written and read",
        "#  by molbuilder, and is NOT part of the calculation.",
        "#",
        "#  Do not hand-edit it.  Each section is fenced by its own",
        "#  '=== molbuilder <name> BEGIN/END ===' markers; edits inside those",
        "#  fences are either overwritten on the next generation or make the",
        "#  file unreadable, and neither is announced.",
        "#",
        "#  To change what is here, change the structure and generate again.",
        rule,
    ])


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


def apply_inbody_atom_metadata(struct: Any, text: str, *,
                               notices: Optional[list] = None) -> bool:
    """If ``text`` carries an ATOM-METADATA block, apply its labels to
    ``struct``.

    ``struct`` is duck-typed: any object with a mutable ``regions`` dict will
    do.  Mirrors the protocol :func:`molbuilder.sidecars.molstruct.
    apply_to_structure` uses for the sidecar, minus the structure_hash check
    (metadata and coordinates are written by the same generator pass and cannot
    drift apart by construction).

    Returns ``True`` when labels were applied, ``False`` otherwise (no block,
    or a block carrying nothing).

    THE VERSION LINE IS READ NOW (2026-08-03), and until then it was not.  The
    block states the version that wrote it -- and this reader took the contents
    at face value regardless, so a block in an older layout was applied, its
    frozen atoms silently dropped, and a run came back with nothing frozen and
    nothing said.  That is how the label store's move lost 50 and 216-atom
    frozen sets out of real run directories.

    On a version it does not recognise this WARNS and TRANSLATES rather than
    refusing (user, 2026-08-03).  Refusing would make a finished run
    unopenable, and the point of these notes is that a run directory explains
    itself.  So: say what was found, convert what can be converted, and let the
    user see both.

    ``notices`` collects ``{level, message, where, about}`` dicts for the
    caller to surface.  A finding never travels as ``warnings.warn`` -- that
    reaches server stderr and no web user at all (delivery contract R5,
    science/validation.md § 4.1), which is the same mistake in a different
    place.
    """
    # Local import -- avoids a circular import via parse.scripts.
    from molbuilder.parse.scripts.atom_metadata import (
        _extract_atom_metadata_dict,
    )
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION
    from molbuilder.structure import FROZEN_LABEL

    payload = _extract_atom_metadata_dict(text)
    if payload is None:
        return False

    regions = dict(payload.get("regions") or {})
    annotations = payload.get("annotations") or {}

    said = payload.get("schema_version")
    if said != SCHEMA_VERSION:
        # THE ONE TRANSLATION WORTH DOING: frozen atoms used to be a key of
        # their own, beside `regions`.  They are an ordinary label inside it
        # now, so an old block's `frozen_atoms` is moved in.  An existing
        # in-regions entry wins -- it is the current shape and the newer truth.
        moved = payload.get("frozen_atoms")
        recovered = 0
        if isinstance(moved, list) and FROZEN_LABEL not in regions:
            regions[FROZEN_LABEL] = list(moved)
            recovered = len(moved)
        if notices is not None:
            detail = (f"; recovered {recovered} frozen atom(s) from the old "
                      f"layout" if recovered else
                      "; nothing needed moving")
            notices.append({
                "level": "warn",
                "message": (
                    f"These atom labels were written by an older molbuilder "
                    f"(the notes say version {said!r}; this build writes "
                    f"{SCHEMA_VERSION}){detail}. Check the labels are what you "
                    f"expect before running anything from them, and re-save "
                    f"the structure to store them in the current form."),
                "where": "labels.atom_metadata_version",
                "about": "labels",
            })

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
        # Extensible channels -> struct.annotations, same round-trip as the
        # sidecar (§ 3 data-model persistence).
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
    "DECL_TYPES", "decl_line", "deck_note",
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


# --------------------------------------------------------------------- #
#  deck_note -- an item's note comes from the CATALOGUE, for EVERY engine #
#                                                                        #
#  `engines/template.md` § 1.0 says the template exists because an engine #
#  input "cannot be read without knowing the engine -- there is nowhere   #
#  in the file to say what it is, what it is measured in, or what a       #
#  sensible value looks like."  Both emitters answered that by writing    #
#  their OWN prose beside each keyword and never consulting the           #
#  catalogue: 392 of a 485-line SIESTA deck were hand-written comments,   #
#  and `pyscf/input.py` carries 179 more.                                #
#                                                                        #
#  Two homes for one explanation drift, and had (found 2026-08-17 by      #
#  reading a generated deck): the comment called 0.02 "typical            #
#  production" while the deck emitted 0.01; it said "3 fine for most      #
#  cases" while shipping 8; and the catalogue stated the § 5.2 DEVIATION  #
#  for `SCF.Mixer.Weight` while the deck -- the file a scientist opens    #
#  before a week of compute -- did not.                                   #
#                                                                        #
#  It lives HERE, not in one engine, because § 2 of `engines/overview.md` #
#  makes this module the shared script-contract wrapper both emitters     #
#  call, and because the note is per-ITEM: an item may serve one engine   #
#  or several, so the lookup takes the ENGINE and asks the one read API   #
#  (`template.one`), which answers None for "not for this engine" and     #
#  raises for "no such item" -- two different answers that must not read  #
#  the same.                                                             #
# --------------------------------------------------------------------- #



def _catalogue():
    """The parsed catalogue, through its one door.

    This kept a module-level cache of its own until 2026-08-18 -- correct, and
    a second answer to *"where do I get the catalogue?"*  ``template.catalogue``
    is that answer now, and it caches for everybody.
    """
    from . import template as _T
    return _T.catalogue()


def deck_note(item_name, engine, *lead, extra=()):
    """One item's catalogue note as deck comment lines: ``["", "# ...", ...]``.

    *engine* is required and is not decoration: an item may declare
    ``engines`` and a note pulled without it would put another engine's
    guidance in this deck.  An item that does not apply here yields NO
    comment rather than a wrong one.

    *lead* is an optional first line naming the keyword, for a reader
    scanning the file.  *extra* carries lines that are the EMITTER's own --
    how THIS deck wires the value -- which do not belong in a catalogue
    whose help is engine-agnostic and is also shown on a form.

    Prose is re-flowed to the deck's width; an INDENTED line is copied
    verbatim with its relative indent, because several items carry a
    hand-aligned tier ladder and re-flowing that destroys the alignment
    that makes it a table.  One source line is one paragraph: the
    catalogue writes help with a hard newline between thoughts.
    """
    import textwrap
    from . import template as _T
    try:
        it = _T.one(_catalogue(), item_name, engine=engine)
    except KeyError:
        it = None
    out = [""]
    for ln in lead:
        out.append("# %s" % ln)
    if it is not None and (it.help or "").strip():
        para = []

        def _flush():
            if para:
                for ln in textwrap.wrap(" ".join(para), width=66):
                    out.append(("# %s" % ln).rstrip())
                para.clear()

        for raw in it.help.strip().splitlines():
            if not raw.strip():
                _flush(); out.append("#")
            elif raw[:1].isspace():
                # An indented line is a LADDER ROW.  Its own indent is kept so
                # the column survives, and a long row is wrapped with a hanging
                # indent rather than left at 180 characters -- verbatim was
                # right for "150   screening" and wrong for a bullet carrying a
                # sentence.
                _flush()
                lead = raw[: len(raw) - len(raw.lstrip())]
                for w in textwrap.wrap(raw.strip(), width=66,
                                       initial_indent=lead,
                                       subsequent_indent=lead + "    "):
                    out.append(("# %s" % w).rstrip())
            else:
                para.append(raw.strip()); _flush()
        _flush()
    out.extend(("# %s" % ln).rstrip() for ln in extra)
    return out if len(out) > 1 else []


# --------------------------------------------------------------------- #
#  § 8.0's read API, for a GENERATOR — one door per parameter           #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Parameter:
    """What a script writer may ask about ONE parameter of the calculation
    it is writing.

    **Why this exists, and what it replaces.** A generator legitimately owns
    its own logic — which parameters to write, in what order, with what
    checks. What it must never own is the *information*: what a parameter
    declares, what this calculation resolved it to, and which engine keywords
    it therefore writes. Before this object each writer got that information
    its own way, and the ways did not agree:

    * ``siesta/input.py`` asked the catalogue for HELP (``deck_note``, 22
      sites) and hand-kept everything else;
    * ``runwrap.py`` asked nothing — it re-parsed the deck with awk for the
      two facts it could cheaply re-parse, and asserted the rest from string
      literals. Every re-parsed fact stayed true and every asserted one went
      stale, five copies of one claim among them;
    * ``pyscf/input.py`` asked nothing at all.

    And the fact itself had three homes: ``[item.restart].expands``,
    ``SIESTA_RESTART_GROUP.keys`` and ``warm-files.toml``'s ``honoured_by``
    rows — the same three keywords in three different orders, with nothing
    comparing them. *"Pull it from the source you know"* is not a thing a
    writer can do when there are three sources.

    So: one object, one question per attribute, and the catalogue is the
    declaration behind all of them.
    """

    #: The item's name in the catalogue (``restart``, ``mesh_cutoff``).
    name: str
    #: Which engine is asking — an item that does not apply to it answers
    #: EMPTY rather than handing over another engine's guidance.
    engine: str
    #: The catalogue's declaration, or ``None`` when this engine has no such
    #: item. Callers test :attr:`known` rather than this.
    declaration: Any = None
    #: What THIS calculation resolved it to, when the caller supplied a
    #: config or a rendered deck to read it from; ``None`` when unknown.
    value: Any = None

    @property
    def known(self) -> bool:
        """Whether the catalogue declares this item for this engine."""
        return self.declaration is not None

    @property
    def writes(self) -> Tuple[str, ...]:
        """The engine keywords this parameter puts in the deck.

        ``expands`` when it writes several (``restart`` →
        ``DM.UseSaveDM`` / ``MD.UseSaveXV`` / ``MD.UseSaveCG``), the single
        ``anchor`` when it writes one, and empty when it writes none — a
        wrapper-only knob, or one whose effect is generated control flow.

        **This is the declaration that had three copies.** A generator asks
        here and there is nothing left to keep in step.
        """
        if not self.known:
            return ()
        if self.declaration.expands:
            return tuple(self.declaration.expands)
        return (self.declaration.anchor,) if self.declaration.anchor else ()

    @property
    def default(self):
        """The catalogue's recommended value — what the item declares, never
        what this calculation chose. :attr:`value` is that."""
        return self.declaration.default if self.known else None

    def note(self, *lead, extra=()) -> List[str]:
        """This parameter's own note, as deck comment lines.

        The same rendering :func:`deck_note` has always done — kept as one
        implementation, reached now through the object rather than by name.
        """
        return deck_note(self.name, self.engine, *lead, extra=extra)


_UNSET = object()


def declarations(engine: str = None):
    """Every item the catalogue declares, for one engine — the LIST door.

    :func:`parameter` answers about one item by name; this answers *which items
    are there*, which is the question a record of the whole configuration asks.

    **It exists so that nothing outside this module loads the catalogue.**  Two
    callers reached through ``_catalogue()`` -- a private -- into
    ``template.select`` to get this, which meant the catalogue OBJECT travelled
    out of the one module that owns reading it.  A caller that holds the object
    can ask it anything, including things the read API deliberately does not
    offer, and that is how a second way of reading a declaration starts.
    """
    from . import template as _T
    return list(_T.select(_catalogue(), engine=engine))


def parameter(name: str, engine: str, *, config=None,
              deck_text: str = None, value=_UNSET) -> "Parameter":
    """The ONE door a script writer opens to ask about a parameter.

    ``config`` is the resolved engine config — the answer a writer has BEFORE
    it renders. ``deck_text`` is a rendered deck — the answer a writer has
    AFTER, and the one the *wrapper* generator must use, because the wrapper
    ships beside a finished deck and the deck is what the engine obeys. Both
    resolve through the same declaration, so the two moments cannot disagree
    about what a parameter is; they can only differ about what it says, which
    is the real question and the one worth being able to ask.

    Neither given, the Parameter still answers :attr:`writes`, :attr:`default`
    and :meth:`note` — the declaration alone.
    """
    from . import template as _T
    try:
        decl = _T.one(_catalogue(), name, engine=engine)
    except KeyError:
        decl = None
    if value is not _UNSET:
        # A DERIVED value -- one the engine worked out rather than one a field
        # answers.  A block size computed from this deck's rank count is nobody's
        # config field, and without this door every such value would fall to the
        # engine's free-form body, where the note-with-the-value rule cannot
        # reach it (script-preparation.md 4.2a).  The declaration, the range and
        # the note still come from the catalogue; only the number is the
        # caller's.
        return Parameter(name=name, engine=engine, declaration=decl, value=value)
    resolved = None
    if decl is not None and config is not None:
        resolved = getattr(config, name, None)
    elif decl is not None and deck_text is not None:
        resolved = _deck_answer(decl, deck_text)
    return Parameter(name=name, engine=engine, declaration=decl, value=resolved)


def _deck_answer(decl, deck_text: str):
    """What a RENDERED deck says for this item, or ``None``.

    Reads the first occurrence of the item's own keyword — libfdf's rule, and
    the rule ``jobset/summarize.deck_value`` already documents: ``fdf_locate``
    walks from the first line and stops at the first label that matches, so a
    deck naming a keyword twice is read with its FIRST value.

    An item that expands to several keywords answers with the FIRST of them:
    they are one field's expansion and a deck that disagreed with itself
    across them would be a defect of the writer, not a state to model.
    """
    keys = (tuple(decl.expands) if decl.expands
            else ((decl.anchor,) if decl.anchor else ()))
    if not keys:
        return None
    want = keys[0].lower().replace(".", "").replace("_", "").replace("-", "")
    for line in deck_text.splitlines():
        toks = line.split("#", 1)[0].split()
        if len(toks) >= 2 and toks[0].lower().replace(".", "").replace(
                "_", "").replace("-", "") == want:
            return toks[1]
    return None


def write_script(path, text: str) -> "Path":
    """Write a generated script, KEEPING the reader's own USER-CUSTOM block.

    **Every writer of a generated script goes through here.**  The deck says,
    in its own words, *"Your own additions go here.  molbuilder will preserve
    this section verbatim across regenerations."*  That promise had exactly one
    keeper until 2026-08-18 — the web file-editor's save route — so a person
    who added ``WriteMullikenPop`` or a ``%block BandLines`` (the very things
    the deck's own post-processing section invites) lost them at the next
    ``prep``, silently, to a file that had promised otherwise.

    The merge itself is :func:`merge_user_custom_from_target` and is unchanged;
    what was missing was a door the generators actually open.  It is safe in
    every degenerate case — no target, no block on either side, unreadable
    target — so a first write behaves exactly like ``write_text``.

    This is the third mechanism in this module to have been present and
    uncalled: :func:`deck_note` (the deck writer uses it, the wrapper and the
    PySCF writer do not) and ``StructureCodec`` (``prep`` and the web route use
    it, the single-shot converters did not) were the others.  A shared writer
    only shares what its callers ask it for.
    """
    from pathlib import Path as _P
    p = _P(path)
    p.write_text(merge_user_custom_from_target(text, p), encoding="utf-8")
    return p


# --------------------------------------------------------------------- #
#  The step-3 runner — `execution/script-preparation.md` § 3 and § 4.2   #
#                                                                       #
#  The framework owns the ORDER of the sub-steps and the four that are   #
#  the same for every engine (validate · reader section · record ·       #
#  write) plus the frame of the fifth (check).  An engine answers three  #
#  doors and never sees a bare value to write, which is what makes       #
#  "a value is written with its reason" structural rather than a habit.  #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Section:
    """One titled run of parameters in a deck — the **layout, as data**.

    ``items`` are CATALOGUE NAMES, not engine keywords: the framework turns
    each into a :class:`Parameter` through :func:`parameter`, so the engine is
    handed the declaration and cannot emit the value without its note.

    Deck layout is engine knowledge and stays with the engine — but as a table
    rather than as control flow.  The catalogue's own ``group`` cannot serve
    here: its vocabulary is the FORM's (setup / stage / profile / budget /
    staging / output) and one group cuts across several deck sections --
    ``stage`` alone holds ``PAO.BasisSize``, ``MeshCutoff`` and
    ``DM.Tolerance``, which land in three different places in a deck.
    """
    title: str
    items: Tuple[str, ...]
    #: Lines that sit BETWEEN the heading and the values — the section's own
    #: explanation, in the engine's comment syntax.
    #:
    #: **It is here because that is where it goes in the deck**, and a walk
    #: that emitted heading-then-values had nowhere to put it.  An engine with
    #: an explanation used to write the heading AND the prose itself and then
    #: suppress the framework's heading — which meant its sections could not
    #: share one spec, and cost the section its NAME in the layout, since a
    #: falsy title was the only way to ask for silence.  A reader of the layout
    #: then could not tell what the section was without going to find the
    #: writer.
    #:
    #: Dropped with the notes when ``verbose`` is off: it is explanation, and
    #: that is what the quiet deck leaves out.
    note: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Block:
    """A part of a deck that no parameter models — **in layout order**.

    A cell, a coordinate table, a run loop, a post-processing template: things
    whose text the engine writes whole.  They are members of the layout rather
    than a separate door because **a deck interleaves them with its
    parameters** — SIESTA's cell sits between the exchange-correlation
    settings and the SCF ones, and a door appended after the sections could
    not put it there.

    ``render`` is ``(struct, cfg) -> str | None``, and ``None`` means *nothing
    to say for this configuration* — the same answer :meth:`DeckSpec.line`
    gives for a parameter, so conditionality is one idea in this framework and
    not two.  ``title`` names the block for a reader of the layout; a block
    writes its own headings, because what a free-form block looks like is the
    thing it is free about.
    """
    title: str
    render: Callable


@dataclass(frozen=True)
class DeckSpec:
    """Everything one engine supplies to the runner — § 4.2's TWO doors, and
    the values only this engine can put in the record.

    Everything NOT here is the framework's, and an engine that needs a third
    door is evidence the seam is wrong rather than a reason to add a callable.

    **Nine slots and two doors is not a contradiction.**  ``layout`` is door 1
    and ``line`` is door 2; ``note_lead`` and ``section_title`` are how this
    engine writes a note and a heading in its own syntax; ``provenance_defaults``
    and ``bench_marks`` are VALUES for blocks the framework assembles;
    ``check_rules`` is what a finished deck of this engine must satisfy; and
    ``engine`` / ``created_by`` are names.  Only the first two are asked at
    every item.

    **A form, not a function** (§ 4.3).  A function can only be called; a form
    can be READ -- which is what lets the framework work out what the deck was
    supposed to contain and compare that with the file, without the writer
    passing it a list to be believed.
    """
    #: Which engine's catalogue rows and notes to read.
    engine: str
    #: Door 1 — the deck's layout in order: :class:`Section` for parameters,
    #: :class:`Block` for the text no parameter models.  One ordered table,
    #: because a deck interleaves the two and an appended door could not.
    #:
    #: **Its MEMBERSHIP is settled here, when the spec is built** (§ 4.2, W9).
    #: A section that only some calculations have is left OUT for the others --
    #: ``spec_for`` holds ``(struct, cfg)`` and can answer that.  A section
    #: chosen inside a :class:`Block` instead is a section this table cannot
    #: name and :func:`render_deck` cannot collect from, and then the check
    #: gate has nothing to compare the file against.  Both engines did exactly
    #: that until 2026-08-19: SIESTA's whole deck was one ``Block``, so a
    #: 728-line file reported zero written keywords.
    layout: Tuple[Any, ...]
    #: Door 2 — ``(Parameter) -> str | None``.  ``None`` means *not emitted
    #: for this configuration*, and that is the whole conditionality
    #: mechanism: ``MD.Steps`` under ``relax_type="none"`` simply returns None.
    line: Callable
    #: ``(cfg) -> {field: description}`` for the PROVENANCE block's
    #: resolved-defaults rows.  The block itself is the framework's.
    provenance_defaults: Optional[Callable] = None
    #: ``(struct, cfg) -> dict`` of :func:`emit_bench_marks` keyword
    #: arguments, or ``None`` for an engine that declares no anchors.
    bench_marks: Optional[Callable] = None
    #: ``(text, struct, cfg) -> [Issue]`` — this engine's answer to *what must
    #: a finished deck of mine satisfy?*  Runs on the file as written.
    check_rules: Optional[Callable] = None
    #: How a section title is written as a comment.  **Neither shipped
    #: engine overrides this** -- both write `#` comments -- and that is
    #: worth knowing rather than assuming: the slot is unexercised, kept
    #: because the comment character is an engine's syntax and the next
    #: engine may not spell it this way.  Both restated the default
    #: verbatim until 2026-08-19, which made it look like a variation.
    section_title: Callable = lambda title: f"# --- {title} ---"
    #: ``(Parameter) -> tuple[str, ...]`` — lines to head this parameter's note
    #: with.  SIESTA heads each with the keyword it writes, because its notes
    #: are long and the keyword otherwise arrives after them; PySCF's line sits
    #: right below a short note and needs no signpost.  Formatting is the
    #: engine's, so the choice is too.
    note_lead: Callable = lambda param: ()
    #: Recorded in ATOM-METADATA as the producer.
    created_by: str = "molbuilder"


class RenderedDeck(str):
    """A deck's text, carrying what the parameters step says it emitted.

    ``emitted`` is what lets the **check** gate close its loop — the engine
    keywords the parameters step actually wrote, so the check can ask whether
    each one survived into the file rather than trusting that it did.  Without
    it that rule has no input and passes silently, which is what it did on
    every production route until 2026-08-18.

    **It IS the text**, a ``str`` subclass rather than a wrapper around one.
    The seam says a deck writer returns deck text (§ 4), three production
    routes and twenty test files already have a string in hand, and changing
    that return type to carry one extra tuple would be a rewrite of the test
    suite wearing a migration's clothes (`archive/2026-08-18-preparation-backend-plan.md`
    § 3.1a).  A caller that wants the text uses it as text; a caller that wants
    to close the loop reads ``.emitted``.
    """
    emitted: Tuple[str, ...]

    def __new__(cls, text: str, emitted=()) -> "RenderedDeck":
        self = super().__new__(cls, text)
        self.emitted = tuple(emitted)
        return self

    @property
    def text(self) -> str:
        """The deck, as a plain ``str``.  Named because the check gate and its
        tests read it that way, and because ``str(deck)`` at a call site reads
        as a conversion rather than as *the deck's text*."""
        return str(self)


def _render_sections(spec: "DeckSpec", cfg, *, verbose: bool = True
                    ) -> Tuple[List[str], List[str]]:
    """The **parameters** sub-step: walk the layout, one Parameter at a time.

    Returns ``(lines, emitted)``.  A section whose parameters all decline to
    emit contributes no title either -- a heading over nothing is the block
    lying, the same rule the BENCH-MARKS defaults row follows.

    :class:`Block` members are SKIPPED here: they are text, and placing them
    in order is :func:`render_deck`'s job.

    **PRIVATE since 2026-08-19, and that is the point.**  Both engines called
    it -- nine times in one SIESTA deck, once in a PySCF one -- each passing a
    one-section spec built with ``dataclasses.replace``.  So the sections were
    rendered from INSIDE a block, where the layout could not name them and
    :func:`render_deck` could not collect what they wrote: SIESTA's deck
    reported zero written keywords for a 728-line file, and the check gate's
    loop-closing rule ran on an empty list and passed.  There is one walk now,
    and no door for an engine to start a second one
    (`script-preparation.md` § 4.1).
    """
    out: List[str] = []
    emitted: List[str] = []
    for section in spec.layout:
        if isinstance(section, Block):
            continue          # a Block is text, not parameters -- render_deck
        body: List[str] = []
        for name in section.items:
            param = parameter(name, spec.engine, config=cfg)
            text = spec.line(param)
            if text is None:
                continue
            if verbose:
                body.extend(param.note(*spec.note_lead(param)))
            body.append(text)
            # WHICH KEYWORD THIS PARAMETER COMMITTED THE DECK TO.
            #
            # An item may declare several and choose between them:
            # ``relax_steps`` declares ``MD.Steps`` and ``MD.FinalTimeStep``
            # and writes whichever the run mode calls for, never both.  Only
            # the TEXT knows which branch was taken, so where the line names
            # one of them, that is the answer.
            #
            # **Where it names none, the single declared keyword is still the
            # answer** *(2026-08-19)*.  A deck that is a PROGRAM binds the
            # value here and hands it to the engine further down --
            # ``_GEOM_ETOL = 1e-06`` at the top, ``convergence_energy =
            # _GEOM_ETOL`` at the ``optimize()`` call -- so the keyword is in
            # the FILE, which is what the gate reads, and not in this line.
            # Asking the line alone made every PySCF geometry target invisible
            # to the gate.  With one candidate there is nothing to resolve; the
            # parameter emitted, so the deck claims that keyword and the file
            # has to show it.  With several and no match the line has told us
            # nothing, and inventing an answer is what the gate exists to stop.
            chosen = [k for k in param.writes if _mentions_keyword(text, k)]
            emitted.extend(
                chosen if chosen
                else param.writes if len(param.writes) == 1
                else ())
        if body:
            out.append("")
            # A SECTION WITH NO TITLE GETS NO HEADING, and the engine is not
            # asked to spell one.  This called ``section_title("")`` and tested
            # the result, so an engine whose heading it writes itself had to
            # pass a suppressing ``section_title`` -- which meant its OTHER
            # sections could not share the spec, which is why one deck needed
            # eight of them.
            title = spec.section_title(section.title) if section.title else ""
            # A falsy title means the caller has already written its own
            # heading -- a section whose explanation must sit between the
            # heading and the values, which the walk has no way to interleave.
            if title:
                out.append(title)
            # The section's own explanation, between the heading and the
            # values — where a reader of the deck needs it, and where a walk
            # that only knew headings and values could not put it.
            if verbose:
                out.extend(section.note)
            out.extend(body)
    return out, emitted


def render_deck(spec: "DeckSpec", struct, cfg, *, verbose: bool = True
                ) -> "RenderedDeck":
    """Sub-steps **structure** through **record**, in that order.

    **The layout is walked in order**, and a member of it is either a
    :class:`Section` — parameters, through the engine's ``line`` — or a
    :class:`Block`, whose text the engine writes whole.  That is the deck's own
    shape: a cell sits between two runs of settings, a run loop after a third,
    and a framework that appended its free-form parts after the sections could
    describe neither engine's deck.

    The engine is asked for its layout and its syntax and nothing else; the
    reader's section, the record blocks and the banner are the framework's, so
    two engines cannot drift about what a generated file looks like below the
    science.
    """
    parts: List[str] = []
    emitted: List[str] = []
    for member in spec.layout:
        if isinstance(member, Block):
            text = member.render(struct, cfg)
            # ``None`` is *nothing to say*; ``""`` is a blank line the block
            # meant to write.  Testing truthiness conflates them, and a block
            # whose whole content is one blank line joins to "" -- so the
            # separator between two runs of settings silently disappeared.
            if text is not None:
                parts.append(text)
            continue
        lines, names = _render_sections(
            _dataclasses.replace(spec, layout=(member,)), cfg, verbose=verbose)
        parts.extend(lines)
        emitted.extend(names)
    science = "\n".join(parts)

    record: List[str] = [emit_provenance(
        generator_version=molbuilder_git_sha(),
        generated_at=generated_at_now(),
        resolved_defaults=(spec.provenance_defaults(cfg)
                           if spec.provenance_defaults else None))]
    if spec.bench_marks is not None:
        marks = spec.bench_marks(struct, cfg)
        if marks:
            record.append(emit_bench_marks(**marks))
    atoms = emit_atom_metadata(
        regions=dict(getattr(struct, "regions", {}) or {}),
        annotations=dict(getattr(struct, "annotations", {}) or {}),
        n_atoms_total=int(getattr(struct, "n_atoms", 0)),
        created_by=spec.created_by,
        created_at=generated_at_now())
    if atoms:
        record.append(atoms)

    text = (science + "\n\n" + emit_user_custom_placeholder()
            + "\n\n" + machine_record_banner()
            + "\n\n" + "\n\n".join(record) + "\n")
    return RenderedDeck(text=text, emitted=tuple(emitted))


# --------------------------------------------------------------------- #
#  The CHECK gate — the file the engine will open                       #
# --------------------------------------------------------------------- #


def check_deck(path, spec: "DeckSpec", rendered: "RenderedDeck",
               struct=None, cfg=None) -> List["Issue"]:
    """Read the WRITTEN file back and report what is wrong with it.

    **The subject is the artifact, and that is what is new.**  Every other
    validator in this tree takes ``(struct, cfg)`` and runs before emission, so
    none of them can see a writer bug: a value that never reached the text, a
    keyword written twice, a generated program that does not parse.  Those are
    exactly the defects that have shipped.

    **The file, not the rendered string.**  ``write_script`` merges the
    reader's own USER-CUSTOM section from whatever was already there, so the
    string handed to the writer is an intermediate and the file is what the
    engine opens.  Checking the intermediate would check something nobody runs.

    Shared rules here; the engine's own are ``spec.check_rules``.  Reported
    through the existing :class:`~molbuilder.issues.Issue` and
    ``validation.report``, so a refusal reads like every other refusal.

    **One door.**  A second entry point, ``check_written``, took the same
    arguments loose rather than as a spec, for a conductor that held an
    engine's rules and a written deck but no ``DeckSpec``.  Both engines build
    one now and `prep` calls :func:`prepare_deck`, so that caller no longer
    exists; it was removed on 2026-08-19 rather than left as a public name with
    one internal caller and an expired reason.
    """
    from pathlib import Path as _P
    from .issues import Issue

    rules, emitted = spec.check_rules, rendered.emitted

    out: List[Issue] = []
    p = _P(path)
    if not p.is_file():
        return [Issue("error", f"the deck was not written: {p}",
                      where="deck.missing")]
    text = p.read_text(encoding="utf-8")

    for block, label in ((BLOCK_USER_CUSTOM, "the section left for a reader"),
                         (BLOCK_PROVENANCE, "the provenance record")):
        n = text.count(begin_marker(block))
        if n != 1:
            out.append(Issue(
                "error",
                f"{label} appears {n} times in {p.name}, expected once",
                where=f"deck.{block}"))

    # THE LOOP CLOSED: every keyword the parameters sub-step says it wrote must
    # be in the file.  Without this the two halves are only related by hope --
    # which is how a deck came to state values nobody had read.
    for key in dict.fromkeys(emitted):
        if not _mentions_keyword(text, key):
            out.append(Issue(
                "error",
                f"{p.name} was written with {key!r} but the file does not "
                f"contain it",
                where="deck.missing_keyword"))

    if rules is not None:
        out.extend(rules(text, struct, cfg) or [])
    return out


def _mentions_keyword(text: str, key: str) -> bool:
    """Whether a deck names this engine keyword, ignoring comments.

    Compared loosely on purpose -- SIESTA's fdf reader itself ignores case and
    the ``.``/``_``/``-`` separators, so a check that insisted on one spelling
    would report a deck the engine reads perfectly well.
    """
    want = key.lower().replace(".", "").replace("_", "").replace("-", "")
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        for tok in code.replace("=", " ").split():
            if tok.lower().replace(".", "").replace("_", "").replace(
                    "-", "") == want:
                return True
    return False


def prepare_deck(spec: "DeckSpec", struct, cfg, path, *,
                 verbose: bool = True):
    """**Validate → render → write → check**, in that order, for one deck.

    The shared spine of `script-preparation.md` § 3's per-deck sub-steps. The
    conductor still owns what surrounds it -- naming the file and stamping the
    identity before, keeping the deck's promises and declaring the job after --
    because those are its business and not an emitter's.

    The two gates are separate on purpose and neither can do the other's job.
    **Validate** reads the resolved config and asks *is this a sound
    calculation?*; it is the existing framework, shared with the form's
    preflight, and nothing upstream of the final deck is re-checked here.
    **Check** reads the written file and asks *does this deck say what it was
    meant to say?* -- a question that can only be asked of an artifact, which is
    why no validator in this tree could ask it before.
    """
    from .validation import report, validate
    report(validate(struct, cfg))                      # 3.3 validate
    rendered = render_deck(spec, struct, cfg, verbose=verbose)
    written = write_script(path, rendered.text)        # 3.10 write
    report(check_deck(written, spec, rendered, struct, cfg))   # 3.11 check
    return written

"""molbuilder.script_emit — the script-contract blocks, BOTH directions.

**One format, one owner.**  This module writes the reserved blocks
(HEADER / PROVENANCE / BENCH-MARKS / ATOM-METADATA / USER-CUSTOM) and
reads them back: the emitters are in the first half, the extractors and
:func:`read_script` in the second.

The read half lived in :mod:`molbuilder.parse.scripts` until 2026-09-05,
wrapped in `TextParser` classes so it could sit in `parse/`'s registry.
That registry exists to answer *"which parser handles this file?"* for
FOREIGN formats; these blocks are molbuilder's own, in a file molbuilder
generated, and every caller already knows which block it wants -- so the
classes were ceremony, and the split forced a circular import that a
lazy-import table had to work around.  `plans/plan.md` § 5d has the
measurement; `execution/job-contracts.md` § 3.1 owns the grammar.

*(The write half was itself absorbed from a retired
:mod:`molbuilder.script_contract` on 2026-06-21.  The name `script_emit`
now understates the module: it emits AND reads.  Renaming it would touch
54 files for no functional gain, so the docstring carries the truth
instead.)*

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
* In-body application — :func:`apply_atom_metadata` (mutates
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
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping,
                    Optional, Tuple)

# The HOOK BOUNDARY (§ 4.6).  A real import, not a TYPE_CHECKING one: this is
# called, not annotated.  `issues` is L1 and imports nothing, so there is no
# cycle to worry about.
from .issues import calling as _calling

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
#:
#: **WHY A SHAPE CANNOT BE DECLARED TO A HARNESS**, keyed by the type and
#: DERIVED from `template.TYPES` rather than re-typed beside it.
#:
#: This was a hand-written tuple until 2026-08-23, and it read as a second
#: vocabulary -- which it never was.  `test_template_declarations` requires a
#: BENCH-MARKS line's type to EQUAL its catalogue item's (bar one listed
#: narrowing), so there has only ever been one vocabulary; this said which
#: members of it a benchmark may be told about.  A permission list wearing a
#: vocabulary's clothes, and it drifted exactly as one: it carried ``bool``
#: and ``int3``, added 2026-08-07 when the template briefly shared this
#: grammar for in-deck item blocks, kept after that sharing ended on
#: 2026-08-11, and declared by no ``field`` line since.
_NOT_BENCHMARKABLE = {
    "int3":    "a shape, not a knob -- a harness has no ordering to sweep",
    "float3":  "same",
    "strlist": "a list has no single value to vary",
    "intlist": "same",
    "text":    "verbatim engine text, copied rather than chosen",
    "bool":    "on/off is a FAMILY of runs, not a knob to optimise -- and the "
               "one live case, `use_gpu`, is the person's choice and never an "
               "override a tool may make (`execution/gpu.md` G2)",
}


def benchmark_declarable_types() -> Tuple[str, ...]:
    """The types a BENCH-MARKS ``field`` line may carry.

    **One vocabulary, narrowed by a stated rule.**  A benchmark turns a knob
    and times the result, so it can be told about a scalar it can vary --
    a number, a bounded number, or a choice from a closed set.  Everything
    `_NOT_BENCHMARKABLE` names is a shape or a family instead, with its reason
    beside it.

    Derived from ``template.TYPES`` at call time (both modules are L2 and
    neither imports the other at module scope; the lazy import keeps it that
    way).  So a type added to the catalogue is a type this rule answers for
    automatically -- and `tests/test_type_vocabulary.py` fails if the answer
    was never decided.

    ⚠ ``str`` survives the rule and is declared by no ``field`` line either.
    A free string has no ordering, so no harness can sweep one -- but
    `job-contracts.md` § 3.3 names it among the five this block accepts, and
    overturning that is a contract change rather than a code one.  Recorded
    here so the next narrowing starts from the document.
    """
    from .template import TYPES
    return tuple(t for t in TYPES if t not in _NOT_BENCHMARKABLE)


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
    type_: str                                # benchmark_declarable_types()
    range_: Optional[Tuple[float, float]] = None
    unit: Optional[str] = None
    group: Optional[str] = None               # workflow_group (§ 3.7)
    choices: Optional[Tuple[str, ...]] = None  # the enum's members (§ 3.7)
    optional: bool = False                    # unset is a distinct state

    def __post_init__(self) -> None:
        _legal = benchmark_declarable_types()
        if self.type_ not in _legal:
            raise ValueError(
                f"field {self.name!r}: type {self.type_!r} is not one of "
                f"{', '.join(_legal)} (job-contracts.md 3.3). A field "
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
                    form_config_hash: Optional[str] = None,
                    engine: Optional[str] = None) -> str:
    """Emit the PROVENANCE block.

    ``resolved_defaults`` is a flat dict of "field -> description";
    e.g. {"BlockSize": "auto -> 256 (10 * 212 atoms / mpi_np)"}.
    Caller decides what to include.  The block has no version tag
    (per the contract — keys are additive and forward-compatible).

    ``engine`` is WHICH ENGINE THIS DECK IS FOR, and it leads the block
    because it is what a reader asks first.  Generation is the only
    moment that answer is certain -- ``DeckSpec.engine`` chose the
    catalogue rows and the layout that produced the body -- and § 3
    exists precisely because the file then gets copied away from
    everything that knew.  Consumers read it back through
    ``running-a-job.md`` § 4.2's resolution order.  Optional so the
    signature stays additive; a caller with nothing to declare emits
    the block exactly as before.
    """
    out: List[str] = [begin_marker(BLOCK_PROVENANCE)]
    if engine:
        out.append(f"#   engine               {engine}")
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

    This is **data-model persistence** (``model/structure-annotations.md`` § 4a) — the
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
        from molbuilder import repo_root as _repo_root   # A11: one owner
        repo_root = _repo_root()
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
#  (The lazy extractor table stood here until 2026-09-05.)              #
# --------------------------------------------------------------------- #
#
#  It re-exported six `_extract_*` functions from `parse/scripts/` under
#  unprefixed names, resolved through a module-level `__getattr__` because
#  -- in its own words -- *"an eager top-level import would deadlock:
#  markers.py re-exports BLOCK_* + MARKER_RE from this module."*
#
#  The deadlock was the split, not the imports.  The readers now live
#  above, beside the emitters and the constants they both need, and are
#  reached through `read_script`.

__all__ = [
    # Block names + markers
    "BLOCK_HEADER", "BLOCK_PROVENANCE", "BLOCK_BENCH_MARKS",
    "BLOCK_ATOM_METADATA", "BLOCK_USER_CUSTOM",
    "MARKER_RE", "begin_marker", "end_marker",
    "benchmark_declarable_types", "decl_line", "deck_note",
    # Bench declarations
    "BenchField", "SIESTA_BENCH_FIELDS",
    # Emitters
    "emit_header", "emit_provenance", "emit_bench_marks",
    "write_validation_report", "VALIDATION_SUFFIX",
    "emit_atom_metadata", "emit_user_custom_placeholder",
    # In-body application
    "apply_atom_metadata",
    # USER-CUSTOM round-trip
    "replace_user_custom_inner", "merge_user_custom_from_target",
    # Git / time
    "molbuilder_git_sha", "generated_at_now",
    # Per-block extractors (read-side; re-export from parse/scripts/)
    # Reading the blocks back -- the ONE door (plan.md § 5d)
    "read_script", "ScriptSource",
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
    #: **W10's one per-render context** — *what this deck derived* — carried on
    #: the form so it can be READ, not only closed over.
    #:
    #: Both engines already keep exactly this dict: SIESTA fills it before the
    #: layout (its MEMBERSHIP depends on ``spin_polarized`` and ``relax_kind``),
    #: PySCF fills it as its blocks render.  Until it was declared here, the
    #: only readers were the engine's own closures — the syntax door, the
    #: layout, the record blocks — so a value like ``block_size = 8`` reached
    #: the deck with no way for anything outside the engine to say where 8 came
    #: from.  W10 says *"every reader takes it whole"*; a reader that
    #: re-derived these numbers instead would be W10's forbidden second
    #: channel, so the context is exposed rather than re-computed.
    #:
    #: The framework never WRITES it and never branches on it — it is a value
    #: the form carries, like ``provenance_defaults``, not a door.
    derived: Mapping[str, Any] = _dataclasses.field(default_factory=dict)
    #: The described calculation KIND this deck renders — a FACT the
    #: settings gate reads (validate(..., calculation=…) composes the
    #: kind's science from it).  A fact, not a hook: the spec states
    #: what it is; the framework decides what that implies.
    calculation: str = "optimization"
    #: ``(text, struct, cfg) -> [Issue]`` — this engine's answer to *what must
    #: a finished deck of mine satisfy?*  Runs on the file as written.
    check_rules: Optional[Callable] = None
    #: ``(struct, cfg) -> (struct, kwargs)`` — WHAT the settings gate should
    #: judge, when that is not the structure as it arrived.  SIESTA wraps
    #: coordinates into the cell and resolves a box before writing, so the deck
    #: expresses a structure the caller never handed in; judging the input
    #: would judge something nobody runs.
    #:
    #: **It exists so step 3.3 has ONE owner.**  Both engines ran the gate
    #: themselves inside ``spec_for`` while ``prepare_deck`` ran it again --
    #: two owners, and on different subjects: the engine's call saw the wrapped
    #: coordinates and the resolved cell, the framework's saw neither.  The
    #: order is the framework's (§ 4.3); what the order is applied TO can be
    #: the engine's, and that is what this carries.
    validate_subject: Optional[Callable] = None
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

    ``findings`` carries step 3.3's verdict OUT, so the companion validation
    file can state what the checker said about this deck.  Those findings are
    produced before a line of text exists and were reported to stderr and
    dropped; the artifact gate's own findings arrive later, in
    :func:`prepare_deck`, and the file wants both — *"the final validation of
    the full script"* is the two halves together, not whichever one the caller
    happened to hold.
    """
    emitted: Tuple[str, ...]
    findings: Tuple["Issue", ...]

    def __new__(cls, text: str, emitted=(), findings=()) -> "RenderedDeck":
        self = super().__new__(cls, text)
        self.emitted = tuple(emitted)
        self.findings = tuple(findings)
        return self

    @property
    def text(self) -> str:
        """The deck, as a plain ``str``.  Named because the check gate and its
        tests read it that way, and because ``str(deck)`` at a call site reads
        as a conversion rather than as *the deck's text*."""
        return str(self)


def _render_sections(spec: "DeckSpec", cfg, *, verbose: bool = True,
                     log=None) -> Tuple[List[str], List[str]]:
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
        # WHICH items spoke and which declined -- the log's business, and only
        # this walk knows.  A conditional item that quietly emitted nothing
        # looks exactly like one that was never in the layout, and telling
        # those apart is most of what a reader comes to the log for.
        spoke: List[str] = []
        silent: List[str] = []
        for name in section.items:
            param = parameter(name, spec.engine, config=cfg)
            # EVERY ENGINE HOOK IS CALLED THROUGH THE BOUNDARY (§ 4.6).  This
            # walk is a walk over the engine's functions, so an exception with
            # no owner on it is the ordinary failure here, not an exotic one.
            with _calling("line", engine=spec.engine,
                          where=f"item {name!r}", log=log):
                text = spec.line(param)
            if text is None:
                silent.append(name)
                continue
            spoke.append(name)
            if verbose:
                with _calling("note_lead", engine=spec.engine,
                              where=f"item {name!r}", log=log):
                    lead = spec.note_lead(param)
                body.extend(param.note(*lead))
            body.append(text)
            # WHAT THIS PARAMETER PUT IN THE DECK -- the LINE, verbatim.
            #
            # Not the keyword.  A keyword search cannot tell a setting from a
            # READ of that setting, and a deck that is a program does both: the
            # effective-parameters record reads ``mf.conv_tol`` back to report
            # it (W8), and that read satisfied the gate for a
            # ``mf.conv_tol = …`` line a writer bug had dropped -- measured
            # 2026-08-19, deleting the setting left the deck passing.  Two
            # features cancelling: the record that makes a deck honest about
            # what the engine holds was what let a writer bug through.
            #
            # The line is exact evidence and needs no matching rules at all --
            # no case folding, no separator folding, no stripping comments or
            # (in a Python deck) string literals, and no deciding WHICH of an
            # item's declared keywords this run took.  All of that existed only
            # to feed this check.  The file is what molbuilder wrote moments
            # earlier, so a verbatim compare is the honest one.
            emitted.append(text)
        if body:
            out.append("")
            # A SECTION WITH NO TITLE GETS NO HEADING, and the engine is not
            # asked to spell one.  This called ``section_title("")`` and tested
            # the result, so an engine whose heading it writes itself had to
            # pass a suppressing ``section_title`` -- which meant its OTHER
            # sections could not share the spec, which is why one deck needed
            # eight of them.
            with _calling("section_title", engine=spec.engine,
                          where=f"section {section.title!r}", log=log):
                title = (spec.section_title(section.title)
                         if section.title else "")
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
        if log is not None:
            log.produced("Section", f"{section.title!r}  "
                                    f"{len(body)} lines  "
                                    f"{', '.join(spoke) or '(none)'}")
            if silent:
                log.note(f"declined for this configuration: "
                         f"{', '.join(silent)}")
    return out, emitted


def render_deck(spec: "DeckSpec", struct, cfg, *, verbose: bool = True,
                log=None) -> "RenderedDeck":
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
    # STEP 3.3, HERE, AND ONLY HERE.  The gate belongs to rendering because
    # its whole job is to refuse before a line exists -- and every route that
    # produces deck text must be gated, not only the one that also writes.
    # It ran in TWO places until 2026-08-19: each engine called it inside
    # `spec_for` and `prepare_deck` called it again, on a different subject.
    from .validation import report as _report, validate as _validate
    with _calling("validate_subject", engine=spec.engine, log=log):
        _subject, _kw = ((spec.validate_subject(struct, cfg))
                         if spec.validate_subject else (struct, {}))
    _issues = _validate(_subject, cfg, calculation=spec.calculation, **_kw)
    if log is not None:
        log.step("STEP 3.3 · VALIDATE — the settings gate")
        log.received("subject", ("the spec's own subject" if spec.validate_subject
                                 else "the structure as it arrived")
                     + (f"  ({', '.join(sorted(_kw))})" if _kw else ""))
        log.produced("verdict",
                     f"{sum(1 for i in _issues if i.severity == 'error')} error, "
                     f"{sum(1 for i in _issues if i.severity != 'error')} warn")
        for i in _issues:
            log.note(f"{i.severity}: [{i.where}] {i.message}")
    # LOGGED BEFORE REPORTED, so a refusal is IN the file with its reason.
    # `report` raises on an error-severity issue; a log written afterwards
    # would be missing exactly the run that most needed explaining.
    _report(_issues)

    parts: List[str] = []
    emitted: List[str] = []
    if log is not None:
        log.step("STEP 3.6 · RENDER — walking the layout in order")
    for member in spec.layout:
        if isinstance(member, Block):
            with _calling("Block.render", engine=spec.engine,
                          where=f"block {member.title!r}", log=log):
                text = member.render(struct, cfg)
            if log is not None:
                log.produced("Block", f"{member.title!r}  " + (
                    f"{len(text.splitlines())} lines" if text is not None
                    else "nothing to say for this configuration"))
            # ``None`` is *nothing to say*; ``""`` is a blank line the block
            # meant to write.  Testing truthiness conflates them, and a block
            # whose whole content is one blank line joins to "" -- so the
            # separator between two runs of settings silently disappeared.
            if text is not None:
                parts.append(text)
            continue
        lines, names = _render_sections(
            _dataclasses.replace(spec, layout=(member,)), cfg,
            verbose=verbose, log=log)
        parts.extend(lines)
        emitted.extend(names)
    science = "\n".join(parts)

    # AFTER THE WALK, because that is when both engines' context is complete:
    # SIESTA fills its own before the layout (membership depends on it), PySCF
    # fills part of it as its blocks render.  Printing at spec time would show
    # one engine's answers and an empty dict for the other.
    if log is not None and spec.derived:
        log.step("this deck's own derived context (W10)")
        for _k in sorted(spec.derived):
            log.chose(_k, spec.derived[_k], "derived from (struct, cfg)")

    with _calling("provenance_defaults", engine=spec.engine, log=log):
        _defaults = (spec.provenance_defaults(cfg)
                     if spec.provenance_defaults else None)
    record: List[str] = [emit_provenance(
        generator_version=molbuilder_git_sha(),
        generated_at=generated_at_now(),
        resolved_defaults=_defaults,
        engine=spec.engine)]
    # NAMED AS THEY GO IN, not counted afterwards.  Re-deriving which blocks
    # a list of three strings holds is the guess this file exists to replace,
    # and the first version of it got the answer wrong.
    in_record: List[str] = ["PROVENANCE"]
    if spec.bench_marks is not None:
        with _calling("bench_marks", engine=spec.engine, log=log):
            marks = spec.bench_marks(struct, cfg)
        if marks:
            record.append(emit_bench_marks(**marks))
            in_record.append("BENCH-MARKS")
    atoms = emit_atom_metadata(
        regions=dict(getattr(struct, "regions", {}) or {}),
        annotations=dict(getattr(struct, "annotations", {}) or {}),
        n_atoms_total=int(getattr(struct, "n_atoms", 0)),
        created_by=spec.created_by,
        created_at=generated_at_now())
    if atoms:
        record.append(atoms)
        in_record.append("ATOM-METADATA")

    text = (science + "\n\n" + emit_user_custom_placeholder()
            + "\n\n" + machine_record_banner()
            + "\n\n" + "\n\n".join(record) + "\n")
    if log is not None:
        log.produced("record", ", ".join(in_record))
        if spec.bench_marks is None:
            log.note("BENCH-MARKS: nothing — this engine declares no "
                     "anchors, and that is a recorded answer (W5)")
        elif "BENCH-MARKS" not in in_record:
            log.note("BENCH-MARKS: this engine declares anchors but had "
                     "none to state for this configuration")
        if "ATOM-METADATA" not in in_record:
            log.note("ATOM-METADATA: nothing — this structure carries no "
                     "regions or annotations")
        log.produced("deck", f"{len(text.splitlines())} lines, "
                             f"{len(emitted)} recorded for the gate")
    return RenderedDeck(text=text, emitted=tuple(emitted),
                        findings=tuple(_issues))


# --------------------------------------------------------------------- #
#  The CHECK gate — the file the engine will open                       #
# --------------------------------------------------------------------- #



#: The companion file's name, beside the deck it is about.  The one spelling
#: is `identity.OUR_FILE_PATTERNS`; this is the suffix that builds it.
VALIDATION_SUFFIX = ".validation.txt"

_VALIDATION_HEADER = """\
# What the checks said about {deck}
# ---------------------------------------------------------------------------
# THESE ARE ADVISORY, and reading them that way is the point.
#
# They are heuristics applied to the settings you chose and to the file that
# was generated from them.  They are NOT a verdict on the physics of your
# system.  Judge each one against what you know about this structure and this
# method: a warning that does not apply to your case is one to override
# deliberately, and a clean report is not a guarantee of a correct answer.
#
#   error   the deck was written and then failed its own check -- do not
#           submit it.  (A refusal BEFORE the deck exists writes no deck and
#           no report; those reasons are on stderr instead.)
#   warn    the deck was written; worth reading before you spend cluster time
#   info    advisory
#
# molbuilder never reads this file back.  It travels beside the deck so the
# two can be opened together months from now.
# ---------------------------------------------------------------------------
"""


def write_validation_report(deck_path, findings) -> "Path":
    """Write the deck's companion validation file, and return its path.

    **A separate file, not a block inside the deck** (user, 2026-08-23), and
    that choice removes a real problem rather than being a matter of taste:
    the artifact gate's subject is the file on disk, so findings written INTO
    that file would mean the bytes that were checked are not the bytes that
    ship — write, check, write again, and the second write is unchecked.
    Beside it, the deck is final the moment it is checked.

    Written BEFORE the artifact gate's `report`, for the same reason the log
    is: that call raises on an error-severity finding, and a deck that was
    written and then failed its own check is exactly the one whose reasons a
    person needs on disk.

    **It does not cover every refusal, and the header says so honestly.**  The
    settings gate (step 3.3) raises before a line of the deck exists, so there
    is no deck for a companion to be about; those reasons travel in the
    exception and on stderr.  This file exists wherever a deck does.

    ``findings`` is both halves — step 3.3's verdict on the settings and step
    3.11's on the artifact — because *"the final validation of the full
    script"* is the two together, not whichever the caller happened to hold.
    """
    out = Path(deck_path).with_suffix(VALIDATION_SUFFIX)
    lines = [_VALIDATION_HEADER.format(deck=Path(deck_path).name)]
    if not findings:
        lines.append("")
        lines.append("The checks had nothing to say about this deck.")
        lines.append("")
        lines.append("That is not a claim that the calculation is right -- "
                     "only that")
        lines.append("nothing molbuilder knows how to check looked wrong.")
    else:
        width = max(len(i.severity) for i in findings)
        for i in findings:
            where = f"[{i.where}] " if i.where else ""
            lines.append("")
            lines.append(f"{i.severity.upper():<{width}}  {where}{i.message}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def check_deck(path, spec: "DeckSpec", rendered: "RenderedDeck",
               struct=None, cfg=None, *, log=None) -> List["Issue"]:
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

    # BOTH markers, not just BEGIN.  A block is delimited by a PAIR, and
    # counting one end let a stray END through: the USER-CUSTOM round-trip
    # carries a person's zone forward verbatim, so a marker line pasted into
    # it lands in the written deck -- and until 2026-09-05 a stray BEGIN was
    # caught here while a stray END was not.  This is the gate that refuses
    # an ambiguous boundary (`job-contracts.md` § 3.5); the merge no longer
    # guesses, it carries the outermost span forward and leaves the verdict
    # here, where a refusal reaches the validation report before `report`
    # raises.
    for block, label in ((BLOCK_USER_CUSTOM, "the section left for a reader"),
                         (BLOCK_PROVENANCE, "the provenance record")):
        for marker, which in ((begin_marker(block), "BEGIN"),
                              (end_marker(block), "END")):
            n = text.count(marker)
            if n != 1:
                out.append(Issue(
                    "error",
                    f"{label}'s {which} marker appears {n} times in "
                    f"{p.name}, expected once"
                    + (" — a marker line inside your own section makes the "
                       "boundary ambiguous; remove it"
                       if n > 1 and block == BLOCK_USER_CUSTOM else ""),
                    where=f"deck.{block}"))

    # THE LOOP CLOSED: every LINE the parameters sub-step produced must be in
    # the file, verbatim.  Without this the two halves are related only by
    # hope -- which is how a deck came to state values nobody had read.
    #
    # Verbatim, and not a keyword search.  A keyword can appear because the
    # deck READS it (the effective-parameters record reads every setting back,
    # by design), and that read satisfied this check for a setting a writer bug
    # had dropped.  A line is the assignment itself, so it cannot be confused
    # with a mention of it -- and it catches a mangled VALUE too, which a
    # keyword search never could.
    present = set(text.splitlines())
    # ONE EMISSION MAY BE SEVERAL LINES, and each of them is evidence.
    # ``line`` returns ``str | None`` and a str may hold a pair: a FIXED total
    # spin needs ``Spin.Fix`` and ``Spin.Total`` together, and the free-energy
    # section is titled *"a PAIR: the value + its switch"* for the same reason.
    # Comparing the emission whole meant a two-line answer could never equal
    # any member of a set of single lines, so the gate refused a deck that was
    # correct -- every spin-polarized SIESTA run, from the moment this rule
    # replaced the keyword search on 2026-08-19 until 2026-08-19.  It went
    # unseen because the reference harness's own spin case was passing a field
    # name ``SiestaConfig`` does not have and pinning the TypeError.
    for line in dict.fromkeys(
            ln for e in emitted for ln in e.splitlines() if ln.strip()):
        if line not in present:
            out.append(Issue(
                "error",
                f"{p.name}: the parameters step wrote {line!r} and the file "
                f"does not contain that line",
                where="deck.missing_line"))

    if rules is not None:
        with _calling("check_rules", engine=spec.engine, where=p.name,
                      log=log):
            out.extend(rules(text, struct, cfg) or [])
    return out


def prepare_deck(spec: "DeckSpec", struct, cfg, path, *,
                 verbose: bool = True, log=None):
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
    from .validation import report
    if log is not None:
        _log_spec(spec, log)
    # 3.3 validate now runs inside `render_deck`, on the subject the SPEC
    # names -- one owner for the step, and the same one for every route that
    # renders (`render_fdf` / `render_script` gate too, not just this one).
    rendered = render_deck(spec, struct, cfg, verbose=verbose, log=log)
    written = write_script(path, rendered.text)        # 3.10 write
    if log is not None:
        log.step("STEP 3.10 · WRITE")
        log.produced(written.name,
                     f"{len(written.read_text(encoding='utf-8').splitlines())} "
                     f"lines -> {written}")
    issues = check_deck(written, spec, rendered, struct, cfg,
                        log=log)                                # 3.11 check
    if log is not None:
        log.step("STEP 3.11 · CHECK — the artifact gate")
        log.received(written.name, "read back from disk")
        log.produced("compared", f"{len(dict.fromkeys(rendered.emitted))} "
                                 f"distinct lines the parameters step wrote")
        log.produced("verdict", f"{len(issues)} issue(s)")
        for i in issues:
            log.note(f"{i.severity}: [{i.where}] {i.message}")
    # The companion file, from BOTH halves of the verdict, and written before
    # `report` -- which raises on an error, and a refused run is the one whose
    # reasons most need to be on disk.
    write_validation_report(written, list(rendered.findings) + list(issues))
    report(issues)
    return written


def _log_spec(spec: "DeckSpec", log) -> None:
    """Write down the FORM the engine handed over, before anything runs on it.

    **This is readable only because a form crosses the seam.**  While engines
    handed back finished text there was nothing here to describe: the layout,
    the record's values and the check rules existed only as whatever the
    writer had already done with them.  The same property that lets the check
    gate re-derive what a deck was supposed to contain lets the log state it
    (`script-preparation.md` § 4.3).
    """
    log.step("STEP 3.4 · SPEC_FOR — the engine describes its deck")
    log.received("engine", spec.engine)
    n_sec = sum(1 for m in spec.layout if isinstance(m, Section))
    n_blk = sum(1 for m in spec.layout if isinstance(m, Block))
    log.produced("layout", f"{len(spec.layout)} members "
                           f"({n_sec} Section, {n_blk} Block)")
    for member in spec.layout:
        if isinstance(member, Block):
            # W11: free text, so this is ALL the framework can say about it
            # until it renders -- no catalogue note reaches inside, and it
            # contributes no line to the gate.
            log.note(f"Block    {member.title!r}  (free text, W11)")
        else:
            log.note(f"Section  {member.title!r}  "
                     f"{', '.join(member.items)}")
    # W5: a slot answering None is answering NOTHING, and that is a real
    # answer -- so it is written down rather than left as a blank.
    for slot in ("provenance_defaults", "bench_marks", "check_rules",
                 "validate_subject"):
        log.produced(slot, "answered" if getattr(spec, slot) is not None
                     else "nothing (W5)")


# --------------------------------------------------------------------- #
#  READING THE BLOCKS BACK                                              #
# --------------------------------------------------------------------- #
#
#  The inverse of the emitters above, and it lives HERE because a format
#  has one owner.  These were `parse/scripts/` until 2026-09-05, six
#  extractor functions each wrapped in a `TextParser` class that existed
#  only to fit `parse/`'s registry -- a registry whose whole purpose is
#  *"query it rather than knowing which parser to call"*, for blocks
#  molbuilder writes itself and whose caller always knows which one it
#  wants.  `ProvenanceTextParser.parse` built a ten-field `ScriptResult`
#  so a caller could read back the one dict the function already returned.
#
#  THE SPLIT COST A CIRCULAR IMPORT.  `parse/scripts/markers.py` was forty
#  lines re-exporting `BLOCK_*` + `MARKER_RE` from this module -- *"so the
#  read-side parsers stay in lock-step with the write-side emitters"* --
#  and this module imported the extractors back through a
#  `_LAZY_EXTRACTORS` table, because *"an eager top-level import would
#  deadlock."*  Both are gone: the constants were always here, and the
#  readers now sit beside them.
#
#  These functions take a STRING and do no I/O.  Reading the file is the
#  caller's job -- the rule that used to be `parse.md` § 7 forbidden #2,
#  kept because it is about this code, not about the ABC that carried it.
#
#  Contract: `execution/job-contracts.md` § 3.1 (the block grammar and the
#  emit matrix), `plans/plan.md` § 5d (why they moved).

#: The atom-metadata schema this build WRITES, and the set it READS.
#: Compared against rather than a literal: a version written down in two
#: places is how the block came to claim v4 while carrying v7.
from molbuilder.sidecars.molstruct import (        # noqa: E402
    READABLE_VERSIONS as _READABLE,
    SCHEMA_VERSION as _CURRENT_SCHEMA,
)

# ---- from parse/scripts/header.py ----
def _extract_header_text(text: str) -> Optional[str]:
    """Find the HEADER block and return its inner content as a single
    string (free-form prose, comment prefixes stripped).

    Returns ``None`` when no HEADER block is present.  The leading
    ``# `` (or ``#``) on each line is removed so the result is the
    raw prose the generator wrote; line ordering is preserved.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m or m.group(1) != BLOCK_HEADER:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    out_lines: List[str] = []
    for raw in lines[begin_idx + 1: end_idx]:
        # Strip the comment prefix the generator emits ("# " or "#").
        if raw.startswith("# "):
            out_lines.append(raw[2:])
        elif raw.startswith("#"):
            out_lines.append(raw[1:])
        else:
            out_lines.append(raw)
    return "\n".join(out_lines)


# ---- from parse/scripts/provenance.py ----
_PROVENANCE_KV_RE = re.compile(
    r"^#\s+(?P<key>[A-Za-z][A-Za-z0-9._-]*)\s{2,}(?P<val>.+?)\s*$")

_PROVENANCE_DEFAULTS_HDR = re.compile(
    r"^#\s+resolved-defaults\s*:\s*$")

_PROVENANCE_DEFAULTS_KV_RE = re.compile(
    r"^#\s{4,}(?P<key>[A-Za-z][A-Za-z0-9._-]*)\s{2,}(?P<val>.+?)\s*$")

def _extract_provenance_dict(text: str) -> Optional[Dict[str, str]]:
    """Find the PROVENANCE block and return its k/v payload as a flat
    dict.  Returns ``None`` when no well-formed PROVENANCE block is
    present.  Empty-but-present block returns ``{}`` (distinct from
    None — `model/parse.md`'s absent-vs-empty rule)."""
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != BLOCK_PROVENANCE:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    out: Dict[str, str] = {}
    in_defaults = False
    for raw in lines[begin_idx + 1: end_idx]:
        if _PROVENANCE_DEFAULTS_HDR.match(raw):
            in_defaults = True
            continue
        if in_defaults:
            dm = _PROVENANCE_DEFAULTS_KV_RE.match(raw)
            if dm:
                out[f"resolved-defaults.{dm.group('key')}"] = dm.group("val")
                continue
            # A non-defaults-shaped line ends the sub-block; the
            # top-level k/v matcher below may still pick it up.
            in_defaults = False
        m = _PROVENANCE_KV_RE.match(raw)
        if m:
            out[m.group("key")] = m.group("val")
    # Block present, no parseable k/v -> {} (NOT None).  Distinguished
    # from "block absent" -- `model/parse.md`'s None-vs-empty-dict
    # semantics.
    return out


# ---- from parse/scripts/user_custom.py ----
def _user_custom_span(text: str) -> Optional[Tuple[int, int]]:
    """``(begin_idx, end_idx)`` of the USER-CUSTOM block, or ``None``.

    **The OUTERMOST pair, and that is not a guess.**
    :func:`emit_user_custom_placeholder` writes exactly ONE ``BEGIN`` and one
    ``END`` on every generation, so the outermost pair in a generated file is
    the framework's.  Everything between them is the person's content --
    including a line that happens to look like a marker, because a person
    pasting a snippet from another deck has pasted TEXT, not a boundary.

    It took the INNERMOST span until 2026-09-05 -- resetting ``begin_idx`` on
    each ``BEGIN`` and breaking on the first ``END`` after it -- so a stray
    ``BEGIN`` silently discarded everything above it and a stray ``END``
    discarded everything below.  No refusal, no warning: the file came back
    well-formed and shorter.  Measured through the real save route, HTTP 200.

    Lossless and idempotent: re-merging the result yields the same content,
    because the outermost pair is stable under splicing.  The stray markers
    then survive INTO the written deck, where `check_deck`'s marker count is
    what refuses -- refusals belong to the check gate, which writes them to
    the validation report before `report` raises (`prepare_deck`'s order).
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m or m.group(1) != BLOCK_USER_CUSTOM:
            continue
        if m.group(2) == "BEGIN":
            if begin_idx is None:          # the FIRST begin, and only it
                begin_idx = i
        elif m.group(2) == "END":
            end_idx = i                    # the LAST end
    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        return None
    return begin_idx, end_idx


def _extract_user_custom_inner(text: str) -> Optional[List[str]]:
    """Return the inner lines of the USER-CUSTOM block in ``text``, or
    ``None`` if there is no well-formed USER-CUSTOM BEGIN/END pair.

    Inner lines are everything STRICTLY between the BEGIN and END
    markers (markers excluded).  Trailing/leading whitespace inside
    the block is preserved.  See :func:`_user_custom_span` for which
    pair, and why.
    """
    span = _user_custom_span(text)
    if span is None:
        return None
    begin_idx, end_idx = span
    return text.splitlines()[begin_idx + 1: end_idx]


# ---- from parse/scripts/atom_metadata.py ----

def _brace_delta(line: str) -> int:
    """``{`` minus ``}`` on one line, counting only braces OUTSIDE strings.

    A plain ``line.count("{") - line.count("}")`` stood here until
    2026-09-05, and a brace inside a JSON *string* closed the walk early.
    Measured: a region named ``a}b`` -- valid JSON on the wire, written
    correctly by :func:`emit_atom_metadata` -- made the whole
    ATOM-METADATA block unreadable, so every reader of that deck got
    ``None`` and the labels AND the frozen set vanished with no message.
    ``{``, ``"`` and ``\\`` in a label were all fine; only ``}`` was fatal,
    which is exactly the shape of bug that survives casual testing.

    JSON strings cannot contain a literal newline, so the in-string state
    never has to carry across lines.
    """
    depth = 0
    in_str = False
    escaped = False
    for ch in line:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return depth


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
        brace_depth += _brace_delta(stripped)
        if brace_depth <= 0:
            break
    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


# ---- from parse/scripts/bench_marks.py ----
def _coerce_scalar(s: str) -> Any:
    """Best-effort numeric coercion for BENCH-MARKS scalar values.
    Returns the original string when neither int nor float parses."""
    s = s.strip()
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s

def _extract_bench_marks_dict(text: str) -> Optional[Dict[str, Any]]:
    """Find the BENCH-MARKS block and return its structured payload.

    Returns ``None`` when no BENCH-MARKS block is present.  See the
    module docstring for the payload shape.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m or m.group(1) != BLOCK_BENCH_MARKS:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    out: Dict[str, Any] = {"fields": []}
    for raw in lines[begin_idx + 1: end_idx]:
        # Strip "#   " comment prefix.
        s = raw.lstrip("#").strip()
        if not s:
            continue
        # `field <name> anchor=<x> type=<y> ...` rows.
        if s.startswith("field "):
            tokens = s.split()
            field: Dict[str, Any] = {"name": tokens[1]}
            for tok in tokens[2:]:
                if "=" not in tok:
                    continue
                k, _, v = tok.partition("=")
                v = v.strip()
                # Coerce numeric range / default values.
                if k == "range" and v.startswith("[") and v.endswith("]"):
                    try:
                        a, b = v[1:-1].split(",")
                        field["range"] = [_coerce_scalar(a), _coerce_scalar(b)]
                    except ValueError:
                        field["range"] = v
                elif k == "default":
                    field["default"] = _coerce_scalar(v)
                else:
                    field[k] = v
            out["fields"].append(field)
            continue
        # Top-level `key value` scalars.
        if " " in s and not s.startswith("field "):
            k, _, v = s.partition(" ")
            v = v.strip()
            if k == "version":
                out["version"] = v
            elif v.lower() in ("true", ".true."):
                out[k] = True
            elif v.lower() in ("false", ".false."):
                out[k] = False
            else:
                out[k] = _coerce_scalar(v)
    return out


# ---- from parse/scripts/source_dict.py ----
def _extract_script_source(text: str) -> Dict[str, Any]:
    """Single-pass extract over a generated-script body for the run
    decoder.  Returns a dict with:

      * ``regions``           dict[str, list[int]] | None
      * ``frozen_atoms``      list[int] | None
      * ``user_custom_lines`` list[str] | None
      * ``provenance``        dict[str, str] | None
      * ``schema_version``    int | None
      * ``notes``             list[str]

    ``None`` distinguishes "block absent" from "block present but
    empty" (``{}`` / ``[]``) — `model/parse.md`'s absent-vs-empty rule.  A block whose
    version is not the one this build writes is READ (a finished run must stay
    readable) and surfaced as a diagnostic note naming what may be missing --
    see the comment at the check itself.
    """
    atom_md = _extract_atom_metadata_dict(text)
    gated = _gate_atom_metadata(atom_md)
    return {
        "regions":           gated["regions"],
        "frozen_atoms":      gated["frozen_atoms"],
        "user_custom_lines": _extract_user_custom_inner(text),
        "provenance":        _extract_provenance_dict(text),
        "schema_version":    gated["schema_version"],
        "notes":             gated["notes"],
    }


def _gate_atom_metadata(atom_md: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ATOM-METADATA schema gate, over a block ALREADY read.

    Split out of :func:`_extract_script_source` on 2026-09-05 so a caller
    that already holds the block does not re-parse the whole script to gate
    it.  :func:`read_script` did exactly that: it called
    ``_extract_script_source(text)``, which re-ran the atom-metadata,
    user-custom and provenance extractors the door had just run itself, so
    three of five blocks were parsed TWICE.  Measured on a real 1272-line
    deck: the door cost **4.8 ms** against **0.5 ms** for a single block --
    and the ``0.55 ms`` this module's own docstring quoted was
    ``_extract_script_source``, the inner call, not the door.

    Returns ``regions`` / ``frozen_atoms`` / ``schema_version`` / ``notes``.
    """
    notes: List[str] = []
    regions: Optional[Dict[str, List[int]]] = None
    frozen: Optional[List[int]] = None
    schema_version: Optional[int] = None
    if atom_md is not None:
        sv = atom_md.get("schema_version")
        if isinstance(sv, int):
            schema_version = sv
            if sv not in _READABLE:
                # REFUSED, NOT READ (2026-08-01, by decision; amended
                # 2026-08-20 to the READABLE SET -- v8 added only optional
                # identity columns, so a v7 block reads whole and refusing
                # it would have made every existing finished run's labels
                # unreadable for a change that loses nothing).
                #
                # It used to READ an older block and attach a warning, on the
                # reasoning that a finished run cannot be re-exported the way a
                # sidecar can.  That reasoning is wrong for a product still
                # being built: an older block stores the same facts in different
                # places, so "read it and warn" hands back a payload that LOOKS
                # complete and quietly is not -- which is how a junction's fifty
                # frozen atoms came back as an empty list.  Supporting both
                # shapes also doubles what every reader, test and debugging
                # session has to hold in its head, for data that will be
                # regenerated anyway.
                #
                # The scripts get regenerated.  That is cheaper than a format
                # nobody can reason about.
                notes.append(
                    f"atom-metadata schema_version {sv}, but this molbuilder "
                    f"writes v{_CURRENT_SCHEMA} and reads "
                    f"{sorted(_READABLE)} only. The block was "
                    f"NOT read -- an older one keeps the same facts in "
                    f"different places (before v7 the frozen atoms sat in a "
                    f"top-level key rather than in `regions`), so reading it "
                    f"would silently drop what it cannot map. Re-generate the "
                    f"script from the structure."
                )
            else:
                # A BLOCK AT ANY OTHER VERSION IS READ, AND SAID SO ABOUT.
                #
                # This is a FINISHED RUN on disk, so refusing it outright would
                # make a user's existing results unreadable -- unlike the
                # sidecar, which is refused because it is still being worked on
                # and can be re-exported. But the note has to be accurate, and
                # this one was not: it said "molbuilder expects 4 — loading with
                # current handler", which reads as a formality.
                #
                # It is not. Before the label store was unified, the reserved
                # `frozen_atoms` list sat in a top-level key; this reader takes
                # the whole store from `regions`, so on an older block the
                # LABELS COME BACK AND THE FROZEN SET DOES NOT. That is how a
                # junction's fifty pinned electrode atoms read back as an empty
                # list. The note says which fact is at risk now, instead of
                # reporting a number.
                raw_regions = atom_md.get("regions")
                if isinstance(raw_regions, dict):
                    regions = {
                        str(k): sorted({int(i) for i in v})
                        for k, v in raw_regions.items()
                    }
                else:
                    regions = {}
                # ONE designated read: v5 keeps the reserved label in
                # `regions`, v3/v4 kept it in a top-level key, and this
                # knows which without the caller spelling the name.
                from molbuilder.sidecars import molstruct as _ms
                frozen = _ms.frozen_atoms(atom_md)
        else:
            notes.append(
                "atom-metadata block has no schema_version; ignored.")
    return {
        "regions":        regions,
        "frozen_atoms":   frozen,
        "schema_version": schema_version,
        "notes":          notes,
    }



@dataclass(frozen=True)
class ScriptSource:
    """What a generated script says about itself -- every reserved block.

    One object, and one parse per block.  A per-block door was considered
    and rejected on measurement -- but the numbers first written here were
    wrong twice over, so they are restated honestly: reading every block
    off a real 1272-line deck costs **3.0 ms** against **0.5 ms** for one.

    *The original claim was "0.55 ms against 0.22 ms".  The 0.55 was
    `_extract_script_source` -- the INNER call -- not this door, which
    then cost 4.8 ms because it asked that function for the gated view
    and re-ran three of the same extractors for the raw one.  Three of
    five blocks were parsed twice.  Split out `_gate_atom_metadata`
    (2026-09-05) so the gate takes a block already read.*

    Fields are ``None`` when the block is ABSENT and empty when the block
    is PRESENT-but-empty -- the distinction the ATOM-METADATA emission rule
    turns on (`job-contracts.md` § 3.1).
    """

    #: Raw blocks, exactly as their extractors return them.
    header:        Optional[str]            = None
    provenance:    Optional[Dict[str, str]] = None
    bench_marks:   Optional[Dict[str, Any]] = None
    atom_metadata: Optional[Dict[str, Any]] = None
    user_custom:   Optional[List[str]]      = None

    #: The ATOM-METADATA block SCHEMA-GATED and unpacked: a block at a
    #: version this build does not read is refused, and `notes` says so.
    #: `schema_version` is the version the block DECLARED -- kept because a
    #: version written down in two places is how one came to claim v4 while
    #: carrying v7.
    regions:        Optional[Dict[str, List[int]]] = None
    frozen_atoms:   Optional[List[int]]            = None
    schema_version: Optional[int]                  = None

    #: Non-fatal notes -- an unreadable schema version, say.
    notes:          Tuple[str, ...]                = ()


def read_script(text: str) -> ScriptSource:
    """Read every reserved block out of a generated ``.fdf`` / ``.py`` body.

    **The one door.**  Callers used to import a private ``_extract_*_dict``
    across a package boundary -- `parse/contract.py`, `jobset/summarize.py`,
    `jobset/agreement.py` and `transport/compose.py` all did -- because the
    public surface was six `TextParser` classes that returned a whole-script
    object to carry one dict.  Ask this instead.

    Takes a STRING: reading the file is the caller's job.
    """
    # ONE parse per block.  This called `_extract_script_source(text)` for
    # the gated view AND every extractor again for the raw one, so
    # atom-metadata, user-custom and provenance were each read twice.
    atom_md = _extract_atom_metadata_dict(text)
    gated = _gate_atom_metadata(atom_md)
    return ScriptSource(
        header=_extract_header_text(text),
        provenance=_extract_provenance_dict(text),
        bench_marks=_extract_bench_marks_dict(text),
        # The RAW block.  `regions` / `frozen_atoms` below are the same
        # block after the schema gate; a caller wanting the gate takes
        # those, one wanting the payload takes this.
        atom_metadata=atom_md,
        user_custom=_extract_user_custom_inner(text),
        regions=gated["regions"],
        frozen_atoms=gated["frozen_atoms"],
        schema_version=gated["schema_version"],
        notes=tuple(gated["notes"] or ()),
    )


# ===================================================================== #
#  ROUND-TRIP — the operations that need BOTH halves                    #
# ===================================================================== #
#
#  These come last because they are the only things here that both read
#  and write, so they sit above everything they depend on rather than
#  in the middle of it.  The file now reads in dependency order:
#
#      vocabulary  ->  emit  ->  read  ->  round-trip
#
#  They lived among the EMITTERS until 2026-09-05, which is where a
#  reader would least expect the one function in this module that opens
#  a file on disk.  Neither emits a block:
#
#    * `merge_user_custom_from_target` reads the PREVIOUS OUTPUT to carry
#      your USER-CUSTOM zone forward across a regeneration
#      (`job-contracts.md` § 3.5).  It is the reason a regenerated deck
#      does not lose what you typed into it.
#    * `apply_atom_metadata` reads a block and writes onto a
#      Structure -- a post-process; it produces no script at all.
#
#  Measured before the move: these are the ONLY two call paths crossing
#  from the write half to the read half, and nothing crosses back.
#  Nothing references them at import time, so the order is free.

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


def apply_atom_metadata(struct: Any, payload: Dict[str, Any]) -> bool:
    """THE reader of an ATOM-METADATA payload.  One block, one reader.

    ``struct`` is duck-typed: any object with a mutable ``regions`` dict will
    do.  Returns ``True`` when labels were applied, ``False`` when the payload
    carries none.  A caller holding a whole script extracts first --
    ``_extract_atom_metadata_dict(text)`` -- and hands the payload here.

    THE RULE: the block is read as it is written today.  Nothing translates an
    older layout, and a block that no longer says what this build reads simply
    carries no labels.

    ONE guard, and it is not about versions: a block whose ``n_atoms_total``
    disagrees with the structure raises
    :class:`~molbuilder.sidecars.molstruct.MolstructPairingError`.  Labels are
    indexed by atom POSITION, so a block written for another structure does
    not fail loudly when applied -- it labels the wrong atoms and says
    nothing.  Same error type as the sidecar's identical guard: one name for
    one condition.

    Until 2026-09-05 there were two readers of this one block and they
    disagreed -- this one translated pre-v7 layouts, while ``/api/build/load``
    applied the block through the sidecar's ``apply_to_structure`` and refused
    it.  The same finished run kept its frozen set through one door and lost
    it through the other.  One reader now, and no translation in it.
    """
    from molbuilder.sidecars.molstruct import MolstructPairingError

    # IS THIS BLOCK EVEN ABOUT THESE ATOMS?  Asked before anything is applied,
    # because every label below is an index into the atom list and means
    # nothing -- or means the wrong thing -- if the answer here is no.
    stated = payload.get("n_atoms_total")
    have = len(getattr(struct, "elements", ()) or ())
    if stated is not None and have and stated != have:
        raise MolstructPairingError(
            f"these atom labels were written for a structure of {stated} "
            f"atoms and this one has {have}.  Region and frozen-atom indices "
            f"are positions in the atom list, so applying them here would "
            f"label the wrong atoms; re-export the labels from the structure "
            f"they belong to.")

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
    # THE SAME SPAN THE EXTRACTOR USES.  A second copy of the boundary rule
    # stood here until 2026-09-05, and two readers of one boundary is how the
    # zone came to mean different things to the half that reads it and the
    # half that replaces it.
    lines = text.splitlines(keepends=False)
    span = _user_custom_span(text)
    if span is None:
        return text
    begin_idx, end_idx = span
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
    # `_extract_user_custom_inner` is defined below in THIS module.
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

"""Helpers shared across the build / modify / watch blueprints.

This module is the SINGLE source of truth for:

* the ``Issue`` -> JSON wire shape
* JSON -> Structure body parsing (xyz + per-atom metadata lists)
* Structure -> JSON response body construction
* JSON -> dataclass coercion (used by Build for SiestaConfig /
  PySCFConfig form values; available to Modify for any future op
  that takes a dataclass-driven body, e.g. M5's electrode panel)

If a helper is genuinely blueprint-specific (e.g. Build's
``/api/build/load`` accepts both multipart and JSON, Modify's body
parsing always carries the canonical state bundle), it stays in
the calling blueprint.  Promote here when at least two callers
need the same behaviour and drift would silently break wire
contracts.
"""

from __future__ import annotations

import dataclasses
import math
import re
import typing
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

from flask import jsonify

from molbuilder.structure import (
    Structure,
    annotations_from_json,
)
from molbuilder.validation import validate_geometry
from molbuilder.cell import resolve_and_check
from molbuilder.periodicity_gate import (notices_for_report,
                                         validate_periodicity)


# J2 2026-06-14: charset guard for region label keys in
# the in-body region labels.  Same charset the wrapper basename
# uses (``runwrap.py::_SAFE_WRAPPER_NAME_RE``) so the labels are
# safe to embed in shell, JSON, sidecar filenames, and Issue
# message bodies without per-consumer escaping.  Length capped
# at 64 to avoid abuse via giant strings.
#: RETIRED 2026-08-03, and worth saying why rather than just deleting.
#:
#: This restricted region labels arriving over the wire to [A-Za-z0-9._-], and
#: it was enforced in exactly ONE place: `apply_labels_to_struct`, which read a
#: top-level `regions` key.  That shape is gone (labels ride inside the
#: structure now), and it turns out the restriction never covered the other two
#: ways a label reaches the application:
#:
#:   * the ENVELOPE -- `test_the_metadata_a_coordinate_file_cannot_hold_arrives
#:     _with_the_atoms` sends `α-helix` and asserts it survives;
#:   * the SIDECAR  -- `test_save_emits_utf8_bytes_for_unicode_region_label`
#:     pins UTF-8 labels on disk, an audit regression fix.
#:
#: So deleting the flat shape deleted the only path this guarded, and putting it
#: back on either surviving path is a NEW restriction, not a preserved one --
#: tried both, and each broke one of those two tests.
#:
#: THE OPEN QUESTION, left open deliberately: the original rationale still
#: holds -- a label reaches Issue messages, TranSIESTA's region-key comparison
#: and the sidecar, so a charset is what protects a future consumer that emits
#: one without escaping.  Whether labels should be ASCII-only EVERYWHERE (and
#: those two tests changed) or Unicode-capable everywhere (and this constant
#: deleted) is a decision about the data model, not a tidy-up to fold into a
#: wiring change.  Kept here, unused, so the question has a location.
_SAFE_REGION_LABEL_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


# --------------------------------------------------------------------- #
#  Issues                                                                #
# --------------------------------------------------------------------- #


def resolve_workflow_group(where: str, cfg) -> Optional[str]:
    """Return the workflow-group binding for an Issue ``where`` field.

    Per docs/web/ui-contract.md Rule 2, validator findings
    should attach to the workflow-group card whose fields they
    concern: a ``config.mesh_cutoff`` finding belongs in the Stage
    card; ``config.spin_polarized`` belongs in the Run profile card;
    ``config.max_scf_iter`` belongs in the Compute & budget card.
    The mapping is the SAME single source of truth that drives the
    form-schema render: each dataclass field's
    ``metadata["workflow_group"]``.

    Returns ``None`` for:
      * Issues that don't target a config field (where prefix isn't
        ``config.``) — geometry / cell / polymer findings render in
        a residual panel below the cards.
      * Issues whose dotted name doesn't resolve to a real dataclass
        field (typo, future-proofing).
      * Issues whose field has no ``workflow_group`` metadata (legacy
        untagged fields; rendered in residual panel).
    """
    if not where or not where.startswith("config."):
        return None
    if not dataclasses.is_dataclass(cfg):
        return None
    # Strip "config." prefix and any further dotted sub-fields:
    # "config.net_charge.makov_payne" → "net_charge".
    tail = where.split(".", 1)[1]
    field_name = tail.split(".", 1)[0]
    for f in fields(cfg):
        if f.name == field_name:
            return f.metadata.get("workflow_group")
    return None


def issues_to_json(issues, cfg=None):
    """Serialise List[Issue] for the JSON wire.

    The web client reads ``issues[].severity / message / where /
    workflow_group`` to decide how to display.  Schema duplicated
    literally in both blueprints' tests; if a key changes here,
    those tests catch it.

    ``cfg`` is the engine config dataclass — when provided, each
    issue's ``workflow_group`` is resolved (per
    :func:`resolve_workflow_group`) so the frontend can attach
    findings to their workflow-group card per web-ui-coherence
    Rule 2.  When ``cfg`` is None, the ``workflow_group`` key is
    omitted from the dict (the Issue's own ``workflow_group`` field
    is still honoured if pre-tagged).
    """
    out = []
    for i in issues:
        d = {"severity": i.severity, "message": i.message,
             "where": i.where}
        # An Issue may pre-tag its workflow_group; if not, derive
        # from the where field via the config dataclass metadata.
        group = i.workflow_group
        if group is None and cfg is not None:
            group = resolve_workflow_group(i.where, cfg)
        if group is not None:
            d["workflow_group"] = group
        out.append(d)
    return out


# --------------------------------------------------------------------- #
#  JSON <-> Structure  (canonical Modify body shape, also reusable for  #
#  any future endpoint that takes "xyz + per-atom metadata" arrays)     #
# --------------------------------------------------------------------- #


def _stated_periodicity(per: Any) -> Dict[str, Any]:
    """Read what a caller STATED about the box, from a ``periodicity`` block.

    THE ONE READER of that block.  Its four names were spelled out at three
    separate call sites, which is the only defect there was here: no one place
    owned the set, so a fifth field would have had to be added three times.

    UNKNOWN KEYS ARE IGNORED, and that is not an oversight.  The block arrives
    from `Structure.to_wire`, which sends the stated values BESIDE the server's
    own derived answers (`resolved_cell`, `resolved_cell_origin`,
    `resolved_vacuum`) so a page can show the box as it will be used -- and
    MolView keeps that block verbatim and hands the whole thing back.  Reading
    the names we set and leaving the rest is what makes that work.

    A stricter reader was tried on 2026-08-04 and reverted the same day.  The
    reasoning was that `apply_metadata_dict` REFUSES an unknown key, so this
    should too -- but that reader is on the SIDECAR path, and the sidecar is a
    FILE.  web-api.md § 1: "the sidecar carries `schema_version` because a file
    outlives the program that wrote it.  The wire does not: client and server
    ship together."  An unrecognised key here is our own client disagreeing
    with our own server in the same build -- a defect to fix in development,
    not a runtime condition to turn into a 400 for the user.

    (No "kgrid" either: k-grid is a SAMPLING knob on SiestaConfig /
    TransportConfig, not geometry -- structure-periodicity.md.  One sent here is
    simply not read.)
    """
    if not isinstance(per, dict):
        return {}
    out: Dict[str, Any] = {}
    if per.get("cell") is not None:
        out["cell"] = per["cell"]
    # cell_origin (§ 3c) rides with the cell so a modify op on an electrode
    # junction doesn't drop the corner that makes the box wrap the atoms.
    if per.get("cell_origin") is not None:
        out["cell_origin"] = per["cell_origin"]
    if per.get("axis_kind") is not None:
        out["axis_kind"] = tuple(per["axis_kind"])
    if per.get("vacuum") is not None:
        out["vacuum"] = tuple(per["vacuum"])
    return out


def _struct_from_envelope(env: Dict[str, Any]) -> Structure:
    """The inverse, and the same rule: ``Structure.from_dict`` is the ONE
    deserialiser, and it validates through the same ``__post_init__`` a freshly
    built Structure runs -- so a malformed envelope is refused here rather than
    becoming a half-built structure downstream.

    ``source_index`` -- present when the envelope describes a SUBSET of a larger
    structure -- is the CALLER's bookkeeping for mapping an answer back onto the
    structure the subset came from.  The receiver answers about the atoms it was
    given and has no use for it, so it is ignored rather than rejected.
    """
    if not isinstance(env, dict):
        raise ValueError("'structure' must be an object")
    # WHAT THE ENVELOPE IS, checked by membership.  A key outside this set is a
    # fact the sender believes it transmitted -- refused rather than dropped.
    # `applyOp` shipped `regions` and `periodicity` at the TOP level for weeks;
    # `from_dict` reads them from `metadata`, so every geometry edit came back
    # with its labels and its cell silently gone, at HTTP 200.
    known = {"title", "elements", "positions", "atom_names", "residue_ids",
             "residue_names", "chain_ids", "metadata",
             "source_index",      # the CALLER's map back onto a larger structure
             "document"}          # outbound only; a request's is ignored
    stray = sorted(k for k in env if k not in known)
    if stray:
        raise ValueError(
            f"structure carries {stray!r}, which the envelope does not define "
            f"(known: {sorted(known)!r}).  Metadata belongs under 'metadata'.")
    if not env.get("elements"):
        raise ValueError("structure.elements must be a non-empty list")
    if not isinstance(env.get("positions"), list):
        raise ValueError("structure.positions must be a list of [x, y, z]")

    return Structure.from_dict(env)


def struct_from_body(body: Dict[str, Any]) -> Structure:
    """Reconstruct a Structure from the canonical body shape::

        {
          "xyz":            "<xyz string>",
          "atom_names":     [...],   # optional; len == n_atoms
          "residue_ids":    [...],   # optional
          "residue_names":  [...],   # optional
          "chain_ids":      [...],   # optional
          "title":          "..."    # optional
        }

    A metadata list is honoured only when its length matches the atom
    count; otherwise the default from ``Structure.from_xyz``
    (atom_names = elements, residue_ids = [1]*n, residue_names =
    ["MOL"]*n, chain_ids = ["A"]*n) is kept so a malformed metadata
    array can't corrupt the result.

    Raises ``ValueError`` when the xyz field is missing or empty;
    callers turn that into an HTTP 400 with the standard error shape.

    Construction goes through a single ``Structure(...)`` call so all
    invariants in ``Structure.__post_init__`` (parallel-array length
    checks, dtype coercion of positions) fire on the final shape --
    NOT via post-construction ``setattr`` that bypasses the contract.

    OR THE ENVELOPE (web-api.md § 1, "The request envelope")::

        {"structure": {"elements": [...], "positions": [[x, y, z], …],
                       "metadata": {"regions": {...}, "cell": [...], ...}}}

    FLAT, with the per-atom facts beside the atoms and everything else under
    ``metadata``.  This said ``geometry`` wrapped the elements and positions --
    a shape ``_struct_from_envelope`` REFUSES, because ``geometry`` is not in
    its known-key set and a stray key fails the whole body.  Written to the
    letter, this docstring produced a 400.  It is also exactly what
    ``molview``'s ``structureForServer`` emits, which is the only shape any
    caller should be sending.

    which is the shape every door is being brought to: the atoms as NUMBERS, and
    the facts beside them, so a caller holding coordinates never has to write a
    coordinate document to ask a question about them.

    **Which shape a body is: it carries a ``structure`` key, or it does not.**
    That is the whole test.  A body carrying BOTH is a caller mid-migration --
    the envelope wins and the legacy fields are ignored entirely, never merged,
    because merging lets a stale field silently override a fresh one.
    """
    envelope = body.get("structure")
    if not isinstance(envelope, dict):
        raise ValueError(
            "no 'structure' provided: a structure crosses in the envelope "
            "(web-api.md § 1) -- {elements, positions, metadata}")
    return _struct_from_envelope(envelope)


def atoms_list(struct: Structure) -> List[Dict[str, Any]]:
    """Build the per-atom payload list — the same shape
    ``/api/selection/atoms`` returns.

    Used by every response that carries a Structure so the front-end's
    selection store stays in sync with the in-memory geometry without
    a separate fetch.  Pre-2026-06-07, modifier-op responses lacked
    this and the selection panel went stale after every Delete / Add /
    Orient / etc — the disk hadn't changed yet so the
    ``/api/selection/atoms`` re-fetch returned pre-op atoms.

    Each row:

        {
            "index":         int,
            "element":       "C" | "H" | ...,
            "x": float, "y": float, "z": float,   # COORDS -- the atom carries its
                                                  # own geometry (workspace-contract
                                                  # §1.2.1); no string re-parse
            "regions":       [str, ...],     # EVERY label the atom carries,
                                             # reserved ones (`frozen`) included --
                                             # one representation, so the panel
                                             # cannot render the same fact twice
            "atom_name":     "CA" | ...,     # optional, PDB-derived
            "residue_name":  "ALA" | ...,    # optional
            "chain_id":      "A"   | ...,    # optional
        }
    """
    n = len(struct.elements)
    atom_to_regions: Dict[int, list] = {}
    regions = getattr(struct, "regions", {}) or {}
    for label, idxs in regions.items():
        for idx in idxs:
            atom_to_regions.setdefault(idx, []).append(label)

    atom_names    = struct.atom_names    or []
    residue_names = struct.residue_names or []
    chain_ids     = struct.chain_ids     or []
    positions     = struct.positions

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        # Coordinates ride ON the atom (workspace-contract.md §1.2.1 -- the atom is
        # the geometric truth, not a re-parsed xyz string).
        pos = positions[i]
        row: Dict[str, Any] = {
            "index":     i,
            "element":   struct.elements[i],
            "x":         float(pos[0]),
            "y":         float(pos[1]),
            "z":         float(pos[2]),
            "regions":   atom_to_regions.get(i, []),
        }
        if i < len(atom_names)    and atom_names[i]:
            row["atom_name"]    = atom_names[i]
        if i < len(residue_names) and residue_names[i]:
            row["residue_name"] = residue_names[i]
        if i < len(chain_ids)     and chain_ids[i]:
            row["chain_id"]     = chain_ids[i]
        rows.append(row)
    return rows


def workspace_payload(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The canonical wire shape for every endpoint that returns a
    ``Structure``.

    Per :doc:`protocols/workspace-state` § 4.4, this is the single
    serialiser that supersedes the four hand-rolled jsonify blobs
    that historically lived inside ``/api/build/load``,
    ``/api/build/molecule``, ``/api/modify/*``, and
    ``/api/selection/atoms``.  Adding a new field that every
    consumer should see (``bonds``, ``dipole``, …) is a one-line
    change here, applied to every endpoint at once.

    The shape (always present):

    .. code-block:: python

        {
            "text":          "<xyz/pdb bytes>",
            "source_format": "xyz" | "pdb",
            "title":         "<title or empty string>",
            "n_atoms":       <int>,
            "atoms":         [<per-atom row>, ...],   # per atoms_list
            "lattice":       [[...3...], [...3...], [...3...]] | None,
            "issues":        [<Issue JSON>, ...],
            "extra":         { ... endpoint-specific add-ons ... },
        }

    Endpoint-specific add-ons (``backend_used``,
    ``add_hydrogens_mode``, ``pdb``, ``summary``, …) belong in
    ``extra``.  The dispatcher
    on the client reads the canonical keys and treats ``extra`` as
    opaque metadata — extending ``extra`` does not break any
    consumer.

    Notes
    -----
    * ``source_format`` defaults to ``"xyz"`` because
      :class:`molbuilder.structure.Structure` round-trips through
      :meth:`Structure.to_xyz`.  PDB-emitting endpoints set
      ``source_format="pdb"`` via ``extra`` plus a ``"text"``
      override at the callsite (see Phase 2 migration in
      :doc:`protocols/workspace-state` § 6).
    * ``lattice`` is always ``None`` here, and NOT because the structure
      has no cell -- it has one.  ``Structure`` grew ``cell`` /
      ``cell_origin`` / ``axis_kind`` / ``vacuum``, and they travel in
      the ``periodicity`` block that :func:`structure_to_dict` takes
      from ``struct.to_wire()``, together with the resolved views.
      ``lattice`` is the older single-field spelling that no consumer
      reads; it stays for the wire shape's sake.  (This note used to
      say Structure "carries geometry only" and that a future cell
      field would land here -- it landed elsewhere, on purpose:
      one block, so a cell cannot half-arrive.)  This helper is the
      one place to add it.
    * ``issues`` is populated via :func:`validate_geometry` — the
      same set the modify-tab response array already exposed.
      Callers that don't want the validation pass (e.g. a
      throughput-sensitive path that already validated) pass
      ``extra={"issues_skipped": True}`` and override ``issues``
      via ``extra`` if needed.
    """
    return {
        "text":          struct.to_xyz(),
        "source_format": "xyz",
        "title":         struct.title or "",
        "n_atoms":       struct.n_atoms,
        "atoms":         atoms_list(struct),
        "lattice":       None,
        # H1 2026-06-14: ``cfg=None`` explicit (not implicit
        # default) so the missing-cfg case is documented at the
        # call site.  ``validate_geometry`` emits only ``where=
        # "struct.*"`` issues -- no engine config field is in
        # scope here -- so workflow_group enrichment correctly
        # short-circuits to None.  A future refactor that moves
        # engine-config validation upstream of this helper MUST
        # pass cfg= or the per-card fan-out (web-ui-coherence
        # Rule 2) silently drops engine issues into the residual
        # panel.  Pinned by ``test_workflow_group_wire_contract``
        # at the wire side; this comment documents WHY this site
        # is fine.
        "issues":        issues_to_json(
            validate_geometry(struct), cfg=None),
        "extra":         dict(extra) if extra else {},
    }


def structure_to_dict(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The canonical-plus-legacy serialised shape for any
    Structure-returning endpoint.

    Routes through :func:`workspace_payload` for the canonical
    keys (``text``, ``source_format``, ``title``, ``n_atoms``,
    ``atoms``, ``lattice``, ``issues``, ``extra``) and adds the
    legacy aliases that the existing modify-tab front-end's
    ``applyStructure(r)`` reads directly (``xyz``, ``elements``,
    flat per-atom columns, ``n_residues``).

    The optional ``extra`` dict (Phase 2 addition, 2026-06-07)
    threads endpoint-specific keys (``pdb``, ``summary``,
    ``backend_used``, ``add_hydrogens_mode``) into BOTH places at
    once:

    * At the top level of the returned dict, for back-compat with
      every existing JS consumer that reads them off the response
      root (Phase 1-3 clients).
    * In the canonical ``extra`` sub-dict, where the Phase 4+
      workspace dispatcher will read them after the client
      migration completes.

    Top-level ``extra`` keys override the canonical defaults — a
    caller emitting ``source_format="pdb"`` from a PDB-parsing
    endpoint replaces the canonical default of ``"xyz"`` at both
    the root level and inside ``extra``.

    One :func:`validate_geometry` pass per call (issues are read
    from the workspace payload; not recomputed in this helper).
    """
    extras = dict(extra) if extra else {}
    base = workspace_payload(struct, extra=extras)
    # The drift-prone block -- the full `periodicity` (raw cell/origin + the
    # server-RESOLVED cell/origin via the ONE resolver) + `annotations` + the
    # identity columns -- is assembled by the Structure itself
    # (`struct.to_wire()`, structure-authority.md § 3.2).  This helper no longer
    # hand-lists a single metadata field or re-runs a resolver, so a field added
    # to `metadata_to_dict` rides onto every endpoint automatically and the
    # `cell_origin` drop-on-repack bug cannot recur.  The web layer only adds its
    # OWN concerns (render `atoms`, `issues`, `text`/`xyz`, `extra`).
    wire = struct.to_wire()
    return {
        # THE ENVELOPE (web-api.md § 1) is the structure's OWN canonical dict --
        # not a wire shape assembled here. `to_dict` is the one serialiser the
        # sidecar, the persistence layer and the CLI already round-trip through,
        # and its rule is that nobody outside the class picks a structure apart,
        # because that is where a field goes missing (`cell_origin` did). So a
        # field added to the structure reaches the wire with no edit in this file.
        #
        # It sits BESIDE the keys below rather than replacing them, and both are
        # derived from this one Structure, so they cannot come to disagree. The
        # legacy keys go when nothing reads them -- a question the code can answer.
        "structure":     struct.to_dict(),
        # Canonical keys (forward-compat with workspace dispatcher).
        "text":          base["text"],
        "source_format": base["source_format"],
        "title":         base["title"],
        "n_atoms":       base["n_atoms"],
        "atoms":         base["atoms"],
        "lattice":       base["lattice"],
        # Structure-owned: full periodicity (incl. resolved_cell/_origin) +
        # annotations ride with the geometry into the store so a captured
        # electrode cell survives the modify op (workspace-contract.md §4.0).
        "periodicity":   wire["periodicity"],
        "annotations":   wire["annotations"],
        "issues":        base["issues"],
        "extra":         base["extra"],
        # Legacy aliases for existing modify-tab consumers (identity columns
        # also sourced from the ONE view so they can't diverge).
        #
        # WHICH OF THESE STILL HAVE A READER, asked 2026-08-03 because the note
        # above says "the legacy keys go when nothing reads them -- a question
        # the code can answer", and nobody had asked it:
        #
        #   `xyz`           -- LIVE.  The generators' answer: modify/structure/
        #                      {peptide,name,dna,rna,smiles}.js read `body.xyz`.
        #   `residue_names` -- LIVE.  MolView folds it in as a parallel array
        #                      (`model-jobs.js::structureFromServer`) because the
        #                      atoms do not carry it.
        #   the other five  -- NO reader, in any client or on the Python side.
        #                      `elements` looks read, but every hit is MolView's
        #                      OWN shape coming back out of `applyOp` /
        #                      `getStructure`, not this key.
        #
        # They stay anyway, and deleting them is NOT cleanup: the envelope was
        # "added not swapped" on purpose (tests/test_structure_envelope_protocol
        # .py), and `test_a_response_carries_the_envelope_beside_todays_keys`
        # guards each one by name.  Retiring them RETIRES THAT TRANSITION -- a
        # decision, not a tidy-up, and one that also has to say what happens to
        # the guard that both views agree.
        "xyz":           base["text"],
        "elements":      wire["elements"],
        "atom_names":    wire["atom_names"],
        "residue_ids":   wire["residue_ids"],
        "residue_names": wire["residue_names"],
        "chain_ids":     wire["chain_ids"],
        "n_residues":    wire["n_residues"],
        # Endpoint-specific keys at the top level for back-compat
        # with existing JS consumers.  Phase 4+ readers go through
        # ``extra`` instead.
        **extras,
    }


def ok_structure_response(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
):
    """Build a Flask jsonify response for any Structure-returning
    endpoint.

    Phase 2 of the workspace-state migration (2026-06-07) —
    ``/api/build/load`` + ``/api/build/molecule`` + every
    ``/api/modify/*`` route through this helper instead of
    hand-rolling their own jsonify blob.  The optional ``extra``
    dict carries per-endpoint add-ons:

    * ``/api/build/load``: ``{"pdb", "summary", "source_format"}``
      (``source_format`` overrides the canonical XYZ default with
      the actually-parsed format).
    * ``/api/build/molecule``: ``{"pdb", "summary",
      "backend_used", "add_hydrogens_mode"}``.
    * ``/api/modify/<op>``: no ``extra`` keys — the client CLEARS
      the selection on any atom-count change (molview.md § 11.1,
      "Effect on atom count"), so no per-op selection remap is emitted.

    Wraps :func:`structure_to_dict` (which routes through
    :func:`workspace_payload`) in ``{"ok": True, ...}``.

    EVERY STRUCTURE LEAVING FOR THE BROWSER IS CHECKED HERE, and this is the
    only place it can be done once (structure-periodicity.md § 8.1).  The EIGHT
    ``/api/modify/*`` ops plus the two build doors return through this helper,
    and until 2026-08-01 none of the eight ran the gate at all: deleting the atom
    that held a clearance, or translating the structure out of an explicit box,
    changed nothing anyone was told about.  (``/api/modify/meta`` is the ninth
    route and not an op -- a GET of the dropdown enums, no structure either way.)

    Note what does NOT need doing here.  In the DERIVED regime the cell is a
    computed view -- ``resolve_cell`` builds it from the bounding box and the
    vacuum on every read -- so it follows the atoms by construction and there is
    nothing to regenerate.  An EXPLICIT cell is returned verbatim, which is the
    whole point of it, and is exactly the case where moving atoms can put them
    outside.  So the op runs, the cell answers for itself, and the check reports.

    ``extra["notices"]`` is the RECEIPTS slot, and it is kept ahead of what is
    computed here for the order the cell door already uses: what the edit did
    first, what is now true after it (molview.md § 6.8).  No caller passes it
    today -- every op's answer is conditions only -- but the merge is not
    decoration: without it, the assignment below would drop a caller's receipts
    without a word, which is the failure this whole helper exists to make
    impossible.
    """
    said = list((extra or {}).get("notices") or [])
    # THE ONE LINE (cell-plan.md § 6a): resolve once, check once, report.
    #
    # No try/except, because there is nothing to catch: these are the
    # loading/modifying doors, and § 8.2 says they REPORT a bad box rather than
    # refusing it -- so they ask the checker directly instead of calling the
    # raising gate and reconstructing a notice from the exception. That
    # reconstruction is what dropped the finding's id: it rebuilt the dict by
    # hand from ``str(exc)``, so the front end received a message it could not
    # identify, and tests had to match on the prose.
    _rc, issues = resolve_and_check(struct)
    said.extend(notices_for_report(issues))
    merged = dict(extra or {})
    if said:
        merged["notices"] = said
    return jsonify({"ok": True, **structure_to_dict(struct, extra=merged or None)})


def err(msg: str, code: int = 400):
    """Standard error response shape for the modify routes."""
    return jsonify({"ok": False, "error": msg}), code


def finite_float(name: str, value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float or raise ``ValueError`` with
    a request-facing message.  Used by the /api/modify/* float fields
    so a JSON body that passes ``"nan"`` / ``"inf"`` (or a stringified
    huge number that parses but breaks downstream geometry) gets
    rejected at the boundary instead of silently producing a
    NaN-coordinate structure.

    Returns ``default`` when ``value`` is None or "" -- mirrors the
    ``body.get(field, default)`` pattern already in the route
    handlers.
    """
    if value is None or value == "":
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must be a finite number; got {value!r}") from exc
    if not math.isfinite(f):
        raise ValueError(f"{name!r} must be finite; got {value!r}")
    return f


# --------------------------------------------------------------------- #
#  JSON -> dataclass coercion (Build's SIESTA / PySCF config; reusable  #
#  for any future dataclass-driven Modify endpoint, e.g. M5 electrode)  #
# --------------------------------------------------------------------- #


def dataclass_to_form_schema(cls, id_prefix: str) -> Dict[str, Any]:
    """Build a JSON form-rendering schema from an L1 config dataclass.

    Closes the last Principle-#1 anti-pattern: the SIESTA + PySCF
    form fields in ``web/templates/index.html`` and the per-field
    parse logic in ``viewer.js`` used to duplicate the dataclass
    field set (~50 fields each side).  This generator walks
    ``dataclasses.fields(cls)`` ONCE and emits everything the JS
    renderer needs to construct the form -- so adding a new field
    is now a one-line metadata change on the dataclass.

    Schema shape::

        {
          "config":    "SiestaConfig",
          "id_prefix": "p",
          "sections":  [
            {"name": "System", "fields": [<field_schema>, ...]},
            ...
          ],
        }

    Per-field shape (subset; only the relevant keys for the field's
    inferred kind are present)::

        {
          "name":     "<dataclass field name>",         # canonical key
          "id":       "<id_prefix>-<id_suffix>",        # HTML id
          "label":    "<human label>",                  # from metadata.label
          "help":     "<help / tooltip>",
          "default":  <JSON-serialisable default>,
          "tier":     "basic" | "advanced",
          "kind":     "checkbox" | "int" | "number" | "text"
                      | "select" | "tri-select" | "int-triple",
          # number / int:
          "min": ..., "max": ..., "step": ...,
          # select / tri-select:
          "choices": [...],
          "null_option": True,
          "null_label":  "<label for the empty option>",
          # int-triple (kgrid):
          "labels": ["x", "y", "z"],
          # display:
          "unit": "Å" | "Ry" | ...,
          "pattern": "<HTML pattern attr>",
        }

    **Opt-in via ``section``**: only fields whose metadata declares a
    ``"section"`` key are exposed.  Fields without a section live on
    the dataclass for the Python API / CLI but stay off the web form
    (psml paths, write_forces always-on flags, MD-only knobs that
    only matter for relax_type=Verlet, etc.).

    **ID override via ``id_suffix``**: by default the HTML id is
    ``"{id_prefix}-{field_name.replace('_', '-')}"``.  A few fields
    have shorter legacy ids (e.g. ``p-temperature`` for
    ``electronic_temperature``); they declare ``"id_suffix"`` so the
    compatibility engine + sessionStorage list stay backwards-
    compatible.

    **Section ordering**: by default sections appear in the order
    their first field is declared in the dataclass.  When the class
    declares an ``_form_section_order`` class attribute (a tuple /
    list of section names), sections appear in that order instead;
    any sections present in field metadata but missing from the
    explicit order get appended after the explicit ones in
    declaration order.  This lets a dataclass control form layout
    without reorganising the (often natural) field declaration
    order.
    """
    hints = typing.get_type_hints(cls)
    sections_in_order: List[str] = []
    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for f in fields(cls):
        section = f.metadata.get("section")
        if not section:
            continue
        if section not in by_section:
            sections_in_order.append(section)
            by_section[section] = []
        by_section[section].append(_field_to_schema(f, hints, id_prefix))

    # Explicit section ordering via _form_section_order on the class.
    # Names not present in that list keep their declaration-order
    # position (appended after the explicit names).
    declared_order = getattr(cls, "_form_section_order", None)
    if declared_order:
        seen = set()
        ordered: List[str] = []
        for name in declared_order:
            if name in by_section and name not in seen:
                ordered.append(name)
                seen.add(name)
        for name in sections_in_order:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        sections_in_order = ordered

    # Per-section descriptions (optional class attribute).  When a
    # class declares ``_form_section_descriptions: Dict[str, str]``,
    # each section's schema entry picks up its blurb and the form
    # renderer surfaces it below the legend.  Sections missing from
    # the dict get no description (omitted from the output) so this
    # is opt-in per class.
    descriptions = getattr(cls, "_form_section_descriptions", {}) or {}

    def _section_entry(s: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"name": s, "fields": by_section[s]}
        desc = descriptions.get(s)
        if desc:
            entry["description"] = desc
        return entry

    return {
        "config":    cls.__name__,
        "id_prefix": id_prefix,
        "sections": [_section_entry(s) for s in sections_in_order],
    }


def _field_to_schema(f: dataclasses.Field,
                     hints: Dict[str, Any],
                     id_prefix: str) -> Dict[str, Any]:
    """One dataclass field -> one schema entry.

    Pure inspection: no I/O, no side effects, only field.type +
    field.metadata + field.default.  Optional[X] unwraps to X with
    ``optional=True`` so the renderer knows to emit an empty/auto
    sentinel option.
    """
    ann = hints.get(f.name, f.type)
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    is_optional = (origin is typing.Union and type(None) in args)
    if is_optional:
        ann = next((a for a in args if a is not type(None)), str)
        origin = typing.get_origin(ann)
        args   = typing.get_args(ann)

    md = dict(f.metadata)
    id_suffix = md.get("id_suffix", f.name.replace("_", "-"))
    out: Dict[str, Any] = {
        "name":     f.name,
        "id":       f"{id_prefix}-{id_suffix}",
        "label":    md.get("label", f.name.replace("_", " ").capitalize()),
        "help":     md.get("help", ""),
        "default":  _serialize_default(f, ann),
        "optional": is_optional,
        "tier":     md.get("tier", "basic"),
    }
    if "unit" in md:
        out["unit"] = md["unit"]
    # Source-of-truth tag: the actual engine keyword this UI field
    # writes into the generated input.  Surfaced next to the form label
    # so the user can map UI -> generated script -> engine manual /
    # error messages without guessing.  Optional metadata key; fields
    # without an obvious 1:1 keyword mapping (kgrid as a block, etc.)
    # leave it unset and the UI shows no tag.
    if "engine_key" in md:
        out["engine_key"] = md["engine_key"]
    # Workflow-group tag (2026-06-13): one of "system" / "stage" /
    # "budget".  Drives the .workflow-group--<kind> card wrappers
    # in form-schema.js + STAGE_PRESETS' restricted write surface
    # in viewer.js.  Fields without this tag render bare (outside
    # any workflow-group wrapper) and STAGE_PRESETS never touches
    # them.  See docs/web/results.md + the
    # .workflow-group framework in lib/form-schema.css.
    if "workflow_group" in md:
        out["workflow_group"] = md["workflow_group"]

    choices = md.get("choices")
    if choices is not None:
        out["kind"] = "select"
        out["choices"] = list(choices)
        # An Optional[str] with explicit choices needs an empty
        # sentinel option in the UI (e.g. the dispersion select
        # whose "none" choice maps to None).
        if is_optional:
            out["null_option"] = True
            out["null_label"] = md.get("null_label", "(default)")
    elif ann is bool:
        if is_optional:
            # Optional[bool] -> tri-select (auto / true / false).
            # Today only parallel_over_k uses this pattern.
            out["kind"] = "tri-select"
            out["choices"] = ["auto", "true", "false"]
        else:
            out["kind"] = "checkbox"
    elif ann is int:
        out["kind"] = "int"
        rng = md.get("range")
        if rng is not None:
            out["min"], out["max"] = rng
        if is_optional:
            out["null_option"] = True
            out["null_label"] = md.get("null_label", "(auto)")
    elif ann is float:
        out["kind"] = "number"
        # step="any" is the HTML "accept any float"; widgets can
        # override with metadata["step"] when they want spinner steps.
        out["step"] = md.get("step", "any")
        rng = md.get("range")
        if rng is not None:
            out["min"], out["max"] = rng
        if is_optional:
            out["null_option"] = True
            out["null_label"] = md.get("null_label", "(auto)")
    elif origin is tuple and args:
        # Tuple[int, int, int] -- kgrid + Transport's k_mesh_transverse.
        # Renderer emits three side-by-side number inputs with sub-ids
        # f"{id}-{labels[i]}".  We pass the labels through so the
        # k-grid UI's "kx / ky / kz" stays declaration-driven.
        out["kind"] = "int-triple"
        out["labels"] = list(md.get("triple_labels", ("x", "y", "z")))
    elif origin in (list, tuple) and args and args[0] is float:
        # Variable-length List[float] -- Transport's bias_voltages_v.
        # No fixed-arity widget makes sense; render as text and let
        # the user enter a comma-separated list.  ``coerce_to_field_type``
        # parses it back into List[float] before the dataclass sees it.
        out["kind"] = "comma-floats"
    elif (origin in (list, tuple) and args
          and dataclasses.is_dataclass(args[0])):
        # List[<dataclass>] — PySCFConfig.stages is the first such
        # case (per-stage rows for the in-script staged optimization,
        # task #534).  Emit ``kind: "stage-table"`` plus the per-row
        # field shape so the JS renderer (commit 3) can lay out a
        # table without knowing the dataclass at compile time.
        elem_cls = args[0]
        out["kind"] = "stage-table"
        out["stage_fields"] = _stagespec_to_field_schemas(elem_cls)
    elif ann is str:
        out["kind"] = "text"
    else:
        # Sequence[str] (species_order) etc. -- not exposed in the
        # form today.  Fall back to text so the schema is at least
        # well-formed for tests, but the field shouldn't have a
        # section anyway.
        out["kind"] = "text"

    if "pattern" in md:
        out["pattern"] = md["pattern"]
    return out


def _serialize_default(f: dataclasses.Field, ann: Any = None) -> Any:
    """JSON-friendly default for the schema.

    Tuples become lists for JSON compatibility.  When the field uses
    ``default_factory`` (Transport's ``bias_voltages_v: List[float]``
    is the first such case), call the factory so the form shows the
    actual default — without this, the form opens with a blank input
    and the user can't see what the production-tuned default is.
    ``ann`` is the already-resolved type hint (per ``hints`` in
    ``_field_to_schema``); used to decide whether a list default
    should serialize as a comma-string (``List[float]`` for the
    comma-floats text input) or stay a list (``List[int]`` for the
    int-triple renderer).
    """
    if f.default is not dataclasses.MISSING:
        v = f.default
    elif f.default_factory is not dataclasses.MISSING:    # type: ignore[misc]
        try:
            v = f.default_factory()                       # type: ignore[misc]
        except Exception:
            return None
    else:
        return None
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, list):
        args = typing.get_args(ann) if ann is not None else ()
        origin = typing.get_origin(ann) if ann is not None else None
        # List[float] -> comma-string (comma-floats text input
        # pre-populates from this).  List[int] -> keep as list (no
        # variable-length int field exposed in the form today, but
        # the contract stays JSON-friendly).
        if origin in (list, tuple) and args and args[0] is float:
            return ", ".join(repr(x) for x in v)
        # List[<dataclass>] (PySCFConfig.stages) — serialize each
        # row to a plain dict so the JSON envelope is faithful.
        # ``asdict`` is recursive (handles nested dataclasses) and
        # keeps the field order from the dataclass declaration.
        if (origin in (list, tuple) and args
                and dataclasses.is_dataclass(args[0])):
            return [dataclasses.asdict(item) for item in v]
        return list(v)
    return v


def _stagespec_to_field_schemas(elem_cls) -> List[Dict[str, Any]]:
    """Walk ``dataclasses.fields(elem_cls)`` and emit a per-field
    schema entry the JS ``stage-table`` renderer (commit 3) lays
    out as one table column per row of the parent ``stages`` field.

    Per-row schema is intentionally smaller than the outer
    ``_field_to_schema`` shape — no HTML ``id`` (the JS composes
    per-stage ids), no ``optional``/``null_*`` (rows are dense),
    no ``section`` / ``tier``.  Carries the minimum the renderer
    needs to pick an input widget + label it: ``name``, ``kind``,
    ``label``, ``help``, ``default``, ``unit?``, ``step?``,
    ``min``/``max``?, ``pattern``?, ``engine_key``?.

    Type-driven dispatch mirrors ``_field_to_schema``: ``bool`` ->
    checkbox, ``int`` -> int, ``float`` -> number, ``str`` -> text.
    Anything fancier raises ``TypeError`` — the JS renderer doesn't
    know how to render select / triple / etc. inside a row, and a
    silent fallback would surface as a broken UI.
    """
    # ``elem_cls`` may use ``from __future__ import annotations`` so
    # ``sf.type`` arrives as a *string* ("'float'", "'bool'", ...);
    # resolve via ``typing.get_type_hints`` for the real classes.
    hints = typing.get_type_hints(elem_cls)
    out: List[Dict[str, Any]] = []
    for sf in dataclasses.fields(elem_cls):
        md = dict(sf.metadata)
        row: Dict[str, Any] = {
            "name":    sf.name,
            "label":   md.get("label", sf.name),
            "help":    md.get("help", ""),
            "default": (sf.default
                        if sf.default is not dataclasses.MISSING
                        else None),
        }
        if "unit"       in md: row["unit"]       = md["unit"]
        if "step"       in md: row["step"]       = md["step"]
        if "pattern"    in md: row["pattern"]    = md["pattern"]
        if "engine_key" in md: row["engine_key"] = md["engine_key"]
        rng = md.get("range")
        if rng is not None:
            row["min"], row["max"] = rng
        t = hints.get(sf.name, sf.type)
        if t is bool:
            row["kind"] = "checkbox"
        elif t is int:
            row["kind"] = "int"
        elif t is float:
            row["kind"] = "number"
            row.setdefault("step", "any")
        elif t is str:
            # `str` with a ``choices`` enum -> dropdown widget; the
            # plain string variant stays as a text input.  #534 6a
            # introduced the first stage-table choice field
            # (``on_nonconvergence``); the JS renderer dispatches on
            # ``kind: "choice"`` to render a ``<select>`` with the
            # provided ``choices`` tuple.
            if "choices" in md:
                row["kind"] = "choice"
                row["choices"] = list(md["choices"])
            else:
                row["kind"] = "text"
        else:
            raise TypeError(
                f"_stagespec_to_field_schemas: field {elem_cls.__name__}."
                f"{sf.name} has unsupported type {t!r} for a stage-table "
                f"row.  Supported: bool / int / float / str.")
        out.append(row)
    return out


def coerce_to_field_type(field: dataclasses.Field, value: Any,
                         resolved_hints: Dict[str, Any]) -> Any:
    """Convert a JSON-arriving value to the field's declared type.

    The form layer can deliver number-typed fields as strings ("300"
    rather than 300) when the request comes from a non-browser HTTP
    client (the in-tree JS frontend coerces with parseFloat/parseInt
    so the test path is fine).  Without coercion, the dataclass
    happily stores the string, downstream the validator's range check
    raises ``TypeError`` on ``string < int`` and the validator-pass
    swallows it as a "skip this validator", quietly losing the
    out-of-range warning.

    Coercion respects ``Optional[X]`` (the empty string and ``None``
    pass through as ``None``).  ``bool`` accepts the JSON literal True
    / False as well as the strings ``"true"`` / ``"false"`` / ``"1"`` /
    ``"0"`` (case-insensitive).  Tuple-typed fields like ``kgrid``
    fall through to per-element int coercion.

    Unknown / unhandled types pass through untouched -- the dataclass
    constructor sees what the caller sent.

    Coercion failures (TypeError / ValueError) propagate to the
    caller so the endpoint can surface them as an error-severity
    Issue rather than HTTP 400.
    """
    ann = resolved_hints.get(field.name, field.type)
    origin = typing.get_origin(ann)
    args   = typing.get_args(ann)
    is_optional = (origin is typing.Union and type(None) in args)
    if is_optional:
        if value is None or value == "":
            return None
        ann = next((a for a in args if a is not type(None)), str)
        origin = typing.get_origin(ann)
        args   = typing.get_args(ann)

    if ann is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)
    if ann is int:
        return int(value)
    if ann is float:
        return float(value)
    if ann is str:
        return str(value)
    # Tuple[int, int, int] (kgrid is the only such field today).
    if origin is tuple and args:
        if not isinstance(value, (list, tuple)):
            return value
        elem_t = args[0]
        return tuple(elem_t(v) for v in value)
    # Sequence[str] (species_order in SiestaConfig) -- accept either
    # a comma-string or an already-list value.
    if origin in (list, tuple) and args and args[0] is str:
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value
    # Sequence[int] (frozen_indices / es_explicit_indices) -- accept
    # comma-separated indices with optional range syntax
    # "0-35, 100, 150-200" -> [0,1,...,35, 100, 150,...,200].  Used by
    # the Spectra tab's frozen-atom + L4 explicit-mode lists.
    if origin in (list, tuple) and args and args[0] is int:
        if isinstance(value, str):
            return _parse_int_list_with_ranges(value)
        if isinstance(value, (list, tuple)):
            # Already a sequence; coerce each element to int.  Reject
            # element-wise rather than silently truncating floats so
            # bad input surfaces.
            return [int(v) for v in value]
        return value
    # Sequence[float] (Transport's bias_voltages_v) -- accept a comma-
    # separated string ("0.0, 0.5, 1.0") or an already-list value.
    # Without this branch the form layer's text-input string passes
    # through unchanged and the dataclass stores it as a str, which
    # then fails downstream when the engine slices ``bias[0]``.
    if origin in (list, tuple) and args and args[0] is float:
        if isinstance(value, str):
            return [float(s.strip()) for s in value.split(",") if s.strip()]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return value
    # List[<dataclass>] (PySCFConfig.stages — task #534).  Form
    # payload arrives as a list of plain dicts; rebuild each row by
    # passing the dict as kwargs to the dataclass constructor.
    # Missing keys fall back to the dataclass's per-field defaults
    # (so a partial UI update — e.g. only changing ``enabled`` —
    # doesn't blank the other knobs).  Items already of the right
    # type pass through unchanged.  Per-field type coercion
    # (string "1e-9" -> float 1e-9 for number inputs) mirrors the
    # scalar branches above; the JS sends typed values but a
    # non-browser HTTP client may send strings.
    if (origin in (list, tuple) and args
            and dataclasses.is_dataclass(args[0])):
        elem_cls = args[0]
        if not isinstance(value, (list, tuple)):
            # Non-list payloads on a List[<dataclass>] field are a
            # wire-format error (a dict on cfg.stages would silently
            # crash validate_stages with AttributeError on
            # ``s.enabled``; #534 commit 5c).  Reject at the
            # boundary so the API surfaces a clean 400 instead of a
            # 500 / corrupt downstream state.
            raise TypeError(
                f"cannot coerce {type(value).__name__} to "
                f"list of {elem_cls.__name__}: expected list, got "
                f"{value!r}"
            )
        # ``from __future__ import annotations`` makes
        # ``sf.type`` arrive as a string; resolve via
        # ``typing.get_type_hints`` for the per-field coerce
        # dispatch below.
        elem_hints = typing.get_type_hints(elem_cls)
        spec_fields = {
            sf.name: (sf, elem_hints.get(sf.name, sf.type))
            for sf in dataclasses.fields(elem_cls)
        }
        out = []
        for item in value:
            if isinstance(item, elem_cls):
                out.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError(
                    f"cannot coerce {type(item).__name__} to "
                    f"{elem_cls.__name__}: expected dict")
            kwargs: Dict[str, Any] = {}
            for k, v in item.items():
                pair = spec_fields.get(k)
                if pair is None:
                    continue   # unknown keys silently ignored
                _sf, _t = pair
                kwargs[k] = _coerce_scalar(_t, v)
            out.append(elem_cls(**kwargs))
        return out
    # Anything else: pass through.
    return value


def _coerce_scalar(t: Any, value: Any) -> Any:
    """Per-row scalar coercion for ``List[<dataclass>]`` items.

    Mirrors the top-level dispatch in :func:`coerce_to_field_type`
    but stays focused on the inner-row primitive types that the
    ``stage-table`` renderer emits: bool / int / float / str.
    Coercion failures raise so the caller can surface a typed
    error rather than silently storing a string in a float field.
    """
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)
    if t is int:
        return int(value)
    if t is float:
        return float(value)
    if t is str:
        return str(value)
    return value


def _parse_int_list_with_ranges(s: str):
    """Parse ``"0-35, 100, 150-200"`` -> ``[0, 1, ..., 35, 100, 150, ..., 200]``.

    Each comma-separated token is either a bare integer or
    ``<lo>-<hi>`` (inclusive on both ends).  Whitespace around
    commas / hyphens is tolerated.  Empty tokens (trailing comma)
    are skipped.

    Raises ``ValueError`` with the offending token so the caller can
    surface it as a typed error to the user.
    """
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            # Negative-prefix support would be ambiguous with the
            # range separator; indices are 0-based so negatives don't
            # need to be supported here.
            lo_s, _, hi_s = tok.partition("-")
            try:
                lo = int(lo_s.strip())
                hi = int(hi_s.strip())
            except ValueError:
                raise ValueError(
                    f"could not parse index range {tok!r}; "
                    f"expected '<int>-<int>'"
                )
            if hi < lo:
                raise ValueError(
                    f"index range {tok!r} is empty (hi < lo)"
                )
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(tok))
            except ValueError:
                raise ValueError(
                    f"could not parse index {tok!r}; "
                    f"expected an integer"
                )
    return out


def config_from_params(cls, params: Dict[str, Any],
                       hints: Dict[str, Any],
                       none_sentinels: Tuple[str, ...] = ()):
    """Build a dataclass instance from a JSON-style params dict.

    Walks dataclass fields, picks the matching key from ``params``,
    coerces to the field's declared type via ``coerce_to_field_type``,
    and constructs the dataclass.

    ``none_sentinels``: per-field rule for "this string means None"
    (e.g. ``("solvent", "auxbasis", "dispersion")`` for PySCFConfig
    where the form sends an empty string for "leave default").
    """
    by_name = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for k, v in params.items():
        f = by_name.get(k)
        if f is None:
            continue
        # Form-sentinel "empty string -> None / drop" handling for
        # specific Optional fields the JS deliberately blanks out.
        if k in none_sentinels and (v == "" or v is None):
            kwargs[k] = None
            continue
        # Backwards-compat: JS sometimes sends "none" for "no
        # dispersion".  Same treatment as the empty-string case.
        if k == "dispersion" and isinstance(v, str) and v.strip().lower() == "none":
            kwargs[k] = None
            continue
        # net_charge: empty string from the form means "auto-detect"
        # (don't pass the kwarg so the dataclass default of None
        # kicks in and render_fdf falls back to the phosphate
        # heuristic).
        if k == "net_charge" and (v == "" or v is None):
            continue
        # Coercion failures (TypeError / ValueError) propagate to the
        # endpoint, which surfaces them as an error-severity Issue
        # rather than HTTP 400 -- so the UI renders the same panel
        # for parse-failure as for validator-failure.
        kwargs[k] = coerce_to_field_type(f, v, hints)
    return cls(**kwargs)


class PeriodicityRefused(Exception):
    """The gate REFUSED the periodicity a request carried.

    Not a warning about a box: a state that cannot be represented at all -- a
    left-handed cell, or one too small to hold the structure whatever origin it
    is given (periodicity_gate, "Errors vs notices").  The user has to change
    something before the request can be answered, which is what a 400 means.

    It exists so that answer cannot be forgotten.  ``validate_periodicity``
    raises ``ValueError``, and every door that runs it on the way IN had to
    remember a try/except.  SIX OF THE SEVEN DID NOT -- only the Cell-page door
    handled it -- so a refusable cell arrived as a 500 and an HTML error page,
    which the browser's ``r.json()`` then reported as a network failure, hiding
    the real message the gate had written.  Raising a
    type ONE handler in ``web/app.py`` knows about means a seventh door inherits
    the right answer instead of inheriting the omission.  (Same reasoning, same
    file, as the 413 handler beside it.)
    """


def checked_periodicity(struct):
    """Run the gate and let a refusal become the door's 400.

    The one wrapper both entry paths use -- ``periodicity_checked_for_emit``
    below and the export door -- so neither owns a copy of the translation.
    Returns the gate's ``(struct, notices)`` unchanged.
    """
    try:
        return validate_periodicity(struct)
    except ValueError as exc:
        raise PeriodicityRefused(str(exc)) from exc


def apply_periodicity_only(struct, body):
    """Apply what the caller STATED about the box, and check nothing.

    ``body["periodicity"]`` is the block :meth:`Structure.to_wire` sends, and
    it has TWO halves that read alike and behave nothing alike:

      * ``cell`` / ``cell_origin`` / ``axis_kind`` / ``vacuum`` -- what the
        caller STATED.  Applied verbatim.  An absent block means the
        Structure's own defaults (isolated, vacuum unset).
      * the ``resolved_*`` answers the server computed and sent back so a page
        can show the box as it will be USED.  They arrive here because MolView
        keeps the block verbatim and hands the whole thing back; they are not
        read, which is all that needs to happen to them.

    WHICH DOORS TAKE THIS BLOCK, and why only they:

      * ``/api/build/load`` (text branch) and ``struct_from_body``'s legacy
        ``xyz`` branch.  Those bodies carry NO envelope -- the structure is a
        file or a paste -- so a stated block is the only way to say what the box
        is.  One key, one door, nothing to rank.

    Every other door takes the envelope, where the cell rides in
    ``structure.metadata`` and reaches the Structure through
    ``Structure.from_dict``.  Applying this block THERE gave the cell two
    sources, and the second won: an envelope stating 8 A plus a block stating
    20 A emitted 20 (fixed 2026-08-04; see
    :func:`periodicity_checked_for_emit`).  The rule the whole envelope exists
    for is that a structure crosses ONCE -- web-api.md § 1.

    There is deliberately NO disk-sidecar fallback: a request reflects the model
    the user is looking at, and inferring state from what happens to be in the
    request is the exact failure that severed this wire on 2026-06-14 (the
    label-presence branch silently skipping the sidecar's cell).

    THE APPLYING AND THE JUDGING ARE TWO STEPS, because the same bad box has
    two right answers depending on what the request is FOR (user decision,
    2026-08-03):

      * a request that GENERATES something you would run -- an .fdf, a PySCF
        script, a transport or spectra job, an exported document -- is REFUSED.
        Those parameters have to be right; there is no point emitting a
        calculation nobody can trust.
      * a request that LOADS or MODIFIES a structure carries on, and the
        problem is REPORTED with the answer, so the user can look at it, fix it
        in the Cell page, and be checked again.  Refusing here would leave a
        structure with a bad box unfixable through the UI -- you could not even
        open it to correct it.

    So this half applies; :func:`periodicity_checked_for_emit` adds the refusal
    for the first kind of door, and :func:`ok_structure_response` reports for
    the second.
    """
    stated = _stated_periodicity(body.get("periodicity"))
    if stated:
        for field, value in stated.items():
            setattr(struct, field, value)
        struct.__post_init__()          # the ONE validator, on the final shape
    return struct


def periodicity_checked_for_emit(struct):
    """REFUSE a bad box -- the emitting doors.  Checks; applies nothing.

    Returns the CHECKED structure -- the same object the gate was given, since
    the gate corrects nothing (clause 1: a resolved value is never written
    back).  Callers rebind so this stays the one seam every emitted structure
    passes through rather than an optional check.  A refusable cell raises
    :class:`PeriodicityRefused`, which the app turns into the door's 400.

    WHY IT NO LONGER APPLIES ANYTHING (2026-08-04).  This ran
    ``apply_periodicity_only(struct, body)`` first, which reads a TOP-LEVEL
    ``body["periodicity"]`` and writes it over the structure -- so an emit
    request had TWO places to say what the box was, and the second one won:
    an envelope stating an 8 A cell plus a top-level block stating 20 A
    emitted 20.  Measured, not inferred.

    That is the cell wearing the shape the LABELS wore until the day before
    (#41): two sources, silently ranked, with no rule that could work -- "the
    envelope stated no cell" and "the envelope stated a different cell" are the
    same input to any precedence rule you can write.  No emit caller was
    sending the top-level key (the tabs send ``molview.exportFile()``, whose
    cell rides in ``metadata``), so nothing was being mis-emitted; a reader for
    a shape nobody sends is exactly how the label version stayed invisible.

    ``apply_periodicity_only`` REMAINS, and is right, on ``/api/build/load``:
    that door takes a structure as TEXT, so there is no envelope and a stated
    block is the only way to say what the box is.  One key, one door, no rank.
    """
    checked, _conditions = checked_periodicity(struct)
    return checked


def apply_sidecar_if_possible(struct, structure_path):
    """Best-effort .molstruct.json sidecar application.

    Sets ``struct.regions`` -- the whole label store -- from the
    sidecar next to ``structure_path`` so engine preflight + emitters
    see the real boundary-condition data.  Cross-blueprint helper:
    used by /api/build/fdf, /api/build/pyscf, and
    /api/spectra/render to honor the contract that /modify's
    boundary-conditions edits flow through to the generators
    (design.md "Sidecar-driven boundary conditions").

    Returns
    -------
    None
      Success (sidecar applied) OR clean no-op (no sidecar, or
      structure_path empty).
    str
      User-facing notice when the sidecar EXISTS but couldn't be
      applied (path rejected, malformed JSON, atom-count mismatch).
      Caller surfaces as a preflight warn-severity Issue so the
      user knows their sidecar didn't take effect.

    Failure to apply is non-fatal because the caller might be
    running against a re-cropped or hand-pasted structure -- the
    form's own freeze/region fields stay authoritative.

    Moved 2026-05-26 from web/blueprints/spectra.py to _shared.py
    because Build's /api/build/fdf + /api/build/pyscf endpoints also
    need to apply the sidecar; importing from spectra blueprint
    created an asymmetric cross-blueprint dependency.
    """
    from .files import _resolve_within_roots, _PickerError
    from molbuilder.sidecars import molstruct as _molstruct_json
    if not structure_path:
        return None
    try:
        resolved = _resolve_within_roots(structure_path)
    except _PickerError as exc:
        return (f"sidecar lookup skipped: structure_path rejected "
                f"({exc.message}); the form's freeze rules are the "
                f"sole boundary condition for this run.")
    if not resolved.exists():
        return (f"sidecar lookup skipped: {resolved.name!s} not on "
                f"disk; the form's freeze rules are the sole "
                f"boundary condition for this run.")
    if resolved.suffix.lower() not in (".xyz", ".pdb"):
        return None
    # In-body wins over sidecar (bundle-contract.md § 4.2): if a
    # same-stem .fdf / .py companion sits next to the structure file
    # AND carries an ATOM-METADATA block, those labels are the
    # authoritative record (the script was the basis of an actual
    # run; the .molstruct.json sidecar is editable in isolation and
    # may have drifted).  apply_companion_labels_if_present mutates
    # struct in place and returns a marker string when it applied;
    # the sidecar branch is skipped in that case.
    _companion_said = []
    if apply_companion_labels_if_present(
            struct, resolved, notices=_companion_said) is not None:
        # Same channel the sidecar branch below uses: a string here becomes a
        # warn-Issue in the caller's preflight.  ALL of them, joined -- taking
        # only the first would be the silent drop this whole path exists to
        # end, and would go unnoticed for as long as there happened to be one.
        return (" ".join(n["message"] for n in _companion_said)
                if _companion_said else None)
    sidecar_path = _molstruct_json.sidecar_path_for(resolved)
    if not sidecar_path.exists():
        return None
    try:
        sidecar_data = _molstruct_json.load(sidecar_path)
        _molstruct_json.apply_to_structure(struct, sidecar_data)
    except _molstruct_json.MolstructJsonError as exc:
        return (f"sidecar at {sidecar_path.name} could not be "
                f"applied ({exc}); the form's freeze rules are "
                f"the sole boundary condition for this run.  "
                f"Re-export the sidecar from the Structure tab to "
                f"re-enable sidecar-driven divergence checks.")
    return None


def apply_companion_labels_if_present(struct, structure_path, *, notices=None):
    """Same-stem ``.fdf`` / ``.py`` companion next to a ``.xyz`` /
    ``.pdb`` wins over a ``.molstruct.json`` sidecar as the label
    source.  See ``docs/execution/job-contracts.md`` and § 5.3.

    Why companion-wins: the script was the actual basis of a
    molbuilder-generated run.  Its in-body ATOM-METADATA block is
    written by the same generator pass that wrote the engine body,
    so the labels and the coordinates cannot drift apart.  The
    ``.molstruct.json`` sidecar, by contrast, is editable in
    isolation and may have been mutated after the script was
    generated.  When both exist, the script is the truth.

    Parameters
    ----------
    struct
        Mutated in place: ``struct.regions``
        are populated when the companion carries them.
    structure_path
        ``Path``-like.  Resolved already by the caller; this
        function does NOT re-validate against the picker roots
        (caller's job).

    Returns
    -------
    None
        No companion present, OR companion present but carrying no
        ATOM-METADATA block.  Caller falls through to the sidecar
        branch.
    str
        ``"applied:fdf"`` or ``"applied:py"`` — labels were applied
        from the named companion.  Caller skips the sidecar branch.
    """
    from molbuilder import script_emit as _sc
    from pathlib import Path as _Path
    base = _Path(structure_path)
    # Iterate by priority: .fdf first (SIESTA / transport are the
    # canonical workflows), .py second (PySCF).
    for ext, marker in ((".fdf", "applied:fdf"), (".py", "applied:py")):
        companion = base.with_suffix(ext)
        if not companion.exists():
            continue
        try:
            text = companion.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # An unreadable companion is not a hard error -- fall
            # through to sidecar.  The caller surfaces sidecar
            # outcomes as warn-Issues; an unreadable companion
            # behaves the same as "no companion present".
            continue
        _said = []
        if _sc.apply_inbody_atom_metadata(struct, text, notices=_said):
            # The version warning rides back with the marker, so the caller can
            # surface it.  Without this it would die here -- the labels would
            # be recovered and the user would never learn they came from an
            # older layout, which is the SAME silence the recovery exists to
            # end (delivery contract R5).
            if _said and notices is not None:
                notices.extend(_said)
            return marker
    return None


def regions_pattern_b_notice(struct, engine_label: str):
    """Three-stage Pattern B (sidecar-contract.md § 6 B): every
    engine that DOESN'T consume the structure's region labels MUST
    explicitly notice them so the user can re-direct to Transport
    when a junction structure was meant.  Pre-task-#308 this block
    was inlined verbatim in /api/build/fdf and /api/build/pyscf;
    each new engine that joins the cohort (Spectra, future
    transport-but-not-NEGF) would re-implement it or — worse —
    silently absorb the labels.

    Returns
    -------
    None
      ``struct.regions`` is empty; nothing to notice.
    Issue
      ``info``-severity, ``where='config.regions'``, message
      enumerates the labels + names the engine + points at the
      Transport tab.  Caller appends to its issues list and
      returns alongside the rendered script / fdf.

    ``engine_label`` is what the user sees in the message — e.g.
    ``"the .fdf"``, ``"the PySCF script"``, ``"the spectra deck"``.
    Keep the phrasing engine-specific so the notice reads
    naturally.

    Added 2026-06-09 (task #308) as the dedupe of the two
    copy-pasted blocks in build.py.  Test coverage:
    tests/test_web.py::test_{fdf,pyscf}_surfaces_info_when_structure
    _carries_regions.
    """
    from molbuilder.issues import Issue
    if not struct.regions:
        return None
    region_labels = sorted(struct.regions.keys())
    return Issue(
        "info",
        ("Structure carries region labels ("
         + ", ".join(region_labels)
         + ") but a Structure-Optimization run does not consume "
         "them — they are reserved for Transport.  Generating "
         + engine_label + " here is OK if you only want "
         "SCF / relaxation / spectroscopy; for a transport "
         "calculation, switch to the Transport tab."),
        "config.regions",
    )


__all__ = [
    "atoms_list",
    "issues_to_json",
    "struct_from_body",
    "structure_to_dict",
    "ok_structure_response",
    "workspace_payload",
    "err",
    "finite_float",
    "coerce_to_field_type",
    "config_from_params",
    "apply_sidecar_if_possible",
    "PeriodicityRefused",
    "checked_periodicity",
    "apply_companion_labels_if_present",
    "regions_pattern_b_notice",
]

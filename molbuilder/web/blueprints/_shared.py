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
)
from molbuilder.validation import validate_geometry
from molbuilder.cell import resolve_and_check
from molbuilder.periodicity_gate import (notices_for_report,
                                         validate_periodicity)



# --------------------------------------------------------------------- #
#  Issues                                                                #
# --------------------------------------------------------------------- #


def resolve_workflow_group(where: str, cfg) -> Optional[str]:
    """Return the workflow-group binding for an Issue ``where`` field.

    Per docs/web/ui-contract.md Rule 2, validator findings
    should attach to the workflow-group card whose fields they
    concern: a ``config.mesh_cutoff`` finding belongs in the Stage
    card; ``config.spin_treatment`` belongs in the Run profile card;
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
        # The stage label rides BESIDE ``where``, never inside it
        # (engines/stages.md § 4 R2): the same check produces the same id
        # whether it fired for a single run or for a stage, and the UI binds
        # behaviour to the id.  Omitted when absent so a single-run response
        # is byte-identical to what it was before ladders existed.
        if i.stage is not None:
            d["stage"] = i.stage
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
             "info",              # the free-form NON-structural store
                                  # (structure-info-plan.md; from_dict reads it)
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
        return list(v)
    return v



# --------------------------------------------------------------------- #
#  The form schema, built from the CATALOGUE                            #
#                                                                       #
#  `web/form-schema.md` § 1: the catalogue is the source of truth.  The  #
#  presentation does not change -- the JS renderer already takes         #
#  whatever schema it is handed -- so this is one function pointed at a  #
#  different source, not a new UI.                                       #
# --------------------------------------------------------------------- #

#: How a template `type` becomes a control (§ 1.1's derived column).
_CONTROL_FOR_TYPE = {
    "bool":    "checkbox",
    "int":     "int",
    "pow2":    "int",          # an int with a constraint the validator holds
    "float":   "number",
    "str":     "text",
    "text":    "text",
    "int3":    "int-triple",
    "float3":  "float-triple",
    # A list renders as a TEXT input holding a comma-separated value, which
    # is what `comma-floats` already is and what ``coerce_to_field_type``
    # already parses back (`Sequence[str]`, `Sequence[float]`).  No new
    # control kind: the renderer has one for this shape already.
    "strlist": "text",
    "intlist": "text",
}



def _control_for(item) -> str:
    """Which widget renders this item.  `choices` wins: an enum is a select
    whatever its underlying type, and a tri-select is an OPTIONAL bool."""
    if item.choices:
        return "select"
    if item.type == "bool" and item.optional:
        return "tri-select"
    return _CONTROL_FOR_TYPE.get(item.type, "text")


def catalogue_to_form_schema(engine: str, id_prefix: str = "p",
                             calculation: str = "optimization",
                             ) -> Dict[str, Any]:
    """The Build form's schema for *engine*, from the catalogue.

    **The two grouping axes** (`form-schema.md` § 1.3), both carried by every
    item and answering different questions:

    * ``group`` -- *when do I set this?* -- is the OUTER card, unchanged since
      2026-06-13, and load-bearing: it exists because the stage selector once
      silently rewrote budget and system fields.
    * ``category`` -- *what question about the calculation is this?* -- is the
      legend INSIDE the card, and it replaces the per-engine free-text
      ``section``.  The six are shared, so SIESTA and PySCF show the same inner
      headings for the first time.

    Sections come out in § 6.2's reading order, which is the order the closed
    vocabulary is declared in -- not alphabetical, and not the order the items
    happen to sit in the file.
    """
    from molbuilder import template as _T

    parsed = _T.catalogue()
    items = _T.select(parsed, engine=engine)

    # ``staging`` is a PANEL THIS SURFACE DOES NOT HAVE.  The stage token is a
    # real parameter -- it reaches the generated script, and the template
    # carries it -- but it is answered by the staging surface, not typed here
    # (user, 2026-08-15: *"no staging related setup at all"*).  Filtered by the
    # item's own declaration rather than by a name this file would have to
    # keep: a second such parameter needs no edit here.
    items = [it for it in items if it.group != "staging"]
    # The form serves ONE calculation kind (`template.md` § 6.3's sibling
    # rule): an item another kind owns stays out by its own declaration.
    # P0 hardcoded "optimization" here; P2 threads the caller's kind --
    # the vibration form is the same renderer over the same catalogue.
    items = [it for it in items
             if not it.calculations or calculation in it.calculations]


    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        panel = it.category[0] if it.category else "procedure"
        by_category.setdefault(panel, []).append(_item_to_field(it, id_prefix))

    sections = [{"name": cat, "title": cat.capitalize(),
                 "fields": by_category[cat]}
                for cat in _T.CATEGORIES if cat in by_category]

    # A stage ladder is NOT here, and that is correct: `stages.md` § 1.1 makes
    # it the user's decision about what VARIES, and it lives in ``task.json``.
    # The catalogue carries parameters; how a ladder is set up is its own
    # design conversation (user, 2026-08-14).
    return {"config": engine, "id_prefix": id_prefix, "sections": sections}


def engine_key_for(item) -> str:
    """How this item is spelled for the engine — the string a SURFACE shows.

    **One writer, because two surfaces disagreed.**  The parameter form used
    this precedence; the task-setup column chooser (`build.py`) read
    ``item.anchor`` directly and published it under the same JSON name.  An
    ``anchor`` is DERIVED — ``_bare_anchor`` takes the leading token of
    ``engine_key`` and nothing checks that the token is a keyword — so for an
    item whose ``engine_key`` leads with a VALUE the column chooser showed the
    value: ``method`` appeared as ``RKS`` (one of its four choices) and
    ``optimizer`` as ``geomeTRIC``, while the form showed the full spelling.

    The order is the honest one: the full spelling if the item has it, then
    the keywords a ``deck`` item expands to, and the bare anchor only when
    there is nothing better (`template.md` § 5).
    """
    if getattr(item, "engine_key", ""):
        return item.engine_key
    if getattr(item, "expands", ()):
        return " + ".join(item.expands)
    return getattr(item, "anchor", "") or ""


def _item_to_field(item, id_prefix: str) -> Dict[str, Any]:
    """One catalogue item as one form field (`form-schema.md` § 1.1)."""
    out: Dict[str, Any] = {
        "name":     item.name,
        "id":       f"{id_prefix}-{item.name.replace('_', '-')}",
        "label":    item.label or item.name.replace("_", " ").capitalize(),
        "help":     item.help,
        "default":  (list(item.default) if isinstance(item.default, tuple)
                     else item.default),
        "optional": item.optional,
        "tier":     item.tier or "basic",
        "kind":     _control_for(item),
    }
    if item.unit:
        out["unit"] = item.unit
    if item.pattern:
        out["pattern"] = item.pattern
    if item.group:
        out["workflow_group"] = item.group
    # The engine-keyword badge.  **The FULL spelling, not the anchor.**
    #
    # This read `item.anchor` from 2026-08-14 until 2026-08-15, and an anchor
    # is deliberately the bare leading keyword (`template.md` § 5) -- so the
    # badge said `gto.M` on four different controls, `mf` on three more, and
    # nothing at all on the eleven whose engine_key is a molbuilder note
    # rather than a keyword.  That note is the only way a reader learns the
    # setting never reaches the deck, which `web/form-schema.md` § 1a requires
    # always be present.  `expands` remains the fallback for a `deck` item
    # whose several keywords are the honest answer.
    spelled = engine_key_for(item)
    if spelled:
        out["engine_key"] = spelled
    if item.choices:
        out["choices"] = list(item.choices)
    elif out["kind"] == "tri-select":
        # An Optional[bool] has three states and the renderer walks
        # ``f.choices`` to build them; they are the CONTROL's vocabulary, not
        # the item's, so the catalogue does not carry them (§ 5's `choices` is
        # an enum's members).  Today only `parallel_over_k` is one.
        out["choices"] = ["auto", "true", "false"]
    if item.range:
        out["min"], out["max"] = item.range
    if out["kind"] in ("int", "number"):
        out["step"] = "1" if item.type in ("int", "pow2") else "any"
    if out["kind"] in ("int-triple", "float-triple"):
        out["labels"] = ["x", "y", "z"]
        # A triple gets a step too (2026-08-15).  It was emitted only for the
        # SCALAR kinds, so the renderer had to pick one itself -- and a bound
        # or a step chosen in the renderer is a second place for the rule to
        # live.  `min`/`max` are already set above from `item.range` and
        # apply PER COMPONENT for a triple: `kgrid` bounds each axis count,
        # not their product.
        out["step"] = "1" if item.type == "int3" else "any"
    if item.optional:
        out["null_option"] = True
        out["null_label"] = item.null_label or "(auto)"
    if item.refs:
        # RESOLVED here, server-side, from the one bibliography
        # (molbuilder/references.py) -- the form shows a real title and
        # DOI, never a bare key.  An unknown key is silently omitted
        # HERE because the test suite is where it must fail
        # (tests/test_catalogue_refs.py); the form is not the CI.
        from molbuilder.references import citation_for
        cites = [c for c in (citation_for(k) for k in item.refs) if c]
        if cites:
            out["refs"] = cites
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
    ``"0"`` (case-insensitive).  A Tuple-typed field like ``kgrid`` takes
    a list OR the comma text a person types (``"4,4,1"``, ``"4x4x1"``,
    ``"4 4 1"`` -- the spellings ``--kgrid`` itself takes), and coerces
    per component.

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
    # Tuple[int, int, int] (kgrid) and Tuple[float, float, float]
    # (kgrid_displacement, Transport's k_mesh_transverse).
    #
    # A COMMA STRING PARSES, for the same reason the Sequence[*] branches
    # just below accept one: a non-browser client sends the text a person
    # would type.  Until 2026-08-25 this was the one sequence branch that
    # did not -- `if not isinstance(value, (list, tuple)): return value`
    # handed the string straight back, while the docstring above claimed it
    # "falls through to per-element int coercion".  A POST carrying
    # `kgrid: "4,4,1"` therefore stored a str in a `Tuple[int, int, int]`
    # field, and the range check downstream could only report it as a
    # programmer bug.
    #
    # A value that is neither a string nor a sequence RAISES here rather
    # than passing through: TypeError is what this function's contract
    # promises the caller for a value it cannot make, and the endpoint
    # turns it into an error Issue.  The LENGTH is not checked -- that is
    # `_validate_kgrid`'s sentence to pass, and it says it better.
    if origin is tuple and args:
        elem_t = args[0]
        if isinstance(value, str):
            value = [s for s in re.split(r"[,\sx]+", value.strip()) if s]
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
    # Anything else: pass through.
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


# (apply_sidecar_if_possible / apply_companion_labels_if_present /
#  regions_pattern_b_notice retired 2026-08-21, C-shared: the emitting
#  doors take the ENVELOPE now -- regions ride structure.metadata through
#  Structure.from_dict -- so the read-a-sidecar-file-beside-the-path flow
#  lost its last caller, and the Pattern-B notice re-homed into the
#  validators (validation/sidecar.py::check_unconsumed_region_labels),
#  where it runs on every deck route instead of two deleted endpoints.)

__all__ = [
    "atoms_list",
    "issues_to_json",
    "struct_from_body",
    "structure_to_dict",
    "ok_structure_response",
    "workspace_payload",
    "err",
    "finite_float",
    "catalogue_to_form_schema",
    "coerce_to_field_type",
    "config_from_params",
    "PeriodicityRefused",
    "checked_periodicity",
]

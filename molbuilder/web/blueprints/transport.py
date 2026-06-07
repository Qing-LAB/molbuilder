"""Transport-calculation blueprint.

Form-schema endpoint for the ``/transport-calculation`` tab.
Engine implementations (TranSIESTA, PySCF-NEGF) and the
script-render endpoint follow in a later phase; today's surface
is just the form so the user can configure parameters even
though Generate is gated.

Routes:

    GET  /api/transport/schema       form-rendering schema for
                                     the TransportConfig dataclass

Mirrors the contract of Build's ``/api/build/schema/<engine>`` and
Spectra's ``/api/build/schema/spectra``: the dataclass field
metadata is the single source of truth; the JS renders the form
directly from the returned schema; no field declarations are
duplicated in the HTML template.
"""

from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, render_template

from ._shared import dataclass_to_form_schema as _dataclass_to_form_schema

from molbuilder.config.transport import TransportConfig


bp = Blueprint("transport", __name__)


# ===================================================================== #
# /api/transport/schema  --  form schema endpoint                       #
# ===================================================================== #


@bp.route("/api/transport/schema", methods=["GET"])
def api_transport_schema() -> Any:
    """Return the TransportConfig form schema.

    Section order is taken from ``TransportConfig._form_section_order``
    so the workflow-order (System → Geometry → Electrodes →
    Transmission → NEGF → Runtime) is stable independent of field-
    declaration order in the dataclass.

    Sidecar-driven defaults (electrode/bridge regions seeding the
    Geometry section) follow the Spectra pattern in a later
    iteration — engine implementations come first.  Today the
    endpoint returns the static schema only.
    """
    schema = _dataclass_to_form_schema(TransportConfig, "t")
    response: Dict[str, Any] = {"ok": True, "schema": schema}
    return jsonify(response)

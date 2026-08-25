"""Bench blueprint -- a whole sweep on one page.

Routes:

    GET /api/bench/summary?path=<a sweep's job-set.json>
                                    the sweep: every trial, what it asked
                                    for / ran / measured, where it is now,
                                    and the verdict.  Read-only and safe to
                                    poll.

Spec: ``docs/web/bench-summary.md``.

**This module is a surface, not a composer.**  All of the reading is
``summarize.sweep_view`` -- an L2 verb over the four doors § 2 names -- and
what is left here is the three things a route owns: which paths may be
read, what the HTTP failures are, and JSON.  The composition sits a layer
down because the readers it needs (``_read_environment``, ``_latest_run_file``)
are private to the module that owns them, and because a verb is testable
without a Flask app -- ``tests/test_prep_bench_fold.py`` exercises it
directly.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

bp = Blueprint("bench", __name__)


@bp.route("/api/bench/summary")
def api_bench_summary():
    """Compose one sweep from its ``job-set.json``.

    Query: ``path`` -- the sweep's ``job-set.json``.  The CALCULATION it
    belongs to is derived from it (``bundle_for_sweep_file``); the file's
    own directory is not the bundle, and using it as one finds no
    artifacts at all rather than failing.

    Errors:

    * 400 -- ``_PickerError``: outside the allowed roots, or ``..``.  The
      same fence the file picker uses, imported the way ``results.py``
      imports it, so there is ONE answer to "may this be read".
    * 404 -- no such file.
    * 400 -- the file is not a readable job-set, or is not a sweep, or its
      calculation cannot be found.
    """
    # Deferred: this blueprint is imported at startup and the jobset
    # readers pull in the parsers (the reason results.py defers its own).
    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset.summarize import bundle_for_sweep_file, sweep_view
    from .files import _PickerError, _resolve_within_roots

    raw = request.args.get("path", "")
    try:
        path = _resolve_within_roots(raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not path.is_file():
        return jsonify({"ok": False, "error": f"no such file: {raw}"}), 404

    try:
        jobset = JobSet.load(path)
    except Exception as exc:
        # The picker lists what is on disk and a person may click any of
        # it, so "you picked something that is not a job-set" is a 400 the
        # UI can show -- never a 500.
        return jsonify({
            "ok": False,
            "error": f"not a readable job-set: {type(exc).__name__}: {exc}",
        }), 400

    if getattr(jobset, "kind", "") != "sweep":
        return jsonify({
            "ok": False,
            "error": (f"{path.name} is a {jobset.kind or 'plain'} job-set, "
                      f"not a sweep -- there is no comparison to draw"),
        }), 400

    try:
        bundle = bundle_for_sweep_file(jobset, path)
        view = sweep_view(jobset, bundle)
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": f"could not read the sweep: {type(exc).__name__}: {exc}",
        }), 400

    view["ok"] = True
    return jsonify(view)

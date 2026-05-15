"""Flask app factory for the molbuilder UI.

The UI has three halves served by one process:

  * Build   at  ``GET /``         (build page; routes under
                                   ``/api/build/*`` -- see
                                   ``web/blueprints/build.py``)
  * Watch   at  ``GET /watch``    (watch page; routes under
                                   ``/api/watch/*`` -- see
                                   ``web/blueprints/watch.py``)
  * Modify  at  ``GET /modify``   (modify page; M2 = read-only
                                   inspection.  Edit ops + their
                                   ``/api/modify/*`` routes land
                                   in M3-M5 -- see
                                   ``docs/tabs/modify.md``.)

Two top-level routes stay on the app rather than on any blueprint
because all three halves consume them:

  * ``GET /api/health``    liveness
  * ``GET /api/backends``  available builder backends (used by both
                           tabs' Backend pickers)

The page templates and static assets live under ``templates/`` and
``static/``; the watch viewer's assets live under ``static/watch/``
to avoid name collisions with the build viewer.
"""

from __future__ import annotations

from flask import Flask, abort, jsonify, render_template, send_file

from ..diagnostics import initialize as _initialize_diagnostics


# Cap multipart uploads at 50 MB.  Build side only needs ~10 MB for
# realistic PDBs (10k atoms ~= 1 MB at 80 bytes/line); the watch side
# accepts trajectory log uploads up to 50 MB.  Flask's
# MAX_CONTENT_LENGTH is a single global cap, so we use the larger.
_MAX_UPLOAD_MB = 50


def create_app() -> Flask:
    # Configure the root logger so warnings from L1 modules (e.g.
    # ``molbuilder.projects.list_projects`` skipping invalid directory
    # names) reach the user.  Without this, Python's root logger has
    # no handler attached and warnings vanish silently.  Mirrors the
    # equivalent call in ``molbuilder.cli.main``.
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s: %(message)s")

    # Bind the diagnostics snapshot before any blueprint registers /
    # before any request handler runs.  All backend availability checks
    # (e.g. ``/api/backends``) then read from a consistent view of the
    # machine's envs / PATH / config.  Cheap (~50 ms once per process).
    _initialize_diagnostics()

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024

    # Build + Watch route groups live on Blueprints so each half is
    # self-contained (handlers, helpers, validation).  Both blueprints
    # use full route paths in their decorators (no url_prefix) -- the
    # paths read clearly at the call site.
    from .blueprints.build   import bp as build_bp
    from .blueprints.watch   import bp as watch_bp
    from .blueprints.modify  import bp as modify_bp
    from .blueprints.spectra import bp as spectra_bp
    app.register_blueprint(build_bp)
    app.register_blueprint(watch_bp)
    app.register_blueprint(modify_bp)
    app.register_blueprint(spectra_bp)

    # 413 Payload Too Large -- without this Flask returns its default
    # HTML 413 page, which the JS uploaders parse as ``r.json()`` and
    # crash with a misleading "Network error".  Returning the same
    # ``{ok: false, error: ...}`` JSON shape every other endpoint
    # uses gives the user an actionable message.
    from werkzeug.exceptions import RequestEntityTooLarge

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        return jsonify({
            "ok":    False,
            "error": (f"Upload exceeds the {_MAX_UPLOAD_MB} MB cap "
                      f"(MAX_CONTENT_LENGTH).  Shrink the file or "
                      f"point the loader at the path on disk."),
        }), 413

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/modify")
    def modify_page():
        # M2: read-only inspection (load XYZ/PDB, atom list, viewer
        # click-sync).  M3-M5 add edits, anchor-pair selection, and
        # the electrode panel.  The shared app-tabs nav lives in the
        # template just like /watch -- no business logic here.
        return render_template("modify.html")

    @app.route("/api/health")
    def api_health():
        from .. import __version__
        return jsonify({"ok": True, "version": __version__})

    @app.route("/api/backends")
    def api_backends():
        # `auto_name` is what dispatch(backend="auto") would pick on
        # this machine -- exposed so the UI can label the dropdown's
        # "auto" option with the resolved backend, and surface a
        # warning when the preferred (3DNA) backend isn't installed.
        from ..backends import auto_backend_name, available_backends
        return jsonify({
            "ok": True,
            "available": available_backends(),
            "auto_name": auto_backend_name(),
        })

    @app.route("/vendor/plotly.min.js")
    def vendor_plotly_js():
        """Serve plotly.min.js from the installed plotly Python
        package so the browser doesn't need a CDN at all.

        Plotly ships the JS bundle as package data; we read it via
        ``importlib.resources`` rather than reaching into
        ``plotly.__file__`` directly so this works for zip-packaged
        installs, editable installs, and any other layout
        ``importlib.resources`` supports.

        Returns 404 if the plotly Python package isn't importable
        or doesn't ship the JS bundle -- the caller (the Spectra
        page) can degrade by showing the chart-unavailable message.
        """
        from importlib import resources
        try:
            # plotly>=5 publishes the bundle at
            # ``plotly/package_data/plotly.min.js``; older versions
            # used the same path.  files() returns a Traversable,
            # which works the same for filesystem + zipfile installs.
            ref = resources.files("plotly").joinpath(
                "package_data", "plotly.min.js")
        except ModuleNotFoundError:
            abort(404)
        if not ref.is_file():
            abort(404)
        # Long-cache: the URL is version-agnostic but the file
        # changes only when the user upgrades the plotly Python
        # package; that's a fresh app start anyway.
        return send_file(
            str(ref),
            mimetype="application/javascript",
            max_age=3600,
        )

    return app

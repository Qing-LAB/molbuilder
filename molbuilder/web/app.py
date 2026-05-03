"""Flask app factory for the molbuilder UI.

The UI has two halves served by one process:

  * Build  at  ``GET /``         (build page; routes under
                                  ``/api/build/*`` -- see
                                  ``web/blueprints/build.py``)
  * Watch  at  ``GET /watch``    (watch page; routes under
                                  ``/api/watch/*`` -- see
                                  ``web/blueprints/watch.py``)

Two top-level routes stay on the app rather than on either blueprint
because both halves consume them:

  * ``GET /api/health``    liveness
  * ``GET /api/backends``  available builder backends (used by both
                           tabs' Backend pickers)

The page templates and static assets live under ``templates/`` and
``static/``; the watch viewer's assets live under ``static/watch/``
to avoid name collisions with the build viewer.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template


# Cap multipart uploads at 50 MB.  Build side only needs ~10 MB for
# realistic PDBs (10k atoms ~= 1 MB at 80 bytes/line); the watch side
# accepts trajectory log uploads up to 50 MB.  Flask's
# MAX_CONTENT_LENGTH is a single global cap, so we use the larger.
_MAX_UPLOAD_MB = 50


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024

    # Build + Watch route groups live on Blueprints so each half is
    # self-contained (handlers, helpers, validation).  Both blueprints
    # use full route paths in their decorators (no url_prefix) -- the
    # paths read clearly at the call site.
    from .blueprints.build import bp as build_bp
    from .blueprints.watch import bp as watch_bp
    app.register_blueprint(build_bp)
    app.register_blueprint(watch_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

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

    return app

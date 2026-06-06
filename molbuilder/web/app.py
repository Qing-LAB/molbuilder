"""Flask app factory for the molbuilder UI.

The UI has five tabs served by one process, each owned by a
blueprint under ``web/blueprints/``:

  * Build    at  ``GET /``         (web/blueprints/build.py)
  * Modify   at  ``GET /modify``   (web/blueprints/modify.py)
  * Spectra  at  ``GET /spectra``  (web/blueprints/spectra.py)
  * Results  at  ``GET /results``  (web/blueprints/results.py)
  * Watch    at  ``GET /watch``    (web/blueprints/watch.py; legacy
                                    after the tab-merge -- /results
                                    is the canonical post-run inspector)

Plus the file-picker + project mutations under
``web/blueprints/files.py``, the inspector partial endpoints under
``results.py`` (``/partials/trajectory-inspector`` etc.), the
optional auth surface in ``auth.py`` + ``auth_providers/``, and a
small set of app-level routes that don't fit any one blueprint:

  * ``GET /api/health``               liveness
  * ``GET /api/backends``             available builder backends
  * ``GET /vendor/plotly.min.js``     plotly JS served from the
                                       installed Python package's
                                       package_data (so /spectra
                                       and /results work offline)

The page templates live under ``templates/``; static assets under
``static/``, with per-tab subdirectories (e.g. ``static/spectra/``,
``static/watch/``) for tab-specific JS/CSS that doesn't belong in
``static/lib/`` (the shared layer).  Inspector cores -- code that
both /results and a legacy tab call into -- live under
``static/lib/<concept>/`` (e.g. ``static/lib/trajectory/core.js``).
"""

from __future__ import annotations

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file

from ..diagnostics import initialize as _initialize_diagnostics


def _maybe_install_auth(app, cfg) -> None:
    """Install the auth layer from the already-loaded ``cfg`` dict.

    Quietly no-op when the config has no ``auth`` section -- the
    localhost-only single-user shape stays auth-free.  Reads from
    the pre-loaded cfg (not the filesystem) so ``create_app(config=...)``
    can swap in a test config without touching ``molbuilder.json``.
    """
    from ..runtime_config import get_auth, get_secret_key_file
    auth_cfg = get_auth(cfg)
    if not auth_cfg:
        return  # no auth configured; nothing to do
    from .auth import init_auth
    init_auth(app, auth_cfg, get_secret_key_file(cfg))


def request_is_https() -> bool:
    """True when the current request reached us over HTTPS, either
    directly (Flask's TLS) or via a reverse proxy that set
    ``X-Forwarded-Proto``.  Used by the HSTS header decision -- we
    only send HSTS over already-HTTPS responses so a misconfigured
    plain-HTTP deploy doesn't lock browsers out."""
    if request.is_secure:
        return True
    fwd = request.headers.get("X-Forwarded-Proto", "").lower()
    return fwd == "https"


# Cap multipart uploads at 50 MB.  Build side only needs ~10 MB for
# realistic PDBs (10k atoms ~= 1 MB at 80 bytes/line); the watch side
# accepts trajectory log uploads up to 50 MB.  Flask's
# MAX_CONTENT_LENGTH is a single global cap, so we use the larger.
_MAX_UPLOAD_MB = 50


def create_app(*, config=None) -> Flask:
    """Build the molbuilder Flask app.

    Parameters:
        config: optional pre-loaded ``runtime_config`` dict.  When
                ``None`` (production default), the config is read
                from ``./molbuilder.json`` via
                :func:`molbuilder.runtime_config.read_config`.  Tests
                pass an explicit dict (often ``{}`` for the no-auth
                / no-TLS default) so they never touch the developer's
                per-machine config file -- removing a hidden coupling
                between CWD state and what the test client sees.

                This parameter is the supported injection seam; do NOT
                add other CWD-reading side effects without honouring it.
    """
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

    # Load config from disk only when the caller didn't pass one.
    if config is None:
        try:
            from ..runtime_config import read_config
            config = read_config()
        except Exception as exc:
            # Bad config is loud, not silent: refuse to start.  Same
            # pattern as the TLS resolver in cli.py.
            logging.error(
                "molbuilder: failed to read config from "
                "molbuilder.json: %s", exc,
            )
            raise

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024

    # Optional auth.  When the cfg has no ``auth`` section the call is
    # a no-op (the localhost-only default).  See molbuilder/web/auth.py
    # + docs/deployment.md for the auth-enabled shape.
    _maybe_install_auth(app, config)

    # Build + Watch route groups live on Blueprints so each half is
    # self-contained (handlers, helpers, validation).  Both blueprints
    # use full route paths in their decorators (no url_prefix) -- the
    # paths read clearly at the call site.
    from .blueprints.build     import bp as build_bp
    from .blueprints.watch     import bp as watch_bp
    from .blueprints.modify    import bp as modify_bp
    from .blueprints.spectra   import bp as spectra_bp
    from .blueprints.files     import bp as files_bp
    from .blueprints.results   import bp as results_bp
    from .blueprints.selection import bp as selection_bp
    app.register_blueprint(build_bp)
    app.register_blueprint(watch_bp)
    app.register_blueprint(modify_bp)
    app.register_blueprint(spectra_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(selection_bp)

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

    # --- Security headers ----------------------------------------- #
    #
    # Defense-in-depth headers for the internet-deployment use case
    # (see docs/deployment.md).  Each header below has a specific
    # threat it mitigates; we set the most restrictive value that
    # still lets the app work.
    #
    # NOTE: these are CODE-LEVEL defaults.  Operators putting molbuilder
    # behind a reverse proxy (the recommended deployment shape) can
    # override / add headers at the proxy layer too -- both belt and
    # suspenders are fine, headers compose.
    @app.after_request
    def _add_security_headers(response):
        # Content-Security-Policy: the heaviest one.  Default-src
        # 'self' means JS / CSS / fonts only load from molbuilder's
        # own origin.  We DON'T allow inline scripts (no script-src
        # 'unsafe-inline') because the project already audits for
        # no inline-event-handler markup -- if XSS ever lands, CSP
        # blocks the attacker's payload from loading additional
        # JS or beaconing data out.
        #
        # 'unsafe-inline' is allowed for STYLE only because 3Dmol +
        # the inspector cards use a small amount of inline style=
        # (e.g., structure-inspector's height = "420px").  Dropping
        # this would mean a wider CSS refactor; the risk delta of
        # inline style is small (CSS injection at worst, not JS).
        #
        # img-src 'self' data: covers Plotly's data-URI tiles.
        # connect-src 'self' covers our /api/* fetches.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'",
        )
        # Prevents the browser from MIME-sniffing a response away from
        # its declared Content-Type.  Cheap; no behavior change.
        response.headers.setdefault(
            "X-Content-Type-Options", "nosniff",
        )
        # Block click-jacking by refusing to render any molbuilder
        # page inside an iframe.  ``frame-ancestors 'none'`` in CSP
        # is the modern equivalent; X-Frame-Options is the legacy
        # fallback for older browsers.
        response.headers.setdefault(
            "X-Frame-Options", "DENY",
        )
        # Don't leak the full request URL to off-site links the user
        # clicks (project paths, file names) -- the molbuilder origin
        # only.
        response.headers.setdefault(
            "Referrer-Policy", "same-origin",
        )
        # Browsers respect HSTS only when delivered over HTTPS; the
        # serve-time TLS guard (docs/deployment.md) refuses non-loop-
        # back binds without --cert/--key, so this header lands the
        # moment the operator actually exposes molbuilder.  1-year
        # max-age is the standard "this is permanent" signal.
        if request_is_https():
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response

    # Phase 7 tab reorganization (Phase A, 2026-06-06): canonical
    # routes match the visible tab labels.  Legacy URLs 301-redirect
    # to their new homes so existing bookmarks and shared links keep
    # working.  See docs/tabs/architecture.md § 3 for the route table.
    #
    # The view-function bodies (the actual templates rendered) don't
    # change in Phase A.  Phase B will rebuild the Structure tab to
    # absorb Build's structure-generation paths.

    @app.route("/structure")
    def structure_page():
        # Renders the legacy modify.html template — Phase B merges
        # Build's generator paths into this tab.
        return render_template("modify.html")

    @app.route("/structure-optimization")
    def structure_optimization_page():
        # Renders the legacy index.html (Build) template — Phase B
        # strips structure-generation from this template; the form +
        # generate flow stay.
        return render_template("index.html")

    @app.route("/transport-calculation")
    def transport_calculation_page():
        # Phase D will replace this placeholder with a form skeleton
        # driven by TransportConfig.  Phase B.3 wires the engine
        # backends.
        return render_template("transport_calculation.html")

    @app.route("/")
    def index():
        # 301: legacy home was the Build tab; the new home is the
        # Structure tab per the Phase 7 reorganization.
        return redirect("/structure", code=301)

    @app.route("/modify")
    def modify_page():
        # 301: legacy /modify is now /structure (same view function,
        # new canonical URL).
        return redirect("/structure", code=301)

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

"""Flask Blueprints registered into the molbuilder web app.

The Build routes are still defined directly on the app in
``web/app.py`` (historical layout).  The blueprint modules in this
package hold the page + API routes for the other tabs:

  * ``watch.py``     -- /api/watch/* trajectory API endpoints
                        (consumed by the /results trajectory
                        inspector; the legacy /watch page route was
                        retired 2026-05-19; the module name is
                        kept for git-history continuity per the
                        2026-06-07 rename-and-no-redirect rule).
  * ``files.py``     -- /api/files/* file IO endpoints
  * ``spectra.py``   -- /spectrum-calculation page (was ``/spectra``
                        pre-2026-06-07) + /api/spectra/* endpoints
  * ``modify.py``    -- /molbuilder page (was ``/modify``
                        pre-2026-06-07) + /api/modify/* endpoints
  * ``transport.py`` -- /transport-calculation page +
                        /api/transport/* endpoints
  * ``results.py``   -- /results page + /partials/* partials +
                        /api/results/bundle (Step-3 PR-E)
  * ``selection.py`` -- /api/selection/eval + /api/selection/atoms +
                        /api/selection/save + /api/selection/save-sidecar
                        + /api/selection/refresh-hash (atom-selection rule
                        eval + atom list + sidecar I/O; Pattern C:
                        stateless, JS holds the rule tree, Python
                        canonicalises + evaluates.  Click-toggle is
                        handled client-side in the selection store)
  * ``system.py``    -- /api/system/load (server-load snapshot for
                        the bottom-strip widget, 2026-06-15)
  * ``auth.py``      -- /login + /oauth-callback/* + provider dispatch

All blueprints are registered into a single Flask app by
``web/app.py::create_app()``.
"""

from . import watch as watch  # re-export for `from .blueprints import watch`

# Adapter modules — imported for their @register_adapter side
# effects so the registry is populated before any HTTP request
# hits /api/structure/analyze.  See
# docs/protocols/scientific-validation.md § 4.3 for the import-site
# convention.
from molbuilder.siesta import auto_defaults as _siesta_auto_defaults  # noqa: F401
from molbuilder.pyscf  import auto_defaults as _pyscf_auto_defaults   # noqa: F401

__all__ = ["watch"]

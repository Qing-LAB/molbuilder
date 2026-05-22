"""Flask Blueprints registered into the molbuilder web app.

The Build routes are still defined directly on the app in
``web/app.py`` (historical layout).  The blueprint modules in this
package hold the page + API routes for the other tabs:

  * ``watch.py``    -- /api/watch/* trajectory API endpoints
                       (consumed by the /results trajectory
                       inspector; the legacy /watch page route was
                       retired 2026-05-19).
  * ``files.py``     -- /api/files/* file IO endpoints
  * ``spectra.py``   -- /spectra page + /api/spectra/* endpoints
  * ``modify.py``    -- /modify page + /api/modify/* endpoints
  * ``results.py``   -- /results page + /partials/* partials
  * ``selection.py`` -- /api/selection/eval + /api/selection/toggle
                       (atom-selection rule eval + click-toggle bookkeeping,
                       Pattern C: stateless, JS holds the rule tree, Python
                       canonicalises + evaluates)
  * ``auth.py``      -- /login + /oauth-callback/* + provider dispatch

All blueprints are registered into a single Flask app by
``web/app.py::create_app()``.
"""

from . import watch as watch  # re-export for `from .blueprints import watch`

__all__ = ["watch"]

"""Flask Blueprints registered into the molbuilder web app.

The blueprint modules in this package hold page + API routes:

  * ``watch.py``     -- /api/watch/* trajectory API endpoints
                        (consumed by the /results trajectory
                        inspector)
  * ``files.py``     -- /api/files/* file IO endpoints
  * ``spectra.py``   -- /spectrum-calculation page + /api/spectra/* endpoints
  * ``modify.py``    -- /api/modify/* endpoints
  * ``transport.py`` -- /api/transport/* endpoints
  * ``results.py``   -- /results page + /partials/* partials
  * ``selection.py`` -- /api/selection/eval + /api/selection/atoms
                        (atom-selection rule
                        eval + atom list; Pattern C:
                        stateless, JS holds the rule tree, Python
                        canonicalises + evaluates.  Click-toggle is
                        handled client-side in the selection store)
  * ``system_load.py`` -- /api/system/load (server-load snapshot for
                        the bottom-strip widget)

All blueprints are registered into a single Flask app by
``web/app.py::create_app()``.
"""

from . import watch as watch  # re-export for `from .blueprints import watch`

__all__ = ["watch"]

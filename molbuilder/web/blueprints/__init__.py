"""Flask Blueprints registered into the molbuilder web app.

The Build routes are still defined directly on the app in `web/app.py`
(historical layout). The Watch routes live in `web/blueprints/watch.py`
and are registered into the same app so a single Flask instance serves
both halves of the merged molbuilder + molwatch UI.
"""

from . import watch as watch  # re-export for `from .blueprints import watch`

__all__ = ["watch"]

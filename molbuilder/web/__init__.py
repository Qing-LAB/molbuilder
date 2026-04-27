"""Browser UI for molbuilder.

Run:
    molbuilder serve --port 8000

then open http://127.0.0.1:8000.
"""

from .app import create_app

__all__ = ["create_app"]

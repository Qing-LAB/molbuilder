"""Entry point so ``python -m molbuilder ...`` works without installing.

Mirrors the ``if __name__ == "__main__":`` block at the bottom of
``cli.py``.  Useful because the project is invoked directly from the
repo root rather than through ``pip install -e .`` — see
``docs/ops/installation.md``.
"""

import sys

from molbuilder.cli import main

sys.exit(main())

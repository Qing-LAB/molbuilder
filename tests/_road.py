"""What a test needs to walk the road with a real engine: the conda hook the
wrapper sources, and an engine env's own ``bin`` -- each through the
product's own resolvers, never a guess at the layout or the host's PATH.
"""
from __future__ import annotations

from pathlib import Path


def conda_hook() -> Path:
    """The ``conda.sh`` the wrapper's preamble sources, beside the conda the
    product detects (``<root>/condabin/conda`` or ``<root>/bin/conda`` ->
    ``<root>/etc/profile.d/conda.sh``); a path that does not exist when no
    conda is detected."""
    from molbuilder import diagnostics
    binary = diagnostics.detect().conda_binary
    if not binary:
        return Path("/nonexistent/conda.sh")
    return Path(binary).parent.parent / "etc" / "profile.d" / "conda.sh"


def env_available(name: str) -> bool:
    """Through ``Capabilities.env_available`` -- the door that knows the
    manager's own env list."""
    from molbuilder import diagnostics
    return diagnostics.detect().env_available(name)


def env_bin(name: str) -> Path:
    """The env's own ``bin``, through the product's resolver; a path that does
    not exist when the env is absent."""
    from molbuilder import diagnostics
    from molbuilder.envs.install import _env_prefix
    caps = diagnostics.detect()
    if not caps.env_available(name):
        return Path("/nonexistent")
    prefix = _env_prefix(name, caps.conda_binary)
    return Path(prefix) / "bin" if prefix else Path("/nonexistent")

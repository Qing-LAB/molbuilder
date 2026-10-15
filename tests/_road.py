"""What a test needs to walk the road with a real engine: the conda hook the
wrapper sources, and an engine env's own ``bin`` -- each through the
product's own resolvers, never a guess at the layout or the host's PATH.
"""
from __future__ import annotations

from contextlib import contextmanager
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


#: The pseudopotentials a live run needs -- H, C and O, one PseudoDojo
#: release: inputs, checked in (`tests/fixtures/psml/README.md`).
PSML_DIR = Path(__file__).resolve().parent / "fixtures" / "psml"
H_PSML = PSML_DIR / "H.psml"


def set_template_value(template: Path, name: str, value: str) -> None:
    """``[item.<name>]``'s ``value``, as a person sets it in the template."""
    text = template.read_text()
    head, sep, tail = text.partition(f"[item.{name}]")
    assert sep, f"no [item.{name}] in {template.name}"
    body, nxt, rest = tail.partition("\n[item.")
    lines = body.split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith("value = "))
    lines[at] = f"value = {value}"
    template.write_text(head + sep + "\n".join(lines) + nxt + rest)


@contextmanager
def live_siesta(tree: Path, tmp_path_factory):
    """A module's road with the real SIESTA: no read of the developer's
    config directory (`conftest.config_root_is_never_the_developers`, which
    a module fixture does not get), this box probed with the conda the
    wrapper sources (`configuration.md` § 4), the env's own ``bin`` ahead of
    the suite's stub toolchain, the working folder beside the tree, and the
    checked-in pseudopotentials -- H, C, O -- in the tree's library.  Yields
    the monkeypatch, undone on exit."""
    import os
    import shutil

    import pytest

    from conftest import write_machine_record
    mp = pytest.MonkeyPatch()
    try:
        mp.delenv("MOLBUILDER_CONFIG_DIR", raising=False)
        mp.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
        write_machine_record(env_init={
            "activation": "conda activate",
            "preamble": f"source {conda_hook()}"})
        mp.chdir(tree.parent)
        bin_ = env_bin("molbuilder-siesta")
        assert (bin_ / "siesta").is_file(), bin_
        mp.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
        (tree / "pseudopotential").mkdir(exist_ok=True)
        for psml in sorted(PSML_DIR.glob("*.psml")):
            shutil.copy(psml, tree / "pseudopotential" / psml.name)
        yield mp
    finally:
        mp.undo()

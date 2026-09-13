"""Which env runs a tool -- the routing policy (`molbuilder.envs`).

**This is not the installer.**  `run_tool` is how molbuilder runs `tleap` when
it builds a structure: the question is whether to use the `tleap` inside
`molbuilder-MDtools` or whatever `tleap` happens to be on the system.

The rule -- **the curated env wins over host PATH** -- is the one worth being
sure of.  A stray system AmberTools in `/usr/local/bin` silently produces
different chemistry from the pinned one, and nothing about the run would say
so.  An explicit `env=` beats both; host PATH is the fallback; anything else is
a refusal that names what was tried.

`route()` answers all of that and dispatches nothing, so these are ordinary
calls.  They used to build a fake manager binary, fake tools, a PATH and a log
file, and read the decision back out of the log -- because the decision was
tangled with the dispatch and there was nothing to ask.
"""

from __future__ import annotations

import pytest

from molbuilder import envs
from molbuilder.diagnostics import Capabilities, set_capabilities


def _machine_with(*env_names):
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs={n: f"/prefix/{n}" for n in env_names}))


@pytest.fixture
def tool_on_host_path(tmp_path, monkeypatch):
    """PATH holding one tool -- the only input that IS about PATH."""
    (tmp_path / "tleap").write_text("")
    (tmp_path / "tleap").chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))


@pytest.fixture
def nothing_on_host_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))


def test_the_routed_env_wins_even_when_the_tool_is_on_host_path(
        tool_on_host_path):
    """The load-bearing one: a system AmberTools must not shadow the curated
    env."""
    _machine_with("molbuilder-MDtools")
    assert envs.route("tleap") == "molbuilder-MDtools"


def test_host_path_is_the_fallback_when_the_routed_env_is_absent(
        tool_on_host_path):
    """``None`` means "run it straight from PATH" -- no conda hop."""
    _machine_with()
    assert envs.route("tleap") is None


def test_an_explicit_env_overrides_the_routing(nothing_on_host_path):
    _machine_with("custom-env")
    assert envs.route("python", env="custom-env") == "custom-env"


@pytest.mark.parametrize("tool,env,envs_present,expected", [
    # routed, but the env is not on this machine and the tool is not on PATH
    ("tleap", None, (), "routed env"),
    # not routed at all, and not on PATH
    ("definitely-not-a-real-tool", None, (), "no conda env routing"),
    # an explicit env that does not exist
    ("tleap", "nonexistent-env", ("something-else",), "does not exist"),
])
def test_a_refusal_names_what_was_tried(nothing_on_host_path, tool, env,
                                        envs_present, expected):
    """A caller cannot see the candidates, so the message has to."""
    _machine_with(*envs_present)
    with pytest.raises(FileNotFoundError, match=expected):
        envs.route(tool, env=env)


def test_run_in_env_refuses_when_no_manager_was_found():
    """Nothing can be dispatched into an env without one, and the message
    names the recorded-manager door rather than a stack trace."""
    set_capabilities(Capabilities(runtime_config={}, conda_binary=None,
                                  conda_envs={}))
    with pytest.raises(RuntimeError, match="conda CLI not found"):
        envs.run_in_env("my-env", ["echo"])

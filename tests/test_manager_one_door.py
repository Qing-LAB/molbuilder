"""One package manager, one door -- resolution is a recorded machine fact.

The user's ruling (2026-08-21, after ASU Sol resolved the wrong
manager): the CLI must find and use the manager through ONE framework
door -- ``envs.manager`` in molbuilder.json first (the recorded fact),
the PATH/env-var probe as fallback -- with the provenance carried and
echoed, and a recorded-but-unusable manager REFUSING rather than
silently falling back.  The last test is the architecture gate: no
module outside diagnostics.py may probe for a manager by name.
"""
from __future__ import annotations

import re
from pathlib import Path

import molbuilder.diagnostics as D


def _detect_with(monkeypatch, cfg):
    monkeypatch.setattr(D, "read_config", lambda: cfg)
    return D.detect()


def test_the_recorded_manager_wins_over_the_path_probe(monkeypatch, tmp_path):
    exe = tmp_path / "my-mamba"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    caps = _detect_with(monkeypatch, {"envs": {"manager": str(exe)}})
    assert caps.conda_binary == str(exe)
    assert caps.conda_binary_source == "molbuilder.json envs.manager"


def test_a_wrong_recorded_manager_refuses_it_never_falls_back(monkeypatch, tmp_path):
    """The Sol failure mode: the recorded pathway must be FOLLOWED or
    the defect NAMED -- a silent fallback to whatever PATH holds is
    exactly 'the script forgot the correct pathway'."""
    missing = tmp_path / "gone" / "mamba"
    caps = _detect_with(monkeypatch, {"envs": {"manager": str(missing)}})
    assert caps.conda_binary is None
    assert "envs.manager" in (caps.conda_binary_source or "")
    assert "not an executable" in caps.conda_binary_source


def test_absent_key_probes_with_provenance(monkeypatch):
    caps = _detect_with(monkeypatch, {})
    if caps.conda_binary is not None:
        assert caps.conda_binary_source.startswith(("PATH (", "$"))


def test_the_manager_key_is_not_a_category(monkeypatch):
    """`envs.manager` must not leak into the category->env map."""
    from molbuilder.runtime_config import get_envs, get_env_manager
    cfg = {"envs": {"manager": "/x/mamba", "siesta": "molbuilder-siesta"}}
    assert "manager" not in get_envs(cfg)
    assert get_env_manager(cfg) == "/x/mamba"


def test_no_manager_probe_outside_the_door():
    """ARCHITECTURE GATE: `shutil.which` on a manager name, or a
    subprocess argv starting with a literal manager name, appears in
    diagnostics.py only -- everything else consumes
    ``caps.conda_binary``.  This is what keeps 'unified and used
    consistently' true tomorrow, not just today."""
    root = Path(__file__).resolve().parents[1] / "molbuilder"
    probe = re.compile(
        r"""shutil\.which\(\s*['"](?:mamba|micromamba|conda)['"]"""
    )
    literal_exec = re.compile(
        r"""subprocess\.\w+\(\s*\[\s*['"](?:mamba|micromamba|conda)['"]"""
    )
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "diagnostics.py":
            continue
        text = py.read_text(encoding="utf-8")
        if probe.search(text) or literal_exec.search(text):
            offenders.append(str(py.relative_to(root)))
    assert not offenders, (
        f"manager resolution outside the one door (diagnostics.py): "
        f"{offenders} -- consume caps.conda_binary instead")

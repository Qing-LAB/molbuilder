"""Tests for the ``scheduler`` config reader
(``runtime_config.get_scheduler``).

Authoritative design: docs/execution/job-system.md (schema),
§ 6 (value sourcing), § 10 (refuse-to-emit).

Pinned contracts:
  * absent ``scheduler`` block -> ``None`` (signal to emit only
    ``.run.sh``; § 10).
  * server-wide + project scopes deep-merge, project wins.
  * a ``slurm`` site MUST carry ``directives.partition`` + ``qos`` or
    the reader raises (refuse-to-emit; § 10) -- even if completeness is
    only reached after the scope merge.
  * type validation on directives / gpu / defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.runtime_config import (
    CONFIG_FILENAME,
    PROJECT_CONFIG_FILENAME,
    RuntimeConfigError,
    get_scheduler,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Clean cwd + isolated $HOME + cleared XDG so the server-wide lookup
    chain lands in tmp_path (mirrors test_runtime_config_multi_scope)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    return tmp_path


def _write_server(sandbox: Path, cfg: dict) -> None:
    (sandbox / CONFIG_FILENAME).write_text(json.dumps(cfg))


def _write_project(project_dir: Path, cfg: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / PROJECT_CONFIG_FILENAME).write_text(json.dumps(cfg))


_ASU_SOL = {
    "scheduler": {
        "kind": "slurm",
        "directives": {
            "partition": "public", "qos": "public",
            "mail_type": "ALL", "mail_user": "%u@asu.edu", "export": "NONE",
        },
        "gpu": {"partition": "public", "default_type": "a100",
                "exclusive": True},
        "defaults": {"time": "0-04:00:00", "cpus_per_task": 12, "mem": None},
    }
}


# --------------------------------------------------------------------- #
#  Absence + happy path                                                 #
# --------------------------------------------------------------------- #


def test_absent_scheduler_returns_none(sandbox):
    assert get_scheduler() is None


def test_absent_scheduler_with_other_config_returns_none(sandbox):
    _write_server(sandbox, {"envs": {"siesta": "molbuilder-siesta"}})
    assert get_scheduler() is None


def test_asu_sol_preset_resolves(sandbox):
    _write_server(sandbox, _ASU_SOL)
    sched = get_scheduler()
    assert sched is not None
    assert sched["kind"] == "slurm"
    assert sched["directives"]["partition"] == "public"
    assert sched["directives"]["qos"] == "public"
    assert sched["directives"]["export"] == "NONE"
    assert sched["gpu"]["default_type"] == "a100"
    assert sched["gpu"]["exclusive"] is True
    assert sched["defaults"]["cpus_per_task"] == 12
    assert sched["defaults"]["mem"] is None


def test_kind_defaults_to_slurm(sandbox):
    _write_server(sandbox, {"scheduler": {
        "directives": {"partition": "public", "qos": "public"}}})
    assert get_scheduler()["kind"] == "slurm"


# --------------------------------------------------------------------- #
#  Scope merge                                                          #
# --------------------------------------------------------------------- #


def test_project_overrides_server(sandbox):
    _write_server(sandbox, _ASU_SOL)
    proj = sandbox / "proj"
    _write_project(proj, {"scheduler": {
        "directives": {"partition": "htc"},
        "defaults": {"time": "0-00:30:00"}}})
    sched = get_scheduler(project_dir=proj)
    # project wins on the overridden keys ...
    assert sched["directives"]["partition"] == "htc"
    assert sched["defaults"]["time"] == "0-00:30:00"
    # ... but server-wide keys survive the deep-merge.
    assert sched["directives"]["qos"] == "public"
    assert sched["gpu"]["default_type"] == "a100"
    assert sched["defaults"]["cpus_per_task"] == 12


def test_completeness_reached_only_after_merge(sandbox):
    """A server scope missing ``qos`` plus a project scope supplying it
    is legal -- completeness is a MERGED-config property (§ 10)."""
    _write_server(sandbox, {"scheduler": {
        "kind": "slurm", "directives": {"partition": "public"}}})
    proj = sandbox / "proj"
    _write_project(proj, {"scheduler": {
        "directives": {"qos": "public"}}})
    sched = get_scheduler(project_dir=proj)
    assert sched["directives"] == {"partition": "public", "qos": "public"}


# --------------------------------------------------------------------- #
#  Refuse-to-emit (§ 10)                                                #
# --------------------------------------------------------------------- #


def test_missing_partition_refuses(sandbox):
    _write_server(sandbox, {"scheduler": {
        "kind": "slurm", "directives": {"qos": "public"}}})
    with pytest.raises(RuntimeConfigError, match="partition"):
        get_scheduler()


def test_missing_qos_refuses(sandbox):
    _write_server(sandbox, {"scheduler": {
        "kind": "slurm", "directives": {"partition": "public"}}})
    with pytest.raises(RuntimeConfigError, match="qos"):
        get_scheduler()


def test_empty_partition_string_refuses(sandbox):
    _write_server(sandbox, {"scheduler": {
        "kind": "slurm",
        "directives": {"partition": "   ", "qos": "public"}}})
    with pytest.raises(RuntimeConfigError, match="partition"):
        get_scheduler()


# --------------------------------------------------------------------- #
#  Type validation                                                     #
# --------------------------------------------------------------------- #


def test_bad_kind_rejected(sandbox):
    _write_server(sandbox, {"scheduler": {
        "kind": "pbs", "directives": {"partition": "p", "qos": "q"}}})
    with pytest.raises(RuntimeConfigError, match="kind"):
        get_scheduler()


def test_directives_must_be_object(sandbox):
    _write_server(sandbox, {"scheduler": {
        "kind": "slurm", "directives": "public"}})
    with pytest.raises(RuntimeConfigError, match="directives.*object"):
        get_scheduler()


def test_gpu_exclusive_must_be_bool(sandbox):
    _write_server(sandbox, {"scheduler": {
        "kind": "slurm",
        "directives": {"partition": "public", "qos": "public"},
        "gpu": {"exclusive": "yes"}}})
    with pytest.raises(RuntimeConfigError, match="exclusive.*bool"):
        get_scheduler()


def test_defaults_cpus_per_task_must_be_int(sandbox):
    _write_server(sandbox, {"scheduler": {
        "kind": "slurm",
        "directives": {"partition": "public", "qos": "public"},
        "defaults": {"cpus_per_task": "twelve"}}})
    with pytest.raises(RuntimeConfigError, match="cpus_per_task"):
        get_scheduler()


def test_scheduler_not_object_rejected(sandbox):
    _write_server(sandbox, {"scheduler": "slurm"})
    with pytest.raises(RuntimeConfigError, match="scheduler.*object"):
        get_scheduler()


# --------------------------------------------------------------------- #
#  The committed asu-sol template parses through the real reader        #
# --------------------------------------------------------------------- #


def test_committed_asu_sol_example_parses(sandbox):
    """The committed molbuilder.asu-sol.json example must stay valid against
    the live reader (it carries _comment_* keys -- they must pass through).

    The example ships beside the doc that explains it:
    docs/ops/deployment.md § 5 (Configuration)."""
    repo_root = Path(__file__).resolve().parent.parent
    example = repo_root / "docs" / "ops" / "examples" / "molbuilder.asu-sol.json"
    raw = json.loads(example.read_text())
    _write_server(sandbox, raw)
    sched = get_scheduler()
    assert sched is not None
    assert sched["directives"]["partition"] == "public"
    assert sched["gpu"]["default_type"] == "a100"


# --------------------------------------------------------------------- #
#  The committed server template stays synced with the live reader      #
# --------------------------------------------------------------------- #
#
# User rule (2026-07-29): the example .json templates are documentation
# that MUST move with the code in the same commit.  The dev-workstation
# missing-.run.sh regression was exactly this class of drift in reverse:
# a live config predating the (templated) script_generation section.

_EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "ops" / "examples"


def test_server_template_parses_and_covers_the_load_bearing_sections():
    from molbuilder.runtime_config import read_config
    cfg = read_config(_EXAMPLES / "molbuilder.json.example")
    missing = {"tls", "envs", "auth", "secret_key_file",
               "script_generation"} - set(cfg)
    assert not missing, (
        f"molbuilder.json.example no longer demonstrates: {sorted(missing)} "
        "-- the template must move with the reader in the same commit")


def test_server_template_activation_is_a_legal_form():
    from molbuilder.runtime_config import read_config
    cfg = read_config(_EXAMPLES / "molbuilder.json.example")
    activation = (cfg.get("script_generation") or {}).get("activation")
    assert activation in ("source activate", "conda activate"), (
        f"script_generation.activation in the template is {activation!r}; "
        "the generator accepts only the two documented forms")


def test_example_templates_cite_only_existing_docs():
    """Every docs/... path named inside the example templates must exist
    (the templates live under docs/, outside test_no_retired_doc_paths'
    scan roots -- this closes that gap for them)."""
    import re
    repo = Path(__file__).resolve().parents[1]
    bad = []
    for name in ("molbuilder.json.example", "molbuilder.asu-sol.json"):
        text = (_EXAMPLES / name).read_text(encoding="utf-8")
        for m in re.finditer(r"docs/[A-Za-z0-9_\-./]+\.md", text):
            if not (repo / m.group(0)).is_file():
                bad.append(f"{name}: {m.group(0)}")
    assert not bad, f"example templates cite missing docs: {bad}"


# ``test_routing_rows_keep_the_operators_own_columns`` stood here until
# 2026-08-17 (N4).  It pinned R10 (review-4 G5): `get_routing` rebuilt each row
# from a known-key list and silently STRIPPED everything else, so an operator
# drafting a `node_type` / `max_cores` / `gpu{}` column in `molbuilder.json`
# could not tell it apart from not writing one.  (The `node_type` named
# there was the scalar retired 2026-08-27 -- scheduler.md R11.)
#
# Retired because its subject moved, not because the lesson did.  A routing row
# is no longer an operator's hand-written description of their cluster -- the
# domains are PROBED into `environment.json`, one typed `Domain` per reachable
# (partition, qos) (`configuration.md` § 5, M-1).  Nobody hand-writes a column
# there, so there is no unknown column to preserve, and
# `test_scheduler_probe.py::test_a_domain_is_never_a_preference` now asserts
# the opposite property deliberately: a domain carries EXACTLY the four fields
# that were measured, and anything else appearing in one is a preference that
# has crept back in.
#
# The stripping hazard itself is still live wherever a reader rebuilds a
# person's object from a key list; `test_routing_domains.py` covers the one
# key that changed houses, and refuses it by name rather than dropping it.

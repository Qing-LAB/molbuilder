"""`jobset probe` -- what the suite can say without a scheduler: an empty
answer (a machine with none), the record's own `set` and its diff.  What the
probe makes of a real scheduler's answers is the field test's
(`tests/field/`).
"""

from molbuilder.scheduler.probe import (derive_domains, parse_allowed_qos,
                                        parse_qos, parse_sinfo)
from support.machines import WORKSTATION


def test_empty_probe_is_safe():
    """Not on a cluster: empty text -> no crash, no domains, a note."""
    domains, notes = derive_domains(parse_sinfo(""), parse_qos(""),
                                    parse_allowed_qos(""))
    assert domains == []
    assert any("no partitions" in n for n in notes)


def test_one_domain_per_allowed_pair_and_no_qos_when_unread():
    """`scheduler.md` § 4, `configuration.md` § 5 M-1: the record lists every
    (partition, QoS) pair the account's associations allow -- an
    association naming no partition holds on every one -- named by the
    partition for `public` or its own name and `<partition>-<qos>` else;
    with no readable list, one domain per partition stating no QoS, and a
    note.  Silent failure otherwise: a probe choosing a QoS for the person
    (one domain per partition, `public` assumed), the preference 2.17.0
    took out."""
    from molbuilder.scheduler.probe import Partition
    parts = [Partition(name="general", timelimit_str="7-00:00:00",
                       timelimit_secs=7 * 86400),
             Partition(name="htc", timelimit_str="4:00:00",
                       timelimit_secs=4 * 3600)]
    allowed = {"": {"public", "debug"}, "htc": {"private", "htc"}}
    domains, notes = derive_domains(parts, parse_qos(""), allowed)
    assert sorted((d.name, d.partition, d.qos) for d in domains) == [
        ("general", "general", "public"),
        ("general-debug", "general", "debug"),
        ("htc", "htc", "public"),
        ("htc-debug", "htc", "debug"),
        ("htc-htc", "htc", "htc"),
        ("htc-private", "htc", "private"),
    ]
    assert len({d.name for d in domains}) == len(domains)
    assert any("names no partition" in n for n in notes)

    domains, notes = derive_domains(parts, parse_qos(""), {})
    assert [(d.name, d.qos) for d in domains] == [("htc", None),
                                                  ("general", None)]
    assert any("states no QoS" in n for n in notes)
    # ...AND THE RECORD KEEPS SUCH A PAIR: `qos: null` on disk, read back as
    # the same domain (a row with no `qos` key is no domain, and was dropped
    # on the read until 2026-10-19).
    from molbuilder.scheduler.record import Domain
    assert all(Domain.from_row(d.to_row()) == d for d in domains)


# --------------------------------------------------------------------- #
#  The probe VERB — declared facts and per-difference consent           #
#  (roadmap § 0.2, N3+; configuration.md § 5 M-1/M-6)                   #
# --------------------------------------------------------------------- #


def _cli(args, **kw):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, ["probe", *args], **kw)


def test_set_refuses_unknown_keys_and_mistyped_values_by_name(tmp_path):
    r = _cli(["--set", "max_gpus=4"])
    assert r.exit_code != 0
    assert "no topology field 'max_gpus'" in r.output
    assert "gpus_per_node" in r.output          # the vocabulary, offered
    r = _cli(["--set", "gpus_per_node=four"])
    assert r.exit_code != 0 and "'four' is not int" in r.output
    r = _cli(["--set", "gpus_per_node"])
    assert r.exit_code != 0 and "KEY=VALUE" in r.output


#: How each machine enters its environment -- a fact the record carries,
#: asked about like the rest (W52: replaced unasked, so a probe over Sol's
#: record from a workstation put the workstation's hook into it).
_SOL_ENTERS = {"activation": "source activate"}
_DESK_ENTERS = {"activation": "conda activate"}


def _two_envs():
    from molbuilder.scheduler import Environment, Topology
    before = Environment(scheduler=WORKSTATION,
                         topology=Topology(gpus_per_node=4, gpu_type="a100"),
                         env_init=dict(_SOL_ENTERS),
                         detected_at="2026-08-01T00:00:00+00:00")
    probed = Environment(scheduler=WORKSTATION,
                         topology=Topology(gpus_per_node=1, gpu_type="rtx"),
                         env_init=dict(_DESK_ENTERS),
                         detected_at="2026-08-19T00:00:00+00:00")
    return before, probed


# The consent tests -- No keeps the record, Yes takes the probe, EOF declines,
# `--yes` asks nothing, an unchanged record is told so -- are rows of
# `tests/data/machine_record.toml`, down the road (W54 T29).


def test_domains_diff_as_one_fact(monkeypatch, capsys):
    """The reachable-domain SET is one question, not one per row
    (`configuration.md` M-6).

    API-LEVEL, and why: a domains difference needs a cluster's `sinfo`, and
    the road's probe runs on this box, which has none -- the rest of M-6 is
    rows of `tests/data/machine_record.toml`."""
    import click
    from molbuilder.scheduler import Domain
    from molbuilder.jobset._cli import _probe_consent_merge
    before, probed = _two_envs()
    probed.topology = before.topology            # isolate the domains diff
    probed.env_init = before.env_init
    probed.domains = [Domain(name="short", partition="p", qos="q",
                             max_time="1:00:00")]
    asked = []
    monkeypatch.setattr(click, "confirm",
                        lambda msg, **k: (asked.append(msg), True)[1])
    out = _probe_consent_merge(before, probed, yes=False)
    assert len(asked) == 1 and "domains" in asked[0]
    assert [d.name for d in out.domains] == ["short"]

"""x86 and ARM are different machines, and the record must say which.

User, 2026-08-26: *architecture matters too. x64 vs arm are different, don't
mix them* — and then *we should know our compiled/installed architecture*.
Two halves, because **a mismatch is what fails, so a check needs both
numbers**:

* `Topology.arch` — what the compute node is.
* `Environment.env_arch` — what the environments we would activate were
  built for.

ASU Sol offers an `arm` partition (Grace-Hopper, aarch64) in the same menu as
eight x86 ones. The
failure is total rather than slow: an x86 conda env does not activate usefully
on aarch64, and an AVX-512 binary does not run there at all. Worse, it arrives
disguised — `envs/builds.py` looks for `x86_64-conda-linux-gnu-gcc` **by
name**, so on aarch64 it finds nothing and reports an unknown compiler version
rather than the actual cause.

**R3 decides the default.** *An unstated limit never bars*, so a record that
states no architecture filters nothing.
"""
from __future__ import annotations

import json

from molbuilder.scheduler.record import Environment, Topology


# --------------------------------------------------------------------- #
#  the other half: what OUR software was built for                       #
# --------------------------------------------------------------------- #

def test_env_arch_round_trips():
    e = Environment(scheduler="slurm",
                    conda_envs=["molbuilder", "molbuilder-siesta"],
                    env_arch="x86_64")
    back = Environment.from_dict(json.loads(e.to_json()))
    assert back.env_arch == "x86_64"
    assert back.conda_envs == ["molbuilder", "molbuilder-siesta"]


def test_the_field_is_absent_rather_than_null_when_unknown():
    """`to_dict` omits it entirely, the same way `conda_envs` and
    `env_init` are omitted: a key that is missing and a key that is
    null are different claims to anything testing for one."""
    d = json.loads(Environment(scheduler="workstation").to_json())
    assert "env_arch" not in d


def test_topology_arch_survives_the_round_trip():
    e = Environment(scheduler="slurm",
                    topology=Topology(sockets=1, cores_per_socket=72,
                                      arch="aarch64"))
    assert Environment.from_dict(json.loads(e.to_json())).topology.arch \
        == "aarch64"

"""x86 and ARM are different machines, and the record must say which.

User, 2026-08-26: *architecture matters too. x64 vs arm are different, don't
mix them* — and then *we should know our compiled/installed architecture*.
Two halves, because **a mismatch is what fails, so a check needs both
numbers**:

* `Topology.arch` — what the compute node is.
* `Environment.env_arch` — what the environments we would activate were
  built for.

ASU Sol offers an `arm` partition (Grace-Hopper, aarch64) in the same menu as
eight x86 ones, and until this the record could not tell them apart. The
failure is total rather than slow: an x86 conda env does not activate usefully
on aarch64, and an AVX-512 binary does not run there at all. Worse, it arrives
disguised — `envs/builds.py` looks for `x86_64-conda-linux-gnu-gcc` **by
name**, so on aarch64 it finds nothing and reports an unknown compiler version
rather than the actual cause.

**R3 decides the default.** *An unstated limit never bars*, so a record
written before these fields states nothing and nothing is filtered — which is
what keeps every existing record working.
"""
from __future__ import annotations

import json

from molbuilder.scheduler.record import (Environment, Topology,
                                         _parse_lscpu, _parse_scontrol_node)


# Real `scontrol show node` from an ASU Sol GPU node (trimmed).
_SCONTROL = (
    "NodeName=sg013 Arch=x86_64 CoresPerSocket=24 "
    "CPUAlloc=48 CPUTot=48 CPULoad=43.21 "
    "AvailableFeatures=a100 ActiveFeatures=a100 "
    "Gres=gpu:a100:4 NodeAddr=sg013 RealMemory=515000 Sockets=2 "
    "ThreadsPerCore=1 State=MIXED"
)

# What an aarch64 node reports instead -- the case that must be
# distinguishable, since Sol's `arm` partition is Grace-Hopper.
_SCONTROL_ARM = (
    "NodeName=sarm01 Arch=aarch64 CoresPerSocket=72 CPUTot=72 "
    "Gres=gpu:gh200:1 RealMemory=580000 Sockets=1 ThreadsPerCore=1"
)

_LSCPU = """\
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Socket(s):                          2
Core(s) per socket:                 10
Thread(s) per core:                 2
NUMA node(s):                       2
"""


# --------------------------------------------------------------------- #
#  reading it off the machine                                            #
# --------------------------------------------------------------------- #

def test_scontrol_arch_is_read(): 
    """`scontrol show node` has printed `Arch=` all along; the parser read
    Sockets, CoresPerSocket, ThreadsPerCore, RealMemory and Gres, and
    skipped it."""
    assert _parse_scontrol_node(_SCONTROL).arch == "x86_64"


def test_an_arm_node_is_distinguishable_from_an_x86_one():
    x86 = _parse_scontrol_node(_SCONTROL)
    arm = _parse_scontrol_node(_SCONTROL_ARM)
    assert (x86.arch, arm.arch) == ("x86_64", "aarch64")
    assert x86.arch != arm.arch, "the two must not read the same"


def test_lscpu_arch_is_read():
    """The workstation path, which had the same gap."""
    assert _parse_lscpu(_LSCPU).arch == "x86_64"


def test_the_name_is_kept_as_the_MACHINE_spells_it():
    """Never normalised. A name we invent is a name nothing else uses --
    not SLURM, not `uname -m`, not conda's platform strings."""
    assert _parse_scontrol_node(_SCONTROL_ARM).arch == "aarch64"


def test_a_node_that_does_not_say_reads_as_unknown(): 
    """R3: an unstated fact never bars. `None`, never a guess and never a
    default of x86 -- a default would make silence indistinguishable from an
    answer."""
    assert _parse_scontrol_node("NodeName=x CPUTot=8 Sockets=1").arch is None
    assert _parse_lscpu("Socket(s): 1\n").arch is None


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


def test_an_older_record_states_no_architecture_and_that_is_fine():
    """Every record on disk today predates both fields. They must read as
    *unknown* rather than as an answer, or an existing record would start
    filtering a menu on a fact nobody measured."""
    old = {"schema": "molbuilder/environment@2", "scheduler": "slurm",
           "topology": {"sockets": 2, "cores_per_socket": 32},
           "conda_envs": ["molbuilder"]}
    back = Environment.from_dict(old)
    assert back.env_arch is None
    assert back.topology.arch is None
    assert back.conda_envs == ["molbuilder"], "the rest still reads"


def test_the_field_is_absent_rather_than_null_when_unknown():
    """`to_dict` omits it entirely, the same way `conda_envs` and
    `script_generation` are omitted: a key that is missing and a key that is
    null are different claims to anything testing for one."""
    d = json.loads(Environment(scheduler="workstation").to_json())
    assert "env_arch" not in d


def test_both_halves_are_needed_to_see_a_mismatch():
    """Neither number alone says anything. The *pair* is the check -- which
    is why the record carries both rather than assuming one from the other.
    """
    node = _parse_scontrol_node(_SCONTROL_ARM)          # aarch64 compute node
    env = Environment(scheduler="slurm", topology=node,
                      conda_envs=["molbuilder-siesta"], env_arch="x86_64")
    assert env.topology.arch != env.env_arch, (
        "an x86 environment aimed at an aarch64 node -- the case that must "
        "be visible in the record before anything can act on it")


def test_topology_arch_survives_the_round_trip():
    e = Environment(scheduler="slurm",
                    topology=Topology(sockets=1, cores_per_socket=72,
                                      arch="aarch64"))
    assert Environment.from_dict(json.loads(e.to_json())).topology.arch \
        == "aarch64"

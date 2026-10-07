"""Emission — a placement, rendered for the scheduler in both its spellings.

The contract is ``docs/execution/scheduler.md`` **R1**: the ``#SBATCH`` header
and the ``sbatch`` command line are two RENDERINGS of one placement, never two
decisions.  They cannot disagree, because there is nothing to disagree about.

A ``Directives`` renders the facts BOTH spellings carry.  Everything else in
the header -- the job name, the account, mail, output paths, the body -- is
the header's own and stays where it is: those are not decisions the command
line also makes, so they cannot drift from it.

Stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Directives:
    """The scheduler-facing facts of one placement.

    ``walltime`` is SLURM text (``D-HH:MM:SS``) because that is what both
    spellings carry; converting it here rather than at each caller is the
    point.  ``None`` fields are simply not rendered -- an unstated fact is
    not a zero.
    """
    partition:     Optional[str] = None
    qos:           Optional[str] = None
    walltime:      Optional[str] = None
    ntasks:        Optional[int] = None
    cpus_per_task: Optional[int] = None
    gres:          Optional[str] = None
    #: ``--gres-flags=enforce-binding`` beside the GPU ask, unless the
    #: calculation turned it off (`execution/gpu.md` G9).
    gpu_binding:   bool = True
    mem:           Optional[str] = None
    exclusive:     bool = False

    @classmethod
    def of(cls, placement, resources=None) -> "Directives":
        """Bind a :class:`~molbuilder.scheduler.place.Placement` to the
        resources it was placed for.

        **Both objects travel whole** (`architecture.md` § 3.1, A8).  The
        fields are read HERE, by the thing that needs them.

        ``resources`` is duck-typed rather than imported: this package
        sits below the layer that defines ``Resources``, so
        it reads the attributes it needs and names none of the type.

        ``placement`` may be ``None`` on a machine with no menu, where the
        queue is simply not stated.
        """
        r = resources
        return cls(
            partition=getattr(placement, "partition", None),
            qos=getattr(placement, "qos", None),
            walltime=getattr(r, "time", None),
            ntasks=getattr(r, "mpi_np", None),
            cpus_per_task=getattr(r, "cpus_per_task", None),
            gres=getattr(r, "gres", None),
            gpu_binding=getattr(r, "gpu_binding", None) is not False,
            mem=getattr(r, "mem", None),
            exclusive=bool(getattr(r, "exclusive", False)),
        )

    # ---- the two spellings -------------------------------------------- #

    @staticmethod
    def lines_of(header_text: str) -> List[str]:
        """A rendered header's ``#SBATCH`` lines, as it carries them -- what
        a surface shows for what a job will be launched with (A13,
        `execution/architecture.md` § 5.2), read back by the writer's side
        and never worked out again."""
        return [ln.strip() for ln in header_text.splitlines()
                if ln.startswith("#SBATCH")]

    def header_lines(self) -> List[str]:
        """``#SBATCH`` lines for the facts a command line also carries.

        Not the whole header: ``-J``, ``-N``, the account, mail and the output
        paths belong to the header alone and are added by its renderer.
        """
        out: List[str] = []
        if self.ntasks is not None:
            out.append(f"#SBATCH -n {self.ntasks}")
        if self.cpus_per_task is not None:
            out.append(f"#SBATCH -c {self.cpus_per_task}")
        if self.walltime:
            out.append(f"#SBATCH -t {self.walltime}")
        if self.partition:
            out.append(f"#SBATCH -p {self.partition}")
        if self.qos:
            out.append(f"#SBATCH -q {self.qos}")
        out.extend(self._gres_lines("#SBATCH "))
        out.extend(self._memory_lines("#SBATCH ", spell_all=True))
        return out

    def sbatch_flags(self) -> List[str]:
        """The same facts as command-line flags, which WIN over the header.

        That is what lets one rendered ``.sbatch`` serve a whole sweep while
        each job still gets its own ranks and cores.
        """
        out: List[str] = []
        if self.partition:
            out += ["-p", self.partition]
        if self.qos:
            out += ["-q", self.qos]
        if self.ntasks is not None:
            out += ["-n", str(self.ntasks)]
        if self.cpus_per_task is not None:
            out += ["-c", str(self.cpus_per_task)]
        out.extend(self._gres_lines(""))
        if self.walltime:
            out += ["-t", self.walltime]
        out.extend(self._memory_lines("", spell_all=False))
        return out

    def _gres_lines(self, prefix: str) -> List[str]:
        """The GPU ask, and the binding that goes with it.

        Both spellings carry ``--gres-flags=enforce-binding`` (R1): it is not
        one of the header-only facts (``-J``, ``-N``, the account, mail and
        the output paths).

        It rides WITH the gres because it is meaningless without one: it
        asks the scheduler to put the task on the socket its GPU is
        attached to, which is the difference between a device on the local
        PCIe root and one across the interconnect.  A calculation may turn
        it off (`allocation.gpu_binding`, `execution/gpu.md` G9) -- for both
        spellings at once, since this is the one place either is written.
        """
        if not self.gres:
            return []
        return [f"{prefix}--gres={self.gres}"] + (
            [f"{prefix}--gres-flags=enforce-binding"] if self.gpu_binding
            else [])

    def _memory_lines(self, prefix: str, *, spell_all: bool) -> List[str]:
        """``--exclusive`` and ``--mem`` are mutually exclusive.

        Whole-node ownership already grants all the node's memory, so a
        ``--mem`` beside it is meaningless and some sites reject the pair
        (`running-a-job.md` § 5.3.1).  One rule, both spellings.

        ``spell_all`` is the one place the two renderings legitimately
        differ, and they differ in VERBOSITY, not in meaning: the header adds
        ``--mem=0`` because a person reads that file and "all the node's
        memory" is worth saying out loud, while a command line that overrides
        nothing has nothing to spell.
        """
        if self.exclusive:
            out = [f"{prefix}--exclusive"]
            if spell_all:
                out.append(f"{prefix}--mem=0")
            return out
        if self.mem:
            return [f"{prefix}--mem={self.mem}"]
        return []

    def queue(self):
        """``(partition, qos)`` — what both spellings must agree on."""
        return (self.partition, self.qos)

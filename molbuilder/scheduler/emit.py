"""Emission — a placement, rendered as the scheduler's ``#SBATCH`` lines.

The contract is ``docs/execution/scheduler.md`` **R1**: a placement is decided
once, at prep, and rendered once -- the header; the ``sbatch`` line sends that
header (`jobset.submit._sbatch_request`).

A ``Directives`` renders the placement's facts: the queue, the wall, the
ranks, the cores, the devices and the memory.  Everything else in the header
-- the job name, the output paths, the body -- is its renderer's
(`runwrap._render_sbatch_for`).

Stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Directives:
    """The scheduler-facing facts of one placement.

    ``walltime`` is SLURM text (``D-HH:MM:SS``), the spelling the header
    carries.  ``None`` fields are simply not rendered -- an unstated fact is
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

    @staticmethod
    def lines_of(header_text: str) -> List[str]:
        """A rendered header's ``#SBATCH`` lines, as it carries them -- what
        a surface shows for what a job will be launched with (A13,
        `execution/architecture.md` § 5.2), read back by the writer's side
        and never worked out again."""
        return [ln.strip() for ln in header_text.splitlines()
                if ln.startswith("#SBATCH")]

    def header_lines(self) -> List[str]:
        """The placement's ``#SBATCH`` lines.

        Not the whole header: ``-J``, ``-N`` and the output paths are added
        by its renderer.
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
        out.extend(self._gres_lines())
        out.extend(self._memory_lines())
        return out

    def _gres_lines(self) -> List[str]:
        """The GPU ask, and the binding that goes with it.

        It rides WITH the gres because it is meaningless without one: it
        asks the scheduler to put the task on the socket its GPU is
        attached to, which is the difference between a device on the local
        PCIe root and one across the interconnect.  A calculation may turn
        it off (`allocation.gpu_binding`, `execution/gpu.md` G9).
        """
        if not self.gres:
            return []
        return [f"#SBATCH --gres={self.gres}"] + (
            ["#SBATCH --gres-flags=enforce-binding"] if self.gpu_binding
            else [])

    def _memory_lines(self) -> List[str]:
        """``--exclusive`` and ``--mem`` are mutually exclusive.

        Whole-node ownership already grants all the node's memory, so a
        ``--mem`` beside it is meaningless and some sites reject the pair
        (`running-a-job.md` § 5.3.1); ``--mem=0`` beside it says "all the
        node's memory" out loud to the person reading the file.
        """
        if self.exclusive:
            return ["#SBATCH --exclusive", "#SBATCH --mem=0"]
        if self.mem:
            return [f"#SBATCH --mem={self.mem}"]
        return []

"""Emission — a placement, rendered as its kind's header directives.

The contract is ``docs/execution/scheduler.md`` **R1**: a placement is decided
once, at prep, and rendered once -- the header; the line a launch sends
carries that header (`jobset.submit._sbatch_request`).  The words are the
kind's (`kinds.toml`, § 6a): this module spells no scheduler of its own.

A ``Directives`` renders the placement's facts: the queue, the wall, the
ranks, the cores, the devices and the memory.  Everything else in the header
-- the job name, the output paths, the body -- is its renderer's
(`runwrap.render_sbatch`).

Stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Directives:
    """The scheduler-facing facts of one placement.

    ``walltime`` and ``mem`` are a job's own spelling (`task.py`'s: ``D-HH:MM:SS``,
    ``120G``), the spelling the header carries.  ``None`` fields are simply not
    rendered -- an unstated fact is not a zero.
    """
    partition:     Optional[str] = None
    qos:           Optional[str] = None
    walltime:      Optional[str] = None
    ntasks:        Optional[int] = None
    cpus_per_task: Optional[int] = None
    #: The GPUs asked for, a count -- never a card (`scheduler.md` R2a).
    gpus:          Optional[int] = None
    #: The binding beside the GPU ask, unless the calculation turned it off
    #: (`execution/gpu.md` G9).
    gpu_binding:   bool = True
    mem:           Optional[str] = None
    exclusive:     bool = False

    @staticmethod
    def lines_of(header_text: str, kind) -> List[str]:
        """A rendered header's directive lines, as it carries them -- what
        a surface shows for what a job will be launched with (A13,
        `execution/architecture.md` § 5.2), read back by the writer's side
        and never worked out again.  ``kind`` is the header's
        (`kinds.Kind`)."""
        return kind.directive_lines_of(header_text)

    @staticmethod
    def said(lines: List[str]) -> str:
        """A header's request in one line, as its directive ``lines`` carry
        it (:meth:`lines_of`) -- every value of what a launch sends, each
        line's directive after its kind's prefix -- the line both doors show
        where a person approves it (`submission.md` S4)."""
        return "header: " + "  ".join(ln.split(None, 1)[1] for ln in lines)

    def header_lines(self, kind) -> List[str]:
        """The placement's directive lines, in ``kind``'s language
        (`kinds.Kind`).

        Not the whole header: the name, the node and the output paths are
        added by its renderer.
        """
        out: List[str] = []
        if self.ntasks is not None:
            out += kind.directive("ranks", ranks=self.ntasks)
        if self.cpus_per_task is not None:
            out += kind.directive("cores", cores=self.cpus_per_task)
        if self.walltime:
            out += kind.directive("wall", wall=self.walltime)
        if self.partition:
            out += kind.directive("partition", partition=self.partition)
        if self.qos:
            out += kind.directive("qos", qos=self.qos)
        out += self._gpu_lines(kind)
        out += self._memory_lines(kind)
        return out

    def _gpu_lines(self, kind) -> List[str]:
        """The GPU ask, and the binding that goes with it.

        It rides WITH the GPU ask because it is meaningless without one: it
        asks the scheduler to put the task on the socket its GPU is
        attached to, which is the difference between a device on the local
        PCIe root and one across the interconnect.  A calculation may turn
        it off (`allocation.gpu_binding`, `execution/gpu.md` G9).
        """
        if not self.gpus:
            return []
        return kind.directive("gpus", gpus=self.gpus) + (
            kind.directive("gpu_binding") if self.gpu_binding else [])

    def _memory_lines(self, kind) -> List[str]:
        """The whole node, or the memory -- never both.

        Whole-node ownership already grants all the node's memory, so a
        memory ask beside it is meaningless and some sites reject the pair
        (`running-a-job.md` § 5.3.1); the kind's ``whole_node`` says "all
        the node's memory" out loud to the person reading the file.
        """
        if self.exclusive:
            return kind.directive("whole_node")
        if self.mem:
            return kind.directive("memory", memory=self.mem)
        return []

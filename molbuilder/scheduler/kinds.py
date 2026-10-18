"""The scheduler kinds -- ``kinds.toml`` read, once, into one :class:`Kind`
per block (`execution/scheduler.md` § 6a).

**Everything molbuilder says to a scheduler is a kind's**: how a job is sent
and asked about, the id it answers, how a sent job is asked after and
stopped, the header's language, the value a launch may restate and what it
tells a running job.  This is the one reader of that file, and no other
module spells a scheduler.  A machine's record names its kind and where the
kind's commands are there (`record.Scheduler`); every step asks the kind
through the record (``Environment.kind``), never compares the word.

What stays code is discovery and the reading of a scheduler's answers, a
reader per kind keyed by its name: the probe reading the machine and its
queues out of the scheduler's own tools (`record._DISCOVERY`), `--mode ask`
reading its start-time answer (`jobset.ask._ANSWER_READERS`), and the jobs
list reading whether it still lists a job and which cluster it serves
(`queue._READERS`).

Floor 1 (`execution/architecture.md` § 2.1): data, and the words it fills.
Stdlib-only.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: The file, beside this module.
KINDS_FILE = Path(__file__).with_name("kinds.toml")

#: The commands a kind with a queue states, by what each does: send a job,
#: ask whether it is still listed, stop it, and say which cluster answers.
ROLES = ("submit", "query", "cancel", "cluster")

#: The header's directives, one per value prep writes -- the keys of a
#: kind's ``header`` beside its ``prefix`` and ``jobid``.
DIRECTIVES = ("name", "nodes", "ranks", "cores", "wall", "partition", "qos",
              "gpus", "gpu_binding", "memory", "whole_node", "stdout",
              "stderr")

#: The values a launch may restate on the line (`commands.LAUNCH_FLAGS`).
RESTATES = ("wall",)

#: The lines a kind's ``query`` states (`scheduler.queue`): asking whether
#: a job is still listed, stopping it, and asking which cluster answers.
QUERIES = ("listed", "cancel", "cluster")

#: MOLBUILDER'S OWN NAMES for what a scheduler tells a running job, each with
#: what it means.  A kind maps its variables onto these (its ``job`` block),
#: and the job's header says each once for everything it starts
#: (:meth:`Kind.job_lines`) -- the run script, a walk, the monitor, the
#: engine's own script -- so each reads one name, whatever the kind.  Unset
#: where no scheduler tells one: a run here has no job id.  The names are
#: molbuilder's protocol, read by name where this module cannot be imported
#: (the monitor and the scripts beside a job); static review keeps them to
#: this list.
JOB_VARIABLES: Dict[str, str] = {
    "MB_JOB_ID":        "the scheduler's id for this job",
    "MB_NTASKS":        "the ranks the scheduler granted the job",
    "MB_CPUS_PER_TASK": "the cores the scheduler granted each rank",
    "MB_CPUS_ON_NODE":  "the cores the job holds on its node",
    "MB_JOB_GPUS":      "the GPUs the scheduler granted the job",
    "MB_JOB_NODELIST":  "the nodes the job runs on",
}

#: A command-line token that is one brace name and stands for several words.
_SPLICED = re.compile(r"^\{([a-z_]+)\}$")


class KindError(ValueError):
    """``kinds.toml`` does not say what a kind must, or a kind named that it
    does not hold -- said with the kind and the key."""


@dataclass(frozen=True)
class Kind:
    """One scheduler kind, as ``kinds.toml`` states it."""
    name: str
    what: str
    #: Whether a job is sent to a queue -- prep writes a header, launch
    #: sends it.  ``False`` is a machine where a job runs directly.
    queue: bool
    #: Each of :data:`ROLES` -> the command's name on PATH, what the probe
    #: looks for.
    commands: Mapping[str, str]
    #: How a job is sent: the line's words, the word that asks instead, and
    #: the pattern of the id a send is answered with.
    line_words: Tuple[str, ...]
    ask: Optional[str]
    job_id: Optional[str]
    #: Each of :data:`RESTATES` -> its words on the line.
    restate: Mapping[str, Tuple[str, ...]]
    #: Each of :data:`QUERIES`.
    query: Mapping[str, object]
    #: ``prefix``, ``jobid`` and each of :data:`DIRECTIVES`.
    header: Mapping[str, object]
    #: Each of :data:`JOB_VARIABLES` it tells -> its own variable.
    job: Mapping[str, str]

    # ----- how a job is sent ------------------------------------------ #

    def line(self, where: Mapping[str, Optional[str]], *, job: str,
             restated: Sequence[str], launched_by: str, header: str,
             args: Sequence[str]) -> List[str]:
        """The one line that sends ``header``: ``where`` is the record's
        command locations (`record.Scheduler.commands`), ``restated`` what a
        launch restates (:meth:`restated`), ``launched_by`` molbuilder's
        launch claim (`job-contracts.md` § 2.6), ``args`` the run's
        arguments.  A command the line names and the record does not is a
        :class:`KindError`."""
        line = _filled(self, self.line_words, where, {
            "job": job, "restated": list(restated),
            "launched_by": launched_by, "header": header,
            "args": list(args)})
        if line is None:
            raise KindError(f"the machine's record names no command the "
                            f"{self.name} line runs: "
                            f"{' '.join(self.line_words)}")
        return line

    def asking(self, line: Sequence[str]) -> List[str]:
        """``line`` asked about without being sent: the kind's ``ask`` set
        after the command."""
        return [line[0], self.ask, *line[1:]]

    def job_id_of(self, answer: str) -> Optional[str]:
        """The id the scheduler answered a send with, or ``None`` when the
        answer carries none."""
        m = re.search(self.job_id, answer or "")
        return m.group(1) if m else None

    def restated(self, what: str, value: str) -> List[str]:
        """The words that restate ``what`` (:data:`RESTATES`) on the line."""
        return _fill(self.restate[what], {what: value})

    # ----- the header's language -------------------------------------- #

    def directive(self, key: str, **values) -> List[str]:
        """Directive ``key`` (:data:`DIRECTIVES`) as header lines, filled
        with ``values`` -- one line, or the several a ``whole_node`` is."""
        spelled = self.header[key]
        words = spelled if isinstance(spelled, list) else [spelled]
        return [f"{self.header['prefix']} {w.format(**values)}"
                for w in words]

    def output_file(self, name: str) -> str:
        """``name`` -- the file catalogue's name for the scheduler's own
        output (`runfiles.SCHEDULER_STDOUT`), carrying ``{scheduler}`` and
        ``{jobid}`` -- as the header asks the scheduler for it: this kind's
        name, and the placeholder it fills with the job's id."""
        return name.format(scheduler=self.name, jobid=self.header["jobid"])

    def directive_lines_of(self, header_text: str) -> List[str]:
        """A written header's directive lines, as it carries them."""
        prefix = self.header["prefix"]
        return [ln.strip() for ln in header_text.splitlines()
                if ln.startswith(prefix)]

    # ----- what it tells a running job -------------------------------- #

    def job_lines(self) -> List[str]:
        """Bash that says each variable the scheduler tells a running job
        once, in molbuilder's own name (:data:`JOB_VARIABLES`), exported for
        everything the script starts; none for a kind that tells none."""
        if not self.job:
            return []
        return ([f"# What {self.what} tells this job, in molbuilder's own "
                 f"names (scheduler/kinds.toml) --",
                 "# read by everything this script starts."]
                + [f'if [ -n "${{{theirs}:-}}" ]; then export '
                   f'{ours}="${theirs}"; fi'
                   for ours, theirs in self.job.items()])

    # ----- a sent job, asked after ------------------------------------ #

    def asked(self, what: str, where: Mapping[str, Optional[str]],
              **values) -> Optional[List[str]]:
        """Query ``what`` (:data:`QUERIES`) as a command line, from the
        record's command locations -- ``None`` where the record names no
        command the line runs."""
        return _filled(self, self.query[what], where, values)


def _filled(kind: Kind, words: Sequence[str],
            where: Mapping[str, Optional[str]],
            values: Mapping[str, object]) -> Optional[List[str]]:
    """``words`` filled with ``values`` and the record's command locations
    -- each command the words name by its role (``{submit}``, ``{query}``,
    ...) where the record says it is; ``None`` when the record names none
    for a command the words name."""
    located = {role: command for role, command in where.items() if command}
    named = {m for w in words for m in re.findall(r"\{([a-z_]+)\}", w)}
    if any(role in named and role not in located for role in ROLES):
        return None
    return _fill(words, {**located, **values})


def _fill(words: Sequence[str], values: Mapping[str, object]) -> List[str]:
    """``words`` filled: a word that is one brace name standing for several
    (a list value) is spliced in, any other word filled in place."""
    out: List[str] = []
    for w in words:
        m = _SPLICED.match(w)
        if m and isinstance(values.get(m.group(1)), list):
            out.extend(str(v) for v in values[m.group(1)])
        else:
            out.append(w.format(**values))
    return out


def _keys(kind: str, block: str, got: Mapping, want: Sequence[str], *,
          all_of: bool = True) -> None:
    """``got``'s keys are ``want`` (all of them, or some of them)."""
    extra = sorted(set(got) - set(want))
    missing = [k for k in want if k not in got] if all_of else []
    if extra or missing:
        raise KindError(
            f"{KINDS_FILE.name}: [{kind}{'.' + block if block else ''}] "
            + "; ".join(([f"names {', '.join(extra)}, which no kind states"]
                         if extra else [])
                        + ([f"leaves out {', '.join(missing)}"]
                           if missing else [])))


def _kind_of(name: str, d: Mapping) -> Kind:
    """One block of the file, checked whole."""
    if not isinstance(d.get("what"), str) or not isinstance(
            d.get("queue"), bool):
        raise KindError(f"{KINDS_FILE.name}: [{name}] states its `what` "
                        f"and whether it has a `queue`")
    if not d["queue"]:
        _keys(name, "", d, ("what", "queue"))
        return Kind(name=name, what=d["what"], queue=False, commands={},
                    line_words=(), ask=None, job_id=None, restate={},
                    query={}, header={}, job={})
    _keys(name, "", d, ("what", "queue", "commands", "send", "restate",
                        "query", "header", "job"))
    _keys(name, "commands", d["commands"], ROLES)
    _keys(name, "send", d["send"], ("line", "ask", "job_id"))
    _keys(name, "restate", d["restate"], RESTATES)
    _keys(name, "query", d["query"], QUERIES)
    _keys(name, "header", d["header"], ("prefix", "jobid") + DIRECTIVES)
    _keys(name, "job", d["job"], tuple(JOB_VARIABLES), all_of=False)
    if re.compile(d["send"]["job_id"]).groups != 1:
        raise KindError(f"{KINDS_FILE.name}: [{name}.send] job_id reads the "
                        f"id as its one group")
    return Kind(name=name, what=d["what"], queue=True,
                commands=dict(d["commands"]),
                line_words=tuple(d["send"]["line"]), ask=d["send"]["ask"],
                job_id=d["send"]["job_id"],
                restate={k: tuple(v) for k, v in d["restate"].items()},
                query=dict(d["query"]), header=dict(d["header"]),
                job=dict(d["job"]))


@lru_cache(maxsize=1)
def _every() -> Dict[str, Kind]:
    """Every kind the file states, in its order -- read once."""
    raw = tomllib.loads(KINDS_FILE.read_text(encoding="utf-8"))
    return {name: _kind_of(name, block) for name, block in raw.items()}


def names() -> Tuple[str, ...]:
    """Every kind's name, in the file's order."""
    return tuple(_every())


def kind(name: str) -> Kind:
    """The kind ``name`` -- :class:`KindError` naming the kinds the file
    states when it holds none by that name."""
    try:
        return _every()[name]
    except KeyError:
        raise KindError(f"no scheduler kind {name!r} -- {KINDS_FILE.name} "
                        f"states {', '.join(names())}") from None


__all__ = ["DIRECTIVES", "JOB_VARIABLES", "KINDS_FILE", "Kind", "KindError",
           "RESTATES", "ROLES", "kind", "names"]

"""What molbuilder prints for a person to type -- every `molbuilder jobset`
line a verb, a refusal, a next step or a status names, composed here once
(`execution/job-system.md` § 5.3: *what molbuilder prints, you can type*).

**Module:** L2 (jobset).  The W52 review read the same lines spelled in a
dozen modules.  Some named the calculation and some did not, so a line pasted
anywhere but inside the folder acted on another calculation or on none; one
printed ``--mode submit|direct``, which bash runs as a pipe; others ended in a
``(note)`` a shell cannot parse, or a ``<stage>`` nobody can type.  A line
composed here is ONE command:

* the calculation named by its address from the projects root, unless the
  reader is standing in it (:func:`bundle_flag`);
* a launch's mode stated where this calculation's config sets none -- one
  line per mode, each typeable (:func:`launch_lines`);
* any prose after ``#``.

What a text read LATER says about a launch -- a deck's header, a result's
remedy -- is `identity.launch_as_typed`: it is read wherever it was copied, so
it names no folder.  A layer below this one (a machine record's refusal, a
wrapper's banner) cannot import it and prints plain text by the same rules.
"""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import List, Optional

#: The program as a person types it.
PROG = "molbuilder jobset"


def bundle_flag(base) -> str:
    """`` --bundle <address from the projects root>``, or ``""`` when the
    working directory already IS the calculation -- or when ``base`` is not
    in the projects tree, where no address names it (a calculation always is,
    `_cli._resolve_bundle`).

    A printed command has to work where it is pasted.  Naming the calculation
    is what makes it work from anywhere, and leaving it off when the reader
    stands in it keeps the common case short (`job-contracts.md` § 2.5b).
    The address is quoted as the shell needs it: a folder name may hold a
    space.  *(It was `_cli._bundle_hint`, which only the command line's own
    next lines asked until 2026-10-01; W52.)*"""
    from ..projects import projects_root
    try:
        here = Path(base).resolve()
        if here == Path.cwd().resolve():
            return ""
        rel = here.relative_to(Path(projects_root()).resolve())
    except (ValueError, OSError):
        return ""
    return f" --bundle {shlex.quote(str(rel))}"


def command(verb: str, *words, base, flags=()) -> str:
    """``molbuilder jobset <verb> <words> <flags>`` and the calculation's
    ``--bundle``: one command.  ``base`` is required -- the calculation a
    line acts on is never left to where it happens to be pasted (D1); pass
    ``None`` for a line that names none (`init`, `probe`).  A word that is
    ``None`` is left out -- the verb's form without it -- never printed as
    ``None``."""
    parts = [PROG, verb, *(str(w) for w in words if w is not None),
             *(str(f) for f in flags)]
    return " ".join(parts) + (bundle_flag(base) if base is not None else "")


def configured_mode() -> Optional[str]:
    """The launch mode this machine's config states, or ``None`` --
    `runtime_config.get_launch_mode`, the reader `launch` falls back on."""
    from ..runtime_config import get_launch_mode
    try:
        return get_launch_mode() or None
    except Exception:                                    # noqa: BLE001
        return None          # a malformed config is `launch`'s to refuse


def launch_lines(kind: str, *words, base) -> List[str]:
    """The launch of ``words`` as a person types it: the bare line where this
    machine's config names a mode -- `launch` takes it -- else one line per
    mode, each a command (never ``--mode submit|direct``, which bash reads as
    a pipe)."""
    if configured_mode():
        return [command("launch", kind, *words, base=base)]
    return [command("launch", kind, *words, base=base,
                    flags=("--mode", "direct")) + "   # here",
            command("launch", kind, *words, base=base,
                    flags=("--mode", "submit")) + "   # to the queue"]


def lines(verb: str, kind: str, *words, base) -> List[str]:
    """The command lines for ``verb`` -- a launch's from
    :func:`launch_lines`, any other verb's one :func:`command` -- so a caller
    that names a verb it was handed cannot print a launch with no mode."""
    if verb == "launch":
        return launch_lines(kind, *words, base=base)
    return [command(verb, kind, *words, base=base)]


def run_first(stage: str, *, base) -> List[str]:
    """Run ``stage`` first -- its prep, then its launch: the remedy every
    refusal that waits on another stage gives (W52: written six times, one
    with `--mode`, none naming the calculation)."""
    return [command("prep", "run", stage, base=base)] + launch_lines(
        "run", stage, base=base)


def name_a_stage(verb: str, kind: str, refs, *, base) -> str:
    """``name one -- coarse ('#1'), medium ('#2'):`` and the command for the
    first -- what every verb that acts on ONE stage says when none was named
    (W52: six wordings, two printing a ``<stage>`` nobody can type).
    ``refs`` are the stages the verb takes, first the one to offer."""
    from ..identity import render_stage_choices
    refs = list(refs)
    if not refs:
        return "name a stage."
    return (f"name one -- {render_stage_choices(refs)}:\n"
            + block(lines(verb, kind, refs[0].name, base=base)))


def enabled_refs(task):
    """The description's stages that run, as refs (`identity.StageRef`) --
    numbered on the whole ladder, so ``#N`` names the same stage everywhere."""
    from ..identity import StageRef
    return [r for r, s in zip(StageRef.ladder([s.name for s in task.stages]),
                              task.stages)
            if getattr(s, "enabled", True) is not False]


def block(text_lines, pad: str = "    ") -> str:
    """Lines indented as a command block under a sentence."""
    return "\n".join(pad + ln for ln in text_lines)


def rollback(what: str, *, base) -> str:
    """How ``what`` is redone: the state saved before it, restored, and a
    prep anew -- a prepped stage is not prepped again (user, 2026-10-02:
    *"refuse it, redo via rollback"*; `job-system.md` § 5.0).

    WHICH STATE is the person's to pick, so it is asked for in words, never
    a ``<state>``; the list that shows them is a command, and both verbs
    name the folder -- by its path, `checkpoint`'s ``-p`` -- unless the
    reader stands in it."""
    here = Path(base).resolve()
    try:
        standing = here == Path.cwd().resolve()
    except OSError:
        standing = False
    p = "" if standing else f" -p {shlex.quote(str(here))}"
    return (f"go back to the state saved before {what}, and prep anew -- "
            f"the folder's saved states:\n"
            + block([f"molbuilder checkpoint list{p}"])
            + f"\n  then `molbuilder checkpoint restore{p}` with the id of "
              f"the one you pick.")


__all__ = ["PROG", "bundle_flag", "command", "configured_mode",
           "launch_lines", "lines", "run_first", "name_a_stage",
           "enabled_refs", "block", "rollback"]

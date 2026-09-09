"""Run-file names — composed and parsed in ONE place.

Contract: `docs/execution/job-contracts.md` § 2.2a (the name grammar), which
          § 2.1 rule 2 (one basename) and § 2.3 (only per-stage files carry the
          token) had both implied without making sayable.
Owns:     the grammar, the role vocabulary's *shape*, and both directions
          across it.
Called by: every writer that names a run file and every reader that looks one
          up.  Nothing composes or splits a run-file name inline.

WHY THIS EXISTS.  The rule was already written down twice and enforced nowhere,
so each site built its own name out of strings.  Measured on 2026-09-07: ONE
file -- geomeTRIC's trajectory -- had SIX spellings across the emitter, the
warm-file declaration, three places in the parser, and the catalogue in the
contract itself.  Only the emitter's was right, so on a staged run the parser's
own error message named a file that does not exist, and the warm-file carry
looked for one too.  Nothing failed: a missing warm file is a legal state, so
the ladder started cold in silence.

    <label>[_<stage>][-run<N>]<role>

    my-job.chk                        carried: no stage, no attempt
    my-job_optimized.xyz              carried, role-style separator
    my-job_01_coarse.molwatch.log     this rung's
    my-job-run2.out                   this attempt's
    my-job_01_coarse-run2.out         this rung's second attempt

THE TWO SEPARATORS ARE THE GRAMMAR (`job-contracts.md` § 6.3): *"a hyphen
announces a counter follows... a stage is not a counter -- it is a name"*.  So
``_`` introduces the stage and ``-run`` the attempt, and neither can be read as
the other.

The token sits IMMEDIATELY AFTER THE LABEL and never inside the role.  Five of
the six per-stage files already did that; the geomeTRIC pair was the one
exception, so the grammar moves one file rather than five (user, 2026-09-07).

A CARRIED FILE HAS NO TOKEN, and that is what carrying means: SIESTA's `.XV` /
`.DM` / `.CG` and PySCF's `.chk` / `_optimized.xyz` are how one rung hands the
next its geometry and density.  A token in them would make every rung look for
a file only its own stage ever wrote.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: A stage artifact token: `01_coarse`, `02_electrode_L`.  The ordinal is
#: **two or more** digits and the name is ``[A-Za-z0-9_]+`` -- `identity.py`'s
#: ``STAGE_NAME_RE`` and ``_STAGE_TOKEN_TAIL``, which own the word itself; this
#: is only the shape a NAME may carry, so a token that module would reject
#: cannot reach a filename through here either.
#:
#: IT WAS `[0-9]{2}_[A-Za-z0-9]+` FOR A DAY, and that is a lesson about
#: re-deriving instead of consulting: a transport ladder's rungs are
#: `02_electrode_L` and `03_device`, so `compose` refused to name any of the
#: transport decks.  The narrow pattern also made `parse` easy, and the real
#: one does not -- see :data:`_CARRIED_ROLES`.
_STAGE = re.compile(r"[0-9]{2,}_[A-Za-z0-9_]+")

#: The underscore-introduced roles an engine's WARM vocabulary contributes.
#:
#: **Why `parse` needs a vocabulary at all.** A stage name may contain ``_``
#: (`02_electrode_L`) and so may a role (`_geom_optim.xyz`), so the separator
#: cannot say where one ends and the other begins:
#: ``job_01_coarse_geom_optim.xyz`` is *either* stage `01_coarse` + role
#: `_geom_optim.xyz` *or* stage `01_coarse_geom_optim` + role `.xyz`, and only
#: knowing the roles decides it.  What molbuilder writes comes from
#: :data:`WRITTEN` below; these are the engines' own, named here because
#: `warm-files.toml` is read by `warmfiles`, which this module may not import
#: -- L1 grammar reaching an L1 TOML reader would put a file read behind every
#: filename.  A caller holding the real vocabulary passes it as ``roles``.
_CARRIED_ROLES = ("_optimized.xyz", "_geom_optim.xyz",
                  "_geom_optim.tmp", "_geom.tmp")

#: The basename rule, restated from `config/siesta.py::_validate_basename`
#: (job-contracts.md § 2.1: "a single token matching [A-Za-z0-9_-]+").  It is
#: restated rather than imported because THIS module must not depend on an
#: engine config to know what a label may look like -- the dependency runs the
#: other way.  A label carrying a dot would make `parse` ambiguous, which is
#: the one thing the pattern is load-bearing for.
_LABEL = re.compile(r"[A-Za-z0-9_-]+")


#: The COUNTERS a hyphen may introduce, in the order they appear in a name.
#:
#: § 6.3 gives the hyphen ONE meaning -- *"a hyphen announces a counter
#: follows... a stage is not a counter, it is a name"* -- and a counter is a
#: keyword plus a number.  ``run`` is the wrapper's attempt and is the only one
#: today; the point of naming the class is that the SECOND one is a line here
#: rather than an edit to the parser, which is where a hand-written
#: ``-run(\d+)`` had put it.
#:
#: A benchmark point is deliberately NOT one.  It is a whole LABEL
#: (``bdt-G1K4C6``), because a trial's warm files must never meet the run's
#: (`project-layout.md` § 2.3.2) -- a qualifier would put them back in one
#: name-space, which is the thing the separate label prevents.  Measured
#: 2026-09-07: ``bdt-G1K4C6_01_coarse-run2.out`` parses as that label with the
#: same three segments, one level down, and returns None against ``bdt``.
QUALIFIERS: "tuple[str, ...]" = ("run",)

_KEYWORDS = "|".join(QUALIFIERS)
#: One counter at the FRONT of what is left, and the same run of them anchored
#: at the END -- the two places a name can put them.  Both are built from
#: :data:`QUALIFIERS`, so neither can know a keyword the declaration does not.
_COUNTER = re.compile(r"-(" + _KEYWORDS + r")([0-9]+)")
_COUNTER_RUN_AT_END = re.compile(r"(?:-(?:" + _KEYWORDS + r")[0-9]+)+$")


class RunFileError(ValueError):
    """A name that cannot be composed, refused rather than guessed at."""


def _take_counters(text: str) -> "tuple[tuple[tuple[str, int], ...], str]":
    """Peel every declared counter off the FRONT of ``text``."""
    found: "list[tuple[str, int]]" = []
    while True:
        m = _COUNTER.match(text)
        if not m:
            return tuple(found), text
        found.append((m.group(1), int(m.group(2))))
        text = text[m.end():]


def _take_counters_at_end(text: str
                          ) -> "tuple[tuple[tuple[str, int], ...], str]":
    """The same, from the END -- where the role has already been taken off."""
    m = _COUNTER_RUN_AT_END.search(text)
    if not m:
        return (), text
    found, left = _take_counters(m.group(0))
    return found, text[:m.start()] if not left else text


def _in_declared_order(found: "tuple[tuple[str, int], ...]") -> bool:
    """Counters appear in :data:`QUALIFIERS` order, and each at most once.

    :func:`compose` emits them that way, so a name that does not is not one
    this module builds -- and reading it would give a `RunFile` whose own
    ``name`` differs from the string it came from.  Refused instead.
    """
    seen = [QUALIFIERS.index(k) for k, _ in found]
    return len(set(seen)) == len(seen) and seen == sorted(seen)


@dataclass(frozen=True)
class RunFile:
    """What a filename says it is.  ``stage`` is None for a carried file.

    ``counters`` holds every declared qualifier the name carried, in order, as
    ``(keyword, number)`` pairs -- a tuple rather than a dict so the record
    stays hashable and comparable.  :attr:`run` reads the one that exists
    today, which is what nearly every caller wants.
    """
    label: str
    stage: Optional[str]
    role:  str
    counters: "tuple[tuple[str, int], ...]" = ()

    @property
    def run(self) -> Optional[int]:
        """The wrapper's attempt, or None when the name carries no counter."""
        return dict(self.counters).get("run")

    @property
    def name(self) -> str:
        return compose(self.label, self.role, self.stage,
                       **dict(self.counters))


def stem(label: str, stage: Optional[str] = None) -> str:
    """``<label>[_<stage>]`` -- what every role attaches to.

    The half of :func:`compose` a caller needs on its own when the tail is not
    a role at all: the prep log appends ``.<engine>.<shape>.log``, and a deck's
    directory is named from the stem before any suffix exists.  Five call sites
    spelled ``f"{label}_{token}" if token else label`` until 2026-09-07, which
    is the same rule written five times and free to drift four ways.

    ``stage`` is None for a file that CARRIES between rungs.  An empty string
    is refused rather than read as None: a caller holding ``token = ""`` for
    *no ladder* should say ``token or None`` and mean it, because the two
    cases produce different filenames and a silent coercion is how a
    ladderless spelling reaches a staged run.
    """
    if not _LABEL.fullmatch(label or ""):
        raise RunFileError(
            f"label {label!r} is not a single [A-Za-z0-9_-]+ token "
            f"(job-contracts.md § 2.1).  A label carrying a dot or a space "
            f"cannot be read back out of a filename.")
    if stage is None:
        return label
    if not isinstance(stage, str):
        # A POSITION IS NOT A TOKEN, and this is the one wrong type worth
        # naming: passing `1` is how the old `-stage<N>` convention was built,
        # and a name keyed on a stage's POSITION silently reassigns outputs the
        # moment the ladder grows a rung (`job-contracts.md` § 6.3).  Caught
        # here rather than formatted in, so the caller hears about it at the
        # call and not in a directory of misfiled results.
        raise RunFileError(
            f"stage must be an artifact token like '01_coarse', not "
            f"{stage!r} ({type(stage).__name__}).  A POSITION is not a token: "
            f"a name keyed on one reassigns itself when the ladder grows.")
    if not _STAGE.fullmatch(stage):
        raise RunFileError(
            f"stage token {stage!r} is not `NN_name` (job-contracts.md "
            f"§ 2.3).  Pass None for a file that CARRIES between rungs; a "
            f"malformed token would produce a name nothing can parse back.")
    return f"{label}_{stage}"


def compose(label: str, role: str, stage: Optional[str] = None,
            run: Optional[int] = None, **counters: int) -> str:
    """The one way a run file gets its name (§ 2.2a).

    ``role`` is what the file IS and begins with ``.`` or ``_`` -- ``".chk"``,
    ``"_optimized.xyz"``, ``".molwatch.log"``.  It is declared once per engine
    in that engine's ``warm-files.toml`` and never spelled at a call site.

    ``stage`` omitted is a CARRIED file, which is a statement and not a
    default: it says this file crosses rungs.

    ``run`` is the wrapper's attempt (``-run2``), spelled out because it is the
    counter that exists; any other keyword in :data:`QUALIFIERS` is passed by
    name and lands in ``counters``.  All of them are COUNTERS and take a
    hyphen, where a stage is a NAME and takes an underscore -- § 6.3's rule,
    and the reason the two cannot be confused when read back.  They are emitted
    in declared order, so one name has one spelling.
    """
    if not role or role[0] not in "._":
        raise RunFileError(
            f"role {role!r} must begin with '.' or '_' -- it is the part that "
            f"says what the file IS, and the separator is what lets `parse` "
            f"find where the label ends.")
    if run is not None:
        counters = {"run": run, **counters}
    for key, value in counters.items():
        if key not in QUALIFIERS:
            raise RunFileError(
                f"{key!r} is not a counter this grammar knows "
                f"({', '.join(QUALIFIERS)}).  A hyphen announces a COUNTER "
                f"(job-contracts.md § 6.3); declare the keyword in "
                f"`runfiles.QUALIFIERS` rather than spelling it at a call.")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunFileError(
                f"{key} must be a non-negative counter, not {value!r}.")
    _counters = "".join(f"-{k}{counters[k]}"
                        for k in QUALIFIERS if k in counters)
    return f"{stem(label, stage)}{_counters}{role}"


#: A stand-in label for :func:`tail`.  Any legal label would do; what matters
#: is that the tail is CUT from a real `compose` result rather than assembled
#: beside one, so the two cannot disagree.
_PLACEHOLDER = "L"


def tail(role: str, stage: Optional[str] = None,
         run: Optional[int] = None, **counters: int) -> str:
    """Everything AFTER the label -- for a writer that has no label yet.

    A generated script names its own outputs at RUN time, from its own ``JOB``
    / ``SystemLabel`` variable, so the emitter can only supply the tail::

        output = _mb_outfile(JOB + '_01_coarse.log')

    That tail was assembled by hand at five emit sites, and one of them put the
    stage token in the wrong place -- geomeTRIC's prefix, giving
    ``<JOB>_geom_<stage>_optim.xyz`` where the declared role is
    ``_geom_optim.xyz``.  Cutting the tail off a real :func:`compose` result
    makes that impossible: whatever `compose` would produce, this is its end.

    For geomeTRIC, which takes a PREFIX and appends its own suffix, ask for the
    trajectory's tail and drop what geomeTRIC will add.
    """
    return compose(_PLACEHOLDER, role, stage, run,
                   **counters)[len(_PLACEHOLDER):]


def parse(filename: str, label: str,
          roles: "tuple[str, ...]" = ()) -> Optional[RunFile]:
    """Read a name back, or None when it is not this label's file.

    ``label`` is required, and that is the whole reason this can be exact: a
    role may contain underscores (``_geom_optim.xyz``) and a label may too, so
    nothing can find the boundary between them by looking at the string alone.
    Splitting on ``_`` is what several call sites did, and it is why
    ``my-job_01_coarse_geom_optim.xyz`` was read as a label of ``my-job`` and a
    stage of ``01`` at one site and ``01_coarse_geom`` at another.

    ``roles`` is the caller's own vocabulary, added to the one this module
    knows (:data:`WRITTEN` plus :data:`_CARRIED_ROLES`).  It only matters for
    roles that begin with ``_``: a stage NAME may contain ``_`` too, so those
    two are the case the separator cannot decide, and the vocabulary is what
    decides it.  A dotted role never needs it.

    A basename is accepted with or without directories in front of it.
    """
    base = filename.rsplit("/", 1)[-1]
    if not base.startswith(label):
        return None
    rest = base[len(label):]
    if not rest:
        return None

    # CARRIED: the role follows the label, possibly behind a counter.
    if rest[0] in ".-":
        counters, tail = _take_counters(rest)
        if tail and tail[0] in "._" and _in_declared_order(counters):
            return RunFile(label, None, tail, counters)
        return RunFile(label, None, rest) if rest[0] == "." else None
    if rest[0] != "_":
        return None                     # a longer label, not this one
    after = rest[1:]
    # PER-STAGE: `NN_name` then the role.  The token's own shape ends it --
    # `_geom_optim.xyz` cannot be mistaken for a token because a token starts
    # with two digits, which is why § 2.2a fixes the token's position rather
    # than trusting a separator.
    # A DECLARED UNDERSCORE ROLE WINS, because it is the only thing that can
    # tell `01_coarse` + `_geom_optim.xyz` from `01_coarse_geom_optim` + `.xyz`.
    # Longest first: `_geom_optim.xyz` must beat nothing, but a vocabulary that
    # grows a shorter suffix of a longer one would otherwise split too late.
    known = sorted({a.role for a in WRITTEN if a.role.startswith("_")}
                   | set(_CARRIED_ROLES)
                   | {r for r in roles if r.startswith("_")},
                   key=len, reverse=True)
    for role in known:
        if not after.endswith(role):
            continue
        counters, head = _take_counters_at_end(after[:-len(role)])
        if _STAGE.fullmatch(head) and _in_declared_order(counters):
            return RunFile(label, head, role, counters)
    # Otherwise the token runs to the role's own separator, which for every
    # remaining role is the first `.`.
    m = re.match(r"([0-9]{2,}_[A-Za-z0-9_]+)", after)
    if m:
        counters, role = _take_counters(after[m.end():])
        if role and role[0] == "." and _in_declared_order(counters):
            return RunFile(label, m.group(1), role, counters)
        if not role:
            return None                 # a token with no role is not a file
    # No token: the role itself began with `_` (`_optimized.xyz`).
    return RunFile(label, None, rest)


def find(directory, label: str, *,
         role: Optional[str] = None,
         roles: "tuple[str, ...]" = (),
         stage: Optional[str] = None,
         run: Optional[int] = None) -> "list[tuple[Path, RunFile]]":
    """Our files in *directory*, read back — the counterpart of :func:`compose`.

    **`project-layout.md` § 4.5: for every name it composes, the framework
    owns the search.**  Two callers spelled ``glob(f"{basename}-run*.{suffix}")``
    for the ``-run<N>`` counter whose one home is this module — `materialize`
    and `summarize` — and each carried its OWN regex to pull ``N`` back out, so
    the counter grammar was written three times to be composed once.  They
    bypassed this module because it offered nothing to bypass it WITH.

    Returns ``(path, RunFile)`` pairs, sorted by run index then name, so a
    caller that wants the latest takes the last and one that wants them all
    walks in order.  **The parsing is :func:`parse`'s**, not a second reader:
    a name this module cannot read is not ours and is left out, which is what
    keeps a foreign file in the same directory from being reported as an
    attempt.

    ``role`` narrows to one (``".out"``); ``roles`` passes a caller's own
    vocabulary through to :func:`parse` for the ``_``-prefixed kind, exactly as
    that function documents.  ``stage`` and ``run`` filter on what was parsed —
    ``run=None`` means *any*, and a file with no counter is matched only by
    ``run=None``, because "the file with no run index" and "run 0" are
    different things.

    Takes a directory and returns paths: stdlib ``pathlib`` only, so this stays
    importable by a monitor shipped beside a job (§ 4.5's floor rule).
    """
    d = Path(directory)
    try:
        # FILES, which is what the first line of this docstring says and what
        # every caller wants.  It iterated everything and checked nothing, so
        # a DIRECTORY whose name happened to parse would have come back as a
        # run file -- unreachable in today's layouts (a stage directory
        # carries no label prefix) and still a claim the function was not
        # keeping.  `find_by_role` below has always checked; the two halves
        # of one door disagreed.
        #
        # Not sorted here: the sort at the end orders by run index and
        # replaces any order this produced.
        entries = [e for e in d.iterdir() if e.is_file()]
    except OSError:
        return []
    out: "list[tuple[Path, RunFile]]" = []
    for entry in entries:
        parsed = parse(entry.name, label, roles)
        if parsed is None:
            continue
        if role is not None and parsed.role != role:
            continue
        if stage is not None and parsed.stage != stage:
            continue
        if run is not None and parsed.run != run:
            continue
        out.append((entry, parsed))
    out.sort(key=lambda pair: (pair[1].run if pair[1].run is not None else -1,
                               pair[0].name))
    return out


def find_by_role(directory, role: str) -> "list[Path]":
    """Every file here in this ROLE, whoever's it is — sorted by name.

    The label-less half of :func:`find`, and it exists because two callers
    genuinely have no label: *"which deck is in this directory"* and *"which
    molwatch logs are here"* are asked of a folder before anything has said
    whose it is.  They globbed ``"*.fdf"`` and ``"*.molwatch.log"``, which is
    the role vocabulary spelled outside the module that declares it.

    **Only a DOTTED role, and that is this module's own rule** rather than a
    limitation invented here — :func:`parse` states it: the vocabulary
    matters *"only for roles that begin with ``_``: a stage NAME may contain
    ``_`` too, so those two are the case the separator cannot decide ... A
    dotted role never needs it."*  So a dotted role is recognisable with no
    label, and an underscore one is refused rather than answered wrongly.

    The role is checked against :data:`WRITTEN`, so a typo is a refusal here
    instead of an empty list at the call site.
    """
    if not role.startswith("."):
        raise RunFileError(
            f"find_by_role({role!r}): only a dotted role can be found without "
            f"a label -- an underscore role cannot be told from a stage name "
            f"without one (see `parse`).  Use `find(dir, label, role=...)`.")
    known = {a.role for a in WRITTEN}
    if role not in known:
        raise RunFileError(
            f"find_by_role({role!r}): not a role molbuilder writes.  "
            f"The catalogue is `runfiles.WRITTEN`; known dotted roles are "
            + ", ".join(sorted(r for r in known if r.startswith("."))))
    d = Path(directory)
    try:
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.name.endswith(role))
    except OSError:
        return []


def latest_run(directory, label: str, *,
               roles: "tuple[str, ...]" = (),
               stage: Optional[str] = None,
               role: Optional[str] = None) -> Optional[int]:
    """The highest ``-run<N>`` present here, or ``None`` when nothing carries one.

    The question both callers actually asked.  `materialize` asked it across
    three roles at once to find the newest attempt's marker; `summarize` asked
    it per role to read the newest log.  Neither wanted the files — they wanted
    the number — and asking for it by name is what stops the next caller
    writing a fourth ``-run*`` glob.
    """
    runs = [rf.run for _p, rf in
            find(directory, label, role=role, roles=roles, stage=stage)
            if rf.run is not None]
    return max(runs) if runs else None


def is_carried(f: "RunFile | str", label: str = "") -> bool:
    """Does this name cross rungs?  A carried file states it by having no token.

    Accepts a parsed :class:`RunFile` or a filename plus its label, so a caller
    holding either can ask without composing a third thing.
    """
    if isinstance(f, RunFile):
        return f.stage is None
    parsed = parse(f, label)
    return parsed is not None and parsed.stage is None


# --------------------------------------------------------------------- #
#  WHAT MOLBUILDER ITSELF WRITES -- the catalogue, and the two views of  #
#  it (`job-contracts.md` § 2.2, § 4.2).                                 #
# --------------------------------------------------------------------- #
#
# THE LIST THAT CAN BE COMPLETE, and that is the whole point of it.  An
# engine's output set depends on its version and on which options are on, so
# enumerating THAT is a snapshot pretending to be a rule.  What WE write is
# knowable, because we write it -- so every question of the form *"did the
# engine leave something here"* is answered by subtraction (§ 4.2).
#
# It lives beside the grammar because both readers need the same table and
# neither may hold its own copy:
#
#   * :func:`patterns` -- the glob family, which `identity.OUR_FILE_PATTERNS`
#     IS and which `runwrap`'s `--cold` sweep derives its exception list from.
#   * :func:`manifest` -- the concrete names, for a person being told what a
#     prep is about to write (user, 2026-09-07: *"a card that list all the
#     generated data file from the setup"*).
#
# The two used to be one hand-written glob list with no name view at all, and
# it had drifted: `{label}_geom_*.log` was geomeTRIC's opt log spelled the way
# the OLD name put the stage token INSIDE the role.  Since the token moved to
# its one position (§ 2.2a) nothing writes that, and the file that IS written
# -- `<label>_<stage>_geom.log` -- matched no row, so molbuilder's own log was
# reported back to the user as the engine's warm state and `--cold` counted it
# among what it would clobber.  Measured 2026-09-07.


@dataclass(frozen=True)
class Artifact:
    """One kind of file molbuilder writes, and the shapes its name takes.

    ``role`` is the suffix :func:`compose` takes.  ``what`` is one line a
    person can read -- these are shown in the Task-setup card, so it says what
    the file HOLDS, not which module writes it.
    """
    role: str
    what: str
    #: Can this file carry a stage token?  False for the three that belong to
    #: the calculation rather than to a rung -- the template and the source
    #: pair are written once, at the bundle root.
    staged: bool = True
    #: The wrapper's attempt counter: "never", "maybe" (both spellings exist --
    #: the wrapper indexes, a hand-run does not) or "always".
    attempt: str = "never"
    #: Which engine writes it, or None for both.
    engine: Optional[str] = None
    #: WHEN it appears -- "setup" (the description hand-over), "prep" (the
    #: deck and its wrapper) or "run" (everything the launch produces).  A
    #: card that lists a run's files has to say which of them exist yet, and
    #: the alternative was for the page to subtract one list from another.
    when: str = "run"
    #: Which CALCULATION writes it, or None for any.  A relaxation has no
    #: spectrum, and a list that promised one would be describing a file the
    #: run will never write -- the same fault in the other direction as a
    #: file that is written and undeclared.
    calculation: Optional[str] = None


#: Ordered as `job-contracts.md` § 2.2 lists them: inputs, the wrapper, the
#: canonical trajectory, then the run-indexed logs -- which are HISTORY and not
#: state, and nothing here may treat them as leftovers.
WRITTEN: "tuple[Artifact, ...]" = (
    # ---- what we generated to run -------------------------------------
    Artifact(".fdf", "the SIESTA deck — every keyword this rung runs with",
             engine="siesta",
             when="prep"),
    Artifact(".py", "the PySCF script this rung runs", engine="pyscf",
             when="prep"),
    # THE SUFFIX IS SPELLED, NOT IMPORTED, and that is the LAYERING rule
    # rather than a lapse: this module is L1 and `template` is L2, so
    # importing it here is the violation `tests/test_layering.py` catches
    # (tried 2026-08-17 in `identity`, and reverted).  The cost is real --
    # the glob view answers *"did the engine leave this, or did we write
    # it"* by subtraction, so a suffix that silently stops matching hands a
    # person their own input back as engine state.  What guards it instead
    # is `test_doc_claims.py`'s template-path test, which exempts this one
    # module BY NAME.
    Artifact(".template.toml", "every parameter, with the value it was given",
             staged=False,
             when="setup"),
    # THE STRUCTURE THE CALCULATION IS OF, written into the bundle by the
    # hand-over (`web/handover-procedure.md`).  Added 2026-08-16, the same
    # day molbuilder started writing them: before that the pair did not
    # exist in a bundle, so the subtraction never saw it -- and the moment
    # it did, `prep` announced a fresh calculation as *"already under way --
    # warm files at the root"* and offered a person's own input back to them
    # as engine state.
    #
    # `.source` is the RESERVATION (`job-contracts.md` § 6.3): identities are
    # validated dot-free, so this is a name no engine output can take, in any
    # shape.  A bare `{label}.xyz` was tried and is wrong -- a FLAT engine
    # runs at the bundle root, and the first flat relaxation whose label
    # matched the structure's stem had `WriteCoorXmol` overwrite the
    # description's own input (2026-08-19).  `{label}.xyz` at the root is
    # therefore the ENGINE's, and it is deliberately absent from this table.
    Artifact(".source.xyz", "the structure the calculation is of",
             staged=False,
             when="setup"),
    Artifact(".source.molstruct.json", "its cell and its region labels",
             staged=False,
             when="setup"),
    # THE DECK'S COMPANION REPORT (`script_emit.VALIDATION_SUFFIX`, added
    # 2026-08-23 with the file itself).  Here for the reason the `.source`
    # pair records: a file molbuilder writes and does not declare reads as
    # ENGINE OUTPUT, and `prep` then greets a fresh calculation with *"already
    # under way"*, offering the user their own report back as run state.
    Artifact(".validation.txt", "what the generator checked before it wrote "
                                "the deck",
             when="prep"),
    # ---- what the DECK writes for itself ------------------------------
    # A file the generated script writes is molbuilder's too -- it is our
    # deck that writes it -- and the three below were on NEITHER list until
    # 2026-09-07 (measured by rendering both PySCF decks and asking `is_ours`
    # of every name they choose).  So the subtraction called them the
    # engine's restart state: `--cold` offered to clobber a run's own
    # spectrum, and `prep` greeted a fresh calculation by naming the input
    # geometry it had written itself.  None of them is warm -- nothing reads
    # them back -- which is why declaring them here is the whole fix.
    Artifact("_initial.xyz", "the input geometry, echoed back before "
                             "anything ran", staged=False, engine="pyscf"),
    Artifact(".constraints.txt", "which atoms are held still, in geomeTRIC's "
                                 "own format — written only when some are",
             staged=False, engine="pyscf"),
    Artifact(".spectra.json", "the spectrum this run computed: frequencies, "
                              "intensities, thermochemistry", staged=False,
             engine="pyscf", calculation="vibration"),
    # ---- the wrapper --------------------------------------------------
    Artifact(".run.sh", "the wrapper — activates the environment, tees "
                        "the output, catches a kill",
             when="prep"),
    Artifact(".sbatch", "the queue header, written only when the run is "
                        "submitted rather than started here",
             when="prep"),
    # ---- the canonical trajectory -------------------------------------
    # Written before the engine even starts.
    Artifact(".molwatch.log", "the run as it happens — coordinates, "
                              "energy and forces, one block per step"),
    # ---- history: stdout and the logs ---------------------------------
    # ALL OF THIS IS HISTORY AND NOT STATE -- it is what a person goes back
    # to read, and nothing may treat it as leftovers.  The rows were missing
    # until 2026-08-13 (final review E-2) and the run's own stdout was being
    # offered back as the ENGINE's restart state, while the `--cold` sweep
    # moved a prior stage's logs aside.
    #
    # `attempt="maybe"` is two real spellings, not indecision: the wrapper
    # indexes its redirect (`-run0.out`) and a hand-started run does not.
    Artifact(".out", "the run's output as the engine printed it",
             attempt="maybe"),
    Artifact(".pyscf.log", "the same, for PySCF under the wrapper — it "
                           "writes here and not to .out", attempt="always",
             engine="pyscf"),
    Artifact(".log", "the engine's verbose log"),
    # GEOMETRIC'S OPT LOG, and the row that recorded the drift this catalogue
    # exists to end.  It was spelled `{label}_geom_*.log` -- the token INSIDE
    # the role, which is how the name read before § 2.2a fixed the token's
    # position.  Nothing has written that since; what IS written,
    # `<label>_<stage>_geom.log`, matched no row, so our own log came back to
    # the user as the engine's warm state (measured 2026-09-07).
    Artifact("_geom.log", "geomeTRIC's optimizer log", engine="pyscf"),
    Artifact(".runwrap-*.log", "the wrapper's own session log — one per "
                               "launch, stamped with the clock"),
    # The monitor's two files gained the wrapper's run index on 2026-08-27,
    # so both spellings are listed: a directory can hold artifacts from
    # before the change, and a cold sweep that misses one leaves it to be
    # appended to or truncated by the next run.
    Artifact(".monitor.log", "the monitor's rolling status", attempt="maybe"),
    Artifact(".util.csv", "processor and memory samples taken while it ran",
             attempt="maybe"),
    Artifact(".scf-timing.log", "wall time per SCF iteration",
             attempt="always"),
    # THE CONCLUSION MARKER -- the wrapper's last act on its main path
    # (`project-layout.md` § 1.6, "the other file", 2026-08-28).  Indexed
    # like the stdout, because a warm-retry chain execs fresh wrappers and
    # only the FINAL process concludes.
    Artifact(".concluded", "the marker the wrapper writes when the job ends",
             attempt="always"),
)

#: The attempt counter as a GLOB, per :attr:`Artifact.attempt`.  Empty string
#: means the unindexed spelling; ``-run*`` the wrapper's.
_ATTEMPT_GLOBS = {"never": ("",), "maybe": ("", "-run*"), "always": ("-run*",)}

#: THE FIRST ATTEMPT IS ZERO (`runwrap`: ``_run_n=0   # first run``), so this
#: is the real name the first launch writes -- not a placeholder.
FIRST_ATTEMPT = 0


def patterns(artifacts: "Optional[tuple[Artifact, ...]]" = None
             ) -> "tuple[str, ...]":
    """The ``{label}``-keyed glob family for :data:`WRITTEN`.

    GLOBS, NOT NAMES, which is why this does not go through :func:`compose`:
    ``{label}_*`` stands for every stage token at once and no single call could
    produce it.  What ties the two together is a property rather than a shared
    call -- every name :func:`compose` can build for one of these roles is
    matched here -- and that is what the suite checks.
    """
    out = []
    for a in artifacts if artifacts is not None else WRITTEN:
        stems = ["{label}"] + (["{label}_*"] if a.staged else [])
        for stem in stems:
            for run in _ATTEMPT_GLOBS[a.attempt]:
                name = f"{stem}{run}{a.role}"
                if name not in out:
                    out.append(name)
    return tuple(out)


def manifest(label: str, stage: Optional[str] = None,
             engine: Optional[str] = None,
             when: "Optional[tuple]" = None,
             calculation: Optional[str] = None) -> "list[dict]":
    """The names one rung writes, each with a line saying what the file is.

    ``stage`` given answers for THAT RUNG and lists only what carries its
    token.  ``stage`` omitted answers for the CALCULATION and lists only what
    does not -- the template and the source pair, written once at the bundle
    root.  The two sets do not overlap, which is the point: a rung's card that
    repeated the calculation's files would say each of them N times and imply
    N copies.

    ``engine`` keeps a SIESTA run from being told about ``.py`` and a PySCF one
    about ``.fdf``.  Passing None answers for both, which is what a caller that
    does not yet know the engine should show.

    ``when`` narrows to the moments asked for (:attr:`Artifact.when`) -- the
    Task-setup page already lists the SETUP's own files with their existence
    state, so its second card asks for everything else rather than subtracting
    one list from the other.

    Every name comes out of :func:`compose`, so a card cannot show a spelling
    the writers do not use -- which is the failure this module exists for.
    """
    rows = []
    for a in WRITTEN:
        if a.staged != (stage is not None):
            continue
        if engine and a.engine and a.engine != engine:
            continue
        if when and a.when not in when:
            continue
        if calculation and a.calculation and a.calculation != calculation:
            continue
        run = None if a.attempt == "never" else FIRST_ATTEMPT
        rows.append({"name": compose(label, a.role, stage, run),
                     "what": a.what,
                     "when": a.when,
                     "carries_attempt": a.attempt != "never"})
    return rows

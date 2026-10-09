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
    my-job_01_coarse.fdf              this rung's
    my-job_01_coarse-run2.out         this rung's run 2

THE TWO SEPARATORS ARE THE GRAMMAR (`job-contracts.md` § 6.3): *"a hyphen
announces a counter follows... a stage is not a counter -- it is a name"*.  So
``_`` introduces the stage and ``-run`` the attempt, and neither can be read as
the other.

The token sits IMMEDIATELY AFTER THE LABEL and never inside the role.

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
#: ``STAGE_NAME_RE`` and ``stage_token``, which own the word itself; this
#: is only the shape a NAME may carry, so a token that module would reject
#: cannot reach a filename through here either.
_STAGE = re.compile(r"[0-9]{2,}_[A-Za-z0-9_]+")

#: The underscore-introduced roles of the files an ENGINE itself writes: what
#: it warm-starts from (``_optimized.xyz``, `pyscf/warm-files.toml`) and what
#: geomeTRIC writes under the prefix the deck hands it -- its trajectory and
#: its scratch, which are not warm state (`engines/stages.md` § 1.1a,
#: consequence 4) and are named here all the same, because a reader still
#: has to tell them apart from a stage.
#:
#: **Why `parse` needs a vocabulary at all.** A stage name may contain ``_``
#: (`02_electrode_L`) and so may a role (`_geom_optim.xyz`), so the separator
#: cannot say where one ends and the other begins:
#: ``job_01_coarse_geom_optim.xyz`` is *either* stage `01_coarse` + role
#: `_geom_optim.xyz` *or* stage `01_coarse_geom_optim` + role `.xyz`, and only
#: knowing the roles decides it.  What molbuilder writes comes from
#: :data:`WRITTEN` below; these are the engines' own, named here because the
#: engines' vocabularies are read by modules this one may not import -- L1
#: grammar reaching an L1 TOML reader would put a file read behind every
#: filename.  A caller holding the real vocabulary passes it as ``roles``.
_ENGINE_ROLES = ("_optimized.xyz", "_geom_optim.xyz", "_geom.tmp")

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
#: rather than an edit to the parser.
#:
#: A benchmark point is deliberately NOT one.  It is a whole LABEL
#: (``bdt-G1K4C6``), because a trial's warm files must never meet the run's
#: (`project-layout.md` § 2.3.2) -- a qualifier would put them back in one
#: name-space, which is the thing the separate label prevents.  Measured
#: 2026-09-07: ``bdt-G1K4C6_01_coarse-run2.out`` parses as that label with the
#: same three segments, one level down, and returns None against ``bdt``.
QUALIFIERS: "tuple[str, ...]" = ("run",)

#: Every declared FIELD, in ONE place -- the same discipline
#: :data:`QUALIFIERS` gives counters.
#:
#: **A field is the bounded flexibility** (`plans/plan.md` § 5l.1). A role names
#: what a file IS; a field carries a value that varies per file and belongs to
#: the NAME rather than to the vocabulary.
#:
#: **Why not a counter.** § 6.3's separator rule is that a hyphen announces a
#: COUNTER and an underscore a NAME; a counter is a declared keyword followed
#: by digits (``-run2``) and is ordered, so two names cannot spell one thing.
#: A stamp is neither a keyword nor a number and carries no ordering claim, so
#: it is a third thing and says so.
@dataclass(frozen=True)
class Field:
    """One declared field: what it means, its shape, and one real example.

    ``example`` is not decoration.  It is what lets anything WALK the catalogue
    and build a name -- the suite composes every role at every stage and every
    attempt, and a template it cannot fill is a row it cannot check.  It is also
    how a reader learns what ``[0-9]{8}-[0-9]{6}`` looks like without decoding
    it, and a test asserts it matches its own shape, so it cannot rot.
    """
    what: str
    shape: str
    example: str


FIELDS: "dict[str, Field]" = {
    "stamp": Field(
        what="when this launch started, from the wrapper's own clock",
        # `runwrap`: `date +%Y%m%d-%H%M%S`
        shape=r"[0-9]{8}-[0-9]{6}",
        example="20260908-120000"),
    # The fields of the catalogue's FIXED-NAME families (B13, 2026-10-04):
    # files molbuilder writes that are not named on the label -- a
    # pseudopotential is named for its element, SLURM's output for its job.
    "element": Field(
        what="the chemical element a pseudopotential is for",
        shape=r"[A-Z][a-z]?",
        example="Au"),
    "jobid": Field(
        what="the scheduler's id for the job",
        shape=r"[0-9]+",
        example="62372574"),
    "pid": Field(
        what="the run script's own process id",
        shape=r"[0-9]+",
        example="48213"),
    "group": Field(
        what="a launch group's name: a benchmark's side and shelf, or a "
             "bias sweep's chain",
        shape=r"[A-Za-z0-9_.-]+",
        example="bench-group-gpu"),
    "random": Field(
        what="a temporary file's random part",
        shape=r"[A-Za-z0-9_.-]+",
        example="k3j9x_2a"),
    "engine": Field(
        what="the engine a prep rendered for",
        shape=r"siesta|pyscf",
        example="siesta"),
    "shape": Field(
        what="the calculation's shape",
        shape=r"flat|hierarchical",
        example="hierarchical"),
}

_FIELD_RE = re.compile(r"\{([a-z_]+)\}")


def _fields_in(role: str) -> "tuple[str, ...]":
    """The field names a role TEMPLATE carries, in the order they appear."""
    return tuple(_FIELD_RE.findall(role))


def _role_pattern(role: str) -> "re.Pattern":
    """A templated role, compiled -- each field replaced by its declared shape."""
    out, last = [], 0
    for m in _FIELD_RE.finditer(role):
        out.append(re.escape(role[last:m.start()]))
        name = m.group(1)
        if name not in FIELDS:
            raise RunFileError(
                f"role {role!r} names the field {name!r}, which has no shape. "
                f"Declare it in `runfiles.FIELDS`; a field whose shape is "
                f"not written down cannot be read back out of a filename.")
        out.append(f"(?P<{name}>{FIELDS[name].shape})")
        last = m.end()
    out.append(re.escape(role[last:]))
    return re.compile("".join(out) + r"\Z")


def canonical_role(concrete: str) -> "tuple[str, dict]":
    """A role read off a filename, mapped back to the TEMPLATE it matches.

    ``(".runwrap-20260908-120000.log")`` -> ``(".runwrap-{stamp}.log",
    {"stamp": "20260908-120000"})``, and anything that matches no template comes
    back unchanged with no fields. This is what lets every comparison in this
    module stay an EQUALITY on the declared role.
    """
    for a in ON_THE_LABEL:
        if not a.fields:
            continue
        m = _role_pattern(a.role).match(concrete)
        if m:
            return a.role, m.groupdict()
    return concrete, {}

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
    #: The declared FIELDS the name carried, as ``(name, value)`` pairs -- a
    #: tuple for the reason ``counters`` is one.  :attr:`role` is the TEMPLATE
    #: for a name that carries fields, so two wrapper logs from two launches
    #: compare equal on role and differ here, which is what a field is for.
    fields: "tuple[tuple[str, str], ...]" = ()

    @property
    def run(self) -> Optional[int]:
        """The wrapper's attempt, or None when the name carries no counter."""
        return dict(self.counters).get("run")

    @property
    def name(self) -> str:
        return compose(self.label, self.role, self.stage,
                       **dict(self.counters), **dict(self.fields))


def stem(label: str, stage: Optional[str] = None) -> str:
    """``<label>[_<stage>]`` -- what every role attaches to.

    The half of :func:`compose` a caller needs on its own: a run's basename,
    which every file of its rung begins with (:attr:`RunNames.stem`).

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
        # naming: a name keyed on a stage's POSITION silently reassigns outputs the
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
    # FIELDS FIRST: a templated role is filled in before anything else looks at
    # it, so everything downstream sees a concrete name.  A field is refused
    # unless the role declares it, and a declared one is refused unless its
    # value matches the shape in `FIELDS` -- a stamp that is not a stamp
    # would produce a name `parse` could not read back.
    _wanted = _fields_in(role)
    if _wanted:
        _given = {k: counters.pop(k) for k in list(counters) if k in _wanted}
        missing = [f for f in _wanted if f not in _given]
        if missing:
            raise RunFileError(
                f"role {role!r} needs the field(s) {', '.join(missing)}; a "
                f"template cannot be composed without them.  Pass "
                f"{missing[0]}=... , or ask `patterns()` if you want the glob.")
        for k, v in _given.items():
            # A FIELD LEFT AS ITS PLACEHOLDER is a template's
            # (`RunNames.template`): the script it is rendered into fills it.
            if v == "{" + k + "}":
                continue
            if not re.fullmatch(FIELDS[k].shape, str(v)):
                raise RunFileError(
                    f"{k}={v!r} does not match the declared shape "
                    f"{FIELDS[k].shape!r} for that field "
                    f"({FIELDS[k].what}; e.g. {FIELDS[k].example}).")
        role = _FIELD_RE.sub(lambda m: str(_given[m.group(1)]), role)
    for key, value in counters.items():
        if key not in QUALIFIERS:
            raise RunFileError(
                f"{key!r} is not a counter this grammar knows "
                f"({', '.join(QUALIFIERS)}).  A hyphen announces a COUNTER "
                f"(job-contracts.md § 6.3); declare the keyword in "
                f"`runfiles.QUALIFIERS` rather than spelling it at a call.")
        if key == "run" and value == RUN_FIELD:
            continue                    # a template's: its run fills it
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunFileError(
                f"{key} must be a non-negative counter, not {value!r}.")
    _counters = "".join(f"-{k}{counters[k]}"
                        for k in QUALIFIERS if k in counters)
    return f"{stem(label, stage)}{_counters}{role}"


def run_name(label: str, stage: Optional[str], run: int) -> str:
    """``<label>_<stage>-run<N>`` -- what every file of ONE run carries, and
    so the flat layout's name for that run: flat tells runs apart by this
    index, never by a directory (`project-layout.md` § 1.5a).  CUT from a
    real :func:`compose` result, like :func:`tail`, so the counter is
    spelled by the grammar and nowhere else."""
    _role = ".concluded"
    return compose(label, _role, stage, run=run)[:-len(_role)]


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

    Cut off a real :func:`compose` result: whatever `compose` would produce,
    this is its end.

    For geomeTRIC, which takes a PREFIX and appends its own suffix, ask for the
    trajectory's tail and drop what geomeTRIC will add.

    **:func:`compose` when you own the label; this when you were handed a stem.**
    """
    return compose(_PLACEHOLDER, role, stage, run,
                   **counters)[len(_PLACEHOLDER):]


def _carrying(label: str, stage: Optional[str], role: str,
              counters: "tuple[tuple[str, int], ...]" = ()) -> "RunFile":
    """A :class:`RunFile` whose role has been mapped back to its template.

    Every arm of :func:`parse` builds its result through here, so a concrete
    role and its declared template can never disagree about which file this is.
    """
    template, values = canonical_role(role)
    return RunFile(label, stage, template, counters,
                   tuple(sorted(values.items())))


def parse(filename: str, label: str,
          roles: "tuple[str, ...]" = ()) -> Optional[RunFile]:
    """Read a name back, or None when it is not this label's file.

    ``label`` is required, and that is the whole reason this can be exact: a
    role may contain underscores (``_geom_optim.xyz``) and a label may too, so
    nothing can find the boundary between them by looking at the string alone.

    ``roles`` is the caller's own vocabulary, added to the one this module
    knows (:data:`WRITTEN` plus :data:`_ENGINE_ROLES`).  It only matters for
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
            return _carrying(label, None, tail, counters)
        return _carrying(label, None, rest) if rest[0] == "." else None
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
                   | set(_ENGINE_ROLES)
                   | {r for r in roles if r.startswith("_")},
                   key=len, reverse=True)
    for role in known:
        if not after.endswith(role):
            continue
        counters, head = _take_counters_at_end(after[:-len(role)])
        if _STAGE.fullmatch(head) and _in_declared_order(counters):
            return _carrying(label, head, role, counters)
    # Otherwise the token runs to the role's own separator, which for every
    # remaining role is the first `.`.
    m = re.match(r"([0-9]{2,}_[A-Za-z0-9_]+)", after)
    if m:
        counters, role = _take_counters(after[m.end():])
        if role and role[0] == "." and _in_declared_order(counters):
            return _carrying(label, m.group(1), role, counters)
        if not role:
            return None                 # a token with no role is not a file
    # No token: the role itself began with `_` (`_optimized.xyz`).
    return _carrying(label, None, rest)


def _tail(name: str, role: str) -> str:
    """The end of *name*, long enough to hold *role* -- for a label-less match.

    A templated role has no fixed length, so the slice is taken from the first
    literal segment of the template (`.runwrap-` for `.runwrap-{stamp}.log`).
    """
    head = role.split("{", 1)[0]
    if head != role:
        i = name.find(head)
        return name[i:] if i >= 0 else name
    return name[-len(role):] if len(role) <= len(name) else name


def role_of(name) -> Optional[str]:
    """The DECLARED role this filename carries, or None — WITHOUT a label.

    :func:`parse` needs the label because a role may contain ``_`` and so may
    a stage name, and nothing can find the boundary between them from the
    string alone.  A DOTTED role escapes that, which is the rule
    :func:`find_by_role` already states — so a caller holding one path and no
    label can still ask *what IS this file*, which is what
    `parse.engines._run_ending.ending_of` dispatches on.

    Underscore roles are not answered, for `find_by_role`'s reason: without a
    label they cannot be told from a stage name.

    **THE LONGEST DECLARED ROLE WINS**, and that is not a tie-break — it is
    what the file IS.  ``job-run0.pyscf.log`` ends with `.pyscf.log` AND with
    `.log`, and only the first says which file this is.
    """
    base = str(name).rsplit("/", 1)[-1]
    best: Optional[str] = None
    for a in WRITTEN:
        if not a.role.startswith("."):
            continue
        if canonical_role(_tail(base, a.role))[0] != a.role:
            continue
        if best is None or len(a.role) > len(best):
            best = a.role
    return best


def find(directory, label: str, *,
         role: Optional[str] = None,
         roles: "tuple[str, ...]" = (),
         stage: Optional[str] = None,
         run: Optional[int] = None) -> "list[tuple[Path, RunFile]]":
    """The label's files in *directory*, read back — the counterpart of
    :func:`compose`: molbuilder's, and the engine's own named on the label
    (SIESTA's ``<label>.XV``, PySCF's ``<label>.chk`` -- the carried files of
    `job-contracts.md` § 2.2a), each in the role its name spells.  Which of
    them molbuilder wrote is the catalogue's question (:func:`row_for`, the
    run door's `about`): what molbuilder did not write, under the run's
    name, is the engine's -- found by that name, never by a list of what an
    engine writes (§ 4.2).

    **`project-layout.md` § 4.5: for every name it composes, the framework
    owns the search.**

    Returns ``(path, RunFile)`` pairs, sorted by run index then name, so a
    caller that wants the latest takes the last and one that wants them all
    walks in order.  **The parsing is :func:`parse`'s**, not a second reader:
    a name that does not read back with this label -- another label's, a
    file foreign to the run -- is left out, which is what keeps a foreign
    file in the same directory from being reported as an attempt.  An
    engine's file can carry a stage too -- geomeTRIC's trajectory is named
    on its rung's token (``<label>_<NN>_<stage>_geom_optim.xyz``) -- so a
    caller asks by role as well when it wants one of molbuilder's.

    ``role`` narrows to one, and it is an EQUALITY on the declared role -- pass
    the catalogue's spelling, template and all (``".runwrap-{stamp}.log"``).
    Every name is mapped back to its template by :func:`canonical_role` before
    the comparison, so one rule reads and one rule matches.  ``roles`` passes a caller's own
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
        # every caller wants.
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
    whose it is.

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
        # EQUALITY on the canonical role, through :func:`role_of` -- which
        # reads each NAME's own role rather than testing it against the one
        # asked for.  The difference is the `.log` case that function
        # records.
        return sorted(p for p in d.iterdir()
                      if p.is_file() and role_of(p.name) == role)
    except OSError:
        return []


def latest_run(directory, label: str, *,
               roles: "tuple[str, ...]" = (),
               stage: Optional[str] = None,
               role: Optional[str] = None) -> Optional[int]:
    """The highest ``-run<N>`` present here, or ``None`` when nothing carries one."""
    runs = [rf.run for _p, rf in
            find(directory, label, role=role, roles=roles, stage=stage)
            if rf.run is not None]
    return max(runs) if runs else None


def at_run(directory, files, stem: str,
           run: Optional[int] = None) -> "list[Path]":
    """Those of ``files`` that belong to run ``run`` of the stage ``stem``
    names -- by default its LATEST run index -- newest first.  A file that
    carries no run index (the hierarchy's folder-wide names) counts for
    every run.

    A per-run file counts only at the HIGHEST index any per-run file of the
    run reached, across every role: a warm-retry chain execs a fresh wrapper
    per run, so an earlier index's file beside a newer run's is a previous
    run's -- its goodbye, or its monitor's -- and says nothing about the run
    that followed it.  Across every role, because an engine that dies before
    printing leaves a marker and no output at all.

    ``stem`` is the run's own, ``<label>[_<stage>]`` -- its caller's, who
    knows which run it asks about (`execution/architecture.md` § 3.2).  The
    run index orders; a file's time does not (`model/parse.md` § 5.1).
    """
    d = Path(directory)
    wanted = latest_run(d, stem) if run is None else run
    kept = []
    for f in files:
        got = parse(f.name, stem)
        idx = got.run if got else None
        if wanted is not None and idx is not None and idx != wanted:
            continue
        kept.append((idx if idx is not None else -1, f))
    return [f for _key, f in sorted(kept, key=lambda k: k[0], reverse=True)]


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


@dataclass(frozen=True)
class Artifact:
    """One kind of file molbuilder writes, and the shapes its name takes.

    ``role`` is the suffix :func:`compose` takes, for a file named on the
    label; ``name`` is the whole name of one that is not (`task.json`, a
    pseudopotential's ``{element}.psml``) -- a row has exactly one of the two.
    ``what`` is one line a person can read -- these are shown in the
    Task-setup card and the Results tab's file card, so it says what the file
    HOLDS; ``writer`` says who writes it, and when.
    """
    role: str = ""
    what: str = ""
    #: The FIELDS this role's template carries, if any -- the bounded
    #: flexibility (`plans/plan.md` § 5l.1).  A role naming ``{stamp}`` is a
    #: family of names, and the field is what tells them apart; the shape of
    #: each is declared once in :data:`FIELDS`.
    fields: "tuple[str, ...]" = ()
    #: Can this file carry a stage token?  False for the three that belong to
    #: the calculation rather than to a rung -- the template and the source
    #: pair are written once, at the bundle root.
    staged: bool = True
    #: The run's number in the name: "never", "maybe" (both spellings exist
    #: -- the wrapper indexes, a hand-run does not), "always", or "shared":
    #: where a stage's runs share one folder -- a flat calculation's stage --
    #: and not where each run has a folder of its own, which names the run
    #: (`project-layout.md` § 1.5a, plan W57 decision 2).  Read by
    #: :class:`RunNames`, the one composer of a stage's names.
    attempt: str = "never"
    #: Which engine writes it, or None for both.
    engine: Optional[str] = None
    #: WHEN it appears -- "setup" (the description hand-over), "prep" (the
    #: deck and its wrapper), "run" (everything the launch produces) or
    #: "summarize" (what `jobset summarize task` writes from results that
    #: exist, never the launch).  A card that lists a run's files has to say
    #: which of them exist yet, and the alternative was for the page to
    #: subtract one list from another.
    when: str = "run"
    #: Which CALCULATION writes it, or None for any.  A relaxation has no
    #: spectrum, and a list that promised one would be describing a file the
    #: run will never write -- the same fault in the other direction as a
    #: file that is written and undeclared.
    calculation: Optional[str] = None
    #: Does this file carry evidence of HOW THE RUN WENT, and of which kind?
    #: (`model/parse.md` § 5.5.)
    #:
    #:   ``"stdout"``   -- exists because the PROCESS started, so it speaks
    #:                     whether or not it has ended.  Block-buffered: its
    #:                     mtime is NOT liveness.
    #:   ``"progress"`` -- SEEDED at prep, so it speaks only once its footer
    #:                     concludes; otherwise a seed outranks a real result.
    #:   ``None``       -- not run evidence.  It may still be VIEWABLE, which
    #:                     is the REGISTRY's question (`detect()`), not this
    #:                     column's: `.spectra.json` is what a person should
    #:                     open and is not output, `.pyscf.log` is output and
    #:                     no parser claims it.
    #:
    #: **Why a column and not a boolean.** ``output == "stdout"`` *is* § 5.1's
    #: rule -- *an engine's stdout speaks before it ends; a seeded log does
    #: not*.  A boolean would force :func:`run_output_roles` to append
    #: `.molwatch.log` as a literal, putting membership back in a function
    #: while the row says nothing.
    output: Optional[str] = None
    # ---- the rest of the row (B13, `job-contracts.md` § 2.2, 2026-10-04):
    # the catalogue is the ONE source of what each file is, and the
    # contract's manifest (`project-layout.md` § 5, rendered by
    # `tools/manifest.py`), the Task setup card and the Results tab's file
    # card are its readings.
    #: The whole name of a file NOT named on the label -- a fixed name, or a
    #: family whose fields are declared in :data:`FIELDS` -- and ``""`` for a
    #: role row.
    name: str = ""
    #: Where it sits: ``calculation`` (the root), ``stage`` (a stage's
    #: folder in the hierarchy), ``run`` (an attempt; in the flat shape every
    #: level is the root), ``bench`` (a benchmark's container), ``launch``
    #: (a launch group's folder), or ``transient`` (only while a write runs).
    level: str = "run"
    #: Its name in the HIERARCHY where that differs from the label's
    #: spelling: an attempt's ``run.json`` -- one file, two spellings, one
    #: row.
    hierarchical: str = ""
    #: Who writes it, and when -- one line.
    writer: str = ""
    #: The one function its readers ask (`architecture.md` § 3.2), as a
    #: dotted path under ``molbuilder``; ``""`` when a person reads it and no
    #: code does.
    door: str = ""
    #: ``source`` · ``input`` · ``derived`` · ``record`` · ``result`` ·
    #: ``transient`` -- what losing the file costs (`project-layout.md` § 5).
    kind: str = "record"
    #: When it is written only sometimes, the condition, in words: *a
    #: machine with a queue*, *MOLBUILDER_PARSE_LOG set*.  A card that listed
    #: such a file unconditionally named a file the run may never write.
    only: str = ""


#: THE FIXED NAMES -- files molbuilder writes that are not named on the label,
#: spelled ONCE, here; each owner imports its own (`job-contracts.md` § 2.2,
#: B13, 2026-10-04).  Plain data, so this module stays stdlib and travels.
TASK_FILE = "task.json"
TASK_HANDOVER_FILE = "task.1st.json"
WARM_FILES_FILE = "warm-files.toml"
MACHINE_RECORD_FILE = "environment.json"
JOBSET_FILE = "job-set.json"
PLAN_FILE = "STAGE-PLAN.md"
LEDGER_FILE = "jobset-decisions.log"
PERMUTATION_FILE = "atom-permutation.json"
PSEUDO_DIR = "pseudos"
JUNCTION_FILE = "junction.xyz"
JUNCTION_SIDECAR_FILE = "junction.molstruct.json"
JUNCTION_CITED_FILE = "junction.cited.fdf"
SLOT_PROVENANCE_FILE = "slot-provenance.json"
CALCDIR_FILE = "calcdir.json"
LAUNCH_RECORD_FILE = "run.json"
CONTINUED_FROM_FILE = ".continued-from"
GATHERED_FROM_FILE = ".gathered-from"
MONITOR_BUNDLE = "mb_monitor.pyz"
VIBRATION_BUNDLE = "mb_vibration.pyz"
PYSCF_BUNDLE = "mb_pyscf.pyz"
MAKOV_PAYNE_SCRIPT = "makov_payne_correction.py"
BENCH_RESULT_FILE = "bench-result.json"
LAUNCH_DIR = "launch"


#: Ordered as `job-contracts.md` § 2.2 lists them: inputs, the wrapper, the
#: canonical trajectory, then the run-indexed logs -- which are HISTORY and not
#: state, and nothing here may treat them as leftovers.
WRITTEN: "tuple[Artifact, ...]" = (
    # ---- what we generated to run -------------------------------------
    Artifact(".fdf", "the SIESTA deck — every keyword this rung runs with",
             engine="siesta",
             when="prep", level="stage", kind="derived",
             writer="prep (`script_emit.prepare_deck`), in the stage's "
                    "folder; a copy in each attempt",
             door="runfiles.find_by_role"),
    Artifact(".py", "the PySCF script this rung runs", engine="pyscf",
             when="prep", level="stage", kind="derived",
             writer="prep (`script_emit.prepare_deck`), in the stage's "
                    "folder; a copy in each attempt",
             door="runfiles.find_by_role"),
    # THE SUFFIX IS SPELLED, NOT IMPORTED, and that is the LAYERING rule
    # rather than a lapse: this module is L1 and `template` is L2, so
    # importing it here is the upward import review refuses
    # (`process/code-audit.md` § 1c (e)).  The cost is real --
    # the glob view answers *"did the engine leave this, or did we write
    # it"* by subtraction, so a suffix that silently stops matching hands a
    # person their own input back as engine state.
    Artifact(".template.toml", "every parameter, with the value it was given",
             staged=False,
             when="setup", level="calculation", kind="source",
             writer="`jobset init`; the hand-over and the Transport tab, "
                    "which the browser writes; `jobset migrate` -- each "
                    "naming it with `template.template_path`",
             door="template.find_template"),
    # THE STRUCTURE THE CALCULATION IS OF, written into the bundle by the
    # hand-over (`web/handover-procedure.md`).
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
             when="setup", level="calculation", kind="input",
             writer="the hand-over and `jobset init` "
                    "(`StructureCodec.source_files`)",
             door="jobset.prep._structure_for"),
    Artifact(".source.molstruct.json", "its cell and its region labels",
             staged=False,
             when="setup", level="calculation", kind="input",
             writer="the hand-over and `jobset init` "
                    "(`StructureCodec.source_files`)",
             door="jobset.prep._structure_for"),
    Artifact(".template.toml.pre-m6", "the template as it was before "
                                      "`jobset migrate` rewrote it",
             staged=False,
             when="setup", level="calculation", kind="record",
             writer="`jobset migrate`",
             only="a template migrated"),
    # THE DECK'S COMPANION REPORT (`script_emit.VALIDATION_SUFFIX`): a file
    # molbuilder writes and does not declare reads as ENGINE OUTPUT.
    Artifact(".validation.txt", "what the generator checked before it wrote "
                                "the deck",
             when="prep", level="stage", kind="record",
             writer="prep (`script_emit.write_validation_report`), beside "
                    "the deck"),
    # ---- what the DECK writes for itself ------------------------------
    # A file the generated script writes is molbuilder's too -- it is our
    # deck that writes it.
    Artifact("_initial.xyz", "the input geometry, echoed back before "
                             "anything ran", staged=False, engine="pyscf",
             kind="result", writer="the PySCF deck"),
    # ITS SIDECAR, the pair's other half -- written by the deck through the
    # codec (`StructureCodec.write_moved`, imported from `mb_pyscf.pyz`).
    Artifact("_initial.molstruct.json", "the input geometry's cell and "
                                        "labels",
             staged=False, engine="pyscf", kind="result",
             writer="the PySCF deck"),
    Artifact("_optimized.molstruct.json", "the relaxed geometry's cell and "
                                          "labels -- the sidecar of the "
                                          "restart file `_optimized.xyz`",
             staged=False, engine="pyscf", kind="result",
             writer="the PySCF deck"),
    Artifact(".constraints.txt", "which atoms are held still, in geomeTRIC's "
                                 "own format",
             staged=False, engine="pyscf", kind="derived",
             writer="the PySCF deck", only="atoms are held"),
    # EITHER ENGINE'S RUN WRITES IT: the PySCF deck, and on SIESTA the job's
    # finish after the force-constant run (`engines/vibration.md` § 5.5).
    Artifact(".spectra.json", "the spectrum this run computed: frequencies, "
                              "the strengths the engine computes, "
                              "thermochemistry -- written by the run itself",
             staged=False, calculation="vibration", kind="result",
             writer="the run itself: the PySCF deck, or a SIESTA "
                    "force-constant job's finish (`mb_vibration.pyz`)",
             door="parse.sidecars.spectra.SpectraSidecarFileParser"),
    # ---- the wrapper --------------------------------------------------
    Artifact(".run.sh", "the wrapper — activates the environment, tees "
                        "the output, catches a kill",
             when="prep", level="stage", kind="derived",
             writer="prep (`runwrap.write_run_wrapper`), beside its deck; "
                    "a copy in each attempt"),
    # ON A MACHINE WITH A QUEUE, whether the run is then submitted or not:
    # the header is written whenever the target's record names a scheduler
    # (`runwrap`).
    Artifact(".sbatch", "the queue header — `sbatch` reads it",
             when="prep", level="stage", kind="derived",
             writer="prep (`runwrap.write_run_wrapper`), beside the run "
                    "script; a copy in each attempt",
             only="a machine with a queue"),
    # ---- the canonical trajectory -------------------------------------
    # Written before the engine even starts.
    # SEEDED AT PREP, which is why it is `progress` and not `stdout`: it
    # exists before the engine does, so an empty one is a prep and says
    # nothing about a run.  It speaks only once its footer concludes.
    # PREP SEEDS IT, so its moment is prep's.
    # EACH FLAT RUN'S OWN (plan W57 decision 2), so a flat stage launched
    # again does not truncate an earlier run's PySCF trajectory.
    Artifact(".molwatch.log", "the run as it happens — coordinates, "
                              "energy and forces, one block per step",
             attempt="shared", output="progress", when="prep", kind="record",
             writer="prep seeds the first run's, in the attempt -- in "
                    "the flat shape numbered for run 0; PySCF's deck "
                    "writes each step into its run's own, SIESTA never "
                    "does",
             door="parse.engines.molwatch.MolwatchLogFileParser"),
    # ---- history: stdout and the logs ---------------------------------
    # ALL OF THIS IS HISTORY AND NOT STATE -- it is what a person goes back
    # to read, and nothing may treat it as leftovers.
    #
    # `attempt="maybe"` is two real spellings, not indecision: the wrapper
    # indexes its redirect (`-run0.out`) and a hand-started run does not.
    #
    # `.out` IS SIESTA'S (`model/parse.md` § 5.5).
    Artifact(".out", "the run's output as the engine printed it",
             attempt="maybe", engine="siesta", output="stdout",
             kind="result", writer="the run script, from the engine's "
                                   "stdout",
             door="parse.engines._run_ending.ending_of"),
    # `attempt="maybe"` for the SAME reason `.out` above has it.
    Artifact(".pyscf.log", "the same, for PySCF — it writes here and not "
                           "to .out", attempt="maybe",
             engine="pyscf", output="stdout",
             kind="result", writer="the run script, from the engine's "
                                   "stdout",
             door="parse.engines._run_ending.ending_of"),
    # PYSCF'S OWN LOG (`mol.output`, `model/parse.md` § 5.5).
    # PySCF OPENS IT WITH 'w', so a flat run's own (plan W57 decision 2): a
    # flat stage launched again would truncate the last run's.
    Artifact(".log", "PySCF's own log", attempt="shared", engine="pyscf",
             kind="result", writer="the PySCF deck",
             door="parse.dirs.record.run_record"),
    # GEOMETRIC'S OPT LOG, `<label>_<stage>_geom.log`.
    # A FLAT RUN'S OWN with the prefix that names it, which carries the run's
    # number there: otherwise geomeTRIC moves the last run's aside as
    # `<prefix>_1.log`, a name nothing declares.
    Artifact("_geom.log", "geomeTRIC's optimizer log", attempt="shared",
             engine="pyscf", kind="result",
             writer="geomeTRIC, under the prefix the deck hands it"),
    # A TEMPLATE, NOT A GLOB.  The stamp is a FIELD: the file it names is real and
    # concrete, and what varies is a value the name carries (§ 5l.1).
    Artifact(".runwrap-{stamp}.log", "the wrapper's own session log — one per "
                                     "launch, stamped with the clock",
             fields=("stamp",), kind="record",
             writer="the run script, at each start",
             door="wrapper_log.log_of_run"),
    # The monitor's two files carry the wrapper's run index, always: the run
    # script starts the monitor with `--run "$_run_n"`.
    #
    # EVERY ENGINE'S, the monitor's two: the wrapper starts it from its shared
    # part (`run-reports.md` § 2.3).  The timing tee is
    # SIESTA's alone -- it reads the SIESTA family's rows -- so its row names
    # the engine: a PySCF rung's card must not promise a file no PySCF run
    # writes.  `patterns()` ignores
    # the column, so the cold sweep still knows them in any directory.
    Artifact(".monitor.log", "the monitor's rolling status", attempt="always",
             kind="record", writer="the monitor",
             door="parse.instruments.monitor.MonitorLogFileParser"),
    Artifact(".util.csv", "processor and memory samples taken while it ran",
             attempt="always", kind="record", writer="the monitor",
             door="parse.instruments.util_csv.UtilCsvFileParser"),
    Artifact(".scf-timing.log", "wall time per SCF iteration — on a "
                                "TranSIESTA device, both its phases",
             attempt="always",
             engine="siesta", kind="record",
             writer="the run script's tee of the output's SCF lines",
             door="parse.instruments.scf_timing_rows.timing_of",
             # THE TEE OPENS IT AT THE FIRST SCF ROW (`runwrap._mb_scf_tee`):
             # a run that stops before one has none.
             only="the engine printed an SCF iteration"),
    # MOLBUILDER'S OWN READING LOG.
    # NAMED OFF THE FILE READ, so the output's `-run<N>` rides it:
    # `<base>-run0.out` gives `<base>-run0.parse.log` (`parse/_log.py`).
    # Off unless asked.
    Artifact(".parse.log", "molbuilder's log of reading the run's output",
             attempt="always", kind="record",
             writer="molbuilder's parser, reading the output "
                    "(`parse._log.ParseLogger`)",
             only="`MOLBUILDER_PARSE_LOG` set"),
    Artifact(".molwatch.parse.log",
             "the same, for the trajectory log", attempt="shared",
             kind="record",
             writer="molbuilder's parser, reading the trajectory log "
                    "(`parse._log.ParseLogger`)",
             only="`MOLBUILDER_PARSE_LOG` set"),
    # The summary `jobset summarize task` writes.  It is the deliverable of the ladder
    # (`engines/transport.md` § 2a.12: "the transmission stage's output ...
    # everything else in the tree exists to make it trustworthy"), written
    # once at the calculation root -- which is what `staged=False` says.
    Artifact(".transport.json", "the transport results, summarised by "
                                "`summarize task` from the transmission "
                                "points that ran",
             staged=False, calculation="transport", when="summarize",
             level="calculation", kind="result",
             writer="`jobset summarize task` (`transport.record.write_record`)",
             door="parse.sidecars.transport.TransportRecordFileParser"),
    # A SIESTA VIBRATION'S DISPLACEMENT SWEEP, summarised (`engines/
    # vibration.md` § 5.9): written at the calculation root by `summarize task`
    # when the ladder holds two or more force-constant stages -- a record of
    # results that exist, each stage's own files staying in its attempt.
    Artifact(".fc-sweep.json", "the force-constant stages compared, when "
                               "the ladder has two or more (a displacement "
                               "sweep), by `summarize task`: each stage and "
                               "what it varied, every mode's frequency per "
                               "stage, the force-constant changes, and "
                               "where each stage's files are",
             staged=False, engine="siesta", calculation="vibration",
             when="summarize", level="calculation", kind="result",
             writer="`jobset summarize task` "
                    "(`spectra.displacement_sweep.write_sweep`)",
             door="parse.sidecars.fc_sweep.FcSweepRecordFileParser",
             only="two or more force-constant stages"),
    # THE CONCLUSION MARKER -- the wrapper's last act on its main path
    # (`project-layout.md` § 1.6, "the other file", 2026-08-28).  Indexed
    # like the stdout, because a warm-retry chain execs fresh wrappers and
    # only the FINAL process concludes.
    Artifact(".concluded", "the marker the wrapper writes when the job ends",
             attempt="always", kind="record",
             writer="the run script, its last act",
             door="runrecord.ending"),
    # A FLAT STAGE'S LAUNCH RECORD (`project-layout.md` § 1.6.3) -- what an
    # attempt's `run.json` is for a stage with no attempt of its own: every
    # stage of a flat calculation shares one directory, so each names its
    # record as it names every other file of it.  Written at launch.
    # AND AN ATTEMPT'S, `run.json`: one file, two spellings, one row
    # (`hierarchical`).
    # ONE PER FLAT RUN, its number in the name (plan W57 decisions 2 and 6),
    # so a warm retry -- a second run of one launch -- has its own.
    Artifact(".run.json", "the launch record: how, where and when the run "
                          "was sent, what it continued from, and the run a "
                          "warm retry retries",
             attempt="shared", hierarchical=LAUNCH_RECORD_FILE, when="launch",
             kind="record",
             writer="launch (`runrecord.write_launch`); a flat run's warm "
                    "retry, the run script through the monitor's bundle "
                    "(`runrecord.record_retry`)",
             door="runrecord.launch_record"),
    # AND THE RUN IT CONTINUES FROM, left by `prep` for `launch` to write into
    # that record -- an attempt's `.continued-from`, for a stage with none of
    # its own (user, 2026-10-01: the flat layout records it too), named for
    # the run that continues.
    Artifact(".continued-from", "the run this one continues from -- a "
                                "walked point's or a stage's start, a "
                                "taken-over point's computing run (one hop) "
                                "-- for launch's record",
             attempt="shared", hierarchical=CONTINUED_FROM_FILE, when="prep",
             kind="record",
             writer="prep, and launch when it runs a stage again, through "
                    "`runrecord.write_continued_from`",
             door="runrecord.read_continued_from",
             only="it continues from an earlier run"),
    # ---- written on the label, beside any file of ours ---------------------
    Artifact(".runtime_info.json", "what a file of the run says about the "
                                   "run, as `molbuilder runtime-info` read it",
             attempt="maybe", kind="record",
             writer="`molbuilder runtime-info`, beside the file it reads",
             only="a person runs `molbuilder runtime-info`"),
    Artifact(".{engine}.{shape}.pipeline.log",
             "what each step of a prep received, decided and produced",
             fields=("engine", "shape"), when="prep", level="calculation",
             kind="record",
             writer="every prep, from either door, beside its "
                    "`STAGE-PLAN.md` (`pipeline_log.PipelineLog`)"),
    # ---- NOT named on the label: fixed names, and families of them --------
    # A FIXED NAME'S OWNER TAKES IT FROM HERE (`job-contracts.md` § 2.2): the
    # constants above are the one spelling.
    Artifact(name=TASK_FILE, what="the description: what varies, the stages, "
                                  "the shape, the structure",
             when="setup", level="calculation", kind="source",
             writer="`jobset init`; Task setup's Save; the Transport tab's "
                    "describe, which the browser writes",
             door="task.read_task"),
    Artifact(name=TASK_HANDOVER_FILE,
             what="a hand-over waiting for its shape and stages; Save "
                  "removes it",
             when="setup", level="calculation", kind="record",
             writer="the hand-over route, which the browser writes",
             door="web.blueprints.build.api_task_setup_folder"),
    Artifact(name=WARM_FILES_FILE,
             what="this calculation's own restart-file list",
             when="setup", level="calculation", kind="source",
             writer="a person, from molbuilder's list for the engine",
             door="warmfiles.warm_list", only="a person writes one"),
    Artifact(name=MACHINE_RECORD_FILE,
             what="the machine record this calculation is set to — its "
                  "first prep's",
             when="prep", level="calculation", kind="record",
             writer="the first prep (`jobset.machine.set_machine`)",
             door="scheduler.record.machine_for"),
    Artifact(name=JOBSET_FILE,
             what="the plan: one job per prepared stage, merged per stage — "
                  "and a benchmark's own, in its container",
             when="prep", level="calculation", kind="derived",
             writer="prep (`jobset.model.JobSet.write`)",
             door="jobset.model.JobSet.load"),
    Artifact(name=PLAN_FILE, what="the plan in reading order",
             when="prep", level="calculation", kind="derived",
             writer="prep (`jobset.prep.prep_jobset`), whole at each prep"),
    Artifact(name=LEDGER_FILE, what="one line per decision of every verb",
             when="prep", level="calculation", kind="record",
             writer="every verb (`jobset.ledger.record`)"),
    Artifact(name=PERMUTATION_FILE,
             what="the atom order the decks were written in",
             when="prep", level="calculation", kind="derived",
             writer="prep (`transport.sort.write_permutation`)",
             door="atom_permutation.read_permutation",
             only="a SIESTA vibration, or a transport calculation"),
    Artifact(name="{element}.psml",
             what="a pseudopotential: the calculation's one copy in "
                  "`pseudos/`, and a real copy beside every deck — SIESTA "
                  "opens only its working directory",
             engine="siesta", when="prep", level="calculation", kind="input",
             writer="prep (`jobset.engines._pseudo_dir`, `materialize`); "
                    "`jobset init --psml-lib`",
             door="pseudos.psml_sources"),
    Artifact(name=JUNCTION_FILE, what="the composed junction",
             calculation="transport", when="prep", level="calculation",
             kind="derived",
             writer="a transport calculation's first prep "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.load_compose_record"),
    Artifact(name=JUNCTION_SIDECAR_FILE,
             what="its cell and its region labels",
             calculation="transport", when="prep", level="calculation",
             kind="derived",
             writer="a transport calculation's first prep "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.load_compose_record"),
    Artifact(name=JUNCTION_CITED_FILE,
             what="the deck the junction was cited from, as it was",
             calculation="transport", when="prep", level="calculation",
             kind="derived",
             writer="a transport calculation's first prep "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.load_compose_record"),
    Artifact(name=SLOT_PROVENANCE_FILE,
             what="where each part of the junction came from, with hashes",
             calculation="transport", when="prep", level="calculation",
             kind="record",
             writer="a transport calculation's first prep "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.load_compose_record"),
    Artifact(name=".gitignore",
             what="which files the saved states keep by content in "
                  "`.binsnapshots/` rather than in git",
             when="setup", level="calculation", kind="record",
             writer="`checkpoint.save_before`, before every Save and prep",
             door="checkpoint.Repo"),
    Artifact(name=".git/", what="the folder's saved states",
             when="setup", level="calculation", kind="record",
             writer="`checkpoint.save_before`, before every Save and prep",
             door="checkpoint.Repo"),
    Artifact(name=".binsnapshots/",
             what="the big files of each saved state, by content",
             when="setup", level="calculation", kind="record",
             writer="`checkpoint.save_before`, before every Save and prep",
             door="checkpoint.Repo"),
    Artifact(name=CALCDIR_FILE,
             what="what this folder is in its calculation — a container or "
                  "a run — and where the calculation is",
             when="prep", level="stage", kind="record",
             writer="the code that makes the folder: prep and launch "
                    "(`materialize.open_run` -- a stage's attempt, a bias "
                    "point's, a benchmark trial's, and the containers above "
                    "each -- and `materialize.open_container`, a "
                    "submission's `launch/` and `pseudos/`)",
             door="calcdirs.read"),
    Artifact(name=MONITOR_BUNDLE,
             what="the monitor, and the readers it runs on — one file",
             when="prep", level="stage", kind="derived",
             writer="prep, beside each run script "
                    "(`runwrap.write_run_wrapper`); a copy in each attempt"),
    Artifact(name=VIBRATION_BUNDLE,
             what="a force-constant job's finish: the modes, from `.FC`",
             engine="siesta", calculation="vibration",
             when="prep", level="stage", kind="derived",
             writer="prep, beside the run script "
                    "(`runwrap.write_run_wrapper`); a copy in each attempt",
             only="a force-constant stage"),
    Artifact(name=PYSCF_BUNDLE,
             what="molbuilder's own code the PySCF script runs, which it "
                  "imports from here -- the core count, the progress-log "
                  "writer, the structure codec, the relaxation, a "
                  "vibration's rules",
             engine="pyscf",
             when="prep", level="stage", kind="derived",
             writer="prep, beside the PySCF script "
                    "(`runwrap.write_run_wrapper`); a copy in each attempt"),
    Artifact(name=MAKOV_PAYNE_SCRIPT,
             what="the energy correction a charged, isolated deck asks a "
                  "person to run afterwards",
             engine="siesta", when="prep", level="stage", kind="derived",
             writer="prep (`siesta.makov_payne.emit_correction_script`); a "
                    "copy in each attempt",
             only="a charged, isolated deck"),
    Artifact(name=GATHERED_FROM_FILE,
             what="what a rung took, from which upstream run",
             calculation="transport", when="prep", level="run",
             kind="record",
             writer="prep (`jobset.prep.transport_inputs`, through "
                    "`runrecord.write_gathered_from`)",
             door="runrecord.read_gathered_from"),
    Artifact(name="slurm.{jobid}.out", what="SLURM's own stdout for the job",
             level="run", kind="record",
             writer="SLURM, as the run's header asks (`-o`)",
             only="launched to a queue"),
    Artifact(name="slurm.{jobid}.err", what="SLURM's own stderr for the job",
             level="run", kind="record",
             writer="SLURM, as the run's header asks (`-e`)",
             only="launched to a queue"),
    Artifact(name=".mb-rank-launch-{pid}.sh",
             what="the per-rank GPU launcher",
             level="run", kind="transient",
             writer="the run script, removed when it exits",
             only="a GPU run"),
    Artifact(name=BENCH_RESULT_FILE,
             what="every trial's timing and the winner — the benchmark's "
                  "archival trace",
             when="summarize", level="bench", kind="record",
             writer="`jobset summarize` (`run_summarize_jobset`)"),
    Artifact(name="{group}.run.sh",
             what="a launch group's sequencer: a benchmark's trials, a "
                  "bias sweep's points, or a task's stages sharing one job, "
                  "in order",
             when="launch", level="launch", kind="derived",
             writer="launch (`jobset/submit.py`), written again at each "
                    "launch"),
    Artifact(name="{group}.sbatch", what="its queue header",
             when="launch", level="launch", kind="derived",
             writer="launch (`jobset/submit.py`), written again at each "
                    "launch -- a task's group's by its prep "
                    "(`prep.prep_group`), and by launch only for stages "
                    "first named together there",
             only="launched to a queue"),
    Artifact(name="{group}.log", what="every member's output, in order",
             when="run", level="launch", kind="record",
             writer="the group's sequencer, as it runs"),
    Artifact(name="{random}.tmp",
             what="a file being written; it replaces its target when the "
                  "write ends",
             when="run", level="transient", kind="transient",
             writer="`persist` and the codec, writing whole or not at all"),
    Artifact(name="{random}.lock", what="a sidecar's lock",
             when="run", level="transient", kind="transient",
             writer="`sidecars.molstruct.with_lock`"),
)

#: The rows NAMED ON THE LABEL -- the role rows every reader of the run-file
#: grammar walks; the fixed names are the rest of :data:`WRITTEN`.
ON_THE_LABEL: "tuple[Artifact, ...]" = tuple(a for a in WRITTEN if a.role)

#: The attempt counter as a GLOB, per :attr:`Artifact.attempt`.  Empty string
#: means the unindexed spelling; ``-run*`` the wrapper's.
_ATTEMPT_GLOBS = {"never": ("",), "maybe": ("", "-run*"), "always": ("-run*",),
                  "shared": ("", "-run*")}

#: THE FIRST RUN IS ZERO -- the number `launch` gives a stage's first run
#: (`runrecord.next_run`) -- so this is the real name the first launch
#: writes, not a placeholder.
FIRST_ATTEMPT = 0

#: The two shapes, by name (`paths.SHAPES`, restated: this module is L1 and
#: travels beside jobs, so it imports nothing of molbuilder's).
_SHAPES = ("flat", "hierarchical")

#: THE RUN'S NUMBER IN A NAME TEMPLATE (:meth:`RunNames.template`): the
#: placeholder a rendered script fills when it runs -- the number `launch`
#: gives it (`--run N`).  A role's fields stay ``{field}`` the same way.
RUN_FIELD = "{run}"


def runs_share_folder(shape: str, trial: bool = False) -> bool:
    """WHETHER A STAGE'S RUNS SHARE ONE FOLDER -- the one rule every name of
    a run's own files follows (`project-layout.md` § 1.5a, § 1.6.1): a flat
    calculation's stage runs again in the folder it ran in, so each run's
    records and logs carry its number; a hierarchical stage opens a folder
    per launch, and a benchmark trial has its own folder in either shape,
    launched once -- the folder is the run."""
    if shape not in _SHAPES:
        raise RunFileError(f"shape {shape!r} is not one of "
                           f"{' / '.join(_SHAPES)}.")
    return shape == "flat" and not trial


def _row(role: str) -> "Artifact":
    """The catalogue's row for ``role`` -- or refused: a name nothing
    declares is a name nothing may compose."""
    for a in ON_THE_LABEL:
        if a.role == role:
            return a
    raise RunFileError(f"{role!r} is no role of the catalogue "
                       f"(`runfiles.WRITTEN`).")


def _numbered(a: "Artifact", shared: bool) -> bool:
    """Whether a file of row ``a`` carries the run's number, its runs
    sharing a folder or not (:attr:`Artifact.attempt`)."""
    return {"never": False, "maybe": True, "always": True,
            "shared": shared}[a.attempt]


def _template(a: "Artifact", label: str, stage: Optional[str],
              shared: bool) -> str:
    """Row ``a``'s name on ``label`` with :data:`RUN_FIELD` where the run's
    number goes and each of its fields as ``{field}``; the hierarchy's fixed
    name where the row has one and the run has a folder of its own (an
    attempt's ``run.json``).  ``stage`` names a staged row's stage; an
    unstaged row is the calculation's and carries none.  Read by
    :meth:`RunNames.template` and :func:`manifest`."""
    if a.hierarchical and not shared:
        return a.hierarchical
    return compose(label, a.role, stage if a.staged else None,
                   run=RUN_FIELD if _numbered(a, shared) else None,
                   **{f: "{" + f + "}" for f in _fields_in(a.role)})


def shared_numbered_roles() -> "tuple[str, ...]":
    """The roles whose names carry the run's number only where a stage's
    runs share a folder (``attempt="shared"``) -- read off the rows, so
    whatever asks which names an older shared folder held unnumbered
    (`runrecord`'s refusal, `jobset migrate`) asks the catalogue, never a
    list of its own."""
    return tuple(a.role for a in ON_THE_LABEL if a.attempt == "shared")


@dataclass(frozen=True)
class RunNames:
    """THE NAMES OF ONE STAGE'S FILES -- the catalogue (:data:`WRITTEN`) read
    for one stage of one calculation (`job-contracts.md` § 2.2a).  Every
    writer of a stage's files, and every script rendered to write them,
    names them through this: prep and launch, the run records, the run
    script and the deck, the readers -- so no two can spell one name two
    ways.  The row decides every part of a name: its stage token
    (``staged``), the run's number (``attempt``), the hierarchy's fixed name
    (``hierarchical``); this object only supplies the stage it names.

    ``shared`` is whether the stage's runs share one folder
    (:func:`runs_share_folder`, :meth:`of`): then a run's own records and
    logs carry its number; else its folder names the run.

    The stage is required: every calculation has at least one, and every
    file of a run carries its token (`stages.md` § 6.5)."""
    label: str
    stage: str
    shared: bool

    def __post_init__(self):
        if self.stage is None:
            raise RunFileError(
                f"{self.label!r}: a run's files are named by its stage, and "
                f"no stage was given -- every calculation has at least one "
                f"(`stages.md` § 6.5).")
        stem(self.label, self.stage)    # a malformed label or token, refused

    @classmethod
    def of(cls, label: str, stage: str, shape: str,
           trial: bool = False) -> "RunNames":
        """The names of ``stage``'s files in a calculation of ``shape`` --
        a benchmark trial's (``trial``) in its own folder."""
        return cls(label, stage, runs_share_folder(shape, trial))

    @property
    def stem(self) -> str:
        """``<label>_<stage>`` -- what every staged role attaches to."""
        return stem(self.label, self.stage)

    def numbered(self, role: str) -> bool:
        """Whether a file in ``role`` carries the run's number here."""
        return _numbered(_row(role), self.shared)

    def template(self, role: str) -> str:
        """``role``'s name with :data:`RUN_FIELD` where the run's number
        goes and each of its fields as ``{field}`` -- what a rendered script
        fills in when it runs; the hierarchy's fixed name where the row has
        one and the run has a folder of its own (an attempt's ``run.json``).
        """
        return _template(_row(role), self.label, self.stage, self.shared)

    def tail(self, role: str) -> str:
        """:meth:`template` after the label -- for a script that names its
        files from its own ``JOB`` / ``SystemLabel`` (:func:`tail`)."""
        name = self.template(role)
        if not name.startswith(self.label):
            raise RunFileError(
                f"{role!r} is named {name!r} here, not on the label: a "
                f"script cannot spell it from its own label.")
        return name[len(self.label):]

    def name(self, role: str, run: Optional[int] = None, **fields) -> str:
        """``role``'s name for run ``run``, the run the file belongs to --
        the row says whether the name carries its number here (then it is
        required), or is the folder's for every run in it."""
        a = _row(role)
        if a.hierarchical and not self.shared:
            return a.hierarchical
        numbered = _numbered(a, self.shared)
        if numbered and run is None:
            raise RunFileError(
                f"{role!r} is named {self.template(role)!r} here, and no "
                f"run was named to fill it.")
        return compose(self.label, role, self.stage if a.staged else None,
                       run=run if numbered else None, **fields)

    def run_name(self, run: int) -> str:
        """``<label>_<stage>-run<N>`` -- the name every file of run ``run``
        carries (:func:`run_name`)."""
        return run_name(self.label, self.stage, run)


@dataclass(frozen=True)
class GroupNames:
    """A LAUNCH GROUP'S FILES -- a benchmark's walk or a bias sweep's chain:
    the catalogue's launch-level rows ``{group}.run.sh``, ``{group}.sbatch``
    and ``{group}.log``, filled with the group's name (`project-layout.md`
    § 5).  The same two questions :class:`RunNames` answers -- :attr:`stem`
    and :meth:`name` -- so a header is rendered alike for a stage's run and
    a group's (`runwrap.render_sbatch`)."""
    group: str

    def __post_init__(self):
        stem(self.group)                # one [A-Za-z0-9_-]+ token, refused

    @property
    def stem(self) -> str:
        return self.group

    def name(self, tail: str) -> str:
        """The group's file ending ``tail``, from its catalogue row."""
        template = "{group}" + tail
        if not any(a.level == "launch" and a.name == template
                   for a in WRITTEN):
            raise RunFileError(
                f"no launch-level row {template!r} in runfiles.WRITTEN -- a "
                f"group's files are the catalogue's")
        return template.format(group=self.group)


def roles(*, engines: bool = True) -> "tuple[str, ...]":
    """Every role this module knows -- the catalogue, read as a vocabulary.

    :data:`WRITTEN`'s roles, plus the engines' own (:data:`_ENGINE_ROLES`)
    unless ``engines=False``.  The address layer (`ref`) validates against this, which
    is what makes an undeclared role a catalogue question rather than something
    a caller can slip past by spelling it (`plans/plan.md` § 5l.2: extensibility
    is a catalogue row, never a new function).

    Ordered and de-duplicated, so it is stable to compare against.
    """
    out = [a.role for a in ON_THE_LABEL]
    if engines:
        out += [r for r in _ENGINE_ROLES if r not in out]
    seen, uniq = set(), []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return tuple(uniq)


def run_output_roles(engine: Optional[str] = None) -> "tuple[str, ...]":
    """Every role that carries evidence of HOW A RUN WENT (§ 5.5).

    The answer to *what is this run's output* -- the catalogue's question.  It
    is NOT *what can a person open*, which is the registry's (`detect()`) and
    gets no column here: `.pyscf.log` is output and no parser claims it,
    `.spectra.json` is what a person should see and is not output.

    ``engine`` narrows to what THAT engine's runs produce, keeping the
    engine-agnostic rows.  Passing None answers for every engine at once,
    which is what a reader of a directory whose engine it does not know needs
    -- and § 5.5's reason that `ending_of` dispatches on the ROLE.
    """
    return tuple(a.role for a in ON_THE_LABEL
                 if a.output is not None
                 and not (engine and a.engine and a.engine != engine))


def stdout_roles(engine: Optional[str] = None) -> "tuple[str, ...]":
    """The subset of :func:`run_output_roles` that is a PROCESS's stdout.

    ``output == "stdout"`` is § 5.1's rule, spelled as a row: such a file
    exists because the process started, so it speaks whether or not the run
    has ended -- where a SEEDED progress log speaks only once its footer
    concludes, and an empty one is a prep.
    """
    return tuple(a.role for a in ON_THE_LABEL
                 if a.output == "stdout"
                 and not (engine and a.engine and a.engine != engine))


def result_roles(calculation: Optional[str] = None) -> "tuple[str, ...]":
    """What a person LOOKS AT for this calculation, most specific first.

    The third question `model/parse.md` § 5.5 splits out — *what should a
    viewer open here* — answered from a column rather than a ladder, which is
    what the other two already do.

    Three kinds of row answer it, and the order between them is the whole rule:

    * a role naming this CALCULATION is what the run is **for**.
      `.spectra.json` for a vibration: the deck rewrites it atomically at
      every phase boundary and it carries its own `phase_*` flags, so it is
      the live view AND the final result — there is no moment when it is the
      wrong file to open.
    * a ``"stdout"`` role is the engine's own account, and it speaks whether
      or not the run has ended (§ 5.5).  For SIESTA it IS the trajectory --
      the `.out` every geometry step is printed into -- and a parser claims
      it; PySCF's `.pyscf.log` no parser claims, so the door skips it.
    * a ``"progress"`` role is the engine-neutral live channel every run has.
      It is SEEDED at prep, so it speaks only once the run writes into it:
      PySCF's deck does, SIESTA's never does (measured 2026-09-24 on every
      SIESTA relaxation in the fixture project: a 613-byte seed holding one
      `initial_preview` block beside a 34 KB `.out`), and a spectrum run
      leaves it a stub either way (measured 2026-09-18 on a CO2 run).  So it
      comes AFTER the stdout roles -- offered first, the seed outranked the
      real result, which is the trap § 5.5 names.

    ``calculation`` None answers with the output roles alone.

    **This is not a preference order to tune.**  Whichever file the
    calculation produces is the one to open, during the run and after it.
    """
    named = [a.role for a in ON_THE_LABEL
             if calculation and a.calculation == calculation]
    spoken = [a.role for a in ON_THE_LABEL
              if a.output == "stdout" and a.role not in named]
    live = [a.role for a in ON_THE_LABEL
            if a.output == "progress" and a.role not in named]
    return tuple(named + spoken + live)


def deck_roles(engine: Optional[str] = None) -> "tuple[str, ...]":
    """The DECKS -- what a stage's engine reads: the stage's own file, derived
    at prep, that one engine owns (`.fdf`, `.py`) -- for ``engine``, or every
    engine's.  Read off the rows, so a new engine's deck is its row."""
    return tuple(a.role for a in ON_THE_LABEL
                 if a.level == "stage" and a.kind == "derived" and a.engine
                 and (engine is None or a.engine == engine))


def engine_of_role(role: str) -> Optional[str]:
    """Which engine writes this role, or None when any of them may.

    The mirror of :func:`stdout_roles`, and what a WRITER asks: the wrapper
    holds a deck suffix (`.py`, `.fdf`) and needs the role its run will write
    its stdout to.  Composing the two is the whole derivation
    (`model/parse.md` § 5.5, R-RO1: the vocabulary binds writers as well as
    readers).
    """
    for a in WRITTEN:
        if a.role == role:
            return a.engine
    return None


def engines() -> "tuple[str, ...]":
    """Every engine the catalogue names, sorted.

    The catalogue is where an engine becomes known to the run-file layer (a
    row with ``engine=``), so this is what a caller iterating engines asks
    rather than restating the pair -- § 5.5's *adding an engine is two edits*.
    """
    return tuple(sorted({a.engine for a in WRITTEN if a.engine}))


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
    for a in artifacts if artifacts is not None else ON_THE_LABEL:
        # A FIELD BECOMES A STAR HERE, and only here.  `.runwrap-{stamp}.log`
        # yields `.runwrap-*.log`.  A glob belongs in the glob view, not in
        # the vocabulary (§ 5l.3).
        role = _FIELD_RE.sub("*", a.role)
        stems = ["{label}"] + (["{label}_*"] if a.staged else [])
        for stem in stems:
            for run in _ATTEMPT_GLOBS[a.attempt]:
                name = f"{stem}{run}{role}"
                if name not in out:
                    out.append(name)
    return tuple(out)


def manifest(label: str, stage: Optional[str], *, shape: str,
             engine: Optional[str] = None,
             when: "Optional[tuple]" = None,
             calculation: Optional[str] = None) -> "list[dict]":
    """The names one rung writes, each with a line saying what the file is.

    ``stage`` given answers for THAT RUNG and lists only what carries its
    token.  ``stage`` None answers for the CALCULATION and lists only what
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

    ``shape`` names each file as that shape spells it: an attempt's
    ``run.json`` in the hierarchy where a flat stage's first run writes
    ``<base>-run0.run.json`` (:attr:`Artifact.hierarchical`);
    a rung's card cannot be drawn without it.  A
    file written only sometimes carries its condition, ``only``; each row
    says the folder it lands in, ``level`` -- a rung's pipeline log is the
    calculation's, at its root.

    Every name comes out of :func:`compose`, so a card cannot show a spelling
    the writers do not use -- which is the failure this module exists for.
    """
    shared = runs_share_folder(shape)
    rows = []
    for a in ON_THE_LABEL:
        if a.staged != (stage is not None):
            continue
        if engine and a.engine and a.engine != engine:
            continue
        if when and a.when not in when:
            continue
        if calculation and a.calculation and a.calculation != calculation:
            continue
        # EACH ROW'S NAME as the stage's names spell it (:func:`_template`,
        # what :class:`RunNames` reads), the first run's where the name
        # carries the run's number.  A FIELD HAS NO VALUE UNTIL THE FILE
        # IS WRITTEN, so the card shows the field's NAME:
        # `<label>_<stage>.runwrap-<stamp>.log`.
        run = FIRST_ATTEMPT if _numbered(a, shared) else None
        name = _FIELD_RE.sub(lambda m: f"<{m.group(1)}>",
                             _template(a, label, stage, shared).replace(
                                 RUN_FIELD, str(run)))
        rows.append({"name": name,
                     "what": a.what,
                     "when": a.when,
                     "level": a.level,
                     "only": a.only})
    return rows


def fixed(level: Optional[str] = None,
          when: "Optional[tuple]" = None,
          calculation: Optional[str] = None,
          engine: Optional[str] = None) -> "list[dict]":
    """The catalogue's FIXED-NAME files -- the ones not named on the label --
    at ``level``, with a line each: what the Task-setup card lists for the
    whole run (`job-set.json`, the plan, the machine record, the ledger),
    narrowed like :func:`manifest`.  A family shows its fields by name:
    ``<element>.psml``."""
    rows = []
    for a in WRITTEN:
        if a.role:
            continue
        if level and a.level != level:
            continue
        if when and a.when not in when:
            continue
        if calculation and a.calculation and a.calculation != calculation:
            continue
        if engine and a.engine and a.engine != engine:
            continue
        rows.append({"name": _FIELD_RE.sub(lambda m: f"<{m.group(1)}>",
                                           a.name),
                     "what": a.what, "when": a.when, "only": a.only})
    return rows


#: THE STRUCTURE PAIR A PERSON SAVES, in a storage topic -- `structure/` holds
#: files, not calculations (`project-layout.md` § 2.6) -- so no row of the
#: catalogue names it: its stem is the person's.  What the Results file card
#: says of each half, read by :func:`saved_pair_row` (`web/results.md` § 3b).
SAVED_PAIR_WRITER = ("Save on the Molbuilder tab, or Export on the Results "
                     "tab (`StructureCodec.write`)")
SAVED_PAIR_WHAT = {
    "structure": ("a structure -- its atoms and coordinates, every frame when "
                  "it has several; its cell, labels and metadata are in the "
                  "sidecar beside it"),
    "sidecar": ("the sidecar of the structure beside it -- its cell, its "
                "region labels and its recorded metadata"),
}


def saved_pair_row(path) -> Optional["tuple[str, str]"]:
    """``(what, writer)`` for one half of a saved structure pair -- a
    structure file with its sidecar beside it, or that sidecar with its
    structure -- or ``None``.  The pairing is the codec's
    (`sidecars.molstruct.sidecar_path_for`); a structure it opens is `.xyz`
    or `.pdb`."""
    from .sidecars.molstruct import SUFFIX, sidecar_path_for
    p = Path(path)
    if p.name.endswith(SUFFIX):
        stem = p.name[:-len(SUFFIX)]
        if any((p.parent / f"{stem}{ext}").is_file()
               for ext in (".xyz", ".pdb")):
            return SAVED_PAIR_WHAT["sidecar"], SAVED_PAIR_WRITER
        return None
    if p.suffix in (".xyz", ".pdb") and sidecar_path_for(p).is_file():
        return SAVED_PAIR_WHAT["structure"], SAVED_PAIR_WRITER
    return None


def row_for(path, label: Optional[str] = None) -> Optional[Artifact]:
    """WHICH ROW OF THE CATALOGUE IS THIS FILE -- or ``None``: not a file
    molbuilder writes.

    ``label`` is the label the file's run names its files on (the run door
    reads it from the description, `architecture.md` § 3.2): a name read back
    with it to a declared role is that role's row.  Otherwise a fixed name,
    or a family of them -- a launch group's files only inside a ``launch/``
    folder, where the group's name is the whole stem, so SIESTA's
    ``fdf.<stamp>.log`` beside a run is not taken for one.  Without a label
    no role row answers: a role cannot be told from the rest of a name that
    way (:func:`parse`).
    """
    p = Path(path)
    base = p.name
    if label:
        rec = parse(base, label)
        if rec is not None:
            for a in ON_THE_LABEL:
                if a.role == rec.role:
                    return a
    in_launch = p.parent.name == LAUNCH_DIR
    for a in WRITTEN:
        if a.role:
            if a.hierarchical and a.hierarchical == base:
                return a
            continue
        if (a.level == "launch") != in_launch and a.level != "transient":
            continue
        if not a.name.endswith("/") and _role_pattern(a.name).match(base):
            return a
    return None

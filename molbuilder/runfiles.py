"""Run-file names — composed and parsed in ONE place.

Contract: `docs/execution/job-contracts.md` § 2.2a (the name grammar), which
          § 2.1 rule 2 (one basename) and § 2.3 (only per-stage files carry the
          token) had both implied without making sayable.
Owns:     the grammar; the vocabulary -- every file molbuilder writes
          (`WRITTEN`, a `ROLE_*` constant each), the engines' files it names
          (`ENGINE_FILES`), each output's readers (`Artifact.reads`); and both
          directions across it.
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

    <label>[_<stage>]<role>

    my-job.chk                        carried: no stage
    my-job_optimized.xyz              carried, role-style separator
    my-job_01_coarse.fdf              this rung's
    my-job_01_coarse.out              this rung's run's output

``_`` introduces the stage, and the token sits IMMEDIATELY AFTER THE LABEL and
never inside the role.  A run's files carry no number of their own: a run is
one start in its own folder (`project-layout.md` § 1.6.1), and the folder
names the run.

A FILE NAMED ON THE LABEL ALONE HAS NO TOKEN -- every name an engine gives by
its label, and a run's own result (`<label>.spectra.json`).  The CARRIED files
are among them, and that is what carrying means: SIESTA's `.XV` / `.DM` /
`.CG` and PySCF's `.chk` / `_optimized.xyz` are how one rung hands the next its
geometry and density.  A token in them would make every rung look for a file
only its own stage ever wrote.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Tuple, Union

#: A stage artifact token: `01_coarse`, `02_electrode_L`.  The ordinal is
#: **two or more** digits and the name is ``[A-Za-z0-9_]+`` -- `identity.py`'s
#: ``STAGE_NAME_RE`` and ``stage_token``, which own the word itself; this
#: is only the shape a NAME may carry, so a token that module would reject
#: cannot reach a filename through here either.
_STAGE = re.compile(r"[0-9]{2,}_[A-Za-z0-9_]+")

#: THE ROLES OF OUR FILES -- one constant per row of :data:`WRITTEN` named on
#: the label, each spelled here and nowhere else (`model/parse.md` § 5.5
#: R-RO1; `job-contracts.md` § 2.2): the rows are written with them, and
#: every writer and reader imports them.  A deck has none: shared code asks
#: an engine's deck by the engine (:func:`deck_role`), so no module names
#: SIESTA's deck where any engine's would do.
ROLE_TEMPLATE      = ".template.toml"      # every parameter, as given
ROLE_SOURCE        = ".source.xyz"         # the structure the calculation is of
ROLE_SOURCE_SIDECAR = ".source.molstruct.json"   # its cell and labels
ROLE_VALIDATION    = ".validation.txt"     # what prep checked, beside the deck
ROLE_INITIAL       = "_initial.xyz"        # a PySCF run's input geometry
ROLE_INITIAL_SIDECAR = "_initial.molstruct.json"
ROLE_OPTIMIZED_SIDECAR = "_optimized.molstruct.json"
ROLE_CONSTRAINTS   = ".constraints.txt"    # geomeTRIC's held atoms
ROLE_POINTS        = ".points.json"        # a run of points: its index
ROLE_SPECTRA       = ".spectra.json"       # a vibration's result
ROLE_SPECTRA_DATA  = ".spectra.nc"         # the same result as structured data
ROLE_OPTICAL       = ".optical.nc"         # a SIESTA vibration's optical calls, as data
ROLE_RUN_SCRIPT    = ".run.sh"             # the run script
ROLE_HEADER        = ".sbatch"             # its queue header
ROLE_MOLWATCH      = ".molwatch.log"       # the run as it happens, every engine
ROLE_SIESTA_OUT    = ".out"                # the SIESTA family's stdout
ROLE_PYSCF_STDOUT  = ".pyscf.log"          # PySCF's stdout
ROLE_PYSCF_LOG     = ".log"                # PySCF's own log, a rung's own
#: geomeTRIC NAMES ITS FILES UNDER THE PREFIX A PySCF DECK HANDS IT, the
#: run's stem and this (`geometric/optimize.py`): ``<prefix>_optim.xyz``,
#: its trajectory, and ``<prefix>.tmp/``, its scratch folder.
GEOMETRIC_PREFIX   = "_geom"
GEOMETRIC_APPENDS  = "_optim.xyz"
ROLE_GEOM_TRAJ     = GEOMETRIC_PREFIX + GEOMETRIC_APPENDS  # a rung's own trajectory
ROLE_SESSION_LOG   = ".runwrap.log"        # the run script's session log
ROLE_MONITOR_LOG   = ".monitor.log"        # the monitor's rolling status
ROLE_UTIL_CSV      = ".util.csv"           # the monitor's samples
ROLE_SCF_TIMING    = ".scf-timing.log"     # the run script's SCF tee
ROLE_PARSE_LOG     = ".parse.log"          # molbuilder's log of reading an output
ROLE_MOLWATCH_PARSE_LOG = ".molwatch.parse.log"
ROLE_TRANSPORT_RECORD = ".transport.json"  # a transport calculation's results
ROLE_TRANSPORT_DATA = ".transport.nc"      # the same results as data
ROLE_TBT_DIGEST    = ".{tbt}.digest.nc"    # a transmission point's data file, read
ROLE_TBT_DIGEST_CLAIM = ROLE_TBT_DIGEST + ".reading"   # its reading under way
ROLE_FC_SWEEP      = ".fc-sweep.json"      # a displacement sweep, compared
ROLE_CONCLUDED     = ".concluded"          # the conclusion marker
ROLE_RUNTIME_INFO  = ".runtime_info.json"  # what `molbuilder runtime-info` read
#: THE ONE LAYOUT'S NAME (`project-layout.md` § 1) -- what every description
#: states as its ``shape`` (`task`) and every prep's log carries in its name.
LAYOUT             = "hierarchical"
ROLE_PIPELINE_LOG  = ".{engine}." + LAYOUT + ".pipeline.log"  # a prep's log

#: THE ENGINES' FILES MOLBUILDER NAMES IN CODE -- one constant per row of
#: :data:`ENGINE_FILES` (below) that a module of ours composes, reads or
#: carries by name, spelled here and nowhere else (`job-contracts.md` § 2.2):
#: an engine module asks its own, shared code an engine's by its record.
SIESTA_XV          = ".XV"                 # the geometry and velocities
SIESTA_DM          = ".DM"                 # the density matrix
SIESTA_CG          = ".CG"                 # the conjugate-gradient optimizer's history
SIESTA_HSX         = ".HSX"                # the Hamiltonian and overlap
SIESTA_FC          = ".FC"                 # a force-constant run's forces
SIESTA_EPSIMG      = ".EPSIMG"             # an optical call's dielectric function
SIESTA_ORB_INDX    = ".ORB_INDX"           # the orbitals each atom carries
SIESTA_MD_NC       = ".MD.nc"              # a run's steps, NetCDF
SIESTA_MD          = ".MD"
SIESTA_MDE         = ".MDE"
SIESTA_ANI         = ".ANI"
SIESTA_BASIS_ENTHALPY = ".BASIS_ENTHALPY"
SIESTA_FDF_LOG     = "fdf.{stamp}.log"     # every setting SIESTA read
SIESTA_TSHS        = ".TSHS"               # an electrode's Hamiltonian
SIESTA_TS_HSX      = ".TS.HSX"             # the device's, read by TBtrans
SIESTA_TSDE        = ".TSDE"               # the device's NEGF density
SIESTA_TSGF        = ".{bias}.TSGF{lead}"  # a lead's Green's function, at a voltage
TBTRANS_NC         = ".{tbt}.nc"           # a transmission point's data file
TBTRANS_CC         = ".TBT.CC"             # its energy contour
TBTRANS_DOS        = ".{tbt}.DOS"          # its DOS at each k-point
TBTRANS_AVTRANS    = ".{tbt}.AVTRANS_{pair}"   # its k-averaged transmission
TBTRANS_AVTEIG     = ".{tbt}.AVTEIG_{pair}"    # ... its eigenchannels
TBTRANS_AVDOS      = ".{tbt}.AVDOS"            # ... the device's DOS
TBTRANS_AVADOS     = ".{tbt}.AVADOS_{lead}"    # ... a lead's spectral DOS
TBTRANS_AVBDOS     = ".{tbt}.AVBDOS_{lead}"    # ... a lead's bulk DOS
PYSCF_CHK          = ".chk"                # PySCF's checkpoint
PYSCF_CUBE         = ".cube"               # a grid `cubegen` writes
PYSCF_OPTIMIZED    = "_optimized.xyz"      # a PySCF relaxation's geometry
GEOMETRIC_SCRATCH  = GEOMETRIC_PREFIX + ".tmp"  # geomeTRIC's scratch folder

#: The basename rule, restated from `config/siesta.py::_validate_basename`
#: (job-contracts.md § 2.1: "a single token matching [A-Za-z0-9_-]+").  It is
#: restated rather than imported because THIS module must not depend on an
#: engine config to know what a label may look like -- the dependency runs the
#: other way.  A label carrying a dot would make `parse` ambiguous, which is
#: the one thing the pattern is load-bearing for.
_LABEL = re.compile(r"[A-Za-z0-9_-]+")


#: TBTRANS'S CHANNELS, each with the tag its files carry after the label
#: (``Util/TS/TBtrans/m_tbt_save.F90`` ``name_save``): ``<label>.TBT.nc``
#: unpolarized, ``<label>.TBT_UP.nc`` / ``<label>.TBT_DN.nc`` for the two
#: channels of a polarized run -- the ``tbt`` field's values, and every
#: reader's channel (`siesta_grammar.TBT_CHANNELS`).
TBT_CHANNEL_TAGS: "dict[str, str]" = {"unpolarized": "TBT", "up": "TBT_UP",
                                      "down": "TBT_DN"}

#: THE LEADS MOLBUILDER'S DECKS NAME, the left one then the right one: each
#: lead's ``TS.Elec.<name>`` block (`transport.transiesta`), and so the name
#: TranSIESTA and TBtrans give a lead's files in a run of ours -- the
#: ``lead`` field's values, and the ``pair`` field's example.
OUR_LEADS: "tuple[str, str]" = ("L", "R")


#: Every declared FIELD, in ONE place.
#:
#: **A field is the bounded flexibility** (`plans/plan.md` § 5l.1). A role names
#: what a file IS; a field carries a value that varies per file and belongs to
#: the NAME rather than to the vocabulary.
@dataclass(frozen=True)
class Field:
    """One declared field: what it means, its shape, and one real example.

    ``example`` is not decoration.  It is what lets anything WALK the catalogue
    and build a name -- the suite composes every role at every stage, and a
    template it cannot fill is a row it cannot check.  It is also how a reader
    learns what ``TBT(?:_UP|_DN)?`` looks like without decoding it, and a test
    asserts it matches its own shape, so it cannot rot.
    """
    what: str
    shape: str
    example: str
    #: EVERY VALUE MOLBUILDER'S OWN RUNS GIVE IT -- the TBtrans channels,
    #: the leads its decks name -- so the names a templated row makes for a
    #: run of ours can be walked; ``()`` where the example stands for them.
    values: "tuple[str, ...]" = ()


FIELDS: "dict[str, Field]" = {
    # The fields of the catalogue's FIXED-NAME families (B13, 2026-10-04):
    # files molbuilder writes that are not named on the label -- a
    # pseudopotential is named for its species label, a scheduler's output
    # for its kind and its job.
    "species": Field(
        what="the species label a pseudopotential is for -- an element and "
             "an optional index, `Au` or `Au1`: SIESTA opens "
             "`<label>.psml`",
        shape=r"[A-Z][a-z]?[0-9]*",
        example="Au1"),
    "jobid": Field(
        what="the scheduler's id for the job",
        shape=r"[0-9]+",
        example="62372574"),
    "scheduler": Field(
        what="the scheduler's kind, by its name in `scheduler/kinds.toml`",
        shape=r"[a-z][a-z0-9_]*",
        example="slurm"),
    "tbt": Field(
        what="a TBtrans data file's channel, as its name tags it: TBT "
             "(unpolarized), TBT_UP or TBT_DN (spin up, spin down)",
        shape="(?:" + "|".join(sorted(TBT_CHANNEL_TAGS.values(), key=len,
                                     reverse=True)) + ")",
        example="TBT", values=tuple(TBT_CHANNEL_TAGS.values())),
    # THE ENGINES' FILES' FIELDS (:data:`ENGINE_FILES`): what TranSIESTA and
    # TBtrans put in a name beside the label, and SIESTA's fdf log's stamp.
    "bias": Field(
        what="a bias point's token, as its folder and its leads' Green's "
             "functions are named (`task.bias_token`): v0, v0.2, v-0.5",
        shape=r"v-?[0-9][0-9.e+-]*",
        example="v0.2"),
    "lead": Field(
        what="a lead as molbuilder's decks name it in their `TS.Elecs` "
             "block, :data:`OUR_LEADS` -- the name TranSIESTA and TBtrans "
             "give its files (`m_tbt_save.F90` `name_save`)",
        shape="(?:" + "|".join(OUR_LEADS) + ")",
        example=OUR_LEADS[0], values=OUR_LEADS),
    "pair": Field(
        what="the two leads a TBtrans quantity runs between, as its file "
             "names them",
        shape="(?:" + "|".join(OUR_LEADS) + ")-(?:" + "|".join(OUR_LEADS)
              + ")",
        example="-".join(OUR_LEADS)),
    "stamp": Field(
        what="the moment SIESTA opened its fdf log, as it names the file",
        shape=r"[0-9]{8}T[0-9]{6}\.[0-9]+",
        example="20260925T194936.695"),
    "pid": Field(
        what="the run script's own process id",
        shape=r"[0-9]+",
        example="48213"),
    "group": Field(
        what="a launch group's name (`GroupNames.of_*`): a task's group "
             "or a sequence of groups, its stages' tokens; a benchmark's "
             "side and shelf; a run of points' walk",
        shape=r"[A-Za-z0-9_.-]+",
        example="bench-group-cpu-G0N4C1"),
    "random": Field(
        what="a temporary file's random part",
        shape=r"[A-Za-z0-9_.-]+",
        example="k3j9x_2a"),
    "target": Field(
        what="the name of the file a temporary one replaces, or a lock "
             "guards",
        shape=r"[^/]+",
        example="H2.molstruct.json"),
    # "engine": the catalogue's own engines, set once its rows are declared
    # (below WRITTEN) -- never a second list typed here.
    "kept": Field(
        what="the name of a record `jobset migrate` rewrote",
        shape=r"[A-Za-z0-9_.-]+",
        example="task.json"),
    "rule": Field(
        what="the earliest rule the kept record was written before -- a "
             "plan's code, a day or a release",
        shape=r"[A-Za-z0-9_.-]+",
        example="2.27.0"),
}

_FIELD_RE = re.compile(r"\{([a-z_]+)\}")

#: THE SCHEDULER'S OWN OUTPUT FILES, as a header asks for them: named for the
#: scheduler's kind and the job's id, the id the scheduler fills in from its
#: own placeholder (`scheduler.kinds.Kind.output_file`).  One name for every
#: kind; the kind spells only its placeholder.
SCHEDULER_STDOUT = "{scheduler}.{jobid}.out"
SCHEDULER_STDERR = "{scheduler}.{jobid}.err"
#: The per-rank GPU launcher the run script writes beside the job, named on
#: the script's own process id (`runwrap`), removed at its exit.
RANK_LAUNCHER = ".mb-rank-launch-{pid}.sh"

#: A SIESTA pseudopotential, named for its species label (`fixed_name`).
PSEUDO_FILE = "{species}.psml"
#: A record `jobset migrate` rewrote, kept beside its rewrite under the
#: earliest rule it was written before (`migrate._kept_as`).
KEPT_FILE = "{kept}.pre-{rule}"
#: A LAUNCH GROUP'S FILES, in its `launch/` folder (:class:`GroupNames`):
#: its sequencer, its queue header, its log, a sequence's launch record.
GROUP_RUN_SCRIPT = "{group}.run.sh"
GROUP_HEADER = "{group}.sbatch"
GROUP_LOG = "{group}.log"
GROUP_LAUNCH_RECORD = "{group}.launch.json"
#: The saved states' big files, by content (`checkpoint`), and the ignore
#: list that sends them there.
#:
#: **DO NOT RENAME THE DIRECTORY TO MATCH THE TOOL'S NAME** (user's ruling).
#: The tool is `molbuilder checkpoint` everywhere a person types it; the
#: directory sits inside calculation folders that already exist, and
#: renaming it orphans real archives -- whole copies of large files -- to no
#: one's benefit.  A verb someone types is the tool's name; a directory on
#: disk is the data's.
SNAPSHOT_DIR = ".binsnapshots"
GITIGNORE_FILE = ".gitignore"
#: The saved states themselves, git's own folder (`checkpoint`).
HISTORY_DIR = ".git"


def _fields_in(role: str) -> "tuple[str, ...]":
    """The field names a role TEMPLATE carries, in the order they appear."""
    return tuple(_FIELD_RE.findall(role))


@functools.lru_cache(maxsize=None)
def _role_pattern(role: str) -> "re.Pattern":
    """A templated role, compiled -- each field replaced by its declared shape;
    once a role, since the roles are the catalogue's fixed set and every
    reader of a folder classifies each of its files by them."""
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

    ``(".TBT_UP.digest.nc")`` -> ``(".{tbt}.digest.nc", {"tbt": "TBT_UP"})``,
    an engine's file's alike (``".v0.2.TSGFL"`` -> ``SIESTA_TSGF``, its
    ``bias`` and ``lead``), and anything that matches no template comes back
    unchanged with no fields. This is what lets every comparison in this
    module stay an EQUALITY on the declared role.
    """
    for template in _templated_roles():
        m = _role_pattern(template).match(concrete)
        if m:
            return template, m.groupdict()
    return concrete, {}


class RunFileError(ValueError):
    """A name that cannot be composed, refused rather than guessed at."""


@dataclass(frozen=True)
class RunFile:
    """What a filename says it is.  ``stage`` is None for a file named on
    the label alone -- a carried file among them."""
    label: str
    stage: Optional[str]
    role:  str
    #: The declared FIELDS the name carried, as ``(name, value)`` pairs -- a
    #: tuple so the record stays hashable and comparable.  :attr:`role` is
    #: the TEMPLATE for a name that carries fields, so two transmission
    #: digests of two channels compare equal on role and differ here, which
    #: is what a field is for.
    fields: "tuple[tuple[str, str], ...]" = ()

    @property
    def name(self) -> str:
        """The name again, by :func:`compose` -- whose role the name it was
        read from declares, whoever's file that was."""
        return compose(self.label, self.role, self.stage, roles=(self.role,),
                       **dict(self.fields))


def split_stem(stem_: str) -> "tuple[str, Optional[str]]":
    """``(label, stage)`` read back out of a stem :func:`stem` composed --
    the token's own shape ends the label (§ 2.2a: a stage token starts with
    two digits after the underscore), so ``H2_01_coarse`` is ``("H2",
    "01_coarse")`` and ``H2`` is ``("H2", None)``.  A label that itself
    holds ``_NN_`` splits at the first such place."""
    m = re.match(r"^(.*?)_(" + _STAGE.pattern + r")$", stem_)
    return (m.group(1), m.group(2)) if m else (stem_, None)


def stem(label: str, stage: Optional[str] = None) -> str:
    """``<label>[_<stage>]`` -- what every role attaches to.

    The half of :func:`compose` a caller needs on its own: a run's basename,
    which every file of its rung begins with (:attr:`RunNames.stem`).

    ``stage`` is None for a file named on the label alone.  An empty string
    is refused rather than read as None: every description has a stage
    (`job-contracts.md` § 2.2a), and a silent coercion is how a name with no
    stage reaches a staged file.
    """
    if not _LABEL.fullmatch(label or ""):
        raise RunFileError(
            f"label {label!r} is not a single [A-Za-z0-9_-]+ token"
            f".  A label carrying a dot or a space "
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
            f"stage token {stage!r} is not `NN_name`.  Pass None for a file that CARRIES between rungs; a "
            f"malformed token would produce a name nothing can parse back.")
    return f"{label}_{stage}"


def compose(label: str, role: str, stage: Optional[str] = None,
            roles: "tuple[str, ...]" = (), **fields) -> str:
    """The one way a run file gets its name (§ 2.2a).

    ``role`` is what the file IS and begins with ``.`` or ``_`` -- ``".chk"``,
    ``"_optimized.xyz"``, ``".molwatch.log"`` -- a constant of this module,
    never spelled at a call site.

    ``stage`` omitted is a file named on the label alone, which is a
    statement and not a default: a carried file, which crosses rungs, or a
    name an engine gives by its label.

    ``fields`` fill a templated role (:data:`FIELDS`): every field the role
    names, and no other.

    ``role`` is a role of :data:`WRITTEN`, of :data:`ENGINE_FILES`, or of
    ``roles`` -- the vocabulary a caller holds that neither table may (a
    calculation's own restart-file list) -- and anything else is refused
    (:func:`declared`): a misspelled role names a file nobody reads.
    """
    if not role or role[0] not in "._":
        raise RunFileError(
            f"role {role!r} must begin with '.' or '_' -- it is the part that "
            f"says what the file IS, and the separator is what lets `parse` "
            f"find where the label ends.")
    declared(role, roles, door="compose")
    # A field is refused unless the role declares it, and a declared one is
    # refused unless its value matches the shape in `FIELDS` -- a channel
    # that is not a channel would produce a name `parse` could not read back.
    wanted = _fields_in(role)
    stray = [k for k in fields if k not in wanted]
    if stray:
        raise RunFileError(
            f"role {role!r} carries no field {stray[0]!r} "
            f"({', '.join(wanted) or 'it carries none'}).")
    missing = [f for f in wanted if f not in fields]
    if missing:
        raise RunFileError(
            f"role {role!r} needs the field(s) {', '.join(missing)}; a "
            f"template cannot be composed without them.  Pass "
            f"{missing[0]}=... .")
    for k, v in fields.items():
        # A FIELD LEFT AS ITS PLACEHOLDER is a template's (:func:`manifest`
        # shows the field by its name).
        if v == "{" + k + "}":
            continue
        if not re.fullmatch(FIELDS[k].shape, str(v)):
            raise RunFileError(
                f"{k}={v!r} does not match the declared shape "
                f"{FIELDS[k].shape!r} for that field "
                f"({FIELDS[k].what}; e.g. {FIELDS[k].example}).")
    role = _FIELD_RE.sub(lambda m: str(fields[m.group(1)]), role)
    return f"{stem(label, stage)}{role}"


#: A stand-in label for :func:`tail`.  Any legal label would do; what matters
#: is that the tail is CUT from a real `compose` result rather than assembled
#: beside one, so the two cannot disagree.
_PLACEHOLDER = "L"


def tail(role: str, stage: Optional[str] = None,
         roles: "tuple[str, ...]" = (), **fields) -> str:
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
    return compose(_PLACEHOLDER, role, stage, roles=roles,
                   **fields)[len(_PLACEHOLDER):]


def _carrying(label: str, stage: Optional[str], role: str) -> "RunFile":
    """A :class:`RunFile` whose role has been mapped back to its template.

    Every arm of :func:`parse` builds its result through here, so a concrete
    role and its declared template can never disagree about which file this is.
    """
    template, values = canonical_role(role)
    return RunFile(label, stage, template, tuple(sorted(values.items())))


def parse(filename: str, label: str,
          roles: "tuple[str, ...]" = ()) -> Optional[RunFile]:
    """Read a name back, or None when it is not this label's file.

    ``label`` is required, and that is the whole reason this can be exact: a
    role may contain underscores (``_geom_optim.xyz``) and a label may too, so
    nothing can find the boundary between them by looking at the string alone.

    ``roles`` is the caller's own vocabulary, added to the one this module
    knows (:data:`WRITTEN` and :data:`ENGINE_FILES`).  It only matters for
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

    # CARRIED: the role follows the label.
    if rest[0] == ".":
        return _carrying(label, None, rest)
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
    known = sorted({r for r in (*_declared_roles(), *roles)
                    if r.startswith("_")},
                   key=len, reverse=True)
    for role in known:
        if after.endswith(role) and _STAGE.fullmatch(after[:-len(role)]):
            return _carrying(label, after[:-len(role)], role)
    # Otherwise the token runs to the role's own separator, which for every
    # remaining role is the first `.`.
    m = re.match(r"([0-9]{2,}_[A-Za-z0-9_]+)", after)
    if m:
        # A TOKEN FOLLOWED BY ANYTHING BUT A ROLE is no file of this label:
        # a run's files are named once, on the stage's stem, so nothing
        # stands between a stage and its role (§ 2.2a; `project-layout.md`
        # § 1.6.1) -- ``_01_coarse-x.out`` is no name of a run's.
        role = after[m.end():]
        return _carrying(label, m.group(1), role) if role[:1] == "." else None
    # No token: the role itself began with `_` (`_optimized.xyz`).
    return _carrying(label, None, rest)


def _tail(name: str, role: str) -> str:
    """The end of *name*, long enough to hold *role* -- for a label-less match.

    A templated role has no fixed length, so the slice is taken from the first
    literal segment of the template (`.` for `.{tbt}.digest.nc`).
    """
    head = role.split("{", 1)[0]
    if head != role:
        i = name.find(head)
        return name[i:] if i >= 0 else name
    return name[-len(role):] if len(role) <= len(name) else name


def role_of(name) -> Optional[str]:
    """The DECLARED role this filename carries (:func:`_role_of`), read once
    a name: a pure function of the name, and every reader of a folder asks
    it of each of its files."""
    return _role_of(str(name).rsplit("/", 1)[-1])


def is_transient(name) -> bool:
    """Whether a file of this name is one the catalogue calls ``transient``
    -- there only while a write or a reading runs (a temp of the one
    writer, a lock, a reading's claim): never copied into another run or
    calculation, where it would speak for a process that is not there."""
    n = str(name).rsplit("/", 1)[-1]
    # A ROW IS ITS ROLE on the label, else its whole name.
    return any(_role_pattern(a.role or a.name).search(n)
               for a in _transient_rows())


@functools.lru_cache(maxsize=1)
def _transient_rows() -> "tuple[Artifact, ...]":
    return tuple(a for a in WRITTEN if a.kind == "transient")


def transient_families() -> "tuple[str, ...]":
    """:func:`is_transient` as families of file names -- each field a star,
    a role on the label after one -- for an ignore list or an exclude line
    (`warmfiles.never_carried_families`)."""
    return tuple(sorted({family(a.role) if a.role
                         else _FIELD_RE.sub("*", a.name)
                         for a in _transient_rows()}))


def family(role: str) -> str:
    """EVERY NAME A ROLE MAKES, as a glob -- a star for the label and one
    for each field: ``*.DM``, ``*.*.AVTRANS_*`` -- for an ignore list, an
    exclude line or a hint of which files are large (`runtime_config`'s
    checkpoint defaults); never a star placed by hand at the call."""
    return "*" + _FIELD_RE.sub("*", declared(role, door="family"))


@functools.lru_cache(maxsize=65536)
def _role_of(name: str) -> Optional[str]:
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
    what the file IS.  ``job_01_relax.pyscf.log`` ends with `.pyscf.log` AND
    with `.log`, and only the first says which file this is.
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
         stage: Optional[str] = None) -> "list[tuple[Path, RunFile]]":
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

    Returns ``(path, RunFile)`` pairs, sorted by name.  **The parsing is
    :func:`parse`'s**, not a second reader:
    a name that does not read back with this label -- another label's, a
    file foreign to the run -- is left out, which is what keeps a foreign
    file in the same directory from being reported as one of the run's.  An
    engine's file can carry a stage too -- geomeTRIC's trajectory is named
    on its rung's token (``<label>_<NN>_<stage>_geom_optim.xyz``) -- so a
    caller asks by role as well when it wants one of molbuilder's.

    ``role`` narrows to one, and it is an EQUALITY on the declared role --
    the catalogue's spelling, template and all (``".{tbt}.digest.nc"``, every
    channel's), or a concrete name of it (``".TBT_UP.digest.nc"``, that
    channel's).
    Every name is mapped back to its template by :func:`canonical_role` before
    the comparison, so one rule reads and one rule matches.  ``roles`` passes a caller's own
    vocabulary through to :func:`parse` for the ``_``-prefixed kind, exactly as
    that function documents.  ``stage`` filters on what was parsed.

    Takes a directory and returns paths: stdlib ``pathlib`` only, so this stays
    importable by a monitor shipped beside a job (§ 4.5's floor rule).
    """
    if role is not None:
        declared(role, roles, door="find")
        # A CONCRETE NAME OF A TEMPLATED ROLE (``.TBT_UP.nc``) asks for its
        # template with those fields; the template itself, for every value.
        want, want_fields = canonical_role(role)
    d = Path(directory)
    try:
        # FILES, which is what the first line of this docstring says and what
        # every caller wants.
        entries = sorted(e for e in d.iterdir() if e.is_file())
    except OSError:
        return []
    out: "list[tuple[Path, RunFile]]" = []
    for entry in entries:
        parsed = parse(entry.name, label, roles)
        if parsed is None:
            continue
        if role is not None and (
                parsed.role != want
                or not set(want_fields.items()) <= set(parsed.fields)):
            continue
        if stage is not None and parsed.stage != stage:
            continue
        out.append((entry, parsed))
    return out


def find_on_label(directory, label: str, role: str) -> "list[Path]":
    """The label's files in ``role`` named on the label ALONE, no stage
    token -- as an engine names its own (SIESTA's ``<label>.FC``,
    TBtrans's ``<label>.TBT.nc``) and a run its result
    (``<label>.spectra.json``) -- sorted by name.  THE ONE BODY behind
    `runs.Run.on_label` and every reader of an engine file on the label
    (`job-contracts.md` § 2.2a): a reader that copied it answered the same
    question a second way."""
    declared(role, door="find_on_label")
    return [p for p, rec in find(directory, label, role=role)
            if rec.stage is None]


#: THE RELEASE SINCE WHICH A RUN IS ONE START, its files named once on its
#: stage's stem (`project-layout.md` § 1.6.1): a run named before it carries
#: the try's number (:func:`older_named`).
ONE_START_SINCE = "2.60.0"

#: WHAT WAS NAMED BY THE TRY AND ITS ENGINE NAMES SINCE 2.60.0:
#: SIESTA's history, appended to ``<SystemLabel><role>`` -- the label alone
#: -- which those run scripts pointed at the run's numbered file while the
#: engine ran (2.59's ``pointed_roles``).
_ENGINE_NAMED_SINCE = (SIESTA_MD_NC, SIESTA_MD, SIESTA_MDE, SIESTA_ANI)


@dataclass(frozen=True)
class OlderName:
    """A run's file as molbuilder named it before :data:`ONE_START_SINCE`
    (:func:`older_named`): its ``path``; the try's number its name carried
    (``-run<N>``), ``None`` for a start's session log, which carried the
    clock; the ``label`` it is named on; the ``role`` it is now; and, for
    an older file an earlier migration kept beside its rewrite
    (:data:`KEPT_FILE`), the rule it was kept under (``kept``)."""
    path: Path
    tried: Optional[int]
    label: str
    role: str
    kept: Optional[str] = None

    @property
    def staged(self) -> bool:
        """Whether its name now carries the stage -- not for a file its
        engine names (SIESTA's history)."""
        return self.role not in _ENGINE_NAMED_SINCE

    def anew(self, stage: str) -> str:
        """Its name now, by the one composer (:func:`compose`), on the
        label it carries and the stage token ``stage`` -- a kept file's
        under its rule."""
        name = compose(self.label, self.role, stage if self.staged else None,
                       roles=(self.role,) if self.kept else ())
        return (fixed_name(KEPT_FILE, kept=name, rule=self.kept)
                if self.kept else name)


def older_named(directory, stage: str) -> "list[OlderName]":
    """THE FILES OF ``directory`` AN OLDER GRAMMAR NAMED -- molbuilder's
    before :data:`ONE_START_SINCE`, when a run could start more than once
    (`project-layout.md` § 1.6.1): the try's number after the stage token
    ``stage`` (``<label>_<stage>-run<N><role>``) and a session log per
    start stamped with the clock (``<label>_<stage>.runwrap-<YYYYMMDD-
    HHMMSS>.log``) -- each also kept beside a rewrite by an earlier
    migration (:data:`KEPT_FILE`) -- sorted by name; ``[]`` for a folder
    that cannot be listed.  Read by `jobset migrate`, which names them once,
    and by the run door, which says a run holding them is not read until
    then."""
    label = _LABEL.pattern
    tried = re.compile(rf"^(?P<label>{label})_" + re.escape(stage)
                       + r"-run(?P<n>[0-9]+)(?P<role>\..+)$")
    # THE CLOCK THOSE RUN SCRIPTS STAMPED, `date +%Y%m%d-%H%M%S`.
    logged = re.compile(rf"^(?P<label>{label})_" + re.escape(stage)
                        + r"\.runwrap-[0-9]{8}-[0-9]{6}\.log$")
    try:
        entries = sorted(Path(directory).iterdir())
    except OSError:
        return []
    out = []
    for f in entries:
        if not f.is_file() or f.is_symlink():
            continue
        kept = fixed_fields(KEPT_FILE, f.name)
        name = kept["kept"] if kept else f.name
        rule = kept["rule"] if kept else None
        m = tried.match(name)
        if m:
            out.append(OlderName(f, int(m["n"]), m["label"], m["role"],
                                 rule))
            continue
        m = logged.match(name)
        if m:
            out.append(OlderName(f, None, m["label"], ROLE_SESSION_LOG,
                                 rule))
    return out


def longest_tail(engine_names: "tuple[str, ...]") -> str:
    """The longest text the catalogue appends to ``<label>_<stage>`` -- every
    role, a templated one filled with the longest value its field takes:
    the longest engine name.  What `identity.MAX_LABEL_BYTES` is derived
    from (`run-identity.md` § 3)."""
    fill = {"engine": max(engine_names, key=len)}
    out = ""
    for r in roles(engines=False):
        try:
            t = r.format(**fill)
        except (KeyError, IndexError):
            t = r
        if len(t) > len(out):
            out = t
    return out


def stem_before(name, role: str) -> Optional[str]:
    """What precedes ``role`` in ``name`` -- the stem every other file of
    that run is named on -- or ``None`` when ``name`` does not end in it, or
    is nothing but it.  ``role`` is one of this module's (the constant,
    never a literal at the call site) or a concrete name of a templated
    one (``.TBT_UP.nc``) (:func:`declared`) -- a templated role read by its
    pattern, so a role :func:`role_of` answers cuts the name
    it was read off; the cut is one place, so a reader handed a path and no
    label (`parse.engines.pyscf`, geomeTRIC's trajectory) does not split the
    name by hand."""
    declared(role, door="stem_before")
    base = str(name).rsplit("/", 1)[-1]
    if _FIELD_RE.search(role):
        m = _role_pattern(role).search(base)
        return base[:m.start()] if m and m.start() > 0 else None
    if len(base) <= len(role) or not base.endswith(role):
        return None
    return base[:-len(role)]


def fields_of(name, role: str) -> "Optional[dict[str, str]]":
    """THE FIELDS A NAME OF THE TEMPLATED ``role`` CARRIES --
    ``fields_of("tsgf/AuBDT.v0.2.TSGFL", SIESTA_TSGF)`` is ``{"bias":
    "v0.2", "lead": "L"}`` -- or ``None`` for a name not of it: what an
    engine's own line names a file by, read back without its label."""
    declared(role, door="fields_of")
    base = str(name).rsplit("/", 1)[-1]
    m = _role_pattern(role).search(base)
    return m.groupdict() if m and m.start() > 0 else None


def named_beside(name, role: str) -> Optional[str]:
    """THE NAME OF THE FILE IN ``role`` BESIDE THE FILE ``name`` -- on the
    stem ``name``'s own role leaves, its label and stage kept
    (:func:`split_stem`), the stage dropped where ``role``'s row names its
    file on the label alone -- or ``None`` for a name that ends in no role
    of either table on a stem a label begins (``notes.txt``,
    ``my.job.out``), and for a fixed name (``coordinates.fdf``).  Its own
    role is the longest that ends it, as :func:`parse` reads an underscore
    role (``H2_01_relax_geom_optim.xyz`` is ``H2_01_relax``'s, not
    ``H2_01_relax_geom_optim``'s ``.xyz``).  What a runtime record, a
    parse log, a deck's validation report and the output a deck names are
    named by; ``role`` refused as :func:`compose` refuses one."""
    declared(role, door="named_beside")
    base = str(name).rsplit("/", 1)[-1]
    if any(a.name == base and not a.role for a in WRITTEN):
        return None
    cut: Optional[int] = None
    for own in roles():
        if _FIELD_RE.search(own):
            m = _role_pattern(own).search(base)
            at = m.start() if m else 0
        else:
            at = len(base) - len(own) if base.endswith(own) else 0
        if at > 0 and _LABEL.fullmatch(split_stem(base[:at])[0]) \
                and (cut is None or at < cut):
            cut = at
    if cut is None:
        return None
    label, stage = split_stem(base[:cut])
    return compose(label, role,
                   stage if _staged(role, "named_beside") else None)


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
    declared(role, door="find_by_role")
    if not role.startswith("."):
        raise RunFileError(
            f"find_by_role({role!r}): only a dotted role can be found without "
            f"a label -- an underscore role cannot be told from a stage name "
            f"without one (see `parse`).  Use `find(dir, label, role=...)`.")
    known = {a.role for a in WRITTEN}
    if role not in known:
        raise RunFileError(
            f"find_by_role({role!r}): a name is read without its label by "
            f"the roles of `runfiles.WRITTEN` alone (`role_of`) -- an "
            f"engine's file, or one name of a templated role, is found by its "
            f"label (`find`).  The dotted roles are "
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
# It lives beside the grammar because every reader of a name needs the same
# table and none may hold its own copy -- :func:`manifest` among them, the
# concrete names for a person being told what a prep is about to write (user,
# 2026-09-07: *"a card that list all the generated data file from the
# setup"*).


@dataclass(frozen=True)
class Artifact:
    """One kind of file molbuilder writes, and the shapes its name takes.

    ``role`` is the suffix :func:`compose` takes, for a file named on the
    label; ``name`` is the whole name of one that is not (`task.json`, a
    pseudopotential's ``{species}.psml``) -- a row has exactly one of the two.
    ``what`` is one line a person can read -- these are shown in the
    Task-setup card and the Results tab's file card, so it says what the file
    HOLDS; ``writer`` says who writes it, and when.
    """
    role: str = ""
    what: str = ""
    #: The FIELDS this role's template carries, if any -- the bounded
    #: flexibility (`plans/plan.md` § 5l.1).  A role naming ``{tbt}`` is a
    #: family of names, and the field is what tells them apart; the shape of
    #: each is declared once in :data:`FIELDS`.
    fields: "tuple[str, ...]" = ()
    #: Does its name carry the stage's token?  False for a file named on the
    #: label alone -- the calculation's own (the template, the source pair)
    #: and a run's result (`.spectra.json`, `.transport.json`, ...).
    staged: bool = True
    #: Which engine writes it, or None for both.  A row cut from an output
    #: (:attr:`cut_from`) takes that output's.
    engine: Optional[str] = None
    #: THE OUTPUT THIS FILE IS CUT FROM, by its role, as the run writes it --
    #: the SCF-timing tee's, a SIESTA-family ``.out`` -- or ``""``.  The one
    #: statement of which output the run script cuts it from
    #: (:func:`cut_from_output`); the row's engine is that output's.
    cut_from: str = ""
    #: WHEN it appears -- "setup" (the description hand-over), "prep" (the
    #: deck and its wrapper), "launch" (what a launch writes as it sends a
    #: run: its launch record, the hand-over's records, a launch group's
    #: sequencer), "run" (everything the run produces) or "summarize" (what
    #: `jobset summarize task` writes from results that exist, never the
    #: launch).  A card that lists a run's files has
    #: to say which of them exist yet, and the alternative was for the page
    #: to subtract one list from another.
    when: str = "run"
    #: Which CALCULATION writes it -- one kind, the kinds (a run of points'
    #: index), or None for any.  A relaxation has no spectrum, and a list
    #: that promised one would be describing a file the run will never
    #: write -- the same fault in the other direction as a file that is
    #: written and undeclared (:func:`written_by`).
    calculation: Union[str, Tuple[str, ...], None] = None
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
    #: folder), ``run`` (a run's folder), ``bench`` (a benchmark's container),
    #: ``launch`` (a launch group's folder), or ``transient`` (only while a
    #: write runs) -- said in words by :data:`LEVEL_WHERE`.
    level: str = "run"
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
    #: WHICH INDEX ITS ATOMS ARE LISTED BY (`model/overview.md` § 2.2b, F3):
    #: ``user_index`` or ``engine_index`` for a file molbuilder writes that
    #: holds atoms -- the answer for a file of this kind written before the
    #: file said so itself; ``""`` for any other row (an engine's own file is
    #: its run's deck's, F3).
    listed_by: str = ""
    #: AN ENGINE'S INPUT counts its atoms as its engine does (F5): every
    #: numbering its own text uses, by its name in `engine_atom_index`
    #: (``siesta``, ``geometric``, ``pyscf``) -- a PySCF deck uses PySCF's
    #: and writes geomeTRIC's -- said wherever its atom index map is shown;
    #: empty for a file that is not one.
    counted_by: "tuple[str, ...]" = ()
    #: IT TRAVELS BESIDE A DECK into every folder the deck runs in
    #: (`execution/architecture.md` § 3.3; :func:`beside_the_deck`): the run
    #: script and its header, the bundles -- with no ``engine`` beside every
    #: job, the monitor's; with one beside that engine's decks, the code a
    #: PySCF script imports -- a charged SIESTA deck's correction script, a
    #: SIESTA vibration call's coordinates; narrowed by the row's ``engine``
    #: and ``calculation``.  A job's finish is chosen by the job
    #: (`Job.finish`), not by this column; which travelling file is BUILT is
    #: the run script builder's own map (`runwrap.bundles_for`).
    travels: bool = False
    #: THE CODE THAT READS IT, by the question each answers
    #: (:data:`READ_QUESTIONS`) -- ``(question, "module:NAME")`` pairs, the
    #: one declaration of which reader a file of this role gets
    #: (`job-contracts.md` § 2.2; :func:`reads_of`), resolved by
    #: `runpoints.part` on the machine and beside a job, and the modules of
    #: those a job asks (:data:`READ_BESIDE_A_JOB`) travel in its bundle by
    #: this row (`runwrap.monitor_companions`).
    reads: "tuple[tuple[str, str], ...]" = ()
    #: A RUN OUTPUT WHOSE ENGINE'S STDERR STAYS APART -- the SIESTA
    #: family's, kept in the session log: ``die`` flushes stdout on node 0
    #: alone, and another rank may say why it died only there
    #: (`running-a-job.md` § 3.5).  Every other output takes its stderr in.
    stderr_apart: bool = False

    @property
    def calculations(self) -> "tuple[str, ...]":
        """The kinds whose runs write it (:attr:`calculation`), read one
        way -- ``()`` for a row naming none."""
        if self.calculation is None:
            return ()
        return ((self.calculation,) if isinstance(self.calculation, str)
                else tuple(self.calculation))


#: WHERE A FILE OF EACH LEVEL LANDS (:attr:`Artifact.level`), in the words a
#: card listing it says them in (:func:`manifest`'s ``where``); a transient
#: file's moment is its ``only``.
LEVEL_WHERE = {"calculation": "in the calculation's folder",
               "stage": "in the stage's folder",
               "run": "in each run's folder",
               "bench": "in a benchmark's container",
               "launch": "in its launch group's folder"}


#: THE FIXED NAMES -- files molbuilder writes that are not named on the label,
#: spelled ONCE, here; each owner imports its own (`job-contracts.md` § 2.2,
#: B13, 2026-10-04).  Plain data, so this module stays stdlib and travels.
TASK_FILE = "task.json"
JOB_REQUEST_FILE = "job-request.json"
TASK_HANDOVER_FILE = "task.1st.json"
WARM_FILES_FILE = "warm-files.toml"
MACHINE_RECORD_FILE = "environment.json"
JOBSET_FILE = "job-set.json"
PREPARED_WITH_FILE = "prepared-with.json"
COORDINATES_FILE = "coordinates.fdf"
PLAN_FILE = "STAGE-PLAN.md"
LEDGER_FILE = "jobset-decisions.log"
PSEUDO_DIR = "pseudos"
JUNCTION_FILE = "junction.xyz"
JUNCTION_SIDECAR_FILE = "junction.molstruct.json"
JUNCTION_CITED_FILE = "junction.cited.fdf"
SLOT_PROVENANCE_FILE = "slot-provenance.json"
CALCDIR_FILE = "calcdir.json"
LAUNCH_RECORD_FILE = "run.json"
#: The run's notification record, in its run folder beside its launch
#: record (`runrecord.notified_path`).
NOTIFIED_RECORD_FILE = "notified.json"
CONTINUED_FROM_FILE = ".continued-from"
GATHERED_FROM_FILE = ".gathered-from"
COPIED_FROM_FILE = ".copied-from"
MONITOR_BUNDLE = "mb_monitor.pyz"
VIBRATION_BUNDLE = "mb_vibration.pyz"
PYSCF_BUNDLE = "mb_pyscf.pyz"
MAKOV_PAYNE_SCRIPT = "makov_payne_correction.py"
BENCH_RESULT_FILE = "bench-result.json"
LAUNCH_DIR = "launch"


def written_by(a: "Artifact", calculation: str, *, any_: bool = False) -> bool:
    """Whether the row ``a`` is written by ``calculation`` -- the kind it
    names, one of the kinds it names; a row naming none answers ``any_``
    (every calculation's file, where a list wants those too)."""
    if a.calculation is None:
        return any_
    return str(calculation) in a.calculations


def travelling(engine: str, calculation: str) -> "tuple[Artifact, ...]":
    """THE ROWS THAT TRAVEL BESIDE ``engine``'s DECK of a ``calculation``
    (:attr:`Artifact.travels`), each narrowed by its own ``engine`` and
    ``calculation`` -- in the catalogue's order."""
    return tuple(a for a in WRITTEN
                 if a.travels and a.engine in (None, engine)
                 and written_by(a, calculation, any_=True))


def beside_the_deck(engine: str, calculation: str, finish: Optional[str],
                    names: "RunNames") -> "tuple[str, ...]":
    """THE FILES THAT TRAVEL BESIDE A DECK into every folder it runs in --
    the one list, the catalogue's travelling rows (:func:`travelling`) by
    the stage's names, then the job's finish when it has one
    (`Job.finish`): what prep renders beside the deck, what a launch brings
    into a run (`materialize.bring_files`), and what a run's own plan step
    brings into a call it opens (`spectra.siesta_vibration`); the
    calculation's shared package (`JobSet.shared`, the pseudopotentials)
    comes beside them."""
    out = [names.name(a.role) if a.role else a.name
           for a in travelling(engine, calculation)]
    if finish is not None:
        out.append(finish)
    return tuple(out)


#: Ordered as `job-contracts.md` § 2.2 lists them: inputs, the wrapper, the
#: canonical trajectory, then the run's logs -- its own account of how it
#: went, which stays in its folder: a run folder is one start, and the next
#: run is a folder of its own (`project-layout.md` § 1.6.1).
WRITTEN: "tuple[Artifact, ...]" = (
    # ---- what we generated to run -------------------------------------
    Artifact(".fdf", "the SIESTA deck — every keyword this rung runs with",
             engine="siesta",
             when="prep", level="stage", kind="derived",
             writer="prep (`script_emit.prepare_deck`), in the stage's "
                    "folder; a copy in each run",
             door="runfiles.find_by_role",
             listed_by="engine_index", counted_by=("siesta",),
             reads=(("contract", "molbuilder.parse.engines.siesta_contract:"
                                 "contract_fields"),)),
    Artifact(".py", "the PySCF script this rung runs", engine="pyscf",
             when="prep", level="stage", kind="derived",
             writer="prep (`script_emit.prepare_deck`), in the stage's "
                    "folder; a copy in each run",
             door="runfiles.find_by_role",
             listed_by="engine_index", counted_by=("pyscf", "geometric")),
    Artifact(ROLE_TEMPLATE, "every parameter, with the value it was given",
             staged=False,
             when="setup", level="calculation", kind="source",
             writer="`molbuilder describe` and the Transport tab's describe "
                    "door (`describe.write_description`); the hand-over, "
                    "which the browser writes -- each naming it with "
                    "`template_file.template_filename`; `jobset migrate`, "
                    "over the file `template_file.of` found",
             door="template_file.of"),
    # THE STRUCTURE THE CALCULATION IS OF, written into the bundle by the
    # hand-over (`web/handover-procedure.md`).
    #
    # `.source` is the RESERVATION (`job-contracts.md` § 6.3): identities are
    # validated dot-free, so this is a name no engine output can take.
    # `{label}.xyz` is the ENGINE's -- SIESTA's `WriteCoorXmol` writes it --
    # and it is deliberately absent from this table.
    Artifact(ROLE_SOURCE, "the structure the calculation is of",
             staged=False,
             when="setup", level="calculation", kind="input",
             writer="the hand-over and `molbuilder describe` "
                    "(`StructureCodec.source_files`)",
             door="jobset.rung.structure_for",
             listed_by="user_index"),
    Artifact(ROLE_SOURCE_SIDECAR, "its cell and its region labels",
             staged=False,
             when="setup", level="calculation", kind="input",
             writer="the hand-over and `molbuilder describe` "
                    "(`StructureCodec.source_files`)",
             door="jobset.rung.structure_for",
             listed_by="user_index"),
    # THE DECK'S COMPANION REPORT: a file molbuilder writes and does not
    # declare reads as ENGINE OUTPUT.
    Artifact(ROLE_VALIDATION, "what the generator checked before it wrote "
                                "the deck",
             when="prep", level="stage", kind="record",
             writer="prep (`script_emit.write_validation_report`), beside "
                    "the deck",
             door="runs.Run.report"),
    # ---- what the DECK writes for itself ------------------------------
    # A file the generated script writes is molbuilder's too -- it is our
    # deck that writes it.
    Artifact(ROLE_INITIAL, "the input geometry, echoed back before "
                             "anything ran", staged=False, engine="pyscf",
             kind="result", writer="the PySCF deck",
             listed_by="engine_index"),
    # ITS SIDECAR, the pair's other half -- written by the deck through the
    # codec (`StructureCodec.write_moved`, imported from `mb_pyscf.pyz`).
    Artifact(ROLE_INITIAL_SIDECAR, "the input geometry's cell and "
                                        "labels",
             staged=False, engine="pyscf", kind="result",
             writer="the PySCF deck",
             listed_by="engine_index"),
    Artifact(ROLE_OPTIMIZED_SIDECAR, "the relaxed geometry's cell and "
                                          "labels -- the sidecar of the "
                                          "restart file `_optimized.xyz`",
             staged=False, engine="pyscf", kind="result",
             writer="the PySCF deck",
             listed_by="engine_index"),
    Artifact(ROLE_CONSTRAINTS, "which atoms are held still, in geomeTRIC's "
                                 "own format",
             staged=False, engine="pyscf", kind="derived",
             writer="the PySCF deck", only="atoms are held",
             listed_by="engine_index", counted_by=("geometric",)),
    # A RUN OF POINTS -- its index (`execution/runs-of-points.md` § 2.3):
    # every point of the run, what it needs and must hold, how it ended as
    # the job saw it -- a transport rung's points, a SIESTA vibration's calls.
    Artifact(ROLE_POINTS, "the run's index: every point, what it needs and "
                          "must hold, whether it was taken over, and how "
                          "each ended as the job saw it -- one per run",
             engine="siesta", when="launch",
             calculation=("vibration", "transport"), kind="record",
             only="a run of points: a SIESTA vibration, a device or "
                  "transmission rung over several frames or voltages",
             writer="launch, for the points it opens (`submit._points_walk`); "
                    "each point's report steps (`mb_monitor.pyz point`); a "
                    "planner's own step for the points it plans "
                    "(`spectra.siesta_vibration`)",
             door="runpoints.read_index"),
    # EITHER ENGINE'S RUN WRITES IT: the PySCF deck, and on SIESTA the job's
    # finish after the force-constant run (`engines/vibration.md` § 5.5).
    Artifact(ROLE_SPECTRA, "the spectrum this run computed: frequencies, "
                              "the strengths the engine computes, "
                              "thermochemistry",
             staged=False, calculation="vibration", kind="result",
             writer="the run itself: the PySCF deck, or a SIESTA "
                    "vibration's job, after the force constants and at its "
                    "end (`mb_vibration.pyz`)",
             door="parse.sidecars.spectra.SpectraSidecarFileParser",
             listed_by="user_index"),
    # THE SAME RESULT AS STRUCTURED DATA, written beside the JSON at every
    # write by the one writer (`engines/vibration.md` § 6.10.5).
    Artifact(ROLE_SPECTRA_DATA, "the spectrum as data on its grids -- the "
                            "structure, the modes, the thermochemistry, "
                            "every variable stating itself (NetCDF-4); "
                            "the JSON's sha256 beside",
             staged=False, calculation="vibration", kind="result",
             writer="the run itself, with the JSON "
                    "(`sidecars.spectra.write_spectra_payload`)",
             door="spectra.datafile.read_spectra_data",
             listed_by="user_index"),
    # A SIESTA VIBRATION'S OPTICAL CALLS AS DATA, rewritten by the run's own
    # record step after each optical call and by its finish
    # (`engines/vibration.md` § 5.11 § 8): the record of the Raman phase so
    # far -- none where the run planned no optical call.
    Artifact(ROLE_OPTICAL, "the optical calls as data on their grids -- each "
                           "frame's eps2 over the photon energies, its mode, "
                           "side and geometry, its SCF facts, and the "
                           "settings the calls read (NetCDF-4)",
             staged=False, engine="siesta", calculation="vibration",
             kind="result",
             writer="the run's own record steps -- the force constants', "
                    "every frame pending, then each optical call's -- and "
                    "its finish (`spectra.siesta_vibration`, "
                    "`spectra.optical_data.write_optical_data`)",
             door="spectra.optical_data.read_optical_data",
             only="Raman asked, and a mode picked for it",
             listed_by="user_index"),
    # ---- the wrapper --------------------------------------------------
    Artifact(ROLE_RUN_SCRIPT, "the run script — activates the environment, "
                              "tees the output, catches a kill",
             when="prep", level="stage", kind="derived", travels=True,
             writer="prep (`runwrap.write_run_wrapper`), beside its deck; "
                    "a copy in each run"),
    # ON A MACHINE WITH A QUEUE, whether the run is then submitted or not --
    # unless prep is told `--no-sbatch`, and never for a run of points, which
    # its walk's header sends (`jobset.prep`).
    Artifact(ROLE_HEADER, "the queue header — `sbatch` reads it",
             when="prep", level="stage", kind="derived", travels=True,
             writer="prep (`runwrap.write_run_wrapper`), beside the run "
                    "script; a copy in each run",
             only="a machine with a queue, no `--no-sbatch`, and a stage run "
                  "once -- a run of points is sent by its walk's header"),
    # ---- the canonical trajectory -------------------------------------
    # Written before the engine even starts.
    # SEEDED AT PREP, which is why it is `progress` and not `stdout`: it
    # exists before the engine does, so an empty one is a prep and says
    # nothing about a run.  It speaks only once its footer concludes.
    # PREP SEEDS IT, so its moment is prep's.
    Artifact(ROLE_MOLWATCH, "the run as it happens — coordinates, "
                              "energy and forces, one block per step",
             output="progress", when="prep", kind="record",
             writer="prep seeds the first run's, in its folder; PySCF's "
                    "deck writes each step into its run's own, SIESTA "
                    "never does",
             door="parse.engines.molwatch.MolwatchLogFileParser",
             listed_by="engine_index",
             reads=(("ending", "molbuilder.parse.engines._run_ending:"
                               "molwatch_reading"),
                    ("live", "molbuilder.parse.engines.molwatch_reader:"
                             "live_pass"),
                    ("timing", "molbuilder.parse.instruments.scf_timing_rows:"
                               "progress_log_timing_metrics"))),
    # ---- history: stdout and the logs ---------------------------------
    # THE RUN'S ACCOUNT OF ITSELF -- what a person goes back to read, each
    # file its run's alone, since a run folder is one start.  SIESTA's own
    # history (`<SystemLabel>.MD.nc`, `.MD`, `.MDE`, `.ANI`) is no row:
    # SIESTA names it, and each run's is its own for the same reason
    # (`job-contracts.md` § 2.2).
    #
    # `.out` IS SIESTA'S (`model/parse.md` § 5.5).
    Artifact(ROLE_SIESTA_OUT, "the run's output as the engine printed it",
             engine="siesta", output="stdout",
             kind="result", writer="the run script, from the engine's "
                                   "stdout",
             door="parse.engines._run_ending.ending_of",
             stderr_apart=True,
             reads=(("ending", "molbuilder.parse.engines._run_ending:"
                               "siesta_reading"),
                    ("stops", "molbuilder.parse.engines.siesta_grammar:STOPS"),
                    ("live", "molbuilder.parse.engines.siesta_reader:"
                             "live_pass"),
                    ("loop_words", "molbuilder.parse.engines.tbtrans:"
                                   "loop_words"),
                    ("scf_rows", "molbuilder.parse.engines.siesta_grammar:"
                                 "SCF_ROWS"),
                    ("used_log", "molbuilder.parse.engines.siesta_fdflog:"
                                 "used_log_of"),
                    ("record", "molbuilder.parse.engines.siesta_record:"
                               "out_fields"))),
    Artifact(ROLE_PYSCF_STDOUT, "the same, for PySCF — it writes here and "
                                "not to .out",
             engine="pyscf", output="stdout",
             kind="result", writer="the run script, from the engine's "
                                   "stdout",
             door="parse.engines._run_ending.ending_of",
             reads=(("ending", "molbuilder.parse.engines._run_ending:"
                               "pyscf_reading"),)),
    # PYSCF'S OWN LOG (`mol.output`, `model/parse.md` § 5.5).
    Artifact(ROLE_PYSCF_LOG, "PySCF's own log", engine="pyscf",
             kind="result", writer="the PySCF deck",
             door="parse.dirs.record.run_record",
             reads=(("record", "molbuilder.parse.engines.pyscf:"
                               "record_fields"),)),
    # GEOMETRIC'S TRAJECTORY, `<label>_<stage>_geom_optim.xyz` -- the row the
    # deck's prefix for geomeTRIC is derived from (`pyscf.input.
    # geometric_prefix_expr`).  geomeTRIC writes no log file here: PySCF
    # hands it a logging setup that prints to the screen
    # (`pyscf/geomopt/log.ini`).
    Artifact(ROLE_GEOM_TRAJ, "geomeTRIC's trajectory: each accepted "
                                "step's geometry, multi-frame XYZ — the name "
                                "the deck's prefix for geomeTRIC is derived "
                                "from; geomeTRIC writes no log file of its "
                                "own here, PySCF handing it a logging setup "
                                "that prints to the screen",
             engine="pyscf", kind="result",
             writer="geomeTRIC, under the prefix the deck hands it",
             listed_by="engine_index"),
    # THE RUN'S ONE START: the run script opens its session log once the
    # launch-door gate lets the start through -- a help request, a dry run
    # and a refused start write none -- so its being there is the start,
    # what a second start in the folder refuses on (`project-layout.md`
    # § 1.6.1, `job-contracts.md` § 2.6).
    Artifact(ROLE_SESSION_LOG, "the run script's own session log — one per "
             "run",
             kind="record",
             writer="the run script, once the launch-door gate lets its "
                    "start through",
             door="wrapper_log.log_of_run"),
    # EVERY ENGINE'S, the monitor's two: the wrapper starts it from its shared
    # part (`run-reports.md` § 2.3).  The timing tee is cut from a
    # SIESTA-family ``.out`` (its row's ``cut_from``), and its engine is the
    # output's, worked out below -- so a PySCF rung's card never promises a
    # file no PySCF run writes.
    Artifact(ROLE_MONITOR_LOG, "the monitor's rolling status",
             kind="record", writer="the monitor",
             door="parse.instruments.monitor.MonitorLogFileParser"),
    Artifact(ROLE_UTIL_CSV, "processor and memory samples taken while it ran",
             kind="record", writer="the monitor",
             door="parse.instruments.util_csv.UtilCsvFileParser"),
    Artifact(ROLE_SCF_TIMING, "wall time per SCF iteration — on a "
                              "TranSIESTA device, both its SCF loops "
                              "(SIESTA's, then TranSIESTA's)",
             cut_from=ROLE_SIESTA_OUT, kind="record",
             writer="the run script's tee of the output's SCF lines",
             door="parse.instruments.scf_timing_rows.timing_of",
             reads=(("timing", "molbuilder.parse.instruments.scf_timing_rows:"
                               "scf_timing_metrics"),),
             # THE TEE OPENS IT AT THE FIRST SCF ROW (`runwrap._mb_scf_tee`):
             # a run that stops before one has none.
             only="the engine printed an SCF iteration"),
    # MOLBUILDER'S OWN READING LOG, named off the file read: `<base>.out`
    # gives `<base>.parse.log` (`parse/_log.py`).  Off unless asked.
    Artifact(ROLE_PARSE_LOG, "molbuilder's log of reading the run's output",
             kind="record",
             writer="molbuilder's parser, reading the output "
                    "(`parse._log.ParseLogger`)",
             only="`MOLBUILDER_PARSE_LOG` set, beside a file whose name the "
                  "catalogue reads back"),
    Artifact(ROLE_MOLWATCH_PARSE_LOG,
             "the same, for the trajectory log",
             kind="record",
             writer="molbuilder's parser, reading the trajectory log "
                    "(`parse._log.ParseLogger`)",
             only="`MOLBUILDER_PARSE_LOG` set"),
    # The summary `jobset summarize task` writes.  It is the deliverable of the ladder
    # (`engines/transport.md` § 2a.12: "the transmission stage's output ...
    # everything else in the tree exists to make it trustworthy"), written
    # once at the calculation root -- which is what `staged=False` says.
    Artifact(ROLE_TRANSPORT_RECORD, "the transport results, summarised by "
                                "`summarize task` from the transmission "
                                "points that ran",
             staged=False, calculation="transport", when="summarize",
             level="calculation", kind="result",
             writer="`jobset summarize task` (`transport.record.write_record`)",
             door="parse.sidecars.transport.TransportRecordFileParser"),
    # THE SAME RESULTS AS DATA (`engines/transport.md` § 2a.12, *the data
    # file*): frame x voltage x energy, raw beside derived, every variable
    # defined -- written beside the record by the same verb.
    Artifact(ROLE_TRANSPORT_DATA, "the transport results as data on their grid "
                              "-- every point's raw outputs beside what is "
                              "derived from them, a frame set's definition "
                              "beside every frame's values (NetCDF-4)",
             staged=False, calculation="transport", when="summarize",
             level="calculation", kind="result",
             writer="`jobset summarize task` "
                    "(`transport.datafile.write_data_file`)",
             door="transport.datafile.read_data_file",
             listed_by="user_index"),
    # A TRANSMISSION POINT'S DATA FILE, READ (`model/parse.md` § 5e): its
    # device DOS per orbital averaged over the k-points, kept beside it per
    # version of the file -- on the person's word, never as a side effect of
    # looking; one beside each channel's data file.
    Artifact(ROLE_TBT_DIGEST,
             "a transmission point's data file, read once: its device DOS "
             "per orbital averaged over the k-points, what the file states "
             "of its geometry and electrodes, and TBtrans's own k-averaged "
             "quantities -- every variable with its units (NetCDF-4)",
             fields=("tbt",), staged=False, calculation="transport",
             when="summarize", level="run", kind="derived",
             writer="the SIESTA family's parser "
                    "(`parse.engines.tbtrans_nc`), on the person's word: "
                    "the Results tab's DOS tab, `jobset summarize task`",
             door="parse.engines.tbtrans_nc.kept",
             only="a transmission point, its data file read",
             listed_by="engine_index"),
    Artifact(ROLE_TBT_DIGEST_CLAIM,
             "the claim of a data file's reading under way: who reads, "
             "since when, how far -- one reading of a file at a time",
             fields=("tbt",), staged=False, calculation="transport",
             when="summarize", level="transient", kind="transient",
             writer="the SIESTA family's parser, removed when its reading "
                    "ends", door="parse.engines.tbtrans_nc.kept_state",
             only="while its data file is read"),
    # A SIESTA VIBRATION'S DISPLACEMENT SWEEP, summarised (`engines/
    # vibration.md` § 5.9): written at the calculation root by `summarize task`
    # when the ladder holds two or more force-constant stages -- a record of
    # results that exist, each stage's own files staying in its runs.
    Artifact(ROLE_FC_SWEEP, "the force-constant stages compared, when "
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
    # THE CONCLUSION MARKER -- the wrapper's last act on its main line, or
    # its exit trap's when the script fails on its own first
    # (`project-layout.md` § 1.6.3).
    Artifact(ROLE_CONCLUDED, "the marker the run script writes when the job "
             "ends",
             kind="record",
             writer="the run script, its last act; or its exit trap, when "
                    "the script fails on its own before it",
             door="runrecord.ending"),
    # ---- written on the stem of the file a person reads ------------------
    Artifact(ROLE_RUNTIME_INFO, "what a file of the run says about the "
                                "run, as `molbuilder runtime-info` read it -- "
                                "one per run",
             kind="record",
             writer="`molbuilder runtime-info`, in the folder of the file it "
                    "reads",
             only="a person runs `molbuilder runtime-info`",
             listed_by="engine_index"),
    # THE LAYOUT'S NAME stays in the name: every prep log on disk carries
    # it, and the description states the same word (:data:`LAYOUT`).
    Artifact(ROLE_PIPELINE_LOG,
             "what each step of a prep received, decided and produced",
             fields=("engine",), when="prep", level="calculation",
             kind="record",
             writer="every prep, from either door, beside its "
                    "`STAGE-PLAN.md` (`pipeline_log.PipelineLog`)"),
    # ---- NOT named on the label: fixed names, and families of them --------
    # A FIXED NAME'S OWNER TAKES IT FROM HERE (`job-contracts.md` § 2.2): the
    # constants above are the one spelling.
    Artifact(name=TASK_FILE, what="the description: what varies, the stages, "
                                  "the layout, the structure",
             when="setup", level="calculation", kind="source",
             writer="`molbuilder describe` and the Transport tab's describe "
                    "door (`describe.write_description`); Task setup's Save; "
                    "`jobset migrate`",
             door="task.of"),
    Artifact(name=JOB_REQUEST_FILE,
             what="the job request: the queue, wall and memory asked, the "
                  "run cards, each stage's group, a benchmark's plan, the "
                  "reports",
             when="setup", level="calculation", kind="source",
             writer="`molbuilder describe` and the Transport tab's describe "
                    "door (`describe.write_description`); Task setup's Save; "
                    "`jobset migrate`",
             door="job_request.of"),
    Artifact(name=KEPT_FILE,
             what="a record as it was before `jobset migrate` rewrote it",
             when="setup", level="calculation", kind="record",
             writer="`jobset migrate`, beside the record it rewrote",
             only="a calculation migrated"),
    Artifact(name=TASK_HANDOVER_FILE,
             what="a hand-over waiting for its stages; Save removes it",
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
             door="jobset.model.JobSet.of"),
    Artifact(name=PREPARED_WITH_FILE,
             what="what each prepared stage was prepared with — its "
                  "settings, points and run card, merged per stage — which "
                  "`status` compares with its records now",
             when="prep", level="calculation", kind="record",
             writer="prep (`jobset.prepared_with.record`)",
             door="jobset.prepared_with.changes"),
    Artifact(name=PLAN_FILE, what="the plan in reading order",
             when="prep", level="calculation", kind="derived",
             writer="prep (`jobset.prep.prep_jobset`), whole at each prep"),
    Artifact(name=LEDGER_FILE, what="one line per decision of every verb",
             when="prep", level="calculation", kind="record",
             writer="every verb (`jobset.ledger.record`)"),
    Artifact(name=PSEUDO_FILE,
             what="a pseudopotential: the calculation's one copy in "
                  "`pseudos/`, and a real copy beside every deck — SIESTA "
                  "opens only its working directory",
             engine="siesta", when="prep", level="calculation", kind="input",
             writer="`molbuilder describe --psml-lib`, loose at the "
                    "calculation root; the first prep moves them into "
                    "`pseudos/` (`siesta.seam.pseudo_dir`) -- a transport "
                    "calculation's first prep copies those of the run it "
                    "cites, or of the calculation it extends "
                    "(`transport.rung.provide_pseudos`, "
                    "`transport.reuse.carry_pseudos`) -- and `materialize` "
                    "copies one beside each deck",
             door="pseudos.psml_sources"),
    Artifact(name=JUNCTION_FILE, what="the composed junction",
             calculation="transport", when="prep", level="calculation",
             kind="derived",
             writer="the prep that composes the junction, or records an "
                    "older one's order "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.recorded_junction",
             listed_by="engine_index"),
    Artifact(name=JUNCTION_SIDECAR_FILE,
             what="its cell, its region labels and the citation's record",
             calculation="transport", when="prep", level="calculation",
             kind="derived",
             writer="the prep that composes the junction, or records an "
                    "older one's order "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.recorded_junction",
             listed_by="engine_index"),
    Artifact(name=JUNCTION_CITED_FILE,
             what="the deck the junction was cited from, as it was",
             calculation="transport", when="prep", level="calculation",
             kind="derived",
             writer="the prep that composes the junction, or records an "
                    "older one's order "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.load_compose_record",
             only="a run is cited",
             listed_by="engine_index", counted_by=("siesta",)),
    Artifact(name=SLOT_PROVENANCE_FILE,
             what="where each part of the junction came from, with hashes",
             calculation="transport", when="prep", level="calculation",
             kind="record",
             writer="the prep that composes the junction, or records an "
                    "older one's order "
                    "(`transport.compose.write_compose_record`)",
             door="transport.compose.read_provenance"),
    Artifact(name=GITIGNORE_FILE,
             what="which files the saved states keep by content in "
                  "`.binsnapshots/` rather than in git, and the engines' "
                  "scratch, which they keep nowhere",
             when="setup", level="calculation", kind="record",
             writer="`checkpoint.Repo` -- `checkpoint init` and `save`, the "
                    "panel, and `save_before` before every prep that "
                    "writes, Save and a group's first launch",
             door="checkpoint.Repo"),
    Artifact(name=HISTORY_DIR + "/", what="the folder's saved states",
             when="setup", level="calculation", kind="record",
             writer="`checkpoint.Repo` -- `checkpoint init` and `save`, the "
                    "panel, and `save_before` before every prep that "
                    "writes, Save and a group's first launch",
             door="checkpoint.Repo"),
    Artifact(name=SNAPSHOT_DIR + "/",
             what="the big files of each saved state, by content",
             when="setup", level="calculation", kind="record",
             writer="`checkpoint.Repo` -- `checkpoint init` and `save`, the "
                    "panel, and `save_before` before every prep that "
                    "writes, Save and a group's first launch",
             door="checkpoint.Repo"),
    # THE COORDINATES OF ONE SIESTA CALL, read by the deck beside it
    # (`%block AtomicCoordinatesAndAtomicSpecies < coordinates.fdf`,
    # `parse.fdf.read_deck`; `engines/vibration.md` § 5.11 § 4).
    Artifact(name=COORDINATES_FILE,
             what="the coordinate rows of one SIESTA call, which the deck "
                  "beside it reads as its AtomicCoordinatesAndAtomicSpecies "
                  "block -- the structure's beside each phase's deck, a "
                  "call's own in its folder",
             when="prep", level="stage", kind="derived", engine="siesta",
             calculation="vibration", travels=True,
             writer="prep, beside each phase's deck "
                    "(`siesta.deck_lines.coordinates_text`); the run's own "
                    "python for each call (`spectra.siesta_vibration`, "
                    "the plan verb)",
             door="parse.fdf.read_deck"),
    # A RUN'S OWN RECORDS, in its folder: the launch record -- written at
    # launch -- the notification record, and the run it continues from.
    Artifact(name=LAUNCH_RECORD_FILE,
             what="the launch record: how, where and when the run was sent, "
                  "and what it continued from",
             when="launch", level="run", kind="record",
             writer="launch (`runrecord.write_launch`)",
             door="runrecord.launch_record"),
    # THE RUN'S NOTIFICATION RECORD -- every event sent for the run, when
    # (`run-reports.md` § 2.5, § 2.9; `engines/vibration.md` § 5.11 § 7a):
    # a run of points speaks as one run, its start, milestones, failed parts
    # and finish sent by its points' report steps, and each member monitor's
    # N-hour clock counts from the last event this record holds.
    Artifact(name=NOTIFIED_RECORD_FILE,
             what="the run's notification record: every event sent for the "
                  "run -- start, a step, the periodic report, its wall near, "
                  "a milestone, a failed part and its reason, finish -- and "
                  "when",
             when="run", level="run", kind="record",
             writer="`mb_monitor.pyz notify` (a run of points' start, "
                    "milestones, failed parts and finish, sent by its "
                    "points' report steps, `runrecord.note_notified`); a "
                    "member monitor, for every event it sends (`monitor`)",
             door="runrecord.read_notified"),
    Artifact(name=CONTINUED_FROM_FILE,
             what="the run this one continues from -- a stage's run, a run "
                  "of points, a point's start (a walked point's, a planned "
                  "point's), a taken-over point's computing run (one hop)",
             when="launch", level="run", kind="record",
             writer="launch, by its one hand-over (`materialize.hand_over`, "
                    "through `runrecord.write_continued_from`) -- with the "
                    "files it takes; a planner's own step for each point it "
                    "plans",
             door="runrecord.read_continued_from",
             only="it continues from an earlier run"),
    Artifact(name=CALCDIR_FILE,
             what="what this folder is in its calculation — a container, "
                  "a run, a run's scratch folder, or a transmission point's "
                  "shared energies — and where the calculation is",
             when="prep", level="stage", kind="record",
             writer="the code that makes the folder: prep and launch "
                    "(`materialize.open_run` -- a stage's run, a bias "
                    "point's, a benchmark trial's, and the containers above "
                    "each -- `naming.open_container`, a "
                    "submission's `launch/`, `pseudos/` and a point's "
                    "folder below its stage, and "
                    "`materialize.open_scratch`, a device run's `tsgf/`; "
                    "and `transport.reuse`, a transmission point's `shared/`)",
             door="calcdirs.where"),
    Artifact(name=MONITOR_BUNDLE,
             what="the monitor, and the readers it runs on — one file",
             when="prep", level="stage", kind="derived", travels=True,
             writer="prep, beside each run script "
                    "(`runwrap.write_run_wrapper`); a copy in each run"),
    Artifact(name=VIBRATION_BUNDLE,
             what="the SIESTA vibration's own steps -- each call's `plan` "
                  "and `record` and the run's `finish-run`: the coordinates, "
                  "the index, the modes from `.FC`, the result and the "
                  "optical calls' data file",
             engine="siesta", calculation="vibration",
             when="prep", level="stage", kind="derived",
             writer="prep, in each phase's folder beside its run script "
                    "(`runwrap.write_run_wrapper`); a copy in each call"),
    Artifact(name=PYSCF_BUNDLE,
             what="molbuilder's own code the PySCF script runs, which it "
                  "imports from here -- the core count, the progress-log "
                  "writer, the structure codec, the relaxation, a "
                  "vibration's rules",
             engine="pyscf",
             when="prep", level="stage", kind="derived", travels=True,
             writer="prep, beside the PySCF script "
                    "(`runwrap.write_run_wrapper`); a copy in each run"),
    Artifact(name=MAKOV_PAYNE_SCRIPT,
             what="the energy correction a charged, isolated deck asks a "
                  "person to run afterwards",
             engine="siesta", when="prep", level="stage", kind="derived",
             travels=True,
             writer="prep (`siesta.makov_payne.emit_correction_script`); a "
                    "copy in each run",
             only="a charged, isolated deck"),
    Artifact(name=GATHERED_FROM_FILE,
             what="what a rung took, from which upstream run",
             calculation="transport", when="launch", level="run",
             kind="record",
             writer="launch, by its one hand-over (`materialize.hand_over`, "
                    "through `runrecord.write_gathered_from`) -- a rung's "
                    "first launch or a `--gather` "
                    "(`transport.gather.transport_inputs`), or a launch again "
                    "keeping what its last run was given",
             door="runrecord.read_gathered_from"),
    Artifact(name=COPIED_FROM_FILE,
             what="the finished calculation's run, or point, this one is a "
                  "copy of, tree-relative",
             calculation="transport", when="prep", level="run",
             kind="record",
             writer="prep, copying a group in from the finished calculation "
                    "it reuses (`transport.reuse.copy_in`), before the run's "
                    "launch record; a rung of points' first launch, in each "
                    "point it takes over (`transport.reuse.take_into`) and each "
                    "point's `shared/`, written last (`transport.reuse."
                    "shared_into`) -- through `runrecord.write_copied_from`",
             door="runrecord.read_copied_from",
             only="a transport calculation that reuses a finished one"),
    Artifact(name=SCHEDULER_STDOUT,
             what="the scheduler's own stdout for the job",
             level="run", kind="record",
             writer="the scheduler, as the run's header asks (its kind's "
                    "`stdout`, `scheduler/kinds.toml`)",
             only="launched to a queue"),
    Artifact(name=SCHEDULER_STDERR,
             what="the scheduler's own stderr for the job",
             level="run", kind="record",
             writer="the scheduler, as the run's header asks (its kind's "
                    "`stderr`, `scheduler/kinds.toml`)",
             only="launched to a queue"),
    Artifact(name=RANK_LAUNCHER,
             what="the per-rank GPU launcher",
             level="run", kind="transient",
             writer="the run script, removed when it exits",
             only="a GPU run with ranks"),
    Artifact(name=BENCH_RESULT_FILE,
             what="every trial's timing and the winner — the benchmark's "
                  "archival trace",
             when="summarize", level="bench", kind="record",
             writer="`jobset summarize` (`run_summarize_jobset`)"),
    Artifact(name=GROUP_RUN_SCRIPT,
             what="a launch group's sequencer: a benchmark's trials, a "
                  "run of points' walk (a vibration's calls, a bias "
                  "sweep's points), a task's stages sharing one job, or a "
                  "task's groups launched one after another in one job, "
                  "in order",
             when="launch", level="launch", kind="derived",
             writer="launch (`jobset/submit.py`), written again at each "
                    "launch"),
    Artifact(name=GROUP_HEADER,
             what="its queue header — a task's group's, a run of points' "
                  "walk's, a benchmark's shelf's, a sequence of groups'",
             when="prep", level="launch", kind="derived",
             writer="prep (`prep.walk_header_of`), written again at each "
                    "prep; launch sends it",
             only="a machine whose scheduler has a queue, no "
                  "`--no-sbatch`"),
    # THE SCHEDULER'S OWN OUTPUT for a job sent from `launch/`: the header
    # there names its files in the same folder (`runwrap.render_sbatch`,
    # ``where``; `scheduler.md` § 6a).
    Artifact(name=SCHEDULER_STDOUT,
             what="the scheduler's own stdout for the job",
             level="launch", kind="record",
             writer="the scheduler, as the group's header asks (its kind's "
                    "`stdout`, `scheduler/kinds.toml`)",
             only="launched to a queue"),
    Artifact(name=SCHEDULER_STDERR,
             what="the scheduler's own stderr for the job",
             level="launch", kind="record",
             writer="the scheduler, as the group's header asks (its kind's "
                    "`stderr`, `scheduler/kinds.toml`)",
             only="launched to a queue"),
    Artifact(name=GROUP_LOG, what="every member's output, in order",
             when="run", level="launch", kind="record",
             writer="the group's sequencer, as it runs"),
    Artifact(name=GROUP_LAUNCH_RECORD,
             what="a sequence of groups' launch record: its groups and "
                  "their stages, how its one job was sent, its id -- what "
                  "each run it sends records of that job",
             when="launch", level="launch", kind="record",
             writer="launch (`jobset/sequence.py`), before the job goes; "
                    "its id once the scheduler gives it",
             door="jobset.sequence.read_record"),
    Artifact(name="{target}.{random}.tmp",
             what="a file being written; it replaces its target when the "
                  "write ends",
             when="run", level="transient", kind="transient",
             writer="`persist` and the codec, writing whole or not at all"),
)

# A ROW CUT FROM AN OUTPUT IS THAT OUTPUT'S ENGINE'S: said once, by the
# output it names (`cut_from`), never typed beside it.
WRITTEN = tuple(
    replace(a, engine=next(b.engine for b in WRITTEN if b.role == a.cut_from))
    if a.cut_from else a for a in WRITTEN)

#: THE ENGINE A PREP RENDERED FOR, as a name carries it: one of the engines
#: the rows above name (:func:`engines` -- the list the engine table checks
#: itself against), so a new engine is its rows, never a second list here.
_ENGINES: "tuple[str, ...]" = tuple(sorted({a.engine for a in WRITTEN
                                             if a.engine}))
FIELDS["engine"] = Field(
    what="the engine a prep rendered for",
    shape="|".join(re.escape(e) for e in _ENGINES),
    example=_ENGINES[0])

#: The rows NAMED ON THE LABEL -- the role rows every reader of the run-file
#: grammar walks; the fixed names are the rest of :data:`WRITTEN`.
ON_THE_LABEL: "tuple[Artifact, ...]" = tuple(a for a in WRITTEN if a.role)


# --------------------------------------------------------------------- #
#  THE CODE THAT READS A FILE OF A ROLE -- the rows' ``reads``            #
#  (`job-contracts.md` § 2.2; `model/parse.md` § 5.5).                    #
# --------------------------------------------------------------------- #

#: THE QUESTIONS A ROW'S READERS ANSWER, and what each reader is:
#: ``ending`` how the run ended, ``(path, stderr=None) -> (RunEnding,
#: final)``; ``stops`` its family's stops, ``(wall causes, {cause: advice},
#: {cause: words})``; ``live`` its reading pass for the monitor,
#: ``(stage) -> pass``; ``loop_words`` its program's loop in words;
#: ``scf_rows`` how it prints its SCF rows, ``(row ERE, prefix ERE)``;
#: ``timing`` the timing figures in its text, ``(text) -> dict``;
#: ``used_log`` the engine's log of what it read, ``(folder, head) -> path``;
#: ``record`` the run record's fields it answers, ``(RunFiles) -> dict``;
#: ``contract`` a deck's recorded settings, ``(deck path) -> dict``.
READ_QUESTIONS: "tuple[str, ...]" = (
    "ending", "stops", "live", "loop_words", "scf_rows", "timing",
    "used_log", "record", "contract")
#: THE QUESTIONS A JOB ASKS, beside it: their readers' modules travel in its
#: bundle by their rows (`runwrap.monitor_companions`).
READ_BESIDE_A_JOB: "tuple[str, ...]" = (
    "ending", "stops", "live", "loop_words", "timing")


def _asked(question: str) -> str:
    """``question``, one of :data:`READ_QUESTIONS` -- else refused."""
    if question not in READ_QUESTIONS:
        raise RunFileError(f"{question!r} is no question a row's readers "
                           f"answer ({', '.join(READ_QUESTIONS)}).")
    return question


def reads_of(role: str, question: str) -> Optional[str]:
    """THE READER A FILE OF ``role`` GETS for ``question``
    (:data:`READ_QUESTIONS`), ``module:NAME`` -- or ``None``, where its
    row names none (a run output whose family keeps no stops); a role no
    row declares is refused (:func:`declared`).  Resolved by
    `runpoints.part`, on the machine and beside a job alike."""
    _asked(question)
    want = canonical_role(declared(role, door="reads_of"))[0]
    for a in ON_THE_LABEL:
        if a.role == want:
            return dict(a.reads).get(question)
    return None


def reader(role: str, question: str) -> Any:
    """THE READER ITSELF that :func:`reads_of` names, resolved where this
    module was imported from -- the package on this machine, the module the
    bundle holds beside a job (`runpoints.part`, the one resolver of a part
    a record names) -- or ``None`` where the row names none.  Every reader
    of an output asks this, never a table of its own (`model/parse.md`
    § 5.5)."""
    ref = reads_of(role, question)
    if ref is None:
        return None
    try:                                    # inside molbuilder
        from .runpoints import part
    except ImportError:                     # beside a job
        from runpoints import part  # type: ignore[no-redef]
    try:
        return part(ref)
    except ValueError as exc:
        raise RunFileError(f"runfiles.WRITTEN row {role!r}: its "
                           f"{question!r} reader {exc}") from exc


def stderr_apart(role: str) -> bool:
    """Whether the run output in ``role`` leaves its engine's stderr apart
    (:attr:`Artifact.stderr_apart`) -- the run script asks it to route the
    stream, and the ending reader to read the session log beside it."""
    want = canonical_role(declared(role, door="stderr_apart"))[0]
    return any(a.role == want and a.stderr_apart for a in ON_THE_LABEL)


def read_roles(question: str) -> "tuple[str, ...]":
    """Every role whose row names a reader for ``question``, in the
    catalogue's order."""
    return tuple(a.role for a in ON_THE_LABEL
                 if _asked(question) in dict(a.reads))


def _check_reads(rows) -> None:
    """Refuse a row whose readers do not say what the catalogue reads it by
    -- at import, so the gap stops the first thing that loads this module,
    not a job asking how its run ended: every run output names its
    ``ending``; a row names only :data:`READ_QUESTIONS`, once each, as
    ``molbuilder.<module>:NAME``; only a run output names ``ending`` or
    ``stops`` or leaves its stderr apart; only a row named on the label
    names readers, which are asked by role."""
    for a in rows:
        asked = [q for q, _ref in a.reads]
        bad = sorted(set(asked) - set(READ_QUESTIONS))
        twice = sorted({q for q in asked if asked.count(q) > 1})
        refs = [ref for _q, ref in a.reads
                if not (ref.startswith("molbuilder.") and ":" in ref)]
        output_only = [q for q in asked if q in ("ending", "stops")] + (
            ["stderr_apart"] if a.stderr_apart else [])
        wrong = (bad and f"names no question of READ_QUESTIONS: {bad}"
                 or twice and f"names a question twice: {twice}"
                 or refs and f"names readers that are no "
                             f"`molbuilder.<module>:NAME`: {refs}"
                 or a.output and "ending" not in asked
                 and "is a run output and names no `ending` reader"
                 or not a.output and output_only
                 and f"is no run output but names {output_only}"
                 or a.reads and not a.role
                 and "names readers but no role to ask them by")
        if wrong:
            raise RunFileError(f"runfiles.WRITTEN row {a.role or a.name!r} "
                               f"{wrong}.")


_check_reads(WRITTEN)


# --------------------------------------------------------------------- #
#  THE ENGINES' FILES MOLBUILDER NAMES -- the second table               #
#  (`job-contracts.md` § 2.2; `project-layout.md` § 5.4).                #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class EngineFile:
    """One file an engine writes that molbuilder composes, reads or
    carries, or explains by name to a person reading a run's folder --
    never complete: an engine's output is its version's
    and its options' (`job-contracts.md` § 2.2), so *did the engine leave
    this?* is still answered by subtraction from :data:`WRITTEN`.

    ``role`` is its role on the label, as :func:`compose` takes it, or
    ``name`` its whole name (SIESTA's ``MESSAGES``, ``fdf.{stamp}.log``) --
    a row has one of the two; ``staged`` is ``True`` for a file named under
    a stage's stem (geomeTRIC's scratch); ``door`` the reader of ours, a
    dotted path under ``molbuilder``, ``""`` when none reads it.  Which of
    them carry, and which are scratch, is the engine's restart-file list's
    (`warmfiles`), checked against this table when the shipped list loads.

    **Why `parse` needs the underscore ones.** A stage name may contain
    ``_`` (`02_electrode_L`) and so may a role (`_optimized.xyz`), so the
    separator cannot say where one ends and the other begins; knowing the
    roles decides it, these beside :data:`WRITTEN`'s."""
    engine: str
    role: str = ""
    name: str = ""
    what: str = ""
    door: str = ""
    staged: bool = False


ENGINE_FILES: "tuple[EngineFile, ...]" = (
    # ---- SIESTA: what a run starts from, and what molbuilder reads ------
    EngineFile("siesta", SIESTA_XV,
               what="the geometry and velocities SIESTA saves at every step "
                    "-- what a continued run starts from",
               door="parse.coords.siesta_xv.SiestaXVFileParser"),
    EngineFile("siesta", SIESTA_DM,
               what="the density matrix -- a continued run's start"),
    EngineFile("siesta", SIESTA_CG,
               what="the conjugate-gradient optimizer's history"),
    EngineFile("siesta", ".xyz",
               what="the geometry, as SIESTA writes it for a viewer "
                    "(`WriteCoorXmol`), read as any structure file with the "
                    "box from the run's own deck",
               door="workingcopy_structure.StructureCodec.load"),
    EngineFile("siesta", name=SIESTA_FDF_LOG,
               what="every setting SIESTA read, and the value it used",
               door="parse.engines.siesta_fdflog.SiestaFdfLogParser"),
    EngineFile("siesta", SIESTA_ORB_INDX,
               what="the orbitals each atom carries -- what a transmission "
                    "point's DOS per orbital is grouped by, at the device "
                    "point the transmission gathered",
               door="parse.engines.tbtrans_nc.read_orbitals"),
    EngineFile("siesta", SIESTA_FC,
               what="a force-constant run's forces at every displacement",
               door="parse.engines.siesta_fc.read_fc"),
    EngineFile("siesta", ".FCC",
               what="the same, for the cell's displacements"),
    EngineFile("siesta", SIESTA_EPSIMG,
               what="an optical call's dielectric function",
               door="parse.engines.siesta_epsimg.SiestaEpsimgFileParser"),
    EngineFile("siesta", ".contrib.EPSIMG",
               what="an optical call's dielectric function by transition -- "
                    "the engine's scratch"),
    # ---- SIESTA: the run's history, its own -----------------------------
    EngineFile("siesta", SIESTA_MD_NC, what="the run's steps, as NetCDF",
               door="parse.engines.siesta_mdnc.SiestaMdNcFileParser"),
    EngineFile("siesta", SIESTA_MD, what="the run's steps"),
    EngineFile("siesta", SIESTA_MDE, what="the run's steps' energies"),
    EngineFile("siesta", SIESTA_ANI, what="the run's steps, as an animation"),
    # ---- SIESTA: what it reads when present, under the same label -------
    EngineFile("siesta", ".LWF", what="localized Wannier functions"),
    EngineFile("siesta", ".ZM", what="the Z-matrix"),
    EngineFile("siesta", ".BONDS", what="the bonds SIESTA found"),
    EngineFile("siesta", ".PARTIAL", what="a partial charge density"),
    EngineFile("siesta", ".EIG", what="the eigenvalues"),
    EngineFile("siesta", SIESTA_HSX, what="the Hamiltonian and overlap"),
    EngineFile("siesta", ".WFSX", what="the wavefunctions"),
    EngineFile("siesta", ".STRUCT_NEXT_ITER",
               what="the next step's structure"),
    # ---- TranSIESTA and TBtrans ----------------------------------------
    EngineFile("siesta", SIESTA_TSHS,
               what="an electrode's Hamiltonian and overlap -- gathered into "
                    "the rungs after it"),
    EngineFile("siesta", SIESTA_TS_HSX,
               what="the device's Hamiltonian and overlap -- read by TBtrans "
                    "through the deck's `TBT.HS` line"),
    EngineFile("siesta", SIESTA_TSDE,
               what="the device's NEGF density -- handed along a bias sweep"),
    EngineFile("siesta", SIESTA_TSGF,
               what="a lead's surface Green's function at a voltage -- the "
                    "engine's scratch, in the device run's `tsgf/`, its name "
                    "the deck's (`transport.transiesta.lead_gf_file`)"),
    EngineFile("siesta", TBTRANS_NC,
               what="a transmission point's data file, per spin channel",
               door="parse.engines.tbtrans_nc.kept"),
    EngineFile("siesta", TBTRANS_CC,
               what="the energy contour TBtrans integrates on "
                    "(`m_tbt_contour.F90`)"),
    EngineFile("siesta", TBTRANS_DOS,
               what="the device's DOS at each k-point, per spin channel "
                    "(`m_tbt_save.F90`)"),
    EngineFile("siesta", TBTRANS_AVTRANS,
               what="its transmission, averaged over the k-points",
               door="parse.engines.tbtrans.read_averaged"),
    EngineFile("siesta", TBTRANS_AVTEIG,
               what="its eigenchannels, averaged over the k-points",
               door="parse.engines.tbtrans.read_averaged"),
    EngineFile("siesta", TBTRANS_AVDOS,
               what="the device's DOS, averaged over the k-points",
               door="parse.engines.tbtrans.read_averaged"),
    EngineFile("siesta", TBTRANS_AVADOS,
               what="a lead's spectral DOS, averaged over the k-points",
               door="parse.engines.tbtrans.read_averaged"),
    EngineFile("siesta", TBTRANS_AVBDOS,
               what="a lead's bulk DOS, averaged over the k-points",
               door="parse.engines.tbtrans.read_averaged"),
    # ---- SIESTA: read by nothing of ours --------------------------------
    EngineFile("siesta", name="0_NORMAL_EXIT",
               what="SIESTA's own end mark -- the engine ended, not the job: "
                    "the run's end is the conclusion marker's "
                    "(`runrecord.ending`)"),
    EngineFile("siesta", name="MESSAGES", what="SIESTA's warnings"),
    EngineFile("siesta", name="CLOCK", what="SIESTA's own timings"),
    EngineFile("siesta", name="FORCE_STRESS",
               what="the last step's forces and stress"),
    EngineFile("siesta", name="BASIS_ENTHALPY", what="the basis enthalpy"),
    EngineFile("siesta", name="BASIS_HARRIS_ENTHALPY",
               what="the basis Harris enthalpy"),
    EngineFile("siesta", name="OUTVARS.yml", what="SIESTA's final values"),
    EngineFile("siesta", name="PARALLEL_DIST",
               what="how the work was spread over the ranks"),
    EngineFile("siesta", name="NON_TRIMMED_KP_LIST",
               what="every k-point before symmetry trimmed them"),
    EngineFile("siesta", name="{species}.ion",
               what="a species' basis, as SIESTA built it"),
    EngineFile("siesta", name="{species}.ion.nc", what="the same, as NetCDF"),
    EngineFile("siesta", name="{species}.ion.xml", what="the same, as XML"),
    EngineFile("siesta", ".alloc", what="SIESTA's memory use"),
    EngineFile("siesta", ".bib", what="what to cite"),
    EngineFile("siesta", SIESTA_BASIS_ENTHALPY,
               what="the basis enthalpy, on the label"),
    EngineFile("siesta", ".BONDS_FINAL", what="the last step's bonds"),
    EngineFile("siesta", ".FA", what="the last step's forces"),
    EngineFile("siesta", ".FAC", what="the constrained forces"),
    EngineFile("siesta", ".KP", what="the k-points"),
    EngineFile("siesta", ".MD_CAR", what="the steps, in VASP's format"),
    EngineFile("siesta", ".STRUCT_OUT", what="the last step's structure"),
    # ---- PySCF and geomeTRIC --------------------------------------------
    EngineFile("pyscf", PYSCF_CHK,
               what="the checkpoint: the converged density and the molecule "
                    "-- read back by the deck itself"),
    EngineFile("pyscf", PYSCF_OPTIMIZED,
               what="a relaxation's final geometry -- what a continued run "
                    "starts from",
               door="parse.coords.pyscf_geom.PySCFGeomFileParser"),
    EngineFile("pyscf", PYSCF_CUBE,
               what="a volume grid -- a density or an orbital -- PySCF's "
                    "`cubegen` writes when a script asks; molbuilder's "
                    "decks ask none"),
    EngineFile("pyscf", GEOMETRIC_SCRATCH, staged=True,
               what="geomeTRIC's scratch folder, under the prefix the deck "
                    "hands it -- made empty, never read back"),
)


def _check_engine_files(rows) -> None:
    """Refuse a row the two tables cannot both hold: an engine no row of
    :data:`WRITTEN` names, a row with both or neither of ``role`` and
    ``name``, a role without its separator, a name either table holds
    twice -- the two tables never overlap (`job-contracts.md` § 2.2)."""
    ours = {a.role or a.name for a in WRITTEN}
    seen: "set[str]" = set()
    for e in rows:
        key = e.role or e.name
        wrong = (e.engine not in _ENGINES
                 and f"names the engine {e.engine!r}, which no row of "
                     f"WRITTEN names ({', '.join(_ENGINES)})"
                 or bool(e.role) == bool(e.name)
                 and "states both a role and a name, or neither"
                 or e.role and e.role[0] not in "._"
                 and "has a role that does not begin with '.' or '_'"
                 or key in ours and "is a file of ours too (WRITTEN)"
                 or key in seen and "is in the table twice")
        if wrong:
            raise RunFileError(f"runfiles.ENGINE_FILES row {key!r} {wrong}.")
        seen.add(key)


_check_engine_files(ENGINE_FILES)


def engine_roles(engine: str) -> "tuple[str, ...]":
    """The roles of ``engine``'s rows of :data:`ENGINE_FILES`, in its
    order -- a templated one as its template."""
    return tuple(e.role for e in ENGINE_FILES
                 if e.engine == engine and e.role)


def names_made(template: str) -> "tuple[str, ...]":
    """EVERY NAME ``template`` MAKES FOR A RUN OF OURS -- a role or a fixed
    name, each field at every value molbuilder's runs give it
    (:attr:`Field.values`), else at its example."""
    import itertools
    names = _fields_in(template)
    return tuple(
        _FIELD_RE.sub(lambda m: dict(zip(names, combo))[m.group(1)],
                      template)
        for combo in itertools.product(*[
            FIELDS[f].values or (FIELDS[f].example,) for f in names]))



@functools.lru_cache(maxsize=1)
def _declared_roles() -> "frozenset[str]":
    """Every role either table declares."""
    return frozenset(roles())


@functools.lru_cache(maxsize=1)
def _templated_roles() -> "tuple[str, ...]":
    """Every templated role, in the catalogue's order."""
    return tuple(r for r in roles() if _FIELD_RE.search(r))


def declared(role: str, roles: "tuple[str, ...]" = (), *,
             door: str = "a name") -> str:
    """``role``, when :data:`WRITTEN` or :data:`ENGINE_FILES` declares it --
    itself, or a concrete name of a templated one (``.TBT_UP.nc``) -- or
    ``roles`` holds it; else refused, naming the ``door`` that asked.  THE
    ONE CHECK every door that takes a role asks (`job-contracts.md`
    § 2.2a)."""
    if role in _declared_roles() or role in roles:
        return role
    if any(_role_pattern(t).fullmatch(role) for t in _templated_roles()):
        return role
    raise RunFileError(
        f"{door}({role!r}): no role molbuilder declares -- neither a row of "
        f"`runfiles.WRITTEN` nor an engine file it names "
        f"(`runfiles.ENGINE_FILES`)"
        + (f", nor the vocabulary handed in ({', '.join(roles)})"
           if roles else "") + ".")


def _row_named(name: str) -> "Optional[Artifact]":
    """The catalogue row of a file of this name: the row of its whole name,
    else of the longest role its name ends with; ``None`` for a file the
    catalogue does not list."""
    for a in WRITTEN:
        if a.name and not a.role and a.name == name:
            return a
    found = [a for a in ON_THE_LABEL
             if (_role_pattern(a.role).search(name) if a.fields
                 else name.endswith(a.role))]
    return max(found, key=lambda a: len(a.role)) if found else None


def listed_by_of(name: str) -> str:
    """WHICH INDEX A FILE OF THIS NAME IS LISTED BY -- its catalogue row's
    (:attr:`Artifact.listed_by`, `model/overview.md` § 2.2b, F3); ``""`` for
    a row that says nothing, and for a file the catalogue does not list."""
    row = _row_named(name)
    return row.listed_by if row is not None else ""


def counted_by_of(name: str) -> "tuple[str, ...]":
    """WHOSE NUMBERINGS A FILE OF THIS NAME'S OWN TEXT COUNTS ITS ATOMS BY --
    an engine's input's (:attr:`Artifact.counted_by`, F5); empty for any
    other file."""
    row = _row_named(name)
    return row.counted_by if row is not None else ()

def _staged(role: str, door: str) -> bool:
    """Whether a file of ``role`` -- a role of either table, or a concrete
    name of a templated one (:func:`declared`) -- is named on its stage's
    stem: its row's ``staged``."""
    want = canonical_role(declared(role, door=door))[0]
    for row in (*ON_THE_LABEL, *ENGINE_FILES):
        if row.role == want:
            return row.staged
    raise RunFileError(f"{door}({role!r}): no row names it on the label.")


def _template(a: "Artifact", label: str, stage: Optional[str]) -> str:
    """Row ``a``'s name on ``label`` with each of its fields as
    ``{field}``.  ``stage`` names a staged row's stage; an unstaged row is
    named on the label alone.  Read by :func:`manifest`."""
    return compose(label, a.role, stage if a.staged else None,
                   **{f: "{" + f + "}" for f in _fields_in(a.role)})


@dataclass(frozen=True)
class RunNames:
    """THE NAMES OF ONE STAGE'S FILES -- the catalogue (:data:`WRITTEN`) read
    for one stage of one calculation (`job-contracts.md` § 2.2a).  Every
    writer of a stage's files, and every script rendered to write them,
    names them through this: prep and launch, the run script and the deck,
    the readers -- so no two can spell one name two ways.  A run's records
    (`run.json`, `notified.json`, `.continued-from`) are fixed names, the
    run's folder naming the run.  The row decides whether a name carries
    the stage token (``staged``); this object only supplies the stage it
    names.

    The stage is required: every calculation has at least one, and every
    file of a run carries its token (`stages.md` § 6.5)."""
    label: str
    stage: str

    def __post_init__(self):
        if self.stage is None:
            raise RunFileError(
                f"{self.label!r}: a run's files are named by its stage, and "
                f"no stage was given -- every calculation has at least one"
                f".")
        stem(self.label, self.stage)    # a malformed label or token, refused

    @property
    def stem(self) -> str:
        """``<label>_<stage>`` -- what every staged role attaches to."""
        return stem(self.label, self.stage)

    def tail(self, role: str, **fields) -> str:
        """:meth:`name` after the label -- for a script that names its files
        from its own ``JOB`` / ``SystemLabel`` (:func:`tail`)."""
        return self.name(role, **fields)[len(self.label):]

    def name(self, role: str, **fields) -> str:
        """``role``'s name in this stage's run -- a role of :data:`WRITTEN`
        or of :data:`ENGINE_FILES` (:func:`declared`), on the stage's stem
        or on the label alone as its row says (``staged``); ``fields`` fill
        a templated role."""
        staged = _staged(role, "RunNames.name")
        return compose(self.label, role, self.stage if staged else None,
                       **fields)

    @property
    def run_script(self) -> str:
        """The stage's run script -- the one file a stage's and a launch
        group's names both name (:attr:`GroupNames.run_script`), so a header
        is rendered alike for either (`runwrap.render_sbatch`)."""
        return self.name(ROLE_RUN_SCRIPT)

    @property
    def header(self) -> str:
        """The stage's queue header (:attr:`GroupNames.header`)."""
        return self.name(ROLE_HEADER)

    def restaged(self, name: str, source: "RunNames") -> str:
        """``name``, a file of ``source``'s stage, named for this stage --
        its role and fields kept: how a point taken over from another
        stage's run is named in this one, so every reader of this stage's
        files finds it (`execution/runs-of-points.md` § 3, op. 3).  A file
        on the label alone, or not of ``source``'s stage, as it is."""
        f = parse(name, source.label)
        if f is None or f.stage != source.stage:
            return name
        return replace(f, label=self.label, stage=self.stage).name



@dataclass(frozen=True)
class GroupNames:
    """A LAUNCH GROUP'S FILES -- a benchmark's walk, a bias sweep's chain, a
    task's group, a sequence of groups: the catalogue's launch-level rows
    ``{group}.run.sh``, ``{group}.sbatch``, ``{group}.log`` and a
    sequence's ``{group}.launch.json``, filled with the group's name
    (`project-layout.md` § 5).  The questions :class:`RunNames` answers too
    -- :attr:`stem`, :attr:`run_script`, :attr:`header` -- so a header is
    rendered alike for a stage's run and a group's
    (`runwrap.render_sbatch`)."""
    group: str

    def __post_init__(self):
        stem(self.group)                # one [A-Za-z0-9_-]+ token, refused

    # WHAT EACH WALK IS CALLED -- the one composer of a group's name, for
    # each kind of walk (`project-layout.md` § 5.5).
    @classmethod
    def of_stages(cls, label: str, tokens) -> "GroupNames":
        """A task's group -- its stages sharing one job: the calculation's
        label and every member's token, in order --
        ``chain-t_group-01_seed-02_electrode_L-03_electrode_R``."""
        return cls(f"{label}_group-{'-'.join(tokens)}")

    @classmethod
    def of_sequence(cls, label: str, tokens) -> "GroupNames":
        """A task's groups launched one after another in one job (a
        sequence, `job-system.md` § 6.0): the calculation's label and every
        member's token, in order --
        ``chain-t_sequence-01_seed-02_electrode_L-03_electrode_R-04_device``."""
        return cls(f"{label}_sequence-{'-'.join(tokens)}")

    @classmethod
    def of_points(cls, stem_: str) -> "GroupNames":
        """A run of points' walk -- the stage's stem, ``<stem>-sweep``."""
        return cls(f"{stem_}-sweep")

    @classmethod
    def of_trials(cls, side: Optional[str] = None,
                  cell: Optional[str] = None) -> "GroupNames":
        """A benchmark's walk: its trials run here, ``bench-group``; a shelf
        sent to a queue, named in full by its side and its machine cell,
        ``bench-group-cpu-G0N4C1`` (`generator.md` § 4.3a)."""
        return cls("bench-group" + (f"-{side}-{cell}" if side else ""))

    @property
    def stem(self) -> str:
        return self.group

    @property
    def run_script(self) -> str:
        """The group's sequencer (:data:`GROUP_RUN_SCRIPT`) -- what a stage's
        names call its run script (:attr:`RunNames.run_script`)."""
        return self.name(GROUP_RUN_SCRIPT)

    @property
    def header(self) -> str:
        """The group's queue header (:data:`GROUP_HEADER`)."""
        return self.name(GROUP_HEADER)

    def name(self, row: str) -> str:
        """The group's file of the launch-level row ``row``
        (:data:`GROUP_RUN_SCRIPT`, :data:`GROUP_HEADER`, :data:`GROUP_LOG`,
        :data:`GROUP_LAUNCH_RECORD`), composed by :func:`fixed_name`: a
        launch-level row named by the group alone."""
        if not any(a.level == "launch" and a.name == row
                   and _fields_in(row) == ("group",) for a in WRITTEN):
            raise RunFileError(
                f"{row!r} is no launch-level row of runfiles.WRITTEN named "
                f"by its group -- a group's files are the catalogue's")
        return fixed_name(row, group=self.group)

    # WHERE THE GROUP'S FILES LIE -- the launch folder of the folder the group
    # is launched from (a calculation's root for a task's group or sequence,
    # a stage's folder for a run of points' walk, a benchmark's container for
    # its trials; `project-layout.md` § 5.5), composed here alone, as
    # `calcdirs.RunPlace` composes a run's folders.
    def path(self, container, row: Optional[str] = None) -> Path:
        """The group's launch folder in ``container`` -- its ``launch/`` --
        or, with ``row``, the group's file of that row in it."""
        folder = Path(container) / LAUNCH_DIR
        return folder / self.name(row) if row is not None else folder

    def walked(self, row: Optional[str] = None) -> str:
        """The group's file of ``row`` as named from its container -- by the
        walk, the scheduler's line and every sentence that names it:
        ``launch/<group><tail>`` -- or, with no ``row``, its launch folder,
        ``launch``."""
        return (f"{LAUNCH_DIR}/{self.name(row)}" if row is not None
                else LAUNCH_DIR)


def roles(*, engines: bool = True) -> "tuple[str, ...]":
    """Every role this module knows -- the catalogue, read as a vocabulary.

    :data:`WRITTEN`'s roles, then the engines' files molbuilder names
    (:data:`ENGINE_FILES`) unless ``engines=False``: what :func:`declared`
    checks every role against, which makes an undeclared role a catalogue
    question rather than something a caller can slip past by spelling it
    (`plans/plan.md` § 5l.2: extensibility is a catalogue row, never a new
    function).

    Ordered and de-duplicated, so it is stable to compare against.
    """
    out = [a.role for a in ON_THE_LABEL]
    if engines:
        out += [e.role for e in ENGINE_FILES if e.role]
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
      wrong file to open.  A row of kind ``record`` naming the calculation
      -- a run of points' index -- is the run's own bookkeeping, listed as
      an aspect of the run and never what a person opens first.
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
             if calculation and written_by(a, calculation)
             and a.kind != "record"]
    spoken = [a.role for a in ON_THE_LABEL
              if a.output == "stdout" and a.role not in named]
    live = [a.role for a in ON_THE_LABEL
            if a.output == "progress" and a.role not in named]
    return tuple(named + spoken + live)


def progress_roles() -> "tuple[str, ...]":
    """THE PROGRESS CHANNEL'S roles -- the logs a run writes as it goes,
    seeded by prep beside the deck (the catalogue's ``output`` column)."""
    return tuple(a.role for a in WRITTEN if a.output == "progress")


def deck_roles(engine: Optional[str] = None) -> "tuple[str, ...]":
    """The DECKS -- what a stage's engine reads: the stage's own file, derived
    at prep, that one engine owns (`.fdf`, `.py`) -- for ``engine``, or every
    engine's.  Read off the rows, so a new engine's deck is its row."""
    return tuple(a.role for a in ON_THE_LABEL
                 if a.level == "stage" and a.kind == "derived" and a.engine
                 and (engine is None or a.engine == engine))


def deck_role(engine: str) -> str:
    """THE deck ``engine`` reads -- its one row (:func:`deck_roles`)."""
    roles = deck_roles(engine)
    if len(roles) != 1:
        raise RunFileError(f"{engine!r}: {len(roles)} deck rows in the "
                           f"catalogue, not one ({', '.join(roles)}).")
    return roles[0]


def prepared_roles() -> "tuple[str, ...]":
    """What every prepared run carries that says what it is -- each engine's
    deck (:func:`deck_roles`) and the run script beside it, both stating
    their engine and molbuilder's provenance: what marks a folder as
    somebody's working directory, and what a reader asks a folder's
    engine."""
    return (*deck_roles(), ROLE_RUN_SCRIPT)


def engine_of_role(role: str) -> Optional[str]:
    """Which engine writes this role -- its catalogue row's ``engine`` --
    or None when any of them may.

    The mirror of :func:`stdout_roles`, asked by a READER holding a file: a
    writer is handed its engine by the job it writes for, never reads it
    off a name (`model/parse.md` § 5.5, R-RO1).
    """
    for a in WRITTEN:
        if a.role == role:
            return a.engine
    return None


def cut_from_output(role: str) -> Optional[str]:
    """The role of the file the run cuts from an output in ``role`` -- the
    SCF-timing tee's from a SIESTA-family ``.out`` -- or ``None``: the run
    script cuts nothing from it (the row's :attr:`Artifact.cut_from`)."""
    return next((a.role for a in WRITTEN if a.cut_from == role), None)


def engines() -> "tuple[str, ...]":
    """Every engine the catalogue names, sorted.

    The catalogue is where an engine becomes known to the run-file layer (a
    row with ``engine=``), so this is what a caller iterating engines asks
    rather than restating the pair -- § 5.5's *adding an engine is two edits*.
    """
    return _ENGINES


def manifest(label: str, stage: Optional[str], *,
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

    A file written only sometimes carries its condition, ``only``; each
    row says the folder it lands in, ``level``, and in words, ``where``
    (:data:`LEVEL_WHERE`) -- a rung's pipeline log is the calculation's, at
    its root.  ``engine`` given fills a name's engine field: the pipeline
    log a SIESTA prep writes is named for SIESTA.

    Every name comes out of :func:`compose`, so a card cannot show a spelling
    the writers do not use -- which is the failure this module exists for.
    """
    rows = []
    for a in ON_THE_LABEL:
        if a.staged != (stage is not None):
            continue
        if engine and a.engine and a.engine != engine:
            continue
        if when and a.when not in when:
            continue
        if calculation and not written_by(a, calculation, any_=True):
            continue
        # EACH ROW'S NAME as the stage's names spell it (:func:`_template`).
        # A FIELD HAS NO VALUE UNTIL THE FILE IS WRITTEN, so the card shows
        # the field's NAME: `<label>_<stage>.<tbt>.digest.nc` -- but the
        # engine, which the description names.
        name = shown(_template(a, label, stage),
                     **({"engine": engine} if engine else {}))
        rows.append(row_words(a, name))
    return rows


def row_words(a: "Artifact", name: str) -> dict:
    """One row of the catalogue as a page shows it, under the name it is
    spelled by there -- the one form of a row on the wire."""
    return {"name": name, "what": a.what, "when": a.when, "level": a.level,
            "where": LEVEL_WHERE.get(a.level, ""), "only": a.only}


def shown(template: str, **fields) -> str:
    """A NAME AS A PAGE OR A DESCRIPTION SHOWS IT -- ``template`` (a role, or
    a name with its fields left open) with each field in ``fields`` written
    as given and every other as ``<field>``: ``<label>.TBT.AVTEIG_<pair>``.
    Words for a reader, never a name a file is given (:func:`compose`)."""
    return _FIELD_RE.sub(lambda m: str(fields.get(m.group(1),
                                                  f"<{m.group(1)}>")),
                         template)


def fixed_name(name: str, **fields) -> str:
    """A file of the catalogue's FIXED-NAME family ``name`` -- a
    :data:`WRITTEN` row's name with fields, as :data:`PSEUDO_FILE`
    (``{species}.psml``) -- composed: every field filled, each checked
    against its declared shape (:data:`FIELDS`).  The counterpart of
    :func:`fixed_fields`; a name the catalogue does not name is
    :class:`RunFileError`."""
    _a_fixed_name(name, "fixed_name")
    wanted = _fields_in(name)
    if set(wanted) != set(fields):
        raise RunFileError(
            f"fixed_name({name!r}) takes {', '.join(wanted) or 'no field'}; "
            f"given {', '.join(sorted(fields)) or 'none'}.")
    for k, v in fields.items():
        if not re.fullmatch(FIELDS[k].shape, str(v)):
            raise RunFileError(
                f"{k}={v!r} does not match the declared shape "
                f"{FIELDS[k].shape!r} for that field ({FIELDS[k].what}; "
                f"e.g. {FIELDS[k].example}).")
    return _FIELD_RE.sub(lambda m: str(fields[m.group(1)]), name)


def _a_fixed_name(name: str, door: str) -> None:
    """Refuse ``name`` unless the catalogue names it as a FIXED name -- a
    :data:`WRITTEN` row's, or an engine's file named whole
    (:data:`ENGINE_FILES`' ``name``, SIESTA's :data:`SIESTA_FDF_LOG`) --
    the one check every fixed-name door makes first."""
    if not (any(a.name == name and not a.role for a in WRITTEN)
            or any(e.name == name for e in ENGINE_FILES)):
        raise RunFileError(
            f"{door}({name!r}): not a fixed name molbuilder's catalogue "
            f"names (`runfiles.WRITTEN`, `runfiles.ENGINE_FILES`).")


def fixed_fields(name: str, filename: str) -> "Optional[dict]":
    """The fields of ``filename`` read back as the catalogue's FIXED-NAME
    family ``name``, by their declared shapes (:data:`FIELDS`) --
    ``{"species": "Au1"}`` for ``Au1.psml`` as :data:`PSEUDO_FILE` -- or
    ``None`` when it is no name of that family.  The counterpart of
    :func:`fixed_name`; a name the catalogue does not name is
    :class:`RunFileError`."""
    _a_fixed_name(name, "fixed_fields")
    m = _role_pattern(name).match(filename)
    return m.groupdict() if m else None


def find_fixed(directory, name: str) -> "list[tuple[Path, dict]]":
    """Every file in ``directory`` of the catalogue's FIXED-NAME family
    ``name`` -- a :data:`WRITTEN` row's name with fields, as
    :data:`SCHEDULER_STDERR` (``{scheduler}.{jobid}.err``) -- each with its
    fields read back (:func:`fixed_fields`), sorted by name:
    ``[(path, {"scheduler": "slurm", "jobid": "66027146"})]``."""
    _a_fixed_name(name, "find_fixed")
    d = Path(directory)
    try:
        entries = sorted(d.iterdir())
    except OSError:
        return []
    out = []
    for p in entries:
        fields = fixed_fields(name, p.name)
        if fields is not None and p.is_file():
            out.append((p, fields))
    return out


def fixed(level: Optional[str] = None,
          when: "Optional[tuple]" = None,
          calculation: Optional[str] = None,
          engine: Optional[str] = None) -> "list[dict]":
    """The catalogue's FIXED-NAME files -- the ones not named on the label --
    at ``level``, with a line each: what the Task-setup card lists for the
    whole run (`job-set.json`, the plan, the machine record, the ledger),
    narrowed like :func:`manifest`.  A family shows its fields by name:
    ``<species>.psml``."""
    rows = []
    for a in WRITTEN:
        if a.role:
            continue
        if level and a.level != level:
            continue
        if when and a.when not in when:
            continue
        if calculation and not written_by(a, calculation, any_=True):
            continue
        if engine and a.engine and a.engine != engine:
            continue
        rows.append({"name": shown(a.name),
                     "what": a.what, "when": a.when, "only": a.only})
    return rows


#: THE STRUCTURE PAIR A PERSON SAVES, in a storage topic -- `structure/` holds
#: files, not calculations (`project-layout.md` § 2.6) -- so no row of the
#: catalogue names it: its stem is the person's.  What the Results file card
#: says of each half, read by `runs.saved_pair_row` (`web/results.md` § 3b).
SAVED_PAIR_WRITER = ("Save on the Molbuilder tab, or Export on the Results "
                     "tab (`StructureCodec.write`)")
SAVED_PAIR_WHAT = {
    "structure": ("a structure -- its atoms and coordinates, every frame when "
                  "it has several; its cell, labels and metadata are in the "
                  "sidecar beside it"),
    "sidecar": ("the sidecar of the structure beside it -- its cell, its "
                "region labels and its recorded metadata"),
}


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
            continue
        if (a.level == "launch") != in_launch and a.level != "transient":
            continue
        if not a.name.endswith("/") and _role_pattern(a.name).match(base):
            return a
    return None


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.monitor_companions`).
TRAVELS_WITH = ("molbuilder.runpoints",)

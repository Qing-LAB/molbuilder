"""Project directory layout: ``projects/<project>/<topic>/<calculation>/``.

The hierarchy organises work scientifically; what is inside a calculation
is ``execution/project-layout.md`` § 1's (``docs/execution/job-contracts.md``
§ 2.5).

Categories of work in this module:

  * **Name validation** — enforce ``[A-Za-z0-9_-]+`` so SIESTA's
    basename-discipline isn't broken by stray dots or spaces.
  * **Topic vocabulary** — the canonical topics
    (:data:`CANONICAL_TOPICS`): the storage topics and ``user``, and each
    calculation kind's topic from the kind table.  Ad-hoc topic names
    aren't accepted.
  * **Path resolution** — :func:`project_dir` / :func:`topic_dir` /
    :func:`structure_dir`.  Pure functions; no I/O.
  * **Directory creation** — :func:`ensure_structure_dir` and the project
    skeleton (:func:`populate_project_skeleton`) are the functions here that
    write.
  * **Discovery** — :func:`list_projects` / :func:`list_topics` /
    :func:`list_structures`; read-only.  With the path functions, what a
    person's script or notebook asks to find a calculation
    (`execution/job-contracts.md` § 2.5; `molbuilder.postprocess`).

This module does not know about scripts, conda envs, run wrappers, or
the web UI.  Those concerns live in their own modules
(``molbuilder.pyscf.input``, ``molbuilder.siesta.input``,
``molbuilder.runwrap``, ``molbuilder.web``).
"""

from __future__ import annotations


import logging
from pathlib import Path
from typing import List, Mapping, NamedTuple, Optional, Tuple

from . import kind_table
from .runfiles import NAME_CHARS, NAME_TOKEN, PSEUDO_FILE, TASK_FILE, fixed_name

_log = logging.getLogger(__name__)


PROJECTS_ROOT_NAME = "projects"

# Canonical direct children of ``projects/<project>/``
# (`job-contracts.md` § 2.5): the storage topics -- the structures a
# project curates and its SIESTA pseudopotentials -- a run topic for each
# calculation kind, its ``topic`` in the kind table in the table's order,
# then ``user``, the free-form workspace with no molbuilder layout inside it.
# Ad-hoc names at depth 1 are rejected to keep the vocabulary one across
# projects; a kind added to the table brings its topic.
CANONICAL_TOPICS: Tuple[str, ...] = (
    ("structure", "pseudopotential")
    + tuple(kind_table.kind(k).topic for k in kind_table.kinds())
    + ("user",))


class InvalidName(ValueError):
    """A project / topic / structure name failed validation."""


def validate_name(name: str, *, kind: str = "name") -> str:
    """Reject names that would break basename-based file discovery.

    Returns the name unchanged on success; raises :class:`InvalidName`
    with a helpful message on failure.  ``kind`` is interpolated into
    the message so users can tell which field is at fault.
    """
    if not isinstance(name, str) or not name:
        raise InvalidName(f"{kind} must be a non-empty string")
    if not NAME_TOKEN.fullmatch(name):
        raise InvalidName(
            f"{kind} {name!r} contains characters outside [{NAME_CHARS}]; "
            f"these would clash with SIESTA's basename-based file "
            f"discovery."
        )
    return name


def validate_topic(topic: str) -> str:
    """Topic must match the canonical vocabulary (:data:`CANONICAL_TOPICS`)."""
    validate_name(topic, kind="topic")
    if topic not in CANONICAL_TOPICS:
        raise InvalidName(
            # NO COUNT IN THE MESSAGE: the list is right there, and a number
            # beside it is a second thing to keep in step and buys nothing.
            f"topic {topic!r} is not one of the canonical topics: "
            f"{', '.join(CANONICAL_TOPICS)}.  Pick the closest match."
        )
    return topic


def validate_folder_name(name: str, inside: int, *, new: bool) -> str:
    """A FOLDER'S NAME BY ITS DEPTH -- the one rule every door that names a
    folder asks (`job-contracts.md` § 2.5; `project-layout.md` § 7,
    invariant 1: every segment from the root to a calculation is one token),
    ``inside`` the depth of the folder that holds it (1: a project's, so the
    name is its topic's; 2: a topic's, a calculation's; deeper: a folder of
    the person's own).  A NEW topic folder takes a topic's name; one renamed
    is an ordinary folder there, as an older project's are.  A project
    (``inside`` 0) is made whole by :func:`populate_project_skeleton` --
    never by naming a new folder at the tree's root, which is refused here;
    one renamed takes a project's name.  Raises :class:`InvalidName`;
    returns the name."""
    if inside < 1 and new:
        raise InvalidName(
            "a folder at the projects root is a project, which is made with "
            "its topic folders -- use New project")
    if inside == 1 and new:
        return validate_topic(name)
    return validate_name(name, kind="project" if inside < 1 else "folder")


# ---------------------------------------------------------------- #
#  Path resolution (pure functions, no I/O)                        #
# ---------------------------------------------------------------- #


#: The env override for the projects tree.  Named here because it is part
#: of the resolution :func:`projects_root` documents, not a loose string.
PROJECTS_ROOT_ENV = "MOLBUILDER_PROJECTS"


def projects_root(base: Optional[Path] = None) -> Path:
    """Where the projects tree is — **the one door** (user, 2026-08-22).

    Every surface that touches the tree resolves it here: the sidebar
    backend, the `jobset` verbs' ``--bundle``, the workspace store, the
    pseudopotential anchor, diagnostics.  That is the point of a single
    door — moving the tree is one setting, and it moves for everyone at
    once instead of for whichever caller remembered.

    **Resolution, in declared order — nothing is probed:**

      1. ``base`` — an explicit argument, for a caller serving another tree
      2. ``$MOLBUILDER_PROJECTS``
      3. ``molbuilder.json`` → ``paths.projects`` (a relative value is read
         from the molbuilder root, so it means the same thing wherever you
         run from)
      4. ``repo_root()/projects`` — the default

    Steps 2 and 3 move the folder, never rename it: what they name must be
    called ``projects`` (:func:`named_projects`).

    **Never the working directory**: that would make the tree a property of
    where the user happened to stand, the same command meaning different
    trees in different shells.  The default comes from the molbuilder root,
    which is
    a fact the package knows about itself (`job-contracts.md` § 2.5), and
    steps 2–3 exist because the default is not always writable — a cluster
    home with a quota, a scratch filesystem, a shared tree.

    Does NOT create the directory.  Read-only path resolution.

    Returns the path alone.  :func:`projects_root_with_source` is the same
    resolution reporting WHICH of the four steps answered -- use that wherever
    the answer is shown to a person, because "the default" and "your config
    says so" are different facts about the same path.
    """
    return projects_root_with_source(base).path


class ProjectsRoot(NamedTuple):
    """Where the projects tree is, and which declaration put it there.

    The second half exists because four things can answer, and a bare path
    cannot say which did.  `configuration.md` § 2.2 already requires that
    *"which file actually took effect is displayed, never inferred"* for the
    config files; the tree is the same question one level over, and the user's
    2026-09-12 instruction is the same rule: the location must be shown so
    nobody has to believe where it is.

    ``source`` is a phrase for a person, not a token to branch on.
    """

    path: Path
    source: str

    def describe(self) -> str:
        """One line, the same words on every surface that shows it."""
        return f"projects  {self.path}   ({self.source})"


def projects_root_with_source(base: Optional[Path] = None) -> ProjectsRoot:
    """:func:`projects_root`'s resolution, carrying which step answered.

    One resolver, so a surface that PRINTS the tree and a surface that USES it
    can never disagree -- which is the whole point of the one door.
    """
    if base is not None:
        return ProjectsRoot(base / PROJECTS_ROOT_NAME, "explicit argument")
    import os
    env = os.environ.get(PROJECTS_ROOT_ENV)
    if env:
        return ProjectsRoot(named_projects(Path(env).expanduser(),
                                           f"${PROJECTS_ROOT_ENV}"),
                            f"${PROJECTS_ROOT_ENV}")
    # Lazy: runtime_config is not needed to answer 1 or 2, and importing it
    # at module scope would tie the tree API to the config reader.
    from . import repo_root
    # NOTHING IS CAUGHT.  `runtime_config` imports only `config_dir`, so there
    # is no cycle to guard, and a RuntimeConfigError -- a typo in `paths` --
    # reaches the caller as the refusal it is, with the known keys named
    # (architecture.md § 8.2a).
    from .runtime_config import get_paths
    configured = (get_paths() or {}).get("projects")
    if configured:
        p = Path(configured).expanduser()
        return ProjectsRoot(
            named_projects(p if p.is_absolute() else repo_root() / p,
                           "molbuilder.json's paths.projects"),
            "molbuilder.json -> paths.projects")
    return ProjectsRoot(repo_root() / PROJECTS_ROOT_NAME,
                        "default -- no paths.projects set")


def named_projects(path: Path, said_by: str) -> Path:
    """``path``, when it is a folder named ``projects`` -- else refused.

    THE TREE IS THE FOLDER NAMED ``projects``, wherever it is
    (`job-contracts.md` § 2.5): a calculation finds its tree by walking up to
    that name (:func:`find_projects_root`), so a setting may move the folder
    but not rename it.  Under any other name the server would serve the tree
    while every calculation in it found none -- its citations and its
    ``psml_lib`` naming nothing.  Asked of every setting that names the tree:
    ``$MOLBUILDER_PROJECTS`` and ``paths.projects`` when read, and the
    installer's answer before it is written.
    """
    from .runtime_config import RuntimeConfigError
    if path.name != PROJECTS_ROOT_NAME:
        raise RuntimeConfigError(
            f"{said_by} names {path}, a folder called {path.name!r}: the "
            f"projects tree is a folder named {PROJECTS_ROOT_NAME!r}, because "
            f"a calculation finds its tree by walking up to that name -- "
            f"under any other name its citations and psml_lib name nothing.  "
            f"Rename the folder to "
            f"{PROJECTS_ROOT_NAME!r}, or set {path / PROJECTS_ROOT_NAME}.")
    return path


class OutsideRoot(ValueError):
    """A path that resolves outside the root it was required to stay in.

    Carries the candidate, the root and the fence's reason so a caller can
    compose its own message -- the CLI wants a sentence about the projects
    tree, the web layer an HTTP 400 naming the root -- and ``outside``:
    whether it resolved outside the root, rather than being refused before
    it resolved (a ``..``, a path that does not resolve).
    """

    def __init__(self, candidate, root, reason: str, *,
                 outside: bool = False):
        self.candidate = candidate
        self.root = root
        self.reason = reason
        self.outside = outside
        super().__init__(f"{candidate} is outside {root}: {reason}")


def contain(candidate, root) -> Path:
    """Resolve ``candidate`` and require that it lies inside ``root``.

    **THE fence — there is one, and this is it**: two fences around one tree
    is one too many.

    Three steps, and each is here for a reason paid for in advance:

      * ``..`` in the RAW spelling is refused before anything else.
        Resolution would normalise it away, but refusing early removes the
        ambiguity of *"did the writer think `..` was harmless?"*.
      * ``expanduser`` only — never ``expandvars``: with variable
        expansion, ``/etc/$SECRET_KEY/foo`` would resolve to a path echoed
        back in the "outside the root" error, leaking the value.  ``~`` stays because that leak is symmetric.
      * **both sides are resolved before comparing**, so a symlinked root
        or a relative working directory cannot make the fence decorative.

    Existence is NOT checked: a listing wants 404 on a missing path, a
    save-as target legitimately does not exist yet, and a calculation
    folder must be a directory.  The caller decides.

    Raises :class:`OutsideRoot`; returns the resolved path on success.
    """
    raw = str(candidate)
    if ".." in Path(raw).parts:
        raise OutsideRoot(raw, root, "the path may not contain '..'")
    try:
        resolved = Path(raw).expanduser().resolve()
        real_root = Path(root).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise OutsideRoot(raw, root, f"could not resolve: {exc}")
    if resolved != real_root and real_root not in resolved.parents:
        raise OutsideRoot(resolved, real_root, "outside the root",
                          outside=True)
    return resolved


def find_projects_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` to the ``projects/`` tree it lives in.

    Returns the tree root (the directory NAMED ``projects``), or ``None`` if
    ``start`` is not inside one.

    **Why the tree is found from the calculation, not from the process.**
    ``projects_root()`` answers *"where would a NEW project go?"* from this
    machine's declarations -- correct for the server, and wrong for a
    calculation, which is a folder that already knows where it lives.  A
    user on a cluster runs a verb from inside the calculation; anchoring
    anywhere else would make a template that worked on the workstation
    resolve to a folder that does not exist on the cluster.

    **The NEAREST ancestor wins, and the walk stops there.**  A tree inside a
    tree is somebody else's tree: continuing past the first match would let a
    calculation reach into a parent installation's pseudopotentials, which is
    exactly the machine-dependence this function exists to remove.

    ``start`` is resolved first, so a relative path still finds its real
    ancestry.  This is the tree half of the anchor rule in
    `job-contracts.md` § 2.5a; the resolver that applies it to a user-typed
    path is `pseudos.resolve_psml_lib`.
    """
    for anc in Path(start).resolve().parents:
        if anc.name == PROJECTS_ROOT_NAME:
            return anc
    return None


def at(address, root=None) -> Path:
    """THE TREE ADDRESS READ THROUGH THE FENCE (`job-contracts.md` § 2.5b):
    ``address`` -- a path from the tree's root, or an absolute one inside
    it -- resolved and held inside ``root`` (:func:`projects_root` unless
    given: the tree a calculation lives in, for a name it holds) by
    :func:`contain`.  Raises :class:`OutsideRoot`, which each caller says in
    its own words."""
    # THE ROOT RESOLVED FIRST: a setting that spells it with `..` names a
    # folder, and the fence's `..` rule is about the address.
    base = Path(root if root is not None
                else projects_root()).expanduser().resolve()
    return contain(base / Path(str(address)).expanduser(), base)


def segments(folder, root=None) -> Optional[Tuple[str, ...]]:
    """``folder``'s place in the tree as its segments from ``root``
    (:func:`projects_root` unless given: the tree a calculation lives in)
    -- ``()`` for the root itself, its length the folder's depth -- or
    ``None`` when it lies outside.  The reverse of :func:`at`."""
    here = Path(folder).resolve()
    base = Path(root if root is not None else projects_root()).resolve()
    try:
        return here.relative_to(base).parts
    except (ValueError, OSError):
        return None


def address(folder, root=None) -> str:
    """``folder``'s TREE ADDRESS -- its :func:`segments`, ``/``-separated
    (``.`` for the root) -- what ``--bundle`` takes and what a line, a
    record or a refusal names it by; its full path when it lies outside
    the tree."""
    parts = segments(folder, root)
    if parts is None:
        return str(Path(folder).resolve())
    return "/".join(parts) or "."


def project_dir(project: str, *, base: Optional[Path] = None) -> Path:
    """The folder of ``project``, under :func:`projects_root`; the name is
    checked (:func:`validate_name`).  Says nothing of whether it exists."""
    return projects_root(base) / validate_name(project, kind="project")


def topic_dir(project: str, topic: str, *,
              base: Optional[Path] = None) -> Path:
    """``<project>/<topic>``, each name checked (:func:`validate_topic`: one of
    :data:`CANONICAL_TOPICS`).  Says nothing of whether it exists."""
    return project_dir(project, base=base) / validate_topic(topic)


def structure_dir(project: str, topic: str, structure: str, *,
                  base: Optional[Path] = None) -> Path:
    """``<project>/<topic>/<structure>``: a calculation's folder, the
    ``structure`` level being the calculation's own name
    (`execution/job-contracts.md` § 2.5).  Each name checked; says nothing of
    whether it exists."""
    return topic_dir(project, topic, base=base) / \
           validate_name(structure, kind="structure")


def ensure_structure_dir(project: str, topic: str, structure: str, *,
                          base: Optional[Path] = None) -> Path:
    """Create ``<project>/<topic>/<structure>/`` under :func:`projects_root`
    and return its path -- the project, when not there yet, made whole
    (:func:`populate_project_skeleton`).  Idempotent (existing dirs OK)."""
    d = structure_dir(project, topic, structure, base=base)
    proj = project_dir(project, base=base)
    if not proj.exists():
        populate_project_skeleton(proj, project_name=project)
    d.mkdir(parents=True, exist_ok=True)
    return d


# The README of each storage topic and of ``user``, written into a new
# project as `<project>/<topic>/README.md` so a person navigating the tree
# (or a colleague handed a project tarball) sees what each folder is for
# without reading docs.  A run topic's is its kinds' (:func:`_topic_readme`).
_TOPIC_READMES: Mapping[str, str] = {
    "structure": (
        "# structure/\n\n"
        "Curated input structure files for this project.\n\n"
        "Drop `.xyz`, `.pdb` files here.  Flat directory --\n"
        "no subdirs by convention.  Use these as starting points for\n"
        "the calculations in the other topics.\n"
    ),
    "pseudopotential": (
        "# pseudopotential/\n\n"
        "SIESTA pseudopotentials for this project.  Drop `.psml`\n"
        f"files here, one per species, named for its label "
        f"(`{fixed_name(PSEUDO_FILE, species='H')}`,\n"
        f"`{fixed_name(PSEUDO_FILE, species='O')}`, "
        f"`{fixed_name(PSEUDO_FILE, species='Au1')}`, ...): SIESTA opens "
        f"`<label>.psml`.\n"
        "A SIESTA calculation whose `psml_lib` is `pseudopotential`\n"
        "takes the files it lacks from here at prep.\n"
    ),
    "user": (
        "# user/\n\n"
        "Free-form workspace -- no molbuilder layout applies inside this\n"
        "directory.  Use it for notes, scratch files, organising\n"
        "experiments that don't fit the other canonical topics, or\n"
        "any non-canonical directory structure you need.  The sidebar's\n"
        "*New folder* button makes folders here at any depth, each named\n"
        "with letters, digits, `-` and `_`, as every folder in the tree.\n"
    ),
}


def _topic_readme(topic: str) -> str:
    """The README of ``topic``: a storage topic's or ``user``'s own, a run
    topic's said from the kind whose topic it is."""
    if topic in _TOPIC_READMES:
        return _TOPIC_READMES[topic]
    of = next(kind_table.kind(k) for k in kind_table.kinds()
              if kind_table.kind(k).topic == topic)
    return (f"# {topic}/\n\n"
            f"Where this project keeps {of.with_article} -- a folder\n"
            f"holding its description (`{TASK_FILE}`) and its runs, grouped\n"
            f"in folders of your own as the work suits.\n")


def _project_root_readme(project: str) -> str:
    """Project-level README content; reminds the user what the layout is."""
    canonical_lines = "\n".join(
        f"  {t}/" + " " * max(1, 18 - len(t)) + "-- see " + t + "/README.md"
        for t in CANONICAL_TOPICS
    )
    return (
        f"# {project}\n\n"
        f"[Project description placeholder -- edit this file with the\n"
        f" purpose, references, and any project-specific conventions.]\n\n"
        f"## Layout\n\n"
        f"This project follows the molbuilder canonical hierarchy.  Each\n"
        f"subdirectory has a brief `README.md` explaining its purpose:\n\n"
        f"```\n{canonical_lines}\n```\n\n"
        f"See `docs/execution/project-layout.md` for the layout a\n"
        f"calculation's folders follow.\n"
    )


def populate_project_skeleton(proj: Path, *, project_name: str) -> None:
    """MAKE A PROJECT, WHOLE -- the one way a project comes to be
    (`project-layout.md` § 1.1: its README *"made with the project, as is
    every topic"*): the folder ``proj`` (:func:`project_dir` names it), the
    project's ``README.md``, and every :data:`CANONICAL_TOPICS` folder with
    its own.  Asked by New project, by `molbuilder describe` for a project
    not there yet, and by :func:`ensure_structure_dir`.  A folder already
    there is refused (:class:`FileExistsError`); anything failing after the
    folder is made removes it, so a project is there whole or not at all.
    """
    import shutil
    proj.mkdir(parents=True, exist_ok=False)
    try:
        # Plain writes, deliberately (`configuration.md` § 2.3's list): every
        # one lands in the folder just made, removed if anything fails.
        # There is no previous content to protect and no credential in them.
        (proj / "README.md").write_text(_project_root_readme(project_name))
        for topic in CANONICAL_TOPICS:
            sub = proj / topic
            sub.mkdir(parents=False, exist_ok=False)
            (sub / "README.md").write_text(_topic_readme(topic))
    except BaseException:
        shutil.rmtree(proj, ignore_errors=True)
        raise


# ---------------------------------------------------------------- #
#  Discovery (read-only filesystem traversal)                       #
# ---------------------------------------------------------------- #


def list_projects(*, base: Optional[Path] = None) -> List[str]:
    """Sorted list of project names, or ``[]`` if no ``projects/`` dir.

    The tree's own folders are no project: its pseudopotential library
    (`job-contracts.md` § 2.5a) and a hidden folder of molbuilder's.  Other
    directories whose names fail :func:`validate_name` (would clash with the
    SIESTA basename rule -- spaces, dots, etc.) are skipped with a warning
    logged at WARNING level so the user can tell why something they created by
    hand isn't appearing.
    """
    from .pseudos import CONVENTIONAL_LIBRARY
    root = projects_root(base)
    if not root.is_dir():
        return []
    valid: List[str] = []
    for p in root.iterdir():
        if (not p.is_dir() or p.name.startswith(".")
                or p.name == CONVENTIONAL_LIBRARY):
            continue
        if NAME_TOKEN.fullmatch(p.name):
            valid.append(p.name)
        else:
            _log.warning(
                "skipping %s: name fails the [%s]+ rule "
                "and won't work with SIESTA's basename discipline",
                p, NAME_CHARS,
            )
    return sorted(valid)


def list_topics(project: str, *,
                 base: Optional[Path] = None) -> List[str]:
    """Topics present under ``project``, in canonical order.

    Returns the canonical topics intersected with what's on disk, in
    :data:`CANONICAL_TOPICS` order.
    Non-canonical directories under the project root are ignored.
    """
    pd = project_dir(project, base=base)
    if not pd.is_dir():
        return []
    on_disk = {p.name for p in pd.iterdir() if p.is_dir()}
    return [t for t in CANONICAL_TOPICS if t in on_disk]


def list_structures(project: str, topic: str, *,
                     base: Optional[Path] = None) -> List[str]:
    """Sorted names under ``project/topic/``: the calculations of a run topic,
    or ``[]``."""
    td = topic_dir(project, topic, base=base)
    if not td.is_dir():
        return []
    return sorted(p.name for p in td.iterdir()
                   if p.is_dir() and NAME_TOKEN.fullmatch(p.name))


__all__ = [
    "PROJECTS_ROOT_NAME",
    "CANONICAL_TOPICS",
    "InvalidName",
    "validate_name",
    "validate_topic",
    "validate_folder_name",
    "projects_root",
    "contain",
    "at",
    "segments",
    "address",
    "project_dir",
    "topic_dir",
    "structure_dir",
    "ensure_structure_dir",
    "populate_project_skeleton",
    "list_projects",
    "list_topics",
    "list_structures",
]

"""Pluggable nucleic-acid builders: the registry (`engines/builders.md` § 4).

A backend is a module ``_<name>.py`` named in :data:`BACKENDS` with the
words a person picks it by, in auto's order, and declaring:

  * ``build(kind, sequence, form, terminal, title, double_strand, strand2)``
    -- the :class:`~molbuilder.structure.Structure` it builds;
  * ``is_available()`` -- whether it runs here, by its own detection
    (asked by the registry before any build: a backend checks nothing of
    its own absence), and, optionally, ``unavailable()`` -- why not, in
    more words than its ``INSTALL``;
  * ``built(kind, form, terminal)`` -- what it builds of an ask (:class:`Built`);
  * ``DUPLEX`` -- whether it builds a duplex; ``INSTALL`` -- how it is
    installed, in words; ``ENV`` -- the env that holds it (a recipe's
    category, ``"host"`` the host env's), whose command
    :func:`install_command` spells, or ``None`` for a program no env holds.

Nothing here names one backend: every rule asks each module.  Every backend
delegates to an external builder, and none grows the chain residue by
residue in-house: doing that gets the 5'-end chemistry wrong and cannot
enforce the tetrahedral phosphate-bridge geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ...chemistry import BackendUnavailable, BuildRefused


#: THE HELICAL FORMS EACH KIND IS BUILT IN, its default first -- the
#: canonical one, B for DNA and A for RNA: no backend builds RNA in another
#: (the user, 2026-11-18: "A only, everywhere") -- and THE TERMINAL STATE
#: built when none is asked, both ends -OH.  Read here by what asks or
#: fills them: :func:`check_ask`, :func:`dispatch`'s defaults, the command
#: line's choices, the page's menus and the check after a build.
KIND_FORMS = {"dna": ("B", "A", "Z"), "rna": ("A",)}
DEFAULT_FORM = {kind: forms[0] for kind, forms in KIND_FORMS.items()}
DEFAULT_TERMINAL = "OH"

#: WHAT A NUCLEIC BUILD MAY ASK, one home beside the forms of each kind: the
#: words a person picks each form by, the backends -- in auto's order, best
#: geometry first -- with theirs, and the terminal states
#: (`engines/builders.md` § 1, § 4).  The nucleic builders refuse another
#: form, terminal or backend before any backend runs (:func:`check_ask`);
#: every surface's choices -- the command line's, the Molbuilder tab's menus
#: -- read them, and each backend's own table covers them.  What a backend
#: builds of them is its own to say (its module's ``built``, ``DUPLEX``).
FORMS = {"B": "right-handed",
         "A": "right-handed, wide",
         "Z": "left-handed"}
TERMINALS = ("OH", "5P", "3P", "PP")
BACKENDS = {"threedna": "3DNA (canonical helix)",
            "amber": "AmberTools (extended chain)",
            "rdkit": "RDKit (folded conformer)"}


def check_ask(kind: str, form: Optional[str], terminal: Optional[str],
              backend: str = "auto") -> None:
    """Refuse a ``kind`` of nucleic acid asked in a ``form``, with a
    ``terminal`` or on a ``backend`` outside :data:`KIND_FORMS` /
    :data:`TERMINALS` / :data:`BACKENDS` (and ``auto``) --
    :class:`BuildRefused`, naming the ones it may be; ``None`` asks the
    kind's default."""
    if backend != "auto" and (not isinstance(backend, str)
                              or backend not in BACKENDS):
        raise BuildRefused(
            f"the backend {backend!r} is not one a nucleic acid is built on; "
            f"ask auto or one of {', '.join(BACKENDS)}")
    if form is not None and form not in KIND_FORMS[kind]:
        raise BuildRefused(
            f"the helical form {form!r} is not one {kind.upper()} is built "
            f"in; ask {' or '.join(KIND_FORMS[kind])}")
    if terminal is not None and terminal not in TERMINALS:
        raise BuildRefused(
            f"the terminal {terminal!r} is not one a nucleic acid is built "
            f"with; ask one of {', '.join(TERMINALS)}")


@dataclass(frozen=True)
class Built:
    """What a backend builds of an ask *(decided with the user, 2026-10-31,
    (b))*: the helical ``form`` -- ``None`` where it shapes no helix -- and
    the ``terminal`` state, each with the words that say why when it is not
    what was asked.  Each backend answers it (its module's ``built``) beside
    the table its own build reads, and the check after the build
    (`validation.builders.check_built`) says what differs."""
    form: Optional[str]
    terminal: str
    form_words: str = ""
    terminal_words: str = ""


def built(backend: str, kind: str, form: str, terminal: str) -> Built:
    """What ``backend`` builds of a ``kind`` asked in ``form`` with
    ``terminal`` -- its own module's answer."""
    return _modules()[backend].built(kind, form, terminal)


def install_command(env: Optional[str]) -> Optional[str]:
    """THE COMMAND THAT INSTALLS what the env ``env`` holds -- a recipe's
    category (`envs.recipes`), ``"host"`` the host env's, whose recipe has
    none -- in the env framework's one spelling (`envs.hints`): the host env
    repaired with its optional packages (PeptideBuilder's and PubChemPy's
    among them), another env installed.  ``None`` for what no env holds."""
    if env is None:
        return None
    from ...envs.hints import fix_cmd, install_cmd
    from ...envs.recipes import recipe_of_category
    category = None if env == "host" else env
    recipe = recipe_of_category(category)
    return (fix_cmd("repair", recipe.name, "--include-optional")
            if category is None else install_cmd(recipe.name))


def with_install(words: str, env: Optional[str]) -> str:
    """``words`` saying what is missing, then the command that installs it
    where an env holds it (:func:`install_command`)."""
    command = install_command(env)
    return f"{words}  To install: `{command}`." if command else words


def unavailable_words(name: str) -> str:
    """Why the backend ``name`` cannot run here: its module's own account
    (``unavailable()``, where it says more -- 3DNA's places looked), else
    its words and how it is installed (``INSTALL``, and its env's command,
    :func:`with_install`)."""
    mod = _modules()[name]
    hook = getattr(mod, "unavailable", None)
    return with_install(
        hook() if hook else f"{BACKENDS[name]} is not installed: "
                            f"{mod.INSTALL}.", mod.ENV)


def duplex_backend(backend: Optional[str]) -> str:
    """The backend a duplex is built on: ``backend`` itself when it builds
    one (its module's ``DUPLEX``).  Else refused, by why: the person asked
    a backend that builds a single strand while one that builds a duplex is
    installed -- :class:`BuildRefused`, naming it; or none that builds one
    is installed -- :class:`BackendUnavailable`, each one's own account of
    why (:func:`unavailable_words`)."""
    mods = _modules()
    if backend in mods and mods[backend].DUPLEX:
        return backend
    able = [name for name in BACKENDS if mods[name].DUPLEX]
    names = ", ".join(BACKENDS[name] for name in able)
    if any(mods[name].is_available() for name in able):
        raise BuildRefused(
            f"a duplex is built by {names}; the {BACKENDS[backend]} "
            f"backend builds a single strand -- choose it, or auto")
    raise BackendUnavailable(
        f"A duplex is built by {names}.\n"
        + "\n".join(unavailable_words(name) for name in able),
        missing=able[0])


def resolved_backend(backend: str) -> Optional[str]:
    """The backend a build with ``backend`` runs: ``auto`` resolved by
    :func:`auto_backend_name` -- ``None`` when none is installed -- any
    other name as given.  Asked once, before the build, by every door that
    builds: the build runs it and the check after it names it."""
    return auto_backend_name() if backend == "auto" else backend


# ---------------------------------------------------------------- #
#  Registry                                                         #
# ---------------------------------------------------------------- #


def _modules():
    """Each backend's module, by its name (:data:`BACKENDS`) -- lazy, so a
    missing dep doesn't break the whole package."""
    from importlib import import_module
    return {name: import_module(f"{__name__}._{name}") for name in BACKENDS}


def available_backends() -> Dict[str, bool]:
    """Each backend's name -> whether it runs on this machine, by its own
    detection (its module's ``is_available``)."""
    return {name: m.is_available() for name, m in _modules().items()}


def auto_backend_name() -> str | None:
    """Which backend ``dispatch(backend='auto')`` would pick on this
    machine, or ``None`` if no backend is available.

    Pure read-only -- doesn't run any backend.  Useful for surfacing
    the resolved choice in CLIs and the web UI before/after a build.
    """
    avail = available_backends()
    for name in BACKENDS:
        if avail.get(name):
            return name
    return None


def dispatch(kind: str, sequence: str, *,
             backend: str = "auto",
             form: Optional[str] = None,
             terminal: Optional[str] = None,
             title=None,
             double_strand: bool = False,
             strand2: "str | None" = None):
    """Build a nucleic acid on ``backend`` -- one of :data:`BACKENDS`, or
    ``auto``, the first installed -- the ask checked already
    (:func:`check_ask`, and :func:`duplex_backend` for a duplex:
    ``double_strand`` a canonical one, ``strand2`` an explicit second strand,
    5'->3').  A form or terminal left ``None`` is the kind's default
    (:data:`DEFAULT_FORM`, :data:`DEFAULT_TERMINAL`), filled here for the
    build -- the check after it reads the same constants.  A backend that
    cannot run here is :class:`BackendUnavailable`, by its module's own
    detection (``is_available``) before it builds, said in its own words
    (:func:`unavailable_words`) -- ``auto`` with none installed, each
    backend's."""
    chosen = resolved_backend(backend)
    mods = _modules()
    if chosen is None:
        raise BackendUnavailable(
            "No nucleic-acid backend is installed; any of these builds one.\n"
            + "\n".join(unavailable_words(name) for name in BACKENDS))
    if not mods[chosen].is_available():
        raise BackendUnavailable(unavailable_words(chosen), missing=chosen)
    return mods[chosen].build(
        kind, sequence, form or DEFAULT_FORM[kind],
        terminal or DEFAULT_TERMINAL, title,
        double_strand=double_strand, strand2=strand2)


__all__ = [
    "BACKENDS", "DEFAULT_FORM", "DEFAULT_TERMINAL", "FORMS", "KIND_FORMS",
    "TERMINALS",
    "Built", "auto_backend_name", "available_backends", "built",
    "check_ask", "dispatch", "duplex_backend", "resolved_backend",
    "unavailable_words",
]

"""The template — a calculation's parameter catalogue, as one TOML file.

**Module:** L1, floor 1 -- the catalogue and the template format.  Imports
the standard library and the ground floor alone (``identity``,
``kind_table``, ``persist``; inside functions ``engine_table``, ``kmesh``);
imported by the engines' producers, ``validation/task``,
``resolve`` and every layer above.  Nothing here touches the filesystem or a
scheduler: the template FILE's name and read are ``template_file``'s.

**Contract:** [`engines/template.md`](?doc=engines/template.md) — § 3 the required
keys, § 4 the TOML format, § 5 the anatomy of an item, § 6 the ``kind``
vocabulary and ``read_by`` · [`execution/generator.md`](?doc=execution/generator.md)
§ 3.1 — why the UI is a reader and which keys serve it.

WHAT IT IS.  Every parameter the engine's schema declares, each carrying its
value, what it is validated as, who owns it, and what we know about it.  **One
artifact, four readers** (`template.md` § 5): a person learns the calculation, a
**surface renders it**, ``prep`` rebuilds a config and renders decks, and
validation gets a real object out of it.

THE VALUE IS ON THE ITEM, ONCE.  A template stores each value exactly once,
which is the whole argument for TOML over the engine's own format (§ 4.1): a
file that parses cleanly and describes a *different* calculation than it appears
to is the failure mode worth designing out, and storing a value twice is how you
get one.

**An absent ``value`` means explicitly unset** (§ 3) — a real state, distinct
from the default.  TOML has no null, so absence is the only encoding, and it is
unambiguous.

THE WRITER CHECKS ITSELF.  ``tomllib`` reads TOML and does not write it, so the
emitter here is hand-rolled — and § 4.1 requires that whatever emits a template
**read its own output back and compare it to what it meant to write**.
:func:`_emit` builds the payload as a plain object, serialises it, parses the
result, and refuses if the two differ.  That turns *"we emitted TOML correctly"*
from an assumption into a property checked on every call.

**THE CATALOGUE IS THE MASTER** (§ 2.1, § 4.3).  ``molbuilder/data/
catalogue.template.toml`` is authored as TOML and holds every parameter both
engines declare; :func:`template_with_values` narrows it to one engine and fills
in what a config holds, which is what a surface produces once a person has
answered.  A config class is a translator on the way OUT: template → config
→ deck.
"""
from __future__ import annotations

import dataclasses
import functools
import re
import tomllib
import types
import typing
from pathlib import Path as _Path
from dataclasses import dataclass, field
from typing import (Any, Dict, List, Mapping, NamedTuple, NoReturn,
                    Optional, Sequence, Tuple, Union)

from .identity import stage_key
from . import kind_table as _kinds
from .persist import check_schema


SCHEMA = "molbuilder/template@2"

#: The suffix of a template's filename; the stem is the ``SystemLabel``.
#: ``<label>.template.toml`` (`job-contracts.md` § 6.3).
SUFFIX = ".template.toml"

#: § 6 — closed. An unknown ``kind`` is refused, never skipped, because a reader
#: that quietly dropped an item it did not understand would emit a deck missing
#: a parameter and say nothing.
KINDS = ("engine", "deck", "wrapper", "produce", "monitor")

#: The closed CATEGORY vocabulary (`engines/template.md` § 6.2), in READING
#: ORDER -- a surface presents panels top to bottom in this order, because it
#: is the order a person decides in and a methods section is written in.
#:
#: A category has NO effect on the generated script: the deck writer filters on
#: ``kind``.  It is a presentation-and-discovery key, which is why an item may
#: carry SEVERAL -- the first is the panel it appears on, the rest make it
#: findable where a user would look.
#:
#: ``execution`` is the exception and is a CLAIM rather than a hint: a benchmark
#: takes it as the sweepable set -- knobs that change speed and not the answer.
#: Tagging something ``execution`` that changes the answer means a sweep
#: silently measures a different calculation at each point.
CATEGORIES = ("system", "method", "accuracy", "convergence",
              "procedure", "execution")


#: **THE machine-answered fact, one home.**  `engines/template.md` § 6.4.
#:
#: An item whose value the SCHEDULER answers -- ranks, threads, memory.  A
#: portable description states the question and never the answer, so a value
#: on one is refused on read.
#:
#: There is no second vocabulary.  Ask ``Item.allocation``, or
#: ``select(t, allocation=True)``.

#: § 5 — the validation vocabulary. TOML types the *storage*; this types what a
#: reader must check, which a parser cannot know: that ``pow2`` is a power of
#: two, that ``enum`` is drawn from ``choices``, that ``text`` is verbatim engine
#: text to be copied rather than interpreted.
TYPES = ("int", "float", "str", "bool", "enum", "pow2",
         "int3", "float3", "strlist", "intlist", "text")

#: The closed GROUP vocabulary -- *when, and on which surface, do I set this?*
#: It is the OUTER card of a form, load-bearing since 2026-06-13 (the stage
#: selector once silently rewrote budget and system fields), and it is a
#: different axis from :data:`CATEGORIES`, which asks what QUESTION about the
#: calculation an item answers.
#:
#: ``output`` answers *what do I get back*, beside *what am I computing*,
#: *how tight*, and *how much compute*.
#:
#: ``staging`` names a parameter set by the STAGING surface rather than by the
#: tab that collects the physics (user, 2026-08-15). The item is a real
#: parameter and the template carries it; the parameter form simply is not
#: where it is answered.
#:
#: **Unlike `category` this is not required on every item** -- it is
#: presentation, and a template read headlessly by ``prep`` never asks. It is
#: required on every item of the CATALOGUE, which is a different claim and is
#: guarded by ``tests/test_catalogue_agreement.py``: an item there with no
#: group is a parameter no surface can place.
#: ``setup`` is FIRST because its members are the answers a calculation
#: cannot be built without: what the run is called, and where its
#: pseudopotentials come from (user, 2026-08-15). A card of their own,
#: because a card is ordered by `category`, and re-ordering within one card
#: would break the shared legend order that every other card depends on.
GROUPS = ("setup", "profile", "stage", "budget", "output", "staging")

#: ...and what each card SAYS -- its title and the one line under it, the one
#: home of those words, served with every form schema in its ``cards`` (as
#: :data:`SOURCE_WORDS` is) so no surface keeps a copy (`form-schema.md`
#: § 1.3).  True on every form: a stage may override any setting that is not
#: shared, so no card says its settings are set once for every stage.
#: ``staging`` has no card on a parameter form -- its members are each
#: stage's run card -- and its words say so.
GROUP_WORDS = {
    # The Setup card's line depends on which of its fields the form shows:
    # `SETUP_WORDS`, below.
    "setup": ("Setup", None),
    "profile": ("Run profile",
                "The method and the physical character of the system: "
                "functional, basis, charge and spin, smearing, solvent -- "
                "and the SCF solver and mixer that suit it."),
    "stage": ("Convergence targets",
              "How fine and how converged -- what a staged sequence "
              "tightens as it goes."),
    "budget": ("Compute & budget",
               "How much compute: the iteration caps.  Scales with system "
               "size; does NOT change what counts as converged."),
    "output": ("Output files",
               "What the run writes -- trajectories, logs, geometry "
               "snapshots -- and how much the generated input explains "
               "itself.  Changes what you get back, never the answer."),
    "staging": ("Run card",
                "How each stage runs -- ranks, threads, memory, GPU, "
                "retries, restart -- answered on that stage's run card on "
                "Task setup, never on a parameter form."),
}
if tuple(GROUP_WORDS) != GROUPS:      # one vocabulary, in one order
    raise RuntimeError("template.GROUP_WORDS must name every group of "
                       "template.GROUPS, in its order")


def _results_of(kind: str) -> Tuple["_kinds.Result", ...]:
    """The results a ``kind`` gives, in the order a person decides them --
    its record's (`kind_table`, `engines/template.md` § 6.2a); ``()`` for
    no kind, a form of no kind's own."""
    return _kinds.kind(kind).results if kind in _kinds.kinds() else ()


def _roles_of(kind: str) -> Tuple[str, ...]:
    """A ``kind``'s rungs' roles (`kind_table`); ``()`` for a kind without
    roles, or none named."""
    return _kinds.kind(kind).role_names if kind in _kinds.kinds() else ()


class Card(NamedTuple):
    """One card a form of a kind draws (`template.cards`): its key -- a
    group's or a result's -- its title and line, and whether its settings
    are drawn under the category legends (a result's card is not: its
    title already says what they are for)."""
    key: str
    title: str
    subtitle: Optional[str]
    legends: bool


def cards_of_every_kind() -> Tuple[Card, ...]:
    """Every card any kind draws -- the groups' and every kind's results'."""
    return tuple({c.key: c for k in (None, *_kinds.kinds())
                  for c in cards(k or "")}.values())


def cards(kind: str) -> Tuple[Card, ...]:
    """The cards a form of a ``kind`` calculation draws, in order
    (`engines/template.md` § 6.2a): *Setup*, the kind's results, then the
    other groups.  The Setup card's line is chosen by the fields a form
    shows (:func:`setup_words`), so its ``subtitle`` here is ``None``."""
    groups = [Card(g, t, sub, True) for g, (t, sub) in GROUP_WORDS.items()]
    results = [Card(r.key, r.title, r.line, False)
               for r in _results_of(kind)]
    return tuple(groups[:1] + results + groups[1:])

#: THE SETUP CARD'S LINE, one for each set of its fields a form shows -- what
#: to fill in, with an example (the user, 2026-10-10: "you can be elaborate
#: for different situations").  A SIESTA form shows the system label and the
#: pseudopotential folder; a PySCF form the job name; the Transport tab's
#: shared panel the folder alone, for a cited structure pair.
SETUP_WORDS = {
    frozenset({"system_label", "psml_lib"}):
        "Give the run a name and pick its pseudopotential folder.  The name "
        "begins every file the run writes -- for example water, so its "
        "files are water_01_<stage>.fdf, water.XV and so on.  "
        "The folder holds one .psml file for each element of the structure "
        "-- for example pseudopotential/, with O.psml and H.psml for water.  "
        "Nothing can be built until the folder is filled in.",
    frozenset({"job_name"}):
        "Give the run a name.  It begins every file the run writes -- for "
        "example water, so its script is water_01_<stage>.py and its log "
        "water_01_<stage>.log.",
    frozenset({"psml_lib"}):
        "Pick the pseudopotential folder: one .psml file for each element of "
        "the junction -- for example pseudopotential/, with Au.psml, C.psml, "
        "H.psml and S.psml for a gold-benzenedithiol junction.  The structure "
        "pair you cited brings none of its own, so the run takes them from "
        "this folder.  Nothing can be built until it is filled in.",
}


def setup_words(shown) -> str:
    """The Setup card's line for a form showing the Setup fields ``shown``
    (their names).  A set with no line here is refused: a form that shows a
    new combination needs its own words, never another's."""
    key = frozenset(shown)
    if key not in SETUP_WORDS:
        raise ValueError(f"the Setup card has no line for a form showing "
                         f"{sorted(key)} -- add one to template.SETUP_WORDS")
    return SETUP_WORDS[key]


#: THE RUN PROFILE CARD'S LINE, by what a form draws in it: the electronic
#: state alone -- the Spectrum tab's part both engines share, drawn in its
#: chemistry card above the answer it is worked into -- or the method
#: without it -- each engine's own part there; a form drawing both says the
#: card's own line (`GROUP_WORDS`).
PROFILE_STATE_WORDS = ("One setting for both engines: a value you type goes "
                       "to the engine you choose, which refuses beside the "
                       "field one it cannot run; a blank is worked out for "
                       "each engine -- its answer is below.")
PROFILE_METHOD_WORDS = ("How the electrons are computed: the method, and "
                        "the solver settings that suit it.")


def card_line(card: str, shown) -> Optional[str]:
    """The line under ``card`` on a form drawing the fields ``shown`` (their
    names) in it -- true of what that form shows (`web/form-schema.md`
    § 1.3): the Setup card's by its fields (:func:`setup_words`), the Run
    profile's by whether it holds the electronic state, the method or both;
    any other card's own line."""
    shown = set(shown)
    if card == "setup":
        return setup_words(shown)
    if card == "profile":
        state = shown & set(STATE_ITEMS) - {"method"}
        if state and shown <= set(STATE_ITEMS):
            return PROFILE_STATE_WORDS
        if not state:
            return PROFILE_METHOD_WORDS
    words = dict((c.key, c.subtitle) for c in cards_of_every_kind())
    return words.get(card)


def stage_words(stage: str) -> str:
    """What a form of ONE stage's own values says above its cards -- the
    transport tab's rung tabs (`engines/transport.md` § 3.8.2a): the same
    settings appear on every stage's form, so each says whose they are."""
    return (f"{stage}'s own values, over the template's: a value set here "
            f"reaches {stage}'s deck alone, and a blank field runs the "
            f"template's value, shown as its hint.")


def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(f"template{': ' + where if where else ''}: {msg}")


# --------------------------------------------------------------------- #
#  One item                                                             #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class NotOffered:
    """WHY A CHOICE IS NOT OFFERED, where it holds (§ 6.3a): ``on`` an
    engine, a calculation kind or ``"<engine>.<kind>"`` -- ``""`` wherever
    a value is withheld; ``values`` the withheld choices it explains --
    ``()`` every one withheld there; ``why`` the reason, read from the
    engine's source."""
    on: str
    values: Tuple[Any, ...]
    why: str

    def holds(self, value: Any, engine: str, kind: str) -> bool:
        """Whether this row explains ``value`` withheld from a ``kind``
        calculation on ``engine``."""
        return (self.on in ("", engine, kind, f"{engine}.{kind}")
                and (not self.values or is_member(value, self.values)))


@dataclass(frozen=True)
class Item:
    """One parameter, with everything all four readers need from it.

    **The four § 3 requires are `kind`, `category`, `type` and `help`** —
    matching :data:`_REQUIRED_ITEM_KEYS`.  ``name`` is not among them because it
    is not a key at all: it is the table header, ``[item.<name>]``.  Everything
    else is
    present when it applies and absent when it does not — *"absent is not a
    failure; it is the honest statement that the parameter has no default, no
    bounds, no unit, or no other reader."*

    ``value`` is ``None`` for **explicitly unset**, which is the state § 3
    distinguishes from the default. TOML cannot express null, so a written
    template simply omits the key.
    """
    name: str
    kind: str
    type: str
    help: str

    value:   Any = None
    default: Any = None
    #: WHERE THE VALUE CAME FROM -- one of :data:`SOURCES` (§ 6.6 obligation
    #: 2, plan § 5w K7): a cited run, the structure's saved record, the
    #: person, or nobody (``default``, *not chosen*).  ``None`` is a template
    #: written before K7, which never recorded it: *not recorded*, never
    #: guessed.
    source:  Optional[str] = None

    # --- reaching the deck (§ 6) ---
    anchor:  str = ""                       # required when kind == "engine"
    expands: Tuple[str, ...] = ()           # required when kind == "deck"
    read_by: Tuple[str, ...] = ()           # § 6.1 — who ELSE derives from it
    #: ``kind = "wrapper"`` -- THE WORDS THE RUN SCRIPT PASSES THE ENGINE on
    #: its command line for this item, by the program it starts
    #: (``Job.program``; the engine's own by default): ``{program:
    #: [word, ...]}``, each word with the placeholders the script fills
    #: (``{left}``, ``{slack}`` for `wall_slack`).  A program not named takes
    #: the item some other way -- a PySCF deck from its environment -- or not
    #: at all (`running-a-job.md` § 3.6).
    launch: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    #: **How the engine SPELLS this**, in full — and a different fact from
    #: :attr:`anchor`, which is the bare leading keyword the deck writer
    #: matches on (§ 5: *"an anchor is a bare keyword, never a sentence"*).
    #:
    #: The two are not interchangeable; keeping only the anchor loses
    #: information:
    #:
    #:   * ``gto.M(basis=...)``, ``(charge=...)``, ``(spin=...)`` and
    #:     ``(symmetry=...)`` all reduce to ``gto.M`` — four controls showing
    #:     one useless badge;
    #:   * ``mf = mf.density_fit()`` / ``.PCM()`` / ``.add_dispersion(...)``
    #:     all reduce to ``mf``;
    #:   * and a **molbuilder note** — ``(molbuilder: .run.sh ``mpirun -np N``
    #:     only; not in .fdf)`` — leads with no keyword at all. That note is
    #:     the ONLY way a
    #:     reader learns the setting never reaches the deck, which is why
    #:     `web/form-schema.md` § 1a requires it always be present.
    #:
    #: A surface shows this. The deck writer never reads it — it takes
    #: ``anchor`` and ``expands``, which is why truncating the anchor was
    #: right and keeping only the anchor was not.
    engine_key: str = ""

    #: **Where the engine's own documentation defines this keyword** — one
    #: string, naming the version so a citation that goes stale is visible
    #: rather than silently wrong: ``SIESTA 5.4.2 §6.9.2 'Mixing options'``.
    #:
    #: A REVIEWER'S field, and deliberately the only one that lives in the
    #: catalogue alone.  Nothing renders it: no surface shows it, no deck
    #: carries it, and no config class mirrors it (user, 2026-08-15: *"this
    #: does not have to show up in the UI or generated config, but serve as a
    #: reference point when review of template or revision of it could
    #: benefit from"*).  Like every other fact about an item it has this
    #: one home (§ 2.1).
    #:
    #: The citations were DERIVED, not recalled: the 5.4.2 manual sources
    #: were parsed for every ``\begin{fdfentry}{...}`` and matched to our
    #: anchors through fdf's own ``labeleq`` normalisation, which is how
    #: ``SaveHS`` finds the entry the manual spells ``Save.HS``.
    manual: str = ""
    #: THE ENGINE'S OWN VALUE for this keyword, and the line of its source
    #: that sets it -- ``(value, source)``, on an item whose keyword one deck
    #: may state and another leave to the engine (`engines/template.md` § 5):
    #: a keyword one deck states at this value and the other leaves out is
    #: the same engine input (`siesta.engine_input`).  Not ``default``, which
    #: is molbuilder's own.  A catalogue fact, never written into a
    #: calculation's template.  Empty: none stated -- then
    #: :attr:`engine_unfixed` says why.
    engine_default: Tuple[Any, ...] = ()
    #: WHAT THE ENGINE'S VALUE FOLLOWS when it has no one value -- another
    #: keyword, the build, the run -- and the line of its source:
    #: ``(what, source)``.  A keyword one deck states and the other leaves
    #: out is then a difference (`siesta.engine_input`).
    engine_unfixed: Tuple[str, ...] = ()

    # --- bounds and presentation ---
    #: Required when type == "enum".  A member is a string or a WHOLE NUMBER
    #: (`engines/template.md` § 5): `unpaired_electrons` is 0-10 or "free".
    choices: Optional[Tuple[Union[str, int], ...]] = None
    range:   Optional[Tuple[float, float]] = None
    unit:    Optional[str] = None
    group:   Optional[str] = None           # the card, unless a result names one

    # --- what a SURFACE needs (generator.md § 3.1a) -------------------- #
    # The UI is built FROM the template rather than merely generated from the
    # same schema, so a template names its own fields and what UNSET is
    # called.
    label:      str = ""                    # "MPI ranks (np)" -- the human name
    null_label: str = ""                    # "(auto)" -- what UNSET is called
    #: WHAT THE ENGINE DOES WHEN THIS ITEM WRITES NOTHING, one sentence from
    #: the engine's source (§ 5, § 6.6 obligation 4): the deck writes
    #: ``# <keyword> not written: <not_set>`` where the line would be, so no
    #: value reaches the engine by omission (`script_emit.not_set_line`).
    #: Required on every optional engine or deck item the scheduler does not
    #: answer -- checked by :func:`catalogue`.
    not_set: str = ""

    # --- § 6.2 / § 6.3 / § 6.4 (schema @2) ---------------------------- #
    #: Which questions this item answers.  FIRST is the panel; the rest make
    #: it findable (§ 6.2).
    category: Tuple[str, ...] = ()
    #: Which engines this item applies to.  EMPTY MEANS ALL (§ 6.3).
    engines:  Tuple[str, ...] = ()
    #: Which CALCULATION KINDS may select this item -- the exact symmetry of
    #: ``engines``: empty means every kind, and a generated per-calculation
    #: template carries the key on no item, because the selection already
    #: happened (the same writer rule that strips ``engines``).
    calculations: Tuple[str, ...] = ()
    #: A KIND'S OWN RECOMMENDATION for this parameter, as ``(kind, value)``
    #: pairs (§ 6.3a): what a form built for that kind shows and what
    #: `molbuilder describe` writes into that kind's template in place of the general
    #: default -- a frequency's relaxation is tighter than an optimization's.
    #: Pairs rather than a mapping so the frozen item stays hashable.
    recommended: Tuple[Tuple[str, Any], ...] = ()
    #: A HARD LIMIT, ``(bound, why)``: the value must be greater than
    #: ``bound``, and ``why`` is the reason every refusal gives (§ 5.3,
    #: 2026-09-30) -- ``(0, "SIESTA divides each force difference by it
    #: (ofc.f90)")``.  Empty means no limit; ``range`` is a recommendation and
    #: never one.  Every door asks :func:`why_not`, so the refusal is one
    #: message wherever it is met; a triple is held per component.
    above: Tuple[Any, ...] = ()
    #: THE SAME, INCLUSIVE: the value must be at least ``bound`` -- an engine
    #: whose rule is *larger than or equal to* (TranSIESTA's Fermi cutoff,
    #: 2.5 kT, `m_ts_contour_neq.f90`), so the limit refuses exactly what the
    #: engine refuses (§ 5.3, 2.1.0).
    at_least: Tuple[Any, ...] = ()
    #: WHICH WAY A LADDER TIGHTENS THIS ITEM (plan § 5w K4): ``"down"`` -- a
    #: tighter setting is a smaller number (a tolerance) -- or ``"up"`` (a
    #: mesh cutoff); ``""`` when no tier table says (`engines/tuning.md`
    #: § 2), and then a ladder's order is not judged on it.  Read by the
    #: ladder check (`engines/stages.md` § 4 R3) on every engine.
    tightens: str = ""
    #: THE CHOICES A KIND MAY TAKE (§ 6.3a, 2026-09-30), as ``(key, members)``
    #: pairs -- ``recommended``'s sibling: that says where a kind's value
    #: starts, this which values the kind can take at all.  A key is a kind
    #: (``"vibration"``, every engine) or ``"<engine>.<kind>"`` (one engine,
    #: for an item two engines share whose choices differ -- the TOML dotted
    #: key ``siesta.transport``).  No entry for the kind means every choice.
    #: Ask :func:`offered`; why a choice is not offered is
    #: :func:`why_not_offered`'s, from :attr:`not_offered`.
    offered: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()
    #: WHY A CHOICE ``offered`` WITHHOLDS IS WITHHELD (§ 6.3a), one row per
    #: reason (:class:`NotOffered`), the first that holds the one a refusal
    #: gives (:func:`why_not_offered`).  Catalogue-only, like ``result``:
    #: never written into a calculation's template.
    not_offered: Tuple[NotOffered, ...] = ()
    #: WHICH RESULT OF THE KIND THIS SETTING IS FOR (§ 6.2a), as ``(kind,
    #: result)`` pairs, each result one its kind's record gives
    #: (`kind_table`) --
    #: the card it is drawn in (:func:`card_of`).  Where it is drawn and
    #: nothing else: a catalogue fact, never written into a calculation's
    #: template.
    result: Tuple[Tuple[str, str], ...] = ()
    #: THE VALUES OTHER SETTINGS MUST HOLD for this one to enter the
    #: calculation (§ 5), as ``(name, values)`` pairs, every one holding --
    #: *Raman for* only while *Compute Raman* is on.  Otherwise it enters
    #: nothing: the form greys it out, keeping its value, and the settings
    #: gate says so (:func:`inactive_why`).  Catalogue-only, like
    #: ``result``.
    active_when: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()
    #: THE WORDS A PERSON READS FOR EACH CHOICE (§ 5), as ``(choice,
    #: words)`` pairs -- *All modes* for ``all``: what a form's list shows
    #: and a message says (:func:`choice_words`); the value written stays
    #: the choice.  Catalogue-only, like ``result``.
    choice_labels: Tuple[Tuple[Any, str], ...] = ()

    #: Citation keys into ``docs/science/references.bib`` -- the one
    #: bibliography the whole validation design argues from (user,
    #: 2026-08-21).  Surfaced under the form's help expander (resolved
    #: to title/DOI server-side) and pinned by test: a key that does
    #: not resolve in the bib fails CI, so an invented citation cannot
    #: reach a user.  The engine-manual half of a citation is the
    #: sibling ``manual`` key above.
    refs: Tuple[str, ...] = ()

    #: **THE machine-answered flag.  One parameter, written to the file.**
    #:
    #: True means the scheduler answers this, not the person -- so a portable
    #: description may state the question and never the answer (§ 2, G1).
    #:
    #: Ask it directly, or filter for it: ``select(t, allocation=True)``.
    allocation: bool = False

    #: **A row on every run card from the start**, blank until stated
    #: (`web/task-setup.md` § 6.2b; user, 2026-10-11) -- marked on every run
    #: setting of the catalogue.  It states no value and decides nothing
    #: prep does; Task setup's run card reads it.
    run_card: bool = False

    #: **The CITATION answers this one** — `allocation`'s sibling on the same
    #: axis (§ 6.4), added 2026-09-15 on the user's ruling for the transport
    #: kind.
    #:
    #: Same three states, different answerer.  A transport calculation takes
    #: its basis, functional, mesh, energy shift, transverse k and electronic
    #: temperature from the **cited relaxation's own deck** — never typed,
    #: because the leads and the device must not be able to disagree about
    #: them.  So the item is DECLARED here (a surface can show it, and a
    #: reader learns the calculation has such a parameter) and never
    #: ANSWERED here; `prep` fills it from the citation on the machine that
    #: runs the job, exactly as it fills an allocation item from what the
    #: scheduler granted.
    #:
    #: Any later kind whose values arrive from a cited result inherits the
    #: behaviour by declaring it (`engines/transport.md` § 3.3.3).
    #:
    #: **A list of CALCULATION KINDS**, like `calculations` — not a boolean.
    #: `basis_size`, `mesh_cutoff`, `kgrid`, `pao_energy_shift`,
    #: `electronic_temperature`, `xc_functional` and `xc_authors` are the
    #: SAME catalogue rows an optimization uses, where the PERSON answers
    #: them.  Only for the transport kind does the citation.  So the item
    #: says *for which kinds*, and absence means *never*.
    #:
    #: Ask it directly, or filter for it: ``select(t, citation=True)``.
    citation: Tuple[str, ...] = ()

    #: **NOBODY IS ASKED: THE RUNG FIXES IT** — the third answerer on
    #: `allocation`'s axis (§ 6.4), added 2026-09-16 for the transport kind.
    #:
    #: Same three states again, and again a different answerer. Some values
    #: are not a choice at all: a transport device solves with the NEGF
    #: method and a bulk lead does not; a lead writes the Hamiltonian the
    #: device will read; and every SIESTA run writes each step's forces and
    #: coordinates, because molbuilder reads them back (2026-09-29). Offering
    #: any of these as a control would present a choice with exactly one
    #: correct answer, and a person who changed it would not be tuning the
    #: run — they would be stopping it being the run it is.
    #:
    #: **A list of CALCULATION KINDS**, like `citation` and for the same
    #: reason: `solution_method` is role-fixed for *transport*, where the rung
    #: decides it, and an ordinary person-answered choice for an
    #: *optimization*, where nothing else does. So the item says *for which
    #: kinds*, and absence means *never*.
    #:
    #: What it buys: a `role` item is not offered — not as a form field, not
    #: as a stage-table column — it carries no value in a template of that
    #: kind, and no door may set it; each rung's answer
    #: (:func:`role_answers`) is put into the rung's config by `resolve`.
    #:
    #: Ask it directly, or filter for it: ``select(t, role=True)``.
    role: Tuple[str, ...] = ()
    #: **A RUNG WHOSE ROLE ANSWERS OTHERWISE**, as ``(rung, value)`` pairs:
    #: every other rung of a `role` kind answers ``value``.  The device's
    #: ``transiesta`` is the case (§ 6.4).  Pairs, as ``recommended``'s, so
    #: the frozen item stays hashable.
    role_values: Tuple[Tuple[str, Any], ...] = ()
    #: **BINDS EVERY RUNG of these kinds** -- a scope, not an answerer
    #: (`engines/transport.md` § 3.8.6, decided 2026-09-24: a sibling marker
    #: beside `citation`, which keeps meaning *who supplies the default*).
    #:
    #: A transport calculation is five rungs on one electronic description:
    #: the lead self-energy attaches to a device Hamiltonian built the same
    #: way, so basis, functional, mesh, energy shift, electronic temperature,
    #: spin treatment, species order, the pseudopotentials and the transverse
    #: k-sampling are decided once and may not be overridden on one rung.
    #: Every `citation` row is shared; Class A is larger -- `species_order`,
    #: `spin_treatment`, `unpaired_electrons` and the pseudopotentials bind
    #: every rung.  Filtering on `citation` for "shared" would offer those as
    #: per-rung overrides, and a species order that differs
    #: between the leads and the device is the disagreement
    #: `model/chemistry.md` § 3a exists to make impossible.
    #:
    #: **And every kind, not only transport** (M6, 2026-09-28): the
    #: electronic state's four items are `shared` for each kind that has
    #: them, because the state belongs to the calculation, never to a stage
    #: (`science/chemistry-correctness.md` § 2a, ES1) -- every rung's warm
    #: files are a density for one electronic state.
    #:
    #: Three readers ask this one declaration: the shared panel (which shows
    #: these and nothing else), the per-rung form (which never offers them)
    #: and `prep` (which refuses a stage override naming one).  A list of
    #: CALCULATION KINDS, like `citation`; absence means *never*.
    #:
    #: Ask it directly, or filter for it: ``select(t, shared=True)``.
    shared: Tuple[str, ...] = ()

    #: **WHICH KINDS' FORMS DRAW THIS OPTIONAL ITEM AS REQUIRED** (plan § 5w
    #: K20).  Unset stays a state the item has; on these kinds the Send
    #: refuses until the question is answered, so the form marks the control
    #: required and red while it is empty -- the pseudopotential folder on a
    #: SIESTA optimization or vibration.  A list of CALCULATION KINDS, like
    #: `shared`; absence means *not required*.  It refuses nothing itself.
    required: Tuple[str, ...] = ()

    #: **WHICH RUNGS may carry their own value for this item.**
    #:
    #: `calculations` one level down: that says which KINDS have this
    #: parameter at all, and this says which STAGES of such a kind may answer
    #: it differently from each other. Absent means *any rung may* — which is
    #: the ordinary case and the optimization ladder's behaviour, where any
    #: promoted field may vary per rung.
    #:
    #: It exists because a composite kind's rungs are different programs on
    #: different cells rather than one calculation tuned N ways
    #: (`engines/transport.md` § 2a.3). A transmission energy window is the
    #: transmission rung's and nothing else's; a lead's transport-axis
    #: k-density is an electrode's. Without this there is no basis on which to
    #: route an override to the rung that owns it, and a surface has to guess.
    #:
    #: THE NAMES ARE ROLES (the kind table's, plan § 5w K4): a transport
    #: rung by its name, a vibration's ``relaxation`` or ``force_constants``
    #: whatever the stage is called (:func:`stage_roles`).  Each kind reads
    #: its own roles here, and a kind without roles reads every item on every
    #: rung -- so the relaxation settings declare ``["relaxation"]`` once, a
    #: vibration's relax rung's alone and every rung's of an optimization.
    #:
    #: Ask it directly, or filter for it: ``select(t, stages="device")``;
    #: whether one rung reads an item is :func:`reads`.
    stages: Tuple[str, ...] = ()

    #: Whether *unset* is a state this item has at all — and it IS written
    #: to the file.  A surface must offer *(auto)* / *(no cap)*, and it
    #: cannot be inferred from ``null_label``: some optional items carry
    #: none, and would silently lose the option.  (How many is stated once,
    #: in ``engines/template.md`` § 5 and ``web/form-schema.md`` § 1.2.)
    optional: bool = False

    #: ``basic`` or ``advanced`` — a judgement about the PARAMETER, so a
    #: surface can dim the advanced ones rather than asking a first-time reader
    #: to weigh every knob at once.  Empty means unclassified.
    tier: str = ""

    #: A regex the value must match.  Two items have one (``system_label``,
    #: ``job_name``); nothing else in § 5's vocabulary can say *"letters,
    #: digits, hyphens, underscores; no dots"*.
    pattern: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            _refuse(f"kind {self.kind!r} is not one of {', '.join(KINDS)}",
                    where=self.name)
        if self.type not in TYPES:
            _refuse(f"type {self.type!r} is not one of {', '.join(TYPES)}",
                    where=self.name)
        # § 3's conditionally-required keys. Each is required *by the item's own
        # declaration*, so the check belongs on the object and not only on the
        # parse -- a producer building Items by hand gets the same refusal.
        for _c in self.calculations:
            if not isinstance(_c, str) or not _c.strip():
                _refuse("'calculations' entries must be non-empty strings",
                        where=self.name)
        for _r in self.refs:
            if not isinstance(_r, str) or not _r.strip():
                _refuse("'refs' entries must be non-empty citation keys",
                        where=self.name)
        if self.kind == "engine" and not self.anchor:
            _refuse("kind='engine' needs an 'anchor' -- an engine item that "
                    "names no keyword cannot reach the deck", where=self.name)
        if self.kind == "deck" and not self.expands:
            _refuse("kind='deck' needs 'expands' -- it is how a reader learns "
                    "which keywords this item produces", where=self.name)
        if self.type == "enum" and not self.choices:
            _refuse("type='enum' needs 'choices' -- an enum with no members "
                    "cannot be validated or rendered as a control",
                    where=self.name)
        # § 3's fourth REQUIRED key, enforced here for the same reason as the
        # conditional three above: a producer building Items by hand must get
        # the same refusal the parser gives.
        if not self.category:
            _refuse("no 'category' -- § 3 requires one on every item, and "
                    "without it a surface has no panel to put this on "
                    f"(the vocabulary is {', '.join(CATEGORIES)})",
                    where=self.name)
        for _c in self.category:
            if _c not in CATEGORIES:
                _refuse(f"category {_c!r} is not one of "
                        f"{', '.join(CATEGORIES)}", where=self.name)
        # § 6.1: `read_by` names LAYERS, so it draws from the same closed
        # vocabulary `kind` does.
        for _r in self.read_by:
            if _r not in KINDS:
                _refuse(f"read_by names {_r!r}, which is not a layer -- the "
                        f"vocabulary is {', '.join(KINDS)} (§ 6.1)",
                        where=self.name)
        # § 6.4: `stages` names ROLES of the kinds that carry the item -- a
        # name no such kind has is a misspelling that would let no rung read
        # it, and a role-bearing kind the item DECLARES that it names none of
        # would carry an item none of its rungs reads.  Both refused, by
        # name.  (A calculation's template is the catalogue narrowed to one
        # kind and states no `calculations`; its names are held to every
        # kind's roles.)
        if self.stages:
            _with = [k for k in (self.calculations or _kinds.kinds())
                     if _roles_of(k)]
            _known = {r for k in _with for r in _roles_of(k)}
            for _s in self.stages:
                if _s not in _known:
                    _refuse(f"stages names {_s!r}, which is no rung role of "
                            f"the kinds that carry this item -- they are "
                            f"{', '.join(sorted(_known)) or '(none)'} "
                            f"(engines/template.md § 6.4)", where=self.name)
            for _k in (k for k in self.calculations if _roles_of(k)):
                if not set(self.stages) & set(_roles_of(_k)):
                    _refuse(f"stages names no rung of {_kinds.kind(_k).with_article}, which "
                            f"carries this item -- none of its rungs would "
                            f"read it (engines/template.md § 6.4)",
                            where=self.name)
        # `group` is OPTIONAL (presentation), but when present it is closed --
        # a typo would put the item on no card at all, which renders it loose
        # below the form and sends every finding about it to the residual
        # panel.  Silent, and indistinguishable from "deliberately untagged".
        if self.group is not None and self.group not in GROUPS:
            _refuse(f"group {self.group!r} is not one of "
                    f"{', '.join(GROUPS)}", where=self.name)
        # A rung's own answer answers a `role` -- without one the table is a
        # value nothing asks for (§ 6.4).
        if self.role_values and not self.role:
            _refuse("'role_values' names a rung's own answer to a `role` item, "
                    "and this item declares no `role`", where=self.name)
        # § 6.2a: a result is one the kind gives -- a misspelling would draw
        # the setting on no card and send its warnings nowhere.
        for _k, _r in self.result:
            if _r not in {r.key for r in _results_of(_k)}:
                _refuse(f"result.{_k} = {_r!r} is not one of the results "
                        f"{_kinds.kind(_k).with_article} gives "
                        f"({', '.join(r.key for r in _results_of(_k)) or 'none listed'}; "
                        f"the kind table)", where=self.name)

    @property
    def is_set(self) -> bool:
        """Whether this item carries a value (§ 3's *absent means unset*)."""
        return self.value is not None


@dataclass(frozen=True)
class Template:
    """A parsed template: the top-level keys, and the items in order.

    ``schema`` and ``engines``, and ``written_by``.
    """
    #: Every engine this calculation can run on (§ 6.3).  A @1 file yields a
    #: one-element list.
    engines: Tuple[str, ...] = ()
    items: Tuple[Item, ...] = ()
    #: THE MOLBUILDER VERSION THAT WROTE THE FILE (§ 3, since 2.1.0) --
    #: ``None`` for a file written before, which said nothing.  What a
    #: reader asks when an item's meaning changed at a version
    #: (:data:`ZERO_WAS_NOT_SET`).
    written_by: Optional[str] = None

    # Read the items through ``select(t, ...)``; ask for one through
    # ``one(t, name, engine=...)`` (`engines/template.md` § 2.2, § 8.0).


# --------------------------------------------------------------------- #
#  From the schema — the edge that makes this data-driven               #
# --------------------------------------------------------------------- #

#: The two spellings of a union.  ``Optional[X]`` and ``X | None`` mean the same
#: thing and report DIFFERENT origins -- ``typing.Union`` and ``types.UnionType``
#: -- so a check for one silently misses the other.  Nothing in the configs uses
#: the newer spelling today, which is exactly why this is worth naming: the first
#: field written that way would be declared NOT optional and then fail to get a
#: type at all, and the error would point at the annotation rather than at the
#: check that could not read it (audit § 1.1).
_UNION_ORIGINS = (typing.Union, types.UnionType)


def _unwrap_optional(ann) -> Tuple[Any, bool]:
    """``Optional[X]`` / ``X | None`` → ``(X, True)``; anything else → ``(ann, False)``.

    ``X`` may itself be a union: ``Optional[Union[int, str]]`` -- the
    electronic state's count, a whole number or ``free`` -- is optional with
    ``Union[int, str]`` inside.
    """
    if typing.get_origin(ann) in _UNION_ORIGINS:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) < len(typing.get_args(ann)):
            return (args[0] if len(args) == 1
                    else typing.Union[tuple(args)]), True
    return ann, False


_BARE_ANCHOR = re.compile(r"^\s*(%block\s+[A-Za-z][\w.-]*|[A-Za-z][\w.-]*)")


def _bare_anchor(engine_key: str) -> str:
    """The keyword an ``engine_key`` leads with, or ``""`` if it leads with none.

    § 5: an anchor is *"a bare keyword, never a sentence"*, but ``engine_key``
    is older than that rule and carries three other shapes, measured against
    ``SiestaConfig``: a **note** (``(molbuilder: ...)``), an **alternation**
    (``A | B``), and a **conjunction** (``A + B``). The last two are ``deck``
    items whose ``expands`` lists every keyword they may produce, declared in
    metadata; the first is not an engine item at all and is refused above.

    What is left for this function is the common case and one tidy-up: a bare
    keyword, possibly followed by a prose note in parentheses, of which only
    the keyword is the anchor.
    """
    m = _BARE_ANCHOR.match(engine_key or "")
    return m.group(1) if m else ""


# --------------------------------------------------------------------- #
#  Writing                                                              #
# --------------------------------------------------------------------- #

def _toml_basic(s: str) -> str:
    """A TOML basic string — one line, escapes processed."""
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return f'"{out}"'


def _toml_multiline(s: str) -> str:
    """A TOML multi-line basic string, for prose that carries newlines.

    Multi-line **basic** (``\"\"\"``) rather than literal (``'''``) because a
    literal string has no escape at all: a body containing ``'''`` could not be
    written, and help text is prose we do not control the punctuation of.
    """
    body = s.replace("\\", "\\\\").replace('"""', '\\"""')
    # A body ending in a quote would merge with the closing delimiter.
    if body.endswith('"'):
        body = body[:-1] + '\\"'
    # TOML trims a newline immediately after the opening delimiter, so the
    # value is exactly ``body`` -- which the round-trip check below proves.
    return f'"""\n{body}"""'


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):                 # before int -- a bool IS an int
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        # repr keeps the decimal point, so a float round-trips as a float and
        # not as an int -- 300.0 must not come back as 300.
        return repr(v)
    if isinstance(v, str):
        return (_toml_multiline(v) if "\n" in v else _toml_basic(v))
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, Mapping):
        # An INLINE TABLE -- ``recommended`` (per kind) and ``role_values``
        # (per rung) are the item keys that hold one, keyed by a name.
        return ("{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}"
                                 for k, x in v.items()) + " }")
    raise TypeError(f"template: cannot write {type(v).__name__} to TOML")


def _toml_key(k: Any) -> str:
    """A table key -- bare when TOML allows it, quoted otherwise."""
    k = str(k)
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _toml_basic(k)


#: The order keys appear inside an item.  Fixed so two templates of the same
#: calculation diff cleanly, and so the file reads the way § 4.2's example does:
#: what it is, then what it is worth, then what bounds it, then the prose.
_ITEM_KEY_ORDER = ("kind", "category", "engines", "calculations", "refs", "anchor", "engine_key",
                   "manual", "expands", "launch", "type",
                   "choices", "value", "source", "default", "engine_default", "recommended", "offered", "not_offered", "result", "active_when", "choice_labels", "optional", "required",
                   "allocation", "run_card",
                   "citation", "shared", "role", "role_values", "stages",
                   "unit", "range", "above", "at_least", "tightens", "tier",
                   "pattern",
                   "group", "label", "null_label", "not_set", "read_by", "help")


def _launch_from(name: str, raw) -> Dict[str, Tuple[str, ...]]:
    """``launch`` read: a table of program -> list of words, every word a
    string; anything else refused naming the item (§ 5)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(
            isinstance(p, str) and isinstance(ws, list)
            and all(isinstance(w, str) for w in ws)
            for p, ws in raw.items()):
        raise ValueError(
            f"template: {name}: launch must be a table of program -> list "
            f"of words (launch.siesta = [\"-fdf\", ...]); got {raw!r}")
    return {p: tuple(ws) for p, ws in raw.items()}


def _item_payload(it: Item) -> Dict[str, Any]:
    """The mapping one item becomes — omitting everything absent (§ 3)."""
    out: Dict[str, Any] = {"kind": it.kind, "type": it.type}
    if it.anchor:
        out["anchor"] = it.anchor
    if it.engine_key:
        out["engine_key"] = it.engine_key
    if it.manual:
        out["manual"] = it.manual
    if it.expands:
        out["expands"] = list(it.expands)
    if it.launch:
        out["launch"] = {p: list(w) for p, w in it.launch.items()}
    if it.choices:
        out["choices"] = list(it.choices)
    # An absent ``value`` is the encoding of *explicitly unset* (§ 3).
    if it.value is not None:
        out["value"] = list(it.value) if isinstance(it.value, tuple) else it.value
    if it.source is not None:
        out["source"] = it.source
    if it.default is not None:
        out["default"] = (list(it.default) if isinstance(it.default, tuple)
                          else it.default)
    if it.optional:
        out["optional"] = True
    if it.tier:
        out["tier"] = it.tier
    if it.pattern:
        out["pattern"] = it.pattern
    if it.unit:
        out["unit"] = it.unit
    if it.range:
        out["range"] = list(it.range)
    if it.above:
        out["above"] = {"value": it.above[0], "why": it.above[1]}
    if it.at_least:
        out["at_least"] = {"value": it.at_least[0], "why": it.at_least[1]}
    if it.tightens:
        out["tightens"] = it.tightens
    if it.group:
        out["group"] = it.group
    if it.category:
        out["category"] = list(it.category)
    if it.engines:
        out["engines"] = list(it.engines)
    if it.calculations:
        out["calculations"] = list(it.calculations)
    if it.recommended:
        out["recommended"] = {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in it.recommended}
    if it.offered:
        # One table per engine for an engine's own entries (the TOML dotted
        # key `siesta.transport` is exactly that table), a list per kind.
        offered_table: Dict[str, Any] = {}
        for key, members in it.offered:
            engine, _, kind = key.rpartition(".")
            (offered_table.setdefault(engine, {}) if engine
             else offered_table)[kind] = list(members)
        out["offered"] = offered_table
    if it.result:
        out["result"] = dict(it.result)
    if it.active_when:
        out["active_when"] = {k: list(v) for k, v in it.active_when}
    if it.choice_labels:
        out["choice_labels"] = {str(c): w for c, w in it.choice_labels}
    if it.refs:
        out["refs"] = list(it.refs)
    if it.allocation:
        out["allocation"] = True
    if it.run_card:
        out["run_card"] = True
    if it.citation:
        out["citation"] = list(it.citation)
    if it.role:
        out["role"] = list(it.role)
    if it.role_values:
        out["role_values"] = {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in it.role_values}
    if it.shared:
        out["shared"] = list(it.shared)
    if it.required:
        out["required"] = list(it.required)
    if it.stages:
        out["stages"] = list(it.stages)
    if it.label:
        out["label"] = it.label
    if it.null_label:
        out["null_label"] = it.null_label
    if it.not_set:
        out["not_set"] = it.not_set
    if it.read_by:
        out["read_by"] = list(it.read_by)
    out["help"] = it.help
    return out


#: The MASTER file — every parameter both engines declare (§ 4.3).  Shipped
#: with the package; authored as TOML, never generated.
CATALOGUE = _Path(__file__).resolve().parent / "data" / "catalogue.template.toml"


def load_catalogue() -> str:
    """The catalogue's text (§ 4.3).

    One master file, authored rather than generated: *"where is this parameter
    defined?"* has one answer for every parameter in every engine.
    """
    return CATALOGUE.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def catalogue() -> "Template":
    """**The parsed catalogue** — the one door for it, parsed once.

    `select` and `one` are the one door for an ITEM, and this is the one door
    for the file they read from.

    Cached because the catalogue ships with the package and does not change
    inside a process.  A caller holding a DIFFERENT catalogue text -- a test
    fixture, a calculation's own template -- calls :func:`read_template` on it
    directly; that is a different file and not this one.
    """
    parsed = read_template(load_catalogue())
    # THE CATALOGUE'S ENGINES AND KINDS ARE THE TABLES' (`execution/
    # architecture.md` § 3.3): the file serves every engine the engine
    # table holds and no other, and every kind an item names is one the
    # kind table holds -- a name with no record is a row nothing asks.
    from .engine_table import engines as _engines
    if set(parsed.engines) != set(_engines()):
        _refuse(f"engines = {list(parsed.engines)}, and the engine table "
                f"holds {', '.join(_engines())} -- one list of engines, the "
                f"engine table's")
    for it in parsed.items:
        for key in ("calculations", "citation", "role", "shared",
                    "required"):
            stray = [k for k in getattr(it, key) if k not in _kinds.kinds()]
            if stray:
                _refuse(f"{key} names {', '.join(map(repr, stray))}, which "
                        f"the kind table does not hold "
                        f"({', '.join(_kinds.kinds())})", where=it.name)
    # A KIND IS NEVER STARTED ON A CHOICE IT REFUSES (§ 6.3a): the value a
    # kind's template starts at -- its recommendation, else the default --
    # is among what that kind is offered.  Asked of the MASTER FILE, once:
    # an item derived for one kind's form (`with_recommended`) carries that
    # kind's start as its default, which other kinds' sets need not hold.
    for it in parsed.items:
        rec = dict(it.recommended)
        for key, members in it.offered:
            kind = key.rpartition(".")[2]
            start = rec.get(kind, it.default)
            if start is not None and not is_member(start, members):
                _refuse(f"a {kind} starts at {start!r}, which offered.{key} "
                        f"does not let it take", where=it.name)
        # WHAT THE ENGINE DOES WHEN NOTHING IS WRITTEN IS SAID (§ 5,
        # § 6.6 obligation 4): an optional item's deck line can be left
        # out, and the deck then says which default applies -- in this
        # item's own words, so none may be missing.
        if (it.optional and it.kind in ("engine", "deck")
                and not it.allocation and not it.not_set):
            _refuse("an optional item the deck may leave out declares no "
                    "not_set -- what the engine does when it writes nothing "
                    "(engines/template.md § 5)", where=it.name)
        # THE ENGINE'S OWN VALUE IS STATED, OR WHAT IT FOLLOWS (§ 5): every
        # SIESTA item whose anchor is a keyword.  Two decks are compared as
        # SIESTA reads them (`siesta.engine_input`), and a keyword one deck
        # states at the engine's own value is the same input as one that
        # leaves it out -- so an item that states neither makes every older
        # run another calculation the day a deck starts writing it.
        if (it.anchor and not it.anchor.startswith("%block")
                and (not it.engines or "siesta" in it.engines)
                and not it.engine_default and not it.engine_unfixed):
            _refuse("a SIESTA item with a keyword states its engine_default "
                    "-- the engine's value and the line of its source, or "
                    "what that value follows (engines/template.md § 5)",
                    where=it.name)
        # A RUN SETTING STATES NO `value` (§ 5.4): no template starts from
        # one -- its default is `default`, asked through `default_of`.
        if is_run_setting(it) and it.value is not None:
            _refuse("a run setting states no value -- its default is "
                    "`default`, the one every reader asks "
                    "(engines/template.md § 5.4)", where=it.name)
        # WHAT GREYS A SETTING OUT IS A SETTING OF THE SAME CALCULATION (§ 5):
        # an item serving every engine and kind this one does, each value
        # one its own type takes -- else the form would wait on a field it
        # never draws, and the gate on a value no config can hold.
        by_name = {o.name: o for o in parsed.items}
        for other_name, values in it.active_when:
            other = by_name.get(other_name)
            engines_here = it.engines or parsed.engines
            if other is None or any(
                    other.engines and e not in other.engines
                    for e in engines_here) or any(
                    other.calculations and k not in other.calculations
                    for k in (it.calculations or _kinds.kinds())):
                _refuse(f"active_when names {other_name!r}, which is not an "
                        f"item of every engine and kind this one serves "
                        f"(engines/template.md § 5)", where=it.name)
            for v in values:
                _check_raw_value(it.name, f"active_when.{other_name}", v,
                                 other.type, other.choices)
    return parsed


def template_with_values(config, *, engine: str = "", catalogue: str = "",
                         calculation: str = "optimization",
                         title: str = "",
                         valueless: Sequence[str] = (),
                         sources: Optional[Mapping[str, str]] = None) -> str:
    """**This calculation's** template: the catalogue, narrowed to one engine,
    carrying the values *config* holds (§ 4.3).

    This is what a surface produces once a person has answered — the questions
    were already asked by the catalogue, so nothing here invents an item.

    **No run setting is written, valueless or not** (§ 2, G1): how a run
    runs -- the machine's answers and a person's alike -- is the job
    request's (`execution/job-request.md` § 2), and a description holding a
    rank count would assert a machine fact and stop being portable.

    **Each value says where it came from** (`source`, § 6.6 obligation 2,
    plan § 5w K7): ``sources`` is what the CALLER knows -- a cited run's
    answers, the structure's record, what the person's form sent -- and every
    other item is ``default``, *not chosen*.  A source of ``None`` writes
    none -- *not recorded*: what a migration says of the values it carries
    from a file that never recorded one, rather than a guess.  An item the
    template holds no value of by rule (the rung's) carries no source.

    Raises ``ValueError`` if the emitted text does not parse back to what it
    meant to write — § 4.1 asks for exactly this.
    """
    parsed = read_template(catalogue or load_catalogue())
    told = dict(sources or {})
    bad = sorted({v for v in told.values() if v is not None} - set(SOURCES))
    if bad:
        raise ValueError(f"source(s) {', '.join(map(repr, bad))} are not "
                         f"one of {', '.join(SOURCES)} (§ 6.6)")
    eng = engine or engine_name(type(config))
    items = [
        dataclasses.replace(
            it,
            # A per-calculation template serves ONE engine AND one kind, so
            # no item narrows anything and none carries `engines` or
            # `calculations` (§ 6.3's writer rule) -- the selection already
            # happened, here.
            engines=(),
            calculations=(),
            # Valueless for the answerer that is not floor 2's: the stage's
            # ROLE (§ 6.4) -- a role item has no value a description could
            # legitimately claim to have set, because the rung decides it.
            #
            # `citation` is NOT among them: the cited run **defaults** these
            # values and the person may change them, everywhere at once
            # (`engines/transport.md` § 2a.7), so what it produces at `molbuilder describe`
            # is an ordinary template value.
            # AND the rows the caller says nobody answered (``valueless``,
            # `engines/transport.md` § 3.8.3): a citation that carries no
            # deck answers no basis, and writing the class default there
            # would claim a run said something no run said.
            value=(None
                   if calculation in it.role or it.name in set(valueless)
                   else getattr(config, it.name, it.value)),
            source=(None if calculation in it.role
                    else told.get(it.name, "default")),
            # THE KIND'S RECOMMENDATION IS THE TEMPLATE'S DEFAULT (§ 6.3a);
            # the table itself does not travel -- a per-kind template has
            # already answered which kind it serves.
            default=(recommended_for(it, calculation)
                     if recommended_for(it, calculation) is not None
                     else it.default),
            recommended=(),
            # ...nor the rungs' own role answers: `resolve` reads them from
            # the catalogue, and the template holds no value for a role item.
            role_values=(),
            # ...nor the kinds' choice sets: `resolve` and the gate hold a
            # value to the catalogue's (§ 6.3a), naming why a choice is not
            # this kind's -- a narrowed `choices` here would refuse it at
            # read time with the reader's bare words instead.
            offered=(),
            # ...nor where the form draws it or what greys it out (§ 6.2a,
            # § 5): the catalogue's, asked of the catalogue.
            result=(),
            active_when=(),
            choice_labels=(),
        )
        for it in select(parsed, engine=eng, calculation=calculation)
        # NO RUN SETTING, valueless or not: it is the job request's
        # (`execution/job-request.md` § 2), and a home that must stay empty
        # is removed, never policed (§ 2).
        if not is_run_setting(it)
    ]
    return _emit(items, engines=(eng,), title=title)


def title_of(text: str) -> str:
    """A template's header comment, without its ``#`` -- the title
    ``molbuilder describe`` wrote, carried over unchanged by every writer
    that writes the file again."""
    lines = []
    for ln in text.splitlines():
        if not ln.startswith("#"):
            break
        lines.append(ln[2:] if ln.startswith("# ") else ln[1:])
    return "\n".join(lines)


def with_values_set(text: str, config_cls, *, calculation: str,
                    values: Mapping[str, Any], source: str) -> str:
    """A calculation's template with ``values`` set, each said to come from
    ``source`` (§ 6.6 obligation 2) -- every other item's value, blank and
    source as ``text`` holds them, and its title, carried over -- through
    the one writer (:func:`template_with_values`)."""
    items = read_template(text).items
    cfg = dataclasses.replace(config_from_template(text, config_cls),
                              **values)
    sources = {it.name: it.source for it in items}
    sources.update({k: source for k in values})
    return template_with_values(
        cfg, engine=engine_name(config_cls), calculation=calculation,
        title=title_of(text), sources=sources,
        valueless=[it.name for it in items
                   if it.value is None and it.name not in values])


def _emit(items, *, engines, title: str = "") -> str:
    """Serialise items as a template file, and read the result back (§ 4.1).

    ``tomllib`` reads TOML and does not write it, so this is the only thing
    standing between a quoting bug and a file that parses cleanly and says
    something else.
    """
    from . import __version__
    payload = {
        "schema": SCHEMA,
        "engines": list(engines),
        # WHICH MOLBUILDER WROTE IT (§ 3): a later reader names a file by it
        # when an item's meaning changed since (`process/conventions.md`
        # § 2, *Versioning*).
        "written_by": __version__,
        "item": {it.name: _item_payload(it) for it in items},
    }

    lines: List[str] = []
    if title:
        for ln in title.splitlines():
            lines.append(f"# {ln}".rstrip())
    lines.append(f"schema      = {_toml_value(SCHEMA)}")
    lines.append(f"engines     = {_toml_value(list(engines))}")
    lines.append(f"written_by  = {_toml_value(payload['written_by'])}")
    for it in items:
        body = _item_payload(it)
        lines.append("")
        lines.append(f"[item.{it.name}]")
        for key in _ITEM_KEY_ORDER:
            if key in body:
                lines.append(f"{key} = {_toml_value(body[key])}")
    text = "\n".join(lines) + "\n"

    try:
        got = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:      # pragma: no cover - defensive
        raise ValueError(
            f"template: emitted TOML does not parse ({exc}). This is a bug in "
            f"the writer, not in the data.") from exc
    if got != payload:
        raise ValueError(
            f"template: emitted TOML does not read back as written -- "
            f"{_first_difference(payload, got)}. This is a bug in the writer "
            f"(engines/template.md § 4.1 asks for exactly this check).")
    return text


def items_of(cls_or_cfg) -> "Dict[str, Item]":
    """THE CATALOGUE'S ITEMS FOR A CONFIG, by field name -- the one home of
    every fact about a parameter (§ 2.1: a config class is a translator; its
    field's name, type, default and validators are its own, and every other
    fact -- the label, the unit, the range, the choices, the engine keyword,
    the card, the category, whether the machine answers it -- is the
    catalogue's).  Asked with the class or an instance; the engine is the
    class's (:func:`engine_name`).  Every reader of such a fact asks this:
    the settings gate, the description check, the form's coercion, finding
    placement, the template's membership."""
    cls = cls_or_cfg if isinstance(cls_or_cfg, type) else type(cls_or_cfg)
    engine = engine_name(cls)
    # THE SAME RULE `one` READS BY: an item naming no engines is every
    # engine's.
    return {it.name: it for it in catalogue().items
            if not it.engines or engine in it.engines}


def engine_name(cls) -> str:
    """Whose schema these items are — from the class, never guessed by a caller."""
    named = getattr(cls, "ENGINE", "")
    if named:
        return str(named)
    stem = cls.__name__
    for suffix in ("Config", "Configuration"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.lower()


def _first_difference(want: Any, got: Any, path: str = "") -> str:
    """Where two payloads first disagree, named so the bug is findable."""
    if isinstance(want, Mapping) and isinstance(got, Mapping):
        for k in want:
            if k not in got:
                return f"{path}.{k} is missing after the round trip".lstrip(".")
            sub = _first_difference(want[k], got[k], f"{path}.{k}")
            if sub:
                return sub
        for k in got:
            if k not in want:
                return f"{path}.{k} appeared from nowhere".lstrip(".")
        return ""
    if want != got:
        return (f"{path or 'the document'}: wrote {want!r}, read back {got!r}")
    return ""


# --------------------------------------------------------------------- #
#  Reading                                                              #
# --------------------------------------------------------------------- #

_REQUIRED_ITEM_KEYS = ("kind", "category", "type", "help")

#: Where a template value came from (§ 6.6 obligation 2's three states, plan
#: § 5w K7; decision 7 retired the recorded-contract lane): *from the run
#: you cited* · *you set this* · *not chosen*.
SOURCES = ("cited", "person", "default")

#: ...in the words every surface says them -- the one vocabulary, served
#: with the form schema and the Task setup hover (`form-schema.md` § 1.1).
#: ``unrecorded`` is a template written before the key existed.
SOURCE_WORDS = {
    "cited":      "from the junction you cited",
    "person":     "you set this",
    "default":    "not chosen",
    "unrecorded": "not recorded",
}


def _source_from(name: str, raw) -> Optional[str]:
    """An item's ``source``, refused by name outside :data:`SOURCES`."""
    if raw is None:
        return None
    if raw not in SOURCES:
        raise ValueError(
            f"item {name!r}: source = {raw!r} is not one of "
            f"{', '.join(SOURCES)} (engines/template.md 6.6)")
    return str(raw)

_KNOWN_ITEM_KEYS = frozenset(_ITEM_KEY_ORDER)


def read_template(text: str) -> Template:
    """Parse a template, refusing rather than guessing.

    The order is § 3's: the schema string first, so a file from a future major
    fails saying so instead of failing on whichever key moved.

    Every refusal names the item it is about, because a template is a file a
    person is invited to edit (§ 4.1: *"hand-editable — yes"*), and *"missing
    required key 'kind'"* with no item name sends them to read the whole file.
    """
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        _refuse(f"not valid TOML -- {exc}")

    check_schema(str(raw.get("schema") or ""), SCHEMA, label="template")

    # § 3 at @2: `engines` is a LIST -- a template describes a CALCULATION and
    # a calculation may run on more than one engine (§ 6.3).  A @1 file names
    # one engine as a string; read it as a single-element list rather than
    # refusing, since that is exactly what it meant.
    engines = raw.get("engines")
    if isinstance(engines, str):            # tolerated @1 spelling
        engines = [engines]
    if engines is None and isinstance(raw.get("engine"), str):
        engines = [raw["engine"]]
    if not engines or not isinstance(engines, list) or \
            not all(isinstance(e, str) and e for e in engines):
        _refuse("missing required key 'engines' -- without it a reader cannot "
                "know which config classes these items belong to, and would "
                "have to infer it from their names (engines/template.md § 3)")
    engines_t = tuple(engines)
    table = raw.get("item") or {}
    if not isinstance(table, Mapping):
        _refuse(f"'item' must be a table of items, got "
                f"{type(table).__name__}")

    items = tuple(_item_from(name, body) for name, body in table.items())

    # § 6.3: an item's `engines` narrows the FILE's list -- it cannot widen
    # it.  An item naming an engine this calculation does not run on is a
    # contradiction between the two, and accepting it would let a surface
    # filter for that engine and be handed items from a calculation that
    # never offered it.  Which half is wrong is not ours to guess, so the
    # reader reports the disagreement and names both sides.
    for _it in items:
        _stray = [e for e in _it.engines if e not in engines_t]
        if _stray:
            _refuse(
                f"declares engines {_stray} that the template does not list "
                f"(engines = {list(engines_t)}). An item's `engines` narrows "
                f"the file's list; it cannot widen it (engines/template.md "
                f"§ 6.3)", where=_it.name)
    written_by = raw.get("written_by")
    if written_by is not None and not (
            isinstance(written_by, str) and _VERSION.fullmatch(written_by)):
        _refuse(f"'written_by' is the molbuilder version that wrote the "
                f"file, three numbers like \"2.1.0\" -- not {written_by!r}")
    return Template(engines=engines_t, items=items, written_by=written_by)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


#: § 5's checkable meaning of each ``type``, for a SET value.
_TYPE_CHECKS = {
    "int":     _is_int,
    "float":   lambda v: _is_int(v) or isinstance(v, float),
    "str":     lambda v: isinstance(v, str),
    "text":    lambda v: isinstance(v, str),
    "bool":    lambda v: isinstance(v, bool),
    # A member is a string or a whole number (§ 5) -- membership, typed, is
    # `_check_raw_value`'s: `2` is a member of 0-10 and `"2"` or `true` is not.
    "enum":    lambda v: isinstance(v, str) or _is_int(v),
    # ``pow2`` CHECKS only that it is an int.  The power-of-two constraint is
    # not a refusal, it is a COERCION -- ``_shape`` snaps the value (user,
    # 2026-08-14: *"instead of accepting and refusing, it should correctly
    # coerce the value to the allowed number"*).  Refusing would send a person
    # who typed 96 away to work out which numbers are legal; snapping tells
    # them, by doing it.
    "pow2":    _is_int,
    "int3":    lambda v: (isinstance(v, (list, tuple)) and len(v) == 3
                          and all(_is_int(x) for x in v)),
    # ``float3`` accepts an int per component: TOML's ``0`` and ``0.0`` are
    # different types to a parser and the same displacement to SIESTA, and a
    # hand-edited file will carry both.  ``_shape`` makes them one.
    "float3":  lambda v: (isinstance(v, (list, tuple)) and len(v) == 3
                          and all(_is_int(x) or isinstance(x, float)
                                  for x in v)),
    "strlist": lambda v: (isinstance(v, list)
                          and all(isinstance(x, str) for x in v)),
    "intlist": lambda v: (isinstance(v, list) and all(_is_int(x) for x in v)),
}


def _table_from(name: str, key: str, raw, type_: str, choices, *,
                keyed_by: str, example: str,
                keys: Optional[Tuple[str, ...]] = None
                ) -> Tuple[Tuple[str, Any], ...]:
    """A per-name table of an item -- ``recommended``, keyed by calculation
    kind (§ 6.3a), or ``role_values``, keyed by rung (§ 6.4) -- checked value
    by value against the item's own declaration: each obeys the same type and
    enum membership as the general default.  ``keys``, when given, is the
    vocabulary a key must come from: a misspelled kind would otherwise be a
    table nothing ever asks."""
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or not all(isinstance(k, str) for k in raw):
        _refuse(f"`{key}` must be a table keyed by {keyed_by}, "
                f"e.g. {example}", where=name)
    if keys is not None:
        unknown = sorted(set(raw) - set(keys))
        if unknown:
            _refuse(f"`{key}` names {', '.join(map(repr, unknown))}, which "
                    f"is not a {keyed_by} ({', '.join(keys)})", where=name)
    out = []
    for k, v in raw.items():
        _check_raw_value(name, f"{key}.{k}", v, type_, choices)
        out.append((str(k), _shape(v, type_)))
    return tuple(out)


def _offered_from(name: str, raw, type_: str, choices,
                  engines) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
    """The ``offered`` table of an item (§ 6.3a) -- a kind's list of the
    choices it may take, or an engine's table of such lists -- each member
    checked against the item's own type and choices, so a kind can only be
    offered what the item declares."""
    if raw is None:
        return ()
    if not choices:
        _refuse("`offered` narrows an item's choices, and this item "
                "declares none", where=name)
    if not isinstance(raw, Mapping) or not raw:
        _refuse('`offered` must be a table keyed by calculation kind, or by '
                'engine and then kind -- e.g. offered = { vibration = '
                '["CG", "FIRE"] } or offered = { siesta.transport = '
                '["restricted"] }', where=name)
    kinds = _kinds.kinds()
    out = []
    for key, val in raw.items():
        if isinstance(val, Mapping):                    # an engine's table
            if engines and key not in engines:
                _refuse(f"`offered.{key}` names an engine this item does not "
                        f"serve ({', '.join(engines)})", where=name)
            if not val:
                _refuse(f"`offered.{key}` names no calculation kind",
                        where=name)
            pairs = [(f"{key}.{kind}", v) for kind, v in val.items()]
        else:
            pairs = [(str(key), val)]
        for k, members in pairs:
            # A MISSPELLED KIND is a table nothing asks, so every refusal
            # for the kind meant would silently go (the K2 review).
            if k.rpartition(".")[2] not in kinds:
                _refuse(f"`offered.{k}` names {k.rpartition('.')[2]!r}, which "
                        f"is not a calculation kind ({', '.join(kinds)})",
                        where=name)
            if not isinstance(members, list) or not members:
                _refuse(f"`offered.{k}` must be a non-empty list of the "
                        f"item's choices", where=name)
            for m in members:
                _check_raw_value(name, f"offered.{k}", m, type_, choices)
            out.append((k, tuple(_shape(m, type_) for m in members)))
    return tuple(out)


def _not_offered_from(name: str, raw, type_: str, choices, engines,
                      offered) -> Tuple[NotOffered, ...]:
    """The ``not_offered`` rows of an item (§ 6.3a), each checked: only on
    an item with ``offered``; ``on`` an engine the item serves, a kind the
    kind table holds, or the pair; every value one of the item's choices;
    ``why`` a sentence."""
    if raw is None:
        return ()
    if not offered:
        _refuse("`not_offered` says why a choice `offered` withholds is "
                "withheld, and this item withholds none", where=name)
    if not isinstance(raw, list) or not raw or not all(
            isinstance(r, Mapping) for r in raw):
        _refuse('`not_offered` is a list of rows -- [[item.<name>.'
                'not_offered]] with on = "siesta", values = [...], why = '
                '"..."', where=name)
    from .engine_table import engines as _all_engines
    served = tuple(engines) or _all_engines()
    kinds = _kinds.kinds()
    out = []
    for i, row in enumerate(raw):
        at = f"not_offered[{i}]"
        unknown = sorted(set(row) - {"on", "values", "why"})
        if unknown:
            _refuse(f"`{at}` has unknown key(s) {', '.join(unknown)} -- a "
                    f"row states on, values and why", where=name)
        why = row.get("why")
        if not isinstance(why, str) or not why.strip():
            _refuse(f"`{at}` states no why -- the reason, read from the "
                    f"engine's source", where=name)
        on = row.get("on", "")
        engine, dot, kind = str(on).partition(".")
        if not isinstance(on, str) or not (
                on == "" or (dot and engine in served and kind in kinds)
                or (not dot and (on in served or on in kinds))):
            _refuse(f"`{at}.on` is {on!r} -- an engine this item serves "
                    f"({', '.join(served)}), a calculation kind "
                    f"({', '.join(kinds)}) or \"<engine>.<kind>\"",
                    where=name)
        values = row.get("values", [])
        if not isinstance(values, list):
            _refuse(f"`{at}.values` is a list of the item's choices",
                    where=name)
        for v in values:
            _check_raw_value(name, f"{at}.values", v, type_, choices)
        out.append(NotOffered(on=on, values=tuple(_shape(v, type_)
                                                  for v in values),
                              why=why.strip()))
    return tuple(out)


def _result_from(name: str, raw) -> Tuple[Tuple[str, str], ...]:
    """The ``result`` table of an item (§ 6.2a): a calculation kind ->
    one of that kind's results.  A key that is no kind is refused, as
    ``offered``'s is -- a table nothing asks; the result itself is checked
    by the item (:class:`Item`)."""
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or not raw or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
        _refuse('`result` must be a table keyed by calculation kind, each '
                'value a result of that kind -- e.g. result = { vibration = '
                '"raman" }', where=name)
    kinds = _kinds.kinds()
    unknown = sorted(set(raw) - set(kinds))
    if unknown:
        _refuse(f"`result` names {', '.join(map(repr, unknown))}, which is "
                f"not a calculation kind ({', '.join(kinds)})", where=name)
    return tuple((str(k), str(v)) for k, v in raw.items())


def _active_when_from(name: str, raw) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
    """The ``active_when`` table of an item (§ 5): another item's name ->
    the values it must hold, a non-empty list.  That each name is an item
    and each value passes its type is the catalogue's to check, holding
    every item (:func:`catalogue`)."""
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or not raw or not all(
            isinstance(k, str) and isinstance(v, list) and v
            for k, v in raw.items()):
        _refuse('`active_when` must be a table of other settings, each a '
                'non-empty list of the values it must hold -- e.g. '
                'active_when = { compute_raman = [true] }', where=name)
    if name in raw:
        _refuse("`active_when` names the item itself", where=name)
    return tuple((str(k), tuple(v)) for k, v in raw.items())


def _choice_labels_from(name: str, raw, choices) -> Tuple[Tuple[Any, str], ...]:
    """The ``choice_labels`` table of an item (§ 5): each choice, as TOML
    spells a key, -> its words.  A key that names no choice, or empty
    words, is refused -- a label nothing shows."""
    if raw is None:
        return ()
    by_key = {str(c): c for c in (choices or ())}
    if not isinstance(raw, Mapping) or not raw or any(
            k not in by_key or not isinstance(v, str) or not v.strip()
            for k, v in raw.items()):
        _refuse("`choice_labels` must be a table of the item's choices, each "
                "with its words -- e.g. choice_labels = { all = \"All "
                "modes\" }", where=name)
    return tuple((by_key[k], v.strip()) for k, v in raw.items())


def _stages_from(name: str, raw) -> Tuple[str, ...]:
    """``stages`` as read (§ 6.4): a list of rung roles, or the name of a
    set of them -- what its rungs compute (`kind_table.COMPUTES`: ``"scf"``,
    every rung that runs an SCF) -- read as its roles.  A name no set has
    is refused, naming the sets."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        if raw not in _kinds.COMPUTES:
            _refuse(f"stages names the set {raw!r}, which is not one -- the "
                    f"sets are what a rung computes, "
                    f"{', '.join(_kinds.COMPUTES)}; or list the roles "
                    f"(engines/template.md § 6.4)", where=name)
        return _kinds.role_set(raw)
    return tuple(raw)


def _tightens_from(name: str, raw) -> str:
    """``tightens`` as read: ``"up"``, ``"down"``, or absent (plan § 5w K4).
    Any other word is refused -- a direction the ladder check cannot read
    would judge a ladder backwards, or not at all, in silence."""
    if raw is None:
        return ""
    if raw not in ("up", "down"):
        _refuse(f"tightens must be \"up\" or \"down\", not {raw!r} -- which "
                f"way a ladder makes this item tighter (engines/stages.md "
                f"§ 4 R3)", where=name)
    return raw


def _above_from(name: str, body: Mapping, type_: str,
                key: str = "above") -> Tuple[Any, ...]:
    """An item's hard limit (§ 5.3), ``(bound, why)``: a number the value
    must exceed and the sentence every refusal gives, on a numeric item --
    and the item's own DEFAULT and each kind's RECOMMENDED value obey it,
    since a limit its declaration breaks is a catalogue that contradicts
    itself.  Its VALUE is not held here: in a
    calculation's template that is a person's answer, and a value past the
    limit is refused where every value is, by the one per-value door
    (:func:`why_not`) with its one message."""
    raw = body.get(key)
    if raw is None:
        return ()
    # `above` is strict, `at_least` inclusive: a value breaks the limit when
    # it is at or below the bound, or below it.
    breaks = ((lambda x, b: x <= b) if key == "above"
              else (lambda x, b: x < b))
    must = "greater than" if key == "above" else "at least"
    bound = raw.get("value") if isinstance(raw, Mapping) else None
    why = raw.get("why") if isinstance(raw, Mapping) else None
    if (not isinstance(raw, Mapping) or set(raw) != {"value", "why"}
            or isinstance(bound, bool) or not isinstance(bound, (int, float))
            or not isinstance(why, str) or not why.strip()):
        _refuse(f"`{key}` is a hard limit: a number the value must be "
                f"{must} and the reason, e.g. {key} = {{ value = 0, why = "
                f"\"SIESTA divides by it\" }}", where=name)
    if type_ not in ("int", "float", "int3", "float3", "pow2"):
        _refuse(f"`{key}` bounds a number, and this item is {type_!r}",
                where=name)
    # ...AND EVERY VALUE THE CATALOGUE STARTS A CALCULATION ON: the default,
    # and each kind's `recommended` -- a kind is never started on a value it
    # refuses (§ 6.3a).
    recommended = body.get("recommended")
    starts = [("default", body.get("default"))] + [
        (f"recommended value for {k}", v)
        for k, v in (recommended.items() if isinstance(recommended, Mapping)
                     else ())]
    for what, v in starts:
        parts = v if isinstance(v, (list, tuple)) else (v,)
        if v is not None and any(breaks(float(x), bound) for x in parts):
            _refuse(f"its {what} {v!r} breaks its own limit -- it must be "
                    f"{must} {bound}: {why}", where=name)
    return (bound, why.strip())


def _engine_default_from(name: str, body: Mapping, type_: str,
                         choices) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """An item's ``engine_default`` (§ 5): ``((value, source), ())`` -- the
    engine's own value for its keyword, passing the item's own type as
    ``value`` and ``default`` do (:func:`_check_raw_value`), and the line of
    the engine's source that sets it -- or ``((), (what, source))`` when the
    engine's value follows something else; ``((), ())`` when none is
    stated."""
    raw = body.get("engine_default")
    if raw is None:
        return (), ()
    source = raw.get("source") if isinstance(raw, Mapping) else None
    keys = set(raw) if isinstance(raw, Mapping) else set()
    good_source = isinstance(source, str) and bool(source.strip())
    if keys == {"value", "source"} and good_source:
        _check_raw_value(name, "engine_default", raw["value"], type_,
                         tuple(choices) if choices else None)
        return (_shape(raw["value"], type_), source.strip()), ()
    what = raw.get("none") if isinstance(raw, Mapping) else None
    if (keys == {"none", "source"} and good_source
            and isinstance(what, str) and what.strip()):
        return (), (what.strip(), source.strip())
    _refuse("`engine_default` is the engine's own value and the line of "
            "its source that sets it, e.g. engine_default = { value = "
            "0.001, source = \"read_options.F90: fdf_get('SCF.H."
            "Tolerance', 1.e-3_dp*eV, 'Ry')\" } -- or, when the engine's "
            "value follows something else (another keyword, the build, the "
            "run), what it follows and that line: engine_default = { none = "
            "\"the ranks and the orbitals\", source = \"...\" }", where=name)


def recommended_for(item: "Item", calculation: str) -> Any:
    """The value ``item`` recommends for ``calculation``, or ``None``."""
    for kind, v in item.recommended:
        if kind == calculation:
            return v
    return None


def with_recommended(item: "Item", calculation: str) -> "Item":
    """``item`` with the kind's recommendation as its value and default --
    what a surface built for that kind shows (§ 6.3a); the item unchanged
    when the kind recommends nothing."""
    rec = recommended_for(item, calculation)
    if rec is None:
        return item
    return dataclasses.replace(item, value=rec, default=rec)


def apply_recommended(config, calculation: str, *, engine: str = ""):
    """A copy of ``config`` carrying every recommendation the catalogue makes
    for ``calculation`` on its engine (§ 6.3a) -- what `molbuilder describe` writes a kind's
    template from, so the template starts where the kind says it should."""
    eng = engine or engine_name(type(config))
    updates = {}
    for it in select(catalogue(), engine=eng, calculation=calculation):
        rec = recommended_for(it, calculation)
        if rec is not None and hasattr(config, it.name):
            updates[it.name] = rec
    return dataclasses.replace(config, **updates) if updates else config


def _check_raw_value(name: str, key: str, raw, type_: str,
                     choices) -> None:
    """Refuse a SET value (or default) that is not what its own
    declaration says it is -- on the parsed TOML, read as the setting door
    reads every value (§ 5.4, :func:`as_declared`: a whole float for a
    count is the count), before ``_shape`` touches it.  ``range`` stays
    advisory (§ 3.3 calls it *"advisory bounds"* -- a value outside it is a
    choice, not a type error); the TYPE and an enum's membership are not
    advice."""
    if raw is None:
        return
    why = _type_refusal(type_, choices, as_declared(type_, raw))
    if why:
        raise ValueError(
            f"template: item {name!r}: its {key} is {raw!r}{why} -- the "
            f"item's declared type, the contract for an edit "
            f"(engines/template.md § 5).")


def is_member(value, choices) -> bool:
    """Whether ``value`` is one of an enum's ``choices`` WITH ITS OWN TYPE.

    An enum's members are strings or whole numbers (§ 5).  Python's ``in``
    compares by value, and ``True == 1``: a template writing
    ``unpaired_electrons = true`` would pass as a count of one, and ``"2"``
    would pass nowhere it should.  So the type is part of the question.
    """
    return any(type(value) is type(c) and value == c for c in choices)


def as_declared(type_: str, v: Any) -> Any:
    """``v`` in its declared type's canonical form, LOSSLESSLY -- the one
    place a described value meets its type (§ 5).  JSON has one number and
    one sequence: an int where a float is declared widens (``150`` ->
    ``150.0``), a WHOLE float where a count is declared narrows (``8.0`` ->
    ``8``, never ``8.5``), and a list where a triple is declared becomes a
    tuple, each component the same way.  Nothing lossy is done: ``100.7``
    for a count stays ``100.7``, and a string stays a string, for the type
    check to refuse by name (`validation/task.py`).  `resolve` lays a
    description's values on through this, and the description's own check
    asks the per-value door (:func:`why_not`) about the value as `resolve`
    will see it -- so ``0.0`` for a count is the count 0 at every door."""
    def one(x, want):
        if isinstance(x, bool):
            return x
        if want == "float" and isinstance(x, int):
            return float(x)
        if want == "int" and isinstance(x, float) and x.is_integer():
            return int(x)
        return x
    if type_ == "float":
        return one(v, "float")
    if type_ in ("int", "pow2"):
        return one(v, "int")
    if type_ in ("int3", "float3") and isinstance(v, (list, tuple)):
        return tuple(one(x, "int" if type_ == "int3" else "float") for x in v)
    return v


def _shape(v: Any, type_: str) -> Any:
    """Give a parsed value the Python shape its declared type implies.

    TOML has one sequence, and the config classes do not: ``kgrid`` is a
    ``Tuple[int, int, int]`` while ``species_order`` is a ``List[str]``, and
    both come back from ``tomllib`` as a plain list. Without this an item
    round-trips to a value that is *equal in content and different in type*,
    which is the quietest kind of loss — and it is what the writer's own
    round-trip check cannot see, because the check compares the TOML payload
    with itself rather than the config with the config.
    """
    if v is None:
        return None
    if type_ == "pow2":
        # Snap DOWN to the nearest power of two, so a snap can never hand an
        # engine a BIGGER block than was asked for.  No catalogue item is
        # ``pow2`` -- BENCH-MARKS keeps the constraint, for its own sweep
        # (`script_emit.benchmark_declarable_types`) -- so this runs only for
        # a hand-written template that declares one.
        n = int(v)
        if n <= 0:
            return 0 if n == 0 else n
        return 1 << (n.bit_length() - 1)
    if type_ == "int3":
        return tuple(int(x) for x in v)
    if type_ == "float3":
        return tuple(float(x) for x in v)
    if type_ == "intlist":
        return [int(x) for x in v]
    if type_ == "strlist":
        return [str(x) for x in v]
    return v


def _item_from(name: str, body: Any) -> Item:
    if not isinstance(body, Mapping):
        _refuse(f"expected a table, got {type(body).__name__}", where=name)
    unknown = sorted(set(body) - _KNOWN_ITEM_KEYS)
    if unknown:
        _refuse(f"unknown key(s) {', '.join(repr(k) for k in unknown)} "
                f"(known: {', '.join(_ITEM_KEY_ORDER)})", where=name)
    for key in _REQUIRED_ITEM_KEYS:
        if key not in body:
            _refuse(f"missing required key {key!r}", where=name)

    # A key that must be a LIST is refused when it is a bare string, BY NAME.
    #
    # ``tuple("accuracy")`` is ``('a','c','c','u','r','a','c','y')``, so a
    # hand-written ``category = "accuracy"`` would reach the vocabulary check
    # one character at a time and refuse with *"category 'a' is not one of
    # system, method, ..."* -- an error that names the wrong problem and sends
    # the reader looking for a category called `a`.  Refusing here, before any
    # of them is read, is what keeps the message about the key the author
    # actually typed.
    for _k in ("category", "engines", "read_by", "expands", "choices"):
        if isinstance(body.get(_k), str):
            _refuse(f"{_k} must be a list, not the string "
                    f"{body[_k]!r} -- write {_k} = [{body[_k]!r}]. A bare "
                    f"string is read one character at a time, so the refusal "
                    f"would name a member you never wrote", where=name)

    rng = body.get("range")
    type_ = str(body["type"])
    # § 5's type check runs on the RAW TOML value, BEFORE _shape gives it
    # a Python shape (R4): after it, a scalar on a strlist would be exploded
    # "Au" into ['A','u'] and pass, and a scalar on int3 would die as a raw
    # TypeError naming no item.
    choices = body.get("choices")
    for key in ("value", "default"):
        _check_raw_value(name, key, body.get(key), type_,
                         tuple(choices) if choices else None)
    return Item(                       # Item.__post_init__ enforces § 3's rest
        name=name,
        kind=str(body["kind"]),
        type=type_,
        help=str(body["help"]),
        value=_shape(body.get("value"), type_),
        source=_source_from(name, body.get("source")),
        default=_shape(body.get("default"), type_),
        anchor=str(body.get("anchor", "") or ""),
        engine_key=str(body.get("engine_key", "") or ""),
        manual=str(body.get("manual", "") or ""),
        **dict(zip(("engine_default", "engine_unfixed"),
                   _engine_default_from(name, body, type_, choices))),
        expands=tuple(body.get("expands", ()) or ()),
        launch=_launch_from(name, body.get("launch")),
        optional=bool(body.get("optional", False)),
        tier=str(body.get("tier", "") or ""),
        pattern=str(body.get("pattern", "") or ""),
        read_by=tuple(body.get("read_by", ()) or ()),
        choices=(tuple(body["choices"]) if body.get("choices") else None),
        range=(tuple(rng) if rng else None),
        unit=body.get("unit"),
        group=body.get("group"),
        label=str(body.get("label", "") or ""),
        category=tuple(body.get("category", ()) or ()),
        engines=tuple(body.get("engines", ()) or ()),
        calculations=tuple(body.get("calculations", ()) or ()),
        recommended=_table_from(name, "recommended", body.get("recommended"),
                                type_, choices, keyed_by="calculation kind",
                                example="recommended = { vibration = 0.01 }",
                                keys=_kinds.kinds()),
        offered=_offered_from(name, body.get("offered"), type_, choices,
                              tuple(body.get("engines", ()) or ())),
        not_offered=_not_offered_from(name, body.get("not_offered"), type_,
                                      choices,
                                      tuple(body.get("engines", ()) or ()),
                                      body.get("offered")),
        result=_result_from(name, body.get("result")),
        active_when=_active_when_from(name, body.get("active_when")),
        choice_labels=_choice_labels_from(name, body.get("choice_labels"),
                                          choices),
        above=_above_from(name, body, type_),
        at_least=_above_from(name, body, type_, key="at_least"),
        tightens=_tightens_from(name, body.get("tightens")),
        refs=tuple(body.get("refs", ()) or ()),
        allocation=bool(body.get("allocation", False)),
        run_card=bool(body.get("run_card", False)),
        citation=tuple(body.get("citation", ()) or ()),
        role=tuple(body.get("role", ()) or ()),
        role_values=_table_from(name, "role_values", body.get("role_values"),
                                type_, choices, keyed_by="rung",
                                example='role_values = { device = "transiesta" }'),
        shared=tuple(body.get("shared", ()) or ()),
        required=tuple(body.get("required", ()) or ()),
        stages=_stages_from(name, body.get("stages")),
        null_label=str(body.get("null_label", "") or ""),
        not_set=str(body.get("not_set", "") or ""),
    )


@functools.lru_cache(maxsize=1)
def _help_index() -> Dict[str, str]:
    """``{item name: help}`` from the master catalogue, read once."""
    try:
        return {i.name: (i.help or "") for i in catalogue().items}
    except Exception:                                        # noqa: BLE001
        return {}          # a broken catalogue must not break a form or --help


def help_for(name: str, *, first_paragraph: bool = False) -> str:
    """The catalogue's help for one item — **the one home for what a user reads.**

    Every surface that shows a person what a setting means asks this, rather
    than reading the copy in a config dataclass's ``metadata["help"]``.

    ``first_paragraph`` takes the opening sentence only, which is what a CLI
    option line wants — the catalogue's entries carry per-tier tables and
    worked consequences below that, and a `--help` screen is not where those
    belong. Item names are unique across the catalogue, so no engine argument
    is needed to resolve one.

    Returns ``""`` for a name the catalogue does not declare.
    """
    text = _help_index().get(name, "")
    if first_paragraph and text:
        return text.split("\n\n", 1)[0].strip()
    return text


def template_fields(config_cls) -> set:
    """The field names a value may be LAID on as a parameter -- the schema
    minus the machine facts: what `resolve` gates a pin, an override and a
    sweep's parameter axis on, and what a template's item may name.  A
    field tagged ``allocation: True`` is § 7's machine fact: it arrives as
    the ALLOCATION at `prep`, on the machine that will run it, never laid
    as a value.  (Which items a template holds is narrower still -- no run
    setting, § 2 -- and is the template file's door's to refuse.)
    """
    items = items_of(config_cls)
    return {f.name for f in dataclasses.fields(config_cls)
            if not (items.get(f.name) is not None and items[f.name].allocation)}


# --------------------------------------------------------------------- #
#  § 8.0 -- THE ONE READ API                                            #
# --------------------------------------------------------------------- #

def _as_tuple(v) -> Tuple[str, ...]:
    if v is None:
        return ()
    return (v,) if isinstance(v, str) else tuple(v)


def _check_engine(t: "Template", engine) -> None:
    """Refuse an engine this template does not serve (§ 3).

    *"This calculation does not run on that engine"* and *"no items
    matched"* are different answers, and an empty list cannot tell them
    apart -- a caller would read a refusal as an absence and carry on.
    """
    if engine and t.engines and engine not in t.engines:
        raise ValueError(
            f"this template describes a calculation for {list(t.engines)}, "
            f"not {engine!r}. An empty result would read as 'no items "
            f"matched' when the truth is 'this calculation does not run on "
            f"that engine' (engines/template.md § 3).")


#: The electronic state's four items (`science/chemistry-correctness.md`
#: § 2a) -- `shared` for every kind that has them (ES1).
STATE_ITEMS = ("net_charge", "spin_treatment", "unpaired_electrons", "method")


def shared_by_every_stage(engine: str, kind: str) -> frozenset:
    """The items that bind every stage of a ``kind`` calculation -- the
    catalogue's `shared` (§ 6.4), asked rather than listed.  A stage override
    naming one is refused (:func:`why_shared` says why), and the Task setup
    table offers none of them as a column."""
    return frozenset(it.name for it in select(catalogue(), engine=engine)
                     if kind in it.shared)


def run_settings(engine: str) -> frozenset:
    """The settings a rung states on its RUN CARD -- the catalogue's
    `execution` items (§ 6.2), the machine's answers among them -- asked
    rather than listed.  Each has one home, the job request's run cards --
    the calculation's with the rung's over it (`execution/job-request.md`
    § 2) -- never the template, a stage column or an override
    (:func:`why_run_setting`), which the description's own check and
    ``resolve`` refuse and the Task setup table does not offer."""
    return frozenset(it.name for it in select(catalogue(), engine=engine)
                     if is_run_setting(it))


def why_run_setting(name: str) -> str:
    """Why a run setting cannot be a column -- the clause a refusal puts
    after *"stage <s> overrides <name>"* or *"'varies' names <name>"*."""
    return (f"a run setting: the rung states it on its run card in "
            f"job-request.json -- \"execution\": {{\"{name}\": ...}} on the "
            f"stage's entry, or on the calculation for every rung "
            f"(execution/job-request.md 2) -- never in the description")


#: The calculation's identity -- `shared` for every SIESTA kind (2026-09-29,
#: K1): one name, one species table, one pseudopotential set.
IDENTITY_ITEMS = ("system_label", "species_order", "psml_lib")


def shared_by(name: str) -> str:
    """WHO SHARES a `shared` item, in words -- *every stage of this
    calculation*, or the stages its ``stages`` names (§ 5, TD11): what a
    refusal says after *"is shared by"*, before :func:`why_shared`."""
    scoped = _scoped_shared(name)
    return (f"the {_rungs_words(scoped.stages)} stages of this calculation"
            if scoped is not None else "every stage of this calculation")


def _scoped_shared(name: str) -> Optional["Item"]:
    """The catalogue item ``name`` when it is `shared` and names its
    ``stages`` -- one value for those rungs only -- else ``None``."""
    return next((it for it in catalogue().items
                 if it.name == name and it.shared and it.stages), None)


def why_shared(name: str) -> str:
    """Why a `shared` item cannot be one stage's -- the clause a refusal
    puts after *"is shared by <who>"* (:func:`shared_by`)."""
    if name in STATE_ITEMS:
        return ("the electronic state belongs to the calculation, never to "
                "a stage -- every stage's warm files are a density for one "
                "state (science/chemistry-correctness.md § 2a, ES1)")
    if name in IDENTITY_ITEMS:
        return ("a calculation has one name, one species table and one set "
                "of pseudopotentials -- each stage reads what the one before "
                "it wrote under that name, numbered by that table (the `.XV` "
                "holds each atom's species index) and built on that set "
                "(engines/template.md § 5, `shared`)")
    if _scoped_shared(name) is not None:
        return ("they are the rungs that read it, and must not be able to "
                "disagree about it (engines/template.md § 5, `shared` with "
                "`stages`)")
    return ("the electrode and the device must not be able to disagree "
            "about it (engines/transport.md § 2a.7)")


def shared_scope_words(item: "Item") -> str:
    """WHICH RUNGS a `shared` value binds, in words -- ``""`` when every rung
    reads it, else the rungs its ``stages`` names: `shared` with `stages` is
    one value carried by the named rungs only (§ 5, TD11).  Said beside the
    value on the transport tab's shared panel and in Task setup's echo."""
    return (f"read by the {_rungs_words(item.stages)} rungs only"
            if item.stages else "")


def _rungs_words(stages) -> str:
    """``device and transmission`` -- rung roles joined as a sentence."""
    names = list(stages)
    return (names[0] if len(names) == 1
            else ", ".join(names[:-1]) + " and " + names[-1])




def stage_role_rule(engine: str, kind: str) -> Optional[Dict[str, Any]]:
    """How a ``kind`` calculation on ``engine`` names its rungs' roles, as
    DATA -- ``{"named": {stage: roles}, "otherwise": roles}``, each
    ``roles`` a list, where an ``"otherwise"`` of ``None`` means *a rung's
    role is its name* -- or ``None`` for a kind without roles.  ``named`` is
    keyed by `identity.stage_key`, so a reader matches a stage's name in any
    case.  The one statement of the rule: :func:`stage_roles` reads it, and
    so does a surface that maps its own rows (the stage table, which is
    handed it with its columns).

    A transport rung's role is its name.  A SIESTA vibration's one stage
    is one run in three phases, and carries every phase's role
    (`engines/vibration.md` § 5.11 § 9: the stage's one resolved
    configuration feeds every phase's deck); a PySCF vibration relaxes
    inside its one deck, so each of its rungs is the force-constant run --
    the pair's own word in the kind table (``stage_roles``).  A kind with
    roles asked on an engine it does not run on is refused by the kind's
    part (`kind_table.KindTableError`, naming the engines it runs on)."""
    if not _roles_of(kind):
        return None
    rule = _kinds.kind(kind).on(str(engine)).stage_roles
    if rule == "own":
        return {"named": {}, "otherwise": None}
    return {"named": {},
            "otherwise": list(_roles_of(kind) if rule == "every" else rule)}


def stage_roles(engine: str, kind: str, stage: Optional[str]
                ) -> Optional[Tuple[str, ...]]:
    """The roles the rung ``stage`` plays in a ``kind`` calculation on
    ``engine`` (:func:`stage_role_rule`) -- the ONE rule every door asks
    which items a rung reads with (§ 6.4): one role for a transport rung,
    every phase's for a SIESTA vibration's stage.  ``None`` for a kind
    without roles, or no stage."""
    rule = stage_role_rule(engine, kind)
    if rule is None or stage is None:
        return None
    got = rule["named"].get(stage_key(stage)) or rule["otherwise"]
    return tuple(got) if got else (stage,)


def reads(item: "Item", engine: str, kind: str, stage: Optional[str]) -> bool:
    """Does the rung ``stage`` of a ``kind`` calculation read ``item``?  Not
    when the kind does not carry it (`calculations`); on a kind without
    roles, always; else when its ``stages`` names one of the rung's roles,
    or names none (§ 6.4)."""
    if item.calculations and kind not in item.calculations:
        return False
    roles = stage_roles(engine, kind, stage)
    return (roles is None or not item.stages
            or bool(set(roles) & set(item.stages)))


def unread_overrides(engine: str, kind: str, stage: str, names) -> List[str]:
    """The names among a rung's overrides that the rung does not read -- the
    one door the description's own check and ``resolve`` ask (§ 6.4, plan
    § 5w K4), so an override nothing reads is refused by name rather than
    written into a deck that ignores it.  A name no catalogue item of the
    engine carries is the name check's, and an item the rung FIXES is
    :func:`why_role`'s; both are left to them."""
    items = {it.name: it for it in select(catalogue(), engine=engine)}
    return [n for n in names if n in items and kind not in items[n].role
            and not reads(items[n], engine, kind, stage)]


def why_unread(name: str, engine: str, kind: str, stage: str, *,
               then: str = "set it there") -> str:
    """Why the rung ``stage`` does not read ``name`` -- the clause both doors
    give after *"stage <s> overrides <name>"*, ending in the way out
    ``then``: an override is set on a rung that reads it, a run-card value
    removed from the card it is stated on."""
    it = one(catalogue(), name, engine=engine)
    if it.calculations and kind not in it.calculations:
        return (f"which {_kinds.kind(kind).with_article} does not carry -- no rung of it reads "
                f"it, so the value would change nothing (engines/template.md "
                f"6.3)")
    roles = stage_roles(engine, kind, stage) or (stage,)
    # ...THE RUNGS OF THIS KIND that read it, never another kind's roles.
    mine = [r for r in it.stages if r in (_roles_of(kind) or it.stages)]
    who = " / ".join(mine)
    rung = (repr(stage) if roles == (stage,)
            else f"{stage!r} (its {' + '.join(roles)} rung)")
    return (f"which only the {who} rung{'s' if len(mine) > 1 else ''} "
            f"of {_kinds.kind(kind).with_article} read{'' if len(mine) > 1 else 's'}, not "
            f"{rung} -- {then} (engines/template.md 6.4, the `stages` "
            f"declaration)")


def fixed_by_role(engine: str, kind: str) -> frozenset:
    """The items the rungs of a ``kind`` calculation fix -- the catalogue's
    `role` (§ 6.4), asked rather than listed.  No stage override, pin, sweep
    axis or template value may set one (:func:`why_role` says why);
    :func:`role_answers` is what each rung's config holds instead."""
    return frozenset(it.name for it in select(catalogue(), engine=engine)
                     if kind in it.role)


def role_answers(engine: str, kind: str,
                 stage: Optional[str] = None) -> Dict[str, Any]:
    """``{item: value}`` -- what the rung ``stage`` of a ``kind``
    calculation answers for its `role` items (§ 6.4): the item's own
    ``role_values`` entry for that rung, else the catalogue's ``value``.

    An item whose ``stages`` does not name the rung is not this rung's and
    is absent -- a lead's ``ts_hs_save`` is no other rung's.  ``resolve``
    lays these on the rung's config as its last layer; the bias point, the
    one answer the catalogue does not hold, is laid on by the point `prep`
    renders (`engines/transport.md` § 2a.10).
    """
    out: Dict[str, Any] = {}
    # ...by the rung's ROLES (`stage_roles`), on transport its name.
    for it in select(catalogue(), engine=engine,
                     stages=stage_roles(engine, kind, stage)):
        if kind not in it.role:
            continue
        if stage is None and it.stages:
            continue                  # some rungs' only -- no rung was named
        answer = dict(it.role_values).get(stage, it.value)
        if answer is None:
            raise ValueError(
                f"catalogue item {it.name!r} is fixed by the rung for "
                f"{kind!r} (`role`) and answers nothing for rung "
                f"{stage!r}: give it a `value`, or the rung's own in "
                f"`role_values` (engines/template.md § 6.4).")
        out[it.name] = answer
    return out


def fixed_answers(engine: str, kind: str) -> Dict[str, Any]:
    """``{item: value}`` -- the `role` items of a ``kind`` calculation with
    ONE answer wherever they are read: no ``role_values`` (a rung's own
    answer) and not answered point by point (:data:`PER_POINT`), so the
    catalogue's ``value`` is the answer of every rung that reads it (§ 6.4).

    `resolve` reads a template value for one of these as that answer when
    it equals it -- what a template written before 2026-09-29 states for
    SIESTA's per-step forces and coordinates -- and refuses any other."""
    return {it.name: it.value
            for it in select(catalogue(), engine=engine)
            if kind in it.role and not it.role_values
            and it.name not in PER_POINT}


def fixed_on_every_rung(engine: str, kind: str) -> Dict[str, Any]:
    """``{item: value}`` -- those of :func:`fixed_answers` that EVERY rung
    of a ``kind`` calculation reads: on a kind without roles all of them;
    on one with roles, those whose ``stages`` names every role
    (the kind table's) -- SIESTA's per-step forces and coordinates on an
    optimization and a vibration, which every rung writes; not on a
    transport calculation, whose transmission writes neither (§ 6.4;
    `engines/transport.md` § 6.1b).

    What a whole-calculation surface echoes, and what the settings gate
    holds a config to: only a render that skipped `resolve` can hold
    another value.
    """
    roles = set(_roles_of(kind))
    stages = {it.name: set(it.stages)
              for it in select(catalogue(), engine=engine)}
    return {name: value
            for name, value in fixed_answers(engine, kind).items()
            if not roles or not stages[name] or roles <= stages[name]}


def why_not(item: "Item", value: Any, *, engine: str,
            kind: str) -> Optional[str]:
    """Why ``value`` cannot stand for ``item`` on a ``kind`` calculation --
    the ONE clause every door a value passes gives (§ 5.3), or ``None``.

    Asked by the description's own check, ``resolve`` and the settings gate,
    so one value draws one refusal: the first that holds of

    1. a component the kind fixes, holding another value
       (``kmesh.fixed`` -- a transport calculation's third k component,
       which no rung reads; `engines/siesta.md` § 6.1);
    2. a choice the kind does not offer (§ 6.3a) -- for a member of the
       item's ``choices``: a value that is not one is the declared type's
       refusal, and the electronic state's items are that family's, which
       judges their RESOLVED values;
    3. a value past its limit -- at or below an ``above``, below an
       ``at_least`` -- each component of a triple, the fixed components
       left to (1).

    The clause reads after *"<who> sets <name> = <value>"*: each door names
    who.  A value that does not hold the item's declared type -- ``"0"``
    for a number, ``0.0`` for a count -- is the type check's business, here
    as everywhere (``_TYPE_CHECKS``, § 5): it draws that refusal alone.
    """
    if value is None:
        return None
    holds = _TYPE_CHECKS.get(item.type)
    if holds is not None and not holds(value):
        return None
    triple = isinstance(value, (list, tuple))
    from .kmesh import AXES, fixed
    held = fixed(item.name, kind) if triple and len(value) == 3 else {}
    for i, (want, why) in held.items():
        if value[i] != want:
            return (f", whose {AXES[i]} component {_kinds.kind(kind).with_article} fixes at "
                    f"{want!r}: {why}")
    if (item.offered and item.name not in STATE_ITEMS and item.choices
            and is_member(value, item.choices)):
        can = offered(item, engine, kind)
        if not is_member(value, can):
            return (f", which {_kinds.kind(kind).with_article} does not offer: "
                    f"{why_not_offered(item.name, value, engine, kind)}.  It "
                    f"offers {', '.join(map(str, can))}")
    parts = [v for i, v in enumerate(value if triple else (value,))
             if i not in held]
    if item.above:
        bound, why = item.above
        if any(v <= bound for v in parts):
            return (f": {'each component' if triple else 'it'} must be "
                    f"greater than {bound} -- {why}")
    if item.at_least:
        bound, why = item.at_least
        if any(v < bound for v in parts):
            return (f": {'each component' if triple else 'it'} must be "
                    f"at least {bound} -- {why}")
    return None


def offered(item: "Item", engine: str, kind: str) -> Tuple[Any, ...]:
    """The choices ``item`` offers a ``kind`` calculation on ``engine``
    (§ 6.3a): the engine's own entry, else the kind's, else every choice
    the item declares.  What the form and the stage table's cells offer,
    and what `resolve` and the settings gate hold a value to."""
    table = dict(item.offered)
    for key in (f"{engine}.{kind}", kind):
        if key in table:
            return table[key]
    return tuple(item.choices or ())


def offered_by_any(item: "Item", engines, kind: str) -> Tuple[Any, ...]:
    """Every choice ``item`` offers a ``kind`` calculation on any of
    ``engines``, in the item's own order -- what a field drawn once for
    every engine of the kind offers (§ 6.2a); the chosen engine's checks
    refuse, naming why, a choice it cannot run."""
    sets = [set(offered(item, e, kind)) for e in engines]
    return tuple(c for c in (item.choices or ())
                 if any(c in s for s in sets))


def result_of(item: "Item", kind: str) -> Optional[str]:
    """The result of a ``kind`` calculation ``item`` is for, or ``None``
    (§ 6.2a)."""
    return dict(item.result).get(kind)


def card_of(item: "Item", kind: str) -> Optional[str]:
    """THE CARD ``item`` IS DRAWN IN on a form of a ``kind`` calculation --
    its result's when it names one for the kind, else its group's (§ 6.2a).
    One rule for the form and for where a warning about it is shown."""
    return result_of(item, kind) or item.group


def shared_by_engines(kind: str) -> frozenset:
    """The items one catalogue row serves on EVERY engine that runs a
    ``kind`` calculation (its parts in the kind table) -- drawn once,
    before the engine is chosen, on a page offering the kind on several
    (§ 6.2a).  Empty for a kind one engine runs: nothing is shared then;
    a kind the kind table does not hold refused by it, naming the kinds."""
    engines = _kinds.kind(kind).engines
    if len(engines) < 2:
        return frozenset()
    cat = catalogue()
    per_engine = [{it.name for it in select(cat, engine=e, calculation=kind)}
                  for e in engines]
    return frozenset(set.intersection(*per_engine))


def choice_words(item: "Item", value: Any) -> str:
    """``value`` of ``item`` as a person reads it -- a tick box *on* or
    *off*; a choice's words (``choice_labels``), quoted; else as a refusal
    quotes a value (:func:`shown`)."""
    if isinstance(value, bool):
        return "on" if value else "off"
    words = dict(item.choice_labels).get(value)
    return f'"{words}"' if words else shown(value)


def active_words(item: "Item", *, names: bool = False) -> Optional[str]:
    """What ``item`` waits on (§ 5, ``active_when``), as a person reads it
    -- *"it takes effect only while Compute Raman intensities is on"* --
    or ``None`` for an item that waits on nothing.  The form shows it
    beside a greyed-out field; the settings gate (:func:`inactive_why`)
    says it with each setting's name, ``names``, as the template spells
    it."""
    if not item.active_when:
        return None
    cat = {it.name: it for it in catalogue().items}

    def _named(n: str) -> str:
        label = getattr(cat.get(n), "label", "") or ""
        if not label:
            return n
        return f"{label} ({n})" if names else label

    def _said(n: str, v: Any) -> str:
        return choice_words(cat[n], v) if n in cat else shown(v)
    return ("it takes effect only while "
            + " and ".join(f"{_named(n)} is "
                           + " or ".join(_said(n, v) for v in vs)
                           for n, vs in item.active_when))


def inactive_why(item: "Item", values: Mapping[str, Any]) -> Optional[str]:
    """Why ``item`` enters nothing given the other settings' ``values``
    (§ 5, ``active_when``) -- *"it takes effect only while Compute Raman
    intensities (compute_raman) is on"* -- or ``None`` when every setting
    it waits on holds one of its values.  Asked by the settings gate; the
    form shows the same words beside the field (:func:`active_words`)."""
    if all(values.get(n) in vs for n, vs in item.active_when):
        return None
    return active_words(item, names=True)


def why_not_offered(name: str, value: Any, engine: str, kind: str) -> str:
    """Why a ``kind`` calculation on ``engine`` is not offered ``value`` for
    ``name`` -- the clause a refusal puts after *"which a <kind> does not
    offer"*: the first of the item's ``not_offered`` rows that holds
    (§ 6.3a), each reason read from the engine's source
    (`science/chemistry-correctness.md` § 2a.3); a value no row explains
    said as the kind on the engine not running it."""
    item = one(catalogue(), name, engine=engine)
    for row in (item.not_offered if item is not None else ()):
        if row.holds(value, engine, kind):
            return row.why
    return f"{_kinds.kind(kind).with_article} on {engine} does not run it"


#: The role item no rung answers with ONE value: each point of the
#: description's bias list is one device run and one transmission, written at
#: that point (`engines/transport.md` § 2a.10) -- the catalogue's ``value`` is
#: the list's first point, not a rung's answer.  A surface echoing the rung's
#: answers shows this one's reason and no number (`form-schema.md` § 1.1).
PER_POINT = frozenset({"bias_voltage_v"})


def why_role(name: str) -> str:
    """Why nobody may set a `role` item -- the clause a refusal puts after
    *"which the rung fixes"*."""
    if name in PER_POINT:
        return ("the bias is the description's bias list -- each rung writes "
                "the point it runs, the device and the transmission alike "
                "(engines/transport.md § 2a.10); change the list instead")
    if name in ("write_forces", "write_coor_step"):
        return ("every SIESTA run writes each step's forces and coordinates, "
                "because molbuilder reads them back -- the relaxation "
                "record, the trajectory and a vibration's finish "
                "(engines/template.md § 6.4)")
    return ("it is what makes the rung that rung -- a choice with one "
            "correct answer (engines/template.md § 6.4)")


# --------------------------------------------------------------------- #
#  § 5.4 -- THE SETTING DOOR: every place a value lives asks it          #
# --------------------------------------------------------------------- #

#: THE PLACES A VALUE OF A SETTING CAN LIVE (§ 5.4): the template, a stage's
#: overrides (task.json), a run card and a benchmark row (job-request.json),
#: a prep flag.  Each asks :func:`accepts` with its own name.
HOMES = ("override", "run_card", "bench", "flag")

#: WHO SET A VALUE, in the words every reader says it (§ 5.4) -- the deck's
#: PROVENANCE rows, the pipeline log, prep's printed lines and its
#: placement, the page.  Keyed by the layer a value came from: `resolve`'s
#: provenance (``template`` ... ``role``) and the launch fold's
#: (``flag``, ``run card``, ``allocation``, `prep_inputs.fold_allocation`).
#: A template value says, after *the template*, its own record
#: (:data:`SOURCE_WORDS`).
SAID = {
    "template":   "the template",
    "stage":      "this stage's override",
    "stage_card": "the stage's run card",
    "calc_card":  "the calculation's run card",
    "run card":   "the run card",
    "flag":       "a prep flag",
    "allocation": "the job request's allocation",
    "default":    "the catalogue's default · not chosen",
    "sweep":      "a sweep point",
    "pin":        "a pin",
    "role":       "the rung",
}


#: WHAT EVERY BENCHMARK TRIAL FIXES -- what makes a trial a MEASUREMENT
#: rather than a run (`execution/job-request.md` § 3.4): its SCF capped,
#: no relaxation step, a cold start, no retry, and ``scf_must_converge:
#: False`` so the cap ends CLEAN.  Laid over every point by the bench's
#: assembly (`jobset/prep_inputs.py`), and no benchmark row may name one
#: (:func:`home_of`): its points would be one measurement under many labels.
#: ``max_scf_iter: 3`` (user, 2026-08-21, from measured experience:
#: iterations 3-5 agree within seconds on a 444-atom junction, and the
#: bench reads SCALING and DEPENDENCY -- the knee -- not tight rankings):
#: iteration 1 never forms a timing delta, the iter-2 delta is dropped as
#: warm-up-adjacent, iteration 3 is the one clean sample (tuning.md § 2.12).
TRIAL_PINS = {"max_scf_iter": 3, "relax_steps": 0, "restart": "clean",
              "continue_retries": 0, "scf_must_converge": False}

#: THE LAUNCH SHAPE A BENCHMARK'S GRID SWEEPS -- ranks, cores per rank,
#: devices (`execution/generator.md` § 4.3a): the machine-answered items a
#: benchmark row may name.  Another (a memory cap) is no axis the grid
#: resolves (:func:`home_of`).
GRID_AXES = ("mpi_np", "omp_threads", "gpu_count")


def is_run_setting(item: "Item") -> bool:
    """Whether ``item`` is a RUN SETTING -- the catalogue's `execution`
    category (§ 6.2): how a run runs, which the job request states
    (`execution/job-request.md` § 2), never the description."""
    return "execution" in (item.category or ())


def home_of(item: "Item", kind: str) -> frozenset:
    """Where a value of ``item`` may live on a ``kind`` calculation (§ 5.4)
    -- the homes :func:`why_not_in` has no refusal for, so the rule is
    written once."""
    return frozenset(h for h in HOMES
                     if why_not_in(item, h, engine="", kind=kind) is None)


def why_not_in(item: "Item", home: str, *, engine: str, kind: str,
               stage: Optional[str] = None) -> Optional[str]:
    """Why ``home`` may not hold a value of ``item`` -- the clause after
    *"<who> sets <name> = <value>"* or *"<who> names <name>"* -- or
    ``None`` (§ 5.4): THE ONE RULE OF WHERE A VALUE LIVES, read off the
    item's own markers.  A stage's overrides hold what is computed -- never
    a run setting, never what the rung fixes (`role`), never what every
    stage shares (`shared`); a run card holds a
    run setting; a benchmark row one a trial may vary -- never what every
    trial fixes (:data:`TRIAL_PINS`), and of the launch shape only the
    grid's axes (:data:`GRID_AXES`); a prep flag the launch shape
    (`allocation`).  An item the kind does not carry has no home on it, and
    a stage's override or run card holds only what that rung reads (§ 6.4's
    ``stages``, :func:`reads`)."""
    if home not in HOMES:
        raise ValueError(f"no home {home!r} -- a setting's homes are "
                         f"{', '.join(HOMES)} (engines/template.md 5.4)")
    name, run = item.name, is_run_setting(item)
    if item.calculations and kind not in item.calculations:
        return (f", which {_kinds.kind(kind).with_article} does not carry -- no rung of it "
                f"reads it, so the value would change nothing "
                f"(engines/template.md 6.3)")
    if home == "flag" and not item.allocation:
        return (", which no prep flag states -- a flag states the launch "
                "shape alone: ranks, threads, GPUs, the memory cap "
                "(execution/architecture.md 5.2)")
    if home == "run_card" and not run:
        return (f", which is not an execution setting of {engine}.  A run "
                f"card says what the RUN uses -- ranks, threads, the device, "
                f"the solver.  A physics value belongs in the template, and "
                f"a per-stage one in that stage's `overrides` in task.json "
                f"(engines/stages.md 6.2)")
    if home == "bench" and not run:
        return (f", which {engine} does not declare as an `execution` "
                f"parameter -- so sweeping it would measure a different "
                f"calculation at each point and the comparison would mean "
                f"nothing (execution/job-request.md 3.4)")
    if home == "bench" and name in TRIAL_PINS:
        return (f", which every benchmark trial fixes at "
                f"{shown(TRIAL_PINS[name])}: a trial is one measurement "
                f"(execution/job-request.md 3.4)")
    if home == "bench" and item.allocation and name not in GRID_AXES:
        return (f", which a benchmark's grid does not sweep -- its machine "
                f"axes are {', '.join(GRID_AXES)} (execution/generator.md "
                f"4.3a)")
    if home == "override" and run:
        return ", " + why_run_setting(name)
    if home == "override" and kind in item.role:
        return (f", which the rung fixes: {why_role(name)}.  Remove the "
                f"override")
    if home == "override" and kind in item.shared:
        return (f", which is shared by {shared_by(name)}: "
                f"{why_shared(name)}.  Change it in the template, where one "
                f"value applies to every stage that reads it")
    if (stage is not None and home in ("override", "run_card")
            and not reads(item, engine, kind, stage)):
        then = ("remove it from that run card -- on Task setup, the × "
                "beside it" if home == "run_card" else "set it there")
        return ", " + why_unread(name, engine, kind, stage, then=then)
    return None


#: A declared type in the words a refusal names it by (§ 5.4).
_TYPE_WORDS = {
    "int": "a whole number", "pow2": "a whole number", "float": "a number",
    "str": "text", "text": "text", "bool": "true or false",
    "int3": "three whole numbers, written as a list",
    "float3": "three numbers, written as a list",
    "strlist": "a list of text values",
    "intlist": "a list of whole numbers",
}


def _type_refusal(type_: str, choices, value: Any) -> Optional[str]:
    """Why ``value`` -- already spelled (:func:`as_declared`) -- is not of
    the declared ``type_``, an enum's ``choices`` among it (§ 5), or
    ``None``: the ONE type check, which the setting door and the template
    reader ask alike.  An enum's member is part of its type: ``"2"`` is not
    the count 2."""
    holds = _TYPE_CHECKS.get(type_)
    if holds is not None and not holds(value):
        if type_ == "enum":
            return (f", which is not one of "
                    f"{', '.join(map(repr, choices or ()))}")
        if (type_ in ("int", "pow2") and isinstance(value, float)
                and not isinstance(value, bool)):
            return (f", which is not a whole number -- this setting counts "
                    f"things, so {value!r} would have to be rounded and the "
                    f"run would not be the one described")
        return f", which is not {_TYPE_WORDS[type_]}"
    if type_ == "enum" and choices and not is_member(value, choices):
        return f", which is not one of {', '.join(map(repr, choices))}"
    return None


def accepts(item: "Item", value: Any, home: str, *, engine: str, kind: str,
            stage: Optional[str] = None) -> Optional[str]:
    """THE SETTING DOOR (§ 5.4): may ``value`` stand for ``item`` in
    ``home`` on a ``kind`` calculation on ``engine`` (``stage``, for a
    stage's override or run card)?  ``None`` when it may, else the ONE
    clause refusing it, read after *"<who> sets <name> = <value>"* -- the
    first that holds of:

    1. the home may not hold this item at all (:func:`why_not_in`);
    2. ``None`` -- *unset* -- for an item whose unset is no answer (not
       ``optional``);
    3. the item's declared type, the catalogue's -- the value judged as
       :func:`spell` reads it -- and an enum's membership;
    4. § 5.3's door (:func:`why_not`): a component the kind fixes, a choice
       it does not offer, a limit.

    Every place a value lives asks this with its own home, and keeps no
    check of its own; a benchmark row's point is a value to try, never
    ``None``.  The template is read by its own two refusals -- a run setting
    (`template_file`), a rung's answer stated otherwise (`resolve`) -- and
    the kind's rules asked of its values where it meets its kind
    (:func:`why_not`)."""
    clause = why_not_in(item, home, engine=engine, kind=kind, stage=stage)
    if clause:
        return clause
    if value is None:
        if home == "bench":
            return (", but a benchmark row's points are values to try -- "
                    "leave the setting off the row to run at its own "
                    "(engines/template.md 5.4)")
        return None if item.optional else (
            ", but this setting has no unset -- null is taken only by a "
            "setting whose unset is an answer; state a value, or leave it "
            "out (engines/template.md 5.4)")
    spelled = spell(item, value)
    return (_type_refusal(item.type, item.choices, spelled)
            or why_not(item, spelled, engine=engine, kind=kind))


def shown(value: Any) -> str:
    """A value as a refusal quotes it (§ 5.4), in the files' own words:
    ``null`` for unset, ``true`` / ``false``, a triple as the list it is
    written as."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(shown(v) for v in value) + "]"
    return repr(value)


def spell(item: "Item", value: Any) -> Any:
    """``value`` in ``item``'s declared type, LOSSLESSLY (§ 5.4,
    :func:`as_declared`): a whole float for a count is the count, a list of
    three a triple.  A value the type cannot hold stays as it was, for
    :func:`accepts` to refuse by name."""
    return as_declared(item.type, value)


#: A NUMBER AS A PERSON TYPES ONE: decimal, an optional exponent -- never
#: ``0x10``, ``1_000``, ``inf`` or ``nan``, which Python's readers take.
_TYPED_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def _typed_number(text: str):
    t = text.strip()
    if not _TYPED_NUMBER.fullmatch(t):
        return None
    return int(t) if re.fullmatch(r"[+-]?\d+", t) else float(t)


def read_text(item: "Item", text: Any) -> Tuple[Any, Optional[str]]:
    """WHAT A PAGE SENT FOR ``item`` -- a cell's or a card row's text, or a
    form's value -- read as its declared type (§ 5.4): ``(value, None)``,
    the value :func:`spell`'d, or ``(text, clause)`` when it is not that
    type, the clause read after *"<name> = <text>"*.  A blank is ``(None,
    None)``, which the caller takes as *nothing typed*; whether the value
    may stand in a home is :func:`accepts`'s, asked after.  Every page
    reads with this, and keeps no reader of its own.

    Text is read as a person types it, and as a file holds it: a number in
    decimal with an exponent if any, as JSON writes one (``1e3`` is 1000),
    never ``0x10`` or ``1_000``; ``true`` / ``false``; an
    enum's member by its spelling, ``"2"`` the count 2; a grid in its three
    spellings -- ``4,4,1``, ``4x4x1``, ``4 4 1`` -- or as the list
    ``[4, 4, 1]``; a list separated by commas or spaces.  A value that
    arrives typed (a JSON number, a list) is spelled as it is."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return None, None
    if isinstance(text, str):
        value, why = _read_typed(item, text)
        if why:
            return text, why
    else:
        value = spell(item, text)
    why = _type_refusal(item.type, item.choices, value)
    return (text, why) if why else (value, None)


def _read_typed(item: "Item", text: str) -> Tuple[Any, Optional[str]]:
    """Typed ``text`` read as ``item``'s type, its shape alone -- the
    number, the switch, the member, the list -- for :func:`read_text`."""
    kind = item.type
    if kind == "text":
        return text, None
    t = text.strip()
    if kind == "str":
        return t, None
    if kind == "bool":
        if t in ("true", "false"):
            return t == "true", None
        return t, ", which is not true or false"
    if kind == "enum":
        member = next((c for c in item.choices or () if str(c) == t), None)
        return (member if member is not None else t), None
    if kind in ("int", "pow2", "float"):
        n = _typed_number(t)
        if n is None:
            return t, f", which is not {_TYPE_WORDS[kind]}"
        return spell(item, n), None
    inner = t.strip("[]() ")
    pieces = [p for p in re.split(r"[,\sx]+" if kind in ("int3", "float3")
                                  else r"[,\s]+", inner) if p]
    if kind == "strlist":
        return [p.strip("'\"") for p in pieces], None
    numbers = [_typed_number(p) for p in pieces]
    if (not pieces or any(n is None for n in numbers)
            or (kind in ("int3", "float3") and len(numbers) != 3)):
        return t, f", which is not {_TYPE_WORDS[kind]}"
    return spell(item, numbers if kind == "intlist" else tuple(numbers)), None


def default_of(item: "Item", kind: str) -> Any:
    """THE ONE DEFAULT (§ 5.4): ``kind``'s `recommended` value for ``item``,
    else the item's `default` -- ``None`` for an item with neither, whose
    not-set line says what applies (§ 6.6, obligation 4).  What `resolve`
    lays for a run setting no card states, what Task setup shows in a blank
    row and what migrate asks whether a value was the default."""
    rec = recommended_for(item, kind)
    return rec if rec is not None else item.default


def select(t: "Template", *, category=None, engine=None,
           kind=None, read_by=None, allocation=None,
           citation=None, role=None, stages=None,
           shared=None, calculation=None) -> List[Item]:
    """The items matching every filter given, **in category order**.

    One function, one file, every reader (`engines/template.md` § 8.0).
    Each argument filters on an axis the item already declares; **omitting
    one means do not filter on it**, so ``select(t)`` is every item -- which
    is exactly what ``prep`` step 2 wants.

        select(t, category="accuracy", engine="siesta")   # one panel
        select(t, kind=("engine", "deck"), engine=e)      # the deck writer
        select(t, category="execution", engine=e)         # a benchmark

    **A filter, not a query language.** The axes are closed vocabularies
    declared on the item, so this is a dict comparison -- no expression to
    parse, no index to build, and a reader in another language does the same
    with ``tomllib.load`` and a comprehension. § 8's *"a reader never asks a
    second source"* holds: this is a convenience over data the caller already
    has in hand, never a service it must call.

    Category order, not declaration order, because the categories ARE the
    reading order (§ 6.2) and a surface renders panels top to bottom. Within
    a category the config's own field order survives, which is § 8's rule
    that a template a person reads is not a third arrangement of the same
    things.
    """
    _check_engine(t, engine)
    want_cat = _as_tuple(category)
    want_kind = _as_tuple(kind)
    # ``read_by`` filters the same way its siblings do -- any-of, scalar or
    # sequence.
    want_read_by = _as_tuple(read_by)
    out: List[Item] = []
    for it in t.items:
        # `engines` empty means ALL engines (§ 6.3) -- absence is not a
        # narrower claim than presence, it is a wider one.
        if engine and it.engines and engine not in it.engines:
            continue
        if want_cat and not (set(want_cat) & set(it.category)):
            continue
        if want_kind and it.kind not in want_kind:
            continue
        if want_read_by and not (set(want_read_by) & set(it.read_by)):
            continue
        # THE machine-answered filter -- `select(t, allocation=True)` is the
        # one API for *"which settings does the scheduler answer?"*.
        if allocation is not None and bool(it.allocation) is not bool(allocation):
            continue
        # THE citation-answered filter -- the one API for *"which settings
        # does the cited run answer?"*, so `prep` asks the template rather
        # than carrying its own list of field names.
        if citation is not None and bool(it.citation) is not bool(citation):
            continue  # `True` = some kind's citation answers it
        if role is not None and bool(it.role) is not bool(role):
            continue  # `True` = some kind's stage role answers it
        if shared is not None and bool(it.shared) is not bool(shared):
            continue  # `True` = binds every rung of some kind
        if stages is not None:
            # A RUNG'S ROLES (`stage_roles`) -- one, or a SIESTA vibration
            # stage's three -- not a boolean: "does this rung read this
            # item?".  An item declaring no stages is read by any -- the
            # ordinary case -- so it matches every role asked about.
            if it.stages and not (set(_as_tuple(stages)) & set(it.stages)):
                continue
        # A CALCULATION KIND, read as the stage is: "does this item apply to
        # it?".  An item declaring no calculations applies to every kind.
        if (calculation is not None and it.calculations
                and calculation not in it.calculations):
            continue
        out.append(it)

    def _rank(it: Item) -> int:
        # The item's PRIMARY category is its panel (§ 6.2); an item with no
        # category sorts last rather than crashing, because a reader's job
        # is to report a malformed file, not to be the thing that breaks.
        if not it.category:
            return len(CATEGORIES)
        try:
            return CATEGORIES.index(it.category[0])
        except ValueError:
            return len(CATEGORIES)

    return sorted(out, key=_rank)      # stable: ties keep declaration order


def one(t: "Template", name: str, *, engine: str = None) -> Optional[Item]:
    """The item called *name*, or ``None`` when it does not apply here.

    **``None`` and an exception mean different things, and that is the
    point.** ``None`` is *"this calculation has no such parameter for this
    engine"* -- ``one(t, "mesh_cutoff", engine="pyscf")`` on a template whose
    item declares ``engines = ["siesta"]``. It raises when the name is not in
    the file at all, because that is a caller asking for something this
    format never had: *"does not apply"* and *"should be here and is not"*
    must never read the same. Law A, applied to a lookup.
    """
    _check_engine(t, engine)
    for it in t.items:
        if it.name != name:
            continue
        if engine and it.engines and engine not in it.engines:
            return None                      # declared, but not for this engine
        return it
    raise KeyError(
        f"no item {name!r} in this template (engines={list(t.engines)}). If the "
        f"parameter exists for another engine it would still be listed, so "
        f"this is a name this template never carried -- not an item that "
        f"does not apply here (engines/template.md § 8.0).")


#: A molbuilder version: three whole numbers.
_VERSION = re.compile(r"\d+\.\d+\.\d+")


def version_key(version: Optional[str]) -> Tuple[int, ...]:
    """A version as numbers to compare -- ``None``, a file written before
    versions were recorded, as the earliest of all."""
    return tuple(int(x) for x in version.split(".")) if version else (0,)


#: TWO ITEMS WHOSE 0 CHANGED MEANING AT 2.1.0 (`engines/transport.md`
#: § 2a.13).  Before, molbuilder wrote nothing for a 0 and the engine's own
#: rule applied -- a tenth of the leads' broadening; since, a 0 is written
#: as 0 -- no broadening on the device, which TranSIESTA and tbtrans take as
#: a value of its own -- and the engine's rule is the item left blank.  A
#: template written before that holds 0 for one is refused, naming
#: ``jobset migrate``, which rewrites it blank: what it meant.
ZERO_WAS_NOT_SET = ("negf_neq_eta_ev", "tbt_contours_eta_ev")
ZERO_MEANING_CHANGED_AT = "2.1.0"


def written_before_zero_changed(tmpl: "Template") -> bool:
    """Was the calculation written before a 0 on :data:`ZERO_WAS_NOT_SET`
    changed meaning -- its template's ``written_by`` older, or absent?"""
    return version_key(tmpl.written_by) < version_key(ZERO_MEANING_CHANGED_AT)


def stage_zeros_written_before(tmpl: "Template", stages=()
                               ) -> List[Tuple[str, str]]:
    """``(stage, item)`` for each stage in ``stages`` whose override holds
    0 for an item of :data:`ZERO_WAS_NOT_SET` in a calculation written
    before :data:`ZERO_MEANING_CHANGED_AT` -- what it meant was *not set*.
    The template's ``written_by`` dates the whole calculation, its stages'
    overrides in `task.json` with it (a rung's tab writes a 0 typed there
    into its bag)."""
    if not written_before_zero_changed(tmpl):
        return []
    return [(st.name, k) for st in stages
            for k, v in (st.overrides or {}).items()
            if k in ZERO_WAS_NOT_SET and v == 0]


def zeros_written_before(tmpl: "Template", stages=()) -> List[str]:
    """Where a calculation written before :data:`ZERO_MEANING_CHANGED_AT`
    holds 0 for an item of :data:`ZERO_WAS_NOT_SET` -- what it meant was
    *not set*: the template's own items by name, and each stage's override
    in ``stages`` as ``stage '<name>' <item>``
    (:func:`stage_zeros_written_before`)."""
    if not written_before_zero_changed(tmpl):
        return []
    return ([it.name for it in tmpl.items
             if it.name in ZERO_WAS_NOT_SET and it.is_set and it.value == 0]
            + [f"stage {name!r} {k}"
               for name, k in stage_zeros_written_before(tmpl, stages)])


def why_zeros_written_before(tmpl: "Template", where: List[str]) -> str:
    """The refusal of :func:`zeros_written_before`'s answer -- one sentence
    for the template reader and the stage door alike."""
    return ("template written by "
            + (f"molbuilder {tmpl.written_by}" if tmpl.written_by
               else f"a molbuilder from before {ZERO_MEANING_CHANGED_AT}")
            + f": {', '.join(where)} holds 0, which before "
            f"{ZERO_MEANING_CHANGED_AT} meant 'not set' (the program's own "
            f"rule, a tenth of the leads' broadening) and since means no "
            f"broadening on the device.  `molbuilder jobset migrate` in the "
            f"calculation's folder rewrites it as not set, which is what it "
            f"meant (docs/process/changelog.md 2.1.0).")


def config_from_template(text: str, config_cls):
    """An ordinary instance of *config_cls*, rebuilt from a template.

    What ``prep`` holds before it applies a stage's ``overrides``
    (`engines/stages.md` § 4).

    **An unset item is omitted rather than passed as ``None``**, so the class
    default applies. That is the correct reading rather than a loss: every
    field for which *unset* is a real state is annotated ``Optional[...]`` and
    defaults to ``None`` already, so omitting and passing ``None`` agree —
    while for a field that is not optional, passing ``None`` would replace a
    real default with a value its own type forbids.

    A field the template does not carry keeps the class default too: a template
    written against an older schema is **missing** fields, not wrong about them,
    and the per-field checks are what name the ones that moved.
    """
    known = template_fields(config_cls)
    # § 6.3: ONE file carries every engine the calculation can run on, so the
    # items are filtered to this engine's BEFORE they are read as its fields.
    # Without this, a catalogue serving SIESTA and PySCF is refused for both --
    # each sees the other's items as names its schema does not declare, which
    # is the very thing the `engines` axis exists to prevent.  `select` refuses
    # an engine the file does not serve, so *"does not run on that engine"*
    # stays distinct from *"no items matched"*.
    parsed = read_template(text)
    eng = engine_name(config_cls)
    mine = select(parsed, engine=eng) if parsed.engines else parsed.items
    vals = {it.name: it.value for it in mine if it.is_set}
    # A 0 WRITTEN BEFORE ITS MEANING CHANGED meant *not set*; read now, it
    # would be written as a real 0 -- another calculation, in silence.
    zeros = zeros_written_before(parsed)
    if zeros:
        raise ValueError(why_zeros_written_before(parsed, zeros))
    # NO CITATION REFUSAL HERE, and its absence is the ruling
    # (`engines/transport.md` § 2a.7): electrode and device cannot disagree,
    # because ONE value shared by every stage cannot disagree with itself,
    # and the value need not come from the cited run -- a template answering
    # `basis_size` for a transport calculation is the ordinary state.
    unknown = sorted(k for k in vals if k not in known)
    if unknown:
        # Refused, never dropped (U16): this is a file people edit by
        # hand, and an item the schema does not know is a typo, a renamed
        # field or a retired one -- either way, silently ignoring it
        # renders a deck missing what the person believes they set.  A
        # retired item is named as retired, so a template written before
        # the retirement is not sent looking for a misspelling.
        migrated = [k for k in unknown if k in MIGRATED_ITEMS]
        gone = [k for k in unknown if k not in MIGRATED_ITEMS
                and why_not_an_item(k) is not None]
        other = [k for k in unknown
                 if k not in MIGRATED_ITEMS and k not in gone]
        why = []
        if migrated:
            why.append(
                "template carries item(s) the electronic state replaced on "
                "2026-09-28 (science/chemistry-correctness.md § 2a): "
                + ", ".join(f"{k!r} ({MIGRATED_ITEMS[k]})" for k in migrated)
                + ".  " + MIGRATE_HINT)
        # A RETIRED OR RENAMED ITEM is named as such, in the one sentence
        # every door says (:func:`why_not_an_item`): migrate takes out what
        # runs as before without it and renames what was renamed; any other
        # value is the person's to drop, since the run would change.
        if gone:
            why.append("template: " + "; ".join(
                why_not_an_item(k, value=vals[k], is_set=True)
                for k in gone) + ".")
        if other:
            why.append(
                f"template names item(s) the {config_cls.__name__} schema "
                f"does not declare: {', '.join(map(repr, other))}.  A "
                f"template item is a schema field (engines/template.md § 7); "
                f"check the spelling against the schema's own names.")
        raise ValueError("  ".join(why))
    # A VALUE MUST BE ONE OF THE CATALOGUE'S CHOICES -- today's, not the ones
    # the file carries beside it.  The file's own declaration is what it was
    # written against, so an enum the catalogue has since changed parses
    # cleanly against itself and would reach the config as a value no
    # reader expects: `method = "RKS"` under the DFT/HF vocabulary is read as
    # Hartree-Fock (M6, 2026-09-28).  Refused by name instead.
    decls = {it.name: it for it in select(catalogue(), engine=eng)}
    stale = [(k, v) for k, v in vals.items()
             if k in decls and decls[k].type == "enum" and decls[k].choices
             and not is_member(v, decls[k].choices)]
    if stale:
        k, v = stale[0]
        raise ValueError(
            f"template's {k} = {v!r} is not one of today's choices "
            f"({', '.join(map(repr, decls[k].choices))})."
            + (f"  It was written before the electronic state's items "
               f"(science/chemistry-correctness.md § 2a).  {MIGRATE_HINT}"
               if k in STATE_ITEMS else ""))
    return config_cls(**vals)


#: Items the electronic state replaced (M6, 2026-09-28) -- a template still
#: carrying one is refused naming the migration, never told to delete a value
#: that decided what the run was (`plan.md` § 5s.2, decision 7).
MIGRATED_ITEMS = {
    "spin": "PySCF's 2S, now `unpaired_electrons`",
    "spin_total": "SIESTA's Spin.Total, now `unpaired_electrons`",
}

#: The one sentence every refusal of a pre-M6 template ends with.
MIGRATE_HINT = ("Run `molbuilder jobset migrate` in this calculation's "
                "folder (or name it with `--bundle`): it rewrites the old "
                "items and values into the electronic state's, keeping what "
                "the run was.")


class Retired(NamedTuple):
    """An item a schema once declared and no longer does."""
    #: Why it went, and where the retirement is recorded.
    why: str
    #: The release -- or, before releases were numbered, the day -- that
    #: retired it: a file `jobset migrate` rewrites without it is kept as
    #: ``<name>.pre-<at>``.
    at: str
    #: The values that run as before without it: `jobset migrate` removes
    #: the item holding one of these (or none), and refuses any other by
    #: name -- what the calculation was is kept, never rounded.  ``None``:
    #: every value -- nothing a calculation ran ever read it, or migrate
    #: states what it said in today's items (:attr:`migrate_does`).
    runs: Optional[Tuple[Any, ...]] = ()
    #: The engines whose schema retired it -- an item one engine keeps and
    #: another retired (`es_mode_selection`: PySCF's probe, no longer
    #: SIESTA's).  Empty: every engine.
    engines: Tuple[str, ...] = ()
    #: What `jobset migrate` does with it, as the refusal says it: takes it
    #: out, or -- an item whose value today's items say -- states them from
    #: it (SIESTA's Raman selector: Compute Raman and Raman for).
    migrate_does: str = "takes it out"

    def retired_on(self, engine: str) -> bool:
        """Whether ``engine``'s schema retired the item (:attr:`engines`)."""
        return not self.engines or engine in self.engines

    def runs_without(self, value: Any) -> bool:
        """Whether a calculation holding ``value`` runs as before without
        the item (:attr:`runs`)."""
        return self.runs is None or value in self.runs


#: SIESTA's Raman selector before 2.45.0 -- retired on SIESTA alone, PySCF's
#: orbital probe keeping it; `jobset migrate` states Compute Raman and Raman
#: for from it (`engines/vibration.md` § 5.11 § 9).
SIESTA_RAMAN_SELECTOR = ("es_mode_selection", "es_explicit_indices",
                         "freq_min_cm1", "freq_max_cm1")


#: Items a schema once declared and no longer does -- read by the refusal
#: above, so a template written before the retirement is told why, and by
#: `jobset migrate`, which removes one whose value runs as before
#: (`engines/template.md` § 3).
RETIRED_ITEMS: Dict[str, Retired] = {
    "es_top_n": Retired(
        "the top_n selector, retired 2026-09-28 -- engines/vibration.md "
        "§ 4.8", "2026-09-28"),
    "es_threshold": Retired(
        "the threshold selector, retired 2026-09-28 -- engines/vibration.md "
        "§ 4.8", "2026-09-28"),
    "optimizer": Retired(
        "geomeTRIC is the one PySCF optimizer; berny and with it the choice "
        "were retired 2026-09-29 -- engines/pyscf.md § 3", "2026-09-29",
        runs=("geometric",)),
    "copy_psml": Retired(
        "the pseudopotential staging switch, retired at 2.10.0: nothing read "
        "it, and prep always copies the calculation's pseudopotentials "
        "beside its decks -- engines/template.md § 3", "2.10.0",
        runs=(True, False)),
    **{name: Retired(
        f"SIESTA's molecular dynamics ({what}) left the optimization kind "
        f"at 2.40.2 with the Verlet and Nose run types: no calculation of "
        f"molbuilder runs dynamics, and no relaxation's deck ever wrote it "
        f"-- engines/template.md § 3", "2.40.2", runs=None)
       for name, what in (("md_initial_temperature", "its initial "
                           "temperature"),
                          ("md_target_temperature", "its thermostat's "
                           "target"),
                          ("md_length_timestep", "its time step"))},
    **{name: Retired(
        f"a SIESTA vibration's {what}, fixed by the kind until 2.45.0, "
        f"when Compute IR became a setting of both engines and the "
        f"force-constant block writes the charges when it is on -- "
        f"engines/vibration.md 5.3", "2.45.0", runs=None)
       for name, what in (("charge_hirshfeld", "Hirshfeld charges"),
                          ("charge_voronoi", "Voronoi charges"))},
    **{name: Retired(
        f"SIESTA's Raman modes are Compute Raman and Raman for since "
        f"2.45.0; {name} is PySCF's orbital probe's alone -- "
        f"engines/vibration.md 3.1", "2.45.0", runs=None,
        engines=("siesta",),
        migrate_does=("states Compute Raman and Raman for from it -- an "
                      "explicit list of modes, or a stage's own value, is "
                      "yours to restate as Raman for"))
       for name in SIESTA_RAMAN_SELECTOR},
}


#: Items renamed, each with its new name, where the rename is recorded and
#: the day it was (a file `jobset migrate` renames in is kept as
#: ``<name>.pre-<day>``) --
#: read by the refusal above, so a template written before the rename is told
#: the new spelling rather than to check it.  The old name is REFUSED, never
#: read as the new one: a template is a file people edit, and a quiet alias
#: would leave two spellings of one parameter alive.
RENAMED_ITEMS = {
    "elecs_bulk": ("electrodes_bulk",
                   "renamed 2026-09-29, and shared by the device and the "
                   "transmission -- engines/transport.md 6.1b",
                   "2026-09-29"),
}


def why_not_an_item(name: str, *, value: Any = None,
                    is_set: bool = False) -> Optional[str]:
    """Why ``name`` names no item when a release made it so -- retired, and
    the way on: `jobset migrate` when its value runs as before without it,
    else deleting it; renamed, and its new name -- or ``None`` for a name no
    release ever held (`engines/template.md` § 3).  The one sentence the
    template reader, the description's check and a stage's overrides say,
    ``value`` the one the name was given (``is_set``)."""
    if name in RETIRED_ITEMS:
        r = RETIRED_ITEMS[name]
        if is_set and not r.runs_without(value):
            return (f"{name} = {value!r} was retired from the schema "
                    f"({r.why}) at a value that no longer runs -- delete it "
                    f"to accept what runs now; the calculation it described "
                    f"cannot be run again")
        return (f"{name} was retired from the schema ({r.why}) -- "
                f"`molbuilder jobset migrate` in this calculation's folder "
                f"{r.migrate_does}")
    if name in RENAMED_ITEMS:
        new, why, _at = RENAMED_ITEMS[name]
        return (f"{name} is now {new} ({why}) -- `molbuilder jobset migrate` "
                f"in this calculation's folder renames it, keeping its value")
    return None


#: The module's public surface.
__all__ = ["SCHEMA", "SUFFIX", "KINDS", "TYPES", "CATEGORIES",
           "Item", "Template",
           "items_of", "help_for",
           "select", "one",
           "CATALOGUE", "catalogue", "load_catalogue",
           "template_with_values",
           "read_template", "config_from_template",
           "template_fields", "engine_name", "is_member",
           "STATE_ITEMS", "shared_by_every_stage", "why_shared",
           "shared_by", "shared_scope_words",
           "run_settings", "why_run_setting",
           "IDENTITY_ITEMS", "fixed_by_role", "role_answers",
           "fixed_answers", "fixed_on_every_rung", "why_role", "offered", "why_not_offered",
           "as_declared", "why_not", "why_not_an_item",
           "HOMES", "SAID", "TRIAL_PINS", "GRID_AXES", "is_run_setting",
           "home_of", "why_not_in",
           "accepts", "spell", "shown", "read_text", "default_of"]

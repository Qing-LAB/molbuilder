"""The transport composite's five stages — `engines/transport.md`
§ 1 and § 2a.11.

This module owns TWO facts and the renders that follow from them:

* **The ladder** (:data:`TRANSPORT_STAGES`): the five stages in
  dependency order.  Fixed by design, not configurable -- every
  description holds all five (`task.fixed_by_kind`).
* **The per-rung bags, as stages** (:func:`stages_for_transport`).  Which
  rung READS an override is the catalogue's ``stages`` declaration, asked
  through the one door every kind's description check and ``resolve`` ask
  (``template.unread_overrides``, `engines/template.md` § 6.4) -- what puts
  a person's T(E) window into the deck ``tbtrans`` runs rather than the one
  ``siesta`` runs.

**NOTHING IN THIS MODULE RENDERS A DECK.**  All five rungs go
through the framework's own pipeline — ``siesta.input.spec_for`` with
``calculation="transport"`` -> :mod:`molbuilder.transport.deck`, whose
``SHAPE_OF_RUNG`` tables the four deck shapes — which is what lets the
template's items reach them; the electronic contract is resolved by
``transport_rung.resolve_transport`` from the calculation's own template.

The cited relaxation DEFAULTS the electronic description at ``jobset
init``; it does not seal it (§ 2a.7).  The shared-contract invariant holds
because ONE value shared by five rungs cannot disagree with itself.

The launch half's facts also live here: the ladder's DAG (§ 1, § 6.1)
(:func:`stage_inputs`, read by prep's gather and the done-door), the
continuation rows (:func:`warm_declaration` — the seed's ``.DM``, the
device's ``.TSDE``), and the sweep's points (:func:`sweep_points`,
:func:`point_folders`, :func:`points_in`; their folder names are the
codec's one spelling, ``task.bias_token``).  A point is a frame and a
voltage (:class:`Point`, :func:`rung_points`; `engines/transport.md` § 2a.9
*Both axes*): every reader of a rung's points asks these doors and none asks
which axis varies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..structure import frame_words
from ..paths import KIND_ROLES
from ..parse.engines import siesta_grammar as _G

#: The composite's fixed ladder (§ 1) -- the five stages in
#: dependency order.  ``jobset init`` writes exactly these into
#: ``task.json``; `prep` names its rungs from the same tuple.  They are the
#: kind's rung ROLES too, so their one home is the catalogue's role
#: vocabulary (`paths.KIND_ROLES`, `engines/template.md` § 6.4).
TRANSPORT_STAGES = KIND_ROLES["transport"]

#: **WHICH FACT EACH RUNG ANSWERS WITH** — a column, because the five rungs
#: do not answer the same question and reading them as if they did is a
#: defect with a visible face.
#:
#: * ``"scf"`` — did the SCF converge, and at what energy.  The seed hands the
#:   device a density; the device hands TBtrans a Hamiltonian.
#: * ``"fermi"`` — an SCF *and* the lead's Fermi level, which is the reference
#:   the whole junction is measured against (§ 2a.13): T(E) is relative to it
#:   and ``G = G0 * T(E_F)`` is evaluated at it.
#: * ``"product"`` — **not an SCF at all.**  TBtrans runs none: there is
#:   nothing to converge and no total energy, and its result is the
#:   transmission the record already carries in its own ``points`` blocks.
STAGE_FACT = {
    "seed":         "scf",
    "electrode_L":  "fermi",
    "electrode_R":  "fermi",
    "device":       "scf",
    "transmission": "product",
}


def resolvable_override_names() -> frozenset:
    """The names a per-stage override may carry — **asked, never listed.**

    `prep` resolves a rung's config through
    :func:`molbuilder.resolve.resolve` against
    :class:`~molbuilder.config.siesta.SiestaConfig`, so a name that is not one
    of its fields is refused there by name, and a RUN SETTING -- the
    catalogue's ``execution`` items, the machine's answers among them -- is
    refused as an override: it is the rung's run card's (`engines/stages.md`
    § 6.8d, plan § 5w K5). Those two facts are the whole vocabulary, and
    this is where they are turned into one question both doors ask.

    Which names are run settings is the catalogue's own answer
    (``template.run_settings``), rather than a fourth frozenset.
    """
    import dataclasses

    from ..config.siesta import SiestaConfig
    from ..template import run_settings

    on_the_card = run_settings("siesta")
    return frozenset(f.name for f in dataclasses.fields(SiestaConfig)
                     if f.name not in on_the_card)


#: How close two voltages are when they are one point's, in V: a voltage is
#: typed to a few decimals and read back as written, so any difference
#: beyond float noise is another voltage.
SAME_VOLTS_V = 1e-9


def same_volts(a: float, b: float) -> bool:
    """``a`` and ``b`` are one voltage -- THE rule a reader asks when it
    finds a voltage it was given along the axis (the record, its average,
    the data file).  Points joined by the values one door gave both sides
    (`rung_points`) carry the same floats and are joined by them."""
    return abs(float(a) - float(b)) < SAME_VOLTS_V


def bias_points(task) -> Tuple[float, ...]:
    """The sweep's points, in the order the description states them —
    the codec already enforced that the list starts at 0.0 (the chain
    starts from equilibrium).  ``()`` and a single entry mean NO sweep:
    the stage keeps its plain single-deck layout, because the v-dir
    layer exists for the axis, not for every calculation
    (architecture § 0: a list with more than one element)."""
    bias = tuple(getattr(task, "bias", ()) or ())
    return bias if len(bias) > 1 else ()


# --------------------------------------------------------------------- #
#  Where a rung's attempts are (`engines/transport.md` § 2a.11; plan    #
#  § 5w K10)                                                            #
# --------------------------------------------------------------------- #

def per_point_rungs() -> frozenset:
    """The rungs a bias sweep runs once per point: the bias item's own
    `stages` (`template.PER_POINT`, the role item no rung answers with one
    value) -- the device and the transmission, declared once, in the
    catalogue."""
    from ..template import PER_POINT, catalogue, one
    return frozenset(rung for name in PER_POINT
                     for rung in one(catalogue(), name, engine="siesta").stages)


def sweep_points(task, stage: str) -> Tuple[float, ...]:
    """The voltages ``stage`` runs at, as a sweep -- the bias list
    (:func:`bias_points`) for a rung the bias item names
    (:func:`per_point_rungs`: the device, the transmission) when the
    calculation states ``low_bias_approximation: false``; ``()`` for every
    other rung, a single bias, and the low-bias approximation, under which
    the device and the transmission each run once, at 0 V
    (`engines/transport.md` § 2a.10, § 2a.11)."""
    if getattr(task, "low_bias_approximation", None) is not False:
        return ()
    return bias_points(task) if stage in per_point_rungs() else ()


def frame_token(frame: int) -> str:
    """A frame's folder token: its index in the cited set, three digits --
    ``f000`` the base, frame index 0 (`engines/transport.md` § 2a.9;
    `model/structure.md` § 2.2e: the code's index everywhere but the UI's
    display, a folder name included)."""
    return f"f{int(frame):03d}"


@dataclass(frozen=True)
class Point:
    """One point of a rung that runs per point (`engines/transport.md`
    § 2a.9 *Both axes*, § 2a.11): a frame of the cited set and a voltage,
    each ``None`` when that axis does not vary for the calculation -- and
    its folder below the stage or a run, a level for each axis that varies:
    ``f001/v0.2``, ``f001``, ``v0.2``."""
    frame: Optional[int] = None
    volts: Optional[float] = None
    #: How many frames the set holds -- said in the point's words, never
    #: part of which point it is.
    of: Optional[int] = field(default=None, compare=False)

    @property
    def rel(self) -> str:
        from ..task import bias_token
        return "/".join(
            ([frame_token(self.frame)] if self.frame is not None else [])
            + ([bias_token(self.volts)] if self.volts is not None else []))

    @property
    def to_run(self) -> str:
        """THE WAY FROM THE POINT'S FOLDER UP TO ITS RUN'S, as a deck in
        the point names a file in the run's scratch folder
        (`engines/transport.md` § 2a.11, *What a run's points share*):
        ``..`` per level the point
        has -- ``../..`` for ``f001/v0.2`` -- and ``.`` for a point with no
        level, which is the run itself."""
        return "/".join([".."] * len(self.rel.split("/"))) if self.rel else "."

    @property
    def bias_v(self) -> float:
        """The voltage the point runs at: its own, or 0 V on a calculation
        that sweeps no bias (`engines/transport.md` § 2a.10)."""
        return 0.0 if self.volts is None else float(self.volts)

    def words(self) -> str:
        """The point as a message names it, in full: ``frame index 1 (of 3)
        · 0.2 V`` (`structure.frame_words`) -- the voltage said where the
        axis does not vary too, the 0 V it runs at; its folder
        ``f001/v0.2``."""
        return " · ".join(
            ([frame_words(self.frame, self.of)] if self.frame is not None
             else []) + [f"{self.bias_v:g} V"])

    def to_wire(self) -> Dict[str, object]:
        """THE POINT ON THE WIRE -- one form, for every answer that names a
        point (prep's, the run list's, the record's): its frame's index, how
        many frames the set holds, the voltage it runs at."""
        return {"frame": self.frame, "of": self.of, "bias_v": self.bias_v}


def frames_of(base) -> int:
    """How many frames the cited junction holds -- the citation's, never
    typed (`engines/transport.md` § 2a.9): the composed record's
    (`slot-provenance.json`, written at the calculation's first prep).  No
    record yet means nothing is prepared, so no rung has points on disk: 1."""
    from .compose import read_provenance
    return int((read_provenance(base) or {}).get("frames") or 1)


def rung_points(task, stage: str, *, frames: int) -> Tuple[Point, ...]:
    """THE POINTS ``stage`` RUNS AT -- one door, asked by prep, the gather,
    the walk, `status` and the record (`engines/transport.md` § 2a.11): for a
    rung that runs per point (:func:`per_point_rungs`: the device, the
    transmission), each of the cited set's ``frames`` when it holds more than
    one, and each voltage of a self-consistent sweep (:func:`sweep_points`),
    frames outer -- each frame's voltages one chain; ``()`` for a rung that
    varies over neither, which runs once."""
    volts = sweep_points(task, stage)
    fs = (tuple(range(int(frames)))
          if int(frames) > 1 and stage in per_point_rungs() else ())
    if not volts and not fs:
        return ()
    return tuple(Point(frame=f, volts=v, of=int(frames) if fs else None)
                 for f in (fs or (None,)) for v in (volts or (None,)))


def point_folders(base, task, stage: str, *, frames: Optional[int] = None
                  ) -> List[Tuple[Path, Point]]:
    """A rung's PREPARED points -- ``[(<stage>/<point>, Point)]``, each
    folder holding that point's deck and run script, what every run of the
    stage copies -- or ``[]`` for a rung that runs once
    (`engines/transport.md` § 2a.11).  ``frames`` is the cited set's count
    when the caller holds the junction (prep, before its record is on
    disk); else the record's (:func:`frames_of`)."""
    from ..jobset.materialize import stage_home
    stage_dir = stage_home(base, task, stage).dir
    n = frames if frames is not None else frames_of(base)
    return [(stage_dir / pt.rel, pt)
            for pt in rung_points(task, stage, frames=n)]


def points_in(run, task, stage: str, *, base) -> List[Tuple[Path, Point]]:
    """A run's points -- ``[(run-<n>/<point>, Point)]`` -- or ``[]`` for a
    run of a rung that runs once (`engines/transport.md` § 2a.11); the
    frames are the record's of the calculation at ``base``."""
    return [(Path(run) / pt.rel, pt)
            for pt in rung_points(task, stage, frames=frames_of(base))]


#: The regions a lead rung is cut on -- its stage name to its region label.
_LEAD_REGION_OF = {"electrode_L": "L-electrode", "electrode_R": "R-electrode"}


def rung_label(task_label: str, stage: Optional[str]) -> str:
    """THE LABEL A RUNG'S FILES ARE NAMED ON (`job-contracts.md` § 2.1, rule
    2): the calculation's own for every rung but a lead's, whose files --
    the engine's ``.TSHS``, ``.DM``, ``.XV`` and ours alike -- are on the
    lead's own stem, its ``SystemLabel`` (`transiesta.electrode_hs_stem`),
    which the device deck's ``TS.Elec.<name> HS`` line names.  Asked by the
    element's label (`resolve._label_for`), the job's names
    (`materialize.run_names`) and the run door's reading of a folder
    (`runs._position`), so no reader composes a lead's name a second way.
    ``stage`` is the rung's name; ``None`` -- no ladder -- is the label."""
    from .transiesta import electrode_hs_stem
    region = _LEAD_REGION_OF.get(stage or "")
    return electrode_hs_stem(task_label, region) if region else task_label


def lead_tail_bytes() -> int:
    """How much longer a lead's stem is than the calculation's label -- what
    the label cap budgets for a transport calculation (`identity._cap_for`)."""
    from .transiesta import electrode_hs_stem
    return max(len(electrode_hs_stem("", r).encode("utf-8"))
               for r in _LEAD_REGION_OF.values())


def stage_inputs(stage: str, task_label: str):
    """The ladder's DAG (§ 1, § 6.1), as data: ``[(upstream stage, filename)]`` this
    stage CONSUMES from the concluded attempts of the stages before it.

    The filenames are derived from the SystemLabel rules the renderers
    already fixed — the seed and device share the task's label, each
    electrode's label IS its ``.TSHS`` stem (`rung_label`, one spelling) —
    so producer and consumer cannot disagree about a name.

    The seed's density starts the device's SCF (user, 2026-10-12: every
    ladder keeps its seed); the electrode and device rows are the physics
    (the self-energies, and the converged H the transmission reads).
    """
    from ..runfiles import compose
    elec = [(s, compose(rung_label(task_label, s), _G.TSHS))
            for s in ("electrode_L", "electrode_R")]
    if stage == "device":
        return [("seed", compose(task_label, _G.DM))] + elec
    if stage == "transmission":
        # TBtrans reads the device's converged H -- which SIESTA 5.x
        # writes as <label>.TS.HSX (the sparse container that replaced
        # the 4.x device .TSHS; measured live 2026-08-29 on 5.4.2) --
        # plus the electrode .TSHS the TS.Elec blocks in the (shared)
        # deck text reference.  The .TSDE is NOT consumed: the TS.HSX
        # already carries the bias point's converged potential.
        return [("device", compose(task_label, _G.TS_HSX))] + elec
    return []


def inputs_of(task, stage: str):
    """:func:`stage_inputs` for ``task``'s ladder."""
    return stage_inputs(stage, task.label)


def hands_on(stage: str, task_label: str, base_dir) -> list:
    """What a point of ``stage`` hands to the next point's start -- the
    calculation's ``along`` row (`warm-files.toml` [transport]), of the
    files this rung itself declares (:func:`warm_declaration`: the device's
    ``.TSDE``; the transmission declares none) -- ``[]`` for a rung with no
    points (`engines/transport.md` § 2a.11).  The one answer the walk hands
    on by and :func:`products_of` asks a point to hold."""
    if stage not in per_point_rungs():
        return []
    from ..runfiles import compose
    from ..warmfiles import warm_list
    own = {w.name for w in warm_declaration(stage, task_label, base_dir)}
    return [f for f in (compose(task_label, suf) for suf
                        in warm_list("siesta", "transport", base_dir).along)
            if f in own]


def products_of(stage: str, task_label: str, *, base_dir=None) -> list:
    """What a run of ``stage`` must hold to be DONE: every file a rung
    downstream takes from it (:func:`stage_inputs`, the DAG, read the other
    way) -- for the transmission, which nothing downstream reads, its own
    deliverable: the ``.TBT.nc``, or a polarized point's ``.TBT_UP.nc`` and
    ``.TBT_DN.nc`` both, as TBtrans names its channels
    (`siesta_grammar.TBT_CHANNELS`; one entry of alternative sets, which
    `continuation.done` reads) -- and, with ``base_dir``, what a point
    hands to the next point: the calculation's ``along`` row, of the files
    this rung itself declares (the device's ``.TSDE``; the transmission
    declares none).  THE ONE LIST `done` is asked with
    (`engines/transport.md` § 2a.11): the gather, the walk and `status`
    read it here and nowhere else."""
    from ..runfiles import compose
    if stage == "transmission":
        from ..parse.engines.siesta_grammar import TBT_CHANNELS
        tag = dict(TBT_CHANNELS)
        out = [((compose(task_label, _G.tbt_nc(tag["unpolarized"])),),
                (compose(task_label, _G.tbt_nc(tag["up"])),
                 compose(task_label, _G.tbt_nc(tag["down"]))))]
    else:
        out = sorted({fn for downstream in TRANSPORT_STAGES
                      for up, fn in stage_inputs(downstream, task_label)
                      if up == stage})
    # WHAT A POINT HANDS TO THE NEXT POINT is a per-point rung's alone: the
    # seed has no points, and its restart state is its own `.DM` (§ 2a.15).
    if base_dir is not None:
        out += [f for f in hands_on(stage, task_label, base_dir)
                if f not in out]
    return out


def warm_declaration(stage: str, task_label: str, base_dir=None):
    """What a transport rung takes from a run of itself it CONTINUES --
    `launch` opening its next attempt after a stop -- the § 4.2a vocabulary
    rows for the transport type, on the stages where continuing means
    anything -- the seed and the device, each with the vocabulary's carry
    rows (`warm-files.toml` [transport]: the ``.DM``, and the ``.TSDE``,
    the NEGF density, read under ``DM.UseSaveDM``, which the device deck
    writes ``.true.``).  A rung takes no
    ``--from`` at prep (`continuation.cannot_be_named`).

    The electrode single-points and the transmission post-processing
    declare NOTHING — re-running them is cheaper than reasoning about a
    half-finished copy, and an empty declaration is what makes a re-launch
    of one refuse by name instead of copying dead weight.
    """
    if stage not in ("seed", "device"):
        return []
    from ..warmfiles import warm_declaration
    return warm_declaration("siesta", task_label, "transport", base_dir)


#: EACH ENGINE'S NAME ON THE TRANSPORT TAB -- the program that runs the
#: open-boundary calculation on it (`engines/transport.md` § 3.8.8).
ENGINE_WORDS = {"siesta": "TranSIESTA", "pyscf": "PySCF-NEGF"}


def engine_tabs() -> List[Dict[str, object]]:
    """The transport tab's engine strip, one entry per engine molbuilder
    knows (`runfiles.engines`), those that run a transport calculation
    first (`electronic_state.engines_for`): its name, its word here,
    whether it runs one, and, for one that does not, why -- the disabled
    tab's title (§ 3.8.8)."""
    from ..electronic_state import engines_for
    from ..runfiles import engines
    runs = engines_for("transport")
    return [{"name": e, "label": ENGINE_WORDS.get(e, e), "runs": e in runs,
             "why": why_no_transport(e)}
            for e in sorted(engines(), key=lambda e: e not in runs)]


def why_no_transport(engine: str) -> Optional[str]:
    """Why ``engine`` cannot describe a transport calculation, or ``None``
    when it can -- one sentence for the disabled tab and every door that
    refuses the engine (§ 3.8.8)."""
    from ..electronic_state import engines_for
    from ..runfiles import engines
    runs = engines_for("transport")
    if engine in runs:
        return None
    ships = " and ".join(ENGINE_WORDS.get(e, e) for e in runs)
    if engine not in engines():
        return (f"{engine!r} is no engine molbuilder knows; transport ships "
                f"on {ships}")
    return (f"no {ENGINE_WORDS.get(engine, engine)} backend is built yet; "
            f"transport ships on {ships}")


#: What each rung computes and hands on -- the ONE-LINE NOTE a surface puts
#: beside the rung's name (`engines/transport.md` § 2a.3 "the results that
#: propagate", § 6.1; the tab strip of § 3.8.2a).  The index a surface shows
#: is the rung's position in TRANSPORT_STAGES, one-based.
RUNG_NOTES = {
    "seed": ("A closed periodic SCF of the whole junction.  It writes the "
             "density the device starts from -- a starting guess only: it "
             "spares the device iterations and changes no result.  "
             "k-points: the shared k-point grid in the electrode plane "
             "(x, y), and one k-point along z, because the cell is long "
             "along z and the device only starts from this density."),
    "electrode_L": ("The left lead as a bulk crystal.  It writes the "
                    "Hamiltonian, overlap and Fermi level the device builds "
                    "the left self-energy from, and the device takes them "
                    "as given.  k-points: the shared k-point grid in the "
                    "electrode plane (x, y), which must be the device's, "
                    "and many k-points along z (the lead's own count, 100 "
                    "unless set), because the lead's Hamiltonian and Fermi "
                    "level depend on them."),
    "electrode_R": ("The right lead, likewise, on the same k-points as the "
                    "left.  Two lead runs even when the leads are "
                    "identical, so the record stays auditable and the "
                    "Results tab compares their Fermi levels."),
    "device": ("The open-boundary NEGF SCF of the junction between the two "
               "leads, at the bias.  It writes the converged Hamiltonian "
               "the transmission reads -- one per bias point.  k-points: "
               "the shared k-point grid in the electrode plane (x, y), and "
               "one k-point along z, where the leads' self-energies stand "
               "in for everything beyond the cell."),
    "transmission": ("T(E) over the energy window, from that Hamiltonian.  "
                     "tbtrans runs no SCF, so this tab has no SCF settings: "
                     "the ones behind this T(E) are the device's and the "
                     "two leads', on their tabs, and T(E) is only as "
                     "converged as those runs.  k-points: its own "
                     "transmission k-point grid in the electrode plane "
                     "(x, y), finer than the shared one -- three times it "
                     "unless you set it -- because T(E) changes more "
                     "sharply from one k-point to the next than the density "
                     "does; and one k-point along z.  Nothing consumes its "
                     "output, so changing the window or the grid re-runs "
                     "the transmission alone, never an NEGF cycle."),
}


def stages_for_transport(bags=None):
    """The composite's five rungs as ``Stage`` objects, each carrying the
    overrides that are ITS OWN -- *bags* maps a rung name to that rung's
    mapping, and a rung not named gets an empty bag.

    One door, so the two construction sites -- `jobset init` and the web
    describe door -- cannot build the ladder differently.  A rung this
    ladder does not have is refused by name.

    **Ownership is not decided here.**  The per-rung form asks the question
    per rung (`engines/transport.md` § 3.8.2a), so a bag arrives already
    placed, and the description's own check and ``resolve`` refuse a value
    on a rung that does not read it (``template.unread_overrides``).
    """
    from ..task import Stage

    bags = dict(bags or {})
    unknown = sorted(set(bags) - set(TRANSPORT_STAGES))
    if unknown:
        raise ValueError(
            f"no such rung {', '.join(map(repr, unknown))} -- a transport "
            f"ladder's rungs are {', '.join(TRANSPORT_STAGES)}.")
    return [Stage(name=n, overrides=dict(bags.get(n) or {}))
            for n in TRANSPORT_STAGES]



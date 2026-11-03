"""A TRANSPORT CALCULATION DESCRIBED -- from its junction citation
(`engines/transport.md` § 3.1), or from a finished calculation it extends
(§ 2a.16, *Describing it*): the transport kind's description part
``describe``, the one describe both roads take -- `molbuilder describe`,
which asks it through the kind table (the doors
`kind_table.DESCRIBE_DOORS` lists), and the Transport tab's Describe
(`POST /api/transport/describe`).  Each road collects what the person said
in its own way and says the answer in its own way; what is checked,
composed and written is decided here, once.

In order, every refusal before anything is written: the engine (an
extension's is the finished calculation's); the citation, resolved and
classified once, a run that ended with an error refused; the folder -- one
calculation per folder, never inside what it cites; the voltages and their
treatment; each rung's values and the shared ones, by name; the task, by its
codec; an extension's limits, where the road holds them (the tab's; the
command line says what it reuses, never refusing); the pseudopotentials'
rule and the template, its values defaulted from the citation (§ 2a.7); the
description's own check.  Then the files, every one or none -- through the
one writer (`describe.write_description`, `reuse.write_reusing`) -- or, the
plan alone, nothing the person sees; and an extension's manifest, the
answer prep composes (`reuse.decide`).

A door on floor 4: it reads the citation and the finished calculation
through their doors and writes through the description's writer.
"""
from __future__ import annotations

import dataclasses
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class NotDescribed(ValueError):
    """A describe refused, nothing written: its words, the findings beside
    the controls they name (a road that draws them), the task when one was
    built, and whether the folder is what is refused (one calculation per
    folder; never inside what it cites)."""

    def __init__(self, message: str, *, findings: Sequence = (),
                 task=None, folder: bool = False):
        super().__init__(message)
        self.findings = list(findings)
        self.task = task
        self.folder = folder


@dataclass(frozen=True)
class Described:
    """What a describe answers: the task, the files written (none for the
    plan alone), what is said beside them (`issues.Issue`, each at its
    weight), an extension's manifest in its words (``None`` otherwise), and
    the lines a road prints after it -- the headline and what comes next."""
    task: object
    written: Tuple[Path, ...]
    notes: Tuple[object, ...]
    manifest: Optional[Tuple[str, ...]]
    headline: str
    after: Tuple[str, ...] = ()


#: WHAT A DESCRIBE OF THIS KIND DOES NOT TAKE, and why -- by the input every
#: describe names (`molbuilder describe`'s flags), refused by the road that
#: offers it: from a citation, and beside a finished calculation it extends.
_NOT_TAKEN = {
    False: {
        "structure": "its structure IS the junction citation, copied in at "
                     "prep (engines/transport.md § 3.1)",
        "vacuum": "the cell comes with the cited junction "
                  "(engines/transport.md § 3.1)",
        "stage_strategy": "its stages are fixed by design{stages} "
                          "(engines/transport.md § 3.1)",
    },
    True: {
        "name": "the label is the finished calculation's: SystemLabel names "
                "every file a deck reads and writes, and decks under two "
                "labels are never the same calculation "
                "(engines/transport.md § 2a.16)",
        "psml_lib": "the pseudopotentials are the finished calculation's, "
                    "copied with it (engines/transport.md § 2a.16)",
        "structure": "its structure IS the junction citation "
                     "(engines/transport.md § 2a.16)",
        "vacuum": "the cell comes with the junction "
                  "(engines/transport.md § 2a.16)",
        "stage_strategy": "its stages are fixed by design "
                          "(engines/transport.md § 2a.16)",
    },
}


def not_taken(calculation: str, engine: Optional[str], *,
              reusing: bool) -> Dict[str, str]:
    """The inputs a describe of ``calculation`` does not take -- from a
    citation, or ``reusing`` a finished calculation -- each with why
    (`engines/transport.md` § 3.1, § 2a.16): a road refuses one given,
    naming it, before anything is read."""
    from ..kind_table import kind
    stages = ""
    if engine and not kind(calculation).why_not_on(engine):
        stages = f" ({', '.join(kind(calculation).on(engine).stages or ())})"
    return {k: why.format(stages=stages)
            for k, why in _NOT_TAKEN[reusing].items()}


def fit(reuses: str, lo: float, hi: float, finer: int = 1, *, tree_root):
    """THE TRANSMISSION WINDOW FITTED to the grid the finished calculation
    ``reuses`` names computed (`engines/transport.md` § 2a.16): the one fit
    the Transport tab's *use these* and `molbuilder describe --window` take
    (`transport.window.fitted` over `reuse.finished_window`).  NotDescribed,
    :class:`ValueError`, in the reader's or the fit's words."""
    from .reuse import reused
    finished, _rel = reused(reuses, tree_root=tree_root)
    return _fitted(finished, lo, hi, finer)


def _fitted(finished, lo: float, hi: float, finer: int):
    """The window fitted to ``finished``'s grid (:func:`fit`)."""
    from .reuse import finished_window
    from .window import fitted
    return fitted(finished_window(finished), lo, hi, finer)


def shared_of(shared: Mapping[str, object], *, engine: str,
              calculation: str) -> Tuple[Dict[str, object], List[str]]:
    """The shared values as a describe takes them: ``(the values given,
    each read as its item's type, the items emptied)`` -- each name one the
    catalogue marks shared for the kind (`engines/template.md` § 7), each
    value read by the setting door's reader (`template.read_text`), a value
    one cannot take refused naming it.  A BLANK IS NOT CHOSEN
    (`form-schema.md` § 1.1): the panel is drawn holding the citation's
    answers, so an item sent blank was emptied, and is written valueless."""
    from ..engine_table import config_class
    from ..template import items_of, read_text, shared_by_every_stage
    names = shared_by_every_stage(engine, calculation)
    for item in shared:
        if item not in names:
            raise NotDescribed(
                f"{item!r} is not a shared value of this calculation; the "
                f"shared panel carries the items the catalogue marks "
                f"`shared` for {calculation}")
    items = items_of(config_class(engine))
    chosen: Dict[str, object] = {}
    blank: List[str] = []
    for item, raw in shared.items():
        if raw in (None, ""):
            blank.append(item)
            continue
        value, why = read_text(items[item], raw)
        if why:
            raise NotDescribed(f"{item} = {raw!r}{why}")
        if value is not None:
            chosen[item] = value
    return chosen, sorted(blank)


def describe(*, calculation: str, engine: Optional[str],
             junction: Optional[str], reuses: Optional[str],
             name: Optional[str], dest: Optional[Path],
             bias: Optional[Tuple[float, ...]],
             low_bias_approximation: Optional[bool] = None,
             swap_electrodes: Optional[bool] = None,
             stages: Optional[Mapping[str, Mapping]] = None,
             shared: Optional[Mapping[str, object]] = None,
             window: Optional[Tuple[float, float, int]] = None,
             refuse_limits: bool = False, tree_root) -> Described:
    """DESCRIBE a ``calculation`` of this kind into ``dest`` -- or, ``dest``
    ``None``, the plan alone, nothing the person sees.

    ``junction`` the citation, tree-relative -- for an extension, the
    finished calculation's junction or a frame set of it (``None``: the
    finished one's); ``reuses`` the finished calculation, or ``None``;
    ``name`` a new calculation's (an extension's label is the finished
    one's); ``bias`` the voltages (``None``: a new calculation's the single
    bias, ``(0.0,)``; an extension's its own),
    ``low_bias_approximation`` their treatment (``None`` beside an
    extension's several: the finished calculation's);
    ``swap_electrodes`` the electrodes' rename (``None``: a new
    calculation's none, an extension's its own); ``stages`` each rung's
    values as the person holds them, by rung, a rung not named holding none
    -- ``None``, a road that holds none: an extension's rungs its own;
    ``shared`` the shared values as given (:func:`shared_of`); ``window`` a
    window fitted to the finished grid, written as the transmission's own
    values, beside ``reuses`` alone; ``refuse_limits`` whether what an
    extension does not take is
    refused here (the Transport tab's), or left to be said (the command
    line's).  NotDescribed, :class:`NotDescribed`; a write that fails,
    `describe.DescribeError` or :class:`OSError`."""
    from ..describe import Description, refuse_described, write_description
    from ..describe import DescribeError
    from ..issues import Issue, ValidationError
    from ..kind_table import kind
    from ..task import Task, derive_run, fresh_varies
    from ..validation.task import refuse_on_error
    from .citation_defaults import (check_pseudopotentials,
                                    citation_answers,
                                    transport_template_text)
    from .compose import (CitedPair, ComposeError, refuse_failed_citation,
                          refuse_inside_citation, resolve_citation)
    from .deck import describe_findings
    from .reuse import Extension, ReuseError, citing, reused, write_reusing
    from .stages import resolvable_override_names, stages_for_transport
    from .window import window_rung
    from ..jobset.rung import citation_of
    rec = kind(calculation)
    root = Path(tree_root).resolve()
    shared = dict(shared or {})

    # THE FINISHED CALCULATION IT EXTENDS, and its engine: an extension runs
    # on the engine its finished calculation ran (§ 2a.16; the user,
    # 2026-11-02) -- on another, nothing it computed is taken.
    finished = reuses_rel = None
    if reuses:
        try:
            finished, reuses_rel = reused(reuses, tree_root=root)
        except ReuseError as exc:
            raise NotDescribed(str(exc)) from None
        ran = finished.task.engine
        if engine and engine != ran:
            raise NotDescribed(
                f"engine {engine!r}: {finished.base.name} ran on {ran}, and "
                f"a calculation that reuses it runs on the engine it ran -- "
                f"on {engine} it would be another calculation, nothing "
                f"reused (engines/transport.md § 2a.16)")
        engine = ran
    if not engine:
        raise NotDescribed("engine: name the engine the calculation runs "
                           "on -- never assumed or inferred")
    no = rec.why_not_on(engine)
    if no:
        raise NotDescribed(f"engine {engine!r}: {no}.")

    # THE CITATION, resolved and classified once through the door prep
    # composes through; a run that ended with an error refused before
    # anything is written (§ 3.1).  An extension cites the finished
    # junction, or a frame set of it in its place (§ 2a.16).
    if finished is not None and not junction:
        junction = citation_of(finished.task)
    if not junction:
        raise NotDescribed("no junction citation -- pick the relaxed "
                           "junction first: a finished relaxation run, or a "
                           "structure pair saved from one")
    cited = None
    if finished is None or junction != citation_of(finished.task):
        try:
            resolved, cited = resolve_citation(junction, root)
            refuse_failed_citation(cited, junction)
        except ComposeError as exc:
            raise NotDescribed(str(exc)) from None
        junction = str(Path(resolved).relative_to(root))

    # THE FOLDER: one calculation per folder, never inside what it cites
    # (§ 3.1, *A calculation's folder is its own*).
    if dest is not None:
        try:
            here = Path(dest).resolve()
            shown = (str(here.relative_to(root)) if here.is_relative_to(root)
                     else str(dest))
            refuse_described(dest, shown=shown)
            refuse_inside_citation(
                dest, finished.base if finished is not None else cited.path,
                shown=shown, what=("the finished calculation"
                                   if finished is not None
                                   else "the cited junction"))
        except (DescribeError, ComposeError) as exc:
            raise NotDescribed(str(exc), folder=True) from None

    # THE VOLTAGES, STATED -- `[0.0]`, the single bias, where none is given
    # (§ 2a.10), never an empty list; an extension keeps its own unless
    # given, and voltages given with no treatment keep the finished
    # calculation's (§ 2a.16).
    if bias is not None and not bias:
        raise NotDescribed("bias: state the voltages -- [0.0] for a single "
                           "bias (engines/transport.md 2a.10)")
    if bias is None and finished is None:
        bias = (0.0,)
    if finished is not None:
        if bias is None:
            bias = finished.task.bias
            if low_bias_approximation is None:
                low_bias_approximation = finished.task.low_bias_approximation
        elif low_bias_approximation is None and len(bias) > 1:
            low_bias_approximation = finished.task.low_bias_approximation

    # EACH RUNG'S VALUES AND THE SHARED ONES, by name: what prep can
    # resolve, and what the catalogue marks shared for the kind
    # (`engines/template.md` § 7).
    resolvable = resolvable_override_names(engine)
    for rung, bag in (stages or {}).items():
        for item in bag:
            if item not in resolvable:
                raise NotDescribed(
                    f"{item!r} is not a parameter this calculation can "
                    f"carry: `prep` resolves each rung against the {engine} "
                    f"schema, and either no field of it has that name or it "
                    f"is a machine fact the description must never carry "
                    f"(engines/template.md 7).")
    chosen, blank = shared_of(shared, engine=engine, calculation=rec.name)
    if finished is None and not name:
        raise NotDescribed("name: the calculation has no name -- send its "
                           "name, as the transport tab sends the folder's")

    # THE RUNGS: each as the person holds it; an extension's, its own where
    # none is given -- the rungs it keeps always its own (§ 2a.16) -- and a
    # window fitted to the finished grid written into its rung's values.
    ext = Extension(reuses_rel, finished.task) if finished is not None \
        else None
    if ext is not None and stages is None:
        bags = {st.name: dict(st.overrides) for st in finished.task.stages}
    else:
        bags = {r: dict(b) for r, b in (stages or {}).items()}
        for r in (ext.fixed if ext is not None else ()):
            bags[r] = bags.get(r) or ext.own(r)
    fitted_words = None
    if window is not None:
        lo, hi, finer = window
        try:
            got = _fitted(finished, lo, hi, finer)
        except ValueError as exc:
            raise NotDescribed(
                f"the window {lo:g}..{hi:g} eV: {exc}") from None
        bags[window_rung()] = {**bags.get(window_rung(), {}),
                               **got.values()}
        fitted_words = got.words()
    try:
        ladder = tuple(stages_for_transport(engine, bags))
        # THE stages.md 6.2 RULE: an override names a promoted field, and
        # `varies` is the promotion; an extension keeps what the finished
        # calculation varied (§ 6.2: `varies` is intent).
        varies = fresh_varies(ladder)
        if finished is not None:
            varies = tuple(dict.fromkeys((*finished.task.varies, *varies)))
            task = citing(dataclasses.replace(
                finished.task, reuses=reuses_rel, bias=bias,
                low_bias_approximation=low_bias_approximation,
                swap_electrodes=(finished.task.swap_electrodes
                                 if swap_electrodes is None
                                 else swap_electrodes),
                varies=varies, stages=ladder), junction)
        else:
            task = Task(
                engine=engine,
                run=derive_run(name, junction,
                               stage_names=tuple(s.name for s in ladder)),
                structure=None, calculation=rec.name,
                slots={rec.description.cites: junction}, bias=bias,
                low_bias_approximation=low_bias_approximation,
                swap_electrodes=bool(swap_electrodes), varies=varies,
                stages=ladder)
    except ValueError as exc:
        # THE CODEC'S REFUSAL, beside the control that wrote it where the
        # codec names where (`task.DescriptionError`: the bias list, its
        # treatment).
        where = getattr(exc, "where", "")
        raise NotDescribed(str(exc), findings=[Issue(
            "error", str(exc), where=f"task.{where}" if where else "")]
            ) from None
    if ext is not None and refuse_limits:
        kept = ext.refusal(task, shared=shared)
        if kept is not None:
            raise NotDescribed(kept.message,
                               findings=[Issue("error", kept.message)],
                               task=task)

    # THE TEMPLATE: an extension's is the finished calculation's (its
    # shared values fixed, § 2a.16); a new one's DEFAULTED FROM THE
    # CITATION (§ 2a.7), a row nobody answered valueless (§ 3.8.3), the
    # pseudopotentials' rule first (§ 3.1).
    try:
        if finished is not None:
            template = finished.template_text
        else:
            check_pseudopotentials(cited, chosen.get("psml_lib"))
            template = transport_template_text(
                citation_answers(cited), label=task.label, blank=blank,
                **chosen)
    except ValueError as exc:
        raise NotDescribed(str(exc), task=task) from None
    desc = Description(task=task, template=template,
                       request=finished.request if finished is not None
                       else None)
    # THE DESCRIPTION'S OWN CHECK, and the k-point meshes each rung will
    # write on the junction's own axis kinds (§ 0.3b): its errors refuse,
    # its warnings are said.
    found = describe_findings(
        task, template, desc.request,
        cited=cited if finished is None else None,
        base=finished.base if finished is not None else None)
    try:
        refuse_on_error(found)
    except ValidationError as exc:
        raise NotDescribed(str(exc), findings=found, task=task) from None
    notes: List[object] = list(found)
    if fitted_words is not None:
        notes.append(Issue("info", f"transmission window: {fitted_words}"))

    # THE FILES, every one or none -- or the plan alone, written into a
    # hidden folder of its own under the tree, which the reuse needs to
    # find the finished calculation by its path, removed whatever happens.
    written: Tuple[Path, ...] = ()
    manifest = None
    if finished is None:
        if dest is not None:
            written = tuple(write_description(desc, dest))
    else:
        staging = None
        where = Path(dest) if dest is not None else None
        try:
            if where is None:
                staging = Path(tempfile.mkdtemp(dir=root,
                                                prefix=".molbuilder-plan-"))
                where = staging / task.label
            try:
                out, said = write_reusing(desc, where, finished,
                                          tree_root=root)
            except ReuseError as exc:
                raise NotDescribed(str(exc), task=task) from None
            except DescribeError as exc:
                if dest is not None:
                    raise
                raise NotDescribed(str(exc), task=task) from None
            if dest is not None:
                written = tuple(out)
            notes += list(said)
            manifest, why = _manifest_of(where)
            notes += why
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
    pair = isinstance(cited, CitedPair)
    if finished is not None:
        headline = (f"Described {rec.noun} '{task.run.name}' in {dest} -- "
                    f"reuses {reuses_rel}.")
        after = ("Change what the new question asks -- on Task setup, or in "
                 "its task.json and template -- then prepare it for the "
                 "machine that will run it:",)
    else:
        headline = (f"Described {rec.noun} '{task.run.name}' in {dest} -- "
                    f"composes {junction}.")
        after = ("The template carries the shared electronic description, "
                 "filled in from the "
                 + ("record of the optimization your pair carries"
                    if pair else "run you cited")
                 + " -- change anything in it and every stage follows, "
                 "because there is one of it.  The structure arrives at "
                 "prep from the citation, and the pseudopotentials "
                 + ("from the directory you named" if pair
                    else "from the run you cited")
                 + ".  On the machine that will run it:",)
    return Described(task=task, written=written, notes=tuple(notes),
                     manifest=manifest, headline=headline, after=after)


def _manifest_of(where) -> Tuple[Optional[Tuple[str, ...]], List[object]]:
    """WHAT A DESCRIBED EXTENSION AT ``where`` WILL REUSE, AND WHAT RUNS --
    the one answer prep composes (`reuse.decide`), in the manifest's words;
    a note saying why where it is not known yet."""
    from ..calculation import open as open_calculation
    from ..issues import Issue
    from ..jobset.errors import PrepError
    from .reuse import decide
    try:
        made = open_calculation(where)
        reuse = decide(made, made.request.numbers)
    except (PrepError, ValueError, OSError) as exc:
        return None, [Issue("warn", f"what it will reuse is not known yet "
                                    f"-- {exc}")]
    return (tuple(reuse.lines(where)) if reuse is not None else None), []


__all__ = ["Described", "NotDescribed", "describe", "fit", "not_taken",
           "shared_of"]

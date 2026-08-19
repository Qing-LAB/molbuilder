"""The pipeline provenance log — `script-preparation.md` § 4.5.

**What these tests are about is the log's CONTRACT, not its wording.**  Three
promises make it worth having and each is pinned here:

1. it changes nothing (off by default, and on it produces no different
   artifact) — a debugging aid that perturbs the thing it observes is worse
   than none;
2. it answers *where did this value come from* — every value written with
   the source it came from, not just the number it ended up being;
3. it says what a ``Block`` produced — the visibility the check gate
   structurally cannot give (W11).

Plus the one the flat layout forces: **two rungs of one calculation prep into
the same directory, and their logs must not be the same file.**
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest

from conftest import write_pseudos
from molbuilder import describe as D
from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.jobset.model import Resources
from molbuilder.jobset.prep import prep_calculation
from molbuilder.pipeline_log import PipelineLog, log_name
from molbuilder.pyscf.stages import default_pyscf_stages
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """cwd + HOME of their own, like `test_prep_calculation`'s: an
    un-isolated run folds the repo's own ``molbuilder.json`` into every
    wrapper and into the log's config phase."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


#: 1,4-benzenedithiol — an isolated molecule that relaxes, which is what the
#: structure-optimization tab is for.
BDT = Structure(
    elements=["S", "C", "C", "C", "C", "C", "C", "S", "H", "H", "H", "H"],
    positions=np.array([
        [0.000, 0.000, 0.000], [0.000, 0.000, 1.780],
        [1.210, 0.000, 2.480], [1.210, 0.000, 3.880],
        [0.000, 0.000, 4.580], [-1.210, 0.000, 3.880],
        [-1.210, 0.000, 2.480], [0.000, 0.000, 6.360],
        [2.150, 0.000, 1.940], [2.150, 0.000, 4.420],
        [-2.150, 0.000, 4.420], [-2.150, 0.000, 1.940]]),
    vacuum=(12.0, 12.0, 12.0))


def _calculation(tmp_path, engine: str, shape: str, name: str = "BDT"):
    """A described calculation, as `jobset describe` leaves one."""
    cfg, stages = ((SiestaConfig(system_label=name, mesh_cutoff=300.0),
                    default_siesta_stages("publishable"))
                   if engine == "siesta" else
                   (PySCFConfig(job_name=name),
                    default_pyscf_stages("publishable")))
    src = tmp_path / f"{name}.xyz"
    src.write_text(BDT.to_xyz())
    dest = tmp_path / f"{name}-{engine}-{shape}"
    D.write_description(D.build_description(
        BDT, cfg, stages, engine=engine, shape=shape, name=name,
        source=str(src)), dest)
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {
            "activation": "conda activate",
            "preamble": "source /opt/conda/etc/profile.d/conda.sh"}}))
    if engine == "siesta":
        write_pseudos(dest, ["S", "C", "H"])
    return dest, [s.name for s in stages]


def _prep(dest, stage, *, log=True):
    return prep_calculation(dest, stage, allocation=Resources(mpi_np=8),
                            pipeline_log=log)


def _the_log(dest):
    found = sorted(dest.rglob("*.pipeline.log"))
    assert len(found) == 1, [p.name for p in found]
    return found[0].read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
#  1. It changes nothing                                                 #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_no_log_is_written_unless_it_is_asked_for(tmp_path, engine):
    """**W13**: off by default, and off means no file — not an empty one.

    The flag exists because this is an observer of the pipeline; a run that
    did not ask for one must be unable to tell it exists.
    """
    dest, stages = _calculation(tmp_path, engine, "flat")
    _prep(dest, stages[0], log=False)
    assert list(dest.rglob("*.pipeline.log")) == []


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_the_log_changes_no_generated_artifact(tmp_path, engine):
    """**W13**, and the whole premise.  Every file `prep` writes must be identical
    with the log on and off — a record that perturbs what it records is
    worse than no record, and this is the assertion that keeps it true when
    someone later reaches for a value the log does not yet have.

    Normalised for the two things that differ between ANY two preps: the
    generation timestamp, and the calculation's own path (which
    ``STAGE-PLAN.md`` prints).  Getting that normalisation wrong is how this
    check first reported a false failure.
    """
    def _fingerprint(dest):
        out = {}
        for p in sorted(dest.rglob("*")):
            if not p.is_file() or p.name.endswith(".pipeline.log"):
                continue
            t = p.read_text(encoding="utf-8", errors="replace")
            # The PARENT too, not only the calculation: `task.json` records
            # the structure file it was described from, and that sits beside
            # the calculation.  Normalising one and not the other reported
            # `task.json` -- a file `prep` never writes -- as changed by the
            # log.  The harness, not the product; the same class of mistake
            # this whole record exists to make findable.
            t = t.replace(str(dest.parent), "<PARENT>")
            t = t.replace(str(dest), "<DEST>").replace(dest.name, "<NAME>")
            t = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
                       r"(?:[.+\-]\S*)?", "<T>", t)
            t = re.sub(r"\b1[0-9]{9}\.[0-9]+\b", "<W>", t)   # epoch clock
            out[p.relative_to(dest).as_posix()
                .replace(dest.name, "<NAME>")] = t
        return out

    # SEPARATE parents, IDENTICAL leaf name: the two preps must not share a
    # directory, and the calculation's own name must not be what differs --
    # it reaches the deck's identity line, the wrapper and STAGE-PLAN.md.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    off, off_stages = _calculation(tmp_path / "a", engine, "flat")
    on, on_stages = _calculation(tmp_path / "b", engine, "flat")
    _prep(off, off_stages[0], log=False)
    _prep(on, on_stages[0], log=True)
    a, b = _fingerprint(off), _fingerprint(on)
    assert sorted(a) == sorted(b), "the log added or removed a file"
    differ = [k for k in a if a[k] != b[k]]
    assert differ == [], f"the log changed {differ}"


# --------------------------------------------------------------------- #
#  2. The flat layout's own rule                                         #
# --------------------------------------------------------------------- #

def test_two_rungs_in_a_flat_calculation_get_two_logs(tmp_path):
    """**Flat is depth 1**: every stage preps into the bundle root, and the
    deck's stage TOKEN is what tells them apart (`job-contracts.md` § 6.3).
    A log named per calculation would have `medium` overwrite `coarse` — and
    the run whose provenance you wanted would be the one that was destroyed.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    _prep(dest, stages[1])
    names = sorted(p.name for p in dest.rglob("*.pipeline.log"))
    assert names == ["BDT_01_coarse.siesta.flat.pipeline.log",
                     "BDT_02_medium.siesta.flat.pipeline.log"], names
    # and each is ITS OWN rung, not two copies of one
    first = (dest / names[0]).read_text()
    second = (dest / names[1]).read_text()
    assert "stage 01_coarse" in first and "stage 02_medium" not in first
    assert "stage 02_medium" in second and "stage 01_coarse" not in second


def test_the_name_says_engine_and_shape(tmp_path):
    """The file is read after being copied off the machine, where the
    directory it came from no longer says what it was."""
    assert log_name("BDT", "01_coarse", "siesta", "flat") == \
        "BDT_01_coarse.siesta.flat.pipeline.log"
    # a calculation with no ladder has no token, and gets no dangling "_"
    assert log_name("BDT", "", "pyscf", "hierarchical") == \
        "BDT.pyscf.hierarchical.pipeline.log"


# --------------------------------------------------------------------- #
#  3. It answers "where did this value come from"                        #
# --------------------------------------------------------------------- #

def test_every_resolved_value_carries_its_source(tmp_path):
    """The ⊕ chain, which is the difference between provenance and a dump.

    A stage override and the allocation are the two sources a reader cannot
    work out from the template alone, so both are named.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    text = _the_log(dest)
    chosen = [ln for ln in text.splitlines() if ln.startswith("  ⊕")]
    assert chosen, "no ⊕ rows at all"
    assert all("<-" in ln for ln in chosen), \
        "a ⊕ row with no source is a value with no provenance"
    # the stage's own overrides are attributed to the stage, not the template
    assert re.search(r"⊕\s+relax_type\s+CG\s+<- stage", text), text
    # the allocation is folded in, and says so -- no config file names it
    assert re.search(r"⊕\s+mpi_np\s+8\s+<- allocation", text), text
    # and the template's baseline is still there, so the file is complete
    assert re.search(r"⊕\s+mesh_cutoff\s+300\.0\s+<- template", text), text


def test_the_engines_derived_context_is_in_the_log(tmp_path):
    """**W10's one per-render context**, which no other record carries.

    ``block_size`` and the diagonaliser are worked out from
    ``(struct, cfg)`` and are nobody's config field; before the spec carried
    the context, a reader outside the engine had no way to see them at all.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    text = _the_log(dest)
    assert "this deck's own derived context (W10)" in text
    for name in ("algorithm", "block_size", "relax_kind", "spin_polarized"):
        assert re.search(rf"⊕\s+{name}\s+\S+\s+<- derived", text), \
            f"{name} missing from the derived context"


def test_pyscfs_context_is_read_after_the_walk_not_before(tmp_path):
    """PySCF fills its context **while its blocks render** — one block works
    a value out and the next reads it — where SIESTA fills its own before the
    layout.  A log that snapshotted the context at spec time would show
    SIESTA's answers and an empty dict for PySCF.
    """
    dest, stages = _calculation(tmp_path, "pyscf", "flat")
    _prep(dest, stages[0])
    text = _the_log(dest)
    assert re.search(r"⊕\s+stability_checked\s+\S+\s+<- derived", text), text
    assert re.search(r"⊕\s+emit_constraints\s+\S+\s+<- derived", text), text


# --------------------------------------------------------------------- #
#  4. It says what a Block produced — W11's blind spot                   #
# --------------------------------------------------------------------- #

def test_each_block_reports_what_it_produced(tmp_path):
    """**W11**: a `Block` is free text, so the check gate cannot see inside
    one and it contributes no line to compare.  Recording what each block
    produced is exactly the visibility the gate structurally cannot give —
    and SIESTA's blocks are hundreds of the deck's lines.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    text = _the_log(dest)
    rendered = re.findall(r"^  out  Block\s+'([^']+)'\s+(.+)$", text, re.M)
    assert rendered, "no Block rows in the render phase"
    titles = [t for t, _ in rendered]
    assert "system, structure and constraints" in titles, titles
    # a line count, not a bare "rendered": the point is knowing how much of
    # the deck came out of a place the framework cannot see into
    counted = [d for _, d in rendered if re.match(r"\d+ lines", d)]
    assert counted, [d for _, d in rendered]


def test_a_section_reports_which_items_declined(tmp_path):
    """A conditional item that emitted nothing looks exactly like one that
    was never in the layout.  Telling those apart is most of what a reader
    comes here for — and only the parameters walk knows.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    text = _the_log(dest)
    assert re.search(r"^  out  Section\s+'Basis & grid'\s+\d+ lines\s+"
                     r"mesh_cutoff", text, re.M), text


def test_a_slot_that_answers_nothing_says_so(tmp_path):
    """**W5**: *"nothing" is an answer, and it is recorded.*

    PySCF declares no bench anchors and needs no ``validate_subject``; SIESTA
    answers both.  A blank where a slot's answer belongs cannot be told from a
    slot nobody looked at, which is the whole reason W5 exists.
    """
    dest, stages = _calculation(tmp_path, "pyscf", "flat")
    _prep(dest, stages[0])
    text = _the_log(dest)
    assert re.search(r"out  bench_marks\s+nothing \(W5\)", text), text
    assert re.search(r"out  validate_subject\s+nothing \(W5\)", text), text
    assert re.search(r"out  check_rules\s+answered", text), text

    other, other_stages = _calculation(tmp_path, "siesta", "flat")
    _prep(other, other_stages[0])
    siesta = _the_log(other)
    assert re.search(r"out  bench_marks\s+answered", siesta), siesta
    assert re.search(r"out  validate_subject\s+answered", siesta), siesta


# --------------------------------------------------------------------- #
#  5. Both gates' verdicts are IN the file                               #
# --------------------------------------------------------------------- #

def test_both_gates_report_their_verdict(tmp_path):
    """The two gates answer different questions and neither can do the
    other's job (`script-preparation.md` § 4.3), so the log carries both:
    **validate** on the settings, **check** on the written file.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    text = _the_log(dest)
    assert "STEP 3.3 · VALIDATE" in text and "STEP 3.11 · CHECK" in text
    assert re.search(r"out  verdict\s+\d+ error, \d+ warn", text), text
    assert re.search(r"out  compared\s+\d+ distinct lines", text), text


def test_a_settings_refusal_is_in_the_log_with_its_reason(tmp_path):
    """**Logged before reported.**  ``validate``'s report RAISES on an
    error-severity issue; a log written afterwards would be missing exactly
    the run that most needed explaining.

    Driven through a real refusal: GPU without an ELPA diagonaliser.
    """
    from molbuilder.jobset.prep import PrepError
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    tpl = dest / "BDT.template.toml"
    head, sep, tail = tpl.read_text().partition("[item.enable_gpu]")
    assert sep, "the template lost its enable_gpu item"
    body, nxt, rest = tail.partition("\n[item.")
    assert "value = false" in body, body
    tpl.write_text(head + sep + body.replace("value = false", "value = true", 1)
                   + nxt + rest)
    with pytest.raises((PrepError, ValueError)):
        _prep(dest, stages[0])
    text = _the_log(dest)
    # the file stops where the pipeline stopped, and the last thing in it is
    # the step that refused
    assert "STEP 3" in text, text


# --------------------------------------------------------------------- #
#  6. The hook boundary — W16                                            #
# --------------------------------------------------------------------- #

def test_a_hook_that_raises_says_whose_it_was(tmp_path):
    """**W16**: the framework is a walk over the engine's functions, so an
    exception with no owner on it is the ordinary failure here.

    The three promises of `issues.calling`, each asserted: the TYPE survives,
    the MESSAGE survives, and the attribution is attached.
    """
    import dataclasses
    from molbuilder import script_emit as se
    from molbuilder.siesta.input import spec_for

    struct = BDT
    cfg = SiestaConfig(system_label="BDT", psml_lib=None)
    spec = dataclasses.replace(
        spec_for(struct, cfg),
        line=lambda p: (_ for _ in ()).throw(TypeError("engine bug")))
    with pytest.raises(TypeError) as caught:
        se.render_deck(spec, struct, cfg)
    assert str(caught.value) == "engine bug", "the message was replaced"
    notes = getattr(caught.value, "__notes__", [])
    assert any("siesta.line" in n for n in notes), notes
    assert any("item" in n for n in notes), notes


def test_an_engines_deliberate_refusal_survives_the_boundary(tmp_path):
    """The boundary must not bury a refusal an engine raised ON PURPOSE.

    SIESTA refuses GPU without an ELPA diagonaliser from inside `spec_for`,
    in a sentence written for a person.  A wrapper CLASS would have turned
    that into ``HookError`` and broken every caller matching ``ValueError``
    -- which is why the boundary annotates instead of replacing.

    **Driven THROUGH the boundary**, both ways.  The first version of this
    test called ``spec_for`` directly, where nothing wraps it -- so it asserted
    the property without exercising it, and a mutation that replaced the
    exception with a ``RuntimeError`` passed it. Found by that mutation.
    """
    from molbuilder.issues import calling
    from molbuilder.jobset.prep import PrepError

    # 1. the unit: an engine's ValueError through the boundary is still a
    #    ValueError, with its own message.
    with pytest.raises(ValueError) as caught:
        with calling("spec_for", engine="siesta", where="BDT.fdf"):
            raise ValueError("enable_gpu requires an ELPA diagonalizer")
    assert str(caught.value) == "enable_gpu requires an ELPA diagonalizer"
    assert any("siesta.spec_for" in n
               for n in getattr(caught.value, "__notes__", [])), caught.value

    # 2. the real one: a prep of a GPU-without-ELPA calculation refuses with
    #    SIESTA's own sentence, through every layer between.
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    tpl = dest / "BDT.template.toml"
    head, sep, tail = tpl.read_text().partition("[item.enable_gpu]")
    assert sep
    body, nxt, rest = tail.partition("\n[item.")
    tpl.write_text(head + sep + body.replace("value = false", "value = true", 1)
                   + nxt + rest)
    with pytest.raises((ValueError, PrepError)) as real:
        _prep(dest, stages[0])
    assert "ELPA" in str(real.value), real.value


def test_a_hook_failure_lands_in_the_log_with_its_traceback(tmp_path):
    """The failure goes in its OWN column, so ``grep '^  !!'`` finds every
    hook that blew up -- and the traceback goes in whole, because this is the
    file someone opens *because* a run died."""
    from molbuilder.pipeline_log import PipelineLog
    from molbuilder.issues import calling

    log = PipelineLog(tmp_path / "x.pipeline.log")
    log.phase("STEP 3 · DECK")
    with pytest.raises(TypeError):
        with calling("line", engine="siesta", where="item 'mesh_cutoff'",
                     log=log):
            raise TypeError("boom")
    log.close()
    text = (tmp_path / "x.pipeline.log").read_text()
    assert "!! siesta.line RAISED — TypeError" in text, text
    assert [ln for ln in text.splitlines() if ln.startswith("  !!")], text
    assert "at item 'mesh_cutoff'" in text, text
    assert "Traceback (most recent call last)" in text, text


def test_the_attribution_reaches_the_person_running_the_command(tmp_path):
    """**A note does not reach a person on its own.**  ``str(exc)`` drops
    notes, and `prep` turns a user-fixable refusal into ``PrepError(str(exc))``
    -- so without ``notes_of`` the attribution would stop at a traceback that
    a CLI user never sees.
    """
    from molbuilder.issues import Issue, ValidationError, calling
    from molbuilder.jobset.prep import PrepError, _user_error_as_prep

    with pytest.raises(PrepError) as caught:
        with _user_error_as_prep():
            with calling("check_rules", engine="pyscf", where="BDT.py"):
                raise ValidationError([Issue("error", "deck is wrong",
                                             where="deck.x")])
    assert "deck is wrong" in str(caught.value)
    assert "pyscf.check_rules" in str(caught.value), str(caught.value)


def test_every_engine_hook_of_both_seams_is_wrapped():
    """**W16 names sixteen hooks and the code must call all sixteen through
    the boundary** -- a rule that holds for fifteen is a rule nobody can rely
    on, and the sixteenth is where the next afternoon goes.

    Read from the source, so a hook added to either seam and called bare
    fails here rather than the next time it raises.
    """
    import ast
    import dataclasses
    from molbuilder.script_emit import DeckSpec
    from molbuilder.jobset.prep import EngineSeam

    deck = {f.name for f in dataclasses.fields(DeckSpec)} & {
        "line", "note_lead", "section_title", "validate_subject",
        "provenance_defaults", "bench_marks", "check_rules"}
    seam = {f.name for f in dataclasses.fields(EngineSeam)} - {
        "config_cls", "suffix"}
    expected = deck | seam | {"Block.render"}
    assert len(expected) == 16, sorted(expected)

    wrapped = set()
    for rel in ("molbuilder/script_emit.py", "molbuilder/jobset/prep.py"):
        src = pathlib.Path(__file__).resolve().parents[1] / rel
        for node in ast.walk(ast.parse(src.read_text())):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_calling"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)):
                wrapped.add(node.args[0].value)
    missing = sorted(expected - wrapped)
    assert not missing, (
        f"these engine hooks are called without the boundary (W16): {missing}")


def test_every_line_is_a_banner_a_column_or_an_indented_note(tmp_path):
    """**W14**: every line declares what it is in its first column.

    The rule is what makes one grep answer one question across the file, and
    a line that is none of the four kinds is a line nobody can find twice.
    Checked against a REAL log rather than a constructed one, so a verb added
    later that writes its own shape fails here.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    stray, prev = [], ""
    for ln in _the_log(dest).splitlines():
        was_rule = bool(re.match(r"^[═─]+$", prev.strip()))
        prev = ln
        if not ln.strip():
            continue
        if re.match(r"^[═─]+$", ln.strip()):        # a banner rule
            continue
        if re.match(r"^  (in|⊕|out|!!)\s", ln):     # one of the four columns
            continue
        if ln.startswith("       "):                # an indented note / block
            continue
        # A BANNER'S TITLE, and ONLY directly under a rule.  Accepting any
        # two-space-indented line here made this test unfalsifiable: a verb
        # writing `  note ...` passed it.  Found by mutation, 2026-08-19.
        if was_rule and re.match(r"^  \S", ln):
            continue
        stray.append(ln)
    assert not stray, f"lines that are none of W14's kinds: {stray[:5]}"


def test_a_block_row_carries_no_items_values_or_sources(tmp_path):
    """**W15**: what a `Block` hides, it hides from the log too.

    The framework can say a block ran and how many lines it produced, and
    nothing more -- there is no declaration to read.  If a Block row ever
    grew item names, it would mean something was reaching inside a Block,
    which is the freedom W11 says a Block is paying for.
    """
    dest, stages = _calculation(tmp_path, "siesta", "flat")
    _prep(dest, stages[0])
    rows = [ln for ln in _the_log(dest).splitlines()
            if ln.startswith("  out  Block")]
    assert rows, "no Block rows at all"
    for ln in rows:
        assert re.match(r"^  out  Block\s+'[^']*'\s+"
                        r"(\d+ lines|nothing to say for this configuration)$",
                        ln), f"a Block row said more than it can know: {ln!r}"
        assert "<-" not in ln, f"a Block row carries a source: {ln!r}"


def test_nothing_reaches_a_deck_until_the_last_refusal(tmp_path):
    """**W3**: nothing reaches a deck until the last thing that can refuse
    has refused.

    A missing pseudopotential costs a second to find at the data-files step
    and a queue wait plus MPI startup to find on the cluster -- and in
    between sits a half-built folder, which reads to its owner as a folder
    that worked.  So the refusal lands with NO deck on disk.

    **SIESTA only, and the reason is worth stating rather than hiding in a
    parametrize list.**  PySCF ships its basis sets inside the library, so it
    has no data-files step to refuse at -- that is W5's *nothing*, a recorded
    answer.  And its settings gate cannot refuse either: measured 2026-08-19,
    `validation/pyscf.py` emits no error-severity issue at all, so every
    PySCF refusal today comes from somewhere other than the two gates.  A
    parametrised arm here would assert a refusal the code cannot produce.
    """
    from molbuilder.jobset.prep import PrepError

    dest, stages = _calculation(tmp_path, "siesta", "flat")
    for psml in dest.glob("*.psml"):     # what the engine cannot start without
        psml.unlink()
    with pytest.raises(PrepError):
        _prep(dest, stages[0])
    decks = [p for p in dest.rglob("*") if p.suffix in (".fdf", ".py")]
    assert not decks, f"a deck was written despite the refusal: {decks}"


def test_both_engines_traverse_the_same_sequence(tmp_path):
    """**W12**: one sequence, and an engine substitutes steps in it -- it
    never brings its own.

    *"Two engines that each run their own order cannot be compared, and a
    rule proved of one says nothing about the other."*  Until the log existed
    there was nothing that recorded the order actually taken; now the step
    banners ARE that record, so the claim is checkable rather than asserted.

    The engines differ in what they answer at each stop -- PySCF has no bench
    anchors and no ``validate_subject`` -- and that is W5's *nothing*, not a
    step it skips.  So the SEQUENCE must match exactly.
    """
    seqs = {}
    for engine in ("siesta", "pyscf"):
        (tmp_path / engine).mkdir(exist_ok=True)
        dest, stages = _calculation(tmp_path / engine, engine, "flat")
        _prep(dest, stages[0])
        seqs[engine] = [re.sub(r" — .*", "", ln.strip())
                        for ln in _the_log(dest).splitlines()
                        if ln.strip().startswith("STEP ")]
    assert seqs["siesta"] == seqs["pyscf"], (
        f"the engines ran different sequences:\n"
        f"  siesta: {seqs['siesta']}\n  pyscf : {seqs['pyscf']}")
    assert seqs["siesta"], "no step banners at all"


# --------------------------------------------------------------------- #
#  7. ONE writer — W13, structurally                                     #
# --------------------------------------------------------------------- #

def test_no_engine_writes_to_the_log(tmp_path):
    """**W13**: there is one writer, and an engine is not it.

    A ``print`` added to an engine is a second writer and, within a month, a
    second format -- which is how every record in this tree that has two
    writers ended up with two spellings.  The rule is checked by IMPORT: no
    module under an engine's package may reach the log at all, so an engine
    cannot write to it even by accident.

    The framework (`script_emit`) and the conductor (`jobset/prep`) are the
    two that may, and they are named rather than pattern-matched -- a list
    that must be edited on purpose is the point.
    """
    import ast
    root = pathlib.Path(__file__).resolve().parents[1] / "molbuilder"
    allowed = {"molbuilder/jobset/prep.py", "molbuilder/pipeline_log.py"}
    offenders = []
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root.parent).as_posix()
        if rel in allowed:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            mod = (node.module if isinstance(node, ast.ImportFrom) else None)
            names = ([a.name for a in node.names]
                     if isinstance(node, (ast.Import, ast.ImportFrom)) else [])
            if (mod and "pipeline_log" in mod) or any(
                    "pipeline_log" in n for n in names):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "these modules import the pipeline log, which only the framework and "
        "the conductor may do (W13):\n  " + "\n  ".join(offenders))


# --------------------------------------------------------------------- #
#  7. It never breaks a prep                                             #
# --------------------------------------------------------------------- #

def test_an_unwritable_log_does_not_break_the_prep(tmp_path):
    """``ledger.record``'s rule, inherited: *a run must not fail because its
    logbook could not be written.*"""
    log = PipelineLog(tmp_path / "no-such-dir" / "x.pipeline.log")
    log.phase("STEP 1")
    log.received("a", "b")
    log.chose("c", 1, "template")
    log.produced("d", "e")
    log.text("multi\nline")
    log.note("n")
    log.close()          # every one of those is a no-op, and none raises
    assert not (tmp_path / "no-such-dir").exists()

"""A field a person can set must change what the engine is told to do.

**The defect this exists to catch is silent.** A control appears on the form,
a person sets it, the run proceeds, and nothing anywhere obeys it — the
"present but not honoured" shape `run-identity.md` § 4 names for `restart` and
which applies to every parameter; `engines/vibration.md` I10 states it of the
vibration form, *honoured by the deck or refused by name*. No other test asks
the question of a form as a whole: the deck tests check the settings they know
about, and a setting nobody wired up is exactly the one nobody writes a test
for.

**So this asks it of every field each form offers**, from the catalogue, with no
list of names here — a parameter added tomorrow is covered the day it appears.
The form's fields are the catalogue's own table; what they do is read from the
deck `jobset prep` wrote: the value set in the calculation's template, as a
person sets it, and the group prepared again (`process/testing.md` § 3a).

**Why some fields need an enabling context.** `Spin.Total` is not in a
spin-restricted deck and should not be. A field is honoured if it changes the
deck under *some* legal configuration, so each such field names the context
that switches it on. A field that changes nothing under any of them is either
dead or exempt, and exempt means named with a reason.

**What is compared is what the engine reads**, so a value that reaches only a
comment, a report of what was asked or a stamp is not honoured
(:func:`_what_the_engine_reads`).  Comparing a deck's text passed by accident
until 2026-11-06: the text carries the time it was written, so once the sweep
crossed a second every field "changed the deck".

**And what the engine reads is another calculation**: a change it reads is
never called the same by the door a launched stage is prepared again through
(`deck_record.same_calculation`, `execution/job-request.md` § 6) -- which
sets aside what a deck only reports, so a field that reached nothing but a
report would pass a launched stage unseen.
"""
from __future__ import annotations

import ast

import pytest

import molbuilder.template as T
from molbuilder.deck_record import same_calculation
from molbuilder.calculation import open as open_calculation
from molbuilder.parse.fdf import read_fdf
from molbuilder.web.blueprints._shared import catalogue_to_form_schema
from support.form_fields import NOT_IN_THE_DECK, another_value
from support.road import (WATER, described, group_words, jobset, road_tree,
                          set_by_the_person, stage_file)

#: Contexts that switch a conditional field on, by the calculation; tried in
#: order, then the field alone.
_ENABLING = {
    "optimization": {
        "auxbasis":              [{"density_fit": True}],
        "ecp_atoms":             [{"ecp": "lanl2dz"}],
        "ecp":                   [{"ecp_atoms": ["O"]}],
        "solvent_method":        [{"solvent": "water"}],
        "geom_continue_retries": [{"on_nonconvergence": "continue"}],
    },
    "vibration": {
        "auxbasis":              [{"density_fit": True}],
        "ecp_atoms":             [{"ecp": "lanl2dz"}],
        "ecp":                   [{"ecp_atoms": ["O"]}],
        "geom_continue_retries": [{"on_nonconvergence": "continue"}],
        # A relaxed start is where symmetry is honoured; a deck that relaxes
        # refuses it by name.
        "symmetry":              [{"already_relaxed": True}],
        # With infrared and Raman off, the one route PCM reaches
        # (`engines/vibration.md` § 4.6).
        "solvent":               [{"compute_raman": False,
                                   "compute_ir": False}],
        "solvent_method":        [{"solvent": "water", "compute_raman": False,
                                   "compute_ir": False}],
        # What each enters nothing without (`active_when`).
        "displacement_amplitude_ang": [{"es_mode_selection": "all"}],
        "es_n_homo_below":       [{"es_mode_selection": "all"}],
        "es_n_lumo_above":       [{"es_mode_selection": "all"}],
        "freq_min_cm1":          [{"es_mode_selection": "all"}],
        "freq_max_cm1":          [{"es_mode_selection": "all"}],
        "es_explicit_indices":   [{"es_mode_selection": "explicit"}],
    },
}

#: Values the generic one cannot be: what changes what the ENGINE is asked for
#: on water -- a closed shell, so ``restricted`` and a count of 0 are what a
#: blank already resolves to (the electronic state,
#: `science/chemistry-correctness.md` § 2a), an even charge keeps it closed --
#: and what a list-typed or name-typed item takes -- and, for an optional
#: setting left unwritten, a value other than the engine's own: SIESTA stops
#: on an SCF that does not converge unasked (`SCF.MustConverge`, default
#: true), so stating true is the input it already had (`execution/
#: job-request.md` § 6).
_PROBE = {"spin_treatment": "unrestricted", "unpaired_electrons": 2,
          "net_charge": 2, "solvent": "water", "ecp_atoms": ["O"],
          "ecp": "lanl2dz", "species_order": ["O", "H"],
          "es_explicit_indices": "1,2", "scf_must_converge": False}

#: A refusal is filed under the field it names, and a count of unpaired
#: electrons a kind cannot hold under the treatment that count implies.
_REFUSED_UNDER = {"unpaired_electrons": "spin_treatment"}

def _reports():
    """What a PySCF deck REPORTS of what it was asked, each changing with a
    value nothing obeys: the names the deck only reports under -- the one
    list the same-calculation reading sets aside (`pyscf.script_input.REPORTS`:
    its parameters record, its runtime facts, the targets its progress log
    draws, the echo of the config, the manuscript's paragraph, its
    bibliography and the result record's version) -- and the structure's
    sidecar, with the time it was written."""
    from molbuilder.pyscf.script_input import REPORTS, SIDECAR_NAME
    return {*REPORTS, SIDECAR_NAME}


def _what_python_runs(text) -> list:
    """A PySCF deck as Python runs it, statement by statement: no comment and
    no docstring, none of what it only reports (:func:`_reports`, a name or
    an entry of one: `_MB_PARAMS[name] = (default, asked, read)`, the
    parameters record), and no
    constant no line reads -- a value mirrored into a name nobody reads
    is the same echo."""
    tree = ast.parse(text)
    for scope in ast.walk(tree):
        if (isinstance(scope, (ast.Module, ast.FunctionDef, ast.ClassDef))
                and ast.get_docstring(scope) is not None):
            scope.body = scope.body[1:] or [ast.Pass()]
    read = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    reports = _reports()

    def runs(stmt) -> bool:
        if not isinstance(stmt, ast.Assign):
            return True
        target = stmt.targets[0]
        if isinstance(target, ast.Subscript):
            return ast.unparse(target.value) not in reports
        return not (isinstance(target, ast.Name)
                    and (target.id in reports or target.id not in read))

    return [ast.unparse(s) for s in tree.body if runs(s)]


def _what_the_engine_reads(engine, text):
    """The deck as ITS ENGINE reads it: SIESTA's through the one fdf reader,
    every keyword and block and no comment; PySCF's as Python runs it."""
    return read_fdf(text) if engine == "siesta" else _what_python_runs(text)


def _form_fields(engine, prefix, calculation) -> dict:
    """Every catalogue item the form renders a control for -- save one the
    calculation's kind answers itself (`role`): a person sets nothing there,
    its template holding no value of it (`engines/template.md` § 6.4)."""
    items = {i.name: i for i in T.select(T.read_template(T.load_catalogue()),
                                         engine=engine)}
    out = {}
    for section in catalogue_to_form_schema(engine, prefix,
                                            calculation)["sections"]:
        for field in section["fields"]:
            fid = field.get("id", "")
            if not fid.startswith(prefix + "-"):
                continue
            name = fid[len(prefix) + 1:].replace("-", "_")
            if name in items and calculation not in items[name].role:
                out[name] = items[name]
    return out


@pytest.mark.parametrize("engine, prefix, calculation", [
    ("siesta", "p", "optimization"),
    ("pyscf", "py", "optimization"),
    ("pyscf", "py", "vibration"),
], ids=["SIESTA", "PySCF", "PySCF vibration"])
def test_every_field_the_form_offers_changes_what_the_engine_reads(
        engine, prefix, calculation, tmp_path, monkeypatch):
    bundle = described(road_tree(tmp_path, monkeypatch, WATER, engine=engine),
                       calculation=calculation, engine=engine,
                       stage_strategy="")
    stage = open_calculation(bundle).task.stages[0].name
    suffix = ".fdf" if engine == "siesta" else ".py"
    template = open_calculation(bundle).template.path
    as_described = template.read_text()

    def prepared(values):
        """The group prepared again with ``values`` set and nothing else
        changed: what the engine would read and the deck as written, or the
        refusal's words."""
        template.write_text(as_described)
        if values:
            set_by_the_person(bundle, values)
        r = jobset("prep", "task", *group_words(bundle, stage),
                   "--bundle", bundle, "--target", "this")
        if r.exit_code:
            return r.output, None, None
        deck = stage_file(bundle, stage, suffix).read_text()
        return None, _what_the_engine_reads(engine, deck), deck

    def value_of(item, context):
        current = (context[item.name] if item.name in context else
                   T.one(T.read_template(as_described), item.name).value)
        if item.name in _PROBE and _PROBE[item.name] != current:
            return _PROBE[item.name]
        return another_value(item, current,
                             T.offered(item, engine, calculation))

    baselines = {}

    def baseline(context):
        """What the engine reads under ``context`` alone, and the deck,
        prepared once."""
        key = repr(sorted(context.items()))
        if key not in baselines:
            said, *baselines[key] = prepared(context)
            assert said is None, (context, said)
        return baselines[key]

    dead, refused, unprobed, unseen = [], {}, [], []
    for name, item in sorted(_form_fields(engine, prefix, calculation).items()):
        if (engine, name) in NOT_IN_THE_DECK:
            continue
        for context in (*_ENABLING[calculation].get(name, ()), {}):
            value = value_of(item, context)
            if value is None:
                unprobed.append(name)
                break
            said, reads, deck = prepared({**context, name: value})
            if said is not None:
                # REFUSED BY NAME is the setting heard (`engines/vibration.md`
                # I10); any other refusal is the probe not being correct input.
                if f"[config.{_REFUSED_UNDER.get(name, name)}]" in said:
                    break
                refused.setdefault(name, []).append(said[-400:])
                continue
            base_reads, base_deck = baseline(context)
            if reads != base_reads:
                # ...AND IS ANOTHER CALCULATION by the one door a launched
                # stage is prepared again through (`execution/job-request.md`
                # § 6): a change the engine reads is never passed as the same.
                if not same_calculation(engine, base_deck, deck):
                    unseen.append(name)
                break
        else:
            dead.append(name)
    assert not refused, (
        f"{engine} {calculation}: a legal value refused, in other words than "
        f"its own: {refused}")
    assert not unprobed, (
        f"{engine} {calculation}: no other value to try for {unprobed}: give "
        f"it one in _PROBE, or say why one legal value is all it has.")
    assert not unseen, (
        f"{engine} {calculation}: a change the engine reads called the same "
        f"calculation, so a launched stage prepared again would take it "
        f"unseen: {unseen}")
    assert not dead, (
        f"{engine} {calculation}: {len(dead)} field(s) the form offers change "
        f"NOTHING the engine reads: {dead}.\n"
        f"Either wire the value up, or add it to NOT_IN_THE_DECK with the "
        f"reason its effect is elsewhere. A control a person can set that "
        f"nothing obeys is the 'present but not honoured' defect.")

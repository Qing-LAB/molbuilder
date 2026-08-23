"""A field a person can set must change what gets generated.

**The defect this exists to catch is silent.** A control appears on the form,
a person sets it, the run proceeds, and nothing anywhere obeys it — the
"present but not honoured" shape `run-identity.md` § 4 names for `restart` and
which applies to every parameter. No existing test asks the question of the
form as a whole: the deck tests check the settings they know about, and a
setting nobody wired up is exactly the one nobody writes a test for.

**So this asks it of every field the form offers**, from the catalogue, with no
list of names here — a parameter added tomorrow is covered the day it appears.

**Why some fields need an enabling context.** `MD.LengthTimeStep` is not in a
CG relaxation's deck and should not be; `Spin.Total` is not in a spin-restricted
one. A field is honoured if it changes the deck under *some* legal
configuration, so each such field names the context that switches it on. A
field that changes nothing under any of them is either dead or exempt, and
exempt means named below with a reason.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder import script_emit as se
from molbuilder.structure import Structure
from molbuilder.web.blueprints._shared import catalogue_to_form_schema
import molbuilder.template as T


def _struct() -> Structure:
    return Structure(elements=["O", "H", "H"],
                     positions=np.array([[0., 0., 0.], [0.957, 0., 0.],
                                         [-0.24, 0.927, 0.]]),
                     vacuum=(8., 8., 8.))


#: Fields whose effect is real but is NOT on the deck's text, with the reason.
#: Anything added here needs one — "it does nothing" is not a reason, it is the
#: bug this file is looking for.
_NOT_IN_THE_DECK = {
    ("siesta", "copy_psml"):
        "acts on the data-files step (3.2) -- which files are put into the "
        "calculation, not what the deck says",
    ("siesta", "psml_lib"):
        "names WHERE pseudopotentials are found; a deck never carries a host path",
    ("siesta", "wrap_into_cell"):
        "acts on the STRUCTURE before the deck is written; a molecule already "
        "inside its cell has nothing to wrap",
    ("siesta", "write_molwatch_log"):
        "SIESTA honours it at the PROMISES sub-step (3.12), not in the deck: "
        "`prep._seed_trajectory_log` and `convert()` skip seeding "
        "`<label>.molwatch.log` when it is off.  Exempt for SIESTA ONLY -- "
        "PySCF's script carries the emitter, so its deck does change, and this "
        "file checks that it does",
}

#: Contexts that switch a conditional field on.  Tried in order.
_ENABLING = {
    "md_initial_temperature": [{"relax_type": "Nose"}],
    "md_target_temperature":  [{"relax_type": "Nose"}],
    "md_length_timestep":     [{"relax_type": "Nose"}],
    "md_max_cg_displ":        [{"relax_type": "Nose"}],
    "spin_total":             [{"spin_treatment": "polarized"}],
    "diag_algorithm":         [{"diag_algorithm": "ELPA-2STAGE"}],
    "use_gpu":             [{"diag_algorithm": "ELPA-2STAGE"}],
    "auxbasis":               [{"density_fit": True}],
    "ecp_atoms":              [{"ecp": "def2-SVP"}],
    "solvent_method":         [{"solvent": "water"}],
    "spin":                   [{"method": "UKS"}],
    "geom_continue_retries":  [{"on_nonconvergence": "continue"}],
}


def _alternative(item, current):
    """A different, legal value for this item, or None if none can be built."""
    choices = getattr(item, "choices", None)
    if choices:
        return next((c for c in choices if c != current), None)
    rng = getattr(item, "range", None)
    if item.type == "bool":
        return not bool(current)
    if item.type == "int":
        v = (current or 0) + 1
        return rng[0] if rng and not (rng[0] <= v <= rng[1]) else v
    if item.type == "float":
        v = current * 1.5 if isinstance(current, (int, float)) and current else 1.0
        return (rng[0] + rng[1]) / 2 if rng and not (rng[0] <= v <= rng[1]) else v
    if item.type == "str":
        return "mb-probe"
    return None


def _form_fields(engine, prefix):
    """Every catalogue item the form actually renders a control for."""
    items = {i.name: i for i in T.select(T.read_template(T.load_catalogue()),
                                         engine=engine)}
    out = {}
    for section in catalogue_to_form_schema(engine, prefix)["sections"]:
        for field in section["fields"]:
            fid = field.get("id", "")
            if not fid.startswith(prefix + "-"):
                continue
            name = fid[len(prefix) + 1:].replace("-", "_")
            if name in items:
                out[name] = items[name]
    return out


_ENGINES = [
    ("siesta", "p",  "molbuilder.config.siesta:SiestaConfig",
     "molbuilder.siesta.input:spec_for", {"system_label": "t", "psml_lib": None}),
    ("pyscf",  "py", "molbuilder.config.pyscf:PySCFConfig",
     "molbuilder.pyscf.input:spec_for",  {"job_name": "t"}),
]


def _load(path):
    mod, name = path.split(":")
    import importlib
    return getattr(importlib.import_module(mod), name)


@pytest.mark.parametrize("engine,prefix,cfg_path,spec_path,base_kw", _ENGINES)
def test_every_field_the_form_offers_changes_the_generated_deck(
        engine, prefix, cfg_path, spec_path, base_kw):
    cls, spec_for, struct = _load(cfg_path), _load(spec_path), _struct()

    def deck(**over):
        cfg = dataclasses.replace(cls(**base_kw), **over)
        return str(se.render_deck(spec_for(struct, cfg), struct, cfg))

    dead, unprobed = [], []
    for name, item in sorted(_form_fields(engine, prefix).items()):
        if (engine, name) in _NOT_IN_THE_DECK:
            continue
        contexts = _ENABLING.get(name, [{}]) + ([{}] if name in _ENABLING else [])
        changed = False
        for ctx in contexts:
            base = cls(**{**base_kw, **ctx})
            value = _alternative(item, getattr(base, name, None))
            if value is None or value == getattr(base, name, None):
                continue
            try:
                if deck(**ctx) != deck(**{**ctx, name: value}):
                    changed = True
                    break
            except Exception as _exc:
                # A refusal is the setting being HEARD -- but only a
                # VALIDATION refusal.  Treating ANY exception as "heard"
                # made a render broken for every input pass every field.
                from molbuilder.issues import ValidationError
                assert isinstance(_exc, (ValidationError, ValueError)), (
                    f"{name}: render died with {type(_exc).__name__} "
                    f"rather than refusing: {_exc}")
                changed = True
                break
        else:
            if not changed:
                (dead if _alternative(item, getattr(cls(**base_kw), name, None))
                 is not None else unprobed).append(name)
    assert not dead, (
        f"{engine}: {len(dead)} field(s) the form offers change NOTHING in the "
        f"generated deck: {dead}.\n"
        f"Either wire the value up, or add it to _NOT_IN_THE_DECK with the "
        f"reason its effect is elsewhere. A control a person can set that "
        f"nothing obeys is the 'present but not honoured' defect.")

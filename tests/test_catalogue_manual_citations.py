"""Every engine parameter cites the engine's own documentation — § 5.1.

`engines/template.md` § 5.1 makes three claims about `manual`, and each is
pinned here:

  1. every item whose ``kind`` is ``engine`` carries one;
  2. it names the RELEASE it was taken against, so a citation that goes stale
     is visible rather than silently wrong;
  3. it lives in the catalogue ALONE — mirroring it into the config classes
     would give a second home to the one fact whose job is to let a reviewer
     check the first.

The third is the one worth a test.  A future edit that "helpfully" adds
``"manual"`` to a field's metadata and to ``MIRRORED`` would look like tidying
and would quietly undo the reason the key exists.
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from molbuilder import template as T
from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig

ENGINES = [("siesta", SiestaConfig), ("pyscf", PySCFConfig)]

#: A citation opens with the engine and the RELEASE.  The versions are pinned
#: deliberately: when either moves, this test fails and the citations are
#: re-derived against the new documentation rather than inherited untouched.
CITES = re.compile(r"^(SIESTA 5\.4\.2|PySCF 2\.13) \S")


def _catalogue():
    return T.read_template(T.load_catalogue())


def test_every_engine_item_cites_its_manual():
    """§ 5.1: an engine item names where the engine documents that keyword."""
    missing = sorted(i.name for i in _catalogue().items
                     if i.kind == "engine" and not i.manual)
    assert not missing, (
        f"{len(missing)} engine item(s) carry no `manual` citation: "
        f"{missing}.  Every engine parameter names where the engine's own "
        f"documentation defines it (engines/template.md § 5.1) -- that is "
        f"what lets someone reviewing this file check the value instead of "
        f"trusting it.")


def test_a_citation_names_the_release_it_was_taken_against():
    """§ 5.1: *"the version is part of the citation, not decoration"*.

    A bare section number is worse than useless once a manual renumbers: it
    points somewhere confidently and wrongly, with nothing in the file to
    reveal that it moved."""
    bad = sorted(f"{i.name}: {i.manual!r}" for i in _catalogue().items
                 if i.manual and not CITES.match(i.manual))
    assert not bad, (
        "citation(s) that do not open with the engine and its release:\n  "
        + "\n  ".join(bad) +
        "\n\nExpected e.g. \"SIESTA 5.4.2 §6.9.2 'Mixing options'\" or "
        "\"PySCF 2.13 pyscf.scf.hf.SCF 'conv_tol'\".")


def test_a_siesta_citation_carries_a_section_number():
    """The SIESTA half cites a NUMBERED section, because the manual has them."""
    bad = sorted(f"{i.name}: {i.manual!r}" for i in _catalogue().items
                 if i.manual.startswith("SIESTA") and "§" not in i.manual)
    assert not bad, ("SIESTA citation(s) with no § section:\n  "
                     + "\n  ".join(bad))


@pytest.mark.parametrize("engine,cls", ENGINES,
                         ids=lambda x: getattr(x, "__name__", x))
def test_the_citation_has_exactly_one_home(engine, cls):
    """§ 5.1: the config classes do NOT mirror `manual`.

    `help`, `label`, `unit`, `range` and `choices` legitimately live in two
    homes until the form is rebuilt from the catalogue (§ 2.1a), and
    `test_catalogue_agreement.py` keeps those in step.  `manual` is outside
    that set ON PURPOSE: a fact duplicated into the thing it exists to check
    is a fact that can disagree with itself."""
    mirrored = sorted(f.name for f in dataclasses.fields(cls)
                      if f.metadata.get("manual"))
    assert not mirrored, (
        f"{cls.__name__} field(s) carry a `manual` in their metadata: "
        f"{mirrored}.  The citation lives in the catalogue alone "
        f"(engines/template.md § 5.1) -- it is the reference a reviewer "
        f"checks the catalogue WITH, so a second copy is a copy that can "
        f"drift from the thing it is checking.")


def test_declaration_for_never_invents_a_citation():
    """The other direction of the same rule: the config->Item bridge leaves
    `manual` empty, so a template emitted from a config cannot claim a
    citation the catalogue never made."""
    for _engine, cls in ENGINES:
        cited = sorted(d.name for d in T.declarations_for(cls) if d.manual)
        assert not cited, (
            f"declarations_for({cls.__name__}) produced citation(s) for "
            f"{cited}; `declaration_for` must not set `manual` (§ 5.1).")


def test_the_citation_survives_the_round_trip():
    """§ 4.1's round-trip, extended to the new key: a template written from a
    config and read back keeps the catalogue's citations.  Without this the
    key could be silently dropped by the writer and nothing would notice,
    because nothing else reads it."""
    cfg = SiestaConfig()
    text = T.template_with_values(cfg, engine="siesta")
    back = {i.name: i for i in T.read_template(text).items}
    cat = {i.name: i for i in T.select(_catalogue(), engine="siesta")}
    lost = sorted(n for n, i in cat.items()
                  if i.manual and not back.get(n, i).manual)
    assert not lost, (
        f"citation(s) lost on the write/read round trip: {lost}.  Nothing "
        f"else reads `manual`, so the writer dropping it would be invisible.")
    disagree = sorted(f"{n}: {cat[n].manual!r} -> {back[n].manual!r}"
                      for n in cat if n in back
                      and cat[n].manual != back[n].manual)
    assert not disagree, ("citation(s) changed on the round trip:\n  "
                          + "\n  ".join(disagree))

"""Tests for molbuilder.validation.chemistry.

Per docs/process/testing.md (test layout mirrors source
layout).  Split from the pre-2026-06-13 flat tests/test_validation.py
on 2026-06-13; no test body was modified.  Shared fixtures
(``water_struct``, ``_vacuum_cell``) live in tests/validation/conftest.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate
from ._helpers import _peptide_struct, _vacuum_cell




# --------------------------------------------------------------------- #
#  Peptide protonation: warn when neutral build vs charged side chains  #
# --------------------------------------------------------------------- #




def test_peptide_with_asp_glu_warns_on_zero_charge():
    """ARNDCEQGHI has 1 Arg(+1), 1 Asp(-1), 1 Glu(-1), 1 His (skipped),
    others neutral.  Expected pH-7 charge = -1.  cfg.net_charge = 0
    (default) should warn."""
    s = _peptide_struct(["ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE"])
    cfg = PySCFConfig()  # net_charge default = None -> auto-detect -> 0 for non-nucleic
    issues = validate(s, cfg)
    msgs = [i for i in issues if i.where == "config.net_charge"]
    assert len(msgs) == 1
    assert "-1" in msgs[0].message
    assert "neutral" in msgs[0].message.lower()



def test_peptide_with_explicit_charge_no_warn():
    """When the user explicitly sets cfg.charge = -1, the validator
    treats them as having opted in and stays silent."""
    s = _peptide_struct(["ALA","ARG","ASP","GLU","GLY"])
    cfg = PySCFConfig(net_charge=-1)
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.net_charge"] == []



def test_peptide_neutral_residues_no_warn():
    """A peptide of only neutral side chains (G, A, V, L, I) has
    estimated pH-7 charge = 0.  No warning."""
    s = _peptide_struct(["GLY","ALA","VAL","LEU","ILE"])
    cfg = PySCFConfig()
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.net_charge"] == []



def test_non_peptide_skips_protonation_check():
    """A nucleic-acid structure (no AA residue names) shouldn't trigger
    the peptide protonation check at all -- expected_pH7_peptide_charge
    returns None for non-peptides."""
    n = 5
    pos = np.column_stack([np.arange(n) * 3.0, np.zeros(n), np.zeros(n)])
    s = Structure(elements=["C"] * n, positions=pos,
                  residue_names=["DA","DT","DG","DC","DA"],
                  residue_ids=list(range(1, n + 1)))
    cfg = PySCFConfig()
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.net_charge"] == []


# --------------------------------------------------------------------- #
#  The ECP hint — it ASKS, it never chooses                             #
#                                                                       #
#  Added 2026-08-13 with T9.  molbuilder used to PICK an ECP here:       #
#  "lanl2dz" whenever any element had Z > 36 and the basis was not       #
#  def2.  That auto-rule is retired -- *"who defines heavy? there is no  #
#  clear reasoning or standard"* -- and the user asked for the other     #
#  half to stay: *"you can still have the validation function to give    #
#  hints - that should be confirmed."*  So the number survives ONLY as   #
#  the bound of a question, and the message prints it so a reader can    #
#  disagree.                                                             #
# --------------------------------------------------------------------- #

def _pt_complex():
    return Structure(
        elements=["Pt", "C", "C", "C", "C"],
        positions=np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0],
                            [-2, 0, 0], [0, -2, 0]]),
        vacuum=(12.0, 12.0, 12.0))


def _ecp_findings(struct, **kw):
    cfg = PySCFConfig(job_name="pt", **kw)
    return [i for i in validate(struct, cfg)
            if getattr(i, "where", "") == "config.ecp"]


def test_all_electron_heavy_atom_is_pointed_out():
    found = _ecp_findings(_pt_complex(), basis="cc-pVDZ")
    assert len(found) == 1
    msg = found[0].message
    assert "Pt" in msg and "ALL-ELECTRON" in msg
    # It must show its own criterion rather than hiding one.
    assert "Z > 36" in msg
    # And it must say how to answer it, in the field's own vocabulary.
    assert "ecp_atoms" in msg


def test_the_hint_is_a_warning_and_never_blocks():
    """A hint the user confirms.  An error would be molbuilder deciding
    that all-electron Pt is not allowed, which is not its call."""
    found = _ecp_findings(_pt_complex(), basis="cc-pVDZ")
    assert found and all(i.severity == "warn" for i in found)


def test_a_declared_ecp_covering_the_element_silences_it():
    assert _ecp_findings(_pt_complex(), basis="cc-pVDZ",
                         ecp="lanl2dz", ecp_atoms=["Pt"]) == []
    assert _ecp_findings(_pt_complex(), basis="cc-pVDZ",
                         ecp="lanl2dz", ecp_atoms=["*"]) == []


def test_a_selector_that_MISSES_the_element_still_warns():
    """The case a coarser check would let through: an ECP is declared,
    but ``["C"]`` does not cover the Pt.  A typo (``["P"]`` for
    ``["Pt"]``) reads exactly like this."""
    found = _ecp_findings(_pt_complex(), basis="cc-pVDZ",
                          ecp="lanl2dz", ecp_atoms=["C"])
    assert len(found) == 1 and "Pt" in found[0].message


def test_def2_brings_its_own_and_the_check_stays_quiet():
    """A fact about that basis family, not a rule applied elsewhere."""
    for basis in ("def2-SVP", "def2_SVP", "def2svp", "DEF2-TZVP"):
        assert _ecp_findings(_pt_complex(), basis=basis) == [], basis


def test_light_elements_are_never_mentioned():
    water = Structure(elements=["O", "H", "H"],
                      positions=np.array([[0.0, 0, 0], [0.96, 0, 0],
                                          [-0.24, 0.93, 0]]),
                      vacuum=(12.0, 12.0, 12.0))
    assert _ecp_findings(water, basis="cc-pVDZ") == []

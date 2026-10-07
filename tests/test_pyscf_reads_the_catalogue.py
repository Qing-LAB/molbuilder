"""PySCF's SCF settings go through the one door — `script-preparation.md` § 4.2.

The value and its reason are one act, and the reason is the catalogue's.
"""
from __future__ import annotations


from molbuilder.pyscf import layout
from molbuilder.script_emit import parameter


# --------------------------------------------------- emission conditions --


def test_the_layout_names_catalogue_items_that_actually_exist():
    """A typo in the layout table would silently emit nothing at all."""
    missing = [n for n in layout.SCF_SECTION.items
               if not parameter(n, "pyscf").known]
    assert missing == [], f"layout names items PySCF does not declare: {missing}"

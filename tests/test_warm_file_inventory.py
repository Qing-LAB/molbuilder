"""A generated run wrapper never changes directory (`job-contracts.md` § 2.1)."""
from __future__ import annotations

import re

import pytest

from support.road import prepared_here

pytestmark = pytest.mark.usefixtures("in_a_folder_of_its_own")


# PySCF's run script: no basic-tier run starts it, so this lint is what
# holds it in the folder it was launched in.
def test_no_generated_wrapper_changes_directory(tmp_path, monkeypatch):
    """`job-contracts.md § 2.1`: **the caller's working directory is the
    contract** — both launchers establish it, and neither the wrapper nor the
    engine ever navigates.
    """
    _bundle, run, names = prepared_here(tmp_path, monkeypatch,
                                        "conda activate", engine="pyscf")
    cds = re.findall(r"(?m)^\s*cd\s+\S+",
                     (run / names.run_script).read_text())
    assert cds == [], f"the run script navigates: {cds}"



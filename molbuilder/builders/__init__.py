"""molbuilder.builders -- structure-building verbs.

Currently hosts ``backends/`` (the per-tool nucleic-acid builders:
``_amber.py``, ``_rdkit.py``, ``_threedna.py``).  The build verbs
themselves (peptide / nucleic / smiles / pubchem) still live at
the top level (``molbuilder/peptide.py`` etc.) and import
``molbuilder.builders.backends`` directly -- the re-export shim at
``molbuilder/backends/`` was deleted 2026-09-03.
"""

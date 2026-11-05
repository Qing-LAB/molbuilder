"""molbuilder.pyscf -- PySCF deck description.

Submodules:
    input  -- spec_for: the deck's spec, which `jobset prep` renders, writes
              and checks (`script_emit.prepare_deck`)

It re-exports nothing, as `jobset/__init__` does, so a reader of what travels
beside a job (`pyscf.end_lines`) loads no deck writer.
"""

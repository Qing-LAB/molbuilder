"""The lines a molbuilder PySCF deck prints when it reaches its own end.

A format molbuilder GENERATES does not get a sniffed reader
(`model/parse.md` § 5.5): the end line is a string WE print, so the emitters'
package declares it, both emitters print it from here, and the reader
(`parse/engines/_run_ending.py`) imports it -- the ``ROLE_GEOM_TRAJ`` pattern
applied to a line.

**Stdlib only, and nothing of ours**: this module travels beside every job
(`runwrap.MONITOR_COMPANIONS`, `execution/run-reports.md` § 2.3), so the
monitor reads how a PySCF run ended with the reader the Results tab uses --
and beside every PySCF script (`runwrap.PYSCF_COMPANIONS`), whose
progress-log writer writes the footer words from it.
The two constants lived in the emitters themselves until 2026-09-26, where a
reader that must run without molbuilder could not reach them.
"""

#: The relaxation deck's (`pyscf/input.py`).  Reached only on the success
#: path, AFTER the final geometry is written, so it means the run ended -- and
#: it outranks anything the script caught and reported on the way (a real log
#: carries *"Frequency analysis FAILED: ..."* three lines above it).
END_MARKER = "Job complete in"

#: The spectrum deck's (`pyscf/vibration_emitters.py`), which does not print
#: the relaxation deck's line.  The two spectrum runs in the tree end with
#: *"Total wall time: 5090.8 s"* -- and they are exactly the two directories
#: that reported `running` for months (`plans/plan.md` § 5c.2), because
#: nothing read this line at all.
SPECTRUM_END_MARKER = "Total wall time:"


#: The progress log's FOOTER, which a PySCF deck's exit hook appends to its
#: `.molwatch.log` (`pyscf/input.py`) and `molwatch_grammar` reads: the error
#: line when the run raised, then the concluded line.  Declared here, beside
#: the decks' end lines, so the writer and the reader share one spelling.
FOOTER_ERROR = "# error:"
FOOTER_CONCLUDED = "# concluded:"

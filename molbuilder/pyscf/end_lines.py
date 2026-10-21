"""The lines a molbuilder PySCF deck prints when it reaches its own end.

A format molbuilder GENERATES does not get a sniffed reader
(`model/parse.md` § 5.5): the end line is a string WE print, so the emitters'
package declares it, both emitters print it from here, and the reader
(`parse/engines/_run_ending.py`) imports it -- the ``ROLE_GEOM_TRAJ`` pattern
applied to a line.

**Stdlib only, and nothing of ours**: this module travels beside every job
(`runwrap.MONITOR_COMPANIONS`, `execution/run-reports.md` § 2.3), so the
monitor reads how a PySCF run ended with the reader the Results tab uses.
The progress log's footer lines are the log's own vocabulary
(`trajectory_log.keys`).
"""

#: The relaxation deck's (`pyscf/input.py`).  Reached only on the success
#: path, AFTER the final geometry is written, so it means the run ended -- and
#: it outranks anything the script caught and reported on the way (a real log
#: carries *"Frequency analysis FAILED: ..."* three lines above it).
END_MARKER = "Job complete in"

#: The spectrum deck's (`pyscf/vibration_emitters.py`), which does not print
#: the relaxation deck's line.
SPECTRUM_END_MARKER = "Total wall time:"

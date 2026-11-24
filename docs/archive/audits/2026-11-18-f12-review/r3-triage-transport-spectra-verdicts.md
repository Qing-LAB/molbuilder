# R3 triage — R2-5 transport (23) + R2-6 spectra (26) against 2.77.16

Fixed: R2-5 #3 (2.77.8). LEAVE: #16, F14. R5: #14. R4: all T rows (transport T1–T7, spectra T1–T11), gf_files.estimate_from test.
VISIBLE (needs the user's yes, plain):
- R2-6 F7: a vibration's spectrum data file says "computed by siesta" while its optical/Raman files say "SIESTA" -> engine words from engine_table (recommended: capitalised everywhere).
- R2-5 #11: extending a finished transport calculation reads "0.0 V, 0.2 V"; everywhere else "0 V" -> one spelling.
DO (drift only, on the road):
- transport #1 citation kind from the class (citation_defaults:299, deck._pseudopotential_note).
- #2 leads ordered by sort.electrode_orientation; axis via kmesh.TRANSPORT_AXIS.
- #4 temperature default from engine_defaults(); #10 gf_files keywords via catalogue anchors.
- #5 deck_window via units.energy_ev; window spacing via energies.spacing.
- #6 vocabularies: spin channels (runfiles.TBT_CHANNEL_TAGS), SCF/PRODUCT (one travelling home), AXIS_KINDS, treatments beside Task.treatment (M).
- #7 frames_had the one door (M).
- #8 SHAPE_OF_RUNG from the roles; gf_files "device" via SHAPE_RUN.
- #9 LIVE_STATES; #12 reuse _at/_given_at one; #13 copy helpers one; #15 delete layer_tol_ang param.
- spectra F1 sidecar readers through persist.object_of (non-finite refusal an option, spectra only) (M).
- F2 drop from_dict's version gate; F3 SUFFIX/PAIRED_SUFFIX in runfiles; F4 state words one table; F5 IR_CHARGES row table; F6 route constants; F8 one step-to-frame function (M); F9 runrecord.stamp; F10 Structure.frozen_atoms; F11 to_dict(view=False); F12 import-time table check; F13 siesta_fc step pattern; F15 SCF log column order in keys.py + one atom-row formatter.

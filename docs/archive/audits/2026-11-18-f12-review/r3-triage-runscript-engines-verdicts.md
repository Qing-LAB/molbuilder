# R3 triage — R2-3 run script/monitor (26) + R2-4 template/engines (31) against 2.77.16

Fixed: F2, F9 (2.77.16), P1, P8 (2.77.5); P5 partly (LEAVE). LEAVE: F17, F25, P16, P12(d), P24(d,e). R4: F12, F13, T1 helper, T2–T7, P19's private-_emit tests.
VISIBLE / needs the user:
- P2: the displacement amplitude outside 0.02–0.2 Å draws two warnings (general range + a hand-typed one explaining SCF noise / anharmonicity) — which one stays; the explanation into the help text?
- P11 (fix, no choice): PySCF deck comment says "(already)" for basis/density fitting whatever was set — false; compose from config or drop.
- P15: PySCF result's relaxation block lacks `record`.
- P3: BENCH-MARKS block's Diag.Algorithm gains its choices.
DO:
- runwrap F1 bundles through the one closure (S) ; F16 deck_text required; F18 JOB_VARIABLES interpolated.
- monitor F15 ALL_CHANNELS; F18 one MB_JOB_ID constant.
- config_dir/serve_daemon/placement F4 one shepherd marker; F5 *_pattern; F24 one verify step.
- runtime_config/record/envs F6, F7, F10 (ACTIVATION_FORMS rows), F11 BuildPaths placeholders (M), F14 one _quiet_run, F20 drop siesta row, F26 Domain.sizes.
- F8, F19 deletions (restart-group constants + RestartGroup; molwatch_log).
- script_emit F21 Parameter.writes; F22 one clock read.
- quantities F23 _MEM_UNITS; P7 canonical_gres.
- template F3 SAID order; P9 axis letters; P12 a–c; P19 catalogue-only keys (M).
- validation P2 (after the user), P6, P13, P14, P24(a).
- engine writers P3 (M), P4, P20, P21, P24(b).
- spectra.results P15 relaxation_block (M).
- kind_table/runpoints/runfiles P10, P17, P18.
- resolve P23 one layering helper (M).
- makov_payne P22 deletion + the test.
- transport.energies T1 product reader (= R2-5 #5).
- bench.result P24(c) AS_RESOURCE.

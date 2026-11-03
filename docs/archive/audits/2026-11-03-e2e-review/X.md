# X — tests/spectra/test_spectrumchart_box_e2e.py (read by me, 2026-11-02)

Contract: docs/web/spectrumchart.md § 11, fifth rule: "The frame states its size; the surface fills it,
carries no padding, and never spills past it."

| test (line) | asserts | contract § | verdict | evidence | revision |
|---|---|---|---|---|---|
| test_the_frame_pads_and_the_surface_does_not (31) | surface padding 0; frame padding non-zero; surface width ≤ frame; painted height = the token; frame overflow hidden | spectrumchart.md § 11 rule 5 | RIGHT, one WRONG DETAIL | :95-98 demands the FRAME pad ("the rule is that the frame pads") — § 11 says only the surface carries no padding; the docstring (:3) carries literal `\n` escapes from a botched edit and quotes "the frame pads while the surface does not", which § 11 no longer says | drop the frame-padding assertion (:95-98); docstring states § 11's rule as it reads; name the test for the rule (`test_the_frame_states_its_size_and_the_surface_fills_it_unpadded`) |

Browser-only (computed style after the cascade): a real e2e.

/* sidecar-labels.js — single client-side helper for fetching a
 * structure file's ``.molstruct.json`` sidecar.
 *
 * 2026-06-14 contract update: every tab that lets the user load a
 * structure and then click Generate (optimization, spectrum,
 * transport, modify) MUST hold ``frozen_atoms`` + ``regions`` in
 * tab-local memory at Load time, then ship them in the Generate
 * POST body.  Pre-fix tabs sent only a ``structure_path`` and the
 * server re-read the sidecar from disk — that path-indirection
 * produced multiple silent-no-op failure modes (path stale, sidecar
 * missing, suffix not recognised, atom-count drift) where the
 * generated script ended up with no frozen atoms even though the
 * user saw the labels in the viewer.
 *
 * The contract is: the viewer is the truth.  This helper makes
 * fetching the labels a one-line call from each tab so all tabs
 * agree on:
 *
 *   * sidecar path computation (``<stem>.molstruct.json`` next to
 *     the loaded XYZ/PDB, mirroring molstruct_json.sidecar_path_for
 *     on the server),
 *   * silent-OK on 404 (no sidecar = no labels; empty result is
 *     the explicit truth, not a drop),
 *   * server-side JSON parsing (parse here once instead of in every
 *     caller).
 *
 * Mount: ``window.molbuilder.sidecarLabels.fetch(path)``.
 */
(function (root) {
    "use strict";

    /**
     * Return the conventional sidecar path next to a structure file.
     *
     * Matches the server's ``parsers.molstruct_json.sidecar_path_for``
     * rule: a structure at ``/foo/bar/baz.xyz`` has its sidecar at
     * ``/foo/bar/baz.molstruct.json``.  The suffix substitution
     * works for ``.xyz`` and ``.pdb`` and is forgiving on
     * mixed-case (``BDT.XYZ`` -> ``BDT.molstruct.json``).
     *
     * Returns "" when ``path`` has no recognised structure suffix —
     * callers can use that to early-return without an HTTP roundtrip.
     */
    function sidecarPathFor(path) {
        if (!path) return "";
        const p = String(path);
        const lc = p.toLowerCase();
        if (lc.endsWith(".xyz")) {
            return p.slice(0, p.length - 4) + ".molstruct.json";
        }
        if (lc.endsWith(".pdb")) {
            return p.slice(0, p.length - 4) + ".molstruct.json";
        }
        return "";
    }

    /**
     * Fetch the sidecar at ``sidecarPathFor(path)`` and return its
     * ``frozen_atoms`` + ``regions`` fields.
     *
     * Returns
     * -------
     * Promise<{frozen_atoms: number[], regions: {<label>: number[]}}>
     *   Always resolves.  Missing sidecar -> ``{frozen_atoms: [],
     *   regions: {}}``.  Malformed JSON -> same empty defaults.
     *   The caller stores the returned object in tab-local state
     *   and ships it verbatim in the Generate POST body.
     *
     * Why "always resolve, never reject":
     *   * No-sidecar-on-disk is the common case (a freshly-built
     *     structure or one without labels).  Treating that as an
     *     error would force every caller to wrap a try/catch
     *     around a routine load.
     *   * The empty default mirrors the server's "no in-body
     *     labels" semantics: an explicit empty list is the
     *     client's "I have nothing to freeze" claim, which the
     *     server uses verbatim (no surprise sidecar re-read).
     */
    async function fetchSidecarLabels(path) {
        const emptyDefault = { frozen_atoms: [], regions: {} };
        const sidecar = sidecarPathFor(path);
        if (!sidecar) return emptyDefault;
        try {
            const r = await fetch(
                "/api/files/read?path=" + encodeURIComponent(sidecar)
            );
            if (!r.ok) return emptyDefault;
            const body = await r.json();
            if (!body || !body.ok || !body.text) return emptyDefault;
            const parsed = JSON.parse(body.text);
            return {
                frozen_atoms: Array.isArray(parsed.frozen_atoms)
                    ? parsed.frozen_atoms.slice() : [],
                regions: (parsed.regions
                    && typeof parsed.regions === "object")
                    ? Object.assign({}, parsed.regions)
                    : {},
            };
        } catch (_) {
            return emptyDefault;
        }
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.sidecarLabels = {
        fetch:           fetchSidecarLabels,
        sidecarPathFor:  sidecarPathFor,
    };
})(typeof window !== "undefined" ? window : globalThis);

/*
 * Bundle-for-next-stage card -- Step 3 PR-E (task #492).
 *
 * Mounts the form in ``_bundle_handoff_panel.html``, POSTs to
 * ``/api/results/bundle``, and surfaces the response.
 *
 *   * Pre-fills ``run_dir`` and ``target_dir`` from the projects-
 *     sidebar's current cursor (when available).  ``stem`` defaults
 *     to the source script's basename which the endpoint reports
 *     back; we don't try to guess it client-side because the
 *     bundle layer picks the .fdf/.py via atom-count rules the
 *     UI can't easily replicate.
 *   * On success: renders a summary panel (engine, coords source,
 *     n_atoms, regions, notes) and calls ``projects.refresh(target_dir)``
 *     so the sidebar lists the new pair.  See bundle-contract.md
 *     § 7.5 for the client-side refresh contract.
 *   * On error: surfaces the endpoint's ``error`` message verbatim.
 *     Most errors carry actionable text (which field is missing,
 *     why the run dir can't be bundled, etc.).
 *
 * Contract: ``docs/protocols/bundle-contract.md`` §§ 5.2 + 7.
 */
(function () {
    'use strict';

    function $(sel, root) { return (root || document).querySelector(sel); }

    function setText(el, txt) {
        if (!el) return;
        el.textContent = txt;
    }

    /** Return the sidebar's current selection as ``{dir, file}``,
     *  or null when the public API isn't on the page.
     *
     *  Audit follow-up: the prior version read ``getState().cursor``
     *  which the projects-sidebar public surface (state.js § C1)
     *  has never exposed.  The actual API is ``getCurrentDir()`` +
     *  ``getCurrentFile()`` (state.js:767-768). */
    function currentSidebarSelection() {
        try {
            var ws = (window.molbuilder && window.molbuilder.projects) || null;
            if (!ws ||
                typeof ws.getCurrentDir  !== 'function' ||
                typeof ws.getCurrentFile !== 'function') return null;
            return {
                dir:  ws.getCurrentDir()  || '',
                file: ws.getCurrentFile() || '',
            };
        } catch (_) {
            return null;
        }
    }

    /** Navigate the sidebar to ``dir`` so the user sees the new
     *  bundle pair without manual refresh.  Per bundle-contract.md
     *  § 7.5: the endpoint does NOT route through writeFile so the
     *  auto-refresh path doesn't fire -- the client invokes
     *  navigateTo (NOT refresh -- refresh() takes no dir arg and
     *  re-lists wherever the sidebar already points; see state.js:296).
     *  Tolerate any missing surface so the bundle write still
     *  counts as a success. */
    function navigateSidebarTo(dir) {
        try {
            var ws = (window.molbuilder && window.molbuilder.projects) || null;
            if (ws && typeof ws.navigateTo === 'function') {
                ws.navigateTo(dir);
            }
        } catch (_) { /* swallow -- the bundle is already on disk */ }
    }

    function renderResult(result, body) {
        result.hidden = false;
        result.className = 'bundle-handoff-result';
        var notConverged = (
            body.final_coords_from === 'fdf-initial' ||
            body.final_coords_from === 'py-initial'
        );
        if (notConverged) {
            result.classList.add('not-converged');
        }
        var html = [
            '<div class="bundle-handoff-summary">',
            '  <div><strong>Wrote</strong> ',
            '    <code>', escapeHtml(body.xyz_path), '</code> + ',
            '    <code>', escapeHtml(body.sidecar_path), '</code>',
            '  </div>',
            '  <div><strong>Engine:</strong> ', escapeHtml(body.source_engine),
            '    &middot; <strong>Coords:</strong> ', escapeHtml(body.final_coords_from),
            (notConverged ? ' <em>(NOT converged geometry)</em>' : ''),
            '    &middot; <strong>Atoms:</strong> ', body.n_atoms,
            '    &middot; <strong>Frozen:</strong> ', body.frozen_atoms_count,
            '  </div>',
        ];
        if (body.regions && body.regions.length) {
            html.push('  <div><strong>Regions:</strong> ',
                      body.regions.map(escapeHtml).join(', '),
                      '</div>');
        } else {
            html.push('  <div class="bundle-handoff-no-labels">',
                      '    No region labels.  The next stage will see ',
                      '    an unlabeled structure.',
                      '  </div>');
        }
        if (body.notes && body.notes.length) {
            html.push('  <ul class="bundle-handoff-notes">');
            body.notes.forEach(function (n) {
                html.push('    <li>', escapeHtml(n), '</li>');
            });
            html.push('  </ul>');
        }
        html.push('</div>');
        result.innerHTML = html.join('');
    }

    function renderError(result, msg) {
        result.hidden = false;
        result.className = 'bundle-handoff-result is-error';
        result.innerHTML = (
            '<div class="bundle-handoff-summary">' +
            '  <strong>Bundle failed:</strong> ' +
            escapeHtml(msg) +
            '</div>'
        );
    }

    function escapeHtml(s) {
        if (s == null) return '';
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function mount() {
        var form   = $('#bundle-handoff-form');
        if (!form) return;       // panel not on this page; bail.
        var runIn  = form.querySelector('input[name="run_dir"]');
        var dstIn  = form.querySelector('input[name="target_dir"]');
        var stemIn = form.querySelector('input[name="stem"]');
        var ow     = form.querySelector('input[name="overwrite"]');
        var btn    = form.querySelector('button[type="submit"]');
        var status = $('[data-status]', form);
        var result = $('[data-result]', form.parentElement || form);

        // Pre-fill from the sidebar's selection.  Use the
        // public API explicitly: getCurrentFile() tells us whether
        // the cursor is on a file (so we want its parent for
        // run_dir) vs. a directory (use verbatim).  Pre-fix this
        // used a "does last segment contain '.'" heuristic which
        // misfired on dotted directory names like ``v1.2/``.
        var sel = currentSidebarSelection();
        var runDirDefault    = '';
        var targetDirDefault = '';
        if (sel) {
            if (sel.file) {
                // Cursor is on a file: bundle from the file's dir.
                var idx = sel.file.lastIndexOf('/');
                runDirDefault = (idx > 0) ? sel.file.substring(0, idx) : sel.dir;
            } else {
                runDirDefault = sel.dir;
            }
            // Default target lands in a DEDICATED sub-dir of the
            // run dir, NOT the run dir itself.  Pre-fix the default
            // was the run dir verbatim, which mixed bundle output
            // with the run's source script -- a sibling .fdf could
            // then shadow the bundle's sidecar via
            // apply_companion_labels_if_present on the next load.
            // bundle-contract.md § 7.2 + audit-IMPORTANT.
            targetDirDefault = runDirDefault
                ? (runDirDefault + '/handoff')
                : '';
        }
        if (runIn  && !runIn.value)  runIn.value  = runDirDefault;
        if (dstIn  && !dstIn.value)  dstIn.value  = targetDirDefault;
        if (stemIn && !stemIn.value) {
            stemIn.value = 'handoff';
        }

        form.addEventListener('submit', function (e) {
            e.preventDefault();
            if (!runIn.value || !dstIn.value || !stemIn.value) {
                setText(status, 'Fill all three paths first.');
                return;
            }
            btn.disabled = true;
            setText(status, 'Bundling…');
            result.hidden = true;
            fetch('/api/results/bundle', {
                method:  'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({
                    run_dir:    runIn.value.trim(),
                    target_dir: dstIn.value.trim(),
                    stem:       stemIn.value.trim(),
                    overwrite:  !!ow.checked,
                }),
            })
                .then(function (r) {
                    return r.json().then(function (body) {
                        return { status: r.status, body: body };
                    });
                })
                .then(function (resp) {
                    btn.disabled = false;
                    setText(status, '');
                    if (resp.body && resp.body.ok) {
                        renderResult(result, resp.body);
                        navigateSidebarTo(dstIn.value.trim());
                    } else {
                        var msg = (resp.body && resp.body.error)
                            || ('HTTP ' + resp.status);
                        renderError(result, msg);
                    }
                })
                .catch(function (err) {
                    btn.disabled = false;
                    setText(status, '');
                    renderError(result, String(err));
                });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mount);
    } else {
        mount();
    }
})();

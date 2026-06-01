/* projects/api.js -- HTTP wrappers for /api/files/* + /api/projects/*.
 *
 * Pure functions: no DOM, no module-level state.  Each function
 * issues one HTTP request and returns the parsed JSON body, ALWAYS
 * shaped as ``{ok: bool, ...}``.  Network errors AND non-JSON
 * responses are caught here and surfaced as
 * ``{ok: false, error: "..."}`` so callers don't need a try/catch
 * around every call.
 *
 * The backend contract is in ``docs/protocols/selection.md`` § and
 * ``docs/protocols/web-api.md`` § ``/api/files/*``.  The uniform-
 * envelope contract is in ``docs/protocols/projects-sidebar.md``
 * Principle 6 ("every public method returns {ok, ...} or {ok:false,
 * error}; NEVER throws").
 */

/**
 * Issue one HTTP request and return the parsed JSON body.  All
 * failure modes (network drop, browser offline, server returning
 * 5xx with non-JSON, DNS failure, ...) map to
 * ``{ok: false, error: "<reason>"}`` so callers can rely on
 * ``response.ok`` without try/catch.
 *
 * ``fetchInit`` is forwarded verbatim to ``fetch`` so callers can
 * set method / headers / body / credentials.  The function does
 * NOT throw under any condition expected in production; an
 * unexpected throw indicates a programmer error (e.g. an invalid
 * URL).
 */
async function _fetchEnvelope(url, fetchInit) {
  let resp;
  try {
    resp = await fetch(url, fetchInit);
  } catch (e) {
    // Distinguish user-initiated cancellation (AbortError) from
    // genuine network failure.  Both look like exceptions to the
    // caller, but the UI usually wants to silently dismiss an
    // abort (it was the user's choice) while showing a banner for
    // a network drop.  The ``aborted: true`` flag lets callers
    // branch without parsing the error string.
    if (e && e.name === "AbortError") {
      return { ok: false, error: "aborted", aborted: true };
    }
    // Network-level failure: TypeError "Failed to fetch", DNS
    // failure, CORS rejection at preflight, etc.  Surface the
    // error name + message; callers usually just need to know
    // "could not reach server".
    return {
      ok:    false,
      error: "network error: " + (e && e.message
                                  ? e.message
                                  : String(e)),
    };
  }
  // Successful network, but the server might have returned non-JSON
  // (5xx with an HTML error page, 501 from a stubbed endpoint, ...).
  try {
    return await resp.json();
  } catch (_e) {
    return {
      ok:    false,
      error: "server returned non-JSON (status " + resp.status + ")",
    };
  }
}


export async function apiRoots() {
  // The /api/files/roots endpoint responds with
  //   ``{roots: [{path, label, ...}]}``
  // (no top-level ``ok`` in the success case; the presence of
  // ``roots`` IS the success signal).  We normalise here so the
  // caller always sees the uniform envelope:
  //   ``{ok: true,  roots: [...]}``           (success)
  //   ``{ok: false, error: "...", roots: []}`` (failure; roots stub
  //   present so callers that destructure ``{roots}`` don't NPE).
  const body = await _fetchEnvelope("/api/files/roots");
  if (body.ok === false) {
    return { ok: false, error: body.error, roots: [] };
  }
  return { ok: true, roots: body.roots || [] };
}

export async function apiList(path, opts) {
  opts = opts || {};
  return await _fetchEnvelope(
    "/api/files/list?path=" + encodeURIComponent(path),
    { signal: opts.signal }
  );
}

export async function apiStat(path, opts) {
  opts = opts || {};
  return await _fetchEnvelope(
    "/api/files/stat?path=" + encodeURIComponent(path),
    { signal: opts.signal }
  );
}

export async function apiRead(path, opts) {
  opts = opts || {};
  return await _fetchEnvelope(
    "/api/files/read?path=" + encodeURIComponent(path),
    { signal: opts.signal }
  );
}

export async function apiMkdir(parent, name, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/files/mkdir", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({parent: parent, name: name}),
    signal:  opts.signal,
  });
}

export async function apiCreateProject(name, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/projects/create", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({name: name}),
    signal:  opts.signal,
  });
}

/* Every endpoint accepts an ``opts.signal`` AbortSignal.  The
 * lock's three-layer recovery (timeout + Cancel button +
 * try/finally) relies on this -- without a signal threaded all
 * the way to fetch, clicking Cancel during a slow read OR write
 * would unlock the UI but leave the request running.  Read
 * endpoints (list / stat / read) added signal support 2026-05-31
 * to close the design § C3 + § C5 contract (#175 follow-up).
 * (See docs/protocols/projects-sidebar.md Layer B + #174.) */

export async function apiUpload(targetDir, file, opts) {
  opts = opts || {};
  const fd = new FormData();
  fd.append("target_dir", targetDir);
  fd.append("file", file);
  return await _fetchEnvelope("/api/files/upload", {
    method: "POST",
    body:   fd,
    signal: opts.signal,
  });
}

export async function apiDelete(path, recursive, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/files/delete", {
    method:  "DELETE",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({path: path, recursive: !!recursive}),
    signal:  opts.signal,
  });
}

export async function apiRename(path, newName, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/files/rename", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({path: path, new_name: newName}),
    signal:  opts.signal,
  });
}

export async function apiWrite(path, text, opts) {
  opts = opts || {};
  const body = {path: path, text: text};
  if (opts.overwrite) body.overwrite = true;
  if (opts.expected_mtime != null) body.expected_mtime = opts.expected_mtime;
  return await _fetchEnvelope("/api/files/write", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify(body),
    signal:  opts.signal,
  });
}

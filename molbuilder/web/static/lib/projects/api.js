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
    // Network-level failure: TypeError "Failed to fetch", DNS
    // failure, CORS rejection at preflight, AbortError, etc.
    // Surface the error name + message; callers usually just
    // need to know "could not reach server".
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

export async function apiList(path) {
  return await _fetchEnvelope(
    "/api/files/list?path=" + encodeURIComponent(path)
  );
}

export async function apiStat(path) {
  return await _fetchEnvelope(
    "/api/files/stat?path=" + encodeURIComponent(path)
  );
}

export async function apiRead(path) {
  return await _fetchEnvelope(
    "/api/files/read?path=" + encodeURIComponent(path)
  );
}

export async function apiMkdir(parent, name) {
  return await _fetchEnvelope("/api/files/mkdir", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({parent: parent, name: name}),
  });
}

export async function apiCreateProject(name) {
  return await _fetchEnvelope("/api/projects/create", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({name: name}),
  });
}

export async function apiUpload(targetDir, file) {
  const fd = new FormData();
  fd.append("target_dir", targetDir);
  fd.append("file", file);
  return await _fetchEnvelope("/api/files/upload", {
    method: "POST",
    body:   fd,
  });
}

export async function apiDelete(path, recursive) {
  return await _fetchEnvelope("/api/files/delete", {
    method:  "DELETE",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({path: path, recursive: !!recursive}),
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
  });
}

import { useCallback, useEffect, useState } from "react";
import { errorText, getJson, sendWithToken } from "./api.js";

// Fetch one or more endpoints, refetch when `paths` or `reloadKey` change and
// every `refreshMs` (0: never). Returns { data, error, loading, refresh }.
export function useApi(paths, { refreshMs = 0, reloadKey = 0 } = {}) {
  const key = JSON.stringify([paths, reloadKey]);
  const [state, setState] = useState({ data: null, error: null, loading: true });

  const refresh = useCallback(async () => {
    const [list] = JSON.parse(key);
    try {
      const results = await Promise.all(list.map(getJson));
      setState({ data: results, error: null, loading: false });
    } catch (e) {
      setState((prev) => ({ ...prev, error: String(e), loading: false }));
    }
  }, [key]);

  useEffect(() => {
    setState((prev) => ({ ...prev, loading: true }));
    refresh();
    if (!refreshMs) return undefined;
    const id = setInterval(refresh, refreshMs);
    return () => clearInterval(id);
  }, [refresh, refreshMs]);

  return { ...state, refresh };
}

// Accept / reject a proposal; same effect as a `/docdebt accept|reject <id>`
// reply on the PR. A rejection always carries its reason (false_positive,
// incorrect_fix, manual, other): it feeds the observed precision. An accept
// commits the field-by-field fix by default; `options.mechanism` can be
// "regeneration" or "edited" (with `options.editedDoc`).
// `results` maps short_id -> { ok, message } for display.
export function useDecision(onDone) {
  const [busyId, setBusyId] = useState(null);
  const [results, setResults] = useState({});

  const decide = useCallback(
    async (proposal, decision, reason = null, options = {}) => {
      const label = `${proposal.method_label} (${proposal.owner}/${proposal.repo}#${proposal.pr_number})`;
      const what = { regeneration: "the regenerated Javadoc", edited: "your edited Javadoc" }[options.mechanism] || "the fix";
      if (decision === "accept" && !options.skipConfirm && !window.confirm(`Commit ${what} for ${label} to the pull request branch?`)) {
        return false;
      }
      const setResult = (ok, message) => setResults((prev) => ({ ...prev, [proposal.short_id]: { ok, message } }));
      setBusyId(proposal.short_id);
      let ok = false;
      try {
        const { status, body } = await sendWithToken("POST", `/api/proposals/${proposal.short_id}/decide`, {
          decision,
          ...(reason ? { reason } : {}),
          ...(options.mechanism ? { mechanism: options.mechanism } : {}),
          ...(options.editedDoc ? { edited_doc: options.editedDoc } : {}),
        });
        if (status === 200) {
          ok = body.status !== "not_applied" && body.status !== "expired";
          setResult(ok, body.message);
        } else if (status !== 0) setResult(false, errorText(body, status));
      } catch (e) {
        setResult(false, `Could not reach the service: ${e}`);
      } finally {
        setBusyId(null);
        onDone?.();
      }
      return ok;
    },
    [onDone],
  );

  return { busyId, results, decide };
}

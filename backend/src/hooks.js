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
// reply on the PR. `results` maps short_id -> { ok, message } for display.
export function useDecision(onDone) {
  const [busyId, setBusyId] = useState(null);
  const [results, setResults] = useState({});

  const decide = useCallback(
    async (proposal, decision) => {
      const label = `${proposal.method_label} (${proposal.owner}/${proposal.repo}#${proposal.pr_number})`;
      if (decision === "accept" && !window.confirm(`Commit the fix for ${label} to the pull request branch?`)) {
        return;
      }
      const setResult = (ok, message) => setResults((prev) => ({ ...prev, [proposal.short_id]: { ok, message } }));
      setBusyId(proposal.short_id);
      try {
        const { status, body } = await sendWithToken("POST", `/api/proposals/${proposal.short_id}/decide`, {
          decision,
        });
        if (status === 200) setResult(body.status !== "not_applied" && body.status !== "expired", body.message);
        else if (status !== 0) setResult(false, errorText(body, status));
      } catch (e) {
        setResult(false, `Could not reach the service: ${e}`);
      } finally {
        setBusyId(null);
        onDone?.();
      }
    },
    [onDone],
  );

  return { busyId, results, decide };
}

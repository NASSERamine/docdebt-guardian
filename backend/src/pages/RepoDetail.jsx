import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { errorText, getJson, sendWithToken } from "../api.js";
import { StatTiles, StateNote, SyncButton, formatDate, timeAgo } from "../components.jsx";
import { useApi } from "../hooks.js";
import UsageChart from "../UsageChart.jsx";
import { ProposalHistory } from "./History.jsx";

function connectionLabel(detail) {
  if (detail.connected) return { cls: "connected", text: `connected${detail.connected_at ? ` since ${formatDate(detail.connected_at)}` : ""}` };
  if (detail.installed_before) return { cls: "disconnected", text: "disconnected (the App was removed from this repository)" };
  return { cls: "disconnected", text: "not connected (activity recorded before installation tracking)" };
}

const POLL_MS = 2000;

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function scanSummary(scan) {
  if (scan.status === "running") return "Scanning the open pull requests…";
  if (scan.status === "failed") return `Scan failed: ${scan.error}`;
  if (scan.prs === 0) return "No open pull request to scan.";
  const parts = [plural(scan.prs, "open PR") + " scanned", plural(scan.proposed, "new proposal")];
  if (scan.already_proposed) parts.push(`${scan.already_proposed} already waiting for a decision`);
  if (scan.already_decided) parts.push(`${scan.already_decided} already accepted or rejected`);
  if (scan.clean) parts.push(plural(scan.clean, "clean method"));
  if (scan.not_proposed) parts.push(`${scan.not_proposed} with debt but no verified fix`);
  if (scan.failed_prs) parts.push(`${plural(scan.failed_prs, "PR")} could not be read`);
  return parts.join(" · ");
}

// "Scan now": reviews every open PR of the repository right away, as if GitHub
// had sent a webhook for each one. Polls the scan until it finishes.
function ScanButton({ owner, repo, lastScan, enabled, onFinished }) {
  const [scan, setScan] = useState(lastScan);
  const [error, setError] = useState(null);
  useEffect(() => setScan(lastScan), [lastScan?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const running = scan?.status === "running";
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(async () => {
      try {
        const next = await getJson(`/api/scans/${scan.id}`);
        setScan(next);
        if (next.status !== "running") onFinished();
      } catch (e) {
        setError(`Lost track of the scan: ${e}`);
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [running, scan?.id, onFinished]);

  async function start() {
    setError(null);
    const { status, body } = await sendWithToken(
      "POST",
      `/api/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/scan`,
    );
    if (status === 202) setScan(body);
    else if (status !== 0) setError(errorText(body, status));
  }

  return (
    <div className="scan-box">
      <button
        type="button"
        className="primary-btn"
        disabled={!enabled || running}
        onClick={start}
        title={enabled ? "Review every open pull request now" : "Set DOCDEBT_DASHBOARD_TOKEN to scan from the dashboard"}
      >
        {running ? "Scanning…" : "Scan now"}
      </button>
      <span className={`scan-status ${scan?.status === "failed" || error ? "error" : ""}`}>
        {error ||
          (scan &&
            `${scanSummary(scan)}${scan.status !== "running" && scan.finished_at ? ` (${timeAgo(scan.finished_at)})` : ""}`) ||
          "Reviews every open pull request now, without waiting for a push."}
      </span>
    </div>
  );
}

export default function RepoDetail() {
  const { owner, repo } = useParams();
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((k) => k + 1), []);
  const { data, error, loading } = useApi(
    [`/api/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}?days=30`, "/api/dashboard-config"],
    { reloadKey },
  );

  if (error && error.endsWith("404")) {
    return (
      <>
        <p className="breadcrumb">
          <Link to="/repositories">Repositories</Link> / {owner}/{repo}
        </p>
        <p className="state-note">Nothing is known about {owner}/{repo} yet.</p>
      </>
    );
  }

  return (
    <>
      <p className="breadcrumb">
        <Link to="/repositories">Repositories</Link> / {owner}/{repo}
      </p>
      <StateNote error={!data && error} loading={!data && loading}>
        {data && (() => {
          const [detail, config] = data;
          const s = detail.stats;
          const conn = connectionLabel(detail);
          return (
            <>
              <div className="page-heading">
                <h1 className="page-title mono">
                  {owner}/{repo}
                </h1>
                <a href={detail.url} target="_blank" rel="noreferrer">
                  Open on GitHub ↗
                </a>
              </div>
              <p className="repo-connection">
                <span className={`dot ${conn.cls}`} /> {conn.text}
              </p>
              {!detail.connected && (
                <div className="field-help sync-hint">
                  <span>
                    Installed the App on this repository already? Its installation was not recorded: sync with GitHub
                    to connect it, then <strong>Scan now</strong> appears.
                  </span>
                  <SyncButton onSynced={reload} />
                </div>
              )}
              {detail.connected && (
                <ScanButton
                  owner={owner}
                  repo={repo}
                  lastScan={detail.last_scan}
                  enabled={config.decisions_enabled}
                  onFinished={reload}
                />
              )}
              <StatTiles
                tiles={[
                  { label: "proposals", value: s.proposals },
                  { label: "pending", value: s.pending, cls: "pending" },
                  { label: "accepted", value: s.accepted, cls: "accepted" },
                  { label: "rejected", value: s.rejected, cls: "rejected" },
                  ...(s.expired ? [{ label: "expired", value: s.expired }] : []),
                  { label: "LLM calls", value: s.llm_calls },
                  { label: "tokens", value: s.tokens.toLocaleString() },
                  { label: "cost", value: `$${s.cost_usd.toFixed(4)}` },
                  { label: "avg run", value: `${s.avg_latency_s.toFixed(1)} s` },
                ]}
              />
              <UsageChart series={detail.usage} />
              <h2 className="section-title">Proposals</h2>
            </>
          );
        })()}
      </StateNote>
      {data && <ProposalHistory fixedRepo={`${owner}/${repo}`} reloadKey={reloadKey} />}
    </>
  );
}

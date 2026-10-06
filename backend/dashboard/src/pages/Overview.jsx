import { Link } from "react-router-dom";
import { ConnectButton, ProposalList, StateNote, repoPath } from "../components.jsx";
import { useApi, useDecision } from "../hooks.js";
import UsageChart from "../UsageChart.jsx";

const REFRESH_MS = 15000;

function StatusLine({ summary }) {
  return (
    <div className="statusline">
      <span className="metric">
        <strong>{summary.repositories}</strong> repos
      </span>
      <span className="metric">
        <strong className="pending">{summary.pending}</strong> pending
      </span>
      <span className="metric">
        <strong className="accepted">{summary.accepted}</strong> accepted
      </span>
      <span className="metric">
        <strong>${summary.cost_usd.toFixed(2)}</strong> total cost
      </span>
    </div>
  );
}

function RepoList({ repos, connectUrl }) {
  if (repos.length === 0) {
    return (
      <div className="empty-state">
        <p>No repository connected yet. Install the App on a repository to start reviewing its documentation.</p>
        <ConnectButton connectUrl={connectUrl} />
      </div>
    );
  }
  return (
    <>
      <div className="panel-actions">
        <span />
        <ConnectButton connectUrl={connectUrl} />
      </div>
      <ul className="repo-list">
        {repos.map((r) => (
          <li key={`${r.owner}/${r.repo}`}>
            <Link className="repo-row" to={repoPath(r.owner, r.repo)}>
              <span
                className={`dot ${r.connected ? "connected" : "disconnected"}`}
                title={r.connected ? "connected" : "not connected"}
              />
              <span className="repo-name">
                {r.owner}/{r.repo}
              </span>
              {r.proposals !== undefined && (
                <span className="repo-stats">
                  {r.pending > 0 && <span className="pending">{r.pending} pending</span>}
                  {r.accepted > 0 && <span className="accepted">{r.accepted} accepted</span>}
                  {r.rejected > 0 && <span className="rejected">{r.rejected} rejected</span>}
                  {r.expired > 0 && <span title="The review run was gone when a decision was made">{r.expired} expired</span>}
                  {r.unknown > 0 && <span title="Resolved before this dashboard could record the decision">{r.unknown} unknown</span>}
                </span>
              )}
            </Link>
          </li>
        ))}
      </ul>
    </>
  );
}

export default function Overview() {
  const { data, error, loading, refresh } = useApi(
    ["/api/summary", "/api/repos", "/api/proposals?limit=10", "/api/usage?days=14", "/api/connect-url", "/api/dashboard-config"],
    { refreshMs: REFRESH_MS },
  );
  const decisions = useDecision(refresh);

  return (
    <StateNote error={!data && error} loading={!data && loading}>
      {data && (() => {
        const [summary, repos, proposals, usage, connect, config] = data;
        return (
          <>
            <StatusLine summary={summary} />
            <div className="columns">
              <div className="panel">
                <h2>Repositories</h2>
                <RepoList repos={repos} connectUrl={connect.url} />
              </div>
              <div className="panel">
                <div className="panel-heading">
                  <h2>Recent activity</h2>
                  <Link to="/proposals" className="see-all">
                    Full history →
                  </Link>
                </div>
                <ProposalList proposals={proposals} decisionsEnabled={config.decisions_enabled} decisions={decisions} />
              </div>
            </div>
            <UsageChart series={usage} />
          </>
        );
      })()}
    </StateNote>
  );
}

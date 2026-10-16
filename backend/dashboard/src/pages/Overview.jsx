import { Link } from "react-router-dom";
import { PrecisionCard, ProposalList, StateNote, repoPath } from "../components.jsx";
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

function AddRepositoryLink() {
  return (
    <Link className="connect-btn" to="/repositories">
      + Add a repository
    </Link>
  );
}

function RepoList({ repos }) {
  if (repos.length === 0) {
    return (
      <div className="empty-state">
        <p>No repository yet. Add one to start reviewing its documentation.</p>
        <AddRepositoryLink />
      </div>
    );
  }
  return (
    <>
      <div className="panel-actions">
        <span />
        <AddRepositoryLink />
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
    ["/api/summary", "/api/repos", "/api/proposals?limit=10", "/api/usage?days=14", "/api/dashboard-config", "/api/precision?days=30"],
    { refreshMs: REFRESH_MS },
  );
  const decisions = useDecision(refresh);

  return (
    <StateNote error={!data && error} loading={!data && loading}>
      {data && (() => {
        const [summary, repos, proposals, usage, config, precision] = data;
        return (
          <>
            <section className="intro">
              <p>
                <strong>DocDebt Guardian</strong> finds Java documentation that no longer matches its code — a renamed
                parameter, a missing <code>@return</code>, a description the code contradicts — in every pull request, and
                proposes a verified fix that a person accepts or rejects.
              </p>
              <Link to="/how-it-works" className="intro-link">
                How it works →
              </Link>
            </section>
            <StatusLine summary={summary} />
            <div className="columns">
              <div className="panel">
                <h2>Repositories</h2>
                <RepoList repos={repos} />
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
            <PrecisionCard precision={precision} />
            <UsageChart series={usage} />
          </>
        );
      })()}
    </StateNote>
  );
}

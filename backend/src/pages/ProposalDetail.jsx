import { Link, useParams } from "react-router-dom";
import { StateNote, formatDate, marker, repoPath } from "../components.jsx";
import { useApi, useDecision } from "../hooks.js";

const ACTION_LABELS = { add: "added", remove: "removed", replace: "rewritten" };

function DiffBlock({ lines }) {
  return (
    <pre className="code-block diff">
      {lines.map((line, i) => {
        const cls = line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : line.startsWith("@@") ? "hunk" : "";
        return (
          <div key={i} className={`diff-line ${cls}`}>
            {line || " "}
          </div>
        );
      })}
    </pre>
  );
}

function Finding({ finding }) {
  return (
    <li className="finding">
      <span className={`rule-chip ${finding.kind}`}>{finding.rule}</span>
      <span>{finding.message}</span>
      {finding.quoted_doc && (
        <div className="quotes">
          <div>
            <div className="quote-label">The documentation says</div>
            <pre className="code-block small">{finding.quoted_doc}</pre>
          </div>
          <div>
            <div className="quote-label">The code does</div>
            <pre className="code-block small">{finding.quoted_code}</pre>
          </div>
        </div>
      )}
    </li>
  );
}

function DecisionBar({ proposal, decisionsEnabled, decisions }) {
  const m = marker(proposal);
  const result = decisions.results[proposal.short_id];
  if (m.label !== "pending") {
    return (
      <div className={`decision-bar done ${m.cls}`}>
        <span>
          {m.icon} This proposal is <strong>{m.label}</strong>.
        </span>
        {result && <span className={`feed-result ${result.ok ? "ok" : "error"}`}>{result.message}</span>}
      </div>
    );
  }
  if (!decisionsEnabled) {
    return (
      <div className="decision-bar">
        <span>
          Reply <span className="command">/docdebt accept {proposal.short_id}</span> or{" "}
          <span className="command">/docdebt reject {proposal.short_id}</span> on the pull request (set
          DOCDEBT_DASHBOARD_TOKEN to decide from here).
        </span>
      </div>
    );
  }
  const busy = decisions.busyId !== null;
  return (
    <div className="decision-bar">
      <span>Commit this fix to the pull request branch?</span>
      <span className="decision-buttons">
        <button type="button" className="decide-btn accept big" disabled={busy} onClick={() => decisions.decide(proposal, "accept")}>
          {busy ? "Working…" : "Accept"}
        </button>
        <button type="button" className="decide-btn reject big" disabled={busy} onClick={() => decisions.decide(proposal, "reject")}>
          Reject
        </button>
      </span>
      {result && <span className={`feed-result ${result.ok ? "ok" : "error"}`}>{result.message}</span>}
    </div>
  );
}

export default function ProposalDetail() {
  const { shortId } = useParams();
  const { data, error, loading, refresh } = useApi([`/api/proposals/${encodeURIComponent(shortId)}`, "/api/dashboard-config"]);
  const decisions = useDecision(refresh);

  if (error && error.endsWith("404")) {
    return (
      <p className="state-note">
        No proposal with id {shortId}. <Link to="/proposals">Back to the history</Link>
      </p>
    );
  }

  return (
    <StateNote error={!data && error} loading={!data && loading}>
      {data && (() => {
        const [p, config] = data;
        const d = p.details;
        return (
          <>
            <p className="breadcrumb">
              <Link to="/proposals">History</Link> / <Link to={repoPath(p.owner, p.repo)}>{p.owner}/{p.repo}</Link> /{" "}
              <a href={p.pr_url} target="_blank" rel="noreferrer">
                #{p.pr_number} ↗
              </a>
            </p>
            <div className="page-heading">
              <h1 className="page-title mono">{p.method_label}</h1>
              <span className="mono id-tag">{p.short_id}</span>
            </div>
            <p className="repo-connection">
              <span className="mono">
                {p.file_path}
                {d ? `:${d.method.start_line}` : ""}
              </span>
              · proposed {formatDate(p.created_at)}
            </p>

            <DecisionBar proposal={p} decisionsEnabled={config.decisions_enabled} decisions={decisions} />

            {!d ? (
              <p className="notice">
                The details of this proposal are no longer available: its review run was not found (the graph database
                was reset or moved). Push to the pull request again to get a fresh review.
              </p>
            ) : (
              <>
                <section className="detail-section">
                  <h2 className="section-title">Debt found</h2>
                  <ul className="findings">
                    {d.debt.map((f, i) => (
                      <Finding key={i} finding={f} />
                    ))}
                  </ul>
                  {d.warnings.length > 0 && (
                    <details className="warnings">
                      <summary>{d.warnings.length} hygiene warning(s), not counted as debt</summary>
                      <ul className="findings">
                        {d.warnings.map((f, i) => (
                          <Finding key={i} finding={f} />
                        ))}
                      </ul>
                    </details>
                  )}
                  {d.detection.model && (
                    <p className="field-help">
                      Semantic check by {d.detection.model}
                      {d.detection.confidence != null && `, confidence ${Math.round(d.detection.confidence * 100)}%`}.
                    </p>
                  )}
                </section>

                <section className="detail-section">
                  <h2 className="section-title">Proposed fix</h2>
                  {d.diff.length > 0 ? <DiffBlock lines={d.diff} /> : <p className="state-note">No change proposed.</p>}
                  {d.repair.changes.length > 0 && (
                    <table className="changes">
                      <thead>
                        <tr>
                          <th>Field</th>
                          <th>Change</th>
                          <th>Before</th>
                          <th>After</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.repair.changes.map((c, i) => (
                          <tr key={i}>
                            <td className="mono">{c.field}</td>
                            <td>{ACTION_LABELS[c.action] || c.action}</td>
                            <td className="muted">{c.before ?? "—"}</td>
                            <td>{c.after ?? "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  <p className="field-help">
                    Written by {d.repair.model || "the model"} in {d.repair.attempts} attempt(s), then {d.repair.status} by the
                    Verifier: no structural error left, every other line of the Javadoc unchanged.
                  </p>
                </section>

                <section className="detail-section">
                  <h2 className="section-title">Current code in the pull request</h2>
                  <p className="field-help">
                    The method as it is now on the PR branch, before the fix: <span className="mono">{d.method.signature}</span>
                  </p>
                  <pre className="code-block">{d.method.source}</pre>
                </section>
              </>
            )}
          </>
        );
      })()}
    </StateNote>
  );
}

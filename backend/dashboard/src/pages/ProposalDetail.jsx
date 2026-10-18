import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { errorText, sendWithToken } from "../api.js";
import { ErrorNote, MECHANISM_LABELS, RejectButton, StateNote, formatDate, marker, money, repoPath } from "../components.jsx";
import { useApi, useDecision } from "../hooks.js";
import { wordDiff } from "../wordDiff.js";

const ACTION_LABELS = { add: "added", remove: "removed", replace: "rewritten" };

const SOURCE_LABELS = {
  siblings: "same file",
  similar: "similar method",
  validated: "accepted fix",
  tests: "test",
  readme: "README",
};

// What the post-validation (layer 4 of the hybrid mechanism) did with each
// field the model answered.
const OUTCOMES = {
  accepted: { cls: "ok", text: "accepted" },
  accepted_after_retry: { cls: "ok", text: "accepted on retry" },
  kept: { cls: "", text: "kept as is (the model judged it correct)" },
  kept_no_verdict: { cls: "", text: "kept (no verdict)" },
  write_failed: { cls: "error", text: "not written" },
  write_flagged: { cls: "warn", text: "written, names an identifier absent from the code" },
  reverted_format: { cls: "warn", text: "reverted: broken format" },
  reverted_hallucination: { cls: "warn", text: "reverted: invented identifier" },
  reverted_cosmetic: { cls: "", text: "reverted: cosmetic change only" },
  reverted_deletion_only: { cls: "warn", text: "reverted: only removed text" },
  ignored_extra_key: { cls: "", text: "ignored: field not asked" },
};

function Badge({ children, cls = "", title }) {
  return (
    <span className={`badge ${cls}`} title={title}>
      {children}
    </span>
  );
}

function ContextBlock({ items, title, help }) {
  if (!items?.length) return null;
  return (
    <details className="rag-context">
      <summary>
        {title} <span className="muted">· {items.length} excerpt(s)</span>
      </summary>
      <p className="field-help">{help}</p>
      <ul className="rag-items">
        {items.map((item, i) => (
          <li key={i}>
            <div className="rag-item-head">
              <Badge cls={`source ${item.source}`}>{SOURCE_LABELS[item.source] || item.source}</Badge>
              <span className="mono">{item.label}</span>
              <span className="muted">
                #{item.rank}
                {item.score != null && ` · RRF ${item.score}`}
                {item.lexical_rank != null && ` · BM25 #${item.lexical_rank}`}
                {item.dense_rank != null && ` · vectors #${item.dense_rank}`}
              </span>
            </div>
            <pre className="code-block small">{item.text}</pre>
          </li>
        ))}
      </ul>
    </details>
  );
}

function FieldLog({ log }) {
  if (!log?.length) return null;
  return (
    <details className="warnings">
      <summary>Post-validation: what was kept from the model&apos;s answer ({log.length} field(s))</summary>
      <ul className="mini-table field-log">
        {log.map((e, i) => {
          const o = OUTCOMES[e.outcome] || { cls: "", text: e.outcome };
          return (
            <li key={i}>
              <span className="mono">{e.key}</span>
              <span className={`feed-result ${o.cls}`}>
                {o.text}
                {e.detail ? ` (${e.detail})` : ""}
              </span>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

const norm = (text) => (text || "").replace(/\s+/g, " ").trim();

// The lines of the method (Javadoc and code) that an explanation points at (U9).
function highlightedLines(source, items) {
  const quotes = items.flatMap((i) => [i.claim, i.code]).map(norm).filter((q) => q.length >= 3);
  if (!quotes.length) return new Set();
  const lines = source.split("\n");
  const marked = new Set();
  lines.forEach((line, n) => {
    const l = norm(line.replace(/^\s*\*\s?/, ""));
    if (l.length >= 3 && quotes.some((q) => q.includes(l) || l.includes(q))) marked.add(n);
  });
  return marked;
}

function SourceBlock({ source, marked }) {
  return (
    <pre className="code-block">
      {source.split("\n").map((line, i) => (
        <div key={i} className={`code-line${marked.has(i) ? " highlighted" : ""}`}>
          {line || " "}
        </div>
      ))}
    </pre>
  );
}

function Explanation({ items, detection }) {
  if (!items?.length) return null;
  return (
    <ul className="findings">
      {items.map((item, i) => (
        <li className="finding" key={i}>
          <span className={`rule-chip ${item.source === "rule" ? "structural" : "semantic"}`}>
            {item.source === "rule" ? `rule ${item.rule}` : "model"}
          </span>
          {item.field && <span className="mono field-ref">{item.field}</span>}
          <span>{item.message}</span>
          {item.source === "model" && (item.claim || item.code) && (
            <div className="quotes">
              {item.claim && (
                <div>
                  <div className="quote-label">The documentation says</div>
                  <pre className="code-block small highlighted">{item.claim}</pre>
                </div>
              )}
              {item.code && (
                <div>
                  <div className="quote-label">The code does</div>
                  <pre className="code-block small highlighted">{item.code}</pre>
                </div>
              )}
            </div>
          )}
          {item.source === "model" && !item.verified && (
            <span className="field-help"> (no exact quote found in the method: nothing highlighted)</span>
          )}
        </li>
      ))}
      {detection?.model && (
        <li className="field-help">
          Judged by <Badge cls="model">{detection.model}</Badge>
          {detection.screened_on && ` · prompts screened on ${detection.screened_on} (same model family)`}
          {detection.screened === false && " · prompt not screened in the research for this model"}. The exact rules
          and the model look at the whole Javadoc: it is debt if a rule fails or the model says so.
        </li>
      )}
    </ul>
  );
}

function WordDiffText({ before, after }) {
  return (
    <pre className="code-block word-diff">
      {wordDiff(before, after).map((part, i) => (
        <span key={i} className={part.kind === "same" ? "" : `wd-${part.kind}`}>
          {part.text}
        </span>
      ))}
    </pre>
  );
}

function Column({ title, help, children, actions, chosen }) {
  return (
    <div className={`compare-col${chosen ? " chosen" : ""}`}>
      <div className="compare-head">
        <strong>{title}</strong>
        {help && <span className="field-help">{help}</span>}
      </div>
      {children}
      {actions && <div className="compare-actions">{actions}</div>}
    </div>
  );
}

// Three columns: the current Javadoc, the field-by-field fix, the regeneration (U7).
function Compare({ proposal, details, decisionsEnabled, decisions, onRegenerated, actionsRef }) {
  const current = details.indent + (details.method.javadoc || "");
  const indent = (text) => (text ? details.indent + text : "");
  const replacement = indent(details.repair.repaired_doc);
  const regen = proposal.regeneration;
  const regenerated = indent(regen.doc);
  const pending = marker(proposal).label === "pending";
  const busy = decisions.busyId !== null;
  const [regenBusy, setRegenBusy] = useState(false);
  const [regenError, setRegenError] = useState(null);
  const [editing, setEditing] = useState(false);
  const [edited, setEdited] = useState(details.repair.repaired_doc || "");
  const accepted = proposal.accepted_mechanism;

  async function regenerate() {
    setRegenBusy(true);
    setRegenError(null);
    try {
      const { status, body } = await sendWithToken("POST", `/api/proposals/${proposal.short_id}/regenerate`);
      if (status === 200 && body.regen_status !== "verified") setRegenError(body.message);
      else if (status !== 200 && status !== 0) setRegenError(errorText(body, status));
      onRegenerated();
    } catch (e) {
      setRegenError(`Could not reach the service: ${e}`);
    } finally {
      setRegenBusy(false);
    }
  }

  const accept = (mechanism, extra = {}) => decisions.decide(proposal, "accept", null, { mechanism, ...extra });
  const canAct = pending && decisionsEnabled;
  // Keyboard shortcuts of the "To review" mode (A, G, E; R opens the reject menu).
  useEffect(() => {
    if (!actionsRef) return;
    actionsRef.current = canAct
      ? {
          accept: () => accept(regen.status === "verified" && accepted === "regeneration" ? "regeneration" : "replacement"),
          regenerate: () => regen.status !== "verified" && !regenBusy && regenerate(),
          edit: () => setEditing(true),
        }
      : null;
  });
  return (
    <>
      <div className="compare">
        <Column title="Current Javadoc" help="as it is on the PR branch">
          <pre className="code-block">{current}</pre>
        </Column>
        <Column
          title="Field-by-field fix"
          help="replacement: only the faulty fields change"
          chosen={accepted === "replacement"}
          actions={
            canAct && (
              <>
                <button type="button" className="decide-btn accept" disabled={busy} onClick={() => accept("replacement")}>
                  Accept this
                </button>
                <button type="button" className="page-btn" disabled={busy} onClick={() => setEditing(true)} title="Shortcut: E">
                  Edit, then accept
                </button>
              </>
            )
          }
        >
          <WordDiffText before={current} after={replacement} />
        </Column>
        <Column
          title="Regeneration"
          help={regen.screened === false ? "whole Javadoc rewritten (prompt not screened for this model)" : "whole Javadoc rewritten, then only the planned fields kept"}
          chosen={accepted === "regeneration"}
          actions={
            canAct &&
            (regen.status === "verified" ? (
              <button type="button" className="decide-btn accept" disabled={busy} onClick={() => accept("regeneration")}>
                Accept this
              </button>
            ) : (
              <button type="button" className="page-btn" disabled={regenBusy || busy} onClick={regenerate} title="Shortcut: G">
                {regenBusy ? "Regenerating…" : "Regenerate"}
              </button>
            ))
          }
        >
          {regen.status === "verified" ? (
            <WordDiffText before={current} after={regenerated} />
          ) : (
            <p className="state-note">
              {regen.status?.startsWith("failed")
                ? `No verified regeneration (${regen.status.replace("failed: ", "")}).`
                : "Not computed yet. One more model call"}
              {!regen.status && proposal.cost_usd != null && ` — this method's review cost ${money(proposal.cost_usd)}`}
              {!regen.status && "."}
            </p>
          )}
          {regenError && <ErrorNote text={regenError} />}
        </Column>
      </div>
      {editing && canAct && (
        <div className="edit-box">
          <label htmlFor="edited-doc" className="field-label">
            Your version of the Javadoc — checked by the exact rules before it is committed
          </label>
          <textarea id="edited-doc" className="code-block edit-area" rows={Math.max(6, edited.split("\n").length + 1)} value={edited} onChange={(e) => setEdited(e.target.value)} />
          <div className="form-actions">
            <button
              type="button"
              className="decide-btn accept"
              disabled={busy}
              onClick={async () => {
                if (await accept("edited", { editedDoc: edited })) setEditing(false);
              }}
            >
              Accept my version
            </button>
            <button type="button" className="page-btn" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
}

function DecisionBar({ proposal, decisionsEnabled, decisions }) {
  const m = marker(proposal);
  const result = decisions.results[proposal.short_id];
  const resultNote = result && (result.ok ? <span className="feed-result ok">{result.message}</span> : <ErrorNote text={result.message} />);
  if (m.label !== "pending") {
    return (
      <div className={`decision-bar done ${m.cls}`}>
        <span>
          {m.icon} This proposal is <strong>{m.label}</strong>
          {proposal.accepted_mechanism && ` (${MECHANISM_LABELS[proposal.accepted_mechanism] || proposal.accepted_mechanism})`}.
          {proposal.commit_url && (
            <>
              {" "}
              <a href={proposal.commit_url} target="_blank" rel="noreferrer">
                See the commit {proposal.commit_sha.slice(0, 7)} ↗
              </a>
            </>
          )}
        </span>
        {resultNote}
      </div>
    );
  }
  if (!decisionsEnabled) {
    return (
      <div className="decision-bar">
        <span>
          Reply <span className="command">/docdebt accept {proposal.short_id}</span>,{" "}
          <span className="command">/docdebt regenerate {proposal.short_id}</span> or{" "}
          <span className="command">/docdebt reject {proposal.short_id} false-positive</span> on the pull request (sign
          in, or set DOCDEBT_DASHBOARD_TOKEN, to decide from here).
        </span>
      </div>
    );
  }
  const busy = decisions.busyId !== null;
  return (
    <div className="decision-bar">
      <span>Choose the version to commit below, or:</span>
      <span className="decision-buttons">
        <RejectButton proposal={proposal} busy={busy} onDecide={decisions.decide} big />
      </span>
      {busy && <span className="field-help">Working…</span>}
      {resultNote}
    </div>
  );
}

// One proposal, everything needed to decide. Used by its own page and by the
// "To review" mode, which passes `onDecided` to move to the next one.
export function ProposalView({ shortId, onDecided, actionsRef }) {
  const { data, error, loading, refresh } = useApi([`/api/proposals/${encodeURIComponent(shortId)}`, "/api/dashboard-config"]);
  const decisions = useDecision(() => {
    refresh();
    onDecided?.();
  });

  if (error && error.endsWith("404")) {
    return (
      <p className="state-note">
        No proposal with id {shortId}. <Link to="/proposals">Back to the history</Link>
      </p>
    );
  }

  return (
    <StateNote error={!data && error} loading={!data && loading}>
      {data &&
        (() => {
          const [p, config] = data;
          const d = p.details;
          const explanation = p.explanation?.length ? p.explanation : d?.explanation || [];
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
                {" · "}
                <a href={p.file_url} target="_blank" rel="noreferrer" title="The changed files of the pull request">
                  open in the pull request ↗
                </a>
                {" · "}proposed {formatDate(p.created_at)}
                {p.cost_usd != null && ` · cost ${money(p.cost_usd)} (this PR: ${money(p.pr_cost_usd)})`}
              </p>

              <DecisionBar proposal={p} decisionsEnabled={config.decisions_enabled} decisions={decisions} />

              {!d ? (
                <p className="notice">
                  The details of this proposal are no longer available: its review run was not found (the graph
                  database was reset or moved). Push to the pull request again to get a fresh review.
                </p>
              ) : (
                <>
                  <section className="detail-section">
                    <h2 className="section-title">Why this is documentation debt</h2>
                    <Explanation items={explanation} detection={d.detection} />
                    {d.warnings.length > 0 && (
                      <details className="warnings">
                        <summary>{d.warnings.length} hygiene warning(s), not counted as debt</summary>
                        <ul className="findings">
                          {d.warnings.map((f, i) => (
                            <li className="finding" key={i}>
                              <span className="rule-chip">{f.rule}</span> <span>{f.message}</span>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </section>

                  <section className="detail-section">
                    <h2 className="section-title">Choose what to commit</h2>
                    <Compare
                      proposal={p}
                      details={d}
                      decisionsEnabled={config.decisions_enabled}
                      decisions={decisions}
                      onRegenerated={refresh}
                      actionsRef={actionsRef}
                    />
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
                      Fix written by <Badge cls="model">{d.repair.model || "the model"}</Badge> field by field, then
                      checked: no structural error left, every untouched line of the Javadoc byte-identical.
                    </p>
                    <FieldLog log={d.repair.field_log} />
                    <ContextBlock
                      items={d.rag_context}
                      title="Project context given to the model (RAG)"
                      help="Always retrieved: up to three documented methods of the same file, then the four best excerpts of the project's knowledge base by hybrid search (BM25 + vectors, fused by rank). Given as reference only; methods with the same name are never included."
                    />
                  </section>

                  <section className="detail-section">
                    <h2 className="section-title">Current code in the pull request</h2>
                    <p className="field-help">
                      The method as it is now on the PR branch, lines the explanation points at highlighted:{" "}
                      <span className="mono">{d.method.signature}</span>
                    </p>
                    <SourceBlock source={d.method.source} marked={highlightedLines(d.method.source, explanation)} />
                  </section>
                </>
              )}
            </>
          );
        })()}
    </StateNote>
  );
}

export default function ProposalDetail() {
  const { shortId } = useParams();
  return <ProposalView shortId={shortId} />;
}

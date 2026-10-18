import { Link } from "react-router-dom";
import { timeAgo } from "./components.jsx";
import { useApi } from "./hooks.js";

// Always-visible status of the service (U2): the model, the knowledge base,
// the webhooks. Each item links to the page that fixes it.

const REFRESH_MS = 30000;

function Item({ state, to, children, title }) {
  const content = (
    <>
      <span className={`status-dot ${state}`} aria-hidden="true" />
      {children}
    </>
  );
  return to ? (
    <Link className="status-item" to={to} title={title}>
      {content}
    </Link>
  ) : (
    <span className="status-item" title={title}>
      {content}
    </span>
  );
}

export function useStatus() {
  return useApi(["/api/status"], { refreshMs: REFRESH_MS });
}

export default function StatusBanner() {
  const { data } = useStatus();
  const s = data?.[0];
  if (!s?.signed_in) return null;
  const model = s.model;
  const kb = s.knowledge;
  const modelState = !model.configured ? "bad" : model.ok === false ? "bad" : "ok";
  const kbState = kb.repositories === 0 ? "idle" : kb.not_indexed.length ? "warn" : "ok";
  const firstUnindexed = kb.not_indexed[0];
  return (
    <div className="status-banner" role="status">
      <Item
        state={modelState}
        to="/model"
        title={model.error || "The model DocDebt calls for detection and repair"}
      >
        {!model.configured ? "No model configured" : model.ok === false ? `Model ${model.model}: last call failed` : `Model ${model.model}`}
      </Item>
      <Item
        state={kbState}
        to={firstUnindexed ? `/repos/${firstUnindexed}` : "/repositories"}
        title={kb.embeddings_error ? `Embeddings unavailable (${kb.embeddings_error}): BM25 only` : `Vectors: ${kb.embeddings}, index: ${kb.vector_index}`}
      >
        {kb.repositories === 0
          ? "No repository yet"
          : kb.not_indexed.length
            ? `Knowledge base: ${kb.indexed}/${kb.repositories} indexed`
            : "Knowledge base up to date"}
        {kb.embeddings_error && " · BM25 only"}
      </Item>
      <Item state={s.webhooks.last_at ? "ok" : "idle"} title="The last event GitHub sent to this service">
        {s.webhooks.last_at ? `Last webhook ${timeAgo(s.webhooks.last_at)} (${s.webhooks.last_event})` : "No webhook received yet"}
      </Item>
    </div>
  );
}

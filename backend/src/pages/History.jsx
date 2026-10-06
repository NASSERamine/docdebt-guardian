import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ProposalList, StateNote } from "../components.jsx";
import { useApi, useDecision } from "../hooks.js";

const PAGE_SIZE = 25;

const STATUSES = [
  ["", "Any status"],
  ["pending", "Pending"],
  ["accept", "Accepted"],
  ["reject", "Rejected"],
  ["expired", "Expired"],
  ["unknown", "Decision unknown"],
];

function Pagination({ page, pages, total, onPage }) {
  if (total === 0) return null;
  const first = (page - 1) * PAGE_SIZE + 1;
  const last = Math.min(page * PAGE_SIZE, total);
  return (
    <div className="pagination">
      <button type="button" className="page-btn" disabled={page <= 1} onClick={() => onPage(page - 1)}>
        ← Previous
      </button>
      <span className="page-info">
        {first}–{last} of {total} · page {page} of {pages}
      </span>
      <button type="button" className="page-btn" disabled={page >= pages} onClick={() => onPage(page + 1)}>
        Next →
      </button>
    </div>
  );
}

// Filters live in the URL (?q=&repo=&status=&since=&until=&page=), so a
// filtered view can be bookmarked or shared. With `fixedRepo` ("owner/repo")
// the repository filter is locked: that is a repository page's history.
export function ProposalHistory({ fixedRepo = null, reloadKey = 0 }) {
  const [params, setParams] = useSearchParams();
  const filters = {
    q: params.get("q") || "",
    repo: fixedRepo || params.get("repo") || "",
    status: params.get("status") || "",
    since: params.get("since") || "",
    until: params.get("until") || "",
    page: Math.max(1, Number(params.get("page")) || 1),
  };

  // The search box updates the URL 300 ms after typing stops.
  const [text, setText] = useState(filters.q);
  useEffect(() => setText(filters.q), [filters.q]);
  useEffect(() => {
    if (text === filters.q) return undefined;
    const id = setTimeout(() => update({ q: text }), 300);
    return () => clearTimeout(id);
  }, [text, filters.q]);

  function update(changes, { keepPage = false } = {}) {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(changes)) {
      if (value === "" || value === null || value === undefined) next.delete(key);
      else next.set(key, String(value));
    }
    if (!keepPage) next.delete("page");
    setParams(next, { replace: true });
  }

  const query = new URLSearchParams({ page: String(filters.page), page_size: String(PAGE_SIZE) });
  for (const key of ["q", "repo", "status", "since", "until"]) if (filters[key]) query.set(key, filters[key]);

  const paths = [`/api/proposals/search?${query}`, "/api/dashboard-config"];
  if (!fixedRepo) paths.push("/api/repos");
  const { data, error, loading, refresh } = useApi(paths, { reloadKey });
  const decisions = useDecision(refresh);

  const filtered = filters.q || filters.status || filters.since || filters.until || (!fixedRepo && filters.repo);

  return (
    <>
      <form className="filters" onSubmit={(e) => { e.preventDefault(); update({ q: text }); }}>
        <input
          type="search"
          className="filter-input grow"
          placeholder="Search method, file or id…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          aria-label="Search"
        />
        {!fixedRepo && (
          <select
            className="filter-input"
            value={filters.repo}
            onChange={(e) => update({ repo: e.target.value })}
            aria-label="Repository"
          >
            <option value="">All repositories</option>
            {(data?.[2] || []).map((r) => (
              <option key={`${r.owner}/${r.repo}`} value={`${r.owner}/${r.repo}`}>
                {r.owner}/{r.repo}
              </option>
            ))}
          </select>
        )}
        <select
          className="filter-input"
          value={filters.status}
          onChange={(e) => update({ status: e.target.value })}
          aria-label="Status"
        >
          {STATUSES.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
        <label className="filter-date">
          From
          <input type="date" className="filter-input" value={filters.since} max={filters.until || undefined} onChange={(e) => update({ since: e.target.value })} />
        </label>
        <label className="filter-date">
          To
          <input type="date" className="filter-input" value={filters.until} min={filters.since || undefined} onChange={(e) => update({ until: e.target.value })} />
        </label>
        {filtered && (
          <button
            type="button"
            className="page-btn"
            onClick={() => {
              setText("");
              update({ q: "", status: "", since: "", until: "", ...(fixedRepo ? {} : { repo: "" }) });
            }}
          >
            Clear
          </button>
        )}
      </form>

      <StateNote error={!data && error} loading={!data && loading}>
        {data && (
          <>
            <ProposalList
              proposals={data[0].items}
              decisionsEnabled={data[1].decisions_enabled}
              decisions={decisions}
              showRepo={!fixedRepo}
              absoluteDates
            />
            <Pagination
              page={data[0].page}
              pages={data[0].pages}
              total={data[0].total}
              onPage={(page) => update({ page }, { keepPage: true })}
            />
          </>
        )}
      </StateNote>
    </>
  );
}

export default function History() {
  return (
    <>
      <h1 className="page-title">History</h1>
      <ProposalHistory />
    </>
  );
}

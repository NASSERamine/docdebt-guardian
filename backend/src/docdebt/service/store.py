"""Maps the short ids posted in PR comments back to a graph thread.

A person replies ``/docdebt accept 7f3a2c1d`` to a proposal comment; GitHub
gives back only the comment's own id and text, with no link to which graph
thread it was about. This tiny SQLite table is that missing link — separate
from the LangGraph checkpointer (``docdebt.graph.build``), which only knows
about thread ids, never about PR comments.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_requests (
    short_id     TEXT PRIMARY KEY,
    thread_id    TEXT NOT NULL,
    owner        TEXT NOT NULL,
    repo         TEXT NOT NULL,
    pr_number    INTEGER NOT NULL,
    file_path    TEXT NOT NULL,
    method_label TEXT NOT NULL,
    head_sha     TEXT NOT NULL,
    installation_id TEXT,
    resolved     INTEGER NOT NULL DEFAULT 0,
    decision     TEXT,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_requests_pr
    ON review_requests (owner, repo, pr_number);

CREATE TABLE IF NOT EXISTS llm_usage (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    owner          TEXT NOT NULL,
    repo           TEXT NOT NULL,
    pr_number      INTEGER NOT NULL,
    kind           TEXT NOT NULL,   -- "review" | "command"
    calls          INTEGER NOT NULL,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    cost_usd       REAL NOT NULL,
    latency_s      REAL NOT NULL,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_repo ON llm_usage (owner, repo);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage (created_at);

CREATE TABLE IF NOT EXISTS scans (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    owner             TEXT NOT NULL,
    repo              TEXT NOT NULL,
    status            TEXT NOT NULL,   -- "running" | "done" | "failed"
    started_at        REAL NOT NULL,
    finished_at       REAL,
    prs               INTEGER NOT NULL DEFAULT 0,
    proposed          INTEGER NOT NULL DEFAULT 0,
    already_proposed  INTEGER NOT NULL DEFAULT 0,
    already_decided   INTEGER NOT NULL DEFAULT 0,
    clean             INTEGER NOT NULL DEFAULT 0,
    not_proposed      INTEGER NOT NULL DEFAULT 0,
    failed_prs        INTEGER NOT NULL DEFAULT 0,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_repo ON scans (owner, repo, started_at);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,   -- JSON
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS installations (
    owner            TEXT NOT NULL,
    repo             TEXT NOT NULL,
    installation_id  TEXT NOT NULL,
    active           INTEGER NOT NULL DEFAULT 1,
    connected_at     REAL NOT NULL,
    PRIMARY KEY (owner, repo)
);
"""


_STATUS_CLAUSES = {
    "pending": "resolved = 0",
    "accept": "decision = 'accept'",
    "reject": "decision = 'reject'",
    "expired": "decision = 'expired'",
    "unknown": "(resolved = 1 AND decision IS NULL)",
}


@dataclass(frozen=True)
class ReviewRequest:
    short_id: str
    thread_id: str
    owner: str
    repo: str
    pr_number: int
    file_path: str
    method_label: str
    head_sha: str
    resolved: bool
    decision: str | None
    created_at: float
    installation_id: str | None = None


class ReviewStore:
    """Thread-safe-enough for this service's single-process use: one short-lived
    SQLite connection per call, matching the pattern already used for the
    LangGraph checkpointer elsewhere in this codebase."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after the first release of this schema.
        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a
        database created before the dashboard (T5) needs this to gain the
        ``decision`` column without losing its existing rows (and, likewise,
        the ``installation_id`` column added for dashboard decisions)."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(review_requests)")}
        if "decision" not in columns:
            conn.execute("ALTER TABLE review_requests ADD COLUMN decision TEXT")
        if "installation_id" not in columns:
            conn.execute("ALTER TABLE review_requests ADD COLUMN installation_id TEXT")
        scan_columns = {row["name"] for row in conn.execute("PRAGMA table_info(scans)")}
        if "already_decided" not in scan_columns:
            conn.execute("ALTER TABLE scans ADD COLUMN already_decided INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(
        self,
        *,
        thread_id: str,
        owner: str,
        repo: str,
        pr_number: int,
        file_path: str,
        method_label: str,
        head_sha: str,
        installation_id: str | None = None,
    ) -> str:
        """Register a pending review request and return its short id.

        ``installation_id`` is what lets the dashboard act on the proposal
        later: a PR comment arrives with its own installation id, a click on
        the dashboard does not.
        """
        short_id = uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO review_requests "
                "(short_id, thread_id, owner, repo, pr_number, file_path, method_label, head_sha, "
                " installation_id, resolved, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    short_id,
                    thread_id,
                    owner,
                    repo,
                    pr_number,
                    file_path,
                    method_label,
                    head_sha,
                    installation_id,
                    time.time(),
                ),
            )
        return short_id

    def get(self, short_id: str) -> ReviewRequest | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM review_requests WHERE short_id = ?", (short_id,)).fetchone()
        return self._to_request(row) if row else None

    def mark_resolved(self, short_id: str, decision: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE review_requests SET resolved = 1, decision = ? WHERE short_id = ?", (decision, short_id)
            )

    def pending_for_thread(self, thread_id: str) -> ReviewRequest | None:
        """The unresolved proposal already posted for this graph thread, if any:
        re-reviewing a PR must not post the same proposal a second time."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_requests WHERE thread_id = ? AND resolved = 0 "
                "ORDER BY created_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        return self._to_request(row) if row else None

    def pending_for_pr(self, owner: str, repo: str, pr_number: int) -> list[ReviewRequest]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_requests WHERE owner = ? AND repo = ? AND pr_number = ? AND resolved = 0",
                (owner, repo, pr_number),
            ).fetchall()
        return [self._to_request(r) for r in rows]

    # -- installations (connected repositories) --------------------------------

    def connect_repository(self, *, owner: str, repo: str, installation_id: str) -> None:
        """Record that the App is now installed on ``owner/repo``. Called from
        the ``installation_repositories`` (and ``installation``) webhook, so the
        dashboard can list connected repositories even before any PR was ever
        opened on them."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO installations (owner, repo, installation_id, active, connected_at) "
                "VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(owner, repo) DO UPDATE SET installation_id = excluded.installation_id, active = 1",
                (owner, repo, installation_id, time.time()),
            )

    def disconnect_repository(self, *, owner: str, repo: str) -> None:
        """Record that the App was removed from ``owner/repo`` (uninstalled, or
        that one repository was deselected from the installation)."""
        with self._connect() as conn:
            conn.execute("UPDATE installations SET active = 0 WHERE owner = ? AND repo = ?", (owner, repo))

    def installation_for(self, owner: str, repo: str) -> str | None:
        """The installation id recorded for ``owner/repo`` by an installation
        webhook, active or not; ``None`` if none was ever received."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT installation_id FROM installations WHERE owner = ? AND repo = ?", (owner, repo)
            ).fetchone()
        return row["installation_id"] if row else None

    def installations(self) -> list[dict]:
        """Every repository the App was ever installed on, with ``active``
        telling whether it still is (``False`` once uninstalled or deselected)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT owner, repo, installation_id, active, connected_at FROM installations "
                "ORDER BY active DESC, connected_at DESC"
            ).fetchall()
        return [{**dict(row), "active": bool(row["active"])} for row in rows]

    def connected_repositories(self) -> list[dict]:
        """Every repository the App is currently (actively) installed on."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT owner, repo, installation_id, connected_at FROM installations "
                "WHERE active = 1 ORDER BY connected_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    # -- usage logging and dashboard queries -----------------------------------

    def log_usage(
        self,
        *,
        owner: str,
        repo: str,
        pr_number: int,
        kind: str,
        calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_s: float,
    ) -> None:
        """Record one graph run's LLM usage (``kind`` is "review" for a
        pull_request event, "command" for an accept/reject). Called once per
        webhook handled, with the *delta* of a shared :class:`UsageTracker`
        (see ``docdebt.service.review_flow.ReviewFlow``), not a running total."""
        if calls == 0:
            return  # nothing was actually called (e.g. structural-only, no debt)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO llm_usage (owner, repo, pr_number, kind, calls, input_tokens, "
                "output_tokens, cost_usd, latency_s, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    owner,
                    repo,
                    pr_number,
                    kind,
                    calls,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_s,
                    time.time(),
                ),
            )

    def summary(self) -> dict:
        """Totals across every repository this instance has ever reviewed."""
        with self._connect() as conn:
            requests = conn.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) AS pending, "
                "SUM(CASE WHEN decision = 'accept' THEN 1 ELSE 0 END) AS accepted, "
                "SUM(CASE WHEN decision = 'reject' THEN 1 ELSE 0 END) AS rejected, "
                "SUM(CASE WHEN decision = 'expired' THEN 1 ELSE 0 END) AS expired "
                "FROM review_requests"
            ).fetchone()
            usage = conn.execute(
                "SELECT COALESCE(SUM(calls), 0) AS calls, COALESCE(SUM(cost_usd), 0) AS cost_usd, "
                "COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens, "
                "COALESCE(AVG(latency_s), 0) AS avg_latency_s FROM llm_usage"
            ).fetchone()
            repos = conn.execute(
                "SELECT COUNT(DISTINCT owner || '/' || repo) AS n FROM review_requests"
            ).fetchone()
        return {
            "repositories": repos["n"],
            "proposals": requests["n"] or 0,
            "pending": requests["pending"] or 0,
            "accepted": requests["accepted"] or 0,
            "rejected": requests["rejected"] or 0,
            "expired": requests["expired"] or 0,
            "llm_calls": usage["calls"],
            "cost_usd": round(usage["cost_usd"], 4),
            "tokens": usage["tokens"],
            "avg_latency_s": round(usage["avg_latency_s"], 2),
        }

    def repo_breakdown(self) -> list[dict]:
        """One row per repository: proposal counts and cost, most active first.

        ``unknown`` counts rows that are resolved but whose decision was never
        recorded -- only possible for a row created before the ``decision``
        column existed (a database migrated from an earlier version of this
        service). It is not "pending": nobody is waiting on it, the outcome
        is just not known. ``expired`` counts proposals whose paused graph run
        was gone when someone tried to decide on them.
        ``pending + accepted + rejected + expired + unknown`` always equals
        ``proposals``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT r.owner, r.repo, COUNT(*) AS proposals, "
                "SUM(CASE WHEN r.resolved = 0 THEN 1 ELSE 0 END) AS pending, "
                "SUM(CASE WHEN r.decision = 'accept' THEN 1 ELSE 0 END) AS accepted, "
                "SUM(CASE WHEN r.decision = 'reject' THEN 1 ELSE 0 END) AS rejected, "
                "SUM(CASE WHEN r.decision = 'expired' THEN 1 ELSE 0 END) AS expired, "
                "SUM(CASE WHEN r.resolved = 1 AND r.decision IS NULL THEN 1 ELSE 0 END) AS unknown, "
                "COALESCE((SELECT SUM(u.cost_usd) FROM llm_usage u "
                "          WHERE u.owner = r.owner AND u.repo = r.repo), 0) AS cost_usd "
                "FROM review_requests r GROUP BY r.owner, r.repo ORDER BY proposals DESC"
            ).fetchall()
        return [
            {
                "owner": row["owner"],
                "repo": row["repo"],
                "proposals": row["proposals"],
                "pending": row["pending"],
                "accepted": row["accepted"],
                "rejected": row["rejected"],
                "expired": row["expired"],
                "unknown": row["unknown"],
                "cost_usd": round(row["cost_usd"], 4),
            }
            for row in rows
        ]

    def recent_proposals(self, limit: int = 20) -> list[ReviewRequest]:
        """The most recently created proposals, across all repositories."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_requests ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_request(row) for row in rows]

    def search_proposals(
        self,
        *,
        owner: str | None = None,
        repo: str | None = None,
        status: str | None = None,
        text: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[ReviewRequest], int]:
        """One page of proposals, newest first, and the total matching count.

        ``status`` is one of ``pending``, ``accept``, ``reject``, ``expired``
        or ``unknown`` (resolved with no recorded decision); ``text`` matches
        the method, the file path or the short id; ``since`` / ``until`` are
        Unix timestamps bounding ``created_at`` (``until`` exclusive).
        """
        clauses: list[str] = []
        params: list[object] = []
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        if repo is not None:
            clauses.append("repo = ?")
            params.append(repo)
        if status is not None:
            if status not in _STATUS_CLAUSES:
                raise ValueError(f"Unknown status '{status}'. Valid: {sorted(_STATUS_CLAUSES)}.")
            clauses.append(_STATUS_CLAUSES[status])
        if text:
            pattern = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            clauses.append(
                "(method_label LIKE ? ESCAPE '\\' OR file_path LIKE ? ESCAPE '\\' OR short_id LIKE ? ESCAPE '\\')"
            )
            params.extend([pattern, pattern, pattern])
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("created_at < ?")
            params.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS n FROM review_requests {where}", params).fetchone()["n"]
            rows = conn.execute(
                f"SELECT * FROM review_requests {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [self._to_request(row) for row in rows], total

    def repo_stats(self, owner: str, repo: str) -> dict:
        """Proposal counts and LLM usage totals for one repository (all zeros
        for a repository nothing was ever recorded for)."""
        with self._connect() as conn:
            counts = conn.execute(
                "SELECT COUNT(*) AS proposals, "
                "SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) AS pending, "
                "SUM(CASE WHEN decision = 'accept' THEN 1 ELSE 0 END) AS accepted, "
                "SUM(CASE WHEN decision = 'reject' THEN 1 ELSE 0 END) AS rejected, "
                "SUM(CASE WHEN decision = 'expired' THEN 1 ELSE 0 END) AS expired, "
                "SUM(CASE WHEN resolved = 1 AND decision IS NULL THEN 1 ELSE 0 END) AS unknown, "
                "MAX(created_at) AS last_proposal_at "
                "FROM review_requests WHERE owner = ? AND repo = ?",
                (owner, repo),
            ).fetchone()
            usage = conn.execute(
                "SELECT COALESCE(SUM(calls), 0) AS calls, COALESCE(SUM(cost_usd), 0) AS cost_usd, "
                "COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens, "
                "COALESCE(AVG(latency_s), 0) AS avg_latency_s FROM llm_usage WHERE owner = ? AND repo = ?",
                (owner, repo),
            ).fetchone()
        return {
            "proposals": counts["proposals"],
            "pending": counts["pending"] or 0,
            "accepted": counts["accepted"] or 0,
            "rejected": counts["rejected"] or 0,
            "expired": counts["expired"] or 0,
            "unknown": counts["unknown"] or 0,
            "last_proposal_at": counts["last_proposal_at"],
            "llm_calls": usage["calls"],
            "cost_usd": round(usage["cost_usd"], 4),
            "tokens": usage["tokens"],
            "avg_latency_s": round(usage["avg_latency_s"], 2),
        }

    def total_cost_usd(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) AS c FROM llm_usage").fetchone()
        return float(row["c"])

    # -- on-demand scans ----------------------------------------------------------

    def start_scan(self, owner: str, repo: str) -> int | None:
        """Record a new running scan and return its id, or ``None`` when one
        is already running for this repository (checked and inserted in one
        transaction, so two clicks cannot start two scans)."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            running = conn.execute(
                "SELECT 1 FROM scans WHERE owner = ? AND repo = ? AND status = 'running'", (owner, repo)
            ).fetchone()
            if running:
                return None
            cursor = conn.execute(
                "INSERT INTO scans (owner, repo, status, started_at) VALUES (?, ?, 'running', ?)",
                (owner, repo, time.time()),
            )
            return int(cursor.lastrowid)

    def finish_scan(self, scan_id: int, *, error: str | None = None, **counts: int) -> None:
        """Close a scan as ``done`` (with its counts) or ``failed`` (with ``error``)."""
        allowed = {"prs", "proposed", "already_proposed", "already_decided", "clean", "not_proposed", "failed_prs"}
        if set(counts) - allowed:
            raise ValueError(f"Unknown scan counts: {sorted(set(counts) - allowed)}")
        assignments = "".join(f", {key} = ?" for key in counts)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE scans SET status = ?, finished_at = ?, error = ?{assignments} WHERE id = ?",
                ("failed" if error else "done", time.time(), error, *counts.values(), scan_id),
            )

    def get_scan(self, scan_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return dict(row) if row else None

    def latest_scan(self, owner: str, repo: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scans WHERE owner = ? AND repo = ? ORDER BY started_at DESC, id DESC LIMIT 1",
                (owner, repo),
            ).fetchone()
        return dict(row) if row else None

    def fail_running_scans(self, reason: str) -> None:
        """Close scans left "running" by a process that stopped mid-scan, so
        they do not block a new scan forever."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE scans SET status = 'failed', finished_at = ?, error = ? WHERE status = 'running'",
                (time.time(), reason),
            )

    # -- settings overrides (edited from the dashboard) -------------------------

    def get_settings(self) -> dict[str, object]:
        """Every setting saved from the dashboard, decoded from JSON."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def save_settings(self, values: dict[str, object]) -> None:
        """Insert or replace each of ``values``; other saved keys are kept."""
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                [(key, json.dumps(value), now) for key, value in values.items()],
            )

    def clear_settings(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM settings")

    def usage_series(self, days: int = 14, *, owner: str | None = None, repo: str | None = None) -> list[dict]:
        """Daily totals (UTC) for the last ``days`` days, oldest first, one row
        per day even where no run happened (cost and calls at 0) so a chart
        does not have to guess at gaps. ``owner``/``repo`` restrict it to one
        repository."""
        import datetime

        today = datetime.datetime.now(datetime.UTC).date()
        start = today - datetime.timedelta(days=days - 1)
        repo_filter = " AND owner = ? AND repo = ?" if owner is not None and repo is not None else ""
        params: list[object] = [time.mktime(start.timetuple())]
        if repo_filter:
            params.extend([owner, repo])
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date(created_at, 'unixepoch') AS day, SUM(calls) AS calls, "
                f"SUM(cost_usd) AS cost_usd FROM llm_usage WHERE created_at >= ?{repo_filter} GROUP BY day",
                params,
            ).fetchall()
        by_day = {row["day"]: (row["calls"], row["cost_usd"]) for row in rows}
        series = []
        for offset in range(days):
            day = start + datetime.timedelta(days=offset)
            calls, cost = by_day.get(day.isoformat(), (0, 0.0))
            series.append({"date": day.isoformat(), "calls": calls or 0, "cost_usd": round(cost or 0.0, 4)})
        return series

    @staticmethod
    def _to_request(row: sqlite3.Row) -> ReviewRequest:
        return ReviewRequest(
            short_id=row["short_id"],
            thread_id=row["thread_id"],
            owner=row["owner"],
            repo=row["repo"],
            pr_number=row["pr_number"],
            file_path=row["file_path"],
            method_label=row["method_label"],
            head_sha=row["head_sha"],
            resolved=bool(row["resolved"]),
            decision=row["decision"],
            created_at=row["created_at"],
            installation_id=row["installation_id"],
        )

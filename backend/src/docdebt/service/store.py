"""Maps the short ids posted in PR comments back to a graph thread.

A person replies ``/docdebt accept 7f3a2c1d`` to a proposal comment; GitHub
gives back only the comment's own id and text, with no link to which graph
thread it was about. This tiny SQLite table is that missing link — separate
from the LangGraph checkpointer (``docdebt.graph.build``), which only knows
about thread ids, never about PR comments.
"""

from __future__ import annotations

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

CREATE TABLE IF NOT EXISTS installations (
    owner            TEXT NOT NULL,
    repo             TEXT NOT NULL,
    installation_id  TEXT NOT NULL,
    active           INTEGER NOT NULL DEFAULT 1,
    connected_at     REAL NOT NULL,
    PRIMARY KEY (owner, repo)
);
"""


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
        ``decision`` column without losing its existing rows."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(review_requests)")}
        if "decision" not in columns:
            conn.execute("ALTER TABLE review_requests ADD COLUMN decision TEXT")

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
    ) -> str:
        """Register a pending review request and return its short id."""
        short_id = uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO review_requests "
                "(short_id, thread_id, owner, repo, pr_number, file_path, method_label, head_sha, "
                " resolved, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (short_id, thread_id, owner, repo, pr_number, file_path, method_label, head_sha, time.time()),
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
                "SUM(CASE WHEN decision = 'reject' THEN 1 ELSE 0 END) AS rejected "
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
        is just not known. ``pending + accepted + rejected + unknown`` always
        equals ``proposals``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT r.owner, r.repo, COUNT(*) AS proposals, "
                "SUM(CASE WHEN r.resolved = 0 THEN 1 ELSE 0 END) AS pending, "
                "SUM(CASE WHEN r.decision = 'accept' THEN 1 ELSE 0 END) AS accepted, "
                "SUM(CASE WHEN r.decision = 'reject' THEN 1 ELSE 0 END) AS rejected, "
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

    def usage_series(self, days: int = 14) -> list[dict]:
        """Daily totals (UTC) for the last ``days`` days, oldest first, one row
        per day even where no run happened (cost and calls at 0) so a chart
        does not have to guess at gaps."""
        import datetime

        today = datetime.datetime.now(datetime.UTC).date()
        start = today - datetime.timedelta(days=days - 1)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date(created_at, 'unixepoch') AS day, SUM(calls) AS calls, "
                "SUM(cost_usd) AS cost_usd FROM llm_usage WHERE created_at >= ? GROUP BY day",
                (time.mktime(start.timetuple()),),
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
        )

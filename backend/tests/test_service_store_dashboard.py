"""Tests for the dashboard-facing parts of ReviewStore: installations
tracking, usage aggregation, and the migration that adds the ``decision``
column to a database created before this table existed.
"""

import sqlite3

from docdebt.service.store import ReviewStore


def test_connect_and_disconnect_repository(tmp_path):
    store = ReviewStore(tmp_path / "state.sqlite3")
    store.connect_repository(owner="acme", repo="demo", installation_id="1")
    store.connect_repository(owner="acme", repo="other", installation_id="1")
    assert {r["repo"] for r in store.connected_repositories()} == {"demo", "other"}

    store.disconnect_repository(owner="acme", repo="demo")
    assert {r["repo"] for r in store.connected_repositories()} == {"other"}


def test_reconnecting_updates_installation_id_and_reactivates(tmp_path):
    store = ReviewStore(tmp_path / "state.sqlite3")
    store.connect_repository(owner="acme", repo="demo", installation_id="1")
    store.disconnect_repository(owner="acme", repo="demo")
    store.connect_repository(owner="acme", repo="demo", installation_id="2")  # reinstalled, new id
    rows = store.connected_repositories()
    assert len(rows) == 1 and rows[0]["installation_id"] == "2"


def test_summary_on_an_empty_store_has_no_none_values(tmp_path):
    store = ReviewStore(tmp_path / "state.sqlite3")
    summary = store.summary()
    assert summary == {
        "repositories": 0,
        "proposals": 0,
        "pending": 0,
        "accepted": 0,
        "rejected": 0,
        "llm_calls": 0,
        "cost_usd": 0.0,
        "tokens": 0,
        "avg_latency_s": 0.0,
    }


def test_log_usage_skips_zero_call_rows(tmp_path):
    store = ReviewStore(tmp_path / "state.sqlite3")
    store.log_usage(
        owner="a",
        repo="b",
        pr_number=1,
        kind="review",
        calls=0,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        latency_s=0.1,
    )
    assert store.summary()["llm_calls"] == 0


def test_usage_series_has_one_row_per_day_including_empty_days(tmp_path):
    store = ReviewStore(tmp_path / "state.sqlite3")
    store.log_usage(
        owner="a",
        repo="b",
        pr_number=1,
        kind="review",
        calls=3,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.02,
        latency_s=1.5,
    )
    series = store.usage_series(days=7)
    assert len(series) == 7
    assert series[-1]["calls"] == 3 and series[-1]["cost_usd"] == 0.02
    assert all(row["calls"] == 0 for row in series[:-1])


def test_migration_adds_decision_column_to_a_pre_existing_database(tmp_path):
    path = tmp_path / "old.sqlite3"
    # Simulate a database created before the "decision" column (and llm_usage /
    # installations tables) existed, matching the schema shipped in T3.
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE review_requests (
            short_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, owner TEXT NOT NULL,
            repo TEXT NOT NULL, pr_number INTEGER NOT NULL, file_path TEXT NOT NULL,
            method_label TEXT NOT NULL, head_sha TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO review_requests VALUES ('abc', 't1', 'acme', 'demo', 1, 'X.java', 'X.f', 'sha', 0, 0.0)"
    )
    conn.commit()
    conn.close()

    store = ReviewStore(path)  # must migrate in place, not lose the existing row
    request = store.get("abc")
    assert request is not None and request.decision is None
    store.mark_resolved("abc", "accept")
    assert store.get("abc").decision == "accept"


def test_repo_breakdown_surfaces_rows_resolved_before_the_decision_column_existed(tmp_path):
    """A row resolved by an older version of this service (before the
    ``decision`` column was added) is ``resolved=1`` with ``decision=NULL``.
    It must not silently vanish from every count: it is not pending (nobody
    is waiting on it), and it is not accepted or rejected (that was never
    recorded) -- it belongs in ``unknown``, and every count must still sum to
    ``proposals``.
    """
    path = tmp_path / "state.sqlite3"
    store = ReviewStore(path)
    store.create(
        thread_id="t1",
        owner="acme",
        repo="demo",
        pr_number=1,
        file_path="X.java",
        method_label="X.f",
        head_sha="s1",
    )
    # Simulate the old mark_resolved(short_id) with no decision argument, by
    # writing resolved=1 directly without ever setting decision.
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("UPDATE review_requests SET resolved = 1 WHERE thread_id = 't1'")
    conn.commit()
    conn.close()

    row = store.repo_breakdown()[0]
    assert (row["pending"], row["accepted"], row["rejected"], row["unknown"]) == (0, 0, 0, 1)
    assert row["pending"] + row["accepted"] + row["rejected"] + row["unknown"] == row["proposals"]

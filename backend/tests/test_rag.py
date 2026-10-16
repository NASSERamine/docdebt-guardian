"""Knowledge base and retrieval (RAG): indexing, ranking, sources and the leak guard."""

from docdebt.core.java_parser import parse_java_source
from docdebt.rag.embeddings import bm25_scores, choose_embedder, cosine, tokenize
from docdebt.rag.knowledge import KnowledgeBase, find_readme
from docdebt.rag.retriever import RagProvider

MAIN = """package demo;

public class Account {

    /**
     * Deposits money on the account.
     * @param amount the amount to deposit, in cents
     */
    public void deposit(long amount) { balance += amount; }

    /**
     * Withdraws money from the account.
     * @param amount the amount to withdraw, in cents
     * @return the new balance
     */
    public long withdraw(long amount) { balance -= amount; audit(amount); return balance; }

    /**
     * Records an operation in the audit log.
     * @param amount the amount moved
     */
    void audit(long amount) { log.add(amount); }

    /**
     * Returns the balance.
     * @param wrong phantom parameter
     * @return the balance
     */
    public long getBalance() { return balance; }

    /**
     * Transfers money to another account.
     * @param target the receiving account
     * @param amount the amount
     */
    public void transfer(Account target, long amount) { withdraw(amount); target.deposit(amount); }
}
"""

OTHER = """package demo;

public class Wallet {
    /**
     * Withdraws coins from the wallet.
     * @param coins how many coins
     * @return the coins left
     */
    public int withdraw(int coins) { return left - coins; }

    /**
     * Adds a deposit fee to an amount, in cents.
     * @param amount the amount in cents
     * @return the amount with the fee
     */
    public long withFee(long amount) { return amount + 30; }
}
"""

TEST = """package demo;

public class AccountTest {
    public void depositAddsMoney() { account.deposit(5); assertEquals(5, account.getBalance()); }
}
"""

README = "# Demo\nA bank account library.\n# Amounts\nAll amounts are expressed in cents, never in euros."


def kb_with_files(tmp_path):
    kb = KnowledgeBase(tmp_path / "state.sqlite3")
    kb.rebuild(
        "acme",
        "bank",
        {
            "src/main/java/demo/Account.java": MAIN,
            "src/main/java/demo/Wallet.java": OTHER,
            "src/test/java/demo/AccountTest.java": TEST,
        },
        readme=README,
        commit_sha="abc",
    )
    return kb


def unit(name, source=MAIN, path="src/main/java/demo/Account.java"):
    return next(u for u in parse_java_source(source, path) if u.name == name)


def test_tokenize_splits_identifiers_and_bm25_ranks_relevant_documents():
    assert tokenize("getBalanceInCents snake_case") == ["get", "balance", "in", "cent", "snake", "case"]
    scores = bm25_scores("withdraw cents", ["deposit money", "withdraw money in cents", "unrelated"])
    assert scores.index(max(scores)) == 1 and scores[2] == 0
    assert round(cosine([1, 0], [1, 0]), 6) == 1.0 and cosine([1, 0], [0, 1]) == 0.0


def test_rebuild_indexes_only_debt_free_documented_methods_tests_and_readme(tmp_path):
    kb = kb_with_files(tmp_path)
    stats = kb.stats("acme", "bank")
    names = {c.name for c in kb.chunks("acme", "bank", ["method"])}
    assert "getBalance" not in names  # phantom @param: structural debt, never an example
    assert {"deposit", "withdraw", "audit", "transfer", "withFee"} <= names
    assert stats["counts"]["test"] == 1 and stats["counts"]["readme"] == 2
    assert stats["embedder"] == "bm25" and stats["commit_sha"] == "abc"
    test_chunk = kb.chunks("acme", "bank", ["test"])[0]
    assert "deposit" in test_chunk.extra.split()


def test_rebuild_keeps_validated_chunks(tmp_path):
    kb = kb_with_files(tmp_path)
    kb.add_validated("acme", "bank", unit("getBalance"), "/**\n * Returns the balance, in cents.\n */")
    kb.rebuild("acme", "bank", {"src/main/java/demo/Account.java": MAIN})
    assert kb.stats("acme", "bank")["counts"]["validated"] == 1
    assert kb.stats("acme", "bank")["counts"]["readme"] == 0


def test_siblings_and_callees_come_from_the_file_under_review(tmp_path):
    rag = RagProvider(None, "acme", "bank", sources=("siblings", "callees"))
    rag.set_files({"src/main/java/demo/Account.java": MAIN})
    items = rag.build(unit("withdraw")).items
    labels = [(i.source, i.label) for i in items]
    assert ("callees", "Account.audit") in labels
    assert all(label != "Account.withdraw" for _, label in labels)  # never the method itself
    assert all(label != "Account.getBalance" for _, label in labels)  # trivial getter skipped
    assert len([i for i in items if i.source == "siblings"]) >= 1


def test_similar_methods_come_from_the_whole_project_with_the_leak_guard(tmp_path):
    kb = kb_with_files(tmp_path)
    rag = RagProvider(kb, "acme", "bank", sources=("similar",), top_k=2)
    items = rag.build(unit("deposit")).items
    assert items and all(i.source == "similar" and i.score > 0 for i in items)
    assert all(not i.label.endswith(".deposit") for i in items)
    # Wallet.withdraw has the same name as Account.withdraw: excluded for it (leak guard)
    withdraw_items = rag.build(unit("withdraw")).items
    assert all(not i.label.endswith(".withdraw") for i in withdraw_items)


def test_tests_readme_and_validated_sources(tmp_path):
    kb = kb_with_files(tmp_path)
    kb.add_validated(
        "acme", "bank", unit("withFee", OTHER, "src/main/java/demo/Wallet.java"), "/** Fee in cents. */"
    )
    rag = RagProvider(kb, "acme", "bank", sources=("tests", "readme", "validated"))
    by_source = {i.source: i for i in rag.build(unit("deposit")).items}
    assert by_source["tests"].label == "AccountTest.depositAddsMoney"
    assert "cents" in by_source["readme"].text
    assert by_source["validated"].label == "Wallet.withFee"


def test_context_text_names_each_source_and_stays_within_budget(tmp_path):
    kb = kb_with_files(tmp_path)
    rag = RagProvider(kb, "acme", "bank", sources=("siblings", "similar"))
    rag.set_files({"src/main/java/demo/Account.java": MAIN})
    text = rag(unit("transfer"))
    assert "Documented method in the same file" in text and "Similar documented method" in text
    assert len(text) < 4000
    assert rag.build(unit("transfer")) is rag.build(unit("transfer"))  # cached per review


def test_unknown_source_is_rejected_and_readme_is_found():
    import pytest

    with pytest.raises(ValueError):
        RagProvider(None, "a", "b", sources=("commits",))
    assert find_readme(["docs/README.md", "README.md", "src/A.java"]) == "README.md"
    assert find_readme(["src/A.java"]) is None


def test_bm25_mode_needs_no_embedding_server():
    assert choose_embedder("bm25") is None
    assert choose_embedder("openai", allow_cloud=False) is None

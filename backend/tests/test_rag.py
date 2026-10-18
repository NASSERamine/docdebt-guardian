"""Knowledge base and retrieval (RAG): indexing, ranking, sources and the leak guard."""

from docdebt.core.java_parser import parse_java_source
from docdebt.rag.embeddings import bm25_scores, choose_embedder, cosine, tokenize
from docdebt.rag.knowledge import KnowledgeBase, find_readme
from docdebt.rag.retriever import RagProvider, rrf_fuse

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


def kb_with_files(tmp_path, embedder=None):
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
        embedder=embedder,
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


class WordEmbedder:
    """A deterministic stand-in for MiniLM: one dimension per vocabulary word."""

    name = "test:words"
    VOCAB = ("deposit", "withdraw", "money", "cent", "fee", "balance", "transfer", "audit", "coin", "account")

    def embed(self, texts):
        out = []
        for text in texts:
            words = tokenize(text)
            out.append([float(words.count(w)) + 0.01 for w in self.VOCAB])
        return out


def test_siblings_are_the_three_closest_documented_methods_of_the_file():
    rag = RagProvider(None, "acme", "bank")
    rag.set_files({"src/main/java/demo/Account.java": MAIN})
    items = rag.build(unit("withdraw")).items
    siblings = [i for i in items if i.source == "siblings"]
    assert {i.label for i in siblings} == {"Account.deposit", "Account.audit", "Account.transfer"}
    assert siblings[-1].label == "Account.transfer"  # the farthest of the three
    assert [i.rank for i in siblings] == [1, 2, 3]
    labels = {i.label for i in items}
    assert "Account.withdraw" not in labels  # never the method itself
    assert "Account.getBalance" not in labels  # trivial getter skipped


def test_rrf_fuses_ranks_and_breaks_ties_by_id():
    fused = rrf_fuse([[3, 1, 2], [1, 4]])
    assert [cid for cid, _ in fused] == [1, 3, 4, 2]
    assert fused[0][1] == 1 / 61 + 1 / 62
    assert rrf_fuse([[5], [7]]) == [(5, 1 / 61), (7, 1 / 61)]  # equal scores: lower id first


def test_hybrid_search_uses_bm25_and_vectors_with_the_leak_guard(tmp_path):
    kb = kb_with_files(tmp_path, embedder=WordEmbedder())
    assert kb.stats("acme", "bank")["embedder"] == "test:words"
    rag = RagProvider(kb, "acme", "bank", embedder=WordEmbedder())
    items = [i for i in rag.build(unit("withdraw")).items if i.source != "siblings"]
    assert 0 < len(items) <= 4
    assert all(not i.label.endswith(".withdraw") for i in items)  # Wallet.withdraw: same name, excluded
    assert all(i.dense_rank is not None or i.lexical_rank is not None for i in items)
    assert [i.rank for i in items] == list(range(1, len(items) + 1))
    assert items[0].score >= items[-1].score


def test_bm25_alone_when_no_embedder(tmp_path):
    kb = kb_with_files(tmp_path)
    rag = RagProvider(kb, "acme", "bank")
    items = [i for i in rag.build(unit("deposit")).items if i.source != "siblings"]
    assert items and all(i.dense_rank is None and i.lexical_rank is not None for i in items)
    assert all(not i.label.endswith(".deposit") for i in items)


def test_tests_readme_and_validated_chunks_compete_in_the_search(tmp_path):
    kb = kb_with_files(tmp_path)
    kb.add_validated(
        "acme", "bank", unit("withFee", OTHER, "src/main/java/demo/Wallet.java"), "/** Deposit fee in cents. */"
    )
    rag = RagProvider(kb, "acme", "bank", top_k=10)
    sources = {i.source: i for i in rag.build(unit("deposit")).items}
    assert sources["tests"].label == "AccountTest.depositAddsMoney"
    assert "cents" in sources["readme"].text
    assert sources["validated"].label == "Wallet.withFee"


def test_same_index_same_method_same_context(tmp_path):
    kb = kb_with_files(tmp_path, embedder=WordEmbedder())
    first = RagProvider(kb, "acme", "bank", embedder=WordEmbedder())
    second = RagProvider(kb, "acme", "bank", embedder=WordEmbedder())
    for rag in (first, second):
        rag.set_files({"src/main/java/demo/Account.java": MAIN})
    assert [i.to_dict() for i in first.build(unit("transfer")).items] == [
        i.to_dict() for i in second.build(unit("transfer")).items
    ]


def test_context_text_names_each_source_and_stays_within_budget(tmp_path):
    kb = kb_with_files(tmp_path)
    rag = RagProvider(kb, "acme", "bank")
    rag.set_files({"src/main/java/demo/Account.java": MAIN})
    text = rag(unit("transfer"))
    assert "Documented method in the same file" in text
    assert len(text) < 4000
    assert rag.build(unit("transfer")) is rag.build(unit("transfer"))  # cached per review


def test_a_push_updates_only_the_changed_files(tmp_path):
    kb = kb_with_files(tmp_path, embedder=WordEmbedder())
    before = {c.name for c in kb.chunks("acme", "bank", ["method"])}
    assert "withFee" in before
    changed = OTHER.replace("withFee", "withCharge")
    kb.update_files(
        "acme", "bank", {"src/main/java/demo/Wallet.java": changed}, [], embedder=WordEmbedder(), commit_sha="def"
    )
    after = {c.name for c in kb.chunks("acme", "bank", ["method"])}
    assert "withCharge" in after and "withFee" not in after and "deposit" in after
    assert kb.stats("acme", "bank")["commit_sha"] == "def"
    kb.update_files("acme", "bank", {}, ["src/main/java/demo/Wallet.java"])
    assert "withCharge" not in {c.name for c in kb.chunks("acme", "bank", ["method"])}


def test_chroma_index_answers_like_the_exact_index(tmp_path):
    import pytest

    pytest.importorskip("chromadb")
    from docdebt.rag.vectors import ChromaIndex

    kb = KnowledgeBase(tmp_path / "state.sqlite3", index=ChromaIndex(tmp_path / "chroma"))
    kb.rebuild("acme", "bank", {"src/main/java/demo/Account.java": MAIN}, embedder=WordEmbedder())
    query = WordEmbedder().embed(["withdraw money"])[0]
    hits = kb.index.query("acme", "bank", query, 3)
    names = {c.id: c.name for c in kb.chunks("acme", "bank")}
    assert names[hits[0][0]] == "withdraw"
    assert kb.index.query("other", "repo", query, 3) == []  # one repository never sees another's chunks


def test_readme_is_found():
    assert find_readme(["docs/README.md", "README.md", "src/A.java"]) == "README.md"
    assert find_readme(["src/A.java"]) is None


def test_bm25_mode_needs_no_embedding_model():
    assert choose_embedder("bm25") is None

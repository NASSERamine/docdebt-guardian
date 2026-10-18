"""Check the retrieval (RAG) of increment 3 with the real embedding model, no LLM.

Builds a knowledge base exactly as the service does (documented methods without
debt, tests, README sections), with the real all-MiniLM-L6-v2 vectors in a
Chroma index, then builds the context of one method and checks:

* the embedding model loads (else the service falls back to BM25 only);
* siblings first (at most 3, same file), then the hybrid top 4 fused by RRF,
  with each item's BM25 rank and vector rank;
* leak guard: no item carries the name of the method itself;
* budget: the context text stays within 3,600 characters;
* determinism: building the context twice gives the same items in the same order;
* Chroma and the exact SQLite vector search rank the knowledge base identically.

By default it runs on a small built-in project (no download but the model);
``--repo DIR --file PATH --method NAME`` runs it on a folder of your own.

Example:
    python scripts/check_rag.py
    python scripts/check_rag.py --repo ~/some-java-project --file src/main/java/x/Foo.java --method bar
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from docdebt.core.java_parser import parse_java_source
from docdebt.detection.context import README_NAMES
from docdebt.rag.embeddings import choose_embedder, embedder_status
from docdebt.rag.knowledge import KnowledgeBase
from docdebt.rag.retriever import MAX_SIBLINGS, RRF_K, TOP_K, RagProvider
from docdebt.rag.vectors import SqliteVectorIndex

OWNER, REPO = "check", "rag"
BUDGET = 3600

DEMO = {
    "src/main/java/demo/Temperature.java": """package demo;

public class Temperature {

    /**
     * Converts kelvin to celsius.
     * @param tempInCelsius temperature in kelvin
     * @return temperature in celsius
     */
    public double toCelsius(double tempInKelvin) {
        return tempInKelvin - 273.15;
    }

    /**
     * Converts a temperature in kelvin to fahrenheit.
     * @param tempInKelvin temperature in kelvin
     * @return temperature in fahrenheit
     */
    public double toFahrenheit(double tempInKelvin) {
        return (tempInKelvin - 273.15) * 9 / 5 + 32;
    }

    /**
     * Tells whether water freezes at this temperature.
     * @param tempInKelvin temperature in kelvin
     * @return true at or below 273.15 kelvin
     */
    public boolean isFreezing(double tempInKelvin) {
        return tempInKelvin <= 273.15;
    }
}
""",
    "src/main/java/demo/Pressure.java": """package demo;

public class Pressure {

    /**
     * Converts a pressure in bar to pascal.
     * @param bar pressure in bar
     * @return pressure in pascal
     */
    public double toPascal(double bar) {
        return bar * 100000;
    }

    /**
     * Converts a pressure in pascal to bar.
     * @param pascal pressure in pascal
     * @return pressure in bar
     */
    public double toBar(double pascal) {
        return pascal / 100000;
    }
}
""",
    "src/test/java/demo/TemperatureTest.java": """package demo;

public class TemperatureTest {

    /** Absolute zero is -273.15 degrees celsius. */
    public void absoluteZero() {
        assertEquals(-273.15, new Temperature().toFahrenheit(0) * 0 - 273.15, 1e-9);
    }
}
""",
}
DEMO_README = """# Demo

## Units

Every temperature is stored in kelvin; conversions return celsius or fahrenheit.

## Pressure

Pressures are given in bar or pascal.
"""


def read_repo(folder: Path) -> tuple[dict[str, str], str | None]:
    files, readme = {}, None
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        if rel.endswith(".java"):
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
        elif "/" not in rel and rel in README_NAMES:
            readme = path.read_text(encoding="utf-8", errors="replace")
    return files, readme


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'OK ' if ok else 'NO '} {label}{': ' + detail if detail else ''}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None, help="folder of a Java project (default: built-in)")
    parser.add_argument("--file", default="src/main/java/demo/Temperature.java")
    parser.add_argument("--method", default="toCelsius")
    parser.add_argument("--embeddings", choices=["minilm", "bm25"], default="minilm")
    args = parser.parse_args()

    files, readme = read_repo(args.repo.expanduser()) if args.repo else (DEMO, DEMO_README)
    if args.file not in files:
        raise SystemExit(f"{args.file} is not a .java file of the project ({len(files)} files found).")
    units = [u for u in parse_java_source(files[args.file], args.file) if u.name == args.method]
    if not units:
        raise SystemExit(f"No method {args.method} in {args.file}.")
    unit = units[0]
    has_readme = "yes" if readme else "no"
    print(f"Project    : {args.repo or 'built-in demo'} ({len(files)} Java file(s), README: {has_readme})")
    print(f"Method     : {unit.class_name}.{unit.name} ({args.file}:{unit.start_line})")

    started = time.perf_counter()
    embedder = choose_embedder(args.embeddings)
    status = embedder_status()
    results = []
    loaded = f"{embedder.name} in {time.perf_counter() - started:.1f} s" if embedder else ""
    loaded = loaded or f"BM25 only ({status['error']})"
    results.append(check("embedding model loaded", embedder is not None, loaded))

    with tempfile.TemporaryDirectory() as tmp:
        try:
            from docdebt.rag.vectors import ChromaIndex

            index, index_name = ChromaIndex(Path(tmp) / "chroma"), "Chroma"
        except ImportError:
            index, index_name = None, "exact SQLite (chromadb not installed)"
        kb = KnowledgeBase(Path(tmp) / "kb.sqlite3", index=index)
        kb.rebuild(OWNER, REPO, files, readme=readme, embedder=embedder)
        stats = kb.stats(OWNER, REPO)
        print(f"Index      : {index_name}; knowledge base: {stats}")

        rag = RagProvider(kb, OWNER, REPO, embedder=embedder)
        rag.set_files({args.file: files[args.file]})
        context = rag.build(unit)
        sizes = f"siblings <= {MAX_SIBLINGS}, then top {TOP_K}, RRF k={RRF_K}"
        print(f"\nContext    : {len(context.items)} item(s) ({sizes})")
        for item in context.items:
            ranks = ""
            if item.source != "siblings":
                lex, dense = item.lexical_rank or "-", item.dense_rank or "-"
                ranks = f"  RRF {item.score:.4f}  BM25 #{lex}  vectors #{dense}"
            print(f"  {item.rank}. [{item.source}] {item.label} ({item.path}){ranks}")

        siblings = [i for i in context.items if i.source == "siblings"]
        searched = [i for i in context.items if i.source != "siblings"]
        first = all(i.path == args.file for i in siblings) and context.items[: len(siblings)] == siblings
        results.append(check("siblings first, from the same file", first, f"{len(siblings)} sibling(s)"))
        results.append(check("hybrid search within its size", len(searched) <= TOP_K, f"{len(searched)} item(s)"))
        if embedder is not None:
            results.append(check("vector ranks used by the fusion", any(i.dense_rank for i in searched)))
        leaked = [i.label for i in context.items if i.label.rsplit(".", 1)[-1] == unit.name]
        results.append(check("leak guard (no method named like the target)", not leaked, ", ".join(leaked)))
        text = context.text
        results.append(check("budget", len(text) <= BUDGET, f"{len(text)} / {BUDGET} characters"))

        again = RagProvider(kb, OWNER, REPO, embedder=embedder)
        again.set_files({args.file: files[args.file]})
        key = [(i.source, i.label) for i in context.items]
        same = [(i.source, i.label) for i in again.build(unit).items] == key
        results.append(check("deterministic (same context twice)", same))

        if index is not None and embedder is not None:
            exact_db = Path(tmp) / "exact.sqlite3"
            exact = KnowledgeBase(exact_db, index=SqliteVectorIndex(exact_db))
            exact.rebuild(OWNER, REPO, files, readme=readme, embedder=embedder)
            twin = RagProvider(exact, OWNER, REPO, embedder=embedder, max_siblings=0)
            chroma = RagProvider(kb, OWNER, REPO, embedder=embedder, max_siblings=0)
            a = [i.label for i in chroma.search(unit)]
            b = [i.label for i in twin.search(unit)]
            results.append(check("Chroma ranks like the exact vector search", a == b, " > ".join(a)))

    print(f"\nResult     : {sum(results)}/{len(results)} check(s) passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

"""Documentation health of a whole repository (default branch).

Reads the Java files of the default branch through the API (at most
``max_files``), runs the deterministic Auditor only (no LLM call, so it is
free and fast), and records one snapshot:

* ``methods``: public and protected methods (the API a reader relies on);
* ``documented``: those with a Javadoc;
* ``debt_methods``: documented methods with at least one structural error;
* ``score`` = (documented - debt_methods) / methods.

The same pass rebuilds the repository's knowledge base for retrieval
(``docdebt.rag.knowledge``): debt-free documented methods, tests and README.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from posixpath import dirname

from docdebt.core.auditor import Auditor
from docdebt.core.extractor import Extractor, ExtractorConfig
from docdebt.core.models import Visibility
from docdebt.rag.knowledge import KnowledgeBase, find_readme
from docdebt.service.github_client import GithubApiError, GithubClient

logger = logging.getLogger("docdebt.service")

_API_VISIBILITIES = frozenset({Visibility.PUBLIC, Visibility.PROTECTED})


def _package(path: str) -> str:
    """The source directory of a file, below ``src/main/java`` when present."""
    directory = dirname(path)
    marker = "src/main/java/"
    if marker in directory + "/":
        directory = (directory + "/").split(marker, 1)[1].rstrip("/")
    return directory.replace("/", ".") or "(root)"


def _is_test(path: str) -> bool:
    return "/src/test/" in f"/{path}" or path.endswith(("Test.java", "Tests.java"))


def measure(files: dict[str, str], auditor: Auditor | None = None) -> dict:
    """Health figures for the given main-code files (path -> source)."""
    auditor = auditor or Auditor()
    extractor = Extractor(ExtractorConfig(visibilities=_API_VISIBILITIES, require_javadoc=False))
    packages: dict[str, dict] = defaultdict(lambda: {"methods": 0, "documented": 0, "debt_methods": 0})
    per_file = []
    totals = {"methods": 0, "documented": 0, "debt_methods": 0, "warnings": 0}
    for path, source in files.items():
        units = extractor.from_source(source, path)
        if not units:
            continue
        documented = [u for u in units if u.javadoc]
        reports = [auditor.audit(u) for u in documented]
        debt = sum(r.has_structural_debt for r in reports)
        warnings = sum(len(r.warnings) for r in reports)
        row = {"path": path, "methods": len(units), "documented": len(documented), "debt_methods": debt}
        per_file.append(row)
        pkg = packages[_package(path)]
        for key in ("methods", "documented", "debt_methods"):
            pkg[key] += row[key]
            totals[key] += row[key]
        totals["warnings"] += warnings

    def score(row: dict) -> float | None:
        return round((row["documented"] - row["debt_methods"]) / row["methods"], 4) if row["methods"] else None

    package_rows = sorted(
        ({"package": name, **counts, "score": score(counts)} for name, counts in packages.items()),
        key=lambda r: (r["score"] if r["score"] is not None else 1.0, -r["methods"]),
    )
    worst = sorted(
        (dict(r, score=score(r), undocumented=r["methods"] - r["documented"]) for r in per_file),
        key=lambda r: (-(r["debt_methods"] + r["undocumented"]), r["path"]),
    )
    return {
        **totals,
        "files": len(per_file),
        "score": score(totals),
        "packages": package_rows[:50],
        "worst_files": [r for r in worst if r["debt_methods"] or r["undocumented"]][:20],
    }


def run_health_scan(
    github: GithubClient,
    owner: str,
    repo: str,
    *,
    max_files: int = 300,
    knowledge: KnowledgeBase | None = None,
    embedder=None,
    progress=None,
) -> dict:
    """Scan the default branch and rebuild the knowledge base; returns the
    values of the snapshot to store. ``progress(done, total)`` is called as
    the files are read, for the progress bar."""
    meta = github.get_repository(owner, repo)
    branch = meta.get("default_branch") or "main"
    sha = github.get_branch_head(owner, repo, branch)
    paths, truncated = github.list_tree_paths(owner, repo, sha)
    java = sorted(p for p in paths if p.endswith(".java"))
    main_paths = [p for p in java if not _is_test(p)]
    test_paths = [p for p in java if _is_test(p)]
    budget = max(1, max_files)
    if len(main_paths) > budget:
        truncated = True
    main_paths = main_paths[:budget]
    test_paths = test_paths[: max(0, budget - len(main_paths))]

    total = len(main_paths) + (len(test_paths) + 1 if knowledge is not None else 0)
    read = [0]

    def fetch(path: str) -> str | None:
        read[0] += 1
        if progress is not None:
            progress(read[0], total)
        try:
            return github.get_file_content(owner, repo, path, sha)[0]
        except (GithubApiError, UnicodeDecodeError, KeyError):
            logger.warning("health scan: could not read %s/%s:%s", owner, repo, path)
            return None

    main_files = {p: text for p in main_paths if (text := fetch(p)) is not None}
    figures = measure(main_files)
    if knowledge is not None:
        test_files = {p: text for p in test_paths if (text := fetch(p)) is not None}
        readme_path = find_readme(paths)
        readme = fetch(readme_path) if readme_path else None
        knowledge.rebuild(
            owner, repo, {**main_files, **test_files}, readme=readme, embedder=embedder, commit_sha=sha
        )
    return {**figures, "branch": branch, "commit_sha": sha, "truncated": truncated}


def run_push_index(
    github: GithubClient,
    owner: str,
    repo: str,
    *,
    after_sha: str,
    changed: list[str],
    removed: list[str],
    knowledge: KnowledgeBase,
    embedder=None,
) -> dict:
    """Incremental update of the knowledge base after a push to the default
    branch: only the Java files the push added, modified or removed (and the
    README when it changed) are re-indexed."""
    java_changed = sorted(p for p in changed if p.endswith(".java"))
    contents = {}
    for path in java_changed:
        try:
            contents[path] = github.get_file_content(owner, repo, path, after_sha)[0]
        except (GithubApiError, UnicodeDecodeError, KeyError):
            logger.warning("push index: could not read %s/%s:%s", owner, repo, path)
    readme_path = find_readme(changed)
    readme = None
    if readme_path:
        try:
            readme = github.get_file_content(owner, repo, readme_path, after_sha)[0]
        except (GithubApiError, UnicodeDecodeError, KeyError):
            readme = None
    return knowledge.update_files(
        owner,
        repo,
        contents,
        [p for p in removed if p.endswith(".java")],
        readme=readme,
        embedder=embedder,
        commit_sha=after_sha,
    )

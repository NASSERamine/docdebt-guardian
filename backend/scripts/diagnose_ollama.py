"""Find why a local model behaves differently from the research runs.

Sends the SAME detection prompts for the SAME benchmark items through five
variants and prints the verdict distribution of each, so the effect of one
factor at a time is visible:

  A research_generate   raw method_code, /api/generate, num_ctx 4096  (the research client)
  B generate_ctx        raw method_code, /api/generate, num_ctx = --num-ctx
  C generate_parsed     tree-sitter method text, /api/generate, num_ctx = --num-ctx
  D chat_raw            raw method_code, /api/chat, num_ctx = --num-ctx
  E tool_path           the tool itself (LangChain ChatOllama, tree-sitter text)

Reading the table:
  A vs B     -> effect of the context window
  B vs C     -> effect of the code normalisation done by the parser
  B vs D     -> effect of /api/generate vs /api/chat
  D vs E     -> anything left in the tool's own path

Example:
    python scripts/diagnose_ollama.py --dataset datasets/verified_final_DEFINITIVE_363.xlsx \\
        --split-file config/split.csv --n-clean 30 --n-debt 20 --out runs/diagnose.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import httpx

from docdebt.benchmark import item_to_unit, load_items
from docdebt.llm.catalog import get_model_spec
from docdebt.llm.client import LLMClient
from docdebt.llm.parsing import parse_binary_verdict
from docdebt.llm.prompts.registry import PromptRegistry

VARIANTS = ["A_research_generate", "B_generate_ctx", "C_generate_parsed", "D_chat_raw", "E_tool_path"]


def call_generate(http, base_url, model, prompt, num_ctx, max_tokens):
    r = http.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": max_tokens, "num_ctx": num_ctx},
        },
    )
    r.raise_for_status()
    return r.json()["response"]


def call_chat(http, base_url, model, prompt, num_ctx, max_tokens):
    r = http.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": max_tokens, "num_ctx": num_ctx},
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def summarize(rows, variant):
    clean = [r for r in rows if r["label"] == "clean" and r[variant] != "none"]
    debt = [r for r in rows if r["label"] == "debt" and r[variant] != "none"]
    none = sum(r[variant] == "none" for r in rows)
    spec = sum(r[variant] == "clean" for r in clean) / len(clean) if clean else float("nan")
    rec = sum(r[variant] == "debt" for r in debt) / len(debt) if debt else float("nan")
    answered = len(clean) + len(debt)
    correct = sum(r[variant] == "clean" for r in clean) + sum(r[variant] == "debt" for r in debt)
    acc = correct / answered if answered else float("nan")
    return acc, rec, spec, none


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--pattern", default="zeroshot")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--n-clean", type=int, default=30)
    parser.add_argument("--n-debt", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = load_items(args.dataset)
    with args.split_file.resolve().open(newline="", encoding="utf-8") as fh:
        keep = {r["item_id"] for r in csv.DictReader(fh) if r["split"] == args.split}
    pool = items[items["item_id"].astype(str).isin(keep)].to_dict("records")
    rng = random.Random(args.seed)
    clean = [r for r in pool if r["label"] == "clean"]
    debt = [r for r in pool if r["label"] == "debt"]
    picked = rng.sample(clean, min(args.n_clean, len(clean))) + rng.sample(debt, min(args.n_debt, len(debt)))

    spec = get_model_spec(args.model)
    max_tokens = spec.budget(args.pattern)
    registry = PromptRegistry()
    client = LLMClient(spec)
    http = httpx.Client(timeout=600.0)

    rows, code_differs = [], 0
    for i, item in enumerate(picked, 1):
        raw_code, doc = str(item["method_code"]), item["stale_doc"]
        unit = item_to_unit(str(item["item_id"]), raw_code, doc)
        if unit is None:
            print(f"  skip {item['item_id']}: code did not parse")
            continue
        code_differs += unit.code != raw_code
        p_raw = registry.build(args.model, args.pattern, raw_code, doc).text
        p_parsed = registry.build(args.model, args.pattern, unit.code, doc).text

        answers = {
            "A_research_generate": call_generate(http, args.base_url, args.model, p_raw, 4096, max_tokens),
            "B_generate_ctx": call_generate(http, args.base_url, args.model, p_raw, args.num_ctx, max_tokens),
            "C_generate_parsed": call_generate(
                http, args.base_url, args.model, p_parsed, args.num_ctx, max_tokens
            ),
            "D_chat_raw": call_chat(http, args.base_url, args.model, p_raw, args.num_ctx, max_tokens),
            "E_tool_path": client.invoke(p_parsed, purpose=args.pattern).text,
        }
        row = {"item_id": item["item_id"], "label": item["label"], "prompt_chars": len(p_raw)}
        for variant, text in answers.items():
            verdict = parse_binary_verdict(text)
            row[variant] = "none" if verdict is None else ("debt" if verdict else "clean")
            row[variant + "_raw"] = text.strip()[:80].replace("\n", " ")
        rows.append(row)
        if i % 10 == 0:
            print(f"  {i}/{len(picked)} items done")

    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.out.resolve().open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_clean = sum(r["label"] == "clean" for r in rows)
    print(
        f"\n{len(rows)} items ({n_clean} clean, {len(rows) - n_clean} debt); "
        f"parser changed the code text on {code_differs} item(s); "
        f"longest prompt {max(r['prompt_chars'] for r in rows)} characters"
    )
    print(f"\n{'variant':<22}{'accuracy':>9}{'recall':>9}{'specificity':>13}{'no-verdict':>12}")
    for variant in VARIANTS:
        acc, rec, spc, none = summarize(rows, variant)
        print(f"{variant:<22}{acc:9.3f}{rec:9.3f}{spc:13.3f}{none:12d}")
    print("\nAnswers per variant:")
    for variant in VARIANTS:
        print(f"  {variant:<22}{dict(Counter(r[variant] for r in rows))}")
    print(f"\nPer-item detail: {args.out.resolve()}")
    print("Research reference for Qwen zero-shot: accuracy 0.7145, recall 0.7291, specificity 0.7000")


if __name__ == "__main__":
    main()

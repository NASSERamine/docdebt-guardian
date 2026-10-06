# DocDebt Guardian

Detect (and, in later phases, repair) **stale method-level Javadoc** in Java
code: a comment that no longer describes the method it documents.

This package covers phases L1 to L7 of the project, plus a local dashboard:

| Lot | Module | What it does |
|---|---|---|
| L1 Extractor | `docdebt.core.java_parser`, `docdebt.core.diff`, `docdebt.core.extractor` | tree-sitter parsing of methods, signatures and Javadoc; unified diff reading; method selection |
| L2 Auditor | `docdebt.core.auditor`, `docdebt check` | exact structural rules (`@param`, `@return`, `@throws`) |
| L3 LLM layer | `docdebt.llm` | LangChain clients (Ollama, OpenAI, optional Anthropic/Google), retries, truncation handling, token budgets, usage tracking, versioned prompts, output parsing |
| L4 Detection | `docdebt.detection`, `docdebt detect` | Judge (votes, confidence, validated citation), Router (local-to-cloud cascade), optional context, threshold calibration |
| L5 Repair | `docdebt.core.javadoc_edit`, `docdebt.repair`, `docdebt repair` | edit plan, research Regeneration prompts, field-level merge, Verifier, retry with feedback, diff and write |
| L6 Graph | `docdebt.graph`, `docdebt review` | LangGraph orchestration of Auditor, Router/Judge and Repairer, with a synchronous human-review pause and a SQLite checkpointer for crash/restart-safe resume |
| L7 Service | `docdebt.service`, GitHub App | FastAPI webhook endpoint; the graph's review pause becomes asynchronous, resumed by a `/docdebt accept\|reject <id>` PR comment instead of a keypress; an accepted repair is committed straight to the PR branch via the GitHub Contents API |
| Dashboard | `dashboard/` (React + Vite), served by the same FastAPI process under `/dashboard` | connected repositories, pending/accepted/rejected proposals, LLM cost and call volume — all read from data the App already collected, no separate database |

It is the production continuation of the research project *Rewriting the
Narrative: Leveraging Large Language Models for Documentation Debt
Management*. The research code was reused where it had been validated:

* `parse_javadoc` and the structural rules are ported unchanged;
  `scripts/parity_check.py` checks that signatures and structural-debt
  decisions are identical to the research implementation (726/726 on the
  paired research dataset).
* The detection wordings are copied verbatim, with the locked wording index
  of each model from the research screening.
* `parse_binary_verdict` (last whole word wins) is unchanged.

Two research issues are fixed here: the `siblings+callees` context no
longer includes the target method or its overloads (label leak), and the
context closing instruction no longer asks CoT prompts for a single word.

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,bench]"          # add ,openai when you move to GPT-5.6, ,service for the GitHub App
pytest                                 # 117 tests, no model needed
```

`tree-sitter` is pinned below 0.26: version 0.26.0 crashed (segmentation
fault) on repeated parsing in our tests, 0.25.x is stable.

## Structural check (no LLM)

```bash
docdebt check path/to/src                       # files or directories
git diff main... > pr.diff && docdebt check --diff pr.diff --root .
docdebt check path/to/src --format json --fail-on never
```

Exit code 1 when an error is found (`--fail-on error|warning|never`).

## Detection with Qwen (local)

```bash
ollama pull qwen2.5-coder:7b
curl http://localhost:11434/api/tags             # server up?
python scripts/smoke_test.py --model qwen2.5-coder:7b --patterns zeroshot cot
docdebt detect path/to/src --model qwen2.5-coder:7b --show-votes
```

Always run the smoke test first and read the RAW answers.

How a verdict is reached:

1. The Auditor runs first. Its errors are exact and always mean debt; an LLM
   answer can never cancel them.
2. The Judge votes with the model's validated prompt(s). Confidence is the
   share of valid votes agreeing with the majority (1.0 with a single vote
   at temperature 0; use several patterns or `samples_per_pattern` for graded
   confidence).
3. On a debt vote, the citation prompt asks for the contradicted claim and the
   contradicting code. Only quotes found verbatim become semantic findings.
   A debt vote with no valid citation and no structural error is an
   abstention (`unsupported_verdict`).
4. The Router escalates to the next model of the cascade only on abstention,
   if cloud models are allowed and the budget is not exhausted.

## Repair

```bash
docdebt repair path/to/src                       # structural debt: Auditor findings, then repair
docdebt repair path/to/src --semantic            # also detect semantic debt with the Judge
docdebt repair --diff pr.diff --root .           # only the methods touched by a diff
docdebt repair path/to/src --write               # write the verified repairs (never without this flag)
docdebt repair path/to/src --model gpt-5.6 --wording 0   # a model the research did not screen
```

How a repair is built (`docdebt.repair.repairer`):

1. An **edit plan** lists the fields that may change: `missing-*` findings add a
   field, `unknown-param` removes one, a semantic finding replaces the field it disputes.
2. The model receives the research *Regeneration* prompt of its locked wording and
   answers with a complete corrected Javadoc.
3. The tool **reads only the planned fields** from that answer and merges them into the
   original. Every other block is copied from the original: in the usual layout its source
   lines stay byte for byte, whatever the model wrote. A renamed `@param` takes the place
   of the old one, so the diff stays minimal.
4. The **Verifier** checks the merged text: no structural error left, every requested
   field present and non-empty, the cited claim gone from a replaced field, and every
   other field identical to the original. A repair that fails is retried with the reasons
   (`repair.max_attempts`) and, if it still fails, never proposed.

Statuses: `verified` (a checked fix is ready), `unchanged` (the model kept every
disputed field: nothing to propose), `failed`, `not_needed`, `skipped` (no Javadoc).

Validate the machinery without any model (the developer's own fix plays the model):

```bash
python scripts/oracle_repair_check.py --dataset datasets/verified_final_DEFINITIVE_363.xlsx
```

Then with a model (read the raw answers first):

```bash
python scripts/smoke_test_repair.py --model qwen2.5-coder:7b
python scripts/run_repair_benchmark.py --model qwen2.5-coder:7b --limit 50 \
    --dataset datasets/verified_final_DEFINITIVE_363.xlsx --split-file config/split.csv \
    --split dev --out runs/repair_qwen.csv          # add --mode pipeline, --similarity
```

Repair wordings: the research screened Qwen, CodeGemma and DeepSeek. GPT-5.6 and Claude have
no locked wording: pass one with `--wording` (a warning says it was not screened).

## Review (L6): the graph, with a human-review pause

```bash
docdebt review path/to/src                       # audit, detect, repair, pause per verified repair
docdebt review path/to/src --structural-only      # skip the Judge entirely (no LLM detection call)
docdebt review --diff pr.diff --root .            # only the methods touched by a diff
docdebt review path/to/src --yes                  # accept every verified repair without asking
docdebt review path/to/src --write                # write accepted repairs into the source files
```

For each method the graph runs, in order: **Auditor** (deterministic), then the
**Judge/Router** (unless `--structural-only`), then, if there is debt, the
**Repairer**. A repair the Verifier accepts pauses the graph and prints its
diff; you answer `y`/`N` and the graph continues from exactly that point —
nothing else re-runs. A repair that fails, or a method with no debt, finishes
without pausing.

**Restart-safe by design.** State is checkpointed to SQLite
(`<config dir>/.docdebt/graph.sqlite3` by default, or `--checkpoint-db`) after
every step. Interrupting `docdebt review` — Ctrl+C, a closed terminal, a
crashed process — and re-running the same command later reattaches to any
paused review exactly where it stopped: the Auditor, Judge and Repairer are
**not** called again for a method already paused or already finished with
unchanged content. Only a method whose code or Javadoc changed since its last
run (same file, same method, but edited) is re-analysed. This is what makes
repeatedly reviewing a whole repository affordable: methods nobody touched
since the last run cost nothing on a second pass.

Verify this yourself: run `docdebt review`, answer nothing (let it hang or
Ctrl+C it), then re-run the identical command. The paused method's diff
reappears unchanged and answering it costs no additional model call — confirm
with your model provider's usage dashboard, or point `base_url` at a request
counter for a local model.

## GitHub App service (L7)

Turns the graph into a bot that reviews pull requests. A proposal is posted
as a PR comment; a person replies `/docdebt accept <id>` or
`/docdebt reject <id>`; on accept, the fix is committed straight to the PR's
branch through the GitHub Contents API (no local checkout, no direct push
from your machine — the GitHub App's own installation token does it).

**Permission change required.** Committing a fix needs the Contents API's
single-file update endpoint, which requires **Contents: Read and write**.
If your App was set up with Contents as read-only (the original scoping,
before this write path existed), update it: on the App's page →
**Permissions & events** → set Contents to *Read and write* → **Save
changes**. Existing installations must then approve the new permission (an
email/notification is sent, or re-visit **Install App** for the same
prompt); no need to recreate the App or reinstall from scratch.

```bash
pip install -e ".[service]"
cp .env.example .env               # fill in your GitHub App's credentials
set -a && source .env && set +a
uvicorn docdebt.service.app:create_app --factory --reload --port 8000
curl localhost:8000/health         # {"status": "ok"}
```

GitHub cannot reach `localhost`, so exposing this port to the internet (a
tunnel in development, a real deployment later) is required before any real
webhook arrives — see the project's setup notes for the `cloudflared` tunnel
used in local development.

**Safety property, tested explicitly** (`tests/test_service_app.py`): a
comment authored by a bot — including this service's own posted proposals,
which spell out `/docdebt accept <id>` as an instruction — is never parsed as
a command. Without this, the service would see its own proposal echoed back
by GitHub's webhook and silently self-accept it, with no human involved.

## Dashboard (local only)

A read-only view of what the GitHub App has done: connected repositories,
pending/accepted/rejected proposals, and LLM cost/call volume. It is served
by the **same** FastAPI process as the webhook endpoint — one process, one
port, no separate frontend server needed once built.

```bash
cd dashboard
npm install
npm run build          # writes dashboard/dist/; re-run after any UI change
cd ..
./run_local.sh          # builds the dashboard if missing, then starts the service
```

Open `http://localhost:8000` — it redirects to `/dashboard`. The **Connect a
repository** button points at GitHub's own installation flow
(`github.com/apps/<slug>/installations/new`): clicking it, choosing a
repository and accepting the permissions is the entire "connect my repo"
step — no token to create, no config file to edit on the user's side. What
happens after (App created, keys, webhook secret) is developer-side, one-time
setup, covered in "GitHub App service" below.

**Sign in with GitHub (multi-user mode).** With `DOCDEBT_GITHUB_CLIENT_ID`
and `DOCDEBT_GITHUB_CLIENT_SECRET` set (the App's own OAuth credentials,
*General* page of the App, with *Callback URL* set to
`<DOCDEBT_PUBLIC_URL>/auth/callback`), the dashboard becomes an application:

* `/dashboard/` is a public home page with **Sign in with GitHub**;
* after signing in, every repository where the App is installed and the
  user can push is added to their list; **Repositories** adds one more by its
  exact `owner/name` or GitHub URL: the service checks it exists, that the
  user has write access (accepting a fix commits to it), and that the App is
  installed, otherwise it links to GitHub's install page for that account;
* every page and API answer is limited to the user's repositories (another
  user's repository answers 404), and the session cookie (HttpOnly,
  SameSite=Lax) authorizes decisions, scans and settings: no dashboard token.

Without them the service stays in local mode, exactly as before: one
operator, everything visible, changes authorized by `DOCDEBT_DASHBOARD_TOKEN`.

**Pages.** The dashboard is a small single-page app (React Router), every
page addressable by URL:

| URL | Page |
|---|---|
| `/dashboard/` | Home page: what DocDebt does, Sign in with GitHub (or Open the dashboard) |
| `/dashboard/overview` | Overview: a two-line introduction to the project, totals, repositories, the 10 latest proposals, 14-day LLM usage |
| `/dashboard/repositories` | Your repositories, and Add a repository by exact name or URL |
| `/dashboard/proposals` | History: every proposal, searchable (method, file, id) and filterable by repository, status and date range, 25 per page. Filters live in the URL (`?repo=&status=&q=&since=&until=&page=`), so a filtered view can be bookmarked |
| `/dashboard/repos/<owner>/<repo>` | One repository: connection state, counts, LLM calls, tokens, cost, 30-day usage, and its own filtered history |
| `/dashboard/proposals/<id>` | One proposal before deciding: the debt found (each rule and message, or the quoted claim and code for semantic debt), the proposed fix as a diff and field by field, the method as it is today, then Accept / Reject |
| `/dashboard/how-it-works` | The pipeline for a newcomer: Auditor → Judge → Repairer → Verifier → human review, each step on one running example, with what DocDebt guarantees; shows whether the Judge is on in the current settings |
| `/dashboard/settings` | Model, repair prompt, semantic detection on/off, confidence threshold, cost cap; connected and disconnected repositories |

**Settings are applied without a restart.** `.env` gives the defaults; values
saved on the Settings page are stored in the state database and override
them, read again for every webhook, so the next pull request uses them.
*Reset to defaults* forgets every saved value. Saving needs the dashboard
token (below). The cost cap counts the LLM cost recorded across all
repositories; once reached, a new push is not reviewed and the PR gets a
comment saying why (local Ollama models record no cost). Connecting and
removing repositories still happens on GitHub, in the App's installation
settings; the list follows the installation webhooks.

**Scan now.** A connected repository's page has a *Scan now* button: it
reviews every open pull request of the repository right away, exactly as if
GitHub had sent a webhook for each one (same graph, same PR comments, same
Accept/Reject), so a demo does not have to wait for a real push. It runs in
the background; the page follows it and shows the totals (PRs scanned, new
proposals, proposals already waiting, clean methods). One scan at a time per
repository; it needs the dashboard token and respects the cost cap. A
proposal still waiting for a decision is never posted twice, whether the PR
is scanned again or receives a push that does not touch that method.

API behind the pages: `GET /api/proposals/search`, `GET /api/proposals/<id>`, `GET /api/repos/<owner>/<repo>`,
`GET|PUT|DELETE /api/settings`, `POST /api/repos/<owner>/<repo>/scan`,
`GET /api/scans/<id>` and `POST /api/installations/sync` (writes need the token).

**Accept or reject from the dashboard.** Each pending proposal in *Recent
activity* has **Accept** / **Reject** buttons. They call
`POST /api/proposals/<id>/decide` (`{"decision": "accept" | "reject"}`), which
runs exactly the same code as a `/docdebt accept|reject <id>` PR comment
(`ReviewFlow.decide`): the paused graph is resumed, an accepted fix is
committed to the PR branch, and the outcome is still posted on the PR, marked
"decided from the dashboard", so the PR keeps the full history. Decisions
from PR comments and from the dashboard are serialized, so the same fix is
never committed twice.

The buttons only appear when `DOCDEBT_DASHBOARD_TOKEN` is set; without it the
dashboard stays read-only and shows the slash command instead. The dashboard
asks for the token on the first click and keeps it in the browser. The token
matters because the cloudflared tunnel exposes the whole port 8000, not only
the webhook: without it, anyone who finds the URL could commit to your PRs.

**Active development on the UI**: run the API and the dashboard as two
processes instead, so UI edits show up without rebuilding:

```bash
set -a && source .env && set +a
uvicorn docdebt.service.app:create_app --factory --reload --port 8000   # terminal 1
cd dashboard && npm run dev                                              # terminal 2, proxies /api to :8000
```

Open `http://localhost:5173/dashboard/` (the Vite dev server's own port).

**Local only, by design for now**: nothing here is hosted or exposed to the
internet. Real GitHub webhooks (a PR opened on a repo you connected) still
need `cloudflared` pointed at port 8000, exactly as in "GitHub App service"
below — the dashboard does not change that part.

## Benchmark workflow (Qwen first, then GPT-5.6)

All paths absolute; every script resumes where it stopped.

```bash
# 1. Few-shot items, in research order (ids taken from the research notebook, never guessed)
python scripts/export_fewshot.py --dataset /abs/verified_final_DEFINITIVE_363.xlsx \
    --item-ids ID0 ID1 ID2 ID3 ID4 ID5 --out /abs/config/fewshot_detection.json

# 2. Frozen dev/test split, calibration items excluded
python scripts/make_split.py --dataset /abs/verified_final_DEFINITIVE_363.xlsx \
    --exclude-file /abs/config/fewshot_detection.json --out /abs/config/split.csv

# 3. Dev run (checkpoint every 20 items, resume on re-run, stops after 3 failed items)
python scripts/run_benchmark.py --model qwen2.5-coder:7b --patterns zeroshot \
    --dataset /abs/verified_final_DEFINITIVE_363.xlsx \
    --split-file /abs/config/split.csv --split dev --out /abs/runs/qwen_dev.csv

# 4. Threshold on dev, then a single final run on test
python scripts/calibrate.py --run /abs/runs/qwen_dev.csv --target-precision 0.85 \
    --out /abs/config/threshold_qwen.json
```

For GPT-5.6: `pip install -e ".[openai]"`, export `OPENAI_API_KEY`, run the
smoke test with `--model gpt-5.6`, then the same benchmark commands.

## Layout

```
src/docdebt/
  core/        models, java_parser, javadoc, javadoc_edit, diff, extractor, auditor   (no LLM)
  llm/         config, catalog, client, parsing, prompts/{detection_wordings, registry, fewshot, explain}
  detection/   citations, judge, router, context, calibration
  repair/      wordings, registry, fewshot, extract, plan, verifier, repairer, apply
  benchmark.py settings.py pipeline.py cli.py
scripts/       smoke_test, export_fewshot, make_split, run_benchmark, calibrate, parity_check,
               oracle_repair_check, smoke_test_repair, run_repair_benchmark, export_repair_fewshot, diagnose_ollama
config/        docdebt.example.yml
tests/         unit and end-to-end tests with a scripted chat model
```

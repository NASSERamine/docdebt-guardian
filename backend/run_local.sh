#!/usr/bin/env bash
# One-shot local run: builds the dashboard if needed, then starts the
# FastAPI service (API + dashboard on the same port). Everything stays on
# your machine -- this does NOT expose anything to the internet. To receive
# real GitHub webhooks you still need cloudflared (see README.md).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and fill in your GitHub App credentials first." >&2
  exit 1
fi

if [ ! -f dashboard/dist/index.html ]; then
  echo "Building the dashboard (first run only; re-run 'npm run build' in dashboard/ after UI changes)..."
  (cd dashboard && npm install && npm run build)
fi

set -a
source .env
set +a

echo "Starting on http://localhost:8000  (dashboard at /dashboard, API at /api/*)"
exec uvicorn docdebt.service.app:create_app --factory --reload --port 8000

#!/bin/bash
# usage: scripts/gateway.sh <cap_usd> <ledger> [port]
# The only process allowed to reach the API, and only api.anthropic.com.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
export PYTHONPATH="$PWD/tools/netguard" NETGUARD_ALLOW_HOSTS=api.anthropic.com NETGUARD_LOG="$PWD/runs/netguard-gateway.log"
.venv/bin/python -m sfb.gateway --env .env --cap "$1" --ledger "$2" --port "${3:-8765}" &
echo $! > runs/gateway.pid
wait

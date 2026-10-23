#!/bin/bash
# usage: scripts/harness.sh <sfb.run args...>   (loopback only: it can reach the gateway, nothing else)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
export PYTHONPATH="$PWD/tools/netguard" NETGUARD_LOG="$PWD/runs/netguard-harness.log" PYTHONDONTWRITEBYTECODE=1
exec sandbox-exec -f tools/netguard/loopback.sb .venv/bin/python -m sfb.run "$@"

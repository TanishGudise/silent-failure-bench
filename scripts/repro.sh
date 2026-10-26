#!/bin/bash
# usage: scripts/repro.sh
# $0 reproduction: no API key, no paid calls. Installs into .venv, then runs the
# test suite, the 40-seed labeling gate and the manifest check.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "needs uv (https://docs.astral.sh/uv/)"; exit 1; }
[ -d .venv ] || uv venv --python 3.11 .venv
# Dependencies only (pinned in pyproject.toml); the code runs from the repo root.
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python -m sfb.gate --seeds 40 | tail -1
.venv/bin/python -m sfb.manifest --check

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .

echo "Environment ready: $REPO_DIR/.venv"
echo "Run: .venv/bin/stretch3-vr smoke"

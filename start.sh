#!/usr/bin/env bash
# One-click start for macOS / Linux: creates a venv on first run, then launches CogniSign.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting up (this takes a minute)..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e .
fi
exec .venv/bin/cognisign "$@"

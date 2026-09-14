#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Environment ready. Activate it with: source .venv/bin/activate"

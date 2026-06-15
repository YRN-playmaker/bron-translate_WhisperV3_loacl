#!/usr/bin/env bash
# ===========================================================================
#  BroN-translate (Local Edition) - macOS / Linux launcher
#  Run:  ./start.sh   (first run sets up a venv + installs deps)
# ===========================================================================
set -e
cd "$(dirname "$0")"

PY=python3
if ! command -v $PY >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[*] Creating virtual environment (first run only)..."
    $PY -m venv .venv
    echo "[*] Installing dependencies... this can take a while."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

if [ ! -f ".env" ]; then
    echo "[!] No .env found. Copying .env.example -> .env"
    cp .env.example .env
    echo "[!] Edit .env and paste your DEEPSEEK_API_KEY, then run ./start.sh again."
    exit 0
fi

echo "[*] Starting BroN-translate (Local)..."
exec .venv/bin/python main.py

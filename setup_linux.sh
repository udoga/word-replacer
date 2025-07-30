#!/usr/bin/env bash
set -euo pipefail

PY_VERSION=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')

sudo apt update -y
sudo apt install -y python"${PY_VERSION}"-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet stopwords
./download_dataset.sh
pytest

if [[ -n "${1:-}" ]]; then
    hf auth login --token "$1"
else
    echo "No HF token supplied – skipping login."
fi

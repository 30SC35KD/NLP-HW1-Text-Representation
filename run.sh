#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

mkdir -p assets/glove
curl -L -o assets/glove/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip -o assets/glove/glove.6B.zip glove.6B.100d.txt -d assets/glove

.venv/bin/python run_all.py "$@"

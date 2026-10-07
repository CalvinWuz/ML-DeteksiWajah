#!/bin/bash
# Antrean pengambilan wajah OpenFake (bisa dijalankan ulang; bagian yang sudah selesai dilewati).
#   bash scripts/harvest_openfake_all.sh
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { echo "=== $(date +%H:%M) $*"; $PY -u scripts/fetch_openfake.py "$@" || echo "!!! gagal: $*"; }
# validation -> val (memilih checkpoint)
run --split validation --out-name val        --shards 0 1 2     --cap-fake 60  --cap-real 1500
# train: 8 shard tersebar di 32 kelompok file
run --split train --shards 0 75 150 225 300 375 450 525 --cap-fake 350 --cap-real 6000
# validation (shard lain) -> test generator dikenal
run --split validation --out-name test_known --shards 10 11  --cap-fake 120 --cap-real 3000
# test OpenFake (generator terbaru) -> test generator baru; z-image/flux dibatasi agar bandwidth ke GPT/Gemini
run --split test --shards 1 2 3 --cap-fake 400 --cap-real 3000
echo "=== $(date +%H:%M) SELESAI"

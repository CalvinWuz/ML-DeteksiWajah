#!/bin/bash
# Seluruh alur dataset modern, otomatis dan bisa dijalankan ulang (melanjutkan dari posisi terakhir):
#   ambil data OpenFake -> susun dataset -> training 4 model -> tabel perbandingan
#   nohup caffeinate -i -s bash scripts/run_modern_pipeline.sh > logs/modern_pipeline.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "=== $(date '+%d/%m %H:%M') mulai"
bash scripts/harvest_openfake_all.sh
echo "=== $(date '+%d/%m %H:%M') susun dataset modern"
$PY -u scripts/prepare_modern.py || { echo "!!! prepare_modern gagal"; exit 1; }
for m in vgg16 efficientnet_b0 maxvit_t clip_vitl14; do
  echo "=== $(date '+%d/%m %H:%M') training $m"
  $PY -u scripts/train.py --model $m --mode modern --resume || echo "!!! training $m gagal"
done
$PY scripts/compare.py --run-name modern
echo "=== $(date '+%d/%m %H:%M') PIPELINE SELESAI"

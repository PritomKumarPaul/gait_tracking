#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
CLEAN_ROOT="$ROOT/clean_demo_v2"
OUT_ROOT="$CLEAN_ROOT/output/pritom_test1_threshold_$TS"
WORK_ROOT="$OUT_ROOT/work"

MODEL="${MODEL:-grew_gaitbase}"
THRESHOLD="${THRESHOLD:-0.97}"

mkdir -p "$OUT_ROOT"

echo "[clean-demo-v2] timestamp=$TS"
echo "[clean-demo-v2] model=$MODEL threshold=$THRESHOLD"
echo "[clean-demo-v2] gallery: $CLEAN_ROOT/gallery/pritomgallery.mp4"
echo "[clean-demo-v2] probe:   $CLEAN_ROOT/probes/test1probe.mp4"

python clean_demo_v2/tools/build_single_target_gallery.py \
  --gallery-video "$CLEAN_ROOT/gallery/pritomgallery.mp4" \
  --target-label pritom \
  --model "$MODEL" \
  --work-root "$WORK_ROOT/gallery_build" \
  --gallery-out "$OUT_ROOT/pritom_gallery.npz" \
  --metadata-out "$OUT_ROOT/pritom_gallery.json"

python clean_demo_v2/tools/threshold_target_probe_with_video.py \
  --gallery-npz "$OUT_ROOT/pritom_gallery.npz" \
  --probe-video "$CLEAN_ROOT/probes/test1probe.mp4" \
  --target-label pritom \
  --threshold "$THRESHOLD" \
  --model "$MODEL" \
  --work-root "$WORK_ROOT/probe_test1" \
  --output-json "$OUT_ROOT/test1_pritom_vs_unknown.json" \
  --summary-txt "$OUT_ROOT/test1_pritom_vs_unknown.txt" \
  --output-video "$OUT_ROOT/test1_pritom_vs_unknown_annotated.mp4"

echo "[clean-demo-v2] finished"
echo "[clean-demo-v2] output folder: $OUT_ROOT"
echo "[clean-demo-v2] result JSON: $OUT_ROOT/test1_pritom_vs_unknown.json"
echo "[clean-demo-v2] summary TXT: $OUT_ROOT/test1_pritom_vs_unknown.txt"
echo "[clean-demo-v2] annotated video: $OUT_ROOT/test1_pritom_vs_unknown_annotated.mp4"

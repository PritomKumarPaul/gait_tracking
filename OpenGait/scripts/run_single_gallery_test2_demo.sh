#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/OpenGait/demo/output/single_gallery_test2_$TS"
OUT_ROOT="$ROOT/OpenGait/output/single_gallery_test2_$TS"

PRITOM_VIDEO="${PRITOM_VIDEO:-Pritomtrain1}"
COCO_VIDEO="${COCO_VIDEO:-Coco1}"
JEEVITH_VIDEO="${JEEVITH_VIDEO:-Jeevith1}"
THRESHOLD="${THRESHOLD:-0.97}"
MARGIN="${MARGIN:-0.005}"

mkdir -p "$RUN_ROOT" "$OUT_ROOT"

echo "[run] timestamp=$TS"
echo "[run] selected gallery: pritom=$PRITOM_VIDEO coco=$COCO_VIDEO jeevith=$JEEVITH_VIDEO"
echo "[run] threshold=$THRESHOLD margin=$MARGIN"

python OpenGait/tools/evaluate_single_gallery_choices.py \
  --output-json "$OUT_ROOT/single_gallery_choice_analysis.json"

python OpenGait/tools/build_selected_gallery.py \
  --pritom-video "$PRITOM_VIDEO" \
  --coco-video "$COCO_VIDEO" \
  --jeevith-video "$JEEVITH_VIDEO" \
  --gallery-out "$OUT_ROOT/selected_gallery.npz" \
  --metadata-out "$OUT_ROOT/selected_gallery.json"

python OpenGait/tools/identify_probe_with_gallery.py \
  --gallery-npz "$OUT_ROOT/selected_gallery.npz" \
  --video-path "$ROOT/newvideos/Test2.mp4" \
  --model grew_gaitbase \
  --threshold -1 \
  --margin 0 \
  --match-mode max \
  --work-root "$RUN_ROOT/probe_demo_style" \
  --output-json "$OUT_ROOT/test2_demo_style.json"

python OpenGait/tools/identify_probe_with_gallery.py \
  --gallery-npz "$OUT_ROOT/selected_gallery.npz" \
  --video-path "$ROOT/newvideos/Test2.mp4" \
  --model grew_gaitbase \
  --threshold "$THRESHOLD" \
  --margin "$MARGIN" \
  --match-mode max \
  --work-root "$RUN_ROOT/probe_safe_style" \
  --output-json "$OUT_ROOT/test2_safe_style.json"

echo "[run] outputs saved under:"
echo "  $OUT_ROOT"
echo "  $RUN_ROOT"

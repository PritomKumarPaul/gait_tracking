#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_baseline_casiab.sh"
"$SCRIPT_DIR/run_gaitset_casiab.sh"
"$SCRIPT_DIR/run_gaitgl_casiab.sh"

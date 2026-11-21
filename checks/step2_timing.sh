#!/usr/bin/env bash
# Step 2, part 5: the twin's latency and throughput per frame on the GPU (cpp/bin/dpd_bench),
# for the BASEBAND frame at 50 samples per symbol (6400) and at 8 (1024, the same 128 symbols),
# back to back and paced at the pipeline's arrival rate (128 symbols at 5000 symbols/s: 25.6 ms),
# with pinned and pageable host memory. Writes results/step2_timing.txt. About 2 minutes.
#   make -C cpp && checks/step2_timing.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=cpp/bin/dpd_bench
W=results/models/transformer_nominal.dpdt
OUT=results/step2_timing.txt
{
  echo "Step 2, part 5: dpd_bench, $(date -u +%Y-%m-%dT%H:%MZ), power mode: $(nvpmodel -q 2>/dev/null | head -1 | sed 's/NV Power Mode: //')"
  echo
  for frame in 6400 1024; do
    for memory in pinned pageable; do
      $BIN $W --frame $frame --frames 1000 --memory $memory; echo
    done
    $BIN $W --frame $frame --frames 1000 --memory pinned --period-ms 25.6; echo
  done
} | tee $OUT

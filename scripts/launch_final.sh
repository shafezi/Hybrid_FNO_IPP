#!/bin/bash
# Final ablation: 4 robot counts × 4 kernels × 3 acquisitions × 5 seeds = 240 configs
# Episode length L=10 (40 days)
# Metrics-only mode: skip pickles/videos/figures, just save per-config metrics CSVs
# Aggregator can then combine them into one master CSV.

set -e
cd "$(dirname "$0")/.."

OUT_BASE=results/dynamic_ipp/final
METRICS_DIR=$OUT_BASE/metrics
mkdir -p $METRICS_DIR

L=10

# (seed, t0) pairs — 5 seeds with their corresponding t0 (max_t0 = 366 - 40 - 1 = 325)
SEED_T0=("42 29" "0 277" "7 308" "100 250" "2024 78")

# ---- Generate config list ----
JOBS=$(mktemp)
for st in "${SEED_T0[@]}"; do
  read -r seed t0 <<< "$st"
  for nbots in 5 10 20 40; do
    for kn in "matern 0.5" "matern 1.5" "matern 2.5" "rbf 1.5"; do
      kernel=$(echo $kn | cut -d' ' -f1)
      nu=$(echo $kn | cut -d' ' -f2)
      for acq in uncertainty_only hybrid_ucb mi; do
        echo "$seed|$t0|$nbots|$kernel|$nu|$acq"
      done
    done
  done
done > $JOBS

TOTAL=$(wc -l < $JOBS)
echo "Total configs: $TOTAL"

# ---- Run all configs (metrics-only mode) ----
# Each config = 5 method runs (sequential within process), so 12 in parallel = 60 episodes simultaneously
echo "=== Running $TOTAL configs (20 in parallel, OMP=4) ==="
cat $JOBS | xargs -I {} -P 20 -d '\n' bash -c '
  IFS="|" read -r seed t0 nbots kernel nu acq <<< "$1"
  OMP_NUM_THREADS=4 python scripts/run_final_config.py \
    --n_robots $nbots --kernel $kernel --nu $nu --acquisition $acq \
    --t0 $t0 --seed $seed --episode_length '$L' \
    --metrics_only --metrics_dir '$METRICS_DIR' 2>&1 | tail -1
' _ {}

echo "=== ALL DONE ==="
echo "CSVs: $(ls $METRICS_DIR | wc -l)"

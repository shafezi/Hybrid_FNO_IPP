#!/bin/bash
# Run n_robots=10 then n_robots=5, 20 episodes each.
set -e
cd "$(dirname "$0")/.."

for N in 10 5; do
    echo ""
    echo "=== Starting n_robots=$N, 20 episodes ($(date)) ==="
    PYTHONUNBUFFERED=1 python scripts/diag_multiepisode.py \
        --n_episodes 20 --n_robots $N 2>&1 \
        | grep -v ConvergenceWarning | grep -v 'warnings.warn' \
        | tee results/dynamic_ipp/diag/multiepisode_n${N}_stdout.log
    echo "=== n_robots=$N done ($(date)) ==="
done

echo ""
echo "=== BOTH RUNS COMPLETE ==="
ls -la results/dynamic_ipp/diag/multiepisode_winning_n*.csv
#!/bin/bash
# Queue: wait for current 40-robot run, then run 20, 10, 5 in sequence.
set -e
cd "$(dirname "$0")/.."

echo "=== Waiting for current 40-robot run (PID 6694) to finish ==="
# Wait until no diag_multiepisode python process is running
while pgrep -f "diag_multiepisode.py" > /dev/null; do
    sleep 30
done
echo "Current run finished."

for N in 20 10 5; do
    echo ""
    echo "=== Starting n_robots=$N, 20 episodes ($(date)) ==="
    PYTHONUNBUFFERED=1 python scripts/diag_multiepisode.py \
        --n_episodes 20 --n_robots $N 2>&1 \
        | grep -v ConvergenceWarning | grep -v 'warnings.warn' \
        | tee results/dynamic_ipp/diag/multiepisode_n${N}_stdout.log
    echo "=== n_robots=$N done ($(date)) ==="
done

echo ""
echo "=== ALL SWEEPS COMPLETE ==="
ls -la results/dynamic_ipp/diag/multiepisode_winning_*.csv
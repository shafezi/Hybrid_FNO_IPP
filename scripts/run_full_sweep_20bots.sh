#!/bin/bash
# Full sweep, same as 10 robots but with --n_robots 20
OUT=/tmp/full_sweep_20bots.csv
echo "t0,forecast_mode,kernel,nu,length_scale,noise_upper,RMSE,ACC,HF,FSS,time" > $OUT
JOBS=$(mktemp)
for t0 in 4 66 141 164 231; do
  for fm in fno persistence none; do
    for kernel_nu in "matern 0.5" "matern 1.5" "matern 2.5" "rbf 1.5"; do
      kernel=$(echo $kernel_nu | cut -d' ' -f1)
      nu=$(echo $kernel_nu | cut -d' ' -f2)
      for ls in 0.02 0.05 0.10 0.20 0.30 0.50; do
        echo "$t0|$fm|$kernel|$nu|$ls"
      done
    done
  done
done > $JOBS

cat $JOBS | xargs -I {} -P 32 -d '\n' bash -c '
  IFS="|" read -r t0 fm kernel nu ls <<< "$1"
  OMP_NUM_THREADS=4 python scripts/run_gp_sweep.py \
    --t0 $t0 --n_robots 20 --forecast_mode $fm \
    --gp_kernel $kernel --gp_matern_nu $nu \
    --gp_length_scale_init $ls --gp_noise_upper 0.01 \
    >> '$OUT' 2>/dev/null
' _ {}
echo "All done"

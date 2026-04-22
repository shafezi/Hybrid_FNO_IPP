#!/bin/bash
# RMSE-optimal hyperparameters per (robot count, method)
# format: nrobots,method,fcast_mode,policy,kernel,nu,ls,tag

JOBS=$(mktemp)

# t0s to run
TS="4 66 141 164 231"

# Format: nbots,fm,kernel,nu,ls,tag (6 fields)
RMSE_CONFIG="
5,fno,matern,0.5,0.10,fnogp_rmseopt
5,persistence,matern,0.5,0.10,perpgp_rmseopt
5,none,matern,0.5,0.02,gponly_rmseopt
10,fno,matern,0.5,0.10,fnogp_rmseopt
10,persistence,matern,1.5,0.20,perpgp_rmseopt
10,none,matern,0.5,0.02,gponly_rmseopt
20,fno,matern,1.5,0.02,fnogp_rmseopt
20,persistence,matern,2.5,0.05,perpgp_rmseopt
20,none,rbf,1.5,0.30,gponly_rmseopt
40,fno,rbf,1.5,0.05,fnogp_rmseopt
40,persistence,matern,1.5,0.30,perpgp_rmseopt
40,none,matern,2.5,0.20,gponly_rmseopt
"

ACC_CONFIG="
5,fno,matern,2.5,0.05,fnogp_accopt
5,persistence,matern,1.5,0.30,perpgp_accopt
5,none,matern,0.5,0.02,gponly_accopt
10,fno,matern,0.5,0.10,fnogp_accopt
10,persistence,rbf,1.5,0.10,perpgp_accopt
10,none,matern,2.5,0.05,gponly_accopt
20,fno,rbf,1.5,0.10,fnogp_accopt
20,persistence,rbf,1.5,0.05,perpgp_accopt
20,none,rbf,1.5,0.30,gponly_accopt
40,fno,rbf,1.5,0.05,fnogp_accopt
40,persistence,rbf,1.5,0.30,perpgp_accopt
40,none,matern,2.5,0.20,gponly_accopt
"

M2_CONFIG="
5,fno,matern,2.5,0.02,fnogp_m2opt
5,persistence,matern,0.5,0.10,perpgp_m2opt
5,none,matern,1.5,0.05,gponly_m2opt
10,fno,matern,0.5,0.02,fnogp_m2opt
10,persistence,matern,1.5,0.20,perpgp_m2opt
10,none,matern,2.5,0.05,gponly_m2opt
20,fno,matern,2.5,0.05,fnogp_m2opt
20,persistence,matern,0.5,0.50,perpgp_m2opt
20,none,rbf,1.5,0.05,gponly_m2opt
40,fno,rbf,1.5,0.05,fnogp_m2opt
40,persistence,matern,1.5,0.30,perpgp_m2opt
40,none,rbf,1.5,0.30,gponly_m2opt
"

(echo "$RMSE_CONFIG"; echo "$ACC_CONFIG"; echo "$M2_CONFIG") | while IFS=',' read -r nbots fm kernel nu ls tag; do
  [ -z "$nbots" ] && continue
  for t0 in $TS; do
    echo "$t0|$nbots|$fm|$kernel|$nu|$ls|$tag"
  done
done > $JOBS

TOTAL=$(wc -l < $JOBS)
echo "Total jobs: $TOTAL"

cat $JOBS | xargs -I {} -P 24 -d '\n' bash -c '
  IFS="|" read -r t0 nbots fm kernel nu ls tag <<< "$1"
  OMP_NUM_THREADS=4 python scripts/run_psd_at_optimal.py \
    --t0 $t0 --n_robots $nbots --forecast_mode $fm \
    --use_policy --gp_kernel $kernel --gp_matern_nu $nu \
    --gp_length_scale_init $ls --tag $tag 2>/dev/null
' _ {}

echo "All done"

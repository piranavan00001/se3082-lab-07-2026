#!/bin/bash
for np in 2 4 8; do
  echo "================ NP = $np ================"
  for prog in sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan; do
    echo "--- $prog (np=$np) ---"
    mpirun --oversubscribe -np $np ./$prog | grep -E "Correct|Time"
  done
done

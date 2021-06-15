#!/bin/bash

if [[ -z "$POTANET_ROOT_DIR" ]]; then
  echo "POTANET_ROOT_DIR environment variable not set"
  echo "Please edit env.sh and run \`source env.sh\`"
  exit 1
fi

run_id=$RANDOM

cd modules

for i in 4 # 3 2
do

    for j in "none" # "baseline" "baseline_smoothing" "all"
    do

        python isotopenet_cv_training_performances.py --id $run_id --epochs 100 --kfold 4 --model base --classes $i --preprocessing $j

    done
done

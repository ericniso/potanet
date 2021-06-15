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
    for j in "none" "baseline" "baseline_smoothing" "all"
    do
        python isotopenet_roi_plot.py \
            --model /data/output/isotopenet_training/base/classes_$i/preprocess_$j/isotopenet.h5 \
            --config /data/output/isotopenet_training/base/classes_$i/preprocess_$j/isotopenet_config.joblib \
            --dinfo /data/output/isotopenet_training/base/classes_$i/preprocess_$j/dataset_config.joblib \
            --id $run_id
    done
done
#!/bin/bash

run_id=$RANDOM

cd model

for i in 4 # 3 2
do
    for j in "none" "all" "baseline" "baseline_smoothing"
    do
        python analyse_grad_cam.py \
            --rootdir /data/output/isotopenet_grad_cam/base/classes_$i/preprocess_$j \
            --rid $run_id
    done
done

#!/bin/bash

run_id=$RANDOM

cd model

for i in 4 # 3 2
do
    for j in "none" "all" "baseline" "baseline_smoothing" # "none" "all" "baseline" "baseline_smoothing"
    do
        for t in "training" "validation"
        do
            python isotopenet_grad_cam.py \
                --model /data/output/isotopenet_training/base/classes_$i/preprocess_$j/isotopenet.h5 \
                --config /data/output/isotopenet_training/base/classes_$i/preprocess_$j/isotopenet_config.joblib \
                --type $t \
                --dinfo /data/output/isotopenet_training/base/classes_$i/preprocess_$j/dataset_config.joblib \
                --rid $run_id
        done
    done
done
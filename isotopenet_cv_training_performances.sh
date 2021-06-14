#!/bin/bash

run_id=$RANDOM

cd modules

for i in 4 # 3 2
do

    for j in "none" # "baseline" "baseline_smoothing" "all"
    do

        python isotopenet_cv_training_performances.py --id $run_id --epochs 100 --kfold 4 --model base --classes $i --preprocessing $j

    done
done

#!/bin/bash

run_id=$RANDOM

cd modules

for i in 4 # 3 2
do

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i # base, no preprocessing

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i --preprocessing all # base, all preprocessing

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i --preprocessing baseline # base, baseline preprocessing

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i --preprocessing baseline_smoothing # base, baseline_smoothing preprocessing

done

#!/bin/bash

if [[ ! -z "$POTANET_ROOT_DIR" ]]; then
  echo "POTANET_ROOT_DIR environment variable not set"
  echo "Please edit env.sh and \`source env.sh\`"
  exit 1
fi

run_id=$RANDOM

cd modules

for i in 4 # 3 2
do

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i # base, no preprocessing

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i --preprocessing all # base, all preprocessing

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i --preprocessing baseline # base, baseline preprocessing

    python isotopenet_training.py --id $run_id --epochs 70 --model base --classes $i --preprocessing baseline_smoothing # base, baseline_smoothing preprocessing

done

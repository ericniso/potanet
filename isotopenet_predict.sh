#!/bin/bash

# run_id=$RANDOM

isotopenet=/data/isotopenet_models/isotopenet_training_run_28900_avg_0_005/base/classes_4/preprocess_none

cd model

python isotopenet_predict.py \
  --model $isotopenet/isotopenet.h5 \
  --config $isotopenet/isotopenet_config.joblib \
  --dinfo $isotopenet/dataset_config.joblib \
  --spectra /data/imzML_roi_single_spectra_raw/validation/1005.imzML

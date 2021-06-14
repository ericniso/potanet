#!/bin/bash

cd model

ROOT_DIR="/data"
SAMPLES_KIND="tesi"
IMZML_ROI_ROOT_DIR="$ROOT_DIR/imzML_roi_single_spectra_raw"
IMZML_ROI_EXTRACTED_ROOT_DIR="$ROOT_DIR/imzML_roi_single_spectra_extracted"

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR"

echo "----- Training"

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR/training"

INPUT="samples/$SAMPLES_KIND/training.txt"

while IFS= read -r P
do
  if [ -f "$IMZML_ROI_ROOT_DIR/training/$P.imzML" ] && [ -f "$IMZML_ROI_ROOT_DIR/training/$P.ibd" ]; then
    # echo $P
    python3 parser.py "$IMZML_ROI_ROOT_DIR/training/$P.imzML" "$IMZML_ROI_EXTRACTED_ROOT_DIR/training/$P"
  fi
done < "$INPUT"

echo "----- Validation"

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation"

INPUT="samples/$SAMPLES_KIND/validation.txt"

while IFS= read -r P
do
  if [ -f "$IMZML_ROI_ROOT_DIR/validation/$P.imzML" ] && [ -f "$IMZML_ROI_ROOT_DIR/validation/$P.ibd" ]; then
    # echo $P
    python3 parser.py "$IMZML_ROI_ROOT_DIR/validation/$P.imzML" "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation/$P"
  fi
done < "$INPUT"

echo "----- Validation exvivo"

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation_exvivo"

INPUT="samples/$SAMPLES_KIND/validation_exvivo.txt"

while IFS= read -r P
do
  if [ -f "$IMZML_ROI_ROOT_DIR/validation_exvivo/$P.imzML" ] && [ -f "$IMZML_ROI_ROOT_DIR/validation_exvivo/$P.ibd" ]; then
    # echo $P
    python3 parser.py "$IMZML_ROI_ROOT_DIR/validation_exvivo/$P.imzML" "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation_exvivo/$P"
  fi
done < "$INPUT"

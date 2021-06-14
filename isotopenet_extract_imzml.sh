#!/bin/bash

if [[ ! -z "$POTANET_ROOT_DIR" ]]; then
  echo "POTANET_ROOT_DIR environment variable not set"
  echo "Please edit env.sh and \`source env.sh\`"
  exit 1
fi

cd modules

ROOT_DIR=$POTANET_ROOT_DIR
SAMPLES_KIND="tesi" # tesi airc_thyroid_paper
IMZML_ROI_ROOT_DIR="$ROOT_DIR/imzML_roi_single_spectra_raw"
IMZML_ROI_EXTRACTED_ROOT_DIR="$ROOT_DIR/imzML_roi_single_spectra_extracted"

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR"

# ----- Training

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR/training"

INPUT="samples/$SAMPLES_KIND/training.txt"

while IFS= read -r P
do
  if [ -f "$IMZML_ROI_ROOT_DIR/training/$P.imzML" ] && [ -f "$IMZML_ROI_ROOT_DIR/training/$P.ibd" ]; then
    python3 parser.py "$IMZML_ROI_ROOT_DIR/training/$P.imzML" "$IMZML_ROI_EXTRACTED_ROOT_DIR/training/$P"
  fi
done < "$INPUT"

# ----- Validation

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation"

INPUT="samples/$SAMPLES_KIND/validation.txt"

while IFS= read -r P
do
  if [ -f "$IMZML_ROI_ROOT_DIR/validation/$P.imzML" ] && [ -f "$IMZML_ROI_ROOT_DIR/validation/$P.ibd" ]; then
    python3 parser.py "$IMZML_ROI_ROOT_DIR/validation/$P.imzML" "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation/$P"
  fi
done < "$INPUT"

# ----- Validation exvivo

mkdir -p "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation_exvivo"

INPUT="samples/$SAMPLES_KIND/validation_exvivo.txt"

while IFS= read -r P
do
  if [ -f "$IMZML_ROI_ROOT_DIR/validation_exvivo/$P.imzML" ] && [ -f "$IMZML_ROI_ROOT_DIR/validation_exvivo/$P.ibd" ]; then
    python3 parser.py "$IMZML_ROI_ROOT_DIR/validation_exvivo/$P.imzML" "$IMZML_ROI_EXTRACTED_ROOT_DIR/validation_exvivo/$P"
  fi
done < "$INPUT"

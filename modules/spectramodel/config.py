import os
from logger import potanet_logger

if not "POTANET_ROOT_DIR" in os.environ:
    potanet_logger.error("POTANET_ROOT_DIR missing, it must be set as an environment variable")
    exit(1)

if not "POTANET_IMZML_RAW_ROOT_DIR" in os.environ:
    potanet_logger.error("POTANET_IMZML_RAW_ROOT_DIR missing, it must be set as an environment variable")
    exit(1)

if not "POTANET_IMZML_EXTRACTED_ROOT_DIR" in os.environ:
    potanet_logger.error("POTANET_IMZML_EXTRACTED_ROOT_DIR missing, it must be set as an environment variable")
    exit(1)

if not "POTANET_SAMPLES_TYPE" in os.environ:
    potanet_logger.error("POTANET_SAMPLES_TYPE missing, it must be set as an environment variable")
    exit(1)

if not "POTANET_PREPROCESSING_BASELINE_MEDIAN" in os.environ:
    potanet_logger.error("POTANET_PREPROCESSING_BASELINE_MEDIAN missing, it must be set as an environment variable")
    exit(1)

if not "POTANET_PREPROCESSING_SMOOTHING_MOVING_AVERAGE" in os.environ:
    potanet_logger.error(
        "POTANET_PREPROCESSING_SMOOTHING_MOVING_AVERAGE missing, it must be set as an environment variable")
    exit(1)

if not "POTANET_PREPROCESSING_NOISE_THRESHOLD" in os.environ:
    potanet_logger.error(
        "POTANET_PREPROCESSING_NOISE_THRESHOLD missing, it must be set as an environment variable")
    exit(1)

if not "POTANET_SPECTRUM_SIZE" in os.environ:
    potanet_logger.error(
        "POTANET_SPECTRUM_SIZE missing, it must be set as an environment variable")
    exit(1)

POTANET_ROOT_DIR = os.environ.get("POTANET_ROOT_DIR")

POTANET_IMZML_RAW_ROOT_DIR = os.environ.get("POTANET_IMZML_RAW_ROOT_DIR")

POTANET_IMZML_EXTRACTED_ROOT_DIR = os.environ.get("POTANET_IMZML_EXTRACTED_ROOT_DIR")

POTANET_THREAD_POOL_SIZE = int(os.environ.get("POTANET_THREAD_POOL_SIZE", 10))

POTANET_SAMPLES_TYPE = os.environ.get("POTANET_SAMPLES_TYPE")

POTANET_SAMPLES_DIR = "samples"

POTANET_PREPROCESSING_BASELINE_MEDIAN = int(os.environ.get("POTANET_PREPROCESSING_BASELINE_MEDIAN"))

POTANET_PREPROCESSING_SMOOTHING_MOVING_AVERAGE = int(os.environ.get("POTANET_PREPROCESSING_SMOOTHING_MOVING_AVERAGE"))

POTANET_PREPROCESSING_NOISE_THRESHOLD = float(os.environ.get("POTANET_PREPROCESSING_NOISE_THRESHOLD"))

POTANET_SPECTRUM_SIZE = int(os.environ.get("POTANET_SPECTRUM_SIZE"))

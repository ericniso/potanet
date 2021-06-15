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

POTANET_ROOT_DIR = os.environ.get("POTANET_ROOT_DIR")

POTANET_IMZML_RAW_ROOT_DIR = os.environ.get("POTANET_IMZML_RAW_ROOT_DIR")

POTANET_IMZML_EXTRACTED_ROOT_DIR = os.environ.get("POTANET_IMZML_EXTRACTED_ROOT_DIR")

POTANET_THREAD_POOL_SIZE = int(os.environ.get("POTANET_THREAD_POOL_SIZE", 10))

POTANET_SAMPLES_TYPE = os.environ.get("POTANET_SAMPLES_TYPE")

POTANET_SAMPLES_DIR = "samples"

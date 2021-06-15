import os
import logging
from pathlib import Path
from logging import config

os.makedirs('logs', exist_ok=True)

logging.config.fileConfig(Path(__file__).parent / 'logging.ini')

potanet_logger = logging.getLogger('root')

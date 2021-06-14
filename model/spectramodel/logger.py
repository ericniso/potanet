import os
import logging
from logging import config

os.makedirs('logs', exist_ok=True)

logging.config.fileConfig('logging.ini')

spectra_logger = logging.getLogger('root')

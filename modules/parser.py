import time
import config
import warnings
import sys
import os
import argparse
import numpy as np
from logger import potanet_logger
from pyimzml.ImzMLParser import ImzMLParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class ParserWorker:

    def __init__(self, path_raw, path_extracted):

        self.path_raw = path_raw
        self.path_extracted = path_extracted

    def parse(self, sample):

        sample_kind, sample_id = sample

        loaded_data = self.__load_data__(sample_kind, sample_id)
        self.__save_data__(sample_kind, sample_id, loaded_data)

        potanet_logger.info("Processed {}/{}".format(sample_kind, sample_id))

        return sample

    def __load_data__(self, sample_kind, sample_id):

        parser = ImzMLParser(self.path_raw / sample_kind / "{}.imzML".format(sample_id))

        coords, masses, spectra = [], [], []

        for idx, idx_coords in enumerate(parser.coordinates):
            mzs, intensities = parser.getspectrum(idx)

            coords.append(list(idx_coords))
            masses.append(mzs)
            spectra.append(intensities)

        coords = np.array(coords, dtype=np.int)
        masses = np.array(masses, dtype=np.float)
        spectra = np.array(spectra, dtype=np.float)

        return coords, masses, spectra

    def __save_data__(self, sample_kind, sample_id, data):

        sample_id_root_dir = self.path_extracted / sample_kind / sample_id

        if sample_id_root_dir.exists():
            os.rmdir(sample_id_root_dir)

        os.makedirs(sample_id_root_dir)

        coords, masses, spectra = data

        np.savetxt(sample_id_root_dir / "coordinates.txt", coords, fmt="%i")
        np.savetxt(sample_id_root_dir / "masses.txt", masses)
        np.savetxt(sample_id_root_dir / "intensities.txt", spectra)


if __name__ == "__main__":

    execution_start_time = time.perf_counter()

    imzml_raw = Path(config.POTANET_IMZML_RAW_ROOT_DIR)
    imzml_extracted = Path(config.POTANET_IMZML_EXTRACTED_ROOT_DIR)
    # TODO: remove ".."
    imzml_samples_list_dir = Path(__file__).parent / ".." / config.POTANET_SAMPLES_DIR / config.POTANET_SAMPLES_TYPE

    os.makedirs(imzml_extracted, exist_ok=True)

    samples_kind_id = []

    potanet_logger.info("Reading sample ids list")

    for sample_type in imzml_samples_list_dir.iterdir():
        if sample_type.is_file():
            sample_kind = sample_type.stem

            with open(sample_type) as samples:
                for sample_id in samples:
                    samples_kind_id.append((sample_kind.strip(), sample_id.strip()))

    potanet_logger.info("Starting data extraction with {} workers".format(config.POTANET_THREAD_POOL_SIZE))

    with ThreadPoolExecutor(max_workers=config.POTANET_THREAD_POOL_SIZE) as executor:

        worker = ParserWorker(imzml_raw, imzml_extracted)
        result = executor.map(worker.parse, samples_kind_id)

    execution_end_time = time.perf_counter()

    potanet_logger.info("Extraction complete in {}s".format(execution_end_time - execution_start_time))

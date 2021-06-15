import config
import warnings
import sys
import os
import argparse
import numpy as np
from logger import potanet_logger
from pyimzml.ImzMLParser import ImzMLParser
from pathlib import Path


def load_data(path):

    parser = ImzMLParser(path)
    
    coords = []
    masses = []
    spectra = []

    for idx, (x, y, z) in enumerate(parser.coordinates):
        mzs, intensities = parser.getspectrum(idx)

        coords.append([x, y, z])
        masses.append(mzs)
        spectra.append(intensities)

    coords = np.array(coords, dtype=np.int)
    masses = np.array(masses, dtype=np.float)
    spectra = np.array(spectra, dtype=np.float)

    return coords, masses, spectra


def save_data(path, coords, masses, spectra):

    coords_path = path / "coordinates.txt"
    masses_path = path / "mzs.txt"
    spectra_path = path / "intensities.txt"

    coords = coords.astype(np.int)

    np.savetxt(coords_path, coords, fmt="%i")
    np.savetxt(masses_path, masses)
    np.savetxt(spectra_path, spectra)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("imzml", type=str, help="")
    parser.add_argument("outdir", type=str, help="")
    args = parser.parse_args()

    imzml_path = Path(args.imzml)

    if not imzml_path.exists():
        potanet_logger.error("Invalid .imzml path: {!r}".format(imzml_path))
        exit(1)

    if not imzml_path.is_file():
        potanet_logger.error("Invalid .imzml filename: {!r}".format(imzml_path))
        exit(1)

    output_folder = Path(args.outdir)

    if not output_folder.exists():
        potanet_logger.info("Output folder doesn't exist, creating")
        os.makedirs(output_folder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        potanet_logger.info("Extracting data from {!r}".format(imzml_path))
        coords, masses, spectra = load_data(imzml_path)

    coords = np.array(coords, dtype=np.int)
    masses = np.array(masses, dtype=np.float)
    spectra = np.array(spectra, dtype=np.float)

    potanet_logger.info("Saving data to {!r}".format(output_folder))
    save_data(output_folder, coords, masses, spectra)

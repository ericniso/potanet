import warnings
import sys
import os
import argparse
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pathlib import Path

def load_data(path):

    parser = ImzMLParser(path)
    
    coords = []
    masses = []
    spectra = []

    for idx, (x,y,z) in enumerate(parser.coordinates):
        mzs, intensities = parser.getspectrum(idx)

        coords.append([x, y, z])
        masses.append(mzs)
        spectra.append(intensities)

    coords = np.array(coords, dtype=np.int)
    masses = np.array(masses, dtype=np.float)
    spectra = np.array(spectra, dtype=np.float)

    return coords, masses, spectra

def save_data(path, coords, masses, spectra, target_size=8000):

    coords_path = path / 'coordinates.txt'
    masses_path = path / 'mzs.txt'
    spectra_path = path / 'intensities.txt'

    if masses.shape[1] < target_size:

        new_masses = np.zeros((masses.shape[0], target_size), dtype=masses.dtype)
        new_masses[:, 0:masses.shape[1]] = masses

        masses = new_masses

    else:

        masses = masses[:, 0:target_size]

    if spectra.shape[1] < target_size:

        new_spectra = np.zeros((spectra.shape[0], target_size), dtype=spectra.dtype)
        new_spectra[:, 0:spectra.shape[1]] = spectra

        spectra = new_spectra

    else:

        spectra = spectra[:, 0:target_size]

    coords = coords.astype(np.int)

    np.savetxt(coords_path, coords, fmt='%i')
    np.savetxt(masses_path, masses)
    np.savetxt(spectra_path, spectra)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('imzml', type=str, help='')
    parser.add_argument('outdir', type=str, help='')
    args = parser.parse_args()

    imzml_path = Path(args.imzml)

    if not imzml_path.exists():
        print('ERROR: Invalid .imzml path')
        exit(1)

    if not imzml_path.is_file():
        print('ERROR: Invalid .imzML filename')
        exit(1)

    output_folder = Path(args.outdir)

    if not output_folder.exists():
        os.makedirs(output_folder)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coords, masses, spectra = load_data(imzml_path)

    coords = np.array(coords, dtype=np.int)
    masses = np.array(masses, dtype=np.float)
    spectra = np.array(spectra, dtype=np.float)

    save_data(output_folder, coords, masses, spectra)

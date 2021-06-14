import sys
import os
import numpy as np
import argparse
from pyimzml.ImzMLParser import ImzMLParser
from pathlib import Path

def load_data(path):

    parser = ImzMLParser(str(path))
    
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('imzml', type=str, help='')
    args = parser.parse_args()

    imzml_path = Path(args.imzml)

    if not imzml_path.exists():
        print('ERROR: Invalid .imzml path')
        exit(1)

    if not imzml_path.is_file():
        print('ERROR: Invalid .imzML filename')
        exit(1)

    coords, masses, spectra = load_data(imzml_path)

    print('Path: {}'.format(imzml_path))
    print('Spectra: {}'.format(spectra.shape))
    print('Masses: {}'.format(masses.shape))
    print('Coordinates: {}'.format(coords.shape))
    print()
    print('Intensities boundaries: [{}, {}]'.format(np.min(spectra), np.max(spectra)))
    print('Mass boundaries: [{}, {}]'.format(np.min(masses), np.max(masses)))

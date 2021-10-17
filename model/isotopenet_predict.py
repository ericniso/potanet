from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import joblib
import warnings
import numpy as np
from pathlib import Path
from spectraio import Files
from spectramodel import IsotopeNet
from pyimzml.ImzMLParser import ImzMLParser

def get_neighbours(coords):

    x = coords[0]
    y = coords[1]

    return np.array([
        [x - 1, y - 1, 1],
        [x, y - 1, 1],
        [x + 1, y - 1, 1],
        [x + 1, y, 1],
        [x + 1, y + 1, 1],
        [x, y + 1, 1],
        [x - 1, y + 1, 1],
        [x - 1, y, 1],
    ])

def load_data(path, target_size):
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

    return coords, masses, spectra


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', type=str, help='')
    parser.add_argument('-c', '--config', type=str, help='')
    parser.add_argument('--dinfo', type=str, help='')
    parser.add_argument('-s', '--spectra', type=str, help='')
    args = parser.parse_args()

    if (not args.model) or (not args.config) or (not args.dinfo):
        print('error: model and corresponding config required together')
        exit(1)

    if (not Path(args.model).exists()) or (not Path(args.config).exists()) or (not Path(args.dinfo).exists()):
        print('error: model or config does not exist')
        exit(1)

    files = Files()

    model_trained_on_dataset_info = joblib.load(Path(args.dinfo))
    config = joblib.load(Path(args.config))
    model = IsotopeNet(lr=config['lr'], model_type=config['type'], n_classes=config['n_classes'])
    model.load(Path(args.model))
    model.compile()

    if config['preprocessing'] == 'all':
        preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=5, normalize_tic=True)

    if config['preprocessing'] == 'baseline':
        preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=None, normalize_tic=None)

    if config['preprocessing'] == 'baseline_smoothing':
        preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=5, normalize_tic=None)

    if config['preprocessing'] == 'none':
        preprocessing = None

    print('Model trained with: ', model_trained_on_dataset_info)
    print('Model: {}, Classes: {}, Preprocessing: {}'.format(config['type'], config['n_classes'], config['preprocessing']))

    # Load data
    imzml_path = Path(args.spectra)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coords, masses, spectra = load_data(imzml_path, model.spectra_size)

    # Average spectra method if needed
    if (model_trained_on_dataset_info["type"] == "avg"):
        avg_spectra = []
        for i in range(spectra.shape[0]):
            current_spectra = spectra[i]
            current_neighbours_spectrum = [spectra[i]]
            current_neighbours_coords = get_neighbours(coords[i])
            for j in range(spectra.shape[0]):
                for k in range(current_neighbours_coords.shape[0]):
                    if (coords[j] == current_neighbours_coords[k]).all():
                        current_neighbours_spectrum.append(spectra[j])

            avg_spectrum = np.array(current_neighbours_spectrum)
            avg_spectrum = np.average(avg_spectrum, axis=0)

            avg_spectra.append(avg_spectrum)

        avg_spectra = np.array(avg_spectra)
        spectra = avg_spectra

    # Preprocessing
    preprocessed_spectra = []
    for i in range(spectra.shape[0]):
        spectrum = spectra[i]
        if preprocessing is not None:
            spectrum = preprocessing(spectrum)

        spectrum = np.reshape(spectrum, (spectrum.shape[0], 1, 1))
        preprocessed_spectra.append(spectrum)

    preprocessed_spectra = np.array(preprocessed_spectra)

    predicted_result = model.raw_predict(preprocessed_spectra)
    print(predicted_result)

    # Samples diagnosis
    # 0: HP
    # 1: PTC
    # 2: Noise
    # 3: HT
    # 4: niftp

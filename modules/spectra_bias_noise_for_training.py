import sys
import os
import argparse
import shutil
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from spectraio import Files
from spectramodel import SpectraNormalizer

if __name__ == '__main__':
    
    BASELINE_MEDIAN = 5
    SMOOTHING_MOVING_AVERAGE = 5

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('type', type=str) # single, avg
    parser.add_argument('threshold', type=float) # 0.005
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.type not in ['single', 'avg']:
        print('ERROR: type arg must be \'single\' or \'avg\'')
        exit(1)

    if args.threshold < 0:
        print('ERROR: threshold must be >= 0')
        exit(1)

    files = Files()
    
    threshold = args.threshold

    if args.type == 'single':
        dataset = pd.read_csv(files.roi_single_spectra_path_data_single) # Single
        intensities_dir = files.roi_single_spectra_path_intensities_single
        mzs_dir = files.roi_single_spectra_path_mzs_single
        coordinates_dir = files.roi_single_spectra_path_coordinates_single

    if args.type == 'avg':
        dataset = pd.read_csv(files.roi_single_spectra_path_data_avg) # Average
        intensities_dir = files.roi_single_spectra_path_intensities_avg
        mzs_dir = files.roi_single_spectra_path_mzs_avg
        coordinates_dir = files.roi_single_spectra_path_coordinates_avg

    out_dir = files.roi_single_spectra_path_root / '_{}_noise_{}'.format(args.type, str(threshold).replace('.', '_'))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    os.makedirs(out_dir)

    shutil.copytree(intensities_dir, out_dir / 'intensities')
    shutil.copytree(mzs_dir, out_dir / 'mzs')
    shutil.copytree(coordinates_dir, out_dir / 'coordinates')

    diagnosis_index = list(dataset.columns).index('diagnosis')

    hp_to_noise = 0
    ptc_to_noise = 0
    ht_to_noise = 0

    for i in tqdm(range(dataset.shape[0])):

        spectrum = np.loadtxt(intensities_dir / '{}.txt'.format(dataset.iloc[i]['spectrum']))
        normalizer = SpectraNormalizer(spectrum)
        
        normalizer.baseline_median(BASELINE_MEDIAN)
        normalizer.smoothing_moving_average(SMOOTHING_MOVING_AVERAGE)
        normalizer.normalize_tic()
        
        spectrum = normalizer.get()
        below_threshold = np.argwhere(spectrum > threshold).shape[0] == 0
        
        if below_threshold:
            if dataset.iloc[i]['diagnosis'] == 0:
                hp_to_noise += 1
            
            if dataset.iloc[i]['diagnosis'] == 1:
                ptc_to_noise += 1

            if dataset.iloc[i]['diagnosis'] == 3:
                ht_to_noise += 1

            dataset.iat[i, diagnosis_index] = 2

    dataset.to_csv(out_dir / 'spectra.csv', index=False)

    config = {
        'type': args.type,
        'threshold': args.threshold,
        'total': dataset.shape[0],
        'HP to Noise': hp_to_noise,
        'PTC to Noise': ptc_to_noise,
        'HT to Noise': ht_to_noise
    }
    joblib.dump(config, out_dir / 'config.joblib')

    if args.verbose:
        print('-- Total: {}'.format(dataset.shape[0]))
        print('-- HP to Noise: {}'.format(hp_to_noise))
        print('-- PTC to Noise: {}'.format(ptc_to_noise))
        print('-- HT to Noise: {}'.format(ht_to_noise))

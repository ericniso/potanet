from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pandas as pd
import numpy as np
import joblib
import datetime
from tqdm import tqdm
from pathlib import Path
from spectraio import Files, SpectraDrawing, balanced_dataset
from spectramodel import csv_loader
from spectramodel import IsotopeNet, SpectraPredictSequence

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', type=str, help='')
    parser.add_argument('-c', '--config', type=str, help='')
    parser.add_argument('--dinfo', type=str, help='')
    parser.add_argument('-i', '--id', type=int, help='')
    args = parser.parse_args()

    if (not args.model) or (not args.config) or (not args.dinfo):
        print('error: model and corresponding config required together')
        exit(1)

    if (not Path(args.model).exists()) or (not Path(args.config).exists()) or (not Path(args.dinfo).exists()):
        print('error: model or config does not exist')
        exit(1)

    files = Files()

    dataset_config = joblib.load(files.roi_single_spectra_path / 'config.joblib')
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

    today = datetime.datetime.today()
    outdir = files.out_path / 'spectra_plot_{}_{}_{}'.format(args.id,  dataset_config['type'], str(dataset_config['threshold']).replace('.', '_')) / config['type'] / 'classes_{}'.format(config['n_classes']) / 'preprocess_{}'.format(config['preprocessing'])

    if not outdir.exists():
        os.makedirs(outdir)

    joblib.dump(dataset_config, outdir / 'dataset_config.joblib')

    loader = csv_loader(files.roi_single_spectra_path)
    dataset = pd.read_csv(files.roi_single_spectra_path_data)

    print('Model trained with: ', model_trained_on_dataset_info)
    print()
    print('Testing now on dataset: ', dataset_config)
    print()
    print('Dataset: {}, Model: {}, Classes: {}, Preprocessing: {}'.format(dataset_config['type'], config['type'], config['n_classes'], config['preprocessing']))
    print()

    drawing = SpectraDrawing()

    classes_4_training_dataset = balanced_dataset(dataset, 4)
    classes_3_training_dataset = balanced_dataset(dataset, 3)
    classes_2_training_dataset = balanced_dataset(dataset, 2)
    
    exclude_patients = list(classes_4_training_dataset['patient'].unique())
    for p in (classes_3_training_dataset['patient'].unique()):
        exclude_patients.append(p)
    for p in (classes_2_training_dataset['patient'].unique()):
        exclude_patients.append(p)

    exclude_patients = list(set(exclude_patients))
    include_patients = [p for p in list(dataset['patient'].unique()) if p not in exclude_patients]

    classes = [{ 'name': 'HP', 'color': (0, 255 / 255, 0, 1) }, { 'name': 'PTC', 'color': (255 / 255, 0, 0, 1) }]

    if config['n_classes'] > 2:
        classes.append({ 'name': 'Noise', 'color': (220 / 255, 220 / 255, 220 / 255, 1) })
    
    if config['n_classes'] > 3:
        classes.append({ 'name': 'HT', 'color': (255 / 255, 255 / 255, 0, 1) })

    for n, l in [('training', exclude_patients), ('validation', include_patients)]:

        print()
        print(n)
        print()

        for p in tqdm(l):

            if not (outdir / n / p).exists():
                os.makedirs(outdir / n / p)

            current_patient = dataset[dataset['patient'] == p]

            diagnosis = current_patient['diagnosis'].values
            diagnosis = np.array(diagnosis, dtype=np.uint8)
            coordinates = []

            predict_generator = SpectraPredictSequence(current_patient, 
                model.csv_loader(files.roi_single_spectra_path),
                preprocessing=preprocessing)

            diagnosis = model.predict(predict_generator.all(), {})

            for i in range(current_patient.shape[0]):
                coordinates.append(loader['coordinates'](current_patient.iloc[i]))

            coordinates = np.array(coordinates)
            drawing.draw(coordinates, np.argmax(diagnosis, axis=1).astype(np.int8), (outdir / n / p / '{}.png'.format(p)))
            drawing.plot_probs(diagnosis, outdir / n / p, classes)

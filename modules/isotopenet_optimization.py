from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import joblib
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from spectraio import Files, SpectraPlot
from spectraio import balanced_dataset
from spectramodel import IsotopeNet, SpectraSequence, exp_decay, SpectraAugmenter, augment_dataset
from imblearn.keras import BalancedBatchGenerator
from tensorflow.keras import callbacks

if __name__ == '__main__':

    exit(0)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--id', type=int, help='')

    args = parser.parse_args()

    if args.id and args.id < 0:
        print('error: id must be a positive integer')
        exit(1)

    files = Files()

    batch_size = 64
    epochs = 100
    learning_rate = 1e-3
    decay_factor = 0.1
    min_learning_rate = 1e-7

    n_classes = 4
    use_preprocessing = False
    model_type = 'base'
    n_folds = 5

    dataset = pd.read_csv(files.roi_single_spectra_path_data)
    train_data = balanced_dataset(dataset, n_classes)

    isotopenet_callbacks = [
        callbacks.LearningRateScheduler(
            exp_decay(learning_rate, decay_factor))
    ]

    cv_results = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for train_index, validation_index in skf.split(np.zeros((train_data.shape[0], )), train_data['diagnosis'].values):
        
        model = IsotopeNet(lr=learning_rate, model_type=model_type, n_classes=n_classes)
        model.compile()

        if use_preprocessing:
            preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=5, normalize_tic=True)
        else:
            preprocessing = None

        # class_weights = class_weight.compute_class_weight('balanced', 
        #     classes=train_data.iloc[train_index]['diagnosis'].unique(), 
        #     y=train_data.iloc[train_index]['diagnosis'].values)

        # class_weights_dict = {i : class_weights[i] for i in range(n_classes)}

        train_generator = SpectraSequence(train_data.iloc[train_index], 
            model.csv_loader(files.roi_single_spectra_path), 
            batch_size=batch_size, 
            preprocessing=preprocessing,
            n_classes=n_classes)
        validation_generator = SpectraSequence(train_data.iloc[validation_index], 
            model.csv_loader(files.roi_single_spectra_path), 
            batch_size=batch_size, 
            preprocessing=preprocessing,
            n_classes=n_classes)

        train_config = dict(
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=isotopenet_callbacks 
        )
        
        result = model.train(train_generator, train_config)
        evaluation_config = dict()
        evaluation_results = model.evaluate(validation_generator, evaluation_config)
        cv_results.append(evaluation_results)

    cv_results = np.mean(np.array(cv_results), axis=0)


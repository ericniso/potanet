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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import class_weight
from spectraio import Files, SpectraPlot
from spectraio import balanced_dataset
from spectramodel import IsotopeNet, SpectraSequence, exp_decay, SpectraAugmenter, augment_dataset
from imblearn.keras import BalancedBatchGenerator
from tensorflow.keras import callbacks

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='')
    parser.add_argument('-k', '--kfold', type=int, default=4, help='')
    parser.add_argument('-m', '--model', type=str, default='base', help='')
    parser.add_argument('-c', '--classes', type=int, default=4, help='')
    parser.add_argument('-p', '--preprocessing', type=str, default='none', help='')
    parser.add_argument('-i', '--id', type=int, help='')

    args = parser.parse_args()

    if args.id and args.id < 0:
        print('error: id must be a positive integer')
        exit(1)

    if args.epochs < 1:
        print('error: epochs must be greater than 0')
        exit(1)

    if args.kfold < 1:
        print('error: kfold must be greater than 0')
        exit(1)

    if args.classes < 2 or args.classes > 4:
        print('error: classes must be 2, 3 or 4')
        exit(1)

    files = Files()

    batch_size = 64
    epochs = args.epochs
    learning_rate = 1e-3
    decay_factor = 0.1
    min_learning_rate = 1e-7

    n_classes = args.classes
    preprocess_type = args.preprocessing if args.preprocessing in ['none', 'baseline', 'baseline_smoothing', 'all'] else 'none'
    model_type = args.model if args.model in ['simple_rescnn', 'simple_cnn', 'base'] else 'base'

    dataset = pd.read_csv(files.roi_single_spectra_path_data)
    train_set = balanced_dataset(dataset, n_classes)
    train_set['diagnosis'] = train_set['diagnosis'].astype(int)

    test_size = 0.10
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0) # 0 for reproducibility
    train_index, test_index = next(sss.split(np.zeros(train_set.shape[0]), train_set['diagnosis'].values))

    train_data = train_set.iloc[train_index]

    isotopenet_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=5e-1,
            patience=2,
            mode='auto',
            min_delta=0.0001,
            min_lr=min_learning_rate),
        # callbacks.LearningRateScheduler(
        #     exp_decay(learning_rate, decay_factor))
    ]

    dataset_config = joblib.load(files.roi_single_spectra_path / 'config.joblib')
    print('Dataset configuration: {}'.format(dataset_config))

    today = datetime.datetime.today()

    run_id = '{}_{}'.format(today.day, today.month)
    if args.id:
        run_id = 'run_{}'.format(args.id)

    out_path = files.out_path / 'isotopenet_cv_training_performances_{}_{}_{}'.format(run_id, dataset_config['type'], str(dataset_config['threshold']).replace('.', '_')) / model_type / 'classes_{}'.format(n_classes) / 'preprocess_{}'.format(preprocess_type)
    if not out_path.exists():
        os.makedirs(out_path)

    test_cv_results = []
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    fold = 0
    for train_index, validation_index in skf.split(np.zeros((train_data.shape[0], )), train_data['diagnosis'].values):

        # Check if no intersection exists between train-val-test sets
        # print('train-val', [v for v in train_index if v in validation_index])
        # print('train-test', [v for v in train_index if v in test_index])
        # print('val-test', [v for v in validation_index if v in test_index])

        model = IsotopeNet(lr=learning_rate, model_type=model_type, n_classes=n_classes)
        model.compile()
        # model.summary()
        # exit(0)

        if preprocess_type == 'all':
            preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=5, normalize_tic=True)

        if preprocess_type == 'baseline':
            preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=None, normalize_tic=None)

        if preprocess_type == 'baseline_smoothing':
            preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=5, normalize_tic=None)

        if preprocess_type == 'none':
            preprocessing = None

        print()
        print('Model: {}, Classes: {}, Preprocessing: {}, Kfold: {}/{}'.format(model_type, n_classes, preprocess_type, fold + 1, args.kfold))
        print()

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
            # use_multiprocessing=True,
            # workers=4,
            # verbose=0
        )
        
        tic = time.perf_counter()

        result = model.train(train_generator, train_config)

        toc = time.perf_counter()

        run_id = '{}_{}'.format(today.day, today.month)
        if args.id:
            run_id = 'run_{}'.format(args.id)

        fold_out_path = out_path / str(fold)
        if not fold_out_path.exists():
            os.makedirs(fold_out_path)

        train_time = toc - tic
        print()
        print('Training time: {}'.format(train_time))
        print()

        with (fold_out_path / 'time.txt').open('w') as f:
            f.write('{}\n'.format(train_time))

        model.save(fold_out_path / 'isotopenet.h5')
        joblib.dump(result.history, fold_out_path / 'isotopenet.joblib')

        test_generator = SpectraSequence(train_set.iloc[test_index], 
            model.csv_loader(files.roi_single_spectra_path), 
            batch_size=batch_size, 
            preprocessing=preprocessing,
            n_classes=n_classes,
            test=True)

        test_results = model.evaluate(test_generator, dict())
        test_cv_results.append(test_results)
        joblib.dump(test_results, fold_out_path / 'isotopenet_test.joblib')
        print()
        print(model.metrics_names)
        print()
        print(test_results)

        model_config = {
            'lr': learning_rate,
            'lr_decay_factor': decay_factor,
            'type': model_type,
            'n_classes': n_classes,
            'preprocessing': preprocess_type,
            'lr_on_plateau': {
                'factor': 5e-1,
                'patience': 2,
                'min_delta': 0.0001,
                'min_lr': 5e-6
            }
        }
        joblib.dump(model_config, fold_out_path / 'isotopenet_config.joblib')

        plot = SpectraPlot()
        plot.plot_metrics(result.history, title_font=14, outfolder=fold_out_path)

        fold += 1

    test_cv_results = np.array(test_cv_results)
    joblib.dump(test_cv_results, out_path / 'isotopenet_test.joblib')
    joblib.dump(dataset_config, out_path / 'dataset_config.joblib')
    print()
    print(model.metrics_names)
    print()
    print(np.mean(test_cv_results, axis=0))
    print()
    print(test_cv_results)
    print()

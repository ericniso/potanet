from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import joblib
import time
import datetime
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from spectraio import Files, SpectraPlot
from spectraio import balanced_dataset
from spectramodel import IsotopeNet, SpectraSequence, exp_decay
from imblearn.keras import BalancedBatchGenerator
from tensorflow.keras import callbacks
from sklearn.metrics import roc_curve, auc, roc_auc_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='')
    parser.add_argument('-m', '--model', type=str, default='base', help='')
    parser.add_argument('-c', '--classes', type=int, default=3, help='')
    parser.add_argument('-p', '--preprocessing', type=str, default='none', help='')
    parser.add_argument('-i', '--id', type=int, help='')

    args = parser.parse_args()

    if args.id and args.id < 0:
        print('error: id must be a positive integer')
        exit(1)
        
    if args.epochs < 1:
        print('error: epochs must be greater than 0')
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
    train_data = balanced_dataset(dataset, n_classes)
    train_data['diagnosis'] = train_data['diagnosis'].astype(int)

    isotopenet_callbacks = [
        # callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=5e-1,
        #     patience=2,
        #     mode='auto',
        #     min_delta=0.0001,
        #     min_lr=min_learning_rate),
        callbacks.LearningRateScheduler(
            exp_decay(learning_rate, decay_factor))
    ]
    
    dataset_config = joblib.load(files.roi_single_spectra_path / 'config.joblib')
    print('Dataset configuration: {}'.format(dataset_config))

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
    print('Model: {}, Classes: {}, Preprocessing: {}'.format(model_type, n_classes, preprocess_type))
    print()

    test_size = 0.30
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    train_index, validation_index = next(sss.split(np.zeros(train_data.shape[0]), train_data['diagnosis'].values))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    validation_index_index, test_index_index = next(sss.split(np.zeros(train_data.iloc[validation_index].shape[0]), train_data.iloc[validation_index]['diagnosis'].values))

    test_index = validation_index[test_index_index]
    validation_index = validation_index[validation_index_index]

    # Check if no intersection exists between train-val-test sets
    # print('train-val', [v for v in train_index if v in validation_index])
    # print('train-test', [v for v in train_index if v in test_index])
    # print('val-test', [v for v in validation_index if v in test_index])

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

    today = datetime.datetime.today()

    tic = time.perf_counter()

    result = model.train(train_generator, train_config)

    toc = time.perf_counter()

    run_id = '{}_{}'.format(today.day, today.month)
    if args.id:
        run_id = 'run_{}'.format(args.id)

    out_path = files.out_path / 'isotopenet_training_{}_{}_{}'.format(run_id, dataset_config['type'], str(dataset_config['threshold']).replace('.', '_')) / model_type / 'classes_{}'.format(n_classes) / 'preprocess_{}'.format(preprocess_type)
    if not out_path.exists():
        os.makedirs(out_path)

    train_time = toc - tic
    print()
    print('Training time: {}'.format(train_time))

    with (out_path / 'time.txt').open('w') as f:
        f.write('{}\n'.format(train_time))

    model.save(out_path / 'isotopenet.h5')
    joblib.dump(result.history, out_path / 'isotopenet.joblib')

    test_generator = SpectraSequence(train_data.iloc[test_index], 
        model.csv_loader(files.roi_single_spectra_path), 
        batch_size=batch_size, 
        preprocessing=preprocessing,
        n_classes=n_classes,
        test=True)

    test_results = model.evaluate(test_generator, dict())
    joblib.dump([model.metrics_names, test_results], out_path / 'isotopenet_test.joblib')
    print()
    print('Test results')
    print([model.metrics_names, test_results])

    test_generator = SpectraSequence(train_data.iloc[test_index], 
        model.csv_loader(files.roi_single_spectra_path), 
        batch_size=batch_size, 
        preprocessing=preprocessing,
        n_classes=n_classes,
        test=True)

    loader = model.csv_loader(files.roi_single_spectra_path)
    predict_results = model.predict(test_generator, dict())
    predict_true_results = []
    rows = train_data.iloc[test_index]
    for i in range(rows.shape[0]):
        p = np.zeros((n_classes,), dtype=np.uint8)
        diagnosis = int(loader['diagnosis'](rows.iloc[i]))
        p[diagnosis] = 1
        predict_true_results.append(p)
    predict_true_results = np.array(predict_true_results)

    n_data = min(predict_results.shape[0], predict_true_results.shape[0])
    predict_results = predict_results[0:n_data]
    predict_true_results = predict_true_results[0:n_data]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(predict_true_results[:, i], predict_results[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print()
    print('ROC AUC results')
    print(roc_auc)
    
    for i in range(n_classes):
        plt.clf()
        plt.plot(fpr[i], tpr[i], color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve for class {}'.format(i))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_path / 'roc_{}.pdf'.format(i))

    joblib.dump({ 'tpr': tpr, 'fpr': fpr, 'auc': auc }, out_path / 'isotopenet_roc.joblib')

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
    joblib.dump(model_config, out_path / 'isotopenet_config.joblib')
    joblib.dump(dataset_config, out_path / 'dataset_config.joblib')

    plot = SpectraPlot()
    plot.plot_metrics(result.history, title_font=14, outfolder=out_path)
        

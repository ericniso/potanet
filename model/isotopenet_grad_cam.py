from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pandas as pd
import numpy as np
import joblib
import datetime

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from spectraio import Files, SpectraDrawing, balanced_dataset
from spectramodel import csv_loader
from spectramodel import IsotopeNet, SpectraPredictSequence
from tensorflow.keras import backend as K

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', type=str, help='')
    parser.add_argument('-c', '--config', type=str, help='')
    parser.add_argument('-t', '--type', type=str, help='', default='training')
    parser.add_argument('--dinfo', type=str, help='')
    parser.add_argument('--rid', type=int, help='', default=42)
    args = parser.parse_args()

    if (not args.model) or (not args.config) or (not args.dinfo):
        print('error: model and corresponding config required together')
        exit(1)

    if (not Path(args.model).exists()) or (not Path(args.config).exists()) or (not Path(args.dinfo).exists()):
        print('error: model or config does not exist')
        exit(1)

    if args.type not in ['training', 'validation']:
        print('error: patient type must be either training or validation')
        exit(1)

    files = Files()

    patient_type = args.type
    dataset_config = joblib.load(files.roi_single_spectra_path / 'config.joblib')
    model_trained_on_dataset_info = joblib.load(Path(args.dinfo))
    config = joblib.load(Path(args.config))

    if config['n_classes'] < 3:
        print('error: classes should be >= 3')
        exit(1)

    model = IsotopeNet(lr=config['lr'], model_type=config['type'], n_classes=config['n_classes'])
    model.load(Path(args.model))
    model.compile()
    # model.summary()

    out_classes = [model.raw_model().output[0, c] for c in range(config['n_classes'])]
    locally_conn_conv_output = model.raw_model().get_layer('locally_connected2d').output
    gradients = [K.gradients(o_c, locally_conn_conv_output)[0] for o_c in out_classes]
    gradient_functions = [K.function([model.raw_model().input], [locally_conn_conv_output, g]) for g in gradients]

    if config['preprocessing'] == 'all':
        preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=5, normalize_tic=True)

    if config['preprocessing'] == 'baseline':
        preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=None, normalize_tic=None)

    if config['preprocessing'] == 'baseline_smoothing':
        preprocessing = model.preprocessing(baseline_median=5, smoothing_moving_average=5, normalize_tic=None)

    if config['preprocessing'] == 'none':
        preprocessing = None

    today = datetime.datetime.today()
    outdir = files.out_path / 'spectra_grad_cam_{}_{}_{}'.format(args.rid,  dataset_config['type'], str(dataset_config['threshold']).replace('.', '_')) / config['type'] / 'classes_{}'.format(config['n_classes']) / 'preprocess_{}'.format(config['preprocessing']) / patient_type

    if not outdir.exists():
        os.makedirs(outdir)
    else:
        print('error: output folder already exists')
        exit(1)

    joblib.dump(dataset_config, outdir / 'dataset_config.joblib')
    joblib.dump(config, outdir / 'isotopenet_config.joblib')

    loader = csv_loader(files.roi_single_spectra_path)
    dataset = pd.read_csv(files.roi_single_spectra_path_data)

    print('Model trained with: ', model_trained_on_dataset_info)
    print()
    print('Testing now on dataset: ', dataset_config)
    print()
    print('Dataset: {}, Model: {}, Classes: {}, Preprocessing: {}, Type: {}'.format(dataset_config['type'], config['type'], config['n_classes'], config['preprocessing'], patient_type))
    print()

    classes_4_training_dataset = balanced_dataset(dataset, 4)
    classes_3_training_dataset = balanced_dataset(dataset, 3)
    classes_2_training_dataset = balanced_dataset(dataset, 2)
    
    exclude_patients = list(classes_4_training_dataset['patient'].unique())
    for p in (classes_3_training_dataset['patient'].unique()):
        exclude_patients.append(p)
    for p in (classes_2_training_dataset['patient'].unique()):
        exclude_patients.append(p)

    # a.k.a in training set
    exclude_patients = list(set(exclude_patients))
    # a.k.a not in training set
    include_patients = [p for p in list(dataset['patient'].unique()) if p not in exclude_patients]

    all_intervals_info = {
        # 'patients': [],
        # 'types': [],
        # 'spectra': [],
        # 'mzs': [],
        # 'diagnosis': [],
        # 'coordinates': [],
        'threshold': [0, 0.25, 0.50, 0.75],
        # 'intervals': [],
        # 'mzs_intervals': [],
        # 'locally_connected_outputs': [],
        # 'gradients': [],
        # 'cams': [],
        # 'weights': [],
        'resize_val': 8000
    }

    for p in tqdm(exclude_patients if patient_type == 'training' else include_patients):
        
        p_type = patient_type
        patient_out_dir = outdir / p

        if not (patient_out_dir).exists():
            os.makedirs(patient_out_dir)

        current_patient = dataset[dataset['patient'] == p]

        predict_generator = SpectraPredictSequence(current_patient, 
            model.csv_loader(files.roi_single_spectra_path),
            preprocessing=preprocessing)

        diagnosis = model.predict(predict_generator.all(), {})

        for i in range(current_patient.shape[0]):

            current_diagnosis = int(np.argmax(diagnosis[i], axis=0))
            current_spectrum = loader['spectrum'](current_patient.iloc[i])
            original_spectrum = np.copy(current_spectrum)

            if preprocessing is not None:
                current_spectrum = preprocessing(current_spectrum)

            current_spectrum = np.reshape(current_spectrum, (current_spectrum.shape[0], 1, 1))

            output, grads_val = gradient_functions[current_diagnosis](np.array([current_spectrum]))
            output, grads_val = output[0, :], grads_val[0, :, :, :]

            weights = np.mean(grads_val, axis=(0, 1))
            cam = np.dot(output, weights)

            original_cam = np.copy(cam)

            cam = cv2.resize(cam, (current_spectrum.shape[1], current_spectrum.shape[0]), cv2.INTER_LINEAR)
            cam = np.maximum(cam, 0)
            cam_max = cam.max() 
            if cam_max != 0: 
                cam = cam / cam_max

            current_out_dir = patient_out_dir / str(current_diagnosis) / current_patient.iloc[i]['spectrum']
            if not (current_out_dir).exists():
                os.makedirs(current_out_dir)

            current_mzs = loader['mzs'](current_patient.iloc[i])

            # TODO: uncomment for plots
            # x_ticks = []
            # multiplier = 1000
            # n_ticks = current_mzs.shape[0] // multiplier + 1

            # for x_val in range(n_ticks):
            #     x_ticks.append('{:.3f}'.format(current_mzs[min(x_val * multiplier, current_mzs.shape[0] - 1)]))

            # plt.clf()
            # plt.plot(original_spectrum, label='raw', color='b', lw=0.5)
            # plt.xticks(np.arange(n_ticks) * multiplier, x_ticks, rotation=30)
            # _, xlabels = plt.xticks()
            # xlabels[-1].set_visible(False)
            # plt.xlabel('m/z')
            # plt.ylabel('Intensity')
            # # plt.tight_layout()
            # plt.legend(loc='upper right')#, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
            # plt.savefig(current_out_dir / 'raw_spectrum.pdf')

            # plt.clf()
            # plt.plot(np.reshape(current_spectrum, (current_spectrum.shape[0])), label='preprocessed', color='r', lw=0.5)
            # plt.xticks(np.arange(n_ticks) * multiplier, x_ticks, rotation=30)
            # _, xlabels = plt.xticks()
            # xlabels[-1].set_visible(False)
            # plt.xlabel('m/z')
            # plt.ylabel('Intensity')
            # # plt.tight_layout()
            # plt.legend(loc='upper right')#, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
            # plt.savefig(current_out_dir / 'preprocessed_spectrum.pdf')

            # plt.clf()
            # plt.plot(original_spectrum, label='raw', color='b', lw=0.5)
            # plt.plot(np.reshape(current_spectrum, (current_spectrum.shape[0])), label='preprocessed', color='r', lw=0.5)
            # plt.xticks(np.arange(n_ticks) * multiplier, x_ticks, rotation=30)
            # _, xlabels = plt.xticks()
            # xlabels[-1].set_visible(False)
            # plt.xlabel('m/z')
            # plt.ylabel('Intensity')
            # # plt.tight_layout()
            # plt.legend(loc='upper right')#, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
            # plt.savefig(current_out_dir / 'combined_spectra.pdf')

            # plt.clf()
            # plt.plot(cam, lw=0.5)
            # plt.xticks(np.arange(n_ticks) * multiplier, x_ticks, rotation=30)
            # _, xlabels = plt.xticks()
            # xlabels[-1].set_visible(False)
            # plt.xlabel('m/z')
            # plt.ylabel('Intensity')
            # # plt.tight_layout()
            # plt.savefig(current_out_dir / 'cam.pdf')

            # img_height = 1000
            # cam_heatmap = []
            # for h in range(img_height):
            #     cam_heatmap.append(cam)

            # cam_heatmap = np.reshape(cam_heatmap, (img_height, current_spectrum.shape[0]))

            # plt.clf()
            # # plt.tight_layout()
            # plt.imshow(cam_heatmap, cmap='YlOrRd')
            # plt.colorbar(orientation="horizontal", pad=0.25)
            # plt.xticks(np.arange(n_ticks) * multiplier, x_ticks, rotation=30)
            # _, xlabels = plt.xticks()
            # xlabels[-1].set_visible(False)
            # plt.yticks([])
            # plt.xlabel('m/z')
            # plt.savefig(current_out_dir / 'cam_heatmap.pdf', bbox_inches='tight')

            threshold_values = all_intervals_info['threshold']
            intervals_threshold = [[] for t in threshold_values]
            mzs_intervals_threshold = [[] for t in threshold_values]
            
            for t in range(len(threshold_values)):
                current_interval = [-1, -1]
                
                for m in range(current_mzs.shape[0]):
                    if cam[m] > threshold_values[t] and current_interval[0] == -1:
                        current_interval[0] = m

                    if cam[m] <= threshold_values[t] and current_interval[0] != -1 and current_interval[1] == -1:
                        current_interval[1] = m - 1
                        intervals_threshold[t].append(current_interval)
                        mzs_intervals_threshold[t].append([
                            current_mzs[current_interval[0]],
                            current_mzs[current_interval[1]]
                        ])
                        current_interval = [-1, -1]

            intervals_info = {
                'patient': current_patient.iloc[i]['patient'],
                'spectrum': current_patient.iloc[i]['spectrum'],
                'mzs': current_patient.iloc[i]['mzs'],
                'diagnosis': diagnosis[i],
                'coordinates': current_patient.iloc[i]['coordinates'],
                'threshold': threshold_values,
                'intervals': intervals_threshold,
                'mzs_intervals': mzs_intervals_threshold,
                # 'locally_connected_output': output,
                # 'gradients': grads_val,
                'cam': original_cam,
                # 'weights': weights,
                'resize_val': current_spectrum.shape[0]
            }

            joblib.dump(intervals_info, current_out_dir / 'intervals.joblib')

    joblib.dump(all_intervals_info, outdir / 'intervals.joblib')
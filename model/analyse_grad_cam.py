from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

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
from spectraio import Files
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--rootdir', type=str, help='')
    parser.add_argument('--rid', type=int, help='', default=42)
    args = parser.parse_args()

    if (not args.rootdir) or (not Path(args.rootdir).exists()):
        print('error: intervals and dataset info files required')
        exit(1)

    files = Files()
    rootdir = Path(args.rootdir)

    n_classes = -1
    all_threshold_values = []
    dataset_config = {}
    config = {}
    outdir = None

    histogram_intervals = { }

    print()
    print('Computing histograms')
    print()
    for patient_type in ['training', 'validation']:

        ptype_dir = (rootdir / patient_type)

        all_intervals = joblib.load(ptype_dir / 'intervals.joblib')
        dataset_config = joblib.load(ptype_dir / 'dataset_config.joblib')
        config = joblib.load(ptype_dir / 'isotopenet_config.joblib')

        n_classes = config['n_classes']

        outdir = files.out_path / 'spectra_grad_cam_analysis_{}_{}_{}'.format(args.rid,  dataset_config['type'], str(dataset_config['threshold']).replace('.', '_')) / config['type'] / 'classes_{}'.format(config['n_classes']) / 'preprocess_{}'.format(config['preprocessing'])

        if not outdir.exists():
            os.makedirs(outdir)

        print('Dataset: {}, Model: {}, Classes: {}, Preprocessing: {}, Patient type: {}'.format(dataset_config['type'], config['type'], config['n_classes'], config['preprocessing'], patient_type))
        print()

        threshold_values = all_intervals['threshold']
        all_threshold_values = threshold_values
        n_bins = 50
        bins = np.linspace(2000, 20000, 50)
        histogram_intervals[patient_type] = [ [ np.zeros(n_bins, dtype=np.int) for _ in threshold_values ] for _ in range(config['n_classes']) ]

        all_patients = [ d for d in ptype_dir.iterdir() if d.is_dir() ]

        for p in tqdm(all_patients):

            for c in range(config['n_classes']):

                class_dir = ptype_dir / p / str(c)

                if not class_dir.exists():
                    continue

                all_spectra = [ d for d in class_dir.iterdir() if d.is_dir() ]

                for s in all_spectra:

                    spectrum_dir = class_dir / s

                    spectrum_info = joblib.load(spectrum_dir / 'intervals.joblib')
                    diagnosis = int(np.argmax(spectrum_info['diagnosis'], axis=0))

                    for t in range(len(threshold_values)):
                        mzs_intervals = spectrum_info['mzs_intervals'][t]

                        for j in range(len(mzs_intervals)):

                            current_interval = mzs_intervals[j]
                            left_added = -1

                            for b in range(n_bins):
                            
                                if (left_added == -1) and (current_interval[0] <= bins[b]):
                                    histogram_intervals[patient_type][diagnosis][t][b] += 1
                                    left_added = b

                                if b > left_added and (current_interval[1] <= bins[b]):
                                    histogram_intervals[patient_type][diagnosis][t][b] += 1
                                    break

                                if b == left_added and (current_interval[1] <= bins[b]):
                                    break

    print()
    print('Plotting histograms')
    print()
    
    xticks = np.arange(n_bins)
    xticks_labels = []
    for i in range(bins.shape[0]):
        xticks_labels.append('{:.3f}'.format(bins[i]))

    histogram_distances = {
        'threshold': all_threshold_values,
        'n_classes': n_classes,
        'distances': {
            'l2': np.ndarray((n_classes, len(all_threshold_values)), dtype=np.float),
            'cosine_similarity': np.ndarray((n_classes, len(all_threshold_values)), dtype=np.float)
        },
        'differences': [ [ [] for _ in range(len(all_threshold_values)) ] for _ in range(n_classes) ]
    }

    for c in tqdm(range(n_classes)):

        for t in range(len(all_threshold_values)):
            
            normalized_histograms = {}

            for patient_type in ['training', 'validation']:

                threshold_histogram = histogram_intervals[patient_type][c][t]

                plt.clf()
                plt.xticks(xticks, xticks_labels, rotation=90)
                _, xlabels = plt.xticks()

                for l in range(1, len(xlabels) - 1):
                    xlabels[l].set_visible(False)
                    if l % 5 == 0:
                        xlabels[l].set_visible(True)

                plt.bar(xticks, threshold_histogram, align='edge', width=1.0)
            
                plt.tight_layout()
                plt.savefig(outdir / '{}_class_{}_threshold_{}.pdf'.format(patient_type, c, str(threshold_values[t])))

                normalized_threshold_histogram = threshold_histogram / np.sum(threshold_histogram, axis=0)
                normalized_histograms[patient_type] = normalized_threshold_histogram

                plt.clf()
                plt.xticks(xticks, xticks_labels, rotation=90)
                _, xlabels = plt.xticks()

                for l in range(1, len(xlabels) - 1):
                    xlabels[l].set_visible(False)
                    if l % 5 == 0:
                        xlabels[l].set_visible(True)

                plt.bar(xticks, normalized_threshold_histogram, align='edge', width=1.0)
            
                plt.tight_layout()
                plt.savefig(outdir / '{}_class_{}_threshold_{}_normalized.pdf'.format(patient_type, c, str(threshold_values[t])))

            diff = normalized_histograms['training'] - normalized_histograms['validation']
            histogram_distances['differences'][c][t].append(np.abs(diff))
            histogram_distances['distances']['l2'][c, t] = np.linalg.norm(diff)
            histogram_distances['distances']['cosine_similarity'][c, t] = cosine_similarity(
                np.reshape(normalized_histograms['training'], (1, -1)), 
                np.reshape(normalized_histograms['validation'], (1, -1)))
            
            plt.clf()
            plt.xticks(xticks, xticks_labels, rotation=90)
            _, xlabels = plt.xticks()

            for l in range(1, len(xlabels) - 1):
                xlabels[l].set_visible(False)
                if l % 5 == 0:
                    xlabels[l].set_visible(True)

            plt.bar(xticks, normalized_histograms['training'], align='edge', width=1.0, label='training', color='b', alpha=0.5)
            plt.bar(xticks, normalized_histograms['validation'], align='edge', width=1.0, label='validation', color='g', alpha=0.5)
            
            # plt.title('cosine similarity = {}'.format(histogram_distances['distances']['cosine_similarity'][c, t]))
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(outdir / 'class_{}_threshold_{}_normalized_combined.pdf'.format(c, str(threshold_values[t])))

    print()
    print(histogram_distances['threshold'])
    print('L2')
    print(histogram_distances['distances']['l2'])
    print('Cosine similarity')
    print(histogram_distances['distances']['cosine_similarity'])
    print()

    histogram_distances['differences'] = np.array(histogram_distances['differences'])

    joblib.dump(histogram_intervals, outdir / 'histograms.joblib')
    joblib.dump(histogram_distances, outdir / 'histogram_distances.joblib')
    joblib.dump(config, outdir / 'isotopenet_config.joblib')
    joblib.dump(dataset_config, outdir / 'dataset_config.joblib')



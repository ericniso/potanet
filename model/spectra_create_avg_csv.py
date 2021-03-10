import sys
import argparse
import os
import numpy as np
import json
import pandas as pd
import uuid
import shutil
from pathlib import Path
from tqdm import tqdm
from spectraio import Dataset, DataHandler, Files
from spectramodel import csv_loader

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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()

    files = Files()
    # dataset = Dataset()
    loader = csv_loader(files.roi_single_spectra_path_data_single_root)
    data_handler = DataHandler()

    dataset_info = []
    patient_info = []
    spectrum_info = []
    mzs_info = []
    coords_info = []
    diagnosis_info = []

    if files.roi_single_spectra_path_data_avg_root.exists():
        shutil.rmtree(files.roi_single_spectra_path_data_avg_root)

    os.makedirs(files.roi_single_spectra_path_data_avg_root)
    os.makedirs(files.roi_single_spectra_path_intensities_avg)
    os.makedirs(files.roi_single_spectra_path_mzs_avg)
    os.makedirs(files.roi_single_spectra_path_coordinates_avg)

    dataset = pd.read_csv(files.roi_single_spectra_path_data_single)

    patients = dataset['patient'].unique()

    all_found = 0

    with tqdm(total=dataset.shape[0]) as pbar:
        for p in patients: # tqdm(patients):
            patient_dataset = dataset[dataset['patient'] == p]
            n_found_total = 0
            for i in range(patient_dataset.shape[0]):

                i_patient = patient_dataset.iloc[i]
                i_coords = loader['coordinates'](i_patient)
                neighbours = get_neighbours(i_coords)

                avg_intensities = []
                avg_intensities.append(loader['spectrum'](i_patient))

                n_found = 0
                for n in neighbours:
                    for j in range(patient_dataset.shape[0]):
                        j_patient = patient_dataset.iloc[j]
                        j_coords = loader['coordinates'](j_patient)

                        if (j_coords == n).all():
                            n_found += 1
                            avg_intensities.append(loader['spectrum'](j_patient))
                
                n_found_total += 1 if n_found == 8 else 0

                avg_intensities = np.array(avg_intensities)
                avg_intensities = np.average(avg_intensities, axis=0)
                
                save_id = uuid.uuid4()
                out_file_name = '{}.txt'.format(save_id)
                out_file = files.roi_single_spectra_path_intensities_avg / out_file_name
                np.savetxt(out_file, avg_intensities)
                spectrum_info.append(save_id)

                save_id = uuid.uuid4()
                out_file_name = '{}.txt'.format(save_id)
                out_file = files.roi_single_spectra_path_mzs_avg / out_file_name
                np.savetxt(out_file, loader['mzs'](i_patient))
                mzs_info.append(save_id)

                save_id = uuid.uuid4()
                out_file_name = '{}.txt'.format(save_id)
                out_file = files.roi_single_spectra_path_coordinates_avg / out_file_name
                np.savetxt(out_file, i_coords.astype(np.int), fmt='%i')
                coords_info.append(save_id)

                dataset_info.append(i_patient['dataset'])
                patient_info.append(i_patient['patient'])
                diagnosis_info.append(i_patient['diagnosis'])

                pbar.update(1)
            
            all_found += n_found_total

    dataset_to_save = pd.DataFrame({
        'dataset': dataset_info,
        'patient': patient_info,
        'spectrum': spectrum_info,
        'mzs': mzs_info,
        'coordinates': coords_info,
        'diagnosis': diagnosis_info
    })
    dataset_to_save = dataset_to_save[['dataset', 'patient', 'spectrum', 'mzs', 'coordinates', 'diagnosis']]
    dataset_to_save['dataset'] = dataset_to_save['dataset'].apply(str)
    dataset_to_save['patient'] = dataset_to_save['patient'].apply(str)
    dataset_to_save['diagnosis'] = dataset_to_save['diagnosis'].apply(int)
    dataset_to_save.to_csv(files.roi_single_spectra_path_data_avg, index=False)

    print('{}/{}'.format(all_found, dataset.shape[0]))
    print('{}'.format(all_found / dataset.shape[0]))

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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()

    files = Files()
    dataset = Dataset()
    data_handler = DataHandler()

    dataset_info = []
    patient_info = []
    spectrum_info = []
    mzs_info = []
    coords_info = []
    diagnosis_info = []

    append = 'mlmlml'
    while append.upper() not in ['y'.upper(), 'n'.upper()]:
        append = input('Do you want to keep all existing data? [y/n] ')

    append = append.upper() == 'y'.upper()

    if not append:
        if files.roi_single_spectra_path_root.exists():
            shutil.rmtree(files.roi_single_spectra_path_root)

        os.makedirs(files.roi_single_spectra_path_root)
        os.makedirs(files.roi_single_spectra_path_intensities_single)
        os.makedirs(files.roi_single_spectra_path_mzs_single)
        os.makedirs(files.roi_single_spectra_path_coordinates_single)

    for dataset_type in ['training', 'validation', 'validation_exvivo']:

        print()
        print('Processing {} dataset...\n'.format(dataset_type))

        current_dataset = dataset.get(dataset_type)
        current_spectra_count = 0

        for f_spectra, f_mz, f_coords, patient_id, diagnosis in tqdm(zip(current_dataset['intensities'], current_dataset['mzs'], current_dataset['coordinates'], current_dataset['patients'], current_dataset['diagnosis'])):
        
            # print('Processing file {}...'.format(patient_id))

            total_spectra = 0
            total_mz = 0
            total_coords = 0

            # print('Processing spectra...')
            check_existing_files = os.listdir(files.roi_single_spectra_path_intensities_single)
            check_existing_files = [f.split('.') for f in check_existing_files]
            check_existing_files = [f[0] for f in check_existing_files]
            for spectrum in data_handler.load_data(f_spectra):

                spectrum_id = uuid.uuid4()
                while str(spectrum_id) in check_existing_files:
                    spectrum_id = uuid.uuid4()

                out_file_name = '{}.txt'.format(spectrum_id)
                out_file = files.roi_single_spectra_path_intensities_single / out_file_name

                np.savetxt(out_file, spectrum)

                spectrum_info.append(spectrum_id)
                total_spectra += 1

            # print('Processing mzs...')
            check_existing_files = os.listdir(files.roi_single_spectra_path_mzs_single)
            check_existing_files = [f.split('.') for f in check_existing_files]
            check_existing_files = [f[0] for f in check_existing_files]
            for mz in data_handler.load_data(f_mz):

                mz_id = uuid.uuid4()
                while str(mz_id) in check_existing_files:
                    mz_id = uuid.uuid4()

                out_file_name = '{}.txt'.format(mz_id)
                out_file = files.roi_single_spectra_path_mzs_single / out_file_name

                np.savetxt(out_file, mz)

                mzs_info.append(mz_id)
                total_mz += 1

            assert total_spectra == total_mz

            # print('Processing coordinates...')
            check_existing_files = os.listdir(files.roi_single_spectra_path_coordinates_single)
            check_existing_files = [f.split('.') for f in check_existing_files]
            check_existing_files = [f[0] for f in check_existing_files]
            for coords in data_handler.load_data(f_coords):

                coords_id = uuid.uuid4()
                while str(coords_id) in check_existing_files:
                    coords_id = uuid.uuid4()

                out_file_name = '{}.txt'.format(coords_id)
                out_file = files.roi_single_spectra_path_coordinates_single / out_file_name

                np.savetxt(out_file, coords.astype(np.int), fmt='%i')

                coords_info.append(coords_id)
                total_coords += 1

            assert total_coords == total_spectra

            patient_info += [patient_id] * total_spectra
            diagnosis_info += [diagnosis] * total_spectra

            current_spectra_count += total_spectra

        dataset_info += [dataset_type] * current_spectra_count

    dataset_to_save = pd.DataFrame({
        'dataset': dataset_info,
        'patient': patient_info,
        'spectrum': spectrum_info,
        'mzs': mzs_info,
        'coordinates': coords_info,
        'diagnosis': diagnosis_info
    })
    dataset_to_save = dataset_to_save[['dataset', 'patient', 'spectrum', 'mzs', 'coordinates', 'diagnosis']]
        
    dataset_to_save['dataset'].apply(str)
    dataset_to_save['patient'].apply(str)

    if append:
        existing_dataset = pd.read_csv(files.roi_single_spectra_path_data_single)
        dataset_to_save = existing_dataset.append(dataset_to_save)

    dataset_to_save.to_csv(files.roi_single_spectra_path_data_single, index=False)

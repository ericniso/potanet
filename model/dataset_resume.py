import pandas as pd
import numpy as np
from spectraio import Files, Dataset, balanced_dataset
from spectramodel import csv_loader

if __name__ == '__main__':
    
    files = Files()

    loader = csv_loader(files.roi_single_spectra_path)
    dataset = pd.read_csv(files.roi_single_spectra_path_data)

    classes_4_training_dataset = balanced_dataset(dataset, 4)
    classes_3_training_dataset = balanced_dataset(dataset, 3)
    classes_2_training_dataset = balanced_dataset(dataset, 2)

    patients = dataset['patient'].unique()
    patients.sort()
    for p in patients:
        
        current_patient = dataset[dataset['patient'] == p]

        print()
        print(p)
        print(current_patient.shape[0])
        print(current_patient['diagnosis'].value_counts(normalize=True))
        print()

    print()
    print('---- 4 classes')
    print()
    print(classes_4_training_dataset['diagnosis'].value_counts())
    print()
    print(classes_4_training_dataset['diagnosis'].value_counts(normalize=True))
    print()

    print()
    print('---- 3 classes')
    print()
    print(classes_3_training_dataset['diagnosis'].value_counts())
    print()
    print(classes_3_training_dataset['diagnosis'].value_counts(normalize=True))
    print()

    print()
    print('---- 2 classes')
    print()
    print(classes_2_training_dataset['diagnosis'].value_counts())
    print()
    print(classes_2_training_dataset['diagnosis'].value_counts(normalize=True))
    print()

    print()
    print('Dataset size: {}'.format(dataset.shape[0]))
    print('4 classes dataset size: {}'.format(classes_4_training_dataset.shape[0]))
    print('3 classes dataset size: {}'.format(classes_3_training_dataset.shape[0]))
    print('2 classes dataset size: {}'.format(classes_2_training_dataset.shape[0]))
    print()

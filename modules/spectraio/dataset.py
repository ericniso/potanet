import config
import pandas as pd
from pathlib import Path


class Dataset:

    def __init__(self):

        self.root = Path(config.POTANET_ROOT_DIR)
        self.imzML_extracted_path = Path(config.POTANET_IMZML_EXTRACTED_ROOT_DIR)

        # TODO: change order
        self.diagnosis_to_index = dict(
            HP=0,
            PTC=1,
            HT=2,
            Noise=3
        )

        self.raw_dataset = pd.read_csv(Path(__file__).parent / "dataset" / "dataset.csv")

    def get(self, dataset_type):

        assert dataset_type in ["training", "validation", "validation_exvivo"]

        patient_column = list(self.raw_dataset.columns).index("patient")
        diagnosis_column = list(self.raw_dataset.columns).index("diagnosis")
        selected_dataset = [p for p in self.raw_dataset[self.raw_dataset["type"] == dataset_type].values if
                            p[diagnosis_column] in self.diagnosis_to_index]

        return dict(
            masses=[self.imzML_extracted_path / dataset_type / p[patient_column] / "masses.txt" for p in
                    selected_dataset],
            coordinates=[self.imzML_extracted_path / dataset_type / p[patient_column] / "coordinates.txt" for p in
                         selected_dataset],
            intensities=[self.imzML_extracted_path / dataset_type / p[patient_column] / "intensities.txt" for p in
                         selected_dataset],
            patients=[p[patient_column] for p in selected_dataset],
            diagnosis=[self.diagnosis_to_index[p[diagnosis_column]] for p in selected_dataset]
        )


def full_dataset(dataset_csv):

    return dataset_csv


def balanced_dataset(dataset_csv, n_classes):

    with (Path(__file__).parent / "dataset" / "dataset_{}_classes.txt".format(n_classes)).open() as f:
        selected_patients = [p.strip() for p in f.readlines()]

    new_balanced_dataset = pd.DataFrame([], columns=dataset_csv.columns)

    for p in selected_patients:
        new_balanced_dataset = new_balanced_dataset.append(
            dataset_csv[dataset_csv["patient"] == p]
        )

    new_balanced_dataset_filtered = pd.DataFrame([], columns=new_balanced_dataset.columns)
    for i in [0, 1, 2, 3]:
        if i >= n_classes:
            break

        new_balanced_dataset_filtered = new_balanced_dataset_filtered.append(
            new_balanced_dataset[new_balanced_dataset["diagnosis"] == i]
        )

    return new_balanced_dataset_filtered


def test_dataset(dataset_csv, filter_csv):

    filtered = filter_csv
    exclude_patients = list(filtered["patient"].unique())

    test_data = pd.DataFrame([], columns=dataset_csv.columns)

    for p in list(dataset_csv["patient"].unique()):

        if p in exclude_patients:
            continue

        current_patient = dataset_csv[dataset_csv["patient"] == p]
        test_data = test_data.append(current_patient, ignore_index=True)

    return test_data

import config
import argparse
import os
import numpy as np
import pandas as pd
import uuid
import shutil
import time
from logger import potanet_logger
from spectraio import Dataset, Files
from concurrent.futures import ThreadPoolExecutor


class RawSample:

    def __init__(self, dataset_type, intensities, masses, coordinates, patient_id, diagnosis):

        self.loaded = False
        self.dataset_type = dataset_type
        self.intensities = intensities
        self.masses = masses
        self.coordinates = coordinates
        self.patient_id = patient_id
        self.diagnosis = diagnosis

    def load(self):

        self.loaded = True
        self.intensities = np.loadtxt(self.intensities, dtype=np.float)
        self.masses = np.loadtxt(self.masses, dtype=np.float)
        self.coordinates = np.loadtxt(self.coordinates, dtype=np.int)

        assert (self.intensities.shape[0] == self.masses.shape[0] and self.intensities.shape[0] == self.coordinates.shape[0]), \
            "Dimensions don't match"

    def __len__(self):
        if not self.loaded:
            raise Exception("Data not loaded, call load() method before asking for length")

        return self.intensities.shape[0]

    def __repr__(self):
        return ""


class ProcessedSample:

    def __init__(self, dataset_type, patient_id, diagnosis, sample_id, coordinates):

        self.dataset_type = dataset_type
        self.patient_id = patient_id
        self.diagnosis = diagnosis
        self.sample_id = sample_id
        self.coordinates = coordinates

    def __repr__(self):
        return "<{} dataset_type={}, patient_id={}, diagnosis={}, sample_id={}, coordinates={}>".format(
            type(self).__name__, self.dataset_type, self.patient_id, self.diagnosis, self.sample_id, self.coordinates)


class ProcessingWorker:

    def __init__(self, out_folders):

        self.out_folders = out_folders

    def __call__(self, *args, **kwargs):
        sample = args[0]

        processed_samples = []

        sample.load()
        for i in range(len(sample)):

            coords = sample.coordinates[i]
            intensities = sample.intensities[i]
            masses = sample.masses[i]

            sample_id = "{}_{}_{}_{}".format(str(sample.patient_id), str(coords[0]), str(coords[1]), str(coords[2]))

            np.savetxt(self.out_folders["intensities"] / "{}.txt".format(sample_id), intensities)
            np.savetxt(self.out_folders["masses"] / "{}.txt".format(sample_id), masses)

            processed_samples.append(
                ProcessedSample(sample.dataset_type, sample.patient_id, sample.diagnosis, sample_id, coords))

        potanet_logger.info("Processed sample {} with {} spectra".format(sample.patient_id, len(sample)))

        return processed_samples


if __name__ == "__main__":

    execution_start_time = time.perf_counter()

    files = Files()
    dataset = Dataset()

    if files.spectra_processed_dataset_single.exists():
        shutil.rmtree(files.spectra_processed_dataset_single)

    os.makedirs(files.spectra_processed_dataset_single)
    os.makedirs(files.spectra_processed_dataset_single_intensities)
    os.makedirs(files.spectra_processed_dataset_single_masses)

    dataset_patients_samples = []

    for dataset_type in ["training", "validation", "validation_exvivo"]:
        current_dataset = dataset.get(dataset_type)

        for intensities, masses, coordinates, patient_id, diagnosis in zip(current_dataset["intensities"],
                                                                           current_dataset["masses"],
                                                                           current_dataset["coordinates"],
                                                                           current_dataset["patients"],
                                                                           current_dataset["diagnosis"]):
            dataset_patients_samples.append(
                RawSample(dataset_type, intensities, masses, coordinates, patient_id, diagnosis))

    potanet_logger.info("Starting dataset creation with {} workers".format(config.POTANET_THREAD_POOL_SIZE))

    with ThreadPoolExecutor(max_workers=config.POTANET_THREAD_POOL_SIZE) as executor:

        out_folders = dict(
            intensities=files.spectra_processed_dataset_single_intensities,
            masses=files.spectra_processed_dataset_single_masses
        )
        result = executor.map(ProcessingWorker(out_folders), dataset_patients_samples)

    dataset_info = []
    patient_info = []
    sample_info = []
    coords_info = dict(x=[], y=[], z=[])
    diagnosis_info = []

    total_spectra = 0
    for res in result:
        for processed_sample in res:
            total_spectra += 1

            dataset_info.append(processed_sample.dataset_type)
            patient_info.append(processed_sample.patient_id)
            sample_info.append(processed_sample.sample_id)
            coords_info["x"].append(processed_sample.coordinates[0])
            coords_info["y"].append(processed_sample.coordinates[1])
            coords_info["z"].append(processed_sample.coordinates[2])
            diagnosis_info.append(processed_sample.diagnosis)

    dataset_to_save = pd.DataFrame({
        "dataset": dataset_info,
        "patient": patient_info,
        "sample": sample_info,
        "coordinates_x": coords_info["x"],
        "coordinates_y": coords_info["y"],
        "coordinates_z": coords_info["z"],
        "diagnosis": diagnosis_info
    })
    dataset_to_save = dataset_to_save[
        ["dataset", "patient", "sample", "coordinates_x", "coordinates_y", "coordinates_z", "diagnosis"]]

    dataset_to_save["dataset"].apply(str)
    dataset_to_save["patient"].apply(str)
    dataset_to_save.to_csv(files.spectra_processed_dataset_single_data, index=False)

    execution_end_time = time.perf_counter()
    potanet_logger.info("Dataset of {} spectra created in {}s".format(total_spectra, execution_end_time - execution_start_time))

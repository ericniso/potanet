import os
import config
import numpy as np
import pandas as pd
import shutil
import time
from spectraio import Files
from concurrent.futures import ThreadPoolExecutor
from logger import potanet_logger


class AverageSpectrum:

    def __init__(self, dataset, patient_id, csv_row, intensities_path):

        self.dataset = dataset
        self.patient_id = patient_id
        self.csv_row = csv_row
        self.intensities_path = intensities_path
        self.original_intensities = np.loadtxt(self.intensities_path / "{}.txt".format(self.csv_row.at["sample"]))

    def __find_neighbours__(self):

        sample = self.csv_row.at["sample"]
        start_x = self.csv_row.at["coordinates_x"] - 1
        end_x = start_x + 2
        start_y = self.csv_row.at["coordinates_y"] - 1
        end_y = start_y + 2
        neighbours_coords = []
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                neighbour = "{}_{}_{}_{}".format(self.patient_id, x, y, self.csv_row.at["coordinates_z"])
                if not neighbour == sample:
                    neighbours_coords.append(neighbour)

        return neighbours_coords

    def get_average_intensities(self):

        neighbours = self.__find_neighbours__()
        avg_intensities = [self.original_intensities]

        for neigh in neighbours:
            neigh_row = self.dataset[self.dataset["sample"] == neigh]

            if neigh_row.shape[0] > 0:
                neigh_row = neigh_row.iloc[0]
                neigh_intensity = np.loadtxt(self.intensities_path / "{}.txt".format(neigh_row.at["sample"]))
                avg_intensities.append(neigh_intensity)

        avg_intensities = np.array(avg_intensities)
        avg_intensities = np.mean(avg_intensities, axis=0)

        return avg_intensities


class ProcessingWorker:

    def __init__(self, dataset, src_intensities_path, dest_intensities_path):

        self.dataset = dataset
        self.src_intensities_path = src_intensities_path
        self.dest_intensities_path = dest_intensities_path

    def __call__(self, *args, **kwargs):
        patient_id = args[0]
        patient_dataset = self.dataset[self.dataset["patient"] == patient_id]

        for i in range(patient_dataset.shape[0]):
            row = patient_dataset.iloc[i]
            average_spectrum = AverageSpectrum(patient_dataset, patient_id, row, self.src_intensities_path)\
                .get_average_intensities()
            np.savetxt(self.dest_intensities_path / "{}.txt".format(row.at["sample"]), average_spectrum)

        potanet_logger.info("Created average spectra of patient {}".format(patient_id))

        return patient_id


if __name__ == "__main__":

    execution_start_time = time.perf_counter()

    files = Files()

    if files.spectra_processed_dataset_avg.exists():
        shutil.rmtree(files.spectra_processed_dataset_avg)

    os.makedirs(files.spectra_processed_dataset_avg)
    os.makedirs(files.spectra_processed_dataset_avg_intensities)

    potanet_logger.info("Starting copy of dataset masses into new directory")

    shutil.copytree(files.spectra_processed_dataset_single_masses, files.spectra_processed_dataset_avg_masses)

    dataset = pd.read_csv(files.spectra_processed_dataset_single_data)
    shutil.copyfile(files.spectra_processed_dataset_single_data, files.spectra_processed_dataset_avg_data)

    patients = dataset["patient"].unique()

    potanet_logger.info("Starting dataset creation with {} workers".format(config.POTANET_THREAD_POOL_SIZE))

    with ThreadPoolExecutor(max_workers=config.POTANET_THREAD_POOL_SIZE) as executor:

        result = executor.map(ProcessingWorker(dataset, files.spectra_processed_dataset_single_intensities,
                                               files.spectra_processed_dataset_avg_intensities), patients)

    execution_end_time = time.perf_counter()
    potanet_logger.info(
        "Dataset of {} spectra created in {}s".format(-1, execution_end_time - execution_start_time))

import config
import numpy as np
from mvgavg import mvgavg
from collections import deque
from bisect import insort, bisect_left
from itertools import islice


class Spectrum:

    def __int__(self, files, csv_row):

        self.target_size = config.POTANET_SPECTRUM_SIZE

        self.dataset = csv_row.at["dataset"]
        self.sample = csv_row.at["sample"]
        self.patient = csv_row.at["patient"]
        self.diagnosis = csv_row.at["diagnosis"]
        self.original_intensities = np.loadtxt(
            files.spectra_processed_dataset_intensities / "{}.txt".format(self.sample))
        self.original_masses = np.loadtxt(files.spectra_processed_dataset_masses / "{}.txt".format(self.sample))
        self.coordinates = np.array([
            csv_row.at["coordinates_x"],
            csv_row.at["coordinates_y"],
            csv_row.at["coordinates_z"]
        ], dtype=np.int)

        # TODO: check for intensities < 0
        self.intensities = np.copy(self.original_intensities)
        self.masses = np.copy(self.original_masses)

        self.__check_shape__()

    def normalize_tic(self):
        tic = np.sum(self.intensities)

        if tic > 0.0:
            self.intensities /= tic

    def baseline_median(self, k):

        tmp_spectrum = self.intensities

        tmp_spectrum = iter(tmp_spectrum)
        d = deque()
        s = []
        result = []

        for item in islice(tmp_spectrum, k):
            d.append(item)
            insort(s, item)
            result.append(s[len(d)//2])

        m = k // 2

        for item in tmp_spectrum:
            old = d.popleft()
            d.append(item)
            del s[bisect_left(s, old)]
            insort(s, item)
            result.append(s[m])

        self.intensities -= np.array(result)

        condition = self.intensities < 0

        self.intensities = np.where(condition, 0.0, self.intensities)

    def smoothing_moving_average(self, k):

        tmp_zeros = np.zeros(self.intensities.shape, dtype=np.float)
        tmp_mvgavg = mvgavg(self.intensities, k)

        for i in range(tmp_mvgavg.shape[0]):
            tmp_zeros[i] = tmp_mvgavg[i]

        self.intensities = tmp_zeros

    def is_below_threshold(self, threshold):

        return np.argwhere(self.intensities > threshold).shape[0] == 0

    def __check_shape__(self):

        if self.intensities.shape[0] > self.target_size:

            self.intensities = self.intensities[0:self.target_size]
            self.masses = self.masses[0:self.target_size]

        if self.intensities.shape[0] < self.target_size:

            tmp_intensities = np.zeros((self.target_size,), dtype=np.float)
            tmp_intensities[0:self.intensities.shape[0]] = self.intensities

            medium_diff = 0
            for i in range(1, self.masses.shape[0]):
                medium_diff += self.masses[i] - self.masses[i - 1]

            medium_diff /= self.masses.shape[0]
            target_size_diff = self.target_size - self.masses.shape[0]

            tmp_masses = np.zeros((self.target_size,), dtype=np.float)
            tmp_masses[0:self.masses.shape[0]] = self.masses

            for i in range(1, target_size_diff):
                current_mass_index = self.masses.shape[0] - 1 + i
                tmp_masses[current_mass_index] = self.masses[-1] + i * medium_diff

            self.intensities = tmp_intensities
            self.masses = tmp_masses

    def __repr__(self):
        return "<{} dataset_type={}, patient_id={}, diagnosis={}, sample_id={}, intensities={}, masses={}, coordinates={}>".format(
            type(self).__name__, self.dataset, self.patient, self.diagnosis, self.sample, self.intensities.shape,
            self.masses.shape, self.coordinates)

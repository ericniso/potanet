import config
import numpy as np

from mvgavg import mvgavg
from collections import deque
from bisect import insort, bisect_left
from itertools import islice


class Spectrum:

    def __int__(self, csv_loader, csv_row):

        self.target_size = config.SPECTRUM_SIZE

        self.patient = csv_loader['patient'](csv_row)
        self.diagnosis = csv_loader['diagnosis'](csv_row)
        self.original_intensities = csv_loader['spectrum'](csv_row)
        self.masses = csv_loader['mzs'](csv_row)
        self.coordinates = csv_loader['coordinates'](csv_row)

        self.__check_shape__()

        self.intensities = np.copy(self.original_intensities)

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

    def __check_shape__(self):

        if self.original_intensities.shape[0] > self.target_size:

            self.original_intensities = self.original_intensities[0:self.target_size]
            self.masses = self.masses[0:self.target_size]

        if self.original_intensities.shape[0] < self.target_size:

            tmp_intensities = np.zeros((self.target_size,), dtype=np.float)
            tmp_intensities[0:self.target_size] = self.original_intensities

            medium_diff = 0
            for i in range(1, self.masses.shape[0]):
                medium_diff += self.masses[i] - self.masses[i - 1]

            medium_diff /= self.masses.shape[0] - 1
            target_size_diff = self.target_size - self.masses.shape[0]

            tmp_masses = np.zeros((self.target_size,), dtype=np.float)
            tmp_masses[0:self.target_size] = self.masses

            for i in range(1, target_size_diff):
                current_mass_index = self.masses.shape[0] - 1 + i
                tmp_masses[current_mass_index] = self.masses[-1] + i * medium_diff

            self.original_intensities = tmp_intensities
            self.masses = tmp_masses


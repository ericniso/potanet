import numpy as np
from mvgavg import mvgavg
from collections import deque
from bisect import insort, bisect_left
from itertools import islice

class SpectraNormalizer():
    
    def __init__(self, intensities):

        self.intensities = np.copy(intensities)
        self.intensities = np.reshape(self.intensities, (-1,))

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

    def cut_threshold(self, t):

        condition = self.intensities <= t
        self.intensities = np.where(condition, 0.0, self.intensities)

    def get(self):

        return self.intensities

    # def scalar_normalize(self, val):

    #     self.intensities /= val

    # def baseline_snip(self, k):

    #     intensities_copy = np.copy(self.intensities)
    #     spectra_median = np.zeros(intensities_copy.shape, dtype=np.float)

    #     for r in range(intensities_copy.shape[0]):

    #         for i in range(1, k + 1):
    #             for j in range(i, intensities_copy.shape[1] - i):
    #                 curr_intensity = intensities_copy[r, j]
    #                 curr_median_intensity = (intensities_copy[r, j - i] + intensities_copy[r, j + i]) / 2.0
    #                 spectra_median[r, j] = min(curr_intensity, curr_median_intensity)

    #             intensities_copy = spectra_median

    #     self.intensities -= spectra_median

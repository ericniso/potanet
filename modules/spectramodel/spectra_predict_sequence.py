import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence

class SpectraPredictSequence(Sequence):
    
    def __init__(self, data, spectra_loader, preprocessing=None):

        self.data = data
        self.indexes = []
        self.spectra_loader = spectra_loader
        self.preprocessing = preprocessing

    def __len__(self):

        return self.data.shape[0]

    def all(self):

        rows = self.data

        spectra = []

        for i in range(rows.shape[0]):
            
            row = rows.iloc[i]

            spectrum = self.spectra_loader['spectrum'](row)

            if self.preprocessing is not None:
                spectrum = self.preprocessing(spectrum)

            spectrum = np.reshape(spectrum, (spectrum.shape[0], 1, 1))
            spectra.append(spectrum)

        spectra = np.array(spectra)

        return spectra

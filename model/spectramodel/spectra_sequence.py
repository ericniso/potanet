import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from imblearn.keras import BalancedBatchGenerator

class SpectraSequence(Sequence):
    
    def __init__(self, data, spectra_loader, batch_size=64, preprocessing=None, n_classes=4, test=False):

        assert n_classes >= 2 and n_classes <= 4

        self.data = data
        self.batch_size = batch_size
        self.indexes = []
        self.spectra_loader = spectra_loader
        self.preprocessing = preprocessing
        self.n_classes = n_classes
        self.test = test

        self.on_epoch_end()

    def __len__(self):

        if self.test:
            return (self.data.shape[0] // self.batch_size)
        else:
            return len(self.balanced_batch)

    def __getitem__(self, index):
        
        if self.test:
            indx = self.indexes[index]
            rows = self.data.iloc[np.arange(indx * self.batch_size, (indx + 1) * self.batch_size)]
        else:
            rows, _ = self.balanced_batch[index]

        spectra_batch = []
        diagnosis_batch = []

        for i in range(rows.shape[0]):
            
            row = rows.iloc[i]

            spectrum = self.spectra_loader['spectrum'](row)

            if self.preprocessing is not None:
                spectrum = self.preprocessing(spectrum)

            spectrum = np.reshape(spectrum, (spectrum.shape[0], 1, 1))
            spectra_batch.append(spectrum)
            
            diagnosis = int(self.spectra_loader['diagnosis'](row))

            if self.n_classes == 2:
                diagnosis_batch.append(diagnosis)
            else:
                diagnosis_bits = np.zeros((self.n_classes,), dtype=np.uint8)
                diagnosis_bits[diagnosis] = 1
                diagnosis_batch.append(diagnosis_bits)

        spectra_batch = np.array(spectra_batch)
        diagnosis_batch = np.array(diagnosis_batch)

        return (spectra_batch, diagnosis_batch)

    def on_epoch_end(self):
        
        if self.test:
            self.indexes = np.arange(len(self))
        else:
            self.balanced_batch = BalancedBatchGenerator(self.data, self.data['diagnosis'].values, batch_size=self.batch_size)


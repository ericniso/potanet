import pandas as pd
import numpy as np

class SpectraAugmenter:

    def __init__(self, ratecoeff):
        
        assert ratecoeff > 0 and ratecoeff < 1

        self.ratecoeff = ratecoeff

    def augment(self, spectrum):

        rnd = np.random.rand(spectrum.shape[0])
        
        sign = np.random.rand(spectrum.shape[0])
        sign = np.where(sign < 0.5, sign, 1)
        sign = np.where(sign == 1, sign, -1)
        
        change = spectrum * self.ratecoeff * rnd * sign

        return spectrum + change

def augment_dataset(dataset, augmenter, augment_classes, augment_amount):
    
    assert len(augment_classes) == len(augment_amount)

    for i in range(dataset.shape[0]):

        row = dataset.iloc[i]

        if row.at['diagnosis'] in augment_classes:
            pass

from pathlib import Path
import numpy as np

class DataHandler():

    def load_data(self, path):

        data = np.loadtxt(path)

        for i in range(data.shape[0]):

            yield data[i]


    def count_rows(self, path):

        i = 0

        with path.open('r') as lines:
            for _ in lines:

                i += 1

        return i

    def save_labels(self, labels, path):

        with path.open('a') as o:
            for l in labels:
                o.write('{}\n'.format(l))

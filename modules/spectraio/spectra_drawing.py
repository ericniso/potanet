import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SpectraDrawing():

    def __init__(self):

        self.class_colors = [
            (0, 255, 0), # 0 HP green
            (255, 0, 0), # 1 PTC red
            (220, 220, 220), # 2 Noise gray
            (255, 255, 0), # 3 HT yellow
            (0, 0, 255) # 4 niftp blue
        ]

    def draw(self, coordinates, diagnosis, outpath, multiplier=100):

        x_boundaries = [np.min(coordinates[:, 0]), np.max(coordinates[:, 0])]
        y_boundaries = [np.min(coordinates[:, 1]), np.max(coordinates[:, 1])]

        width = (x_boundaries[1] - x_boundaries[0] + 1) * multiplier
        height = (y_boundaries[1] - y_boundaries[0] + 1) * multiplier

        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        pixels = img.load()

        for i in range(coordinates.shape[0]):

            x = int(coordinates[i][0] - x_boundaries[0])
            y = int(coordinates[i][1] - y_boundaries[0])

            for j in range(multiplier):
                for k in range(multiplier):
                    color = self.class_colors[diagnosis[i]]
                    pixels[multiplier * x + j, multiplier * y + k] = color

        img.save(outpath, format='PNG')

    def plot_probs(self, diagnosis_probs, outdir, classes, n_bins=10):

        diagnosis_class = np.argmax(diagnosis_probs, axis=1)

        bins_x = np.linspace(0, 1, n_bins + 1)
        x_ticks = np.arange(n_bins + 1)
        xticks_labels = []
        for i in range(n_bins + 1):
            xticks_labels.append('{:.1f}'.format(bins_x[i]))

        bins_count = []
        for i in range(len(classes)):
            bins_count.append(np.zeros((n_bins + 1,), dtype=np.int32))

        for i in range(diagnosis_class.shape[0]):
            
            current_class = diagnosis_class[i]
            current_prob = diagnosis_probs[i, current_class]

            for j in range(1, bins_x.shape[0]):

                if current_prob <= bins_x[j]:
                    bins_count[current_class][j - 1] += 1
                    break

        bins_count = np.array(bins_count)
        bins_columns_count = np.sum(bins_count, axis=0) # + np.finfo(np.float32).eps
        # bins_count = bins_count / bins_columns_count
        max_y = np.max(bins_columns_count)

        for i in range(len(classes)):

            legend_handle = [plt.Rectangle((0, 0), 1, 1, color=classes[i]['color'])]
            plt.xticks(x_ticks, xticks_labels)
            plt.bar(x_ticks, bins_count[i], width=1.0, align='edge', color=classes[i]['color'])
            plt.xlabel('Class probability')
            plt.ylabel('Pixel count')
            # plt.xlim((0.0, 1.0))
            plt.ylim((0.0, max_y + 0.1 * max_y))
            # plt.ylim((0.0, 1.0))
            
            plt.legend(legend_handle, [classes[i]['name']], loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=False, ncol=1)
            plt.savefig(outdir / '{}.pdf'.format(classes[i]['name']))
            plt.close()

        # print(bins_count)


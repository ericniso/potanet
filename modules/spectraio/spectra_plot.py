from .files import Files
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class SpectraPlot():

    def __init__(self):

        self.files = Files()

    def plot_metrics(self, series, title_font=12, legend_font=16, outfolder=None):

        validation = False

        for k, _ in series.items():

            if 'val_' in k:
                validation = True
                break

        metrics = [k for k, _ in series.items()]
        metrics = [m for m in metrics if 'val_' not in m]

        for m in metrics:

            val_m = 'val_{}'.format(m)

            if outfolder is None:
                _ = plt.figure()
                plt.title(m)
            else:
                plt.clf()
            
            if m in series:
                plt.plot(series[m], linestyle='--', label=m)
            
            if validation == True and val_m in series:
                plt.plot(series[val_m], label=val_m)

            if m != 'loss' and m != 'lr':
                plt.ylim((0.0, 1.0))
            
            if m == 'lr':
                lr_max = np.max(series[m])
                lr_max += lr_max / 10
                plt.ylim((0.0, lr_max))

            if outfolder is None:
                plt.yticks(fontsize=title_font)
                plt.xticks(fontsize=title_font)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=legend_font)
                plt.tight_layout()
            else:
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

            if outfolder is not None:
                plt.tight_layout()
                plt.savefig(outfolder / '{}.pdf'.format(m))

        if outfolder is None:
            plt.show()

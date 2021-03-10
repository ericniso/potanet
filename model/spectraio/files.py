from pathlib import Path

class Files():

    def __init__(self, root=None):

        if root == None:
            self.root = Path('/data')
        else:
            self.root = Path(root)

        self.out_path = self.root / 'output'

        self.spectra_plot_out_dir = self.out_path / 'spectra_plot'

        self.isotopenet_out_dir = self.out_path / 'isotopenet'
        self.isotopenet_out_path = self.isotopenet_out_dir / 'isotopenet.h5'
        self.isotopenet_result_path = self.isotopenet_out_dir / 'isotopenet.joblib'

        # ROI single spectrum

        self.roi_single_spectra_path_root = self.root / 'imzML_roi_single_spectra'
        
        self.roi_single_spectra_path = self.roi_single_spectra_path_root / 'dataset'
        self.roi_single_spectra_path_data = self.roi_single_spectra_path / 'spectra.csv'
        self.roi_single_spectra_path_intensities = self.roi_single_spectra_path / 'intensities'
        self.roi_single_spectra_path_mzs = self.roi_single_spectra_path / 'mzs'
        self.roi_single_spectra_path_coordinates = self.roi_single_spectra_path / 'coordinates'

        self.roi_single_spectra_path_data_single_root = self.roi_single_spectra_path_root / '_single'
        self.roi_single_spectra_path_data_single = self.roi_single_spectra_path_data_single_root / 'spectra.csv'
        self.roi_single_spectra_path_intensities_single = self.roi_single_spectra_path_data_single_root / 'intensities'
        self.roi_single_spectra_path_mzs_single = self.roi_single_spectra_path_data_single_root / 'mzs'
        self.roi_single_spectra_path_coordinates_single = self.roi_single_spectra_path_data_single_root / 'coordinates'
        
        self.roi_single_spectra_path_data_avg_root = self.roi_single_spectra_path_root / '_avg'
        self.roi_single_spectra_path_data_avg = self.roi_single_spectra_path_data_avg_root / 'spectra.csv'
        self.roi_single_spectra_path_intensities_avg = self.roi_single_spectra_path_data_avg_root / 'intensities'
        self.roi_single_spectra_path_mzs_avg = self.roi_single_spectra_path_data_avg_root / 'mzs'
        self.roi_single_spectra_path_coordinates_avg = self.roi_single_spectra_path_data_avg_root / 'coordinates'


import config
from pathlib import Path


class Files:

    def __init__(self):

        self.root = Path(config.POTANET_ROOT_DIR)

        self.out_path = self.root / "output"

        self.spectra_plot_out_dir = self.out_path / "spectra_plot"

        self.isotopenet_out_dir = self.out_path / "isotopenet"
        self.isotopenet_out_path = self.isotopenet_out_dir / "isotopenet.h5"
        self.isotopenet_result_path = self.isotopenet_out_dir / "isotopenet.joblib"

        self.spectra_processed_path_root = self.root / "imzML_processed"
        
        self.spectra_processed_dataset = self.spectra_processed_path_root / "dataset"
        self.spectra_processed_dataset_data = self.spectra_processed_dataset / "spectra.csv"
        self.spectra_processed_dataset_intensities = self.spectra_processed_dataset / "intensities"
        self.spectra_processed_dataset_masses = self.spectra_processed_dataset / "masses"

        self.spectra_processed_dataset_single = self.spectra_processed_path_root / "_single"
        self.spectra_processed_dataset_single_data = self.spectra_processed_dataset_single / "spectra.csv"
        self.spectra_processed_dataset_single_intensities = self.spectra_processed_dataset_single / "intensities"
        self.spectra_processed_dataset_single_masses = self.spectra_processed_dataset_single / "masses"

        self.spectra_processed_dataset_avg = self.spectra_processed_path_root / "_avg"
        self.spectra_processed_dataset_avg_data = self.spectra_processed_dataset_avg / "spectra.csv"
        self.spectra_processed_dataset_avg_intensities = self.spectra_processed_dataset_avg / "intensities"
        self.spectra_processed_dataset_avg_masses = self.spectra_processed_dataset_avg / "masses"


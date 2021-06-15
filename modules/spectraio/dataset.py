import config
import pandas as pd
from pathlib import Path

class Dataset():

    def __init__(self, root=config.POTANET_ROOT_DIR):

        if root is None:
            self.root = Path('/data')
        else:
            self.root = Path(root)

        # ROI single spectrum
        self.roi_single_spectra_extracted_path = self.root / 'imzML_roi_single_spectra_extracted'

        # Samples diagnosis
        # 0: HP
        # 1: PTC
        # 2: Noise
        # 3: HT
        # 4: niftp
        self.diagnosis_to_index = {
            'HP': 0,
            'PTC': 1,
            'Noise': 2,
            'HT': 3,
            'niftp': 4
        }

        self.raw_dataset = pd.read_csv(Path(__file__).parent / 'dataset' / 'dataset.csv')

        # self.roi_single_specta_extracted_patients_training = [
        #     { 'id': '213', 'diagnosis': 1 }, { 'id': '250', 'diagnosis': 1 }, { 'id': '262', 'diagnosis': 0 }, { 'id': '268', 'diagnosis': 0 }, 
        #     { 'id': '302', 'diagnosis': 0 }, { 'id': '308', 'diagnosis': 0 }, { 'id': '381', 'diagnosis': 0 }, { 'id': '384', 'diagnosis': 0 }, 
        #     { 'id': '442', 'diagnosis': 1 }, { 'id': '475', 'diagnosis': 0 }, { 'id': '565', 'diagnosis': 0 }, { 'id': '598', 'diagnosis': 3 },
        #     { 'id': '647', 'diagnosis': 3 }, { 'id': '922', 'diagnosis': 3 }, { 'id': '992', 'diagnosis': 1 }, { 'id': '995', 'diagnosis': 1 }, 
        #     { 'id': '1012', 'diagnosis': 1 }, {'id': '1046', 'diagnosis': 0 }, { 'id': '1047', 'diagnosis': 3 }, { 'id': '1074', 'diagnosis': 3 }, 
        #     { 'id':'1076', 'diagnosis': 1 }, { 'id': '1083', 'diagnosis': 3 }, { 'id': '1144', 'diagnosis': 3 }, { 'id': '1145', 'diagnosis': 3 }, 
        #     { 'id': '1208', 'diagnosis': 3 }, { 'id': '1126_exvivo', 'diagnosis': 1 }, { 'id': '1187_exvivo', 'diagnosis': 1 }
        # ]

        # self.roi_single_specta_extracted_patients_validation = [
        #     { 'id': '278', 'diagnosis': 0 }, { 'id': '290', 'diagnosis': 4 }, { 'id': '316', 'diagnosis': 3 }, { 'id': '387', 'diagnosis': 4 }, 
        #     { 'id': '436', 'diagnosis': 1 }, { 'id': '440', 'diagnosis': 1 }, { 'id': '520', 'diagnosis': 3 }, { 'id': '525', 'diagnosis': 3 }, 
        #     { 'id': '609', 'diagnosis': 4 }, { 'id': '621', 'diagnosis': 4 }, { 'id': '935', 'diagnosis': 4 }, { 'id': '987', 'diagnosis': 4 }, 
        #     { 'id': '1075', 'diagnosis': 3 }, {'id': '1081', 'diagnosis': 3 }, {'id': '1082', 'diagnosis': 3 }, {'id':'1084', 'diagnosis': 1 }, 
        #     { 'id': '1122', 'diagnosis': 3 }, {'id': '1123', 'diagnosis': 0 }, {'id': '1126', 'diagnosis': 1 }, {'id':'1147', 'diagnosis': 1 }, 
        #     { 'id': '1149', 'diagnosis': 1 }, {'id': '1156', 'diagnosis': 0 }, {'id': '1172', 'diagnosis': 1 }, {'id':'1187', 'diagnosis': 1 }, 
        #     { 'id': '1188', 'diagnosis': 1 }, {'id': '1202', 'diagnosis': 1 }, {'id': '1234', 'diagnosis': 3 }, {'id':'1283', 'diagnosis': 4 }, 
        #     { 'id': '1294', 'diagnosis': 4 }, {'id': '1328', 'diagnosis': 4 }, {'id': '1331', 'diagnosis': 3 }
        # ]

        # self.roi_single_specta_extracted_patients_validation_exvivo = [
        #     { 'id': '1331_exvivo', 'diagnosis': 3 }, { 'id': '1084_exvivo', 'diagnosis': 1 }, { 'id': '1188_exvivo', 'diagnosis': 1 }, { 'id': '1283_exvivo', 'diagnosis': 4 }, 
        #     { 'id': '1294_exvivo', 'diagnosis': 4 }, { 'id': '1328_exvivo', 'diagnosis': 4 }, { 'id': '250_exvivo', 'diagnosis': 1 }, { 'id': '290_exvivo', 'diagnosis': 4 }, 
        #     { 'id': '992_exvivo', 'diagnosis': 1 }, { 'id': '995_exvivo', 'diagnosis': 1 }, { 'id': '1076_exvivo', 'diagnosis': 1 }, { 'id': '1147_exvivo', 'diagnosis': 1 }, 
        #     { 'id': '1172_exvivo', 'diagnosis': 1 }
        # ]

    def get(self, dataset_type):

        assert dataset_type in ['training', 'validation', 'validation_exvivo']

        selected_dataset = self.raw_dataset[self.raw_dataset['type'] == dataset_type].values
        patient_column = list(self.raw_dataset.columns).index('patient')
        diagnosis_column = list(self.raw_dataset.columns).index('diagnosis')

        if dataset_type == 'training':

            return {
                # 'mzs' : [self.roi_single_spectra_extracted_path / 'training' / p['id'] / 'mzs.txt' for p in self.roi_single_specta_extracted_patients_training],
                # 'coordinates' : [self.roi_single_spectra_extracted_path / 'training' / p['id'] / 'coordinates.txt' for p in self.roi_single_specta_extracted_patients_training],
                # 'intensities' : [self.roi_single_spectra_extracted_path / 'training' / p['id'] / 'intensities.txt' for p in self.roi_single_specta_extracted_patients_training],
                # 'patients' : [p['id'] for p in self.roi_single_specta_extracted_patients_training],
                # 'diagnosis' : [p['diagnosis'] for p in self.roi_single_specta_extracted_patients_training],

                'mzs': [self.roi_single_spectra_extracted_path / 'training' / p[patient_column] / 'mzs.txt' for p in selected_dataset],
                'coordinates': [self.roi_single_spectra_extracted_path / 'training' / p[patient_column] / 'coordinates.txt' for p in selected_dataset],
                'intensities': [self.roi_single_spectra_extracted_path / 'training' / p[patient_column] / 'intensities.txt' for p in selected_dataset],
                'patients': [p[patient_column] for p in selected_dataset],
                'diagnosis': [self.diagnosis_to_index[p[diagnosis_column]] for p in selected_dataset]
            }

        if dataset_type == 'validation':

            return {
                # 'mzs' : [self.roi_single_spectra_extracted_path / 'validation' / p['id'] / 'mzs.txt' for p in self.roi_single_specta_extracted_patients_validation],
                # 'coordinates' : [self.roi_single_spectra_extracted_path / 'validation' / p['id'] / 'coordinates.txt' for p in self.roi_single_specta_extracted_patients_validation],
                # 'intensities' : [self.roi_single_spectra_extracted_path / 'validation' / p['id'] / 'intensities.txt' for p in self.roi_single_specta_extracted_patients_validation],
                # 'patients' : [p['id'] for p in self.roi_single_specta_extracted_patients_validation],
                # 'diagnosis' : [p['diagnosis'] for p in self.roi_single_specta_extracted_patients_validation],

                'mzs': [self.roi_single_spectra_extracted_path / 'validation' / p[patient_column] / 'mzs.txt' for p in selected_dataset],
                'coordinates': [self.roi_single_spectra_extracted_path / 'validation' / p[patient_column] / 'coordinates.txt' for p in selected_dataset],
                'intensities': [self.roi_single_spectra_extracted_path / 'validation' / p[patient_column] / 'intensities.txt' for p in selected_dataset],
                'patients': [p[patient_column] for p in selected_dataset],
                'diagnosis': [self.diagnosis_to_index[p[diagnosis_column]] for p in selected_dataset]
            }

        if dataset_type == 'validation_exvivo':

            return {
                # 'mzs' : [self.roi_single_spectra_extracted_path / 'validation_exvivo' / p['id'] / 'mzs.txt' for p in self.roi_single_specta_extracted_patients_validation_exvivo],
                # 'coordinates' : [self.roi_single_spectra_extracted_path / 'validation_exvivo' / p['id'] / 'coordinates.txt' for p in self.roi_single_specta_extracted_patients_validation_exvivo],
                # 'intensities' : [self.roi_single_spectra_extracted_path / 'validation_exvivo' / p['id'] / 'intensities.txt' for p in self.roi_single_specta_extracted_patients_validation_exvivo],
                # 'patients' : [p['id'] for p in self.roi_single_specta_extracted_patients_validation_exvivo],
                # 'diagnosis' : [p['diagnosis'] for p in self.roi_single_specta_extracted_patients_validation_exvivo],

                'mzs': [self.roi_single_spectra_extracted_path / 'validation_exvivo' / p[patient_column] / 'mzs.txt' for p in selected_dataset],
                'coordinates': [self.roi_single_spectra_extracted_path / 'validation_exvivo' / p[patient_column] / 'coordinates.txt' for p in selected_dataset],
                'intensities': [self.roi_single_spectra_extracted_path / 'validation_exvivo' / p[patient_column] / 'intensities.txt' for p in selected_dataset],
                'patients': [p[patient_column] for p in selected_dataset],
                'diagnosis': [self.diagnosis_to_index[p[diagnosis_column]] for p in selected_dataset]
            }

def full_dataset(d):

    return d

def balanced_dataset(d, n_classes):

    with (Path(__file__).parent / 'dataset' / 'dataset_{}_classes.txt'.format(n_classes)).open() as f:
        selected_patients = [p.strip() for p in f.readlines()]

    new_balanced_dataset = pd.DataFrame([], columns=d.columns)

    for p in selected_patients:
        new_balanced_dataset = new_balanced_dataset.append(
            d[d['patient'] == p]
        )

    new_balanced_dataset_filtered = pd.DataFrame([], columns=new_balanced_dataset.columns)
    for i in [0, 1, 2, 3, 4]:
        if i >= n_classes:
            break

        new_balanced_dataset_filtered = new_balanced_dataset_filtered.append(
            new_balanced_dataset[new_balanced_dataset['diagnosis'] == i]
        )

    return new_balanced_dataset_filtered

def test_dataset(d, f):

    filtered = f
    exclude_patients = list(filtered['patient'].unique())

    test_data = pd.DataFrame([], columns = d.columns)

    for p in list(d['patient'].unique()):

        if p in exclude_patients:
            continue

        current_patient = d[d['patient'] == p]
        test_data = test_data.append(current_patient, ignore_index=True)

    return test_data

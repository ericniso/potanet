import numpy as np

def csv_loader(root):

    spectra_loader = lambda r: np.loadtxt(root / 'intensities' / '{}.txt'.format(r.at['spectrum']))
    diagnosis_loader = lambda r: int(r.at['diagnosis'])
    patient_loader = lambda r: str(r.at['patient'])
    mzs_loader = lambda r: np.loadtxt(root / 'mzs' / '{}.txt'.format(r.at['mzs']))
    coordinates_loader = lambda r: np.loadtxt(root / 'coordinates' / '{}.txt'.format(r.at['coordinates']), dtype=np.int)

    return {
        'spectrum': spectra_loader, 
        'diagnosis': diagnosis_loader, 
        'patient': patient_loader, 
        'mzs': mzs_loader, 
        'coordinates': coordinates_loader
    }

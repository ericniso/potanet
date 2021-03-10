import pandas as pd
import numpy as np
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from model import Files, SpectraNormalizer, csv_loader
from model import balanced_dataset, test_dataset, full_dataset
from model import SpectraAugmenter

files = Files()

loader = csv_loader(files.roi_single_spectra_path)
dataset = pd.read_csv(files.roi_single_spectra_path_data)

# loader = csv_loader(files.roi_single_spectra_path_data_single_root)
# dataset = pd.read_csv(files.roi_single_spectra_path_data_single)

# loader = csv_loader(files.roi_single_spectra_path_data_avg_root)
# dataset = pd.read_csv(files.roi_single_spectra_path_data_avg)

training_dataset = balanced_dataset(dataset, 4)
test_dataset = test_dataset(dataset, training_dataset)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex='https?://.*',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/{dataset_type}/spectra')
def spectra(response: Response, dataset_type: str = 'training', page: int = -1, page_size: int = 15):

    if dataset_type == 'test':
        current_dataset = test_dataset
    else:
        current_dataset = training_dataset

    if page == -1:
        return {
            'current_page': -1,
            'total_pages': -1,
            'data': current_dataset.to_dict('records')
        }

    total_pages = current_dataset.shape[0] // page_size
    
    if page < -1 or page > total_pages or page_size < 1:
        response.status_code = status.HTTP_404_NOT_FOUND
        return

    start_idx = page * page_size
    end_idx = min(current_dataset.shape[0], start_idx + page_size)
    dataset_paged = current_dataset.iloc[start_idx : end_idx]

    return {
        'current_page': page,
        'total_pages': total_pages,
        'data': dataset_paged.to_dict('records')
    }

@app.get('/{dataset_type}/spectra/filter')
def spectra_filter(response: Response, dataset_type: str = 'training', spectrum: str = '', patient: str = '', diagnosis: int = -1, page: int = -1, page_size: int = 15):

    if dataset_type == 'test':
        current_dataset = test_dataset
    else:
        current_dataset = training_dataset

    filtered = current_dataset

    if patient:
        filtered = filtered[filtered['patient'].str.contains(patient)]

    if spectrum:
        filtered = filtered[filtered['spectrum'].str.contains(spectrum)]

    if diagnosis in [0, 1, 2]:
        filtered = filtered[filtered['diagnosis'] == diagnosis]

    if page == -1:
        return {
            'current_page': -1,
            'total_pages': -1,
            'data': filtered.to_dict('records')
        }

    total_pages = filtered.shape[0] // page_size
    
    if page < -1 or page > total_pages or page_size < 1:
        response.status_code = status.HTTP_404_NOT_FOUND
        return

    start_idx = page * page_size
    end_idx = min(filtered.shape[0], start_idx + page_size)
    filtered_dataset_paged = filtered.iloc[start_idx : end_idx]

    return {
        'current_page': page,
        'total_pages': total_pages,
        'data': filtered_dataset_paged.to_dict('records')
    }

@app.get('/{dataset_type}/spectra/{spectrum_id}')
def spectrum_data(response: Response, spectrum_id: str, dataset_type: str = 'training'):

    if dataset_type == 'test':
        current_dataset = test_dataset
    else:
        current_dataset = training_dataset

    items = current_dataset[current_dataset['spectrum'] == spectrum_id]
    if items.shape[0] > 0:
        intensities = loader['spectrum'](items.iloc[0])
        mzs = loader['mzs'](items.iloc[0])
        coordinates = loader['coordinates'](items.iloc[0])
        return { 
            'intensities': intensities.tolist(), 
            'mzs': mzs.tolist(), 
            'coordinates': coordinates.astype(np.int32).tolist() 
        }
    else:
        response.status_code = status.HTTP_404_NOT_FOUND

@app.get('/{dataset_type}/spectra/{spectrum_id}/preprocess')
def spectrum_process(response: Response, spectrum_id: str, dataset_type: str = 'training', normalize_tic: bool = False, baseline_median: int = 0, smoothing_moving_average: int = 0, cut_threshold: float = -1.0):
    
    if dataset_type == 'test':
        current_dataset = test_dataset
    else:
        current_dataset = training_dataset

    items = current_dataset[current_dataset['spectrum'] == spectrum_id]
    if items.shape[0] > 0:
        original_intensities = loader['spectrum'](items.iloc[0])
        mzs = loader['mzs'](items.iloc[0])
        coordinates = loader['coordinates'](items.iloc[0])

        normalizer = SpectraNormalizer(np.copy(original_intensities))

        if baseline_median > 0:
            normalizer.baseline_median(baseline_median)

        if smoothing_moving_average > 0:
            normalizer.smoothing_moving_average(smoothing_moving_average)

        if normalize_tic:
            original_normalizer = SpectraNormalizer(original_intensities)
            original_normalizer.normalize_tic()
            original_intensities = original_normalizer.get()
            normalizer.normalize_tic()

        if cut_threshold > 0:
            normalizer.cut_threshold(cut_threshold)

        intensities = normalizer.get()

        return { 
            'original_intensities': original_intensities.tolist(),
            'preprocessed_intensities': intensities.tolist(),
            'mzs': mzs.tolist(), 
            'coordinates': coordinates.astype(np.int32).tolist() 
        }
    else:
        response.status_code = status.HTTP_404_NOT_FOUND

@app.get('/{dataset_type}/spectra/{spectrum_id}/augment')
def spectrum_augment(response: Response, spectrum_id: str, dataset_type: str = 'training', augment_rate: float = 0.5):
    
    if dataset_type == 'test':
        current_dataset = test_dataset
    else:
        current_dataset = training_dataset

    if augment_rate <= 0 or augment_rate >= 1:
        response.status_code = status.HTTP_404_NOT_FOUND
        return

    items = current_dataset[current_dataset['spectrum'] == spectrum_id]
    if items.shape[0] > 0:
        original_intensities = loader['spectrum'](items.iloc[0])
        mzs = loader['mzs'](items.iloc[0])
        coordinates = loader['coordinates'](items.iloc[0])

        augmenter = SpectraAugmenter(augment_rate)

        intensities = augmenter.augment(original_intensities)

        return { 
            'original_intensities': original_intensities.tolist(),
            'preprocessed_intensities': intensities.tolist(),
            'mzs': mzs.tolist(), 
            'coordinates': coordinates.astype(np.int32).tolist() 
        }
    else:
        response.status_code = status.HTTP_404_NOT_FOUND

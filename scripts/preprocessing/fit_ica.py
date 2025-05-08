import os
import mne
from autoreject import read_auto_reject, validation_curve, get_rejection_threshold, set_matplotlib_defaults
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from joblib import cpu_count

from utils.preprocessing_tools import prepare_ica_epochs, ica_fit
from utils.file_io import load_raw_data, save_ica, log_reject_threshold
from utils.config import DATASETS, set_plot_style, EEG_SETTINGS, PLOTS_PATH
from utils.helpers import iterate_dataset_items, format_numbers

set_plot_style()

VERBOSE = True
SHOW_PLOTS = False

#DATASETS.pop('braboszcz2017')
DATASETS.pop('jin2019')
DATASETS.pop('touryan2022')

for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    # Load and prepare raw data
    raw = load_raw_data(
        dataset, 
        subject, 
        **kwargs, 
        verbose=False
    )

    if raw is None:
        tqdm.write(f"    No data found for subject {subject} with {label} {item}.")
        continue
    
    # Prepare epochs for ICA
    epochs = prepare_ica_epochs(
        raw, 
        dataset, 
        EEG_SETTINGS, 
        subject, 
        **kwargs, 
        verbose=VERBOSE,
        min_threshold=300e-6,
        reject_scale_factor=1.5
    )

    # Fit ICA
    ica = ica_fit(
        epochs,
        EEG_SETTINGS,
        verbose=VERBOSE,
    )

    # Save ICA
    save_ica(
        ica,
        dataset,
        subject,
        **kwargs,
        verbose=VERBOSE
    )

    # Save epochs
    save_path = os.path.join(dataset.path_epochs, 'ica_epochs')
    os.makedirs(save_path, exist_ok=True)
    epochs.save(os.path.join(save_path, f"sub-{subject}_{label}-{item}_ica-epo.fif"), overwrite=True)

    
import os

from tqdm import tqdm

from utils.helpers import iterate_dataset_items, format_numbers
from utils.preprocessing_tools import prepare_raw_data, fix_bad_channels, autoreject_raw
from utils.file_io import load_raw_data, save_autoreject
from utils.config import DATASETS, set_plot_style, EEG_SETTINGS

set_plot_style()

VERBOSE = True

DATASETS['braboszcz2017'].subjects = []
DATASETS['jin2019'].subjects = []


for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    raw = load_raw_data(dataset, subject, **kwargs, verbose=VERBOSE)
    if raw is None:
        tqdm.write(f"    No data found for subject {subject} with {label} {item}.")
        continue

    raw = prepare_raw_data(raw, dataset, EEG_SETTINGS)

    # Fix bad channels in the raw data
    raw = fix_bad_channels(raw, dataset, subject, **kwargs, verbose=VERBOSE)

    # Average reference
    raw.set_eeg_reference(ref_channels='average', projection=False)

    # Fit autoreject
    ar = autoreject_raw(
        raw,
        EEG_SETTINGS,
        verbose=VERBOSE,
    )

    # Save the model
    save_path = os.path.join(dataset.path_derivatives, 'pre_ica_autoreject_models')
    os.makedirs(save_path, exist_ok=True)
    save_autoreject(ar, os.path.join(save_path, f"sub-{subject}_{label}-{item}_autoreject_pre_ica.h5"))
import os
import mne
from mne.preprocessing import create_eog_epochs
from mne import set_bipolar_reference
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from joblib import cpu_count

from utils.preprocessing_tools import prepare_raw_data, prepare_ica_epochs
from utils.file_io import load_raw_data, load_ica
from utils.config import DATASETS, set_plot_style, EEG_SETTINGS, PLOTS_PATH
from utils.helpers import iterate_dataset_items, format_numbers

set_plot_style()

VERBOSE = True
SHOW_PLOTS = False

# Subjects to inspect in braboszcz2017 dataset
subjects = []
# subjects.extend(range(25, 56)) # control group
# subjects.extend(range(60, 79)) # vipassana group
# subjects = format_numbers(subjects, 3)  # convert to list of strings with leading zeros like ["060", "061", ...]

DATASETS.pop('braboszcz2017')
DATASETS.pop('jin2019')


for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    epochs = mne.read_epochs(os.path.join(dataset.path_epochs, 'ica_epochs', f"sub-{subject}_{label}-{item}_ica-epo.fif"))

    ica = load_ica(
        dataset,
        subject,
        **kwargs,
        verbose=VERBOSE
    )

    # Set bipolar reference for EOG channels
    epochs = set_bipolar_reference(
        epochs,
        anode='UVEOG',
        cathode='LVEOG',
        ch_name='VEOG',
        drop_refs=False,  # Keep UVEOG and LVEOG in case you want them too
        copy=True
    )
    # Set bipolar reference for EOG channels
    epochs = set_bipolar_reference(
        epochs,
        anode='LHEOG',
        cathode='RHEOG',
        ch_name='HEOG',
        drop_refs=False, 
        copy=True
    )

    # --- BLINKS (Vertical EOG) ---

    blink_inds, blink_scores = ica.find_bads_eog(
        epochs,
        ch_name=['VEOG'],
        threshold=4.5,
        measure='zscore'
    )
    print(f"Subject: {subject}, Item: {item} - Blink components (z>3): {blink_inds}")

    # --- SACCADES (Horizontal EOG) ---

    saccade_inds, saccade_scores = ica.find_bads_eog(
        epochs,
        ch_name=['HEOG'],
        threshold=5,
        measure='zscore'
    )
    print(f"Subject: {subject}, Item: {item} - Saccade components (z>4.5): {saccade_inds}")


    # Set ICA to exclude detected components
    ica.exclude = blink_inds + saccade_inds

    # --- Visual inspection (strongly recommended) ---
    if blink_inds:
        ica.plot_properties(epochs, picks=blink_inds, psd_args={'fmax': 40.}, show=True)

    if saccade_inds:
        ica.plot_properties(epochs, picks=saccade_inds, psd_args={'fmax': 40.}, show=True)


    # --- Apply ICA ---
    # epochs_clean = ica.apply(epochs.copy())

    # (Optional) Save cleaned epochs for downstream analyses
    # epochs_clean.save(os.path.join(dataset.path_epochs, 'clean_epochs', f"sub-{subject}_{label}-{item}_clean-epo.fif"), overwrite=True)

    # blinks
    # ica.plot_overlay(epochs, exclude=eog_indices, picks="eeg")

    # barplot of ICA component "EOG match" scores
    #ica.plot_scores(eog_scores)

    # plot diagnostics
    #ica.plot_properties(epochs, picks=eog_indices)

    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(epochs, block=True, show_scrollbars=True)


    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    #ica.plot_sources(eog_evoked)
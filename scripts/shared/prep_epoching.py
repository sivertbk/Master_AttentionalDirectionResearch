import os
import mne
import numpy as np
import warnings
import matplotlib.pyplot as plt

from utils.config import DATASETS, set_plot_style, EEG_SETTINGS
from utils.helpers import iterate_dataset_items
from utils.file_io import load_ica, load_raw_data, load_ica_excluded_components
from utils.preprocessing_tools import prepare_raw_data, fix_bad_channels

set_plot_style()







for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    ##----------------------------------------------------------------------------##
    #                    1. LOAD AND PREPARE RAW EEG DATA                          #
    ##----------------------------------------------------------------------------##
    ''' Load the raw data and create copies for further processing. All channel
        types will be set correctly. Data will be downsampled to sampling rate
        defined in config.py.

        RESULTS: 
        - Raw data objects from the respective dataset ready to be preprocessed.
    '''
    raw = load_raw_data(dataset, subject, **kwargs, verbose=True)

    if raw is None:
        print(f"[WARN] No data found for subject {subject} with {label} {item}.")
        continue

    prepare_raw_data(raw, dataset, EEG_SETTINGS)
    raw.set_eeg_reference('average', verbose=True, projection=False)

    ##----------------------------------------------------------------------------##
    #               2. INTERPOLATE BAD CHANNELS IN ORIGINAL RAW                    #
    ##----------------------------------------------------------------------------##
    ''' Interpolate bad channels in the original raw data based on RANSAC results.

        RESULTS:
        - Cleaned raw object with interpolated channels
    '''
    fix_bad_channels(raw, dataset, subject=subject, **kwargs, verbose=True)

    ##----------------------------------------------------------------------------##
    #            3. APPLY ICA TO FULL UNFILTERED RAW (CLEANED VERSION)             #
    ##----------------------------------------------------------------------------##
    ''' Apply the ICA solution (trained on clean synthetic epochs) to the full,
        unfiltered raw data. This ensures all data benefits from artifact removal.

        RESULTS:
        - ICA-cleaned raw object
    '''
    ica = load_ica(dataset, subject, **kwargs, verbose=True)
    if ica is None:
        print(f"[WARN] No ICA found for subject {subject} with {label} {item}.")

    components_to_exclude = load_ica_excluded_components(dataset, subject, **kwargs, verbose=True)
    if components_to_exclude is None:
        print(f"[WARN] No excluded components found for subject {subject} with {label} {item}.")

    raw = ica.apply(raw, exclude=components_to_exclude, verbose=True)

    # eog_epochs = create_eog_epochs(raw)
    # Do some plotting


    ##----------------------------------------------------------------------------##
    #      4. EPOCH THE CLEAN RAW FOR ANALYSIS AND APPLY DETRENDING                #
    ##----------------------------------------------------------------------------##
    ''' Epoch the ICA-cleaned raw based on experimental events and apply linear
        detrending to preserve frequency information while removing slow drifts.

        RESULTS:
        - Analysis-ready epochs
    '''



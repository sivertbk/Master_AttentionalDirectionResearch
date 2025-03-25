import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pickle
from joblib import cpu_count

import utils.config as config
from utils.config import DATASETS, EEG_SETTINGS
from utils.file_io import save_psd_data, load_epochs


# Specify the subfolder for the epochs path
EPOCHS_SUBFOLDER = "ica_cleaned"

def detect_data_to_compute(dataset_name):
    """
    Detect and return list of subject files to compute PSD for.
    Currently, this is a placeholder that returns all subjects.

    Returns:
        list: list of subject identifiers to process
    """
    dataset_config = DATASETS[dataset_name]
    return dataset_config.subjects

def separate_epochs_by_class(epochs, event_classes):
    """
    Separate epochs by event classes.

    Parameters:
        epochs (mne.Epochs): EEG epochs data
        event_classes (dict): Dictionary mapping class names to event IDs

    Returns:
        dict: Dictionary with class names as keys and corresponding epochs as values
    """
    epochs_by_class = {}
    for class_name, event_ids in event_classes.items():
        epochs_by_class[class_name] = epochs[event_ids]
    return epochs_by_class

def compute_psd(epochs, fmin=EEG_SETTINGS["PSD_FMIN"], fmax=EEG_SETTINGS["PSD_FMAX"], 
                n_fft=EEG_SETTINGS["PSD_N_FFT"], n_overlap=EEG_SETTINGS["PSD_N_OVERLAP"], 
                window=EEG_SETTINGS["PSD_WINDOW"], remove_dc=EEG_SETTINGS["PSD_REMOVE_DC"]):
    """
    Compute PSD using Welch's method for given epochs.

    Parameters:
        epochs (mne.Epochs): EEG epochs data
        fmin, fmax (float): frequency range for PSD
        n_fft, n_overlap (int): FFT parameters matching Rayleigh frequency
        window (str): window taper to use

    Returns:
        tuple: PSD array (epochs x channels x frequencies), frequency array
    """
    n_jobs = cpu_count()  # Determine the number of jobs based on CPU cores

    psd, freqs = mne.time_frequency.psd_array_welch(
        epochs,
        sfreq=epochs.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        window=window,
        average='mean',
        picks='eeg',
        output='power',
        n_jobs=n_jobs,
        remove_dc=remove_dc  
    )

    return psd, freqs

def compute_psd_data(dataset_name):
    """
    Compute the PSD data for all subjects in a given dataset.

    Parameters:
        dataset_name (str): Identifier for dataset configuration.
    """
    dataset_config = DATASETS[dataset_name]
    subjects = detect_data_to_compute(dataset_name)

    for subject in subjects:
        for session in dataset_config.sessions:
            print(f"Processing subject {subject}, session {session}")

            # Load epochs
            epochs_path = os.path.join(dataset_config.path_epochs, EPOCHS_SUBFOLDER)
            epochs = load_epochs(epochs_path, subject, session)

            if epochs is None:
                continue

            # Separate epochs by class
            epochs_by_class = separate_epochs_by_class(epochs, dataset_config.event_classes)

            # Compute PSD for each class
            psd_data = {}
            for class_name, class_epochs in epochs_by_class.items():
                psd, freqs = compute_psd(class_epochs)
                psd_data[class_name] = psd

            # Save computed PSD data
            save_psd_data(
                psd_data,
                dataset_name,
                subject,
                freqs,
                epochs.ch_names,
                epochs.info['sfreq']
            )

def main():
    dataset_names = ["braboszcz2017"]  # or set dynamically
    for dataset_name in dataset_names:
        compute_psd_data(dataset_name)
    # compute_spectrograms()  # Implement later

if __name__ == "__main__":
    main()
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
from utils.helpers import remap_events_to_original, generate_metadata_df


# Specify the subfolder for the epochs path
EPOCHS_SUBFOLDER = "ica_cleaned"


def save_psd_data(psds, freqs, channels, metadata_df, output_dir, fname_prefix):
    """
    Save PSD data and metadata explicitly to the given directory.

    Parameters:
        psds (ndarray): PSD data (epochs × channels × frequencies).
        freqs (ndarray): Frequency values.
        channels (list[str]): Channel names.
        metadata_df (pd.DataFrame): Metadata DataFrame.
        output_dir (str): Path to the directory to save files.
        fname_prefix (str): Filename prefix for saved files (without extension).
    """

    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, f"{fname_prefix}_psd.npz"),
        psd=psds,
        freqs=freqs,
        channels=channels
    )

    metadata_df.to_csv(
        os.path.join(output_dir, f"{fname_prefix}_metadata.csv"),
        index=False
    )

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
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        window=window,
        average=None,
        output='power',
        n_jobs=n_jobs,
        remove_dc=remove_dc,
        verbose=False  
    )

    return psd, freqs

from tqdm import tqdm

def compute_psd_data(dataset_name):
    """
    Compute the PSD data for all subjects in a given dataset, showing a progress bar.

    Parameters:
        dataset_name (str): Identifier for dataset configuration.
    """
    dataset_config = DATASETS[dataset_name]

    # Create a clearly structured iteration list
    subject_session_pairs = [
        (subject, session)
        for subject in dataset_config.subjects
        for session in dataset_config.sessions
    ]

    for subject, session in tqdm(subject_session_pairs, desc="Computing PSD", unit="subject-session"):
        epochs_path = os.path.join(dataset_config.path_epochs, EPOCHS_SUBFOLDER)
        epochs = load_epochs(epochs_path, subject, session, verbose=False)

        if epochs is None:
            continue

        channels = epochs.ch_names

        metadata_df = generate_metadata_df(epochs, dataset_config, subject, session)

        psd, freqs = compute_psd(epochs)

        fname_prefix = f'sub-{subject}_ses-{session}'

        save_psd_data(psd, freqs, channels, metadata_df, dataset_config.path_psd, fname_prefix)


def main():
    dataset_names = ["jin2019", "braboszcz2017"]  # or set dynamically
    for dataset_name in dataset_names:
        compute_psd_data(dataset_name)
    # compute_spectrograms()  # Implement later

if __name__ == "__main__":
    main()
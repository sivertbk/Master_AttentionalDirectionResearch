"""
This file contains functions for reading and writing files.
"""
import mne
import os
import pickle

from utils.config import DATASETS
import utils.config as config

def load_epochs(epochs_dir, subject, session=1, preload=True, verbose=True):
    """
    Load MNE epochs from a file with structured filename.

    Parameters:
    -----------
    epochs_dir : str
        Directory where the epoch files are stored.
    subject : str or int
        Subject identifier.
    session : str or int, optional
        Session identifier. Defaults to 1 assumes only one session.
    preload : bool, optional
        If True, preload the data into memory. Defaults to True.
    verbose : bool, optional
        If True, print detailed information. Defaults to True.

    Returns:
    --------
    mne.Epochs or None
        The loaded epochs object, or None if the file does not exist.
    """
    # Construct filename
    filename = f"sub-{subject}_ses-{session}_epo.fif"
    file_path = os.path.join(epochs_dir, filename)
    if not os.path.exists(file_path):
        if verbose:
            print(f"File not found: {file_path}")
        return None

    # Load the epochs
    epochs = mne.read_epochs(file_path, preload=preload, verbose=verbose)

    if verbose:
        print(f"Loaded epochs from: {file_path}")
    return epochs

def save_epochs(epochs, output_dir, subject, session=1, subfolder=None):
    """
    Save MNE epochs with structured filenames.

    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs object to save.
    output_dir : str
        Directory to save the epoch files.
    subject : str or int
        Subject identifier.
    session : str or int
        Session identifier. Defaults to 1 assumes only one session.
    subfolder : str, optional
        Subfolder within the output directory to save the file.

    Returns:
    --------
    str
        Path to the saved file.
    """
    # Construct the full output directory path
    if subfolder:
        output_dir = os.path.join(output_dir, subfolder)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct filename
    filename = f"sub-{subject}_ses-{session}_epo.fif"
    file_path = os.path.join(output_dir, filename)

    # Save the epochs
    epochs.save(file_path, overwrite=True)

    print(f"Saved epochs to: {file_path}")
    return file_path

def save_psd_data(psd_data, dataset_name, subject, session, freqs, ch_names, sfreq):
    """
    Save PSD data to a file.

    Args:
        psd_data (dict): The PSD data to save.
        dataset_name (str): Name of the dataset to save.
        subject (int): Subject identifier.
        freqs (array): Frequency array.
        ch_names (list): Channel names.
        sfreq (float): Sampling frequency.
    """
    dataset_config = DATASETS[dataset_name]
    psd_file = os.path.join(dataset_config.path_psd, f"sub-{subject}_ses-{session}_psd.pkl")
    with open(psd_file, "wb") as f:
        pickle.dump({
            "psd_data": psd_data,
            "freqs": freqs,
            "ch_names": ch_names,
            "sfreq": sfreq
        }, f)
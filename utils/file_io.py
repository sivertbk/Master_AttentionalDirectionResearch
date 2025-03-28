"""
This file contains functions for reading and writing files.
"""
import mne
import os
import numpy as np
import json

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

def save_psd_data(psds, freqs, channels, metadata_epochs_df, metadata_config, output_root, subject, session, variant):
    """
    Save PSD data and split metadata to a structured directory.

    Parameters:
        psds (ndarray): PSD data (epochs × channels × frequencies).
        freqs (ndarray): Frequency values.
        channels (list[str]): Channel names.
        metadata_epochs_df (pd.DataFrame): Per-epoch metadata (states, task labels).
        metadata_config (dict): Global PSD configuration metadata.
        output_root (str): Root output directory (e.g., 'psd_data/').
        subject (str or int): Subject identifier.
        session (str or int): Session identifier.
        variant (str): Variant name (e.g., 'avg-mean').
    """
    sub = f"sub-{subject}"
    ses = f"ses-{session}"
    output_dir = os.path.join(output_root, sub, ses, variant)
    os.makedirs(output_dir, exist_ok=True)

    # Save PSD data
    np.savez_compressed(
        os.path.join(output_dir, "psd.npz"),
        psd=psds,
        freqs=freqs,
        channels=channels
    )

    # Save per-epoch metadata
    metadata_epochs_df.to_csv(
        os.path.join(output_dir, "metadata_epochs.csv"),
        index=False
    )

    # Save global config metadata
    with open(os.path.join(output_dir, "metadata_config.json"), "w") as f:
        json.dump(metadata_config, f, indent=2)
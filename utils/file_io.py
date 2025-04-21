"""
This file contains functions for reading and writing files.
"""
import mne
import os
from pathlib import Path
import numpy as np
import json


from utils.config import DATASETS
import utils.config as config
from utils.dataset_config import DatasetConfig

def _load_braboszcz(dataset: DatasetConfig, subject, task='med2', preload=True):
    subject_str = f"sub-{int(subject):03d}"
    fname = f"{subject_str}_task-{task}_eeg.{dataset.extension}"
    path = os.path.join(dataset.path_raw, subject_str, "eeg", fname)
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        return None

    return mne.io.read_raw_bdf(path, preload=preload)

def _load_jin(dataset: DatasetConfig, subject, session=1, preload=True):
    fname = f"sub{int(subject)}_{session}.{dataset.extension}"
    path = os.path.join(dataset.path_raw, fname)
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        return None

    return mne.io.read_raw_bdf(path, preload=preload)

def _load_touryan(dataset: DatasetConfig, subject, run=2, preload=True):
    subject_str = f"sub-{int(subject):02d}"
    fname = f"{subject_str}_ses-01_task-DriveWithTaskAudio_run-{run}_eeg.{dataset.extension}"
    path = os.path.join(dataset.path_raw, subject_str, "ses-01", "eeg", fname)
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        return None

    return mne.io.read_raw_eeglab(path, preload=preload)


def load_raw_data(dataset: DatasetConfig, subject, session=1, task=None, run=None, preload=True):
    """
    Load MNE raw data from a file with structured filename.

    Parameters:
    -----------
    dataset : Dataset
        The dataset object containing paths and metadata.
    subject : str or int
        Subject identifier.
    session : str or int, optional
        Session identifier. Defaults to 1 assumes only one session.
    preload : bool, optional
        If True, preload the data into memory. Defaults to True.

    Returns:
    --------
    mne.io.Raw or None
        The loaded raw object, or None if the file does not exist.
    """
    if dataset.f_name == "braboszcz2017":
        return _load_braboszcz(dataset, subject, task=task, preload=preload)
    elif dataset.f_name == "jin2019":
        return _load_jin(dataset, subject, session=session, preload=preload)
    elif dataset.f_name == "touryan2022":
        return _load_touryan(dataset, subject, run=run, preload=preload)
    else:
        raise ValueError(f"Unsupported dataset: {dataset.f_name}")

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

def update_bad_channels_json(
    save_dir, dataset, subject, session=None, task=None, run=None, bad_chs=None, mode='ransac'
):
    """
    Update or create a JSON file with bad channels detected by RANSAC.

    Parameters:
    - save_dir: str or Path. Directory where the JSON should be stored.
    - dataset: str. Dataset name (e.g., 'jin2019', 'braboszcz2017').
    - subject: str. Subject ID (e.g., '001').
    - session: str or int, optional. Only used for 'jin2019'.
    - task: str, optional. Only used for 'braboszcz2017'.
    - run: str or int, optional. Only used for 'touryan2022'.
    - bad_chs: list of str. Channel names flagged as bad by RANSAC.
    - mode: str. Mode of operation ('ransac' or 'inspect').
    """

    if mode not in ['ransac', 'inspect']:
        raise ValueError("Mode must be 'ransac' or 'inspect'.")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{mode}_bad_channels.json")

    # Load existing file if it exists
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            bad_chans_data = json.load(f)
    else:
        bad_chans_data = {}

    # Build hierarchical key path
    if dataset not in bad_chans_data:
        bad_chans_data[dataset] = {}

    subj_key = f"sub-{subject}"
    if subj_key not in bad_chans_data[dataset]:
        bad_chans_data[dataset][subj_key] = {}

    # Determine meta key (session/task/run)
    if dataset == "jin2019" and session is not None:
        meta_key = f"ses-{session}"
    elif dataset == "braboszcz2017" and task is not None:
        meta_key = f"task-{task}"
    elif dataset == "touryan2022" and run is not None:
        meta_key = f"run-{run}"
    else:
        meta_key = "unknown"

    bad_chans_data[dataset][subj_key][meta_key] = bad_chs or []

    # Write updated JSON
    with open(save_path, "w") as f:
        json.dump(bad_chans_data, f, indent=2)

    print(f"[BAD CHS UPDATE] Updated bad channels at: {save_path}")

def load_bad_channels(save_dir, dataset, subject, session=None, task=None, run=None):
    """
    Check if bad channels from RANSAC are already saved for a subject/session/task/run.

    Parameters:
    - save_dir: str or Path. Directory where 'ransac_bad_channels.json' is stored.
    - dataset: str. Dataset name ('jin2019', 'braboszcz2017', 'touryan2022').
    - subject: str. Subject ID.
    - session: str or int, optional. Used for 'jin2019'.
    - task: str, optional. Used for 'braboszcz2017'.
    - run: str or int, optional. Used for 'touryan2022'.

    Returns:
    - bad_chs: list of bad channel names, or None if not found.
    """
    save_path = os.path.join(save_dir, "ransac_bad_channels.json")
    if not os.path.exists(save_path):
        print("No RANSAC bad channel file found.")
        return None

    with open(save_path, "r") as f:
        bad_chans_data = json.load(f)

    subj_key = f"sub-{subject}"
    if dataset == "jin2019" and session is not None:
        meta_key = f"ses-{session}"
    elif dataset == "braboszcz2017" and task is not None:
        meta_key = f"task-{task}"
    elif dataset == "touryan2022" and run is not None:
        meta_key = f"run-{run}"
    else:
        meta_key = "unknown"

    try:
        bad_chs = bad_chans_data[dataset][subj_key][meta_key]
        print(f"Found existing bad channels for {subj_key} - {meta_key}")
        return bad_chs
    except KeyError:
        print(f"No bad channels found for {subj_key} - {meta_key}")
        return None
    
def save_autoreject(ar, save_path):
    """
    Save an AutoReject object only if it has been successfully fitted.

    Parameters:
    - ar : autoreject.AutoReject instance
    - save_path : str or Path to .h5 or .hdf5 file
    """
    try:
        # Check for fitted state: threshes_ must exist and not be empty
        if getattr(ar, "threshes_", None) and len(ar.threshes_) > 0:
            ar.save(save_path, overwrite=True)
            print(f"[AutoReject] Model saved at: {save_path}")
        else:
            print(f"[AutoReject] Not saving: No thresholds learned (empty model).")
    except Exception as e:
        print(f"[AutoReject] Save failed with exception:\n{e}")

def load_reject_log(path, subject, session=None, task=None, run=None):
    """
    Load a saved RejectLog if it exists at the given path using read_reject_log().

    Parameters:
    - path: str, directory where the file is saved
    - subject: str or int, subject ID (e.g., '001')
    - session: str or int, used for datasets like jin2019
    - task: str, used for datasets like braboszcz2017
    - run: str or int, used for datasets like touryan2022

    Returns:
    - reject_log: RejectLog instance if found, otherwise None
    """
    from autoreject import read_reject_log
    if session is not None:
        fname = os.path.join(path, f"sub-{subject}_ses-{session}_autoreject_log.npz")
    elif task is not None:
        fname = os.path.join(path, f"sub-{subject}_task-{task}_autoreject_log.npz")
    elif run is not None:
        fname = os.path.join(path, f"sub-{subject}_run-{run}_autoreject_log.npz")
    else:
        raise ValueError("Either session, task, or run must be provided.")

    if os.path.exists(fname):
        reject_log = read_reject_log(fname)
        print(f"[AutoReject] Loaded reject log from: {fname}")
        return reject_log
    else:
        print(f"[AutoReject] No reject log found at: {fname}")
        return None

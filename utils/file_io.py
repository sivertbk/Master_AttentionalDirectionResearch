"""
This file contains functions for reading and writing files.
"""
import mne
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from mne.preprocessing import ICA, read_ica
from autoreject import read_auto_reject
from datetime import datetime
from typing import Union


from utils.config import DATASETS
import utils.config as config
from utils.dataset_config import DatasetConfig

def _load_braboszcz(dataset: DatasetConfig, subject, task='med2', preload=True, verbose=True):
    subject_str = f"sub-{int(subject):03d}"
    fname = f"{subject_str}_task-{task}_eeg.{dataset.extension}"
    path = os.path.join(dataset.path_raw, subject_str, "eeg", fname)
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        return None

    return mne.io.read_raw_bdf(path, preload=preload, verbose=verbose)

def _load_jin(dataset: DatasetConfig, subject, session=1, preload=True, verbose=True):
    fname = f"sub{int(subject)}_{session}.{dataset.extension}"
    path = os.path.join(dataset.path_raw, fname)
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        return None

    return mne.io.read_raw_bdf(path, preload=preload, verbose=verbose)

def _load_touryan(dataset: DatasetConfig, subject, run=2, preload=True, verbose=True):
    subject_str = f"sub-{int(subject):02d}"
    fname = f"{subject_str}_ses-01_task-DriveWithTaskAudio_run-{run}_eeg.{dataset.extension}"
    path = os.path.join(dataset.path_raw, subject_str, "ses-01", "eeg", fname)
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        return None

    return mne.io.read_raw_eeglab(path, preload=preload, verbose=verbose)


def load_raw_data(dataset: DatasetConfig, subject, session=1, task=None, run=None, preload=True, verbose=True):
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
    verbose : bool, optional
        If True, print detailed information. Defaults to True.

    Returns:
    --------
    mne.io.Raw or None
        The loaded raw object, or None if the file does not exist.
    """
    if dataset.f_name == "braboszcz2017":
        return _load_braboszcz(dataset, subject, task=task, preload=preload, verbose=verbose)
    elif dataset.f_name == "jin2019":
        return _load_jin(dataset, subject, session=session, preload=preload, verbose=verbose)
    elif dataset.f_name == "touryan2022":
        return _load_touryan(dataset, subject, run=run, preload=preload, verbose=verbose)
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

def save_psd_data(psd, freqs, channels, metadata, output_root, subject, session, task, state, variant):
    """
    Save PSD data and metadata to a structured directory.

    Parameters:
        psd (ndarray): PSD data (epochs × channels × frequencies).
        freqs (ndarray): Frequency values.
        channels (list[str]): Channel names.
        metadata (dict): Metadata (subject, states, task labels, etc.).
        output_root (str): Root output directory (e.g., 'psd_data/').
        subject (str or int): Subject identifier.
        session (int): Session identifier.
        task (str): Task identifier.
        state (str): State identifier.
        variant (str): Variant name (e.g., 'avg-mean').
    """
    sub = f"subject-{subject}"
    ses = f"session-{session}"
    task = f"task-{task}"
    state = f"state-{state}"
    output_dir = os.path.join(output_root, sub, ses, task, state, variant)
    os.makedirs(output_dir, exist_ok=True)

    # Save PSD data
    np.savez_compressed(
        os.path.join(output_dir, "psd.npz"),
        psd=psd,
        freqs=freqs,
        channels=channels
    )

    # Save metadata dict
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    tqdm.write(f"Saved PSD data and metadata to: {output_dir}")


def save_spectrogram_data(psd, freqs, channels, metadata, output_root, subject, session, task, state):
    """
    Save PSD data and metadata to a structured directory.

    Parameters:
        psd (ndarray): PSD data (epochs × channels × frequencies).
        freqs (ndarray): Frequency values.
        channels (list[str]): Channel names.
        metadata (dict): Metadata (subject, states, task labels, etc.).
        output_root (str): Root output directory (e.g., 'psd_data/').
        subject (str or int): Subject identifier.
        session (int): Session identifier.
        task (str): Task identifier.
        state (str): State identifier.
    """
    sub = f"subject-{subject}"
    ses = f"session-{session}"
    task = f"task-{task}"
    state = f"state-{state}"
    output_dir = os.path.join(output_root, sub, ses, task, state, )
    os.makedirs(output_dir, exist_ok=True)

    # Save PSD data
    np.savez_compressed(
        os.path.join(output_dir, "psd.npz"),
        psd=psd,
        freqs=freqs,
        channels=channels
    )

    # Save metadata dict
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    tqdm.write(f"Saved PSD data and metadata to: {output_dir}")

def update_bad_channels_json(
    save_dir, dataset, subject, session=None, task=None, run=None, bad_chs=None, mode='ransac'
):
    """
    Update or create a JSON file with bad channels detected by RANSAC or INSPECT.

    Parameters:
    - save_dir: str or Path. Directory where the JSON should be stored.
    - dataset: str. Dataset name (e.g., 'jin2019', 'braboszcz2017').
    - subject: str. Subject ID (e.g., '001').
    - session: str or int, optional. Only used for 'jin2019'.
    - task: str, optional. Only used for 'braboszcz2017'.
    - run: str or int, optional. Only used for 'touryan2022'.
    - bad_chs: list of str. Channel names flagged as bad by RANSAC or INSPECT.
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

    # If mode is 'inspect', load the ransac JSON file as a base
    if mode == 'inspect':
        ransac_path = os.path.join(save_dir, "ransac_bad_channels.json")
        if os.path.exists(ransac_path):
            with open(ransac_path, "r") as f:
                ransac_data = json.load(f)
            # Merge ransac data into inspect data
            bad_chans_data = {**ransac_data, **bad_chans_data}

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

def load_bad_channels(save_dir, dataset, subject, session=None, task=None, run=None, mode='ransac'):
    """
    Load the json of bad channels for interpolation for a subject/session/task/run.

    Parameters:
    - save_dir: str or Path. Directory where 'inspect_bad_channels.json' is stored.
    - dataset: str. Dataset name ('jin2019', 'braboszcz2017', 'touryan2022').
    - subject: str. Subject ID.
    - session: str or int, optional. Used for 'jin2019'.
    - task: str, optional. Used for 'braboszcz2017'.
    - run: str or int, optional. Used for 'touryan2022'.

    Returns:
    - bad_chs: list of bad channel names, or None if not found.
    """
    if mode not in ['ransac', 'inspect']:
        raise ValueError("Mode must be 'ransac' or 'inspect'.")
    
    save_path = os.path.join(save_dir, f"{mode}_bad_channels.json")
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

def load_ar(dataset, subject, label, item, verbose=True):
    """
    Load a saved AutoReject model for a specific subject and item.

    Parameters
    ----------
    dataset : DatasetConfig
        The dataset configuration object.
    subject : str or int
        Subject identifier.
    label : str
        The iteration label ('session', 'task', or 'run').
    item : str or int
        The iteration value corresponding to the label.
    verbose : bool, optional
        If True, prints status messages.

    Returns
    -------
    ar : autoreject.AutoReject or None
        The loaded AutoReject model, or None if not found.
    """
    fname = f"sub-{subject}_{label}-{item}_autoreject_pre_ica.h5"
    ar_model_path = os.path.join(
        dataset.path_derivatives,
        "pre_ica_autoreject_models",
        fname
    )

    if not os.path.exists(ar_model_path):
        if verbose:
            print(f"[AutoReject] Model not found: {ar_model_path}")
        return None

    if verbose:
        print(f"[AutoReject] Loading model: {ar_model_path}")
    return read_auto_reject(ar_model_path)

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

def save_ica(ica, dataset, subject, session=None, task=None, run=None, verbose=True):
    if dataset.f_name == "braboszcz2017":
        fname = f"sub-{subject}_task-{task}_ica.fif"
    elif dataset.f_name == "jin2019":
        fname = f"sub{subject}_{session}_ica.fif"
    elif dataset.f_name == "touryan2022":
        fname = f"sub-{subject}_ses-01_task-DriveWithTaskAudio_run-{run}_ica.fif"
    else:
        raise ValueError(f"Unsupported dataset: {dataset.f_name}")

    save_dir = os.path.join(dataset.path_derivatives, "ica")
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, fname)
    ica.save(path, overwrite=True, verbose=verbose)

    if verbose:
        print(f"[ICA] Saved ICA to: {path}")
    return path

def load_ica(dataset, subject, session=None, task=None, run=None, verbose=True):
    if dataset.f_name == "braboszcz2017":
        fname = f"sub-{subject}_task-{task}_ica.fif"
    elif dataset.f_name == "jin2019":
        fname = f"sub{subject}_{session}_ica.fif"
    elif dataset.f_name == "touryan2022":
        fname = f"sub-{subject}_ses-01_task-DriveWithTaskAudio_run-{run}_ica.fif"
    else:
        raise ValueError(f"Unsupported dataset: {dataset.f_name}")

    ica_dir = os.path.join(dataset.path_derivatives, "ica")
    path = os.path.join(ica_dir, fname)

    if not os.path.exists(path):
        print(f"[ICA] File not found: {path}")
        return None

    ica = read_ica(path, verbose=verbose)

    if verbose:
        print(f"[ICA] Loaded ICA from: {path}")
    return ica

def log_dropped_epochs(epochs, dataset, subject, log_root=config.PREPROCESSING_LOG_PATH, stage='pre_ica',
                       session=None, task=None, run=None, threshold=None, verbose=True):
    """
    Log the number of dropped epochs for a given subject/task/run to a file.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object after `drop_bad()` has been applied.
    dataset : DatasetConfig
        The dataset config object with .f_name.
    subject : str or int
        Subject identifier.
    log_root : str
        Root directory where logs are saved.
    stage : str
        Identifier for the processing stage (e.g., 'pre_ica').
    session : str or int, optional
        Session identifier.
    task : str, optional
        Task identifier.
    run : str or int, optional
        Run identifier.
    threshold : float, optional
        Threshold used for rejection (in Volts).
    verbose : bool
        Whether to print the log line to stdout.
    """
    n_total = len(epochs.drop_log)
    n_dropped = sum(len(e) > 0 for e in epochs.drop_log)
    n_kept = n_total - n_dropped

    # Construct log directory and filename
    log_dir = os.path.join(log_root, f"{stage}_dropped_epochs")
    os.makedirs(log_dir, exist_ok=True)

    filename = f"{dataset.f_name}_dropped_epochs.log"

    log_path = os.path.join(log_dir, filename)
    threshold_str = f"{threshold * 1e6:.1f} µV" if threshold is not None else "N/A"

    log_line = (
        f"{datetime.now().isoformat()} | "
        f"Subject: {subject}, Session: {session}, Task: {task}, Run: {run} | "
        f"Dropped: {n_dropped}/{n_total} epochs (Kept: {n_kept}) | "
        f"Threshold: {threshold_str}\n"
    )

    with open(log_path, "a") as f:
        f.write(log_line)

    if verbose:
        print(f"[EPOCH DROP LOG] {log_line.strip()}")

def log_reject_threshold(reject, dataset, subject, save_dir=config.PREPROCESSING_LOG_PATH, stage='pre_ica',
                         session=None, task=None, run=None, verbose=True):
    """
    Log a single subject's reject threshold into a shared JSON file.

    Parameters
    ----------
    reject : dict
        Dictionary with thresholds per channel type (e.g., {'eeg': 0.00034}).
    dataset : DatasetConfig
        Dataset configuration object with `.f_name`.
    subject : str or int
        Subject ID.
    save_dir : str
        Root directory where threshold logs should be stored.
    stage : str
        Processing stage, used in filename.
    session : str or int, optional
        Session ID (only used if present).
    task : str, optional
        Task label (only used if present).
    run : str or int, optional
        Run number (only used if present).
    verbose : bool
        If True, print status messages.
    """
    thresholds_dir = os.path.join(save_dir, "reject_thresholds", dataset.f_name)
    os.makedirs(thresholds_dir, exist_ok=True)

    filename = f"{stage}_reject_thresholds.json"
    path = os.path.join(thresholds_dir, filename)

    key = f"sub-{subject}"
    if session is not None:
        key += f"_ses-{session}"
    if task is not None:
        key += f"_task-{task}"
    if run is not None:
        key += f"_run-{run}"

    # Load existing JSON (if exists)
    if os.path.exists(path):
        with open(path, "r") as f:
            all_thresholds = json.load(f)
    else:
        all_thresholds = {}

    # Update with new entry (ensure JSON-safe float)
    all_thresholds[key] = {k: float(v) for k, v in reject.items()}

    # Save back to file
    with open(path, "w") as f:
        json.dump(all_thresholds, f, indent=2)

    if verbose:
        print(f"[REJECTION] Logged threshold for {key} to {path}")

def load_reject_threshold(dataset, subject, label, item, save_dir, stage='pre_ica'):
    """
    Load the reject threshold for a specific subject+item from the dataset JSON file.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset object with .f_name
    subject : str or int
        Subject ID
    label : str
        Type of iteration ('task', 'run', etc.)
    item : str or int
        Item value (e.g., 2)
    save_dir : str
        Base path where the thresholds file is saved
    stage : str
        Stage name (used in filename)

    Returns
    -------
    dict or None
        The reject dict (e.g., {"eeg": 0.00034}) or None if not found
    """
    fname = f"{stage}_reject_thresholds.json"
    path = os.path.join(save_dir, "reject_thresholds", dataset.f_name, fname)

    if not os.path.exists(path):
        print(f"[WARN] No threshold file found: {path}")
        return None

    with open(path, "r") as f:
        thresholds_all = json.load(f)

    key = f"sub-{subject}_{label}-{item}"
    return thresholds_all.get(key, None)

def save_ica_excluded_components(dataset, subject, label, item, blink_components, saccade_components):
    """Save or update ICA excluded components information into a JSON file."""
    # Ensure components are native Python ints
    blink_components = [int(c) for c in blink_components]
    saccade_components = [int(c) for c in saccade_components]
    all_excluded = sorted(list(set(blink_components + saccade_components)))

    # Define the subject and item labels
    subject_key = f"subject-{subject}"
    item_key = f"{label}-{item}"

    # Full file path
    save_path = dataset.path_derivatives
    filename = f"{dataset.f_name}_ica_excluded_components.json"
    filepath = os.path.join(save_path, filename)

    # Load existing data if file exists
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Navigate into the label dictionary, creating it if it doesn't exist
    if dataset.f_name not in data:
        data[dataset.f_name] = {}

    if subject_key not in data[dataset.f_name]:
        data[dataset.f_name][subject_key] = {}

    # Update the specific entry
    data[dataset.f_name][subject_key][item_key] = {
        "blink_components": blink_components,
        "saccade_components": saccade_components,
        "all_excluded": all_excluded
    }

    # Save back to JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_ica_excluded_components(dataset, subject, label, item) -> Union[list, None]:
    """
    Load excluded ICA components (blink + saccade).

    Returns
    -------
    excluded_components : list of int
        List of all excluded components (blink + saccade merged).
    """
    save_path = os.path.join(dataset.path_derivatives, f"{dataset.f_name}_ica_excluded_components.json")
    if not os.path.exists(save_path):
        print(f"No exclusion file found at {save_path}. Returning None.")
        return None

    with open(save_path, "r") as f:
        data = json.load(f)

    try:
        exclusion_info = data[dataset.f_name][f'subject-{subject}'][f"{label}-{item}"]
        excluded_components = exclusion_info.get("all_excluded", [])
    except KeyError:
        print(f"No exclusions found for {dataset.f_name} {subject} {label}-{item}.")
        return None

    print(f"Loaded {len(excluded_components)} excluded components for {dataset.name} {subject} {label}-{item}.")
    return excluded_components


def load_new_events(path, subject, session):
    file_name = f'subject-{subject}_session-{session}_events.csv'
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        column_names = df.columns.values
        return df.values, column_names
    else:
        raise FileNotFoundError(f"No such file: {file_path}")
    

def save_epochs_dict(epochs_dict, reject, dataset, subject, session):
    """
    Save each Epochs object in the dictionary using key information file names.

    NOTE: from now on, all datasets will be treated with session instead of label and item

    Parameters
    ----------
    epochs_dict : dict
        Dictionary of class_label -> Epochs.
    reject : dict
        Rejection threshold for each channel type.
    dataset : DatasetConfig
        Dataset metadata config.
    subject : str or int
        Subject ID.
    session : str
        session ID.

    Returns
    -------
    saved_paths : list of Path
        List of saved file paths.
    """
    # Checking the passed session
    if session not in dataset.sessions:
        # As of now this applies only to the braboszcz dataset and we map to session-1
        session = 1
    saved_paths = []
    subject_str = f"subject-{subject}"
    session_str = f"session-{session}"

    for class_label, epochs in epochs_dict.items():
        task, state = class_label.split('/')
        condition = None

        # Build filename
        fname_parts = [f"task-{task}"]
        if condition:
            fname_parts.append(f"condition-{condition}")
        fname_parts.append(f"state-{state}")
        fname = "_".join(fname_parts) + "_epo.fif"

        # Build full path
        base_path = Path(dataset.path_epochs) / "analysis_epochs" / subject_str / session_str
        base_path.mkdir(parents=True, exist_ok=True)
        out_path = base_path / fname

        # deop bad epochs
        epochs.drop_bad(reject, verbose=True)

        # Save
        epochs.save(out_path, overwrite=True)
        saved_paths.append(out_path)

    return saved_paths
import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import ICA
import gc

import utils.config as config
from utils.config import DATASETS

def print_directory_tree(
    root_dir=None,
    indent="",
    file_limit=10,
    dir_limit=10,
    exclude_dirs=None,
    max_depth=None,
    current_depth=0,
    skip_folders=None,
    output_lines=None,
    save_to_file=False,
    filename="directory_structure.txt"
):
    """
    Prints or saves a directory tree from the root_dir with various options.

    Args:
        root_dir (str): Starting directory (defaults to current working directory).
        indent (str): Indentation string for pretty print.
        file_limit (int): Max number of files per folder to show.
        dir_limit (int): Max number of folders per folder to show.
        exclude_dirs (list): List of dir names to exclude completely.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current level (used internally for recursion).
        skip_folders (list): List of folder names to skip diving into.
        output_lines (list): Internal list of lines being built.
        save_to_file (bool): If True, writes to file instead of printing.
        filename (str): Filename to save the output (if save_to_file=True).
    """
    if root_dir is None:
        root_dir = os.getcwd()

    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__"]

    if skip_folders is None:
        skip_folders = ["data"]

    if output_lines is None:
        output_lines = []

    if max_depth is not None and current_depth > max_depth:
        output_lines.append(f"{indent}... (max depth reached)")
        return output_lines

    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        output_lines.append(f"{indent}🔒 [Access Denied]")
        return output_lines

    dirs = [e for e in entries if os.path.isdir(os.path.join(root_dir, e)) and e not in exclude_dirs]
    files = [e for e in entries if os.path.isfile(os.path.join(root_dir, e))]

    display_dirs = dirs[:dir_limit]
    remaining_dirs = len(dirs) - len(display_dirs)

    display_files = files[:file_limit]
    remaining_files = len(files) - len(display_files)

    for d in display_dirs:
        output_lines.append(f"{indent}📂 {d}/")
        full_path = os.path.join(root_dir, d)
        if d in skip_folders:
            output_lines.append(f"{indent}│   ... (skipped)")
        else:
            print_directory_tree(
                full_path,
                indent + "│   ",
                file_limit,
                dir_limit,
                exclude_dirs,
                max_depth,
                current_depth + 1,
                skip_folders,
                output_lines,
                save_to_file,
                filename
            )

    if remaining_dirs > 0:
        output_lines.append(f"{indent}... ({remaining_dirs} more directories)")

    for f in display_files:
        output_lines.append(f"{indent}📄 {f}")

    if remaining_files > 0:
        output_lines.append(f"{indent}... ({remaining_files} more files)")

    # Final output
    if current_depth == 0:
        if save_to_file:
            with open(filename, "w") as f:
                f.write("\n".join(output_lines))
            print(f"✅ Directory structure saved to: {filename}")
        else:
            for line in output_lines:
                print(line)

    return output_lines




# =============================================================================
#                 Jin et al. (2019) EEG Dataset Helper Functions
# =============================================================================

def load_new_events(path, subject, session):
    file_name = f'subject-{subject}_session-{session}_events.csv'
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        column_names = df.columns.values
        return df.values, column_names
    else:
        raise FileNotFoundError(f"No such file: {file_path}")
    
def load_epochs(subject, session, bids_root="./epochs/external_task"):
    """
    Load MNE epochs from a file with structured filename.

    Parameters:
    -----------
    subject : str or int
        Subject identifier.
    session : str or int
        Session identifier.
    bids_root : str, optional
        Directory where the epoch files are stored. Defaults to "./epochs".

    Returns:
    --------
    mne.Epochs
        The loaded epochs object.
    """
    # Construct filename
    filename = f"sub-{subject}_ses-{session}_epo.fif"
    file_path = os.path.join(bids_root, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Load the epochs
    epochs = mne.read_epochs(file_path, preload=True)

    print(f"Loaded epochs from: {file_path}")
    return epochs

def encode_events(events):
    """
    Encode a 2D array of events with 3 integers into a 1D array of encoded integers.
    
    Parameters:
    events (np.ndarray): 2D array of shape (n_events, 3) where each row contains 3 integers.
    
    Returns:
    np.ndarray: 1D array of shape (n_events, 1) with encoded integers.
    """
    return events[:, 0] * 100 + events[:, 1] + events[:, 2]* 10 

def decode_events(encoded_events):
    """
    Decode a 1D array of encoded integers back into a 2D array of events with 3 integers.
    
    Parameters:
    encoded_events (np.ndarray): 1D array of shape (n_events, 1) with encoded integers.
    
    Returns:
    np.ndarray: 2D array of shape (n_events, 3) where each row contains 3 integers.
    """
    events = np.zeros((encoded_events.shape[0], 3), dtype=int)
    events[:, 0] = encoded_events // 100
    events[:, 2] = (encoded_events % 100) // 10
    events[:, 1] = encoded_events % 10
    return events

def decode_event(event_id, event_info):
    """
    Decode an encoded event ID back into the original event.
    
    Parameters:
    event_id (int): Encoded event ID.
    event_info (list): List of 3 strings representing the event information.
    
    Returns:
    dict: Event as a dict of 3 integers.
    """
    return {event_info[0]: event_id // 100, event_info[1]: event_id % 10, event_info[2]: (event_id % 100) // 10}

def save_epochs(epochs, subject, session, output_dir=config.EPOCHS_PATH):
    """
    Save MNE epochs with structured filenames.

    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs object to save.
    subject : str or int
        Subject identifier.
    session : str or int
        Session identifier.
    output_dir : str, optional
        Directory to save the epoch files. Defaults to "./epochs".

    Returns:
    --------
    str
        Path to the saved file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct filename
    filename = f"sub-{subject}_ses-{session}_epo.fif"
    file_path = os.path.join(output_dir, filename)

    # Save the epochs
    epochs.save(file_path, overwrite=True)

    print(f"Saved epochs to: {file_path}")
    return file_path


def log_warning(subject, session, msg, path="./logs/warnings.log"):
    """
    Log warnings to a specified file.

    Parameters:
    -----------
    subject : int
        Subject identifier.
    session : int
        Session identifier.
    msg : str
        Warning message to log.
    path : str, optional
        Path to the log file. Defaults to "./logs/warnings.log".
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Construct the log message
    log_msg = f"Subject: {subject}, Session: {session}, Message: {msg}\n"

    # Write the log message to the file
    with open(path, "a") as log_file:
        log_file.write(log_msg)

    print(f"Logged warning for Subject {subject}, Session {session}: {msg}")


def log_msg(subject, session, msg, path="./logs/info.log"):
    """
    Log messages to a specified file.

    Parameters:
    -----------
    subject : int
        Subject identifier.
    session : int
        Session identifier.
    msg : str
        Message to log.
    path : str, optional
        Path to the log file. Defaults to "./logs/info.log".
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Construct the log message
    log_msg = f"Subject: {subject}, Session: {session}, Message: {msg}\n"

    # Write the log message to the file
    with open(path, "a") as log_file:
        log_file.write(log_msg)

    print(f"Logged message for Subject {subject}, Session {session}: {msg}")



def mark_bad_channels(subject_list, bids_root, dataset_name="Jin2019", bad_channels_dict=None):
    """
    Interactively mark bad channels for each subject and session using dataset configuration.

    Parameters:
        subject_list (list): List of subject IDs.
        bids_root (str): Directory where raw files are stored.
        dataset_name (str): Name of the dataset in config (default: "Jin2019").
        bad_channels_dict (dict): Existing dictionary to update (default: None).

    Returns:
        dict: Dictionary with bad channels for each subject and session.
    """
    if bad_channels_dict is None:
        bad_channels_dict = {}

    dataset_config = DATASETS[dataset_name]
    subject_session_df = dataset_config.extra_info['subject_session_df']

    for subject in subject_list:
        print(f"\nProcessing subject: {subject}")

        if subject not in bad_channels_dict:
            bad_channels_dict[subject] = {}

        for session in dataset_config.sessions:
            try:
                raw_file = os.path.join(bids_root, subject_session_df.loc[subject, session] + ".bdf")
                raw = mne.io.read_raw_bdf(raw_file, preload=True)
                print(f"Loaded: {raw_file}")

                old_ch_names = raw.ch_names

                # Temporary renaming for alignment
                temp_ch_names = ['temp_' + ch for ch in old_ch_names[:-9]] + old_ch_names[-9:]
                mapping_old_to_temp = dict(zip(old_ch_names, temp_ch_names))

                # Renaming channels using dataset config mappings
                raw.rename_channels(mapping_old_to_temp)
                raw.rename_channels(dataset_config.mapping_channels)
                raw.rename_channels(dataset_config.mapping_non_eeg)

                # Set channel types for EXG channels
                raw.set_channel_types({
                    'sacc_EOG1': 'eog',
                    'sacc_EOG2': 'eog',
                    'blink_EOG1': 'eog',
                    'blink_EOG2': 'eog',
                    'EXG5': 'misc',
                    'EXG6': 'misc',
                    'EXG7': 'misc',
                    'EXG8': 'misc'
                })

                # Identify non-EEG channels
                non_eeg_channels = [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type != 'eeg']

                # Channels to retain
                channels_to_keep = list(dataset_config.mapping_channels.keys()) + non_eeg_channels
                raw.pick(channels_to_keep)

                # Interactive marking
                print("Mark bad channels interactively and close the plot window to continue...")
                raw.plot(highpass=1, lowpass=40, block=True)

                # Save bad channels
                bad_channels_dict[subject][session] = raw.info['bads']
                print(f"Bad channels for subject {subject}, session {session}: {raw.info['bads']}")

                del raw
                gc.collect()

            except FileNotFoundError:
                print(f"File not found: {raw_file}. Skipping this session.")
                continue

    return bad_channels_dict


def perform_ica_cleaning(epochs, subject, n_components=20, method='infomax', random_state=97):
    """
    Perform ICA on the provided epochs and allow manual selection of components for exclusion.

    Parameters:
        epochs (mne.Epochs): The epochs object to clean.
        n_components (int): Number of ICA components to compute (default: 20).
        method (str): ICA method to use ('infomax' by default).
        random_state (int): Random seed for ICA initialization (default: 42).

    Returns:
        mne.Epochs: Cleaned epochs after ICA.
        ICA: The ICA object used for the analysis.
    """
    print("Fitting ICA...")
    # Initialize ICA
    ica = ICA(n_components=n_components, method=method, random_state=random_state)

    # Fit ICA to the epochs
    ica.fit(epochs)
    print("ICA fitting complete.")

    # Plot ICA components
    print("Displaying ICA components. Use the interactive GUI to select components for exclusion.")
    ica.plot_components(title=f"ICA decomposition on subject {subject}") 

    # Plot sources to identify artifacts
    print("Displaying ICA sources. Use this to examine the time-series for artifact-related components.")
    ica.plot_sources(epochs, block = True)  # Opens an interactive source plot

    # Apply ICA to remove the selected components
    print("Applying ICA to remove selected components...")
    cleaned_epochs = ica.apply(epochs.copy())
    print("ICA cleaning complete.")

    return cleaned_epochs, ica

if __name__ == "__main__":
    print("Helper functions loaded successfully.")
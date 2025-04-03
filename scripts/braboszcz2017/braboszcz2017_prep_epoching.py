import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from autoreject import AutoReject, Ransac, get_rejection_threshold

import utils.config as config
from utils.config import DATASETS
from utils.file_io import save_epochs
from utils.helpers import format_numbers

### Defining constants and preparing stuff ###

DATASET = DATASETS["braboszcz2017"]

bids_root = DATASET.path_raw
path_epochs = os.path.join(DATASET.path_epochs, "preprocessed")
os.makedirs(path_epochs, exist_ok=True)

# EEG settings
subjects = DATASET.subjects
sessions = DATASET.sessions
tasks = ["med2", "think2"]
event_id = DATASET.event_id_map
subject_groups = DATASET.extra_info["subject_groups"]
epoch_length = config.EEG_SETTINGS["EPOCH_LENGTH_SEC"] 
tmin = config.EEG_SETTINGS["EPOCH_START_SEC"]
tmax = tmin + epoch_length
sfreq = config.EEG_SETTINGS["SAMPLING_RATE"]
h_cut = config.EEG_SETTINGS["HIGH_CUTOFF_HZ"]
l_cut = config.EEG_SETTINGS["LOW_CUTOFF_HZ"]
reject_threshold = config.EEG_SETTINGS["REJECT_THRESHOLD"]
montage = mne.channels.make_standard_montage('biosemi64')


def load_task_data(bids_root, subject_id, tasks):
    """
    Load specific task EEG data for a subject from a BIDS-like dataset.
    
    Parameters:
    - bids_root: str, path to the root of the dataset.
    - subject_id: str, subject identifier (e.g., '088').
    - tasks: list of str, the task names to load (e.g., ['med2', 'think2']).
    
    Returns:
    - raw_dict: dict, containing MNE Raw objects for each task.
    """
    # Define the subject's EEG folder
    subject_path = os.path.join(bids_root, f"sub-{subject_id}", "eeg")
    
    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Subject folder not found: {subject_path}")
    
    raw_dict = {}  # Dictionary to hold Raw objects for each task
    
    for task in tasks:
        # Locate the BDF file for the specific task
        bdf_file = os.path.join(subject_path, f"sub-{subject_id}_task-{task}_eeg.bdf")
        if not os.path.exists(bdf_file):
            print(f"WARNING: BDF file not found for task '{task}'. Skipping...")
            continue
        
        # Load the BDF file
        raw = mne.io.read_raw_bdf(bdf_file, preload=True)
        print(f"Loaded BDF file for task '{task}': {bdf_file}")
        
        # Optionally, load metadata from JSON/TSV files
        # Metadata file paths
        json_file = os.path.join(subject_path, f"sub-{subject_id}_task-{task}_eeg.json")
        tsv_file = os.path.join(subject_path, f"sub-{subject_id}_task-{task}_channels.tsv")
        
        metadata = {}
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                metadata['json'] = f.read()
            print(f"Loaded metadata JSON for task '{task}'")
        
        if os.path.exists(tsv_file):
            metadata['tsv'] = tsv_file  # Store TSV file path for later use if needed
            print(f"Loaded metadata TSV for task '{task}'")
        
        # Add the loaded data and metadata to the dictionary
        raw_dict[task] = {
            'raw': raw,
            'metadata': metadata
        }
    
    if not raw_dict:
        raise ValueError(f"No data was loaded for subject {subject_id} with tasks {tasks}.")
    
    return raw_dict



for subject_id in subjects:
    print(f"\n##### Preprocessing subject {subject_id} #####")

    subject_data = load_task_data(bids_root, f"sub-{subject_id}", tasks)
    subject_group = subject_groups[subject_id]

    # Access raw data for a specific task
    raw_med = subject_data['med2']['raw']
    raw_mw = subject_data['think2']['raw']

    print('\n##### Channel types #####')
    # Set channel types
    raw_med.set_channel_types({'EXG1': 'misc',
                        'EXG2': 'misc',
                        'EXG3': 'misc',
                        'EXG4': 'misc',
                        'EXG5': 'misc',
                        'EXG6': 'misc',
                        'EXG7': 'misc',
                        'EXG8': 'misc',
                        'GSR1': 'misc',
                        'GSR2': 'misc',
                        'Erg1': 'misc',
                        'Erg2': 'misc',
                        'Resp': 'bio',
                        'Plet': 'bio',
                        'Temp': 'bio'
    })
    raw_mw.set_channel_types({'EXG1': 'misc',
                        'EXG2': 'misc',
                        'EXG3': 'misc',
                        'EXG4': 'misc',
                        'EXG5': 'misc',
                        'EXG6': 'misc',
                        'EXG7': 'misc',
                        'EXG8': 'misc',
                        'GSR1': 'misc',
                        'GSR2': 'misc',
                        'Erg1': 'misc',
                        'Erg2': 'misc',
                        'Resp': 'bio',
                        'Plet': 'bio',
                        'Temp': 'bio'
    })

    # Drop non-eeg channels
    raw_med.pick('eeg')
    raw_mw.pick('eeg')

    # Verify the remaining channels
    print(raw_med.info['ch_names'])

    #Load the BioSemi 64-channel montage
    raw_med.set_montage(montage)
    raw_mw.set_montage(montage)

    # Exclude bad channels
    if subject_id == "077":
        bad_channels = ['PO4', 'O2', 'P2']
        raw_med.info['bads'] = bad_channels
        raw_mw.info['bads'] = bad_channels

    print('Bad channels in meditation block:', raw_med.info['bads']) 
    print('Bad channels in thinking block:', raw_mw.info['bads'])  

    # Interpolate bad channels
    raw_med.interpolate_bads()
    raw_mw.interpolate_bads()

    # Verify that the bad channels have been interpolated
    print(raw_med.info['bads'])  # Should be an empty list
    print(raw_mw.info['bads'])  # Should be an empty list

    # Plot the data
    #raw_med.plot()
    #raw_mw.plot()

    print("\n##### Downsampling #####")

    # Downsample the data to 128 Hz
    raw_med.resample(sfreq=sfreq, npad="auto", verbose=False)
    raw_mw.resample(sfreq=sfreq, npad="auto", verbose=False)

    # Print the new sampling frequency
    print(f"New sampling frequency: {raw_med.info['sfreq']} Hz")

    print("\n##### Averaging and filtering raw data #####")

    preprocessed_data_med = raw_med.copy()
    preprocessed_data_mw = raw_mw.copy()
    # Apply an average reference to the data
    preprocessed_data_med.set_eeg_reference(ref_channels='average', verbose=False)
    preprocessed_data_mw.set_eeg_reference(ref_channels='average', verbose=False)

    # Filter out low frequencies and high frequencies
    preprocessed_data_med.filter(l_freq=l_cut, h_freq=h_cut, verbose=False)
    preprocessed_data_mw.filter(l_freq=l_cut, h_freq=h_cut, verbose=False)

    # average reference
    preprocessed_data_med.set_eeg_reference(ref_channels='average', verbose=False)
    preprocessed_data_mw.set_eeg_reference(ref_channels='average', verbose=False)

    # Verify the new reference
    print(preprocessed_data_med.info['custom_ref_applied'])  # Should be 1 if referencing was applied
    print(preprocessed_data_mw.info['custom_ref_applied'])  # Should be 1 if referencing was applied


    preprocessed_data_med.plot_psd(fmin=l_cut, fmax=h_cut, average=True)
    preprocessed_data_mw.plot_psd(fmin=l_cut, fmax=h_cut, average=True)
    ICA_cleaned_data_med = preprocessed_data_med.copy()
    ICA_cleaned_data_mw = preprocessed_data_mw.copy()

    # Create custom events: start at 0, 5, 10, ... seconds
    n_samples = ICA_cleaned_data_med.n_times
    epochs_samples = np.arange(0, n_samples, epoch_length * sfreq, dtype=int)

    # Create event array with [sample, 0, event_id] for each epoch
    event_code = event_id[f"{subject_group}/med2"]
    events = np.column_stack([epochs_samples, np.zeros(len(epochs_samples), dtype=int), event_code * np.ones(len(epochs_samples), dtype=int)])

    # Create event id for meditation task
    event_id_med = {f"{subject_group}/med2": event_code}

    # Define the epochs object using the custom events
    epochs_med = mne.Epochs(ICA_cleaned_data_med, events, event_id=event_id_med, tmin=0, tmax=epoch_length, baseline=None, detrend=1, preload=True)

    print("\n##### Autorejecting bad epochs #####")

    epochs_original_med = epochs_med.copy()

    ar = AutoReject() ################################                                    REMEMBER TO CITE
    epochs_med, reject_log = ar.fit_transform(epochs_med, return_log=True)
    get_rejection_threshold(epochs_med)  # Get the rejection threshold dictionary

    rsc = Ransac()
    epochs_med = rsc.fit_transform(epochs_med)

    #  Plot the epoch average before and after autorejection and RANSAC
    evoked_before = epochs_original_med.average()
    evoked_after = epochs_med.average()

    # Dropping bad epochs after autorejection
    reject = dict(eeg=reject_threshold)
    epochs_med.drop_bad(reject=reject)
    epochs_med.plot_drop_log()

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))  # 1 row, 2 columns

    # Plot the averages
    evoked_before.plot(axes=axes[0], show=False, spatial_colors=True, time_unit='s')
    axes[0].set_title("Average Before Auto-Rejection")

    evoked_after.plot(axes=axes[1], show=False, spatial_colors=True, time_unit='s')
    axes[1].set_title("Average After Auto-Rejection")

    # save the plot
    output_dir = os.path.join(config.PLOTS_PATH, "autoreject/")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,f"{DATASET.name}_sub-{subject_id}_meditation_task_epoch_averages_before_after.png"))
    plt.show()

    epochs_med.plot_psd(fmin=l_cut, fmax=h_cut, average=True, spatial_colors=False)

    # Create custom events: start at 0, 5, 10, ... seconds
    n_samples = ICA_cleaned_data_mw.n_times
    epochs_samples = np.arange(0, n_samples, epoch_length * sfreq, dtype=int)

    # Create event array with [sample, 0, event_id] for each epoch
    event_code = event_id[f"{subject_group}/think2"]
    events = np.column_stack([epochs_samples, np.zeros(len(epochs_samples), dtype=int), event_code * np.ones(len(epochs_samples), dtype=int)])

    # create event_id for think task
    event_id_think = {f"{subject_group}/think2": event_code}

    # Define the epochs object using the custom events
    epochs_mw = mne.Epochs(ICA_cleaned_data_mw, events, event_id=event_id_think, tmin=0, tmax=epoch_length, baseline=None, detrend=1, preload=True)

    print("\n##### Autorejecting bad epochs #####")

    epochs_original_mw = epochs_mw.copy()

    epochs_mw, reject_log = ar.fit_transform(epochs_mw, return_log=True)
    get_rejection_threshold(epochs_mw)  # Get the rejection threshold dictionary
    epochs_mw = rsc.fit_transform(epochs_mw)

    #  Plot the epoch average before and after autorejection and RANSAC
    evoked_before = epochs_original_mw.average()
    evoked_after = epochs_mw.average()

    # Dropping bad epochs after autorejection
    reject = dict(eeg=reject_threshold)
    epochs_mw.drop_bad(reject=reject)
    epochs_mw.plot_drop_log()

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))  # 1 row, 2 columns

    # Plot the averages
    evoked_before.plot(axes=axes[0], show=False, spatial_colors=True, time_unit='s')
    axes[0].set_title("Average Before Auto-Rejection")

    evoked_after.plot(axes=axes[1], show=False, spatial_colors=True, time_unit='s')
    axes[1].set_title("Average After Auto-Rejection")

    # Adjust layout
    plt.tight_layout()

    # save the plot
    output_dir = os.path.join(config.PLOTS_PATH, "autoreject/")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,f"{DATASET.name}_sub-{subject_id}_think_task_epoch_averages_before_after.png"))
    plt.show()


    # Merge the two epochs objects
    epochs_combined = mne.concatenate_epochs([epochs_med, epochs_mw])


    save_epochs(epochs_combined, path_epochs, subject_id, subfolder="preprocessed")
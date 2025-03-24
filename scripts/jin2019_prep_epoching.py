import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from autoreject import AutoReject, Ransac
import psutil
import gc
import json

import utils.config as config
from utils.config import DATASETS
from utils.helpers import save_epochs



### Defining constants and preparing stuff ###

bids_root = os.path.join(DATASETS["Jin2019"].path, "study1/raw/eeg/")
path_events = os.path.join(DATASETS["Jin2019"].path, "study1/raw/beh/new_events/")
path_bad_channels = config.PREPROCESSING_LOG_PATH
path_epochs = os.path.join(config.EPOCHS_PATH, "external_task/jin2019/")
# df used to localize .bdf files
df_subject_data = DATASETS["Jin2019"].extra_info["subject_session_df"]

# EEG settings
subjects = DATASETS["Jin2019"].subjects
sessions = DATASETS["Jin2019"].sessions
tmin = config.EEG_SETTINGS["EPOCH_START_SEC"]
tmax = tmin + config.EEG_SETTINGS["EPOCH_LENGTH_SEC"]
sfreq = config.EEG_SETTINGS["SAMPLING_RATE"]
h_cut = config.EEG_SETTINGS["HIGH_CUTOFF_HZ"]
l_cut = config.EEG_SETTINGS["LOW_CUTOFF_HZ"]
reject_threshold = config.EEG_SETTINGS["REJECT_THRESHOLD"]
mapping_128_to_64 = DATASETS["Jin2019"].mapping_channels
mapping_non_eeg = DATASETS["Jin2019"].mapping_non_eeg
event_id_map = DATASETS["Jin2019"].event_id_map
event_classes = DATASETS["Jin2019"].event_classes
montage = mne.channels.make_standard_montage('biosemi64')




def load_new_events(path, subject, session):
    file_name = f'subject-{subject}_session-{session}_events.csv'
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        column_names = df.columns.values
        return df.values, column_names
    else:
        raise FileNotFoundError(f"No such file: {file_path}")

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



if __name__ == "__main__":

    process = psutil.Process()
    rsc = Ransac()
    ar = AutoReject()

    # Loop through subjects and sessions
    for subject in subjects:
        for session in sessions:
            bids_path = bids_root + df_subject_data.iloc[subject - 1, session - 1] + ".bdf"
            if not os.path.exists(bids_path):
                log_warning(subject, session, f"File not found: {bids_path}")
                continue
            print(f"Memory before loading: {process.memory_info().rss / 1e6:.2f} MB")
            raw = mne.io.read_raw_bdf(bids_path, preload=True)

            # Extracting the channel names
            old_ch_names = raw.ch_names

            ### Temporary channel names ###

            # Placeholder names for the old channels
            temp_ch_names = ['temp_' + ch for ch in old_ch_names[:-9]]
            temp_ch_names.extend(old_ch_names[-9:])
            mapping_old_to_temp = dict(zip(old_ch_names, temp_ch_names))

            # Rename the channels in the dataset
            raw.rename_channels(mapping_old_to_temp)
            raw.rename_channels(mapping_128_to_64)
            raw.rename_channels(mapping_non_eeg)

            # Set the channel types for the EXG channels
            raw.set_channel_types({
                'sacc_EOG1': 'eog',
                'sacc_EOG2': 'eog',
                'blink_EOG1': 'eog',
                'blink_EOG2': 'eog',
                'EXG5': 'misc',  # Could be a mastoid, set as misc
                'EXG6': 'misc',  # Could be a mastoid, set as misc
                'EXG7': 'misc',  # Could be a mastoid, set as misc
                'EXG8': 'misc'   # Could be a mastoid, set as misc
            })

            # Identify non-EEG channels
            non_eeg_channels = [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type != 'eeg']

            # Get a list of the channels you want to retain (the new 64-channel names and the non-EEG channels)
            channels_to_keep = list(mapping_128_to_64.values()) + non_eeg_channels

            # Drop the channels not in the 64-channel system
            raw.pick(channels_to_keep)

            # Load the BioSemi 64-channel montage
            raw.set_montage(montage)   

            # Set the bad channels for the current subject if it exists in the dictionary
            # Load bad channels from JSON file
            with open(os.path.join(path_bad_channels, "jin2019_bad_channels.json")) as f:
                bad_channels_dict = json.load(f)
            
            if subject in bad_channels_dict:
                if session in bad_channels_dict[subject]:
                    raw.info['bads'] = bad_channels_dict[subject][session]

            # Interpolate bad channels
            raw.interpolate_bads()

            # Downsampling to 128 Hz
            raw.resample(sfreq=sfreq, npad="auto", verbose=False)

            # create epochs with stimuli channel
            events_old = mne.find_events(raw, stim_channel='Status', verbose=False)

            # Load the dataframes
            new_events, event_info = load_new_events(path_events, subject, session)

            # Encode and decode the new events
            encoded_new_events = encode_events(new_events)
            decoded_new_events = decode_events(encoded_new_events)

            try:
                # Verify that the decoded events match the original new events
                assert np.array_equal(new_events, decoded_new_events), "ERROR: Decoded events do not match the original events"

                # Filter events to keep only IDs in range 9–22
                filtered_events = events_old[(events_old[:, 2] >= 10) & (events_old[:, 2] <= 21)]

                # Verify alignment
                assert len(filtered_events) == len(encoded_new_events), "ERROR: Mismatch between filtered events and CSV rows"
            except AssertionError as e:
                log_warning(subject, session, str(e))
                continue

            # Replace the third column in the filtered events array
            filtered_events[:, 2] = encoded_new_events

            # Create an Info object
            info = mne.create_info(ch_names=['STIM'], sfreq=raw.info['sfreq'], ch_types=['stim'])

            # Create a RawArray object for the events
            stim_data = np.zeros((1, raw.n_times))  # Create empty data with one channel
            for event in filtered_events:
                stim_data[0, event[0]] = event[2]  # Place the encoded event value at the event timepoint

            # Create a Raw object for the events
            raw_events = mne.io.RawArray(stim_data, info)

            # Add the events to the raw data
            raw.add_channels([raw_events], force_update_info=True)

            # Drop the Status channel
            raw.drop_channels(['Status'])

            # Finding new events with new stim channel
            events = mne.find_events(raw, stim_channel='STIM', verbose=False)

            # Remove unused event ids from the event_id dictionary
            unused_event_ids = set(event_id_map.values()) - set(events[:, 2])
            event_id = {k: v for k, v in event_id_map.items() if v not in unused_event_ids}

            # Compare number of events before and after new events
            if len(filtered_events) != len(events):
                # Log the difference in number of events
                log_warning(subject, session, f"Number of events have changed from {len(filtered_events)} to {len(events)}")

            # Update the second column with the value from the third column of the previous row
            events[1:, 1] = events[:-1, 2]  # Shift the third column down by one row

            # First row stays zero in the second column
            events[0, 1] = 0

            print("\n##### Averaging and filtering raw data #####")

            preprocessed_data = raw.copy()
            # Apply an average reference to the data
            preprocessed_data.set_eeg_reference(ref_channels='average', verbose=False)

            # Filter out low frequencies and high frequencies
            preprocessed_data.filter(l_freq=l_cut, h_freq=h_cut, verbose=False)

            # average reference
            preprocessed_data.set_eeg_reference(ref_channels='average', verbose=False)
                    
            # Create a new event_id dictionary for the classes
            class_event_id = {label: idx + 1 for idx, label in enumerate(event_classes.keys())}

            # Map the events to the new classes
            classified_events = events.copy()
            for class_label, event_codes in event_classes.items():
                for code in event_codes:
                    classified_events[classified_events[:, 2] == code, 2] = class_event_id[class_label]

            # Create epochs for each class
            epochs = mne.Epochs(preprocessed_data, classified_events, event_id=class_event_id, tmin=tmin, tmax=tmax, 
                                        baseline=None, reject=None, preload=True)

            # Count the number of epochs in each class
            epochs_counts = epochs.__len__()
            epochs_MW_counts = epochs['vs/MW'].__len__() + epochs['sart/MW'].__len__()
            epochs_OT_counts = epochs['vs/OT'].__len__() + epochs['sart/OT'].__len__()

            # Find the ratio of MW vs OT in the data
            ratio_MW_OT = epochs_MW_counts / (epochs_MW_counts + epochs_OT_counts)

            print(f"Ratio of MW vs OT in epochs: {ratio_MW_OT:.2f}")

            # Log the balance
            log_msg(subject, session, f"Ratio of MW vs OT in epochs: {ratio_MW_OT:.2f}"
                    f"\n Number of mind wandering epochs: {epochs_MW_counts}"
                    f"\n Number of on target epochs: {epochs_OT_counts}")

            class_imbalance_range = [0.25, 0.75]  # A 1:4 ratio

            if ratio_MW_OT > class_imbalance_range[1] or ratio_MW_OT < class_imbalance_range[0]:
                log_warning(subject, session, f"WARNING: Ratio of MW vs OT in epochs is not within range. Balance is {ratio_MW_OT}")

            print("\n##### Autorejecting bad epochs #####")

            epochs_original = epochs.copy()

            epochs, reject_log = ar.fit_transform(epochs, return_log=True)
            epochs = rsc.fit_transform(epochs)

            #  Plot the epoch average before and after autorejection
            evoked_before = epochs_original.average()
            evoked_after = epochs.average()

            # Dropping bad epochs after autorejection
            reject = dict(eeg=reject_threshold)
            epochs.drop_bad(reject=reject)

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
            plt.savefig(os.path.join(output_dir,f"{DATASETS["Jin2019"].name}_sub-{subject}-ses-{session}_epoch_averages_before_after.png"))

            print('\n##### Class balance after auto-rejection #####')

            # Count the number of epochs in each class
            epochs_counts = epochs.__len__()
            epochs_MW_vs_counts = epochs['vs/MW'].__len__()
            epochs_MW_sart_counts = epochs['sart/MW'].__len__()
            epochs_OT_vs_counts = epochs['vs/OT'].__len__()
            epochs_OT_sart_counts = epochs['sart/OT'].__len__()

            # Plot the class balance between sart and vs with double bars
            fig, ax = plt.subplots()
            bar_width = 0.35
            index = np.arange(2)

            # Plotting the bars
            bar1 = ax.bar(index, [epochs_MW_vs_counts, epochs_MW_sart_counts], bar_width, label='Mind-Wandering')
            bar2 = ax.bar(index + bar_width, [epochs_OT_vs_counts, epochs_OT_sart_counts], bar_width, label='On Task')

            # Adding labels and title
            ax.set_xlabel('Class')
            ax.set_ylabel('Number of epochs')
            ax.set_title(f'Class balance for subject {subject} session {session}')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(['visual search', 'SART'])
            ax.legend()

            plt.show()

            # save the plot
            output_dir = os.path.join(config.PLOTS_PATH, "class_balance/")
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f'{DATASETS["Jin2019"].name}_Class_balance_sub{subject}_sess{session}.png'))

            # Find the ratio of MW vs OT in the data
            ratio_vs = epochs_MW_vs_counts / (epochs['vs'].__len__())
            ratio_sart = epochs_MW_sart_counts / (epochs['sart'].__len__())

            # Log the balance
            log_msg(subject, session, f"Ratio of MW vs OT in visual search epochs: {ratio_vs:.2f}"
                    f"\n Ratio of MW vs OT in SART epochs: {ratio_sart:.2f}")

            # Save the cleaned epochs
            save_epochs(epochs, path_epochs, subject=subject, session=session)

            del raw, preprocessed_data, epochs
            gc.collect()
            print(f"Memory after cleanup: {process.memory_info().rss / 1e6:.2f} MB")
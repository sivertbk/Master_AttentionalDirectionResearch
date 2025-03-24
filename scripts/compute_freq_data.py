import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pickle

import utils.config as config
from utils.config import DATASETS

path_epochs = config.EPOCHS_PATH

# Identify all subjects in each dataset/task type from the epochs folder
internal_subjects = [f.split('_')[0].split('-')[1] for f in os.listdir(os.path.join(path_epochs, 'internal_task/braboszcz2017')) if f.endswith('cleaned.fif')]
external_subjects = [f.split('_')[0].split('-')[1] for f in os.listdir(os.path.join(path_epochs, 'external_task/jin2019')) if f.endswith('cleaned.fif')]


def load_epochs(task_type, subject, task, state=None, preload=True):
    """
    Load MNE epochs for a specified task type, subject, task, and state.
    
    Parameters:
    - task_type (str): Type of task ('internal_task' or 'external_task').
    - subject (str): Subject identifier (e.g., '001').
    - task (str): Task name (e.g., 'task_name').
    - state (str): State associated with the task (e.g., 'state_name').
    - preload (bool, optional): Whether to preload the epochs into memory. Default is True.
    
    Returns:
    - mne.Epochs: The filtered epochs object.
    
    Raises:
    - ValueError: If `task_type` is invalid or required keys are missing.
    - FileNotFoundError: If the specified files do not exist.
    """

    if task_type not in ['internal_task', 'external_task']:
        raise ValueError("task_type must be either 'internal_task' or 'external_task'")

    if task_type == 'external_task':
        # Handle external task sessions
        sessions = ['ses-1', 'ses-2']
        epochs_list = []
        state = state.upper()

        for session in sessions:
            file_path = f'./epochs/external_task/sub-{subject}_{session}_epo.fif'
            if os.path.exists(file_path):
                print(f"Loading {file_path}")
                epochs = mne.read_epochs(file_path, preload=preload)
                epochs_list.append(epochs)
            else:
                print(f"Warning: Session {session} not found for subject {subject}")
        
        if not epochs_list:
            raise FileNotFoundError(f"No sessions found for subject {subject}")
        
        epochs = mne.concatenate_epochs(epochs_list)

        # Filter epochs based on task and state
        if task == 'combined':
            combined_event_ids = [key for key in epochs.event_id if any(f"{t}/{state}" in key for t in ['vs', 'sart'])]
            if not combined_event_ids:
                raise ValueError(f"State '{state}' not found in event_id of epochs for tasks 'vs' and 'sart'")
            epochs = epochs[combined_event_ids]
        else:
            event_id = next((key for key in epochs.event_id if f"{task}/{state}" in key), None)
            if not event_id:
                raise ValueError(f"Task '{task}' with state '{state}' not found in event_id of epochs")
            epochs = epochs[event_id]

    else:  # Handle internal task
        file_path = f'./epochs/internal_task/sub-{subject}_epo.fif'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No epochs found for subject {subject} at {file_path}")

        print(f"Loading {file_path}")
        epochs = mne.read_epochs(file_path, preload=preload)

        # Retrieve the group from event_id
        group = next((key.split('/')[0] for key in epochs.event_id if '/' in key), None)
        if not group:
            raise ValueError("No group found in event_id of the epochs")

        # Filter epochs based on task
        event_id = next((key for key in epochs.event_id if f"{group}/{task}" in key), None)
        if not event_id:
            raise ValueError(f"Task '{task}' not found in event_id of the epochs")
        epochs = epochs[event_id]

    return epochs


def compute_psd_data():
    """
    Compute the power spectral density (PSD) data for all subjects.
    """

    data_to_load = detect_data_to_compute()

    epochs = load_epochs(data_to_load)

    psd_focused = compute_psd()
    psd_mw = compute_psd()

    save_psd()

def main():
    compute_psd_data()
    #compute_spectrograms()

if __name__ == "__main__":
    main()
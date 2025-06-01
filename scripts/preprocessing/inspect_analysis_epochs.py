"""
    This script has the purpose of plotting and inspecting the epochs
    that are used for the analysis of the EEG data. It is useful to
    visually inspect the epochs and ensure that they are correctly
    preprocessed and ready for analysis. The script will load the epochs
    from the specified path, plot them, and enable the inspector to
    interactively inspect the epochs and make adjustments if necessary.
    """

import mne
import os
from joblib import cpu_count
from tqdm import tqdm

from utils.config import DATASETS
from utils.helpers import iterate_dataset_sessions
from utils.file_io import log_dropped_epochs

# Specify the subfolder for the epochs path
EPOCHS_SUBFOLDER = "analysis_epochs"

def inspect_epochs(dataset_config, dataset_name=None, subject=None, session=None):
    """
    Load the epochs for all datasets. Inspect the epochs and plot them.
    """
    if dataset_name is not None and subject is not None and session is not None:
        dataset_config = {dataset_name: dataset_config[dataset_name]}
        dataset_config[dataset_name].subjects = [subject]
        dataset_config[dataset_name].sessions = [session]
    
    for dataset, subject, session, task, state in iterate_dataset_sessions(dataset_config):
        # Path to the epochs
        epochs_path = os.path.join(dataset.path_epochs, EPOCHS_SUBFOLDER, f"subject-{subject}", f"session-{session}")
        # epochs file name
        epochs_file_name = f"task-{task}_state-{state}_epo.fif"
        # Full path to the epochs file
        file_path = os.path.join(epochs_path, epochs_file_name)
        # Check if the file exists
        if not os.path.exists(file_path):
            continue
        # Load epochs
        epochs = mne.read_epochs(file_path, verbose=False)
        # Check if epochs are empty
        if len(epochs) == 0:
            print(f"Empty epochs for {file_path}. Skipping...")
            continue
        # Pick EEG channels
        epochs.pick(picks="eeg", verbose=False)

        # Plot epochs
        print(f"Inspecting epochs for {dataset.name}, subject {subject}, session {session}, task {task}, state {state}")
        epochs.plot_psd()
        epochs.plot_psd_topomap()
        epochs.plot(block=True, scalings=dict(eeg=100e-6), show=True, title=f"{dataset.name} - {subject} - {session} - {task} - {state}")

        # If any bad epochs are marked, save the updated epochs
        if len(epochs.drop_log) > 0:
            print(f"Bad epochs found for {file_path}. Saving updated epochs...")
            log_dropped_epochs(
                epochs=epochs, 
                dataset=dataset,
                subject=subject,
                stage="analysis_epochs",
                session=session,
                threshold=None,  # No threshold for inspection
            )
            epochs.drop_bad()
            # Save the updated epochs
            epochs.save(file_path, overwrite=True, verbose=False)
        else:
            print(f"No bad epochs found for {file_path}. No changes made.")

if __name__ == "__main__":
    dataset_name = "braboszcz2017"  # Specify dataset name if needed
    subject = "065"  # Specify subject if needed
    session = 1  # Specify session if needed

    inspect_epochs(dataset_config=DATASETS,
                   dataset_name=dataset_name,
                   subject=subject,
                   session=session)

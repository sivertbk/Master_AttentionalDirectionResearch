import mne
import os
from joblib import cpu_count
from tqdm import tqdm

from utils.config import DATASETS, EEG_SETTINGS
from utils.file_io import save_psd_data
from utils.helpers import generate_metadata_epochs, iterate_dataset_sessions


# Specify the subfolder for the epochs path
EPOCHS_SUBFOLDER = "analysis_epochs"


def compute_psd(epochs, eeg_settings):
    """
    Compute the Power Spectral Density (PSD) using Welch's method for the given EEG epochs.

    Parameters:
        epochs (mne.Epochs): EEG epochs data.
        eeg_settings (dict): Dictionary with PSD parameters including scaling factor.

    Returns:
        tuple: 
            - psd (ndarray): PSD array (epochs × channels × frequencies), scaled to µV²/Hz.
            - freqs (ndarray): Frequency values in Hz.
    """
    psd, freqs = mne.time_frequency.psd_array_welch(
        epochs.get_data(),
        sfreq=epochs.info['sfreq'],
        fmin=eeg_settings["PSD_FMIN"],
        fmax=eeg_settings["PSD_FMAX"],
        n_fft=eeg_settings["PSD_N_FFT"],
        n_per_seg=eeg_settings["PSD_N_PER_SEG"],
        n_overlap=eeg_settings["PSD_N_OVERLAP"],
        window=eeg_settings["PSD_WINDOW"],
        average=eeg_settings["PSD_AVERAGE_METHOD"],
        output=eeg_settings["PSD_OUTPUT"],
        n_jobs=cpu_count(),
        remove_dc=eeg_settings["PSD_REMOVE_DC"],
        verbose=False
    )

    # Apply unit scaling (e.g., V² → µV²)
    psd *= eeg_settings["PSD_UNIT_CONVERT"]

    return psd, freqs

def compute_psd_multitaper(epochs, eeg_settings):
    """
    Compute the Power Spectral Density (PSD) using the multitaper method for the given EEG epochs.

    Parameters:
        epochs (mne.Epochs): EEG epochs data.
        eeg_settings (dict): Dictionary with PSD parameters including scaling factor.

    Returns:
        tuple: 
            - psd (ndarray): PSD array (epochs × channels × frequencies), scaled to µV²/Hz.
            - freqs (ndarray): Frequency values in Hz.
    """
    psd, freqs = mne.time_frequency.psd_array_multitaper(
        epochs.get_data(),
        sfreq=epochs.info['sfreq'],
        fmin=eeg_settings["PSD_FMIN"],
        fmax=eeg_settings["PSD_FMAX"],
        bandwidth=None,  # Use default bandwidth
        adaptive=True,  # Use adaptive method
        low_bias=True,  # Use low bias method
        normalization='full',  # Full normalization
        output=eeg_settings["PSD_OUTPUT"],
        n_jobs=cpu_count(),
        remove_dc=eeg_settings["PSD_REMOVE_DC"],
        verbose=False
    )

    # Apply unit scaling (e.g., V² → µV²)
    psd *= eeg_settings["PSD_UNIT_CONVERT"]

    return psd, freqs
            
def compute_psd_data():
    """
    Compute the PSD data for all datasets.
    """
    for dataset, subject, session, task, state in iterate_dataset_sessions(DATASETS):
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
        # Compute PSD
        for average_method in ['mean', 'median', None]:
            variant = f"avg-{average_method}"
            eeg_settings = EEG_SETTINGS.copy()
            eeg_settings["PSD_AVERAGE_METHOD"] = average_method

            psd, freqs = compute_psd(epochs, eeg_settings)

            # Generate metadata
            metadata = generate_metadata_epochs(
                psd_data=psd,
                eeg_settings=eeg_settings,
                dataset=dataset,
                subject=subject,
                session=session,
                task=task,
                state=state
            )

            save_psd_data(
                psd=psd,
                freqs=freqs,
                channels=epochs.ch_names,
                metadata=metadata,
                output_root=dataset.path_psd,
                subject=subject,
                session=session,
                task=task,
                state=state,
                variant=variant
            )



def main():
    compute_psd_data()

if __name__ == "__main__":
    main()
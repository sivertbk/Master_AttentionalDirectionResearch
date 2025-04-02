import mne
import os
from joblib import cpu_count
from tqdm import tqdm

from utils.config import DATASETS, EEG_SETTINGS
from utils.file_io import save_psd_data, load_epochs
from utils.helpers import remap_events_to_original, generate_metadata_epochs, generate_metadata_config


# Specify the subfolder for the epochs path
EPOCHS_SUBFOLDER = "ica_cleaned"




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



def compute_psd_data(dataset_name):
    """
    Compute the PSD data for all subjects in a given dataset, for both mean and median averaging.
    """
    dataset_config = DATASETS[dataset_name]

    subject_session_pairs = [
        (subject, session)
        for subject in dataset_config.subjects
        for session in dataset_config.sessions
    ]

    for subject, session in tqdm(subject_session_pairs, desc="Computing PSD", unit="subject-session"):
        epochs_path = os.path.join(dataset_config.path_epochs, EPOCHS_SUBFOLDER)
        epochs = load_epochs(epochs_path, subject, session, verbose=False)

        if epochs is None:
            continue

        if dataset_name == "jin2019":
            epochs = remap_events_to_original(epochs)

        epochs = epochs.pick_types(eeg=True, verbose=False)
        channels = epochs.ch_names

        for average_method in ['test']:
            variant = f"avg-{average_method}"
            eeg_settings = EEG_SETTINGS.copy()
            eeg_settings["PSD_AVERAGE_METHOD"] = 'mean'

            psd, freqs = compute_psd(epochs, eeg_settings)

            metadata_epochs_df = generate_metadata_epochs(epochs, dataset_config, subject, session)
            metadata_config = generate_metadata_config(eeg_settings, psd)

            save_psd_data(psd, freqs, channels, metadata_epochs_df, metadata_config,
              output_root=dataset_config.path_psd,
              subject=subject, session=session, variant=variant)



def main():
    for dataset_name in DATASETS.keys():
        compute_psd_data(dataset_name)
        # compute_spectrograms()  # Implement later

if __name__ == "__main__":
    main()
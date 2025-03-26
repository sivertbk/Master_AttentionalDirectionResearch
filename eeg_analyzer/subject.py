import os
import numpy as np
import pandas as pd
from eeg_analyzer.metrics import Metrics
from eeg_analyzer.visualizer import Visualizer

class Subject:
    """
    Class to represent an individual subject's PSD data, associated metadata, and EEG metrics.
    """

    def __init__(self, subject_id, session_id, dataset_config):
        """
        Initialize the Subject object.

        Parameters:
            subject_id (str or int): Unique identifier for the subject.
            session_id (str or int): Session identifier.
            dataset_config (DatasetConfig): Configuration object for the dataset.
        """
        self.subject_id = subject_id
        self.session_id = session_id
        self.dataset_config = dataset_config

        self.psd = None
        self.freqs = None
        self.channels = None
        self.metadata = None

        self.metrics = None
        self.viz = Visualizer(self)

    def _generate_filenames(self):
        """Generate filenames for PSD and metadata based on subject and session."""
        fname_prefix = f"sub-{self.subject_id}_ses-{self.session_id}"
        psd_file = os.path.join(self.dataset_config.path_psd, f"{fname_prefix}_psd.npz")
        metadata_file = os.path.join(self.dataset_config.path_psd, f"{fname_prefix}_metadata.csv")
        return psd_file, metadata_file

    def load_data(self):
        """
        Load PSD data and metadata explicitly from saved files.
        """
        psd_file, metadata_file = self._generate_filenames()

        if not os.path.exists(psd_file):
            raise FileNotFoundError(f"PSD file not found: {psd_file}")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load PSD data
        data = np.load(psd_file)
        self.psd = data['psd']
        self.freqs = data['freqs']
        self.channels = data['channels'].tolist() # Convert to list to enable indexing

        # Load metadata
        self.metadata = pd.read_csv(metadata_file)

        # Initialize the Metrics object clearly and explicitly here
        self.metrics = Metrics(self.psd, self.freqs, self.channels)

    def get_epochs_by_state(self, state):
        """
        Get PSD data for epochs matching a given state.

        Parameters:
            state (str): State name (e.g., "mind-wandering" or "focused").

        Returns:
            ndarray: PSD data for epochs matching the specified state.
        """
        if self.metadata is None or self.psd is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        epoch_indices = self.metadata[self.metadata['state'] == state].index.values
        return self.psd[epoch_indices]

    def normalize_psd(self, method='z-score'):
        """
        Normalize PSD data without overwriting original PSD.

        Parameters:
            method (str): Normalization method ('z-score' or 'min-max').

        Returns:
            ndarray: Normalized PSD data.
        """
        if self.psd is None:
            raise ValueError("PSD data not loaded. Call load_data() first.")

        if method == 'z-score':
            mean = self.psd.mean(axis=0, keepdims=True)
            std = self.psd.std(axis=0, keepdims=True)
            normalized_psd = (self.psd - mean) / std
        elif method == 'min-max':
            psd_min = self.psd.min(axis=0, keepdims=True)
            psd_max = self.psd.max(axis=0, keepdims=True)
            normalized_psd = (self.psd - psd_min) / (psd_max - psd_min)
        else:
            raise ValueError("Invalid normalization method.")

        return normalized_psd

    def __repr__(self):
        return (f"Subject(subject_id={self.subject_id}, session_id={self.session_id}, "
                f"dataset='{self.dataset_config.name}')")

"""
Subject
-------

Represents a single participant in a dataset.

Responsibilities:
- Store subject ID, group membership (if applicable), and a collection of recordings.
- Provide access to recordings.
- Encapsulate all subject-specific metadata and logic.

Notes:
- Group can be 'control', 'vipassana', etc., or None if not applicable.
- Recordings are typically stored in a dictionary by name.
"""

import os
import numpy as np
import pandas as pd
from eeg_analyzer.metrics import Metrics
from eeg_analyzer.visualizer import Visualizer

class Subject:
    """
    Class to represent an individual subject's PSD data, associated metadata, and EEG metrics.
    """

    def __init__(self, subject_id, session_id, dataset_config, variant="avg-mean"):
        """
        Initialize the Subject object.

        Parameters:
            subject_id (str or int): Unique identifier for the subject.
            session_id (str or int): Session identifier.
            dataset_config (DatasetConfig): Configuration object for the dataset.
            variant (str): Variant name for the PSD data (e.g., 'avg-mean', 'avg-median').
        """
        self.subject_id = subject_id
        self.session_id = session_id
        self.dataset_config = dataset_config
        self.variant = variant

        self.psd = None
        self.freqs = None
        self.channels = None
        self.metadata = None

        self.metrics = None
        self.viz = Visualizer(self)

    def _get_variant_path(self):
        """Construct the path to the PSD variant directory for the subject and session."""
        sub = f"sub-{self.subject_id}"
        ses = f"ses-{self.session_id}"
        task = f""
        return os.path.join(self.dataset_config.path_psd, sub, ses, task, state, self.variant)

    def load_data(self):
        """
        Load PSD data and metadata from structured variant directory.
        """
        variant_path = self._get_variant_path()
        psd_file = os.path.join(variant_path, "psd.npz")
        metadata_file = os.path.join(variant_path, "metadata_epochs.csv")

        if not os.path.exists(psd_file):
            raise FileNotFoundError(f"PSD file not found: {psd_file}")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load PSD data
        data = np.load(psd_file)
        self.psd = data["psd"]
        self.freqs = data["freqs"]
        self.channels = data["channels"].tolist()  # Ensure it's a list for indexing

        # Load metadata
        self.metadata = pd.read_csv(metadata_file)

        # Initialize Metrics
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
    
    @staticmethod
    def list_variants(path_psd, subject_id, session_id, summarize=True):
        """
        List all available PSD variants for a given subject-session and optionally summarize metadata.

        Parameters:
            path_psd (str): Root PSD directory.
            subject_id (str or int): Subject identifier.
            session_id (str or int): Session identifier.
            summarize (bool): If True, load and return summary metadata for each variant.

        Returns:
            If summarize:
                Dict[str, dict]: Mapping from variant name to metadata summary.
            Else:
                List[str]: List of available variant names.
        """
        sub_dir = os.path.join(path_psd, f"sub-{subject_id}", f"ses-{session_id}")
        if not os.path.isdir(sub_dir):
            return {} if summarize else []

        variants = [
            name for name in os.listdir(sub_dir)
            if os.path.isdir(os.path.join(sub_dir, name))
        ]

        if not summarize:
            return sorted(variants)

        summaries = {}
        for variant in variants:
            metadata_path = os.path.join(sub_dir, variant, "metadata.csv")
            if not os.path.exists(metadata_path):
                continue

            try:
                meta = pd.read_csv(metadata_path, nrows=1)

                # Safely extract values from DataFrame row
                row = meta.iloc[0]
                summaries[variant] = {
                    "average_method": row.get("psd_average_method", "?"),
                    "freq_range_hz": row.get("psd_freq_range_hz", "?"),
                    "n_fft": row.get("psd_n_fft", "?"),
                    "n_per_seg": row.get("psd_n_per_seg", "?"),
                    "n_overlap": row.get("psd_n_overlap", "?"),
                    "freq_resolution_hz": row.get("psd_freq_resolution_hz", "?")
                }
            except Exception as e:
                print(f"Warning: Could not read metadata for variant '{variant}': {e}")

        return summaries
    
    @staticmethod
    def print_variant_summaries(path_psd, subject_id, session_id):
        """
        Print available PSD variants and their metadata summaries for a subject-session.

        Parameters:
            path_psd (str): Root PSD directory.
            subject_id (str or int): Subject identifier.
            session_id (str or int): Session identifier.
        """
        summaries = Subject.list_variants(path_psd, subject_id, session_id, summarize=True)

        if not summaries:
            print(f"No PSD variants found for sub-{subject_id}, ses-{session_id}.")
            return

        print(f"Available PSD variants for sub-{subject_id}, ses-{session_id}:")

        for variant, info in summaries.items():
            print(f"\nVariant: {variant}")
            for key, value in info.items():
                print(f"  {key}: {value}")


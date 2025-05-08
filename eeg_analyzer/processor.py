"""
Processor
---------

Handles transformations and preprocessing of EEG features for analysis.

Responsibilities:
- Apply outlier detection and rejection (e.g., z-score filtering).
- Normalize data across conditions or subjects.
- Support exclusion rules based on configurable criteria.

Notes:
- Should remain agnostic of dataset structure.
- Ideally takes raw metric data as input and returns cleaned versions.
"""

import numpy as np
from eeg_analyzer.metrics import Metrics
from eeg_analyzer.subject import Subject


class Processor:
    def __init__(self, data):
        self.data = data

    def process(self):
        return self.data

    @staticmethod
    def detect_outliers_alpha_power(alpha_power: np.ndarray, threshold: float = 4.0, by_state: bool = False, states: dict = None) -> dict:
        """
        Detect outliers in alpha power across epochs for each channel.

        Parameters:
            alpha_power (ndarray): Alpha power data (epochs × channels).
            threshold (float): Z-score threshold for detecting outliers.
            by_state (bool): Whether to process states independently.
            states (dict): Dictionary mapping state names to epoch indices (optional, required if by_state=True).

        Returns:
            dict: A dictionary with channel names as keys and lists of outlier epoch indices as values.
                  If by_state=True, returns a nested dictionary with states as keys.
        """
        if by_state and states is None:
            raise ValueError("States must be provided when by_state is True.")

        def compute_outliers(data):
            z_scores = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            return {ch_idx: np.where(np.abs(z_scores[:, ch_idx]) > threshold)[0].tolist() for ch_idx in range(data.shape[1])}

        if by_state:
            outliers_by_state = {}
            for state, indices in states.items():
                state_alpha_power = alpha_power[indices]
                outliers_by_state[state] = compute_outliers(state_alpha_power)
            return outliers_by_state
        else:
            return compute_outliers(alpha_power)
        
    @staticmethod
    def summarize_subject_alpha(subject: Subject, threshold=4.0) -> list[dict]:
        rows = []
        for rec in subject.get_all_recordings():
            for task in rec.get_available_tasks():
                for state in rec.get_available_states(task):
                    psd = rec.get_psd(task, state)
                    freqs = rec.get_freqs(task, state)
                    alpha = Metrics.alpha_power(psd, freqs)  # shape [epochs, channels]

                    # Outlier detection
                    outlier_indices = Processor.detect_outliers_alpha_power(alpha, threshold=threshold)
                    mask = np.ones(alpha.shape[0], dtype=bool)
                    for idx_list in outlier_indices.values():
                        if idx_list:  # skip empty
                            idx_arr = np.array(idx_list, dtype=int)
                            mask[idx_arr] = False

                    clean_alpha = alpha[mask]
                    if clean_alpha.shape[0] == 0:
                        continue  # Skip if no data left

                    mean_alpha = clean_alpha.mean(axis=0)  # shape [channels]
                    channel_names = rec.channels

                    print(f"→ Subject {subject.id} session {rec.session_id}, task {task}, state {state}, alpha shape: {alpha.shape}")
                    # if any outliers:
                    print("  Outlier indices:", outlier_indices)


                    for ch_name, val in zip(channel_names, mean_alpha):
                        rows.append({
                            "dataset": subject.dataset.f_name,
                            "subject": subject.id,
                            "session": rec.session_id,
                            "task": task,
                            "state": state,
                            "channel": ch_name,
                            "alpha_power": val
                        })
        return rows
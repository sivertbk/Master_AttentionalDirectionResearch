"""
Recording
---------

Represents a single EEG recording session for a subject.

Responsibilities:
- Store preprocessed PSD data (e.g., per task and state).
- Provide methods to access PSDs or derived features like alpha power.
- Manage loading/saving of EEG data at the session level.

Notes:
- Even if a subject has only one recording, this abstraction keeps the structure consistent.
- Can hold references to metadata like task labels or state mappings.
"""

from collections import defaultdict
import numpy as np

class Recording:
    def __init__(self, session_id: int, psd_entries: list[np.ndarray], metadata_entries: list[dict], freq_entries: list[np.ndarray], channels: list[str]):
        self.session_id = session_id
        self.psd_map = defaultdict(dict)     # task -> state -> PSD
        self.meta_map = defaultdict(dict)    # task -> state -> metadata
        self.freq_map = defaultdict(dict)    # task -> state -> frequencies
        self.channels = channels             # List of channel names

        for psd, meta, freqs in zip(psd_entries, metadata_entries, freq_entries):
            task = meta["task"]
            state = meta["state"]
            self.psd_map[task][state] = psd
            self.meta_map[task][state] = meta
            self.freq_map[task][state] = freqs

    def __repr__(self):
        total_conditions = sum(len(states) for states in self.psd_map.values())
        return f"<Recording session-{self.session_id} with {total_conditions} condition(s)>"
    
    #                                           Public API
    ##########################################################################################################
    
    def get_psd(self, task: str, state: str):
        try:
            return self.psd_map[task][state]
        except KeyError:
            raise ValueError(f"No PSD found for task '{task}' and state '{state}' in session {self.session_id}")

    def get_freqs(self, task: str, state: str) -> np.ndarray:
        try:
            return self.freq_map[task][state]
        except KeyError:
            raise ValueError(f"No frequency data for task '{task}' and state '{state}' in session {self.session_id}")

    def get_metadata(self, task: str, state: str):
        try:
            return self.meta_map[task][state]
        except KeyError:
            raise ValueError(f"No metadata found for task '{task}' and state '{state}' in session {self.session_id}")

    def get_available_tasks(self):
        return list(self.psd_map.keys())

    def get_available_states(self, task: str):
        return list(self.psd_map[task].keys()) if task in self.psd_map else []
    
    def list_conditions(self) -> list[tuple[str, str]]:
        """List all available (task, state) condition pairs."""
        return [
            (task, state)
            for task, states in self.psd_map.items()
            for state in states
        ]

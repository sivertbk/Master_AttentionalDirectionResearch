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
import json
import numpy as np

from utils.dataset_config import DatasetConfig
from eeg_analyzer.recording import Recording

class Subject:
    def __init__(self, dataset_config: DatasetConfig, subject_id: str):
        self.dataset = dataset_config
        self.id = subject_id
        self.group = dataset_config.extra_info.get('subject_groups', {}).get(subject_id, None)
        self.recordings = {}

    def __repr__(self):
        return f"<Subject {self.id} ({self.group or 'no group'}) - {len(self.recordings)} sessions>"
    
    #                                  Public API
    ##########################################################################################################


    def load_data(self, variant="mean"):
        """
        Loads all session recordings for this subject by scanning the standardized folder structure.
        Assumes all data is organized as:
        data/<dataset>/psd_data/subject-<id>/session-<n>/task-<task>/state-<state>/avg-<variant>/
        """
        base_path = os.path.join(self.dataset.path_psd, f"subject-{self.id}")
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        session_dirs = [d for d in os.listdir(base_path) if d.startswith("session-")]

        for session_dir in session_dirs:
            session_id = int(session_dir.split("-")[1])
            session_path = os.path.join(base_path, session_dir)

            psd_entries = []
            metadata_entries = []
            freq_entries = []
            channels = None

            for task in self.dataset.tasks or []:
                for state in self.dataset.states or ["MW", "OT"]:
                    avg_path = os.path.join(session_path, f"task-{task}", f"state-{state}", f"avg-{variant}")
                    psd_file = os.path.join(avg_path, "psd.npz")
                    meta_file = os.path.join(avg_path, "metadata.json")

                    if os.path.exists(psd_file) and os.path.exists(meta_file):
                        psd_data = np.load(psd_file)
                        psd = psd_data["psd"]
                        freqs = psd_data["freqs"]
                        if channels is None:
                            channels = psd_data["channels"].tolist()  # Load channels once
                        with open(meta_file, "r") as f:
                            metadata = json.load(f)

                        psd_entries.append(psd)
                        metadata_entries.append(metadata)
                        freq_entries.append(freqs)

            if psd_entries:
                recording = Recording(
                    session_id=session_id,
                    psd_entries=psd_entries,
                    metadata_entries=metadata_entries,
                    freq_entries=freq_entries,
                    channels=channels
                )
                self.recordings[session_id] = recording

    def get_recording(self, session_id: int):
        return self.recordings.get(session_id)

    def get_all_recordings(self):
        return list(self.recordings.values())
    
    def get_group(self):
        return self.group
    
    def get_id(self):
        return self.id
    
    def get_dataset(self):
        return self.dataset


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
    
    def check_hypothesis(self) -> bool:
        """
        Check if the subject supports the hypothesis of alpha power difference between states.
        This method uses all available sessions and tasks for the subject and disregards differences in tasks.

        If multiple sessions are available, the method will check if the alpha power difference is consistent across all sessions.
        The method will return True if the alpha power difference is consistent across all sessions, otherwise it will return False.
        The method will merge the alpha power data across all tasks within each session.

        The hypothesis states that task orientation should be evident in the alpha power difference between states.
        If task orientation is external, the alpha power should be higher in the "MW" state.
        If task orientation is internal, the alpha power should be higher in the "OT" state.

        Returns:
            bool: True if the subject supports the hypothesis, False otherwise.
        """
        # Check if the subject has at least one session
        if not self.recordings:
            return False
        
        # define the task orientation
        task_orientation = self.dataset.task_orientation

        # Iterate through all sessions and check the hypothesis for each session
        session_results = []
        for recording in self.recordings.values():
            # Initialize lists to store alpha power data for "MW" and "OT" states
            alpha_power_mw = []
            alpha_power_ot = []

            # Iterate through all tasks in the recording
            for task in recording.get_available_tasks():
                # Collect alpha power for "MW" state
                if "MW" in recording.get_available_states(task):
                    alpha_power_mw.append(recording.alpha_power(task=task, state="MW"))

                # Collect alpha power for "OT" state
                if "OT" in recording.get_available_states(task):
                    alpha_power_ot.append(recording.alpha_power(task=task, state="OT"))

            # Check if there is sufficient data for both states in this session
            if not alpha_power_mw or not alpha_power_ot:
                session_results.append(False)
                continue

            # Compute the mean alpha power across all tasks for each state in this session
            mean_alpha_power_mw = np.mean(np.concatenate(alpha_power_mw), axis=0)
            mean_alpha_power_ot = np.mean(np.concatenate(alpha_power_ot), axis=0)

            if task_orientation == "external":
                # For external task orientation, we expect higher alpha power in "MW" state
                session_results.append(np.mean(mean_alpha_power_mw) > np.mean(mean_alpha_power_ot))
            elif task_orientation == "internal":
                # For internal task orientation, we expect higher alpha power in "OT" state
                session_results.append(np.mean(mean_alpha_power_mw) < np.mean(mean_alpha_power_ot))

        # Return True only if all sessions support the hypothesis
        return all(session_results)


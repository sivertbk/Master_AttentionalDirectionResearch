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
        self.group = dataset_config.extra_info.get('subject_groups', {}).get(subject_id, self.dataset.f_name) # Default to dataset name if no group specified
        self.recordings = {}

    def __repr__(self):
        return f"<Subject {self.id} ({self.group or 'no group'}) - {len(self.recordings)} sessions>"
    
    #                                  Public API
    ##########################################################################################################
    def load_data(self, variant="mean", band=(8, 12)):
        """
        Loads all session recordings for this subject by scanning the standardized folder structure.
        Assumes all data is organized as:
        data/<dataset>/psd_data/subject-<id>/session-<n>/task-<task>/state-<state>/avg-<variant>/
        """
        base_path = os.path.join(self.dataset.path_psd, f"subject-{self.id}")
        if not os.path.exists(base_path):
            return None

        # Get all session directories at once instead of listing repeatedly
        try:
            all_dirs = os.listdir(base_path)
            session_dirs = [d for d in all_dirs if d.startswith("session-")]
        except OSError:
            return None

        for session_dir in session_dirs:
            session_id = int(session_dir.split("-")[1])
            session_path = os.path.join(base_path, session_dir)

            psd_entries = []
            metadata_entries = []
            freq_entries = []
            channels = None

            # Pre-compute all file paths to reduce path operations
            tasks = self.dataset.tasks or []
            states = self.dataset.states or ["MW", "OT"]
            
            for task in tasks:
                for state in states:
                    avg_path = os.path.join(session_path, f"task-{task}", f"state-{state}", f"avg-{variant}")
                    psd_file = os.path.join(avg_path, "psd.npz")
                    meta_file = os.path.join(avg_path, "metadata.json")

                    # Batch existence checks - both files must exist
                    if not (os.path.exists(psd_file) and os.path.exists(meta_file)):
                        continue
                    
                    try:
                        # Load PSD data
                        psd_data = np.load(psd_file)
                        psd = psd_data["psd"]
                        freqs = psd_data["freqs"]
                        
                        # Load channels only once per session
                        if channels is None:
                            channels = psd_data["channels"].tolist()
                        
                        # Load metadata
                        with open(meta_file, "r") as f:
                            metadata = json.load(f)

                        psd_entries.append(psd)
                        metadata_entries.append(metadata)
                        freq_entries.append(freqs)
                        
                    except (IOError, KeyError, json.JSONDecodeError) as e:
                        # Skip corrupted files instead of crashing
                        print(f"Warning: Failed to load data for {task}-{state}: {e}")
                        continue

            if psd_entries:
                recording = Recording(
                    session_id=session_id,
                    psd_entries=psd_entries,
                    metadata_entries=metadata_entries,
                    freq_entries=freq_entries,
                    channels=channels,
                    band=band
                )
                self.recordings[session_id] = recording

    def get_recording(self, session_id: int) -> Recording:
        return self.recordings.get(session_id)

    def get_all_recordings(self) -> list[Recording]:
        return list(self.recordings.values())

    def get_group(self) -> str:
        return self.group

    def get_id(self) -> str:
        return self.id

    def get_dataset(self) -> DatasetConfig:
        return self.dataset
    
    def get_total_epochs(self):
        """
        Returns the number of epochs in all sessions.
        """
        n_epochs = 0
        for recording in self.recordings.values():
            # More efficient: get epoch counts once and sum them
            if recording.channels:
                # All channels have the same epoch count, so just use the first one
                channel = recording.channels[0]
                for task, state in recording.list_conditions():
                    epoch_count = recording.get_stat('epoch_count', channel, task=task, state=state)
                    if not np.isnan(epoch_count):
                        n_epochs += int(epoch_count)
        return n_epochs
    
    def get_epochs_per_condition(self):
        """
        Returns a dictionary with the number of epochs for each task and state.
        The keys are tuples (task, state) and the values are the number of epochs
        for that condition in all available sessions.  
        """
        epochs_per_condition = {}
        for recording in self.recordings.values():
            if recording.channels:
                # All channels have the same epoch count, so just use the first one
                channel = recording.channels[0]
                for task, state in recording.list_conditions():
                    epoch_count = recording.get_stat('epoch_count', channel, task=task, state=state)
                    if not np.isnan(epoch_count):
                        condition = (task, state)
                        epochs_per_condition[condition] = epochs_per_condition.get(condition, 0) + int(epoch_count)
        return epochs_per_condition
    
    def get_mean_alpha_diff_by_state(self, session_id: int):
        """
        Computes the mean alpha power difference between the two states ("OT" and "MW") for a given session,
        combining all available epochs across all tasks for each state.

        Args:
            session_id (int): The session ID to retrieve data from.

        Returns:
            np.ndarray: Array where the value is the difference (OT - MW)
              of mean alpha power across all epochs and tasks, shape (n_channels,).
        """
        recording = self.get_recording(session_id)
        if recording is None:
            print(f"Warning: No recording found for subject {self.id} session {session_id}")
            return np.array([])

        # Check if recording has the appropriate band range for alpha power (8-12 Hz)
        if recording.band_range != (8, 12):
            print(f"Warning: Recording does not have alpha band power calculated. "
                  f"Current band: {recording.band_range}, expected: (8, 12)")
            return np.array([])

        # Use the new get_stat method to get state-level mean statistics
        # Get mean band power for each state across all tasks
        mean_mw_per_channel = []
        mean_ot_per_channel = []
        
        for channel in recording.channels:
            mw_mean = recording.get_stat('mean', channel, data_type='band_power', state='MW')
            ot_mean = recording.get_stat('mean', channel, data_type='band_power', state='OT')
            
            if np.isnan(mw_mean) or np.isnan(ot_mean):
                # Skip channels with invalid data
                continue
                
            mean_mw_per_channel.append(mw_mean)
            mean_ot_per_channel.append(ot_mean)

        if not mean_mw_per_channel or not mean_ot_per_channel:
            return np.array([])

        mean_mw = np.array(mean_mw_per_channel)
        mean_ot = np.array(mean_ot_per_channel)

        return mean_ot - mean_mw

    def get_channel_names(self):
        """
        Retrieves the channel names for the subject. Since channel names are consistent across sessions,
        this method uses the first valid session to extract the channel names.

        Returns:
            list: A list of channel names.
        """
        for recording in self.recordings.values():
            return recording.get_channel_names()
        raise ValueError("No valid recordings available to retrieve channel names.")


"""
Dataset
-------

Represents a full EEG dataset, including multiple subjects, tasks, and sessions.

Responsibilities:
- Store metadata and subject list for a given dataset (e.g., 'jin2019').
- Filter subjects by group membership or other metadata.
- Provide access to individual subjects and their recordings.

Notes:
- Subject group assignments are dataset-specific and loaded from config.
"""

from typing import Optional, List, Set, Union
import numpy as np
from tqdm import tqdm

from utils.dataset_config import DatasetConfig
from utils.config import EEG_SETTINGS, channel_positions, cortical_regions, ROIs
from . import Subject


class Dataset:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.name = config.name
        self.f_name = config.f_name
        self.subject_IDs = config.subjects  # List of subject IDs in the dataset
        self.tasks = config.tasks
        self.states = config.states
        self.task_orientation = config.task_orientation

        # Set of unique subject groups present in this dataset
        subject_groups = config.extra_info.get('subject_groups', {})
        self.subject_groups = set(
            subject_groups[subject_id]
            for subject_id in self.subject_IDs
            if subject_id in subject_groups
        )

        self._subject_group_map = subject_groups  # Store for internal use
        self.subjects = {}  # subject_id: Subject

    def __repr__(self):
        return f"<Dataset {self.name} with {len(self.subject_IDs)} subjects>"
    
    def __str__(self):
        subjects_str = f"\n Subjects:\n    - {len(self.subject_IDs)} subjects, {len(self.subject_groups)} groups: [{', '.join(self.subject_groups)}]"
        tasks_str = f"\n Tasks:\n    - {', '.join(self.tasks)}"
        states_str = f"\n States:\n    - {', '.join(self.states)}"
        # Robust handling for empty or missing epochs_info
        if hasattr(self, 'epochs_info') and self.epochs_info:
            median_strs, min_strs, max_strs = [], [], []
            for (task, state), values in self.epochs_info.items():
                if values:
                    median_strs.append(f'{task} - {state}: {np.median(values)}')
                    min_strs.append(f'{task} - {state}: {np.min(values)}')
                    max_strs.append(f'{task} - {state}: {np.max(values)}')
                else:
                    median_strs.append(f'{task} - {state}: N/A')
                    min_strs.append(f'{task} - {state}: N/A')
                    max_strs.append(f'{task} - {state}: N/A')
            epochs_median_str = f"\n Epochs:\n    - Median epochs per condition: [{', '.join(median_strs)}]"
            epochs_min_str = f"\n    - Min epochs per condition: [{', '.join(min_strs)}]"
            epochs_max_str = f"\n    - Max epochs per condition: [{', '.join(max_strs)}]"
        else:
            epochs_median_str = "\n Epochs:\n    - Median epochs per condition: N/A"
            epochs_min_str = "\n    - Min epochs per condition: N/A"
            epochs_max_str = "\n    - Max epochs per condition: N/A"
        # Robust handling for empty or missing state_ratios
        if hasattr(self, 'state_ratios') and self.state_ratios:
            state_ratios_values = list(self.state_ratios.values())
            state_ratios_keys = list(self.state_ratios.keys())
            min_index = np.argmin(state_ratios_values)
            max_index = np.argmax(state_ratios_values)
            state_ratios_str = (
                f"\n    - State ratios: "
                f"mean:{np.mean(state_ratios_values):.3f}, "
                f"min:{state_ratios_values[min_index]:.3f} (subject_id: {state_ratios_keys[min_index]}), "
                f"max:{state_ratios_values[max_index]:.3f} (subject_id: {state_ratios_keys[max_index]})"
            )
        else:
            state_ratios_str = "\n    - State ratios: N/A"
        return f"Dataset: {self.name}" + subjects_str + tasks_str + states_str + epochs_median_str + epochs_min_str + epochs_max_str + state_ratios_str

    def __len__(self):
        # Return the number of subjects in the dataset that are loaded
        return len(self.subjects)

    
    #                                 Public API
    ##########################################################################################################

    def load_subjects(self, variant: str = "mean"):
        """
        Load all subjects and their recordings into self.subjects.
        This calls subject.load_data(variant) for each subject.
        Also gather other stats about the dataset.
        """
        for subject_id in tqdm(self.subject_IDs, desc="Loading subjects"):
            subject = Subject(self.config, subject_id)
            subject.load_data(variant=variant)
            if subject.recordings == {}:
                print(f"Warning: Subject {subject_id} has no recordings. Skipping.")
                # remove subject id from subject_IDs
                if subject_id in self.subject_IDs:
                    self.subject_IDs.remove(subject_id)
                continue
            self.subjects[subject_id] = subject
        # Recalculate stats after loading
        self._recalculate_dataset_stats()

    def filter_recordings_by_state_ratio(self, min_ratio: float, max_ratio: float):
        """
        Remove recordings (sessions) from subjects if their MW/OT epoch ratio is outside [min_ratio, max_ratio].
        Returns a log of removed recordings: list of dicts with subject_id, session_id, and ratio.
        After removal, updates dataset stats.
        """
        removed_log = []
        for subject in self.subjects.values():
            to_remove = []
            for session_id, recording in list(subject.recordings.items()):
                ratio = recording.get_mw_ot_epoch_ratio()
                if (ratio < min_ratio) or (ratio > max_ratio):
                    to_remove.append((session_id, ratio))
            for session_id, ratio in to_remove:
                del subject.recordings[session_id]
                removed_log.append({
                    "subject_id": subject.id,
                    "session_id": session_id,
                    "ratio": ratio
                })
        # Remove subjects with no recordings left
        empty_subjects = [sid for sid, subj in self.subjects.items() if not subj.recordings]
        for sid in empty_subjects:
            del self.subjects[sid]
        # Update subject_IDs and subject_groups
        self.subject_IDs = list(self.subjects.keys())
        self.subject_groups = set(
            self._subject_group_map[sub_id]
            for sub_id in self.subject_IDs
            if sub_id in self._subject_group_map and self._subject_group_map[sub_id] is not None
        )
        # Recalculate stats after filtering
        self._recalculate_dataset_stats()
        return removed_log

    def _recalculate_dataset_stats(self):
        """
        Recalculate self.epochs_info and self.state_ratios based on current subjects/recordings.
        """
        self.epochs_info = {}
        epochs_per_condition = []
        for subject in self.subjects.values():
            epochs_per_condition.append(subject.get_epochs_per_condition())
        for subject_epochs in epochs_per_condition:
            for (task, state), num_epochs in subject_epochs.items():
                if (task, state) not in self.epochs_info:
                    self.epochs_info[(task, state)] = []
                self.epochs_info[(task, state)].append(num_epochs)
        self.state_ratios = {}
        for id, subject in self.subjects.items():
            self.state_ratios[id] = None

    def get_subject_group(self, subject_id: str) -> Optional[str]:
        """Return the group name of a given subject, or None if not assigned."""
        self._ensure_data_loaded()
        return self._subject_group_map.get(subject_id)

    def get_group_subjects(self, group: str) -> list[str]:
        """Return a list of subject IDs that belong to a specified group."""
        self._ensure_data_loaded()
        return [
            sid for sid in self.subject_IDs
            if self.get_subject_group(sid) == group
        ]
    
    def get_channel_names(self) -> list[str]:
        """Return the list of channel names from the first subject."""
        self._ensure_data_loaded()
        if self.subjects:
            return self.subjects[next(iter(self.subjects))].get_channel_names()
        return []

    def get_subject(self, subject_id: str) -> Optional['Subject']:
        """Return a loaded Subject object by ID, or None if not found."""
        self._ensure_data_loaded()
        return self.subjects.get(subject_id)

    def get_subjects(self, group: Optional[str] = None) -> List['Subject']:
        """
        Return a list of Subject objects.
        If `group` is specified, filter only those in the given group.
        """
        self._ensure_data_loaded()
        if group is None:
            return list(self.subjects.values())
        return [
            subj for subj in self.subjects.values()
            if subj.group == group
        ]
    
    def get_total_epochs(self, subject_id: str) -> int:
        """
        Returns the total number of epochs for a given subject.
        """
        self._ensure_data_loaded()
        if subject_id in self.subjects:
            return self.subjects[subject_id].get_total_epochs()
        else:
            raise ValueError(f"Subject {subject_id} not found in dataset.")

    def get_epochs_per_condition(self, subject_id = None) -> dict:
        """
        Returns a dictionary with the number of epochs for each task and state.
        The keys are tuples (task, state) and the values are the number of epochs
        for that condition in all available sessions. If subject_id is None,
        return the epochs info for all subjects.
        """
        self._ensure_data_loaded()
        if subject_id is None:
            return self.epochs_info
        else:
            return self.subjects[subject_id].get_epochs_per_condition()
    
    def get_state_ratio(self, subject_id = None) -> dict:
        """
        Returns the ratio between the number of epochs in each state (there is always two states).
        If subject_id is None, return the state ratio for all subjects.
        """
        self._ensure_data_loaded()
        if subject_id is None:
            return self.state_ratios
        else:
            return self.state_ratios.get(subject_id, None)

    def get_groups(self) -> Set[str]:
        """Return the set of all subject groups in the dataset."""
        return self.subject_groups
    
    def remove_subjects(self, subject_ids: Union[str, List[str]]) -> List[str]:
        """
        Remove subjects from the dataset.
        If a single subject ID is provided, remove it.
        If a list of subject IDs is provided, remove all of them.
        Returns a list of subject IDs that were actually removed.
        """
        if isinstance(subject_ids, str):
            subject_ids_list = [subject_ids]
        else:
            subject_ids_list = subject_ids
        
        actually_removed_ids = []
        for subject_id in subject_ids_list:
            if subject_id in self.subjects:
                del self.subjects[subject_id]
                actually_removed_ids.append(subject_id)
            else:
                print(f"Subject {subject_id} not found in dataset, not removed.")
                pass
                
        # Update the subject IDs and groups if any subjects were removed
        if actually_removed_ids:
            self.subject_IDs = list(self.subjects.keys())
            # Re-calculate subject_groups based on remaining subjects
            self.subject_groups = set(
                self._subject_group_map[sub_id] 
                for sub_id in self.subject_IDs 
                if sub_id in self._subject_group_map and self._subject_group_map[sub_id] is not None
            )
        return actually_removed_ids

    def get_mean_alpha_powers(self, session_id: int) -> dict:
        """
        Returns the mean alpha power for each subject in the dataset.
        """
        self._ensure_data_loaded()
        return {
            subj.id: subj.get_mean_alpha_power(session_id)
            for subj in self.subjects.values()
        }

    def get_mean_alpha_diff_by_condition(self, session_id: int) -> dict:
        """
        Returns the mean alpha power difference between states within task for each subject in the dataset.

        exception: if the dataset is braboszcz2017, the mean alpha power difference is between tasks.
        """
        self._ensure_data_loaded()
        return {
            subj.id: subj.get_mean_alpha_diff_by_condition(session_id)
            for subj in self.subjects.values()
        }
    
    def get_mean_alpha_diff_by_state(self, session_id: int) -> dict:
        """
        Returns the mean alpha power difference between states for each subject in the dataset.
        """
        self._ensure_data_loaded()
        return {
            subj.id: subj.get_mean_alpha_diff_by_state(session_id)
            for subj in self.subjects.values()
        }

    def to_long_band_power_list(self, freq_band: tuple[float, float]) -> list[dict]:
        """
        Extract per-epoch band power data from all subjects and recordings in the dataset.

        Args:
            freq_band (tuple): Frequency band as (low, high) in Hz.

        Returns:
            list[dict]: Each dict contains:
                - "dataset"
                - "subject_session": "<subject_id>_<session_id>"
                - "subject_id"
                - "session_id"
                - "group"
                - "epoch_idx"
                - "channel"
                - "cortical_region"
                - "hemisphere"
                - "task"
                - "state" (int: 0=OT, 1=MW)
                - "band_power" (float: band power in µV²)
                - "band_log_power" (float: band power in natural logarithm)
                - "is_bad" (bool: True if the epoch is marked as bad)
        """
        from eeg_analyzer.metrics import Metrics  # local import to avoid circular
        long_list = []
        self._ensure_data_loaded()
        
        cortical_regions_map = {}
        for region, ch_names in cortical_regions.items():
            for ch_name in ch_names:
                cortical_regions_map[ch_name] = region

        for subject in self.subjects.values():
            subj_id = subject.id
            for rec in subject.get_all_recordings():
                sess_id = rec.session_id
                subject_session = f"{subj_id}_{sess_id}"
                epoch_counter = 0  # Unique epoch index within subject_session
                for task in rec.get_available_tasks():
                    for state in rec.get_available_states(task):
                        psd = rec.get_psd(task, state)
                        freqs = rec.get_freqs(task, state)
                        band_power = Metrics.band_power(psd, freqs, freq_band, operation='mean')  # shape: (epochs, channels)
                        band_log = Metrics.band_log(psd, freqs, freq_band, operation='mean')  # shape: (epochs, channels)
                        n_epochs, n_channels = band_power.shape
                        for epoch_idx in range(n_epochs):
                            for ch_idx, ch_name in enumerate(rec.channels):
                                long_list.append({
                                    "dataset": self.f_name,
                                    "subject_session": subject_session,
                                    "subject_id": subj_id,
                                    "session_id": sess_id,
                                    "group": subject.group,
                                    "epoch_idx": epoch_counter,
                                    "channel": ch_name,
                                    "cortical_region": cortical_regions_map.get(ch_name, None),
                                    "hemisphere": "central" if channel_positions[ch_name][0] == 0 else "left" if channel_positions[ch_name][0] < 0 else "right",
                                    "task": task,
                                    "task_orientation": self.task_orientation,
                                    "state": state,
                                    "band_power": float(band_power[epoch_idx, ch_idx]),
                                    "band_log": float(band_log[epoch_idx, ch_idx]),
                                    "is_bad": False
                                })
                                if ch_idx == n_channels - 1:
                                    epoch_counter += 1
        return long_list

    def estimate_long_band_length(self, variant: str = "mean") -> int:
        """
        Estimate the total number of rows in the long-format band power DataFrame
        (i.e., sum of epochs × channels for all subject-session-task-state combinations).

        Args:
            variant (str): PSD variant to load (default: "mean").

        Returns:
            int: Total expected number of rows.
        """
        # Ensure subjects are loaded
        if not self.subjects:
            self.load_subjects(variant=variant)
        total = 0
        for subject in self.subjects.values():
            for rec in subject.get_all_recordings():
                for task in rec.get_available_tasks():
                    for state in rec.get_available_states(task):
                        psd = rec.get_psd(task, state)
                        # psd shape: (epochs, channels, freqs) or (epochs, channels)
                        n_epochs = psd.shape[0]
                        n_channels = psd.shape[1]
                        total += n_epochs * n_channels
        return total

    #                                 Private API
    ##########################################################################################################
    def _load_subject(self, subject_id: str, variant: str):
        """Load a single subject's data."""
        subject = Subject(self.config, subject_id)
        subject.load_data(variant=variant)
        return subject
    
    def _is_loaded(self, subject_id: str) -> bool:
        return subject_id in self.subjects

    def _ensure_data_loaded(self):
        """Raise an error if subjects are not loaded."""
        if not self.subjects:
            raise RuntimeError("Subjects data is not loaded. Call `load_subjects()` first.")



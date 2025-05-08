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

from utils.dataset_config import DatasetConfig
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
        return f"Dataset: {self.name} ({len(self.subject_IDs)} subjects, {len(self.subject_groups)} groups)"
    
    def __len__(self):
        # Return the number of subjects in the dataset that are loaded
        return len(self.subjects)

    
    #                                 Public API
    ##########################################################################################################

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

    def load_subjects(self, variant: str = "mean"):
        """
        Load all subjects and their recordings into self.subjects.
        This calls subject.load_data(variant) for each subject.
        """
        for subject_id in self.subject_IDs:
            subject = Subject(self.config, subject_id)
            subject.load_data(variant=variant)
            self.subjects[subject_id] = subject

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

    def get_groups(self) -> Set[str]:
        """Return the set of all subject groups in the dataset."""
        return self.subject_groups
        
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



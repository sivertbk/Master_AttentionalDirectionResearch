"""
EEGAnalyzer
-----------

Top-level coordinator class for managing EEG data analysis across multiple datasets.

Responsibilities:
- Manage and access all loaded datasets.
- Provide unified interfaces for retrieving subjects and groups across datasets.
- Coordinate workflows such as loading data, running metrics, and executing analysis pipelines.

Usage:
- Entry point for running full analysis across datasets.
"""
from eeg_analyzer.dataset import Dataset


class EEGAnalyzer:
    def __init__(self, dataset_configs):
        """
        Initialize the EEGAnalyzer with a list of dataset configurations.
        """
        self.datasets = {}
        for config in dataset_configs:
            dataset = Dataset(config)
            self.datasets[dataset.f_name] = dataset
            dataset.load_data()

    def __repr__(self):
        return f"<EEGAnalyzer with {len(self.datasets)} datasets>"
    
    def get_dataset(self, name):
        """
        Get a dataset by its name.
        """
        return self.datasets.get(name)
    
    def get_subject(self, dataset_name, subject_id):
        """
        Get a subject by its ID from a specific dataset.
        """
        dataset = self.get_dataset(dataset_name)
        if dataset:
            return dataset.get_subject(subject_id)
        return None
    
    def remove_subjects(self, dataset_name, subject_ids):
        """
        Remove subjects from a specific dataset.
        """
        dataset = self.get_dataset(dataset_name)
        if dataset:
            for subject_id in subject_ids:
                dataset.remove_subject(subject_id)

    
    

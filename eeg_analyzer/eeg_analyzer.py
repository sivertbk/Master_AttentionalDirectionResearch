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
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
import pickle # Added for saving/loading analyzer objects
from typing import Union # Import Union for older Python versions

from eeg_analyzer.dataset import Dataset
from eeg_analyzer.statistics import Statistics # Import the Statistics class
from utils.config import EEGANALYZER_OBJECT_DERIVATIVES_PATH, NAME_LIST


class EEGAnalyzer:
    def __init__(self, dataset_configs: dict, analyzer_name: str = None, description: str = None):
        """
        Initialize the EEGAnalyzer.
        - If analyzer_name is provided: uses/creates a directory with that name.
          If the directory exists, it's reused (a message is printed).
        - If analyzer_name is None: generates a random name. If a directory for
          that random name already exists, it keeps generating new random names
          until a unique one (without an existing directory) is found. Then,
          a new directory is created for this unique random name.
        - description: An optional user-provided description for the analyzer instance.
        """
        self._history = [] # Initialize history log first
        
        if analyzer_name is None:
            # Generate a unique random name if no name is provided
            while True:
                potential_name = random.choice(NAME_LIST)
                potential_derivatives_path = os.path.join(EEGANALYZER_OBJECT_DERIVATIVES_PATH, potential_name)
                if not os.path.exists(potential_derivatives_path):
                    self.analyzer_name = potential_name
                    self.derivatives_path = potential_derivatives_path
                    os.makedirs(self.derivatives_path, exist_ok=True)
                    log_msg = f"EEGAnalyzer instance '{self.analyzer_name}' (randomly named): Created new derivatives directory at {self.derivatives_path}"
                    print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
                    self._log_event("Initialization", {"status": "New (random name)", "name": self.analyzer_name, "path": self.derivatives_path, "message": log_msg})
                    break
        else:
            # User provided a name
            self.analyzer_name = analyzer_name
            self.derivatives_path = os.path.join(EEGANALYZER_OBJECT_DERIVATIVES_PATH, self.analyzer_name)
            if os.path.exists(self.derivatives_path):
                log_msg = f"EEGAnalyzer instance '{self.analyzer_name}': Directory already exists at {self.derivatives_path}. Will use this directory."
                print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
                self._log_event("Initialization", {"status": "Existing directory (user-defined name)", "name": self.analyzer_name, "path": self.derivatives_path, "message": log_msg})
            else:
                os.makedirs(self.derivatives_path, exist_ok=True)
                log_msg = f"EEGAnalyzer instance '{self.analyzer_name}': Created new derivatives directory at {self.derivatives_path}"
                print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
                self._log_event("Initialization", {"status": "New directory (user-defined name)", "name": self.analyzer_name, "path": self.derivatives_path, "message": log_msg})

        self.description = description if description is not None else "No description provided for this EEGAnalyzer instance."
        if description is not None:
             self._log_event("Description Set", {"description": self.description})

        self.datasets = {}
        self.df = None # Initialize df attribute
        loaded_datasets_info = {}
        for name, config in dataset_configs.items():
            dataset = Dataset(config)
            self.datasets[name] = dataset
            dataset.load_subjects()
            loaded_datasets_info[name] = f"{len(dataset.subjects)} subjects"
            print(f"[EEGAnalyzer - {self.analyzer_name}] Loaded dataset: {name} with {len(dataset.subjects)} subjects.")
        self._log_event("Datasets Loaded on Init", {"datasets_loaded": loaded_datasets_info})


    def _log_event(self, event_name: str, details: dict = None):
        """Appends an event to the analyzer's history."""
        if not hasattr(self, '_history'): # Ensure history is initialized, e.g. if loaded from older pickle
            self._history = []
        log_entry = {
            "timestamp": datetime.now(),
            "event": event_name,
            "details": details if details is not None else {}
        }
        self._history.append(log_entry)

    def set_description(self, new_description: str):
        """Sets or updates the description of the EEGAnalyzer instance."""
        self.description = new_description
        self._log_event("Description Updated", {"new_description": new_description})
        print(f"[EEGAnalyzer - {self.analyzer_name}] Analyzer description updated.")

    def get_description(self) -> str:
        """Returns the current description of the EEGAnalyzer instance."""
        return self.description

    def get_history(self, formatted: bool = True) -> Union[list, str]:
        """
        Returns the history log of the EEGAnalyzer instance.
        
        Parameters:
        - formatted (bool): If True, returns a human-readable string. 
                            If False, returns the raw list of log dictionaries.
        """
        if not hasattr(self, '_history'):
            return "No history available." if formatted else []
            
        if formatted:
            if not self._history:
                return "No history events recorded."
            history_str = f"--- History for EEGAnalyzer '{self.analyzer_name}' ---\n"
            history_str += "=" * (len(f"History for EEGAnalyzer '{self.analyzer_name}':") + 6) + "\n"
            for entry in self._history:
                history_str += f"[{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] Event: {entry['event']}\n"
                if entry['details']:
                    for key, value in entry['details'].items():
                        history_str += f"    - {key}: {value}\n"
                history_str += "---\n"
            return history_str
        return self._history

    def __repr__(self):
        df_info = "No DataFrame"
        if self.df is not None:
            df_info = f"DataFrame with shape {self.df.shape}"
        return (f"<EEGAnalyzer(name='{self.analyzer_name}', "
                f"datasets={len(self.datasets)}, "
                f"{df_info}, "
                f"derivatives_path='{self.derivatives_path}')>")
    
    def __str__(self):
        dataset_names = ', '.join(self.datasets.keys()) if self.datasets else "None"
        df_status = f"DataFrame with shape {self.df.shape}" if self.df is not None else "No DataFrame"
        return (f"--- EEGAnalyzer Instance: '{self.analyzer_name}' ---\n"
                f"  Description: {self.description}\n"
                f"  Datasets ({len(self.datasets)}): {dataset_names}\n"
                f"  DataFrame Status: {df_status}\n"
                f"  Derivatives Path: {self.derivatives_path}\n"
                f"--------------------------------------")
    
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
            removed_subjects_log = []
            for subject_id in subject_ids:
                dataset.remove_subject(subject_id)
                removed_subjects_log.append(subject_id)
            self._log_event("Subjects Removed", {"dataset": dataset_name, "removed_ids": removed_subjects_log})
        else:
            self._log_event("Remove Subjects Attempt Failed", {"dataset": dataset_name, "reason": "Dataset not found"})


    def create_dataframe(self, freq_band = (8,12)) -> pd.DataFrame:
        """
        Create a DataFrame containing all subjects for all datasets.
        Uses to_long_band_power_list() to get the data.
        """
        #empty dataframe
        self.df = pd.DataFrame()
        print(f"[EEGAnalyzer - {self.analyzer_name}] Creating DataFrame from datasets...")
        dataset_details_log = {}
        for name, dataset in self.datasets.items():
            print(f"[EEGAnalyzer - {self.analyzer_name}] Processing dataset: {name}")
            # get list of dicts for each dataset
            dataset_data = dataset.to_long_band_power_list(freq_band)
            # Convert list of dicts to DataFrame
            dataset_df = pd.DataFrame(dataset_data)
            dataset_details_log[name] = f"{len(dataset_df)} rows"
            # Add to df
            self.df = pd.concat([self.df, dataset_df], ignore_index=True)
    
        log_msg = f"DataFrame created with {len(self.df)} rows and {len(self.df.columns)} columns."
        print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
        self._log_event("DataFrame Created", {
            "freq_band": freq_band, 
            "rows": len(self.df), 
            "columns": len(self.df.columns),
            "dataset_contributions": dataset_details_log,
            "message": log_msg
        })
        return self.df

    def save_dataframe(self, dir: str = None, filename: str = "eeganalyzer_default_df.csv"):
        """
        Save the DataFrame to a CSV file.
        Defaults to the object's specific derivatives directory and a default filename.
        """
        if dir is None:
            dir = self.derivatives_path
        
        if hasattr(self, 'df') and self.df is not None:
            os.makedirs(dir, exist_ok=True)  # Ensure the directory exists
            filepath = os.path.join(dir, filename)
            self.df.to_csv(filepath, index=False)
            log_msg = f"DataFrame saved to {filepath}"
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("DataFrame Saved", {"path": filepath, "rows": len(self.df), "columns": len(self.df.columns), "message": log_msg})
        else:
            log_msg = "DataFrame not created yet or is empty. Cannot save."
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Save DataFrame Attempt Failed", {"reason": log_msg})


    def load_dataframe(self, dir: str = None, filename: str = "eeganalyzer_default_df.csv"):
        """
        Load the DataFrame from a CSV file.
        Defaults to the object's specific derivatives directory and a default filename.
        """
        if dir is None:
            dir = self.derivatives_path

        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            self.df = pd.read_csv(filepath)
            log_msg = f"DataFrame loaded from {filepath}"
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("DataFrame Loaded", {"path": filepath, "rows": len(self.df), "columns": len(self.df.columns), "message": log_msg})
        else:
            log_msg = f"File {filepath} does not exist. DataFrame not loaded."
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Load DataFrame Attempt Failed", {"path": filepath, "reason": "File not found", "message": log_msg})

    def generate_summary_table(self, 
                               groupby_cols: list, 
                               target_col: str = 'band_power', 
                               filter_type: str = "unfiltered", 
                               output_filename_suffix: str = None,
                               source_df: pd.DataFrame = None) -> Union[pd.DataFrame, None]:
        """
        Generates a summary table with descriptive statistics.

        Parameters:
        - groupby_cols (list): List of columns to group by.
        - target_col (str): The column to calculate statistics on (default: 'band_power').
        - filter_type (str): A descriptor for the data filtering state (e.g., "unfiltered", "z_scored").
                             Used for logging and filename generation.
        - output_filename_suffix (str): Optional suffix for the output CSV file. If None,
                                        a default name based on groupby_cols and filter_type is used.
        - source_df (pd.DataFrame, optional): DataFrame to use for generating the summary. 
                                              If None, self.df is used.

        Returns:
        - pd.DataFrame: The generated summary table, or None if an error occurs.
        """
        df_to_process = source_df if source_df is not None else self.df

        if df_to_process is None or df_to_process.empty:
            log_msg = "DataFrame is not available or is empty. Cannot generate summary table."
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Generate Summary Table Failed", {"reason": log_msg, "groupby_cols": groupby_cols, "target_col": target_col, "filter_type": filter_type})
            return None

        print(f"[EEGAnalyzer - {self.analyzer_name}] Generating summary table for '{target_col}', grouped by {groupby_cols}, filter: {filter_type}...")

        try:
            summary_df = Statistics.calculate_descriptive_stats(df_to_process, target_col, groupby_cols)
            
            if output_filename_suffix is None:
                group_str = "_".join(groupby_cols).replace(" ", "")
                output_filename = f"summary_{group_str}_{filter_type}_{target_col}.csv"
            else:
                output_filename = f"summary_{output_filename_suffix}_{filter_type}_{target_col}.csv"
            
            filepath = os.path.join(self.derivatives_path, output_filename)
            summary_df.to_csv(filepath, index=False)
            
            log_msg = f"Summary table saved to {filepath}"
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Summary Table Generated", {
                "groupby_cols": groupby_cols,
                "target_col": target_col,
                "filter_type": filter_type,
                "output_file": filepath,
                "rows": len(summary_df),
                "message": log_msg
            })
            return summary_df
        except Exception as e:
            log_msg = f"Error generating summary table: {e}"
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Generate Summary Table Failed", {
                "groupby_cols": groupby_cols,
                "target_col": target_col,
                "filter_type": filter_type,
                "error": str(e),
                "message": log_msg
            })
            return None

    def save_analyzer(self, filename: str = "analyzer_state.pkl"):
        """
        Save the current state of the EEGAnalyzer instance using pickle.
        This includes the datasets, DataFrame, and other attributes.
        """
        filepath = os.path.join(self.derivatives_path, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            log_msg = f"EEGAnalyzer state saved to {filepath}"
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Analyzer State Saved", {"path": filepath, "message": log_msg})
        except Exception as e:
            log_msg = f"Error saving analyzer state: {e}"
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Save Analyzer State Failed", {"path": filepath, "error": str(e), "message": log_msg})

    @classmethod
    def load_analyzer(cls, analyzer_name: str, filename: str = "analyzer_state.pkl") -> "EEGAnalyzer":
        """
        Load an EEGAnalyzer instance from a saved state using pickle.
        
        Parameters:
        - analyzer_name (str): The name of the analyzer to load.
        - filename (str): The name of the pickle file (default: "analyzer_state.pkl").

        Returns:
        - EEGAnalyzer: The loaded EEGAnalyzer instance, or None if loading fails.
        """
        derivatives_path = os.path.join(EEGANALYZER_OBJECT_DERIVATIVES_PATH, analyzer_name)
        filepath = os.path.join(derivatives_path, filename)

        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    analyzer = pickle.load(f)
                print(f"[EEGAnalyzer - {analyzer_name}] EEGAnalyzer state loaded from {filepath}")
                # Ensure the loaded object knows its derivatives_path, in case it wasn't set correctly during unpickling
                # or if the path structure changed (though it shouldn't with this setup)
                analyzer.derivatives_path = derivatives_path 
                analyzer.analyzer_name = analyzer_name
                # Log the loading event on the loaded analyzer instance
                if not hasattr(analyzer, '_history'): # Ensure history is initialized if loading older object
                    analyzer._history = []
                analyzer._log_event("Analyzer State Loaded", {"path": filepath, "message": f"Successfully loaded state for '{analyzer_name}'."})
                return analyzer
            except Exception as e:
                print(f"[EEGAnalyzer - {analyzer_name}] Error loading analyzer state from {filepath}: {e}")
                return None
        else:
            print(f"[EEGAnalyzer - {analyzer_name}] Could not find saved analyzer state at {filepath}")
            return None







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
from eeg_analyzer.processor import Processor # Import the Processor class
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
        self.fitted_models = {} # Initialize storage for all fitted models (channel or ROI)


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

    def apply_zscore_flagging(self, group_cols: list, value_col: str, threshold: float = 3.0, zscore_col_name: str = None):
        """
        Applies Z-score based outlier flagging to self.df.
        Adds a Z-score column and updates the 'is_bad' column.
        """
        if self.df is None or self.df.empty:
            print(f"[EEGAnalyzer - {self.analyzer_name}] DataFrame is empty. Cannot apply Z-score flagging.")
            self._log_event("Z-score Flagging Skipped", {"reason": "DataFrame empty"})
            return

        if zscore_col_name is None:
            zscore_col_name = f"zscore_{value_col}"

        print(f"[EEGAnalyzer - {self.analyzer_name}] Applying Z-score flagging to '{value_col}' (threshold: {threshold})...")
        
        initial_bad_count = self.df['is_bad'].sum() if 'is_bad' in self.df.columns else 0
        
        processed_df = Processor.flag_outliers_zscore(
            self.df, 
            group_cols=group_cols, 
            value_col=value_col, 
            threshold=threshold,
            zscore_col_name=zscore_col_name
        )
        
        final_bad_count = processed_df['is_bad'].sum()
        newly_flagged = final_bad_count - initial_bad_count
        
        self.df = processed_df
        
        log_details = {
            "value_col": value_col,
            "group_cols": group_cols,
            "threshold": threshold,
            "zscore_col_name": zscore_col_name,
            "initial_bad_rows": initial_bad_count,
            "final_bad_rows": final_bad_count,
            "newly_flagged_rows": newly_flagged,
            "df_shape": self.df.shape
        }
        print(f"[EEGAnalyzer - {self.analyzer_name}] Z-score flagging applied. {newly_flagged} new rows flagged as bad.")
        self._log_event("Z-score Flagging Applied", log_details)

    def apply_iqr_flagging(self, group_cols: list, value_col: str, multiplier: float = 1.5):
        """
        Applies IQR based outlier flagging to self.df.
        Updates the 'is_bad' column.
        """
        if self.df is None or self.df.empty:
            print(f"[EEGAnalyzer - {self.analyzer_name}] DataFrame is empty. Cannot apply IQR flagging.")
            self._log_event("IQR Flagging Skipped", {"reason": "DataFrame empty"})
            return

        print(f"[EEGAnalyzer - {self.analyzer_name}] Applying IQR flagging to '{value_col}' (multiplier: {multiplier})...")
        
        initial_bad_count = self.df['is_bad'].sum() if 'is_bad' in self.df.columns else 0

        processed_df = Processor.flag_outliers_iqr(
            self.df, 
            group_cols=group_cols, 
            value_col=value_col, 
            multiplier=multiplier
        )
        
        final_bad_count = processed_df['is_bad'].sum()
        newly_flagged = final_bad_count - initial_bad_count

        self.df = processed_df

        log_details = {
            "value_col": value_col,
            "group_cols": group_cols,
            "multiplier": multiplier,
            "initial_bad_rows": initial_bad_count,
            "final_bad_rows": final_bad_count,
            "newly_flagged_rows": newly_flagged,
            "df_shape": self.df.shape
        }
        print(f"[EEGAnalyzer - {self.analyzer_name}] IQR flagging applied. {newly_flagged} new rows flagged as bad.")
        self._log_event("IQR Flagging Applied", log_details)


    def _get_df_for_summary(self, source_df: pd.DataFrame = None, exclude_bad_rows: bool = True):
        """Helper to get the DataFrame to be used for summary, optionally filtering bad rows."""
        df_to_process = source_df if source_df is not None else self.df

        if df_to_process is None or df_to_process.empty:
            return None

        if exclude_bad_rows and 'is_bad' in df_to_process.columns:
            original_rows = len(df_to_process)
            df_to_process = df_to_process[~df_to_process['is_bad']].copy()
            rows_after_exclusion = len(df_to_process)
            if original_rows != rows_after_exclusion:
                 print(f"[EEGAnalyzer - {self.analyzer_name}] Excluded {original_rows - rows_after_exclusion} rows where is_bad=True for summary.")
        return df_to_process

    def generate_summary_table(self, 
                               groupby_cols: list, 
                               target_col: str = 'band_power', 
                               filter_type: str = "processed",
                               output_filename_suffix: str = None,
                               source_df: pd.DataFrame = None,
                               exclude_bad_rows: bool = True,
                               state_col: str = 'state', 
                               perform_state_comparison: bool = True,
                               output_subfolder: str = None 
                               ) -> Union[pd.DataFrame, None]:
        """
        Generates a summary table with descriptive statistics.
        Can perform detailed state comparisons if perform_state_comparison is True.
        Assumes state_col contains 0 for 'OT' and 1 for 'MW' for state comparison.

        Parameters:
        - groupby_cols (list): List of columns to group by.
        - target_col (str): The column to calculate statistics on (default: 'band_power').
        - filter_type (str): A descriptor for the data state (e.g., "unfiltered", "zscore_flagged").
                             Used for logging and filename generation.
        - output_filename_suffix (str): Optional suffix for the output CSV file. If None,
                                        a default name based on groupby_cols and filter_type is used.
        - source_df (pd.DataFrame, optional): DataFrame to use for generating the summary. 
                                              If None, self.df is used.
        - exclude_bad_rows (bool): If True (default), rows where 'is_bad' is True are excluded 
                                   before calculating statistics.
        - state_col (str): Column name for states (default: 'state').
        - perform_state_comparison (bool): If True, generates detailed stats including state comparisons.
                                           If False, generates simpler group-wise statistics.
        - output_subfolder (str, optional): Subfolder within the analyzer's derivatives path to save the table.
                                            If None, saves directly in the derivatives path.

        Returns:
        - pd.DataFrame: The generated summary table, or None if an error occurs.
        """
        df_to_process = self._get_df_for_summary(source_df, exclude_bad_rows)

        if df_to_process is None or df_to_process.empty:
            log_msg = "DataFrame is not available or is empty (possibly after excluding bad rows). Cannot generate summary table."
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Generate Summary Table Failed", {"reason": log_msg, "groupby_cols": groupby_cols, "target_col": target_col, "filter_type": filter_type, "exclude_bad_rows": exclude_bad_rows})
            return None

        print(f"[EEGAnalyzer - {self.analyzer_name}] Generating summary table for '{target_col}', grouped by {groupby_cols}, data state: {filter_type}, excluding bad rows: {exclude_bad_rows}, state comparison: {perform_state_comparison}...")

        try:
            if perform_state_comparison:
                if state_col not in df_to_process.columns:
                    log_msg = f"State column '{state_col}' not found in DataFrame. Cannot perform state comparison."
                    print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
                    self._log_event("Generate Summary Table Failed", {"reason": log_msg, "state_col": state_col})
                    return None
                summary_df = Statistics.calculate_descriptive_stats_detailed(
                    df_to_process, 
                    value_col=target_col, 
                    group_cols=groupby_cols,
                    state_col=state_col
                )
            else:
                summary_df = Statistics.calculate_descriptive_stats(df_to_process, target_col, groupby_cols)
            
            clean_suffix = "_cleaned" if exclude_bad_rows and ('is_bad' in (source_df if source_df is not None else self.df).columns) else ""
            state_comp_suffix = "_statecomp" if perform_state_comparison else ""
            
            if output_filename_suffix is None:
                group_str = "_".join(groupby_cols).replace(" ", "")
                output_filename = f"summary_{group_str}_{filter_type}_{target_col}{clean_suffix}{state_comp_suffix}.csv"
            else:
                # Ensure output_filename_suffix doesn't lead to excessively long or redundant names if it already contains info
                output_filename = f"summary_{output_filename_suffix}_{filter_type}_{target_col}{clean_suffix}{state_comp_suffix}.csv"

            
            output_dir = self.derivatives_path
            if output_subfolder:
                output_dir = os.path.join(self.derivatives_path, output_subfolder)
                os.makedirs(output_dir, exist_ok=True) # Ensure subfolder exists
            
            filepath = os.path.join(output_dir, output_filename)
            summary_df.to_csv(filepath, index=False)
            
            log_msg = f"Summary table saved to {filepath}"
            print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
            self._log_event("Summary Table Generated", {
                "groupby_cols": groupby_cols,
                "target_col": target_col,
                "filter_type": filter_type,
                "exclude_bad_rows": exclude_bad_rows,
                "state_comparison": perform_state_comparison,
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
                "exclude_bad_rows": exclude_bad_rows,
                "state_comparison": perform_state_comparison,
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
                analyzer._log_event("Analyzer State Loaded", {
                    "path": filepath, 
                    "message": f"Successfully loaded state for '{analyzer_name}'.",
                    "loaded_by": os.path.abspath(__file__),  # Log the file that loaded the instance
                    "loaded_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Log the timestamp
                })
                return analyzer
            except Exception as e:
                print(f"[EEGAnalyzer - {analyzer_name}] Error loading analyzer state from {filepath}: {e}")
                return None
        else:
            print(f"[EEGAnalyzer - {analyzer_name}] Could not find saved analyzer state at {filepath}")
            return None
        
    def _get_data_slice_for_model(self, 
                                 dataset_name: str, 
                                 value_col: str, 
                                 state_col: str, 
                                 group_col: str, 
                                 exclude_bad_rows: bool = True,
                                 channel_name: str = None,
                                 roi_cortical_region: str = None,
                                 roi_hemisphere: str = None 
                                 ) -> pd.DataFrame:
        """
        Private helper to get a data slice for model fitting.
        Filters self.df for a specific dataset and channel, or dataset and ROI.
        Selects necessary columns and handles exclusion of bad rows.
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        # Start with dataset filter
        df_slice = self.df[self.df['dataset'] == dataset_name].copy()

        if channel_name:
            df_slice = df_slice[df_slice['channel'] == channel_name]
        elif roi_cortical_region:
            df_slice = df_slice[df_slice['cortical_region'] == roi_cortical_region]
            if roi_hemisphere and roi_hemisphere.lower() != "full":
                # Ensure case-insensitivity for hemisphere matching if needed,
                # assuming DataFrame 'hemisphere' column has consistent casing (e.g., 'Left', 'Right', 'Midline')
                df_slice = df_slice[df_slice['hemisphere'].str.lower() == roi_hemisphere.lower()]
        else:
            # Neither channel nor ROI specified, this shouldn't happen if called correctly
            print(f"[EEGAnalyzer - {self.analyzer_name}] _get_data_slice_for_model called without channel_name or roi_cortical_region.")
            return pd.DataFrame()


        if exclude_bad_rows and 'is_bad' in df_slice.columns:
            df_slice = df_slice[~df_slice['is_bad']]

        # Define required columns for the model itself, plus identifiers
        model_value_cols = [value_col, state_col, group_col]
        identifier_cols = ['dataset'] # Keep dataset for context
        if channel_name:
            identifier_cols.append('channel')
        if roi_cortical_region:
            identifier_cols.extend(['cortical_region', 'hemisphere']) # Keep these for verification of slice

        required_cols_for_slice = list(set(model_value_cols + identifier_cols))


        if not all(col in df_slice.columns for col in model_value_cols):
            missing_model_cols = [col for col in model_value_cols if col not in df_slice.columns]
            print(f"[EEGAnalyzer - {self.analyzer_name}] Missing one or more required model columns {missing_model_cols} for data slice for {dataset_name}.")
            return pd.DataFrame()
        
        # Ensure state_col is categorical if it's used as C(state) in formula
        if state_col in df_slice.columns:
             df_slice[state_col] = pd.Categorical(df_slice[state_col])

        # Return only the columns needed for modeling plus identifiers for clarity if debugging
        # The actual modeling function (Statistics.fit_mixedlm) will use formula to pick columns from this slice.
        return df_slice[list(set(model_value_cols + ['dataset', 'channel', 'cortical_region', 'hemisphere', 'subject_session']))].dropna(subset=model_value_cols)


    def fit_models_by_channel(self, 
                              formula: str, 
                              value_col: str = 'band_power', 
                              state_col: str = 'state', 
                              group_col: str = 'subject_session', 
                              re_formula: str = None, 
                              vc_formula: dict = None, # Corrected type to dict
                              exclude_bad_rows: bool = True):
        """
        Fits a mixed-effects model for each channel within each dataset using self.df.

        Parameters:
        - formula (str): The formula for the fixed effects (e.g., 'band_power ~ C(state)').
                         The value_col will be dynamically inserted if formula uses 'value_col_placeholder'.
        - value_col (str): The dependent variable for the model.
        - state_col (str): Column representing state, often used as a predictor.
        - group_col (str): Column for grouping random effects (e.g., 'subject_session').
        - re_formula (str, optional): Random effects formula part. If None, a simple random intercept on group_col is assumed.
        - vc_formula (dict, optional): Variance components formula for statsmodels.
        - exclude_bad_rows (bool): Whether to exclude rows marked as 'is_bad'.
        """
        if self.df is None or self.df.empty:
            print(f"[EEGAnalyzer - {self.analyzer_name}] DataFrame is empty. Cannot fit models.")
            self._log_event("Fit Models By Channel Skipped", {"reason": "DataFrame empty"})
            return

        if not hasattr(self, 'fitted_models'):
            # Initialize fitted_models if it doesn't exist (e.g. after loading older object)
            print(f"[EEGAnalyzer - {self.analyzer_name}] Initializing fitted_models dictionary.")
            self.fitted_models = {}
            self._log_event("Fitted Models Dictionary Initialized", {"message": "fitted_models dictionary created for storing model results."})
        
        log_msg_start = f"Starting model fitting for each dataset and channel. Formula: {formula}"
        print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg_start}")
        self._log_event("Fit Models By Channel Started", {
            "formula_template": formula, "value_col": value_col, "state_col": state_col, 
            "group_col": group_col, "re_formula": re_formula, "exclude_bad_rows": exclude_bad_rows
        })

        unique_datasets = self.df['dataset'].unique()
        total_models_to_fit = 0
        for ds_name in unique_datasets:
            total_models_to_fit += self.df[self.df['dataset'] == ds_name]['channel'].nunique()
        
        fitted_count = 0
        failed_count = 0

        for dataset_name in unique_datasets:
            if dataset_name not in self.fitted_models:
                self.fitted_models[dataset_name] = {}
            unique_channels_in_dataset = self.df[self.df['dataset'] == dataset_name]['channel'].unique()
            
            print(f"[EEGAnalyzer - {self.analyzer_name}] Processing dataset: {dataset_name} ({len(unique_channels_in_dataset)} channels)")

            for channel_name in unique_channels_in_dataset:
                current_formula = formula.replace("value_col_placeholder", value_col) # Allow dynamic value_col in formula

                data_slice = self._get_data_slice_for_model(
                    dataset_name=dataset_name, 
                    value_col=value_col, 
                    state_col=state_col, 
                    group_col=group_col, 
                    exclude_bad_rows=exclude_bad_rows,
                    channel_name=channel_name
                )

                if data_slice.empty or len(data_slice) < 10: # Basic check for sufficient data
                    log_msg = f"Skipping model for {dataset_name}-{channel_name}: Insufficient data after filtering (rows: {len(data_slice)})."
                    print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}")
                    self.fitted_models[dataset_name][channel_name] = None
                    self._log_event("Model Fit Skipped", {"dataset": dataset_name, "identifier": channel_name, "type": "channel", "reason": "Insufficient data", "rows": len(data_slice)})
                    failed_count +=1
                    continue
                
                # Default re_formula if not provided (random intercept for the group_col)
                current_re_formula = re_formula
                if current_re_formula is None:
                    current_re_formula = f"1" # Results in "1 | group_col" effectively if groups is specified in mixedlm

                # print(f"[EEGAnalyzer - {self.analyzer_name}] Fitting model for {dataset_name}-{channel_name}...")
                model_result = Statistics.fit_mixedlm(data_slice, 
                                                      current_formula, 
                                                      groups_col=group_col,
                                                      re_formula=current_re_formula,
                                                      vc_formula=vc_formula)
                
                self.fitted_models[dataset_name][channel_name] = model_result
                if model_result:
                    fitted_count += 1
                    self._log_event("Model Fit Successful", {"dataset": dataset_name, "identifier": channel_name, "type": "channel", "formula": current_formula})
                else:
                    failed_count += 1
                    self._log_event("Model Fit Failed", {"dataset": dataset_name, "identifier": channel_name, "type": "channel", "formula": current_formula})
            print(f"[EEGAnalyzer - {self.analyzer_name}] Finished dataset: {dataset_name}. Models fitted: {fitted_count}/{fitted_count+failed_count} (for this dataset so far for all datasets)")


        log_msg_end = f"Finished model fitting. Total models attempted: {total_models_to_fit}. Successful: {fitted_count}. Failed/Skipped: {failed_count}."
        print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg_end}")
        self._log_event("Fit Models By Channel Finished", {"total_attempted": total_models_to_fit, "successful": fitted_count, "failed_skipped": failed_count})

    def fit_models_by_roi(self,
                         formula: str,
                         value_col: str = 'band_power',
                         state_col: str = 'state',
                         group_col: str = 'subject_session',
                         re_formula: str = None,
                         vc_formula: dict = None,
                         exclude_bad_rows: bool = True):
        """
        Fits a mixed-effects model for each ROI within each dataset using self.df.
        ROIs are defined by cortical_region and hemisphere (Left, Right, Midline, Full).

        Parameters:
        - formula (str): The formula for the fixed effects (e.g., 'value_col_placeholder ~ C(state)').
        - value_col (str): The dependent variable for the model.
        - state_col (str): Column representing state, often used as a predictor.
        - group_col (str): Column for grouping random effects (e.g., 'subject_session').
        - re_formula (str, optional): Random effects formula part.
        - vc_formula (dict, optional): Variance components formula for statsmodels.
        - exclude_bad_rows (bool): Whether to exclude rows marked as 'is_bad'.
        """
        if self.df is None or self.df.empty:
            print(f"[EEGAnalyzer - {self.analyzer_name}] DataFrame is empty. Cannot fit models by ROI.")
            self._log_event("Fit Models By ROI Skipped", {"reason": "DataFrame empty"})
            return

        if not hasattr(self, 'fitted_models'):
            self.fitted_models = {}
            self._log_event("Fitted Models Dictionary Initialized", {})
        
        log_msg_start = f"Starting ROI model fitting. Formula: {formula}"
        print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg_start}")
        self._log_event("Fit Models By ROI Started", {
            "formula_template": formula, "value_col": value_col, "state_col": state_col,
            "group_col": group_col, "re_formula": re_formula, "exclude_bad_rows": exclude_bad_rows
        })

        unique_datasets = self.df['dataset'].unique()
        hemisphere_specs = ["Left", "Right", "Midline", "Full"]
        
        fitted_count = 0
        failed_count = 0
        total_rois_attempted = 0

        for dataset_name in unique_datasets:
            if dataset_name not in self.fitted_models:
                self.fitted_models[dataset_name] = {}
            dataset_df_slice = self.df[self.df['dataset'] == dataset_name]
            unique_cortical_regions_in_dataset = dataset_df_slice['cortical_region'].unique()
            
            print(f"[EEGAnalyzer - {self.analyzer_name}] Processing ROIs for dataset: {dataset_name}")

            for cortical_region in unique_cortical_regions_in_dataset:
                if pd.isna(cortical_region): continue # Skip if cortical_region is NaN

                for hemi_spec in hemisphere_specs:
                    roi_identifier = f"{cortical_region}_{hemi_spec.lower()}"
                    total_rois_attempted += 1

                    # Skip Midline spec for regions that don't typically have midline channels,
                    # or if no midline channels exist for that region in the data.
                    if hemi_spec.lower() == "midline":
                        has_midline_data = not dataset_df_slice[
                            (dataset_df_slice['cortical_region'] == cortical_region) &
                            (dataset_df_slice['hemisphere'].str.lower() == "midline")
                        ].empty
                        if not has_midline_data:
                            # print(f"[EEGAnalyzer - {self.analyzer_name}] Skipping {roi_identifier} for {dataset_name}: No midline data.")
                            failed_count +=1 # Count as skipped/failed attempt
                            continue
                    
                    current_formula = formula.replace("value_col_placeholder", value_col)
                    
                    data_slice = self._get_data_slice_for_model(
                        dataset_name=dataset_name,
                        value_col=value_col,
                        state_col=state_col,
                        group_col=group_col,
                        exclude_bad_rows=exclude_bad_rows,
                        roi_cortical_region=cortical_region,
                        roi_hemisphere=hemi_spec 
                    )

                    if data_slice.empty or len(data_slice) < 10: # Basic check
                        log_msg = f"Skipping model for {dataset_name}-{roi_identifier}: Insufficient data (rows: {len(data_slice)})."
                        # print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg}") # Can be verbose
                        self.fitted_models[dataset_name][roi_identifier] = None
                        self._log_event("Model Fit Skipped", {"dataset": dataset_name, "identifier": roi_identifier, "type": "ROI", "reason": "Insufficient data", "rows": len(data_slice)})
                        failed_count += 1
                        continue

                    current_re_formula = re_formula if re_formula is not None else "1"
                    
                    model_result = Statistics.fit_mixedlm(
                        data_slice,
                        current_formula,
                        groups_col=group_col,
                        re_formula=current_re_formula,
                        vc_formula=vc_formula
                    )
                    
                    self.fitted_models[dataset_name][roi_identifier] = model_result
                    if model_result:
                        fitted_count += 1
                        self._log_event("Model Fit Successful", {"dataset": dataset_name, "identifier": roi_identifier, "type": "ROI", "formula": current_formula})
                    else:
                        failed_count += 1
                        self._log_event("Model Fit Failed", {"dataset": dataset_name, "identifier": roi_identifier, "type": "ROI", "formula": current_formula})
            
            print(f"[EEGAnalyzer - {self.analyzer_name}] Finished ROIs for dataset: {dataset_name}.")

        log_msg_end = f"Finished ROI model fitting. Total ROIs attempted: {total_rois_attempted}. Successful: {fitted_count}. Failed/Skipped: {failed_count}."
        print(f"[EEGAnalyzer - {self.analyzer_name}] {log_msg_end}")
        self._log_event("Fit Models By ROI Finished", {"total_attempted": total_rois_attempted, "successful": fitted_count, "failed_skipped": failed_count})


    def get_fitted_model(self, dataset_name: str, model_identifier: str):
        """
        Retrieves a specific fitted model result (channel or ROI).
        
        Parameters:
        - dataset_name (str): The name of the dataset.
        - model_identifier (str): The identifier for the model (e.g., channel name or ROI string like 'frontal_left').
        """
        if not hasattr(self, 'fitted_models'):
            return None
        return self.fitted_models.get(dataset_name, {}).get(model_identifier, None)

    def get_all_fitted_models(self):
        """Returns the dictionary of all fitted models (channel and ROI based)."""
        if not hasattr(self, 'fitted_models'):
            return {}
        return self.fitted_models
    
    def get_significant_models(self, p_value_threshold: float = 0.05) -> dict:
        """
        Returns a dictionary of significant models based on p-value threshold.
        Significance is determined by the p-value of the second parameter in the model results,
        which is assumed to be the slope or main effect of interest (e.g., C(state)).
        
        Parameters:
        - p_value_threshold (float): The threshold for significance (default: 0.05).
        
        Returns:
        - dict: A dictionary where keys are dataset names and values are dictionaries of 
                significant models (key: model_identifier) with the name and p-value of the 
                second parameter, and the model result itself.
        """
        significant_models_summary = {}
        all_fitted_models = self.get_all_fitted_models()

        for dataset_name, models_in_dataset in all_fitted_models.items():
            significant_identifiers_in_dataset = {}
            for model_identifier, model_result in models_in_dataset.items():
                if model_result is not None and hasattr(model_result, 'pvalues'):
                    if len(model_result.pvalues) > 1:
                        # Assume the second parameter is the slope of interest
                        slope_param_name = model_result.pvalues.index[1]
                        slope_p_value = model_result.pvalues.iloc[1]

                        if slope_p_value < p_value_threshold:
                            significant_identifiers_in_dataset[model_identifier] = {
                                "model": model_result,
                                "slope_param_name": slope_param_name,
                                "slope_p_value": slope_p_value
                            }
                    else:
                        # Log or print if a model doesn't have at least two parameters
                        print(f"[EEGAnalyzer - {self.analyzer_name}] Model for {dataset_name}-{model_identifier} has fewer than 2 parameters. Cannot assess significance based on the second parameter.")
            
            if significant_identifiers_in_dataset:
                significant_models_summary[dataset_name] = significant_identifiers_in_dataset
                print(f"[EEGAnalyzer - {self.analyzer_name}] Found significant models in dataset '{dataset_name}' for identifiers: {', '.join(significant_identifiers_in_dataset.keys())} with p-value threshold {p_value_threshold} (based on the second model parameter).")
                self._log_event("Significant Models Found", {
                    "dataset": dataset_name, 
                    "identifiers": list(significant_identifiers_in_dataset.keys()), 
                    "p_value_threshold": p_value_threshold
                })
        return significant_models_summary

    def summarize_fitted_models(self, extract_info_func=None, save: bool = False, filename: str = "fitted_models_summary.csv") -> pd.DataFrame:
        """
        Summarizes information from all fitted models (channel and ROI) into a DataFrame.

        Parameters:
        - extract_info_func (callable, optional): A function that takes a fitted model result 
          (statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper) and returns a 
          dictionary of the information to extract. If None, extracts model parameters.
        - save (bool): If True, saves the summary DataFrame to a CSV file in the analyzer's
                       derivatives_path. Defaults to False.
        - filename (str): Filename to use if saving the summary. Defaults to "fitted_models_summary.csv".


        Returns:
        - pd.DataFrame: A DataFrame where each row corresponds to a model,
                        with columns for dataset, identifier (channel/ROI), and extracted model info.
        """
        all_fitted_models = self.get_all_fitted_models()
        if not all_fitted_models:
            print(f"[EEGAnalyzer - {self.analyzer_name}] No fitted models available to summarize.")
            self._log_event("Fitted Models Summarization Skipped", {"reason": "No fitted models"})
            return pd.DataFrame()

        if extract_info_func is None:
            def default_extract_info(model_result):
                if model_result is None:
                    return {"error": "Fitting failed or skipped"}
                try:
                    # Extract parameters and p-values
                    summary = {"converged": model_result.converged}
                    summary["n_observations"] = model_result.nobs
                    for param, value in model_result.params.items():
                        summary[f"param_{param}"] = value
                    for param, p_value in model_result.pvalues.items():
                        summary[f"pvalue_{param}"] = p_value
                    summary["loglike"] = model_result.llf
                    summary["aic"] = model_result.aic
                    summary["bic"] = model_result.bic
                    return summary
                except Exception as e:
                    return {"error": str(e)}
            extract_info_func = default_extract_info
        
        summary_list = []
        for dataset_name, models_in_dataset in all_fitted_models.items():
            for model_identifier, model_result in models_in_dataset.items():
                model_info = {"dataset": dataset_name, "identifier": model_identifier}
                extracted = extract_info_func(model_result)
                model_info.update(extracted)
                summary_list.append(model_info)
        
        summary_df = pd.DataFrame(summary_list)
        # Log this action
        log_details = {"num_models_in_summary": len(summary_df)}
        
        if save and not summary_df.empty:
            try:
                filepath = os.path.join(self.derivatives_path, filename)
                summary_df.to_csv(filepath, index=False)
                save_msg = f"Fitted models summary saved to {filepath}"
                print(f"[EEGAnalyzer - {self.analyzer_name}] {save_msg}")
                log_details["saved_to"] = filepath
                log_details["save_status"] = "Success"
            except Exception as e:
                save_err_msg = f"Error saving fitted models summary: {e}"
                print(f"[EEGAnalyzer - {self.analyzer_name}] {save_err_msg}")
                log_details["save_error"] = str(e)
                log_details["save_status"] = "Failed"
        elif save and summary_df.empty:
            print(f"[EEGAnalyzer - {self.analyzer_name}] Fitted models summary is empty. Nothing to save.")
            log_details["save_status"] = "Skipped (empty summary)"

        self._log_event("Fitted Models Summarized", log_details)
        return summary_df


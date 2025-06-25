"""
Recording
---------

Represents a single EEG recording session for a subject.

Responsibilities:
- Store preprocessed PSD data (e.g., per task and state).
- Provide methods to access PSDs or derived features like alpha power.


Notes:
- Even if a subject has only one recording, this abstraction keeps the structure consistent.
- Can hold references to metadata like task labels or state mappings.
"""

from collections import defaultdict
from typing import Optional, Tuple, Any, Iterator, Iterable
import numpy as np
import os
import seaborn as sns
import pandas as pd

from eeg_analyzer.metrics import Metrics
from eeg_analyzer.band_power_stats import BandPowerStats
from utils.config import EEG_SETTINGS, set_plot_style, PLOTS_PATH, OUTLIER_DETECTION, QUALITY_CONTROL
import mne
import matplotlib.pyplot as plt

set_plot_style()  # Set the plotting style for MNE and Matplotlib
class Recording:
    def __init__(self, session_id: int, psd_entries: list[np.ndarray], metadata_entries: list[dict], freq_entries: list[np.ndarray], channels: list[str], band: Optional[tuple[float, float]] = None):
        self.session_id = session_id
        self.psd_map = defaultdict(dict)     # task -> state -> PSD (epochs, channels, freqs)
        self.meta_map = defaultdict(dict)    # task -> state -> metadata
        self.freq_map = defaultdict(dict)    # task -> state -> frequencies
        self.band_power_map = defaultdict(dict) # task -> state -> band_power (epochs, channels)
        self.log_band_power_map = defaultdict(dict) # task -> state -> log band_power (epochs, channels)
        self.z_band_power_map = defaultdict(dict) # task -> state -> z-scored band_power (epochs, channels)
        self.outlier_mask_map = defaultdict(dict) # task -> state -> outlier mask (epochs, channels)       
        self.channels = channels             # List of channel names
        self.band_power_stats: Optional[BandPowerStats] = None  # Statistics calculator
        self.exclude: bool = False          # Flag to exclude session from analysis

        for psd, meta, freqs in zip(psd_entries, metadata_entries, freq_entries):
            task = meta["task"]
            state = meta["state"]
            self.psd_map[task][state] = psd
            self.meta_map[task][state] = meta
            self.freq_map[task][state] = freqs

        if band:
            self.calculate_band_power(band)

        # Automatic quality control, outlier filtering, and normalization during initialization
        self.update_exclude_flag()
        self.apply_outlier_filtering()
        self.normalize()

    def __repr__(self):
        total_conditions = sum(len(states) for states in self.psd_map.values())
        return f"<Recording session-{self.session_id} with {total_conditions} condition(s)>"
    
    def __str__(self):
        return f"Recording session-{self.session_id} with {len(self.channels)} channels and {len(self.psd_map)} tasks"


    #                                           Public API
    ##########################################################################################################
    
    def calculate_band_power(self, band: tuple[float, float] = (8, 12)):
        """
        Calculate power for a given frequency band for all conditions.
        The result is stored in `self.band_power_map`.
        Also calculates log-transformed band power, initializes an outlier mask,
        and computes comprehensive statistics.

        Parameters:
        - band: tuple (low_freq, high_freq) specifying the frequency band in Hz.
        """
        self.band_range = band
        self.band_power_map.clear()  # Clear previous calculations
        self.log_band_power_map.clear()
        self.z_band_power_map.clear()
        self.outlier_mask_map.clear()
        
        for task, states in self.psd_map.items():
            for state, psd in states.items():
                freqs = self.get_freqs(task, state)
                # The result of Metrics.band_power has shape (n_epochs, n_channels)
                band_power = Metrics.band_power(psd, freqs, band)
                self.band_power_map[task][state] = band_power
                self.log_band_power_map[task][state] = Metrics.log_transform(band_power)
                self.outlier_mask_map[task][state] = np.ones_like(band_power, dtype=bool)
        
        # Calculate comprehensive statistics
        self.band_power_stats = BandPowerStats(self.channels)
        self.band_power_stats.calculate_all_stats(
            self.band_power_map, 
            self.log_band_power_map, 
            self.outlier_mask_map
        )

    def normalize(self):
        """
        Normalize the filtered (no outliers) log-transformed band power data using z-score normalization.
        Normalization is done per channel across all epochs. Outliers are set to np.nan.
        """
        if self.band_power_stats is None:
            raise ValueError("Band power statistics have not been calculated. Call calculate_band_power() first.")

        for task, states in self.log_band_power_map.items():
            for state, log_power in states.items():
                mask = self.outlier_mask_map[task][state]
                z_scores = np.full_like(log_power, np.nan, dtype=np.float64)  # same shape, fill with nan

                for ch_idx, channel in enumerate(self.channels):
                    mean = self.get_stat('mean', channel, data_type='log_band_power', filtered=True)
                    std = np.sqrt(self.get_stat('variance', channel, data_type='log_band_power', filtered=True))
                    # Avoid division by zero
                    if np.isnan(mean) or np.isnan(std) or std == 0:
                        continue
                    # Only normalize non-outlier values
                    valid_idx = mask[:, ch_idx]
                    z_scores[valid_idx, ch_idx] = (log_power[valid_idx, ch_idx] - mean) / std

                self.z_band_power_map[task][state] = z_scores
        
        # Recalculate statistics after normalization
        self.band_power_stats._calculate_stats_for_data_type(
            data_type='z_band_power',
            data_map=self.z_band_power_map,
            outlier_mask_map=None # No outliers in z_band_power, so mask is not needed
        )


    def recalculate_stats(self):
        """
        Recalculate statistics after outlier mask changes.
        Call this method after modifying outlier masks.
        """
        if self.band_power_stats is None:
            raise ValueError("Band power has not been calculated. Call calculate_band_power() first.")
        
        self.band_power_stats.calculate_all_stats(
            self.band_power_map,
            self.log_band_power_map,
            self.outlier_mask_map
        )

    def get_stat(self, stat_name: str, channel: Optional[str] = None, data_type: str = 'log_band_power',
                 task: Optional[str] = None, state: Optional[str] = None, 
                 filtered: bool = False) -> Any:
        """
        Unified method to get any statistic using the BandPowerStats class.
        
        Parameters:
        - stat_name: Name of the statistic ('mean', 'variance', 'median', etc.)
        - channel: Channel name (if None, returns an array of values for all channels)
        - data_type: 'band_power' or 'log_band_power'
        - task: Task name (if None, uses state-level or all_data stats)
        - state: State name (if None, uses all_data stats)
        - filtered: Whether to use filtered (outlier-removed) data
        
        Returns:
        - The requested statistic value or an array of values for all channels
        """
        if self.band_power_stats is None:
            raise ValueError("Band power statistics have not been calculated. Call calculate_band_power() first.")

        if channel is None:
            # Return an array of values for all channels
            return np.array([
                self.band_power_stats.get_stat(stat_name, ch, data_type, task, state, filtered)
                for ch in self.channels
            ])

        return self.band_power_stats.get_stat(stat_name, channel, data_type, task, state, filtered)

    def calculate_state_ratio(self, filtered: bool = False) -> float:
        """
        Calculate the state ratio for quality control assessment.
        
        The state ratio is defined as:
        r = min(n_MW, n_OT) / max(n_MW, n_OT)
        
        where n_MW and n_OT are the total number of valid epochs for 
        mind-wandering and on-task states respectively.
        
        Parameters:
        - filtered: If True, uses outlier-filtered data. If False, uses all data.
        
        Returns:
        - float: State ratio between 0 and 1
        """
        n_mw = 0
        n_ot = 0
        
        for task, states in self.psd_map.items():
            for state, psd in states.items():
                if filtered and self.outlier_mask_map:
                    # Count valid epochs after filtering
                    mask = self.outlier_mask_map[task][state]
                    valid_epochs = np.sum(np.any(mask, axis=1))  # At least one valid channel per epoch
                else:                    # Count all epochs
                    valid_epochs = psd.shape[0]
                
                if state == QUALITY_CONTROL["MW_OT_STATES"][0]:  # "MW"
                    n_mw += valid_epochs
                elif state == QUALITY_CONTROL["MW_OT_STATES"][1]:  # "OT"
                    n_ot += valid_epochs
        
        if n_mw == 0 and n_ot == 0:
            return 0.0
        elif n_mw == 0 or n_ot == 0:
            return 0.0
        else:
            return min(n_mw, n_ot) / max(n_mw, n_ot)

    def check_state_imbalance(self, threshold: float = None, filtered: bool = False) -> bool:
        """
        Check if the session has state imbalance based on the state ratio.
        
        Parameters:
        - threshold: Minimum ratio threshold (uses config default if None)
        - filtered: If True, uses outlier-filtered data. If False, uses all data.
        
        Returns:
        - bool: True if state_imbalance (ratio < threshold), False otherwise
        """
        if threshold is None:
            threshold = QUALITY_CONTROL["STATE_RATIO_THRESHOLD"]
        
        ratio = self.calculate_state_ratio(filtered=filtered)
        return ratio < threshold

    def get_epoch_counts_by_state(self, filtered: bool = False) -> dict:
        """
        Get the number of valid epochs for each state.
        
        Parameters:
        - filtered: If True, uses outlier-filtered data. If False, uses all data.
        
        Returns:
        - dict: Dictionary with state names as keys and epoch counts as values
        """
        state_counts = {state: 0 for state in QUALITY_CONTROL["MW_OT_STATES"]}
        
        for task, states in self.psd_map.items():
            for state, psd in states.items():
                if filtered and self.outlier_mask_map:
                    # Count valid epochs after filtering
                    mask = self.outlier_mask_map[task][state]
                    valid_epochs = np.sum(np.any(mask, axis=1))  # At least one valid channel per epoch
                else:
                    # Count all epochs
                    valid_epochs = psd.shape[0]
                
                if state in state_counts:
                    state_counts[state] += valid_epochs
        
        return state_counts

    def check_minimum_epochs(self, min_epochs_per_state: int = None, filtered: bool = False) -> bool:
        """
        Check if each state has the minimum required number of epochs.
        
        Parameters:
        - min_epochs_per_state: Minimum number of epochs required per state (uses config default if None)
        - filtered: If True, uses outlier-filtered data. If False, uses all data.
        
        Returns:
        - bool: True if minimum epoch requirement is met for all states, False otherwise
        """
        if min_epochs_per_state is None:
            min_epochs_per_state = QUALITY_CONTROL["MIN_EPOCHS_PER_STATE"]
        
        state_counts = self.get_epoch_counts_by_state(filtered=filtered)
          # Check if both MW and OT states have minimum epochs
        mw_state, ot_state = QUALITY_CONTROL["MW_OT_STATES"]
        return (state_counts.get(mw_state, 0) >= min_epochs_per_state and 
                state_counts.get(ot_state, 0) >= min_epochs_per_state)

    def get_quality_control_summary(self, ratio_threshold: float = None, 
                                   min_epochs_per_state: int = None) -> dict:
        """
        Get a comprehensive quality control summary for the session.
        
        Parameters:
        - ratio_threshold: Minimum ratio threshold for state balance (uses config default if None)
        - min_epochs_per_state: Minimum epochs required per state (uses config default if None)
        
        Returns:
        - dict: Quality control summary with metrics for both filtered and unfiltered data
        """
        if ratio_threshold is None:
            ratio_threshold = QUALITY_CONTROL["STATE_RATIO_THRESHOLD"]
        if min_epochs_per_state is None:
            min_epochs_per_state = QUALITY_CONTROL["MIN_EPOCHS_PER_STATE"]
        
        summary = {}
        
        for filtered in [False, True]:
            filter_key = "filtered" if filtered else "unfiltered"
            
            state_counts = self.get_epoch_counts_by_state(filtered=filtered)
            state_ratio = self.calculate_state_ratio(filtered=filtered)
            state_imbalance = self.check_state_imbalance(ratio_threshold, filtered=filtered)
            meets_min_epochs = self.check_minimum_epochs(min_epochs_per_state, filtered=filtered)
            
            summary[filter_key] = {
                "epoch_counts": state_counts,
                "state_ratio": state_ratio,
                "state_imbalance": state_imbalance,
                "meets_minimum_epochs": meets_min_epochs,                "passes_quality_control": not state_imbalance and meets_min_epochs
            }
        
        return summary

    def update_exclude_flag(self, ratio_threshold: float = None,
                           min_epochs_per_state: int = None, 
                           use_filtered: bool = False) -> bool:
        """
        Update the exclude flag based on quality control criteria.
        
        Parameters:
        - ratio_threshold: Minimum ratio threshold for state balance (uses config default if None)
        - min_epochs_per_state: Minimum epochs required per state (uses config default if None)
        - use_filtered: Whether to use filtered data for the assessment
        
        Returns:
        - bool: The updated exclude flag value
        """
        if ratio_threshold is None:
            ratio_threshold = QUALITY_CONTROL["STATE_RATIO_THRESHOLD"]
        if min_epochs_per_state is None:
            min_epochs_per_state = QUALITY_CONTROL["MIN_EPOCHS_PER_STATE"]
        
        qc_summary = self.get_quality_control_summary(ratio_threshold, min_epochs_per_state)
        data_key = "filtered" if use_filtered else "unfiltered"        
        # Exclude if quality control fails
        self.exclude = not qc_summary[data_key]["passes_quality_control"]
        
        return self.exclude

    def apply_outlier_filtering(self, M: int = None, K: float = None, S: float = None) -> bool:
        """
        Apply outlier filtering to log-band power data using global IQR and skewness statistics.
        Updates the outlier mask per channel, task, and state.

        Parameters:
            M (int): Minimum number of valid epochs required to apply filtering
            K (float): IQR multiplier to define outlier bounds
            S (float): Skewness threshold to switch to one-sided filtering

        Returns:
            bool: True if any outliers were newly detected, False otherwise.
        """
        if self.band_power_stats is None:
            raise ValueError("Log band power statistics have not been calculated. Call calculate_log_band_power() first.")
        if self.log_band_power_map is None:
            raise ValueError("Log band power has not been calculated. Call calculate_band_power() first.")

        # Use defaults if not provided
        M = M or OUTLIER_DETECTION["MIN_EPOCHS_FOR_FILTERING"]
        K = K or OUTLIER_DETECTION["IQR_MULTIPLIER"]
        S = S or OUTLIER_DETECTION["SKEWNESS_THRESHOLD"]

        data_map = self.log_band_power_map
        outliers_detected = False

        for task, state_dict in data_map.items():
            for state, data in state_dict.items():
                mask = self.outlier_mask_map[task][state]
                for ch_idx, channel in enumerate(self.channels):
                    x = data[:, ch_idx]
                    m = mask[:, ch_idx]
                    valid_x = x[m]

                    if len(valid_x) < M:
                        continue  # Skip channel if not enough valid data

                    median = self.get_stat('median', channel, 'log_band_power')
                    iqr = self.get_stat('iqr', channel, 'log_band_power')
                    skewness = self.get_stat('skewness', channel, 'log_band_power')

                    # Determine outlier bounds
                    if abs(skewness) > S:
                        if skewness > 0:
                            lower, upper = -np.inf, median + K * iqr
                        else:
                            lower, upper = median - K * iqr, np.inf
                    else:
                        lower, upper = median - K * iqr, median + K * iqr

                    new_channel_mask = m & (x >= lower) & (x <= upper)

                    if not np.array_equal(new_channel_mask, m):
                        mask[:, ch_idx] = new_channel_mask
                        outliers_detected = True

        if outliers_detected:
            self.recalculate_stats()

        return outliers_detected

    def detect_suspicious_distributions(self) -> dict:
        """
        Detect channels with suspicious statistical distributions.
        
        Returns:
        - dict: Dictionary with flags for suspicious distributions by data type and level
        """
        if self.band_power_stats is None:
            raise ValueError("Band power statistics have not been calculated. Call calculate_band_power() first.")
        
        suspicious_flags = {
            'band_power': {
                'all_data': {},
                'by_condition': {},
                'by_state': {}
            },
            'log_band_power': {
                'all_data': {},
                'by_condition': {},
                'by_state': {}
            }
        }
        
        # Thresholds from config
        skew_threshold = OUTLIER_DETECTION["SUSPICIOUS_SKEWNESS_THRESHOLD"]
        kurt_threshold = OUTLIER_DETECTION["SUSPICIOUS_KURTOSIS_THRESHOLD"]
        min_epochs = OUTLIER_DETECTION["MIN_EPOCH_COUNT_THRESHOLD"]
        p_value_threshold = OUTLIER_DETECTION["NORMALITY_P_VALUE_THRESHOLD"]
        
        for data_type in ['band_power', 'log_band_power']:
            # Check all_data level
            for channel in self.channels:
                flags = []
                
                # Get unfiltered stats for all data
                try:
                    skewness = self.get_stat('skewness', channel, data_type, filtered=False)
                    kurtosis = self.get_stat('kurtosis', channel, data_type, filtered=False)
                    epoch_count = self.get_stat('epoch_count', channel, data_type, filtered=False)
                    is_normal = self.get_stat('is_normal', channel, data_type, filtered=False)
                    normality_p = self.get_stat('normality_p_value', channel, data_type, filtered=False)
                    
                    # Check for suspicious characteristics
                    if abs(skewness) > skew_threshold:
                        flags.append('high_skewness')
                    if abs(kurtosis) > kurt_threshold:
                        flags.append('high_kurtosis')
                    if epoch_count < min_epochs:
                        flags.append('low_epoch_count')
                    if not is_normal and normality_p < p_value_threshold:
                        flags.append('non_normal')
                    
                except (KeyError, ValueError):
                    flags.append('missing_data')
                
                suspicious_flags[data_type]['all_data'][channel] = flags
              # Check by_state level
            for state in QUALITY_CONTROL["MW_OT_STATES"]:
                for channel in self.channels:
                    flags = []
                    
                    try:
                        skewness = self.get_stat('skewness', channel, data_type, state=state, filtered=False)
                        kurtosis = self.get_stat('kurtosis', channel, data_type, state=state, filtered=False)
                        epoch_count = self.get_stat('epoch_count', channel, data_type, state=state, filtered=False)
                        is_normal = self.get_stat('is_normal', channel, data_type, state=state, filtered=False)
                        normality_p = self.get_stat('normality_p_value', channel, data_type, state=state, filtered=False)
                        
                        if abs(skewness) > skew_threshold:
                            flags.append('high_skewness')
                        if abs(kurtosis) > kurt_threshold:
                            flags.append('high_kurtosis')
                        if epoch_count < min_epochs:
                            flags.append('low_epoch_count')
                        if not is_normal and normality_p < p_value_threshold:
                            flags.append('non_normal')
                        
                    except (KeyError, ValueError):
                        flags.append('missing_data')
                    
                    if state not in suspicious_flags[data_type]['by_state']:
                        suspicious_flags[data_type]['by_state'][state] = {}
                    suspicious_flags[data_type]['by_state'][state][channel] = flags
            
            # Check by_condition level
            for task, states in self.psd_map.items():
                for state in states.keys():
                    condition_key = (task, state)
                    
                    for channel in self.channels:
                        flags = []
                        
                        try:
                            skewness = self.get_stat('skewness', channel, data_type, task, state, filtered=False)
                            kurtosis = self.get_stat('kurtosis', channel, data_type, task, state, filtered=False)
                            epoch_count = self.get_stat('epoch_count', channel, data_type, task, state, filtered=False)
                            is_normal = self.get_stat('is_normal', channel, data_type, task, state, filtered=False)
                            normality_p = self.get_stat('normality_p_value', channel, data_type, task, state, filtered=False)
                            
                            if abs(skewness) > skew_threshold:
                                flags.append('high_skewness')
                            if abs(kurtosis) > kurt_threshold:
                                flags.append('high_kurtosis')
                            if epoch_count < min_epochs:
                                flags.append('low_epoch_count')
                            if not is_normal and normality_p < p_value_threshold:
                                flags.append('non_normal')
                        
                        except (KeyError, ValueError):
                            flags.append('missing_data')
                        
                        if condition_key not in suspicious_flags[data_type]['by_condition']:
                            suspicious_flags[data_type]['by_condition'][condition_key] = {}
                        suspicious_flags[data_type]['by_condition'][condition_key][channel] = flags
        
        return suspicious_flags

    def has_outliers_after_filtering(self) -> bool:
        """
        Quick check to see if any outliers were detected after filtering.
        
        Returns:
        - bool: True if any data points are marked as outliers (False in mask), False otherwise
        """
        if not self.outlier_mask_map:
            return False
        
        for task, states in self.outlier_mask_map.items():
            for state, mask in states.items():
                # If any value in the mask is False, outliers were detected
                if not np.all(mask):
                    return True
        
        return False

    def get_outlier_summary(self) -> dict:
        """
        Get a summary of outliers detected across all conditions.
        Reports the number of band power values (channel-epoch pairs) marked as outliers.
        Returns:
        - dict: Summary of outlier detection results
        """
        summary = {
            'has_outliers': self.has_outliers_after_filtering(),
            'outlier_counts': {},  # Number of band power values (epochs * channels) marked as outliers
            'outlier_percentages': {},
            'total_band_power_values_removed': 0,
            'total_band_power_values': 0
        }
        for task, states in self.outlier_mask_map.items():
            for state, mask in states.items():
                condition_key = f"{task}_{state}"
                total_points = mask.size  # total band power values (epochs * channels)
                outliers = np.sum(~mask)  # Count False values (outliers)
                valid_points = np.sum(mask)  # Count True values
                summary['outlier_counts'][condition_key] = outliers
                summary['outlier_percentages'][condition_key] = (outliers / total_points * 100) if total_points > 0 else 0
                summary['total_band_power_values_removed'] += outliers
                summary['total_band_power_values'] += total_points
                
        # Overall percentage
        if summary['total_band_power_values'] > 0:
            summary['overall_outlier_percentage'] = (summary['total_band_power_values_removed'] / summary['total_band_power_values'] * 100)
        else:
            summary['overall_outlier_percentage'] = 0
        return summary

    def get_band_power(self, task: Optional[str] = None, state: Optional[str] = None) -> np.ndarray:
        """
        Get pre-calculated band power, with flexible filtering.

        Can filter by task and/or state. If a parameter is None, data is concatenated
        across that dimension. If both are None, all data is returned.
        The returned array has shape (n_epochs, n_channels).

        Raises:
            ValueError: If band power has not been calculated, or if a specific
                        task-state pair is requested but does not exist.
        """
        if not self.band_power_map:
            raise ValueError("Band power has not been calculated. Call `calculate_band_power()` first.")

        # Specific lookup if both task and state are provided
        if task is not None and state is not None:
            try:
                return self.band_power_map[task][state]
            except KeyError:
                raise ValueError(f"Band power not calculated for task '{task}' and state '{state}'. "
                                 f"Call `calculate_band_power()` first.")

        # Concatenation logic for flexible filtering
        data_to_concatenate = []
        conditions_to_check = self.list_conditions()

        if task is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if t == task]
        if state is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if s == state]

        for t, s in conditions_to_check:
            if t in self.band_power_map and s in self.band_power_map[t]:
                data_to_concatenate.append(self.band_power_map[t][s])

        if not data_to_concatenate:
            return np.empty((0, len(self.channels)), dtype=np.float64)

        return np.concatenate(data_to_concatenate, axis=0)

    def get_log_band_power(self, task: Optional[str] = None, state: Optional[str] = None) -> np.ndarray:
        """
        Get pre-calculated log-transformed band power, with flexible filtering.

        Can filter by task and/or state. If a parameter is None, data is concatenated
        across that dimension. If both are None, all data is returned.
        The returned array has shape (n_epochs, n_channels).

        Raises:
            ValueError: If band power has not been calculated, or if a specific
                        task-state pair is requested but does not exist.
        """
        if not self.log_band_power_map:
            raise ValueError("Log band power has not been calculated. Call `calculate_band_power()` first.")

        # Specific lookup if both task and state are provided
        if task is not None and state is not None:
            try:
                return self.log_band_power_map[task][state]
            except KeyError:
                raise ValueError(f"Log band power not calculated for task '{task}' and state '{state}'. "
                                 f"Call `calculate_band_power()` first.")

        # Concatenation logic for flexible filtering
        data_to_concatenate = []
        conditions_to_check = self.list_conditions()

        if task is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if t == task]
        if state is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if s == state]

        for t, s in conditions_to_check:
            if t in self.log_band_power_map and s in self.log_band_power_map[t]:
                data_to_concatenate.append(self.log_band_power_map[t][s])

        if not data_to_concatenate:
            return np.empty((0, len(self.channels)), dtype=np.float64)

        return np.concatenate(data_to_concatenate, axis=0)

    def get_outlier_mask(self, task: Optional[str] = None, state: Optional[str] = None) -> np.ndarray:
        """
        Get the outlier mask for band power data, with flexible filtering.

        Can filter by task and/or state. If a parameter is None, the mask is concatenated
        across that dimension. If both are None, the full mask is returned.
        The returned array has shape (n_epochs, n_channels).

        Raises:
            ValueError: If the outlier mask is not available, or if a specific
                        task-state pair is requested but does not exist.
        """
        if not self.outlier_mask_map:
            raise ValueError("Outlier mask not available. Call `calculate_band_power()` first.")

        # Specific lookup if both task and state are provided
        if task is not None and state is not None:
            try:
                return self.outlier_mask_map[task][state]
            except KeyError:
                raise ValueError(f"Outlier mask not available for task '{task}' and state '{state}'. "
                                 f"Call `calculate_band_power()` first.")

        # Concatenation logic for flexible filtering
        data_to_concatenate = []
        conditions_to_check = self.list_conditions()

        if task is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if t == task]
        if state is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if s == state]

        for t, s in conditions_to_check:
            if t in self.outlier_mask_map and s in self.outlier_mask_map[t]:
                data_to_concatenate.append(self.outlier_mask_map[t][s])

        if not data_to_concatenate:
            return np.empty((0, len(self.channels)), dtype=bool)

        return np.concatenate(data_to_concatenate, axis=0)
    
    def get_z_band_power(self, task: Optional[str] = None, state: Optional[str] = None) -> np.ndarray:
        """
        Get pre-calculated z-scored band power, with flexible filtering.

        Can filter by task and/or state. If a parameter is None, data is concatenated
        across that dimension. If both are None, all data is returned.
        The returned array has shape (n_epochs, n_channels).

        Raises:
            ValueError: If z-scored band power has not been calculated, or if a specific
                        task-state pair is requested but does not exist.
        """
        if not self.z_band_power_map:
            raise ValueError("Z-scored band power has not been calculated. Call `calculate_band_power()` first.")

        # Specific lookup if both task and state are provided
        if task is not None and state is not None:
            try:
                return self.z_band_power_map[task][state]
            except KeyError:
                raise ValueError(f"Z-scored band power not calculated for task '{task}' and state '{state}'. "
                                 f"Call `calculate_band_power()` first.")

        # Concatenation logic for flexible filtering
        data_to_concatenate = []
        conditions_to_check = self.list_conditions()

        if task is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if t == task]
        if state is not None:
            conditions_to_check = [(t, s) for t, s in conditions_to_check if s == state]

        for t, s in conditions_to_check:
            if t in self.z_band_power_map and s in self.z_band_power_map[t]:
                data_to_concatenate.append(self.z_band_power_map[t][s])

        if not data_to_concatenate:
            return np.empty((0, len(self.channels)), dtype=np.float64)

        return np.concatenate(data_to_concatenate, axis=0)

    def set_outlier_mask(self, task: str, state: str, mask: np.ndarray):
        """
        Set the outlier mask for a given task and state.
        
        Parameters:
        - task: Task name
        - state: State name  
        - mask: Boolean array of shape (epochs, channels). True = valid data, False = outlier
        """
        if (task, state) not in [(t, s) for t, states in self.band_power_map.items() for s in states]:
            raise ValueError(f"No band power data for task '{task}' and state '{state}'.")
        
        expected_shape = self.band_power_map[task][state].shape
        if mask.shape != expected_shape:
            raise ValueError(f"Mask shape {mask.shape} does not match data shape {expected_shape}")
        
        self.outlier_mask_map[task][state] = mask

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
        
    def get_channel_names(self):
        """Return the list of channel names."""
        return self.channels

    def get_available_tasks(self):
        return list(self.psd_map.keys())

    def get_available_states(self, task: str):
        return list(self.psd_map[task].keys()) if task in self.psd_map else []
    
    def get_num_epochs(self):
        """Return a dictionary with number of epochs for each task and state."""
        num_epochs = {}
        for task, states in self.psd_map.items():
            for state, psd in states.items():
                num_epochs[(task, state)] = psd.shape[0]
        return num_epochs

    def list_conditions(self) -> list[tuple[str, str]]:
        """List all available (task, state) condition pairs."""
        return [
            (task, state)
            for task, states in self.psd_map.items()
            for state in states
        ]

    def get_mw_ot_epoch_ratio(self) -> float:
        """
        Calculate the ratio of total MW epochs to total OT epochs across all tasks.
        Returns np.inf if OT epochs count is zero.
        """
        total_mw_epochs = 0
        total_ot_epochs = 0
        
        for task, states_data in self.psd_map.items():
            for state, psd_array in states_data.items():
                num_epochs_for_condition = psd_array.shape[0]
                if state == QUALITY_CONTROL["MW_OT_STATES"][0]:  # "MW"
                    total_mw_epochs += num_epochs_for_condition
                elif state == QUALITY_CONTROL["MW_OT_STATES"][1]:  # "OT"
                    total_ot_epochs += num_epochs_for_condition
        
        if total_ot_epochs == 0:
            return np.inf  # Avoid division by zero
        
        return total_mw_epochs / total_ot_epochs
    
    def alpha_power(self, task: str, state: str) -> np.ndarray:
        """Compute alpha power for a given task and state using pre-calculated band power if available."""
        if self.band_range == (8, 12) and (task, state) in [(t, s) for t, states in self.band_power_map.items() for s in states]:
            # Use pre-calculated alpha band power
            return self.get_band_power(task, state)
        else:
            # Fall back to direct calculation
            psd = self.get_psd(task, state)
            freqs = self.get_freqs(task, state)
            return Metrics.alpha_power(psd, freqs)
    
    def mean_alpha_power_per_channel(self, task: str, state: str) -> np.ndarray:
        """Compute mean alpha power per channel for a given task and state using pre-calculated data."""
        alpha_power = self.alpha_power(task, state)
        return alpha_power.mean(axis=0)
    
    def stats_band_power_per_channel(self, task: str, state: str, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean, variance, and standard error of band power for a given task and state.
        Uses pre-calculated band power if the band matches, otherwise calculates on-demand.

        Parameters:
        - task: Task name (e.g., 'task1').
        - state: State name (e.g., 'MW', 'OT').
        - band: Frequency band as a tuple (low_freq, high_freq) in Hz.

        Returns:
        - A tuple containing:
            - Mean band power per channel.
            - Variance of band power per channel.
            - Standard error of band power per channel.
        """
        if self.band_range == band and (task, state) in [(t, s) for t, states in self.band_power_map.items() for s in states]:
            # Use pre-calculated band power
            band_power = self.get_band_power(task, state)
        else:
            # Calculate on-demand
            psd = self.get_psd(task, state)
            freqs = self.get_freqs(task, state)
            band_power = Metrics.band_power(psd, freqs, band)
        
        mean = band_power.mean(axis=0)
        var = band_power.var(axis=0)
        std_err = band_power.std(axis=0) / np.sqrt(band_power.shape[0])
        return mean, var, std_err

    #                                           Visualization
    ##########################################################################################################

    def plot_topo_power(self, condition: tuple[str, str], band: tuple[float, float] = (8, 12), show: bool = True) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot topographic power map for a given task and state or for OT–MW difference.
        Uses pre-calculated band power when available.

        Parameters:
        - condition: tuple (task, state), where state can be 'MW', 'OT', or 'difference'.
                    If task is 'all', averages across all tasks in the recording.
        - band: tuple (low_freq, high_freq) specifying the frequency band in Hz.
        """
        task_input, state = condition
        montage_type = EEG_SETTINGS["MONTAGE"]

        # Get all available conditions
        available_conditions = self.list_conditions()

        # Determine which task–state pairs to use
        if task_input == "all":
            tasks = set(t for (t, s) in available_conditions)
        else:
            tasks = [task_input]

        # Prepare to store power values
        power_arrays = []

        for task in tasks:
            if state == "difference":
                available_tasks = self.get_available_tasks()
                mw_state, ot_state = QUALITY_CONTROL["MW_OT_STATES"]
                mw_tasks = [t for t in available_tasks if mw_state in self.get_available_states(t)]
                ot_tasks = [t for t in available_tasks if ot_state in self.get_available_states(t)]

                mw_arrays, ot_arrays = [], []

                for t in mw_tasks:
                    if self.band_range == band and (t, mw_state) in [(task, s) for task, states in self.band_power_map.items() for s in states]:
                        # Use pre-calculated band power
                        alpha = self.get_band_power(t, mw_state).mean(axis=0)
                    else:
                        # Calculate on-demand
                        alpha = self.stats_band_power_per_channel(t, mw_state, band)[0]
                    mw_arrays.append(alpha)

                for t in ot_tasks:
                    if self.band_range == band and (t, ot_state) in [(task, s) for task, states in self.band_power_map.items() for s in states]:
                        # Use pre-calculated band power
                        alpha = self.get_band_power(t, ot_state).mean(axis=0)
                    else:
                        # Calculate on-demand
                        alpha = self.stats_band_power_per_channel(t, ot_state, band)[0]
                    ot_arrays.append(alpha)

                if not mw_arrays or not ot_arrays:
                    print("Insufficient data for computing OT–MW difference.")
                    return

                alpha_mw = np.mean(mw_arrays, axis=0)
                alpha_ot = np.mean(ot_arrays, axis=0)

                power_arrays.append(alpha_ot - alpha_mw)
            else:
                if (task, state) not in available_conditions:
                    continue
                
                if self.band_range == band and (task, state) in [(t, s) for t, states in self.band_power_map.items() for s in states]:
                    # Use pre-calculated band power
                    alpha = self.get_band_power(task, state).mean(axis=0)
                else:
                    # Calculate on-demand
                    psd = self.get_psd(task, state)
                    freqs = self.get_freqs(task, state)
                    alpha = Metrics.band_power(psd, freqs, band, operation='mean').mean(axis=0)
                power_arrays.append(alpha)

        if not power_arrays:
            print("No valid data available for the requested condition.")
            return

        # Average across tasks if multiple arrays
        topo_data = np.mean(power_arrays, axis=0)

        # Create MNE info with channel locations
        info = mne.create_info(ch_names=self.channels, sfreq=128, ch_types="eeg")
        info.set_montage(montage_type)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot and get the image object
        im, _ = mne.viz.plot_topomap(
            topo_data, info, axes=ax, show=False, contours=4
        )

        # Add vertical colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

        # Unit for colorbar
        unit = "µV²" 

        cbar.set_label(unit, rotation=270, labelpad=15)

        # Title and layout
        title = f"State: {state} | Band: {band[0]}–{band[1]} Hz"
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_topo_power_comparison(self, task_input: str, band: tuple[float, float] = (8, 12), show: bool = True, use_decibel: bool = False):
        """
        Plot topographic power maps for OT, MW, and OT-MW difference in one figure.

        Parameters:
        - task_input: str, task name or 'all' to average across all tasks.
        - band: tuple (low_freq, high_freq), frequency band in Hz.
        - show: bool, whether to display the plot.
        - use_decibel: bool, whether to plot power in decibels.
        """
        montage_type = EEG_SETTINGS["MONTAGE"]
        info = mne.create_info(ch_names=self.channels, sfreq=128, ch_types="eeg")
        info.set_montage(montage_type)

        available_conditions = self.list_conditions()
        if task_input == "all":
            tasks_to_process = sorted(list(set(t for (t, s) in available_conditions)))
        else:
            tasks_to_process = [task_input]

        # Helper function to get averaged power for a given state
        def get_averaged_power_for_state(state_str: str, tasks_list: list[str], band_tuple: tuple[float, float]) -> np.ndarray | None:
            power_arrays_for_state = []
            for task_iter in tasks_list:
                if (task_iter, state_str) in available_conditions:
                    try:
                        if self.band_range == band_tuple and (task_iter, state_str) in [(t, s) for t, states in self.band_power_map.items() for s in states]:
                            # Use pre-calculated band power
                            if use_decibel:
                                power_val = self.get_log_band_power(task_iter, state_str).mean(axis=0)
                            else:
                                power_val = self.get_band_power(task_iter, state_str).mean(axis=0)
                        else:
                            # Calculate on-demand
                            psd_val = self.get_psd(task_iter, state_str)
                            freqs_val = self.get_freqs(task_iter, state_str)
                            if use_decibel:
                                power_val = Metrics.band_log(psd_val, freqs_val, band_tuple, operation='mean').mean(axis=0)
                            else:
                                power_val = Metrics.band_power(psd_val, freqs_val, band_tuple, operation='mean').mean(axis=0)
                        power_arrays_for_state.append(power_val)
                    except ValueError:
                        # This might happen if data is corrupted or unexpectedly missing
                        print(f"Warning: Could not retrieve data for task '{task_iter}', state '{state_str}'. Skipping.")
                        continue
            
            if not power_arrays_for_state:
                return None
            return np.mean(power_arrays_for_state, axis=0)

        topo_data_ot = get_averaged_power_for_state("OT", tasks_to_process, band)
        topo_data_mw = get_averaged_power_for_state("MW", tasks_to_process, band)

        topo_data_diff = None
        if topo_data_ot is not None and topo_data_mw is not None:
            topo_data_diff = topo_data_ot - topo_data_mw
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plot_data_list = [topo_data_ot, topo_data_mw, topo_data_diff]
        plot_titles = ["On-target", "Mind-wandering", "Difference (OT-MW)"]

        # Determine vlim for OT and MW plots (shared scale)
        vlim_ot_mw = None
        valid_ot_mw_data = [d for d in [topo_data_ot, topo_data_mw] if d is not None]
        if valid_ot_mw_data:
            min_val = np.min([np.min(d) for d in valid_ot_mw_data])
            max_val = np.max([np.max(d) for d in valid_ot_mw_data])
            vlim_ot_mw = (min_val, max_val)
            if min_val == max_val: # Avoid vlim=(x,x) if data is flat
                vlim_ot_mw = (min_val - 0.1, max_val + 0.1) if min_val != 0 else (-0.1, 0.1)


        # Determine vlim for difference plot (symmetrical around 0)
        vlim_diff = None
        if topo_data_diff is not None:
            abs_max_diff = np.max(np.abs(topo_data_diff))
            if abs_max_diff == 0: # Data is all zeros
                 vlim_diff = (-0.1, 0.1) # Small range for visual clarity
            else:
                 vlim_diff = (-abs_max_diff, abs_max_diff)
        
        vlims_list = [vlim_ot_mw, vlim_ot_mw, vlim_diff]

        colorbar_label = "ln(µV²)" if use_decibel else "µV²"

        for i, ax in enumerate(axes):
            data = plot_data_list[i]
            title = plot_titles[i]
            current_vlim = vlims_list[i]
            
            # Determine cmap: explicit for difference, default for others
            current_cmap = None
            if title == "Difference (OT-MW)":
                current_cmap = 'coolwarm' # Explicitly set for the difference plot
            else:
                current_cmap = "Reds"

            if data is not None:
                im, _ = mne.viz.plot_topomap(
                    data, 
                    info, 
                    axes=ax, 
                    show=False, 
                    contours=4, 
                    vlim=current_vlim, 
                    cmap=current_cmap  # Use the determined colormap
                )
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label(colorbar_label, rotation=270, labelpad=15)
                ax.set_title(title, fontsize=12)
            else:
                ax.set_title(title, fontsize=12)
                ax.text(0.5, 0.5, "Data not available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')

        main_title_parts = [
            f"Session: {self.session_id}",
            f"Band: {band[0]}–{band[1]} Hz"
        ]
        if use_decibel:
            main_title_parts.append("Scale: dB")
        if task_input != "all":
            main_title_parts.append(f"Task: {task_input}")
        
        fig.suptitle(" | ".join(main_title_parts), fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and bottom

        # Save figure to Plots path
        if use_decibel:
            save_dir = os.path.join(PLOTS_PATH, "topo_power_comparison", f"band-{band[0]}-{band[1]}", "decibel")
        else:
            save_dir = os.path.join(PLOTS_PATH, "topo_power_comparison", f"band-{band[0]}-{band[1]}")
        os.makedirs(save_dir, exist_ok=True)
        decibel_suffix = "_db" if use_decibel else ""
        file_name = f"topo_power_comparison_ses-{self.session_id}_task-{task_input}_band-{band[0]}-{band[1]}{decibel_suffix}.svg"
        file_path = os.path.join(save_dir, file_name)
        fig.savefig(file_path, format='svg', bbox_inches='tight')
        print(f"Topographic power comparison saved to {file_path}")
        
        if show:
            plt.show()
        plt.close()
        
        return fig, axes

    def plot_psd(self, show: bool = True):
        """
        Plot the Power Spectral Density (PSD) in decibel for all task-state pairs.
        The area under each line is filled.
        """
        plt.figure(figsize=(12, 7))
        
        conditions = self.list_conditions()
        if not conditions:
            print(f"No conditions to plot for session {self.session_id}.")
            plt.close() # Close the empty figure
            return

        colors = plt.cm.get_cmap('tab10', len(conditions))
        
        all_mean_psd_db = []
        all_freqs = []

        for i, (current_task, current_state) in enumerate(conditions):
            psd = self.get_psd(current_task, current_state)
            freqs = self.get_freqs(current_task, current_state)
            psd_db = Metrics.to_db(psd)
            mean_psd_db = psd_db.mean(axis=(0, 1))  # Average across epochs and channels
            all_mean_psd_db.append(mean_psd_db)
            all_freqs.append(freqs)

        # Determine a global minimum for the y-axis fill baseline
        # This ensures fill is always "under" the line, even for negative values
        global_min_val = np.min([np.min(psd_data) for psd_data in all_mean_psd_db if psd_data.size > 0])
        # Add a small padding below the global minimum for the fill
        fill_baseline = global_min_val - (np.abs(global_min_val) * 0.01 if global_min_val != 0 else 0.1)


        for i, (current_task, current_state) in enumerate(conditions):
            mean_psd_db = all_mean_psd_db[i]
            freqs = all_freqs[i]
            line_color = colors(i)
            
            plt.plot(freqs, mean_psd_db, label=f"{current_task} - {current_state}", color=line_color)
            plt.fill_between(freqs, mean_psd_db, y2=fill_baseline, alpha=0.3, color=line_color, interpolate=True)

        plt.title(f"Session: {self.session_id} | All Task-State PSDs")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB)")
        plt.legend()
        plt.grid(True)

        # save the figure to Plots path
        save_dir = os.path.join(PLOTS_PATH, "psd_plots")
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"psd_plot_ses-{self.session_id}.svg"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, format='svg', bbox_inches='tight')
        print(f"PSD plot saved to {file_path}")

        if show:
            plt.show()
        
        plt.close()

    def plot_distribution(self, band: tuple[float, float] = (8, 12), show: bool = True, use_decibel: bool = True):
        """
        Plot the distribution of alpha power for each channel in the specified frequency band.
        Produces a histogram for each channel showing the distribution of alpha power with KDE.
        The histogram does not separate tasks or states, but rather shows the overall distribution
        of alpha power across all epochs.
        If use_decibel is True, uses decibel-transformed band power.
        """
        channel_names = self.get_channel_names()
        if not channel_names:
            print(f"No channels available for session {self.session_id}.")
            return
        
        # Prepare data for Seaborn
        data = []
        for task, states in self.psd_map.items():
            for state, psd in states.items():
                if self.band_range == band and (task, state) in [(t, s) for t, states in self.band_power_map.items() for s in states]:
                    # Use pre-calculated band power
                    if use_decibel:
                        band_power = self.get_log_band_power(task, state)
                    else:
                        band_power = self.get_band_power(task, state)
                else:
                    # Calculate on-demand
                    freqs = self.freq_map[task][state]
                    if use_decibel:
                        band_power = Metrics.band_log(psd, freqs, band, operation='mean')
                    else:
                        band_power = Metrics.band_power(psd, freqs, band, operation='mean')
                
                # band_power shape: (epochs, channels)
                for epoch_idx in range(band_power.shape[0]):
                    for ch_idx, ch_name in enumerate(channel_names):
                        data.append({
                            'band_power': band_power[epoch_idx, ch_idx],
                            'condition': f"{task} - {state}",
                            'channel': ch_name
                        })

        if not data:
            print(f"No data available for plotting distribution for session {self.session_id}.")
            return

        df = pd.DataFrame(data)

        # Plotting
        for channel_name in channel_names:
            plt.figure(figsize=(10, 6))
            
            channel_data = df[df['channel'] == channel_name]
            
            sns.histplot(
                data=channel_data,
                x='band_power',
                kde=True,
                stat='density',
                alpha=0.6,
                common_norm=False
            )
            
            plt.title(f'Band Power Distribution for Channel {channel_name} (Session {self.session_id})')
            plt.xlabel('Band Power (dB)' if use_decibel else 'Band Power (µV²/Hz)')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the figure to Plots path
            save_dir = os.path.join(PLOTS_PATH, "distribution_plots", f"band-{band[0]}-{band[1]}")
            os.makedirs(save_dir, exist_ok=True)
            decibel_suffix = "_db" if use_decibel else ""
            file_name = f"distribution_plot_ses-{self.session_id}_channel-{channel_name}{decibel_suffix}.svg"
            file_path = os.path.join(save_dir, file_name)
            plt.savefig(file_path, format='svg', bbox_inches='tight')
            print(f"Distribution plot saved to {file_path}")

            if show:
                plt.show()
            
            plt.close()








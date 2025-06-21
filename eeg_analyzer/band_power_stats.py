from typing import Union, Dict, Any, Optional, Tuple
from collections import defaultdict

from scipy.stats import mannwhitneyu, iqr, skew, kurtosis, wilcoxon, PermutationMethod, mode, shapiro, sem
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


class BandPowerStats:
    """
    A class to compute and store comprehensive statistics for band power data.
    
    Computes statistics across epochs for each channel, with support for:
    - By condition (task, state)
    - By state only (combined across tasks)
    - All data combined
    - Both filtered and unfiltered data
    """
    
    def __init__(self, channels: list[str]):
        """
        Initialize the statistics calculator.
        
        Parameters:
        - channels: List of channel names
        """
        self.channels = channels
        self.stats = {
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
    
    def calculate_all_stats(self, band_power_map: dict, log_band_power_map: dict, 
                           outlier_mask_map: dict):
        """
        Calculate all statistics for both band power and log band power.
        
        Parameters:
        - band_power_map: Dictionary with structure task -> state -> array(epochs, channels)
        - log_band_power_map: Dictionary with structure task -> state -> array(epochs, channels)
        - outlier_mask_map: Dictionary with structure task -> state -> boolean_mask(epochs, channels)
        """
          # Calculate stats for band power
        self._calculate_stats_for_data_type('band_power', band_power_map, outlier_mask_map)
        
        # Calculate stats for log band power
        self._calculate_stats_for_data_type('log_band_power', log_band_power_map, outlier_mask_map)
        
    
    def _calculate_stats_for_data_type(self, data_type: str, data_map: dict, 
                                      outlier_mask_map: dict):
        """Calculate statistics for a specific data type (band_power or log_band_power)."""
        
        # 1. Calculate by_condition stats
        
        for task, states in data_map.items():
            for state, data in states.items():
                condition_key = (task, state)
                mask = outlier_mask_map[task][state]

                self.stats[data_type]['by_condition'][condition_key] = {
                    'unfiltered': self._calculate_channel_stats(data),
                    'filtered': self._calculate_channel_stats(data, mask)
                }
        
        # 2. Calculate by_state stats (combine across tasks)
        state_data = defaultdict(list)
        state_masks = defaultdict(list)
        
        for task, states in data_map.items():
            for state, data in states.items():
                state_data[state].append(data)
                state_masks[state].append(outlier_mask_map[task][state])
        
        for state, data_list in state_data.items():
            combined_data = np.concatenate(data_list, axis=0)
            combined_mask = np.concatenate(state_masks[state], axis=0)
            
            self.stats[data_type]['by_state'][state] = {
                'unfiltered': self._calculate_channel_stats(combined_data),
                'filtered': self._calculate_channel_stats(combined_data, combined_mask)
            }
          # 3. Calculate all_data stats (combine everything)
        all_data_list = []
        all_masks_list = []
        
        for task, states in data_map.items():
            for state, data in states.items():
                all_data_list.append(data)
                all_masks_list.append(outlier_mask_map[task][state])
        
        if all_data_list:
            combined_all_data = np.concatenate(all_data_list, axis=0)
            combined_all_masks = np.concatenate(all_masks_list, axis=0)
            
            unfiltered_stats = self._calculate_channel_stats(combined_all_data)
            filtered_stats = self._calculate_channel_stats(combined_all_data, combined_all_masks)
            
            self.stats[data_type]['all_data'] = {
                'unfiltered': unfiltered_stats,
                'filtered': filtered_stats
            }
    
    def _calculate_channel_stats(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
        """
        Calculate statistics for each channel across epochs.
        
        Parameters:
        - data: Array of shape (epochs, channels)
        - mask: Optional boolean mask of shape (epochs, channels). True = valid data
          Returns:
        - Dictionary with channel -> statistics mapping
        """
        channel_stats = {}
        
        for ch_idx, channel in enumerate(self.channels):
            channel_data = data[:, ch_idx]
            
            if mask is not None:
                channel_mask = mask[:, ch_idx]
                channel_data = channel_data[channel_mask]
            
            channel_stats[channel] = self._compute_stats(channel_data)
        
        return channel_stats
    
    def _compute_stats(self, data: np.ndarray) -> dict:
        """
        Compute comprehensive statistics for a 1D array.
        
        Parameters:
        - data: 1D array of values
        
        Returns:
        - Dictionary with statistical measures
        """
        if len(data) == 0:
            return self._empty_stats()
        
        # Remove NaN values for calculations
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return self._empty_stats()
        
        stats = {
            'mean': np.mean(clean_data),
            'variance': np.var(clean_data, ddof=1) if len(clean_data) > 1 else 0.0,
            'std_error': sem(clean_data) if len(clean_data) > 1 else np.nan,
            'min_value': np.min(clean_data),
            'lower_quartile': np.percentile(clean_data, 25),
            'median': np.median(clean_data),
            'upper_quartile': np.percentile(clean_data, 75),
            'max_value': np.max(clean_data),
            'iqr': iqr(clean_data),
            'skewness': skew(clean_data),
            'kurtosis': kurtosis(clean_data, fisher=True),
            'epoch_count': len(clean_data)
        }
        
        # Calculate modes (requires minimum data points)
        stats.update(self._calculate_modes(clean_data))
        
        # Test for normality
        stats.update(self._test_normality(clean_data))
        
        return stats
    
    def _calculate_modes(self, data: np.ndarray, min_points: int = 10) -> dict:
        """
        Calculate mode statistics with robust handling.
        
        Parameters:
        - data: 1D array of values
        - min_points: Minimum number of points required for mode calculation
        
        Returns:
        - Dictionary with mode-related statistics
        """
        if len(data) < min_points:
            return {
                'modes': [],
                'mode_count': 0,
                'mode_frequency': 0
            }
        
        try:
            # Use scipy.stats.mode for continuous data
            mode_result = mode(data, keepdims=True)
            modes = mode_result.mode.flatten()
            mode_freq = mode_result.count[0] if len(mode_result.count) > 0 else 0
            
            return {
                'modes': modes.tolist(),
                'mode_count': len(modes),
                'mode_frequency': mode_freq
            }
        except Exception:
            return {
                'modes': [],
                'mode_count': 0,
                'mode_frequency': 0
            }
    
    def _test_normality(self, data: np.ndarray, min_points: int = 3) -> dict:
        """
        Test for normality using Shapiro-Wilk test.
        
        Parameters:
        - data: 1D array of values
        - min_points: Minimum number of points required for the test
        
        Returns:
        - Dictionary with normality test results
        """
        if len(data) < min_points:
            return {
                'is_normal': False,
                'normality_p_value': np.nan,
                'normality_statistic': np.nan
            }
        
        try:
            stat, p_value = shapiro(data)
            return {
                'is_normal': p_value > 0.05,  # Common significance level
                'normality_p_value': p_value,
                'normality_statistic': stat
            }
        except Exception:
            return {
                'is_normal': False,
                'normality_p_value': np.nan,
                'normality_statistic': np.nan
            }
    
    def _empty_stats(self) -> dict:
        """Return empty/NaN statistics for cases with no valid data."""
        return {
            'mean': np.nan,
            'variance': np.nan,
            'median': np.nan,
            'iqr': np.nan,
            'min_value': np.nan,
            'max_value': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'epoch_count': 0,
            'modes': [],
            'mode_count': 0,
            'mode_frequency': 0,
            'is_normal': False,
            'normality_p_value': np.nan,
            'normality_statistic': np.nan
        }
    
    def get_stat(self, stat_name: str, channel: str, data_type: str = 'band_power',
                 task: Optional[str] = None, state: Optional[str] = None, 
                 filtered: bool = False) -> Any:
        """
        Unified method to get any statistic.
        
        Parameters:
        - stat_name: Name of the statistic to retrieve
        - channel: Channel name
        - data_type: 'band_power' or 'log_band_power'
        - task: Task name (if None, uses state-level or all_data stats)
        - state: State name (if None, uses all_data stats)
        - filtered: Whether to use filtered (outlier-removed) data
        
        Returns:
        - The requested statistic value
        """
        filter_key = 'filtered' if filtered else 'unfiltered'
        
        try:
            if task is not None and state is not None:
                # By condition
                condition_key = (task, state)
                return self.stats[data_type]['by_condition'][condition_key][filter_key][channel][stat_name]
            elif state is not None:
                # By state
                return self.stats[data_type]['by_state'][state][filter_key][channel][stat_name]
            else:
                # All data
                return self.stats[data_type]['all_data'][filter_key][channel][stat_name]
        except KeyError:
            return np.nan

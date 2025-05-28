"""
Processor
---------

Handles transformations and preprocessing of EEG features for analysis.

Responsibilities:
- Apply outlier detection and rejection (e.g., z-score filtering).
- Normalize data across conditions or subjects.
- Support exclusion rules based on configurable criteria.

Notes:
- Should remain agnostic of dataset structure.
- Ideally takes raw metric data as input and returns cleaned versions.
"""

import numpy as np
import pandas as pd  # Ensure pandas is imported
from eeg_analyzer.metrics import Metrics
from eeg_analyzer.subject import Subject


class Processor:
    @staticmethod
    def flag_outliers_zscore(df: pd.DataFrame, group_cols: list, value_col: str, threshold: float = 3.0, zscore_col_name: str = None) -> pd.DataFrame:
        """
        Adds a Z-score column and flags outliers in the 'is_bad' column based on Z-scores
        within specified groups. Operates on a copy of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame. Must contain an 'is_bad' column.
        - group_cols (list): List of column names to group by for Z-score calculation.
        - value_col (str): The name of the column to calculate Z-scores on.
        - threshold (float): The Z-score threshold. Rows with abs(Z-score) > threshold are flagged.
        - zscore_col_name (str, optional): Name for the new Z-score column. 
                                           Defaults to f"zscore_{value_col}".

        Returns:
        - pd.DataFrame: A new DataFrame with added Z-score column and updated 'is_bad' column.
        """
        if df.empty:
            return df

        df_copy = df.copy()
        if 'is_bad' not in df_copy.columns:
            df_copy['is_bad'] = False # Initialize if not present

        if zscore_col_name is None:
            zscore_col_name = f"zscore_{value_col}"

        def calculate_zscore(group):
            return (group[value_col] - group[value_col].mean()) / group[value_col].std(ddof=0)

        if group_cols:
            df_copy[zscore_col_name] = df_copy.groupby(group_cols, group_keys=False).apply(calculate_zscore).reset_index(level=0, drop=True)
        else: # Global z-score
            df_copy[zscore_col_name] = (df_copy[value_col] - df_copy[value_col].mean()) / df_copy[value_col].std(ddof=0)
        
        # Flag outliers: update 'is_bad' to True where it's already True OR where new outlier condition is met
        df_copy['is_bad'] = df_copy['is_bad'] | (np.abs(df_copy[zscore_col_name]) > threshold)
        
        return df_copy

    @staticmethod
    def flag_outliers_iqr(df: pd.DataFrame, group_cols: list, value_col: str, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Flags outliers in the 'is_bad' column based on the IQR method within specified groups.
        Operates on a copy of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame. Must contain an 'is_bad' column.
        - group_cols (list): List of column names to group by for IQR calculation.
        - value_col (str): The name of the column to check for outliers.
        - multiplier (float): The IQR multiplier (typically 1.5).

        Returns:
        - pd.DataFrame: A new DataFrame with an updated 'is_bad' column.
        """
        if df.empty:
            return df

        df_copy = df.copy()
        if 'is_bad' not in df_copy.columns:
            df_copy['is_bad'] = False # Initialize if not present


        def get_iqr_bounds(group):
            q1 = group[value_col].quantile(0.25)
            q3 = group[value_col].quantile(0.75)
            iqr_val = q3 - q1
            lower_bound = q1 - multiplier * iqr_val
            upper_bound = q3 + multiplier * iqr_val
            # Create a boolean series for outliers within the group
            is_outlier_group = (group[value_col] < lower_bound) | (group[value_col] > upper_bound)
            return is_outlier_group

        if group_cols:
            # Apply to get boolean series for outliers per group, then combine
            # The result of apply here should have an index aligned with df_copy.index
            outlier_flags_series = df_copy.groupby(group_cols, group_keys=False).apply(get_iqr_bounds)
            df_copy['is_bad'] = df_copy['is_bad'] | outlier_flags_series
        else: # Global IQR
            q1 = df_copy[value_col].quantile(0.25)
            q3 = df_copy[value_col].quantile(0.75)
            iqr_val = q3 - q1
            lower_bound = q1 - multiplier * iqr_val
            upper_bound = q3 + multiplier * iqr_val
            df_copy['is_bad'] = df_copy['is_bad'] | ((df_copy[value_col] < lower_bound) | (df_copy[value_col] > upper_bound))
            
        return df_copy
    


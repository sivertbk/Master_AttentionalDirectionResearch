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
    def filter_outliers_zscore(df: pd.DataFrame, group_cols: list, value_col: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Filters outliers from a DataFrame based on Z-scores within specified groups.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - group_cols (list): List of column names to group by for Z-score calculation.
        - value_col (str): The name of the column to check for outliers.
        - threshold (float): The Z-score threshold. Rows with abs(Z-score) > threshold are removed.

        Returns:
        - pd.DataFrame: A new DataFrame with outliers removed.
        """
        if df.empty:
            return df

        def calculate_zscore(group):
            return (group[value_col] - group[value_col].mean()) / group[value_col].std(ddof=0)  # ddof=0 for population std

        df_copy = df.copy()
        if group_cols:
            df_copy['z_score'] = df_copy.groupby(group_cols, group_keys=False).apply(calculate_zscore).reset_index(level=0, drop=True)
        else:  # Global z-score
            df_copy['z_score'] = (df_copy[value_col] - df_copy[value_col].mean()) / df_copy[value_col].std(ddof=0)

        return df_copy[np.abs(df_copy['z_score']) <= threshold].drop(columns=['z_score'])

    @staticmethod
    def filter_outliers_iqr(df: pd.DataFrame, group_cols: list, value_col: str, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Filters outliers from a DataFrame based on the IQR method within specified groups.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - group_cols (list): List of column names to group by for IQR calculation.
        - value_col (str): The name of the column to check for outliers.
        - multiplier (float): The IQR multiplier (typically 1.5).

        Returns:
        - pd.DataFrame: A new DataFrame with outliers removed.
        """
        if df.empty:
            return df

        def get_iqr_bounds(group):
            q1 = group[value_col].quantile(0.25)
            q3 = group[value_col].quantile(0.75)
            iqr_val = q3 - q1
            lower_bound = q1 - multiplier * iqr_val
            upper_bound = q3 + multiplier * iqr_val
            return pd.Series({'lower_bound': lower_bound, 'upper_bound': upper_bound})

        df_copy = df.copy()
        if group_cols:
            bounds = df_copy.groupby(group_cols, group_keys=False).apply(get_iqr_bounds)
            df_copy = df_copy.join(bounds, on=group_cols)
        else:  # Global IQR
            q1 = df_copy[value_col].quantile(0.25)
            q3 = df_copy[value_col].quantile(0.75)
            iqr_val = q3 - q1
            df_copy['lower_bound'] = q1 - multiplier * iqr_val
            df_copy['upper_bound'] = q3 + multiplier * iqr_val

        filtered_df = df_copy[(df_copy[value_col] >= df_copy['lower_bound']) & (df_copy[value_col] <= df_copy['upper_bound'])]
        return filtered_df.drop(columns=['lower_bound', 'upper_bound'])

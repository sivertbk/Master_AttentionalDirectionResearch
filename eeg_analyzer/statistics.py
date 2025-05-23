"""
Statistics
----------

Implements statistical tests and models for analyzing EEG data.

Responsibilities:
- a container for statistical tests and models
- a container for statistical measures

Used by EEGAnalyzer to perform statistical analysis on EEG data.

NOTE: Ensures logging of all statistical tests, measures, and models.
"""

from scipy.stats import mannwhitneyu, iqr, skew, kurtosis, wilcoxon
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd



class Statistics:
    @staticmethod
    def calculate_descriptive_stats(df: pd.DataFrame, value_col: str, group_cols: list) -> pd.DataFrame:
        """
        Calculates descriptive statistics for a given column, grouped by specified columns.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - value_col (str): The name of the column to calculate statistics on.
        - group_cols (list): A list of column names to group by.

        Returns:
        - pd.DataFrame: A DataFrame with descriptive statistics.
        """
        if not group_cols:
            # Calculate global statistics if no group_cols are provided
            stats = {
                f'mean_{value_col}': [df[value_col].mean()],
                f'std_{value_col}': [df[value_col].std()],
                f'median_{value_col}': [df[value_col].median()],
                f'iqr_{value_col}': [iqr(df[value_col], nan_policy='omit')],
                f'min_{value_col}': [df[value_col].min()],
                f'max_{value_col}': [df[value_col].max()],
                f'skewness_{value_col}': [skew(df[value_col], nan_policy='omit')],
                f'kurtosis_{value_col}': [kurtosis(df[value_col], fisher=True, nan_policy='omit')],
                'count': [len(df)] # General count of rows in the group
            }
            # Add subject count if column exists
            if 'subject_id' in df.columns:
                stats['subject_count'] = [df['subject_id'].nunique()]
            # Removed epoch_count for global stats as 'count' (len(df)) serves this.
            
            return pd.DataFrame(stats)

        # Define aggregation functions
        def q1(x):
            return x.quantile(0.25)

        def q3(x):
            return x.quantile(0.75)

        def iqr_custom_func(x): # Renamed to avoid conflict with imported iqr
            return q3(x) - q1(x)

        # Initialize agg_funcs with operations on the value_col
        agg_funcs = {
            f'mean_{value_col}': pd.NamedAgg(column=value_col, aggfunc='mean'),
            f'std_{value_col}': pd.NamedAgg(column=value_col, aggfunc='std'),
            f'median_{value_col}': pd.NamedAgg(column=value_col, aggfunc='median'),
            f'iqr_{value_col}': pd.NamedAgg(column=value_col, aggfunc=iqr_custom_func),
            f'min_{value_col}': pd.NamedAgg(column=value_col, aggfunc='min'),
            f'max_{value_col}': pd.NamedAgg(column=value_col, aggfunc='max'),
            f'skewness_{value_col}': pd.NamedAgg(column=value_col, aggfunc=lambda x: skew(x, nan_policy='omit')),
            f'kurtosis_{value_col}': pd.NamedAgg(column=value_col, aggfunc=lambda x: kurtosis(x, fisher=True, nan_policy='omit')),
            # 'count' can be ambiguous if value_col has NaNs. 
            # Using 'size' on the group directly is more robust for row count per group.
        }
        
        # Group first, then aggregate. 'size' is a GroupBy method.
        grouped = df.groupby(group_cols)
        summary_df = grouped.agg(**agg_funcs)
        
        # Add count of rows per group (more robust than count on a specific column if it has NaNs)
        summary_df['group_size'] = grouped.size() 

        # Add subject and epoch counts if columns exist
        if 'subject_id' in df.columns:
            subject_counts = grouped['subject_id'].nunique()
            summary_df = summary_df.join(subject_counts.rename('subject_count'))

        return summary_df.reset_index()

    # Placeholder for other statistical methods like Cohen's d, t-tests, etc.
    # @staticmethod
    # def cohen_d(group1, group2):
    #     # ... implementation ...
    #     pass

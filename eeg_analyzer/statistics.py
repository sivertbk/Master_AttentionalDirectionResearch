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
    def _get_stats_dict(series: pd.Series, value_col_name: str, suffix: str = "") -> dict:
        """Helper to compute descriptive stats for a series and return as a dict."""
        if series.empty or series.isnull().all():
            return {
                f'mean{suffix}': np.nan,
                f'std{suffix}': np.nan,
                f'median{suffix}': np.nan,
                f'iqr{suffix}': np.nan,
                f'min{suffix}': np.nan,
                f'max{suffix}': np.nan,
                f'skewness{suffix}': np.nan,
                f'kurtosis{suffix}': np.nan,
                f'count{suffix}': 0
            }
        
        # For scipy functions, ensure input is not all NaNs after dropping NaNs for some functions
        valid_series_for_scipy = series.dropna()
        if valid_series_for_scipy.empty:
            iqr_val = np.nan
            skew_val = np.nan
            kurt_val = np.nan
        else:
            iqr_val = iqr(valid_series_for_scipy, nan_policy='omit')
            skew_val = skew(valid_series_for_scipy, nan_policy='omit')
            kurt_val = kurtosis(valid_series_for_scipy, fisher=True, nan_policy='omit')


        return {
            f'mean{suffix}': series.mean(),
            f'std{suffix}': series.std(),
            f'median{suffix}': series.median(),
            f'iqr{suffix}': iqr_val,
            f'min{suffix}': series.min(),
            f'max{suffix}': series.max(),
            f'skewness{suffix}': skew_val,
            f'kurtosis{suffix}': kurt_val,
            f'count{suffix}': series.count() # Non-NaN count
        }

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
            # Using _get_stats_dict for consistency
            stats_dict = Statistics._get_stats_dict(df[value_col], value_col_name=value_col)
            # Rename keys to match original format f'{key}_{value_col}'
            stats = {f'{k}_{value_col}' if k != 'count' else k: v for k,v in stats_dict.items()}
            
            # Add subject count if column exists
            if 'subject_id' in df.columns:
                stats['subject_count'] = [df['subject_id'].nunique()]
            # Original 'count' was len(df), _get_stats_dict 'count' is non-NaN count of value_col.
            # For global, len(df) is more like 'group_size'. Let's keep original 'count' as len(df) for global.
            stats['count'] = [len(df)]
            stats[f'count_non_nan_{value_col}'] = stats_dict['count']


            return pd.DataFrame(stats)

        # Initialize agg_funcs with operations on the value_col
        agg_funcs = {
            f'mean_{value_col}': pd.NamedAgg(column=value_col, aggfunc='mean'),
            f'std_{value_col}': pd.NamedAgg(column=value_col, aggfunc='std'),
            f'median_{value_col}': pd.NamedAgg(column=value_col, aggfunc='median'),
            f'iqr_{value_col}': pd.NamedAgg(column=value_col, aggfunc=lambda x: iqr(x, nan_policy='omit')),
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

    @staticmethod
    def calculate_descriptive_stats_detailed(
        df: pd.DataFrame, 
        value_col: str, 
        group_cols: list, 
        state_col: str,
        positive_state: str,
        negative_state: str
    ) -> pd.DataFrame:
        """
        Calculates detailed descriptive statistics including per-state measures and differences.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - value_col (str): The name of the column to calculate statistics on.
        - group_cols (list): A list of column names to group by.
        - state_col (str): The name of the column indicating state (e.g., 'state').
        - positive_state (str): The name of the 'positive' state (e.g., 'OT').
        - negative_state (str): The name of the 'negative' state (e.g., 'MW').

        Returns:
        - pd.DataFrame: A DataFrame with detailed descriptive statistics.
        """
        if not group_cols: # Handle global case if necessary, or raise error
            # For simplicity, this detailed function expects group_cols.
            # Global detailed stats can be achieved by passing a dummy group_col.
            raise ValueError("group_cols must be provided for calculate_descriptive_stats_detailed.")

        results = []
        grouped = df.groupby(group_cols)

        for name, group_df in grouped:
            row_stats = {}
            if isinstance(name, tuple):
                for i, col_name in enumerate(group_cols):
                    row_stats[col_name] = name[i]
            else:
                row_stats[group_cols[0]] = name

            # Overall stats for the group
            overall_stats = Statistics._get_stats_dict(group_df[value_col], value_col, suffix="")
            row_stats.update(overall_stats)
            
            # Stats for positive state
            positive_df = group_df[group_df[state_col] == positive_state]
            pos_stats = Statistics._get_stats_dict(positive_df[value_col], value_col, suffix=f"_{positive_state}")
            row_stats.update(pos_stats)

            # Stats for negative state
            negative_df = group_df[group_df[state_col] == negative_state]
            neg_stats = Statistics._get_stats_dict(negative_df[value_col], value_col, suffix=f"_{negative_state}")
            row_stats.update(neg_stats)

            # State differences
            mean_pos = row_stats.get(f'mean_{positive_state}', np.nan)
            mean_neg = row_stats.get(f'mean_{negative_state}', np.nan)
            median_pos = row_stats.get(f'median_{positive_state}', np.nan)
            median_neg = row_stats.get(f'median_{negative_state}', np.nan)

            row_stats[f'state_diff_mean'] = mean_pos - mean_neg
            row_stats[f'state_diff_median'] = median_pos - median_neg
            
            # Group size and subject count
            row_stats['group_size'] = len(group_df)
            if 'subject_id' in df.columns: # Check original df for subject_id column
                row_stats['subject_count'] = group_df['subject_id'].nunique()
            
            results.append(row_stats)
            
        summary_df = pd.DataFrame(results)
        
        # Reorder columns to have group_cols first, then general stats, then state stats, then diffs
        ordered_cols = group_cols.copy()
        base_stat_names = ['mean', 'std', 'median', 'iqr', 'min', 'max', 'skewness', 'kurtosis', 'count']
        ordered_cols.extend(base_stat_names)
        ordered_cols.extend([f'{stat}_{positive_state}' for stat in base_stat_names])
        ordered_cols.extend([f'{stat}_{negative_state}' for stat in base_stat_names])
        ordered_cols.extend(['state_diff_mean', 'state_diff_median', 'group_size'])
        if 'subject_count' in summary_df.columns:
             ordered_cols.append('subject_count')
        
        # Filter to existing columns in summary_df to avoid KeyError if some stat wasn't computed (e.g. subject_count)
        ordered_cols = [col for col in ordered_cols if col in summary_df.columns]
        
        return summary_df[ordered_cols]


    # Placeholder for other statistical methods like Cohen's d, t-tests, etc.
    # @staticmethod
    # def cohen_d(group1, group2):
    #     # ... implementation ...
    #     pass

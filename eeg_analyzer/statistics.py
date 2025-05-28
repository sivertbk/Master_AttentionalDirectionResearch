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
from typing import Union

from scipy.stats import mannwhitneyu, iqr, skew, kurtosis, wilcoxon, PermutationMethod
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
        state_col: str
    ) -> pd.DataFrame:
        """
        Calculates detailed descriptive statistics including per-state measures and differences.
        Assumes state_col contains "OT" (positive) and "MW" (negative) string values.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - value_col (str): The name of the column to calculate statistics on.
        - group_cols (list): A list of column names to group by.
        - state_col (str): The name of the column indicating state (e.g., 'state').

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
            
            # Stats for positive state (OT)
            positive_df = group_df[group_df[state_col] == "OT"] # OT state
            pos_stats = Statistics._get_stats_dict(positive_df[value_col], value_col, suffix=f"_OT")
            row_stats.update(pos_stats)

            # Stats for negative state (MW)
            negative_df = group_df[group_df[state_col] == "MW"] # MW state
            neg_stats = Statistics._get_stats_dict(negative_df[value_col], value_col, suffix=f"_MW")
            row_stats.update(neg_stats)

            # State differences
            mean_pos = row_stats.get(f'mean_OT', np.nan)
            mean_neg = row_stats.get(f'mean_MW', np.nan)
            median_pos = row_stats.get(f'median_OT', np.nan)
            median_neg = row_stats.get(f'median_MW', np.nan)

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
        ordered_cols.extend([f'{stat}_OT' for stat in base_stat_names])
        ordered_cols.extend([f'{stat}_MW' for stat in base_stat_names])
        ordered_cols.extend(['state_diff_mean', 'state_diff_median', 'group_size'])
        if 'subject_count' in summary_df.columns:
             ordered_cols.append('subject_count')
        
        # Filter to existing columns in summary_df to avoid KeyError if some stat wasn't computed (e.g. subject_count)
        ordered_cols = [col for col in ordered_cols if col in summary_df.columns]
        
        return summary_df[ordered_cols]

    @staticmethod
    def perform_wilcoxon_test(df: pd.DataFrame, col1: str, col2: str, group_cols: list, 
                              tail: str = 'two-sided', 
                              n_permutations: int = 9999, 
                              random_state: Union[int, np.random.RandomState, None] = None) -> pd.DataFrame:
        """
        Performs a Wilcoxon signed-rank test for paired samples, grouped by specified columns,
        optionally using a permutation method.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - col1 (str): The name of the first column for paired samples.
        - col2 (str): The name of the second column for paired samples.
        - group_cols (list): A list of column names to group by.
        - tail (str): Specifies the alternative hypothesis ('two-sided', 'greater', 'less'). Default is 'two-sided'.
        - n_permutations (int): Number of permutations for the permutation test. Default is 9999.
                                If 0 or None, scipy's default method selection is used.
        - random_state (Union[int, np.random.RandomState, None]): Seed for random number generator for permutations.

        Returns:
        - pd.DataFrame: A DataFrame with group identifiers, Wilcoxon statistic, and p-value.
                          Returns an empty DataFrame if input df is empty or required columns are missing.
        """
        if df.empty:
            print("Input DataFrame is empty. Cannot perform Wilcoxon test.")
            return pd.DataFrame()

        required_data_cols = [col1, col2]
        all_required_cols = group_cols + required_data_cols
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns in DataFrame for Wilcoxon test: {missing_cols}.")
            return pd.DataFrame()

        results = []

        if not group_cols:
            # Perform test on the entire DataFrame if no group_cols are provided
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()
            
            # Align data for paired test - keep only common indices
            common_idx = data1.index.intersection(data2.index)
            data1_aligned = data1.loc[common_idx]
            data2_aligned = data2.loc[common_idx]

            wilcoxon_method_arg = 'auto'
            if n_permutations and n_permutations > 0:
                perm_method = PermutationMethod(n_resamples=n_permutations, random_state=random_state)
                wilcoxon_method_arg = perm_method

            if len(data1_aligned) < 10: # Arbitrary threshold, scipy might have its own internal checks
                print(f"Not enough paired samples for Wilcoxon test (found {len(data1_aligned)}).")
                stat, p_value = np.nan, np.nan
            else:
                try:
                    stat, p_value = wilcoxon(data1_aligned, data2_aligned, alternative=tail, method=wilcoxon_method_arg)
                except ValueError as e:
                    print(f"Could not perform Wilcoxon test: {e}")
                    stat, p_value = np.nan, np.nan
            
            results.append({'wilcoxon_statistic': stat, 'p_value': p_value})

        else:
            grouped = df.groupby(group_cols)
            for name, group_df in grouped:
                group_id_dict = {}
                if isinstance(name, tuple):
                    for i, col_name in enumerate(group_cols):
                        group_id_dict[col_name] = name[i]
                else:
                    group_id_dict[group_cols[0]] = name

                data1 = group_df[col1].dropna()
                data2 = group_df[col2].dropna()

                # Align data for paired test within the group
                common_idx = data1.index.intersection(data2.index)
                data1_aligned = data1.loc[common_idx]
                data2_aligned = data2.loc[common_idx]
                
                wilcoxon_method_arg = 'auto'
                if n_permutations and n_permutations > 0:
                    perm_method = PermutationMethod(n_resamples=n_permutations, random_state=random_state)
                    wilcoxon_method_arg = perm_method

                # Wilcoxon test requires at least some non-zero differences.
                # Scipy's wilcoxon handles small sample sizes and zero differences by raising ValueError.
                if len(data1_aligned) < 8: # A common heuristic minimum, though scipy might be more specific
                    # print(f"Group {name}: Not enough paired samples for Wilcoxon test (found {len(data1_aligned)}). Skipping.")
                    stat, p_value = np.nan, np.nan
                else:
                    try:
                        # Perform test on the difference if you want one-sample test on differences
                        # Or directly on two samples if that's the interpretation
                        stat, p_value = wilcoxon(data1_aligned, data2_aligned, alternative=tail, method=wilcoxon_method_arg)
                    except ValueError as e:
                        # This can happen if all differences are zero, or too few samples.
                        # print(f"Group {name}: Could not perform Wilcoxon test: {e}. Assigning NaN.")
                        stat, p_value = np.nan, np.nan
                
                result_row = group_id_dict.copy()
                result_row['wilcoxon_statistic'] = stat
                result_row['p_value'] = p_value
                results.append(result_row)

        return pd.DataFrame(results)


    @staticmethod
    def fit_mixedlm(df: pd.DataFrame, 
                    formula: str, 
                    groups_col: str, 
                    re_formula: str = None, 
                    vc_formula: str = None):
        """
        Fits a mixed-effects linear model using statsmodels.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - formula (str): The formula for the fixed effects (e.g., 'value ~ C(state)').
        - groups_col (str): The column name specifying the groups for random effects.
        - re_formula (str, optional): The formula for the random effects. Defaults to a random intercept for groups_col.
        - vc_formula (dict, optional): Variance components formula.

        Returns:
        - statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper or None: 
          The fitted model results, or None if fitting fails.
        """
        if df.empty:
            print("DataFrame is empty. Cannot fit MixedLM.")
            return None
        
        required_cols = [groups_col] + [term.strip() for term in formula.replace('~', ' ').replace('+', ' ').replace('*', ' ').replace('C(', '').replace(')', '').split() if term.strip().isalpha()]
        missing_cols = [col for col in required_cols if col not in df.columns and col != '1'] # '1' for intercept
        if any(missing_cols):
            print(f"Missing columns in DataFrame for MixedLM: {missing_cols}. Formula: {formula}")
            return None
        
        # Drop rows with NaNs in critical columns for the model
        # Identify columns involved in the formula (simple parsing)
        formula_vars = [v.strip() for v in formula.replace("~", " ").replace("+", " ").replace("*", " ").split(" ") if v.strip() and v.strip() != "C"]
        # Remove potential function calls like C(state) -> state
        formula_vars = [v.split('(')[-1].replace(')', '') for v in formula_vars] 
        cols_to_check_for_nans = list(set(formula_vars + [groups_col]))
        
        df_cleaned = df.dropna(subset=cols_to_check_for_nans)

        if df_cleaned.empty:
            print(f"DataFrame became empty after dropping NaNs in columns: {cols_to_check_for_nans}. Cannot fit MixedLM.")
            return None
        
        if df_cleaned[groups_col].nunique() < 2 and re_formula is not None : # Need multiple groups for random effects
            print(f"Not enough unique groups in '{groups_col}' ({df_cleaned[groups_col].nunique()}) for random effects. Skipping MixedLM for this slice.")
            # Potentially fit a simpler model like OLS or return None
            return None


        try:
            model = smf.mixedlm(formula, 
                                df_cleaned, 
                                groups=df_cleaned[groups_col], 
                                re_formula=re_formula, 
                                vc_formula=vc_formula)
            result = model.fit(reml=False) # Want full maximum likelihood estimation to compare models 
            return result
        except Exception as e:
            print(f"Error fitting MixedLM: {e}")
            print(f"Formula: {formula}, Groups: {groups_col}, RE Formula: {re_formula}")
            print(f"Data head:\n{df_cleaned.head()}")
            return None


    # Placeholder for other statistical methods like Cohen's d, t-tests, etc.
    # @staticmethod
    # def cohen_d(group1, group2):
    #     # ... implementation ...
    #     pass

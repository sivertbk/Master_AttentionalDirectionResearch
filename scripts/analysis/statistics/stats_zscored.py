"""
Z-Score Processing and Summary Statistics Script
-------------------------------------------------

This script demonstrates:
1. Loading or creating an EEGAnalyzer instance.
2. Applying Z-score based outlier flagging to its DataFrame using the Processor.
   - A new column with Z-scores is added.
   - The 'is_bad' column is updated for outliers.
3. Generating summary statistics tables:
   - On the full dataset (including Z-scores and all original data points).
   - On the dataset excluding rows flagged as 'is_bad'.
"""
import pandas as pd

from utils.config import DATASETS
from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from eeg_analyzer.processor import Processor

# --- Configuration ---
ANALYZER_NAME = "ZScoreProcessedData"
ANALYZER_DESCRIPTION = (
    "This analyzer holds data processed with Z-score outlier flagging. "
    "Its internal DataFrame (self.df) is modified to include a Z-score column "
    "for 'band_power' and an updated 'is_bad' column. "
    "Summary tables can be generated on the full or cleaned data."
)
ZSCORE_THRESHOLD = 3.0
VALUE_COL_TO_PROCESS = 'band_power' # The column to calculate Z-scores on
# Grouping for Z-score calculation (e.g., per channel within each subject-session and state)
ZSCORE_GROUP_COLS = ['dataset', 'subject_session', 'channel', 'state'] 
# Grouping for the final summary tables
SUMMARY_GROUP_COLS = ['dataset'] 


# --- Load or Create Analyzer ---
analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)

if analyzer is None:
    print(f"Analyzer '{ANALYZER_NAME}' not found. Creating a new one.")
    analyzer = EEGAnalyzer(
        DATASETS,
        analyzer_name=ANALYZER_NAME,
        description=ANALYZER_DESCRIPTION
    )
    analyzer.create_dataframe() # Create the base DataFrame
    # Initial save after creation
    analyzer.save_analyzer() 
    print(f"Analyzer '{ANALYZER_NAME}' created and saved with initial DataFrame.")
else:
    print(f"Successfully loaded existing analyzer: '{ANALYZER_NAME}'")
    # If analyzer is loaded, ensure its df is also loaded if not already part of the pickled object
    if analyzer.df is None:
        print(f"Loaded analyzer '{ANALYZER_NAME}' has no DataFrame. Loading default or creating...")
        analyzer.load_dataframe() # Tries to load eeganalyzer_default_df.csv
        if analyzer.df is None: # If still none
            analyzer.create_dataframe()
            analyzer.save_dataframe() # Save the newly created df
            analyzer.save_analyzer() # And update the analyzer pkl

print(analyzer)
if analyzer.df is not None:
    print(f"Initial DataFrame shape: {analyzer.df.shape}")
    # Ensure 'is_bad' column exists for accurate initial count
    if 'is_bad' not in analyzer.df.columns:
        analyzer.df['is_bad'] = False
    print(f"Initial 'is_bad' counts:\n{analyzer.df['is_bad'].value_counts(dropna=False)}")
else:
    print("Analyzer DataFrame is missing.")
    exit()

# --- Process Data: Add Z-scores and Flag Outliers ---
# This will modify analyzer.df directly.

zscore_col_name_generated = f'zscore_{VALUE_COL_TO_PROCESS}' # Define it once

analyzer.apply_zscore_flagging(
    group_cols=ZSCORE_GROUP_COLS,
    value_col=VALUE_COL_TO_PROCESS,
    threshold=ZSCORE_THRESHOLD,
    zscore_col_name=zscore_col_name_generated
)

# Now analyzer.df is updated
print(f"\nShape of analyzer.df after Z-score processing: {analyzer.df.shape}")
if zscore_col_name_generated in analyzer.df.columns:
    print(f"Sample Z-scores in analyzer.df for '{VALUE_COL_TO_PROCESS}':\n{analyzer.df[zscore_col_name_generated].describe()}")
print(f"'is_bad' counts in analyzer.df after Z-score flagging:\n{analyzer.df['is_bad'].value_counts(dropna=False)}")

# Save the analyzer as its df has been modified
analyzer.set_description(f"{ANALYZER_DESCRIPTION} - Z-score flagging applied on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}.")
analyzer.save_analyzer()

# --- Generate Summary Tables ---
# The 'filter_type' now describes the state of analyzer.df

# 1. Summary table on the full processed data (all rows from analyzer.df, which now has z-scores and flags)
print(f"\nGenerating summary table on FULL data (from analyzer.df) grouped by {SUMMARY_GROUP_COLS}...")
summary_full_flagged_data = analyzer.generate_summary_table(
    groupby_cols=SUMMARY_GROUP_COLS,
    target_col=VALUE_COL_TO_PROCESS, 
    filter_type="zscore_flagged", # Describes the state of analyzer.df
    exclude_bad_rows=False, # Include rows marked as bad
    output_filename_suffix="dataset_zscore_flagged_all_rows" 
)
if summary_full_flagged_data is not None:
    print(f"\n--- Summary Table (Full Z-score Flagged Data, on '{VALUE_COL_TO_PROCESS}') ---")
    print(summary_full_flagged_data)

# If you also want a summary of the z-scores themselves (on all rows):
if zscore_col_name_generated in analyzer.df.columns:
    print(f"\nGenerating summary table on Z-SCORES themselves (all rows) grouped by {SUMMARY_GROUP_COLS}...")
    summary_zscores_themselves = analyzer.generate_summary_table(
        groupby_cols=SUMMARY_GROUP_COLS,
        target_col=zscore_col_name_generated, 
        filter_type="zscore_flagged", 
        exclude_bad_rows=False, # Summarize all z-scores
        output_filename_suffix="dataset_zscore_values_all_rows"
    )
    if summary_zscores_themselves is not None:
        print(f"\n--- Summary Table (Z-score values from '{VALUE_COL_TO_PROCESS}') ---")
        print(summary_zscores_themselves)


# 2. Summary table on "clean" data (where is_bad == False from analyzer.df)
print(f"\nGenerating summary table on CLEAN data (is_bad == False from analyzer.df) after Z-score, grouped by {SUMMARY_GROUP_COLS}...")
summary_clean_flagged_data = analyzer.generate_summary_table(
    groupby_cols=SUMMARY_GROUP_COLS,
    target_col=VALUE_COL_TO_PROCESS, 
    filter_type="zscore_flagged", # Describes the state of analyzer.df
    exclude_bad_rows=True, # Default, but explicit here
    output_filename_suffix="dataset_zscore_flagged_cleaned_rows"
)
if summary_clean_flagged_data is not None:
    print(f"\n--- Summary Table (Clean Data after Z-score flagging, on '{VALUE_COL_TO_PROCESS}') ---")
    print(summary_clean_flagged_data)

print("\nScript finished.")

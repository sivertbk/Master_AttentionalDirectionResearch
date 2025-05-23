"""
Granular Channel-Level Statistics Script
-----------------------------------------

This script produces detailed statistical summaries at a granular level:
per dataset, per subject-session, and per channel.

It demonstrates:
1. Loading or creating an EEGAnalyzer instance specifically for these detailed stats.
2. Generating a summary table with statistics for each channel within each
   subject-session of each dataset.
"""
import pandas as pd

from utils.config import DATASETS
from eeg_analyzer.eeg_analyzer import EEGAnalyzer

# --- Configuration ---
ANALYZER_NAME = "GranularChannelStats"
ANALYZER_DESCRIPTION = (
    "This analyzer is used to generate highly granular summary statistics "
    "for band power, calculated per dataset, per subject-session, and per channel. "
    "This allows for detailed inspection of data at the lowest common levels before epoch aggregation."
)
VALUE_COL_TO_SUMMARIZE = 'band_power'
# Define the grouping for the granular summary table
GRANULAR_SUMMARY_GROUP_COLS = ['dataset', 'subject_session', 'channel'] 


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
    # Ensure 'is_bad' column exists if it's expected by generate_summary_table's default
    if 'is_bad' not in analyzer.df.columns:
        analyzer.df['is_bad'] = False
    analyzer.save_analyzer() 
    print(f"Analyzer '{ANALYZER_NAME}' created and saved with initial DataFrame.")
else:
    print(f"Successfully loaded existing analyzer: '{ANALYZER_NAME}'")
    if analyzer.df is None:
        print(f"Loaded analyzer '{ANALYZER_NAME}' has no DataFrame. Loading default or creating...")
        analyzer.load_dataframe()
        if analyzer.df is None:
            analyzer.create_dataframe()
            if 'is_bad' not in analyzer.df.columns:
                analyzer.df['is_bad'] = False
            analyzer.save_dataframe()
            analyzer.save_analyzer()
    elif 'is_bad' not in analyzer.df.columns: # df exists but no is_bad
        print(f"Adding missing 'is_bad' column to loaded DataFrame for analyzer '{ANALYZER_NAME}'.")
        analyzer.df['is_bad'] = False
        analyzer.save_dataframe() # Save df with new column
        analyzer.save_analyzer() # Save analyzer state


print(analyzer)
if analyzer.df is not None:
    print(f"DataFrame shape: {analyzer.df.shape}")
    if 'is_bad' in analyzer.df.columns:
        print(f"'is_bad' counts:\n{analyzer.df['is_bad'].value_counts(dropna=False)}")
else:
    print("Analyzer DataFrame is missing.")
    exit()

# --- Generate Granular Summary Table ---
# We'll generate this on all data, including any rows that might be flagged as 'is_bad'
# if this analyzer instance was previously processed. 
# If it's a fresh instance, 'is_bad' will be all False.

print(f"\nGenerating granular summary table for '{VALUE_COL_TO_SUMMARIZE}', grouped by {GRANULAR_SUMMARY_GROUP_COLS}...")
granular_summary = analyzer.generate_summary_table(
    groupby_cols=GRANULAR_SUMMARY_GROUP_COLS,
    target_col=VALUE_COL_TO_SUMMARIZE, 
    filter_type="unfiltered_or_raw", # Indicates the source data state for this summary
    exclude_bad_rows=False, # Important: Generate stats on ALL data for this specific table
    output_filename_suffix="granular_channel_summary_all_rows" 
)

if granular_summary is not None:
    print(f"\n--- Granular Summary Table (All Data, on '{VALUE_COL_TO_SUMMARIZE}') ---")
    print(granular_summary.head()) # Print head due to potentially large size
    print(f"Total rows in granular summary: {len(granular_summary)}")

# Optionally, generate a summary excluding bad rows if 'is_bad' has been populated by other processes
if 'is_bad' in analyzer.df.columns and analyzer.df['is_bad'].any():
    print(f"\nGenerating granular summary table (excluding bad rows) for '{VALUE_COL_TO_SUMMARIZE}', grouped by {GRANULAR_SUMMARY_GROUP_COLS}...")
    granular_summary_cleaned = analyzer.generate_summary_table(
        groupby_cols=GRANULAR_SUMMARY_GROUP_COLS,
        target_col=VALUE_COL_TO_SUMMARIZE,
        filter_type="processed_state", # Reflects current state of analyzer.df
        exclude_bad_rows=True, # Default, but explicit
        output_filename_suffix="granular_channel_summary_cleaned_rows"
    )
    if granular_summary_cleaned is not None:
        print(f"\n--- Granular Summary Table (Cleaned Data, on '{VALUE_COL_TO_SUMMARIZE}') ---")
        print(granular_summary_cleaned.head())
        print(f"Total rows in cleaned granular summary: {len(granular_summary_cleaned)}")

# Save the analyzer state (e.g., if description was updated or if new logs are important)
# analyzer.set_description(f"{ANALYZER_DESCRIPTION} - Granular summaries generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}.")
# analyzer.save_analyzer() # Uncomment if you want to save changes to the analyzer object itself

print("\nScript finished.")
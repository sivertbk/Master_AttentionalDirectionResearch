"""
DATASET STATISTICS:
This script produces statistical measures for analysis of datasets.
It includes functions to calculate the measures such as; mean, median,
and standard deviation (and more) of each dataset before, and after various 
filtering techniques. 

**Datasets is the top level object in the script**

This script produces statistical insights for better decision making.
"""
from utils.config import DATASETS
from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from eeg_analyzer.processor import Processor # Import Processor

# Define the name and description for the analyzer
ANALYZER_NAME = "DatasetSummaryTable"
ANALYZER_DESCRIPTION = (
    "This analyzer is intended for generating a summary table of statistics "
    "(mean, std, median, IQR, min, max, skewness, kurtosis, subject count, epoch count) "
    "aggregated at the dataset level. It will support creating these summaries for "
    "unfiltered data, as well as data processed with z-score and IQR filtering methods. "
    "The initial DataFrame is created from raw band power, and subsequent filtering and "
    "aggregation will be performed in dedicated analysis scripts."
)

# Attempt to load the analyzer
analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)

if analyzer is None:
    print(f"Analyzer '{ANALYZER_NAME}' not found. Creating a new one.")
    analyzer = EEGAnalyzer(
        DATASETS,
        analyzer_name=ANALYZER_NAME,
        description=ANALYZER_DESCRIPTION
    )
    # Create the initial full dataframe for this analyzer (unfiltered)
    analyzer.create_dataframe() 
    # Save the analyzer state immediately after creation
    analyzer.save_analyzer()
    print(f"Analyzer '{ANALYZER_NAME}' created and saved.")
else:
    print(f"Successfully loaded existing analyzer: '{ANALYZER_NAME}'")

print(analyzer) # Print the analyzer's string representation
print(f"DataFrame shape: {analyzer.df.shape if analyzer.df is not None else 'No DataFrame'}")

# --- Calculate and Display Dataset Level Statistics ---

if analyzer.df is not None and not analyzer.df.empty:
    group_by_cols_for_summary = ['dataset']
    value_col_for_summary = 'band_power'
    
    # --- 1. Unfiltered Data Summary ---
    print("\nCalculating dataset level statistics for unfiltered data...")
    dataset_summary_unfiltered = analyzer.generate_summary_table(
        groupby_cols=group_by_cols_for_summary,
        target_col=value_col_for_summary,
        filter_type="unfiltered",
        source_df=analyzer.df # Explicitly pass the source df
    )
    if dataset_summary_unfiltered is not None:
        print("\n--- Unfiltered Dataset Summary Table ---")
        print(dataset_summary_unfiltered)

    # Define grouping for outlier detection (more granular)
    filter_group_cols = ['dataset', 'subject_session', 'channel', 'state']

    # --- 2. Z-score Filtered Data Summary ---
    print("\nApplying Z-score filtering...")
    df_zscore_filtered = Processor.filter_outliers_zscore(
        analyzer.df, 
        group_cols=filter_group_cols, 
        value_col=value_col_for_summary, 
        threshold=3.0
    )
    print(f"Shape after Z-score filtering: {df_zscore_filtered.shape}")
    # Removed: analyzer._log_event("Z-score Filtering Applied", { ... })
    # This information is implicitly logged by generate_summary_table via 'filter_type'

    if not df_zscore_filtered.empty:
        print("\nCalculating dataset level statistics for Z-score filtered data...")
        dataset_summary_zscore = analyzer.generate_summary_table(
            groupby_cols=group_by_cols_for_summary,
            target_col=value_col_for_summary,
            filter_type="zscore_filtered",
            source_df=df_zscore_filtered
        )
        if dataset_summary_zscore is not None:
            print("\n--- Z-score Filtered Dataset Summary Table ---")
            print(dataset_summary_zscore)
    else:
        print("DataFrame is empty after Z-score filtering. Skipping summary table generation.")


    # --- 3. IQR Filtered Data Summary ---
    print("\nApplying IQR filtering...")
    df_iqr_filtered = Processor.filter_outliers_iqr(
        analyzer.df, 
        group_cols=filter_group_cols, 
        value_col=value_col_for_summary, 
        multiplier=1.5
    )
    print(f"Shape after IQR filtering: {df_iqr_filtered.shape}")
    # Removed: analyzer._log_event("IQR Filtering Applied", { ... })
    # This information is implicitly logged by generate_summary_table via 'filter_type'

    if not df_iqr_filtered.empty:
        print("\nCalculating dataset level statistics for IQR filtered data...")
        dataset_summary_iqr = analyzer.generate_summary_table(
            groupby_cols=group_by_cols_for_summary,
            target_col=value_col_for_summary,
            filter_type="iqr_filtered",
            source_df=df_iqr_filtered
        )
        if dataset_summary_iqr is not None:
            print("\n--- IQR Filtered Dataset Summary Table ---")
            print(dataset_summary_iqr)
    else:
        print("DataFrame is empty after IQR filtering. Skipping summary table generation.")

    # Save the analyzer state as filtering operations might be considered significant
    # or if any new attributes were added to the analyzer itself (not the case here).
    # For now, the main df in analyzer is unchanged, so saving might not be strictly necessary
    # unless the logs themselves are critical to persist with the .pkl object.
    analyzer.save_analyzer()

else:
    print("DataFrame is not available or is empty. Cannot calculate statistics.")

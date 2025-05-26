from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS


def generate_all_summary_tables(analyzer: EEGAnalyzer, 
                                base_filter_type_label: str, 
                                target_value_col: str = 'band_power'):
    """
    Generates a suite of summary tables at different aggregation levels.

    Parameters:
    - analyzer (EEGAnalyzer): An initialized EEGAnalyzer object with data loaded.
    - base_filter_type_label (str): A label describing the state of analyzer.df 
                                   (e.g., "unprocessed", "zscore_flagged").
    - target_value_col (str): The primary value column to summarize (e.g., 'band_power').
    """
    if analyzer.df is None or analyzer.df.empty:
        print(f"Analyzer DataFrame for '{analyzer.analyzer_name}' is empty. Attempting to create/load.")
        analyzer.create_dataframe() 
        if analyzer.df is None or analyzer.df.empty:
            print(f"Failed to create/load DataFrame for '{analyzer.analyzer_name}'. Cannot generate summary tables.")
            return
        # Ensure 'is_bad' column exists after creation/loading
        if 'is_bad' not in analyzer.df.columns:
            analyzer.df['is_bad'] = False
            print("Added 'is_bad' column to newly created/loaded DataFrame.")
        analyzer.save_dataframe() # Save if it was just created
        analyzer.save_analyzer() # Save analyzer state

    print(f"\n--- Generating All Summary Tables for Analyzer: {analyzer.analyzer_name} ---")
    print(f"--- Data State (filter_type): {base_filter_type_label} ---")

    # Define configurations for each summary table type
    table_configs = [
        {
            "level_name": "Dataset Level",
            "groupby_cols": ['dataset'],
            "output_filename_suffix": "dataset_level",
            "perform_state_comparison": True,
        },
        {
            "level_name": "Subject-Session Level",
            "groupby_cols": ['dataset', 'subject_session'],
            "output_filename_suffix": "subses_level",
            "perform_state_comparison": True,
        },
        {
            "level_name": "Channel Level (Aggregated across subjects)",
            "groupby_cols": ['dataset', 'channel', 'hemisphere', 'cortical_region'], 
            "output_filename_suffix": "channel_level_agg",
            "perform_state_comparison": True,
        },
        {
            "level_name": "ROI Level - Full Region (Aggregated across subjects)",
            "groupby_cols": ['dataset', 'cortical_region'], 
            "output_filename_suffix": "roi_full_region_level_agg",
            "perform_state_comparison": True,
        },
        {
            "level_name": "ROI Level - Hemisphere Specific (Aggregated across subjects)",
            "groupby_cols": ['dataset', 'hemisphere', 'cortical_region'], 
            "output_filename_suffix": "roi_hemi_specific_level_agg",
            "perform_state_comparison": True,
        },
        {
            "level_name": "Subject-Session-Channel Level",
            "groupby_cols": ['dataset', 'subject_session', 'channel', 'hemisphere', 'cortical_region'], 
            "output_filename_suffix": "subses_channel_level",
            "perform_state_comparison": True,
        },
        {
            "level_name": "Subject-Session-ROI Level - Full Region",
            "groupby_cols": ['dataset', 'subject_session', 'cortical_region'], 
            "output_filename_suffix": "subses_roi_full_region_level",
            "perform_state_comparison": True,
        },
        {
            "level_name": "Subject-Session-ROI Level - Hemisphere Specific",
            "groupby_cols": ['dataset', 'subject_session', 'hemisphere', 'cortical_region'], 
            "output_filename_suffix": "subses_roi_hemi_specific_level",
            "perform_state_comparison": True,
        }
    ]

    # Check for required columns
    all_required_cols = set([target_value_col, 'state', 'is_bad'])
    for config in table_configs:
        all_required_cols.update(config['groupby_cols'])
    
    missing_cols = [col for col in all_required_cols if col not in analyzer.df.columns]
    if missing_cols:
        print(f"CRITICAL ERROR: Missing required columns in DataFrame for analyzer '{analyzer.analyzer_name}': {missing_cols}.")
        print("Please ensure 'cortical_region' and 'hemisphere' are in the DataFrame for ROI tables.")
        print("Other missing columns might indicate issues with DataFrame creation or column naming.")
        return

    output_subfolder = "summary_tables" # All tables go into this subfolder

    for config in table_configs:
        print(f"\nGenerating: {config['level_name']} Summary Table")
        
        # Generate table including all rows (even if flagged bad)
        analyzer.generate_summary_table(
            groupby_cols=config['groupby_cols'],
            target_col=target_value_col,
            output_filename_suffix=config['output_filename_suffix'],
            filter_type=base_filter_type_label,
            exclude_bad_rows=False, 
            perform_state_comparison=config['perform_state_comparison'],
            output_subfolder=output_subfolder
        )

        # Generate table excluding bad rows (if 'is_bad' column has any True values)
        if 'is_bad' in analyzer.df.columns and analyzer.df['is_bad'].any():
            print(f"Generating: {config['level_name']} Summary Table (Cleaned Data)")
            analyzer.generate_summary_table(
                groupby_cols=config['groupby_cols'],
                target_col=target_value_col,
                output_filename_suffix=config['output_filename_suffix'], # Suffix will be added by generate_summary_table
                filter_type=base_filter_type_label, 
                exclude_bad_rows=True,
                perform_state_comparison=config['perform_state_comparison'],
                output_subfolder=output_subfolder
            )
        elif not ('is_bad' in analyzer.df.columns and analyzer.df['is_bad'].any()):
             print(f"Skipping cleaned version for {config['level_name']}; no 'is_bad' rows or 'is_bad' column missing/all False.")


if __name__ == "__main__":
    # --- CHOOSE THE ANALYZER TO PROCESS ---
    # Option 1: Use an analyzer with unprocessed data
    ANALYZER_NAME = "UnprocessedAnalyzer" 
    ANALYZER_FILTER_TYPE_LABEL = "unprocessed"

    # Option 2: Use an analyzer that has undergone Z-score flagging
    # ANALYZER_NAME_TO_PROCESS = "ZScoreProcessedData"
    # ANALYZER_FILTER_TYPE_LABEL = "zscore_flagged"
    
    # Option 3: Use an analyzer that has undergone IQR flagging
    # ANALYZER_NAME_TO_PROCESS = "IQRProcessedData"
    # ANALYZER_FILTER_TYPE_LABEL = "iqr_flagged"

    # --- ---

    print(f"Attempting to load EEGAnalyzer instance: {ANALYZER_NAME}")
    analyzer_instance = EEGAnalyzer.load_analyzer(analyzer_name=ANALYZER_NAME)

    if analyzer_instance is None:
        print(f"Analyzer '{ANALYZER_NAME}' not found.")
    else:
        print(f"Successfully loaded existing analyzer: '{analyzer_instance.analyzer_name}'")

    if analyzer_instance:
        # Ensure DataFrame is loaded and 'is_bad' column exists
        if analyzer_instance.df is None or analyzer_instance.df.empty:
            print(f"DataFrame not found in loaded analyzer '{analyzer_instance.analyzer_name}'. Attempting to load/create.")
            analyzer_instance.load_dataframe() # Try loading default CSV
            if analyzer_instance.df is None or analyzer_instance.df.empty:
                analyzer_instance.create_dataframe()
                print("DataFrame created.")
            if analyzer_instance.df is not None and 'is_bad' not in analyzer_instance.df.columns:
                analyzer_instance.df['is_bad'] = False
                print("Added 'is_bad' column to DataFrame.")
            analyzer_instance.save_dataframe() # Save if changed
            analyzer_instance.save_analyzer()  # Save analyzer if df was loaded/created
        elif 'is_bad' not in analyzer_instance.df.columns:
             analyzer_instance.df['is_bad'] = False
             print("Added 'is_bad' column to existing DataFrame.")
             analyzer_instance.save_dataframe()
             analyzer_instance.save_analyzer()


        if analyzer_instance.df is not None and not analyzer_instance.df.empty:
            generate_all_summary_tables(analyzer_instance, 
                                        base_filter_type_label=ANALYZER_FILTER_TYPE_LABEL)
        else:
            print(f"Could not obtain DataFrame for analyzer '{analyzer_instance.analyzer_name}'. Cannot proceed.")



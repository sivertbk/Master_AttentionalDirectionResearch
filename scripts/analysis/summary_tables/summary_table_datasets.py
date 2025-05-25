from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS, EEGANALYZER_OBJECT_DERIVATIVES_PATH # For loading analyzer


def generate_dataset_summary_tables(analyzer: EEGAnalyzer, filter_type_label: str = "raw"):
    """
    Generates dataset-level summary tables using the EEGAnalyzer instance.

    Parameters:
    - analyzer (EEGAnalyzer): An initialized EEGAnalyzer object with data loaded.
    - filter_type_label (str): A label for the data state (e.g., "raw", "processed_zscore").
                               This is used in the output filename.
    """
    if analyzer.df is None or analyzer.df.empty:
        print(f"Analyzer DataFrame is empty. Cannot generate summary tables. analyzer name: {analyzer.analyzer_name}")
        analyzer.create_dataframe() # Attempt to create if not existing
        if analyzer.df is None or analyzer.df.empty:
             print(f"Failed to create DataFrame. Exiting. analyzer name: {analyzer.analyzer_name}")
             return

    print(f"\n--- Generating Dataset Summary Table ({filter_type_label} data) ---")
    
    # Define grouping for DatasetSummaryTable
    # Assuming 'dataset_name' column exists in analyzer.df
    dataset_groupby_cols = ['dataset_name'] 
    
    # Check if necessary columns exist
    required_cols_for_grouping = dataset_groupby_cols + ['state', 'band_power'] # state and band_power are defaults
    missing_cols = [col for col in required_cols_for_grouping if col not in analyzer.df.columns]
    if missing_cols:
        print(f"Missing required columns in DataFrame: {missing_cols}. Cannot generate dataset summary.")
        return

    summary_df_dataset = analyzer.generate_summary_table(
        groupby_cols=dataset_groupby_cols,
        target_col='band_power', # As per user's spec, this is the value_col
        output_filename_suffix=f"dataset_level_{filter_type_label}",
        filter_type=filter_type_label, # Pass the label for consistent naming
        exclude_bad_rows=True, # Default, can be parameterized if needed
        perform_state_comparison=True, # Generate detailed stats with OT/MW
        positive_state='OT', # Standard positive state
        negative_state='MW'  # Standard negative state
    )

    if summary_df_dataset is not None:
        print(f"Dataset summary table generated successfully for {analyzer.analyzer_name}.")
        print(summary_df_dataset.head())
    else:
        print(f"Failed to generate dataset summary table for {analyzer.analyzer_name}.")

if __name__ == "__main__":
    # This is an example of how to use the function.
    # You'll need to specify the name of the EEGAnalyzer instance to load.
    # Replace 'your_analyzer_name' with the actual name of your analyzer instance.
    analyzer_name_to_load = "UnfilteredSummary"

    print(f"Attempting to load EEGAnalyzer instance: {analyzer_name_to_load}")
    analyzer = EEGAnalyzer.load_analyzer(analyzer_name=analyzer_name_to_load)

    if analyzer:
        print(f"Successfully loaded EEGAnalyzer: {analyzer.analyzer_name}")
        
        # Example: Generate summary tables for the "raw" (as-loaded) data
        # Ensure analyzer.df is populated. If not, create it.
        if analyzer.df is None or analyzer.df.empty:
            print("DataFrame not found in loaded analyzer. Attempting to create it.")
            analyzer.create_dataframe() # Default freq_band=(8,12)
            if analyzer.df is None or analyzer.df.empty:
                print("Failed to create DataFrame. Cannot proceed.")
            else:
                print("DataFrame created successfully.")
                generate_dataset_summary_tables(analyzer, filter_type_label="raw_initial_df")
        else:
            print("Using existing DataFrame from loaded analyzer.")
            generate_dataset_summary_tables(analyzer, filter_type_label="loaded_df")

        # Later, if you process the data (e.g., apply outlier flagging),
        # you can call this function again with a different filter_type_label:
        # Example:
        # analyzer.apply_zscore_flagging(group_cols=['dataset_name', 'channel'], value_col='band_power')
        # analyzer.save_dataframe(filename="df_after_zscore.csv") # Optional: save processed df
        # generate_dataset_summary_tables(analyzer, filter_type_label="processed_zscore")

    else:
        print(f"Could not load EEGAnalyzer instance: {analyzer_name_to_load}. "
              f"Please ensure an analyzer with this name exists in: {EEGANALYZER_OBJECT_DERIVATIVES_PATH}")
        print("To run this script, you need a saved EEGAnalyzer object. ")
        print("You might need to run a script that creates and saves an EEGAnalyzer instance first.")


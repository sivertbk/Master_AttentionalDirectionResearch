from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS


def run_summary_table_generation(analyzer: EEGAnalyzer, 
                                 base_filter_type_label: str, 
                                 target_value_col: str = 'band_db'):
    """
    Calls the EEGAnalyzer method to generate a suite of standard summary tables.

    Parameters:
    - analyzer (EEGAnalyzer): An initialized EEGAnalyzer object.
    - base_filter_type_label (str): A label describing the state of analyzer.df 
                                   (e.g., "unprocessed", "zscore_flagged").
    - target_value_col (str): The primary value column to summarize (e.g., 'band_power').
    """
    if not isinstance(analyzer, EEGAnalyzer):
        print("Error: Invalid analyzer object provided.")
        return

    print(f"\n--- Requesting Standard Summary Table Generation for Analyzer: {analyzer.analyzer_name} ---")
    analyzer.generate_standard_summary_tables(
        base_filter_type_label=base_filter_type_label,
        target_value_col=target_value_col
    )
    print(f"\n--- Finished Standard Summary Table Generation for Analyzer: {analyzer.analyzer_name} ---")


if __name__ == "__main__":
    # --- CHOOSE THE ANALYZER TO PROCESS ---
    # Option 1: Use an analyzer with unprocessed data
    ANALYZER_NAME = "UnprocessedAnalyzer" 
    ANALYZER_FILTER_TYPE_LABEL = "unprocessed"

    # Option 2: Use an analyzer that has undergone Z-score flagging
    # ANALYZER_NAME = "ZScoreProcessedData" # Example if you create such an analyzer
    # ANALYZER_FILTER_TYPE_LABEL = "zscore_flagged"
    
    # Option 3: Use an analyzer that has undergone IQR flagging
    # ANALYZER_NAME = "IQRProcessedData" # Example
    # ANALYZER_FILTER_TYPE_LABEL = "iqr_flagged"

    # --- ---

    print(f"Attempting to load EEGAnalyzer instance: {ANALYZER_NAME}")
    analyzer_instance = EEGAnalyzer.load_analyzer(analyzer_name=ANALYZER_NAME)

    if analyzer_instance is None:
        print(f"Analyzer '{ANALYZER_NAME}' not found. You might need to create it first if it doesn't exist.")
        # Example: Create if not found (adjust dataset_configs as needed)
        # print(f"Creating a new analyzer: {ANALYZER_NAME}")
        # from utils.config import DATASETS # Ensure DATASETS is defined with your dataset configurations
        # analyzer_instance = EEGAnalyzer(dataset_configs=DATASETS, analyzer_name=ANALYZER_NAME)
        # analyzer_instance.set_description(f"Analyzer for {ANALYZER_FILTER_TYPE_LABEL} data.")
        # # analyzer_instance.create_dataframe() # Create df immediately
        # # analyzer_instance.save_dataframe()
        # analyzer_instance.save_analyzer()
        # print(f"New analyzer '{ANALYZER_NAME}' created and saved. Please re-run the script if you want to generate tables now.")
    else:
        print(f"Successfully loaded existing analyzer: '{analyzer_instance.analyzer_name}'")

        # The generate_standard_summary_tables method now handles DataFrame loading/creation if needed.
        run_summary_table_generation(analyzer_instance, 
                                     base_filter_type_label=ANALYZER_FILTER_TYPE_LABEL)
        
        # Optionally, save the analyzer again if its state might have changed (e.g., df hash updated)
        # This is good practice if generate_standard_summary_tables modified the analyzer (like updating the hash)
        # or if it created/loaded a df.
        print(f"Saving analyzer '{analyzer_instance.analyzer_name}' state after summary table generation attempt.")
        analyzer_instance.save_analyzer()



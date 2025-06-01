import pandas as pd
import matplotlib.pyplot as plt
import os

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from eeg_analyzer.statistics import Statistics



def perform_wilcoxon_test(file_names):
    """
    Loads data from a summary CSV files and performs Wilcoxon signed-rank tests
    to compare state-dependent power trends (MW vs. OT) for each subject_session,
    grouped by dataset and identifier (channel and ROIs).
    """
    base_required_cols = ['dataset', 'subject_session', 'mean_MW', 'mean_OT']

    for csv_file_name in file_names:
        print(f"\nProcessing file: {csv_file_name}...")
        # Construct the path to the CSV file relative to the project root
        csv_file_path = os.path.join(path_to_analyzer, "summary_tables", csv_file_name)

        # Load the dataframe
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"Error: The file {csv_file_path} was not found. Skipping.")
            continue
        except Exception as e:
            print(f"Error loading CSV file '{csv_file_path}': {e}. Skipping.")
            continue

        # Determine group_cols and analysis_key_suffix based on csv_file_name
        current_group_cols = []
        analysis_key_suffix = ""
        if "channel_level" in csv_file_name:
            current_group_cols = ['channel']
            analysis_key_suffix = "by_channel"
        elif "roi_full_region_level" in csv_file_name:
            current_group_cols = ['cortical_region']
            analysis_key_suffix = "by_roi_full_region"
        elif "roi_hemi_specific_level" in csv_file_name:
            current_group_cols = ['cortical_region', 'hemisphere']
            analysis_key_suffix = "by_roi_hemi_specific"
        else:
            print(f"Warning: Could not determine grouping strategy for {csv_file_name}. Skipping.")
            continue
        
        print(f"Using group_cols: {current_group_cols} for {csv_file_name}")

        # Validate required columns
        required_cols = list(set(base_required_cols + current_group_cols))
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: The CSV file {csv_file_name} is missing required columns: {missing_cols}.")
            print(f"Available columns: {df.columns.tolist()}. Skipping this file.")
            continue

        # Initialize a dictionary to store significance results for  greater/less -> each dataset -> channel/ROI
        significance = {"greater": {}, "less": {}}

        analysis_params = {
            "col1": 'mean_OT',
            "col2": 'mean_MW',
            "group_cols": current_group_cols,
            "n_permutations": 9999,
            "random_state": 42,
            "source_csv_file_name": csv_file_name 
        }

        # Get unique datasets
        datasets = df['dataset'].unique()
        for dataset_name in datasets:
            df_dataset = df[df['dataset'] == dataset_name].copy()

            # testing for both tail sides
            results_greater = Statistics.perform_wilcoxon_test(
                df=df_dataset,
                col1=analysis_params['col1'],
                col2=analysis_params['col2'],
                group_cols=analysis_params['group_cols'],
                tail='greater',
                n_permutations=analysis_params['n_permutations'],
                random_state=analysis_params['random_state']
            )
            
            significance['greater'][dataset_name] = results_greater

            # testing for less tail side
            results_less = Statistics.perform_wilcoxon_test(
                df=df_dataset,
                col1=analysis_params['col1'],
                col2=analysis_params['col2'],
                group_cols=analysis_params['group_cols'],
                tail='less',
                n_permutations=analysis_params['n_permutations'],
                random_state=analysis_params['random_state']
            )
            significance['less'][dataset_name] = results_less

        # Print the significance results for the current file
        print(f"\nSignificance results for {csv_file_name}:")
        for tail, results in significance.items():
            print(f"  Tail '{tail}':")
            for dataset, stats_df in results.items():
                if stats_df is not None and not stats_df.empty:
                    print(f"    Dataset: {dataset}")
                    print(stats_df.to_string()) # Pretty print DataFrame
                elif stats_df is not None and stats_df.empty:
                    print(f"    Dataset: {dataset} - No significant results or data.")
                else:
                    print(f"    Dataset: {dataset} - Results are None.")


        # Store the results in the analyzer object
        analysis_name = f"wilcoxon_meanOT_vs_meanMW_{analysis_key_suffix}"
        analyzer.store_statistical_analysis(
            analysis_name=analysis_name,
            data=significance,
            parameters=analysis_params
        )
        print(f"Stored results for {csv_file_name} under analysis name: {analysis_name}")

    # Save the analyzer object once after processing all files
    analyzer.save_analyzer()
    print(f"\nAll Wilcoxon test results stored in analyzer object '{analyzer.analyzer_name}' and saved.")


if __name__ == "__main__":
    # CSV's to be used for the analysis
    # csv_file_names = ["summary_subses_channel_level_IQR_Filtered_band_power_cleaned_statecomp.csv", 
    #                   "summary_subses_roi_full_region_level_IQR_Filtered_band_power_cleaned_statecomp.csv", 
    #                   "summary_subses_roi_hemi_specific_level_IQR_Filtered_band_power_cleaned_statecomp.csv"]
    #csv_file_names = ["summary_subses_channel_level_unprocessed_band_power_statecomp.csv", 
    #                  "summary_subses_roi_full_region_level_unprocessed_band_power_statecomp.csv", 
    #                  "summary_subses_roi_hemi_specific_level_unprocessed_band_power_statecomp.csv"]
    csv_file_names = ["summary_subses_channel_level_unprocessed_band_db_statecomp.csv", 
                     "summary_subses_roi_full_region_level_unprocessed_band_db_statecomp.csv", 
                     "summary_subses_roi_hemi_specific_level_unprocessed_band_db_statecomp.csv"]

    analyzer = EEGAnalyzer.load_analyzer("UnprocessedAnalyzer")
    path_to_analyzer = analyzer.derivatives_path
    perform_wilcoxon_test(csv_file_names)
import pandas as pd
import matplotlib.pyplot as plt
import os

from eeg_analyzer.eeg_analyzer import EEGAnalyzer

analyzer = EEGAnalyzer.load_analyzer("UnprocessedAnalyzer")

path_to_analyzer = analyzer.derivatives_path


def plot_state_power_trends():
    """
    Loads data from a summary CSV file and plots state-dependent power trends
    (MW vs. OT) for each subject_session, grouped by dataset and channel.
    Includes standard deviation as error bars.
    """
    # Construct the path to the CSV file relative to the project root
    csv_file_name = "summary_subses_channel_level_unprocessed_band_power_statecomp.csv"
    csv_file_path = os.path.join(path_to_analyzer, "summary_tables", csv_file_name)

    # Load the dataframe
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        print("Please ensure the CSV file exists at the specified path.")
        return
    except Exception as e:
        print(f"Error loading CSV file '{csv_file_path}': {e}")
        return

    # Validate required columns
    required_cols = ['dataset', 'channel', 'subject_session', 'mean_MW', 'mean_OT', 'std_MW', 'std_OT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The CSV file is missing required columns: {missing_cols}.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Get unique datasets
    datasets = df['dataset'].unique()

    for dataset_name in datasets:
        df_dataset = df[df['dataset'] == dataset_name].copy()
        
        channels = df_dataset['channel'].unique()

        for channel_name in channels:
            df_channel_specific = df_dataset[df_dataset['channel'] == channel_name]

            if df_channel_specific.empty:
                print(f"No data found for Dataset: {dataset_name}, Channel: {channel_name}. Skipping.")
                continue

            plt.figure(figsize=(12, 7))
            
            states = ['MW', 'OT'] # X-axis categories

            for _, row in df_channel_specific.iterrows():
                subject_session_id = row['subject_session']
                mean_values = [row['mean_MW'], row['mean_OT']]
                std_values = [row['std_MW'], row['std_OT']]
                
                plt.plot(states, mean_values, marker='o', linestyle='-', label=f"{subject_session_id}")
                plt.errorbar(states, mean_values, yerr=std_values, fmt='none', ecolor='gray', capsize=4, alpha=0.6, elinewidth=1.5)

            plt.title(f'Alpha Power Trend (MW to OT)\nDataset: {dataset_name} - Channel: {channel_name}', fontsize=14)
            plt.xlabel('Cognitive State', fontsize=12)
            plt.ylabel('Mean Alpha Power (e.g., µV²)', fontsize=12) # Added unit placeholder
            plt.xticks(states, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.7)
            
            num_lines = len(df_channel_specific)
            if num_lines > 10 and num_lines * 0.8 > plt.gcf().get_size_inches()[1] : # Heuristic for moving legend
                ax = plt.gca()
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.12, box.width, box.height * 0.88])
                
                legend_ncol = min(5, (num_lines + 4) // 5) 
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), 
                           fancybox=True, shadow=False, ncol=legend_ncol, fontsize=9)
            else:
                plt.legend(fontsize=9)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap
            plt.savefig(os.path.join(path_to_analyzer, "plots", f"{dataset_name}_{channel_name}_state_power_trends.png"), dpi=300)
            plt.close()  # Close the figure to free memory
            print(f"Plot saved for Dataset: {dataset_name}, Channel: {channel_name}.")

if __name__ == '__main__':
    plot_state_power_trends()

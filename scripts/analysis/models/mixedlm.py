import numpy as np
import pandas as pd
import os
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test
from mne import create_info
from mne.viz import plot_topomap
import matplotlib.pyplot as plt

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS, set_plot_style

set_plot_style()  # Set the plotting style for matplotlib





if __name__ == "__main__":

    ANALYZER_NAME = "eeg_analyzer_test"

    # Trying to load the EEGAnalyzer
    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(DATASETS, ANALYZER_NAME)

    analyzer.save_analyzer()  # Ensure the analyzer is saved

    df = pd.DataFrame()

    for dataset in analyzer: 
        for subject in dataset: 
            for recording in subject:
                if recording.exclude:
                    print(f"Skipping excluded recording for subject {subject.id} in dataset {dataset.name}.")
                    continue
                # make structured dictionary for each recording
                # # This will be used to create a DataFrame later 
                mean_OT = recording.get_stat('mean',
                                            data_type='log_band_power',
                                            state='OT',
                                            filtered=True)   
                mean_MW = recording.get_stat('mean',
                                            data_type='log_band_power',
                                            state='MW',
                                            filtered=True)
                mean_diff = mean_OT - mean_MW
                data_dict = {
                    "dataset": dataset.name,
                    "subject": subject.id,
                    "group": subject.group,
                    "session": recording.session_id,
                    "task_orientation": dataset.task_orientation,
                    "mean_OT_alpha": mean_OT,
                    "mean_MW_alpha": mean_MW,
                    "mean_diff": mean_diff,
                }
                # Add the data_dict to the DataFrame
                df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
                

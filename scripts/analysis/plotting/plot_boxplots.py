import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import EEGANALYZER_SETTINGS, set_plot_style, PLOTS_PATH

set_plot_style()

if __name__ == "__main__":
    eeganalyzer_kwargs = EEGANALYZER_SETTINGS.copy()
    ANALYZER_NAME = EEGANALYZER_SETTINGS["analyzer_name"]

    # Trying to load the EEGAnalyzer
    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(**eeganalyzer_kwargs)
        analyzer.save_analyzer()
    
    # Creating a DataFrame with the data
    analyzer.create_dataframe(exclude_bad_recordings=False, exclude_outliers=False)

    analyzer.viz.plot_boxplot(group_by_region=True, output_subfolder='boxplots_unfiltered')
    analyzer.viz.plot_boxplot(output_subfolder='boxplots_unfiltered',)

    analyzer.create_dataframe(exclude_bad_recordings=False, exclude_outliers=True)

    analyzer.viz.plot_boxplot(group_by_region=True, output_subfolder='boxplots_filtered')
    analyzer.viz.plot_boxplot(output_subfolder='boxplots_filtered')

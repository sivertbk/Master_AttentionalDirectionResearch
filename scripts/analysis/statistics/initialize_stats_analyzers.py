"""
This script makes objects for each desired statistical measure dataframe to be extracted.


"""
import pandas as pd

from utils.config import DATASETS
from eeg_analyzer.eeg_analyzer import EEGAnalyzer

# Creating 
# 1. Analyzer for detailed unfiltered statistics per channel/state
analyzer_unfiltered_stats_name = "UnfilteredChannelStateStats"
analyzer_unfiltered_stats_desc = (
    "This analyzer is intended for extracting detailed statistical measures "
    "(Mean, Std Dev, Median, IQR, Min-Max, Skewness, Kurtosis) "
    "from the unfiltered band power data. Statistics are to be calculated per dataset, "
    "subject-session, channel, and state. The initial DataFrame will be created "
    "using the default to_long_band_power_list() method."
)

analyzer_unfiltered = EEGAnalyzer(
    DATASETS, 
    analyzer_name=analyzer_unfiltered_stats_name, 
    description=analyzer_unfiltered_stats_desc
)

# Create the initial full dataframe for this analyzer
analyzer_unfiltered.create_dataframe() # Using default alpha band (8-12 Hz) for now, can be specified if needed

# Save the analyzer state so it can be loaded in other scripts
analyzer_unfiltered.save_analyzer()

print(f"\nAnalyzer '{analyzer_unfiltered.analyzer_name}' created and saved.")
print(analyzer_unfiltered.get_description())
print(f"DataFrame shape: {analyzer_unfiltered.df.shape if analyzer_unfiltered.df is not None else 'No DataFrame'}")


# 2. Analyzer for Dataset Summary Table
analyzer_dataset_summary_name = "DatasetSummaryTable"
analyzer_dataset_summary_desc = (
    "This analyzer is intended for generating a summary table of statistics "
    "(mean, std, median, IQR, min, max, skewness, kurtosis, subject count, epoch count) "
    "aggregated at the dataset level. It will support creating these summaries for "
    "unfiltered data, as well as data processed with z-score and IQR filtering methods. "
    "The initial DataFrame is created from raw band power, and subsequent filtering and "
    "aggregation will be performed in dedicated analysis scripts."
)

analyzer_dataset_summary = EEGAnalyzer(
    DATASETS,
    analyzer_name=analyzer_dataset_summary_name,
    description=analyzer_dataset_summary_desc
)

# Create the initial full dataframe for this analyzer (unfiltered)
# Specific filtering (z-score, IQR) will be applied in the analysis script that uses this object.
analyzer_dataset_summary.create_dataframe() 

# Save the analyzer state
analyzer_dataset_summary.save_analyzer()

print(f"\nAnalyzer '{analyzer_dataset_summary.analyzer_name}' created and saved.")
print(analyzer_dataset_summary.get_description())
print(f"DataFrame shape: {analyzer_dataset_summary.df.shape if analyzer_dataset_summary.df is not None else 'No DataFrame'}")


# You can add more analyzer initializations here for other statistical tasks
# analyzer_roi_stats_name = "ROIAggregatedStats"
# analyzer_roi_stats_desc = "Analyzer for statistics aggregated by ROI..."
# analyzer_roi = EEGAnalyzer(DATASETS, analyzer_name=analyzer_roi_stats_name, description=analyzer_roi_stats_desc)
# analyzer_roi.create_dataframe()
# analyzer_roi.save_analyzer()
# print(f"\nAnalyzer '{analyzer_roi.analyzer_name}' created and saved.")

from eeg_analyzer.eeg_analyzer import EEGAnalyzer

ANALYSIS_NAME = "VarianceAnalysis"
ANALYZER_DESCRIPTION = (
    "Variance analysis of EEG data across different datasets, subjects, "
    "sessions, channels, ROI, tasks, states, and groups (for braboszcz  "
    "et al. dataset). This analysis focuses on the variance of band power "
    "across these dimensions to identify significant differences in EEG "
    "activity. The pipeline includes loading the EEGAnalyzer instance, "
    "performing a simple variance analysis, and saving the results. Then,"
    "Do some filtering of the data including removing outliers, excluding "
    "subjects with potential bad data quality, and applying "
    "z-score normalization where applicable. Finally, it generates "
    "summary tables for the variance analysis results."
)
# Load the EEGAnalyzer instance
analyzer = EEGAnalyzer(analysis_name=ANALYSIS_NAME)
if analyzer is None:
    #create a new instance if it does not exist
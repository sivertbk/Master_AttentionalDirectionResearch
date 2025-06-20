from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS

def gather_data(analyzer: EEGAnalyzer = None):
    """
    Gathers the alpha power data in micro volts squared per Hz and decibels
    for each subject and channel in the EEG data.
    """
    if analyzer is None:
        raise ValueError("Analyzer must be provided.")

    data = []
    for dataset in analyzer:
        for subject in dataset:
            for recording in subject:
                if recording.exclude:
                    continue
                # Gather alpha power data
                alpha_power = recording.get_band_power()
    return data

def plot_histogram(analyzer: EEGAnalyzer = None):
    """
    Plots the distribution of alpha power in micro volts squared per Hz and decibels
    for each subject and channel in the EEG data.
    """



if __name__ == "__main__":
    ANALYZER_NAME = "eeg_analyzer_test"

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(DATASETS, ANALYZER_NAME)
        analyzer.save_analyzer()

    data = gather_data(analyzer)
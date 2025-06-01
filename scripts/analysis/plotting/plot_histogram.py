from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS


def plot_histogram(analyzer: EEGAnalyzer = None):
    """
    Plots the distribution of alpha power in micro volts squared per Hz and decibels
    for each subject and channel in the EEG data.
    """
    if analyzer is None:
        raise ValueError("EEGAnalyzer instance is required to plot histogram.")

    for dataset_name, dataset in analyzer.datasets.items():
        print(f"Processing dataset: {dataset_name}")
        for subject_id, subject in dataset.subjects.items():
            for session_id, session in subject.recordings.items():
                print(f"Subject: {subject_id}, Session: {session_id}")
                # Plot histogram
                session.plot_distribution(show=False)
                print(f"Histogram plots for {subject_id} - {session_id} completed.")


if __name__ == "__main__":
    ANALYZER_NAME = "UnprocessedAnalyzer"
    analyzer = EEGAnalyzer(DATASETS)
    plot_histogram(analyzer)
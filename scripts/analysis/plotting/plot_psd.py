from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS


def plot_psd(analyzer: EEGAnalyzer = None):
    """
    Plots the Power Spectral Density (PSD) for each subject in the EEG data.
    """
    if analyzer is None:
        raise ValueError("EEGAnalyzer instance is required to plot PSD.")

    for dataset_name, dataset in analyzer.datasets.items():
        print(f"Processing dataset: {dataset_name}")
        for subject_id, subject in dataset.subjects.items():
            for session_id, session in subject.recordings.items():
                print(f"Subject: {subject_id}, Session: {session_id}")
                # Plot PSD
                session.plot_psd(show=False)
                print(f"PSD plots for {subject_id} - {session_id} completed.")


if __name__ == "__main__":
    ANALYZER_NAME = "UnprocessedAnalyzer"
    analyzer = EEGAnalyzer(DATASETS)
    plot_psd(analyzer)

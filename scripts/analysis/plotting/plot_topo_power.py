
from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS


def plot_topo_power(analyzer: EEGAnalyzer = None, band: tuple[float, float] = (8, 12)):
    """
    Plots the topographic distribution of power for each channel in the EEG data.
    The figures inlcude both states and a difference plot for OT–MW.
    """
    if analyzer is None:
        raise ValueError("EEGAnalyzer instance is required to plot topographic power.")

    for dataset_name, dataset in analyzer.datasets.items():
        print(f"Processing dataset: {dataset_name}")
        for subject_id, subject in dataset.subjects.items():
            for session_id, session in subject.recordings.items():
                print(f"Subject: {subject_id}, Session: {session_id}")
                # Plot topographic power
                session.plot_topo_power_comparison(task_input="all", band=band, show=False)
                session.plot_topo_power_comparison(task_input="all", band=band, show=False, use_decibel=True)
                print(f"Topographic power plots for {subject_id} - {session_id} completed.")


if __name__ == "__main__":
    ANALYZER_NAME = "UnprocessedAnalyzer"
    analyzer = EEGAnalyzer(DATASETS)
    bands = [(8, 12), (8, 9), (9, 10), (10, 11), (11, 12)]

    for band in bands:
        plot_topo_power(analyzer, band=band)

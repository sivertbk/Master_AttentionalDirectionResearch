"""
Use this script to change the path of the EEGAnalyzer derivatives.
This is useful when you want to change the location of the derivatives folder
without having to recreate the EEGAnalyzer object.

If you try to run a analyzer not made with this PC, you may encounter issues with file paths.
This script will update the derivatives path of the EEGAnalyzer object.
"""

import os

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import EEGANALYZER_SETTINGS, EEGANALYZER_OBJECT_DERIVATIVES_PATH


if __name__ == "__main__":
    eeganalyzer_kwargs = EEGANALYZER_SETTINGS.copy()
    ANALYZER_NAME = EEGANALYZER_SETTINGS["analyzer_name"]

    # Trying to load the EEGAnalyzer
    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(**eeganalyzer_kwargs)
        analyzer.save_analyzer()

    analyzer.derivatives_path = os.path.join(EEGANALYZER_OBJECT_DERIVATIVES_PATH, ANALYZER_NAME)
    analyzer.save_analyzer()
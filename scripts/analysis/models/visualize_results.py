import matplotlib.pyplot as plt
import os

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import PLOTS_PATH






if __name__ == "__main__":
    ANALYZER_NAME = "UnprocessedAnalyzer"

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print("Analyzer not found. Create the analyzer first.")
        exit()
    print(f"Loaded analyzer: {analyzer.analyzer_name}")
    if analyzer.df is None:
        print("Analyzer DataFrame is missing. Load or create the DataFrame first.")
        exit()  

    model_summaries = analyzer.summarize_all_fitted_models(save=True)
    print(model_summaries)

    

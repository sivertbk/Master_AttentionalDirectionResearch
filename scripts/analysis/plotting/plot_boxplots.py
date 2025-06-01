import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import set_plot_style, channel_positions, PLOTS_PATH

set_plot_style()

if __name__ == "__main__":
    ANALYZER_NAME = "UnprocessedAnalyzer"
    analyzer = EEGAnalyzer.load_analyzer(analyzer_name=ANALYZER_NAME)

    analyzer.viz.plot_boxplot(group_by_region=True)
    analyzer.viz.plot_boxplot()
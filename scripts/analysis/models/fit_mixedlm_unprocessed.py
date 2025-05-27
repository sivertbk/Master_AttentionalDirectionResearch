from eeg_analyzer.eeg_analyzer import EEGAnalyzer

def fit_mixedlm_unprocessed(analyzer: EEGAnalyzer):
    """
    Fit a mixed linear model to the unprocessed EEG data.
    This function assumes that the EEGAnalyzer instance has been loaded
    and contains the necessary DataFrame with unprocessed data.
    """
    if analyzer is None:
        print("Analyzer not found. Create the analyzer first.")
        return
    
    if analyzer.df is None:
        print("Analyzer DataFrame is missing. Load or create the DataFrame first.")
        return
    
    analyzer.fit_models_by_channel(
        formula="band_power ~ C(state)",
        value_col="band_power",
        state_col="state",
        group_col="subject_session"
    )
    analyzer.fit_models_by_roi(
        formula="band_power ~ C(state)",
        value_col="band_power",
        state_col="state",
        group_col="subject_session"
    )


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

    fit_mixedlm_unprocessed(analyzer)
    print("Mixed linear model fitting completed.")

    analyzer.summarize_fitted_models(save=True)
    print("Model summaries generated.")

    analyzer.save_analyzer()  # Save the analyzer state after fitting models


from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS

analyzers = {'UnprocessedAnalyzer': 'Unprocessed data', 'ZScoreFilteredAnalyzer': 'Z-score filtered data', 'IQRFilteredAnalyzer': 'IQR filtered data'}

if __name__ == "__main__":
    for analyzer_name, description in analyzers.items():
        print(f"Processing analyzer: {analyzer_name}")
        existing_analyzer = EEGAnalyzer.load_analyzer(analyzer_name)
        if existing_analyzer is not None:
            print(f"Analyzer '{analyzer_name}' already exists. Skipping creation.")
            continue

        print(f"Creating analyzer: {analyzer_name} - {description}")
        analyzer = EEGAnalyzer(
            DATASETS,
            analyzer_name=analyzer_name,
            description=description
        )
        # Create the initial full dataframe for this analyzer (unfiltered)
        analyzer.create_dataframe() 
        # Save the analyzer state immediately after creation
        analyzer.save_analyzer()
        print(f"Analyzer '{analyzer_name}' created and saved.")
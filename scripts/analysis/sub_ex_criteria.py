from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from eeg_analyzer.processor import Processor

ANALYZER_NAME = "SubExAnalyzer"


if __name__ == "__main__":
    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    

    # remove subjects where group==ctr
    exclude = {}
    exclude_control = analyzer.df[analyzer.df['group'] == 'ctr'].groupby('dataset')['subject_id'].unique().to_dict()
    exclude.update(exclude_control)
    analyzer.exclude_subjects(exclude=exclude)

    # apply state ratio filter
    analyzer.apply_recording_state_ratio_filter(min_ratio=1/9, max_ratio=9)

    # Generate summary tables
    analyzer.create_dataframe()

    # Save the analyzer state after processing
    analyzer.save_analyzer()
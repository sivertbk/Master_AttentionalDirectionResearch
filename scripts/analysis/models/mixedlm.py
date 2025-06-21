import statsmodels.formula.api as smf
import pandas as pd

from eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS

ANALYZER_NAME = "eeg_analyzer_test"

# Trying to load the EEGAnalyzer
analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
if analyzer is None:
    print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
    analyzer = EEGAnalyzer(DATASETS, ANALYZER_NAME)
    analyzer.save_analyzer()

# Creating a DataFrame with the data
df = analyzer.create_dataframe()

# Print out information about the DataFrame
print("DataFrame Information:")
print("="*30)
print(f"Total number of rows: {len(df)}")
print(f"Datasets: {df['dataset'].unique().tolist()}")

for dataset_name in df['dataset'].unique():
    df_dataset = df[df['dataset'] == dataset_name]
    print(f"\n--- Dataset: {dataset_name} ---")
    print(f"  Task orientation: {df_dataset['task_orientation'].iloc[0]}")
    print(f"  Subjects: {df_dataset['subject_id'].nunique()} ({df_dataset['subject_id'].unique().tolist()})")
    print(f"  Sessions: {df_dataset['session_id'].nunique()}")
    print(f"  Channels: {df_dataset['channel'].nunique()}")
    print(f"  Groups: {df_dataset['group'].unique().tolist()}")
    print(f"  States: {df_dataset['state'].unique().tolist()}")
    print(f"  Total data points: {len(df_dataset)}")
    
    print("\n  Data points per channel:")
    print(df_dataset.groupby('channel')['log_band_power'].count())
    
    print("\n  Data points per subject:")
    print(df_dataset.groupby('subject_id')['log_band_power'].count())


# Fitting a linear mixed effects model for each dataset
for dataset_name in df['dataset'].unique():
    print(f"\n\nFitting model for dataset: {dataset_name}")
    df_dataset = df[df['dataset'] == dataset_name].copy()

    # Fitting a linear mixed effects model
    model = smf.mixedlm(
        "log_band_power ~ C(state)",
        df_dataset,
        groups=df_dataset["subject_id"],
        re_formula="1",  # Random intercept per subject
        vc_formula={
            "session": "0 + C(session_id)",
            "channel": "0 + C(channel)",
            "channel_state": "0 + C(channel):C(state)"
        }
    )

    result = model.fit(method='lbfgs')
    print(result.summary())

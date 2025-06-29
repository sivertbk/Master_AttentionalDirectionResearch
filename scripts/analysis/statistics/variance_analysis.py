import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import EEGANALYZER_SETTINGS

eeganalyzer_kwargs = EEGANALYZER_SETTINGS.copy()
ANALYZER_NAME = EEGANALYZER_SETTINGS["analyzer_name"]

# Trying to load the EEGAnalyzer
analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
if analyzer is None:
    print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
    analyzer = EEGAnalyzer(**eeganalyzer_kwargs)
    analyzer.save_analyzer()
    

# create the DataFrame if it does not exist
if analyzer.df is None:
    print("Creating DataFrame for the variance analysis...")
    analyzer.create_dataframe()
    analyzer.save_analyzer()

if analyzer.are_summary_tables_outdated():
    # create new summary tables for the variance analysis
    analyzer.generate_standard_summary_tables(
        base_filter_type_label="IQRprocessed",
        target_value_col="log_band_power"
    )
    analyzer.save_analyzer()

# Inspect the DataFrame
print("DataFrame Info:")
print(analyzer.df.info())
print("\nDataFrame Head:")
print(analyzer.df.head())

# Check for missing values
print("\nMissing values:")
print(analyzer.df[['dataset', 'subject_id', 'session_id', 'channel', 'task', 'state']].isnull().sum())

# Handle missing values (example: remove rows with any missing values in relevant columns)
analyzer.df.dropna(subset=['dataset', 'subject_id', 'session_id', 'channel', 'task', 'state'], inplace=True)

# Remove the 'collision' task
analyzer.df = analyzer.df[analyzer.df['task'] != 'collision']

# Create a composite subject ID
analyzer.df['composite_subject_id'] = analyzer.df['dataset'] + '_' + analyzer.df['subject_id'].astype(str)

# Verify data types (example: convert to categorical if needed)
analyzer.df['dataset'] = analyzer.df['dataset'].astype('category')
analyzer.df['composite_subject_id'] = analyzer.df['composite_subject_id'].astype('category')
analyzer.df['session_id'] = analyzer.df['session_id'].astype('category')
analyzer.df['channel'] = analyzer.df['channel'].astype('category')
analyzer.df['task'] = analyzer.df['task'].astype('category')
analyzer.df['state'] = analyzer.df['state'].astype('category')

# Remove unused categories after dropping NAs
analyzer.df['dataset'] = analyzer.df['dataset'].cat.remove_unused_categories()
analyzer.df['composite_subject_id'] = analyzer.df['composite_subject_id'].cat.remove_unused_categories()
analyzer.df['session_id'] = analyzer.df['session_id'].cat.remove_unused_categories()
analyzer.df['channel'] = analyzer.df['channel'].cat.remove_unused_categories()
analyzer.df['task'] = analyzer.df['task'].cat.remove_unused_categories()
analyzer.df['state'] = analyzer.df['state'].cat.remove_unused_categories()

# Inspect categories
print("\nUnique categories:")
print("dataset:", analyzer.df['dataset'].cat.categories)
print("composite_subject_id:", analyzer.df['composite_subject_id'].cat.categories)
print("session_id:", analyzer.df['session_id'].cat.categories)
print("channel:", analyzer.df['channel'].cat.categories)
print("task:", analyzer.df['task'].cat.categories)
print("state:", analyzer.df['state'].cat.categories)

# Check the number of unique values for each categorical variable
print("Number of unique values per categorical variable:")
for col in ['dataset', 'composite_subject_id', 'session_id', 'channel', 'task', 'state']:
    print(f"{col}: {analyzer.df[col].nunique()}")

# Check for near-constant variables (variables with very few unique values)
print("\nValue counts for each categorical variable:")
for col in ['dataset', 'composite_subject_id', 'session_id', 'channel', 'task', 'state']:
    print(f"\n{col}:\n{analyzer.df[col].value_counts()}")

# You can define interaction or additive model
model = ols('log_band_power ~ C(dataset) + C(composite_subject_id) + C(session_id) + C(channel) + C(task) + C(state)', data=analyzer.df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # Type 2 is common for unbalanced designs
print(anova_table)










import pandas as pd
import matplotlib.pyplot as plt
import os
import re

import utils.config as config
from utils.config import DATASETS

DATASET = DATASETS['jin2019']
path = os.path.join(DATASET.path_root, "author_folder", "study1", "raw", "beh")

# Function to replace block_name content
def replace_block_name(block_name):
    if isinstance(block_name, str):
        if re.match(r'^vs$', block_name):
            return 0
        elif re.match(r'^sart\d*$', block_name):
            return 1
    return block_name

# Function to replace NaN values and invalid values in 'response_probe_content' only
def clean_response_probe_content(x):
    try:
        if pd.isna(x):
            return 0
        x = int(x)
        return x if 1 <= x <= 6 else 0
    except (ValueError, TypeError):
        return 0
    
def clean_probe_on(x):
    try:
        return int(float(x))  # Handles string, float, int safely
    except (ValueError, TypeError):
        return 0  # If invalid, set to 0

# List of relevant columns
relevant_columns = ['probe_on', 'response_probe_content', 'block_name']

# Dict to store dataframes
dataframes = {}

# Load and process files
for subject in DATASET.subjects:
    for session in DATASET.sessions:
        file_name = f'subject-{subject}_s{session}.csv'
        file_path = os.path.join(path, file_name)
        key = f'subject-{subject}'
        
        if os.path.exists(file_path):
            if key not in dataframes:
                dataframes[key] = {}
            df = pd.read_csv(file_path)

            # Keep only relevant columns
            df = df[relevant_columns].copy()

            # Clean 'block_name'
            df['block_name'] = df['block_name'].apply(replace_block_name)

            # Clean 'response_probe_content'
            df['response_probe_content'] = df['response_probe_content'].apply(clean_response_probe_content)

            # Ensure 'probe_on' is numeric
            df['probe_on'] = df['probe_on'].apply(clean_probe_on)

            dataframes[key][f'session-{session}'] = df
        else:
            print(f"File not found: {file_path}")

# Define save path
save_path = os.path.join(DATASET.path_derivatives, 'events')
os.makedirs(save_path, exist_ok=True)

# Export cleaned data
for subject, sessions in dataframes.items():
    for session, df in sessions.items():
        file_name = f'{subject}_{session}_events.csv'
        file_path = os.path.join(save_path, file_name)
        df.to_csv(file_path, index=False)

import pandas as pd
import matplotlib.pyplot as plt
import os
import re

import utils.config as config

path = os.path.join(config.DATASETS_PATH, "jin2019/study1/raw/beh/")

# Function to replace block_name content
def replace_block_name(block_name):
    if re.match(r'^vs$', block_name):
        return 0
    elif re.match(r'^sart\d*$', block_name):
        return 1
    else:
        return block_name


# Function to replace NaN values and ints outside [1,5] with zero and ensure all values are integers
def replace_nan_and_strings(df):
    def convert_and_replace(x):
        if pd.isna(x):
            return 0
        try:
            x = int(x)
        except ValueError:
            return 0
        return x if 1 <= x <= 6 else 0

    df = df.applymap(convert_and_replace)
    return df


# Dict to store dataframes
dataframes = {}

# Generate all the keys for the dict
for i in range(1, 31):
    for j in range(1, 3):
        file_name = f'subject-{i}_s{j}.csv'
        file_path = os.path.join(path, file_name)
        key = f'subject-{i}'
        if os.path.exists(file_path):
            if key not in dataframes:
                dataframes[key] = {}
            dataframes[key][f'session-{j}'] = pd.read_csv(file_path)

# List of relevant columns
relevant_columns = ['probe_on', 'response_probe_content', 'practice_on', 'block_name']

# Drop all irrelevant columns in the dataframes
for subject, sessions in dataframes.items():
    for session, df in sessions.items():
        dataframes[subject][session] = dataframes[subject][session][relevant_columns]


# Initialize an empty list to store the subject and session numbers
subject_session_with_practice = []

# Iterate through the dataframes to find the required datasets
for subject, sessions in dataframes.items():
    for session, df in sessions.items():
        if df['practice_on'].eq(1).any():
            subject_number = int(subject.split('-')[1])
            session_number = int(session.split('-')[1])
            subject_session_with_practice.append((subject_number, session_number))

#Print out the subject and session numbers with practice
print('Subjects and sessions with practice on:')
for subject, session in subject_session_with_practice:
    print(f'Subject: {subject}, Session: {session}')

# remove the subjects and sessions with practice and drop the column as it is unnecessary at this point.
for subject, sessions in dataframes.items():
    for session, df in sessions.items():
        if (int(subject.split('-')[1]), int(session.split('-')[1])) in subject_session_with_practice:
            dataframes[subject][session] = dataframes[subject][session][dataframes[subject][session].practice_on.eq(0)]
        dataframes[subject][session] = dataframes[subject][session].drop(columns=['practice_on'])

dataframes['subject-1']['session-1']



# Apply the function to all dataframes
for subject, sessions in dataframes.items():
    for session, df in sessions.items():
        df['block_name'] = df['block_name'].apply(replace_block_name)


# Apply the function to all dataframes
for subject, sessions in dataframes.items():
    for session, df in sessions.items():
        dataframes[subject][session] = replace_nan_and_strings(df)

# Display the updated dataframe for subject-3 session-1
dataframes['subject-3']['session-1']

# Define the path to the new directory
new_path = os.path.join(path, 'new_events')

# Create the directory if it does not exist
if not os.path.exists(new_path):
    os.makedirs(new_path)

# Export each dataframe to a CSV file in the new directory
for subject, sessions in dataframes.items():
    for session, df in sessions.items():
        file_name = f'{subject}_{session}_events.csv'
        file_path = os.path.join(new_path, file_name)
        df.to_csv(file_path, index=False)
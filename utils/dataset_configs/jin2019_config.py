import pandas as pd

from utils.dataset_config import DatasetConfig


# =============================================================================
#                 Jin et al. (2019) EEG Dataset Configuration
# =============================================================================

subjects = list(range(1, 31)) # All 30 subjects
sessions = [1, 2] # Both sessions
task_orientation = "external"

# Dataframe for file localizations (encapsulate creation in function)
def create_subject_session_dataframe(subjects, sessions):
    df = pd.DataFrame("", index=subjects, columns=sessions, dtype=str)
    for sub in subjects:
        for sess in sessions:
            df.at[sub, sess] = f"sub{sub}_{sess}"
    return df

# Create a mapping from 64 to 128 channel names (Realised that i should have done this the other way around)
mapping_64_to_128 ={'Fp1':'C29',
                    'AF7':'C30',
                    'AF3':'C28',
                    'F1':'C25',
                    'F3':'D4',
                    'F5':'D5',
                    'F7':'D7',
                    'FT7':'D8',
                    'FC5':'D10',
                    'FC3':'D12',
                    'FC1':'C24',
                    'C1':'D14',
                    'C3':'D19',
                    'C5':'D21',
                    'T7':'D23',
                    'TP7':'D24',
                    'CP5':'D26',
                    'CP3':'D28',
                    'CP1':'D16',
                    'P1':'A5',
                    'P3':'A7',
                    'P5':'D29',
                    'P7':'D31',
                    'P9':'D32',
                    'PO7':'A10',
                    'PO3':'A17',
                    'O1':'A15',
                    'Iz':'A25',
                    'Oz':'A23',
                    'POz':'A21',
                    'Pz':'A19',
                    'CPz':'A3',
                    'Fpz':'C17',
                    'Fp2':'C16',
                    'AF8':'C8',
                    'AF4':'C15',
                    'AFz':'C19',
                    'Fz':'C21',
                    'F2':'C12',
                    'F4':'C4',
                    'F6':'C5',
                    'F8':'C7',
                    'FT8':'B27',
                    'FC6':'B29',
                    'FC4':'B31',
                    'FC2':'C11',
                    'FCz':'C23',
                    'Cz':'A1',
                    'C2':'B20',
                    'C4':'B22',
                    'C6':'B24',
                    'T8':'B26',
                    'TP8':'B14',
                    'CP6':'B16',
                    'CP4':'B18',
                    'CP2':'B2',
                    'P2':'A32',
                    'P4':'B4',
                    'P6':'B13',
                    'P8':'B11',
                    'P10':'B10',
                    'PO8':'B7',
                    'PO4':'A30',
                    'O2':'A28'}
# Invert the dictionary and add 'temp_'
mapping_128_to_64 = {'temp_' + v: k for k, v in mapping_64_to_128.items()}

# Create a mapping for non-EEG channels
mapping_non_eeg = {'EXG1': 'LHEOG',
                'EXG2': 'RHEOG',
                'EXG3': 'UVEOG',
                'EXG4': 'LVEOG'}

# create event id dictionary
event_id_map = {'probe_off/vs/0/undefined': 0,
            'probe_off/vs/1/OT': 1,
            'probe_off/vs/2/OT': 2,
            'probe_off/vs/3/MW': 3,
            'probe_off/vs/4/undefined': 4,
            'probe_off/vs/5/MW': 5,
            'probe_off/vs/6/undefined': 6,
            'probe_off/sart/0/undefined': 10,
            'probe_off/sart/1/OT': 11,
            'probe_off/sart/2/OT': 12,
            'probe_off/sart/3/MW': 13,
            'probe_off/sart/4/undefined': 14,
            'probe_off/sart/5/MW': 15,
            'probe_off/sart/6/undefined': 16,
            'probe_on/vs/0/undefined': 100,
            'probe_on/vs/1/OT': 101,
            'probe_on/vs/2/OT': 102,
            'probe_on/vs/3/MW': 103,
            'probe_on/vs/4/undefined': 104,
            'probe_on/vs/5/MW': 105,
            'probe_on/vs/6/undefined': 106,
            'probe_on/sart/0/undefined': 110,
            'probe_on/sart/1/OT': 111,
            'probe_on/sart/2/OT': 112,
            'probe_on/sart/3/MW': 113,
            'probe_on/sart/4/undefined': 114,
            'probe_on/sart/5/MW': 115,
            'probe_on/sart/6/undefined': 116}

# Define the event classes
event_classes = {
    'vs/MW': [103, 105],
    'sart/MW': [113, 115],
    'vs/OT': [101, 102],
    'sart/OT': [111, 112]
}

# Construct dataset-specific configuration object
DATASET_CONFIG = DatasetConfig(
    name="Jin et al. (2019)",
    f_name="jin2019",
    extension="bdf",
    task_orientation=task_orientation,
    subjects=subjects,
    sessions=sessions,
    tasks=["vs", "sart"],
    states=["MW", "OT"],
    mapping_channels=mapping_128_to_64,
    mapping_non_eeg=mapping_non_eeg,
    event_id_map=event_id_map,
    event_classes=event_classes,
    state_classes={
        "focused": event_classes["vs/OT"] + event_classes["sart/OT"],
        "mind-wandering": event_classes["vs/MW"] + event_classes["sart/MW"],
    },
    task_classes={
        "visual_search": event_classes["vs/MW"] + event_classes["vs/OT"],
        "SART": event_classes["sart/MW"] + event_classes["sart/OT"],
    },
    extra_info={
        "subject_session_df": create_subject_session_dataframe(subjects, sessions)
    }
)


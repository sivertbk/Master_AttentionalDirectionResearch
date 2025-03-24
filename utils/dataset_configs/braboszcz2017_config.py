from utils.dataset_config import DatasetConfig


# =============================================================================
#               Braboszcz et al. (2017) EEG Dataset Configuration
# =============================================================================

subjects = list(range(60, 79)) # All 30 subjects
sessions = [1] # Only one session


# Initialize an empty dictionary for subject grouping
subject_groups = {}

# Add subjects to the dictionary based on their group
for i in range(1, 25):
    subject_groups[f"sub-{i:03d}"] = "htr" #Himalayan

for i in range(25, 56):
    subject_groups[f"sub-{i:03d}"] = "ctr" #Control

for i in range(56, 60):
    subject_groups[f"sub-{i:03d}"] = "tm" #Transcendental???

for i in range(60, 79):
    subject_groups[f"sub-{i:03d}"] = "vip" #Vipassana

for i in range(79, 99):
    subject_groups[f"sub-{i:03d}"] = "sny" #Shoonya Yoga


# Creating event id for each task and practice
event_id = {'htr/med1':1, 'htr/med2':2, 'htr/think1':3, 'htr/think2':4,
          'ctr/med1':5, 'ctr/med2':6, 'ctr/think1':7, 'ctr/think2':8,
          'tm/med1': 9, 'tm/med2' :10, 'tm/think1':11, 'tm/think2':12,
          'vip/med1':13, 'vip/med2':14, 'vip/think1':15, 'vip/think2':16,
          'sny/med1':17, 'sny/med2':18, 'sny/think1':19, 'sny/think2':20}

braboszcz2017_config = DatasetConfig(
    name="Braboszcz et al. (2017)",
    subjects=subjects,
    sessions=sessions,  # or define explicitly if relevant
    mapping_channels={},  # specify if available
    mapping_non_eeg={},   # specify if available
    event_id_map=event_id,
    event_classes=None,   # specify if needed
    path=None,            # specify if available
    extra_info={
        "subject_groups": subject_groups
    }
)